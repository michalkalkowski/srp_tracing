"""
Regression/snapshot test for a small, weld-like SimplRectGrid model.

SimplRectGrid counterpart of test_regression_small_model.py (RectGrid): a
small chamfer with a slower "weld" column between two "parent" regions,
run through trim_to_chamfer -> simplify_grid -> add_points ->
calculate_graph -> Solver -> shortest_path. Pins the current numerical
output so a refactor that changes behaviour gets caught; see that file for
the general rationale (anisotropic materials out of scope here for the same
reason).

This specific combination of methods is worth its own snapshot: three real
bugs were found and fixed here during tidy-up (a tie_link crash, a mirror-
plane filter that discarded every point in a pixel row sitting exactly on
y=0, and -- most importantly -- simplify_grid() silently zeroing the entire
material map when called with its default arguments, which every real
script does). test_ogilvy_simpl_validation.py confirms the fixed behaviour
agrees with finite-element data on the real weld case; this test pins the
exact numbers on a small, fast, synthetic one so any future change to this
path is caught immediately rather than only when someone next runs the
slower FE validation.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


def _build_small_weld_model(isotropic_material):
    a, b, c = 10.0, 2.0, 14.0
    dx = 2.0
    no_seeds = 4
    nx, ny = 8, 6
    cx, cy = 0.0, a / 2

    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, nx // 2 - 1: nx // 2 + 1] = 1  # slower "weld" columns
    property_map = np.zeros((ny, nx))

    parent = isotropic_material(vp=5.9)
    weld = isotropic_material(vp=3.24)

    g = grid.SimplRectGrid(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx, no_seeds=no_seeds)
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: parent, 1: weld})
    g.trim_to_chamfer(a, b, c)
    g.simplify_grid()

    # Two points on each side of the chamfer, deliberately clear of the
    # zone-boundary edge cases exercised in test_optimized_graph_path.py.
    sources = np.array([[-5.0, 2.0], [-5.0, 5.0], [5.0, 2.0], [5.0, 5.0]])
    s_ix = np.array([0, 1, 2, 3])
    t_ix = np.array([0, 1, 2, 3])
    g.add_points(points=sources, sources=s_ix, targets=t_ix)
    g.calculate_graph()
    return g


EXPECTED_TOFS = np.array([
    [0.0, 0.50847458, 2.13326224, 2.46054839],
    [0.50847458, 0.0, 2.46054839, 2.73429622],
    [2.13326224, 2.46054839, 0.0, 0.50847458],
    [2.46054839, 2.73429622, 0.50847458, 0.0],
])

EXPECTED_PATH_0_TO_2 = np.array([
    [-5.0, 2.0],
    [-1.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [5.0, 2.0],
])


def test_small_weld_model_travel_times_snapshot(isotropic_material):
    g = _build_small_weld_model(isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    tofs = s.tfs[:, g.target_idx]
    np.testing.assert_allclose(tofs, EXPECTED_TOFS, rtol=1e-7, atol=1e-9)


def test_small_weld_model_is_symmetric(isotropic_material):
    # Structural sanity check independent of the hardcoded snapshot: single
    # isotropic materials on each side give an undirected metric, so the
    # travel-time matrix must equal its own transpose (this would also
    # catch a reintroduction of the edge-duplication double-counting bug
    # for source/source or target/target pairs sharing a zone boundary).
    g = _build_small_weld_model(isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    tofs = s.tfs[:, g.target_idx]
    np.testing.assert_allclose(tofs, tofs.T, rtol=1e-9)


def test_small_weld_model_graph_shape_snapshot(isotropic_material):
    g = _build_small_weld_model(isotropic_material)
    assert g.grid.shape == (237, 2)
    assert g.edges.shape == (237, 237)
    assert g.edges.nnz == 6284


def test_small_weld_model_ray_path_snapshot(isotropic_material):
    # The shortest path from the bottom-left source to the bottom-right
    # target cuts straight through the weld root rather than around it --
    # pin the actual node coordinates so a refactor that silently changes
    # routing/material lookup is caught even if the total travel time
    # happened to stay the same.
    g = _build_small_weld_model(isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx, with_points=True)

    paths = s.calculate_ray_paths(end=list(g.target_idx))
    path_nodes = paths[0][2]  # source #0 (bottom-left) -> target #2 (bottom-right)
    coords = g.grid[path_nodes][:, :2]

    assert coords.shape == EXPECTED_PATH_0_TO_2.shape
    np.testing.assert_allclose(coords, EXPECTED_PATH_0_TO_2, atol=1e-6)


if __name__ == "__main__":
    # Regenerate the golden values above after a deliberate, reviewed
    # behaviour change:
    #   python tests/test_regression_simpl_rect_grid.py
    import conftest as _conftest

    def _isotropic_material(vp, vs=None, rho=1.0):
        if vs is None:
            vs = vp / 2
        m = grid.WaveBasis(anisotropy=0, velocity_variant="group")
        m.set_material_props(_conftest.isotropic_stiffness(vp, vs, rho), rho)
        m.calculate_wavespeeds()
        return m

    g = _build_small_weld_model(_isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx, with_points=True)
    print("EXPECTED_TOFS =")
    print(repr(s.tfs[:, g.target_idx]))
    paths = s.calculate_ray_paths(end=list(g.target_idx))
    print("EXPECTED_PATH_0_TO_2 =")
    print(repr(g.grid[paths[0][2]][:, :2]))
