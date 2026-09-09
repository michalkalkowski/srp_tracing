"""
Regression/snapshot test for a small, weld-like RectGrid model.

This does not check anything analytically; it pins the *current* numerical
output of the full pipeline (RectGrid -> calculate_graph -> Solver ->
scipy.sparse.csgraph.shortest_path, plus ray-path reconstruction) on a small
but structurally realistic case: two isotropic materials arranged as a
slower "weld" column between two "parent" regions, with sources on the top
surface and targets on the bottom surface -- the same layout used by the
real validation scripts in examples/ (see also
tests/test_ogilvy_validation.py, which runs one of those end to end against
its finite-element reference data).

If any of these numbers change after a refactor, either the refactor changed
behaviour (bad, unless intended) or this snapshot needs to be regenerated
deliberately (see the "regenerate" block below).

Regenerated once already: calculate_graph() used to silently double-count
the cost of any edge connecting two nodes that both sit exactly on the
shared boundary between adjacent pixels (each pixel's independent neighbour
search finds the same pair and adds it, and coo_matrix sums duplicate
entries instead of deduplicating). Fixed via build_edge_matrix(), which
keeps the minimum cost for a repeated (row, col) pair. The diagonal-ish
entries above changed accordingly (cheaper, since a spurious 2x edge no
longer inflates some paths); the off-diagonal ones didn't, because those
particular shortest paths never happened to use a doubled boundary edge.

Anisotropic materials are intentionally out of scope here, to keep this
test fast and focused on the grid/graph/solver pipeline rather than the
wave-velocity model. Both materials below are isotropic, so
WaveBasis.get_wavespeed always returns a direction-independent value.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


def _build_small_weld_model(isotropic_material):
    nx, ny = 5, 4
    dx = 2.0
    no_seeds = 3

    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, 2] = 1  # middle column: slower "weld" material
    property_map = np.zeros((ny, nx))

    parent = isotropic_material(vp=5.9)
    weld = isotropic_material(vp=3.24)

    g = grid.RectGrid(nx=nx, ny=ny, cx=0.0, cy=0.0, pixel_size=dx,
                      no_seeds=no_seeds)
    g.assign_model(mode="orientations", property_map=property_map)

    sources = np.array([[-3.0, 3.0], [0.0, 3.0], [3.0, 3.0]])
    targets = np.array([[-3.0, -3.0], [0.0, -3.0], [3.0, -3.0]])
    g.add_points(sources=sources, targets=targets)
    g.assign_materials(material_map, {0: parent, 1: weld})
    g.calculate_graph()
    return g


EXPECTED_TOFS = np.array([
    [1.01694915, 1.35482815, 1.78508455],
    [1.35482815, 1.55093744, 1.35482815],
    [1.78508455, 1.35482815, 1.01694915],
])

EXPECTED_PATH_1_TO_1 = np.array([
    [0.0, 3.0],
    [1.0, 2.0],
    [1.0, 0.0],
    [1.0, -2.0],
    [0.0, -3.0],
])


def test_small_weld_model_travel_times_snapshot(isotropic_material):
    g = _build_small_weld_model(isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    tofs = s.tfs[:, g.target_idx]
    np.testing.assert_allclose(tofs, EXPECTED_TOFS, rtol=1e-7)


def test_small_weld_model_is_left_right_symmetric(isotropic_material):
    # Structural sanity check independent of the hardcoded snapshot: the
    # material map and source/target x-positions are mirror-symmetric about
    # x=0, so the travel-time matrix must be symmetric under simultaneous
    # row/column reversal.
    g = _build_small_weld_model(isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    tofs = s.tfs[:, g.target_idx]
    np.testing.assert_allclose(tofs, tofs[::-1, ::-1], rtol=1e-9)


def test_small_weld_model_graph_shape_snapshot(isotropic_material):
    g = _build_small_weld_model(isotropic_material)
    assert g.grid.shape == (145, 2)
    assert g.edges.shape == (145, 145)
    assert g.edges.nnz == 2476


def test_small_weld_model_ray_path_snapshot(isotropic_material):
    # The shortest path bends away from the slow weld column rather than
    # cutting straight through it -- pin the actual node coordinates so a
    # refactor that silently changes routing/material lookup is caught even
    # if the total travel time happened to stay the same.
    g = _build_small_weld_model(isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx, with_points=True)

    paths = s.calculate_ray_paths(end=list(g.target_idx))
    path_nodes = paths[1][1]  # source #1 (top centre) -> target #1 (bottom centre)
    coords = g.grid[path_nodes]

    assert coords.shape == EXPECTED_PATH_1_TO_1.shape
    np.testing.assert_allclose(coords, EXPECTED_PATH_1_TO_1, atol=1e-6)


if __name__ == "__main__":
    # Regenerate the golden values above after a deliberate, reviewed
    # behaviour change:
    #   python tests/test_regression_small_model.py
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
    print("EXPECTED_PATH_1_TO_1 =")
    print(repr(g.grid[paths[1][1]]))
