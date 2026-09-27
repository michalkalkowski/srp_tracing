"""
Characterization tests for the RectGrid/SimplRectGrid unification work (see
the "Unified chamfer='smooth'|'staircase' grid API" plan).

RectGrid.calculate_graph and SimplRectGrid.calculate_graph both build a
tie_link block by hand (grid.py, currently duplicated byte-for-byte between
the two classes) -- Phase 1 of that plan extracts it into one shared helper.
Unlike the plain-geometry path (already pinned tightly by
test_regression_small_model.py and test_regression_simpl_rect_grid.py), the
tie_link block itself has no direct test for calculate_graph() on either
class today (only ZonesGrid.calculate_graph and RectGrid/SimplRectGrid's
separate update_edges() fast path are covered elsewhere) -- these tests
close that gap first, so the extraction has something to break.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


def _isolated_source_rect_grid(isotropic_material):
    """A single-pixel RectGrid plus a source placed far outside any pixel's
    search radius, so it starts out with zero geometric edges -- makes a
    tie_link's effect on calculate_graph()'s connectivity unambiguous (no
    pre-existing edge to confuse it with). Mirrors
    test_optimized_graph_path.py's _isolated_source_grid, but left
    ungraphed (no set_up_graph()/update_edges()) since these tests exercise
    calculate_graph()'s own tie_link handling instead."""
    material_map = np.zeros((1, 1), dtype=int)
    property_map = np.zeros((1, 1))
    material = isotropic_material(vp=2.0)

    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=4.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=property_map)
    g.add_points(sources=np.array([[100.0, 100.0]]), targets=np.array([[0.0, 0.0]]))
    g.assign_materials(material_map, {0: material})
    return g


def test_rect_grid_calculate_graph_tie_link_connects_otherwise_unreachable_nodes(
    isotropic_material,
):
    g = _isolated_source_rect_grid(isotropic_material)
    g.calculate_graph()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert not np.isfinite(s.tfs[0, g.target_idx[0]])

    g = _isolated_source_rect_grid(isotropic_material)
    g.calculate_graph(tie_link=[[g.source_idx[0]], [g.target_idx[0]]])
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert s.tfs[0, g.target_idx[0]] == pytest.approx(0.0)


def test_rect_grid_calculate_graph_mismatched_tie_link_raises(isotropic_material):
    g = _isolated_source_rect_grid(isotropic_material)
    with pytest.raises(ValueError):
        g.calculate_graph(tie_link=[[g.source_idx[0]], [g.target_idx[0], g.target_idx[0]]])


def _small_simpl_rect_grid(isotropic_material):
    """Same small chamfer geometry as test_regression_simpl_rect_grid.py's
    _build_small_weld_model, with one source and one target point.

    Unlike RectGrid, SimplRectGrid.add_points() routes every added point
    into the left/right iso zone by which side of the chamfer it falls on
    (grid.py:1416-1434, a pure x-position test against the weld angle/
    outline, with no distance cutoff), so an added point can never be made
    geometrically unreachable the way RectGrid's KDTree-radius search
    allows -- there's no "far outside the search radius" for this class.
    tie_link is therefore verified directly against the edge matrix below
    rather than via a before/after reachability contrast."""
    a, b, c = 10.0, 2.0, 14.0
    dx = 2.0
    no_seeds = 4
    nx, ny = 8, 6
    cx, cy = 0.0, a / 2

    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, nx // 2 - 1: nx // 2 + 1] = 1
    property_map = np.zeros((ny, nx))
    material = isotropic_material(vp=2.0)

    g = grid.SimplRectGrid(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx, no_seeds=no_seeds)
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: material, 1: material})
    g.trim_to_chamfer(a, b, c)
    g.simplify_grid()

    points = np.array([[-5.0, 2.0], [5.0, 2.0]])
    g.add_points(points=points, sources=np.array([0]), targets=np.array([1]))
    return g


def test_simpl_rect_grid_calculate_graph_tie_link_adds_zero_cost_edge(isotropic_material):
    g = _small_simpl_rect_grid(isotropic_material)
    g.calculate_graph()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert s.tfs[0, g.target_idx[0]] > 1e-9  # a real, nonzero travel time exists first

    g = _small_simpl_rect_grid(isotropic_material)
    g.calculate_graph(tie_link=[[g.source_idx[0]], [g.target_idx[0]]])
    assert g.edges[g.source_idx[0], g.target_idx[0]] == pytest.approx(0.0, abs=1e-12)

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert s.tfs[0, g.target_idx[0]] == pytest.approx(0.0)


def test_simpl_rect_grid_calculate_graph_mismatched_tie_link_raises(isotropic_material):
    g = _small_simpl_rect_grid(isotropic_material)
    with pytest.raises(ValueError):
        g.calculate_graph(tie_link=[[g.source_idx[0]], [g.target_idx[0], g.target_idx[0]]])


def test_simpl_rect_grid_add_points_with_no_point_in_a_parent_zone(isotropic_material):
    """
    Regression: add_points's per-side mask_receiver list defaulted to float64 when empty (no added point at
    all fell in that side's parent-metal zone -- e.g. every source and target sits inside the weld itself, as
    a TFM delay law's array and region-of-interest points typically do), and ~ on a float array raised
    TypeError. Both sides empty here: source and target are placed well inside the chamfer, away from both
    flanks.
    """
    a, b, c = 10.0, 2.0, 14.0
    dx = 2.0
    nx, ny = 8, 6
    cx, cy = 0.0, a/2
    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, nx//2 - 1: nx//2 + 1] = 1
    property_map = np.zeros((ny, nx))
    material = isotropic_material(vp=2.0)

    g = grid.SimplRectGrid(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx, no_seeds=4)
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: material, 1: material})
    g.trim_to_chamfer(a, b, c)
    g.simplify_grid()

    points = np.array([[-0.5, 3.0], [0.5, 6.0]])   # both inside the weld, neither iso zone
    g.add_points(points=points, sources=np.array([0]), targets=np.array([1]))
    g.calculate_graph()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert np.isfinite(s.tfs[0, g.target_idx[0]])
