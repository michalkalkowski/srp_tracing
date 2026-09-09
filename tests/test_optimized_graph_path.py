"""
Tests for RectGrid.set_up_graph() + update_edges() -- the "fast" alternative
to calculate_graph(): it precomputes per-pixel node geometry once
(set_up_graph) and fills in edge costs separately (update_edges), so
update_edges() can be re-run cheaply after material properties change,
without repeating the neighbour search.

This used to have two confirmed bugs:
1. A shortcut for isotropic "standard" cells returned the material's own
   wavespeed instead of consulting per-pixel property_map, silently wrong
   for mode='slowness_iso'.
2. update_edges(tie_link=...) called .extend() on numpy arrays (built by
   set_up_graph via np.concatenate), raising AttributeError.
Both are fixed now. These tests confirm the fast path produces the same
graph as calculate_graph() (the path every script in examples/ actually
uses), and that update_edges() -- including with tie_link -- is safe to
call repeatedly without accumulating stale state.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


def _two_pixel_slowness_iso_grid(no_seeds=4, pixel_size=4.0):
    """Two homogeneous cells, same isotropic material, but different
    per-pixel slowness via mode='slowness_iso' (v=3 in pixel 0, v=6 in
    pixel 1). The single material object's own wavespeed (v=1) is
    deliberately different from both, so a path that wrongly falls back to
    the material's own speed instead of property_map is easy to spot."""
    material_map = np.zeros((1, 2), dtype=int)
    property_map = np.array([[1 / 3.0, 1 / 6.0]])
    return material_map, property_map


def test_calculate_graph_honours_per_pixel_slowness(isotropic_material):
    material_map, property_map = _two_pixel_slowness_iso_grid()
    base_material = isotropic_material(vp=1.0)

    g = grid.RectGrid(nx=2, ny=1, cx=0.0, cy=0.0, pixel_size=4.0, no_seeds=4)
    g.assign_model(mode="slowness_iso", property_map=property_map)
    g.add_points()
    g.assign_materials(material_map, {0: base_material})
    g.calculate_graph()

    dense = g.edges.toarray()
    # an edge strictly inside pixel 0 must reflect v=3 (property_map), not
    # the material's own v=1
    within_pixel_0 = dense[np.ix_(range(0, 4), range(0, 4))]
    nonzero = within_pixel_0[within_pixel_0 > 0]
    assert nonzero.size > 0
    # edge cost = distance / v; the smallest possible distance between two
    # of the 4 seeds on one edge of a pixel_size=4, no_seeds=4 cell is 1.0
    assert np.min(nonzero) == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_set_up_graph_update_edges_matches_calculate_graph_for_slowness_iso(
    isotropic_material,
):
    """The "fast" path must honour per-pixel slowness for standard cells
    too, matching calculate_graph() exactly (this is exactly the case the
    iso_tofs shortcut used to get wrong)."""
    material_map, property_map = _two_pixel_slowness_iso_grid()
    base_material = isotropic_material(vp=1.0)

    g_ref = grid.RectGrid(nx=2, ny=1, cx=0.0, cy=0.0, pixel_size=4.0, no_seeds=4)
    g_ref.assign_model(mode="slowness_iso", property_map=property_map)
    g_ref.add_points()
    g_ref.assign_materials(material_map, {0: base_material})
    g_ref.calculate_graph()

    g_fast = grid.RectGrid(nx=2, ny=1, cx=0.0, cy=0.0, pixel_size=4.0, no_seeds=4)
    g_fast.assign_model(mode="slowness_iso", property_map=property_map)
    g_fast.add_points()
    g_fast.assign_materials(material_map, {0: base_material})
    g_fast.set_up_graph()
    g_fast.update_edges()

    np.testing.assert_allclose(g_ref.edges.toarray(), g_fast.edges.toarray())


def test_set_up_graph_update_edges_matches_calculate_graph_for_orientations(
    isotropic_material,
):
    """Sanity check on the more common mode='orientations' path too, using
    the same small weld-like layout as test_regression_small_model.py."""
    nx, ny = 5, 4
    dx = 2.0
    no_seeds = 3
    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, 2] = 1  # slower "weld" column
    property_map = np.zeros((ny, nx))

    parent = isotropic_material(vp=5.9)
    weld = isotropic_material(vp=3.24)
    sources = np.array([[-3.0, 3.0], [0.0, 3.0], [3.0, 3.0]])
    targets = np.array([[-3.0, -3.0], [0.0, -3.0], [3.0, -3.0]])

    g_ref = grid.RectGrid(nx=nx, ny=ny, cx=0.0, cy=0.0, pixel_size=dx, no_seeds=no_seeds)
    g_ref.assign_model(mode="orientations", property_map=property_map)
    g_ref.add_points(sources=sources, targets=targets)
    g_ref.assign_materials(material_map, {0: parent, 1: weld})
    g_ref.calculate_graph()

    g_fast = grid.RectGrid(nx=nx, ny=ny, cx=0.0, cy=0.0, pixel_size=dx, no_seeds=no_seeds)
    g_fast.assign_model(mode="orientations", property_map=property_map)
    g_fast.add_points(sources=sources, targets=targets)
    g_fast.assign_materials(material_map, {0: parent, 1: weld})
    g_fast.set_up_graph()
    g_fast.update_edges()

    np.testing.assert_allclose(g_ref.edges.toarray(), g_fast.edges.toarray(), atol=1e-10)


def _isolated_source_grid(isotropic_material):
    """A single-pixel grid plus a source placed far outside any pixel's
    search radius, so it starts out with zero geometric edges -- makes a
    tie_link's effect on connectivity unambiguous (no pre-existing edge to
    confuse it with)."""
    material_map = np.zeros((1, 1), dtype=int)
    property_map = np.zeros((1, 1))
    material = isotropic_material(vp=2.0)

    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=4.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=property_map)
    g.add_points(sources=np.array([[100.0, 100.0]]), targets=np.array([[0.0, 0.0]]))
    g.assign_materials(material_map, {0: material})
    g.set_up_graph()
    return g


def test_update_edges_with_tie_link_connects_otherwise_unreachable_nodes(
    isotropic_material,
):
    g = _isolated_source_grid(isotropic_material)

    g.update_edges()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert not np.isfinite(s.tfs[0, g.target_idx[0]])

    g.update_edges(tie_link=[[g.source_idx[0]], [g.target_idx[0]]])
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert s.tfs[0, g.target_idx[0]] == pytest.approx(0.0)


def test_update_edges_does_not_accumulate_tie_links_across_calls(isotropic_material):
    """self.rows/self.cols (the base geometry from set_up_graph) must stay
    untouched by tie_link -- calling update_edges() again without tie_link
    must restore the original (dis)connectivity, not carry the previous
    call's tie_link forward."""
    g = _isolated_source_grid(isotropic_material)

    g.update_edges(tie_link=[[g.source_idx[0]], [g.target_idx[0]]])
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert s.tfs[0, g.target_idx[0]] == pytest.approx(0.0)

    g.update_edges()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert not np.isfinite(s.tfs[0, g.target_idx[0]])


def test_update_edges_with_mismatched_tie_link_raises(isotropic_material):
    g = _isolated_source_grid(isotropic_material)
    with pytest.raises(ValueError):
        g.update_edges(tie_link=[[g.source_idx[0]], [g.target_idx[0], g.target_idx[0]]])
