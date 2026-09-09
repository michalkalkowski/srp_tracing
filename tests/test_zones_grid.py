"""
Pinning tests for ZonesGrid.

ZonesGrid had no test coverage at all before this file -- partly because it
couldn't even be constructed in this environment: voronoi_finite_polygons_2d
used the ndarray.ptp() method, removed in NumPy 2.0 (fixed to np.ptp(...)),
and __init__ jittered its Voronoi seed points with an unseeded
np.random.randn(), making construction non-deterministic (fixed by adding
an optional random_state parameter, defaulting to the previous unseeded
behaviour).

calculate_graph() also used to build self.edges/zone_labels/distances/
ray_angles as dense NumPy arrays (self.edges[rows, cols] = edges), unlike
RectGrid/SimplRectGrid's sparse csr_matrix -- an O(N^2) memory cost for any
non-trivial zone mesh. It's sparse now, built via the same
dedup_min_edge_indices() helper used to fix RectGrid/SimplRectGrid's
duplicate-boundary-edge double-counting bug (two nodes on a shared zone
boundary get discovered once from each adjacent zone; the minimum cost is
kept for all four matrices consistently, not just self.edges).
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


def _build(isotropic_material, random_state=42):
    material = isotropic_material(vp=2.0)
    g = grid.ZonesGrid(a=10.0, b=2.0, c=14.0, zone_height=3.0,
                       max_width_split=4, max_dx=0.5, random_state=random_state)
    # Sources/targets placed well outside the chamfer (|x| >> c/2) so they
    # fall into the isotropic left/right zones rather than needing to land
    # exactly on a zone boundary edge.
    sources = np.array([[-20.0, 3.0], [20.0, 3.0]])
    targets = np.array([[-20.0, 6.0], [20.0, 6.0]])
    g.add_points(sources=sources, targets=targets)
    material_map = np.zeros(len(g.zones_edge_ind), dtype=int)
    property_map = np.zeros(len(g.zones_edge_ind))
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: material})
    return g


def test_construction_is_deterministic_with_a_seed(isotropic_material):
    g1 = _build(isotropic_material, random_state=42)
    g2 = _build(isotropic_material, random_state=42)
    g3 = _build(isotropic_material, random_state=43)

    np.testing.assert_array_equal(g1.grid, g2.grid)
    assert g1.grid.shape != g3.grid.shape or not np.array_equal(g1.grid, g3.grid)


def test_calculate_graph_matrices_are_sparse(isotropic_material):
    import scipy.sparse

    g = _build(isotropic_material)
    g.calculate_graph()

    for name in ("edges", "zone_labels", "distances", "ray_angles"):
        matrix = getattr(g, name)
        assert scipy.sparse.issparse(matrix), f"{name} is not sparse"
        assert matrix.shape == (g.grid.shape[0], g.grid.shape[0])


def test_calculate_graph_edges_are_symmetric(isotropic_material):
    """A single isotropic material gives an undirected metric: travel time
    from i to j must equal travel time from j to i."""
    g = _build(isotropic_material)
    g.calculate_graph()

    dense = g.edges.toarray()
    np.testing.assert_allclose(dense, dense.T, atol=1e-9)


def test_calculate_graph_snapshot(isotropic_material):
    g = _build(isotropic_material)
    g.calculate_graph()

    assert g.grid.shape == (187, 3)
    assert g.edges.shape == (187, 187)
    assert g.edges.nnz == 9880

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    tfs = s.tfs[:, g.target_idx]

    # source 0 and target 0 are both in the left isotropic zone, 3 units
    # apart, material vp=2.0 -> exact analytic travel time
    assert tfs[0, 0] == pytest.approx(1.5, rel=1e-9)
    assert tfs[1, 1] == pytest.approx(1.5, rel=1e-9)
    # cross paths (left source to right target and back) go through the
    # zone mesh; pin the current values, and confirm the near-symmetry
    # expected from the domain's left/right geometric symmetry (not exact,
    # since the Voronoi jitter differs between the two sides even with a
    # shared seed).
    assert tfs[0, 1] == pytest.approx(20.0586381, rel=1e-6)
    assert tfs[1, 0] == pytest.approx(20.05931492, rel=1e-6)
    assert tfs[0, 1] == pytest.approx(tfs[1, 0], rel=1e-3)


def test_calculate_graph_with_tie_link(isotropic_material):
    """tie_link used to have a length mismatch bug here too: rows/cols/edges
    were extended for the tie link, but distances/zone_labels/glob_angles
    were not, which would misalign (or crash) the sparse matrix
    construction. Confirm it now works and adds a zero-cost connection."""
    g = _build(isotropic_material)
    a, b = g.source_idx[0], g.source_idx[1]
    g.calculate_graph(tie_link=[[a], [b]])

    assert g.edges[a, b] == 0.0
    assert g.zone_labels.shape == g.edges.shape
    assert g.distances.shape == g.edges.shape
    assert g.ray_angles.shape == g.edges.shape
