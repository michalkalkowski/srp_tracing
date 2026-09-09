"""
Pinning tests for RectGrid.calculate_graph()'s water_links parameter.

water_links used to have a real indexing bug: whenever the two linked node
groups had different sizes, the forward-direction edges (link0 -> link1)
got the right cost, but roughly half the reverse-direction edges
(link1 -> link0) were paired with the wrong distance. This came from
building the forward cost matrix with shape (len(link1), len(link0)) but
reusing its *transpose*'s flattened values, shape (len(link0), len(link1)),
for the reverse edges against row/col index arrays still in the original
(len(link1), len(link0)) flattening order -- those two flattenings only
walk the pairs in the same order when the two groups happen to be the same
size. Fixed by reusing one single, symmetric flattened cost array for both
directions instead of two different flattenings that were assumed (wrongly)
to agree.
"""
import numpy as np
import pytest

from srp_tracing import grid


def test_water_links_gives_correct_distance_for_every_pair(isotropic_material):
    material = isotropic_material(vp=5.9)  # any material: water_links ignores it
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=4.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))

    # extra points, deliberately in two differently-sized groups (this is
    # exactly the case that exposed the bug -- equal-sized groups happened
    # to work by coincidence)
    group_a = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])
    group_b = np.array([[10.0, 5.0], [13.0, 6.0]])
    g.add_points(sources=group_a, targets=group_b)
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})

    c0 = 1.48
    link0, link1 = g.source_idx, g.target_idx
    g.calculate_graph(water_links=[(link0, link1)], c0=c0)

    dense = g.edges.toarray()
    for a in link0:
        for b in link1:
            expected = np.linalg.norm(g.grid[a] - g.grid[b]) / c0
            assert dense[a, b] == pytest.approx(expected, rel=1e-9), f"{a}->{b}"
            assert dense[b, a] == pytest.approx(expected, rel=1e-9), f"{b}->{a}"


def test_water_links_symmetric_group_sizes_still_correct(isotropic_material):
    """The bug happened to not manifest for equal-sized groups; pin that
    this case (already implicitly covered, but worth being explicit about)
    still works after the fix."""
    material = isotropic_material(vp=5.9)
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=4.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))

    group_a = np.array([[10.0, 0.0], [11.0, 0.0]])
    group_b = np.array([[10.0, 5.0], [13.0, 6.0]])
    g.add_points(sources=group_a, targets=group_b)
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})

    c0 = 1.48
    link0, link1 = g.source_idx, g.target_idx
    g.calculate_graph(water_links=[(link0, link1)], c0=c0)

    dense = g.edges.toarray()
    for a in link0:
        for b in link1:
            expected = np.linalg.norm(g.grid[a] - g.grid[b]) / c0
            assert dense[a, b] == pytest.approx(expected, rel=1e-9)
            assert dense[b, a] == pytest.approx(expected, rel=1e-9)
