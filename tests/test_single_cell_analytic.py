"""
Pinning tests: a single, homogeneous, isotropic RectGrid cell.

RectGrid.calculate_graph() connects every node found inside a pixel's
bounding box to every other node in that same pixel (see
grid.py:RectGrid.calculate_graph, the ``points``/``neighbours`` loop). With
only one pixel in the whole domain, source and target end up directly
connected by a single edge whose cost is euclidean_distance / wavespeed, so
the scipy shortest_path solve must reproduce the analytic straight-line
travel time exactly (up to floating point), regardless of ``no_seeds``: the
graph is a Euclidean-distance metric divided by a constant, so no multi-hop
detour can ever beat the direct edge (triangle inequality).
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


@pytest.mark.parametrize("no_seeds", [2, 4, 8])
def test_single_cell_matches_analytic_straight_line(isotropic_material, no_seeds):
    v = 5.9
    material = isotropic_material(vp=v)

    pixel_size = 10.0
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=pixel_size,
                      no_seeds=no_seeds)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))

    source = np.array([[-4.0, -3.0]])
    target = np.array([[3.5, 4.0]])
    g.add_points(sources=source, targets=target)
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    expected = np.linalg.norm(target[0] - source[0]) / v
    actual = s.tfs[0, g.target_idx[0]]
    assert actual == pytest.approx(expected, rel=1e-9)


def test_single_cell_travel_time_scales_with_inverse_velocity(isotropic_material):
    pixel_size = 8.0
    source = np.array([[-3.0, -2.0]])
    target = np.array([[3.0, 2.5]])
    distance = np.linalg.norm(target[0] - source[0])

    tofs = {}
    for v in (2.0, 4.0):
        material = isotropic_material(vp=v)
        g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=pixel_size,
                          no_seeds=4)
        g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))
        g.add_points(sources=source, targets=target)
        g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})
        g.calculate_graph()
        s = solver.Solver(g)
        s.solve(source_indices=g.source_idx)
        tofs[v] = s.tfs[0, g.target_idx[0]]

    # doubling the velocity must exactly halve the travel time
    assert tofs[4.0] == pytest.approx(tofs[2.0] / 2, rel=1e-9)
    assert tofs[2.0] == pytest.approx(distance / 2.0, rel=1e-9)


def test_source_to_itself_has_zero_travel_time(isotropic_material):
    material = isotropic_material(vp=5.0)
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=6.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))
    source = np.array([[-1.0, -1.0]])
    g.add_points(sources=source, targets=source)
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    assert s.tfs[0, g.source_idx[0]] == 0.0
