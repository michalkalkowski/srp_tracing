"""
Tests for pulse-echo/reflection via combine_via_boundary(), the replacement
for the old mirrored-domain approach (Solver.fold_paths): instead of
doubling the whole domain to get reflected paths "for free" from a single
shortest_path solve, place a grid of points along the backwall (or
frontwall) as ordinary added nodes in the *original* domain, solve once
from the transducers to everything (including those boundary points), and
combine each pair's one-way times to every boundary point via
combine_via_boundary(). Fermat's principle (the realised path minimises
travel time) means the combination is just a min over the shared boundary
axis -- no second solve, and no assumption that the boundary is flat or
that the domain is mirror-symmetric, unlike the old approach.

Uses a single-pixel RectGrid so calculate_graph() connects every node to
every other node directly with cost = euclidean_distance / v (see
test_single_cell_analytic.py for why): this makes the one-way times exactly
analytic, so the combined pulse-echo times can be checked against a
closed-form answer, not just self-consistency.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


def _single_cell_grid(isotropic_material, vp, pixel_size, transducers, backwall_points):
    material = isotropic_material(vp=vp)
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=pixel_size, no_seeds=4)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))
    g.add_points(sources=transducers, targets=backwall_points)
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})
    g.calculate_graph()
    return g


def test_pulse_echo_normal_incidence_matches_analytic(isotropic_material):
    """A transducer reflecting straight back off a point directly below it:
    pulse-echo time is exactly 2 * height / v."""
    vp = 4.0
    transducer = np.array([[-3.0, 5.0]])
    backwall = np.array([[-3.0, -5.0], [0.0, -5.0], [3.0, -5.0]])

    g = _single_cell_grid(isotropic_material, vp, pixel_size=20.0,
                          transducers=transducer, backwall_points=backwall)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    times_to_backwall = s.tfs[:, g.target_idx]
    relay_times, via_index = solver.combine_via_boundary(times_to_backwall)

    height = transducer[0, 1] - backwall[0, 1]
    assert relay_times[0, 0] == pytest.approx(2 * height / vp, rel=1e-9)
    assert g.target_idx[via_index[0, 0]] == g.target_idx[0]  # the point directly below


def test_pulse_echo_oblique_matches_mirror_image_analytic(isotropic_material):
    """Two different transducers: the fastest via-backwall path matches the
    classic mirror-image construction exactly (dist(A, mirror(B)) / v),
    since the analytically optimal reflection point is included as one of
    the discretized backwall nodes."""
    vp = 4.0
    a = np.array([-3.0, 5.0])
    b = np.array([3.0, 5.0])
    y_backwall = -5.0
    transducers = np.array([a, b])
    # includes x=0, the analytically optimal reflection point for this
    # symmetric pair, and x=-3/x=3 for the normal-incidence self-terms
    backwall = np.column_stack((np.linspace(-8, 8, 17),
                               np.full(17, y_backwall)))

    g = _single_cell_grid(isotropic_material, vp, pixel_size=20.0,
                          transducers=transducers, backwall_points=backwall)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    times_to_backwall = s.tfs[:, g.target_idx]
    relay_times, via_index = solver.combine_via_boundary(times_to_backwall)

    b_mirror = np.array([b[0], 2*y_backwall - b[1]])
    expected = np.linalg.norm(a - b_mirror) / vp
    assert relay_times[0, 1] == pytest.approx(expected, rel=1e-9)
    assert relay_times[0, 1] == pytest.approx(relay_times[1, 0], rel=1e-12)  # symmetric
    assert g.grid[g.target_idx[via_index[0, 1]]][0] == pytest.approx(0.0, abs=1e-9)


def test_combine_via_boundary_matches_old_mirrored_domain_approach(isotropic_material):
    """Cross-validation against the approach this replaces: for a flat
    backwall and a domain that's actually mirror-symmetric (the case the
    old doubled-domain trick handles), combine_via_boundary on the
    original domain must agree with a plain shortest-path solve on the
    doubled domain."""
    vp = 3.7
    a = np.array([-4.0, 6.0])
    b = np.array([2.0, 6.0])
    transducers = np.array([a, b])
    backwall = np.column_stack((np.linspace(-9, 9, 37), np.full(37, 0.0)))

    # New approach: single (non-doubled) domain, relay via backwall nodes.
    g = _single_cell_grid(isotropic_material, vp, pixel_size=24.0,
                          transducers=transducers, backwall_points=backwall)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    relay_times, _ = solver.combine_via_boundary(s.tfs[:, g.target_idx])

    # Old approach: doubled domain (mirrored about y=0), sources on top,
    # targets at the mirror image of the real receivers, single solve.
    # no_seeds=48 (spacing 0.5, same as `backwall` above, and aligned to
    # the same x-coordinates) so both approaches choose from the same
    # candidate reflection points -- the reflection resolution in the old
    # approach is otherwise coupled to the grid's no_seeds, not
    # independently controllable the way backwall's density is here, so a
    # coarser no_seeds would just be comparing two different
    # discretizations rather than validating the method itself.
    g_mirror = grid.RectGrid(nx=1, ny=2, cx=0.0, cy=0.0, pixel_size=24.0, no_seeds=48)
    g_mirror.assign_model(mode="orientations", property_map=np.zeros((2, 1)))
    mirrored_targets = np.column_stack((transducers[:, 0], -transducers[:, 1]))
    g_mirror.add_points(sources=transducers, targets=mirrored_targets)
    g_mirror.assign_materials(np.zeros((2, 1), dtype=int),
                              {0: isotropic_material(vp=vp)})
    g_mirror.calculate_graph()
    s_mirror = solver.Solver(g_mirror)
    s_mirror.solve(source_indices=g_mirror.source_idx)
    old_times = s_mirror.tfs[:, g_mirror.target_idx]

    np.testing.assert_allclose(relay_times, old_times, rtol=1e-6)


def test_calculate_relay_ray_path_reconstructs_reflection(isotropic_material):
    vp = 4.0
    a = np.array([-3.0, 5.0])
    b = np.array([3.0, 5.0])
    transducers = np.array([a, b])
    backwall = np.column_stack((np.linspace(-8, 8, 17), np.full(17, -5.0)))

    g = _single_cell_grid(isotropic_material, vp, pixel_size=20.0,
                          transducers=transducers, backwall_points=backwall)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx, with_points=True)

    times_to_backwall = s.tfs[:, g.target_idx]
    relay_times, via_index = solver.combine_via_boundary(times_to_backwall)

    source_node = g.source_idx[0]
    target_node = g.source_idx[1]
    via_node = g.target_idx[via_index[0, 1]]
    path = s.calculate_relay_ray_path(source_node, target_node, via_node)

    assert path[0] == source_node
    assert path[-1] == target_node
    assert via_node in path
    # single-pixel domain: each leg is a direct edge, so the full reflected
    # path is exactly [source, via, target]
    assert path == [source_node, via_node, target_node]

    path_length = sum(
        np.linalg.norm(g.grid[path[k+1], :2] - g.grid[path[k], :2])
        for k in range(len(path) - 1))
    assert path_length / vp == pytest.approx(relay_times[0, 1], rel=1e-9)


def test_combine_via_boundary_transmission_mode(isotropic_material):
    """times_b != times_a: a different receiver array on the other side of
    a relay boundary (e.g. transmission through a frontwall)."""
    vp = 5.0
    transmitters = np.array([[-3.0, 5.0]])
    receivers = np.array([[3.0, -5.0]])
    frontwall = np.column_stack((np.linspace(-8, 8, 33), np.full(33, 0.0)))

    material = isotropic_material(vp=vp)
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=0.0, pixel_size=20.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))
    g.add_points(sources=np.vstack([transmitters, receivers]), targets=frontwall)
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    times = s.tfs[:, g.target_idx]
    times_a = times[:1]  # transmitter row
    times_b = times[1:]  # receiver row

    relay_times, via_index = solver.combine_via_boundary(times_a, times_b)
    assert relay_times.shape == (1, 1)

    # straight line transmitter -> receiver crosses y=0 at the midpoint
    # in x (since both are equidistant from y=0 here); confirm the relay
    # time equals the direct (unobstructed) straight-line time, since a
    # single-medium "transmission" with no refraction is just a straight
    # line that happens to pass through a frontwall node.
    direct = np.linalg.norm(transmitters[0] - receivers[0]) / vp
    assert relay_times[0, 0] == pytest.approx(direct, rel=1e-6)
