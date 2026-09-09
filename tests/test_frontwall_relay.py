"""
Tests for the water-coupled/immersion extension of the backwall-relay
technique in tests/test_backwall_relay.py: transducers sit in water above a
(possibly complicated) frontwall rather than directly on the solid. The
water leg is a single known, unobstructed, isotropic medium, so it doesn't
need graph tracing at all -- solver.straight_line_times() gives it directly
-- and it combines with the solid-side graph times via the same
solver.combine_via_boundary() used for the backwall, through the frontwall
as the shared via-boundary.

The key result demonstrated here: combine_via_boundary() composes. A full
pulse-echo path (water -> frontwall -> solid -> backwall -> solid ->
frontwall -> water) is just two chained calls -- relay to the backwall via
the frontwall, then relay transducer-to-transducer via the backwall -- with
no new machinery needed for a second boundary.
"""
import numpy as np
import pytest
import scipy.optimize

from srp_tracing import grid, solver


def test_straight_line_times_basic():
    points_a = np.array([[0.0, 0.0], [1.0, 0.0]])
    points_b = np.array([[0.0, 3.0], [4.0, 3.0]])
    times = solver.straight_line_times(points_a, points_b, speed=2.0)
    assert times.shape == (2, 2)
    assert times[0, 0] == pytest.approx(3.0 / 2.0)
    assert times[0, 1] == pytest.approx(5.0 / 2.0)  # 3-4-5 triangle


def test_single_interface_transmission_matches_fermat_minimum(isotropic_material):
    """A transducer in water refracting through a flat frontwall to a point
    in the solid: the fastest via-frontwall time must match the true
    (continuous) Fermat-principle minimum -- the refraction/Snell's-law
    answer -- to within the frontwall discretization spacing."""
    v_water = 1.48
    v_solid = 5.9
    transducer = np.array([-6.0, 8.0])   # in water, y > 0
    solid_point = np.array([4.0, -7.0])  # in the solid, y < 0
    frontwall_x = np.arange(-20.0, 20.0, 0.05)
    frontwall = np.column_stack((frontwall_x, np.zeros_like(frontwall_x)))

    # single pixel must be large enough to contain every point used (both
    # frontwall's x range and solid_point) within its bounding box, or
    # calculate_graph's per-pixel neighbour search silently drops whatever
    # falls outside it
    material = isotropic_material(vp=v_solid)
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=-15.0, pixel_size=50.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))
    g.add_points(sources=frontwall, targets=np.array([solid_point]))
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})
    g.calculate_graph()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    solid_leg = s.tfs[:, g.target_idx]  # (n_frontwall, 1): frontwall -> solid_point

    water_leg = solver.straight_line_times(transducer[None, :], frontwall, v_water)
    relay_times, via_index = solver.combine_via_boundary(water_leg, solid_leg.T)
    discretized_time = relay_times[0, 0]
    discretized_x = frontwall[via_index[0, 0], 0]

    def total_time(x_cross):
        water_dist = np.hypot(transducer[0] - x_cross, transducer[1] - 0.0)
        solid_dist = np.hypot(solid_point[0] - x_cross, solid_point[1] - 0.0)
        return water_dist / v_water + solid_dist / v_solid

    fermat = scipy.optimize.minimize_scalar(total_time, bounds=(-20, 20), method="bounded")
    assert discretized_time == pytest.approx(fermat.fun, abs=1e-3)
    assert discretized_x == pytest.approx(fermat.x, abs=0.1)  # within ~2 grid spacings


def test_double_relay_pulse_echo_normal_incidence_matches_analytic(isotropic_material):
    """Full water -> frontwall -> solid -> backwall -> solid -> frontwall
    -> water pulse-echo, composed from two chained combine_via_boundary
    calls, for a transducer directly above matching frontwall/backwall
    points: exact closed-form normal-incidence time."""
    v_water = 1.48
    v_solid = 5.9
    water_height = 5.0
    solid_height = 10.0

    transducers = np.array([[-3.0, water_height], [3.0, water_height]])
    frontwall = np.column_stack((np.arange(-10.0, 10.0, 0.5),
                                np.zeros(40)))
    backwall = np.column_stack((np.arange(-10.0, 10.0, 0.5),
                               np.full(40, -solid_height)))

    # single pixel must be large enough to contain the full frontwall/
    # backwall x range within its bounding box (see the note in
    # test_single_interface_transmission_matches_fermat_minimum)
    material = isotropic_material(vp=v_solid)
    g = grid.RectGrid(nx=1, ny=1, cx=0.0, cy=-solid_height/2,
                      pixel_size=40.0, no_seeds=4)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 1)))
    g.add_points(sources=frontwall, targets=backwall)
    g.assign_materials(np.zeros((1, 1), dtype=int), {0: material})
    g.calculate_graph()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    solid_leg = s.tfs[:, g.target_idx]  # (n_frontwall, n_backwall)

    water_leg = solver.straight_line_times(transducers, frontwall, v_water)
    to_backwall, _ = solver.combine_via_boundary(water_leg, solid_leg.T)
    pulse_echo, via_backwall = solver.combine_via_boundary(to_backwall)

    expected_self = 2*water_height/v_water + 2*solid_height/v_solid
    assert pulse_echo[0, 0] == pytest.approx(expected_self, rel=1e-6)
    assert pulse_echo[1, 1] == pytest.approx(expected_self, rel=1e-6)
    assert pulse_echo[0, 1] == pytest.approx(pulse_echo[1, 0], rel=1e-9)  # reciprocity


def test_chained_combine_via_boundary_matches_brute_force_double_relay():
    """The two-call chaining pattern above must exactly equal a brute-force
    search over every (frontwall-in, backwall, frontwall-out) triple --
    confirms the composition itself is mathematically correct, independent
    of any particular physical setup."""
    rng = np.random.default_rng(0)
    n_a, n_fw, n_bw = 3, 5, 4
    # arbitrary positive "times" standing in for water_leg/solid_leg
    water_leg = rng.uniform(1, 10, size=(n_a, n_fw))
    solid_leg = rng.uniform(1, 10, size=(n_fw, n_bw))

    to_backwall, _ = solver.combine_via_boundary(water_leg, solid_leg.T)
    pulse_echo, _ = solver.combine_via_boundary(to_backwall)

    brute = np.full((n_a, n_a), np.inf)
    for i in range(n_a):
        for j in range(n_a):
            best = np.inf
            for p_in in range(n_fw):
                for q in range(n_bw):
                    for p_out in range(n_fw):
                        total = (water_leg[i, p_in] + solid_leg[p_in, q]
                                + solid_leg[p_out, q] + water_leg[j, p_out])
                        best = min(best, total)
            brute[i, j] = best

    np.testing.assert_allclose(pulse_echo, brute, rtol=1e-9)
