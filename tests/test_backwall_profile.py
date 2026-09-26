"""
A backwall that is not flat: the relay (solver.combine_via_boundary) with the
points of a backwall profile as the targets, and the domain trimmed to the
plate above it (RectGrid.trim_to_backwall, SimplRectGrid.trim_to_weld(backwall=...)).

With an isotropic material everywhere the time of a pair is known exactly:
the shortest straight path source -> point of the profile -> receiver, found
here by brute force over the points of the profile (every straight leg from
the top surface to the profile stays in the plate when the profile is a
shallow recess). The profile is a machined recess like a manufactured weld's
backwall: a floor, a raised flat and sloping shoulders.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver

VP = 5.


def recess_profile(flat=1.5, half_flat=4., half_base=8., half_width=30., step=0.25):
    """Floor at 0, raised to `flat` for |x| < half_flat, linear shoulders."""
    x = np.arange(-half_width, half_width + step/2, step)
    y = np.clip((half_base - np.abs(x))/(half_base - half_flat), 0., 1.)*flat
    return np.c_[x, y]


def analytic_times(sources, receivers, profile):
    """min over the points of the profile of (|s - p| + |p - r|)/vp: [source, receiver]."""
    def leg(points):
        return np.hypot(points[:, None, 0] - profile[None, :, 0],
                        points[:, None, 1] - profile[None, :, 1])
    to = leg(sources)
    back = leg(receivers)
    return np.min(to[:, None, :] + back[None, :, :], axis=2)/VP


def relay_times(g, sources, profile):
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    times, _ = solver.combine_via_boundary(s.tfs[:, g.target_idx])
    return times


def test_rect_grid_relay_off_a_recessed_backwall(isotropic_material):
    material = isotropic_material(vp=VP)
    profile = recess_profile()
    x_elements = np.arange(-9., 9.1, 1.)
    sources = np.c_[x_elements, np.full(len(x_elements), 24.)]
    g = grid.RectGrid(nx=30, ny=12, cx=0., cy=12., pixel_size=2., no_seeds=8)
    g.trim_to_backwall(profile)
    # nothing is left below the profile
    below = g.grid_1[:, 1] < np.interp(g.grid_1[:, 0], profile[:, 0], profile[:, 1]) - 1e-9
    assert not below.any()
    g.assign_model(mode='orientations', property_map=np.zeros((12, 30)))
    g.add_points(sources=sources, targets=profile)
    g.assign_materials(np.zeros((12, 30), dtype=int), {0: material})
    g.calculate_graph()

    times = relay_times(g, sources, profile)
    exact = analytic_times(sources, sources, profile)
    np.testing.assert_allclose(times, exact, rtol=0.006)
    # the recess matters: a flat backwall would give times up to 6% too long
    flat = analytic_times(sources, sources, recess_profile(flat=0.))
    assert np.max(np.abs(flat - exact)/exact) > 0.03


def _weld_geometry(isotropic_material, backwall):
    material = isotropic_material(vp=VP)
    nx, ny, pixel = 32, 9, 1.
    outline = np.array([[-6., 9.], [-3., 5.], [0., 1.], [3., 5.], [6., 9.]])
    weld_mask = np.zeros((ny, nx))
    weld_mask[:, 10:22] = 1
    x_elements = np.arange(-8., 8.1, 1.)
    sources = np.c_[x_elements, np.full(len(x_elements), 9.)]
    g = grid.SimplRectGrid(nx, ny, 0., ny*pixel/2, pixel, 8)
    g.assign_model(mode='orientations', property_map=np.zeros((ny, nx)))
    g.assign_materials(weld_mask, {0: material, 1: material})
    g.trim_to_weld(outline, mirror_domain=False, backwall=backwall)
    g.simplify_grid(left_add=8, right_add=8)
    return g, sources, outline


def test_weld_grid_relay_off_a_recessed_backwall(isotropic_material):
    """The parent metal beside the weld is a straight-ray zone; its chords and
    the backwall points in it (the shoulders and the floor) follow the
    profile, the weld root sits on the raised flat."""
    profile = recess_profile(flat=1., half_flat=3., half_base=7., half_width=15.5, step=0.25)
    g, sources, _ = _weld_geometry(isotropic_material, profile)
    g.add_points(points=np.r_[sources, profile], sources=np.arange(len(sources)),
                 targets=np.arange(len(sources), len(sources) + len(profile)))
    g.calculate_graph()
    times = relay_times(g, sources, profile)
    exact = analytic_times(sources, sources, profile)
    np.testing.assert_allclose(times, exact, rtol=0.008)
    # the fast update agrees with the rebuild on this grid too
    fast, sources2, _ = _weld_geometry(isotropic_material, profile)
    fast.add_points(points=np.r_[sources2, profile], sources=np.arange(len(sources2)),
                    targets=np.arange(len(sources2), len(sources2) + len(profile)))
    fast.set_up_graph()
    fast.update_edges()
    np.testing.assert_allclose(relay_times(fast, sources2, profile), times, rtol=1e-6)


def test_zone_chords_do_not_dip_below_the_backwall(isotropic_material):
    """No edge between two points of a parent zone runs under the profile."""
    profile = recess_profile(flat=1., half_flat=3., half_base=7., half_width=15.5, step=0.25)
    g, sources, _ = _weld_geometry(isotropic_material, profile)
    g.add_points(points=np.r_[sources, profile], sources=np.arange(len(sources)),
                 targets=np.arange(len(sources), len(sources) + len(profile)))
    for side in ('left', 'right'):
        rows, cols, _ = g._visible_zone_edges(side)
        a, b = g.grid[rows], g.grid[cols]
        t = np.linspace(0, 1, 40)[None, :]
        x = a[:, 0:1] + t*(b[:, 0:1] - a[:, 0:1])
        y = a[:, 1:2] + t*(b[:, 1:2] - a[:, 1:2])
        floor = np.interp(x, profile[:, 0], profile[:, 1])
        assert np.all(y >= floor - 1e-6)


def test_profile_and_geometry_are_checked(isotropic_material):
    profile = recess_profile(flat=1., half_flat=3., half_base=7., half_width=15.5)
    with pytest.raises(ValueError, match='strictly'):
        grid.prepare_backwall_profile(np.array([[0., 0.], [0., 1.], [1., 1.]]))
    with pytest.raises(ValueError, match='n >= 2'):
        grid.prepare_backwall_profile(np.array([[0., 0.]]))
    # a mirrored domain has no such backwall
    g = grid.SimplRectGrid(32, 18, 0., 0., 1., 4)
    with pytest.raises(ValueError, match='mirror_domain'):
        g.trim_to_weld(np.array([[-6., 9.], [0., 1.], [6., 9.]]), mirror_domain=True,
                       backwall=profile)
    # the weld outline must not go below the profile
    g = grid.SimplRectGrid(32, 9, 0., 4.5, 1., 4)
    with pytest.raises(ValueError, match='outline'):
        g.trim_to_weld(np.array([[-6., 9.], [0., 0.], [6., 9.]]), backwall=profile)
