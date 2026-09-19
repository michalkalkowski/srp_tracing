"""
Phase 4 of the WeldGrid unification plan: proves the full wedge/immersion
pulse-echo chain (wedge -> frontwall -> smooth-chamfer solid -> backwall ->
solid -> frontwall -> wedge) works end-to-end through WeldGrid, using the
same chained combine_via_boundary composition already proven correct in
test_frontwall_relay.py::test_chained_combine_via_boundary_matches_brute_force_double_relay.
A rexolite wedge is the same shape as a water immersion setup -- a
homogeneous, unobstructed isotropic medium between transducer and
frontwall -- so it uses solver.straight_line_times exactly like water does,
just with rexolite's wavespeed instead.

No finite-element reference data exists yet for a wedge case (unlike the
Ogilvy/MINA validation tests), so these are self-consistency and
equivalence checks rather than FE-validated ones. What Phase 4 actually
needs to prove is that WeldGrid doesn't disturb the already-proven relay
chain, not new physics, so the primary check is equivalence against the
plain SimplRectGrid WeldGrid(chamfer='smooth') wraps.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver

REXOLITE_V = 2.34  # mm/us, typical bulk longitudinal wavespeed


def _build_smooth_solid(cls, isotropic_material):
    """Same small chamfer geometry as test_regression_simpl_rect_grid.py's
    _build_small_weld_model (already pinned/verified there and in
    test_weld_grid_unified_api.py), with the 4 source/target points split
    into a "frontwall" pair (nearer the outer surface, y=5) and a
    "backwall" pair (nearer the root, y=2), so add_points' sources/targets
    map directly onto the two relay boundaries this test chains through."""
    a, b, c = 10.0, 2.0, 14.0
    dx = 2.0
    no_seeds = 4
    nx, ny = 8, 6
    cx, cy = 0.0, a / 2

    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, nx // 2 - 1: nx // 2 + 1] = 1
    property_map = np.zeros((ny, nx))
    parent = isotropic_material(vp=5.9)
    weld = isotropic_material(vp=3.24)

    if cls is grid.WeldGrid:
        g = cls(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx, no_seeds=no_seeds,
               chamfer='smooth')
    else:
        g = cls(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx, no_seeds=no_seeds)
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: parent, 1: weld})
    g.trim_to_chamfer(a, b, c)
    g.simplify_grid()

    frontwall = np.array([[-5.0, 5.0], [5.0, 5.0]])
    backwall = np.array([[-5.0, 2.0], [5.0, 2.0]])
    if cls is grid.WeldGrid:
        g.add_points(sources=frontwall, targets=backwall)
    else:
        points = np.concatenate((frontwall, backwall), axis=0)
        s_ix = np.arange(len(frontwall))
        t_ix = np.arange(len(frontwall), len(frontwall) + len(backwall))
        g.add_points(points=points, sources=s_ix, targets=t_ix)
    g.calculate_graph()
    return g, frontwall, backwall


def test_weld_grid_smooth_matches_simpl_rect_grid_for_relay_construction(isotropic_material):
    """WeldGrid(chamfer='smooth') must reproduce the exact same solid-leg
    travel times as the SimplRectGrid it wraps, for the frontwall/backwall
    relay construction the wedge/pulse-echo chain below needs -- this is
    what Phase 4 actually needs to demonstrate: the unified API doesn't
    disturb the relay chain, not new physics."""
    g_weld, _, _ = _build_smooth_solid(grid.WeldGrid, isotropic_material)
    g_simpl, _, _ = _build_smooth_solid(grid.SimplRectGrid, isotropic_material)

    s_weld = solver.Solver(g_weld)
    s_weld.solve(source_indices=g_weld.source_idx)
    s_simpl = solver.Solver(g_simpl)
    s_simpl.solve(source_indices=g_simpl.source_idx)

    np.testing.assert_array_equal(
        s_weld.tfs[:, g_weld.target_idx], s_simpl.tfs[:, g_simpl.target_idx])


def test_wedge_pulse_echo_chain_reciprocity(isotropic_material):
    """Two wedge-coupled transducers at different positions: the full
    wedge -> frontwall -> solid -> backwall -> solid -> frontwall -> wedge
    pulse-echo time between them must be symmetric -- a real physical
    invariant (reciprocity), not a tautology of the composition."""
    g, frontwall, _ = _build_smooth_solid(grid.WeldGrid, isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    solid_leg = s.tfs[:, g.target_idx]  # (n_frontwall, n_backwall)

    transducers = np.array([[-6.0, 12.0], [4.0, 13.0]])
    wedge_leg = solver.straight_line_times(transducers, frontwall, REXOLITE_V)

    to_backwall, _ = solver.combine_via_boundary(wedge_leg, solid_leg.T)
    pulse_echo, _ = solver.combine_via_boundary(to_backwall)

    assert pulse_echo.shape == (2, 2)
    assert np.all(np.isfinite(pulse_echo))
    assert pulse_echo[0, 1] == pytest.approx(pulse_echo[1, 0], rel=1e-12)
    assert pulse_echo[0, 0] > 0 and pulse_echo[1, 1] > 0


def test_wedge_pulse_echo_chain_matches_brute_force(isotropic_material):
    """The two-call chaining pattern must exactly equal a brute-force
    search over every (frontwall-in, backwall, frontwall-out) triple, fed
    by real WeldGrid(chamfer='smooth')-derived solid-leg times rather than
    the arbitrary random array used in test_frontwall_relay.py's equivalent
    check -- confirms the composition is correct for the actual
    construction this chain uses, not just in the abstract."""
    g, frontwall, _ = _build_smooth_solid(grid.WeldGrid, isotropic_material)
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    solid_leg = s.tfs[:, g.target_idx]

    transducers = np.array([[-6.0, 12.0], [4.0, 13.0]])
    wedge_leg = solver.straight_line_times(transducers, frontwall, REXOLITE_V)

    to_backwall, _ = solver.combine_via_boundary(wedge_leg, solid_leg.T)
    pulse_echo, _ = solver.combine_via_boundary(to_backwall)

    n_a, n_fw = wedge_leg.shape
    n_bw = solid_leg.shape[1]
    brute = np.full((n_a, n_a), np.inf)
    for i in range(n_a):
        for j in range(n_a):
            best = np.inf
            for p_in in range(n_fw):
                for q in range(n_bw):
                    for p_out in range(n_fw):
                        total = (wedge_leg[i, p_in] + solid_leg[p_in, q]
                                + solid_leg[p_out, q] + wedge_leg[j, p_out])
                        best = min(best, total)
            brute[i, j] = best

    np.testing.assert_allclose(pulse_echo, brute, rtol=1e-9)
