"""
The isotropic parent zones of a SimplRectGrid in a domain that is not mirrored
about the backwall (the real domain of a backwall relay, or transducers on
both surfaces): set_up_graph()/update_edges() must work there and give the
same graph as calculate_graph() up to the chords that would cross the weld.

set_up_graph() used to split each zone's outline nodes into a top half and a
mirrored bottom half, which only the mirrored domain has; on any other grid
it raised an IndexError with an odd number of outline nodes and otherwise
built a wrong set of edges. _visible_zone_edges() joins two nodes of a zone
by a straight ray (one edge in each direction) if the segment does not cross
the outline.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver

THICKNESS = 8.
PIXEL = 1.
VP = 5.
OUTLINE = np.array([[-6., THICKNESS], [-3., 4.], [0., 0.], [3., 4.], [6., THICKNESS]])
PAD = 4
NX = 12 + 2*PAD + 4


def _relay_grid(isotropic_material, elements, seeds=8):
    material = isotropic_material(vp=VP)
    weld_mask = np.zeros((int(THICKNESS/PIXEL), NX))
    weld_mask[:, PAD + 2:NX - PAD - 2] = 1
    sources = np.c_[elements, np.full(len(elements), THICKNESS)]
    backwall_x = np.arange(-9.5, 9.6, 0.5)
    targets = np.c_[backwall_x, np.zeros(len(backwall_x))]
    g = grid.SimplRectGrid(NX, weld_mask.shape[0], 0., THICKNESS/2, PIXEL, seeds)
    g.assign_model(mode='orientations', property_map=np.zeros(weld_mask.shape))
    g.assign_materials(weld_mask, {0: material, 1: material})
    g.trim_to_weld(OUTLINE, mirror_domain=False)
    g.simplify_grid(left_add=PAD, right_add=PAD)
    g.add_points(points=np.r_[sources, targets],
                 sources=np.arange(len(sources)),
                 targets=np.arange(len(sources), len(sources) + len(targets)))
    return g


def _relay_times(g):
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx, with_points=True)
    times, _ = solver.combine_via_boundary(s.tfs[:, g.target_idx])
    return times


def test_set_up_graph_works_on_a_relay_grid_with_an_odd_number_of_outline_nodes(
        isotropic_material):
    g = _relay_grid(isotropic_material, np.arange(-8., 8.1, 1.))
    assert len(g.left_iso_chamfer) % 2 == 1 or len(g.right_iso_chamfer) % 2 == 1
    g.set_up_graph()
    g.update_edges()
    assert g.edges.nnz > 0


def test_fast_update_matches_the_rebuild_and_the_straight_line_on_a_relay_grid(
        isotropic_material):
    elements = np.arange(-8., 8.1, 1.)
    rebuilt = _relay_grid(isotropic_material, elements)
    rebuilt.calculate_graph()
    fast = _relay_grid(isotropic_material, elements)
    fast.set_up_graph()
    fast.update_edges()
    analytic = np.hypot(elements[:, None] - elements[None, :],
                        2*THICKNESS)/VP
    for name, g in (('rebuild', rebuilt), ('fast update', fast)):
        np.testing.assert_allclose(_relay_times(g), analytic, rtol=0.005,
                                   err_msg=name)
    np.testing.assert_allclose(_relay_times(fast), _relay_times(rebuilt),
                               rtol=0.005)


@pytest.mark.parametrize('outline_chords', [False, True])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_zone_edges_never_cross_the_outline_and_are_not_repeated(
        isotropic_material, side, outline_chords):
    g = _relay_grid(isotropic_material, np.arange(-8., 8.1, 1.))
    g.zone_outline_chords = outline_chords
    rows, cols, edges = g._visible_zone_edges(side)
    zone = getattr(g, side + '_iso_zone')
    chamfer = getattr(g, side + '_iso_chamfer')
    assert set(rows) <= set(zone) and set(cols) <= set(zone)
    # one edge per directed pair, and each one has its reverse at the same cost
    pairs = list(zip(rows.tolist(), cols.tolist()))
    assert len(pairs) == len(set(pairs))
    cost = dict(zip(pairs, edges))
    for (i, j), value in cost.items():
        assert cost[(j, i)] == pytest.approx(value)
    # no edge between two outline nodes unless asked for
    on_outline = set(chamfer.tolist())
    between_outline_nodes = [(i, j) for i, j in pairs
                             if i in on_outline and j in on_outline]
    assert bool(between_outline_nodes) == outline_chords
    # the segments stay on the zone's side of the outline (checked by
    # sampling, independently of the rule that chose them)
    outline_y = g.grid[chamfer, 1]
    outline_x = g.grid[chamfer, 0]
    order = np.argsort(outline_y)
    sign = 1 if side == 'left' else -1
    for i, j in pairs[::7]:
        a, b = g.grid[i], g.grid[j]
        t = np.linspace(0, 1, 60)
        x = a[0] + t*(b[0] - a[0])
        y = a[1] + t*(b[1] - a[1])
        inside_range = (y >= outline_y.min()) & (y <= outline_y.max())
        boundary = np.interp(y[inside_range], outline_y[order], outline_x[order])
        assert np.all(sign*(x[inside_range] - boundary) <= 1e-6), (i, j)


def _mirrored_times(isotropic_material, how):
    material = isotropic_material(vp=VP)
    weld_mask = np.zeros((int(THICKNESS/PIXEL), NX))
    weld_mask[:, PAD + 2:NX - PAD - 2] = 1
    mask = np.vstack((weld_mask[::-1], weld_mask))
    elements = np.arange(-8., 8.1, 2.)
    sources = np.c_[elements, np.full(len(elements), THICKNESS)]
    targets = np.c_[elements, np.full(len(elements), -THICKNESS)]
    g = grid.SimplRectGrid(NX, mask.shape[0], 0., 0., PIXEL, 8)
    g.assign_model(mode='orientations', property_map=np.zeros(mask.shape))
    g.assign_materials(mask, {0: material, 1: material})
    g.trim_to_weld(OUTLINE, mirror_domain=True)
    g.simplify_grid(left_add=PAD, right_add=PAD)
    g.add_points(points=np.r_[sources, targets],
                 sources=np.arange(len(sources)),
                 targets=np.arange(len(sources), 2*len(sources)))
    if how == 'rebuild':
        g.calculate_graph()
    else:
        if how != 'default':
            g.zone_edges = how
        g.set_up_graph()
        g.update_edges()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    return s.tfs[:, g.target_idx]


def test_the_general_rule_is_the_default_in_a_mirrored_domain_too(
        isotropic_material):
    """Also in the mirrored domain the fast update reproduces calculate_graph()
    with the default rule; the mirrored rule (zone_edges = 'mirrored') leaves
    out the chords between outline nodes and can be longer (by up to 0.06 us
    on the EDF weld)."""
    rebuilt = _mirrored_times(isotropic_material, 'rebuild')
    np.testing.assert_allclose(_mirrored_times(isotropic_material, 'default'),
                               rebuilt, rtol=1e-6)
    np.testing.assert_allclose(_mirrored_times(isotropic_material, 'visible'),
                               rebuilt, rtol=1e-6)
    legacy = _mirrored_times(isotropic_material, 'mirrored')
    assert np.all(np.isfinite(legacy))
    assert np.all(legacy >= rebuilt - 1e-9)      # never shorter than the exact chords


def test_zone_edges_option_is_checked(isotropic_material):
    with pytest.raises(ValueError, match='zone_edges'):
        _mirrored_times(isotropic_material, 'other')
    g = _relay_grid(isotropic_material, np.arange(-8., 8.1, 1.))
    g.zone_edges = 'mirrored'
    with pytest.raises(ValueError, match='mirrored domain'):
        g.set_up_graph()
