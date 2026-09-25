"""
The backwall relay on a SimplRectGrid trimmed to a weld outline
(trim_to_weld) must give the same times as the mirrored domain.

The isotropic parent metal of such a grid is not discretised: it is one
straight-ray zone on each side, made of the outline's nodes and of the
added points (transducers, backwall points) that lie in it. add_points()
used to assign an added point to a zone only if it lay beyond the horizontal
extent of the outline, so the backwall points in the parent metal *under* a
flank of the weld were left out and no reflection could use them: the relay
times were too long for every path that reflects there (0.4 us on a 14 us
path in the EDF example), while the mirrored domain, which has no such
points, was right. With an isotropic material everywhere the answer is known
in closed form: a straight line to the mirror image of the receiver.
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


def _times(isotropic_material, layout, elements):
    material = isotropic_material(vp=VP)
    weld_mask = np.zeros((int(THICKNESS/PIXEL), NX))
    weld_mask[:, PAD + 2:NX - PAD - 2] = 1
    sources = np.c_[elements, np.full(len(elements), THICKNESS)]
    if layout == 'relay':
        ny, cy = weld_mask.shape[0], THICKNESS/2
        mask = weld_mask
        backwall_x = np.arange(-9.5, 9.6, 0.5)
        targets = np.c_[backwall_x, np.zeros(len(backwall_x))]
    else:
        ny, cy = 2*weld_mask.shape[0], 0.
        mask = np.vstack((weld_mask[::-1], weld_mask))
        targets = np.c_[elements, np.full(len(elements), -THICKNESS)]
    g = grid.SimplRectGrid(NX, ny, 0., cy, PIXEL, 8)
    g.assign_model(mode='orientations', property_map=np.zeros(mask.shape))
    g.assign_materials(mask, {0: material, 1: material})
    g.trim_to_weld(OUTLINE, mirror_domain=(layout != 'relay'))
    g.simplify_grid(left_add=PAD, right_add=PAD)
    g.add_points(points=np.r_[sources, targets],
                 sources=np.arange(len(sources)),
                 targets=np.arange(len(sources), len(sources) + len(targets)))
    g.calculate_graph()
    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx, with_points=True)
    times = s.tfs[:, g.target_idx]
    if layout == 'relay':
        times, _ = solver.combine_via_boundary(times)
    return times


@pytest.mark.parametrize('layout', ['mirror', 'relay'])
def test_isotropic_reflection_times_are_the_straight_line_to_the_mirror_image(
        isotropic_material, layout):
    """Elements over the weld, over the parent on both sides: every pair's
    reflection point, including those under the flanks, is available."""
    elements = np.arange(-8., 8.1, 1.)
    times = _times(isotropic_material, layout, elements)
    analytic = np.hypot(elements[:, None] - elements[None, :],
                        2*THICKNESS)/VP
    np.testing.assert_allclose(times, analytic, rtol=0.005)


def test_relay_and_mirror_agree(isotropic_material):
    elements = np.arange(-8., 8.1, 1.)
    np.testing.assert_allclose(_times(isotropic_material, 'relay', elements),
                               _times(isotropic_material, 'mirror', elements),
                               rtol=0.005)


def test_backwall_points_under_the_flanks_join_the_parent_zones(
        isotropic_material):
    """The points of the backwall between the weld's top extent and its root
    are in the parent metal: they belong to the left/right straight-ray
    zones (as do the transducers beyond the outline), the points at the weld
    root and the transducers over the weld do not."""
    material = isotropic_material(vp=VP)
    mask = np.zeros((int(THICKNESS/PIXEL), NX))
    mask[:, PAD + 2:NX - PAD - 2] = 1
    g = grid.SimplRectGrid(NX, mask.shape[0], 0., THICKNESS/2, PIXEL, 8)
    g.assign_model(mode='orientations', property_map=np.zeros(mask.shape))
    g.assign_materials(mask, {0: material, 1: material})
    g.trim_to_weld(OUTLINE, mirror_domain=False)
    g.simplify_grid(left_add=PAD, right_add=PAD)
    top = np.array([[-8., THICKNESS], [0., THICKNESS], [8., THICKNESS]])
    backwall = np.array([[-7.5, 0.], [-4.5, 0.], [-1.5, 0.], [1.5, 0.],
                         [4.5, 0.], [7.5, 0.]])
    g.add_points(points=np.r_[top, backwall], sources=np.arange(3),
                 targets=np.arange(3, 9))
    zone = set(g.left_iso_trans) | set(g.right_iso_trans)
    left, right = set(g.left_iso_trans), set(g.right_iso_trans)
    sources, targets = g.source_idx, g.target_idx
    assert sources[0] in left and sources[2] in right      # beyond the outline
    assert sources[1] not in zone                          # over the weld
    assert set(targets[:3]) <= left                        # the whole backwall
    assert set(targets[3:]) <= right                       # is parent metal
