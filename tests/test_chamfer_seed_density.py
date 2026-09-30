"""
Tests for SimplRectGrid.trim_to_weld's chamfer seed spacing/density.

Chamfer seeds used to be spaced node_spacing/seeds_vs_node_sp apart, with
seeds_vs_node_sp defaulting to 0.25 -- 4x sparser than the standard
(pixel-boundary) seeds, which are node_spacing = pixel_size/no_seeds
apart. Seeds were also placed evenly in x across the *whole* outline
rather than evenly along its arc length, so a multi-slope outline (more
than one bevel angle -- a multi-level/triple-V weld, or any real fitted
profile) got seeded unevenly regardless of the density factor: too many
seeds on shallow segments, too few on steep ones. Both are fixed: the
default now matches the standard spacing, and seeds are placed evenly
along each outline segment's own length. See trim_to_weld's docstring.
"""
import numpy as np
import pytest

from srp_tracing import grid

# A two-level (different bevel angles per side) outline, deliberately not
# a single straight V: left side (-10, 10) -> (-6, 4) is steep (length
# ~7.21), (-6, 4) -> (-1, 0) is shallower (length ~6.40) -- evenly-spaced-
# in-x seeding puts more seeds per unit length on the shallow segment than
# the steep one; evenly-spaced-in-arc-length must not.
OUTLINE = np.array([[-10., 10.], [-6., 4.], [-1., 0.],
                    [1., 0.], [6., 4.], [10., 10.]])
PIXEL = 2.0
NO_SEEDS = 4
NODE_SPACING = PIXEL/NO_SEEDS


def _trimmed_grid(isotropic_material, nx=14, ny=6, seeds_vs_node_sp=1.0):
    parent = isotropic_material(vp=5.0)
    weld = isotropic_material(vp=3.0)
    material_map = np.zeros((ny, nx), dtype=int)
    # a central band, present in every row, so simplify_grid's per-row NaN
    # fill (it expects at least one weld pixel per row) has something to
    # work from -- same pattern as test_regression_simpl_rect_grid.py
    material_map[:, nx//2 - 2:nx//2 + 2] = 1
    property_map = np.zeros((ny, nx))
    g = grid.SimplRectGrid(nx=nx, ny=ny, cx=0.0, cy=5.0, pixel_size=PIXEL,
                           no_seeds=NO_SEEDS)
    g.assign_model(mode='orientations', property_map=property_map)
    g.assign_materials(material_map, {0: parent, 1: weld})
    g.trim_to_weld(OUTLINE, mirror_domain=False, seeds_vs_node_sp=seeds_vs_node_sp)
    g.simplify_grid()
    # add_points(points=None) is what calculate_graph's real callers rely on
    # to set self.grid = self.grid_1, but it crashes (an unrelated, pre-
    # existing bug: added_points_idx is only defined in the points-is-not-
    # None branch, yet referenced unconditionally afterwards) -- do the same
    # assignment directly instead, since no extra points are needed here.
    g.grid = g.grid_1
    return g


def _chamfer_seed_spacings(g):
    """Consecutive-seed distances along each side's chamfer, in the order
    trim_to_weld built them (left_iso_zone/right_iso_zone, arc-length
    order)."""
    spacings = []
    for zone in (g.left_iso_zone, g.right_iso_zone):
        pts = g.grid[zone]
        spacings.append(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return np.concatenate(spacings)


def test_default_chamfer_seed_spacing_matches_standard_seed_spacing(isotropic_material):
    g = _trimmed_grid(isotropic_material)
    spacings = _chamfer_seed_spacings(g)
    # int() truncation of seeds_per_chamfer and a shared linspace endpoint
    # keep this from being exact; it must still be close, not off by the
    # old ~4x factor
    np.testing.assert_allclose(spacings, NODE_SPACING, rtol=0.15)


def test_chamfer_seed_spacing_is_uniform_across_segments_of_different_slope(
        isotropic_material):
    """The steep and shallow segments of OUTLINE must end up with the same
    seed spacing as each other (arc-length-uniform), not the same seed
    *count* (x-uniform, the old bug): x-uniform seeding would visibly skew
    the two segments' spacings relative to each other despite their
    similar lengths, because their y-extents (and so their length-per-
    unit-x) differ a lot."""
    g = _trimmed_grid(isotropic_material)
    spacings = _chamfer_seed_spacings(g)
    assert spacings.std() < 0.1*NODE_SPACING


def test_sparser_chamfer_seeding_is_still_available(isotropic_material):
    g = _trimmed_grid(isotropic_material, seeds_vs_node_sp=0.25)
    spacings = _chamfer_seed_spacings(g)
    np.testing.assert_allclose(spacings, NODE_SPACING/0.25, rtol=0.15)


def test_every_chamfer_seed_is_captured_by_a_weld_pixel(isotropic_material):
    """Reproduces calculate_graph()'s own per-pixel point-capture test
    (query_ball_point + a tight bounding-box filter) for every kept pixel,
    and checks every chamfer seed -- now much denser, and off the regular
    pixel-boundary lattice the capture radius/tolerance were tuned for --
    is captured by at least one of them. A seed missed by every pixel
    would still get edges from the isotropic zone (_visible_zone_edges),
    but would be cut off from the interior weld graph at exactly that
    point on the boundary."""
    g = _trimmed_grid(isotropic_material)
    chamfer_seeds = np.concatenate((g.left_iso_zone, g.right_iso_zone))

    captured = set()
    tree = grid.cKDTree(g.grid)
    for pixel in range(len(g.image_grid_trim)):
        centre = g.image_grid_trim[pixel]
        candidates = np.array(tree.query_ball_point(
            centre, grid.PIXEL_SEARCH_RADIUS_FACTOR*g.pixel_size*2**0.5), dtype=int)
        take = (np.abs(g.grid[candidates] - centre)
               <= g.pixel_size/2*grid.PIXEL_BOUNDARY_TOL_FACTOR).all(axis=1)
        captured.update(candidates[take].tolist())

    missing = sorted(set(chamfer_seeds.tolist()) - captured)
    assert not missing, (
        f"{len(missing)}/{len(chamfer_seeds)} chamfer seeds are not inside "
        f"any kept pixel's capture box: node indices {missing[:10]}")
