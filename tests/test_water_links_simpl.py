"""
SimplRectGrid counterpart of test_water_links.py: water_links used to be
supported only by RectGrid.calculate_graph, leaving SimplRectGrid (the
smooth-chamfer grid) with no way to model an immersion/water-coupled setup
at all. Both classes now share the same _ModeDependentMaterial._append_
water_links helper, so this pins the same correctness property -- distinct
group sizes, both edge directions -- against SimplRectGrid.
"""
import numpy as np
import pytest

from srp_tracing import grid


def _small_simpl_rect_grid(isotropic_material):
    a, b, c = 10.0, 2.0, 14.0
    dx = 2.0
    no_seeds = 4
    nx, ny = 8, 6
    cx, cy = 0.0, a / 2

    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, nx // 2 - 1: nx // 2 + 1] = 1
    property_map = np.zeros((ny, nx))
    material = isotropic_material(vp=5.9)

    g = grid.SimplRectGrid(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx, no_seeds=no_seeds)
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: material, 1: material})
    g.trim_to_chamfer(a, b, c)
    g.simplify_grid()
    return g, material


def test_water_links_gives_correct_distance_for_every_pair(isotropic_material):
    g, _ = _small_simpl_rect_grid(isotropic_material)

    # distinct group sizes, same as test_water_links.py's equivalent case
    group_a = np.array([[-5.0, 2.0], [-5.0, 5.0], [-5.0, 8.0]])
    group_b = np.array([[5.0, 2.0], [5.0, 5.0]])
    points = np.concatenate((group_a, group_b), axis=0)
    s_ix = np.arange(len(group_a))
    t_ix = np.arange(len(group_a), len(group_a) + len(group_b))
    g.add_points(points=points, sources=s_ix, targets=t_ix)

    c0 = 1.48
    link0, link1 = g.source_idx, g.target_idx
    g.calculate_graph(water_links=[(link0, link1)], c0=c0)

    dense = g.edges.toarray()
    for a in link0:
        for b in link1:
            expected = np.linalg.norm(g.grid[a] - g.grid[b]) / c0
            assert dense[a, b] == pytest.approx(expected, rel=1e-9), f"{a}->{b}"
            assert dense[b, a] == pytest.approx(expected, rel=1e-9), f"{b}->{a}"


def test_calculate_graph_without_water_links_is_unaffected_by_new_parameter(
    isotropic_material,
):
    """Adding water_links/c0 as new, defaulted parameters must not change
    behaviour for existing callers who never pass them."""
    g, _ = _small_simpl_rect_grid(isotropic_material)
    points = np.array([[-5.0, 2.0], [5.0, 2.0]])
    g.add_points(points=points, sources=np.array([0]), targets=np.array([1]))
    g.calculate_graph()
    assert g.edges.nnz > 0
