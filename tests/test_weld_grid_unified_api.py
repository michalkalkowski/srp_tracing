"""
Smoke tests for WeldGrid, the unified chamfer='smooth'|'staircase' API
(see the "Unified chamfer='smooth'|'staircase' grid API" plan). WeldGrid
wraps a plain RectGrid ('staircase') or SimplRectGrid ('smooth') internally
and forwards to it -- these tests confirm the same small weld geometry,
built through WeldGrid, reproduces its corresponding RectGrid/SimplRectGrid
snapshot from test_regression_small_model.py / test_regression_simpl_rect_grid.py
exactly, and that water_links works under chamfer='smooth'.
"""
import numpy as np
import pytest

from srp_tracing import grid, solver

from test_regression_small_model import (
    EXPECTED_TOFS as RECT_EXPECTED_TOFS,
)
from test_regression_simpl_rect_grid import (
    EXPECTED_TOFS as SIMPL_EXPECTED_TOFS,
)


def test_weld_grid_staircase_matches_rect_grid_snapshot(isotropic_material):
    nx, ny = 5, 4
    dx = 2.0
    no_seeds = 3

    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, 2] = 1
    property_map = np.zeros((ny, nx))

    parent = isotropic_material(vp=5.9)
    weld = isotropic_material(vp=3.24)

    g = grid.WeldGrid(nx=nx, ny=ny, cx=0.0, cy=0.0, pixel_size=dx,
                      no_seeds=no_seeds, chamfer='staircase')
    g.assign_model(mode="orientations", property_map=property_map)

    sources = np.array([[-3.0, 3.0], [0.0, 3.0], [3.0, 3.0]])
    targets = np.array([[-3.0, -3.0], [0.0, -3.0], [3.0, -3.0]])
    g.add_points(sources=sources, targets=targets)
    g.assign_materials(material_map, {0: parent, 1: weld})
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    tofs = s.tfs[:, g.target_idx]
    np.testing.assert_allclose(tofs, RECT_EXPECTED_TOFS, rtol=1e-7)


def test_weld_grid_smooth_matches_simpl_rect_grid_snapshot(isotropic_material):
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

    g = grid.WeldGrid(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx,
                      no_seeds=no_seeds, chamfer='smooth')
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: parent, 1: weld})
    g.trim_to_chamfer(a, b, c)
    g.simplify_grid()

    sources = np.array([[-5.0, 2.0], [-5.0, 5.0], [5.0, 2.0], [5.0, 5.0]])
    g.add_points(sources=sources, targets=sources)
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)
    tofs = s.tfs[:, g.target_idx]
    np.testing.assert_allclose(tofs, SIMPL_EXPECTED_TOFS, rtol=1e-7, atol=1e-9)


def test_weld_grid_staircase_rejects_smooth_only_methods(isotropic_material):
    g = grid.WeldGrid(nx=5, ny=4, cx=0.0, cy=0.0, pixel_size=2.0, no_seeds=3,
                      chamfer='staircase')
    with pytest.raises(NotImplementedError):
        g.trim_to_chamfer(10.0, 2.0, 14.0)
    with pytest.raises(NotImplementedError):
        g.simplify_grid()


def test_weld_grid_smooth_supports_water_links(isotropic_material):
    a, b, c = 10.0, 2.0, 14.0
    dx = 2.0
    no_seeds = 4
    nx, ny = 8, 6
    cx, cy = 0.0, a / 2

    material_map = np.zeros((ny, nx), dtype=int)
    material_map[:, nx // 2 - 1: nx // 2 + 1] = 1
    property_map = np.zeros((ny, nx))
    material = isotropic_material(vp=5.9)

    g = grid.WeldGrid(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx,
                      no_seeds=no_seeds, chamfer='smooth')
    g.assign_model(mode="orientations", property_map=property_map)
    g.assign_materials(material_map, {0: material, 1: material})
    g.trim_to_chamfer(a, b, c)
    g.simplify_grid()

    group_a = np.array([[-5.0, 2.0], [-5.0, 5.0], [-5.0, 8.0]])
    group_b = np.array([[5.0, 2.0], [5.0, 5.0]])
    g.add_points(sources=group_a, targets=group_b)

    c0 = 1.48
    link0, link1 = g.source_idx, g.target_idx
    g.calculate_graph(water_links=[(link0, link1)], c0=c0)

    dense = g.edges.toarray()
    for a_ix in link0:
        for b_ix in link1:
            expected = np.linalg.norm(g.grid[a_ix] - g.grid[b_ix]) / c0
            assert dense[a_ix, b_ix] == pytest.approx(expected, rel=1e-9)
            assert dense[b_ix, a_ix] == pytest.approx(expected, rel=1e-9)
