"""
Pinning tests: a symmetric two-cell RectGrid, one homogeneous isotropic
material, crossing the shared cell boundary.

With ny=1 and an even no_seeds, the boundary shared by the two pixels is
seeded at y=0 exactly (see RectGrid.__init__: seeds start at Gy.min() and
step by pixel_size/no_seeds, so for a domain centred on cy=0 the midpoint
lands on a seed iff no_seeds is even). Placing source and target on that
same y=0 line means the straight path source -> (0, 0) -> target is exactly
representable in the graph (each leg lives inside a single pixel, so
RectGrid.calculate_graph gives it a direct edge), and by the triangle
inequality no cross-boundary detour can beat it. That gives us both an exact
analytic value and a genuine multi-hop shortest-path solve (unlike the
single-cell case, this exercises the graph joining logic across pixels).
"""
import numpy as np
import pytest

from srp_tracing import grid, solver


def _two_cell_grid(v, no_seeds=4, pixel_size=10.0):
    material = None  # set by caller via isotropic_material fixture
    g = grid.RectGrid(nx=2, ny=1, cx=0.0, cy=0.0, pixel_size=pixel_size,
                      no_seeds=no_seeds)
    g.assign_model(mode="orientations", property_map=np.zeros((1, 2)))
    return g


def test_two_cell_straight_line_matches_analytic(isotropic_material):
    v = 4.3
    pixel_size = 10.0
    material = isotropic_material(vp=v)
    g = _two_cell_grid(v, no_seeds=4, pixel_size=pixel_size)

    source = np.array([[-0.9 * pixel_size, 0.0]])
    target = np.array([[0.9 * pixel_size, 0.0]])
    g.add_points(sources=source, targets=target)
    g.assign_materials(np.zeros((1, 2), dtype=int), {0: material})
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    expected = np.linalg.norm(target[0] - source[0]) / v
    actual = s.tfs[0, g.target_idx[0]]
    assert actual == pytest.approx(expected, rel=1e-9)


def test_two_cell_reciprocity(isotropic_material):
    """T(A -> B) must equal T(B -> A): edge costs are symmetric (same
    distance, same material both ways), so the shortest path cost is too."""
    material = isotropic_material(vp=3.7)
    pixel_size = 10.0
    g = _two_cell_grid(3.7, no_seeds=4, pixel_size=pixel_size)

    a = np.array([-0.8 * pixel_size, 0.3 * pixel_size])
    b = np.array([0.7 * pixel_size, -0.2 * pixel_size])
    g.add_points(sources=np.array([a, b]), targets=np.array([a, b]))
    g.assign_materials(np.zeros((1, 2), dtype=int), {0: material})
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    t_a_to_b = s.tfs[0, g.target_idx[1]]
    t_b_to_a = s.tfs[1, g.target_idx[0]]
    assert t_a_to_b == pytest.approx(t_b_to_a, rel=1e-12)
    assert t_a_to_b > 0


def test_two_cell_mirror_symmetry(isotropic_material):
    """Two source/target pairs that are mirror images of each other about
    x=0 must give the same travel time, since the domain and material are
    symmetric about the shared cell boundary."""
    material = isotropic_material(vp=5.1)
    pixel_size = 10.0
    g = _two_cell_grid(5.1, no_seeds=4, pixel_size=pixel_size)

    left_src = np.array([-0.85 * pixel_size, 0.4 * pixel_size])
    left_tgt = np.array([-0.2 * pixel_size, -0.3 * pixel_size])
    right_src = np.array([0.85 * pixel_size, 0.4 * pixel_size])
    right_tgt = np.array([0.2 * pixel_size, -0.3 * pixel_size])

    sources = np.array([left_src, right_src])
    targets = np.array([left_tgt, right_tgt])
    g.add_points(sources=sources, targets=targets)
    g.assign_materials(np.zeros((1, 2), dtype=int), {0: material})
    g.calculate_graph()

    s = solver.Solver(g)
    s.solve(source_indices=g.source_idx)

    t_left = s.tfs[0, g.target_idx[0]]
    t_right = s.tfs[1, g.target_idx[1]]
    assert t_left == pytest.approx(t_right, rel=1e-9)
