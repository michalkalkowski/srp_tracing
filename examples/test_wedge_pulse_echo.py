"""
Part of srp_tracing.
Demonstrates the wedge/immersion pulse-echo chain from the "Unified
chamfer='smooth'|'staircase' grid API" tidy-up (Phase 4): a transducer
sits in a rexolite wedge above the weld's outer surface, rather than in
direct contact. A rexolite wedge is the same shape as a water immersion
setup -- a homogeneous, unobstructed isotropic medium between transducer
and frontwall -- so the wedge leg is analytic (solver.straight_line_times),
exactly like water. The frontwall/backwall relay through the solid uses
the same solver.combine_via_boundary() machinery as
test_ogilvy_backwall_relay.py, chained twice (frontwall, then backwall,
then frontwall again) for the full pulse-echo round trip -- see
tests/test_wedge_relay.py::test_wedge_pulse_echo_chain_matches_brute_force
for a proof that this two-call chaining is exact.

Uses a small synthetic chamfer weld (no unpublished ogilvy/mina package
needed) via grid.WeldGrid(chamfer='smooth') -- there is no finite-element
reference data for a wedge case yet, so this is illustrative, not
FE-validated (see tests/test_wedge_relay.py for the self-consistency
checks that stand in for validation until real data exists).

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from srp_tracing import grid, solver


def _isotropic_material(vp, vs=None, rho=1.0):
    if vs is None:
        vs = vp / 2
    c = np.zeros((6, 6))
    c[0, 0] = c[1, 1] = c[2, 2] = vp**2 * rho
    c[3, 3] = c[4, 4] = c[5, 5] = vs**2 * rho
    m = grid.WaveBasis(anisotropy=0, velocity_variant='group')
    m.set_material_props(c, rho)
    m.calculate_wavespeeds()
    return m


# Weld geometry (chamfer defined by outer width a, root width b, top width c)
a, b, c = 10.0, 2.0, 14.0
dx = 2.0
no_seeds = 4
nx, ny = 8, 6
cx, cy = 0.0, a / 2

material_map = np.zeros((ny, nx), dtype=int)
material_map[:, nx // 2 - 1: nx // 2 + 1] = 1  # slower weld columns
property_map = np.zeros((ny, nx))

parent = _isotropic_material(vp=5.9)
weld = _isotropic_material(vp=3.24)
REXOLITE_V = 2.34  # mm/us, typical bulk longitudinal wavespeed

weld_grid = grid.WeldGrid(nx=nx, ny=ny, cx=cx, cy=cy, pixel_size=dx,
                          no_seeds=no_seeds, chamfer='smooth')
weld_grid.assign_model(mode='orientations', property_map=property_map)
weld_grid.assign_materials(material_map, {0: parent, 1: weld})
weld_grid.trim_to_chamfer(a, b, c)
weld_grid.simplify_grid()

# Frontwall (near the outer surface, y=5) and backwall (near the root,
# y=2) point grids -- density here is independent of no_seeds, same as
# test_ogilvy_backwall_relay.py's backwall.
frontwall = np.column_stack((np.linspace(-6.5, 6.5, 9), np.full(9, 5.0)))
backwall = np.column_stack((np.linspace(-6.5, 6.5, 9), np.full(9, 2.0)))
weld_grid.add_points(sources=frontwall, targets=backwall)
weld_grid.calculate_graph()

test = solver.Solver(weld_grid)
test.solve(source_indices=weld_grid.source_idx, with_points=True)
solid_leg = test.tfs[:, weld_grid.target_idx]  # (n_frontwall, n_backwall)

# Two wedge-coupled transducers above the frontwall
transducers = np.array([[-6.0, 12.0], [4.0, 13.0]])
wedge_leg = solver.straight_line_times(transducers, frontwall, REXOLITE_V)

to_backwall, via_frontwall = solver.combine_via_boundary(wedge_leg, solid_leg.T)
pulse_echo, via_backwall = solver.combine_via_boundary(to_backwall)

print('Pulse-echo times (us):')
print(pulse_echo)


def _full_path_coords(i, j):
    """Physical coordinates from transducer i to transducer j via the best
    frontwall-in -> backwall -> frontwall-out relay, including the two
    (non-graph) wedge legs at each end."""
    q = via_backwall[i, j]
    p_in = via_frontwall[i, q]
    p_out = via_frontwall[j, q]
    solid_path = test.calculate_relay_ray_path(
        weld_grid.source_idx[p_in], weld_grid.source_idx[p_out],
        weld_grid.target_idx[q])
    return np.vstack((transducers[i], weld_grid.grid[solid_path], transducers[j]))


# Self (pulse-echo) path for transducer 0, and a pitch-catch path between
# transducer 0 and transducer 1
path_self = _full_path_coords(0, 0)
path_cross = _full_path_coords(0, 1)

fig, ax = plt.subplots()
ax.imshow(material_map, origin='lower',
          extent=[cx - nx*dx/2, cx + nx*dx/2, cy - ny*dx/2, cy + ny*dx/2],
          cmap='gray_r', alpha=0.3, vmin=0, vmax=1)
ax.plot(weld_grid.grid[:, 0], weld_grid.grid[:, 1], 'o', ms=1, c='gray')
ax.plot(frontwall[:, 0], frontwall[:, 1], '.', ms=6, c='cyan', label='frontwall')
ax.plot(backwall[:, 0], backwall[:, 1], '.', ms=6, c='magenta', label='backwall')
ax.plot(transducers[:, 0], transducers[:, 1], '^', ms=10, c='black', label='transducers (in wedge)')
ax.plot(path_self[:, 0], path_self[:, 1], '-x', ms=4, c='red', label='pulse-echo (self)')
ax.plot(path_cross[:, 0], path_cross[:, 1], '-x', ms=4, c='orange', label='pitch-catch (0->1)')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, 'wedge_pulse_echo_paths.png'), dpi=150)

plt.show()
