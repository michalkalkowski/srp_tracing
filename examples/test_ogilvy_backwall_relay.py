"""
Part of srp_tracing.
Test of the SRP ray tracing routine for a pulse-echo configuration with a
planar backwall -- the backwall-relay replacement for test_ogilvy_backwall.py.

An austenitic stainless steel weld with an Ogilvy orientation map. Sources
and receivers positioned centrally on the top surface of the weld, signal
reflected off the backwall.

Unlike test_ogilvy_backwall.py, this does *not* mirror the domain around
y=0 to get reflected paths "for free" from a single shortest_path solve.
Instead, a grid of points is added along the real backwall (y=0) as
ordinary nodes in the *real* (single, un-mirrored) domain, one solve is run
from the transducers to everything (including those backwall points), and
solver.combine_via_boundary() combines each pair's one-way travel time to
every backwall point into the fastest reflected pulse-echo time -- Fermat's
principle means that's just a min-sum over the shared backwall axis, no
second solve needed. This also works for a non-flat backwall (this
particular weld's backwall is flat, so the two approaches can be compared
directly -- see the numbers below and tests/test_ogilvy_backwall_relay_validation.py),
and lets the backwall's own discretisation density be chosen independently
of the grid's no_seeds, which the mirrored-domain approach does not.

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""
import numpy as np
import matplotlib.pyplot as plt
from srp_tracing import grid, solver
from ogilvy.ogilvy_model import Ogilvy_weld as o_weld

# Define the weld
a = 36.8
b = 1.
c = 40

oweld_parameters = dict([('T', 2),
                        ('n_weld', 1.3),
                        ('a', a),
                        ('b', b),
                        ('c', c)])
oweld = o_weld(oweld_parameters)
oweld.define_grid_size(2, use_centroids=True, add_boundary_cells=True,
                       boundary_offset=1.)
oweld.solve()

# Define the domain -- the real, single-sided domain (not mirrored/doubled)
orientations = oweld.grain_orientations_full[:]
bb = orientations[:]
weld_mask = np.copy(oweld.in_weld)
aa = np.zeros([bb.shape[0], 8])
cc = np.zeros([bb.shape[0], 7])
orientations = np.column_stack((aa, bb, cc))
new_wm = np.column_stack((aa, weld_mask, cc))

# Move sources and sensors away from the edge of the domain
orientations = np.concatenate((orientations,
                               np.zeros([1, orientations.shape[1]])),
                              axis=0)
new_wm = np.concatenate((new_wm,
                         np.zeros([1, orientations.shape[1]])),
                        axis=0)
orientations[new_wm != 1] = 0

nx = orientations.shape[1]
ny = orientations.shape[0]
dx = 2

# Sensors, on the top surface
start_gen = -32.55
pitch = 2.05
element_width = 1.55

sx = (element_width/2 + np.arange(start_gen, start_gen + 32*pitch, pitch))
sy = np.array(len(sx)*[a])
sources = np.column_stack((sx, sy))

# A grid of points along the (flat, for this weld) backwall at y=0. Density
# here is independent of no_seeds -- this is the whole point: refine this
# without touching the bulk grid resolution, or swap in an arbitrarily
# shaped backwall outline instead of a flat line.
backwall_x = np.arange(-33, 33, 0.5)
backwall = np.column_stack((backwall_x, np.zeros_like(backwall_x)))

# Properties
orientation_map = orientations
rho_parent = 7.9
rho_weld = 8.0
c_parent = np.array(
    [[255.61, 95.89, 95.89, 0., 0., 0.],
     [95.89, 255.61, 95.89, 0., 0., 0.],
     [95.89, 95.89, 255.61, 0., 0., 0.],
     [0., 0., 0., 79.86, 0., 0.],
     [0., 0., 0., 0., 79.86, 0.],
     [0., 0., 0., 0., 0., 79.86]])
c_weld = np.array([[262, 148, 160, 0, 0, 0],
                       [148, 262, 160, 0, 0, 0],
                       [160, 160, 229, 0, 0, 0],
                       [0, 0, 0, 82, 0, 0],
                       [0, 0, 0, 0, 82, 0],
                       [0, 0, 0, 0, 0, 57]])

parent_basis = grid.WaveBasis(anisotropy=0, velocity_variant='group')
parent_basis.set_material_props(c_parent, rho_parent)
parent_basis.calculate_wavespeeds()

weld_basis = grid.WaveBasis(anisotropy=1, velocity_variant='group')
weld_basis.set_material_props(c_weld, rho_weld)
weld_basis.calculate_wavespeeds(angles_from_ray=True)

cx = -1
cy = ny*dx/2  # domain spans y in [0, ny*dx]: y=0 is the real backwall

no_seeds = 10

test_grid = grid.RectGrid(nx, ny, cx, cy, dx, no_seeds)
test_grid.assign_model(mode='orientations', property_map=orientation_map)
test_grid.add_points(sources=sources, targets=backwall)
test_grid.assign_materials(new_wm,
                           dict([(0, parent_basis),
                                 (1, weld_basis)]))
test_grid.calculate_graph()
test = solver.Solver(test_grid)
test.solve(source_indices=test_grid.source_idx, with_points=True)

times_to_backwall = test.tfs[:, test_grid.target_idx]
tofs_srp, via_index = solver.combine_via_boundary(times_to_backwall)

target = np.load('../data/SRP_validation_ogilvy_backwall.npy')

plt.figure()
plt.plot(target[:, 10], lw=1, c='C1', label='FE 4 MHz')
plt.plot(target[:, 14], lw=1, c='C1')
plt.plot(target[:, 31 - 5], lw=1, c='C1')
plt.plot(tofs_srp[:, 10], lw=1, c='red', label='SRP (backwall relay)')
plt.plot(tofs_srp[:, 14], lw=1, c='red')
plt.plot(tofs_srp[:, 31 - 5], lw=1, c='red')
plt.xlabel('sensor #')
plt.ylabel('time of flight in us')
plt.tight_layout()
plt.legend()
plt.show()


# Plotting a reflected ray path
source_node = test_grid.source_idx[10]
target_node = test_grid.source_idx[14]
via_node = test_grid.target_idx[via_index[10, 14]]
path = test.calculate_relay_ray_path(source_node, target_node, via_node)

gx = -dx/2 + np.append(np.unique(test_grid.image_grid[:, 0]),
                       test_grid.image_grid[-1, 0] + dx)
gy = -dx/2 + np.append(np.unique(test_grid.image_grid[:, 1]),
                       test_grid.image_grid[-1, 1] + dx)
Gx, Gy = np.meshgrid(gx, gy)

fig, ax = plt.subplots()
ax.imshow(np.rad2deg(orientation_map), origin='lower',
          extent=[gx.min(), gx.max(), gy.min(), gy.max()])
ax.plot(test_grid.grid[:, 0], test_grid.grid[:, 1], 'o', ms=1, c='gray')
ax.plot(Gx, Gy, c='gray')
ax.plot(Gx.T, Gy.T, c='gray')
ax.plot(backwall[:, 0], backwall[:, 1], '.', ms=2, c='cyan')
ax.plot(test_grid.grid[path, 0], test_grid.grid[path, 1], '-x', ms=4, c='yellow')
ax.set_aspect('equal')
plt.show()
