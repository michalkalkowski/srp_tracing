"""
Part of srp_tracing.
Test of the SRP ray tracing routine.

An austenitic stainless steel weld with Ogilvy map orientations across the
weld. Sources and receivers positioned centrally on both the top and the bottom
surfaces of the weld.

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

# initial map

oweld_parameters = dict([('T', 2),
                        ('n_weld', 1.3),
                        ('a', a),
                        ('b', b),
                        ('c', c)])
oweld = o_weld(oweld_parameters)
oweld.define_grid_size(2, use_centroids=True, add_boundary_cells=True,
                       boundary_offset=1.)
oweld.solve()

# Define the domain
# Add cells around the weld (not necessary for this test, needed for ray
# projections in time of flight tomography)
orientations = oweld.grain_orientations_full[:]
bb = orientations[:]
weld_mask = np.copy(oweld.in_weld)
aa = np.zeros([bb.shape[0], 6])
cc = np.zeros([bb.shape[0], 6])
orientations = np.column_stack((aa, bb, cc))
new_wm = np.column_stack((aa, weld_mask, cc))

# Move sources and sensors away from the edge of the domain
orientations[new_wm != 1] = 0

nx = orientations.shape[1]
ny = orientations.shape[0]
dx = 2

# Sensors
start_gen = -32.55
pitch = 2.05
element_width = 1.55

sx = (element_width/2 + np.arange(start_gen, start_gen + 32*pitch, pitch))
sy = np.array(len(sx)*[a])
sources = targets = np.r_[np.column_stack((sx, sy)),
                          np.column_stack((sx, np.zeros(len(sy))))]

# Properties
orientation_map = orientations
rho_parent = 7.9
# Define weld material
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

cx = 0
cy = 19

no_seeds = 8

test_grid = grid.RectGrid(nx, ny, cx, cy, dx, no_seeds)
test_grid.assign_model(mode='orientations', property_map=orientation_map)
test_grid.add_points(sources=sources, targets=targets)
test_grid.assign_materials(new_wm,
                           dict([(0, parent_basis),
                                 (1, weld_basis)]))
test_grid.calculate_graph()
test = solver.Solver(test_grid)
test.solve(source_indices=test_grid.source_idx, with_points=True)

tofs_srp = test.tfs[:, test_grid.target_idx].T
tofs_srp[:32, :32] = np.nan
tofs_srp[32:, 32:] = np.nan
target = np.load('../data/SRP_validation_ogilvy.npy')*1e6
target[:32, :32] = np.nan

plt.figure()
plt.plot(target[:, 6], lw=1, c='gray', label='FE 2 MHz')
plt.plot(target[:, 15], lw=1, c='gray')
plt.plot(target[:, 31 - 5], lw=1, c='gray')
plt.plot(tofs_srp[:, 6], lw=1,  c='red', label='SRP')
plt.plot(tofs_srp[:, 15], lw=1, c='red')
plt.plot(tofs_srp[:, 31 - 5], lw=1, c='red')
plt.xlabel('sensor #')
plt.ylabel('time of flight in us')
plt.tight_layout()
plt.legend()
plt.show()


# Plotting paths
paths = test.calculate_ray_paths(test.grid.target_idx)
gx = -dx/2 + np.append(np.unique(test_grid.image_grid[:, 0]),
                       test_grid.image_grid[-1, 0] + dx)
gy = -dx/2 + np.append(np.unique(test_grid.image_grid[:, 1]),
                       test_grid.image_grid[-1, 1] + dx)
Gx, Gy = np.meshgrid(gx, gy)

fig, ax = plt.subplots()
ax.imshow(new_wm, origin='lower',
          extent=[gx.min(), gx.max(), gy.min(), gy.max()])
ax.plot(test_grid.grid[:, 0], test_grid.grid[:, 1], 'o', ms=1, c='gray')
ax.plot(Gx, Gy, c='gray')
ax.plot(Gx.T, Gy.T, c='gray')
ax.plot(test_grid.image_grid[:, 0], test_grid.image_grid[:, 1], 'x', ms=4)
for path in paths[26]:
    plt.plot(test.grid.grid[path, 0], test.grid.grid[path, 1], '-x', ms=1)
plt.show()
