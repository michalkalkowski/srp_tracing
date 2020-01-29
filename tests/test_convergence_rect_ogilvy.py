"""
Test of the SRP ray tracing routine.

A tomography setup with top and bottom transducers and the weld described by
the Ogilvy map
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from srp_tracing import grid, solver
import pogopy.misc as misc
from ogilvy.ogilvy_model import Ogilvy_weld as o_weld

# Define the weld
a = 36.8
b = 1.
c = 39


# Create a MINA model for the weld first
weld_parameters = dict([('T', 1),
                        ('n_weld', 1),
                        ('a', a),
                        ('b', b),
                        ('c', c)])
weld = o_weld(weld_parameters)
weld.define_grid_size(2, use_centroids=True, add_boundary_cells=True,
                      boundary_offset=1.)
weld.solve()

# Define the domain
orientations = weld.grain_orientations_full[:]
bb = orientations[:]
weld_mask = np.copy(weld.in_weld)
aa = np.zeros([bb.shape[0], 14])
cc = np.zeros([bb.shape[0], 14])
orientations = np.column_stack((aa, bb, cc))
weld_mask = np.column_stack((aa, weld_mask, cc))

# Move sources and sensors away from the edge of the domain
orientations = np.concatenate((np.zeros([2, orientations.shape[1]]),
                               orientations,
                               np.zeros([2, orientations.shape[1]])),
                              axis=0)

weld_mask = np.concatenate((np.zeros([2, orientations.shape[1]]),
                            weld_mask,
                            np.zeros([2, orientations.shape[1]])),
                           axis=0)
orientations[weld_mask != 1] = 0

nx = orientations.shape[1]
ny = orientations.shape[0]
dx = .002
nodes_per_pixel = 6
dev = 10

start_gen = -47.25
pitch = 1.5
sx = np.arange(start_gen, start_gen + 64*pitch, pitch)*1e-3
sy = np.array(len(sx)*[a/1e3])
sources = targets = np.r_[np.column_stack((sx, sy)),
                          np.column_stack((sx, np.zeros(len(sy))))]

# Properties
orientation_map = orientations
rho_parent = 7.9e3
# Define weld material
rho_weld = 8.0e3
c_parent = 1e9*np.array(
    [[255.61, 95.89, 95.89, 0., 0., 0.],
     [95.89, 255.61, 95.89, 0., 0., 0.],
     [95.89, 95.89, 255.61, 0., 0., 0.],
     [0., 0., 0., 79.86, 0., 0.],
     [0., 0., 0., 0., 79.86, 0.],
     [0., 0., 0., 0., 0., 79.86]])
# This is the stiffness matrix in material coordinates
c_weld = 1e9*np.array([[234, 118, 148, 0, 0, 0],
                       [118, 240, 146, 0, 0, 0],
                       [148, 146, 210, 0, 0, 0],
                       [0, 0, 0, 99, 0, 0],
                       [0, 0, 0, 0, 110, 0],
                       [0, 0, 0, 0, 0, 95]])

parent_basis = grid.WaveBasis(anisotropy=0, velocity_variant='group')
parent_basis.set_material_props(c_parent, rho_parent)
parent_basis.calculate_wavespeeds()

weld_basis = grid.WaveBasis(anisotropy=1, velocity_variant='group')
weld_basis.set_material_props(c_weld, rho_weld)
weld_basis.calculate_wavespeeds(angles_from_ray=True)


cx = 0.5e-3
cy = 19e-3

node_densities = np.array([4, 8, 10, 15, 20])

tofs_conv = np.zeros([len(node_densities), 64, 64])
for i in range(len(node_densities)):
    jump_level = node_densities[i]
    test_grid = grid.RectGrid(nx, ny, cx, cy, dx,
                              nodes_per_pixel, node_densities[i])
    test_grid.assign_model(mode='orientations', property_map=orientation_map,
                           weld_model=None, only_weld=False)
    test_grid.add_points(sources=sources, targets=targets)
    test_grid.assign_materials(weld_mask,
                               dict([(0, parent_basis),
                                     (1, weld_basis)]))
    test_grid.calculate_graph()
    test = solver.Solver(test_grid)
    test.solve(source_indices=test_grid.source_idx, with_points=False)
    tofs_conv[i] = test.tfs[:, test_grid.target_idx].T[64:, :64]

target = np.load('../data/correct_SRP_weld_ogilvy.npy')
target_4MHz = np.load('../data/correct_SRP_weld_ogilvy_4MHz.npy')
# ignore sensors close to the root
tofs_no_root = np.copy(tofs_conv)
tofs_no_root[:, 28:35] = np.nan
tofs_org = np.copy(tofs_conv)

tofs_conv = tofs_no_root
relative = abs(tofs_conv - target)/target
absolute = abs(tofs_conv - target)

# Relative error
fig, ax = plt.subplots()
ax.plot(node_densities, np.nanmean(relative*100, axis=(1, 2)), '-o')
ax2 = ax.twinx()
ax2.set_ylabel('absolute time of flight mean error in us', color='C1')
ax2.plot(node_densities, 1e6*np.nanmean(absolute, axis=(1, 2)), '-o', ms=10,
         mfc='None', c='C1')
ax.set_xlabel('nodes per pixel edge')
ax.set_ylabel('relative time of flight mean error in %')
#ax.set_title('Relative ToF error vs no. of nodes per pixel and jump level')
plt.tight_layout()
plt.show()
ax2.set_yticklabels(ax2.get_yticklabels(), c='C1')


# Absolute error
fig, ax = plt.subplots()
ax.plot(node_densities, 1e6*np.nanmean(absolute, axis=(1, 2)), '-o')
ax.set_xlabel('nodes per pixel edge')
ax.set_ylabel('absolute time of flight mean error in us')
#ax.set_title('Absolute ToF error vs no. of nodes per pixel and jump level')
plt.tight_layout()


#time_srp[:32, :32] = np.nan
#time_srp[32:, 32:] = np.nan
gx = -dx/2 + np.append(np.unique(test_grid.image_grid[:, 0]), test_grid.image_grid[-1, 0] + dx)
gy = -dx/2 + np.append(np.unique(test_grid.image_grid[:, 1]), test_grid.image_grid[-1, 1] + dx)
Gx, Gy = np.meshgrid(gx, gy)

fig, ax = plt.subplots()
ax.plot(test_grid.grid[:, 0], test_grid.grid[:, 1], 'o', ms=2, c='gray')
ax.plot(Gx, Gy, c='gray')
ax.plot(Gx.T, Gy.T, c='gray')
ax.plot(test_grid.image_grid[:, 0], test_grid.image_grid[:, 1], 'x', ms=4)
ax.set_aspect('equal')
