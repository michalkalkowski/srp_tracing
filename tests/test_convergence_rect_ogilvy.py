"""
Part of srp_tracing.
Test of the convergence of the SRP ray tracing routine.

An austenitic stainless steel weld with Ogilvy orientations across the weld.
Sources and receivers positioned centrally on both the top and the bottom
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

# Create a MINA model for the weld first
weld_parameters = dict([('T', 2),
                        ('n_weld', 1.3),
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

# Sensors
start_gen = -32.55
pitch = 2.05
element_width = 1.55

sx = (element_width/2 + np.arange(start_gen, start_gen + 32*pitch, pitch))*1e-3
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
c_weld = 1e9*np.array([[262, 148, 160, 0, 0, 0],
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

cx = -1e-3
cy = 18e-3

nos_seeds = np.array([1, 2, 4, 8, 10, 15])

tofs_conv = np.zeros([len(nos_seeds), 32, 32])
for i in range(len(nos_seeds)):
    jump_level = nos_seeds[i]
    test_grid = grid.RectGrid(nx, ny, cx, cy, dx,
                              nos_seeds[i])
    test_grid.assign_model(mode='orientations', property_map=orientation_map)
    test_grid.add_points(sources=sources, targets=targets)
    test_grid.assign_materials(weld_mask,
                               dict([(0, parent_basis),
                                     (1, weld_basis)]))
    test_grid.calculate_graph()
    test = solver.Solver(test_grid)
    test.solve(source_indices=test_grid.source_idx, with_points=False)
    tofs_conv[i] = test.tfs[:, test_grid.target_idx].T[32:, :32]

target = np.load('../data/SRP_validation_ogilvy.npy')[32:, :32]

relative = abs(tofs_conv - target)/target
absolute = abs(tofs_conv - target)

# Relative error
fig, ax = plt.subplots()
ax.plot(nos_seeds, np.nanmean(relative*100, axis=(1, 2)), '-o')
ax2 = ax.twinx()
ax2.set_ylabel('absolute time of flight mean error in us', color='C1')
ax2.plot(nos_seeds, 1e6*np.nanmean(absolute, axis=(1, 2)), '-o', ms=10,
         mfc='None', c='C1')
ax.set_xlabel('nodes per pixel edge')
ax.set_ylabel('relative time of flight mean error in %')
plt.tight_layout()
plt.show()
ax2.set_yticklabels(ax2.get_yticklabels(), c='C1')


# Relative error
fig, ax = plt.subplots()
ax.plot(nos_seeds, np.nanmax(relative*100, axis=(1, 2)), '-o')
ax2 = ax.twinx()
ax2.set_ylabel('absolute time of flight max error in us', color='C1')
ax2.plot(nos_seeds, 1e6*np.nanmax(absolute, axis=(1, 2)), '-o', ms=10,
         mfc='None', c='C1')
ax.set_xlabel('nodes per pixel edge')
ax.set_ylabel('relative time of flight max error in %')
plt.tight_layout()
plt.show()
ax2.set_yticklabels(ax2.get_yticklabels(), c='C1')
