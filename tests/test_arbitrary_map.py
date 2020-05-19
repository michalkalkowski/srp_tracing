"""
Part of srp_tracing.
Test of the SRP ray tracing routine.

An austenitic stainless steel weld with an arbitrary set of orientations
across the weld. Sources and receivers positioned centrally on both the
top and the bottom surfaces of the weld.

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""
import numpy as np
import matplotlib.pyplot as plt
from srp_tracing import grid, solver
from mina.original import MINA_weld
# Define the weld
a = 36.8
b = 1.
c = 40


# Create a MINA model for the weld first
weld_parameters = dict([('remelt_h', 0.255),
                        ('remelt_v', 0.135),
                        ('theta_b', np.deg2rad(11.5)),
                        ('theta_c', np.deg2rad(0)),
                        ('number_of_layers', 11),
                        ('number_of_passes', np.array([1]*4 + 3*[2] + 3*[3] +
                                                      [4]*1)),
                        ('electrode_diameter', np.array([2.4, 4] +
                                                        [5]*7 + [4]*2)),
                        ('a', a),
                        ('b', b),
                        ('c', c)])
weld = MINA_weld(weld_parameters)
weld.define_order_of_passes('right_to_left')
weld.define_grid_size(2, use_centroids=True, add_boundary_cells=True,
                      boundary_offset=1.)
weld.solve()

# Define the domain
orientations = weld.grain_orientations_full[:]
bb = orientations[:]
weld_mask = np.copy(weld.in_weld)
aa = np.zeros([bb.shape[0], 8])
cc = np.zeros([bb.shape[0], 7])
orientations = np.column_stack((aa, bb, cc))
new_wm = np.column_stack((aa, weld_mask, cc))

# Move sources and sensors away from the edge of the domain
orientations = np.concatenate((np.zeros([9, orientations.shape[1]]),
                               orientations,
                               np.zeros([8, orientations.shape[1]])),
                              axis=0)

new_wm = np.concatenate((np.zeros([9, orientations.shape[1]]),
                         new_wm,
                         np.zeros([8, orientations.shape[1]])),
                        axis=0)
orientations[new_wm != 1] = 0

# Load the arbitrary map to fill mina cells
arb_map = np.load('../data/arbitrary_map.npy')
orient = np.zeros(new_wm.shape)
orient[new_wm == 1] = arb_map

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

no_seeds = 10

test_grid = grid.RectGrid(nx, ny, cx, cy, dx, no_seeds)
test_grid.assign_model(mode='orientations', property_map=orient)
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
target = np.load('../data/SRP_validation_wp2.npy')

plt.figure()
plt.plot(target[:, 6], lw=1, c='gray', label='FE 2 MHz')
plt.plot(target[:, 15], lw=1, c='gray')
plt.plot(target[:, 31 - 5], lw=1, c='gray')
plt.plot(tofs_srp[:, 6], lw=1,  c='red', label='SRP')
plt.plot(tofs_srp[:, 15], lw=1, c='red')
plt.plot(tofs_srp[:, 31 - 5], lw=1, c='red')
plt.xlabel('sensor #')
plt.ylabel('time of flight in s')
plt.tight_layout()
plt.legend()
plt.show()
