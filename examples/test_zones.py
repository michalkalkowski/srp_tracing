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
from advise_support.plot_orientations import plot_grain_orientations
# Define the weld
a = 36.
b = 5.
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

nx = orientations.shape[1]
ny = orientations.shape[0]
dx = .1

# Sensors
no_el = [32, 64, 128]
case = 0
pitch = 1.5
start_gen = -(no_el[case] - 1)*pitch/2

sx = np.arange(start_gen, start_gen + no_el[case]*pitch, pitch)
sy = np.array(len(sx)*[a])
sy0 = np.array(len(sx)*[0])
sources = np.column_stack((sx, sy))
targets = np.column_stack((sx, sy0))

# Properties
rho_parent = 7.9
# Define weld material
rho_weld = 8.0
c_parent = 1*np.array(
    [[255.61, 95.89, 95.89, 0., 0., 0.],
     [95.89, 255.61, 95.89, 0., 0., 0.],
     [95.89, 95.89, 255.61, 0., 0., 0.],
     [0., 0., 0., 79.86, 0., 0.],
     [0., 0., 0., 0., 79.86, 0.],
     [0., 0., 0., 0., 0., 79.86]])
c_weld = 1*np.array([[262, 148, 160, 0, 0, 0],
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


test_grid = grid.ZonesGrid(a, b, c, 6, 8, dx)
test_grid.add_points(sources=sources, targets=targets)
# Generate some angles for the zones (Ogilvy)
theta = np.arctan(oweld.T*abs(test_grid.b/2
                              + test_grid.zone_centroids[:, 1]
                              *np.tan(oweld.alpha))/abs(test_grid.zone_centroids[:, 0])**oweld.n_weld)
theta[test_grid.zone_centroids[:, 0] >= 0] *= -1
theta = theta - np.pi/2
theta[theta <= np.pi/2] += np.pi
theta[theta > np.pi/2] -= np.pi

test_grid.assign_model(mode='orientations', property_map=theta)
materials = np.zeros(theta.shape, 'int')
isotropic = [test_grid.left_iso_zone, test_grid.right_iso_zone]
materials[[i for i in range(len(test_grid.zones_edge_ind))
          if i not in isotropic]] = 1
test_grid.assign_materials(materials,
                           dict([(0, parent_basis),
                                 (1, weld_basis)]))
test_grid.calculate_graph()
test = solver.Solver(test_grid)
test.solve(source_indices=test_grid.source_idx, with_points=True)
tofs_srp = test.tfs[:, test_grid.target_idx].T
paths = test.calculate_ray_paths(test.grid.target_idx)
fig, ax = plt.subplots()
ax.plot(test_grid.grid[:, 0], test_grid.grid[:, 1], 'o', ms=1, c='gray')
for path in paths[5]:
    plt.plot(test.grid.grid[path, 0], test.grid.grid[path, 1], '-x', ms=1)
plt.show()
plot_grain_orientations(theta[test.grid.active], test_grid.zone_centroids[:, 0][test.grid.active], test_grid.zone_centroids[:, 1][test.grid.active], scale=3, ax=plt.gca(), weld_parameters=oweld_parameters, color='C7')

