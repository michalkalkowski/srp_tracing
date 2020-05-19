"""
Test of the SRP ray tracing routine. Isotropic domain - checking arrival times
vs analytical calculation and verifying the effect of nodes per pixel and the
jump level.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from tqdm import trange

from srp_tracing import grid, solver
import pogopy.misc as misc


# Define the domain
nx = 200
ny = 100
dx = .0005
nodes_per_pixel = 4
dev = 0.5
cx = cy = 0

orientation_map = np.zeros([nx, ny])
material_map = np.zeros([nx, ny])
c_aniso = misc.cubic_c_matrix(210e9, 135e9, 95e9).real
rho_aniso = 7.9e3

parent_basis = grid.WaveBasis(anisotropy=1)
parent_basis.set_material_props(c_aniso, rho_aniso)
parent_basis.calculate_wavespeeds()

devs = np.arange(0.1, 0.4, 0.05)
jump_levels = np.arange(2, 9, 2)
time_srp = []

for i in trange(len(devs)):
    time_this_dev = []
    for j in range(len(jump_levels)):
        test_grid = grid.RandomGrid(nx, ny, cx, cy, dx, nodes_per_pixel,
                                    devs[i])
        sources = np.array([[0., 0.025]])
        targets = 1e-3*np.column_stack((np.arange(-50, 51), np.array([-0.025]*101)))
        test_grid.add_points(sources=sources, targets=targets)
        # Properties
        test_grid.assign_model('orientations', property_map=orientation_map)
        test_grid.assign_materials(material_map, dict([(0, parent_basis)]))
        # Define cases to compute
        test_grid.build_graph(jump_level=jump_levels[j])
        test_grid.calculate_edges()
        test = solver.Solver(test_grid)
        test.solve(source_indices=test_grid.source_idx, with_points=False)
        time_this_dev.append(test.tfs[:, test_grid.target_idx])
    time_srp.append(time_this_dev)


# Calculate theoretical ToFs
dist = test_grid.grid[test_grid.target_idx][:, np.newaxis, :]\
        - test_grid.grid[test_grid.source_idx][np.newaxis, :, :]
angles = np.arctan2(dist[:, :, 1], dist[:, :, 0])
cg = parent_basis.get_wavespeed(0, angles)**0.5
abs_dist = (dist[:, :, 0]**2 + dist[:, :, 1]**2)**0.5
time_theoretical = (abs_dist/cg).squeeze()

# Plot results
time_srp = np.array(time_srp).squeeze()
time_srp[np.isinf(time_srp)] = np.nan
relative = abs(time_srp - time_theoretical)/time_theoretical
absolute = time_srp - time_theoretical

labels = ['JL {}'.format(i) for i in jump_levels]
# Relative error
fig, ax = plt.subplots()
ax.plot(devs, np.nanmean(relative*100, axis=(2)), '-o')
ax.set_xlabel('grid dev')
ax.set_ylabel('relative time of flight error in %')
ax.legend(labels, ncol=6, loc=1)
ax.set_title('Relative ToF error vs no. of nodes per pixel and jump level')
plt.tight_layout()

# Absolute error
fig, ax = plt.subplots()
ax.plot(devs, 1e6*np.nanmean(absolute, axis=(2)), '-o')
ax.set_xlabel('grid dev')
ax.set_ylabel('absolute time of flight error in us')
ax.legend(labels, ncol=6, loc=1)
ax.set_title('Absolute ToF error vs no. of nodes per pixel and jump level')
plt.tight_layout()

# Relative error
fig, ax = plt.subplots()
ax.plot(devs, np.nanmax(relative*100, axis=(2)), '-o')
ax.set_xlabel('grid dev')
ax.set_ylabel('relative time of flight error in %')
ax.legend(labels, ncol=6, loc=1)
ax.set_title('Relative ToF error vs no. of nodes per pixel and jump level')
plt.tight_layout()

# Absolute error
fig, ax = plt.subplots()
ax.plot(devs, 1e6*np.nanmax(absolute, axis=(2)), '-o')
ax.set_xlabel('grid dev')
ax.set_ylabel('absolute time of flight error in us')
ax.legend(labels, ncol=6, loc=1)
ax.set_title('Absolute ToF error vs no. of nodes per pixel and jump level')
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(1e2*relative[:, 2].squeeze().T)
ax.set_ylabel('relative time of flight error in %')
ax.legend(['std: {0:.2f}'.format(devs[i]) for i in range(len(devs))])
