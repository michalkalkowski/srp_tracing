"""
Part of srp_tracing.
Test of the SRP ray tracing routine.

Simulation on a heterogeneous isotropic domain with three materials:
background and a two circular inclusions.

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""

import numpy as np
import matplotlib.pyplot as plt
from srp_tracing import grid, solver
from mina.original import MINA_weld


c2 = 5.970
c1 = 4.400
c3 = 5.200
rho = 8.0

# initial map
slowness_map = np.ones([36, 36])*1/c1
nx = slowness_map.shape[1]
ny = slowness_map.shape[0]
cylinder_centre = np.array([30, 20])
cylinder_r = 12

cx, cy = 36, 36
dx = 2
# Image coordinates
mx = dx/2 + np.arange(nx)*dx - (cx - nx*dx/2)
my = dx/2 + np.arange(nx)*dx - (cy - ny*dx/2)
Mx, My = np.meshgrid(mx, my)
xxyy = np.c_[Mx.flatten(), My.flatten()]

cyl = np.where((xxyy[:, 0] - cylinder_centre[0])**2 + (xxyy[:, 1] - cylinder_centre[1])**2 <= cylinder_r**2)[0]
inds = np.unravel_index(cyl, shape=Mx.shape)

slowness_map[inds] = 1/c2
cyl = np.where((xxyy[:, 0] - 52)**2 + (xxyy[:, 1] - 50)**2 <= 6**2)[0]
inds = np.unravel_index(cyl, shape=Mx.shape)

slowness_map[inds] = 1/c3
# Sensors
start_gen = 1#-32.55
pitch = 2.0
element_width = 1.55

a = 70
sx = (np.arange(start_gen, start_gen + 32*pitch, pitch))
sy = np.array(len(sx)*[a])
sources = targets = np.r_[np.column_stack((sx, sy)),
                          np.column_stack((sx, np.zeros(len(sy))))]

# Properties
c_parent = np.zeros([6, 6])
c_weld = np.zeros([6, 6])

parent_basis = grid.WaveBasis(anisotropy=0, velocity_variant='group')
parent_basis.set_material_props(c_parent, rho)

weld_basis = grid.WaveBasis(anisotropy=1, velocity_variant='group')
weld_basis.set_material_props(c_weld, rho)

no_seeds = 12

test_grid = grid.RectGrid(nx, ny, cx, cy, dx, no_seeds)
test_grid.assign_model(mode='slowness_iso', property_map=slowness_map)
test_grid.add_points(sources=sources, targets=targets)
test_grid.assign_materials(np.ones(slowness_map.shape),
                           dict([(0, parent_basis),
                                 (1, weld_basis)]))
test_grid.calculate_graph()
test = solver.Solver(test_grid)
test.solve(source_indices=test_grid.source_idx, with_points=True)

tofs_srp = test.tfs[:, test_grid.target_idx].T
tofs_srp[:32, :32] = np.nan
tofs_srp[32:, 32:] = np.nan
target = np.load('../data/SRP_validation_anisotropic.npy')
target[:32, :32] = np.nan

plt.figure()
plt.plot(target[:, 6], lw=1, c='gray', label='FE 2 MHz')
plt.plot(target[:, 15], lw=1, c='gray')
plt.plot(target[:, 31 - 5], lw=1, c='gray')
plt.plot(tofs_srp[:, 6], lw=1,  c='red', label='SRP')
plt.plot(tofs_srp[:, 15], lw=1, c='red')
plt.plot(tofs_srp[:, 31 - 5], lw=1, c='red')
plt.xlabel('sensor #')
plt.ylabel('time of flight in s')
plt.legend()
plt.tight_layout()
plt.show()
