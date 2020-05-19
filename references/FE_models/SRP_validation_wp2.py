#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of the SRP routine
- MINA map

@author: Michal K Kalkowski
@email: m.kalkowski@imperial.ac.uk
"""
from tqdm import trange
import numpy as np
import pogopy.generate as pg
import pogopy.model as pm
import pogopy.write as pw
import pogopy.misc as misc
from mina.original import MINA_weld
from ogilvy.ogilvy_model import Ogilvy_weld as o_weld
from argparse import ArgumentParser
from scipy.spatial import cKDTree

rho_parent = 7.9e-9
rho_weld = 8.0e-9

c_parent = 1e3*np.array(
    [[255.61,  95.89,  95.89,   0.  ,   0.  ,   0.  ],
    [ 95.89, 255.61,  95.89,   0.  ,   0.  ,   0.  ],
    [ 95.89,  95.89, 255.61,   0.  ,   0.  ,   0.  ],
    [  0.  ,   0.  ,   0.  ,  79.86,   0.  ,   0.  ],
    [  0.  ,   0.  ,   0.  ,   0.  ,  79.86,   0.  ],
    [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,  79.86]])
c_weld = 1e3*np.array([[262, 160, 148, 0, 0, 0],
                       [160, 229, 160, 0, 0, 0],
                       [148, 160, 262, 0, 0, 0],
                       [0, 0, 0, 82, 0, 0],
                       [0, 0, 0, 0, 57, 0],
                       [0, 0, 0, 0, 0, 82]])


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

wp2_map = np.load('WP2_EBSD_avg_orients.npy')

input_orientations = np.copy(wp2_map)

# Model geometry
length = 110
abs_width = 20
top_height = a
# mesh size
d = 0.025

freq = 2e6

# Determine the number of elements
nx = ny = int(np.ceil(length/d))
# Adjust element size to match the desired model geometry
dx = dy = length/nx
print('Creating the mesh...')
# Create mesh object
mesh = pg.Mesh(2)
mesh.define_rect(dx=dx, dy=dy, nx=nx, ny=int(np.ceil(a/dx)),
                 start=(-length/2, 0))

mesh.get_element_centroids()

# Create the Pogo model
model = pm.PogoModel(mesh)

# Add materials to the model
model.add_material(num_label=0, material_type=2, density=rho_parent,
                   elasticity=misc.c2tricl_list(c_parent), label='Parent')
# model.add_material(num_label=1, material_type=2, density=rho_weld,
#                    elasticity=misc.c2tricl_list(c_weld), label='Weld')
base_materials = len(model.materials.keys())
# Compute parent p-wave velocity
cl = (c_parent[0, 0]/rho_parent)**0.5

# Determine the list of elements belonging to the weld
in_weld = []
limited_view = np.where(abs(mesh.centroids[:, 0]) < c/2 + 2)[0]
for i in trange(len(weld.mina_grid_points)):
    around = np.where((abs(mesh.centroids[limited_view, 0] - weld.mina_grid_points[i, 0]) < 1)
    & (abs(mesh.centroids[limited_view, 1] - weld.mina_grid_points[i, 1]) <
       1))[0]
    in_weld.extend(limited_view[around])
in_weld = np.array(in_weld)
weld_elements = np.unique(in_weld)

# Assign orientatinos (create new material objects with rotated properties)
print('Assigning orientations...')
sections = []
orientations = []
# Initialise element orientation matrix with (-1)s
el_orient = np.ones(model.mesh.assigned_material.shape)*(-1)
# Determine which elements belong to which MNIA cell
vec = (mesh.centroids[weld_elements].reshape(1, len(weld_elements), 2)
       - weld.mina_grid_points.reshape(-1, 1, 2))
indices = ((vec > -weld.grid_size/2).all(axis=2)
           & (vec < weld.grid_size/2).all(axis=2))

# Loop through MINA cells and assign materials with rotated properties.
for i in trange(len(weld.mina_grid_points)):
    orientation = input_orientations[i]
    # The angle is negative because the out-of-plane axis has been flipped
    # while reorienting the elasticity matrix to match desired orientation
    c_rot = misc.rotate_c(c_weld, -np.rad2deg(orientation), 2)
    el_orient[weld_elements[indices[i]]] = orientation
    # Add the 'new' material
    # Add section corresponding to the current MINA grid region
    model.add_material(num_label=i + base_materials, material_type=2,
                       density=rho_weld,
                       elasticity=misc.c2tricl_list(c_rot),
                       label='weld_{}'.format(i))
    sections.append(pg.Section(material_reference=i + base_materials,
                               orientation_reference=-1))
    model.mesh.assign_section(weld_elements[indices[i]], sections[-1], 'CPE4')

# Remaining elements are assigned to the parent material section
parent_elements = np.where((el_orient == -1))[0]
sections.append(pg.Section(material_reference=0, orientation_reference=-1))
model.mesh.assign_section(parent_elements, sections[-1], 'CPE4')

# Determine simulation parameters
# dx = model.mesh.node_locations[1][0] - model.mesh.node_locations[0][0]
CFL = 0.8
model.simulation_parameters['time_step'] = dx/(cl)*CFL
no_steps = np.round(30e-6/model.simulation_parameters['time_step'])
model.simulation_parameters['no_steps'] = int(no_steps)


# Define excitation
excit_parameters = dict([('f', freq), ('n_cycles', 3)])
no_of_generators = 32

start_gen = -32.55 + 1.55/2
pitch = 2.05
nodes = cKDTree(model.mesh.node_locations)

transducers = []
for i in range(no_of_generators):
    _, idx = nodes.query(np.array([start_gen + i*pitch, a]))
    model.add_node_excitation(np.array([idx]), excit_parameters,
                              amps=np.array([0, -10]))
    transducers.append(model.mesh.node_locations[idx])
    model.add_receivers('sens_element_{}'.format(i), np.array([idx]), [1])
for i in range(no_of_generators, 2*no_of_generators):
    _, idx = nodes.query(np.array([start_gen + (i - no_of_generators)*pitch,
                                   0]))
    transducers.append(model.mesh.node_locations[idx])
    model.add_receivers('sens_element_{}'.format(i), np.array([idx]), [1])
# Add an absorbing boundary
model.add_absorbing_boundary(
    x_lims=np.array([-55, -35, 35, 55]))
model.define_measurement_frequency()
model.define_field_save_increment(total_steps=50,
                                  field_store_nodes=-1)
pw.write_input_file(model, 'SRP_validation_wp2.pogo-inp')
