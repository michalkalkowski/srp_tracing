#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:49:57 2019

@author: michal
"""

import numpy as np
# import matplotlib.pyplot as plt
from tqdm import trange
import heapq
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, coo_matrix
import scipy.interpolate as interpolate
import raytracer.refraction as rf

def get_chamfer_profile(y, weld):
    """
    Outputs the profile of the weld chamfer (a simple mapping from the horizontal coordinate(y)
    to the vertical coordinate(z)). Used for checking whether a certain point is inside the weld or not.

    Parameters:
    ---
    y: float, horizontal coordinate
    a: float, weld thickness in mm
    b: float, weld root width in mm
    c: float, weld cap width in mm

    Returns:
    ---
    f: ndarray, z coordinates of the weld boundary
    """
    boundary_gradient = 2*weld.a/(weld.c - weld.b)
    f = boundary_gradient*(abs(y) - weld.b/2) - weld.a/2
    #f *= (f >= 0)
    return f

class WaveBasis:
    def __init__(self):
        pass
    def set_material_props(self, c, rho):
        self.c = c
        self.rho = rho
    def calculate_wavespeeds(self, wave_type=0):
        angles = np.linspace(0, 2*np.pi, 200)
        basis = rf.calculate_slowness(self.c, self.rho, angles)
        cgy, cgz = (basis[2][:, 1, wave_type].real,
                    basis[2][:, 2, wave_type].real)
        self.int_cgy = interpolate.UnivariateSpline(angles, (cgy)**2, s=0)
        self.int_cgz = interpolate.UnivariateSpline(angles, (cgz)**2, s=0)
        self.int_cg2 = interpolate.UnivariateSpline(angles, (cgy)**2 +
                                                    (cgz)**2, s=0)

    def rotate_velocity(self, my, mz, gamma):
        """
        Rotates a point (my, mz) (or a series of points) around (0, 0) by
        gamma. Used in rotating slowness curves.

        Parameters:
        ---
        my: ndarray, first coordinate(s)
        mz: ndarray, second coordinate(s)
        gamma: float, rotation angle in radians

        Returns:
        ---
        my_rot: ndarray, rotated first coordinate(s)
        mz_rot: ndarray, rotated second coordinate(s)
        """

        my_rot = my * np.cos(gamma) - mz * np.sin(gamma)
        mz_rot = my * np.sin(gamma) + mz * np.cos(gamma)
        return my_rot, mz_rot

    def get_wavespeed(self, orientation, direction):
        incident_angle_abs = (direction - orientation + 2*np.pi) % (2*np.pi)
        #cgy2 = self.int_cgy(incident_angle_abs)
        #cgz2 = self.int_cgz(incident_angle_abs)
        cg2 = self.int_cg2(incident_angle_abs)
        #cgy, cgz = self.rotate_velocity(cgy_rot, cgz_rot, -orientation)
        return cg2
    
    def define_parent_velo(self, c_parent):
        self.c_parent2 = c_parent

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]


class RandomGrid:
    def __init__(self, nx, ny, cx, cy, image_spacing, spacing, dev,
                 mode='weld'):
        self.image_spacing = image_spacing
        self.mx = (nx - 1)/2
        self.my = (ny - 1)/2
        self.xlim = [cx, cx + nx*image_spacing]
        self.ylim = [cy, cy + ny*image_spacing]
        x_image, y_image = np.meshgrid(cx + (np.arange(1, nx +1)
                                             - (nx + 1)/2)*image_spacing,
                                       cy + (np.arange(1, ny +1)
                                             - (ny + 1)/2)*image_spacing)
        if mode == 'slowness':
            x_image, y_image = np.meshgrid(cx + (np.arange(1, nx +1)
                                                 - (nx + 1)/2)*image_spacing,
                                           cy + (np.arange(1, ny +1)
                                                 - (ny + 1)/2)*image_spacing)
        #    x_image, y_image = np.meshgrid(np.arange(cx,
        #                                             cx + nx*image_spacing,
        #                                            image_spacing),
        #                                   np.arange(cy,
        #                                             cy + ny*image_spacing,
        #                                            image_spacing))

        self.image_grid = np.c_[x_image.flatten(), y_image.flatten()]
        self.image_tree = cKDTree(self.image_grid)
        self.spacing = spacing
        self.dev = dev
        self.mode = mode
        self.cx = cx
        self.cy = cy

    def generate_realisation(self, weld_model=None, only_weld=False):
        x = np.arange(-self.mx*self.image_spacing,
                      self.mx*self.image_spacing + self.spacing,
                      self.spacing)
        y = np.arange(-self.my*self.image_spacing,
                      self.my*self.image_spacing + self.spacing,
                      self.spacing)
        X, Y = np.meshgrid(x, y)
        self.structured_gird = np.column_stack((X.flatten(), Y.flatten()))
        self.tree_strctured = cKDTree(self.structured_gird)
        X += np.random.randn(X.shape[0], X.shape[1])*self.spacing*self.dev
        Y += np.random.randn(Y.shape[0], Y.shape[1])*self.spacing*self.dev
        self.grid_1 = np.column_stack((X.flatten(), Y.flatten()))

        if self.mode == 'weld':
            self.weld = weld_model
            self.orientations = weld_model.grain_orientations[:]
        elif self.mode == 'orientations':
            self.orientations = weld_model
        elif self.mode == 'slowness':
            self.orientations = (1/weld_model)**2
        self.only_weld = only_weld

        if only_weld and self.mode == 'weld':
            weld_boundary_grid = np.arange(-self.weld.c/2, self.weld.c/2
                                           + self.spacing/20, self.spacing/4)
            weld_boundary_grid = weld_boundary_grid[
                    abs(weld_boundary_grid) >= self.weld.b/2]
            weld_boundary_z = get_chamfer_profile(weld_boundary_grid,
                                                  self.weld)
            weld_boundary = np.column_stack((weld_boundary_grid,
                                             weld_boundary_z))
            to_take = self.grid_1[:, 1] >= get_chamfer_profile(
                self.grid_1[:, 0], self.weld)
            self.grid_1 = np.concatenate((self.grid_1[to_take],
                                          weld_boundary), axis=0)
            self.boundary_left_idx = list(range(to_take.sum(), to_take.sum()
                                          + len(weld_boundary)//2))
            self.boundary_right_idx = list(range(to_take.sum()
                                                 + len(weld_boundary)//2,
                                           to_take.sum() + len(weld_boundary)))

    def add_points(self, sources=None, targets=None):
        if sources is None:
            self.grid = self.grid_1
        else:
            to_add = []
            if sources is not None:
                to_add.append(sources)
                self.source_idx = np.arange(self.grid_1.shape[0],
                                            self.grid_1.shape[0]
                                            + len(sources)).astype(np.int)
            if targets is not None:
                to_add.append(targets)
                self.target_idx = np.arange(self.grid_1.shape[0]
                                            + len(sources),
                                            self.grid_1.shape[0]
                                            + len(sources)
                                            + len(targets)).astype(np.int)
            if len(to_add) > 0:
                conc = [self.grid_1] + to_add
                self.grid = np.concatenate(conc, axis=0)

    def get_tree(self, dijkstra_level):
        self.tree = cKDTree(self.grid)
        dists, heights = self.tree.query(self.grid, 2)
        avg = np.mean(dists[:, 1:], axis=1)
        self.r = 2**0.5*np.mean(avg)*dijkstra_level
        self.r_closest = self.tree.query_ball_point(self.grid, self.r)
        _, self.parent_grid_point = self.tree_strctured.query(self.grid, 1)

        # Assign all weld boundary nodes to closest to start
        # and all targets to all weld boundary nodes
        if self.only_weld:
            for sensor in self.target_idx:
                self.r_closest[sensor] = np.array([])
            for source in self.source_idx:
                if self.grid[source, 0] < 0:
                    self.r_closest[source] = self.boundary_left_idx
                else:
                    self.r_closest[source] = self.boundary_right_idx
            for target in self.target_idx:
                if self.grid[target, 0] < 0:
                    for boundary_node in self.boundary_left_idx:
                        self.r_closest[boundary_node].append(target)
                else:
                    for boundary_node in self.boundary_right_idx:
                        self.r_closest[boundary_node].append(target)

    def set_weld_flag(self, a=0, b=0, c=0):
        self.weld_flag = np.zeros(len(self.grid), np.uint8)
        if self.mode is 'weld':
            self.weld_flag[self.grid[:, 1] >= get_chamfer_profile(self.grid[:, 0],
                           self.weld)] = 1
            _, self.closest_weld_grid = self.weld.tree.query(self.grid, 1)
            if self.only_weld:
                if (self.grid[self.source_idx, 0] < 0).all():
                    self.weld_flag[self.boundary_right_idx] = 0
                else:
                    self.weld_flag[self.boundary_left_idx] = 0
        else:
            _, self.parent_pixel = self.image_tree.query(self.grid, 1)
            if self.mode == 'orientations':
                self.weld_flag = self.mask.flatten()[self.parent_pixel]

    def add_mask(self, mask):
        self.mask = mask

    def calculate_edges(self, wave_basis=None):
        edges = []
        rows = []
        cols = []
        for current_idx in range(self.grid.shape[0]):
            neighbours = self.r_closest[current_idx]
            if current_idx in neighbours:
                neighbours.remove(current_idx)
                self.r_closest[current_idx] = neighbours
            dist = (- self.grid[current_idx]
                    + self.grid[neighbours])
            if self.mode == 'slowness':
                edge_cost = (
                    dist[:, 0]**2 + dist[:, 1]**2)/self.orientations.flatten()[
                        self.parent_pixel[current_idx]]
            else:
                if self.weld_flag[current_idx] != 1:
                    edge_cost = (
                        dist[:, 0]**2 + dist[:, 1]**2)/wave_basis.c_parent2
                else:
                    angles = np.arctan2(dist[:, 1], dist[:, 0])
                    orientation = self.orientations.flatten()[
                        self.parent_pixel[neighbours]]
                    cg = wave_basis.get_wavespeed(orientation, angles)
                    edge_cost = (dist[:, 0]**2 + dist[:, 1]**2)/cg
            rows_local = [current_idx]*len(neighbours)
            edges_local = edge_cost**0.5
            rows.extend(rows_local)
            cols.extend(list(neighbours))
            edges.extend(edges_local)
#            edges[current_idx, neighbours] = edge_cost**0.5
        self.edges = coo_matrix((edges, (cols, rows))).transpose().tocsr()


    def neighbours(self, current_idx, wave_basis, c_per_node=False):
        neighbours = self.edges[current_idx].indices
        nodal_cost = self.edges[current_idx].data
        return zip(neighbours, nodal_cost)


def dijkstra_search(graph, wave_basis, start, goal, early_exit=False,
                    c_per_node=False):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    while not frontier.empty():
        current = frontier.get()
        if early_exit:
            if current == goal:
                break

        for next in graph.neighbours(current, wave_basis, c_per_node):
            new_cost = cost_so_far[current] + next[1]
        # not sure here - I am not repriotising, just duplicating entries, if
        # needed...
            if next[0] not in cost_so_far or new_cost < cost_so_far[next[0]]:
                cost_so_far[next[0]] = new_cost
                priority = new_cost
                frontier.put(next[0], priority)
                came_from[next[0]] = current

    return [came_from, cost_so_far]

