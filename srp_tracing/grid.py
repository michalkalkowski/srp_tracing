#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:49:57 2019

Part of srp_tracing.

A shortest ray path (SRP) solver for ray tracing in heterogeneous media,
austenitic stainless steel welds in particular.

Grid definitions

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""

from collections import defaultdict
import numpy as np
from tqdm import trange
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import scipy.interpolate as interpolate
import raytracer.refraction as rf


def get_chamfer_profile(y, weld):
    """
    Outputs the profile of the weld chamfer (a simple mapping from the
    horizontal coordinate(y) to the vertical coordinate(z)).
    Used for checking whether a certain point is inside the weld or not.

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
    # f *= (f >= 0)
    return f


class WaveBasis:
    """
    Defines a wave basis for a given material. The wave basis is an
    interpolator for the group velocity as a function of the angle.

    Parameters:
    ---
    anisotropy: int, level of anisotropy: 0 - isotropic, 1 - anisotropic
    velocity_variant: str, which velocity is used in the calculations for
                      travel time; 'group' for group velocity, 'phase' for
                      phase velocity
    """
    def __init__(self, anisotropy=0, velocity_variant='group'):
        self.anisotropy = anisotropy
        self.variant = velocity_variant

    def set_material_props(self, c, rho):
        """
        Defines material properties.

        Parameters:
        ---
        c: ndarray, elasticit matrix (6x6)
        rho: float, density
        """
        self.c = c
        self.rho = rho

    def calculate_wavespeeds(self, wave_type=0, angles_from_ray=True):
        """
        Calculates group velocities and their interpolators.

        Parameters:
        wave_type: int, wave type of interest; 0 - P, 1 - SH, 2 - SV
        """
        angles = np.linspace(0, 2*np.pi, 200)
        basis = rf.calculate_slowness(self.c, self.rho, angles)
        if self.variant == 'group':
            cgy, cgz = (basis[2][:, 1, wave_type].real,
                        basis[2][:, 2, wave_type].real)
            my, mz = (basis[1][:, 1, wave_type].real,
                      basis[1][:, 2, wave_type].real)

            if angles_from_ray:
                angles = np.arctan2(cgz, cgy)
                angles[angles < 0] += 2*np.pi
            self.int_cgy = interpolate.UnivariateSpline(angles, (cgy)**2, s=0)
            self.int_cgz = interpolate.UnivariateSpline(angles, (cgz)**2, s=0)
            self.int_cg2 = interpolate.UnivariateSpline(angles, (cgy)**2 +
                                                        (cgz)**2, s=0)
            # what is below is only useful for time of flight tomography
            # and not necessary for forward modelling
            temp = 1/(cgy**2 + cgz**2)**0.5
            angles = np.linspace(0, 2*np.pi, 200)
            ang2 = np.concatenate((-angles[1:][::-1], angles, angles[1:] +
                                   2*np.pi))
            slowness_group = np.concatenate((temp[1:][::-1], temp, temp[1:]))
            self.sgp = interpolate.UnivariateSpline(ang2, slowness_group,
                                                    s=0)
            self.dsgp = self.sgp.derivative(1)
            self.int_my = interpolate.UnivariateSpline(angles, my, s=0)
            self.int_mz = interpolate.UnivariateSpline(angles, mz, s=0)
            self.int_m = interpolate.UnivariateSpline(angles,
                                                      (my**2 + mz**2)**0.5,
                                                      s=0)

        elif self.variant == 'phase':
            cp = basis[0][:, wave_type].real
            self.int_cg2 = interpolate.UnivariateSpline(angles, cp**2, s=0)
        else:
            print('Error! Velocity variant can only be "phase" or "group"!')

        if self.anisotropy == 0:
            self.wavespeed = self.int_cg2(0)

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

    def get_wavespeed(self, orientation=0, direction=0):
        """
        Calculates wavespeed for the given material orientation and direction
        of the incoming wave in model coordinates.

        Parameters:
        ---
        orientation: float, material (grain) orientation in rad, measured from
        the vertical, anticlockwise positive.
        direction: float, incident angle of the ray in rad, measured with
        respect to the global frame of reference.
        """
        if self.anisotropy != 0:
            incident_angle_abs = (
                direction - orientation + 2*np.pi) % (2*np.pi)
        else:
            incident_angle_abs = 0
        cg2 = self.int_cg2(incident_angle_abs)
        return cg2


class RectGrid:
    """
    Defines a rectangular grid object. The domain is divided into pixels, nodes
    of the grid are placed along the boundaries of the pixels, with a specified
    number of seeds per pixel boundary (each side of the square). The number of
    seeds defines the angular resolution of the solver.
    """
    def __init__(self, nx, ny, cx, cy, pixel_size, no_seeds):
        """
        Parameters:
        ---
        nx: int, number of pixels along the x direction
        ny: int, number of pixels along the y direction
        cx: float, x-coordinate of the centre of the domain
        cy: float, y-coordinate of the centre of the domain
        pixel_size: float, the length of one pixel
        no_seeds: int, number of seeds through an edge
        added to the structured grid.
        """

        self.no_seeds = no_seeds
        self.nx = nx
        self.ny = ny
        self.cx = cx
        self.cy = cy
        self.pixel_size = pixel_size
        self.mx = (nx - 1)/2
        self.my = (ny - 1)/2
        # Image grid defines the center of every pixel
        x_image, y_image = np.meshgrid(cx + (np.arange(1, nx + 1)
                                       - (nx + 1)/2)*self.pixel_size,
                                       cy + (np.arange(1, ny + 1)
                                       - (ny + 1)/2)*self.pixel_size)
        self.image_grid = np.c_[x_image.flatten(), y_image.flatten()]
        gx = -self.pixel_size/2 + np.append(np.unique(self.image_grid[:, 0]),
                                            self.image_grid[-1, 0] +
                                            self.pixel_size)
        gy = -self.pixel_size/2 + np.append(np.unique(self.image_grid[:, 1]),
                                            self.image_grid[-1, 1] +
                                            self.pixel_size)
        Gx, Gy = np.meshgrid(gx, gy)
        gdx = np.arange(Gx.min(), Gx.max() + self.pixel_size/self.no_seeds,
                        self.pixel_size/self.no_seeds)
        gdy = np.arange(Gy.min(), Gy.max() + self.pixel_size/self.no_seeds,
                        self.pixel_size/self.no_seeds)
        Gdx, Gdy = np.meshgrid(Gx, gdy)
        Gdx2, Gdy2 = np.meshgrid(gdx, Gy)
        grid_points = np.r_[np.column_stack((Gdx.flatten(), Gdy.flatten())),
                            np.column_stack((Gdx2.flatten(), Gdy2.flatten()))]

        sorted_idx = np.lexsort(grid_points.T)
        sorted_grid = grid_points[sorted_idx]
        changes = np.diff(sorted_grid, axis=0)
        changes[abs(changes) < 1e-10] = 0
        row_mask = np.append([True], np.any(changes, axis=1))
        self.grid_1 = sorted_grid[row_mask]

    def assign_model(self, mode, property_map=None, weld_model=None,
                     only_weld=False):
        """
        Assigns material model to the grid. This can either be 'orientations'
        (material orientation specified per pixel), 'slowness' (slowness
        specified per pixel (isotropic only).
        Parameters:
        ---
        mode: string, assigned material model type
              ('orientations'|'slowness')
        property_map: ndarray, map of properties (either orientations or
                      slowness)
        """
        self.mode = mode

        if self.mode == 'orientations':
            self.property_map = property_map
        elif self.mode == 'slowness':
            self.property_map = (1/property_map)**2

    def assign_materials(self, material_map, materials):
        """
        Assign a dictionary mapping material map indices to WaveBasis objects
        and material map assigning the index from materials dictionary to each
        pixel.

        Parameters:
        ---
        material_map: ndarray, map of materials of the shape corresponding to
        the image grid.
        materials: dict, dictionary of materials ({identifier: WaveBasis
                   object})
        """
        self.materials = materials
        self.material_map = material_map

    def add_points(self, sources=None, targets=None):

        """
        Adds additional points (usually sources and receivers) which have
        prescribed locations.

        Parameters:
            sources: ndarray, nx2 array of sources
            receivers: ndarray, nx2 array of receivers
        """

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

    def calculate_graph(self, tie_link=[None, None]):
        """
        Defines the connections between the nodes (graph edges) and calculates
        travel times for each edge.

        Parameters:
        ---
        """
        edges = []
        rows = []
        cols = []

        self.image_tree = cKDTree(self.image_grid)
        self.tree = cKDTree(self.grid)
        self.r_closest = defaultdict(list)
        for pixel in trange(len(self.image_grid)):
            # identify points within a pixel
            points = self.tree.query_ball_point(self.image_grid[pixel],
                                                0.501*self.pixel_size*2**0.5)
            # In case the search circle went outside the pixel, filter out
            take = (abs(self.grid[points] - self.image_grid[pixel])
                    <= self.pixel_size/2*1.001).all(axis=1)
            points = list(np.array(points)[take])
            for point in points:
                neighbours = points[:]
                neighbours.remove(point)
                if len(neighbours) == 0:
                    break
                neighbours = np.array(neighbours)
                dist = (-self.grid[point] + self.grid[neighbours])
                angles = np.arctan2(dist[:, 1], dist[:, 0])
                to_take = np.array([True]*len(angles))
                this_material = self.materials[
                        self.material_map.flatten()[pixel]]
                # If anisotropic, calculate incident angle
                angles = angles[to_take]
                orientation = self.property_map.flatten()[
                    pixel]
                # Calculate group velocity based on the orientation and
                # incident ray angles
                cg = this_material.get_wavespeed(orientation, angles)
                # Calculate cost (time) for edges originating from the current
                # node
                edge_cost = (dist[to_take, 0]**2 + dist[to_take, 1]**2)/cg
                rows_local = [point]*len(neighbours[to_take])
                edges_local = edge_cost**0.5
                rows.extend(rows_local)
                cols.extend(list(neighbours[to_take]))
                edges.extend(edges_local)
        if tie_link[0] is not None and tie_link[1] is not None:
            if len(tie_link[0]) == len(tie_link[1]):
                rows.extend(list(tie_link[0]))
                cols.extend(list(tie_link[1]))
                edges.extend([0]*len(tie_link[0]))
            else:
                print('Tie link misdefined')

        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = coo_matrix((edges, (cols, rows))).transpose().tocsr()
