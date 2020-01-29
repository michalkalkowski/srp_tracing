#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:49:57 2019

@author: michal
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


class RandomGrid:
    """
    Defines a random grid object. In this grid, points are perturbed from the
    original, structured position using a random 'noise' with defined
    properties.
    """
    def __init__(self, nx, ny, cx, cy, pixel_size, nodes_per_pixel, dev):
        """
        Parameters:
        ---
        nx: int, number of pixels along the x direction
        ny: int, number of pixels along the y direction
        cx: float, x-coordinate of the centre of the domain
        cy: float, y-coordinate of the centre of the domain
        pixel_size: float, the length of one pixel
        nodes_per_pixel: int, number of nodes per pixel
        dev: float, standard deviation of the normally distributed 'noise'
        added to the structured grid.
        """

        self.dev = dev
        self.nx = nx
        self.ny = ny
        self.cx = cx
        self.cy = cy
        self.pixel_size = pixel_size
        self.mx = (nx - 1)/2
        self.my = (ny - 1)/2
        self.nodes_per_pixel = nodes_per_pixel
        self.spacing = self.pixel_size/self.nodes_per_pixel
        # Attention!
        # Image grid defines the center of every pixel
        x_image, y_image = np.meshgrid(cx + (np.arange(1, nx + 1)
                                        - (nx + 1)/2)*self.pixel_size,
                                       cy + (np.arange(1, ny + 1)
                                        - (ny + 1)/2)*self.pixel_size)
        self.image_grid = np.c_[x_image.flatten(), y_image.flatten()]

        # Create a structured grid
        x = np.arange((-self.mx - 0.5)*self.pixel_size,
                      (self.mx + 0.5)*self.pixel_size + self.spacing,
                      self.spacing)
        y = np.arange((-self.my - 0.5)*self.pixel_size,
                      (self.my + 0.5)*self.pixel_size + self.spacing,
                      self.spacing)
        X, Y = np.meshgrid(cx + x, cy + y)
        self.structured_gird = np.column_stack((X.flatten(), Y.flatten()))
        self.image_tree = cKDTree(self.image_grid)
        # Add random components
        X += np.random.randn(X.shape[0], X.shape[1])*self.spacing*self.dev
        Y += np.random.randn(Y.shape[0], Y.shape[1])*self.spacing*self.dev
        self.grid_1 = np.column_stack((X.flatten(), Y.flatten()))

    def assign_model(self, mode, property_map=None, weld_model=None,
                     only_weld=False):
        """
        Assigns material model to the grid. This can either be 'orientations'
        (material orientation specified per pixel), 'slowness' (slowness
        specified per pixel (isotropic only), or 'weld' (similar to
        orientations', but additionally the parent plate is not seeded, but a
        number of nodes at weld boundaries are added instead)

        Parameters:
        ---
        mode: string, assigned material model type
              ('orientatiions'|'slowness'|'weld')
        weld_model: object, if mode is 'weld', this should be a relevant
                    MINA model object (or another weld model)
        property_map: ndarray, map of properties (either orientations or
                      slowness)
        only_weld: bool, if True, only weld boundary is discretised and parent
                   plates are assumed isotropic with travel times calculated
                   explicitly - in general not recommended, the computational
                   time is reduced but the accuracy is lower.

        """
        self.mode = mode

        if self.mode == 'weld':
            self.weld = weld_model
            self.property_map = weld_model.grain_orientations[:]
        elif self.mode == 'orientations':
            self.property_map = property_map
        elif self.mode == 'slowness':
            self.property_map = (1/property_map)**2
        self.only_weld = only_weld

        if only_weld and self.mode == 'weld':
            # Create dense nodes at weld boundaries
            weld_boundary_grid = np.arange(-self.weld.c/2, self.weld.c/2
                                           + self.spacing*1e3/20,
                                           self.spacing*1e3/10)
            weld_boundary_grid = weld_boundary_grid[
                    abs(weld_boundary_grid) >= self.weld.b/2]
            bound_length = sum(weld_boundary_grid < 0)
            weld_boundary_z = get_chamfer_profile(weld_boundary_grid,
                                                  self.weld)
            weld_boundary = np.column_stack((weld_boundary_grid/1e3,
                                             weld_boundary_z/1e3))
            to_take = self.grid_1[:, 1] >= get_chamfer_profile(
                self.grid_1[:, 0]*1e3, self.weld)/1e3
            self.grid_1 = np.concatenate((self.grid_1[to_take],
                                          weld_boundary), axis=0)
            # Save left and right boundary nodes indices
            self.boundary_left_idx = list(range(to_take.sum(), to_take.sum()
                                          + bound_length))
            self.boundary_right_idx = list(range(to_take.sum()
                                                 + bound_length,
                                           to_take.sum() + 2*bound_length))

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

    def build_graph(self, jump_level=2):
        """
        Defines the connections between the nodes (graph edges) based on the
        specified jump level.

        Parameters:
        ---
        jump_level: int, a number defining how far a ray from a given node can
                    shoot.
        """
        self.tree = cKDTree(self.grid)
        dists, heights = self.tree.query(self.grid, 2)
        self.avg_distance = np.mean(dists[:, 1])
        self.jump_radius = self.spacing*(jump_level)
        self.r_closest = self.tree.query_ball_point(self.grid,
                                                    self.jump_radius)
        # Calculate which point belongs to which pixel
        _, self.parent_pixel = self.image_tree.query(self.grid, 1)

        # If using the `only_weld` option, assign all weld boundary
        # nodes to closest to start and all targets to all weld boundary nodes
        if self.only_weld:
            self.sources_top_in = []
            self.sources_top_out_left = []
            self.sources_top_out_right = []
            self.sources_bot_in = []
            self.sources_bot_out_left = []
            self.sources_bot_out_right = []
            for source in self.source_idx:
                if np.isclose(self.grid[source, 1], self.weld.a/2e3):
                    if self.grid[source, 0] > self.weld.c/2e3:
                        self.sources_top_out_right.append(source)
                    elif self.grid[source, 0] < -self.weld.c/2e3:
                        self.sources_top_out_left.append(source)
                    else:
                        self.sources_top_in.append(source)
                elif np.isclose(self.grid[source, 1], -self.weld.a/2e3):
                    if self.grid[source, 0] > self.weld.b/2e3:
                        self.sources_bot_out_right.append(source)
                    elif self.grid[source, 0] < -self.weld.b/2e3:
                        self.sources_bot_out_left.append(source)
                    else:
                        self.sources_bot_in.append(source)
            self.targets_top_in = []
            self.targets_top_out_left = []
            self.targets_top_out_right = []
            self.targets_bot_in = []
            self.targets_bot_out_left = []
            self.targets_bot_out_right = []
            for target in self.target_idx:
                if np.isclose(self.grid[target, 1], self.weld.a/2e3):
                    if self.grid[target, 0] > self.weld.c/2e3:
                        self.targets_top_out_right.append(target)
                    elif self.grid[target, 0] < -self.weld.c/2e3:
                        self.targets_top_out_left.append(target)
                    else:
                        self.targets_top_in.append(target)
                elif np.isclose(self.grid[target, 1], -self.weld.a/2e3):
                    if self.grid[target, 0] > self.weld.b/2e3:
                        self.targets_bot_out_right.append(target)
                    elif self.grid[target, 0] < -self.weld.b/2e3:
                        self.targets_bot_out_left.append(target)
                    else:
                        self.targets_bot_in.append(target)
            for source in self.source_idx:
                if source in self.sources_top_out_left:
                    self.r_closest[source] = (self.boundary_left_idx +
                        self.targets_bot_out_left + self.targets_top_out_left
                        + self.targets_top_out_right + self.targets_top_in)
                elif source in self.sources_bot_out_left:
                    self.r_closest[source] = (self.boundary_left_idx +
                        self.targets_top_out_left + self.targets_bot_out_left
                        + self.targets_bot_out_right + self.targets_bot_in)
                elif source in self.sources_bot_out_right:
                    self.r_closest[source] = (self.boundary_right_idx +
                        self.targets_top_out_right + self.targets_bot_out_left
                        + self.targets_bot_out_right + self.targets_bot_in)
                elif source in self.sources_top_out_right:
                    self.r_closest[source] = (self.boundary_right_idx +
                        self.targets_bot_out_right + self.targets_top_out_left
                        + self.targets_top_out_right + self.targets_top_in)
            for target in self.target_idx:
                if target in (self.targets_bot_out_left +
                              self.targets_top_out_left):
                    for boundary_node in self.boundary_left_idx:
                        self.r_closest[boundary_node].append(target)
                elif target in (self.targets_bot_out_right +
                                self.targets_top_out_right):
                    for boundary_node in self.boundary_right_idx:
                        self.r_closest[boundary_node].append(target)

    def set_weld_flag(self, a=0, b=0, c=0):
        """
        Assigns a weld mask to the grid. Used only if `mode` is 'weld'.
        """
        self.weld_flag = np.zeros(len(self.grid), np.uint8)
        if self.mode is 'weld':
            self.weld_flag[self.grid[:, 1] >= get_chamfer_profile(
                self.grid[:, 0]*1e3, self.weld)/1e3] = 1
            _, self.closest_weld_grid = self.weld.tree.query(self.grid*1e3, 1)
            if self.only_weld:
                # if (self.grid[self.source_idx, 0] < 0).all():
                self.weld_flag[self.boundary_right_idx] = 0
                # else:
                self.weld_flag[self.boundary_left_idx] = 0
        else:
            if self.mode == 'orientations':
                self.weld_flag = self.mask.flatten()[self.parent_pixel]

    def add_mask(self, mask):
        """
        Defines a weld mask (array with ones at pixels occupied by the weld and
        zeros elsewhere.

        Parameters:
        mask: ndarray, matrix of zeros and ones (for pixels occupied by the
        weld.
        """

        self.mask = mask

    def calculate_edges(self):
        """
        Calculate graph edges (connections between the points) based on the
        neighbours of the nodes calculated based on a jump radius.
        """

        edges = []
        rows = []
        cols = []
        # Iterate through nodes
        for current_idx in trange(self.grid.shape[0]):
            neighbours = self.r_closest[current_idx]
            # Remove current point from its neighbours
            if current_idx in neighbours:
                neighbours.remove(current_idx)
                self.r_closest[current_idx] = neighbours
            # Calculate distance between the current point all its neighbours
            dist = (- self.grid[current_idx]
                    + self.grid[neighbours])
            # For isotropic models where slowness is specified directly, the
            # calculation is a straightforward (time=distance/group velocity)
            if self.mode == 'slowness':
                edge_cost = (
                    dist[:, 0]**2 + dist[:, 1]**2)/self.orientations.flatten()[
                        self.parent_pixel[current_idx]]
            else:
                # If weld only case is used, everything edge outside of the
                # weld has group velocity independent of the angle
                if self.only_weld and self.weld_flag[current_idx] == 0:
                    this_material = self.materials[0]
                    cg = this_material.wavespeed
                else:
                    # Read properites from the current pixel
                    this_parent_pixel = self.parent_pixel[current_idx]
                    this_material = self.materials[
                        self.material_map.flatten()[this_parent_pixel]]
                    if this_material.anisotropy == 0:
                        # If isotropic, take group velocity directly
                        cg = this_material.wavespeed
                    else:
                        # If anisotropic, calculate incident angle
                        angles = np.arctan2(dist[:, 1], dist[:, 0])
                        if self.only_weld:
                            orientation = self.weld.grain_orientations[
                                self.closest_weld_grid[neighbours]]
                        else:
                            orientation = self.property_map.flatten()[
                                self.parent_pixel[neighbours]]
                        # Calculate group velocity based on the orientation and
                        # incident ray angles
                        cg = this_material.get_wavespeed(orientation, angles)
                # Calculate cost (time) for edges originating from the current
                # node
                edge_cost = (dist[:, 0]**2 + dist[:, 1]**2)/cg
            rows_local = [current_idx]*len(neighbours)
            edges_local = edge_cost**0.5
            rows.extend(rows_local)
            cols.extend(list(neighbours))
            edges.extend(edges_local)
        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = coo_matrix((edges, (cols, rows))).transpose().tocsr()


class RectGrid:
    """
    Defines a random grid object. In this grid, points are perturbed from the
    original, structured position using a random 'noise' with defined
    properties.
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
        #self.nodes_per_pixel = nodes_per_pixel
        #self.spacing = self.pixel_size/self.nodes_per_pixel
        # Attention!
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
        

        # Create a structured grid
        # x = np.arange((-self.mx - 0.5)*self.pixel_size,
        #               (self.mx + 0.5)*self.pixel_size + self.spacing,
        #               self.spacing)
        # y = np.arange((-self.my - 0.5)*self.pixel_size,
        #               (self.my + 0.5)*self.pixel_size + self.spacing,
        #               self.spacing)
        # X, Y = np.meshgrid(cx + x, cy + y)
        # self.structured_gird = np.column_stack((X.flatten(), Y.flatten()))
        # self.image_tree = cKDTree(self.image_grid)
        # # Add random components
        # X += np.random.randn(X.shape[0], X.shape[1])*self.spacing*self.dev
        # Y += np.random.randn(Y.shape[0], Y.shape[1])*self.spacing*self.dev
        # self.grid_1 = np.column_stack((X.flatten(), Y.flatten()))

    def assign_model(self, mode, property_map=None, weld_model=None,
                     only_weld=False):
        """
        Assigns material model to the grid. This can either be 'orientations'
        (material orientation specified per pixel), 'slowness' (slowness
        specified per pixel (isotropic only), or 'weld' (similar to
        orientations', but additionally the parent plate is not seeded, but a
        number of nodes at weld boundaries are added instead)

        Parameters:
        ---
        mode: string, assigned material model type
              ('orientatiions'|'slowness'|'weld')
        weld_model: object, if mode is 'weld', this should be a relevant
                    MINA model object (or another weld model)
        property_map: ndarray, map of properties (either orientations or
                      slowness)
        only_weld: bool, if True, only weld boundary is discretised and parent
                   plates are assumed isotropic with travel times calculated
                   explicitly - in general not recommended, the computational
                   time is reduced but the accuracy is lower.

        """
        self.mode = mode

        if self.mode == 'weld':
            self.weld = weld_model
            self.property_map = weld_model.grain_orientations[:]
        elif self.mode == 'orientations':
            self.property_map = property_map
        elif self.mode == 'slowness':
            self.property_map = (1/property_map)**2
        self.only_weld = only_weld

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

    def calculate_graph(self):
        """
        Defines the connections between the nodes (graph edges) based on the
        specified jump level.

        Parameters:
        ---
        jump_level: int, a number defining how far a ray from a given node can
                    shoot.
        """
        edges = []
        rows = []
        cols = []

        self.tree = cKDTree(self.grid)
        self.r_closest = defaultdict(list)
        for pixel in trange(len(self.image_grid)):
            # identify points within a pixel 
            points = self.tree.query_ball_point(self.image_grid[pixel],
                                                0.501*self.pixel_size*2**0.5)
            for point in points:
                neighbours = points[:]
                neighbours.remove(point)
                neighbours = np.array(neighbours)
                dist = (-self.grid[point] + self.grid[neighbours])
                angles = np.arctan2(dist[:, 1], dist[:, 0])
                to_take = np.array([True]*len(angles))#~(np.isclose(abs(angles) % (np.pi/2), 0))

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
        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = coo_matrix((edges, (cols, rows))).transpose().tocsr()
