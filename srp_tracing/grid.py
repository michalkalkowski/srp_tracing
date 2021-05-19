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
from scipy.sparse import csr_matrix, coo_matrix
import scipy.interpolate as interpolate
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point

import raytracer.refraction as rf

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

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

    def set_tensor_range(self, ranges, rho):
        """
        Defines material properties.

        Parameters:
        ---
        ranges: list of lists with ranges of respective elastic constants
                [[c11_min, c11_max], [c13_min, c13_max],[c33_min, c33_max], [c44_min, c44_max]]
        rho: float, density
        """
        self.ranges = ranges
        self.rho = rho

    def read_tensor_elements(self):
        c11 = self.c[1, 1]
        c13 = self.c[1, 2]
        c33 = self.c[2, 2]
        c44 = self.c[3, 3]
        return c11, c13, c33, c44

    def update_tensor_elements(self, elements):
        self.c[[0, 1], [0, 1]] = elements[0]
        self.c[[0, 1, 2, 2], [2, 2, 0, 1]] = elements[1]
        self.c[2, 2] = elements[2]
        self.c[[3, 4], [3, 4]] = elements[3]
        self.calculate_wavespeeds()

    def calculate_wavespeed_space(self, wave_type=0, angles_from_ray=True):
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

        self.property_map = property_map

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
        for pixel in range(len(self.image_grid)):
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
                if self.mode == 'orientations':
                    angles = angles[to_take]
                    orientation = self.property_map.flatten()[
                        pixel]
                    # Calculate group velocity based on the orientation and
                    # incident ray angles
                    cg = this_material.get_wavespeed(orientation, angles)
                elif self.mode == 'slowness_iso':
                    # self.property_map contains per-cell slowness (isotropic)
                    # Consequently, material properties do not matter that much
                    cg = 1/self.property_map.flatten()[pixel]**2
                else:
                    print('Mode not implemented.')
                    break
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

class ZonesGrid:
    """
    Defines an irregular grid object. The domain is divided into zones, nodes
    of the grid are placed along the boundaries of the zones, with an approximately specified
    seed spacing along each edge. The number of
    seeds links to the angular resolution of the solver.
    """
    def __init__(self, a, b, c, zone_height, max_width_split, max_dx=0.1):
        """
        Parameters:
        ---

        """
        self.a = a
        self.b = b
        self.c = c
        self.max_dx = max_dx
        grad = 2*self.a/(self.c - self.b)
        corners = np.array([[-self.c/2, self.a], [self.c/2, self.a],
                            [self.b/2, 0], [-self.b/2, 0]])
        weld_boundary = Polygon(corners)

        noy = int(self.a//zone_height)
        ny = np.arange(noy + 1)*self.a/noy
        my = ny[:-1] + self.a/noy/2

        splits = np.round(np.linspace(max_width_split, 2, noy + 1)).astype('int')[::-1]
        mx = my/grad + self.b/2
        x_splits = []; y_splits = []
        for i in range(len(mx)):
            x_splits.append(np.linspace(-mx[i] + 2, mx[i] - 2, splits[i]))
            y_splits.append(np.array([my[i]]*len(x_splits[-1])))

        p_regular = np.c_[np.concatenate(x_splits), np.concatenate(y_splits)]
        flucty = np.random.randn(p_regular.shape[0])
        fluctx = np.random.randn(p_regular.shape[0])
        points = p_regular + np.c_[fluctx, flucty]

        vor = Voronoi(points)

        regions, vertices = voronoi_finite_polygons_2d(vor)
        self.zone_polygons = []
        repeated_vertices = []
        for region in regions:
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)
            poly = poly.intersection(weld_boundary)
            self.zone_polygons.append(poly)
            repeated_vertices.append(np.array(poly.exterior.coords))

        edges = []
        global_nodes = np.zeros([0, 2])
        global_nodes = np.concatenate((global_nodes, repeated_vertices[0][:-1]), axis=0)
        # this_perimeter = [[i, i + 1] for i in range(len(global_nodes) - 1)] + [
        #     [len(global_nodes) - 1, 0]]
        # [per.sort() for per in this_perimeter]
        edges = np.zeros([0, 2], 'int')
        #edges = np.concatenate((edges, np.array(this_perimeter)), axis=0)
        # perimeters = [list(range(len(this_perimeter)))]
        perimeters = []

        for reg in range(len(regions)):
            this_perimeter = []
            for i in range(len(repeated_vertices[reg])):
                tester = np.where((repeated_vertices[reg][i].reshape(1, 2)
                                   == global_nodes).all(axis=1))[0]
                if len(tester) == 0:
                    global_nodes = np.concatenate((global_nodes,
                                                   repeated_vertices[reg][i].reshape(1, 2)),
                                                  axis=0)
                    this_perimeter.append(global_nodes.shape[0] - 1)
                else:
                    which_point = tester[0]
                    this_perimeter.append(which_point)
            this_edges = [[this_perimeter[xx], this_perimeter[xx + 1]]
                          for xx in range(len(this_perimeter) - 1)]
            [per.sort() for per in this_edges]
            this_shape = []
            for edge in this_edges:
                tester = np.where((edge  == edges).all(axis=1))[0]
                if len(tester) == 0:
                    edges = np.concatenate((edges, np.array(edge).reshape(1 ,2)),
                                           axis=0)
                    this_shape.append(edges.shape[0] - 1)
                else:
                    which_edge = tester[0]
                    this_shape.append(which_edge)
            perimeters.append(this_shape)

        # seeding edges
        grid_edges = np.zeros([0, 3])

        for ix, edge in enumerate(edges):
            this_edge = global_nodes[edge]
            l = np.linalg.norm(this_edge[1] - this_edge[0])
            new_dx = l/np.ceil(l/self.max_dx)
            no_div = int(np.round(l/new_dx))
            angle = np.arctan2(this_edge[1, 1] - this_edge[0, 1],
                               this_edge[1, 0] - this_edge[0, 0])
            direction = (np.array([np.cos(angle), np.sin(angle)]))
            a1 = np.linspace(this_edge[0, 0] + direction[0]*new_dx/2,
                             this_edge[1, 0] - direction[0]*new_dx/2, no_div + 1)
            a2 = np.linspace(this_edge[0, 1] + direction[1]*new_dx/2,
                             this_edge[1, 1] - direction[1]*new_dx/2, no_div + 1)
            divd = np.column_stack([a1, a2, np.array(len(a1)*[ix])])
            grid_edges = np.concatenate((grid_edges, divd), axis=0)

        # which edges belong to the chamfer?
        edge_vector = np.diff(global_nodes[edges],
                              axis=1).squeeze()
        left_chamfer_v = np.array([[-self.b/2, 0], [-self.c/2, self.a]])
        self.left_chamfer_ind = np.where(
            np.isclose(0, np.cross(edge_vector,
                                   np.diff(left_chamfer_v, axis=0))))[0]
        right_chamfer_v = np.array([[self.b/2, 0], [self.c/2, self.a]])
        self.right_chamfer_ind = np.where(
            np.isclose(0, np.cross(edge_vector,
                                   np.diff(right_chamfer_v, axis=0))))[0]


        # Image grid defines the center of every pixel
        # x_image, y_image = np.meshgrid(cx + (np.arange(1, nx + 1)
        #                                - (nx + 1)/2)*self.pixel_size,
        #                                cy + (np.arange(1, ny + 1)
        #                                - (ny + 1)/2)*self.pixel_size)
        # self.image_grid = np.c_[x_image.flatten(), y_image.flatten()]
        # gx = -self.pixel_size/2 + np.append(np.unique(self.image_grid[:, 0]),
        #                                     self.image_grid[-1, 0] +
        #                                     self.pixel_size)
        # gy = -self.pixel_size/2 + np.append(np.unique(self.image_grid[:, 1]),
        #                                     self.image_grid[-1, 1] +
        #                                     self.pixel_size)
        # Gx, Gy = np.meshgrid(gx, gy)
        # gdx = np.arange(Gx.min(), Gx.max() + self.pixel_size/self.no_seeds,
        #                 self.pixel_size/self.no_seeds)
        # gdy = np.arange(Gy.min(), Gy.max() + self.pixel_size/self.no_seeds,
        #                 self.pixel_size/self.no_seeds)
        # Gdx, Gdy = np.meshgrid(Gx, gdy)
        # Gdx2, Gdy2 = np.meshgrid(gdx, Gy)
        # grid_points = np.r_[np.column_stack((Gdx.flatten(), Gdy.flatten())),
        #                     np.column_stack((Gdx2.flatten(), Gdy2.flatten()))]

        # sorted_idx = np.lexsort(grid_points.T)
        # sorted_grid = grid_points[sorted_idx]
        # changes = np.diff(sorted_grid, axis=0)
        # changes[abs(changes) < 1e-10] = 0
        # row_mask = np.append([True], np.any(changes, axis=1))

        self.grid = grid_edges
        self.global_edges = edges
        self.zones_edge_ind = perimeters
        self.global_nodes = global_nodes

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

        self.property_map = property_map

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
        print('Adding sources and receivers to the grid...')

        self.source_idx = np.zeros([0])
        self.target_idx = np.zeros([0])
        if sources is None:
            self.grid = self.grid
        else:
            edge_vector = np.diff(self.global_nodes[self.global_edges],
                                  axis=1).squeeze()
            nodes_to_add = np.zeros([0, 3])
            left_iso_zone_id = -99
            right_iso_zone_id = -99
            self.left_iso_edge = None
            self.right_iso_edge = None
            self.left_iso_zone = None
            self.right_iso_zone = None
            for s in sources:
                s_1 = s - self.global_nodes[self.global_edges[:, 0]]
                in_line_inds = np.where(np.cross(s_1, edge_vector) == 0)[0]
                edge_flag = ((np.sign(edge_vector[in_line_inds])
                                * np.sign(s_1[in_line_inds]))[:, 0] == 1) \
                        & (abs(s_1[in_line_inds][:, 0])
                           <= abs(edge_vector[in_line_inds][:, 0]))
                parent_edge = in_line_inds[edge_flag]
                if len(parent_edge) != 0:
                    nodes_to_add = np.concatenate((nodes_to_add,
                                                   np.append(s, parent_edge).reshape(1, 3)),
                                                   axis=0)
                else:
                    if s[0] < 0:
                        if left_iso_zone_id == -99:
                            self.left_iso_edge = self.global_edges.shape[0]
                            # Create isotropic zone to the left of the weld
                            left_iso_zone = self.left_chamfer_ind.tolist() + [self.left_iso_edge]
                            self.zones_edge_ind.append(left_iso_zone)
                            left_iso_zone_id = len(self.zones_edge_ind) - 1
                            self.left_iso_zone = left_iso_zone_id
                            nodes_to_add = np.concatenate((
                                nodes_to_add, np.append(s, self.left_iso_edge).reshape(1, 3)),
                                                       axis=0)
                        else:
                            left_edge_id = self.left_iso_edge
                            nodes_to_add = np.concatenate(
                                (nodes_to_add, np.append(s, left_edge_id).reshape(1, 3)),
                                                       axis=0)

                    if s[0] >= 0:
                        if right_iso_zone_id == -99:
                            self.right_iso_edge = self.global_edges.shape[0] + (self.left_iso_edge is not None)
                            # Create isotropic zone to the right of the weld
                            right_iso_zone = self.right_chamfer_ind.tolist() + [self.right_iso_edge]
                            self.zones_edge_ind.append(right_iso_zone)
                            right_iso_zone_id = len(self.zones_edge_ind) - 1
                            self.right_iso_zone = right_iso_zone_id
                            nodes_to_add = np.concatenate(
                                (nodes_to_add, np.append(s, self.right_iso_edge).reshape(1, 3)),
                                                       axis=0)
                        else:
                            right_edge_id = self.right_iso_edge
                            nodes_to_add = np.concatenate(
                                (nodes_to_add, np.append(s, right_edge_id).reshape(1, 3)),
                                                       axis=0)


            self.grid = np.append(self.grid, nodes_to_add, axis=0)
            self.source_idx = np.arange(self.grid.shape[0]
                                        - len(sources),
                                        self.grid.shape[0]).astype(np.int)
        if targets is None:
            self.grid = self.grid
        else:
            edge_vector = np.diff(self.global_nodes[self.global_edges],
                                  axis=1).squeeze()
            nodes_to_add = np.zeros([0, 3])
            for t in targets:
                t_1 = t - self.global_nodes[self.global_edges[:, 0]]
                in_line_inds = np.where(np.cross(t_1, edge_vector) == 0)[0]
                edge_flag = ((np.sign(edge_vector[in_line_inds])
                                * np.sign(t_1[in_line_inds]))[:, 0] == 1) \
                        & (abs(t_1[in_line_inds][:, 0])
                           <= abs(edge_vector[in_line_inds][:, 0]))
                parent_edge = in_line_inds[edge_flag]
                if len(parent_edge) != 0:
                    nodes_to_add = np.concatenate((nodes_to_add,
                                                   np.append(t, parent_edge).reshape(1, 3)),
                                                   axis=0)
                else:
                    if t[0] < 0:
                        if left_iso_zone_id == -99:
                            self.left_iso_edge = self.global_edges.shape[0] + (self.right_iso_edge is not None)
                            # Create isotropic zone to the left of the weld
                            left_iso_zone = self.left_chamfer_ind.tolist() + [self.left_iso_edge]
                            self.zones_edge_ind.append(left_iso_zone)
                            left_iso_zone_id = len(self.zones_edge_ind) - 1
                            self.left_iso_zone = left_iso_zone_id
                            nodes_to_add = np.concatenate((nodes_to_add,
                                                   np.append(t, self.left_iso_edge).reshape(1, 3)),
                                                   axis=0)
                        else:
                            left_edge_id = self.left_iso_edge
                            nodes_to_add = np.concatenate((nodes_to_add,
                                                   np.append(t, left_edge_id).reshape(1, 3)),
                                                   axis=0)
                    if t[0] > 0:
                        if right_iso_zone_id == -99:
                            self.right_iso_edge = self.global_edges.shape[0] + (self.left_iso_edge is not None)
                            # Create isotropic zone to the right of the weld
                            right_iso_zone = self.right_chamfer_ind.tolist() + [self.right_iso_edge]
                            self.zones_edge_ind.append(right_iso_zone)
                            right_iso_zone_id = len(self.zones_edge_ind) - 1
                            self.right_iso_zone = right_iso_zone_id
                            nodes_to_add = np.concatenate((nodes_to_add,
                                                   np.append(t, self.right_iso_edge).reshape(1, 3)),
                                                   axis=0)
                        else:
                            right_edge_id = self.right_iso_edge
                            nodes_to_add = np.concatenate((nodes_to_add,
                                                   np.append(t, right_edge_id).reshape(1, 3)),
                                                   axis=0)

            self.grid = np.append(self.grid, nodes_to_add, axis=0)
            self.target_idx = np.arange(self.grid.shape[0]
                                        - len(targets),
                                        self.grid.shape[0]).astype(np.int)

        # Calculate zone centroids
        self.zone_centroids = np.zeros([len(self.zones_edge_ind), 2])
        for z in range(len(self.zones_edge_ind)):
            if z != left_iso_zone_id and z != right_iso_zone_id:
                cent_point = self.global_nodes[
                    self.global_edges[
                        self.zones_edge_ind[z]]].mean(axis=(0, 1))
                self.zone_centroids[z] = cent_point
        self.active = np.array(len(self.zones_edge_ind)*[True])
        dont_update = []
        if self.left_iso_zone is not None:
            dont_update.append(self.left_iso_zone)
        if self.right_iso_zone is not None:
            dont_update.append(self.right_iso_zone)
        self.active[dont_update] = False


    def calculate_graph(self, tie_link=[None, None]):
        """
        Defines the connections between the nodes (graph edges) and calculates
        travel times for each edge.

        Parameters:
        ---
        """
        print('Calculating graph edges...')
        edges = []
        rows = []
        cols = []
        distances = []
        zone_labels = []
        glob_angles = []
        for this_zone in trange(len(self.zones_edge_ind)):
            # pull grid_1 indices from the edges
            indices = []
            for local_edge in self.zones_edge_ind[this_zone]:
                indices.append(np.where(self.grid[:, 2] == local_edge)[0])
            indices = np.concatenate(indices)
            local_grid = self.grid[indices]
            # Calculate distance vector
            r = -local_grid[:, :2][:, np.newaxis, :] + local_grid[:, :2][np.newaxis, :, :]
            dist = np.linalg.norm(r, axis=2)
            angles = np.arctan2(r[:, :, 1], r[:, :, 0])
            # Reject connecting to the same node
            # bangles = angles[dist != 0]
            # dist = dist[dist != 0]
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[self.material_map[this_zone]]
            # If anisotropic, calculate incident angle
            if self.mode == 'orientations':
                orientation = self.property_map[this_zone]
                # Calculate group velocity based on the orientation and
                # incident ray angles
                cg = this_material.get_wavespeed(orientation, angles)
            elif self.mode == 'slowness_iso':
                # self.property_map contains per-cell slowness (isotropic)
                # Consequently, material properties do not matter that much
                cg = 1/self.property_map[this_zone]**2
            else:
                print('Mode not implemented.')
                break
                # Calculate cost (time) for edges originating from the current
                # node
            edge_cost = dist/cg**0.5
            
            temp_col, temp_row = np.meshgrid(indices, indices)
            mask = (temp_col != temp_row).flatten()
            col_indices = list(temp_col[temp_col != temp_row])
            row_indices = list(temp_row[temp_row != temp_col])
            edge_cost = edge_cost.reshape(temp_col.shape)[temp_col != temp_row]
            rows.extend(row_indices)
            cols.extend(col_indices)
            edges.extend(edge_cost)
            distances.extend(dist[mask])
            zone_labels.extend(np.array([this_zone]*len(row_indices)))
            glob_angles.extend(angles[mask])
        if tie_link[0] is not None and tie_link[1] is not None:
            if len(tie_link[0]) == len(tie_link[1]):
                rows.extend(list(tie_link[0]))
                cols.extend(list(tie_link[1]))
                edges.extend([0]*len(tie_link[0]))
            else:
                print('Tie link misdefined')

        print('Saving edge cost matrices...')
        # Create a sparse matrix of graph edge lengths (times of flight)
        # self.edges = csr_matrix((edges, (rows, cols)))
        # self.distances = csr_matrix((distances, (rows, cols)))
        # self.zone_labels = csr_matrix((zone_labels, (rows, cols)))
        # self.ray_angles = csr_matrix((glob_angles, (rows, cols)))
        self.edges = np.zeros([temp_row.max() + 1, temp_col.max() + 1])
        self.edges[rows, cols] = edges

        self.zone_labels = np.zeros([temp_row.max() + 1, temp_col.max() + 1])
        self.zone_labels[rows, cols] = zone_labels
        self.distances = np.zeros([temp_row.max() + 1, temp_col.max() + 1])
        self.distances[rows, cols] = distances
        self.ray_angles = np.zeros([temp_row.max() + 1, temp_col.max() + 1])
        self.ray_angles[rows, cols] = glob_angles


class SimplRectGrid:
    """
    Defines a simplified rectangular grid object. The domain is divided into pixels, nodes
    of the grid are placed along the boundaries of the pixels, with a specified
    number of seeds per pixel boundary (each side of the square). The number of
    seeds defines the angular resolution of the solver. The isotropic regions are not discretised
    but taken as one cell depending on the neighbourhood.
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

    def trim_to_chamfer(self, a, b, c, mirror_domain=False):
        weld_angle = np.arctan((c - b)/2/a)
        
        left_chamfer = np.array([[-c/2, a], [-b/2, 0]])
        node_spacing = self.pixel_size/self.no_seeds
        seeds_per_chamfer = int(np.linalg.norm(left_chamfer[1] -
                                               left_chamfer[0])/node_spacing)
        seed_x = np.linspace(-c/2, -b/2, seeds_per_chamfer + 1)
        seed_y = a - (seed_x - (-c/2))/np.tan(weld_angle)
        if not mirror_domain:
            take_left = (self.grid_1[:, 1] + node_spacing*1e-6
                           > np.tan(np.pi/2 + weld_angle)*(self.grid_1[:, 0] + b/2))
            take_right = (self.grid_1[:, 1] + node_spacing*1e-6
                           > np.tan(np.pi/2 - weld_angle)*(self.grid_1[:, 0] - b/2))
        else:
            take_left = (abs(self.grid_1[:, 1]) + node_spacing*1e-6
                           > np.tan(np.pi/2 + weld_angle)*(self.grid_1[:, 0] + b/2))
            take_right = (abs(self.grid_1[:, 1]) + node_spacing*1e-6
                           > np.tan(np.pi/2 - weld_angle)*(self.grid_1[:, 0] - b/2))
        take = take_left & take_right
        first_new_idx = self.grid_1[take].shape[0]
        if not mirror_domain:
            self.grid_1 = np.concatenate((self.grid_1[take], np.column_stack((seed_x, seed_y)),
                                   np.column_stack((-seed_x, seed_y))), axis=0)
            self.left_iso_zone = np.arange(first_new_idx, first_new_idx + seed_x.shape[0])
            self.right_iso_zone = np.arange(first_new_idx + seed_x.shape[0],
                                            first_new_idx + 2*seed_x.shape[0])
        else:
            self.grid_1 = np.concatenate((self.grid_1[take], np.column_stack((seed_x, seed_y)),
                                   np.column_stack((seed_x, -seed_y)), np.column_stack((-seed_x, seed_y)),
                                        np.column_stack((-seed_x, -seed_y))), axis=0)
            self.left_iso_zone = np.arange(first_new_idx, first_new_idx + 2*seed_x.shape[0])
            self.right_iso_zone = np.arange(first_new_idx + 2*seed_x.shape[0],
                                            first_new_idx + 4*seed_x.shape[0])
        
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

        self.property_map = property_map

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

    def simplify_grid(self):
        """
        Simplifes the grid be keeping only the cells covering the weld. The isotropic areas
        to the left and to the right will be  modelled with straight rays.

        """
        new_grid = []
        new_pixels = []
        tree = cKDTree(self.grid_1)
        mat_flat = self.material_map.flatten()
        self.image_grid_lookup = np.zeros(self.image_grid.shape[0], 'int') - 1
        cnt = 0
        for pixel in range(len(self.image_grid)):
            if mat_flat[pixel] != 0:
                points = tree.query_ball_point(self.image_grid[pixel], 0.501*self.pixel_size*2**0.5)
                take = (abs(self.grid_1[points] - self.image_grid[pixel]) <= self.pixel_size/2*1.001).all(axis=1)
                new_grid.extend(list(np.array(points)[take]))
                new_pixels.append(pixel)
                self.image_grid_lookup[pixel] = cnt
                cnt += 1
        #trim_grid = self.grid_1[new_grid]
        #ssi = np.lexsort((trim_grid[:, 0], trim_grid[:, 1]))
        #translator = np.zeros(max(new_grid) + 1, dtype='int')
        #translator[new_grid] = np.arange(len(new_grid))

        TOL = 1e-6
        # d = np.append(True, np.diff(trim_grid[ssi][:, 1]))
        # self.left_iso_zone = translator[np.array(new_grid)[ssi][d>TOL]]
        # ssi = np.lexsort((-trim_grid[:, 0], trim_grid[:, 1]))
        # d = np.append(True, np.diff(trim_grid[ssi][:, 1]))
        # self.right_iso_zone = translator[np.array(new_grid)[ssi][d>TOL]]

#        self.grid_1 = trim_grid
        self.image_grid_trim = np.array(self.image_grid[new_pixels])
        self.image_tree_trim = cKDTree(self.image_grid_trim)
        self.grid_tree_trim = cKDTree(self.grid_1)

    def add_points(self, points=None, sources=None, targets=None):

        """
        Adds additional points (usually sources and receivers) which have
        prescribed locations.

        Parameters:
            sources: ndarray, nx2 array of sources
            receivers: ndarray, nx2 array of receivers
        """
        to_add = []
        indices = []
        cnt = 0
        if points is None:
            self.grid = self.grid_1
        else:
            for i in range(len(points)):
                dist, ix = self.grid_tree_trim.query(points[i])
                if dist > 0.0:
                    to_add.append(points[i])
                    indices.append([i, self.grid_1.shape[0] + cnt])
                    cnt += 1
                else:
                    indices.append(i, ix)
            self.grid = np.concatenate((self.grid_1, np.row_stack(to_add)), axis=0)
            indices = np.row_stack(indices)

            self.source_idx = indices[np.in1d(indices[:, 0], sources), 1]
            self.target_idx = indices[np.in1d(indices[:, 0], targets), 1]
            added_points_idx = np.arange(self.grid_1.shape[0], self.grid_1.shape[0] + len(to_add))
        # Check which sources are in the homogeneous regios
        dd, _ = self.image_tree_trim.query(self.grid[added_points_idx])
        outside = dd > 0.5*2**0.5*self.pixel_size
        self.left_iso_zone = np.append(self.left_iso_zone,
                                       added_points_idx[outside][self.grid[added_points_idx][outside][:, 0] < 0])
        self.right_iso_zone = np.append(self.right_iso_zone,
                                       added_points_idx[outside][self.grid[added_points_idx][outside][:, 0] > 0])

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
        for full_pixel in trange(len(self.image_grid)):
            if self.image_grid_lookup[full_pixel] == -1:
                continue
            else:
                pixel = self.image_grid_lookup[full_pixel]

            # identify points within a pixel
            points = self.tree.query_ball_point(self.image_grid_trim[pixel],
                                                0.501*self.pixel_size*2**0.5)
            # In case the search circle went outside the pixel, filter out
            take = (abs(self.grid[points] - self.image_grid_trim[pixel])
                    <= self.pixel_size/2*1.001).all(axis=1)
            if len(np.array(points)[take]) == 0:
                print('no points')
                continue
                
            local_grid = self.grid[(np.array(points)[take])]
            points = np.array(points)
            # Calculate distance vector
            r = -local_grid[:, :2][:, np.newaxis, :] + local_grid[:, :2][np.newaxis, :, :]
            dist = np.linalg.norm(r, axis=2)
            angles = np.arctan2(r[:, :, 1], r[:, :, 0])
            # Reject connecting to the same node
            # bangles = angles[dist != 0]
            # dist = dist[dist != 0]
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[self.material_map.flatten()[
                    full_pixel]]
            # If anisotropic, calculate incident angle
            if self.mode == 'orientations':
                orientation = self.property_map.flatten()[
                    full_pixel]
                # Calculate group velocity based on the orientation and
                # incident ray angles
                cg = this_material.get_wavespeed(orientation, angles)
            elif self.mode == 'slowness_iso':
                # self.property_map contains per-cell slowness (isotropic)
                # Consequently, material properties do not matter that much
                cg = 1/self.property_map.flatten()[full_pixel]**2
            else:
                print('Mode not implemented.')
                break


#            edge_cost = (dist[to_take, 0]**2 + dist[to_take, 1]**2)/cg
            edge_cost = dist/cg**0.5
            temp_col, temp_row = np.meshgrid(points[take], points[take])
            mask = (temp_col != temp_row).flatten()
            col_indices = list(temp_col[temp_col != temp_row])
            row_indices = list(temp_row[temp_row != temp_col])
            edge_cost = edge_cost.reshape(temp_col.shape)[temp_col != temp_row]
            rows.extend(row_indices)
            cols.extend(col_indices)
            edges.extend(edge_cost)
            #distances.extend(dist[mask])
        # Add left homogeneous zone
        if self.left_iso_zone is not None:
            local_grid = self.grid[self.left_iso_zone]
            # Calculate distance vector
            r = -local_grid[:, :2][:, np.newaxis, :] + local_grid[:, :2][np.newaxis, :, :]
            dist = np.linalg.norm(r, axis=2)
            angles = np.arctan2(r[:, :, 1], r[:, :, 0])
            # Reject connecting to the same node
            # bangles = angles[dist != 0]
            # dist = dist[dist != 0]
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[
                    self.material_map[self.ny//2, 0]]
            # If anisotropic, calculate incident angle
            if self.mode == 'orientations':
                orientation = self.property_map[self.ny//2, 0]
                # Calculate group velocity based on the orientation and
                # incident ray angles
                cg = this_material.get_wavespeed(orientation, angles)
            elif self.mode == 'slowness_iso':
                # self.property_map contains per-cell slowness (isotropic)
                # Consequently, material properties do not matter that much
                cg = 1/self.property_map[self.ny//2, 0]**2
            else:
                print('Mode not implemented.')

#            edge_cost = (dist[to_take, 0]**2 + dist[to_take, 1]**2)/cg
            edge_cost = dist/cg**0.5
            
            temp_col, temp_row = np.meshgrid(self.left_iso_zone, self.left_iso_zone)
            mask = (temp_col != temp_row).flatten()
            col_indices = list(temp_col[temp_col != temp_row])
            row_indices = list(temp_row[temp_row != temp_col])
            edge_cost = edge_cost.reshape(temp_col.shape)[temp_col != temp_row]
            rows.extend(row_indices)
            cols.extend(col_indices)
            edges.extend(edge_cost)
            #distances.extend(dist[mask])

        # Add right homogeneous zone
        if self.right_iso_zone is not None:
            local_grid = self.grid[self.right_iso_zone]
            # Calculate distance vector
            r = -local_grid[:, :2][:, np.newaxis, :] + local_grid[:, :2][np.newaxis, :, :]
            dist = np.linalg.norm(r, axis=2)
            angles = np.arctan2(r[:, :, 1], r[:, :, 0])
            # Reject connecting to the same node
            # bangles = angles[dist != 0]
            # dist = dist[dist != 0]
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[
                    self.material_map[self.ny//2, -1]]
            # If anisotropic, calculate incident angle
            if self.mode == 'orientations':
                orientation = self.property_map[self.ny//2, -1]
                # Calculate group velocity based on the orientation and
                # incident ray angles
                cg = this_material.get_wavespeed(orientation, angles)
            elif self.mode == 'slowness_iso':
                # self.property_map contains per-cell slowness (isotropic)
                # Consequently, material properties do not matter that much
                cg = 1/self.property_map[self.ny//2, -1]**2
            else:
                print('Mode not implemented.')

#            edge_cost = (dist[to_take, 0]**2 + dist[to_take, 1]**2)/cg
            edge_cost = dist/cg**0.5
            
            temp_col, temp_row = np.meshgrid(self.right_iso_zone, self.right_iso_zone)
            mask = (temp_col != temp_row).flatten()
            col_indices = list(temp_col[temp_col != temp_row])
            row_indices = list(temp_row[temp_row != temp_col])
            edge_cost = edge_cost.reshape(temp_col.shape)[temp_col != temp_row]
            rows.extend(row_indices)
            cols.extend(col_indices)
            edges.extend(edge_cost)
            #distances.extend(dist[mask])

        if tie_link[0] is not None and tie_link[1] is not None:
            if len(tie_link[0]) == len(tie_link[1]):
                rows.extend(list(tie_link[0]))
                cols.extend(list(tie_link[1]))
                edges.extend([0]*len(tie_link[0]))
            else:
                print('Tie link misdefined')

        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = coo_matrix((edges, (cols, rows))).transpose().tocsr()



            # Calculate distance vector
