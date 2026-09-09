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
from typing import Any, Optional
import warnings
import numpy as np
from tqdm import trange
from itertools import combinations
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, coo_matrix
import scipy.interpolate as interpolate
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from ._wave_physics import calculate_slowness

# Geometric tolerances shared across grid construction/graph-building below.
# Grouped and named here (rather than left as scattered literals) because a
# divergence between two sites that should agree was a real, hard-to-find
# bug: RectGrid/SimplRectGrid's per-pixel neighbour searches must all use
# the same radius/boundary factors, or "which points belong to this pixel"
# silently differs between methods (see the tidy-up plan history).
#
# Multiplier on pixel_size*sqrt(2) (the pixel's half-diagonal) for the
# radius of the search that finds a pixel's own boundary nodes; slightly
# over 0.5 so nodes exactly on the boundary are reliably included despite
# floating point error, without reaching into the next pixel.
PIXEL_SEARCH_RADIUS_FACTOR = 0.501
# Multiplier on pixel_size/2 for the bounding-box check that filters the
# above search back down to points actually belonging to this pixel.
PIXEL_BOUNDARY_TOL_FACTOR = 1.001
# Coordinates are rounded to this many places (via round(SCALE*coord)) before
# lexsort-based point deduplication, so that two floating point
# representations of "the same point" sort adjacently.
GRID_DEDUP_ROUND_SCALE = 1e8
# Below this, two *rounded* coordinates (see GRID_DEDUP_ROUND_SCALE) are
# treated as the same point during deduplication.
GRID_DEDUP_TOL = 1e-10
# atol for np.isclose when testing whether a point on the weld chamfer
# boundary lies exactly on an interpolated ray between two other boundary
# points (SimplRectGrid's isotropic-zone edge wiring).
CHAMFER_BOUNDARY_MATCH_TOL = 1e-8


def voronoi_finite_polygons_2d(vor: Voronoi,
                               radius: Optional[float] = None) -> tuple[list, np.ndarray]:
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
        # np.ptp(vor.points) (no axis) matches the old vor.points.ptp()
        # ndarray method, removed in NumPy 2.0; the trailing .max() in the
        # original was a no-op (ptp with no axis already returns a scalar).
        radius = np.ptp(vor.points)*2

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


def dedup_min_edge_indices(rows: np.ndarray, cols: np.ndarray,
                           edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For (row, col, cost) triples where the same (row, col) pair can appear
    more than once, finds the minimum-cost occurrence of each unique pair.

    Two nodes that both sit on the shared boundary between adjacent
    pixels/zones get discovered independently from each side of that
    boundary, so the same (row, col) pair can be added more than once.
    Taking the minimum reflects that a shortest-path search only ever wants
    the cheapest available connection between two nodes anyway (the natural
    alternative -- coo_matrix's default behaviour of summing duplicate
    entries -- would silently multiply-count that edge's cost instead).

    Parameters:
    ---
    rows, cols: ndarray, node indices the edge runs from/to
    edges: ndarray, edge cost (time of flight) for each (rows[i], cols[i])

    Returns:
    ---
    unique_rows, unique_cols: ndarray, one entry per distinct (row, col) pair
    keep: ndarray, index into the *original* rows/cols/edges (and any other
          same-length per-edge array, e.g. distances or angles) of the
          minimum-cost occurrence of each pair in unique_rows/unique_cols --
          use this to select consistent values from parallel arrays.
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    edges = np.asarray(edges)
    # sort by row, then col, then (ascending) edge cost, so the first entry
    # in each (row, col) group is the minimum-cost one
    order = np.lexsort((edges, cols, rows))
    sorted_rows, sorted_cols = rows[order], cols[order]
    is_new_pair = np.ones(len(sorted_rows), dtype=bool)
    is_new_pair[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (sorted_cols[1:] != sorted_cols[:-1])
    return sorted_rows[is_new_pair], sorted_cols[is_new_pair], order[is_new_pair]


def build_edge_matrix(rows: np.ndarray, cols: np.ndarray,
                      edges: np.ndarray) -> csr_matrix:
    """
    Builds a sparse graph-edge-cost matrix from (row, col, cost) triples,
    keeping the minimum cost for any (row, col) pair that appears more than
    once (see dedup_min_edge_indices).

    Parameters:
    ---
    rows, cols: ndarray, node indices the edge runs from/to
    edges: ndarray, edge cost (time of flight) for each (rows[i], cols[i])

    Returns:
    ---
    csr_matrix, edges[i, j] = shortest known cost from node i to node j
    """
    edges = np.asarray(edges)
    unique_rows, unique_cols, keep = dedup_min_edge_indices(rows, cols, edges)
    return coo_matrix((edges[keep], (unique_rows, unique_cols))).tocsr()


def cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Scalar (z-component) cross product of 2D vectors: a[..., 0]*b[..., 1] -
    a[..., 1]*b[..., 0]. Equivalent to the old np.cross(a, b) for
    last-axis-length-2 inputs, which NumPy 2.0 deprecates in favour of
    spelling it out explicitly.
    """
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def build_boundary_seed_grid_points(nx: int, ny: int, cx: float, cy: float,
                                    pixel_size: float,
                                    no_seeds: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Shared by RectGrid.__init__ and SimplRectGrid.__init__: builds the
    pixel image grid (one point per pixel centre) and the raw, not yet
    deduplicated, set of candidate boundary-seed points along every pixel
    edge. Each caller does its own point deduplication afterwards -- the
    two classes' dedup passes differ in ways that affect stored floating
    point precision, so unifying them isn't a safe pure refactor and is
    left alone.

    Returns:
    ---
    image_grid: ndarray, one (x, y) row per pixel centre
    grid_points: ndarray, candidate boundary-seed points (not yet deduplicated)
    """
    x_image, y_image = np.meshgrid(cx + (np.arange(1, nx + 1)
                                   - (nx + 1)/2)*pixel_size,
                                   cy + (np.arange(1, ny + 1)
                                   - (ny + 1)/2)*pixel_size)
    image_grid = np.c_[x_image.flatten(), y_image.flatten()]
    gx = -pixel_size/2 + np.append(np.unique(image_grid[:, 0]),
                                   image_grid[-1, 0] + pixel_size)
    gy = -pixel_size/2 + np.append(np.unique(image_grid[:, 1]),
                                   image_grid[-1, 1] + pixel_size)
    Gx, Gy = np.meshgrid(gx, gy)
    gdx = np.arange(Gx.min(), Gx.max() + pixel_size/no_seeds, pixel_size/no_seeds)
    gdy = np.arange(Gy.min(), Gy.max() + pixel_size/no_seeds, pixel_size/no_seeds)
    Gdx, Gdy = np.meshgrid(Gx, gdy)
    Gdx2, Gdy2 = np.meshgrid(gdx, Gy)
    grid_points = np.r_[np.column_stack((Gdx.flatten(), Gdy.flatten())),
                        np.column_stack((Gdx2.flatten(), Gdy2.flatten()))]
    return image_grid, grid_points


def get_chamfer_profile(y: np.ndarray, weld: Any) -> np.ndarray:
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
    def __init__(self, anisotropy: int = 0, velocity_variant: str = 'group') -> None:
        self.anisotropy = anisotropy
        self.variant = velocity_variant

    def set_material_props(self, c: np.ndarray, rho: float) -> None:
        """
        Defines material properties.

        Parameters:
        ---
        c: ndarray, elasticity matrix (6x6, Voigt notation)
        rho: float, density

        Units: srp_tracing works throughout in millimetres and microseconds
        (coordinates and pixel_size are mm; get_wavespeed_squared must
        return (mm/us)**2). Giving c in GPa and rho in g/cm**3 makes
        sqrt(c/rho) come out directly in mm/us with no extra conversion
        factor -- a standard convention in ultrasonics, and the one the
        FE-validated examples use (see tests/test_ogilvy_validation.py).
        Scaling c and rho by *different* powers of ten from there (rather
        than genuinely converting both to another consistent unit system)
        is a common mistake; calculate_wavespeeds warns if the resulting
        velocity is implausible for a solid.
        """
        self.c = c
        self.rho = rho

    def calculate_wavespeeds(self, wave_type: int = 0, angles_from_ray: bool = True) -> None:
        """
        Calculates group velocities and their interpolators.

        Parameters:
        wave_type: int, wave type of interest; 0 - P, 1 - SH, 2 - SV
        """
        angles = np.linspace(0, 2*np.pi, 200)
        basis = calculate_slowness(self.c, self.rho, angles)
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
            raise ValueError(
                f'Unknown velocity_variant: {self.variant!r}; must be "phase" or "group"')

        if self.anisotropy == 0:
            self.wavespeed = self.int_cg2(0)

        # Sanity check for the unit convention documented in
        # set_material_props: catches the common mistake of scaling c and
        # rho by different powers of ten (giving an absurd velocity) rather
        # than silently producing nonsense travel times downstream.
        representative_speed = float(np.sqrt(self.int_cg2(0)))
        if not (0.1 < representative_speed < 20):
            warnings.warn(
                f"WaveBasis: computed wavespeed {representative_speed:.4g} mm/us "
                "is outside the plausible range for a solid (~0.1-20 mm/us). "
                "This usually means c and rho were not given in consistent "
                "units -- see set_material_props's docstring.",
                stacklevel=2)

    def get_wavespeed_squared(self, orientation: float | np.ndarray = 0,
                              direction: float | np.ndarray = 0) -> float | np.ndarray:
        """
        Calculates the squared wavespeed for the given material orientation
        and direction of the incoming wave in model coordinates. Squared,
        rather than the wavespeed itself, because that is what every caller
        in this module actually needs (edge_cost = distance**2 / cg2).

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

    def get_wavespeed(self, orientation: float | np.ndarray = 0,
                      direction: float | np.ndarray = 0) -> float | np.ndarray:
        """
        Deprecated alias for get_wavespeed_squared -- despite the name, this
        returns velocity squared, not velocity. Kept for backwards
        compatibility with existing external callers.
        """
        warnings.warn(
            "WaveBasis.get_wavespeed() returns velocity squared and has "
            "been renamed to get_wavespeed_squared() accordingly; "
            "get_wavespeed() will be removed in a future version.",
            DeprecationWarning, stacklevel=2)
        return self.get_wavespeed_squared(orientation, direction)


class _ModeDependentMaterial:
    """
    Shared behaviour for RectGrid/ZonesGrid/SimplRectGrid: how a grid's
    per-cell "mode" (set via assign_model) turns a material and a
    property_map entry into an edge's squared wavespeed. Pulled out here
    because the mode dispatch itself was duplicated near-verbatim across
    about a dozen call sites across the three classes below.
    """

    def assign_model(self, mode: str, property_map: Optional[np.ndarray] = None,
                     weld_model: Any = None, only_weld: bool = False) -> None:
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

    def wavespeed_squared_for(self, material: WaveBasis,
                              property_value: float | np.ndarray,
                              direction: float | np.ndarray) -> float | np.ndarray:
        """
        Squared wavespeed for one cell/pixel/zone, given its material and
        its entry from property_map, following the mode set by
        assign_model(): 'orientations' treats property_value as a grain
        orientation and asks the material for its (possibly
        direction-dependent) wavespeed; 'slowness_iso' treats property_value
        directly as an isotropic slowness, ignoring the material and
        direction (self.property_map contains per-cell slowness in this
        mode, so material properties don't matter beyond having *a*
        WaveBasis assigned).
        """
        if self.mode == 'orientations':
            return material.get_wavespeed_squared(property_value, direction)
        elif self.mode == 'slowness_iso':
            return 1/property_value**2
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    def assign_materials(self, material_map: np.ndarray, materials: dict) -> None:
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


class RectGrid(_ModeDependentMaterial):
    """
    Defines a rectangular grid object. The domain is divided into pixels, nodes
    of the grid are placed along the boundaries of the pixels, with a specified
    number of seeds per pixel boundary (each side of the square). The number of
    seeds defines the angular resolution of the solver.
    """
    def __init__(self, nx: int, ny: int, cx: float, cy: float, pixel_size: float,
                no_seeds: int) -> None:
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
        self.image_grid, grid_points = build_boundary_seed_grid_points(
            nx, ny, cx, cy, pixel_size, no_seeds)

        sorted_idx = np.lexsort(np.round(GRID_DEDUP_ROUND_SCALE*grid_points).T)
        sorted_grid = grid_points[sorted_idx]
        changes = np.diff(sorted_grid, axis=0)
        self.grid_1 = np.vstack((sorted_grid[0], sorted_grid[1:][
            ~(abs(changes) < GRID_DEDUP_TOL).all(axis=1)]))

    def assign_materials(self, material_map: np.ndarray, materials: dict,
                         left_add: Optional[int] = None,
                         right_add: Optional[int] = None) -> None:
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
        if right_add is not None:
            self.original_weld_mask = self.material_map[:, left_add:-right_add]
        else:
            self.original_weld_mask = self.material_map[:, left_add:right_add]

    def add_points(self, sources: Optional[np.ndarray] = None,
                   targets: Optional[np.ndarray] = None) -> None:

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
                                            + len(sources)).astype(np.int64)
            if targets is not None:
                to_add.append(targets)
                self.target_idx = np.arange(self.grid_1.shape[0]
                                            + len(sources),
                                            self.grid_1.shape[0]
                                            + len(sources)
                                            + len(targets)).astype(np.int64)
            if len(to_add) > 0:
                conc = [self.grid_1] + to_add
                self.grid = np.concatenate(conc, axis=0)

    def plot(self) -> None:
        plt.figure()
        plt.plot(self.grid[:, 0], self.grid[:, 1], 'x')
        plt.gca().set_aspect('equal')
        plt.show()

    def set_up_graph(self) -> None:
        rows = []
        cols = []

        self.image_tree = cKDTree(self.image_grid)
        self.tree = cKDTree(self.grid)
        self.r_closest = defaultdict(list)
        self.pixel_type = np.zeros(self.image_grid.shape[0])
        first = True
        self.r_closest = defaultdict(list)
        self.pixels_with_sources = defaultdict(list)
        for pixel in range(len(self.image_grid)):
            # identify points within a pixel
            points = self.tree.query_ball_point(self.image_grid[pixel],
                                                PIXEL_SEARCH_RADIUS_FACTOR*self.pixel_size*2**0.5)
            # In case the search circle went outside the pixel, filter out
            take = (abs(self.grid[points] - self.image_grid[pixel])
                    <= self.pixel_size/2*PIXEL_BOUNDARY_TOL_FACTOR).all(axis=1)
            points = np.array(points)[take]
            if len(points) < 2:
                continue
            
            points = points[np.lexsort(np.round(GRID_DEDUP_ROUND_SCALE*self.grid[points]).T)]
            # now are sorted from slow y fast x from bottom to top, from left to right
            self.r_closest[pixel] = points
            if points.shape[0] == self.no_seeds*4:
                self.pixel_type[pixel] = 0
                # Standard cell:
                if first is True:
                    # pre calculate angles and distances
                    local_ind = np.arange(points.shape[0])
                    pairs = np.array(list(combinations(local_ind, 2)))
                    local_edges = self.grid[points][pairs, :]
                    dist = -local_edges[:, 0] + local_edges[:, 1]
                    self.angles = np.arctan2(dist[:, 1], dist[:, 0])
                    self.travel_d_squared = (dist**2).sum(axis=1)
                    self.pairs = pairs
                    self.row_pairs = pairs.flatten('F')
                    self.col_pairs = pairs[:, ::-1].flatten('F')
                    first = False
                row_pairs = self.row_pairs
                col_pairs = self.col_pairs
            else:
                self.pixel_type[pixel] = 1
                local_ind = np.arange(points.shape[0])
                pairs = np.array(list(combinations(local_ind, 2)))
                local_edges = self.grid[points][pairs, :]
                dist = -local_edges[:, 0] + local_edges[:, 1]
                angles = np.arctan2(dist[:, 1], dist[:, 0])
                travel_d_squared = (dist**2).sum(axis=1)
                self.pixels_with_sources[pixel] = dict([('angles', angles),
                                                        ('travel_d_squared', travel_d_squared),
                                                        ('pairs', pairs)])
                row_pairs = pairs.flatten('F')
                col_pairs = pairs[:, ::-1].flatten('F')

            rows.extend(points[row_pairs])
            cols.extend(points[col_pairs])
        self.rows = np.array(rows)
        self.cols = np.array(cols)
        # if there is an isotropic material in the domain *and* at least one
        # standard cell was found (first was flipped to False below), precompute
        # ToFs for a standard cell, using that one material as the reference for
        # the update_edges shortcut below. If every pixel turned out irregular
        # (e.g. a source/target lands in every pixel of a small grid), there is
        # no standard cell for iso_tofs to describe, and it is never read
        # anyway (update_edges only consults it when pixel_type == 0).
        mats = [mat.anisotropy for k, mat in self.materials.items()]
        self._iso_shortcut_material = None
        if mats.count(0) > 0 and not first:
            self._iso_shortcut_material = self.materials[mats.index(0)]
            cg_iso = self._iso_shortcut_material.get_wavespeed_squared(0, 0)
            self.iso_tofs = (self.travel_d_squared/cg_iso)**0.5



    def update_edges(self, tie_link: list = [None, None]) -> None:
        if type(self.rows) == np.ndarray:
            edges = np.zeros(self.rows.shape)
        else:
            edges = np.zeros(len(self.rows))
        def assign_wavespeeds(pixel):
            this_material = self.materials[
                    self.material_map.flatten()[pixel]]
            if (self.mode == 'orientations' and self.pixel_type[pixel] == 0
                    and this_material is self._iso_shortcut_material):
                # Standard cell, and the assigned material is exactly the one
                # iso_tofs was precomputed from: velocity does not depend on
                # orientation for an isotropic material, so the precomputed
                # iso_tofs is exact. Comparing the material object itself
                # (not just anisotropy == 0) matters because the domain can
                # have more than one isotropic material at different speeds
                # -- iso_tofs is only valid for the one it was derived from.
                # This shortcut must also not be taken for
                # mode == 'slowness_iso', where velocity is per-pixel
                # (self.property_map), not per-material.
                return np.tile((self.iso_tofs), 2)
            if self.pixel_type[pixel] == 0:
                local_ang = self.angles
                local_travel_d_squared = self.travel_d_squared
            else:
                local_ang = self.pixels_with_sources[pixel]['angles']
                local_travel_d_squared = self.pixels_with_sources[pixel]['travel_d_squared']
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(
                this_material, self.property_map.flatten()[pixel], local_ang)

            # Calculate cost (time) for edges originating from the current
            # node
            return np.tile((local_travel_d_squared/cg)**0.5, 2)

        position = 0
        for pixel in range(len(self.image_grid)):
            update = assign_wavespeeds(pixel)
            edges[position:position + update.shape[0]] = update
            position += update.shape[0]

        # self.rows/self.cols (from set_up_graph) are the reusable base
        # geometry -- tie_link additions are kept local so repeated calls
        # to update_edges() (e.g. with updated material properties) don't
        # keep appending more copies of the same tie links.
        rows, cols = self.rows, self.cols
        if tie_link[0] is not None and tie_link[1] is not None:
            if len(tie_link[0]) == len(tie_link[1]):
                rows = np.concatenate((self.rows, tie_link[0]))
                cols = np.concatenate((self.cols, tie_link[1]))
                edges = np.concatenate((edges, np.zeros(len(tie_link[0]))))
            else:
                raise ValueError("tie_link[0] and tie_link[1] must have the same length")

        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = build_edge_matrix(rows, cols, edges)


    def calculate_graph(self, tie_link: list = [None, None], water_links: list = [],
                        c0: float = 1.480) -> None:
        """
        Defines the connections between the nodes (graph edges) and calculates
        travel times for each edge.

        Parameters:
        ---
        water_links: list, optional [(indices_a, indices_b), ...] pairs of
                     node index arrays; every node in indices_a is directly
                     connected to every node in indices_b (both directions)
                     with an edge cost of straight_line_distance / c0,
                     embedded directly into this grid's graph. For water-
                     coupled/immersion setups with a single flat relay
                     boundary this works fine; for anything with more than
                     one relay boundary (e.g. a frontwall *and* a backwall)
                     solver.combine_via_boundary()/straight_line_times() are
                     usually a better fit -- they compose across any number
                     of boundaries without pre-enumerating every node pair
                     into the graph itself, and don't require a flat/
                     mirror-symmetric boundary. See tests/test_frontwall_relay.py.
        c0: float, wavespeed in the water_links medium, mm/us.
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
                                                PIXEL_SEARCH_RADIUS_FACTOR*self.pixel_size*2**0.5)
            # In case the search circle went outside the pixel, filter out
            take = (abs(self.grid[points] - self.image_grid[pixel])
                    <= self.pixel_size/2*PIXEL_BOUNDARY_TOL_FACTOR).all(axis=1)
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
                cg = self.wavespeed_squared_for(
                    this_material, self.property_map.flatten()[pixel], angles)
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
                raise ValueError("tie_link[0] and tie_link[1] must have the same length")
        if len(water_links) > 0:
            for link0, link1 in water_links:
                link0 = np.asarray(link0)
                link1 = np.asarray(link1)
                # calculate flight times in water c0 must be in mm/us, positions are in mm
                # shape (len(link0), len(link1)), matching row0/col0 below exactly, so
                # the same flattened cost array is valid for both edge directions
                # (distance is symmetric) without needing a separately-flattened
                # transpose -- a previous version paired row_i/col_i (built from
                # np.meshgrid, shape (len(link1), len(link0))) with tof.flatten() for
                # the forward edges but tof.T.flatten() for the reverse ones; those
                # have different shapes whenever len(link0) != len(link1), so the two
                # flattenings walk the pairs in different orders and silently mismatch
                # roughly half the reverse-direction edges with the wrong distance.
                tof = np.linalg.norm(
                    self.grid[link0][:, np.newaxis, :] - self.grid[link1][np.newaxis, :, :],
                    axis=2)/c0
                row0, col0 = np.meshgrid(link0, link1, indexing='ij')
                rows.extend(row0.flatten().tolist() + col0.flatten().tolist())
                cols.extend(col0.flatten().tolist() + row0.flatten().tolist())
                edges.extend(tof.flatten().tolist()*2)
        self.cols = cols
        self.rows = rows
        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = build_edge_matrix(rows, cols, edges)

class ZonesGrid(_ModeDependentMaterial):
    """
    Defines an irregular grid object. The domain is divided into zones, nodes
    of the grid are placed along the boundaries of the zones, with an approximately specified
    seed spacing along each edge. The number of
    seeds links to the angular resolution of the solver.
    """
    def __init__(self, a: float, b: float, c: float, zone_height: float,
                max_width_split: int, max_dx: float = 0.1,
                random_state: Optional[int] = None) -> None:
        """
        Parameters:
        ---
        random_state: int, optional seed for the random jitter applied to
                      the Voronoi seed points (see below). Pass a fixed
                      value for reproducible grid construction (e.g. tests);
                      leave as None for the previous, non-deterministic
                      behaviour.
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
        rng = np.random.default_rng(random_state)
        flucty = rng.standard_normal(p_regular.shape[0])
        fluctx = rng.standard_normal(p_regular.shape[0])
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

        global_nodes = np.zeros([0, 2])
        global_nodes = np.concatenate((global_nodes, repeated_vertices[0][:-1]), axis=0)
        edges = np.zeros([0, 2], 'int')
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
            np.isclose(0, cross2d(edge_vector,
                                  np.diff(left_chamfer_v, axis=0))))[0]
        right_chamfer_v = np.array([[self.b/2, 0], [self.c/2, self.a]])
        self.right_chamfer_ind = np.where(
            np.isclose(0, cross2d(edge_vector,
                                  np.diff(right_chamfer_v, axis=0))))[0]

        self.grid = grid_edges
        self.global_edges = edges
        self.zones_edge_ind = perimeters
        self.global_nodes = global_nodes

    def add_points(self, sources: Optional[np.ndarray] = None,
                   targets: Optional[np.ndarray] = None) -> None:

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
                in_line_inds = np.where(cross2d(s_1, edge_vector) == 0)[0]
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
                                        self.grid.shape[0]).astype(np.int64)
        if targets is None:
            self.grid = self.grid
        else:
            edge_vector = np.diff(self.global_nodes[self.global_edges],
                                  axis=1).squeeze()
            nodes_to_add = np.zeros([0, 3])
            for t in targets:
                t_1 = t - self.global_nodes[self.global_edges[:, 0]]
                in_line_inds = np.where(cross2d(t_1, edge_vector) == 0)[0]
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
                                        self.grid.shape[0]).astype(np.int64)

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


    def calculate_graph(self, tie_link: list = [None, None]) -> None:
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
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[self.material_map[this_zone]]
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(this_material, self.property_map[this_zone], angles)
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
                # distances/zone_labels/glob_angles must stay the same
                # length as rows/cols/edges (dedup_min_edge_indices below
                # uses one index to select from all of them together); a
                # tie link isn't part of any real zone, so -1/nan are
                # sentinels rather than meaningful values.
                distances.extend([0]*len(tie_link[0]))
                zone_labels.extend([-1]*len(tie_link[0]))
                glob_angles.extend([np.nan]*len(tie_link[0]))
            else:
                raise ValueError("tie_link[0] and tie_link[1] must have the same length")

        print('Saving edge cost matrices...')
        edges = np.asarray(edges)
        unique_rows, unique_cols, keep = dedup_min_edge_indices(rows, cols, edges)
        self.edges = coo_matrix((edges[keep], (unique_rows, unique_cols))).tocsr()
        self.zone_labels = coo_matrix(
            (np.asarray(zone_labels)[keep], (unique_rows, unique_cols))).tocsr()
        self.distances = coo_matrix(
            (np.asarray(distances)[keep], (unique_rows, unique_cols))).tocsr()
        self.ray_angles = coo_matrix(
            (np.asarray(glob_angles)[keep], (unique_rows, unique_cols))).tocsr()


class SimplRectGrid(_ModeDependentMaterial):
    """
    Defines a simplified rectangular grid object. The domain is divided into pixels, nodes
    of the grid are placed along the boundaries of the pixels, with a specified
    number of seeds per pixel boundary (each side of the square). The number of
    seeds defines the angular resolution of the solver. The isotropic regions are not discretised
    but taken as one cell depending on the neighbourhood.
    """

    def __init__(self, nx: int, ny: int, cx: float, cy: float, pixel_size: float,
                no_seeds: int) -> None:
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
        self.image_grid, grid_points = build_boundary_seed_grid_points(
            nx, ny, cx, cy, pixel_size, no_seeds)
        grid_points = np.unique(np.round(grid_points, 10), axis=0)
        sorted_idx = np.lexsort(grid_points.T)
        sorted_grid = grid_points[sorted_idx]
        changes = np.diff(sorted_grid, axis=0)
        changes[abs(changes) < GRID_DEDUP_TOL] = 0
        row_mask = np.append([True], np.any(changes, axis=1))
        self.grid_1 = sorted_grid[row_mask]

    def trim_to_chamfer(self, a: float, b: float, c: float,
                        mirror_domain: bool = False) -> None:
        self.a, self.b, self.c = a, b, c
        points_x = np.array([-c/2, -b/2, b/2, c/2])
        points_y = np.array([a, 0, 0, a])
        inter = interpolate.interp1d(points_x, points_y, kind='linear', bounds_error=False, fill_value=a)
        self.weld_outline_int = inter
        weld_angle = np.arctan((c - b)/2/a)
        self.weld_angle = weld_angle
        
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
        self.trimmed_by_outline = False
    
    def trim_to_weld(self, weld_outline: np.ndarray, mirror_domain: bool = False,
                     seeds_vs_node_sp: float = 0.25) -> None:
        self.trimmed_by_outline = True
        self.weld_outline = weld_outline
        weld_centre = np.argmin(weld_outline[:, 1]) 
        self.weld_outline_int = interpolate.interp1d(weld_outline[:, 0], weld_outline[:, 1], 
                                       kind='linear', fill_value=np.max(weld_outline[:, 1]) +
                                                     self.pixel_size*0.2,
                                       bounds_error=False)
        left_chamfer = weld_outline[:weld_centre + 1]
        node_spacing = self.pixel_size/self.no_seeds
        # total length of the weld outline
        total_length = np.linalg.norm(np.diff(weld_outline, axis=0), axis=1).sum()
        seeds_per_chamfer = int(total_length/node_spacing*seeds_vs_node_sp)
        seed_x = np.linspace(weld_outline[0, 0], weld_outline[-1, 0],
                             seeds_per_chamfer + 1)
        left_chamfer_seeds = sum(seed_x < weld_outline[weld_centre, 0])
        right_chamfer_seeds = sum(seed_x >= weld_outline[weld_centre, 0])
        seed_y = self.weld_outline_int(seed_x)
        if not mirror_domain:
            take = ((self.grid_1[:, 1] + node_spacing*1e-6
                         > self.weld_outline_int(self.grid_1[:, 0]))
                    & (self.grid_1[:, 0] >= weld_outline.min(axis=0)[0])
                    & (self.grid_1[:, 0] <= weld_outline.max(axis=0)[0]))
        else:
            take = ((abs(self.grid_1[:, 1]) + node_spacing*1e-6
                           > self.weld_outline_int(self.grid_1[:, 0]))
                    & (self.grid_1[:, 0] >= weld_outline.min(axis=0)[0])
                    & (self.grid_1[:, 0] <= weld_outline.max(axis=0)[0]))
        first_new_idx = self.grid_1[take].shape[0]
        if not mirror_domain:
            self.grid_1 = np.concatenate((self.grid_1[take], np.column_stack((seed_x, seed_y))), axis=0)
            self.left_iso_zone = np.arange(first_new_idx, first_new_idx + left_chamfer_seeds)
            self.right_iso_zone = np.arange(first_new_idx + left_chamfer_seeds,
                                            first_new_idx + left_chamfer_seeds + right_chamfer_seeds)
        else:
            self.grid_1 = np.concatenate((self.grid_1[take],
                                          np.column_stack((seed_x[:left_chamfer_seeds ],
                                                           seed_y[:left_chamfer_seeds ])),
                                          np.column_stack((seed_x[:left_chamfer_seeds ][::-1],
                                                           -seed_y[:left_chamfer_seeds][::-1])),
                                          np.column_stack((seed_x[left_chamfer_seeds:][::-1],
                                                           seed_y[left_chamfer_seeds:][::-1])),
                                          np.column_stack((seed_x[left_chamfer_seeds:],
                                                           -seed_y[left_chamfer_seeds:]))),axis=0)
            self.left_iso_zone = np.arange(first_new_idx, first_new_idx + 2*left_chamfer_seeds)
            self.right_iso_zone = np.arange(first_new_idx + 2*left_chamfer_seeds,
                                            first_new_idx + 2*left_chamfer_seeds +
                                            2*right_chamfer_seeds)
            # add weld centre point to the left iso zone

    def simplify_grid(self, left_add: int = 0, right_add: int = 0) -> None:
        """
        Simplifes the grid be keeping only the cells covering the weld. The isotropic areas
        to the left and to the right will be  modelled with straight rays.

        """
        new_grid = []
        new_pixels = []
        tree = cKDTree(self.grid_1)
        # Some pixels may not have the orientations and the material props set (those which contain
        # the chamfer); Fix this by using neighbouring properties
        temp_props = np.copy(self.property_map)
        temp_materials = np.copy(self.material_map)
        temp_props[self.material_map == 0] = np.nan
        for i in range(temp_props.shape[0]):
            filled = np.where(~np.isnan(temp_props[i]))[0]
            temp_props[i, :filled[0]] = temp_props[i, filled[0]]
            temp_props[i, filled[-1]:] = temp_props[i, filled[-1]]
        mat_flat = self.material_map.flatten()
        prop_flat = self.property_map.flatten()
        self.image_grid_lookup = np.zeros(self.image_grid.shape[0], 'int') - 1
        cnt = 0
        for pixel in range(len(self.image_grid)):
            # check all corner points of a cell; if any of them is above the weld outline, the cell
            # belogs to the weld
            half_size = self.pixel_size/2
            cell_corners = self.image_grid[pixel] + np.array([[-half_size, -half_size, half_size,
                                                               half_size], [-half_size, half_size,
                                                                            -half_size, half_size]]).T
            if (abs(cell_corners[:, 1]) > self.weld_outline_int(cell_corners[:, 0])).any():
                mat_flat[pixel] = 1
                prop_flat[pixel] = temp_props.flatten()[pixel]

                points = tree.query_ball_point(self.image_grid[pixel], PIXEL_SEARCH_RADIUS_FACTOR*self.pixel_size*2**0.5)
                take = (abs(self.grid_1[points] - self.image_grid[pixel]) <= self.pixel_size/2*PIXEL_BOUNDARY_TOL_FACTOR).all(axis=1)
                new_grid.extend(list(np.array(points)[take]))
                new_pixels.append(pixel)
                self.image_grid_lookup[pixel] = cnt
                cnt += 1
        self.image_grid_trim = np.array(self.image_grid[new_pixels])
        self.image_tree_trim = cKDTree(self.image_grid_trim)
        self.grid_tree_trim = cKDTree(self.grid_1)
        self.property_map = prop_flat.reshape(self.property_map.shape)
        self.material_map = mat_flat.reshape(self.property_map.shape)
        # -right_add: is not "the last right_add columns" when right_add == 0
        # (Python's -0 == 0, so material_map[:, -0:] is the *entire* array,
        # not an empty slice) -- guard both the slice used for
        # original_weld_mask and the zeroing below against that.
        self.original_weld_mask = self.material_map[
            :, left_add:(-right_add if right_add else None)]
        if left_add:
            self.material_map[:, :left_add] = 0
        if right_add:
            self.material_map[:, -right_add:] = 0

    def add_points(self, points: Optional[np.ndarray] = None,
                   sources: Optional[np.ndarray] = None,
                   targets: Optional[np.ndarray] = None) -> None:

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
                    indices.append([i, ix])
            self.grid = np.concatenate((self.grid_1, np.vstack(to_add)), axis=0)
            indices = np.vstack(indices)

            self.source_idx = indices[np.isin(indices[:, 0], sources), 1]
            self.target_idx = indices[np.isin(indices[:, 0], targets), 1]
            added_points_idx = np.arange(self.grid_1.shape[0], self.grid_1.shape[0] + len(to_add))
        # Check which sources are in the homogeneous regios
        add_neg = added_points_idx[self.grid[added_points_idx][:, 0] < 0]
        add_pos = added_points_idx[self.grid[added_points_idx][:, 0] >= 0]
        if self.trimmed_by_outline is False:
            take_neg = (abs(self.grid[add_neg, 1]) 
                      < np.tan(np.pi/2 + self.weld_angle)*(self.grid[add_neg, 0] + self.b/2))
            take_pos = (abs(self.grid[add_pos, 1]) 
                      < np.tan(np.pi/2 - self.weld_angle)*(self.grid[add_pos, 0] - self.b/2))
        else:
            take_neg = self.grid[add_neg, 0] < self.weld_outline[0, 0]
            take_pos = self.grid[add_pos, 0] > self.weld_outline[-1, 0]

        self.left_iso_chamfer = np.copy(self.left_iso_zone)
        self.right_iso_chamfer = np.copy(self.right_iso_zone)
        self.left_iso_zone = np.append(self.left_iso_zone,
                                       add_neg[take_neg])
        self.left_iso_trans = add_neg[take_neg]
        self.right_iso_trans = add_pos[take_pos]
        self.right_iso_zone = np.append(self.right_iso_zone,
                                       add_pos[take_pos])

        mask_receiver = [self.left_iso_trans[ix] in self.target_idx
                         for ix in range(len(self.left_iso_trans))]
        self.left_iso_targets = self.left_iso_trans[mask_receiver]
        self.left_iso_sources = self.left_iso_trans[~np.array(mask_receiver)]
        mask_receiver = [self.right_iso_trans[ix] in self.target_idx
                         for ix in range(len(self.right_iso_trans))]
        self.right_iso_targets = self.right_iso_trans[mask_receiver]
        self.right_iso_sources = self.right_iso_trans[~np.array(mask_receiver)]
        self.left_iso_nodes = np.array(list(set(self.left_iso_zone)
                                            - set(self.left_iso_targets)))
        self.right_iso_nodes = np.array(list(set(self.right_iso_zone)
                                            - set(self.right_iso_targets)))

    def set_up_graph(self) -> None:
        rows = []
        cols = []

        self.image_tree = cKDTree(self.image_grid)
        self.tree = cKDTree(self.grid)
        self.r_closest = defaultdict(list)
        self.pixel_type = np.zeros(self.image_grid_trim.shape[0])
        first = True
        self.r_closest = defaultdict(list)
        self.irregular_pixels = dict()
        single_counter = 0
        for full_pixel in trange(len(self.image_grid)):
            if self.image_grid_lookup[full_pixel] == -1:
                continue
            else:
                pixel = self.image_grid_lookup[full_pixel]

            # identify points within a pixel
            points = self.tree.query_ball_point(self.image_grid_trim[pixel],
                                                PIXEL_SEARCH_RADIUS_FACTOR*self.pixel_size*2**0.5)
            # In case the search circle went outside the pixel, filter out
            take = (abs(self.grid[points] - self.image_grid_trim[pixel])
                    <= self.pixel_size/2*PIXEL_BOUNDARY_TOL_FACTOR).all(axis=1)
            points = np.array(points)[take]
            # Restrict to points on the same side of the mirror plane (y=0)
            # as this pixel, so a point from the mirrored half-domain that
            # happens to be geometrically close isn't picked up. Guarded
            # against pixel y == 0 exactly (e.g. the weld root in a
            # non-mirrored domain): the sign-product test is degenerate
            # there (zero times anything is zero, so "> 0" would wrongly
            # discard every point), and there is no meaningful "other side"
            # to exclude for a pixel sitting on the plane itself anyway.
            if self.image_grid_trim[pixel, 1] != 0:
                points = points[self.grid[points, 1]*self.image_grid_trim[pixel, 1] > 0]
            points = points[np.lexsort(np.round(GRID_DEDUP_ROUND_SCALE*self.grid[points]).T)]
            if len(points) < 2:
                single_counter += 1
                self.pixel_type[pixel] = -1
                continue
            # now are sorted from slow y fast x from bottom to top, from left to right
            self.r_closest[pixel] = points
            # to check is this is a standard cell, verify if there is an additional node
            # (source/receiver) and whether the shape is square
            cell_is_square = np.isclose(abs(self.grid[points] - self.image_grid_trim[pixel]),
                    self.pixel_size/2).any(axis=1).all()

            if points.shape[0] == self.no_seeds*4 and cell_is_square:
                self.pixel_type[pixel] = 0
                # Standard cell:
                if first is True:
                    # pre calculate angles and distances
                    local_ind = np.arange(points.shape[0])
                    pairs = np.array(list(combinations(local_ind, 2)))
                    local_edges = self.grid[points][pairs, :]
                    dist = -local_edges[:, 0] + local_edges[:, 1]
                    self.angles = np.arctan2(dist[:, 1], dist[:, 0])
                    self.travel_d_squared = (dist**2).sum(axis=1)
                    self.pairs = pairs
                    self.row_pairs = pairs.flatten('F')
                    self.col_pairs = pairs[:, ::-1].flatten('F')
                    first = False
                row_pairs = self.row_pairs
                col_pairs = self.col_pairs
            else:
                self.pixel_type[pixel] = 1
                local_ind = np.arange(points.shape[0])
                pairs = np.array(list(combinations(local_ind, 2)))
                local_edges = self.grid[points][pairs, :]
                dist = -local_edges[:, 0] + local_edges[:, 1]
                angles = np.arctan2(dist[:, 1], dist[:, 0])
                travel_d_squared = (dist**2).sum(axis=1)
                self.irregular_pixels[pixel] = dict([('angles', angles),
                                                     ('travel_d_squared', travel_d_squared),
                                                     ('pairs', pairs)])
                row_pairs = pairs.flatten('F')
                col_pairs = pairs[:, ::-1].flatten('F')

            rows.extend(points[row_pairs])
            cols.extend(points[col_pairs])
        # if there is an isotropic material in the domain *and* at least one
        # standard cell was found (see RectGrid.set_up_graph for why "not first"
        # is needed here), precompute ToFs for a standard cell.
        mats = [mat.anisotropy for k, mat in self.materials.items()]
        if mats.count(0) > 0 and not first:
            cg_iso = self.materials[mats.index(0)].get_wavespeed_squared(0, 0)
            self.iso_tofs = (self.travel_d_squared/cg_iso)**0.5
        self.left_iso_rows, self.left_iso_cols = [], []
        self.left_iso_edges = []
        # Add left homogeneous zone
        if self.left_iso_zone is not None:
            # Go through possible connections; first top chamfer vs bottom chamfer (assumes
            # pulse echo)
            cham_x = self.grid[self.left_iso_chamfer, 0]
            cham_y = self.grid[self.left_iso_chamfer, 1]           
            mid_chamfer = self.left_iso_chamfer.shape[0]//2
            mid_trans = self.left_iso_trans.shape[0]//2
            top_n = np.arange(mid_chamfer)
            bot_n = np.arange(mid_chamfer, mid_chamfer*2)
            tt, bb = np.meshgrid(top_n, bot_n)
            pairs = np.c_[tt.flatten(), bb.flatten()]
            local_edges = self.grid[self.left_iso_chamfer[pairs]]
            dist = -local_edges[:, 0] + local_edges[:, 1]
            interp_ray = local_edges[:, 0, 0].reshape(-1, 1) \
                + (dist[:, 0]/dist[:, 1]).reshape(-1, 1)*(cham_y.reshape(1, -1)
                                                          - local_edges[:, 0, 1].reshape(-1, 1))
            flag = cham_x.reshape(1, -1) >= interp_ray
            flag[np.isclose(cham_x.reshape(1, -1), interp_ray, atol=CHAMFER_BOUNDARY_MATCH_TOL)] = True
            ch_edge_is_good = []
            for edge in range(flag.shape[0]):
                ch_edge_is_good.append(flag[edge][pairs[edge, 0]:pairs[edge, 1]].all())
            ch_edge_is_good = np.array(ch_edge_is_good).reshape(mid_chamfer, mid_chamfer).T
            # do the same for tranducer vs chamfer 

            top_n = np.arange(mid_trans) + 2*mid_chamfer
            bot_n = np.arange(mid_chamfer*2)
            tt, bb = np.meshgrid(top_n, bot_n)
            pairs = np.c_[tt.flatten(), bb.flatten()]
            local_edges = self.grid[self.left_iso_zone[pairs]]
            dist = -local_edges[:, 0] + local_edges[:, 1]
            interp_ray = local_edges[:, 0, 0].reshape(-1, 1) \
                + (dist[:, 0]/dist[:, 1]).reshape(-1, 1)*(cham_y.reshape(1, -1)
                                                          - local_edges[:, 0, 1].reshape(-1, 1))
            flag = cham_x.reshape(1, -1) >= interp_ray
            flag[np.isclose(cham_x.reshape(1, -1), interp_ray, atol=CHAMFER_BOUNDARY_MATCH_TOL)] = True
            tr_edge_is_good = []
            for edge in range(flag.shape[0]):
                tr_edge_is_good.append(flag[edge][:pairs[edge, 1]].all())
            tr_edge_is_good = np.array(tr_edge_is_good).reshape(2*mid_chamfer, mid_trans).T


            cs = self.left_iso_chamfer.shape[0]
            ts = self.left_iso_trans.shape[0]
            big_row, big_col = np.meshgrid(self.left_iso_zone, self.left_iso_zone)
            mask = np.ones([cs//2, cs//2])
            iu = np.triu_indices(cs//2, 2)
            mask[iu] = 0
            mask[iu[1], iu[0]] = 0
            big_row[:cs//2, :cs//2] = -99
            big_row[cs//2:-ts, cs//2:-ts] = -99

            big_row[:-ts, cs:-ts//2][~tr_edge_is_good.T] = -99
            big_row[:-ts, -ts//2:][~tr_edge_is_good.T] = -99
            big_row[cs:-ts//2, :-ts][~tr_edge_is_good] = -99
            big_row[-ts//2:, :-ts][~tr_edge_is_good[::-1]] = -99
            big_row[:cs//2, cs//2:cs][~ch_edge_is_good] = -99
            big_row[cs//2:-ts, :cs//2][~ch_edge_is_good.T] = -99
            big_col[big_row == -99] = -99

            
            valid_pairs = np.where(big_row != -99)
            valid_pairs = self.left_iso_zone[
                np.c_[valid_pairs[0], valid_pairs[1]]]
            all_nodes = self.grid[valid_pairs]
            # Calculate distance vector
            r = all_nodes[:, 1] - all_nodes[:, 0]
            dist = np.sum(r**2, axis=1)
            angles = np.arctan2(r[:, 1], r[:, 0])
            # Reject connecting to the same node
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[
                    self.material_map[self.ny//2, 0]]
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(
                this_material, self.property_map[self.ny//2, 0], angles)

            edge_cost = (dist/cg)**0.5
            # edge_cost only contains rays from transducers to chamfer
            # construct edge matrix with edges in both directions (but no edges between chamfer
            # elements)
            edge = np.zeros(big_row.shape)
            edge[big_row != -99] = edge_cost

            edge = edge[big_row != -99]
            big_col = big_col[big_row != -99]
            big_row = big_row[big_row != -99]
            self.left_iso_rows = big_row.astype(int)
            self.left_iso_cols = big_col.astype(int)
            self.left_iso_edges = edge
        
        self.right_iso_rows, self.right_iso_cols = [], []
        self.right_iso_edges = []
        # Add right homogeneous zone
        if self.right_iso_zone is not None:
            # Go through possible connections; first top chamfer vs bottom chamfer (assumes
            # pulse echo)
            cham_x = self.grid[self.right_iso_chamfer, 0]
            cham_y = self.grid[self.right_iso_chamfer, 1]           
            mid_chamfer = self.right_iso_chamfer.shape[0]//2
            mid_trans = self.right_iso_trans.shape[0]//2
            top_n = np.arange(mid_chamfer)
            bot_n = np.arange(mid_chamfer, mid_chamfer*2)
            tt, bb = np.meshgrid(top_n, bot_n)
            pairs = np.c_[tt.flatten(), bb.flatten()]
            local_edges = self.grid[self.right_iso_chamfer[pairs]]
            dist = -local_edges[:, 0] + local_edges[:, 1]
            interp_ray = local_edges[:, 0, 0].reshape(-1, 1) \
                + (dist[:, 0]/dist[:, 1]).reshape(-1, 1)*(cham_y.reshape(1, -1)
                                                          - local_edges[:, 0, 1].reshape(-1, 1))
            flag = cham_x.reshape(1, -1) <= interp_ray
            flag[np.isclose(cham_x.reshape(1, -1), interp_ray, atol=CHAMFER_BOUNDARY_MATCH_TOL)] = True
            ch_edge_is_good = []
            for edge in range(flag.shape[0]):
                ch_edge_is_good.append(flag[edge][pairs[edge, 0]:pairs[edge, 1]].all())
            ch_edge_is_good = np.array(ch_edge_is_good).reshape(mid_chamfer, mid_chamfer).T
            # do the same for tranducer vs chamfer 

            top_n = np.arange(mid_trans) + 2*mid_chamfer
            bot_n = np.arange(mid_chamfer*2)
            tt, bb = np.meshgrid(top_n, bot_n)
            pairs = np.c_[tt.flatten(), bb.flatten()]
            local_edges = self.grid[self.right_iso_zone[pairs]]
            dist = -local_edges[:, 0] + local_edges[:, 1]
            interp_ray = local_edges[:, 0, 0].reshape(-1, 1) \
                + (dist[:, 0]/dist[:, 1]).reshape(-1, 1)*(cham_y.reshape(1, -1)
                                                          - local_edges[:, 0, 1].reshape(-1, 1))
            flag = cham_x.reshape(1, -1) <= interp_ray
            flag[np.isclose(cham_x.reshape(1, -1), interp_ray, atol=CHAMFER_BOUNDARY_MATCH_TOL)] = True
            tr_edge_is_good = []
            for edge in range(flag.shape[0]):
                tr_edge_is_good.append(flag[edge][:pairs[edge, 1]].all())
            tr_edge_is_good = np.array(tr_edge_is_good).reshape(2*mid_chamfer, mid_trans).T


            cs = self.right_iso_chamfer.shape[0]
            ts = self.right_iso_trans.shape[0]
            big_row, big_col = np.meshgrid(self.right_iso_zone, self.right_iso_zone)
            mask = np.ones([cs//2, cs//2])
            iu = np.triu_indices(cs//2, 2)
            mask[iu] = 0
            mask[iu[1], iu[0]] = 0
            big_row[:cs//2, :cs//2] = -99
            big_row[cs//2:-ts, cs//2:-ts] = -99

            big_row[:-ts, cs:-ts//2][~tr_edge_is_good.T] = -99
            big_row[:-ts, -ts//2:][~tr_edge_is_good.T] = -99
            big_row[cs:-ts//2, :-ts][~tr_edge_is_good] = -99
            big_row[-ts//2:, :-ts][~tr_edge_is_good[::-1]] = -99
            big_row[:cs//2, cs//2:cs][~ch_edge_is_good] = -99
            big_row[cs//2:-ts, :cs//2][~ch_edge_is_good.T] = -99
            big_col[big_row == -99] = -99
            
            valid_pairs = np.where(big_row != -99)
            valid_pairs = self.right_iso_zone[
                np.c_[valid_pairs[0], valid_pairs[1]]]
            all_nodes = self.grid[valid_pairs]
            # Calculate distance vector
            r = all_nodes[:, 1] - all_nodes[:, 0]
            dist = np.sum(r**2, axis=1)
            angles = np.arctan2(r[:, 1], r[:, 0])
            this_material = self.materials[
                    self.material_map[self.ny//2, 0]]
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(
                this_material, self.property_map[self.ny//2, 0], angles)

            edge_cost = (dist/cg)**0.5
            # edge_cost only contains rays from transducers to chamfer
            # construct edge matrix with edges in both directions (but no edges between chamfer
            # elements)
            edge = np.zeros(big_row.shape)
            edge[big_row != -99] = edge_cost

            edge = edge[big_row != -99]
            big_col = big_col[big_row != -99]
            big_row = big_row[big_row != -99]
            self.right_iso_rows = big_row.astype(int)
            self.right_iso_cols = big_col.astype(int)
            self.right_iso_edges = edge



        self.rows_w = np.array(rows)
        self.cols_w = np.array(cols)
        self.rows = np.concatenate((self.rows_w, self.left_iso_rows, self.right_iso_rows))
        self.cols = np.concatenate((self.cols_w, self.left_iso_cols, self.right_iso_cols))


    def update_edges(self, tie_link: list = [None, None]) -> None:
        edges = np.zeros(self.rows_w.shape)
        def assign_wavespeeds(pixel, trim_pixel):
            this_material = self.materials[
                    self.material_map.flatten()[pixel]]
            if self.pixel_type[trim_pixel] == 0:
                local_ang = self.angles
                local_travel_d_squared = self.travel_d_squared
            else:
                local_ang = self.irregular_pixels[trim_pixel]['angles']
                local_travel_d_squared = self.irregular_pixels[trim_pixel]['travel_d_squared']
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(
                this_material, self.property_map.flatten()[pixel], local_ang)

            # Calculate cost (time) for edges originating from the current
            # node
            return np.tile((local_travel_d_squared/cg)**0.5, 2)

        position = 0
        for full_pixel in trange(len(self.image_grid)):
            if self.image_grid_lookup[full_pixel] == -1:
                continue
            else:
                pixel = self.image_grid_lookup[full_pixel]
                if self.pixel_type[pixel] == -1:
                    continue
            update = assign_wavespeeds(full_pixel, pixel)
            edges[position:position + update.shape[0]] = update
            position += update.shape[0]
        # add isotropic regions
        update = np.array(self.left_iso_edges)
        edges = np.concatenate((edges, update))
        position += update.shape[0]
        update = np.array(self.right_iso_edges)
        edges = np.concatenate((edges, update))

        # self.rows/self.cols (from set_up_graph) are the reusable base
        # geometry -- tie_link additions are kept local so repeated calls
        # to update_edges() (e.g. with updated material properties) don't
        # keep appending more copies of the same tie links.
        rows, cols = self.rows, self.cols
        if tie_link[0] is not None and tie_link[1] is not None:
            if len(tie_link[0]) == len(tie_link[1]):
                rows = np.concatenate((self.rows, tie_link[0]))
                cols = np.concatenate((self.cols, tie_link[1]))
                edges = np.concatenate((edges, np.zeros(len(tie_link[0]))))
            else:
                raise ValueError("tie_link[0] and tie_link[1] must have the same length")

        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = build_edge_matrix(rows, cols, edges)


    def calculate_graph(self, tie_link: list = [None, None]) -> None:
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
                                                PIXEL_SEARCH_RADIUS_FACTOR*self.pixel_size*2**0.5)
            # In case the search circle went outside the pixel, filter out
            take = (abs(self.grid[points] - self.image_grid_trim[pixel])
                    <= self.pixel_size/2*PIXEL_BOUNDARY_TOL_FACTOR).all(axis=1)
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
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[self.material_map.flatten()[
                    full_pixel]]
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(
                this_material, self.property_map.flatten()[full_pixel], angles)

            edge_cost = dist/cg**0.5
            temp_col, temp_row = np.meshgrid(points[take], points[take])
            mask = (temp_col != temp_row).flatten()
            col_indices = list(temp_col[temp_col != temp_row])
            row_indices = list(temp_row[temp_row != temp_col])
            edge_cost = edge_cost.reshape(temp_col.shape)[temp_col != temp_row]
            rows.extend(row_indices)
            cols.extend(col_indices)
            edges.extend(edge_cost)
        # Add left homogeneous zone
        if self.left_iso_zone is not None:
            local_grid = self.grid[self.left_iso_zone]
            # Calculate distance vector
            r = -local_grid[:, :2][:, np.newaxis, :] + local_grid[:, :2][np.newaxis, :, :]
            dist = np.linalg.norm(r, axis=2)
            angles = np.arctan2(r[:, :, 1], r[:, :, 0])
            # Reject connecting to the same node
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[
                    self.material_map[self.ny//2, 0]]
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(
                this_material, self.property_map[self.ny//2, 0], angles)

            edge_cost = dist/cg**0.5
            
            temp_col, temp_row = np.meshgrid(self.left_iso_zone, self.left_iso_zone)
            mask = (temp_col != temp_row).flatten()
            col_indices = list(temp_col[temp_col != temp_row])
            row_indices = list(temp_row[temp_row != temp_col])
            edge_cost = edge_cost.reshape(temp_col.shape)[temp_col != temp_row]
            rows.extend(row_indices)
            cols.extend(col_indices)
            edges.extend(edge_cost)


        # Add right homogeneous zone
        if self.right_iso_zone is not None:
            local_grid = self.grid[self.right_iso_zone]
            # Calculate distance vector
            r = -local_grid[:, :2][:, np.newaxis, :] + local_grid[:, :2][np.newaxis, :, :]
            dist = np.linalg.norm(r, axis=2)
            angles = np.arctan2(r[:, :, 1], r[:, :, 0])
            # Reject connecting to the same node
            dist = dist.flatten()
            angles = angles.flatten()
            this_material = self.materials[
                    self.material_map[self.ny//2, -1]]
            # If anisotropic, calculate incident angle
            cg = self.wavespeed_squared_for(
                this_material, self.property_map[self.ny//2, -1], angles)

            edge_cost = dist/cg**0.5
            
            temp_col, temp_row = np.meshgrid(self.right_iso_zone, self.right_iso_zone)
            mask = (temp_col != temp_row).flatten()
            col_indices = list(temp_col[temp_col != temp_row])
            row_indices = list(temp_row[temp_row != temp_col])
            edge_cost = edge_cost.reshape(temp_col.shape)[temp_col != temp_row]
            rows.extend(row_indices)
            cols.extend(col_indices)
            edges.extend(edge_cost)




        if tie_link[0] is not None and tie_link[1] is not None:
            if len(tie_link[0]) == len(tie_link[1]):
                rows.extend(list(tie_link[0]))
                cols.extend(list(tie_link[1]))
                edges.extend([0]*len(tie_link[0]))
            else:
                raise ValueError("tie_link[0] and tie_link[1] must have the same length")

        # Create a sparse matrix of graph edge lengths (times of flight)
        self.edges = build_edge_matrix(rows, cols, edges)



            # Calculate distance vector
