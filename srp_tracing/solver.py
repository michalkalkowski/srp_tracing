#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:49:57 2019

A shortest ray path (SRP) solver for ray tracing in heterogeneous media,
austenitic stainless steel welds in particular.

Solver functions.

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""

from typing import Any, Optional

import numpy as np
from tqdm import trange
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import Delaunay, cKDTree


def interp_weights(xyz: np.ndarray, uvw: np.ndarray,
                   d: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast interpolation of multiple datasets over the same grid
    from: https://stackoverflow.com/a/20930910/2197375

    Parameters:
    ---
    xyz: ndarray, array of starting irregular grid coordinates
    uvw: ndarray, array of target regular grid coordinates
    d: int, dimension (d=2 in 2D)

    Returns:
    ---
    vertices: ndarray, vertices coordinates
    weights: ndarray, interpolation weights
    """

    tri = Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values: np.ndarray, vtx: np.ndarray, wts: np.ndarray,
                dim: int = 3) -> np.ndarray:
    """
    Fast interpolation of multiple datasets over the same grid
    from: https://stackoverflow.com/a/20930910/2197375

    Parameters:
    ---
    values: ndarray, values to interpolate (corresponding to the previously
    used xyz points.
    vtx: ndarray, vertices
    wts, ndarray, interpolation weights
    """

    if dim == 2:
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    elif dim == 3:
        return np.einsum('inj,nj->in', np.take(values, vtx, axis=1), wts)


def straight_line_times(points_a: np.ndarray, points_b: np.ndarray,
                        speed: float) -> np.ndarray:
    """
    Pairwise straight-line travel time between two sets of points in a
    single, unobstructed, homogeneous medium -- e.g. the water leg of a
    water-coupled/immersion setup, from an array of transducers to a grid
    of points along the (possibly complicated) frontwall, before relaying
    the rest of the way through the solid with combine_via_boundary().
    Nothing here is graph-based; it's just distance / speed, broadcast
    over every pair.

    Parameters:
    ---
    points_a: ndarray (n_a, 2), coordinates
    points_b: ndarray (n_b, 2), coordinates
    speed: float, wavespeed in the medium (e.g. ~1.48 mm/us for water)

    Returns:
    ---
    times: ndarray (n_a, n_b), times[i, j] = |points_a[i] - points_b[j]| / speed
    """
    points_a = np.asarray(points_a)
    points_b = np.asarray(points_b)
    return np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2) / speed


def combine_via_boundary(times_a: np.ndarray,
                         times_b: Optional[np.ndarray] = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Combines one-way travel times to a shared set of "via" nodes -- e.g. a
    grid of points along a backwall or frontwall, which may have an
    arbitrarily complex shape -- into the fastest two-leg travel time
    between every pair of endpoints, and which via node realises it.

    This is the array-based way to do pulse-echo/reflection (or
    transmission through an intermediate boundary) without doubling the
    domain the way the old mirrored-domain approach did (see
    Solver.fold_paths): travel time is symmetric (T(a, k) == T(k, a) for a
    single medium), so the two-leg time from endpoint i to endpoint j via
    node k is simply times_a[i, k] + times_b[j, k], and Fermat's principle
    says the physically realised path is whichever k minimises that sum --
    no further search is needed once times_a/times_b are known, and unlike
    mirroring, this works just as well for a non-flat boundary. The one-way
    times themselves are ordinary Solver.solve() output columns
    (Solver.tfs[:, boundary_indices], after placing the via points as
    regular nodes via Grid.add_points -- no grid changes are needed for
    this at all), but don't have to be: for a water-coupled setup with a
    complicated frontwall, times_a/times_b for the water leg can instead be
    plain straight-line-distance / water_speed, since water is a single
    known, unobstructed, isotropic medium.

    Parameters:
    ---
    times_a: ndarray (n_a, n_via), one-way travel time from each of n_a
             endpoints to each via node.
    times_b: ndarray (n_b, n_via), optional; the equivalent for a second
             set of endpoints (e.g. separate receivers, for transmission
             through an intermediate boundary). Defaults to times_a itself,
             for the common pulse-echo case where the same array acts as
             both transmitter and receiver.

    Returns:
    ---
    relay_times: ndarray (n_a, n_b), relay_times[i, j] = fastest i -> via
                 -> j travel time.
    via_index: ndarray (n_a, n_b), int, index into the via-node axis (i.e.
               into times_a's/times_b's columns, not a node index in the
               full grid) of the via node that achieved that minimum --
               pass the corresponding node index to
               Solver.calculate_relay_ray_path to reconstruct the path.
    """
    if times_b is None:
        times_b = times_a
    combined = times_a[:, None, :] + times_b[None, :, :]
    via_index = np.argmin(combined, axis=2)
    relay_times = np.take_along_axis(combined, via_index[:, :, None], axis=2).squeeze(-1)
    return relay_times, via_index


class Solver:
    """
    Defines a solver object used for a shortest ray path simulation.

    Parameters:
    ---
    grid: object, an SRP grid
    """

    def __init__(self, grid: Any):
        self.grid = grid

    def solve(self, source_indices: np.ndarray, with_points: bool = False) -> None:
        """
        Runs the shortest path solver on the previously defined grid from
        specified source indices. It may optionally return points to
        reconstruct the path and interpolatpe the time of flight image over a
        regular grid.

        Parameters:
        ---
        source_indices: ndarray, indices of source nodes
        with_points: bool, if True, predecessor index is returned for each node
                     allowing for ray path reconstruction.
        """
        self.sources = source_indices
        #print('SRP search...')
        if with_points:
            self.tfs, self.points = shortest_path(
                self.grid.edges,
                return_predecessors=with_points,
                indices=source_indices)
        else:
            self.tfs = shortest_path(self.grid.edges,
                                     return_predecessors=with_points,
                                     indices=source_indices)
        #print('Search ended.')

    def interpolate_tf_field(self, external: bool = False,
                             external_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Interpolates the time of flight field to a regular grid.

        Parameters:
        ---
        external: bool, if False, self.grid.image_grid is used; if True,
                  external grid is used.
        external_grid: ndarray, external grid, if interpolation over a
                       different than the image grid native to the grid
                       attribute is desired.

        Returns:
        ---
        tf_grid: ndarray, interpolated time of flight field
        """
        if external:
            vtx, wts = interp_weights(self.grid.grid, external_grid)
        else:
            vtx, wts = interp_weights(self.grid.grid, self.grid.image_grid)
        no_of_sources = len(self.sources)
        # Interpolate
        tf_grid = interpolate(
            self.tfs, vtx, wts, dim=3).reshape(
                no_of_sources, self.grid.ny, self.grid.nx).transpose(1, 2, 0)
        # Make sure that the corners do not give interpolation artefacts
        tf_grid[0, 0] = tf_grid[
            [1, 1, 0], [0, 1, 1]].mean(axis=0)
        tf_grid[0, -1] = tf_grid[
            [1, 1, 0], [-1, -2, -2]].mean(axis=0)
        tf_grid[-1, -1] = tf_grid[
            [-2, -2, -1], [-1, -2, -2]].mean(axis=0)
        tf_grid[-1, 0] = tf_grid[
            [-2, -2, -1], [0, 1, 1]].mean(axis=0)
        return tf_grid

    def calculate_ray_paths(self, end: list = []) -> dict:
        """
        Extracts ray paths from TFT data based on the SRP solver outcome

        Parameters:
        ---
        end: list, list of target nodes
        """

        paths = {}
        #print('Tracing...')
        for i in range(len(self.sources)):
            from_this_source = []
            for j in range(len(end)):
                current = end[j]
                path = []
                while current != self.sources[i]:
                    path.append(current)
                    current = self.points[i][current]
                path.append(self.sources[i])
                path.reverse()
                from_this_source.append(path)
            paths[i] = from_this_source
        return paths

    def calculate_relay_ray_path(self, source_index: int, target_index: int,
                                 via_node: int) -> list:
        """
        Reconstructs a two-leg ray path from source_index to target_index
        via via_node -- e.g. the best backwall/frontwall relay node
        returned by combine_via_boundary -- by concatenating the
        source_index -> via_node and target_index -> via_node shortest
        paths found by solve(with_points=True). This is the non-mirrored-
        domain equivalent of fold_paths.

        Parameters:
        ---
        source_index, target_index: int, node indices of the two endpoints
                                     (both must be among the source_indices
                                     passed to solve())
        via_node: int, node index of the relay point

        Returns:
        ---
        path: list, node indices from source_index to via_node to target_index
        """
        sources = list(self.sources)

        def _leg_to(origin_index, node):
            i = sources.index(origin_index)
            path = []
            current = node
            while current != origin_index:
                path.append(current)
                current = self.points[i][current]
            path.append(origin_index)
            path.reverse()
            return path

        outbound = _leg_to(source_index, via_node)
        inbound = _leg_to(target_index, via_node)
        return outbound + inbound[-2::-1]

    def fold_paths(self, paths: dict) -> list:
        """
        Folds paths for calculations over a mirrored domain (simulating
        backwall reflected signals).
        """
        # Create lookup table - which node in the mirror corresponds to original node
        bot_grid = np.where(self.grid.grid[:, 1] < 0)[0]
        lookup = np.arange(self.grid.grid.shape[0])

        for point in bot_grid:
            test_point = np.copy(self.grid.grid[point])
            test_point[1] *= -1
            _, org = self.grid.tree.query(test_point)
            lookup[point] = org

        folded_paths = []
        for src in range(len(paths.keys())):
            one_source = paths[src]
            local = []
            for path in one_source:
                local.append(list(lookup[path]))
            folded_paths.append(local)
        return folded_paths

    def calculate_gradient(self, residue, slowness_der=True):
        paths = calculate_ray_paths(self.grid.grid.target_idx)
        tree = cKDTree(self.grid.grid.image_grid)
        grad_proj = np.zeros(self.grid.grid.image_grid.shape[0])
        for src in trange(len(paths)):
            for rec in range(len(paths[src])):
                mid_points = (self.grid.grid[paths[src][rec][1:]] +
                              self.grid.grid[paths[src][rec][:-1]])/2
                segments = (self.grid.grid[paths[src][rec][1:]] -
                            self.grid.grid[paths[src][rec][:-1]])
                lengths = np.linalg.norm(segments,
                                         axis=1)
                _, ind = tree.query(mid_points, k=1)
                grad_proj[ind] += res[rec, src]*lengths
                if slowness_der:
                    ray_angles = np.arctan2(segments[:, 1], segments[:, 0])
                    orientations = self.grid.property_map[ind]
                    alpha = (ray_angles - orientations + 2*np.pi) % (2*np.pi)
                    ds_dalpha = self.grid.materials[1].dgsp(alpha)
                    grad_proj[ind] *= -ds_alpha
                
        return grad_proj
