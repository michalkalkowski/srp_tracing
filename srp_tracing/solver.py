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

import numpy as np
from tqdm import trange
import scipy.sparse.csgraph._shortest_path as sp
import scipy.spatial.qhull as qhull
from scipy.spatial import cKDTree


def interp_weights(xyz, uvw, d=2):
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

    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, dim=3):
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


class Solver:
    """
    Defines a solver object used for a shortest ray path simulation.

    Parameters:
    ---
    grid: object, an SRP grid
    """

    def __init__(self, grid):
        self.grid = grid

    def solve(self, source_indices, with_points=False):
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
            self.tfs, self.points = sp.shortest_path(
                self.grid.edges,
                return_predecessors=with_points,
                indices=source_indices)
        else:
            self.tfs = sp.shortest_path(self.grid.edges,
                                        return_predecessors=with_points,
                                        indices=source_indices)
        #print('Search ended.')

    def interpolate_tf_field(self, external=False, external_grid=None):
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

    def calculate_ray_paths(self, end=[]):
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

    def fold_paths(self, paths):
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
