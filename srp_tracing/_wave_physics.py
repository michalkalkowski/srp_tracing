#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part of srp_tracing.

Anisotropic phase/group velocity calculation, used by grid.WaveBasis to
turn an elasticity matrix into a direction-dependent wave speed.

Ported from the external `raytracer` package (raytracer.refraction /
raytracer.miscallaneous), which srp_tracing used to depend on for just this
one function. Only calculate_slowness and the two helpers it calls are
included here -- raytracer itself is a much larger package covering
reflection/transmission/refraction at interfaces, none of which srp_tracing
uses. Logic is unchanged from the original.

Reference:
- Rokhlin, S.I., Bolland, T.K., Adler, L., 1986. Reflection and refraction
  of elastic waves on a plane interface between two generally anisotropic
  media. The Journal of the Acoustical Society of America 79, 906-918.
  https://doi.org/10.1121/1.393764

@author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) Michal K Kalkowski (MIT License)
"""
import itertools

import numpy as np


def matrix_to_tensor(c):
    """
    Converts the Voigt elasticity matrix to the full 3x3x3x3 elasticity
    tensor. Some inspiration taken from: https://github.com/andreww/theia_tools

    Parameters:
    ---
    c : array_like
        6x6 elasticity matrix (Voigt notation).

    Returns:
    ---
    c_out : array_like 3x3x3x3 elasticity tensor.
    """
    voigt_lookup = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
    tensor = np.zeros([3, 3, 3, 3], 'complex')
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        voigt_i = voigt_lookup[i, j]
        voigt_j = voigt_lookup[k, l]
        tensor[i, j, k, l] = c[voigt_i, voigt_j]
    return tensor


def christoffel_matrix(c, n):
    """
    Calculates the Christoffel tensor.
    Parameters:
    ---
    c : ndarray, stiffness tensor
    n : ndarray, wavefront normal

    Returns:
    ---
    chr : ndarray, Christoffel matrix
    """
    n = n.squeeze()
    if len(n.shape) == 1:
        return np.sum(c*n[None, :, None, None]
                      *n[None, None, :, None], axis=(1, 2))
    elif len(n.shape) == 2:
        return np.sum(c[None, :]*n[:, None, :, None, None]
                      *n[:, None, None, :, None], axis=(2, 3))


def calculate_group_velocity(c, rho, m, polarisation):
    """
    Calculates group velocity for a given polarisation, slowness and
    material properties.

    Parameters:
    ---
    c: ndarray, elasticity tensor
    rho: float, mass density
    m: ndarray, slowness
    polarisation: ndarray, polarisation vector

    Returns:
    ---
    group_velocity: ndarray, the group velocity vector
    """
    return np.sum(c[:, :, :, :, None]
                  *polarisation[None, :, None, None, :]
                  *polarisation[None, None, :, None, :]
                  *m[None, None, None, :, :],
                  axis=(1, 2, 3))/rho


def calculate_slowness(c, rho, angles, n=0, previous_pol=0):
    """
    Calculates slowness diagram.
    DOCString to be completed.

    Parameters:
    ---

    Returns:
    ---

    """
    # define the range for incident angle (preliminary)
    if type(n) != np.ndarray:
        if type(angles) != np.ndarray:
            angles = np.array([angles])
        n = np.column_stack([np.zeros(len(angles)), np.cos(angles), np.sin(angles)])
    c_p = np.zeros([len(n), 3], 'complex')
    m = np.zeros([len(n), 3, 3], 'complex')
    group_velocity = np.zeros([len(n), 3, 3], 'complex')
    polarisation = np.zeros([len(n), 3, 3], 'complex')
    for i in range(len(n)):
        christoffel = christoffel_matrix(matrix_to_tensor(c), n[i])
        val, vec = np.linalg.eig(christoffel)
        if i > 0:
            orthogonality = polarisation[i - 1].conj().T.dot(vec)
            order = np.argmax(abs(orthogonality), axis=1)
            order_values = np.max(abs(orthogonality), axis=1)
            ideal = [0, 1, 2]
            if set(ideal) != set(order):
                repetitions = [list(order).count(i) for i in ideal]
                which_idx_repeats = repetitions.index(2)
                which_wave_repeats = np.where(np.array(order) == which_idx_repeats)
                highest_match = np.argmax(order_values[which_wave_repeats])
                to_change = np.delete(which_wave_repeats, highest_match)
                order[to_change] = list(set(ideal) - set(order))[0]
            polarisation[i] = vec[:, order]
            c_p[i] = (val[order]/rho)**0.5
        elif i == 0 and type(previous_pol) == np.ndarray:
            orthogonality = previous_pol.conj().T.dot(vec)
            order = np.argmax(abs(orthogonality), axis=1)
            order_values = np.max(abs(orthogonality), axis=1)
            ideal = [0, 1, 2]
            if set(ideal) != set(order):
                repetitions = [list(order).count(i) for i in ideal]
                which_idx_repeats = repetitions.index(2)
                which_wave_repeats = np.where(np.array(order) == which_idx_repeats)
                highest_match = np.argmax(order_values[which_wave_repeats])
                to_change = np.delete(which_wave_repeats, highest_match)
                order[to_change] = list(set(ideal) - set(order))[0]
            polarisation[i] = vec[:, order]
            c_p[i] = (val[order]/rho)**0.5
        else:
            order = np.argsort((val/rho)**0.5)[::-1]
            polarisation[i] = vec[:, order]
            c_p[i] = (val[order]/rho)**0.5
        m[i] = n[i].reshape(-1, 1)/c_p[i].reshape(1, -1)
        group_velocity[i] = calculate_group_velocity(matrix_to_tensor(c), rho, m[i], polarisation[i])
    # final sorting in an descending speed order
    order = np.argsort(np.mean(np.linalg.norm(m, axis=1), axis=0))
    c_p = c_p[:, order]
    m = m[:, :, order]
    polarisation = polarisation[:, :, order]
    group_velocity = group_velocity[:, :, order]
    return c_p.squeeze(), m.squeeze(), \
            group_velocity.squeeze(), polarisation.squeeze()
