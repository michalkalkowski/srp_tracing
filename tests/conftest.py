"""
Shared pytest infrastructure for srp_tracing.

(The demo/validation scripts that used to live in this directory --
test_ogilvy.py, test_mina.py, etc. -- were not automated tests: no
assertions, gated behind further unpublished packages, ``plt.show()`` at
import time. They have moved to examples/.)
"""
import numpy as np
import pytest


def isotropic_stiffness(vp, vs, rho=1.0):
    """A diagonal stiffness matrix giving an exact, angle-independent P
    speed of vp (for vp > vs): for propagation along y (angle=0, the only
    angle WaveBasis.get_wavespeed ever queries for an isotropic material),
    the Christoffel matrix for this diagonal c is itself diagonal with
    eigenvalues {vs**2*rho, vp**2*rho, vs**2*rho}, so the sorted P-branch
    velocity is exactly vp."""
    c = np.zeros((6, 6))
    c[0, 0] = c[1, 1] = c[2, 2] = vp**2 * rho
    c[3, 3] = c[4, 4] = c[5, 5] = vs**2 * rho
    return c


@pytest.fixture
def isotropic_material():
    """Factory fixture: isotropic_material(vp, vs=..., rho=...) -> WaveBasis."""
    from srp_tracing import grid

    def _make(vp, vs=None, rho=1.0):
        if vs is None:
            vs = vp / 2
        material = grid.WaveBasis(anisotropy=0, velocity_variant="group")
        material.set_material_props(isotropic_stiffness(vp, vs, rho), rho)
        material.calculate_wavespeeds()
        return material

    return _make
