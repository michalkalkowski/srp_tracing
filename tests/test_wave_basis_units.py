"""
Pinning tests for WaveBasis's unit-convention sanity check (see
set_material_props's docstring): calculate_wavespeeds warns when the
resulting velocity is implausible for a solid, which is what would happen
if c and rho were scaled by different powers of ten from the documented
GPa / g*cm**-3 convention -- exactly the mistake found in
examples/test_mina.py (c scaled by 1e3, rho by 1e-9, six orders of
magnitude off from examples/test_ogilvy.py's convention for the same
material).
"""
import numpy as np
import pytest

from srp_tracing import grid


def test_correct_unit_convention_does_not_warn(isotropic_material, recwarn):
    isotropic_material(vp=5.9)
    assert len(recwarn) == 0


def test_inconsistent_scaling_warns():
    # Same steel-like numbers as examples/test_mina.py's c/rho scaling,
    # relative to examples/test_ogilvy.py's for the same material.
    c = 1e3 * np.diag([255.61, 255.61, 255.61, 79.86, 79.86, 79.86])
    rho = 7.9e-9

    material = grid.WaveBasis(anisotropy=0, velocity_variant="group")
    material.set_material_props(c, rho)
    with pytest.warns(UserWarning, match="outside the plausible range"):
        material.calculate_wavespeeds()


def test_invalid_velocity_variant_raises():
    material = grid.WaveBasis(anisotropy=0, velocity_variant="not_a_real_variant")
    material.set_material_props(np.eye(6), 1.0)
    with pytest.raises(ValueError, match="Unknown velocity_variant"):
        material.calculate_wavespeeds()
