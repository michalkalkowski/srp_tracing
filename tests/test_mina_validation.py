"""
Validation test: runs the real MINA weld example (examples/test_mina.py)
end to end against its finite-element reference data, checks the SRP
solver's error stays within bounds, and saves an error-map plot to
output/.

examples/test_mina.py used to scale its material properties by 1e3 (c) and
1e-9 (rho), six orders of magnitude off the mm/us convention
WaveBasis.set_material_props actually expects (see examples/test_ogilvy.py
for the same material given correctly). It also compared against the FE
reference data without converting seconds to microseconds first. Both are
fixed now; this test is what confirms the fix produces physically correct,
FE-agreeing results rather than just "no crash" -- the same role
test_ogilvy_validation.py plays for the Ogilvy weld.

Needs the unpublished ``mina`` sibling package (see environment.yml);
skips cleanly if it isn't installed. Runs in-process -- srp_tracing has no
unpublished dependency of its own (see srp_tracing/_wave_physics.py).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("mina", reason="requires the unpublished mina package")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")


def _error_stats(tofs_srp, target):
    diff = tofs_srp - target
    abs_diff = np.abs(diff)
    valid = ~np.isnan(diff)
    rel = abs_diff / np.abs(target)
    return {
        "abs_diff": abs_diff,
        "valid": valid,
        "mae": np.nanmean(abs_diff),
        "rmse": np.sqrt(np.nanmean(diff**2)),
        "bias": np.nanmean(diff),
        "max_abs_error": np.nanmax(abs_diff),
        "mean_rel_error_pct": np.nanmean(rel) * 100,
        "max_rel_error_pct": np.nanmax(rel) * 100,
    }


@pytest.fixture(scope="module")
def mina_run():
    import runpy
    import sys

    cwd = os.getcwd()
    os.chdir(EXAMPLES_DIR)
    sys.path.insert(0, EXAMPLES_DIR)
    try:
        ns = runpy.run_path(os.path.join(EXAMPLES_DIR, "test_mina.py"))
    finally:
        os.chdir(cwd)
    plt.close("all")
    return ns


def test_mina_srp_matches_fe_reference_2mhz(mina_run):
    tofs_srp = np.asarray(mina_run["tofs_srp"])
    target = np.asarray(mina_run["target"])
    stats = _error_stats(tofs_srp, target)
    assert stats["valid"].sum() > 0

    # Generous bounds around the currently observed error (MAE ~0.012 us,
    # RMSE ~0.015 us). A regression back to the old unit-scaling bug would
    # blow this bound by roughly six orders of magnitude, so it's not a
    # brittle threshold.
    assert stats["mae"] < 0.1
    assert stats["rmse"] < 0.1
    assert stats["max_rel_error_pct"] < 5.0


def test_mina_srp_matches_fe_reference_4mhz(mina_run):
    tofs_srp = np.asarray(mina_run["tofs_srp"])
    target_4mhz = np.asarray(mina_run["target_4MHz"])
    stats = _error_stats(tofs_srp, target_4mhz)
    assert stats["valid"].sum() > 0
    assert stats["mae"] < 0.1
    assert stats["rmse"] < 0.1


def test_mina_error_map_plot(mina_run):
    tofs_srp = np.asarray(mina_run["tofs_srp"])
    target = np.asarray(mina_run["target"])
    stats = _error_stats(tofs_srp, target)
    abs_diff = np.where(stats["valid"], stats["abs_diff"], np.nan)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mina_validation_error_map.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(abs_diff, origin="lower", cmap="viridis")
    ax.set_xlabel("target index")
    ax.set_ylabel("source index")
    ax.set_title(
        "SRP vs FE absolute error (MINA weld)\n"
        f"mean={stats['mae']:.4f} us   max={stats['max_abs_error']:.4f} us   "
        f"mean rel={stats['mean_rel_error_pct']:.3f}%   "
        f"max rel={stats['max_rel_error_pct']:.3f}%",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="absolute error (us)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    assert os.path.exists(out_path)
