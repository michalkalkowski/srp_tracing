"""
Validation test: runs the real backwall-relay Ogilvy weld example
(examples/test_ogilvy_backwall_relay.py) end to end against its
finite-element pulse-echo reference data, checks the error stays within
bounds, and saves an error-map plot to output/.

This is the pulse-echo counterpart of test_ogilvy_validation.py. It exists
to confirm that solver.combine_via_boundary() -- the backwall/frontwall
relay mechanism that replaces the old mirrored-domain trick (see that
example's docstring, and tests/test_backwall_relay.py for the isotropic,
closed-form validation) -- reproduces real, previously-validated pulse-echo
results on the actual anisotropic weld, not just an idealised isotropic
case.

Needs the unpublished ``ogilvy_weld`` sibling package (see
environment.yml); skips cleanly if it isn't installed. Runs in-process --
srp_tracing has no unpublished dependency of its own (see
srp_tracing/_wave_physics.py).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("ogilvy", reason="requires the unpublished ogilvy_weld package")

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
def ogilvy_backwall_relay_run():
    import runpy
    import sys

    cwd = os.getcwd()
    os.chdir(EXAMPLES_DIR)
    sys.path.insert(0, EXAMPLES_DIR)
    try:
        ns = runpy.run_path(os.path.join(EXAMPLES_DIR, "test_ogilvy_backwall_relay.py"))
    finally:
        os.chdir(cwd)
    plt.close("all")
    return np.asarray(ns["tofs_srp"]), np.asarray(ns["target"])


def test_backwall_relay_srp_matches_fe_reference(ogilvy_backwall_relay_run):
    tofs_srp, target = ogilvy_backwall_relay_run
    stats = _error_stats(tofs_srp, target)
    assert stats["valid"].sum() > 0

    # Generous bounds around the currently observed error (MAE ~0.030 us,
    # max relative error ~0.58 %). Larger absolute error than the direct
    # pitch-catch validations (test_ogilvy_validation.py: MAE ~0.009 us) is
    # expected -- pulse-echo paths are roughly twice as long -- but the
    # relative error is comparable, which is the more meaningful figure
    # here.
    assert stats["mae"] < 0.15
    assert stats["rmse"] < 0.15
    assert stats["max_rel_error_pct"] < 3.0


def test_backwall_relay_error_map_plot(ogilvy_backwall_relay_run):
    tofs_srp, target = ogilvy_backwall_relay_run
    stats = _error_stats(tofs_srp, target)
    abs_diff = np.where(stats["valid"], stats["abs_diff"], np.nan)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "ogilvy_backwall_relay_validation_error_map.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(abs_diff, origin="lower", cmap="viridis")
    ax.set_xlabel("target index")
    ax.set_ylabel("source index")
    ax.set_title(
        "SRP vs FE absolute error (Ogilvy weld, backwall-relay pulse-echo)\n"
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
