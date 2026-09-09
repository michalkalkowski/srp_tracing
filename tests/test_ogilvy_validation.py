"""
Validation test: runs the real Ogilvy weld example (examples/test_ogilvy.py)
end to end against its finite-element reference data, checks the SRP
solver's error against that reference stays within bounds, and saves an
error-map plot to output/.

Needs the unpublished ``ogilvy_weld`` sibling package (see
environment.yml); skips cleanly if it isn't installed, so it doesn't break
the suite on machines without it (fresh clones, CI without the extra pip
install). Unlike that package, srp_tracing itself has no unpublished
dependency any more -- WaveBasis's anisotropic velocity solver is
self-contained (see srp_tracing/_wave_physics.py), so this test runs
in-process rather than needing the subprocess isolation an earlier version
of this file required.
"""
import os
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("ogilvy", reason="requires the unpublished ogilvy_weld package")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")


@pytest.fixture(scope="module")
def ogilvy_run():
    """Executes examples/test_ogilvy.py (weld setup, graph construction,
    shortest-path solve) once and returns its (tofs_srp, target) arrays.
    Module-scoped since the run is expensive and both tests below want the
    same result."""
    cwd = os.getcwd()
    os.chdir(EXAMPLES_DIR)
    sys.path.insert(0, EXAMPLES_DIR)
    try:
        ns = runpy.run_path(os.path.join(EXAMPLES_DIR, "test_ogilvy.py"))
    finally:
        os.chdir(cwd)
    plt.close("all")
    return np.asarray(ns["tofs_srp"]), np.asarray(ns["target"])


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


def test_ogilvy_srp_matches_fe_reference(ogilvy_run):
    tofs_srp, target = ogilvy_run
    stats = _error_stats(tofs_srp, target)
    assert stats["valid"].sum() > 0

    # Generous bounds around the currently observed error (MAE ~0.008 us,
    # max relative error ~0.6 %): enough headroom to not be brittle to
    # floating point noise, tight enough to catch a real regression.
    assert stats["mae"] < 0.05
    assert stats["rmse"] < 0.05
    assert stats["max_rel_error_pct"] < 3.0


def test_ogilvy_error_map_plot(ogilvy_run):
    tofs_srp, target = ogilvy_run
    stats = _error_stats(tofs_srp, target)
    abs_diff = np.where(stats["valid"], stats["abs_diff"], np.nan)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "ogilvy_validation_error_map.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(abs_diff, origin="lower", cmap="viridis")
    ax.set_xlabel("target index")
    ax.set_ylabel("source index")
    ax.set_title(
        "SRP vs FE absolute error (Ogilvy weld)\n"
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
