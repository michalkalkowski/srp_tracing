"""
Validation test: runs the real chamfer-trimmed Ogilvy weld example
(examples/test_ogilvy_simpl.py) end to end against its finite-element
reference data, checks the SRP solver's error stays within bounds, and
saves an error-map plot to output/.

This is the SimplRectGrid counterpart of test_ogilvy_validation.py (which
covers the same weld via the plain RectGrid). It matters as a validation
in its own right: SimplRectGrid.simplify_grid() used to silently zero out
the *entire* material map whenever called with its default arguments (see
the tidy-up plan / grid.py history), which every real script -- including
this one -- does. That means this exact scenario would previously have run
the whole domain as a single (parent) material, with no weld region at
all, while still "succeeding" with no error. This test is what confirms
the fix actually produces physically correct, FE-agreeing results rather
than just "no crash".

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
def ogilvy_simpl_run():
    import runpy
    import sys

    cwd = os.getcwd()
    os.chdir(EXAMPLES_DIR)
    sys.path.insert(0, EXAMPLES_DIR)
    try:
        ns = runpy.run_path(os.path.join(EXAMPLES_DIR, "test_ogilvy_simpl.py"))
    finally:
        os.chdir(cwd)
    plt.close("all")
    return np.asarray(ns["tofs_srp"]), np.asarray(ns["target"])


def test_ogilvy_simpl_srp_matches_fe_reference(ogilvy_simpl_run):
    tofs_srp, target = ogilvy_simpl_run
    stats = _error_stats(tofs_srp, target)
    assert stats["valid"].sum() > 0

    # Generous bounds around the currently observed error (MAE ~0.0096 us,
    # max relative error ~0.78 %): enough headroom to not be brittle to
    # floating point noise, tight enough to catch a real regression (e.g.
    # a reintroduction of the material_map wipeout bug, which would push
    # these numbers up sharply since the weld region would stop being
    # modelled at all).
    assert stats["mae"] < 0.05
    assert stats["rmse"] < 0.05
    assert stats["max_rel_error_pct"] < 3.0


def test_ogilvy_simpl_error_map_plot(ogilvy_simpl_run):
    tofs_srp, target = ogilvy_simpl_run
    stats = _error_stats(tofs_srp, target)
    abs_diff = np.where(stats["valid"], stats["abs_diff"], np.nan)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "ogilvy_simpl_validation_error_map.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(abs_diff, origin="lower", cmap="viridis")
    ax.set_xlabel("target index")
    ax.set_ylabel("source index")
    ax.set_title(
        "SRP vs FE absolute error (Ogilvy weld, chamfer-trimmed SimplRectGrid)\n"
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
