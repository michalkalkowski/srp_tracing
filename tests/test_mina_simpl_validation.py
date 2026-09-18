"""
Validation test: runs the real chamfer-trimmed MINA weld example
(examples/test_mina_simpl.py) end to end against its finite-element
reference data, checks the SRP solver's error stays within bounds, and
saves an error-map plot to output/.

SimplRectGrid counterpart of test_mina_validation.py, same relationship as
test_ogilvy_simpl_validation.py is to test_ogilvy_validation.py. Validated
against ../data/SRP_validation_mina.npy (the same full 64x64 reference
test_mina.py uses), not the chamfer-specific SRP_validation_mina_chamfer*
files: those are (64, 32) rather than (64, 64) -- a real, pre-existing
shape difference from the other reference files whose exact column
semantics weren't investigated here, so no full-matrix comparison is made
against them. The example script itself still uses them for their original
purpose (per-sensor line plots), unaffected by this.

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
def mina_simpl_run():
    import runpy
    import sys

    cwd = os.getcwd()
    os.chdir(EXAMPLES_DIR)
    sys.path.insert(0, EXAMPLES_DIR)
    try:
        ns = runpy.run_path(os.path.join(EXAMPLES_DIR, "test_mina_simpl.py"))
    finally:
        os.chdir(cwd)
    plt.close("all")
    return ns


def test_mina_simpl_srp_matches_fe_reference(mina_simpl_run):
    tofs_srp = np.asarray(mina_simpl_run["tofs_srp"])
    target = np.asarray(mina_simpl_run["target"])
    stats = _error_stats(tofs_srp, target)
    assert stats["valid"].sum() > 0

    # Generous bounds around the currently observed error (MAE ~0.018 us,
    # RMSE ~0.023 us) -- a bit higher than the plain-RectGrid case
    # (test_mina_validation.py: MAE ~0.012 us), consistent with the same
    # RectGrid-vs-SimplRectGrid gap seen for the Ogilvy weld.
    assert stats["mae"] < 0.1
    assert stats["rmse"] < 0.1
    assert stats["max_rel_error_pct"] < 5.0


def test_mina_simpl_error_map_plot(mina_simpl_run):
    tofs_srp = np.asarray(mina_simpl_run["tofs_srp"])
    target = np.asarray(mina_simpl_run["target"])
    stats = _error_stats(tofs_srp, target)
    abs_diff = np.where(stats["valid"], stats["abs_diff"], np.nan)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mina_simpl_validation_error_map.png")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(abs_diff, origin="lower", cmap="viridis")
    ax.set_xlabel("target index")
    ax.set_ylabel("source index")
    ax.set_title(
        "SRP vs FE absolute error (MINA weld, chamfer-trimmed SimplRectGrid)\n"
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
