"""
MountainCar entry point for the NN+GP EM-loop pipeline.

Usage:
    # Full EM run (GP + NN, 3 iterations):
    python benchmarks/nn_gp/run_mc.py

    # Quick sanity check – baseline NN only, no GP:
    python benchmarks/nn_gp/run_mc.py --baseline-only

    # Custom number of EM iterations:
    python benchmarks/nn_gp/run_mc.py --iterations 2

The MountainCar benchmark predicts 'action' (0/1/2) from
'cartPos' and 'cartVel'.  Results are written to
```.results/nn_gp/<timestamp>/```.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Make sure project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.nn_gp.data_utils import normalize_dataset
from benchmarks.nn_gp.em_loop import GPConfig, NNConfig, run_baseline_only, run_em_loop
from benchmarks.nn_gp.experiment_tracker import ExperimentMeta, ExperimentTracker
from benchmarks.nn_gp.paper_blueprint import generate_blueprint
from benchmarks.nn_gp.paper_figures import generate_all_figures

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MC_CSV = _PROJECT_ROOT / "benchmarks" / "mc" / "gp_files" / "samples200.csv"
RESULTS_BASE = _PROJECT_ROOT / ".results" / "nn_gp"

TARGET_COL = "action"
SYMBOLS = ["cartPos", "cartVel"]


# ---------------------------------------------------------------------------
# MountainCar operator set (matches demo_minimal in plagih_gp.py)
# ---------------------------------------------------------------------------


def _mc_operators():
    from plagih.trees import Abs, Add, Cos, Div, Ifte, Le, Lt, Max, Min, Mul, Sin, Square, Sub

    return {
        Add: 2,
        Mul: 2,
        Sub: 1,
        Div: 1,
        Abs: 1,
        Square: 1,
        Sin: 0.5,
        Cos: 0.5,
        Min: 1,
        Max: 1,
        Lt: 1,
        Le: 1,
        Ifte: 1,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    n_iterations: int = 3,
    gp_pop_size: int = 50,
    gp_gen_end: int = 20,
    baseline_only: bool = False,
    show_figures: bool = False,
    fast: bool = False,
) -> Path:
    """Run the full NN+GP EM pipeline on MountainCar data.

    Args:
        n_iterations: Number of EM iterations (GP + NN each).
        gp_pop_size: GP population size per iteration.
        gp_gen_end: Number of GP generations per iteration.
        baseline_only: If True, only run the baseline NN (no GP).
        show_figures: If True, call plt.show() after figure generation.
        fast: If True, use reduced epochs/architectures for quick dev runs.

    Returns:
        Path to the generated PAPER_BLUEPRINT.md.
    """
    nn_cfg = NNConfig(
        epochs=100 if fast else 400,
        lr=1e-3,
        batch_size=64,
        patience=20 if fast else 40,
        tolerance=0.05,
    )
    # How many baseline architectures to try (fewer = faster for dev)
    baseline_max_arch = 5 if fast else 9

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = RESULTS_BASE / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  NN+GP EM Pipeline — MountainCar{'  [FAST]' if fast else ''}")
    print(f"  Results: {results_dir}")
    print(f"{'=' * 60}\n")

    # ----------------------------------------------------------------
    # Load & normalise
    # ----------------------------------------------------------------
    df_raw = pd.read_csv(MC_CSV)
    print(f"Loaded {len(df_raw)} rows from {MC_CSV.name}")
    print(f"Columns: {list(df_raw.columns)}")

    df_norm, feat_scaler, tgt_scaler = normalize_dataset(df_raw, TARGET_COL)
    feature_cols = [c for c in df_norm.columns if c != TARGET_COL]
    n_features = len(feature_cols)

    print(f"Normalised — features: {feature_cols}  target: {TARGET_COL}")

    # ----------------------------------------------------------------
    # Configs
    # ----------------------------------------------------------------
    gp_cfg = GPConfig(
        symbols=SYMBOLS,
        target_col=TARGET_COL,
        rootdir_base=results_dir / "gp_runs",
        pop_max_size=gp_pop_size,
        gen_end=gp_gen_end,
        depth_max=6,
        nodes_max=30,
        operators=_mc_operators(),
        clip_range=(0.0, 1.0),
        error_metric="mse",
        parallel=0,
        enable_analysis=False,
    )

    # ----------------------------------------------------------------
    # Experiment tracker
    # ----------------------------------------------------------------
    tracker = ExperimentTracker(results_dir, benchmark="MountainCar")

    # ----------------------------------------------------------------
    # Baseline NN
    # ----------------------------------------------------------------
    baseline_result = run_baseline_only(df_norm, TARGET_COL, nn_cfg, max_arch_count=baseline_max_arch)

    meta = ExperimentMeta(
        benchmark="MountainCar",
        target_col=TARGET_COL,
        n_train_rows=len(df_norm),
        n_features=n_features,
        baseline_nn_mse=baseline_result.final_mse,
        baseline_nn_param_count=baseline_result.param_count,
        baseline_nn_hidden_sizes=baseline_result.hidden_sizes,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        gp_config={
            "pop_max_size": gp_cfg.pop_max_size,
            "gen_end": gp_cfg.gen_end,
            "depth_max": gp_cfg.depth_max,
            "nodes_max": gp_cfg.nodes_max,
            "preset": gp_cfg.preset,
            "clip_range": list(gp_cfg.clip_range) if gp_cfg.clip_range else None,
            "error_metric": gp_cfg.error_metric,
            "parallel": gp_cfg.parallel,
        },
        nn_config={
            "epochs": nn_cfg.epochs,
            "lr": nn_cfg.lr,
            "batch_size": nn_cfg.batch_size,
            "patience": nn_cfg.patience,
            "tolerance": nn_cfg.tolerance,
        },
    )
    tracker.set_meta(meta)

    if baseline_only:
        print("\n[run_mc] --baseline-only: stopping after baseline NN.")
        tracker.finalize(notes="baseline-only run")
        generate_all_figures(tracker, show=show_figures)
        blueprint_path = generate_blueprint(tracker)
        return blueprint_path

    # ----------------------------------------------------------------
    # EM Loop
    # ----------------------------------------------------------------
    pareto_per_iter = []

    # Monkey-patch the loop so we can capture Pareto fronts for figure generation.
    _orig_run_gp = None
    try:
        import benchmarks.nn_gp.em_loop as _em_mod

        _orig_run_gp = _em_mod.run_gp_phase

        _captured_paretos = pareto_per_iter

        def _run_gp_and_capture(df_train_with_target, gp_cfg_, iter_id):
            gp, pareto = _orig_run_gp(df_train_with_target, gp_cfg_, iter_id)
            _captured_paretos.append(pareto)
            return gp, pareto

        _em_mod.run_gp_phase = _run_gp_and_capture

        run_em_loop(df_norm, gp_cfg, nn_cfg, tracker, n_iterations=n_iterations)
    finally:
        if _orig_run_gp is not None:
            import benchmarks.nn_gp.em_loop as _em_mod

            _em_mod.run_gp_phase = _orig_run_gp

    tracker.finalize()

    # ----------------------------------------------------------------
    # Figures + Blueprint
    # ----------------------------------------------------------------
    generate_all_figures(tracker, pareto_per_iter=pareto_per_iter, show=show_figures)
    blueprint_path = generate_blueprint(tracker)

    print(f"\n{'=' * 60}")
    print(f"  Done!  Paper blueprint: {blueprint_path}")
    print(f"{'=' * 60}\n")
    return blueprint_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN+GP EM pipeline — MountainCar")
    parser.add_argument("--iterations", type=int, default=3, help="Number of EM iterations")
    parser.add_argument("--gp-pop", type=int, default=50, help="GP population size")
    parser.add_argument("--gp-gen", type=int, default=20, help="GP generations per iteration")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline NN, skip GP")
    parser.add_argument("--fast", action="store_true", help="Faster dev run (fewer epochs/architectures)")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    blueprint = run(
        n_iterations=args.iterations,
        gp_pop_size=args.gp_pop,
        gp_gen_end=args.gp_gen,
        baseline_only=args.baseline_only,
        show_figures=args.show,
        fast=args.fast,
    )
    print(f"Blueprint: {blueprint}")
