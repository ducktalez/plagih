"""
EM-Loop runner for the NN+GP pipeline.

Each iteration:
  1. GP phase  – evolve symbolic candidates against current target (residual)
  2. NN phase  – find the smallest MLP that matches baseline accuracy when
                 augmented with GP candidate outputs as extra features
  3. Residual  – compute the per-row error the GP candidates could not explain;
                 this becomes the target for the next GP iteration

All GP hyperparameters are fixed across iterations (same Evolution config).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from benchmarks.nn_gp.data_utils import (
    build_gp_feature_matrix,
    build_nn_input,
    compute_residual,
)
from benchmarks.nn_gp.experiment_tracker import ExperimentTracker, IterationResult
from benchmarks.nn_gp.nn_models import _ARCHITECTURE_GRID, TrainResult, find_minimal_nn, train_nn

# ---------------------------------------------------------------------------
# GP config dataclass (keeps the call-site clean)
# ---------------------------------------------------------------------------


@dataclass
class GPConfig:
    """Parameters forwarded to ``ExplainableGP.create()`` for every iteration.

    Attributes:
        symbols: Input feature column names.
        target_col: Name of the target column in the training DataFrame.
        rootdir_base: Base directory; per-iteration subdirs are created automatically.
        pop_max_size: GP population size.
        gen_end: Generations per GP phase.
        depth_max: Maximum tree depth.
        nodes_max: Maximum nodes per tree.
        preset: Operator preset (``'math_simple'``, ``'math_full'``, …).
        operators: Custom operator dict (overrides ``preset`` when provided).
        clip_range: Optional prediction clipping range.
        error_metric: ``'mse'``, ``'rmse'``, ``'mae'``.
        parallel: Worker count (0 = sequential).
        enable_analysis: Enable GP plots/backups (slow; set False for speed).
        extra_kwargs: Any additional kwargs passed to ``ExplainableGP.create()``.
    """

    symbols: List[str]
    target_col: str = "action"
    rootdir_base: Path = Path(".results/nn_gp")
    pop_max_size: int = 100
    gen_end: int = 30
    depth_max: int = 6
    nodes_max: int = 30
    preset: str = "math_simple"
    operators: Optional[Dict] = None
    clip_range: Optional[Tuple[float, float]] = (0.0, 1.0)
    error_metric: str = "mse"
    parallel: int = 0
    enable_analysis: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NNConfig:
    """Parameters forwarded to the NN grid-search / training."""

    epochs: int = 400
    lr: float = 1e-3
    batch_size: int = 64
    patience: int = 40
    tolerance: float = 0.05
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Strategies used in every GP generation
# ---------------------------------------------------------------------------

_DEFAULT_STRATEGIES_SPEC = [
    dict(name="reproduction", rate=0.2, tournament_n=3),
    dict(name="mutation", rate=0.4, depth_goal=3, p_term=0.3),
    dict(name="random_new", rate=0.2, depths=[2, 3, 4], p_term=0.1),
    dict(name="crossover", rate=0.2, crossover=True, tournament_n=3),
]


def _make_strategies():
    from plagih.parallel import Strategy

    return [Strategy(**s) for s in _DEFAULT_STRATEGIES_SPEC]


# ---------------------------------------------------------------------------
# Single GP phase
# ---------------------------------------------------------------------------


def run_gp_phase(
    df_train_with_target: pd.DataFrame,
    gp_cfg: GPConfig,
    iter_id: int,
) -> Tuple[Any, List[Any]]:
    """Run one GP phase and return ``(gp_instance, pareto_candidates)``.

    The DataFrame must already have the correct (possibly residual) target column.
    A per-iteration subdirectory is created under ``gp_cfg.rootdir_base``.
    """
    from plagih.trees import ExplainableGP

    run_dir = Path(gp_cfg.rootdir_base) / f"iter_{iter_id:02d}_gp"

    create_kwargs: Dict[str, Any] = dict(
        symbols=gp_cfg.symbols,
        df_train=df_train_with_target,
        rootdir=run_dir,
        depth_max=gp_cfg.depth_max,
        nodes_max=gp_cfg.nodes_max,
        pop_max_size=gp_cfg.pop_max_size,
        gen_end=gp_cfg.gen_end,
        clip_range=gp_cfg.clip_range,
        error_metric=gp_cfg.error_metric,
        target_column=gp_cfg.target_col,
        parallel=gp_cfg.parallel,
        enable_analysis=gp_cfg.enable_analysis,
        verbose=True,
    )
    if gp_cfg.operators is not None:
        create_kwargs["operators"] = gp_cfg.operators
    else:
        create_kwargs["preset"] = gp_cfg.preset
    create_kwargs.update(gp_cfg.extra_kwargs)

    gp = ExplainableGP.create(**create_kwargs)
    strategies = _make_strategies()
    gp.gen_create_initial()
    for _ in range(gp_cfg.gen_end):
        gp.run_generation(strategies)

    return gp, list(gp.paretofront)


# ---------------------------------------------------------------------------
# EM loop
# ---------------------------------------------------------------------------


def run_em_loop(
    df_norm: pd.DataFrame,
    gp_cfg: GPConfig,
    nn_cfg: NNConfig,
    tracker: ExperimentTracker,
    n_iterations: int = 3,
) -> List[IterationResult]:
    """Run the full EM loop.

    Args:
        df_norm: Fully normalised DataFrame (features + target column).
        gp_cfg: GP hyperparameters (fixed across all iterations).
        nn_cfg: NN search hyperparameters.
        tracker: ExperimentTracker instance (pre-seeded with baseline metrics).
        n_iterations: Number of EM iterations to run.

    Returns:
        List of IterationResult, one per iteration.
    """
    target_col = gp_cfg.target_col
    target_norm = df_norm[target_col].to_numpy(dtype=np.float32)
    feature_cols = [c for c in df_norm.columns if c != target_col]

    current_target = target_norm.copy()  # Updated per iteration
    all_results: List[IterationResult] = []

    baseline_mse = tracker.baseline_mse

    for it in range(n_iterations):
        print(f"\n{'=' * 60}")
        print(
            f"  EM ITERATION {it}  (residual_mse going in: {np.mean((current_target - current_target.mean()) ** 2):.5f})"
        )
        print(f"{'=' * 60}")

        iter_result = IterationResult(iteration=it)

        # ----------------------------------------------------------------
        # Build training DataFrame with current (residual) target
        # ----------------------------------------------------------------
        df_gp = df_norm[feature_cols].copy()
        df_gp[target_col] = current_target

        # ----------------------------------------------------------------
        # GP phase
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        gp, pareto = run_gp_phase(df_gp, gp_cfg, it)
        iter_result.gp_runtime_s = time.perf_counter() - t0
        iter_result.gp_pareto_size = len(pareto)
        iter_result.gp_expressions = [str(c.tree) for c in pareto]

        # GP best MSE (best candidate on current target)
        if pareto:
            best_cand = min(pareto, key=lambda c: c.fitness)
            iter_result.gp_best_mse = float(best_cand.fitness)

        print(f"  GP done in {iter_result.gp_runtime_s:.1f}s  |  pareto={len(pareto)}")

        # ----------------------------------------------------------------
        # Build GP feature matrix from Pareto candidates
        # ----------------------------------------------------------------
        gp_feats = build_gp_feature_matrix(pareto, df_norm)
        X_enriched = build_nn_input(df_norm, target_col, gp_feats)

        # ----------------------------------------------------------------
        # NN phase: find minimal architecture that matches baseline accuracy
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        nn_result, _ = find_minimal_nn(
            X_enriched,
            target_norm,  # always optimise against original target
            target_mse=baseline_mse,
            tolerance=nn_cfg.tolerance,
            epochs=nn_cfg.epochs,
            lr=nn_cfg.lr,
            batch_size=nn_cfg.batch_size,
            patience=nn_cfg.patience,
            device=nn_cfg.device,
        )
        iter_result.nn_runtime_s = time.perf_counter() - t0
        iter_result.nn_hidden_sizes = nn_result.hidden_sizes
        iter_result.nn_param_count = nn_result.param_count
        iter_result.nn_mse = nn_result.final_mse
        iter_result.nn_loss_curve = nn_result.loss_curve

        print(
            f"  NN done in {iter_result.nn_runtime_s:.1f}s  |  "
            f"arch={nn_result.hidden_sizes}  params={nn_result.param_count:,}  "
            f"mse={nn_result.final_mse:.5f}"
        )

        # ----------------------------------------------------------------
        # Compute residual for next iteration
        # ----------------------------------------------------------------
        current_target = compute_residual(gp_feats, target_norm)
        iter_result.residual_mse = float(np.mean((current_target - 0.5) ** 2))

        # ----------------------------------------------------------------
        # Persist
        # ----------------------------------------------------------------
        tracker.add_iteration(iter_result)
        all_results.append(iter_result)

    return all_results


# ---------------------------------------------------------------------------
# Baseline-only helper
# ---------------------------------------------------------------------------


def run_baseline_only(
    df_norm: pd.DataFrame,
    target_col: str,
    nn_cfg: NNConfig,
    max_arch_count: int = 9,
) -> TrainResult:
    """Train a baseline NN on raw features only (no GP).

    Tests up to ``max_arch_count`` architectures from ``_ARCHITECTURE_GRID``
    (smallest first) and returns the one with the lowest MSE.  That MSE is
    used as the reference target for all EM-loop NN searches.

    Args:
        df_norm: Normalised DataFrame.
        target_col: Target column name.
        nn_cfg: NN training config.
        max_arch_count: How many architectures to try (default 9, up to [64,32]).
                        Set to len(_ARCHITECTURE_GRID) for the full search.

    Returns:
        TrainResult of the best architecture found.
    """
    from benchmarks.nn_gp.data_utils import build_nn_input

    target_norm = df_norm[target_col].to_numpy(dtype=np.float32)
    X_raw = build_nn_input(df_norm, target_col)
    grid = _ARCHITECTURE_GRID[:max_arch_count]

    print(f"\n[baseline] Training baseline NN on raw features ({len(grid)} architectures) ...")
    t0 = time.perf_counter()
    all_results: List[TrainResult] = []
    for hidden_sizes in grid:
        r = train_nn(
            X_raw,
            target_norm,
            hidden_sizes,
            epochs=nn_cfg.epochs,
            lr=nn_cfg.lr,
            batch_size=nn_cfg.batch_size,
            patience=nn_cfg.patience,
            device=nn_cfg.device,
        )
        all_results.append(r)
        print(f"  [baseline] {r}")
    result = min(all_results, key=lambda r_: r_.final_mse)
    elapsed = time.perf_counter() - t0
    print(
        f"[baseline] done in {elapsed:.1f}s  |  "
        f"arch={result.hidden_sizes}  params={result.param_count:,}  mse={result.final_mse:.5f}"
    )
    return result
