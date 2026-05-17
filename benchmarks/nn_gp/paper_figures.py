"""
Automated figure generation for the NN+GP pipeline.

All figures are saved as both PDF (vector, for paper) and SVG (for web/markdown).
Functions accept an ``ExperimentTracker`` and write figures to
``tracker.results_dir / "figures/"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _savefig(fig, path: Path, stem: str) -> None:
    """Save a matplotlib figure as both PDF and SVG."""
    path.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path / f"{stem}.{ext}", bbox_inches="tight", dpi=150)


# ---------------------------------------------------------------------------
# Figure 1: EM progress (residual MSE + NN param count per iteration)
# ---------------------------------------------------------------------------


def plot_em_progress(tracker, save: bool = True) -> "plt.Figure":
    """Dual-Y plot: residual MSE (left) and NN param count (right) across iterations."""
    import matplotlib.pyplot as plt

    iters = [it.iteration for it in tracker.iterations]
    residuals = [it.residual_mse for it in tracker.iterations]
    params = [it.nn_param_count for it in tracker.iterations]
    nn_mses = [it.nn_mse for it in tracker.iterations]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_res = "#e74c3c"
    color_par = "#2980b9"
    color_nn = "#27ae60"

    ax1.set_xlabel("EM Iteration")
    ax1.set_ylabel("MSE", color=color_res)
    l1 = ax1.plot(iters, residuals, "o-", color=color_res, label="Residual MSE", linewidth=2)
    l2 = ax1.plot(iters, nn_mses, "s--", color=color_nn, label="NN MSE", linewidth=1.5)
    if tracker.baseline_mse < float("inf"):
        ax1.axhline(
            tracker.baseline_mse,
            color=color_nn,
            linestyle=":",
            linewidth=1.2,
            label=f"Baseline NN MSE ({tracker.baseline_mse:.4f})",
        )
    ax1.tick_params(axis="y", labelcolor=color_res)

    ax2 = ax1.twinx()
    ax2.set_ylabel("NN Parameters", color=color_par)
    l3 = ax2.bar(iters, params, color=color_par, alpha=0.25, label="NN params")
    if tracker.baseline_param_count:
        ax2.axhline(
            tracker.baseline_param_count,
            color=color_par,
            linestyle=":",
            linewidth=1.2,
            label=f"Baseline params ({tracker.baseline_param_count:,})",
        )
    ax2.tick_params(axis="y", labelcolor=color_par)

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title("EM Loop Progress: Residual & NN Size")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iters)
    fig.tight_layout()

    if save:
        _savefig(fig, tracker.results_dir / "figures", "em_progress")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: NN size comparison (baseline vs each EM iteration)
# ---------------------------------------------------------------------------


def plot_nn_size_comparison(tracker, save: bool = True) -> "plt.Figure":
    """Bar chart comparing baseline NN param count vs each EM iteration."""
    import matplotlib.pyplot as plt

    labels = ["Baseline"] + [f"Iter {it.iteration}" for it in tracker.iterations]
    counts = [tracker.baseline_param_count] + [it.nn_param_count for it in tracker.iterations]
    colors = ["#95a5a6"] + ["#3498db"] * len(tracker.iterations)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("NN Size: Baseline vs GP-Enriched Iterations")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        _savefig(fig, tracker.results_dir / "figures", "nn_size_comparison")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Pareto scatter for a given iteration
# ---------------------------------------------------------------------------


def plot_pareto_scatter(
    pareto_candidates,
    iter_id: int,
    tracker,
    save: bool = True,
) -> "plt.Figure":
    """Scatter plot of the Pareto front (complexity vs fitness)."""
    import matplotlib.pyplot as plt

    if not pareto_candidates:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Pareto candidates", ha="center", va="center")
        return fig

    parsims = [c.parsimony for c in pareto_candidates]
    fitnesses = [c.fitness for c in pareto_candidates]
    labels = [str(c.tree)[:40] for c in pareto_candidates]

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(parsims, fitnesses, c=fitnesses, cmap="plasma_r", s=80, edgecolors="k", linewidths=0.5, zorder=3)
    for x, y, lbl in zip(parsims, fitnesses, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Parsimony (complexity)")
    ax.set_ylabel("Fitness (MSE)")
    ax.set_title(f"Pareto Front — Iteration {iter_id}")
    plt.colorbar(sc, ax=ax, label="Fitness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        _savefig(fig, tracker.results_dir / "figures", f"pareto_iter_{iter_id:02d}")
    return fig


# ---------------------------------------------------------------------------
# Figure 4: GP tree renders (top-3 Pareto candidates)
# ---------------------------------------------------------------------------


def plot_top_gp_trees(
    pareto_candidates,
    iter_id: int,
    tracker,
    top_n: int = 3,
    save: bool = True,
) -> Optional["plt.Figure"]:
    """Render the top-n GP trees using plagih's tree_renderer."""
    try:
        # tree_renderer was moved into the plagih package; keep import lazy
        # so the rest of the figure generation still works without matplotlib
        # backends configured for tree rendering.
        from plagih.visualization.tree_renderer import _render_tree_on_axes
    except ImportError:
        print("  [figures] tree_renderer not available  skipping tree renders.")
        return None

    import matplotlib.pyplot as plt

    candidates = sorted(pareto_candidates, key=lambda c: c.fitness)[:top_n]
    if not candidates:
        return None

    n = len(candidates)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, cand in zip(axes, candidates):
        try:
            _render_tree_on_axes(ax, cand.tree)
            ax.set_title(f"MSE={cand.fitness:.4f}  par={cand.parsimony}", fontsize=9)
        except Exception as exc:
            ax.text(0.5, 0.5, f"render error:\n{exc}", ha="center", va="center", fontsize=8)

    fig.suptitle(f"Top-{n} GP Candidates — Iteration {iter_id}", fontsize=11)
    fig.tight_layout()

    if save:
        _savefig(fig, tracker.results_dir / "figures", f"gp_trees_iter_{iter_id:02d}")
    return fig


# ---------------------------------------------------------------------------
# Figure 5: NN loss curves (last iteration)
# ---------------------------------------------------------------------------


def plot_nn_loss_curve(tracker, iter_id: Optional[int] = None, save: bool = True) -> "plt.Figure":
    """Plot the validation loss curve for the chosen NN in a given iteration."""
    import matplotlib.pyplot as plt

    if iter_id is None:
        iter_id = tracker.iterations[-1].iteration if tracker.iterations else 0

    matching = [it for it in tracker.iterations if it.iteration == iter_id]
    if not matching or not matching[0].nn_loss_curve:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center")
        return fig

    curve = matching[0].nn_loss_curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve, color="#2980b9", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"NN Training Curve — Iteration {iter_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        _savefig(fig, tracker.results_dir / "figures", f"nn_loss_curve_iter_{iter_id:02d}")
    return fig


# ---------------------------------------------------------------------------
# Generate all figures at once
# ---------------------------------------------------------------------------


def generate_all_figures(
    tracker,
    pareto_per_iter: Optional[List] = None,
    show: bool = False,
) -> None:
    """Generate and save all standard figures for an experiment.

    Args:
        tracker: Populated ExperimentTracker.
        pareto_per_iter: Optional list of Pareto candidate lists, one per iteration.
                         If provided, Pareto scatter and tree render figures are created.
        show: If True, call matplotlib.pyplot.show() at the end.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    print("\n[figures] Generating all figures ...")
    plot_em_progress(tracker)
    plot_nn_size_comparison(tracker)

    if tracker.iterations:
        last_iter = tracker.iterations[-1].iteration
        plot_nn_loss_curve(tracker, last_iter)

    if pareto_per_iter:
        for iter_id, pareto in enumerate(pareto_per_iter):
            plot_pareto_scatter(pareto, iter_id, tracker)
            plot_top_gp_trees(pareto, iter_id, tracker)

    print(f"[figures] All figures saved to {tracker.results_dir / 'figures'}")
    if show:
        plt.show()
