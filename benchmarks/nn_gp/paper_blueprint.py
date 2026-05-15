"""
Paper blueprint generator for the NN+GP pipeline.

Reads ``experiment.json`` produced by ExperimentTracker and fills in the
``docs/nn_gp_paper_template.md`` template with measured values.

Usage::

    from benchmarks.nn_gp.paper_blueprint import generate_blueprint

    generate_blueprint(tracker)

Or from the CLI::

    python -m benchmarks.nn_gp.paper_blueprint .results/nn_gp/20260515-120000/experiment.json
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

TEMPLATE_PATH = Path(__file__).parent.parent.parent / "docs" / "nn_gp_paper_template.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct_change(new: int, old: int) -> float:
    """Percentage change from old to new (positive = increase, negative = reduction)."""
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def _direction(pct: float) -> str:
    return "decreased" if pct < 0 else "increased"


def _build_iteration_table_rows(tracker) -> str:
    rows = []
    for it in tracker.iterations:
        rows.append(
            f"| {it.iteration} "
            f"| {it.gp_pareto_size} "
            f"| {it.gp_best_mse:.5f} "
            f"| `{it.nn_hidden_sizes}` "
            f"| {it.nn_param_count:,} "
            f"| {it.nn_mse:.5f} "
            f"| {it.residual_mse:.5f} "
            f"| {it.gp_runtime_s:.1f} "
            f"| {it.nn_runtime_s:.1f} |"
        )
    return "\n".join(rows) if rows else "| — | — | — | — | — | — | — | — | — |"


def _build_pareto_expressions_block(tracker) -> str:
    if not tracker.iterations:
        return "*No GP iterations yet.*"
    lines = []
    for it in tracker.iterations:
        if it.gp_expressions:
            lines.append(f"**Iteration {it.iteration}** ({len(it.gp_expressions)} candidates):")
            for expr in it.gp_expressions:
                lines.append(f"  - `{expr}`")
    return "\n".join(lines) if lines else "*No GP expressions recorded.*"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_blueprint(
    tracker,
    template_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Fill the paper template with values from the experiment tracker.

    Args:
        tracker: Populated ExperimentTracker instance.
        template_path: Path to the Markdown template. Defaults to
                       ``docs/nn_gp_paper_template.md``.
        output_path: Where to write the blueprint. Defaults to
                     ``tracker.results_dir / 'PAPER_BLUEPRINT.md'``.

    Returns:
        Path to the generated blueprint file.
    """
    tmpl = Path(template_path or TEMPLATE_PATH)
    with open(tmpl, encoding="utf-8") as fh:
        text = fh.read()

    meta = tracker.meta
    iters = tracker.iterations
    last = iters[-1] if iters else None

    # --- Derived values -------------------------
    baseline_params = meta.baseline_nn_param_count if meta else 0
    baseline_mse = meta.baseline_nn_mse if meta else float("inf")
    baseline_arch = str(meta.baseline_nn_hidden_sizes) if meta else "?"

    final_params = last.nn_param_count if last else 0
    final_mse = last.nn_mse if last else float("inf")
    final_arch = str(last.nn_hidden_sizes) if last else "?"
    final_residual = last.residual_mse if last else float("inf")
    last_iter_id = last.iteration if last else 0

    pct = _pct_change(final_params, baseline_params)
    pct_reduction = abs(pct)

    n_iterations = len(iters)
    benchmark = meta.benchmark if meta else "unknown"
    n_train_rows = meta.n_train_rows if meta else 0
    n_features = meta.n_features if meta else 0
    exp_id = tracker.results_dir.name

    # GP config fields (read from first iteration result notes or defaults)
    gp_pop_size = "?"
    gp_gen_end = "?"
    nn_epochs = "?"

    # --- Substitutions --------------------------
    replacements = {
        "{{experiment_id}}": exp_id,
        "{{benchmark}}": benchmark,
        "{{date}}": time.strftime("%Y-%m-%d %H:%M"),
        "{{results_dir}}": str(tracker.results_dir),
        "{{n_train_rows}}": str(n_train_rows),
        "{{n_features}}": str(n_features),
        "{{n_iterations}}": str(n_iterations),
        "{{baseline_nn_params}}": str(baseline_params),
        "{{baseline_nn_arch}}": baseline_arch,
        "{{baseline_nn_mse:.5f}}": f"{baseline_mse:.5f}",
        "{{final_nn_mse:.5f}}": f"{final_mse:.5f}",
        "{{final_nn_params}}": str(final_params),
        "{{final_nn_arch}}": final_arch,
        "{{param_reduction_pct:.1f}}": f"{pct_reduction:.1f}",
        "{{param_reduction_direction}}": _direction(pct),
        "{{iteration_table_rows}}": _build_iteration_table_rows(tracker),
        "{{pareto_expressions_block}}": _build_pareto_expressions_block(tracker),
        "{{last_iter_id}}": str(last_iter_id),
        "{{final_residual_mse:.5f}}": f"{final_residual:.5f}",
        "{{gp_pop_size}}": gp_pop_size,
        "{{gp_gen_end}}": gp_gen_end,
        "{{nn_epochs}}": nn_epochs,
    }

    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    # --- Write output ---------------------------
    out = Path(output_path or tracker.results_dir / "PAPER_BLUEPRINT.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)

    print(f"\n[blueprint] Paper blueprint written → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m benchmarks.nn_gp.paper_blueprint <results_dir_or_experiment.json>")
        sys.exit(1)

    p = Path(sys.argv[1])
    results_dir = p.parent if p.suffix == ".json" else p

    from benchmarks.nn_gp.experiment_tracker import ExperimentTracker

    t = ExperimentTracker.load(results_dir)
    out = generate_blueprint(t)
    print(f"Blueprint: {out}")
