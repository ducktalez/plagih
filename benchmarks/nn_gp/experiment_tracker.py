"""
Experiment tracker for the NN+GP pipeline.

Persists all per-iteration results incrementally to a JSON file so that a
crash mid-run does not lose earlier iterations.  Can also reconstruct a
summary DataFrame from the JSON for downstream analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Per-iteration result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    """All metrics collected during one EM-loop iteration.

    Attributes:
        iteration: 0-based iteration index.
        gp_pareto_size: Number of Pareto-optimal GP candidates found.
        gp_best_mse: MSE of the single best GP candidate on the training set.
        gp_expressions: String representations of Pareto-front candidates.
        nn_hidden_sizes: Hidden layer widths of the chosen minimal NN.
        nn_param_count: Total trainable parameters of the chosen NN.
        nn_mse: MSE of the chosen NN on the full training set.
        residual_mse: MSE of the per-row best-GP-prediction vs. target
                      (= lower bound the next GP iteration must beat).
        gp_runtime_s: Wall-clock seconds for the GP phase.
        nn_runtime_s: Wall-clock seconds for the NN search phase.
        nn_loss_curve: Validation loss per epoch of the final chosen architecture.
        notes: Free-form extra notes.
    """

    iteration: int
    gp_pareto_size: int = 0
    gp_best_mse: float = float("inf")
    gp_expressions: List[str] = field(default_factory=list)
    nn_hidden_sizes: List[int] = field(default_factory=list)
    nn_param_count: int = 0
    nn_mse: float = float("inf")
    residual_mse: float = float("inf")
    gp_runtime_s: float = 0.0
    nn_runtime_s: float = 0.0
    nn_loss_curve: List[float] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert any remaining numpy scalars for JSON serialisation
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = v.item()
        return d


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------


@dataclass
class ExperimentMeta:
    """High-level metadata about the whole experiment run."""

    benchmark: str
    target_col: str
    n_train_rows: int
    n_features: int
    baseline_nn_mse: float = float("inf")
    baseline_nn_param_count: int = 0
    baseline_nn_hidden_sizes: List[int] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    notes: str = ""
    # Frozen hyperparameter snapshots (free-form dicts to avoid coupling
    # the tracker to concrete config classes). Resolved by the blueprint
    # generator to fill {{gp_pop_size}}, {{gp_gen_end}}, {{nn_epochs}}, ...
    gp_config: Dict[str, Any] = field(default_factory=dict)
    nn_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """Incrementally persists experiment results to a JSON file.

    Usage::

        tracker = ExperimentTracker(results_dir, benchmark="mc")
        tracker.set_meta(meta)
        tracker.add_iteration(iter_result)
        tracker.finalize()

    The JSON is written after every ``add_iteration`` call, so a crash
    mid-experiment does not lose data.

    Args:
        results_dir: Directory where ``experiment.json`` (and figures) are stored.
        benchmark: Short benchmark name, e.g. ``"mc"``.
    """

    def __init__(self, results_dir: Path, benchmark: str = "unknown"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.results_dir / "experiment.json"
        self.benchmark = benchmark

        self.meta: Optional[ExperimentMeta] = None
        self.iterations: List[IterationResult] = []

    # ------------------------------------------------------------------

    def set_meta(self, meta: ExperimentMeta) -> None:
        """Store experiment metadata and immediately write to disk."""
        self.meta = meta
        self._save()

    def add_iteration(self, result: IterationResult) -> None:
        """Append an iteration result and persist to disk."""
        self.iterations.append(result)
        self._save()
        print(
            f"  [tracker] iter={result.iteration}  "
            f"gp_pareto={result.gp_pareto_size}  "
            f"gp_best_mse={result.gp_best_mse:.5f}  "
            f"nn_params={result.nn_param_count:,}  "
            f"nn_mse={result.nn_mse:.5f}  "
            f"residual_mse={result.residual_mse:.5f}"
        )

    def finalize(self, notes: str = "") -> None:
        """Mark the experiment as complete and write final JSON."""
        if self.meta:
            self.meta.end_time = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.meta.notes = notes
        self._save()
        print(f"  [tracker] experiment saved → {self.json_path}")

    # ------------------------------------------------------------------

    def _save(self) -> None:
        data: Dict[str, Any] = {
            "meta": self.meta.to_dict() if self.meta else {},
            "iterations": [it.to_dict() for it in self.iterations],
        }
        with open(self.json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, results_dir: Path) -> "ExperimentTracker":
        """Reconstruct a tracker instance from an existing ``experiment.json``."""
        path = Path(results_dir) / "experiment.json"
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        tracker = cls(results_dir)
        if data.get("meta"):
            m = data["meta"]
            # Tolerate older JSONs that lack the gp_config / nn_config fields.
            known = {f.name for f in ExperimentMeta.__dataclass_fields__.values()}
            m = {k: v for k, v in m.items() if k in known}
            tracker.meta = ExperimentMeta(**m)
        for it in data.get("iterations", []):
            tracker.iterations.append(IterationResult(**it))
        return tracker

    # ------------------------------------------------------------------

    @property
    def baseline_param_count(self) -> int:
        return self.meta.baseline_nn_param_count if self.meta else 0

    @property
    def baseline_mse(self) -> float:
        return self.meta.baseline_nn_mse if self.meta else float("inf")
