"""Configuration schema for a GP run that the GUI can edit.

The schema is split into:

- :class:`RunConfig` — all values that influence a run, including factory
  arguments for :func:`plagih.trees.ExplainableGP.create` and writable
  fields on :class:`plagih.config.PlagihConfig`.
- :class:`StrategySpec` — JSON-friendly mirror of
  :class:`plagih.parallel.Strategy`.

The split between *live-editable* and *reload-required* fields is captured
in :data:`LIVE_EDITABLE_FIELDS`.  The :class:`RunController` consults this
set when applying pending changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Operator catalogue  (string keys → used by OperatorPanel + ResolveFn)
# ---------------------------------------------------------------------------

#: All operators grouped by visual category.
OPERATOR_CATALOGUE: Dict[str, List[str]] = {
    "Math": [
        "Add",
        "Mul",
        "Sub",
        "Div",
        "Scale",
        "Square",
        "Sqrt",
        "Abs",
        "Sign",
        "Log",
        "Exp",
        "Exp2",
        "Pow",
        "NthRoot",
        "Usub",
        "Round",
        "PowRounded",
    ],
    "Trigonometry": ["Sin", "Cos", "Tan", "Tanh", "Sinh", "Cosh", "Asin", "Acos", "Atan"],
    "MinMax": ["Min", "Max", "Clip"],
    "Logic": ["Not", "And", "Or", "Xor"],
    "Relational": ["Lt", "Le", "Eq", "Ne"],  # Gt/Ge are PleaseUsePartnerOp - discouraged
    "Conditional": ["Ifte", "Piecewise"],
}

#: Named operator presets as ``{op_name: weight}`` dicts.
#: Mirrors the weights in ``Evolution.operator_presets`` but as plain strings.
OPERATOR_PRESET_WEIGHTS: Dict[str, Dict[str, float]] = {
    "math_simple": {
        "Add": 2.0,
        "Mul": 2.0,
        "Scale": 0.5,
        "Div": 1.0,
        "Square": 0.75,
        "Abs": 0.5,
        "Sign": 0.5,
        "Sqrt": 0.1,
        "Log": 0.1,
        "Sin": 0.5,
        "Not": 0.5,
        "Lt": 0.5,
        "Le": 0.5,
        "And": 1.0,
        "Or": 1.0,
        "Min": 1.0,
        "Max": 1.0,
    },
    "math_full": {
        "Add": 2.0,
        "Mul": 2.0,
        "Sub": 1.0,
        "Div": 1.0,
        "Scale": 0.5,
        "Square": 0.75,
        "Abs": 0.5,
        "Sign": 0.5,
        "Sqrt": 0.2,
        "Log": 0.2,
        "Exp": 0.1,
        "Pow": 0.1,
        "Sin": 0.5,
        "Cos": 0.5,
        "Tan": 0.1,
        "Tanh": 0.1,
        "Not": 0.5,
        "Lt": 0.5,
        "Le": 0.5,
        "And": 1.0,
        "Or": 1.0,
        "Min": 1.0,
        "Max": 1.0,
    },
    "with_logic": {
        "Add": 2.0,
        "Mul": 2.0,
        "Scale": 0.5,
        "Div": 1.0,
        "Square": 0.75,
        "Abs": 0.5,
        "Sign": 0.5,
        "Sqrt": 0.1,
        "Log": 0.1,
        "Sin": 0.5,
        "Not": 1.0,
        "And": 2.0,
        "Or": 2.0,
        "Xor": 0.5,
        "Lt": 1.0,
        "Le": 1.0,
        "Ifte": 1.0,
        "Piecewise": 0.5,
        "Min": 1.0,
        "Max": 1.0,
    },
}

# ---------------------------------------------------------------------------
# Strategy spec
# ---------------------------------------------------------------------------


@dataclass
class StrategySpec:
    """JSON-friendly mirror of :class:`plagih.parallel.Strategy`."""

    name: str
    rate: float = 0.0
    count: Optional[int] = None
    crossover: bool = False
    simplicate: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def to_strategy(self):
        """Materialise as the runtime ``plagih.parallel.Strategy`` object."""
        from plagih.parallel import Strategy

        return Strategy(
            name=self.name,
            rate=self.rate,
            count=self.count,
            crossover=self.crossover,
            simplicate=self.simplicate,
            **self.params,
        )


# Sensible default — mirrors ``demo_minimal`` in plagih_gp.py
DEFAULT_STRATEGIES: List[StrategySpec] = [
    StrategySpec("reproduction", rate=0.2, params={"tournament_n": 3}),
    StrategySpec("mutation", rate=0.4, params={"depth_goal": 3, "p_term": 0.3}),
    StrategySpec("random_new", rate=0.2, params={"depths": [2, 3, 4], "p_term": 0.1}),
    StrategySpec("crossover", rate=0.2, crossover=True, params={"tournament_n": 3}),
]


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """All settings influencing a GP run.

    The defaults match :func:`ExplainableGP.create` and :class:`PlagihConfig`.
    """

    # -- Data & output ---------------------------------------------------
    df_train_csv: Optional[str] = None  # CSV path; loaded via pandas.read_csv
    rootdir: str = "./.results/gui_run"
    target_column: str = "action"
    symbols: List[str] = field(default_factory=list)  # input column names

    # -- Engine factory args (mirrors ExplainableGP.create) -------------
    preset: str = "math_full"
    depth_max: int = 7
    nodes_max: int = 40
    pop_max_size: int = 100
    gen_end: int = 50
    clip_range: Optional[Tuple[float, float]] = None
    error_metric: str = "rmse"
    allow_chain: bool = False
    parallel: int = 0  # 0 = sequential
    enable_analysis: bool = False  # plots/backups during run

    # -- PlagihConfig overrides (cfg.*) ---------------------------------
    verbosity: str = "wwaaggiiffpp"
    simplification: bool = False
    visualization: bool = False
    merged_tree: bool = False
    origin_tree: bool = False
    lut_enabled: bool = True
    plots_interval: int = 1
    backup_interval: int = 10
    tree_min_parsimony: int = 3
    float_precision: int = 3

    # -- Operator configuration -----------------------------------------
    #: Custom operator weights as {op_name: weight}.  Empty dict → use ``preset``.
    operator_weights: Dict[str, float] = field(default_factory=dict)

    # -- Origin tree (optional seed) ------------------------------------
    #: Path to a pickled :class:`~plagih.trees._nodes.Node` used as origin seed.
    #: When set, the tree is added to the initial population and used for TED parsimony.
    origin_tree_path: Optional[str] = None

    # -- Strategies (per-generation evolution plan) ---------------------
    strategies: List[StrategySpec] = field(default_factory=lambda: list(DEFAULT_STRATEGIES))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # tuples → lists for JSON
        if d.get("clip_range") is not None:
            d["clip_range"] = list(d["clip_range"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RunConfig:
        d = dict(d)  # shallow copy; don't mutate caller
        clip = d.get("clip_range")
        if clip is not None:
            d["clip_range"] = tuple(clip)  # type: ignore[arg-type]
        strats = d.get("strategies") or []
        d["strategies"] = [StrategySpec(**s) for s in strats]
        return cls(**d)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Live-editable fields
# ---------------------------------------------------------------------------

# Fields that can be changed mid-run without rebuilding ExplainableGP.
# Everything else triggers a "reload required" warning in the GUI and, when
# Apply is pressed, a backup_save → recreate → backup_load cycle.
LIVE_EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        # GP-loop scalars
        "gen_end",
        "pop_max_size",
        "enable_analysis",
        # PlagihConfig fields (applied directly to plagih.config.cfg)
        "verbosity",
        "simplification",
        "merged_tree",
        "origin_tree",
        "plots_interval",
        "backup_interval",
        "tree_min_parsimony",
        "float_precision",
        # Per-generation evolution plan
        "strategies",
    }
)
