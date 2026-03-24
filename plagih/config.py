"""
Centralized Configuration for the plagih GP Framework.

Loads settings from a `.env` file (project root) and environment variables.
Environment variables override `.env` values. Code-level overrides (e.g.
``ExplainableGP.create(parallel=4)``) override everything.

Usage::

    from plagih.config import cfg

    # Read a setting
    if cfg.debug:
        ...

    # Verbosity check (backwards-compatible with PRINT_DUMMY substring check)
    if "gg" in cfg.verbosity:
        ...

    # Runtime override (e.g. in benchmarks)
    cfg.verbosity = "ww"

See ``.env.example`` for all available keys with documentation.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path


def _find_dotenv() -> Path | None:
    """Walk up from this file to find the project-root ``.env``."""
    current = Path(__file__).resolve().parent  # plagih/
    for parent in [current, current.parent, *current.parent.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
        # Stop at the git root (if present)
        if (parent / ".git").exists():
            return candidate  # may not exist yet — that's fine
    return None


def _load_dotenv() -> None:
    """Load ``.env`` file if present. Environment variables take precedence."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        # python-dotenv not installed → settings come from env vars or defaults
        return
    path = _find_dotenv()
    if path is not None and path.is_file():
        load_dotenv(path, override=False)  # env vars win over .env


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value)


# ---------------------------------------------------------------------------
# Load .env on first import
# ---------------------------------------------------------------------------
_load_dotenv()


# ---------------------------------------------------------------------------
# PlagihConfig — singleton configuration object
# ---------------------------------------------------------------------------


class PlagihConfig:
    """Central configuration for the plagih GP framework.

    All defaults correspond to the **minimal profile**: no simplification,
    no visualisation, no parallelisation. LUT is enabled by default for
    performance. Enable other features explicitly via ``.env``, environment
    variables, or code-level overrides.

    Attributes (grouped):
        **Verbosity / Debug**
        verbosity:  Substring-membership string for ``printpl``/``printez``.
        debug:      Enable debug-level checks (e.g. sympy comparison).

        **GP Feature Flags**
        simplification:  Use SymPy simplification during evolution.
        visualization:   Generate plots/renderings during evolution.
        merged_tree:     Build & render merged population tree each generation.
        origin_tree:     Track origin-tree metadata on candidates.
        lut_enabled:     Enable expression look-up tables (LUT) for
                         duplicate-fitness avoidance.
        parallel:        Number of parallel workers (0 = sequential).

        **Numeric / IO**
        float_precision:      Decimal places for terminal formatting.
        plots_interval:       Generate monitoring plots every N generations.
        backup_interval:      Save backup every N generations.
        tree_min_parsimony:   Minimum complexity for a tree to be kept.
    """

    # -- Verbosity / Debug --------------------------------------------------
    verbosity: str
    debug: bool

    # -- GP Feature Flags ---------------------------------------------------
    simplification: bool
    visualization: bool
    merged_tree: bool
    origin_tree: bool
    lut_enabled: bool
    parallel: int

    # -- Numeric / IO -------------------------------------------------------
    float_precision: int
    plots_interval: int
    backup_interval: int
    tree_min_parsimony: int

    def __init__(self) -> None:
        env = os.environ.get

        # Verbosity / Debug
        self.verbosity = env("PLAGIH_VERBOSITY", "wwaaggiiffpp")
        self.debug = _bool(env("PLAGIH_DEBUG", "false"))

        # GP Feature Flags — defaults are MINIMAL (simplification, visualization, merged_tree, origin_tree, parallel off; lut on)
        self.simplification = _bool(env("PLAGIH_SIMPLIFICATION", "false"))
        self.visualization = _bool(env("PLAGIH_VISUALIZATION", "false"))
        self.merged_tree = _bool(env("PLAGIH_MERGED_TREE", "false"))
        self.origin_tree = _bool(env("PLAGIH_ORIGIN_TREE", "false"))
        self.lut_enabled = _bool(env("PLAGIH_LUT_ENABLED", "true"))
        self.parallel = _int(env("PLAGIH_PARALLEL", "0"))

        # Numeric / IO
        self.float_precision = _int(env("PLAGIH_FLOAT_PRECISION", "3"))
        self.plots_interval = _int(env("PLAGIH_PLOTS_INTERVAL", "1"))
        self.backup_interval = _int(env("PLAGIH_BACKUP_INTERVAL", "10"))
        self.tree_min_parsimony = _int(env("PLAGIH_TREE_MIN_PARSIMONY", "3"))

        # -- Emit warnings for performance-critical defaults ----------------
        if not self.lut_enabled:
            warnings.warn(
                "plagih: LUT (Look-Up Tables) is DISABLED (PLAGIH_LUT_ENABLED=false). "
                "This means every expression will be re-evaluated even if it was "
                "already seen.  For any non-trivial run this will be significantly "
                "slower.  Set PLAGIH_LUT_ENABLED=true in your .env or environment "
                "to enable caching.\n"
                "Discussion: LUT is now enabled by default because the performance "
                "benefit is large and the memory cost is usually small for typical "
                "population sizes (<10 000).  This warning therefore only appears "
                "when caching was explicitly disabled.",
                stacklevel=2,
            )

    # -- pretty repr --------------------------------------------------------

    def __repr__(self) -> str:
        flags = (
            f"verbosity={self.verbosity!r}, debug={self.debug}, "
            f"simplification={self.simplification}, visualization={self.visualization}, "
            f"merged_tree={self.merged_tree}, origin_tree={self.origin_tree}, "
            f"lut_enabled={self.lut_enabled}, parallel={self.parallel}, "
            f"float_precision={self.float_precision}, plots_interval={self.plots_interval}, "
            f"backup_interval={self.backup_interval}, tree_min_parsimony={self.tree_min_parsimony}"
        )
        return f"PlagihConfig({flags})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
cfg = PlagihConfig()
