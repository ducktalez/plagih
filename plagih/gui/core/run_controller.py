"""Run controller — bridges :class:`ExplainableGP` and the GUI.

The controller owns the GP instance and a worker thread that drives the
generation loop.  It exposes thread-safe commands (``start``, ``pause``,
``resume``, ``stop``, ``apply_changes``, ``save_backup``, ``load_backup``)
and emits :class:`RunEvent` objects via an :class:`EventBus`.

Lifecycle
=========

1. Construct ``RunController(config)`` — no GP is created yet.
2. Call ``start()`` — spawns worker thread, builds the GP, runs the
   initial population, then loops generations.
3. Between generations the worker waits on a pause :class:`Event` and
   drains a pending-changes queue.
4. ``stop()`` requests cooperative termination after the current generation.

Pause semantics are *between generations only* — see ``docs/IMPLEMENTATION_PLAN.md``
"GUI-Ideen" for finer granularity ideas.
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from plagih.gui.core.config_schema import LIVE_EDITABLE_FIELDS, RunConfig
from plagih.gui.core.events import EventBus, EventType, RunEvent, RunState


class RunController:
    """Drives a single :class:`ExplainableGP` instance from the GUI."""

    def __init__(self, config: RunConfig, bus: Optional[EventBus] = None) -> None:
        self.config = config
        self.bus = bus or EventBus()

        # Worker thread + lifecycle primitives
        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()  # "set" == running; cleared == paused
        self._stop_event = threading.Event()
        self._state = RunState.IDLE

        # Pending-changes queue: each item is a (partial-config-dict, force_reload)
        self._pending: queue.Queue[tuple[Dict[str, Any], bool]] = queue.Queue()

        # Owned GP instance (created inside the worker thread)
        self.gp = None  # type: ignore[assignment]

        # Pareto-tracking — used to derive "latest_added" candidate
        self._known_pareto_keys: set[tuple] = set()
        self._latest_pareto_candidate = None

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Commands (callable from any thread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread.  No-op if already running."""
        if self.is_alive:
            return
        self._stop_event.clear()
        self._pause_event.set()
        self._set_state(RunState.STARTING)
        self._thread = threading.Thread(target=self._run_loop, name="plagih-gp-worker", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Pause after the current generation completes."""
        if self._state == RunState.RUNNING:
            self._pause_event.clear()
            self._set_state(RunState.PAUSED)

    def resume(self) -> None:
        """Resume from a paused state."""
        if self._state == RunState.PAUSED:
            self._pause_event.set()
            self._set_state(RunState.RUNNING)

    def stop(self, wait: bool = False, timeout: Optional[float] = None) -> None:
        """Request cooperative termination after the current generation."""
        self._stop_event.set()
        self._pause_event.set()  # release the pause wait so we exit
        self._set_state(RunState.STOPPING)
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)

    def apply_changes(self, changes: Dict[str, Any], *, force_reload: bool = False) -> None:
        """Queue partial config changes; applied at the next generation boundary.

        Args:
            changes: Partial dict of ``RunConfig`` field overrides.
            force_reload: If True, always rebuild the GP, even for
                live-editable-only fields.
        """
        self._pending.put((dict(changes), force_reload))

    def save_backup(self, path: Optional[Path] = None) -> None:
        """Persist a backup of the current run state."""
        if self.gp is None:
            return
        self.gp.backup_save(opt_path_backup=path)
        self._emit_log("info", f"Backup saved ({path or 'default path'}).")

    def load_backup(self, path: Optional[Path] = None) -> None:
        """Restore a previous run state into the live GP instance."""
        if self.gp is None:
            self._emit_log("warning", "load_backup: no GP instance yet; start a run first.")
            return
        self.gp.backup_load(opt_path_backup=path)
        # Refresh the pareto-tracking after restore.
        self._known_pareto_keys = self._pareto_key_set(self.gp.paretofront)
        self._emit_log("info", f"Backup loaded ({path or 'default path'}).")

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        try:
            self._build_gp_from_config()
            self._wire_monitor_callbacks()
            self._set_state(RunState.RUNNING)

            # Initial population (generation 0)
            if self.gp.gen_id == 0:
                self.gp.gen_create_initial()

            # Main generation loop
            while not self._stop_event.is_set() and self.gp.gen_id < self.config.gen_end:
                # Honour pause request
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                # Apply any pending config changes between generations
                self._drain_pending_changes()
                if self._stop_event.is_set():
                    break

                strategies = [s.to_strategy() for s in self.config.strategies]
                self.gp.run_generation(strategies)

            self._set_state(RunState.FINISHED, message=f"Reached gen {self.gp.gen_id}/{self.config.gen_end}")
        except Exception as exc:
            tb = traceback.format_exc()
            self._emit_log("error", f"Worker crashed: {exc}\n{tb}")
            self._set_state(RunState.ERROR, message=str(exc))
        finally:
            # Best-effort pool cleanup
            try:
                if self.gp is not None:
                    self.gp.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # GP construction & monitor wiring
    # ------------------------------------------------------------------

    def _build_gp_from_config(self) -> None:
        """Instantiate :class:`ExplainableGP` from :attr:`config`."""
        import pandas as pd

        from plagih.config import cfg as _cfg
        from plagih.trees import ExplainableGP

        cfg = self.config

        # Apply PlagihConfig-side settings first (some are read during init).
        self._push_cfg_to_singleton(_cfg)

        if not cfg.df_train_csv:
            raise ValueError("RunConfig.df_train_csv must point to a CSV file.")
        df_train = pd.read_csv(cfg.df_train_csv)
        if cfg.target_column not in df_train.columns:
            raise ValueError(f"Target column {cfg.target_column!r} not in CSV columns {list(df_train.columns)}.")

        # If symbols are unset, default to all non-target columns.
        symbols = cfg.symbols or [c for c in df_train.columns if c != cfg.target_column]

        self.gp = ExplainableGP.create(
            symbols=symbols,
            df_train=df_train,
            rootdir=cfg.rootdir,
            preset=cfg.preset,
            depth_max=cfg.depth_max,
            nodes_max=cfg.nodes_max,
            pop_max_size=cfg.pop_max_size,
            gen_end=cfg.gen_end,
            clip_range=cfg.clip_range,
            error_metric=cfg.error_metric,
            allow_chain=cfg.allow_chain,
            target_column=cfg.target_column,
            verbose=False,
            parallel=cfg.parallel,
            enable_analysis=cfg.enable_analysis,
        )

    def _push_cfg_to_singleton(self, _cfg) -> None:
        """Apply :class:`PlagihConfig` overrides from :attr:`config`."""
        cfg = self.config
        _cfg.verbosity = cfg.verbosity
        _cfg.simplification = cfg.simplification
        _cfg.visualization = cfg.visualization
        _cfg.merged_tree = cfg.merged_tree
        _cfg.origin_tree = cfg.origin_tree
        _cfg.lut_enabled = cfg.lut_enabled
        _cfg.plots_interval = cfg.plots_interval
        _cfg.backup_interval = cfg.backup_interval
        _cfg.tree_min_parsimony = cfg.tree_min_parsimony
        _cfg.float_precision = cfg.float_precision

    def _wire_monitor_callbacks(self) -> None:
        """Bridge :class:`GPMonitor` callbacks to the :class:`EventBus`."""
        monitor = self.gp.monitor

        @monitor.on_generation
        def _on_gen(metrics):
            self.bus.emit(
                RunEvent(
                    EventType.GENERATION_DONE,
                    {
                        "gen_id": metrics.gen_id,
                        "metrics": dict(metrics.metrics),
                        "timestamp": metrics.timestamp,
                    },
                )
            )

        @monitor.on_improvement
        def _on_imp(metrics, delta):
            self.bus.emit(
                RunEvent(
                    EventType.IMPROVEMENT,
                    {"fitness": metrics.get("fit_best"), "delta": delta, "gen_id": metrics.gen_id},
                )
            )

        @monitor.on_pareto_update
        def _on_pareto(metrics):
            front = self.gp.paretofront or []
            new_keys = self._pareto_key_set(front)
            added_keys = new_keys - self._known_pareto_keys
            self._known_pareto_keys = new_keys

            # Pick the "latest added" candidate: prefer the best-fitness new one.
            latest = None
            if added_keys:
                added = [c for c in front if self._pareto_key(c) in added_keys]
                latest = min(added, key=lambda c: (c.get_fitness(), c.get_parsim()))
            self._latest_pareto_candidate = latest or self._latest_pareto_candidate

            self.bus.emit(
                RunEvent(
                    EventType.PARETO_CHANGED,
                    {
                        "front_size": len(front),
                        "added_count": len(added_keys),
                        "latest_added": self._latest_pareto_candidate,
                        "gen_id": metrics.gen_id,
                    },
                )
            )

    @staticmethod
    def _pareto_key(c) -> tuple:
        return (c.get_parsim(), c.get_fitness(), c.full_string())

    @classmethod
    def _pareto_key_set(cls, front) -> set[tuple]:
        return {cls._pareto_key(c) for c in (front or [])}

    # ------------------------------------------------------------------
    # Pending changes
    # ------------------------------------------------------------------

    def _drain_pending_changes(self) -> None:
        """Consume queued config changes; reload the GP if necessary."""
        merged: Dict[str, Any] = {}
        force_reload = False
        while True:
            try:
                changes, force = self._pending.get_nowait()
            except queue.Empty:
                break
            merged.update(changes)
            force_reload = force_reload or force

        if not merged:
            return

        requires_reload = force_reload or any(k not in LIVE_EDITABLE_FIELDS for k in merged)

        # Always update our stored config object first.
        for k, v in merged.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        if requires_reload:
            self._reload_gp_preserving_state()
        else:
            self._apply_live_changes(merged)

        self.bus.emit(
            RunEvent(
                EventType.CONFIG_APPLIED,
                {"changes": dict(merged), "requires_reload": requires_reload},
            )
        )

    def _apply_live_changes(self, changes: Dict[str, Any]) -> None:
        """Apply changes that don't require rebuilding the GP."""
        from plagih.config import cfg as _cfg

        # Engine-side scalars
        if "gen_end" in changes:
            self.gp.gen_end = int(changes["gen_end"])
        if "pop_max_size" in changes:
            self.gp.pop_max_size = int(changes["pop_max_size"])
        if "enable_analysis" in changes:
            self.gp.enable_analysis = bool(changes["enable_analysis"])

        # PlagihConfig-side scalars
        for key in (
            "verbosity",
            "simplification",
            "merged_tree",
            "origin_tree",
            "plots_interval",
            "backup_interval",
            "tree_min_parsimony",
            "float_precision",
        ):
            if key in changes:
                setattr(_cfg, key, changes[key])

        # strategies handled implicitly — the loop re-reads self.config.strategies

    def _reload_gp_preserving_state(self) -> None:
        """Rebuild the GP from scratch, preserving population + pareto via backup."""
        self._emit_log("info", "Reloading GP with new settings (state preserved via backup).")
        backup_path = Path(self.gp.rootdir) / "backup" / "gui_reload_snapshot.pkl"
        try:
            self.gp.backup_save(opt_path_backup=backup_path)
            try:
                self.gp.close()
            except Exception:
                pass
            self._build_gp_from_config()
            self._wire_monitor_callbacks()
            self.gp.backup_load(opt_path_backup=backup_path)
            self._known_pareto_keys = self._pareto_key_set(self.gp.paretofront)
        except Exception as exc:
            self._emit_log("error", f"Reload failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: RunState, message: str = "") -> None:
        self._state = state
        self.bus.emit(RunEvent(EventType.STATE_CHANGED, {"state": state, "message": message}))

    def _emit_log(self, level: str, message: str) -> None:
        self.bus.emit(RunEvent(EventType.LOG, {"level": level, "message": message}))
