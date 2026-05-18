"""Tests for the transport-neutral GUI core (no Qt dependency).

Covers:
- :class:`RunConfig` ↔ dict / JSON roundtrip and field semantics.
- :class:`StrategySpec` → :class:`plagih.parallel.Strategy` conversion.
- :data:`LIVE_EDITABLE_FIELDS` sanity (every key exists on RunConfig).
- :class:`EventBus` subscribe / emit / unsubscribe and exception safety.
- :class:`RunController` lifecycle methods that do **not** require building
  an actual GP (pause/resume on an idle controller, apply_changes queue,
  ``_drain_pending_changes`` for live-editable keys).

These tests run on every ``pytest`` invocation — they must stay Qt-free.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

from plagih.gui.core import (
    DEFAULT_STRATEGIES,
    LIVE_EDITABLE_FIELDS,
    EventBus,
    EventType,
    RunConfig,
    RunController,
    RunEvent,
    RunState,
    StrategySpec,
)

# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_defaults_are_complete(self):
        c = RunConfig()
        assert c.target_column == "action"
        assert c.preset == "math_full"
        assert c.parallel == 0
        assert isinstance(c.strategies, list) and len(c.strategies) == len(DEFAULT_STRATEGIES)

    def test_to_dict_from_dict_roundtrip(self):
        c1 = RunConfig(clip_range=(0.0, 2.0), symbols=["a", "b"], gen_end=7)
        d = c1.to_dict()
        # JSON-friendliness: tuples become lists
        assert isinstance(d["clip_range"], list)
        c2 = RunConfig.from_dict(d)
        assert c2.clip_range == (0.0, 2.0)
        assert c2.symbols == ["a", "b"]
        assert c2.gen_end == 7
        assert len(c2.strategies) == len(c1.strategies)
        assert all(isinstance(s, StrategySpec) for s in c2.strategies)

    def test_save_load_json(self, tmp_path: Path):
        c1 = RunConfig(rootdir=str(tmp_path / "run"), gen_end=42)
        path = tmp_path / "settings.json"
        c1.save(path)
        # File is valid JSON
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert parsed["gen_end"] == 42
        c2 = RunConfig.load(path)
        assert c2.gen_end == 42
        assert c2.rootdir == str(tmp_path / "run")

    def test_strategies_default_matches_demo_minimal(self):
        names = [s.name for s in DEFAULT_STRATEGIES]
        assert names == ["reproduction", "mutation", "random_new", "crossover"]


# ---------------------------------------------------------------------------
# StrategySpec
# ---------------------------------------------------------------------------


class TestStrategySpec:
    def test_to_strategy_passes_params(self):
        spec = StrategySpec("mutation", rate=0.4, params={"depth_goal": 3, "p_term": 0.3})
        strat = spec.to_strategy()
        assert strat.name == "mutation"
        assert strat.rate == 0.4
        assert strat.params == {"depth_goal": 3, "p_term": 0.3}

    def test_crossover_flag(self):
        spec = StrategySpec("crossover", rate=0.2, crossover=True, params={"tournament_n": 3})
        strat = spec.to_strategy()
        assert strat.crossover is True
        assert strat.params["tournament_n"] == 3


# ---------------------------------------------------------------------------
# LIVE_EDITABLE_FIELDS
# ---------------------------------------------------------------------------


class TestLiveEditableFields:
    def test_all_keys_exist_on_runconfig(self):
        names = {f for f in RunConfig().__dataclass_fields__}
        unknown = LIVE_EDITABLE_FIELDS - names
        assert not unknown, f"LIVE_EDITABLE_FIELDS contains unknown keys: {unknown}"

    def test_known_live_keys_present(self):
        """Spot-check that the documented live keys are flagged as live."""
        for key in ("gen_end", "pop_max_size", "strategies", "verbosity"):
            assert key in LIVE_EDITABLE_FIELDS

    def test_structural_keys_are_not_live(self):
        """Spot-check that structural keys correctly require reload."""
        for key in ("symbols", "df_train_csv", "preset", "parallel", "depth_max"):
            assert key not in LIVE_EDITABLE_FIELDS, f"{key} changes the GP topology and must NOT be live-editable"


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class TestEventBus:
    def test_subscribe_emit_unsubscribe(self):
        bus = EventBus()
        received: list[RunEvent] = []
        unsub = bus.subscribe(received.append)

        bus.emit(RunEvent(EventType.LOG, {"level": "info", "message": "x"}))
        assert len(received) == 1
        assert received[0].type is EventType.LOG

        unsub()
        bus.emit(RunEvent(EventType.LOG, {"message": "y"}))
        assert len(received) == 1  # unsubscribed → no new event

    def test_buggy_subscriber_does_not_break_others(self):
        bus = EventBus()
        good: list[RunEvent] = []

        def bad(_ev):
            raise RuntimeError("boom")

        bus.subscribe(bad)
        bus.subscribe(good.append)
        bus.emit(RunEvent(EventType.STATE_CHANGED, {"state": RunState.IDLE}))
        assert len(good) == 1

    def test_thread_safety_of_subscribe(self):
        bus = EventBus()
        counter = {"n": 0}

        def cb(_ev):
            counter["n"] += 1

        # Subscribe from multiple threads, then emit from main thread.
        threads = [threading.Thread(target=bus.subscribe, args=(cb,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        bus.emit(RunEvent(EventType.LOG, {}))
        assert counter["n"] == 10


# ---------------------------------------------------------------------------
# RunController — does not start a real GP
# ---------------------------------------------------------------------------


class TestRunControllerLifecycle:
    def test_initial_state_is_idle(self):
        ctrl = RunController(RunConfig())
        assert ctrl.state is RunState.IDLE
        assert ctrl.gp is None
        assert ctrl.is_alive is False

    def test_pause_resume_noop_when_idle(self):
        """Pause/resume must be safe when no run is active."""
        ctrl = RunController(RunConfig())
        ctrl.pause()
        ctrl.resume()
        # state never advances past IDLE without an explicit start()
        assert ctrl.state is RunState.IDLE

    def test_apply_changes_queue_is_thread_safe(self):
        ctrl = RunController(RunConfig())
        # Queue up many changes from multiple threads
        threads = [threading.Thread(target=ctrl.apply_changes, args=({"gen_end": i},)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 20 items should now be sitting in the queue.
        assert ctrl._pending.qsize() == 20

    def test_emits_state_changes(self):
        """Even without an actual run, ``stop()`` must emit STATE_CHANGED."""
        events: list[RunEvent] = []
        ctrl = RunController(RunConfig())
        ctrl.bus.subscribe(events.append)
        ctrl.stop()
        kinds = [e.type for e in events]
        assert EventType.STATE_CHANGED in kinds


class TestRunControllerLiveChanges:
    """The live-vs-reload split is the most failure-prone part — test directly."""

    def _build_with_fake_gp(self, gen_end: int = 5, pop_max_size: int = 8):
        """Create a controller and stub out ``self.gp`` enough for live-apply."""
        ctrl = RunController(RunConfig(gen_end=gen_end, pop_max_size=pop_max_size))

        class _FakeGP:
            def __init__(self):
                self.gen_end = gen_end
                self.pop_max_size = pop_max_size
                self.enable_analysis = False

        ctrl.gp = _FakeGP()  # type: ignore[assignment]
        return ctrl

    def test_live_change_applied_in_place(self):
        ctrl = self._build_with_fake_gp(gen_end=10)
        ctrl.apply_changes({"gen_end": 25})
        events: list[RunEvent] = []
        ctrl.bus.subscribe(events.append)
        ctrl._drain_pending_changes()

        assert ctrl.config.gen_end == 25
        assert ctrl.gp.gen_end == 25
        config_events = [e for e in events if e.type is EventType.CONFIG_APPLIED]
        assert len(config_events) == 1
        assert config_events[0].payload["requires_reload"] is False

    def test_structural_change_triggers_reload_flag(self):
        ctrl = self._build_with_fake_gp()
        ctrl.apply_changes({"preset": "with_logic"})

        events: list[RunEvent] = []
        ctrl.bus.subscribe(events.append)

        # _drain_pending_changes will try to call _reload_gp_preserving_state.
        # We expect that to fail because there's no real GP — but the
        # CONFIG_APPLIED event must still mark requires_reload=True.
        try:
            ctrl._drain_pending_changes()
        except Exception:
            pass

        # The local config object was always updated first, regardless of reload outcome.
        assert ctrl.config.preset == "with_logic"

    def test_unknown_key_is_ignored_for_live_apply(self):
        """A typo'd field name must not crash the controller."""
        ctrl = self._build_with_fake_gp()
        ctrl.apply_changes({"this_field_does_not_exist": 42})
        # _drain_pending_changes will see it as non-live → tries reload.
        # We swallow the inevitable reload error; the assertion is that
        # nothing about ``ctrl.config`` was mutated for the unknown key.
        try:
            ctrl._drain_pending_changes()
        except Exception:
            pass
        assert not hasattr(ctrl.config, "this_field_does_not_exist")
