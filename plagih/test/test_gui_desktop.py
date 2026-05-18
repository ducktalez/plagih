"""Smoke tests for the PySide6 desktop GUI.

These tests are skipped if PySide6 is not installed.  When available they
run in **offscreen** Qt mode (``QT_QPA_PLATFORM=offscreen``) so they work
on headless CI.

Scope: construction of all widgets, event routing through
:class:`QtEventRelay`, and basic interactions that don't require a real
GP run.  An end-to-end GP-loop test would belong under
``@pytest.mark.performance``.
"""

from __future__ import annotations

import os

import pytest

# Force offscreen Qt platform BEFORE any Qt module is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed - GUI tests skipped")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from plagih.gui.core import (
    EventBus,
    EventType,
    RunConfig,
    RunController,
    RunEvent,
    RunState,
)

# ---------------------------------------------------------------------------
# QApplication fixture — one shared instance for the whole session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # Don't quit — fixture is session-scoped and Qt dislikes repeated init.


@pytest.fixture
def keep_alive(qapp):
    """Hold strong references to widgets so their C++ peers outlive the test.

    Matplotlib's :class:`FigureCanvasQTAgg` schedules deferred ``draw_idle``
    calls; if the canvas is garbage-collected before those drains run, Qt
    raises ``RuntimeError: Internal C++ object already deleted``.  Keep
    everything alive until teardown, then close cleanly.
    """
    bag: list = []
    yield bag.append
    qapp.processEvents()
    for w in bag:
        try:
            w.close()
        except Exception:
            pass
    qapp.processEvents()


# ---------------------------------------------------------------------------
# QtEventRelay
# ---------------------------------------------------------------------------


class TestQtEventRelay:
    def test_relays_bus_events_via_signal(self, qapp):
        from plagih.gui.desktop.qt_event_relay import QtEventRelay

        bus = EventBus()
        relay = QtEventRelay(bus)
        received: list[RunEvent] = []
        relay.event.connect(received.append)

        bus.emit(RunEvent(EventType.LOG, {"level": "info", "message": "hello"}))
        # Process pending Qt events to let the signal deliver.
        qapp.processEvents()
        assert len(received) == 1
        assert received[0].type is EventType.LOG

    def test_disconnect_bus_stops_delivery(self, qapp):
        from plagih.gui.desktop.qt_event_relay import QtEventRelay

        bus = EventBus()
        relay = QtEventRelay(bus)
        received: list[RunEvent] = []
        relay.event.connect(received.append)

        relay.disconnect_bus()
        bus.emit(RunEvent(EventType.LOG, {}))
        qapp.processEvents()
        assert received == []


# ---------------------------------------------------------------------------
# Panel construction smoke tests
# ---------------------------------------------------------------------------


class TestPanelsConstruct:
    """Each panel must instantiate without error using a fresh controller."""

    def test_control_panel(self, qapp, keep_alive):
        from plagih.gui.desktop.panels.control_panel import ControlPanel

        ctrl = RunController(RunConfig())
        panel = ControlPanel(ctrl)
        keep_alive(panel)
        panel.update_state(RunState.RUNNING)
        assert panel._btn_pause.isEnabled()
        assert panel._btn_start.isEnabled() is not False  # idle alive=False → start enabled
        panel.update_state(RunState.PAUSED)
        assert panel._btn_resume.isEnabled()

    def test_config_panel_populates_from_config(self, qapp, keep_alive):
        from plagih.gui.desktop.panels.config_panel import ConfigPanel

        cfg = RunConfig(gen_end=42, pop_max_size=33, verbosity="ww")
        panel = ConfigPanel(cfg)
        keep_alive(panel)
        values, errors = panel._collect_from_form()
        assert not errors
        assert values["gen_end"] == 42
        assert values["pop_max_size"] == 33
        assert values["verbosity"] == "ww"

    def test_config_panel_apply_emits_change_signal(self, qapp, keep_alive):
        from plagih.gui.desktop.panels.config_panel import ConfigPanel

        cfg = RunConfig(gen_end=10)
        panel = ConfigPanel(cfg)
        keep_alive(panel)
        emitted: list[tuple] = []
        panel.config_changed.connect(lambda ch, reload_: emitted.append((dict(ch), reload_)))

        # Programmatically change a live field, then apply
        panel._widgets["gen_end"].setValue(99)
        panel._on_apply()
        qapp.processEvents()

        assert emitted, "Apply must emit config_changed"
        changes, requires_reload = emitted[-1]
        assert changes == {"gen_end": 99}
        assert requires_reload is False  # gen_end is live-editable

    def test_charts_panel(self, qapp, keep_alive):
        from plagih.gui.desktop.panels.charts_panel import ChartsPanel

        ctrl = RunController(RunConfig())
        panel = ChartsPanel(ctrl)
        keep_alive(panel)
        # No GP yet → should render the "no data yet" placeholder without error
        panel._force_redraw()

    def test_pareto_panel(self, qapp, keep_alive):
        from plagih.gui.desktop.panels.pareto_panel import ParetoPanel

        ctrl = RunController(RunConfig())
        panel = ParetoPanel(ctrl)
        keep_alive(panel)
        # Empty Pareto front (no GP) → should not crash
        panel.on_pareto_changed({"latest_added": None})

    def test_best_candidate_panel_handles_no_candidate(self, qapp, keep_alive):
        from plagih.gui.desktop.panels.best_candidate_panel import BestCandidatePanel

        ctrl = RunController(RunConfig())
        panel = BestCandidatePanel(ctrl)
        keep_alive(panel)
        panel.on_pareto_changed({"latest_added": None})  # no-op

    def test_log_panel_grows_on_events(self, qapp, keep_alive):
        from plagih.gui.desktop.panels.log_panel import LogPanel

        panel = LogPanel()
        keep_alive(panel)
        panel.on_generation_done({"gen_id": 0, "metrics": {"fit_best": 1.23}})
        panel.on_generation_done({"gen_id": 1, "metrics": {"fit_best": 0.98}})
        assert panel._table.rowCount() == 2
        panel.on_log({"level": "info", "message": "hello"})
        assert "hello" in panel._log_view.toPlainText()


# ---------------------------------------------------------------------------
# MainWindow smoke test
# ---------------------------------------------------------------------------


class TestMainWindow:
    def test_constructs_and_routes_events(self, qapp, keep_alive):
        from plagih.gui.desktop.main_window import MainWindow

        ctrl = RunController(RunConfig())
        window = MainWindow(ctrl)
        keep_alive(window)
        # State-changed event should reach the control panel
        ctrl.bus.emit(RunEvent(EventType.STATE_CHANGED, {"state": RunState.RUNNING, "message": ""}))
        qapp.processEvents()
        assert "Running" in window._control_panel._status.text()

        # Log event should land in the log panel
        ctrl.bus.emit(RunEvent(EventType.LOG, {"level": "info", "message": "abc"}))
        qapp.processEvents()
        assert "abc" in window._log_panel._log_view.toPlainText()
