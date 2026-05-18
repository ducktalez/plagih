"""Main window: dockable panels driven by a :class:`RunController`.

Layout
======
- Left dock:   :class:`ControlPanel` + :class:`ConfigPanel` (stacked).
- Centre:      tabbed monitoring view (charts, pareto, best candidate, log).
- Bottom dock: live log (re-uses the log tab inline as a separate dock).

Layout is persisted between sessions via ``QMainWindow.saveState``.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QByteArray, QSettings, Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from plagih.gui.core.events import EventType, RunEvent, RunState
from plagih.gui.core.run_controller import RunController
from plagih.gui.desktop.panels.best_candidate_panel import BestCandidatePanel
from plagih.gui.desktop.panels.charts_panel import ChartsPanel
from plagih.gui.desktop.panels.config_panel import ConfigPanel
from plagih.gui.desktop.panels.control_panel import ControlPanel
from plagih.gui.desktop.panels.log_panel import LogPanel
from plagih.gui.desktop.panels.pareto_panel import ParetoPanel
from plagih.gui.desktop.qt_event_relay import QtEventRelay

_ORG = "plagih"
_APP = "plagih-gui"


class MainWindow(QMainWindow):
    """Top-level window for the plagih GP monitoring GUI."""

    def __init__(self, controller: RunController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("plagih GP — Monitoring")
        self.resize(1280, 800)

        self._build_panels()
        self._build_layout()
        self._wire_events()
        self._restore_layout()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_panels(self) -> None:
        self._control_panel = ControlPanel(self._ctrl)
        self._config_panel = ConfigPanel(self._ctrl.config)
        self._charts_panel = ChartsPanel(self._ctrl)
        self._pareto_panel = ParetoPanel(self._ctrl)
        self._best_panel = BestCandidatePanel(self._ctrl)
        self._log_panel = LogPanel()

    def _build_layout(self) -> None:
        # Left dock: control + config in a vertical splitter
        left_container = QWidget()
        v = QVBoxLayout(left_container)
        v.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._control_panel)
        splitter.addWidget(self._config_panel)
        splitter.setSizes([180, 600])
        v.addWidget(splitter)

        dock_left = QDockWidget("Control & Configuration", self)
        dock_left.setObjectName("dock_left")
        dock_left.setWidget(left_container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_left)

        # Centre: tabbed monitoring view
        self._tabs = QTabWidget()
        self._tabs.addTab(self._charts_panel, "Charts")
        self._tabs.addTab(self._pareto_panel, "Pareto")
        self._tabs.addTab(self._best_panel, "Best candidate")
        self._tabs.addTab(self._log_panel, "Log & metrics")
        self.setCentralWidget(self._tabs)

    def _wire_events(self) -> None:
        # Forward bus → Qt thread via relay.
        self._relay = QtEventRelay(self._ctrl.bus, parent=self)
        self._relay.event.connect(self._on_event)

        # Config panel → controller
        self._config_panel.config_changed.connect(self._ctrl.apply_changes)

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def _on_event(self, ev: RunEvent) -> None:
        et = ev.type
        if et is EventType.STATE_CHANGED:
            state = ev.payload.get("state")
            if isinstance(state, RunState):
                self._control_panel.update_state(state)
            msg = ev.payload.get("message", "")
            if msg:
                self._log_panel.on_log({"level": "state", "message": f"{state}: {msg}"})
        elif et is EventType.GENERATION_DONE:
            self._charts_panel.on_generation_done(ev.payload)
            self._log_panel.on_generation_done(ev.payload)
        elif et is EventType.PARETO_CHANGED:
            self._pareto_panel.on_pareto_changed(ev.payload)
            self._best_panel.on_pareto_changed(ev.payload)
            added = ev.payload.get("added_count", 0)
            if added:
                self._log_panel.on_log({"level": "info", "message": f"Pareto front updated (+{added} candidates)"})
        elif et is EventType.IMPROVEMENT:
            fit = ev.payload.get("fitness")
            delta = ev.payload.get("delta")
            self._log_panel.on_log({"level": "info", "message": f"Improvement: fit={fit} (Δ={delta})"})
        elif et is EventType.CONFIG_APPLIED:
            ch = ev.payload.get("changes", {})
            reload_ = ev.payload.get("requires_reload")
            self._log_panel.on_log(
                {
                    "level": "info",
                    "message": f"Config applied ({'reload' if reload_ else 'live'}): {list(ch.keys())}",
                }
            )
        elif et is EventType.LOG:
            self._log_panel.on_log(ev.payload)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _restore_layout(self) -> None:
        s = QSettings(_ORG, _APP)
        geom = s.value("geometry")
        state = s.value("state")
        if isinstance(geom, QByteArray):
            self.restoreGeometry(geom)
        if isinstance(state, QByteArray):
            self.restoreState(state)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        s = QSettings(_ORG, _APP)
        s.setValue("geometry", self.saveGeometry())
        s.setValue("state", self.saveState())
        try:
            self._ctrl.stop(wait=True, timeout=2.0)
        except Exception:
            pass
        super().closeEvent(event)
