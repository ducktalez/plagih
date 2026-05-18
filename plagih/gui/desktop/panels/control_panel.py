"""Start / Pause / Resume / Stop / Backup controls.

Reflects current :class:`RunState` via a coloured status label and toggles
button enabled-state accordingly.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from plagih.gui.core.events import RunState
from plagih.gui.core.run_controller import RunController

_STATE_STYLES = {
    RunState.IDLE: ("#888888", "Idle"),
    RunState.STARTING: ("#1e88e5", "Starting…"),
    RunState.RUNNING: ("#43a047", "Running"),
    RunState.PAUSED: ("#fb8c00", "Paused"),
    RunState.STOPPING: ("#e53935", "Stopping…"),
    RunState.FINISHED: ("#5e35b1", "Finished"),
    RunState.ERROR: ("#c62828", "Error"),
}


class ControlPanel(QWidget):
    """Run lifecycle control widget."""

    def __init__(self, controller: RunController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()
        self.update_state(controller.state)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        # Status label
        self._status = QLabel("Idle")
        self._status.setStyleSheet("font-weight: bold; padding: 4px;")
        outer.addWidget(self._status)

        # Buttons
        row = QHBoxLayout()
        self._btn_start = QPushButton("Start")
        self._btn_pause = QPushButton("Pause")
        self._btn_resume = QPushButton("Resume")
        self._btn_stop = QPushButton("Stop")
        for b in (self._btn_start, self._btn_pause, self._btn_resume, self._btn_stop):
            row.addWidget(b)
        outer.addLayout(row)

        # Backup row
        row_bkp = QHBoxLayout()
        self._btn_save_bkp = QPushButton("Save backup…")
        self._btn_load_bkp = QPushButton("Load backup…")
        row_bkp.addWidget(self._btn_save_bkp)
        row_bkp.addWidget(self._btn_load_bkp)
        outer.addLayout(row_bkp)

        outer.addStretch(1)

        # Wiring
        self._btn_start.clicked.connect(self._on_start)
        self._btn_pause.clicked.connect(self._ctrl.pause)
        self._btn_resume.clicked.connect(self._ctrl.resume)
        self._btn_stop.clicked.connect(lambda: self._ctrl.stop())
        self._btn_save_bkp.clicked.connect(self._on_save_backup)
        self._btn_load_bkp.clicked.connect(self._on_load_backup)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def update_state(self, state: RunState) -> None:
        color, label = _STATE_STYLES.get(state, ("#888888", str(state)))
        self._status.setText(f"State: {label}")
        self._status.setStyleSheet(f"font-weight: bold; padding: 4px; color: {color};")

        running = state == RunState.RUNNING
        paused = state == RunState.PAUSED
        alive = self._ctrl.is_alive

        self._btn_start.setEnabled(not alive)
        self._btn_pause.setEnabled(running)
        self._btn_resume.setEnabled(paused)
        self._btn_stop.setEnabled(alive and state not in (RunState.STOPPING, RunState.FINISHED, RunState.ERROR))

    def _on_start(self) -> None:
        # The controller will rebuild state inside the worker thread.
        self._ctrl.start()

    def _on_save_backup(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save backup", "backup.pkl", "Pickle (*.pkl)")
        if path:
            from pathlib import Path

            self._ctrl.save_backup(Path(path))

    def _on_load_backup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load backup", "", "Pickle (*.pkl)")
        if path:
            from pathlib import Path

            self._ctrl.load_backup(Path(path))
