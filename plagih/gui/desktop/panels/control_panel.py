"""Start / Pause / Stop / Backup controls.

Layout (top → bottom):
    1. Status line                  ("⬤  Running")
    2. Backup directory field       (read-only QLineEdit)
    3. Save backup / Load backup    (two buttons)
    4. ──────── separator ────────
    5. Play-Pause toggle + Stop     (large, prominent buttons)

Design notes
------------
- Previously there were four buttons (Start / Pause / Resume / Stop).
  Start, Pause and Resume have been merged into a **single toggle button**
  (``self._btn_play``) that re-labels itself based on :class:`RunState`:

    IDLE / FINISHED / ERROR  →  "▶  Start"
    RUNNING                  →  "⏸  Pause"
    PAUSED                   →  "▶  Resume"
    STARTING / STOPPING      →  disabled, shows current text

  This matches how every media player works and removes redundant buttons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

_BIG_BTN_BASE = (
    "QPushButton {"
    "  font-size: 16px; font-weight: bold;"
    "  padding: 10px 14px; border-radius: 6px; border: 1px solid #555;"
    "  min-height: 36px;"
    "}"
    "QPushButton:disabled { color: #666; border-color: #444; background: #3a3a3a; }"
)
_PLAY_STYLE = (
    _BIG_BTN_BASE + "QPushButton{background:#1b5e20;color:#c8e6c9;}QPushButton:hover:!disabled{background:#2e7d32;}"
)
_PAUSE_STYLE = (
    _BIG_BTN_BASE + "QPushButton{background:#e65100;color:#ffe0b2;}QPushButton:hover:!disabled{background:#f57c00;}"
)
_STOP_STYLE = (
    _BIG_BTN_BASE + "QPushButton{background:#b71c1c;color:#ffcdd2;}QPushButton:hover:!disabled{background:#c62828;}"
)

_SMALL_BTN_STYLE = (
    "QPushButton{font-size:12px;padding:4px 10px;border:1px solid #555;border-radius:4px;}"
    "QPushButton:hover{background:palette(midlight);}"
)


class ControlPanel(QWidget):
    """Run lifecycle control widget."""

    def __init__(self, controller: RunController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()
        self.update_state(controller.state)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        # 1. Status label
        self._status = QLabel("⬤  Idle")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("font-weight:bold; font-size:13px; padding:4px; border-radius:3px;")
        outer.addWidget(self._status)

        # 2. Backup directory field
        bkp_path_row = QHBoxLayout()
        bkp_path_row.addWidget(QLabel("Backup dir:"))
        self._bkp_path_edit = QLineEdit()
        self._bkp_path_edit.setReadOnly(True)
        self._bkp_path_edit.setPlaceholderText("(no backup path yet)")
        self._bkp_path_edit.setToolTip("Directory of the most recent save / load backup operation.")
        bkp_path_row.addWidget(self._bkp_path_edit, 1)
        outer.addLayout(bkp_path_row)

        # 3. Save / Load backup buttons
        bkp_btn_row = QHBoxLayout()
        self._btn_save_bkp = QPushButton("💾  Save backup…")
        self._btn_load_bkp = QPushButton("📂  Load backup…")
        for b in (self._btn_save_bkp, self._btn_load_bkp):
            b.setStyleSheet(_SMALL_BTN_STYLE)
            bkp_btn_row.addWidget(b)
        outer.addLayout(bkp_btn_row)

        # 4. Separator
        sep = QLabel("─────────  Run control  ─────────")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("color: palette(mid); font-size: 10px; padding: 4px 0;")
        outer.addWidget(sep)

        # 5. Play / Stop buttons (below the folder section)
        run_btn_row = QHBoxLayout()
        run_btn_row.setSpacing(8)
        self._btn_play = QPushButton("▶  Start")
        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_play.setStyleSheet(_PLAY_STYLE)
        self._btn_stop.setStyleSheet(_STOP_STYLE)
        run_btn_row.addWidget(self._btn_play, 2)
        run_btn_row.addWidget(self._btn_stop, 1)
        outer.addLayout(run_btn_row)

        outer.addStretch(1)

        # Wiring
        self._btn_play.clicked.connect(self._on_play_clicked)
        self._btn_stop.clicked.connect(lambda: self._ctrl.stop())
        self._btn_save_bkp.clicked.connect(self._on_save_backup)
        self._btn_load_bkp.clicked.connect(self._on_load_backup)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def update_state(self, state: RunState) -> None:
        color, label = _STATE_STYLES.get(state, ("#888888", str(state)))
        self._status.setText(f"⬤  {label}")
        self._status.setStyleSheet(f"font-weight:bold; font-size:13px; padding:4px; border-radius:3px; color:{color};")

        alive = self._ctrl.is_alive
        if state == RunState.RUNNING:
            self._btn_play.setText("⏸  Pause")
            self._btn_play.setStyleSheet(_PAUSE_STYLE)
            self._btn_play.setEnabled(True)
        elif state == RunState.PAUSED:
            self._btn_play.setText("▶  Resume")
            self._btn_play.setStyleSheet(_PLAY_STYLE)
            self._btn_play.setEnabled(True)
        elif state in (RunState.IDLE, RunState.FINISHED, RunState.ERROR) and not alive:
            self._btn_play.setText("▶  Start")
            self._btn_play.setStyleSheet(_PLAY_STYLE)
            self._btn_play.setEnabled(True)
        else:
            # STARTING / STOPPING — transient
            self._btn_play.setEnabled(False)

        self._btn_stop.setEnabled(alive and state not in (RunState.STOPPING, RunState.FINISHED, RunState.ERROR))

    def _on_play_clicked(self) -> None:
        """Toggle: Start ↔ Pause ↔ Resume depending on current state."""
        state = self._ctrl.state
        if state == RunState.RUNNING:
            self._ctrl.pause()
        elif state == RunState.PAUSED:
            self._ctrl.resume()
        else:
            self._ctrl.start()
            if self._ctrl.config.rootdir:
                self._bkp_path_edit.setText(str(Path(self._ctrl.config.rootdir) / "backup"))

    def _on_save_backup(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save backup", "backup.pkl", "Pickle (*.pkl)")
        if path:
            p = Path(path)
            self._ctrl.save_backup(p)
            self._bkp_path_edit.setText(str(p.parent))

    def _on_load_backup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load backup", "", "Pickle (*.pkl)")
        if path:
            p = Path(path)
            self._ctrl.load_backup(p)
            self._bkp_path_edit.setText(str(p.parent))
