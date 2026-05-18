"""Log + per-generation metrics panel.

Two stacked widgets:
- A read-only text view that streams :class:`EventType.LOG` messages.
- A table that gains one row per :class:`EventType.GENERATION_DONE`.

The log view keeps at most :data:`_MAX_LOG_LINES` entries (ring buffer
behaviour via line trimming).
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_MAX_LOG_LINES = 1000


class LogPanel(QWidget):
    """Combined log + per-generation metrics view."""

    _METRIC_COLS = [
        ("gen_id", "Gen"),
        ("fit_best", "fit_best"),
        ("fit_mean", "fit_mean"),
        ("parsim_best", "parsim_best"),
        ("parsim_mean", "parsim_mean"),
        ("pop_size", "pop"),
        ("pop_unique", "uniq"),
        ("gen_time", "time_s"),
        ("lut_size", "lut"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(splitter, 1)

        # Metrics table
        gb_metrics = QGroupBox("Generation metrics")
        v1 = QVBoxLayout(gb_metrics)
        self._table = QTableWidget(0, len(self._METRIC_COLS))
        self._table.setHorizontalHeaderLabels([label for _, label in self._METRIC_COLS])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v1.addWidget(self._table)
        splitter.addWidget(gb_metrics)

        # Log stream
        gb_log = QGroupBox("Log")
        v2 = QVBoxLayout(gb_log)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(_MAX_LOG_LINES)
        v2.addWidget(self._log_view)
        splitter.addWidget(gb_log)

        splitter.setSizes([300, 200])

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def on_generation_done(self, payload: dict) -> None:
        gen_id = payload.get("gen_id", "?")
        metrics = payload.get("metrics", {})
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, (key, _label) in enumerate(self._METRIC_COLS):
            value = gen_id if key == "gen_id" else metrics.get(key, "")
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            self._table.setItem(row, col, QTableWidgetItem(text))
        self._table.scrollToBottom()

    def on_log(self, payload: dict) -> None:
        level = payload.get("level", "info")
        msg = payload.get("message", "")
        ts = time.strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{ts}] {level.upper():<7} {msg}")
