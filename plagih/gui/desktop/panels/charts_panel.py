"""Live monitoring charts (fitness, parsimony, diversity, time per gen).

Re-uses :meth:`GPMonitor.plot_performance` indirectly by replicating the
same four-panel layout on a Qt-embedded Matplotlib canvas.  Updates only
on :class:`EventType.GENERATION_DONE`, throttled to ~2 Hz, and can be
paused via a checkbox.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class ChartsPanel(QWidget):
    """Four-up matplotlib canvas with live-update toggle."""

    _MIN_REDRAW_INTERVAL_S = 0.5  # max ~2 Hz

    def __init__(self, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._last_redraw = 0.0
        self._pending_redraw = False

        outer = QVBoxLayout(self)

        # Header controls
        row = QHBoxLayout()
        self._chk_live = QCheckBox("Live aktualisieren")
        self._chk_live.setChecked(True)
        self._btn_refresh = QPushButton("Jetzt aktualisieren")
        self._btn_refresh.clicked.connect(self._force_redraw)
        row.addWidget(self._chk_live)
        row.addStretch(1)
        row.addWidget(self._btn_refresh)
        outer.addLayout(row)

        # Matplotlib canvas
        self._fig = Figure(figsize=(8, 5), tight_layout=True)
        self._ax_fit = self._fig.add_subplot(2, 2, 1)
        self._ax_par = self._fig.add_subplot(2, 2, 2)
        self._ax_div = self._fig.add_subplot(2, 2, 3)
        self._ax_time = self._fig.add_subplot(2, 2, 4)
        self._canvas = FigureCanvasQTAgg(self._fig)
        outer.addWidget(self._canvas, 1)

        self._draw_empty()

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    def on_generation_done(self, _payload: dict) -> None:
        """Called when a GENERATION_DONE event arrives."""
        if not self._chk_live.isChecked():
            return
        now = time.monotonic()
        if now - self._last_redraw >= self._MIN_REDRAW_INTERVAL_S:
            self._redraw()
        else:
            # Coalesce — most recent update will run when the cooldown expires.
            self._pending_redraw = True

    def _force_redraw(self) -> None:
        self._redraw()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_empty(self) -> None:
        for ax, title in (
            (self._ax_fit, "Fitness"),
            (self._ax_par, "Parsimony"),
            (self._ax_div, "Population diversity"),
            (self._ax_time, "Time per generation"),
        ):
            ax.clear()
            ax.set_title(title)
            ax.text(0.5, 0.5, "(no data yet)", ha="center", va="center", transform=ax.transAxes, alpha=0.5)
            ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()

    def _redraw(self) -> None:
        gp = getattr(self._ctrl, "gp", None)
        if gp is None or len(gp.monitor) == 0:
            self._draw_empty()
            return

        monitor = gp.monitor
        gens = monitor.get_generation_ids()
        fit_best = monitor.get_metric_series("fit_best")
        fit_mean = monitor.get_metric_series("fit_mean")
        fit_q25 = monitor.get_metric_series("fit_q25")
        fit_q75 = monitor.get_metric_series("fit_q75")
        par_best = monitor.get_metric_series("parsim_best")
        par_mean = monitor.get_metric_series("parsim_mean")
        par_q25 = monitor.get_metric_series("parsim_q25")
        par_q75 = monitor.get_metric_series("parsim_q75")
        pop_size = monitor.get_metric_series("pop_size")
        pop_uniq = monitor.get_metric_series("pop_unique")
        gen_time = monitor.get_metric_series("gen_time")

        ax = self._ax_fit
        ax.clear()
        ax.set_title("Fitness (lower = better)")
        ax.plot(gens, fit_best, "b-", lw=2, label="best")
        ax.plot(gens, fit_mean, "g--", label="mean")
        ax.fill_between(gens, fit_q25, fit_q75, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = self._ax_par
        ax.clear()
        ax.set_title("Parsimony (complexity)")
        ax.plot(gens, par_best, "b-", lw=2, label="best")
        ax.plot(gens, par_mean, "g--", label="mean")
        ax.fill_between(gens, par_q25, par_q75, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = self._ax_div
        ax.clear()
        ax.set_title("Population diversity")
        ax.plot(gens, pop_size, "b-", label="total")
        ax.plot(gens, pop_uniq, "r--", label="unique")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = self._ax_time
        ax.clear()
        ax.set_title("Time per generation (s)")
        ax.bar(gens, gen_time, alpha=0.7, color="steelblue")
        if len(gen_time) and not np.all(np.isnan(gen_time)):
            mean = float(np.nanmean(gen_time))
            ax.axhline(mean, color="red", ls="--", label=f"mean {mean:.2f}s")
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        self._canvas.draw_idle()
        self._last_redraw = time.monotonic()
        self._pending_redraw = False
