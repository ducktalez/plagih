"""Pareto-front view: table + scatter plot.

Updated only on :class:`EventType.PARETO_CHANGED` (rare).
"""

from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class ParetoPanel(QWidget):
    """Pareto-front table + scatter."""

    def __init__(self, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = controller

        outer = QHBoxLayout(self)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Parsimony", "Fitness", "Expression"])
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        outer.addWidget(self._table, 2)

        self._fig = Figure(figsize=(4, 4), tight_layout=True)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._canvas = FigureCanvasQTAgg(self._fig)
        outer.addWidget(self._canvas, 1)

        self._draw_empty()

    def _draw_empty(self) -> None:
        self._ax.clear()
        self._ax.set_xlabel("Parsimony")
        self._ax.set_ylabel("Fitness")
        self._ax.set_title("Pareto front")
        self._ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()

    def on_pareto_changed(self, _payload: dict) -> None:
        gp = getattr(self._ctrl, "gp", None)
        front = list(gp.paretofront) if gp is not None else []

        self._table.setRowCount(len(front))
        xs: list[float] = []
        ys: list[float] = []
        for i, c in enumerate(front):
            par = c.get_parsim()
            fit = c.get_fitness()
            xs.append(par)
            ys.append(fit)
            try:
                expr = str(c.tree)
            except Exception:
                expr = "<unrenderable>"
            self._table.setItem(i, 0, QTableWidgetItem(f"{par}"))
            self._table.setItem(i, 1, QTableWidgetItem(f"{fit:.6f}"))
            self._table.setItem(i, 2, QTableWidgetItem(expr))

        self._ax.clear()
        self._ax.set_xlabel("Parsimony")
        self._ax.set_ylabel("Fitness")
        self._ax.set_title(f"Pareto front ({len(front)} candidates)")
        if xs:
            self._ax.scatter(xs, ys, c="steelblue")
        self._ax.grid(True, alpha=0.3)
        self._canvas.draw_idle()
