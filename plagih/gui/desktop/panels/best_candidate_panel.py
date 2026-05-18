"""Display the latest Pareto-front addition.

Renders the candidate tree to PNG via :func:`render_tree` in a background
:class:`QRunnable` to keep the GUI responsive.  A single-slot coalescing
queue ensures only the most recent request is processed if events arrive
in quick succession.

Updates only on :class:`EventType.PARETO_CHANGED`.  Disabled by the
"Baum rendern" checkbox.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class _RenderSignals(QObject):
    """Signals fired by background render jobs."""

    finished = Signal(str, str)  # (job_id, image_path)
    failed = Signal(str, str)  # (job_id, error message)


class _RenderJob(QRunnable):
    """QRunnable that renders a tree to PNG off the GUI thread."""

    def __init__(self, job_id: str, tree, output_dir: Path, filename: str) -> None:
        super().__init__()
        self.job_id = job_id
        self._tree = tree
        self._output_dir = output_dir
        self._filename = filename
        self.signals = _RenderSignals()

    def run(self) -> None:  # type: ignore[override]
        try:
            # Late import keeps Qt import optional for plagih.gui.core.
            from plagih.visualization.tree_renderer import render_tree

            path = render_tree(
                self._tree,
                filename=self._filename,
                output_dir=self._output_dir,
                orientation="TB",
                title="",
                show=False,
            )
            self.signals.finished.emit(self.job_id, str(path))
        except Exception as exc:
            self.signals.failed.emit(self.job_id, str(exc))


class BestCandidatePanel(QWidget):
    """Latest-Pareto-addition panel with on-demand tree rendering."""

    def __init__(self, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._thread_pool = QThreadPool.globalInstance()
        self._output_dir = Path(tempfile.gettempdir()) / "plagih_gui_trees"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Coalescing slots: only the most recent request is honoured.
        self._pending_tree = None
        self._pending_meta: dict[str, Any] = {}
        self._active_job_id: Optional[str] = None
        self._queued_job_id: Optional[str] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        # Header controls
        header = QHBoxLayout()
        self._chk_render = QCheckBox("Baum rendern")
        self._chk_render.setChecked(True)
        self._chk_render.toggled.connect(self._on_toggle_render)
        self._btn_render_now = QPushButton("Jetzt rendern")
        self._btn_render_now.clicked.connect(self._render_pending_if_any)
        header.addWidget(self._chk_render)
        header.addStretch(1)
        header.addWidget(self._btn_render_now)
        outer.addLayout(header)

        # Metadata
        gb_meta = QGroupBox("Latest Pareto entry")
        meta_layout = QVBoxLayout(gb_meta)
        self._lbl_meta = QLabel("(no candidate yet)")
        self._lbl_meta.setWordWrap(True)
        meta_layout.addWidget(self._lbl_meta)
        self._expr_view = QPlainTextEdit()
        self._expr_view.setReadOnly(True)
        self._expr_view.setMaximumHeight(120)
        meta_layout.addWidget(self._expr_view)
        outer.addWidget(gb_meta)

        # Tree image area
        self._image_label = QLabel("(no tree rendered yet)")
        self._image_label.setAlignment(self._image_label.alignment())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._image_label)
        outer.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def on_pareto_changed(self, payload: dict) -> None:
        candidate = payload.get("latest_added")
        if candidate is None:
            return

        # Update text immediately (cheap).
        try:
            expr = str(candidate.tree)
        except Exception:
            expr = "<unrenderable>"
        try:
            sympy_expr = candidate.tree.get_sympy_expr()
        except Exception:
            sympy_expr = None
        self._lbl_meta.setText(
            f"Parsimony: {candidate.get_parsim()}  |  "
            f"Fitness: {candidate.get_fitness():.6f}  |  "
            f"Tag: {getattr(candidate, 'tag', '-')}"
        )
        body = expr if sympy_expr is None else f"{expr}\n\n[sympy] {sympy_expr}"
        self._expr_view.setPlainText(body)

        # Queue render
        self._pending_tree = candidate.tree
        self._pending_meta = {
            "parsim": candidate.get_parsim(),
            "fitness": candidate.get_fitness(),
        }
        if self._chk_render.isChecked():
            self._render_pending_if_any()

    # ------------------------------------------------------------------
    # Render queue
    # ------------------------------------------------------------------

    def _on_toggle_render(self, on: bool) -> None:
        if on:
            self._render_pending_if_any()

    def _render_pending_if_any(self) -> None:
        if self._pending_tree is None:
            return

        # Coalesce: if a job is in flight, just remember that we want
        # another one when it finishes (and overwrite the queued slot).
        if self._active_job_id is not None:
            self._queued_job_id = uuid.uuid4().hex
            return

        self._start_job_now()

    def _start_job_now(self) -> None:
        if self._pending_tree is None:
            return
        job_id = uuid.uuid4().hex
        self._active_job_id = job_id
        filename = f"best_{int(time.time() * 1000)}"
        job = _RenderJob(job_id, self._pending_tree, self._output_dir, filename)
        job.signals.finished.connect(self._on_render_finished)
        job.signals.failed.connect(self._on_render_failed)
        # Snapshot pending so a subsequent event still queues a follow-up.
        self._pending_tree = None
        self._thread_pool.start(job)

    def _on_render_finished(self, job_id: str, path: str) -> None:
        if job_id != self._active_job_id:
            return
        self._active_job_id = None
        try:
            pix = QPixmap(path)
            if pix.isNull():
                self._image_label.setText(f"(could not load image: {path})")
            else:
                self._image_label.setPixmap(pix)
                self._image_label.adjustSize()
        finally:
            # Honour queued request, if any
            if self._queued_job_id is not None:
                self._queued_job_id = None
                self._render_pending_if_any()

    def _on_render_failed(self, job_id: str, message: str) -> None:
        if job_id != self._active_job_id:
            return
        self._active_job_id = None
        self._image_label.setText(f"(tree render failed: {message})")
        if self._queued_job_id is not None:
            self._queued_job_id = None
            self._render_pending_if_any()
