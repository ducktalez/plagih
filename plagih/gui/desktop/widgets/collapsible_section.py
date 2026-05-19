"""Collapsible section widget for the plagih desktop GUI.

A QPushButton arrow-toggle + content area that can be expanded/collapsed.
Works in both light and dark mode via QPalette-relative colours.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A titled section that can be toggled open or closed.

    Usage::

        sec = CollapsibleSection("Engine", expanded=True)
        inner = QFormLayout()
        inner.addRow("foo", QSpinBox())
        sec.set_content_layout(inner)
        outer_layout.addWidget(sec)
    """

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._is_expanded = expanded

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 1, 0, 1)
        vbox.setSpacing(0)

        # ── Toggle button ─────────────────────────────────────────────────
        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left;"
            "  font-weight: bold;"
            "  font-size: 12px;"
            "  padding: 5px 8px;"
            "  border: none;"
            "  border-bottom: 1px solid palette(mid);"
            "  background: palette(button);"
            "  color: palette(buttonText);"
            "}"
            "QPushButton:hover { background: palette(midlight); }"
            "QPushButton:checked { background: palette(shadow); }"
        )
        self._update_btn_text()
        vbox.addWidget(self._btn)

        # ── Content area ──────────────────────────────────────────────────
        self._content = QWidget()
        self._content.setVisible(expanded)
        vbox.addWidget(self._content)

        self._btn.toggled.connect(self._on_toggle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_content_layout(self, layout: QLayout) -> None:
        """Attach *layout* as the section body."""
        self._content.setLayout(layout)

    @property
    def content(self) -> QWidget:
        """The inner content widget (use :meth:`set_content_layout` to fill it)."""
        return self._content

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically expand or collapse."""
        self._btn.setChecked(expanded)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_btn_text(self) -> None:
        arrow = "▼" if self._is_expanded else "▶"
        self._btn.setText(f"  {arrow}  {self._title}")

    def _on_toggle(self, checked: bool) -> None:
        self._is_expanded = checked
        self._update_btn_text()
        self._content.setVisible(checked)
        # Propagate resize hint up the widget hierarchy
        parent = self.parent()
        if parent is not None:
            parent.adjustSize()
