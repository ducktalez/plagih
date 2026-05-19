"""Operator & variable configuration sub-panel.

Redesigned (2026-05):
- Operators are **directly clickable** toggle buttons (no separate checkbox).
- Related operators are grouped on the **same row** (Basic arithmetic /
  Roots / MinMax / Trigonometry / Logic / Relational / Conditional).
- **Chainable operators** (Add, Mul, And, Or, Min, Max, Xor, Piecewise)
  are fully **disabled & greyed** when "Allow chained operators" is off.
- A *disabled-but-clickable* button (=not currently selected) gets a softer
  grey style; a *hard-disabled* button (chainable + chaining off) gets
  Qt's standard disabled appearance.
- **Weight editing** happens via right-click context menu → small input
  dialog. Active operators show their weight in the button text.

User-facing rules:
- The catalogue (`OPERATOR_LAYOUT` below) is the single source of truth for
  category order and per-row grouping.
- ``CHAINABLE_OPS`` mirrors the ``ChainableOp`` mixin in
  ``plagih.trees._nodes`` — keep in sync when adding new chainable ops.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from plagih.gui.core.config_schema import (
    OPERATOR_PRESET_WEIGHTS,
    RunConfig,
)

# ---------------------------------------------------------------------------
# Visual catalogue: [(category_label, [(row_label, [op_name, ...]), ...]), ...]
# ---------------------------------------------------------------------------

OPERATOR_LAYOUT: List[Tuple[str, List[Tuple[str, List[str]]]]] = [
    (
        "Math  (num → num)",
        [
            ("Basic arithmetic", ["Add", "Sub", "Mul", "Div", "Scale", "Usub"]),
            ("Powers / Logs", ["Square", "Pow", "PowRounded", "Exp", "Exp2", "Log"]),
            ("Roots", ["Sqrt", "NthRoot"]),
            ("Misc", ["Abs", "Sign", "Round"]),
            ("Min / Max", ["Min", "Max", "Clip"]),
            ("Trigonometry", ["Sin", "Cos", "Tan", "Tanh", "Sinh", "Cosh", "Asin", "Acos", "Atan"]),
        ],
    ),
    (
        "Logic  (bool → bool)",
        [
            ("Logic", ["Not", "And", "Or", "Xor"]),
        ],
    ),
    (
        "Relational  (num → bool)",
        [
            ("Relational", ["Lt", "Le", "Eq", "Ne"]),
        ],
    ),
    (
        "Conditional",
        [
            ("Conditional", ["Ifte", "Piecewise"]),
        ],
    ),
]

#: Operators marked with the ``ChainableOp`` mixin in ``plagih.trees._nodes``.
CHAINABLE_OPS = frozenset({"Add", "Mul", "And", "Or", "Min", "Max", "Xor", "Piecewise"})

# ── Button stylesheets ─────────────────────────────────────────────────────
_BTN_BASE = (
    "QToolButton {  font-size: 12px; padding: 5px 10px; border-radius: 5px;  border: 1px solid #555; min-width: 56px;}"
)
# Active (checked) → vivid blue/green
_STYLE_ACTIVE = _BTN_BASE + (
    "QToolButton{background:#1565c0;color:#e3f2fd;font-weight:bold;border-color:#1976d2;}"
    "QToolButton:hover{background:#1976d2;}"
)
# Inactive but available → muted grey (user can click to enable)
_STYLE_INACTIVE = _BTN_BASE + (
    "QToolButton{background:#3a3a3a;color:#aaa;border-color:#555;}QToolButton:hover{background:#4a4a4a;color:#ddd;}"
)
# Hard-disabled (chainable while allow_chain=False) → almost invisible
_STYLE_HARD_DISABLED = _BTN_BASE + (
    "QToolButton{background:#2a2a2a;color:#555;border-color:#3a3a3a; border-style:dashed;}"
)


class OperatorPanel(QWidget):
    """Operator & variable configurator.

    Emits :pyattr:`operators_changed` whenever the user edits a weight or
    toggles an operator, and :pyattr:`variables_changed` whenever the
    variable-selection changes.  The config panel calls
    :meth:`get_operator_weights` and :meth:`get_symbols` when *Apply* is pressed.
    """

    operators_changed = Signal(dict)  # {op_name: weight}
    variables_changed = Signal(list)  # list[str]

    def __init__(self, config: RunConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        # {op_name: QToolButton}.  Each button stores its weight in property "weight".
        self._op_buttons: Dict[str, QToolButton] = {}
        self._var_widgets: Dict[str, QCheckBox] = {}
        # Pristine: True ↔ the user has not manually changed anything since
        # the last ``set_from_config``.  get_operator_weights() returns {}
        # while pristine = "use preset from RunConfig.preset".
        self._pristine: bool = True
        # _updating: True during programmatic updates — blocks change events.
        self._updating: bool = False
        self._build_ui()
        self._load_from_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # Create the chain-toggle FIRST (un-parented for now) so that
        # ``_apply_button_style`` — which queries ``_allow_chain_chk`` while
        # building each operator button — has a valid widget to read from.
        self._allow_chain_chk = QCheckBox("Allow chained operators (Add, Mul, And, Or, Min, Max, Xor, Piecewise)")
        self._allow_chain_chk.setToolTip(
            "When enabled, chainable operators may have more than the standard 2 children.\n"
            "When disabled, the chainable operators are not used at all and appear dashed/greyed."
        )
        self._allow_chain_chk.toggled.connect(self._on_allow_chain_toggled)

        # ── Preset row ─────────────────────────────────────────────────
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(OPERATOR_PRESET_WEIGHTS.keys()))
        self._preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preset_row.addWidget(self._preset_combo, 1)
        btn_load_preset = QPushButton("Load preset")
        btn_load_preset.clicked.connect(self._on_load_preset)
        preset_row.addWidget(btn_load_preset)
        outer.addLayout(preset_row)

        # Quick-action buttons
        sel_row = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_none = QPushButton("Deselect all")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        outer.addLayout(sel_row)

        # Hint label
        hint = QLabel(
            "Click an operator to toggle • Right-click to edit its weight • Dashed = unavailable while chaining is off"
        )
        hint.setStyleSheet("color: palette(mid); font-size: 10px;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        # ── Operator groups (scrollable) ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_vbox = QVBoxLayout(scroll_content)
        scroll_vbox.setSpacing(8)
        scroll_vbox.setContentsMargins(2, 2, 2, 2)

        for category_label, rows in OPERATOR_LAYOUT:
            cat_lbl = QLabel(category_label)
            cat_lbl.setStyleSheet("font-weight: bold; font-size: 12px; color: palette(light); padding: 6px 0 2px 0;")
            scroll_vbox.addWidget(cat_lbl)

            for row_label, op_names in rows:
                row_widget = QWidget()
                row_h = QHBoxLayout(row_widget)
                row_h.setContentsMargins(8, 0, 0, 0)
                row_h.setSpacing(4)

                row_lbl = QLabel(row_label + ":")
                row_lbl.setStyleSheet("color: palette(mid); font-size: 10px;")
                row_lbl.setMinimumWidth(110)
                row_h.addWidget(row_lbl)

                for name in op_names:
                    btn = self._make_op_button(name)
                    self._op_buttons[name] = btn
                    row_h.addWidget(btn)
                row_h.addStretch(1)
                scroll_vbox.addWidget(row_widget)

            # subtle divider between categories
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet("color: #3a3a3a;")
            scroll_vbox.addWidget(line)

        scroll_vbox.addStretch(1)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)

        # ── Chained operators toggle ───────────────────────────────────
        # Widget was created at top of _build_ui; insert into layout here.
        outer.addWidget(self._allow_chain_chk)

        # ── Variables ──────────────────────────────────────────────────
        var_box = QFrame()
        var_box.setFrameShape(QFrame.Shape.StyledPanel)
        var_vbox = QVBoxLayout(var_box)
        var_vbox.setContentsMargins(6, 6, 6, 6)

        var_title = QLabel("Variables (input symbols)")
        var_title.setStyleSheet("font-weight: bold;")
        var_vbox.addWidget(var_title)

        var_hint = QLabel("Comma-separated list of input columns, or use Detect-from-CSV for checkboxes.")
        var_hint.setStyleSheet("color: palette(mid); font-size: 10px;")
        var_hint.setWordWrap(True)
        var_vbox.addWidget(var_hint)

        self._var_edit = QLineEdit()
        self._var_edit.setPlaceholderText("x, y, z  or  cartPos, cartVel")
        var_vbox.addWidget(self._var_edit)

        btn_detect = QPushButton("Detect from CSV…")
        btn_detect.setToolTip("Read column names from the training CSV and show checkboxes.")
        btn_detect.clicked.connect(self._detect_vars_from_csv)
        var_vbox.addWidget(btn_detect)

        self._var_checkbox_area = QWidget()
        self._var_checkbox_vbox = QVBoxLayout(self._var_checkbox_area)
        self._var_checkbox_vbox.setContentsMargins(0, 0, 0, 0)
        var_vbox.addWidget(self._var_checkbox_area)

        outer.addWidget(var_box)

    def _make_op_button(self, name: str) -> QToolButton:
        """Create a single checkable operator button with right-click weight edit."""
        btn = QToolButton()
        btn.setCheckable(True)
        btn.setText(name)
        btn.setProperty("op_name", name)
        btn.setProperty("weight", 1.0)
        btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        btn.customContextMenuRequested.connect(lambda _, b=btn: self._edit_weight(b))
        btn.toggled.connect(lambda _checked, b=btn: self._on_btn_toggled(b))
        if name in CHAINABLE_OPS:
            tip = f"{name} (chainable — needs 'Allow chained operators'). Right-click to set weight."
        else:
            tip = f"{name} — right-click to set weight."
        btn.setToolTip(tip)
        self._apply_button_style(btn)
        return btn

    # ------------------------------------------------------------------
    # Slots / event handlers
    # ------------------------------------------------------------------

    def _on_btn_toggled(self, btn: QToolButton) -> None:
        self._apply_button_style(btn)
        if not self._updating:
            self._pristine = False
            self.operators_changed.emit(self.get_operator_weights())

    def _edit_weight(self, btn: QToolButton) -> None:
        if not btn.isEnabled():
            return
        name = btn.property("op_name")
        current = float(btn.property("weight") or 1.0)
        new_val, ok = QInputDialog.getDouble(
            self,
            "Edit operator weight",
            f"Weight for '{name}':",
            current,
            0.0,
            20.0,
            2,
        )
        if not ok:
            return
        btn.setProperty("weight", float(new_val))
        if not btn.isChecked() and new_val > 0:
            # Setting a positive weight implicitly activates the operator
            self._updating = True
            btn.setChecked(True)
            self._updating = False
        self._apply_button_style(btn)
        self._pristine = False
        self.operators_changed.emit(self.get_operator_weights())

    def _on_load_preset(self) -> None:
        name = self._preset_combo.currentText()
        weights = OPERATOR_PRESET_WEIGHTS.get(name, {})
        self._apply_weights(weights)
        self._pristine = False

    def _on_allow_chain_toggled(self, _checked: bool) -> None:
        # Re-style every chainable button (enabled state + look).
        for name in CHAINABLE_OPS:
            btn = self._op_buttons.get(name)
            if btn is not None:
                self._apply_button_style(btn)
        if not self._updating:
            self._pristine = False

    def _set_all_checked(self, checked: bool) -> None:
        self._updating = True
        try:
            for btn in self._op_buttons.values():
                if checked and not btn.isEnabled():
                    continue  # don't try to enable hard-disabled buttons
                btn.setChecked(checked)
                self._apply_button_style(btn)
        finally:
            self._updating = False
        self._pristine = False
        self.operators_changed.emit(self.get_operator_weights())

    def _detect_vars_from_csv(self) -> None:
        csv_path = self._config.df_train_csv or ""
        if not csv_path:
            csv_path, _ = QFileDialog.getOpenFileName(self, "Select training CSV", "", "CSV (*.csv)")
        if not csv_path or not Path(csv_path).is_file():
            return

        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = _csv.reader(fh)
                headers = next(reader)
        except Exception:
            return

        target_col = self._config.target_column or "action"
        cols = [h.split(":")[0].strip() for h in headers]
        non_target = [c for c in cols if c != target_col]

        while self._var_checkbox_vbox.count():
            item = self._var_checkbox_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._var_widgets.clear()

        current_symbols = set(self.get_symbols())
        for col in non_target:
            chk = QCheckBox(col)
            chk.setChecked(col in current_symbols or not current_symbols)
            self._var_checkbox_vbox.addWidget(chk)
            self._var_widgets[col] = chk
            chk.toggled.connect(lambda _: self.variables_changed.emit(self.get_symbols()))

        self._var_edit.clear()
        self._var_edit.setPlaceholderText("(using checkbox selection above)")

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------

    def _apply_button_style(self, btn: QToolButton) -> None:
        """Update button text + stylesheet based on (checked, weight, allow_chain)."""
        name = btn.property("op_name")
        weight = float(btn.property("weight") or 1.0)
        is_chainable = name in CHAINABLE_OPS
        chain_off = is_chainable and not self._allow_chain_chk.isChecked()

        if chain_off:
            # Hard-disabled: cannot be used, fully greyed
            prev_updating = self._updating
            self._updating = True
            try:
                if btn.isChecked():
                    btn.setChecked(False)
            finally:
                self._updating = prev_updating
            btn.setEnabled(False)
            btn.setText(name)
            btn.setStyleSheet(_STYLE_HARD_DISABLED)
            return

        btn.setEnabled(True)
        if btn.isChecked():
            btn.setText(f"{name}  x{weight:g}")
            btn.setStyleSheet(_STYLE_ACTIVE)
        else:
            btn.setText(name)
            btn.setStyleSheet(_STYLE_INACTIVE)

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------

    def get_operator_weights(self) -> Dict[str, float]:
        """Return the current operator config as ``{name: weight}``.

        Returns an **empty dict** while in the pristine state (= "use the
        preset named in :attr:`RunConfig.preset`").
        """
        if self._pristine:
            return {}
        result: Dict[str, float] = {}
        for name, btn in self._op_buttons.items():
            if btn.isChecked() and btn.isEnabled():
                result[name] = float(btn.property("weight") or 1.0)
        return result

    def get_symbols(self) -> List[str]:
        if self._var_widgets:
            return [name for name, chk in self._var_widgets.items() if chk.isChecked()]
        raw = self._var_edit.text().strip()
        if not raw:
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    def get_allow_chain(self) -> bool:
        return self._allow_chain_chk.isChecked()

    def set_from_config(self, config: RunConfig) -> None:
        self._config = config
        self._pristine = True
        self._load_from_config()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_from_config(self) -> None:
        cfg = self._config
        # allow_chain must be set BEFORE _apply_weights so chainable buttons
        # have the right enabled-state when their weight is applied.
        self._updating = True
        try:
            self._allow_chain_chk.setChecked(cfg.allow_chain)
        finally:
            self._updating = False

        if cfg.operator_weights:
            self._apply_weights(cfg.operator_weights)
        else:
            preset_name = cfg.preset
            idx = self._preset_combo.findText(preset_name)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)
            weights = OPERATOR_PRESET_WEIGHTS.get(preset_name, OPERATOR_PRESET_WEIGHTS["math_simple"])
            self._apply_weights(weights)

        symbols_str = ", ".join(cfg.symbols) if cfg.symbols else ""
        self._var_edit.setText(symbols_str)

    def _apply_weights(self, weights: Dict[str, float]) -> None:
        """Set buttons from a ``{name: weight}`` dict (string keys)."""
        self._updating = True
        try:
            # Reset every button
            for name, btn in self._op_buttons.items():
                btn.setProperty("weight", 1.0)
                btn.setChecked(False)
            # Apply the new weights
            for op_name, w in weights.items():
                btn = self._op_buttons.get(op_name)
                if btn is None:
                    continue
                btn.setProperty("weight", max(0.0, float(w)))
                if w > 0:
                    btn.setChecked(True)
            # Re-style everything (handles chain-off greying)
            for btn in self._op_buttons.values():
                self._apply_button_style(btn)
        finally:
            self._updating = False
