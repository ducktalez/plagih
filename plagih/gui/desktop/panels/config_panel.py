"""Config & strategy editor panel.

Builds a form widget from :class:`RunConfig` dataclass fields arranged in
collapsible sections.  Fields listed in :data:`LIVE_EDITABLE_FIELDS` are
tagged ``[live]``; others are tagged ``[reload]``.

Layout:
  - Header row: "Settings" title + "Save settings" + "Load settings"  (same line)
  - Scrollable collapsible sections: Data & output | Engine | PlagihConfig |
    Operators & Variables | Strategies
  - Footer: "Apply changes" button
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from plagih.gui.core.config_schema import (
    DEFAULT_STRATEGIES,
    LIVE_EDITABLE_FIELDS,
    RunConfig,
    StrategySpec,
)
from plagih.gui.desktop.panels.operator_panel import OperatorPanel
from plagih.gui.desktop.widgets.collapsible_section import CollapsibleSection

_ERROR_METRICS = ["rmse", "mse", "mae"]


class ConfigPanel(QWidget):
    """Form-based editor for :class:`RunConfig`.

    Emits :pyattr:`config_changed` on every Apply click.
    """

    config_changed = Signal(dict, bool)  # (partial changes, force_reload)

    def __init__(self, config: RunConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._widgets: Dict[str, QWidget] = {}
        self._build_ui()
        self._populate_from_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(4)

        # ── Header row: "Settings"  [Save settings…]  [Load settings…] ──
        header_row = QHBoxLayout()
        title_lbl = QLabel("Settings")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; padding-right: 8px;")
        header_row.addWidget(title_lbl)
        btn_save = QPushButton("Save settings…")
        btn_load = QPushButton("Load settings…")
        btn_save.clicked.connect(self._on_save)
        btn_load.clicked.connect(self._on_load)
        header_row.addWidget(btn_save)
        header_row.addWidget(btn_load)
        header_row.addStretch(1)
        outer.addLayout(header_row)

        # ── Scrollable content ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        form_outer = QVBoxLayout(content)
        form_outer.setSpacing(2)

        # ── 1. Data & output ──────────────────────────────────────────
        sec_data = CollapsibleSection("Data & output", expanded=True)
        f_data = QFormLayout()
        self._add_path_field(f_data, "df_train_csv", "Training CSV", file_filter="CSV (*.csv)")
        self._add_path_field(f_data, "rootdir", "Output directory", is_dir=True)
        self._add_text_field(f_data, "target_column", "Target column")
        sec_data.set_content_layout(f_data)
        form_outer.addWidget(sec_data)

        # ── 2. Engine ──────────────────────────────────────────────────
        sec_eng = CollapsibleSection("Engine", expanded=True)
        f_eng = QFormLayout()
        self._add_int_field(f_eng, "depth_max", "depth_max", 1, 32)
        self._add_int_field(f_eng, "nodes_max", "nodes_max", 1, 500)
        self._add_int_field(f_eng, "pop_max_size", "pop_max_size", 1, 100_000)
        self._add_int_field(f_eng, "gen_end", "gen_end", 1, 1_000_000)
        self._add_text_field(f_eng, "clip_range", "clip_range (min,max or empty)", placeholder="0,2")
        self._add_combo_field(f_eng, "error_metric", "error_metric", _ERROR_METRICS)
        self._add_int_field(f_eng, "parallel", "parallel workers (0=seq)", 0, 256)
        self._add_bool_field(f_eng, "enable_analysis", "enable_analysis (plots/backups)")
        sec_eng.set_content_layout(f_eng)
        form_outer.addWidget(sec_eng)

        # ── 3. PlagihConfig ────────────────────────────────────────────
        sec_cfg = CollapsibleSection("PlagihConfig (cfg.*)", expanded=False)
        f_cfg = QFormLayout()
        self._add_text_field(f_cfg, "verbosity", "verbosity")
        self._add_bool_field(f_cfg, "simplification", "simplification")
        self._add_bool_field(f_cfg, "visualization", "visualization")
        self._add_bool_field(f_cfg, "merged_tree", "merged_tree")
        self._add_bool_field(f_cfg, "origin_tree", "origin_tree (track metadata)")
        self._add_bool_field(f_cfg, "lut_enabled", "lut_enabled")
        self._add_int_field(f_cfg, "plots_interval", "plots_interval", 1, 10_000)
        self._add_int_field(f_cfg, "backup_interval", "backup_interval", 1, 10_000)
        self._add_int_field(f_cfg, "tree_min_parsimony", "tree_min_parsimony", 0, 1_000)
        self._add_int_field(f_cfg, "float_precision", "float_precision", 0, 12)
        sec_cfg.set_content_layout(f_cfg)
        form_outer.addWidget(sec_cfg)

        # ── 4. Operators & Variables ───────────────────────────────────
        sec_ops = CollapsibleSection("Operators & Variables", expanded=False)
        ops_vbox = QVBoxLayout()
        ops_vbox.setContentsMargins(0, 0, 0, 0)

        # origin_tree path (seed tree file)
        origin_row = QHBoxLayout()
        origin_edit = QLineEdit()
        origin_edit.setPlaceholderText("(optional) path to seed-tree .pkl")
        origin_edit.setToolTip(
            "Pickled Node tree (.pkl) to add as origin candidate and use for TED parsimony.\nLeave empty to disable."
        )
        btn_origin = QPushButton("Browse…")

        def _pick_origin():
            p, _ = QFileDialog.getOpenFileName(self, "Select origin tree", "", "Pickle (*.pkl)")
            if p:
                origin_edit.setText(p)

        btn_origin.clicked.connect(_pick_origin)
        origin_row.addWidget(QLabel("Origin tree:"))
        origin_row.addWidget(origin_edit, 1)
        origin_row.addWidget(btn_origin)
        ops_vbox.addLayout(origin_row)
        self._widgets["origin_tree_path"] = origin_edit

        # Operator panel
        self._operator_panel = OperatorPanel(config=self._config)
        ops_vbox.addWidget(self._operator_panel)
        sec_ops.set_content_layout(ops_vbox)
        form_outer.addWidget(sec_ops)

        # ── 5. Strategies ──────────────────────────────────────────────
        sec_strat = CollapsibleSection("Strategies (JSON — one per generation)", expanded=False)
        strat_vbox = QVBoxLayout()
        self._strategies_edit = QPlainTextEdit()
        self._strategies_edit.setPlaceholderText('[{"name": "mutation", "rate": 0.4, ...}]')
        strat_vbox.addWidget(self._strategies_edit)
        btn_default = QPushButton("Reset to default strategies")
        btn_default.clicked.connect(self._reset_default_strategies)
        strat_vbox.addWidget(btn_default)
        sec_strat.set_content_layout(strat_vbox)
        form_outer.addWidget(sec_strat)

        form_outer.addStretch(1)

        # ── Footer: Apply button ───────────────────────────────────────
        apply_row = QHBoxLayout()
        self._btn_apply = QPushButton("Apply changes")
        self._btn_apply.setToolTip(
            "Pushes changed fields to the running controller.\n"
            "Fields tagged [reload] trigger a backup→recreate→restore cycle."
        )
        self._btn_apply.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 16px;"
            " border: 1px solid #555; border-radius: 4px; }"
            "QPushButton:hover { background: palette(midlight); }"
        )
        self._btn_apply.clicked.connect(self._on_apply)
        apply_row.addStretch(1)
        apply_row.addWidget(self._btn_apply)
        outer.addLayout(apply_row)

    # -- field helpers -----------------------------------------------------

    @staticmethod
    def _badge(field_name: str) -> str:
        return "[live]" if field_name in LIVE_EDITABLE_FIELDS else "[reload]"

    def _add_text_field(self, form: QFormLayout, name: str, label: str, *, placeholder: str = "") -> None:
        w = QLineEdit()
        w.setPlaceholderText(placeholder)
        self._widgets[name] = w
        form.addRow(f"{label}  {self._badge(name)}", w)

    def _add_int_field(self, form: QFormLayout, name: str, label: str, lo: int, hi: int) -> None:
        w = QSpinBox()
        w.setRange(lo, hi)
        self._widgets[name] = w
        form.addRow(f"{label}  {self._badge(name)}", w)

    def _add_bool_field(self, form: QFormLayout, name: str, label: str) -> None:
        w = QCheckBox()
        self._widgets[name] = w
        form.addRow(f"{label}  {self._badge(name)}", w)

    def _add_combo_field(self, form: QFormLayout, name: str, label: str, items: List[str]) -> None:
        w = QComboBox()
        w.addItems(items)
        self._widgets[name] = w
        form.addRow(f"{label}  {self._badge(name)}", w)

    def _add_path_field(
        self,
        form: QFormLayout,
        name: str,
        label: str,
        *,
        file_filter: str = "",
        is_dir: bool = False,
    ) -> None:
        edit = QLineEdit()
        btn = QPushButton("Browse…")

        def _pick() -> None:
            if is_dir:
                p = QFileDialog.getExistingDirectory(self, label)
            else:
                p, _ = QFileDialog.getOpenFileName(self, label, "", file_filter)
            if p:
                edit.setText(p)

        btn.clicked.connect(_pick)
        wrap = QWidget()
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        self._widgets[name] = edit
        form.addRow(f"{label}  {self._badge(name)}", wrap)

    # ------------------------------------------------------------------
    # Populate / collect
    # ------------------------------------------------------------------

    def _populate_from_config(self) -> None:
        c = self._config
        for f in dataclasses.fields(c):
            if f.name in ("strategies", "operator_weights", "origin_tree_path"):
                continue
            w = self._widgets.get(f.name)
            if w is None:
                continue
            v = getattr(c, f.name)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QComboBox):
                idx = w.findText(str(v))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QLineEdit):
                w.setText(self._format_value(f.name, v))

        # origin_tree_path widget is in _widgets (keyed by field name) but not a dataclass field
        # handled separately here:
        path_w = self._widgets.get("origin_tree_path")
        if isinstance(path_w, QLineEdit):
            path_w.setText(c.origin_tree_path or "")

        # strategies → JSON
        self._strategies_edit.setPlainText(json.dumps([dataclasses.asdict(s) for s in c.strategies], indent=2))
        # operator panel
        self._operator_panel.set_from_config(c)

    @staticmethod
    def _format_value(name: str, value: Any) -> str:
        if value is None:
            return ""
        if name == "clip_range" and isinstance(value, (tuple, list)):
            return ",".join(str(v) for v in value)
        if name == "symbols" and isinstance(value, list):
            return ",".join(value)
        return str(value)

    def _collect_from_form(self) -> Tuple[Dict[str, Any], List[str]]:
        """Read widget state into a dict; return (values, errors)."""
        errors: List[str] = []
        out: Dict[str, Any] = {}
        c = self._config
        for f in dataclasses.fields(c):
            if f.name in ("strategies", "operator_weights", "symbols", "allow_chain", "origin_tree_path", "preset"):
                continue
            w = self._widgets.get(f.name)
            if w is None:
                continue
            try:
                if isinstance(w, QCheckBox):
                    out[f.name] = w.isChecked()
                elif isinstance(w, QSpinBox):
                    out[f.name] = w.value()
                elif isinstance(w, QComboBox):
                    out[f.name] = w.currentText()
                elif isinstance(w, QLineEdit):
                    out[f.name] = self._parse_value(f.name, w.text())
            except ValueError as exc:
                errors.append(f"{f.name}: {exc}")

        # origin_tree_path
        path_w = self._widgets.get("origin_tree_path")
        if isinstance(path_w, QLineEdit):
            out["origin_tree_path"] = path_w.text().strip() or None

        # operator panel: operator_weights, symbols, allow_chain
        out["operator_weights"] = self._operator_panel.get_operator_weights()
        out["symbols"] = self._operator_panel.get_symbols()
        out["allow_chain"] = self._operator_panel.get_allow_chain()

        # strategies
        try:
            raw = self._strategies_edit.toPlainText().strip() or "[]"
            data = json.loads(raw)
            out["strategies"] = [StrategySpec(**s) for s in data]
        except Exception as exc:
            errors.append(f"strategies (JSON): {exc}")

        return out, errors

    @staticmethod
    def _parse_value(name: str, raw: str) -> Any:
        raw = raw.strip()
        if name == "df_train_csv":
            return raw or None
        if name == "clip_range":
            if not raw:
                return None
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError("expected 'min,max'")
            return (float(parts[0]), float(parts[1]))
        return raw

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_apply(self) -> None:
        new_values, errors = self._collect_from_form()
        if errors:
            QMessageBox.warning(self, "Invalid configuration", "\n".join(errors))
            return

        changes: Dict[str, Any] = {}
        for k, v in new_values.items():
            old = getattr(self._config, k, None)
            if v != old:
                changes[k] = v

        if not changes:
            QMessageBox.information(self, "Apply changes", "No changes detected.")
            return

        for k, v in changes.items():
            setattr(self._config, k, v)

        force_reload = any(k not in LIVE_EDITABLE_FIELDS for k in changes)
        self.config_changed.emit(changes, force_reload)

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "settings.json", "JSON (*.json)")
        if not path:
            return
        values, errors = self._collect_from_form()
        if errors:
            QMessageBox.warning(self, "Invalid configuration", "\n".join(errors))
            return
        cfg = RunConfig.from_dict(
            {
                **self._config.to_dict(),
                **values,
                "strategies": [dataclasses.asdict(s) for s in values.get("strategies", self._config.strategies)],
            }
        )
        cfg.save(Path(path))

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            new_cfg = RunConfig.load(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        for f in dataclasses.fields(self._config):
            setattr(self._config, f.name, getattr(new_cfg, f.name))
        self._populate_from_config()

    def _reset_default_strategies(self) -> None:
        self._strategies_edit.setPlainText(json.dumps([dataclasses.asdict(s) for s in DEFAULT_STRATEGIES], indent=2))
