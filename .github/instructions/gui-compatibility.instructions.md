---
applyTo: "plagih/config.py,plagih/monitoring.py,plagih/trees/_gp_engine.py"
---

# GUI Compatibility – Copilot Instructions

The desktop GUI in `plagih/gui/` consumes these modules.  When you edit
them, keep them GUI-compatible:

1. **New `PlagihConfig` field or `ExplainableGP.create()` argument** that
   influences a run → mirror it in `RunConfig`
   (`plagih/gui/core/config_schema.py`) and tag it as either
   - *live-editable*: add to `LIVE_EDITABLE_FIELDS` **and** extend
     `RunController._apply_live_changes`, or
   - *reload-required*: no extra wiring needed (rebuild handles it).
2. **New `GPMonitor` metric** appears in the GUI automatically.
   If you **rename** a metric, update `_BACKWARDS_COMPAT_RENAME` in
   `monitoring.py` so older backups stay loadable.
3. **Dependency direction is one-way**: the GP engine must never import
   from `plagih.gui`. The GUI depends on the engine, never the reverse.

After such changes, run `pytest plagih/test/test_gui_*.py`.

