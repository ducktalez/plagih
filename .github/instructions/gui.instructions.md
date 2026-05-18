---
applyTo: "plagih/gui/**"
---

# GUI – Copilot Instructions

## Layering

- `plagih/gui/core/` — pure-Python, no Qt import. Holds `RunController`,
  `RunConfig`, `EventBus`. **Never import PySide6 here.**
- `plagih/gui/desktop/` — PySide6 widgets. Lazy-import Qt at module top
  level is fine, but defer GP-heavy imports (`plagih.trees`,
  `plagih.visualization`) to function bodies if they aren't always used.

## Threading

- All GP work runs inside `RunController._run_loop` on a daemon thread.
- Qt widgets must only be touched from the GUI thread — use
  `QtEventRelay` (a `QObject` with a signal) to bridge `EventBus`
  callbacks back onto the GUI thread.

## Live vs reload-required fields

- The single source of truth is `LIVE_EDITABLE_FIELDS` in
  `core/config_schema.py`. Adding a new field to `RunConfig`?
  Update that set if the field can be applied without rebuilding the GP.
- `_apply_live_changes()` must handle every key in `LIVE_EDITABLE_FIELDS`.

## Tree rendering

- Always render via `plagih.visualization.tree_renderer.render_tree`
  in a `QRunnable` on `QThreadPool.globalInstance()`. Never in the GUI
  thread — it can take seconds for large trees.
- Use the coalescing pattern from `BestCandidatePanel`: at most one job
  in flight, at most one queued — drop intermediates.

## Backwards compatibility

- The GUI is opt-in. The `plagih.gui` package must **never** be imported
  by the core framework (`plagih.trees`, `plagih.parallel`, …).

