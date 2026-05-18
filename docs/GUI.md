# plagih Desktop GUI

A PySide6 desktop application for configuring, controlling and monitoring
GP runs.  Lives in [`plagih/gui/`](../plagih/gui/) and is split into a
transport-neutral **core** (`plagih.gui.core`) and a Qt **desktop**
frontend (`plagih.gui.desktop`).

## Install & launch

```bash
pip install -e .[gui]      # adds PySide6
plagih-gui                 # or: python -m plagih.gui.desktop
```

The window opens with default settings — pick a training CSV in the
**Configuration** panel, then press **Start**.

## Layout

| Area                | Contents                                                                 |
|---------------------|--------------------------------------------------------------------------|
| Left dock           | `ControlPanel` (Start/Pause/Resume/Stop/Backup) + `ConfigPanel`          |
| Centre (tabs)       | Charts • Pareto • Best candidate • Log & metrics                         |

The dock layout is persisted between sessions via `QSettings`.

## Control flow

The GUI never blocks on GP work.  Everything runs in a background thread
managed by [`RunController`](../plagih/gui/core/run_controller.py):

```
GUI thread          ──┐                              ┌── GP worker thread
  ConfigPanel ─────►  │                              │
  ControlPanel ────►  │      apply_changes()         │
  ParetoPanel ◄────┐  ├──────────────────────────────►│  ExplainableGP.run_generation(...)
  ChartsPanel ◄────┤  │      RunController            │  GPMonitor callbacks
  BestPanel  ◄─────┤  │      ─ pause_event             │       │
  LogPanel   ◄─────┘  │      ─ stop_event              │       ▼
                       │      ─ pending_changes queue   │  EventBus.emit(RunEvent)
                       │                              │       │
                       │  QtEventRelay (queued sig)   │       │
                       └──────────────────────────────└───────┘
```

## Configuration

Every field in [`RunConfig`](../plagih/gui/core/config_schema.py) is
editable in the form.  Each field carries a badge:

- **`[live]`** — applied without rebuilding the GP (e.g. `gen_end`,
  `pop_max_size`, `verbosity`, strategy rates).
- **`[reload]`** — applying triggers `backup_save → recreate → backup_load`
  to keep the population + Pareto front intact across the rebuild.

The current live-editable set is defined in
[`LIVE_EDITABLE_FIELDS`](../plagih/gui/core/config_schema.py).

Settings can be saved / loaded as JSON via the buttons at the top of the
Configuration panel.

## Tabs

- **Charts** — four-panel matplotlib view (fitness, parsimony, diversity,
  time per generation).  Updates only on `GENERATION_DONE`, throttled to
  ~2 Hz, gated by the "Live aktualisieren" checkbox.
- **Pareto** — table + scatter of the current Pareto front, updated on
  `PARETO_CHANGED`.
- **Best candidate** — shows the **most recently added** Pareto entry:
  expression, fitness/parsimony, and a tree-renderer image.  The image
  is produced by a background `QRunnable` to keep the GUI responsive;
  toggle off via "Baum rendern" if not needed.
- **Log & metrics** — per-generation metric table (top) + log stream
  (bottom).

## Pause semantics

Pause takes effect *between* generations — the current generation always
finishes first.  Finer-grained pause is tracked as item I15.5 in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Browser frontend?

Out of scope for the initial implementation.  The
`plagih.gui.core.RunController` is intentionally transport-neutral so a
future FastAPI/WebSocket adapter can drive a browser UI alongside the
desktop one.  Tracked as I15.6.

## Testing

GUI tests live in two files:

- `plagih/test/test_gui_core.py` — Qt-free, always runs.  Covers
  `RunConfig` ↔ JSON, `EventBus`, `LIVE_EDITABLE_FIELDS`, and
  `RunController` lifecycle without a real GP.
- `plagih/test/test_gui_desktop.py` — uses PySide6.  Skipped if PySide6
  is not installed.  Runs against `QT_QPA_PLATFORM=offscreen`, set
  automatically by `plagih/test/conftest.py`.

Run them with:

```bash
pytest plagih/test/test_gui_core.py plagih/test/test_gui_desktop.py -v
```

When you add a new config field or monitor metric, please re-run these
to make sure the GUI still binds them correctly (see "GUI compatibility"
in `.github/copilot-instructions.md`).

