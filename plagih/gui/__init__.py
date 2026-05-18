"""plagih.gui — Optional desktop GUI for monitoring and controlling GP runs.

Two layers:
- ``plagih.gui.core``: transport-neutral run controller, config schema, events.
  Pure Python, no Qt dependency. Reusable for future web/CLI adapters.
- ``plagih.gui.desktop``: PySide6 desktop application built on the core.

Entry point: ``python -m plagih.gui`` or ``plagih-gui`` console script.
"""
