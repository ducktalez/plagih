"""Console entry-point: ``plagih-gui`` and ``python -m plagih.gui.desktop``.

Launches the PySide6 main window with a fresh :class:`RunController`.
"""

from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    # Import Qt lazily so importing ``plagih.gui`` doesn't require PySide6.
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - import-time UX
        sys.stderr.write(
            "PySide6 is required for the plagih desktop GUI.\n"
            "Install with: pip install PySide6 matplotlib\n"
            f"Original error: {exc}\n"
        )
        return 2

    from plagih.gui.core import RunConfig, RunController
    from plagih.gui.desktop.main_window import MainWindow

    app = QApplication(argv or sys.argv)
    app.setApplicationName("plagih GP")

    controller = RunController(RunConfig())
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
