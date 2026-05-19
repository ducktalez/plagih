"""Console entry-point: ``plagih-gui`` and ``python -m plagih.gui.desktop``.

Launches the PySide6 main window with a fresh :class:`RunController`.
Dark mode is applied globally via the Fusion style + a dark QPalette.
"""

from __future__ import annotations

import sys
from typing import Optional


def _apply_dark_palette(app) -> None:
    """Set a dark Fusion palette on *app*."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    p.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button, QColor(58, 58, 60))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.Mid, QColor(80, 80, 80))
    p.setColor(QPalette.ColorRole.Midlight, QColor(70, 70, 72))
    p.setColor(QPalette.ColorRole.Shadow, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Dark, QColor(35, 35, 35))
    app.setPalette(p)


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
    _apply_dark_palette(app)

    controller = RunController(RunConfig())
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
