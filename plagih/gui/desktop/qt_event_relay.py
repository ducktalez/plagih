"""Bridge that re-marshals :class:`RunEvent` deliveries onto the GUI thread.

Subscribers registered with :class:`RunController.bus` are normally called
from the worker thread.  Qt widgets must only be touched from the main
thread, so this helper wraps a Qt signal that fires on the GUI thread.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal

from plagih.gui.core.events import EventBus, RunEvent


class QtEventRelay(QObject):
    """Receives bus events in any thread, re-emits them as a Qt signal."""

    event = Signal(object)  # carries a RunEvent

    def __init__(self, bus: EventBus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Bus callback runs on whichever thread emitted the event.
        # Qt::QueuedConnection ensures slots run on the GUI thread.
        self._unsub = bus.subscribe(self._on_bus_event)

    def _on_bus_event(self, ev: RunEvent) -> None:
        # Emitting a signal across threads requires only that the
        # connection type be queued, which Qt does automatically when
        # sender and receiver live on different threads.
        self.event.emit(ev)

    def disconnect_bus(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None


# Re-export connection type for convenience
__all__ = ["Qt", "QtEventRelay"]
