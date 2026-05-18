"""Event bus and event types for the GUI core.

The bus is a tiny synchronous observer that fans out :class:`RunEvent`
instances to subscribed callbacks.  Callbacks are invoked from whichever
thread emitted the event (typically the GP worker thread); Qt frontends
must re-marshal them onto the GUI thread.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List


class RunState(str, Enum):
    """Lifecycle states reported via :class:`EventType.STATE_CHANGED`."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FINISHED = "finished"
    ERROR = "error"


class EventType(str, Enum):
    """All event types the :class:`RunController` may emit."""

    STATE_CHANGED = "state_changed"
    GENERATION_DONE = "generation_done"
    PARETO_CHANGED = "pareto_changed"
    IMPROVEMENT = "improvement"
    LOG = "log"
    CONFIG_APPLIED = "config_applied"


@dataclass
class RunEvent:
    """A single event carried by the bus.

    ``payload`` is event-specific:
    - GENERATION_DONE: ``{"gen_id", "metrics": dict, "monitor_df_row": dict}``
    - PARETO_CHANGED: ``{"latest_added": Optional[Candidate], "front_size": int}``
    - IMPROVEMENT: ``{"fitness": float, "delta": float}``
    - STATE_CHANGED: ``{"state": RunState, "message": str}``
    - LOG: ``{"level": str, "message": str}``
    - CONFIG_APPLIED: ``{"changes": dict, "requires_reload": bool}``
    """

    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)


Subscriber = Callable[[RunEvent], None]


class EventBus:
    """Tiny thread-safe observer bus.  No queuing, no priorities."""

    def __init__(self) -> None:
        self._subs: List[Subscriber] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register *callback*.  Returns an unsubscribe function."""
        with self._lock:
            self._subs.append(callback)

        def _unsub() -> None:
            with self._lock:
                if callback in self._subs:
                    self._subs.remove(callback)

        return _unsub

    def emit(self, event: RunEvent) -> None:
        """Fan out *event* to all subscribers.

        Subscriber exceptions are swallowed so a buggy listener cannot
        kill the GP worker thread.
        """
        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass
