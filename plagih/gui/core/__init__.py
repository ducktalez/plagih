"""Transport-neutral GUI core: events, config schema, run controller."""

from plagih.gui.core.config_schema import (
    DEFAULT_STRATEGIES,
    LIVE_EDITABLE_FIELDS,
    RunConfig,
    StrategySpec,
)
from plagih.gui.core.events import EventBus, EventType, RunEvent, RunState
from plagih.gui.core.run_controller import RunController

__all__ = [
    "DEFAULT_STRATEGIES",
    "LIVE_EDITABLE_FIELDS",
    "EventBus",
    "EventType",
    "RunConfig",
    "RunController",
    "RunEvent",
    "RunState",
    "StrategySpec",
]
