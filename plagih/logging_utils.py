"""
Logging and console output for the plagih GP framework.

Provides a unified logging system that combines Python's ``logging`` module
with verbosity-gated, colored console output.

**Quick start**::

    from plagih.logging_utils import setup_logging, log, log_info, log_warning

    setup_logging(log_file=Path("./logs/run.log"))
    log("gg", "Generation 5 done")  # verbosity-gated, colored
    log_info("Pareto front: %d candidates", 12)  # standard logging

**Verbosity system** (P5):

The ``log()`` function checks ``msg_type in cfg.verbosity`` before emitting.
``cfg.verbosity`` defaults to ``"wwaaggiiffpp"``; substring membership controls
which message types are printed.  See ``docs/ARCHITECTURE.md`` §6 for all keys.

**Migration from legacy functions**:

``printpl``, ``printez``, ``print_warning``, ``print_caution`` are re-exported
as thin aliases for backward compatibility but new code should use ``log()``,
``log_info()``, ``log_warning()``, ``log_error()``.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logger singleton
# ---------------------------------------------------------------------------
logger = logging.getLogger("plagih")


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
class _Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    WARNING = "\033[93m"  # yellow
    RED = "\033[91m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class _ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter that respects the ``print_type`` attribute."""

    _TYPE_COLORS = {
        "i": (_Colors.CYAN, "Info"),
        "f": (_Colors.WHITE, "File"),
        "a": (_Colors.GREEN, ""),
        "g": (_Colors.MAGENTA, "[Gen]"),
        "p": (_Colors.WHITE, ""),
        "w": (_Colors.WARNING, "Warning"),
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        ptype = getattr(record, "print_type", None)
        if ptype:
            key = ptype[0]  # first char determines color family
            color, prefix = self._TYPE_COLORS.get(key, (_Colors.RESET, ""))
            label = f"{prefix}: " if prefix else ""
            return f"{color}{label}{msg}{_Colors.RESET}"
        # Standard log-level coloring
        if record.levelno >= logging.ERROR:
            return f"{_Colors.RED}ERROR: {msg}{_Colors.RESET}"
        if record.levelno >= logging.WARNING:
            return f"{_Colors.WARNING}Warning: {msg}{_Colors.RESET}"
        if record.levelno >= logging.INFO:
            return f"{_Colors.CYAN}Info: {msg}{_Colors.RESET}"
        return f"{_Colors.WHITE}Debug: {msg}{_Colors.RESET}"


class _FileFormatter(logging.Formatter):
    """Detailed file formatter without colors."""

    def __init__(self) -> None:
        super().__init__(
            "[%(asctime)s][%(levelname)-7s][%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ---------------------------------------------------------------------------
# Progress-line helpers (in-place overwrite on terminal)
# ---------------------------------------------------------------------------
_progress_line_open = False
_progress_line_len = 0
_last_progress_key = None
_last_progress_bucket = None
_last_progress_fail = None
_last_progress_emit_time = 0.0


def _write_progress_line(message: str, *, close: bool = False) -> None:
    """Write or update the in-place progress line.

    Args:
        message: Text to display.
        close: If True, terminate the line with ``\n`` and mark it closed.
    """
    global _progress_line_open, _progress_line_len

    pad = max(0, _progress_line_len - len(message))
    ending = "\n" if close else ""
    sys.stdout.write(f"\r{message}{' ' * pad}{ending}")
    sys.stdout.flush()
    _progress_line_open = not close
    _progress_line_len = 0 if close else len(message)


def _reset_progress_tracking() -> None:
    """Reset throttling state for generation progress updates."""
    global _last_progress_key, _last_progress_bucket, _last_progress_fail, _last_progress_emit_time
    _last_progress_key = None
    _last_progress_bucket = None
    _last_progress_fail = None
    _last_progress_emit_time = 0.0


def _flush_progress_line() -> None:
    """If a progress start-line is open, move to a new line first."""
    global _progress_line_open, _progress_line_len
    if _progress_line_open:
        sys.stdout.write("\n")
        sys.stdout.flush()
        _progress_line_open = False
        _progress_line_len = 0


def print_generation_start(gen_id: int, gen_end: int) -> None:
    """Print an in-place progress line overwritten by ``print_generation_done``."""
    from plagih.config import cfg as _cfg

    if "gg" not in _cfg.verbosity:
        return
    _reset_progress_tracking()
    ts = time.strftime("%H:%M:%S", time.localtime())
    _write_progress_line(f"[{ts}] generation {gen_id}/{gen_end} start")


def print_generation_progress(
    gen_id: int,
    gen_end: int,
    created: int,
    total: int,
    *,
    label: Optional[str] = None,
    fail: int = 0,
    elapsed_s: Optional[float] = None,
) -> None:
    """Update the in-place generation progress bar.

    Args:
        gen_id: Current generation id.
        gen_end: Final generation id.
        created: Number of successfully created candidates so far.
        total: Target number of candidates for this step/generation.
        label: Optional short phase or strategy label.
        fail: Number of failed creation attempts/tasks so far.
        elapsed_s: Optional elapsed wall-clock time in seconds.
    """
    from plagih.config import cfg as _cfg

    if "gg" not in _cfg.verbosity:
        return

    global _last_progress_key, _last_progress_bucket, _last_progress_fail, _last_progress_emit_time

    total_display = max(0, total)
    created_display = max(0, created)
    if total_display > 0:
        created_display = min(created_display, total_display)
        ratio = created_display / total_display
        percent = round(ratio * 100)
    else:
        ratio = 1.0 if created_display > 0 else 0.0
        percent = 100 if created_display > 0 else 0

    bar_width = 24
    filled = max(0, min(bar_width, round(ratio * bar_width)))
    bar = "#" * filled + "-" * (bar_width - filled)
    progress_key = (gen_id, gen_end, label, total_display)
    progress_bucket = 10 if total_display == 0 and created_display > 0 else percent // 10
    now = time.perf_counter()
    should_emit = False
    is_boundary_update = (created_display == 0) or (total_display > 0 and created_display == total_display)

    if (
        progress_key != _last_progress_key
        or is_boundary_update
        or fail != _last_progress_fail
        or progress_bucket != _last_progress_bucket
        or (elapsed_s is not None and now - _last_progress_emit_time >= 2.0)
    ):
        should_emit = True

    if not should_emit:
        return

    ts = time.strftime("%H:%M:%S", time.localtime())
    label_part = f" {label}" if label else ""
    msg = f"[{ts}] generation {gen_id}/{gen_end}{label_part} [{bar}] {created_display}/{total_display} ({percent:3d}%)"
    if fail:
        msg += f" | fail={fail}"
    if elapsed_s is not None:
        msg += f" | {elapsed_s:.1f}s"

    _last_progress_key = progress_key
    _last_progress_bucket = progress_bucket
    _last_progress_fail = fail
    _last_progress_emit_time = now
    _write_progress_line(msg)


def print_generation_done(
    gen_id: int,
    gen_end: int,
    time_ms: float,
    created: int,
    pareto_pre: int,
    ok: int,
    fail: int,
    tracker_total_ms: float,
) -> None:
    """Overwrite the start-line with the final summary."""
    from plagih.config import cfg as _cfg

    if "gg" not in _cfg.verbosity:
        global _progress_line_open, _progress_line_len
        _progress_line_open = False
        _progress_line_len = 0
        _reset_progress_tracking()
        return
    ts = time.strftime("%H:%M:%S", time.localtime())
    msg = (
        f"[{ts}] generation {gen_id}/{gen_end} done: {time_ms:.1f}ms"
        f" | created={created} | pareto_pre={pareto_pre}"
        f" | ok={ok}, fail={fail}, tracker_total={tracker_total_ms:.1f}ms"
    )
    _reset_progress_tracking()
    _write_progress_line(msg, close=True)


# ---------------------------------------------------------------------------
# Public API — setup
# ---------------------------------------------------------------------------
def setup_logging(
    log_file: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    verbose: bool = False,
) -> None:
    """Initialize the plagih logging system.  Call once at script start.

    Args:
        log_file: Optional path to a log file.
        console_level: Logging level for console output.
        file_level: Logging level for file output.
        verbose: If *True*, show DEBUG messages on the console.
    """
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level if not verbose else logging.DEBUG)
    console.setFormatter(_ColoredConsoleFormatter())
    logger.addHandler(console)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(_FileFormatter())
        logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Public API — verbosity-gated output  (replaces printpl / printez)
# ---------------------------------------------------------------------------
#  Level mapping: first char → log level
_LEVEL_FOR_CHAR = {"w": logging.WARNING, "e": logging.ERROR}


def log(msg_type: str, message: str) -> None:
    """Verbosity-gated, colored log message.

    Checks ``msg_type in cfg.verbosity`` before emitting.  The first
    character of *msg_type* determines the log level (``w`` → WARNING,
    everything else → INFO).

    This is the **single replacement** for the legacy ``printpl``,
    ``printez``, and ``print_warning`` functions.

    Args:
        msg_type: Verbosity key (e.g. ``"gg"``, ``"i"``, ``"w"``).
        message: The message string.
    """
    from plagih.config import cfg as _cfg

    if msg_type not in _cfg.verbosity:
        return
    _flush_progress_line()
    level = _LEVEL_FOR_CHAR.get(msg_type[0], logging.INFO)
    record = logging.LogRecord(
        name=logger.name,
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.print_type = msg_type  # type: ignore[attr-defined]
    logger.handle(record)


# ---------------------------------------------------------------------------
# Public API — standard Python logging wrappers
# ---------------------------------------------------------------------------
def log_debug(msg: str, *args, **kwargs) -> None:
    """Log a DEBUG message (file only by default)."""
    _flush_progress_line()
    logger.debug(msg, *args, **kwargs)


def log_info(msg: str, *args, **kwargs) -> None:
    """Log an INFO message."""
    _flush_progress_line()
    logger.info(msg, *args, **kwargs)


def log_warning(msg: str, *args, **kwargs) -> None:
    """Log a WARNING message."""
    _flush_progress_line()
    logger.warning(msg, *args, **kwargs)


def log_error(msg: str, *args, **kwargs) -> None:
    """Log an ERROR message."""
    _flush_progress_line()
    logger.error(msg, *args, **kwargs)


# ---------------------------------------------------------------------------
# Legacy aliases (backward compatibility)
# ---------------------------------------------------------------------------
def printpl(msg_type: str, message_str: str) -> None:
    """**Deprecated** — use ``log(msg_type, message_str)``."""
    log(msg_type, message_str)


def printez(message_type: str, text: str) -> None:
    """**Deprecated** — use ``log(message_type, text)``."""
    log(message_type, text)


def print_warning(msg_type: str, text: str) -> None:
    """**Deprecated** — use ``log("w", text)``."""
    log(msg_type, f"({msg_type}) {text}")


def print_caution(txt: str) -> None:
    """**Deprecated** — use ``log_error(text)``."""
    logger.error(f"CAUTION! {txt}")
