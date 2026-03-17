# Logging in Plagih

Plagih uses a hybrid logging system that combines Python's `logging` module
with user-friendly console output.

The implementation lives in `plagih/logging_utils.py`. The legacy functions
are re-exported from `plagih/util.py` for backward compatibility.

## Quick Start

```python
from plagih.logging_utils import setup_logging, log, log_info
from pathlib import Path

# Call once at the beginning of your script
setup_logging(
    log_file=Path('./logs/my_run.log'),
    console_level=logging.INFO,
    verbose=False
)
```

## What gets logged?

### Console (user feedback)
- **INFO**: Important progress messages (generations, Pareto front updates)
- **WARNING**: Non-critical problems (oversized trees, SymPy errors)
- **ERROR**: Critical errors

### Log file (debug information)
- All console messages
- **DEBUG**: Detailed internal information (tree construction, evaluations)
- Timestamps, module names, function names

## Usage

### 1. Verbosity-gated output (recommended for framework code)

Use `log()` — the single replacement for the legacy `printpl`, `printez`,
and `print_warning` functions:

```python
from plagih.logging_utils import log

log("gg", "Generation 5 completed")    # gated by "gg" in cfg.verbosity
log("w", "Tree too complex")           # gated by "w", logged as WARNING
log("i", "Pareto front updated")       # gated by "i", logged as INFO
```

### 2. Standard Python logging (for framework development)

```python
from plagih.logging_utils import log_debug, log_info, log_warning

log_debug("Created tree with %d nodes", len(tree))
log_info("Pareto front: %d candidates", len(front))
log_warning("Sympy simplification failed for: %s", expr)
```

## Legacy aliases

The following functions are **deprecated** but still work via re-export:

| Legacy | Replacement |
|---|---|
| `printpl(type, msg)` | `log(type, msg)` |
| `printez(type, msg)` | `log(type, msg)` |
| `print_warning(type, msg)` | `log("w", msg)` |
| `print_caution(msg)` | `log_error(msg)` |

## Configuration

### Verbose mode
Also shows DEBUG messages in the console:

```python
setup_logging(log_file=Path('./run.log'), verbose=True)
```

### Console only (no log file)
```python
setup_logging(console_level=logging.INFO)
```

### Different log levels
```python
setup_logging(
    log_file=Path('./run.log'),
    console_level=logging.WARNING,  # Only warnings in console
    file_level=logging.DEBUG         # Everything in file
)
```

## Example log file

```
[2026-01-27 14:23:10][INFO   ][plagih] Starting test run: MTC200_RMSE_scratch
[2026-01-27 14:23:10][DEBUG  ][plagih] Options - chained_on=False, simplicate=False
[2026-01-27 14:23:11][INFO   ][plagih] Preparing to create first Generation. Gen 0.
[2026-01-27 14:23:11][DEBUG  ][plagih] Created tree with 15 nodes
[2026-01-27 14:23:12][INFO   ][plagih] Created 10/3 (10 unique) in generation 0
[2026-01-27 14:23:12][WARNING][plagih] (w) Tree too complex: 52 > 50, pruning 2 nodes
```

## Migrating existing prints

All internal `printpl()`, `printez()`, `print_warning()`, and `print_caution()`
calls have been migrated to `log()` / `log_error()` (M5 complete).
The legacy aliases still exist in `logging_utils.py` for external code.

For new code, prefer:
```python
# Instead of:
printpl('i', f'Generation {gen} completed')

# Better:
from plagih.logging_utils import log
log("i", f"Generation {gen} completed")

# Or for unconditional logging:
from plagih.logging_utils import log_info
log_info("Generation %d completed", gen)
```

## Best Practices

1. **Setup once** at the beginning of your script
2. **log_debug()** for internal details
3. **log_info()** for important milestones
4. **log_warning()** for non-critical problems
5. **printpl()** when you need both console and log output
