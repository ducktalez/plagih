---
applyTo: "plagih/logging_utils.py"
---

# Logging Utils – Copilot Instructions

- **`log()` is the single output function**: Replaces `printpl`, `printez`,
  `print_warning`. All new code should use `log(type, msg)`.
- **Verbosity gating** (P5): `log()` checks `msg_type in cfg.verbosity`
  before emitting. **Not lazy** — guard expensive f-strings:
  ```python
  if "gggg" in cfg.verbosity:
      log("gggg", f"...{expensive_call()}...")
  ```
- **Lazy import of `cfg`**: `log()` imports `cfg` inside the function body
  to avoid circular imports (`config.py` → `logging_utils.py` → `config.py`).
- **Legacy aliases**: `printpl`, `printez`, `print_warning`, `print_caution`
  are thin wrappers around `log()` / `log_error()`. Do not add new logic to them.

Full docs: `docs/LOGGING.md`

