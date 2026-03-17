---
applyTo: "plagih/config.py"
---

# Config – Copilot Instructions

## PlagihConfig singleton

All settings in `PlagihConfig` (`cfg`). Override hierarchy:
`.env` → environment variables → code-level parameters.

## Verbosity (P5)

Uses **substring membership** on `cfg.verbosity` (default `wwaaggiiffpp`).
`printpl()` is **not lazy** — guard expensive f-strings:
```python
if "gggg" in cfg.verbosity:
    printpl("gggg", f"...{tree.str_as_expr()}...")
```
New code should use `cfg.verbosity` directly, not `util.PRINT_DUMMY`.

Full `.env` key reference: `docs/ARCHITECTURE.md` §6.
