---
applyTo: "plagih/evaluation_context.py"
---

# Evaluation Context – Copilot Instructions

- **Optional & backward-compatible**: This system wraps the three existing
  evaluation methods on `Node` — it does **not** replace them.
- **Three modes**: `sympy` (slow, symbolic), `numpy_eager` (fast, debug-friendly),
  `numpy_lambda` (reusable graph-based).
- **LUT caching**: When `use_lut=True`, results are cached by tree identity.
  Cache key is `tree.get_lut_id()` — never `get_sympy_expr()`.
- **Gradient tracking**: Placeholder only (`FutureWarning`). Not implemented yet.

Full docs: `docs/EVALUATION.md`

