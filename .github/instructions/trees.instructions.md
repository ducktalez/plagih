---
applyTo: "plagih/trees/**"
---

# Trees – Copilot Instructions

## Critical pitfalls

1. **Pickle & back-references** (P1/P2): `parent_node`/`root_node` excluded
   from pickle. Call `repair_all()` after any deserialization (deepcopy, IPC, backup).
2. **`get_sympy_expr()` is slow** (P10): Never call in hot paths. Use `str(tree)`
   or `tree.get_lut_id()` for identification.
3. **Scale ↔ SymPy** (P13): After any SymPy round-trip, `tree_node_grouping()` must run.
4. **`canonicalize_children()` timing** (P14): Post-processing only — never in
   `set_childs()` or `__init__`.
5. **Relational on Piecewise** (P12): `Lt(Ifte(...), 0)` can hang SymPy → fail fast.
6. **`gen_create_initial()` is sequential** (P8): Intentional.

## New operator checklist

→ `docs/ARCHITECTURE.md` §7 for full steps.

