---
applyTo: "plagih/tree_complexity/**"
---

# Tree Complexity – Copilot Instructions

- **Architectural rule**: Keep algorithms here, not inside `Node` subclasses.
- **TED modes** (P15): `"structural"` (default for parsimony), `"full"` (diversity),
  `"structural_plus_leaf_diff"`. Use `compute_ted()` — `apted_distance()` is deprecated.
- **Bytecode complexity** (P16): CPython-based heuristic, not hardware-stable.
  Values differ across Python versions.
- **Metric axioms**: Pairwise distances should satisfy non-negativity, identity,
  symmetry, triangle inequality. Unary scores are not distances.
