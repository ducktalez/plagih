---
applyTo: "plagih/paretofront.py"
---

# Paretofront – Copilot Instructions

- **Dominance**: A dominates B iff `A.parsimony ≤ B.parsimony` AND
  `A.fitness ≤ B.fitness` with at least one strict inequality.
- **Algorithm**: `pareto_from_pop()` sorts by (parsimony, fitness), then
  single-pass filter. O(n²) worst case but population sizes are small.
- **Code smell**: Two similar `analyze_pareto`-like functions exist —
  consolidate when touching this module.

