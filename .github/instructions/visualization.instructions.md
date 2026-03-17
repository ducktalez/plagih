---
applyTo: "visualization/**"
---

# Visualization – Copilot Instructions

## Rendering attributes

Visual properties (`_viz_color`, `_viz_border`, `_viz_text`, `_viz_shape`,
`latex_fmt`, `latex_inline`) are **inherited** from base classes in `trees.py`.
No `isinstance` checks in renderers — override on a subclass only for distinct style.

## Modules

- `tree_renderer.py` — Matplotlib trees + merged-graph rendering
- `latex_renderer.py` — LaTeX/TikZ export
