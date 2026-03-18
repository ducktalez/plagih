"""
LaTeX tree export (forest/expression format).

Rendering data comes from node class attributes:
- ``latex_fmt``    → format string for special layout (e.g. Pow → ``{{}^{}}``)
- ``latex_inline`` → separator for infix operators  (e.g. Add → ``+``)
Nodes without either attribute fall back to function-style: ``label(children)``.
"""

import re


def latex_treeviz_full_document(tex_input, doc_border=",border=5pt"):
    if isinstance(tex_input, list):
        tex_body = "\n\\newpage\n".join(tex_input)
    else:
        tex_body = tex_input

    latex_newcommand_forest = r"""
\newcommand{\plforest}[1]{
  \begin{forest}
    for tree={child anchor=north, rounded corners, align=center, draw=black!100, fill=blue!20},
    terminal/.style={rectangle,},
    fixnode/.style={fill=blue!60,},
    observation/.style={rectangle,},
    variable/.style={rectangle,},
    samenode/.style={fill=blue!40,},
    newnode/.style={fill=green!60,},
    changenode/.style={orange=green!60,},
    nodeinsert/.style={fill=green!50,},
    nodechanged/.style={fill=orange!50,},
    #1
  \end{forest}
}"""

    latex_doc_forest = rf"""
\documentclass[varwidth=\maxdimen{doc_border}]{{standalone}}
\usepackage{{forest}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{amsmath}}
{latex_newcommand_forest}
\begin{{document}}
{tex_body}
\end{{document}}
"""
    return latex_doc_forest


def _tex_trunc_digits(label: str) -> str:
    """Truncate overly precise decimals for LaTeX display (e.g. 1.23456 → 1.234)."""
    label = re.sub(r"0\.000[0]+[1-9]+", "0.001", label)
    label = re.sub(r"(?<=[0-9]\.[0-9]{3})(\d+)", "", label)
    return label


def get_expr_latex(node, klammern=False):
    """Return a compact LaTeX math expression for *node*, recursively.

    Uses ``node.latex_fmt`` / ``node.latex_inline`` when present,
    otherwise falls back to ``showme(children)`` function-style.
    """
    if node.is_term():
        return _tex_trunc_digits(str(node.get_childs()[0]))

    children = [get_expr_latex(c, klammern=True) for c in node.get_childs()]

    # Special format (Pow → {x}^{y}, Abs → |x|, Sqrt → √x, Min/Max → join children)
    fmt = getattr(node, "latex_fmt", None)
    if fmt:
        if fmt.count("{}") == 1 and len(children) > 1:
            return fmt.format(", ".join(children))
        return fmt.format(*children)

    # Inline operator (Add -> +, Mul -> \cdot, Or -> \vee, ...)
    sep = getattr(node, "latex_inline", None)
    if sep:
        expr = sep.join(children)
        return f"({expr})" if klammern else expr

    # Fallback: function-style
    label = getattr(node, "showme", type(node).__name__)
    return f"{label}({', '.join(children)})"


def latex_brackettree(node, force_node=True):
    """Return a recursive LaTeX forest string ``[Label [Child1] [Child2] ...]``.

    *force_node=True*  → every node gets its own forest bracket.
    *force_node=False* → the whole tree is rendered as a single expression node.
    """
    if not force_node:
        return f"[$ {get_expr_latex(node)} $]"

    label = f"${get_expr_latex(node)}$"
    if not node.has_childs():
        return f"[{label}]"
    children_str = "".join(latex_brackettree(c, force_node=True) for c in node.get_childs())
    return f"[{label}{children_str}]"


if __name__ == "__main__":
    from plagih.trees import plagih_sympify, sympy_to_tree

    syex = plagih_sympify("a + 2.3")
    t = sympy_to_tree(syex, allow_chain=True)
    print(rf"\plforest{{{latex_brackettree(t, force_node=False)}}}")
    print(rf"\plforest{{{latex_brackettree(t, force_node=True)}}}")
