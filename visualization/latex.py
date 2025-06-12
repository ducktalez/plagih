"""
Visualising Trees with latex.
"""
from plagih.sympy_extras import plagih_sympify
from plagih.trees import *
import re


def latex_treeviz_full_document(tex_input, doc_border=',border=5pt'):
    if isinstance(tex_input, list):
        tex_body = '\n\\newpage\n'.join(tex_input)
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



def label_tex_replace_digits(label):
    # 1.23456 + sdf -> 1.234 + sdf (remove +3 digits with regex)
    label = re.sub(r'0\.000[0]+[1-9]+', '0.001', label)  # displaying very small values as '0.001'
    label = re.sub(r'(?<=[0-9]\.[0-9]{3})(\d+)', '', label)  # removing over-accurate decimals
    return label


def label_tex_replace_digits(label):
    label = re.sub(r'0\.000[0]+[1-9]+', '0.001', label)
    label = re.sub(r'(?<=[0-9]\.[0-9]{3})(\d+)', '', label)
    return label


latex_custom_inline = {
    Mul: r'\cdot',
    Or: r'\vee',
    And: r'\wedge',
    Xor: r'\oplus',
    Add: r'+',
    Sub: r'-',
    Div: r'/',
}

latex_custom = {
    Pow: r'{{{}}}^{{{}}}',
    Abs: r'\left|{} \right|',
    Sqrt: r'\sqrt{{{}}}',
    Max: r'\max\left({}\right)',
    Min: r'\min\left({}\right)',
}

def get_expr_latex(node, klammern=False):
    """
    Gibt einen LaTeX-Ausdruck im kompakten, formatierten Stil zurück.
    """

    # Terminale Zahl oder Symbol
    if node.is_term():
        val = str(node.get_childs()[0])
        return label_tex_replace_digits(val)

    # Kinder rekursiv
    children = [get_expr_latex(c, klammern=True) for c in node.get_childs()]

    # Spezialformate (Pow, Abs, etc.)
    for cls, fmt in latex_custom.items():
        if isinstance(node, cls):
            return fmt.format(*children)

    # Inline-Operatoren (Add, Mul, ...)
    for cls, sep in latex_custom_inline.items():
        if isinstance(node, cls):
            expr = f" {sep} ".join(children)
            return f"({expr})" if klammern else expr

    # Fallback: Funktionsstil
    label = getattr(node, 'sy_str', str(node))
    return f"{label}({', '.join(children)})"


def latex_brackettree(node, force_node=True):
    """
    Gibt einen rekursiven LaTeX-forest-String in [Label [Child1] [Child2] ...] Notation zurück.

    - Wenn force_node == True: Jeder Knoten wird rekursiv dargestellt.
    - Wenn force_node == False: Der gesamte Baum wird als ein einziger Ausdruck dargestellt.
    """

    if not force_node:
        # Kompakter Ausdruck in einem Knoten
        expr = get_expr_latex(node)
        return f"[$ {expr} $]"

    # Ansonsten rekursiv mit echten Knoten
    label = f"${get_expr_latex(node)}$"
    if not node.has_childs():
        return f"[{label}]"
    else:
        children_str = ''.join([latex_brackettree(c, force_node=True) for c in node.get_childs()])
        return f"[{label}{children_str}]"


# Beispiel
if __name__ == "__main__":
    syex = plagih_sympify('a + 2.3')
    t = sympy_to_tree(syex, allow_chain=True)
    print(fr"\plforest{{{latex_brackettree(t, force_node=False)}}}")
    print(fr"\plforest{{{latex_brackettree(t, force_node=True)}}}")
