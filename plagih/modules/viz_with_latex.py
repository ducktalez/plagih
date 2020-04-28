"""
Visualising Trees with latex.
"""
from plagih.modules.operators import op

tree_sep = ''  # ''\\newpage'


def latex_get_forest_title(parsim, fitness, tikz_code, tree_sep):
    return 'Pareto entry at parsimony {} with fitness {}.\n{}\n{}\n'.format(parsim, fitness, tikz_code, tree_sep)


def latex_complete_tree_summary(tikz_forest_list, preamble=''):
    """
    Latex standalone document of forest trees.
    Possible \documentclass options:
    [varwidth=\\maxdimen,convert,border=5pt]{{standalone}}  # -> newpage does not exist
    {{article}}     # -> tree_sep should be \newpage
    {{beamer}}      # -> tree_sep should be \newpage
    todo schow tex plots aswell?
    """

    tikz_combined = ''

    for tikz in tikz_forest_list:
        tikz_combined += tikz

    # \documentclass[varwidth,convert]{standalone}
    latex_doc_forest = '\\documentclass[varwidth=\\maxdimen,convert,border=5pt]{{standalone}}' \
                       '\n\\usepackage{{forest}}' \
                       '\n\\begin{{document}}' \
                       '\n{}' \
                       '\n\\end{{document}}'.format(tikz_combined)
    return latex_doc_forest


def latex_wrap_forest(tikz_forest_tree):
    """
    tikz-forest wrap for a tree in latex-tikz-forest notation.
    - Wraps the 'forest' for latex
    - prepares a number of node styles
        - terminal:
            - variable
            - constant
        - non-terminal
        - fixnode: If user specified this as fix node
        - originalnode: todo, if node is the same as in origin_meta (exactly the same, changed variable?, ...)
        - point: sfeh, guess this is currently not used
    """

    # '\n  point/.style={{coordinate,}},' \
    # '\n  symbol/.style={{text height=1.5ex,text depth=.25ex,}},' \

    latex_tikz_forest = '\n\\begin{{forest}}' \
                        '\n  for tree={{rounded corners,align=center,draw=black!100,fill=blue!20}},' \
                        '\n  terminal/.style={{}},' \
                        '\n  nonterminal/.style={{rectangle}},' \
                        '\n  operation/.style={{}},' \
                        '\n  fixnode/.style={{fill=blue!60,}},' \
                        '\n  terminal/.style={{rectangle,}},' \
                        '\n  variable/.style={{rounded corners,}},' \
                        '\n  constant/.style={{rectangle,}},' \
                        '\n {}' \
                        '\n\\end{{forest}}\n'.format(tikz_forest_tree)
    return latex_tikz_forest