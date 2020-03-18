"""
Visualising Trees with latex.
"""

tree_sep = ''  # ''\\newpage'


def latex_get_sepline(parsim, fitness, tikz_code, tree_sep):
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
    latex_tikz_forest = '\n\\begin{{forest}}' \
                        '\n  for tree={{symbol, rounded corners,draw=black!100, fill=green!20}}' \
                        '\n  point/.style={{coordinate,}},' \
                        '\n  symbol/.style={{draw=black,text height=1.5ex,text depth=.25ex,}},' \
                        '\n  terminal/.style={{symbol,}},' \
                        '\n  nonterminal/.style={{rectangle, symbol, rounded corners,fill=green!20}},' \
                        '\n  operation/.style={{symbol, rounded rectangle,}},' \
                        '\n  fixnode/.style={{draw=black!100, fill=red!20,}},' \
                        '\n  terminal/.style={{rectangle, symbol,draw=black!100, fill=green!20,}},' \
                        '\n  variable/.style={{rounded corners, symbol,draw=black!100, fill=green!20,}},' \
                        '\n  constant/.style={{rectangle, symbol,}},' \
                        '\n {}' \
                        '\n\\end{{forest}}\n'.format(tikz_forest_tree)
    return latex_tikz_forest
