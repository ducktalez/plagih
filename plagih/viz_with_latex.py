"""
Visualising Trees with latex.
"""
import re

from plagih.util import file_dump, path_make_dir


def latex_treeviz_full_document(tex_input, doc_border=',border=5pt'):
    """
    Creating Latex standalone document of forest trees.
    Possible \documentclass options:
    [varwidth=\\maxdimen,convert,border=5pt]{standalone}  # -> newpage does not exist
    {article}     # -> tree_sep should be \newpage
    {beamer}      # -> tree_sep should be \newpage
    sfeh: would be nice to show dimension.difference plots, maybe? (currently: no.)
    """

    if isinstance(tex_input, list):
        tex_body = ' '.join(tex_input)  # sfeh there was a \n does this work now? color used variables (allow colormap?)
    else:
        tex_body = tex_input

    latex_newcommand_forest = '\\newcommand{\\plforest}[1]{{\\begin{forest}   ' \
                              'for fintree={child anchor=north, rounded corners,align=center,draw=black!100,fill=blue!20},   ' \
                              'terminal/.style={rectangle,},   ' \
                              'fixnode/.style={fill=blue!60,},   ' \
                              'observation/.style={rectangle,},   ' \
                              'variable/.style={rectangle,},   ' \
                              'samenode/.style={fill=blue!40,},   ' \
                              'newnode/.style={fill=green!60,},   ' \
                              'changenode/.style={orange=green!60,},   ' \
                              'nodeinsert/.style={fill=green!50,},   ' \
                              'nodechanged/.style={fill=orange!50,}, #1 ' \
                              '\\end{forest}}}'

    latex_doc_forest = f'\\documentclass[varwidth=\\maxdimen,convert{doc_border}]{{standalone}}\n' \
                       '\\usepackage{forest}\n' \
                       '\\usepackage{array}\n' \
                       '\\usepackage{longtable}\n' \
                       '\\usepackage{amsmath}\n' \
                       f'{latex_newcommand_forest}\n' \
                       '\\begin{document}\n' \
                       f'{tex_body}\n' \
                       '\\end{document}'
    return latex_doc_forest


def file_pareto_latex(self, parsim, fintree):
    """
    Generates latex-file with the computational fintree structure of all paretofront agents
    - build fintree from expression
    - fill fintree meta-data, just in case we want to visualise anything of it
    - create latex-forest representation
    """

    """
    whole procedure from fintree to forest core
    tight_viz:
        0: display every node
        1: clever tight-visualisation where possible
        2: one single mathematical expression
    """

    fintree.set_fix_nodes(self.origin)
    fintree = fintree.get_oldtree()

    pl_forest = lambda x: f'\\plforest{{{x}}}\n'

    forest_tree_full = None  # pl_forest(latex_brackettree(fintree))
    forest_tree_tight = None  # pl_forest(latex_brackettree_tight(latex_tree_semitight(fintree)))
    # sfeh workaround delete this
    tex_expr_raw = f'${fintree.export_visualization_latex()}$'  # sfeh dollars
    tex_expr_forest = pl_forest(f'[{tex_expr_raw}]')

    path_subfolder_tex = path_make_dir(self.rootdir / 'visualisation')  # sfeh running this every fintree seems dull

    """
    The following lines delete this
    """
    # sfeh
    file_dump(path_subfolder_tex / f'full_{parsim:02d}.tex', forest_tree_full, verbose='ff', print_type=self.print_type)
    file_dump(path_subfolder_tex / f'full_{parsim:02d}_tight.tex', forest_tree_tight, verbose='ff', print_type=self.print_type)
    file_dump(path_subfolder_tex / f'{parsim:02d}_input.tex', tex_expr_raw, print_type=self.print_type)
    file_dump(path_subfolder_tex / f'{parsim:02d}_input_forest.tex', tex_expr_forest, verbose='ff', print_type=self.print_type)
    file_dump(path_subfolder_tex / f'{parsim:02d}_doc.tex', latex_treeviz_full_document(forest_tree_full), verbose='ff', print_type=self.print_type)  # delete this

    return forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest


def label_tex_replace_digits(label):
    # 1.23456 + sdf -> 1.234 + sdf (remove +3 digits with regex)
    label = re.sub('0\.000[0]+[1-9]+', '0.001', label)  # displaying very small values as '0.001'; for decimals=6
    label = re.sub('(?<=[0-9]\.[0-9]{3})(\d+)', '', label)  # removing over-accurate decimals
    return label
