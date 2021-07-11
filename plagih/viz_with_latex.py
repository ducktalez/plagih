"""
Visualising Trees with latex.
"""
from plagih.plagih_tree import *
import re


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
                       '\\end{forest}}}' \

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



def label_tex_replace_digits(label):
    # 1.23456 + sdf -> 1.234 + sdf (remove +3 digits with regex)
    label = re.sub('0\.000[0]+[1-9]+', '0.001', label)  # displaying very small values as '0.001'
    label = re.sub('(?<=[0-9]\.[0-9]{3})(\d+)', '', label)  # removing over-accurate decimals
    return label
