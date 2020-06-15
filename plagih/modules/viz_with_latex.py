"""
Visualising Trees with latex.
"""
# from plagih.modules.operators import op
from plagih.modules.plagih_tree import *

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
    sfeh: would be nice to show dimension.difference plots, maybe? (currently: no.)
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



##########
# Latex tree visualisation
##########


def latex_tree_get_vistree(tree):
    """
    reduce
    # todo idee: alle teil-terme, die eine einzige variable beinhalten?
    """

    node_dict = dict()  # key: node_id, value: number of nodes to paste
    tree_ids = list(tree_iterate_range(tree))

    # before calculating more, shorten floats to a 3-decimal form
    for node_id in tree_ids:
        if tree_node_is_numeric_constant(tree, node_id):
            label = float(tree_node_get_label(tree, node_id))

            if label % 1 > 0.001:
                label = '{:0.3f}'.format(label)
            else:
                label = '{}'.format(int(label))
            tree = tree_node_set_label(tree, node_id, label)

    open_sym = []  # the nodes where the expression can be sympified
    open_fix = tree_ids[:]
    while open_fix:
        node_id = open_fix[-1]
        if visualize_tree_node_force_show(tree, node_id):  # sfeh a < 5 can actually be shown. just the parent needs a split
            parents = tree_node_get_parents(tree, node_id)
            for x in parents:
                if x in open_fix:
                    node_dict[x] = 1
                    open_fix.remove(x)
        else:
            open_sym.append(node_id)
            open_fix.remove(node_id)

    open_sym = sorted(open_sym)
    while open_sym:
        node_id = open_sym[0]
        branch_ids = tree_node_get_branch(tree, node_id)
        node_dict[node_id] = len(branch_ids)
        for x in branch_ids:
            open_sym.remove(x)

    # Building the new tree
    vis_label_list = []
    vis_arity_list = []
    vis_xtype_list = []
    vis_modify_list = []

    # tex_replace = latex_get_replace_tupels()  # sfeh quick code

    for node_id in tree_ids:
        if node_id in node_dict:
            arity = 0
            if node_dict[node_id] == 1:
                arity = tree_node_get_arity(tree, node_id)
                label = tree_node_get_label(tree, node_id)
                # label = latex_string_replace(label, tex_replace)
                vis_label_list.append(label)
            elif node_dict[node_id] > 1:
                expr_raw = tree_get_expr_raw(tree, node_id)
                label = expr_sympify(expr_raw)
                # label = latex_string_replace(label, tex_replace)
                vis_label_list.append(label)

            vis_arity_list.append(arity)

            xtype = tree_node_get_xtype(tree, node_id)
            vis_xtype_list.append(xtype[-2:])

            modify = tree_node_get_modify(tree, node_id)
            vis_modify_list.append(modify)

    longest_label = 10
    for label in vis_label_list:
        longest_label = max(longest_label, len(label))

    vis_tree = latex_vistree_from_labellist(vis_label_list, vis_xtype_list, modify_list=vis_modify_list, arity_list=vis_arity_list, force_np_size=longest_label)
    return vis_tree


def latex_tree_get_forest(tree):
    """
    Creates forest tree representation (based on tikz) for LaTeX.
    The file can easily ne included in a .tex file with '\input{file_name}'
    optional: stand_alone = True for a complete latex file
    """

    bracket_tree = latex_tree_node_get_forest(tree)
    forest_viz = latex_wrap_forest(bracket_tree)

    return forest_viz


def latex_vistree_from_labellist(label_list, xtype_list, modify_list=None, arity_list=None, force_np_size=None):
    """
    returns: tree, from label_list (newest version)
    """

    if force_np_size:
        np_dtype_size = 'U' + str(force_np_size)  # todo sfeh
    else:
        np_dtype_size = None

    if not arity_list:
        arity_list = [label_get_arity(label) for label in label_list]  # ~- problem: fine. [-, 1, 2] vs [*, 1, -2]
    core = core_from_labels(label_list, arity_list, xtype_list, force_np_dtype=np_dtype_size)
    if modify_list:
        for i, val in enumerate(modify_list):
            core[N_modify][i] = val
    else:  # all can be modified
        for i, val in enumerate(label_list):
            core[N_modify][i] = 1
    tree = tree_convert_pcore_to_karoo(core)
    return tree


def latex_get_replace_tupels():
    label_string_replace = {}
    for key, value in op.items():
        if isinstance(key, str):
            latex_replace = value['latex1']
            if latex_replace is not None:
                label_string_replace[key] = latex_replace
    return label_string_replace
