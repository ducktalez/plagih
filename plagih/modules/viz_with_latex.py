"""
Visualising Trees with latex.
"""
# from plagih.modules.operators import op
from plagih.modules.plagih_tree import *
import re

tree_sep = ''  # ''\\newpage'


def latex_get_forest_title(parsim, fitness, tikz_code, tree_sep):
    return


def latex_treeviz_full(tikz_forest_list, preamble=''):
    """
    Latex standalone document of forest trees.
    Possible \documentclass options:
    [varwidth=\\maxdimen,convert,border=5pt]{{standalone}}  # -> newpage does not exist
    {{article}}     # -> tree_sep should be \newpage
    {{beamer}}      # -> tree_sep should be \newpage
    sfeh: would be nice to show dimension.difference plots, maybe? (currently: no.)
    """

    forest_trees = '\n'.join(tikz_forest_list)

    latex_doc_forest = '\\documentclass[varwidth=\\maxdimen,convert,border=5pt]{{standalone}}' \
                       '\n\\usepackage{{forest}}' \
                       '\n\\begin{{document}}' \
                       '\n{}' \
                       '\n\\end{{document}}'.format(forest_trees)
    return latex_doc_forest


##########
# Latex tree visualisation
##########

def tree_viz_get_nel(tree):
    """
    Deprecated.
    Returns nodes, edges and labels to visualize a tree with NetworkX or pygraphviz. Similar to deap gp visualisation.
    Deprecated? -> Used for NetworkX- which is not used as pygrapviz could not be installed on windows. Latex is used now.
    E. g. the tree with labels [+, 1, 2]
    -> node_list = [1, 2, 3], edge_list = [[1, 2],[1, 3]], label_list = [x, 1, 2]
    """
    # iteratte over all nodes
    # save nodes in list, all edges in list
    node_list, edge_list, label_list = [], [], []
    for node_id in tree_nodes_get_ids(tree):
        node_list.append(node_id)  # node id
        for child_id in tree_node_get_childs(tree, node_id):
            edge_list.append([node_id, child_id])
        label_list.append(tree_node_get_label(tree, node_id))
    return node_list, edge_list, label_list


def latex_tree_get_brackets(tree, node_id=root_id):
    """
    creates a tex file with a tikz figure of a tree.

    Labeling edges: , edge label = {node[midway, font =\scriptsize]{If...}}
    """
    extras = ''
    label, arity, xtype = tree_node_get_lax_v3(tree, node_id)

    # Get the best math-like representation for functions
    if label in op:
        bracket_string = op[label]['latex1']
    else:
        bracket_string = '${{{}}}$'.format(label)

    # todo float labels too long
    # todo underline makes lower indices... good or bad?

    # custom node design
    if arity == 0:
        extras += ',terminal'

        if tree_node_is_variable(tree, node_id):
            extras += ',variable'
        else:
            extras += ',constant'

    if not tree_node_is_modifiable(tree, node_id):
        extras += ',fixnode'

    label += extras

    child_ids = tree_node_get_childs(tree, node_id)
    for child_id in child_ids:
        label += (latex_tree_get_brackets(tree, child_id))
    else:
        bracket_string = '[{}]'.format(label)

    return bracket_string


def visualize_tree_node_force_show(tree, node_id):
    """
    Check if a node must be displayed as full node
    changes_xtypes or complex_label or close_to_root or fix_node
    - if it changes datatypes
    """

    modifiable = tree_node_get_modify(tree, node_id)
    fix_node = True if modifiable == 0 else False  # show (at least) the root node as tree?
    if fix_node:
        return True

    label = tree_node_get_label(tree, node_id)
    complex_label = label in ['Ifte', 'Maxi', 'Mini']  # ideas: min, max, if, abs (sfeh: or let the user specify)
    if complex_label:
        return True

    xtype = tree_node_get_xtype(tree, node_id)
    changes_xtypes = 'b' in xtype and 'f' in xtype  # Attention: (xtype in ['f2b', 'b2f', 'b2f2f']) is slower
    if changes_xtypes:
        return True

    depth = tree_node_get_depth(tree, node_id)
    close_to_root = depth < 0  # show (at least) the root node as tree?
    if close_to_root:
        return True

    return False


def tree_node_is_numeric_constant(tree, node_id):
    """
    returns if the label is float/int constant (aka numeric value)
    """
    if tree_node_get_xtype(tree, node_id) == '2f':
        try:
            label = float(tree_node_get_label(tree, node_id))
            return True
        except:
            pass

    return False


# def latex_tree_shorten_labels(tree):
#     """
#     replace
#     """
#     # First, shorten floats to a 3-decimal form
#     tree_ids = list(tree_iterate_range(tree))
#
#     for node_id in tree_ids:
#         if tree_node_is_numeric_constant(tree, node_id):
#             label = float(tree_node_get_label(tree, node_id))
#             if label == 0:  # e.g. '0'
#                 pass
#             elif label % 1 != 0:  # e.g. '2.0442'
#                 label = '{:0.3f}'.format(label)
#             else:  # e.g. 2
#                 label = '{}'.format(int(label))
#             tree = tree_node_set_label(tree, node_id, label)
#     return tree

def label_tex_replace_all(label, tex_replace):
    return label


def label_tex_replace_digits(label):
    # 1.23456 + sdf -> 1.234 + sdf (remove +3 digits with regex)
    label = re.sub('0\.000[0]+[1-9]+', '0.001', label)  # displaying very small values as '0.001'
    label = re.sub('(?<=[0-9]\.[0-9]{3})(\d+)', '', label)  # removing over-accurate decimals
    return label


def label_tex_replace_opwhat(label, tex_replace):
    # replace all occurences of operations with op
    for find, replace in tex_replace.items():
        label = label.replace(find, replace)
    return label


def get_tex_replace():
    tex_replace = {}
    for key, value in op.items():
        if isinstance(key, str):
            replace = value['latex1']
            tex_replace[key] = replace

    # tex_replace.pop('*')
    # tex_replace['*'] = op_what['*']['latex1'][1]

    return tex_replace


def tree_stringreplace_labels(tree, tex_replace=None):
    """

    """

    # get all the replacements from the op-dict
    # replace all the labels with the latex1 representation

    return tree


def latex_tree_get_forest(tree, tight_viz=True):
    """
    whole procedure from tree to forest core
    """

    # tree = latex_tree_shorten_labels(tree)
    tree = tree.copy()

    # first, shorten all long labels (e.g. 2.44423443344534 -> 2.444)
    if tight_viz:
        viztree = latex_tree_get_tighttree(tree)
    else:
        tex_replace = get_tex_replace()
        for node_id in tree_iterate_range(tree):
            label = tree_node_get_label(tree, node_id)
            if label in op:
                label = op[label]['latex1']
            else:
                label = label_tex_replace_digits(label)
            label = '{{{}}}'.format(label)
            tree = tree_node_set_label(tree, node_id, label)
        viztree = tree_stringreplace_labels(tree, tex_replace=tex_replace)

    bracket_tree = latex_tree_get_brackets(viztree)

    forest_complete = '\n\\begin{{forest}}' \
                      '\n  for tree={{child anchor=north, rounded corners,align=center,draw=black!100,fill=blue!20}},' \
                      '\n  terminal/.style={{rectangle,}},' \
                      '\n  fixnode/.style={{fill=blue!60,}},' \
                      '\n  constant/.style={{rectangle,}},' \
                      '\n {}' \
                      '\n\\end{{forest}}\n'.format(bracket_tree)

    return forest_complete


def latex_tree_get_tighttree(tree):
    """
    reduce
    # todo idee: alle teil-terme, die eine einzige variable beinhalten?
    """

    node_dict = dict()  # key: node_id, value: number of nodes to paste into the viz-node
    tree_ids = list(tree_iterate_range(tree))

    # All nodes, that have to be shown -> node_dict
    # All nodes, that can be sympified -> open_sym
    # node_dict -> {1: 1, 2: 1, ...}
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

    # node_dict -> {5: 8, 6: 9, ...} (updating the above version)
    open_sym = sorted(open_sym)
    while open_sym:
        node_id = open_sym[0]
        branch_ids = tree_node_get_branch(tree, node_id)
        node_dict[node_id] = len(branch_ids)
        for x in branch_ids:
            try:
                open_sym.remove(x)
            except:
                pass  # removing an element that is not in list

    # Building the new tree
    vis_label_list = []
    vis_arity_list = []
    vis_xtype_list = []
    vis_modify_list = []

    # tex_replace = get_tex_replace()

    for node_id in tree_ids:
        if node_id in node_dict:
            arity = 0
            if node_dict[node_id] == 1:
                arity = tree_node_get_arity(tree, node_id)
                label = tree_node_get_label(tree, node_id)
            elif node_dict[node_id] > 1:
                expr_raw = tree_get_expr_raw(tree, node_id)
                label = expr_sympify(expr_raw)
                label = label_tex_replace_digits(label)
                # label = label_tex_replace_opwhat(label, tex_replace)  # todo
            else:
                raise

            vis_label_list.append(label)
            vis_arity_list.append(arity)
            vis_xtype_list.append(tree_node_get_xtype(tree, node_id)[-2:])
            vis_modify_list.append(tree_node_get_modify(tree, node_id))

    longest_label = max(10, max([len(x) for x in vis_label_list]))

    tight_tree = latex_vistree_from_labellist(vis_label_list, vis_xtype_list, modify_list=vis_modify_list, arity_list=vis_arity_list, force_np_size=longest_label)

    return tight_tree


def latex_vistree_from_labellist(label_list, xtype_list, modify_list=None, arity_list=None, force_np_size=None):
    """
    returns: tree, from label_list (newest version)
    """

    if force_np_size:
        np_dtype_size = 'U{}'.format(force_np_size)  # sfeh
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
    """
    'Mini(a, 2.3)' -> min(a, 2.3)
    """
