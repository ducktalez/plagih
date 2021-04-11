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
                       'for tree={child anchor=north, rounded corners,align=center,draw=black!100,fill=blue!20},   ' \
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


def tree_viz_get_nel(tree):
    """
    Not in use.
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


def tree_node_is_numericobservation(tree, node_id):
    """
    returns if the label is float/int observation (aka numeric value)
    """
    if tree_node_get_xtype(tree, node_id) == '2f':
        try:
            label = float(tree_node_get_label(tree, node_id))
            return True
        except:
            pass

    return False


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

    return tex_replace


def label_bracket_beautification(label):
    """
    For single labels, (+, *, 1.234000000, cartVel, Fatigue_4)
    returns tex-version ($+$, $\cdot$, $1.234$, )
    """
    if label in op:  #
        label = f"${op[label]['latex1']}$"
    elif terminal_label_is_observation(label):  # node is a terminal - either observation or variable
        obs_family, obs_time, prelabel = observation_get_family_and_time(label, none_return=None)
        if obs_time is not None:
            label = f"{prelabel}{obs_family}$_{{{obs_time}}}$"
    else:
        label = f"${label_tex_replace_digits(label)}$"
    return label


def latex_brackettree(tree, node_id=root_id):
    """
    creates a tex file with a tikz figure of a tree.

    Labeling edges: , edge label = {node[midway, font =\scriptsize]{If...}}
    """
    label, arity, xtype = tree_node_get_lax_v3(tree, node_id)
    modifiable = tree_node_is_modifiable(tree, node_id)
    extras = '' if modifiable else ',fixnode'
    label_bra = label_bracket_beautification(label)
    label_bra = "{" + label_bra + "}" + extras  # works better in latex-

    # now, append the recursion
    child_ids = tree_node_get_childs(tree, node_id)
    for child_id in child_ids:
        label_bra += (latex_brackettree(tree, child_id))
    else:
        bracket_string = f'[{label_bra}]'

    return bracket_string


def latex_brackettree_tight(tree, node_id=root_id):
    """
    creates a tex file with a tikz figure of a tree.

    Labeling edges: , edge label = {node[midway, font =\\scriptsize]{If...}}
    """
    label, arity, xtype = tree_node_get_lax_v3(tree, node_id)

    child_ids = tree_node_get_childs(tree, node_id)
    for child_id in child_ids:
        label += (latex_brackettree_tight(tree, child_id))
    bracket_string = f'[{label}]'

    return bracket_string


def tex_label_beautify_end(label):

    if label in op:
        label = op[label]['latex1']

    label = f'{label}'
    return label


def latex_tight_node(tree, node_id=root_id, klammern=False):
    """
    Fit the tree in one latex expression (aka a single node)
    """
    label = tree_node_get_label(tree, node_id)

    if tree_node_get_arity(tree, node_id) >= 1:
        childs = tree_node_get_childs(tree, node_id)
        if len(childs) >= 2 and any([tree_node_get_arity(tree, cc) >= 1 in [] for cc in childs]):
            next_klammern = True
        else:
            next_klammern = False

        child_tex_list = [latex_tight_node(tree, cc, klammern=next_klammern) for cc in childs]

        return_label = f"{op[label]['latexF'].format(*child_tex_list)}"  # sfeh deleted {{}} around the
        if klammern and label in latex_inline:  # ['+', '-', '*', '**', '==', '!=', '<', '<=', '>', '>=', 'Andb', 'Orb', 'Xor']
            return_label = f'({return_label})'

    else:
        # sfehsfeh family colored? tex color?
        if terminal_label_is_observation(label):  # node is a terminal - either observation or variable
            obs_family, obs_time, prelabel = observation_get_family_and_time(label, none_return=None)
            if obs_time is not None:
                return_label = f"{prelabel}\\text{{{obs_family}}}_{obs_time}"  # workaround
            else:
                return_label = f"{prelabel}\\text{{{obs_family}}}"  # workaround
        else:
            return_label = f"{label_tex_replace_digits(label)}"

    return return_label


def latex_tree_semitight(tree):
    """
    reduce expressions of large trees where it makes sense (according to me)
    """

    node_dict = dict()  # key: node_id, value: number of nodes to paste into the viz-node
    tree_ids = list(tree_iterate_range(tree))

    # All nodes, that have to be shown -> node_dict
    # All nodes, that can be sympified -> open_sym
    # node_dict -> {1: 1, 2: 1, ...}
    open_sym = []  # nodes where the expression can be sympified
    open_fix = tree_ids[:]  # nodes that have to be displayed completely
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

    for node_id in tree_ids:
        if node_id in node_dict:
            if node_dict[node_id] == 1:  # single node
                arity = tree_node_get_arity(tree, node_id)
                label = tree_node_get_label(tree, node_id)
                label = label_bracket_beautification(label)
            elif node_dict[node_id] > 1:  # complete expression node
                label = latex_tight_node(tree, node_id=node_id)
                # label = helper_format_brackets(label)
                label = f'${label}$'
                arity = 0
            else:
                raise

            label = tex_label_beautify_end(label)

            vis_label_list.append(label)
            vis_arity_list.append(arity)
            vis_xtype_list.append(tree_node_get_xtype(tree, node_id)[-2:])
            vis_modify_list.append(tree_node_get_modify(tree, node_id))
        else:
            pass  # if a node is not in the dict, it must be part of a reducable tree
    longest_label = max(10, max([len(x) for x in vis_label_list]))

    tight_tree = latex_tight_from_labellist(vis_label_list, vis_xtype_list, modify_list=vis_modify_list, arity_list=vis_arity_list, force_np_size=longest_label)

    return tight_tree


def latex_tight_from_labellist(vizlabel_list, xtype_list, modify_list=None, arity_list=None, force_np_size=None):
    """
    returns: tree, from label_list (newest version)
    """

    if force_np_size:
        np_dtype_size = f'U{force_np_size}'  # sfeh
    else:
        np_dtype_size = None

    if not arity_list:
        arity_list = [label_get_arity(label) for label in vizlabel_list]

    core = Core_From_Labels(vizlabel_list, arity_list, xtype_list, force_np_dtype=np_dtype_size).get_uninstanced_core()

    if modify_list:
        for i, val in enumerate(modify_list):
            core[N_modify][i] = val
    else:  # all can be modified
        for i, val in enumerate(vizlabel_list):
            core[N_modify][i] = 1
    tree = tree_convert_pcore_to_karoo(core)
    return tree


if __name__ == "__main__":
    """
    test stuff
    """
    testtr = karoo_tree_from_expr('sin(4+vel)')
    viszviz = latex_tight_node(testtr)
    print(viszviz)
