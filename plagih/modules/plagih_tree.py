import os
import numpy as np
from plagih.modules.plagih_sympy_extras import plagih_sympify
from plagih.modules.dicts import *
from plagih.modules.plagih_types import *

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees


#
# class QwerTree:
#     def __init__(self, label, *children):
#         self.label = label
#         self.children = list(children)
#
#     def from_expr(self, expr_childs):
#         """Create tree from bracket notation
#
#         Bracket notation encodes the trees with nested parentheses, for example,
#         in tree {A{B{X}{Y}{F}}{C}} the root node has label A and two children
#         with labels B and C. Node with label B has three children with labels
#         X, Y, F.
#         """
#         # +, [1, 2]
#         for label, childs in expr_childs:
#             self.label = label
#             for child in range(0, arity):
#                 self.children.append(self.from_expr())
#
#         return


def karoo_tree_clear_meta(tree):
    tree[T_fitness][1] = ''
    tree[T_parsimony][1] = ''
    return tree


def tree_branch_labels(tree, branch):

    """
    This method prepares a stand-alone Tree as a copy of the given branch.

    """
    label_list = []
    arity_list = []
    type_list = []
    for node_id in branch:

        label_list.append(tree[N_label][node_id])
        arity_list.append(tree[N_arity][node_id])
        type_list.append(tree[N_type][node_id])

    return label_list, arity_list, type_list


def evolve_node_arity_fix(tree):
    """
    In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.

    This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and
    Crossover.

    """

    for n in range(1, len(tree[N_id])):  # increment through all nodes (exclude 0) in array 'tree'
        if len(tree[N_id]) <= 2:
            print('FUCK {}'.format(tree))
        if tree[N_type][n] == 'term':  # check for discrepency
            tree[N_arity][n] = '0'  # set arity to 0
            tree[9][n] = ''  # wipe 'node_c1'
            tree[10][n] = ''  # wipe 'node_c2'
            tree[11][n] = ''  # wipe 'node_c3'
            tree[N_modify][n] = '1'

    return tree


#
#
# def evolve_node_arity_fix(tree):
#     """
#     In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.
#
#     This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and
#     Crossover.
#
#     """
#
#     for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'
#         if len(tree[3]) <= 2:
#             print('FUCK {}'.format(tree))
#         if tree[N_type][n] == 'term':  # check for discrepency
#             tree[N_arity][n] = '0'  # set arity to 0
#             tree[9][n] = ''  # wipe 'node_c1'
#             tree[10][n] = ''  # wipe 'node_c2'
#             tree[11][n] = ''  # wipe 'node_c3'
#             tree[N_modify][n] = '1'
#
#     return tree


def tree_node_add_fromvalues(tree, node_id, node_depth,
                             node_type, node_label, node_parent, node_arity, node_c1,
                             node_c2, node_c3):
    np.append(tree,
              ['', '', '', [node_id], [node_depth], [node_type],
               [node_label], [node_parent], [node_arity], [node_c1], [node_c2], [node_c3],
               '', '', ''], 1)
    return tree


def tree_init_first_column():
    tree = np.array(
        [['tree_id'],
         ['tree_type'],
         ['tree_depth_base'],
         ['node_id'],
         ['node_depth'],
         ['node_type'],
         ['node_label'],
         ['node_parent'],
         ['node_arity'],
         ['node_c1'],
         ['node_c2'],
         ['node_c3'],
         ['fitness'],
         ['node_modify'],
         ['parsimony']])

    return tree


def tree_init_core(node_amount):
    """
    returns an empty tree with an amount of nodes, auto fills
    """
    tree = np.zeros((T_num_lines, node_amount), dtype=np.dtype('U12'))  # U12: longest is observation1
    # tree = np.concatenate((tree, empty_node_array), axis=1)

    return tree


def tree_core_depth(tree, parent_list=None):
    """
    Automatically filly node depth
    - Tree needs:
        - c1, c2, c3 filled
    """

    if not parent_list:
        parent_list = tree_row_int(tree, N_c1)

    tree[N_depth][0] = 0  # the root is always here

    for my_id, _ in enumerate(parent_list):

        child_depth = int(tree[N_depth][my_id]) + 1
        c1 = tree[N_c1][my_id]
        c2 = tree[N_c2][my_id]
        c3 = tree[N_c3][my_id]
        for c in [c1, c2, c3]:
            if c != '':
                tree[N_depth][int(c)] = child_depth
    return tree


def tree_core_parents(tree, arity_lst=None):
    """
    Automatically fill the node_parent of a tree
    - arity list in tree or
    """
    if not arity_lst:
        arity_lst = [int(x) for x in tree[N_arity]]

    parent_list = [-1]
    for i, arity in enumerate(arity_lst):
        parent_list.extend([i] * arity)
        tree[N_parent][i] = parent_list[i]
    return tree, parent_list


def tree_core_c(tree, parent_list=None):
    """
    automaticalls fills c1, c2, c3 for each node
    Needed: node_parent
    """
    if not parent_list:
        parent_list = tree_row_int(tree, N_parent)

    c_iter = 0
    last_parent = -1

    # parent_list [-1, 0, 0, 0, 1, 1]
    for i, val in enumerate(parent_list):
        my_id = i  # + 1  # nodeone
        parent_id = val  # - 1  # nodeone
        if val >= 0:  # +1 nodeone

            if val == last_parent:
                c_iter += 1
            else:
                last_parent = val
                c_iter = 0
            tree[N_c1 + c_iter][parent_id] = my_id
    return tree


def tree_row_int(tree, row_id):
    row = []
    for x in tree[row_id]:
        row.append(int(x))
    return row


def tree_core_setrow(tree, row_id, row):
    for i, x in enumerate(row):
        tree[row_id][i] = x
    return tree


def tree_plusnode(tree, add_or_sub, firstrow=1):
    """
    returns a tree where the nodes start at 1 instead of 0
    """
    nodes = len(tree[1])

    for row_id in [N_id, N_c1, N_c2, N_c3, N_parent]:
        for value in range(firstrow, nodes):
            if tree[row_id][value] != '':
                tree[row_id][value] = int(tree[row_id][value]) + add_or_sub
    tree[N_parent][firstrow] = -1
    return tree


def tree_from_labels(label_list, arity_list, type_list):
    """
    Given the labels (and label infos) as list
    this function builds the core of a tree (no node_modify)
    """

    # tree = tree_init_first_column()
    size = len(label_list)
    tree = tree_init_core(size)

    tree = tree_core_setrow(tree, N_id, [x for x in range(0, size)])
    # tree = tree_core_insert(tree, N_id, [x for x in range(1, size + 1)])
    tree = tree_core_setrow(tree, N_label, label_list)
    tree = tree_core_setrow(tree, N_arity, arity_list)
    tree = tree_core_setrow(tree, N_type, type_list)

    tree, parent_list = tree_core_parents(tree)
    tree = tree_core_c(tree)
    tree = tree_core_depth(tree, parent_list)

    return tree


def evolve_c_buffer_karoo(tree, node):
    """
    Generates the c_buffer for a node of a ptree

    """

    parent_arity_sum = 0
    prior_sibling_arity = 0
    prior_siblings = 0

    for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'

        if int(tree[N_depth][n]) == int(tree[N_depth][node]) - 1:  # find parent nodes at the prior depth
            if tree[N_arity][n] != '':
                parent_arity_sum = parent_arity_sum + int(tree[N_arity][n])  # sum arities of all parent nodes at the prior depth

        if int(tree[N_depth][n]) == int(tree[N_depth][node]) and int(tree[3][n]) < int(tree[3][node]):  # find prior siblings at the current depth
            if tree[N_arity][n] != '':
                prior_sibling_arity = prior_sibling_arity + int(tree[N_arity][n])  # sum prior sibling arity
            prior_siblings = prior_siblings + 1  # sum quantity of prior siblings

    c_buffer = node + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

    return c_buffer


def evolve_c_buffer(tree, node_id, wrapper=False):
    """
    Generates the c_buffer for a node_id of a tree
    The c_buffer is:
    """
    if wrapper:
        tree = tree_convert_karoo_to_plagih(tree)
        node_id -= 1

    parent_arity_sum = 0
    prior_sibling_arity_sum = 0
    prior_siblings = 0

    if node_id == 0:
        return 1

    for n in range(0, len(tree[N_id])):  # increment through all nodes in array 'tree'

        # sum up all arities of the parent level
        if int(tree[N_depth][n]) == int(tree[N_depth][node_id]) - 1:  # find parent nodes at the prior depth
            if tree[N_arity][n] != '':
                parent_arity_sum += int(tree[N_arity][n])  # sum arities of all parent nodes at the prior depth

        # add the arities of nodes on the left (siblings)
        elif int(tree[N_depth][n]) == int(tree[N_depth][node_id]) and int(tree[N_id][n]) < int(tree[N_id][node_id]):  # find prior siblings at the current depth
            if tree[N_arity][n] != '':
                prior_sibling_arity_sum += int(tree[N_arity][n])  # sum prior sibling arity
            prior_siblings += 1  # sum quantity of prior siblings

    # node_id = the position from where we start counting
    # (parent_arity_sum - prior_siblings - 1) = the amount of nodes after our node
    # prior_sibling_arity_sum = the amount of children before our children
    # + 1 = our first child node, not the last child of the prior sibling
    c_buffer = node_id + (parent_arity_sum - prior_siblings - 1) + prior_sibling_arity_sum + 1

    if wrapper:
        c_buffer += 1

    return c_buffer


def tree_convert_karoo_to_plagih(karoo_tree):
    """
    karoo has a first row with nonsense and nodes start with 1
    plagih has no first row and nodes start with 0
    """

    tree_nofirst = np.delete(karoo_tree, 0, 1)
    tree_nofirst_nodezero = tree_plusnode(tree_nofirst, add_or_sub=-1, firstrow=0)
    tree_nofirst_nodezero[N_parent][0] = -1

    return tree_nofirst_nodezero


def tree_convert_plagih_to_karoo(plagih_tree):
    """
    karoo has a first row with nonsense and nodes start with 1
    plagih has no first row and nodes start with 0
    """
    first_col = tree_init_first_column()
    tree_withfirst = np.concatenate((first_col, plagih_tree), axis=1)
    tree_karoo = tree_plusnode(tree_withfirst, add_or_sub=1, firstrow=1)
    tree_karoo[N_parent][1] = ''
    return tree_karoo


def tree_insert_subtree(tree, insert_core, delete_ids, wrapper=False):
    """
    insert a prepared subtree in a node-spot
    """
    if wrapper:
        tree = tree_convert_karoo_to_plagih(tree)
        for i, val in enumerate(delete_ids):
            delete_ids[i] -= 1

    # 1. insert the top node
    top_node_id = int(delete_ids[0])

    tree[N_label][top_node_id] = insert_core[N_label][0]  # --label
    tree[N_arity][top_node_id] = insert_core[N_arity][0]  # --arity
    tree[N_type][top_node_id] = 'term' if int(insert_core[N_arity][0]) == 0 else 'func'  # --type

    tree = np.delete(tree, delete_ids[1:], axis=1)  # delete all branches below

    c_buffer = evolve_c_buffer(tree, top_node_id)  # child nr.1 at c_buffer
    tree = tree_insert_node_child_dummies(tree, top_node_id, c_buffer)  # --child: id, depth, parent
    tree = evolve_node_renum(tree)  # --all: ids
    tree = tree_fix_link_child(tree)

    # 2. insert all following nodes
    insert_count = 1  # set node count to +1 as the new root has already replaced 'branch_top' (above)

    while insert_count < len(insert_core[3]):  # increment through all nodes in the new Tree, leaving out the root... +1??

        for j in range(0, len(tree[N_id])):  # increment through all nodes in og tree ('tree')

            if tree[N_type][j] == '':  # aka: is this a dummy?
                tree[N_type][j] = insert_core[N_type][insert_count]  # --type
                tree[N_label][j] = insert_core[N_label][insert_count]  # --label
                tree[N_arity][j] = insert_core[N_arity][insert_count]  # --arity

                if int(tree[N_arity][j]) == 0:
                    tree = tree_fix_link_child(tree)  # fix all child links
                    tree = evolve_node_renum(tree)  # renumber all 'NODE_ID's

                elif int(tree[N_arity][j]) > 0:
                    c_buffer = evolve_c_buffer(tree, j)  # generate 'c_buffer' for point of mutation ('branch_top')
                    tree = tree_insert_node_child_dummies(tree, j, c_buffer)  # insert new nodes
                    tree = tree_fix_link_child(tree)  # fix all child links
                    tree = evolve_node_renum(tree)  # renumber all 'NODE_ID's

                insert_count = insert_count + 1  # exit loop when 'node_count' reaches the number of columns in the array
    if wrapper:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def evolve_fix_link_child_doit(tree, node_id, c_buffer, wrapper=False):
    """
    Link each parent node_id to its children.

    """
    if wrapper:
        tree = tree_convert_karoo_to_plagih(tree)
        node_id -= 1
        c_buffer -= 1

    if int(tree[N_id][node_id]) == 0:
        c_buffer = 1  # if root (node_id 1) is passed through this method

    if tree[N_arity][node_id] != '':

        if int(tree[N_arity][node_id]) == 0:  # if arity = 0
            tree[9][node_id] = ''
            tree[10][node_id] = ''
            tree[11][node_id] = ''

        elif int(tree[N_arity][node_id]) == 1:  # if arity = 1
            tree[9][node_id] = c_buffer
            tree[10][node_id] = ''
            tree[11][node_id] = ''

        elif int(tree[N_arity][node_id]) == 2:  # if arity = 2
            tree[9][node_id] = c_buffer
            tree[10][node_id] = c_buffer + 1
            tree[11][node_id] = ''

        elif int(tree[N_arity][node_id]) == 3:  # if arity = 3
            tree[9][node_id] = c_buffer
            tree[10][node_id] = c_buffer + 1
            tree[11][node_id] = c_buffer + 2

        else:
            print('e', 'evolve_child_link: node_id', node_id, 'has arity', tree[N_arity][node_id])
            raise  # self.plagih_pause()  # consider special instructions for this (pause)
    if wrapper:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def tree_fix_link_child_karoo(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.

    """

    for node_id in range(1, len(tree[3])):
        c_buffer = evolve_c_buffer(tree, node_id, wrapper=True)  # generate c_buffer for each node
        tree = evolve_fix_link_child_doit(tree, node_id, c_buffer, wrapper=True)  # update child links for each node

    return tree


def tree_fix_link_child(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.

    """

    for node in range(0, len(tree[N_id])):
        c_buffer = evolve_c_buffer(tree, node)  # generate c_buffer for each node
        tree = evolve_fix_link_child_doit(tree, node, c_buffer)  # update child links for each node

    return tree


def tree_insert_node_child_dummies(tree, node_id, c_buffer, wrapper=False):
    """
    evolve_subtree_insert_child
    Insert child node_id into the copy of a parent Tree.

    """
    if wrapper:
        tree = tree_convert_karoo_to_plagih(tree)
        node_id -= 1
        c_buffer -= 1

    if int(tree[N_arity][node_id]) == 0:  # if arity = 0
        tree[N_c1][node_id], tree[N_c2][node_id], tree[N_c3][node_id] = '', '', ''
        return tree

    # for arity, insert nodes with correct id, depth, parent
    for c in range(0, int(tree[N_arity][node_id])):  # 0 to 3
        tree = np.insert(tree, c_buffer + c, '', axis=1)  # insert node_id for 'node_c1'
        tree[N_id][c_buffer + c] = c_buffer + c  # node_id ID
        tree[N_depth][c_buffer + c] = int(tree[N_depth][node_id]) + 1  # node_depth
        tree[N_parent][c_buffer + c] = int(tree[N_id][node_id])  # parent ID

    if wrapper:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def evolve_node_renum(tree):
    """
    Renumber all 'node_id' in a given tree.

    This is required after a new generation is evolved as the node_id numbers are carried forward from the previous
    generation but are no longer in order.

    """

    for n in range(0, len(tree[N_id])):
        tree[N_id][n] = n  # renumber all nodes

    return tree


def evolve_node_renum_karoo(tree):
    """
    Renumber all 'node_id' in a given tree.

    This is required after a new generation is evolved as the node_id numbers are carried forward from the previous
    generation but are no longer in order.

    """

    for n in range(0, len(tree[N_id])):
        tree[N_id][n] = n  # renumber all nodes

    return tree


def test_cases(number):
    if number == 0:
        label_list = ['Ifte', '<', '0', '2', 'observation1', '0']
        arity_list = [3, 2, 0, 0, 0, 0]
        type_list = ['func', 'func', 'term', 'term', 'term', 'term']
    elif number == 1:
        label_list = ['+', '+', '+', '+', '1', '2', '3', '4', '5']
        arity_list = [2, 2, 2, 2, 0, 0, 0, 0, 0]
        type_list = ['func', 'func', 'func', 'func', 'term', 'term', 'term', 'term', 'term']
    elif number == 2:
        label_list = ['+', '+', '+', '0', '1', 'Ifte', '2', '3', '+', '4', '5', '6']  # 12 nodes
        arity_list = [2, 2, 2, 0, 0, 3, 0, 0, 2, 0, 0, 0]
        type_list = ['func', 'func', 'func', 'term', 'term', 'func', 'term', 'term', 'func', 'term', 'term', 'term']
    elif number == 3:
        label_list = ['True']  # 12 nodes
        arity_list = [0]
        type_list = ['term']
    elif number == 4:
        label_list = ['Ifte', '<', '0', 'Ifte', 'observation1', '0', 'True', '2', '0']
        arity_list = [3, 2, 0, 3, 0, 0, 0, 0, 0]
        type_list = ['func', 'func', 'term', 'func', 'term', 'term', 'term', 'term', 'term']
    else:
        label_list = ['0']
        arity_list = [0]
        type_list = ['term']
        # solution = np.array([['tree_id', '', '', '', '', '', '', '', '', '', '', '', ''],
        #                      ['tree_type', '', '', '', '', '', '', '', '', '', '', '', ''],
        #                      ['tree_depth_base', '', '', '', '', '', '', '', '', '', '', '', ''],
        #                      ['node_id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
        #                      ['node_depth', '0', '1', '1', '2', '2', '2', '2', '3', '3', '3', '4', '4'],
        #                      ['node_type', 'func', 'func', 'func', 'term', 'term', 'func', 'term', 'term', 'func', 'term', 'term', 'term'],
        #                      ['node_label', '+', '+', '+', '0', '1', 'Ifte', '2', '3', '+', '4', '5', '6'],
        #                      ['node_parent', '-1', '0', '0', '1', '1', '2', '2', '5', '5', '5', '8', '8'],
        #                      ['node_arity', '2', '2', '2', '0', '0', '3', '0', '0', '2', '0', '0', '0'],
        #                      ['node_c1', '1', '3', '5', '', '', '7', '', '', '10', '', '', ''],
        #                      ['node_c2', '2', '4', '6', '', '', '8', '', '', '11', '', '', ''],
        #                      ['node_c3', '', '', '', '', '', '9', '', '', '', '', '', ''],
        #                      ['fitness', '', '', '', '', '', '', '', '', '', '', '', ''],
        #                      ['node_modify', '', '', '', '', '', '', '', '', '', '', '', ''],
        #                      ['parsimony', '', '', '', '', '', '', '', '', '', '', '', '']])
    return label_list, arity_list, type_list


def tree_test_plausibility(tree, wrapper=True):
    """
    A method to check if a tree is plausible. aka:
    - do the values in c1, c2, c3 link to correkt
    """
    if not wrapper:
        pass
    id_list = []
    c_list = []
    for n in range(1, len(tree[3])):
        for c in range(0, 3):
            if tree[N_c1 + c][n] != '':
                c_list.append(int(tree[N_c1 + c][n]))
        id_list.append(int(tree[N_id][n]))
    if sum(c_list) == sum(id_list) - 1:
        return True
    else:
        return False


def karoo_tree_from_user(label_list, modify_list=None):
    arity_list = [op_label_get_arity(label) for label in label_list]
    type_list = ['term' if arity == 0 else 'func' for arity in arity_list]
    tree = tree_from_labels(label_list, arity_list, type_list)
    if modify_list:
        for i, val in enumerate(modify_list):
            tree[N_modify][i] = val
    else:  # all can be modified
        for i, val in enumerate(label_list):
            tree[N_modify][i] = 1
    tree = tree_convert_plagih_to_karoo(tree)

    return tree


def test():
    label_list, arity_list, type_list = test_cases(4)
    core = tree_from_labels(label_list, arity_list, type_list)
    tree = tree_convert_plagih_to_karoo(core)
    print(tree)

# test()
