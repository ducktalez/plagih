import os
import numpy as np
from plagih.modules.plagih_sympy_extras import plagih_sympify
from plagih.tree_distances.tree_edit_distance import apted_distance
from plagih.modules.plagih_types import *

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees


def util_tree_copy(population, tree_id):
    """
    copy a tree from a population
    """
    return np.copy(population[tree_id])


def pop_random(population):
    """
    Returns a random tree_id from a population
    """
    return np.random.randint(1, len(population))


def pop_copy_genepool(population_new, gene_pool, gen_id):
    """
    Copy the genepool of a gen
    """
    pop_y = ['Population Selection in Generation {}.'.format(str(gen_id))]  # empty list

    for i, (tree_num, tree_meta) in enumerate(gene_pool.items()):
        tree_copy = util_tree_copy(population_new, tree_num)
        tree_copy = tree_set_id(tree_copy, i + 1)
        pop_y.append(tree_copy)

    return pop_y


class Plagih_Tree():

    def __init__(self, expr):
        self.fitness = None
        self.parsimony = None
        self.expr = expr

    def write_to_file(self, path):
        pass


def tree_init_core(node_amount):
    """
    returns an empty tree with an amount of nodes, auto fills
    """
    tree = np.zeros((T_num_lines, node_amount), dtype=np.dtype('U12'))  # U12: longest is observation1

    return tree


def tree_modifyable_nodes_set(chosen_tree, origin_tree):
    """
    Sets all the origin core nodes back to non-modifyable
    """
    # Set all nodes to be modifiable (=1)
    for i, tmp in enumerate(chosen_tree[N_modify][1:]):
        chosen_tree[N_modify][i + 1] = '1'

    # Find no-modifyables in Origin
    non_modifiable_nodes = []
    if origin_tree[N_modify][1] == '0':  # check is modifiable nodes are specified
        non_modifiable_nodes.extend(tree_permanent_nodes_get(1, chosen_tree, 1, origin_tree))

    for non_modifiable in non_modifiable_nodes:
        chosen_tree[N_modify][non_modifiable] = '0'

    return chosen_tree


def tree_permanent_nodes_get(origin_node, chosen_tree, chosen_node, origin_tree):
    """
    Returns a list of nodes that are not supposed to be modified
    """

    if origin_tree[N_modify][origin_node] == '0':
        permanent_nodes = [int(chosen_tree[N_id][chosen_node])]
        for child in [N_c1, N_c2, N_c3]:
            if origin_tree[child][origin_node] != '':
                next_origin_node = int(origin_tree[child][origin_node])
                next_chosen_node = int(chosen_tree[child][chosen_node])
                tmp = tree_permanent_nodes_get(next_origin_node, chosen_tree, next_chosen_node, origin_tree)
                if tmp is not None:
                    permanent_nodes.extend(tmp)
        return permanent_nodes
    else:
        return


def tree_set_id(tree, tree_id):
    tree[TR_ID][1] = tree_id
    return tree


def tree_set_history(tree, last_modification):
    tree[TR_type][1] = last_modification
    return tree


def tree_node_get_arity(tree, node_id, karoo=False):
    # if karoo:
    #     node_id = int(node_id) - 1

    return int(tree[N_arity][int(node_id)])


def round_constant(constant, accuracy):
    """
    Rounding float constants
    """
    constant = float(constant)
    new_const = round(constant * accuracy) / accuracy
    if new_const == 0 and constant > 0:
        new_const = 1 / accuracy
    elif new_const == 0 and constant < 0:
        new_const = -1 / accuracy

    return new_const


def tree_round_constants(tree, accuracy, karoo=False):
    """
    rounds the values in constant float nodes
    """

    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    for node_id in tree_get_leafes(tree):
        if tree_node_get_nodekind(tree, node_id) == 'term-float':
            tmp = round_constant(tree[N_label][node_id], accuracy)
            tree[N_label][node_id] = tmp

    if karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def tree_store_fitness(tree, fitness, precision=6):
    """
    Store the fitness within the tree np-array

    """

    fitness = float(fitness)
    fitness = round(fitness, precision)

    tree[T_fitness][1] = fitness  # store the fitness with each tree

    return tree


def tree_get_fitness(tree, precision=None, karoo=True):
    if not karoo:
        raise
    fitness = float(tree[T_fitness][1])

    if precision:
        fitness = round(fitness, precision)
    return fitness


def tree_expr_raw(tree, node_id):
    """
    Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').

    """
    node_id = int(node_id)

    if tree[N_arity, node_id] == '0':  # arity of 0 for the pattern '[term]'
        return '(' + tree[N_label, node_id] + ')'  # 'node_label' (function or terminal)

    elif tree[N_arity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        return '(' + tree_expr_raw(tree, tree[9, node_id]) + tree[N_label, node_id] + ')'

    elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
        # This if case is for 2-ary ops that is prefix. like Min(a, b)
        if tree[N_label, node_id] not in functions_infix_dict:
            return '(' + tree[N_label, node_id] + '(' + tree_expr_raw(tree, tree[9, node_id]) + ', ' + tree_expr_raw(tree, tree[10, node_id]) + '))'
        else:
            return '(' + tree_expr_raw(tree, tree[9, node_id]) + tree[N_label, node_id] + tree_expr_raw(tree, tree[10, node_id]) + ')'  # Klammern, da sympify sonst abkacnen könnte

    elif tree[N_arity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return '(Ifte(' + tree_expr_raw(tree, tree[9, node_id]) + ', ' + tree_expr_raw(tree, tree[10, node_id]) + ', ' + tree_expr_raw(tree, tree[11, node_id]) + '))'


def tree_parsimony_ted(tree1, tree2):
    """
    The Tree Edit distance (TED) ('coolest' distance)
    - the amount of changes that have to be applied to the origin to equality are counted
    """
    # TODO TED soll geänderte Werte ignorieren
    apted_tree1 = tree_raw_depth_prefix(tree1, 1)
    apted_tree2 = tree_raw_depth_prefix(tree2, 1)
    distance, mapping = apted_distance(apted_tree1, apted_tree2)
    # sfeh the mapping could be handy somewhere
    return distance


def tree_parsimony_relari(tree, origin_tree):
    """
    This distance penalizes non-original functions with its arity
    - ignore node[0] [description]
    - look within the subtree if the original function is on origin_tree spot
    """

    # If the new tree is actually less complex than the original one, just return 1
    if len(tree[N_label]) < len(origin_tree[N_label]):
        return 1

    distance = 0

    # iterate over every node in the new tree
    for i, arity in enumerate(tree[N_arity]):
        if i == 0:  # skip node 0. the description
            continue
        elif i < len(origin_tree[N_label]):  # Make sure we stay within the tree index. <= does not work
            if origin_tree[N_label][i] != tree[N_label][i]:  # is it different from the origin_tree?
                distance = distance + int(arity)  # add the nodes arity. double-punishes large trees
        else:
            distance = distance + int(arity)

    return max(distance, 1)  # make sure, it does not return 0


def tree_expr_sympify(algo_raw=None, tree=None):
    """
    returns the sympifyed expression
    """
    if tree is not None:  # If we got a tree, we generate the expression
        algo_raw = str(tree_expr_raw(tree, 1))

    try:
        expr_sym = plagih_sympify(algo_raw)
        expr_sym_str = str(expr_sym)
    except:
        print('w', 'In sympify. Caused by this raw algorithm: ' + str(algo_raw))
        raise

    for fail_reason in ['zoo', 'inf', '*I', 'nan']:
        if fail_reason in expr_sym_str:
            raise
    return expr_sym_str


def tree_raw_depth_prefix(tree, node_id):
    """
    Does the same as tree_expr_raw, but evaluates infix functions in prefix notation (functional form)

    """

    node_id = int(node_id)

    if tree[N_arity, node_id] == '0':  # arity of 0 for the pattern '[term]'
        return '{' + tree[N_label, node_id] + '}'  # 'node_label' (function or terminal)

    elif tree[N_arity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        return '{' + tree[N_label, node_id] + tree_raw_depth_prefix(tree, tree[9, node_id]) + '}'

    elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
        return '{' + tree[N_label, node_id] + '' + tree_raw_depth_prefix(tree, tree[9, node_id]) + tree_raw_depth_prefix(tree, tree[10, node_id]) + '' + '}'

    elif tree[N_arity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return '{Ifte' + tree_raw_depth_prefix(tree, tree[9, node_id]) + tree_raw_depth_prefix(tree, tree[10, node_id]) + tree_raw_depth_prefix(tree, tree[11, node_id]) + '' + '}'


def tree_store_parsimony(tree, parsimony):
    """
    Store the parsimony within the tree np-array
    """
    if parsimony < 0:
        print('Warning: Parsimony is: {}'.format(parsimony))
    tree[T_parsimony][1] = parsimony
    return tree


def tree_node_get_nodekind(tree, node, karoo=False):
    """
    'func', 'term-variable', 'term-float', 'term-bool'
    """
    arity = tree_node_get_arity(tree, node, karoo=False)
    if arity > 0:
        nodekind = 'func'
    else:
        label = tree[N_label][node]
        if 'observation' in label:
            nodekind = 'term-variable'
        elif 'True' in label or 'False' in label:
            nodekind = 'term-bool'
        else:
            try:
                float(label)
                nodekind = 'term-float'
            except ValueError:
                print('No good. This label is completely unknown: {} (or arity {} is not correct).'.format(label, arity))
                raise
    return nodekind


def tree_get_leafes(tree, karoo=False):
    """
    Just return leaf nodes of a tree
    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    node_ids = []
    for node_id in tree[N_id]:
        if tree_node_get_arity(tree, int(node_id), karoo=False) == 0:
            node_ids.append(int(node_id))

    if karoo:
        node_ids = [x - 1 for x in node_ids]

    return node_ids


def tree_branch_get_label_list(tree, node_ids, karoo=False):
    """
    This method prepares a stand-alone Tree as a copy of the given node_ids.

    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)
        node_ids = [x - 1 for x in node_ids]

    label_list = []
    arity_list = []

    for node_id in node_ids:
        label_list.append(tree[N_label][node_id])
        arity_list.append(tree[N_arity][node_id])

    if karoo:
        pass

    return label_list, arity_list


def evolve_node_arity_fix(tree):
    """
    In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.

    This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and
    Crossover.

    """

    for n in range(1, len(tree[N_id])):  # increment through all nodes (exclude 0) in array 'tree'
        if len(tree[N_id]) <= 2:
            print('FUCK {}'.format(tree))
        if tree[N_arity][n] == '0':  # check for discrepency
            # tree[N_arity][n] = '0'  # set arity to 0
            tree[N_c1][n] = ''  # wipe 'node_c1'
            tree[N_c2][n] = ''  # wipe 'node_c2'
            tree[N_c3][n] = ''  # wipe 'node_c3'
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
#             tree[N_c1][n] = ''  # wipe 'node_c1'
#             tree[N_c2][n] = ''  # wipe 'node_c2'
#             tree[N_c3][n] = ''  # wipe 'node_c3'
#             tree[N_modify][n] = '1'
#
#     return tree

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


def core_from_labels(label_list, arity_list):
    """
    Given the labels (and label infos) as list
    this function builds the core of a tree (no node_modify)
    """
    if len(label_list) == 0:
        print('label list is empty. Please handle this error earlier in code.')
        raise
    size = len(label_list)
    tree = tree_init_core(size)

    # set all the rows that are super easy
    tree = tree_core_setrow(tree, N_id, [x for x in range(0, size)])
    tree = tree_core_setrow(tree, N_label, label_list)
    tree = tree_core_setrow(tree, N_arity, arity_list)

    # and also, fill all the leftover rows
    tree, parent_list = tree_core_parents(tree)
    tree = tree_core_c(tree)
    tree = tree_core_depth(tree, parent_list)

    if not tree_test_check_children(tree, karoo=False):
        print(tree)
        raise

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


def evolve_c_buffer(tree, node_id, karoo=False):
    """
    Generates the c_buffer for a node_id of a tree
    The c_buffer is:
    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)
        node_id -= 1

    parent_arity_sum = 0
    prior_sibling_arity_sum = 0
    prior_siblings = 0

    if node_id == 0:
        return 1

    for n in range(0, len(tree[N_id])):  # increment through all nodes in array 'tree'

        # sum up all arities of the parent dim_y
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

    if karoo:
        c_buffer += 1

    return c_buffer


def tree_convert_karoo_to_plagih(karoo_tree):
    """
    tests has a first row with nonsense and nodes start with 1
    plagih has no first row and nodes start with 0
    """

    tree_nofirst = np.delete(karoo_tree, 0, 1)
    tree_nofirst_nodezero = tree_plusnode(tree_nofirst, add_or_sub=-1, firstrow=0)
    tree_nofirst_nodezero[N_parent][0] = -1

    return tree_nofirst_nodezero


def tree_convert_plagih_to_karoo(plagih_tree):
    """
    tests has a first row with nonsense and nodes start with 1
    plagih has no first row and nodes start with 0
    """
    first_col = tree_init_first_column()
    tree_withfirst = np.concatenate((first_col, plagih_tree), axis=1)
    tree_karoo = tree_plusnode(tree_withfirst, add_or_sub=1, firstrow=1)
    tree_karoo[N_parent][1] = ''
    return tree_karoo


def tree_insert_subtree(tree, insert_core, delete_ids, karoo=False):
    """
    insert a prepared subtree in a node-spot
    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)
        for i, val in enumerate(delete_ids):
            delete_ids[i] -= 1

    # 1. insert the top node
    top_node_id = int(delete_ids[0])

    tree[N_label][top_node_id] = insert_core[N_label][0]  # --label
    tree[N_arity][top_node_id] = insert_core[N_arity][0]  # --arity
    # tree[N_type][top_node_id] = 'term' if int(insert_core[N_arity][0]) == 0 else 'func'  # --type

    tree = np.delete(tree, delete_ids[1:], axis=1)  # delete all branches below

    c_buffer = evolve_c_buffer(tree, top_node_id)  # child nr.1 at c_buffer
    tree = tree_insert_node_child_dummies(tree, top_node_id, c_buffer)  # --child: id, depth, parent
    tree = evolve_node_renum(tree)  # --all: ids
    tree = tree_fix_link_child(tree)

    # 2. insert all following nodes
    insert_count = 1  # set node count to +1 as the new root has already replaced 'branch_top' (above)

    while insert_count < len(insert_core[3]):  # increment through all nodes in the new Tree, leaving out the root... +1??

        for j in range(0, len(tree[N_id])):  # increment through all nodes in og tree ('tree')

            if tree[N_label][j] == '':  # aka: is this a dummy?
                # tree[N_type][j] = insert_core[N_type][insert_count]  # --type
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
    if karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def evolve_fix_link_child_doit(tree, node_id, c_buffer, karoo=False):
    """
    Link each parent node_id to its children.

    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)
        node_id -= 1
        c_buffer -= 1

    if int(tree[N_id][node_id]) == 0:
        c_buffer = 1  # if root (node_id 1) is passed through this method

    if tree[N_arity][node_id] != '':

        if int(tree[N_arity][node_id]) == 0:  # if arity = 0
            tree[N_c1][node_id] = ''
            tree[N_c2][node_id] = ''
            tree[N_c3][node_id] = ''

        elif int(tree[N_arity][node_id]) == 1:  # if arity = 1
            tree[N_c1][node_id] = c_buffer
            tree[N_c2][node_id] = ''
            tree[N_c3][node_id] = ''

        elif int(tree[N_arity][node_id]) == 2:  # if arity = 2
            tree[N_c1][node_id] = c_buffer
            tree[N_c2][node_id] = c_buffer + 1
            tree[N_c3][node_id] = ''

        elif int(tree[N_arity][node_id]) == 3:  # if arity = 3
            tree[N_c1][node_id] = c_buffer
            tree[N_c2][node_id] = c_buffer + 1
            tree[N_c3][node_id] = c_buffer + 2

        else:
            print('e', 'evolve_child_link: node_id', node_id, 'has arity', tree[N_arity][node_id])
            raise
    if karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def tree_fix_link_child_karoo(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'config.tree' has been modified, as with both Grow and Full mutation.

    """

    for node_id in range(1, len(tree[3])):
        c_buffer = evolve_c_buffer(tree, node_id, karoo=True)  # generate c_buffer for each node
        tree = evolve_fix_link_child_doit(tree, node_id, c_buffer, karoo=True)  # update child links for each node

    return tree


def tree_fix_link_child(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'config.tree' has been modified, as with both Grow and Full mutation.

    """

    for node in range(0, len(tree[N_id])):
        c_buffer = evolve_c_buffer(tree, node)  # generate c_buffer for each node
        tree = evolve_fix_link_child_doit(tree, node, c_buffer)  # update child links for each node

    return tree


def tree_insert_node_child_dummies(tree, node_id, c_buffer, karoo=False):
    """
    evolve_subtree_insert_child
    Insert child node_id into the copy of a parent Tree.

    """
    if karoo:
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

    if karoo:
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


def tree_get_label(tree, node_id):
    """

    """
    label = tree[N_label][int(node_id)]
    return label


def xtype_get_constant(label, node_arity=None, only_float=True):
    """

    """
    const_xtype = None

    if not node_arity:
        node_arity = label_get_arity(label)

    if node_arity == 0:  # arity=0 -> terminal
        if 'True' in label or 'False' in label:
            if not only_float:
                const_xtype = '2b'
        elif 'observation' in label or 'action' in label:
            pass
        else:
            # now it MUST be float
            const_xtype = '2f'

    return const_xtype


def tree_get_mutatable_nodes(tree, no_root=False):
    """
    Returns a list with mutatable ids
    """

    node_ids = []
    for i, node_id in enumerate(tree[N_id]):
        if tree[N_modify][i] == '1':
            node_ids.append(int(node_id))

    if no_root and root_id in node_ids:
        node_ids.remove(root_id)
    return node_ids


def tree_get_fix_nodes(tree):
    """
    Returns a list with mutatable ids
    """

    node_ids = []
    for i, node_id in enumerate(tree[N_id]):
        if tree[N_modify][i] == '0':
            node_ids.append(int(node_id))

    return node_ids


def tree_node_get_idstring(tree, node_id):
    """
    return a list of s nodes childs.
    + Evaluate all or part of a Tree and

    This method generates a list of all 'node_id's from the given Node and below. It is used primarily to generate
    'branch' for the multi-generational mutation of Trees.
    """

    node_id = int(node_id)

    if tree[N_arity, node_id] == '0':  # arity of 0 for the pattern '[node_id]'
        return tree[3, node_id]

    elif tree[N_arity, node_id] == '1':  # arity of 1 for the pattern '[node_id], [node_id]'
        return '{}, {}'.format(tree[3, node_id], tree_node_get_idstring(tree, tree[9, node_id]))

    elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[node_id], [node_id], [node_id]'
        return '{}, {}, {}'.format(
            tree[3, node_id],
            tree_node_get_idstring(tree, tree[9, node_id]),
            tree_node_get_idstring(tree, tree[10, node_id]))

    elif tree[N_arity, node_id] == '3':  # arity of 3 for the pattern '[node_id], [node_id], [node_id], [node_id]'
        return '{}, {}, {}, {}'.format(
            tree[3, node_id],
            tree_node_get_idstring(tree, tree[9, node_id]),
            tree_node_get_idstring(tree, tree[10, node_id]),
            tree_node_get_idstring(tree, tree[11, node_id]))


def labels_get_aritys_list(label_list, karoo=False):
    """
    returns an arity list for a label list
    """

    arity_list = [label_get_arity(x) for x in label_list]

    if karoo:
        arity_list.pop(0)
    return arity_list


def tree_get_branchinfo(tree, node_id, karoo=True):
    """
    returns all ids, labels and arities for a node in a tree
    """
    ids = tree_get_branch(tree, node_id)
    labels = [tree[N_label][i] for i in ids]
    aritys = [tree[N_arity][i] for i in ids]
    return ids, labels, aritys


def tree_pretty_print(tree, karoo=False):
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    depth = 0
    layer_labels = []
    for i, n_depth in enumerate(tree[N_depth]):
        label = tree_get_label(tree, i)
        if int(n_depth) == depth:
            layer_labels.append(label)
        else:
            print(layer_labels)
            layer_labels = [label]
            depth += 1
    else:
        print(layer_labels)

    return


def tree_get_branch(tree, node, karoo=True):
    """
    return all child-nodes as list
    """
    if not karoo:
        raise Exception

    branch = np.array([])  # the array is necessary in order to len(branch) when 'branch' has only one element

    # 2. Also return all child nodes
    branch_eval = tree_node_get_idstring(tree, node)  # generate tuple of 'branch_top' and subsequent nodes
    branch_symp = plagih_sympify(branch_eval)  # convert string into something useful

    branch = np.append(branch, branch_symp)
    branch = np.sort(branch)  # sort nodes in branch for Crossover.

    return branch


def tree_labels(tree):
    """
    Just helps printing trees better
    """
    label_list = tree[N_label]
    return label_list


def tree_test_check_children(tree, karoo=True):
    """
    A method to check if a tree is plausible. aka:
    - do the values in c1, c2, c3 link to correct
    """
    if not karoo:
        tree = tree_convert_plagih_to_karoo(tree)

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


def tree_delete_nodes(tree, node_list):
    tree = np.delete(tree, node_list, axis=1)  # delete all branches below
    return tree


def tree_check_expression(tree, karoo=True):
    """

    """

    label_list = tree[N_label]
    arity_list = tree[N_arity]

    if karoo:
        label_list = label_list[1:]
        arity_list = arity_list[1:]

    try:
        core = core_from_labels(label_list, arity_list)
    except:
        return False

    return True


def tree_check_all(tree, karoo=True):
    label_list = tree[N_label]
    arity_list = tree[N_arity]

    if karoo:
        label_list = label_list[1:]
        arity_list = arity_list[1:]

    try:
        core = core_from_labels(label_list, arity_list)
        result = tree_test_check_children(core, karoo=False)
    except:
        return False

    return result


def karoo_tree_from_labellist(label_list, modify_list=None):
    """
    create a tree from user input
    """
    arity_list = [label_get_arity(label) for label in label_list]
    core = core_from_labels(label_list, arity_list)
    if modify_list:
        for i, val in enumerate(modify_list):
            core[N_modify][i] = val
    else:  # all can be modified
        for i, val in enumerate(label_list):
            core[N_modify][i] = '1'
    tree = tree_convert_plagih_to_karoo(core)

    return tree


def test_trees(number):
    if number == 0:
        label_list = ['Ifte', '<', '0', '2', 'observation1', '0']
        arity_list = [3, 2, 0, 0, 0, 0]
    elif number == 1:
        label_list = ['+', '+', '+', '+', '1', '2', '3', '4', '5']
        arity_list = [2, 2, 2, 2, 0, 0, 0, 0, 0]
    elif number == 2:
        label_list = ['+', '+', '+', '0', '1', 'Ifte', '2', '3', '+', '4', '5', '6']  # 12 nodes
        arity_list = [2, 2, 2, 0, 0, 3, 0, 0, 2, 0, 0, 0]
    elif number == 3:
        label_list = ['True']  # 12 nodes
        arity_list = [0]
    elif number == 4:
        label_list = ['Ifte', '<', '0', 'Ifte', 'observation1', '0', 'True', '2', '0']
        arity_list = [3, 2, 0, 3, 0, 0, 0, 0, 0]
    elif number == 5:
        label_list = ['+', '+', '+', '0', '0', '0', '0']
        arity_list = [2, 2, 2, 0, 0, 0, 0]
    else:
        label_list = ['0']
        arity_list = [0]

    core = core_from_labels(label_list, arity_list)
    return core
#
# def tree_node_get_arity(tree, node_id, tests=False):
#     node_id = int(node_id) - 1
#
#     return int(tree[N_arity][int(node_id)])


def tree_set_label(tree, node_id, label):
    tree[N_label][int(node_id)] = label
    return tree


def tree_iterate_ids(tree, karoo=False):
    if karoo:
        start = 1
    else:
        start = 0
    node_id_list = [int(node_id) for node_id in tree[N_id][start:]]
    return node_id_list


def tree_iterate_range(tree, karoo=False):
    """
    iterates- but over the range.
    This is useful if the tree is currently not in a state, where
    the node id is at the corresponding position in the tree
    """
    if karoo:
        start = 1
    else:
        start = 0
    np_list = range(start, len(tree[N_label]))
    return np_list


def tree_node_get_child(tree, node_id, child_num, karoo=False):
    if karoo:
        child_id = tree[N_c1 + child_num][node_id]
    else:
        raise
    return child_id


def tree_normalize_exponentiation(tree):

    # 1. ** should have an int as second number
    for node_id in tree_iterate_ids(tree, karoo=True):
        if tree_get_label(tree, node_id) == '**':
            child_id = tree_node_get_child(tree, node_id, 1, karoo=True)  # get second argument
            old_power = tree_get_label(tree, child_id)
            try:
                new_power = float(int(float(old_power)))
                tree = tree_set_label(tree, child_id, new_power)
            except ValueError:
                pass  # sfeh: This may actually take some time. Every tree gets checked any many have '**'.
    return tree


def tree_get_mutatable_leaves_lv0(tree, karoo=True):
    """
    Returns a list with mutatable ids on the outside
    """

    node_ids = []
    fix_ids = tree_get_fix_nodes(tree)
    if len(fix_ids) == 0:
        return root_id + 1
    else:
        for node_id in fix_ids:

            arity = tree_node_get_arity(tree, node_id, karoo=karoo)
            for c in range(0, arity):
                child_id = int(tree[N_c1+c][node_id])
                # if tree_node_modifiable(tree, node_id):
                if int(tree[N_modify][child_id]) == 1:
                    node_ids.append(int(tree[N_c1+c][node_id]))

    return node_ids


def tree_get_mutatable_extendables(tree, karoo=True):
    """
    Returns a list with mutatable ids on the outside
    """
    fix_ids = tree_get_fix_nodes(tree)
    leaf_ids = []
    for node_id in fix_ids:

        arity = tree_node_get_arity(tree, node_id, karoo=karoo)
        for c in range(0, arity):
            child_id = int(tree[N_c1+c][node_id])
            # if tree_node_modifiable(tree, node_id):
            if int(tree[N_modify][child_id]) == 1:
                leaf_ids.append(int(tree[N_c1+c][node_id]))

    core_ids = []
    core_ids.extend(fix_ids)
    core_ids.extend(leaf_ids)
    core_ids.sort()

    return core_ids


def tree_node_get_childs(tree, node_id, karoo=True):
    """

    """
    child_list = []
    arity = tree_node_get_arity(tree, node_id, karoo=karoo)
    for c in range(0, arity):
        child_list.append(int(tree[N_c1 + c][node_id]))
    return child_list


def tree_node_get_parent(tree, node_id, karoo=True):
    """

    """
    return tree[N_parent][node_id]


def tree_node_get_parent_functype(tree, node_id, karoo=True):
    """

    """
    parent_id = tree[N_parent][node_id]
    if tree_node_get_arity(tree, parent_id, karoo=karoo) > 0:
        parent_label = tree_get_label(tree, parent_id)
        fun_type = xtype_get_v2(parent_label)
        return fun_type
    else:
        print_e('That was not a function.')
        raise


def tree_get_mutatable_leaves(tree, level, karoo=True):
    """
    Returns a list with mutatable ids on the outside
    """

    lvl_count = 0
    node_ids = []
    while lvl_count <= level:
        if lvl_count == 0:
            node_ids = tree_get_mutatable_leaves_lv0(tree, karoo=karoo)
        elif lvl_count > 0:
            new_node_ids = []
            for node_id in node_ids:
                child_list = tree_node_get_childs(tree, node_id)
                new_node_ids.extend(child_list)
            node_ids = new_node_ids.copy()
        lvl_count += 1

    return node_ids


