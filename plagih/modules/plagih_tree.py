import os
import numpy as np
from plagih.modules.plagih_sympy_extras import plagih_sympify
from plagih.tree_distances.tree_edit_distance import apted_distance
from plagih.modules.plagih_types import *
from plagih.modules.plagih_eval import *
import random
import csv

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees

TR_ID = 0
TR_type = 1
N_id = 3
N_depth = 4
N_type = 5
N_label = 6
N_parent = 7
N_arity = 8
N_c1 = 9
N_c2 = 10
N_c3 = 11
T_fitness = 12
N_modify = 13
T_parsimony = 14

T_num_lines = 15
P_first_node = 1
root_id = 1
node_is_modifiable = '1'


class Plagih_Tree():
    #
    # def __init__(self, expr=None):
    #     self.fitness = None
    #     self.parsimony = None
    #     self.expr = expr
    #     self.numpy_nodes = None

    # def karoo_tree_from_labellist(label_list, modify_list=None):
    def __init__(self, label_list, modify_list=None):
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
        self.tree = tree_convert_plagih_to_karoo(core)

        return

    def get_uninstanced_tree(self):
        return self.tree

    def write_to_file(self, path):
        pass



class Plagih_node():

    def __init__(self, n_id, depth, n_type, label, parent, arity, c1, c2, c3):
        return


def tree_get_size(tree, karoo=True):
    if karoo:
        size = len(tree[0])
        return size
    else:
        return 0


def tree_set_id(tree, tree_id):
    tree[TR_ID][1] = tree_id
    return tree


def tree_set_history(tree, last_modification):
    tree[TR_type][1] = last_modification
    return tree


def tree_set_fitness(tree, fitness, precision=6):
    """
    Store the fitness within the tree np-array

    """

    fitness = float(fitness)
    fitness = round(fitness, precision)

    tree[T_fitness][1] = fitness  # store the fitness with each tree

    return tree


def tree_set_parsimony(tree, parsimony):
    """
    Store the parsimony within the tree np-array
    """
    if parsimony < 0:
        print_warning('w', 'Warning: Parsimony is: {}'.format(parsimony))
    tree[T_parsimony][1] = parsimony
    return tree


def tree_set_modifyable_nodes_true(tree, karoo=True):
    """

    """
    if karoo:
        start = 1
    else:
        start = 0

    for node_id in range(start, len(tree[N_modify])):
        tree[N_modify][node_id] = '1'
    return tree


def tree_set_modifyable_nodes(chosen_tree, origin_tree):
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


def tree_node_set_childs_ids(tree, node_id, c_buffer, karoo=False):
    """
    Link each parent node_id to its children.

    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)
        node_id -= 1
        c_buffer -= 1

    if node_id == 0:
        c_buffer = 1  # if root (node_id 1) is passed through this method

    arity = tree_node_get_arity(tree, node_id)

    for i in range(arity):
        tree[N_c1 + i][node_id] = c_buffer + i

    for i in range(arity, 3):
        tree[N_c1 + i][node_id] = ''
    #
    # if arity == 0:  # if arity = 0
    #     tree[N_c1][node_id] = ''
    #     tree[N_c2][node_id] = ''
    #     tree[N_c3][node_id] = ''
    #
    # elif arity == 1:  # if arity = 1
    #     tree[N_c1][node_id] = c_buffer
    #     tree[N_c2][node_id] = ''
    #     tree[N_c3][node_id] = ''
    #
    # elif arity == 2:  # if arity = 2
    #     tree[N_c1][node_id] = c_buffer
    #     tree[N_c2][node_id] = c_buffer + 1
    #     tree[N_c3][node_id] = ''
    #
    # elif arity == 3:  # if arity = 3
    #     tree[N_c1][node_id] = c_buffer
    #     tree[N_c2][node_id] = c_buffer + 1
    #     tree[N_c3][node_id] = c_buffer + 2
    #
    # else:
    #     print_e('evolve_child_link: node_id {} has arity {}.'.format(node_id, tree[N_arity][node_id]))
    #     raise

    if karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def tree_node_set_label(tree, node_id, label):
    tree[N_label][int(node_id)] = label
    return tree


def tree_node_set_arity(tree, node_id, arity):
    tree[N_arity][int(node_id)] = int(arity)
    return tree


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


def tree_node_get_xtype(tree, node_id):
    return tree[N_type][node_id]


def tree_node_get_arity(tree, node_id):
    arity = tree[N_arity][int(node_id)]
    if arity == '':
        arity = 0
    else:
        arity = int(arity)

    return arity


def tree_node_get_nodekind(tree, node):
    """
    'func', 'term-variable', 'term-float', 'term-bool'
    """
    arity = tree_node_get_arity(tree, node)
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
                print_e('No good. This label is completely unknown: {} (or arity {} is not correct).'.format(label, arity))
                raise
    return nodekind


def tree_node_get_label(tree, node_id):
    """

    """
    label = tree[N_label][int(node_id)]
    return label


def tree_node_get_depth(tree, node_id):
    """

    """
    depth = tree[N_depth][int(node_id)]
    return int(depth)


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


def tree_node_get_lax(tree, node_id, variables_dict):
    label = tree_node_get_label(tree, node_id)
    arity = tree_node_get_arity(tree, node_id)
    xtype = xtype_get(label, variables_dict)
    return label, arity, xtype


def tree_node_get_child(tree, node_id, child_num):
    """
    returns ONE specified child of a node.
    For a list with all childs, search for the plural version
    """
    child_id = tree[N_c1 + child_num][node_id]
    child_id = int(child_id)

    return child_id


def tree_node_get_childs(tree, node_id):
    """

    """
    child_list = []
    arity = tree_node_get_arity(tree, node_id)
    for c in range(arity):
        child_list.append(tree_node_get_child(tree, node_id, c))
    return child_list


def tree_node_get_parent(tree, node_id):
    """

    """
    return tree[N_parent][node_id]


def tree_node_get_modify(tree, node_id):

    return tree[N_modify][node_id]


def tree_node_is_modifiable(tree, node_id):
    modify = tree_node_get_modify(tree, node_id)
    return modify == node_is_modifiable


def tree_node_get_parent_functype(tree, node_id, variables_dict):
    """

    """
    parent_id = tree[N_parent][node_id]
    if tree_node_get_arity(tree, parent_id) > 0:
        parent_label = tree_node_get_label(tree, parent_id)
        fun_type = xtype_get(parent_label, variables_dict)
        return fun_type
    else:
        print_e('That was not a function.')
        raise


def pop_tree_copy(population, tree_id):
    """
    copy a tree from a population
    """
    return np.copy(population[tree_id])


def pop_tree_choose(population):
    """
    Returns a random tree_id from a population
    """
    return np.random.randint(1, len(population))


def pop_copy_genepool(population_tmp, gene_pool, gen_id):
    """
    Copy the genepool of a gen
    """
    pop_y = ['Population Selection in Generation {}.'.format(str(gen_id))]  # empty list

    for i, (tree_num, tree_meta) in enumerate(gene_pool.items()):
        tree_copy = pop_tree_copy(population_tmp, tree_num)
        tree_copy = tree_set_id(tree_copy, i + 1)
        pop_y.append(tree_copy)

    return pop_y


def tree_init_core(node_amount):
    """
    returns an empty tree with an amount of nodes, auto fills
    """
    tree = np.zeros((T_num_lines, node_amount), dtype=np.dtype('U12'))  # U12: longest is observation1

    return tree


def insert_function_or_term(depth, depth_goal):
    """
    with a certain probability, insert terminals or functions
    """
    if np.random.choice(['50', 'larger', 'larger', 'larger', 'larger']) == 'larger':
        probability = np.random.uniform(0, depth_goal)
        if probability > min(depth, depth_goal / 2):
            decision = 'func'
        else:
            decision = 'term'
    else:
        decision = np.random.choice(['term', 'func'])

    return decision


def tree_parsimony(tree, origin_tree=None, parsimony_distance='ted'):
    """
    parsimony_distance: compute the chosen distance by the user.

    """
    if parsimony_distance == 'ted':
        return tree_parsimony_ted(tree, origin_tree)
    elif parsimony_distance == 'total_count_nodes':
        return int(tree[3][-1:])  # returns the tree size
    elif parsimony_distance == 'total_tree_depth':
        return tree[N_depth][1]  # returns the tree size
    elif parsimony_distance == 'total_karoo_original':  # do not use with long variable names
        algo_raw_str = str(tree_get_expr_raw(tree, root_id))
        return len(str(algo_raw_str))
    # elif parsimony_distance == 'total_simplified':
    #     algo_sym = self.tree_expr_sympify(tree=tree)
    #     return count_ops(algo_sym)
    elif parsimony_distance == 'rel_ari_1':  # Does this work?
        return tree_parsimony_relari(tree, origin_tree)
    else:
        raise Exception('Parsimony distance not specified!')


def invent_label_list_depth_random(xtype_root, depth_goal, variables_dict, func_array, min_depth=0):
    """
    build a random, but within itself consistent label list
    Also, return the arities aswell (they are searched anyways)
    """
    todo_xtypes = [xtype_root]
    result_label_list = []
    result_arity_list = []

    # Build a list with labels in row, and a list with their arities
    for depth in range(min_depth, depth_goal):
        next_xtype_list = []

        if depth < depth_goal - 1:
            for xtype in todo_xtypes:

                if insert_function_or_term(depth, depth_goal) == 'term' and depth >= min_depth:
                    label = xtype_choose_term_v2(xtype, variables_dict)
                    arity = 0
                else:
                    label, arity = xtype_choose_func(func_array, xtype=xtype, arity=None)

                # xtype-'To-do' list for the next depth to give values to these functions
                if label == 'Ifte':
                    next_xtype_list.extend(['2b', '2f', '2f'])
                else:
                    tmp_xtype = xtype_get(label, variables_dict)
                    child_type = tmp_xtype[:2][::-1]  # the input of our function "reverted" is the xtype
                    for _ in range(0, arity):  # when arity==2, add 2 times
                        next_xtype_list.append(child_type)

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)
        else:  # now, we are on the lowest dim_y.

            for xtype in todo_xtypes:  # Build terminals now.
                label = xtype_choose_term_v2(xtype, variables_dict)
                arity = 0

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)

        # Finally, update the list for the next round
        todo_xtypes = next_xtype_list[:]

    return result_label_list, result_arity_list


def tree_evolve_insert_branch_v1(tree, branch_ids, variables_dict, func_array, depth_max=None, depth_min=None, depth_goal=None):

    """
    The old depth based version

    """

    # Get information about the top-node we have to replace
    old_label = tree_node_get_label(tree, branch_ids[0])
    old_xtype = xtype_get(old_label, variables_dict)

    # calculate depth restriction
    depth_upper_bound = depth_max - tree_node_get_depth(tree, branch_ids[0])
    depth_goal = min(depth_goal, depth_upper_bound)

    # Build a new tree
    label_list, arity_list = invent_label_list_depth_random(old_xtype, depth_goal, variables_dict, func_array, min_depth=depth_min)

    if label_list:
        core_insert = core_from_labels(label_list, arity_list)
        tree = tree_insert_subtree(tree, core_insert, branch_ids, karoo=True)

    return tree


def tree_evolve_branch_multiple(tree, max_nodes, variables_dict, func_array):
    """
    todo test this function
    todo the layer on a new tree is always root
    # todo could get last non modify layer instead
    insert a (random) number of branches at the first possible "layer"
    (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
    - get these nodes, randomly choose a subset of those
    - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
    - split the amount of nodes up (randomly) and add these new branches to the tree
    # todo idea crossover with same layer functions? backpropagated?
    """

    tree_origin = tree.copy()

    node_ids = tree_get_mutatable_layer(tree, 0)
    max_branches = len(node_ids)
    num_branches = np.random.randint(1, max_branches + 1)

    # get the node ids where we can insert new branches
    insert_ids = []
    num_del_nodes = 0
    insert_indices = random.sample(range(0, max_branches), num_branches)
    for index in insert_indices:
        insert_id = node_ids.pop(index)
        insert_ids.append(insert_id)
        num_del_nodes += len(tree_get_branch(tree, insert_id, karoo=True))

    # split the total amount of nodes we can insert up in several branches
    new_nodes_left = max_nodes - (tree_get_size(tree, karoo=True) - num_del_nodes)
    num_nodes = []
    for i in range(num_branches-1):
        num_new_nodes = np.random.randint(1, (1/i)*new_nodes_left)
        new_nodes_left -= num_new_nodes
        num_nodes.append(num_new_nodes)
    else:
        num_nodes.append(new_nodes_left)

    # finally, insert branches. need to get layer every time as node ids might have changed.
    for enum, i in enumerate(insert_ids):
        node_ids = tree_get_mutatable_layer(tree, 0)
        index = node_ids.index(i)
        node_id = node_ids[index]
        old_branch = tree_get_branch(tree, node_id, karoo=True)
        tree = tree_evolve_insert_branch_v2(tree_origin, old_branch, variables_dict, func_array, max_nodes=num_nodes[enum])  # tree with new branch

    return tree


def tree_evolve_insert_branch_v2(tree, branch_ids, variables_dict, func_array, max_nodes):

    """
    # TODO would be nicer is this just returned a new branch and insert it separately
    replaces the branch_ids in a tree with a new branch
    Given: Tree and a list of node ids
    - checks how far to build down
    - checks the old nodes xtype, etc.
    - checks if we are not too far down the tree
    -

    returns: new tree

    We allow a certain amount of new nodes instead tree depth.
    This could be calculated respectively to the parsimony dim_y
    which the tree might have up his sleeve
    """

    # Get information about the top-node we have to replace
    old_label = tree_node_get_label(tree, branch_ids[0])
    old_xtype = xtype_get(old_label, variables_dict)

    label_list, arity_list = invent_label_list_nodes_grow(old_xtype, max_nodes, variables_dict, func_array)

    if label_list:
        core_insert = core_from_labels(label_list, arity_list)
        tree = tree_insert_subtree(tree, core_insert, branch_ids, karoo=True)

    return tree


def invent_label_list_nodes_grow(xtype, max_nodes, variables_dict, func_array):
    """
    build a random, function (as label list)
    -> labels, arities: ['+', '1.23', '2.34'], [2, 0, 0]
    E. g.: 'float', 5 nodes, min_nodes = 2
    - tbd list: ['2b', '2f']
    - random term_fun_list: ['func', 'term']

    """
    todo_xtypes = [xtype]
    todo_node_amount = 1
    result_label_list = []
    result_arity_list = []
    done = False

    while not done:

        functerm_list = ['func']
        for _ in range(todo_node_amount - 1):  # 1 -> at least one function
            functerm_list.append(np.random.choice(['func', 'term']))
        np.random.shuffle(functerm_list)

        tmp_label_list = ['dummy'] * todo_node_amount
        tmp_arity_list = [8] * todo_node_amount

        func_indices = [i for i, x in enumerate(functerm_list) if x == 'func']
        term_indices = [i for i, x in enumerate(functerm_list) if x == 'term']
        np.random.shuffle(func_indices)

        for enum, index in enumerate(func_indices):
            xtype = todo_xtypes[index]

            label, arity = xtype_choose_func(func_array, xtype=xtype)
            # print('GG', result_label_list, tmp_label_list, '(', len(result_label_list), todo_node_amount, '>', arity, ')', (len(result_label_list) + todo_node_amount + arity), max_nodes)
            if max_nodes > (len(result_label_list)+todo_node_amount) + arity + 1:  # +1 = the start node which we must not forget
                tmp_label_list[index] = label
                tmp_arity_list[index] = arity
                todo_node_amount += arity - 1
            else:
                term_indices.extend(func_indices[enum:])
                done = True
                break

        for index in term_indices:
            label, arity = xtype_choose_term_v2(todo_xtypes[index], variables_dict), 0
            tmp_label_list[index] = label
            tmp_arity_list[index] = arity
            todo_node_amount -= 1

        # prepare next loop
        todo_xtypes = []
        for index, label in enumerate(tmp_label_list):
            if label == 'Ifte':
                todo_xtypes.extend(['2b', '2f', '2f'])
            else:
                xtype = xtype_get(label, variables_dict)
                child_type = xtype[:2][::-1]  # e. g. 'f2b' requires '2f' input
                arity = tmp_arity_list[index]
                todo_xtypes.extend([child_type] * arity)

        result_label_list.extend(tmp_label_list)
        result_arity_list.extend(tmp_arity_list)

    else:
        # Fix the last leftover nodes
        for xtype in todo_xtypes:
            label, arity = xtype_choose_term_v2(xtype, variables_dict), 0
            result_label_list.append(label)
            result_arity_list.append(arity)

    return result_label_list, result_arity_list


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


def tree_single_from_csv(origin_tree_file_path):
    # Load origin from file
    with Path.open(origin_tree_file_path, 'r') as csv_file:
        target = csv.reader(csv_file, delimiter=',')
        tree = np.array([[]])
        for row in target:
            if tree.shape[1] == 0:  # looks if tree is empty
                tree = np.append(tree, [row], axis=1)  # append first row to Tree ('tree_id')
            else:
                tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree
        if tree.shape[0] == T_num_lines:  # (+ row 0)
            pass  # ('Origin Tree is: \n' + str(tree))
        else:
            print_e('Tree could not be imported correctly from .csv file.')
            raise
    return tree


def tree_round_constants(tree, accuracy, karoo=True):
    """
    rounds the values in constant float nodes
    """

    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    for node_id in tree_get_leaves(tree):
        if tree_node_get_nodekind(tree, node_id) == 'term-float':
            tmp = round_constant(tree[N_label][node_id], accuracy)
            tree[N_label][node_id] = tmp

    if karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def tree_get_fitness(tree, precision=None, karoo=True):
    if not karoo:
        raise
    fitness = float(tree[T_fitness][1])

    if precision:
        fitness = round(fitness, precision)
    return fitness


def tree_get_expr_raw(tree, node_id):
    """
    Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').

    """
    node_id = int(node_id)

    if tree[N_arity, node_id] == '0':  # arity of 0 for the pattern '[term]'
        return '(' + tree[N_label, node_id] + ')'  # 'node_label' (function or terminal)

    elif tree[N_arity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        fun = tree[N_label, node_id]
        if fun == '~':  # ~- workaround
            return '(-({}))'.format(tree_get_expr_raw(tree, tree[9, node_id]))
        else:
            return '(' + fun + tree_get_expr_raw(tree, tree[9, node_id]) + ')'

    elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
        # This if case is for 2-ary ops that is prefix. like Min(a, b)
        if tree[N_label, node_id] not in functions_infix_dict:
            return '(' + tree[N_label, node_id] + '(' + tree_get_expr_raw(tree, tree[9, node_id]) + ', ' + tree_get_expr_raw(tree, tree[10, node_id]) + '))'
        else:
            return '(' + tree_get_expr_raw(tree, tree[9, node_id]) + tree[N_label, node_id] + tree_get_expr_raw(tree, tree[10, node_id]) + ')'  # Klammern, da sympify sonst abkacnen könnte

    elif tree[N_arity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return '(Ifte(' + tree_get_expr_raw(tree, tree[9, node_id]) + ', ' + tree_get_expr_raw(tree, tree[10, node_id]) + ', ' + tree_get_expr_raw(tree, tree[11, node_id]) + '))'


def tree_parsimony_ted(tree1, tree2):
    """
    The Tree Edit distance (TED) ('coolest' distance)
    - the amount of changes that have to be applied to the origin to equality are counted
    """
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
    if tree is None and algo_raw is None:
        print_e('Either tree or algo raw have to be set.')
    if algo_raw is None:  # If we got a tree, we generate the expression
        algo_raw = str(tree_get_expr_raw(tree, root_id))

    try:
        expr_sym = plagih_sympify(algo_raw)
        expr_sym_str = str(expr_sym)
    except Exception as ex:
        raise Exception('In sympify. Caused by this raw algorithm: {}. Ex: {}'.format(algo_raw, ex))

    for fail_reason in ['zoo', 'inf', '*I', 'nan']:
        if fail_reason in expr_sym_str:
            raise Exception('Sympify failed due to a fail reason: {}.'.format(fail_reason))
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


def tree_get_last_nodeid(tree):
    """
    returns the total amount of nodes in a tree
    """
    return int(tree[3][-1:])  # returns last node_id


def tree_get_leaves(tree, karoo=False):
    """
    Just return leaf nodes of a tree
    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    node_ids = []
    for node_id in tree[N_id]:
        if tree_node_get_arity(tree, int(node_id)) == 0:
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
            print_warning('ww', 'Tree has <=2 nodes. Change configuration. {}'.format(tree))
        if tree[N_arity][n] == '0':  # check for discrepency
            # tree[N_arity][n] = '0'  # set arity to 0
            tree[N_c1][n] = ''  # wipe 'node_c1'
            tree[N_c2][n] = ''  # wipe 'node_c2'
            tree[N_c3][n] = ''  # wipe 'node_c3'
            tree[N_modify][n] = '1'

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


def tree_core_init_depth(tree, parent_list=None):
    """
    Automatically filly node depth
    - Tree needs:
        - c1, c2, c3 filled
    """

    if not parent_list:
        parent_list = tree_row_int(tree, N_c1)

    if len(tree) > 0:
        tree[N_depth][0] = 0  # the root is always here
    else:
        print_warning('w', 'Tree is completely empty')

    for my_id, _ in enumerate(parent_list):

        child_depth = int(tree[N_depth][my_id]) + 1
        c1 = tree[N_c1][my_id]
        c2 = tree[N_c2][my_id]
        c3 = tree[N_c3][my_id]
        for c in [c1, c2, c3]:
            if c != '':
                tree[N_depth][int(c)] = child_depth
    return tree


def tree_core_init_parents(tree, arity_lst=None):
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


def tree_core_init_c(tree, parent_list=None):
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
        my_id = i  # + 1  #
        parent_id = val  # - 1
        if val >= 0:

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


def tree_core_init_row(tree, row_id, row):
    """
    sets any complete row in a np-tree
    """
    for i, x in enumerate(row):
        tree[row_id][i] = x
    return tree


def tree_convert_plusnode(tree, add_or_sub, firstrow=1):
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


def core_from_labels(label_list, arity_list=None):
    """
    Given the labels (and label infos) as list
    this function builds the core of a tree (no node_modify)
    """
    if len(label_list) == 0:
        print_warning('w', 'label list is empty')

    if not arity_list:
        arity_list = [label_get_arity(label) for label in label_list]

    size = len(label_list)
    tree = tree_init_core(size)

    # set all the rows that are super easy
    tree = tree_core_init_row(tree, N_id, [x for x in range(0, size)])
    tree = tree_core_init_row(tree, N_label, label_list)
    tree = tree_core_init_row(tree, N_arity, arity_list)

    # and also, fill all the leftover rows
    tree, parent_list = tree_core_init_parents(tree)
    tree = tree_core_init_c(tree)
    tree = tree_core_init_depth(tree, parent_list)

    if not tree_test_check_children(tree, karoo=False):
        print_e('Tree from label_list {} is not correct: {}'.format(label_list, tree))
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
                parent_arity_sum = parent_arity_sum + int(tree[N_arity][n])  # sum arities of parents at  prior depth

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
    tree_nofirst_nodezero = tree_convert_plusnode(tree_nofirst, add_or_sub=-1, firstrow=0)
    tree_nofirst_nodezero[N_parent][0] = -1

    return tree_nofirst_nodezero


def tree_convert_plagih_to_karoo(plagih_tree):
    """
    tests has a first row with nonsense and nodes start with 1
    plagih has no first row and nodes start with 0
    """
    first_col = tree_init_first_column()
    tree_withfirst = np.concatenate((first_col, plagih_tree), axis=1)
    tree_karoo = tree_convert_plusnode(tree_withfirst, add_or_sub=1, firstrow=1)
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


def tree_fix_link_child_karoo(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'config.tree' has been modified, as with both Grow and Full mutation.

    """

    for node_id in range(root_id, len(tree[3])):
        c_buffer = evolve_c_buffer(tree, node_id, karoo=True)  # generate c_buffer for each node
        tree = tree_node_set_childs_ids(tree, node_id, c_buffer, karoo=True)  # update child links for each node

    return tree


def tree_fix_link_child(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'config.tree' has been modified, as with both Grow and Full mutation.

    """

    for node in range(0, len(tree[N_id])):
        c_buffer = evolve_c_buffer(tree, node)  # generate c_buffer for each node
        tree = tree_node_set_childs_ids(tree, node, c_buffer)  # update child links for each node

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


def treegp_reduce_branch(tree, node_id, karoo=False):
    delete_ids = tree_get_branch(tree, node_id, karoo=karoo)
    expr_raw = tree_get_expr_raw(tree, node_id)
    try:
        expr_sym = tree_expr_sympify(algo_raw=expr_raw)
        label_list = ast_convert_from_expr(expr_sym, build=True)
        arity_list = [label_get_arity(label) for label in label_list]  # todo zeile auslagern?
        core = core_from_labels(label_list, arity_list)
        tree_sympified = tree_insert_subtree(tree, core, delete_ids, karoo=True)

        return tree_sympified
    except:
        print_warning('w', 'reducing expr raw: {}'.format(expr_raw))
        print_warning('w', 'Delete this tree! nan tree or other error.')
        return None


def tree_evolve_mutate_point(tree, func_array, variables_dict):

    """
    Mutate a single mutatable point in any Tree.
    """

    # 1. choose a node
    node_ids = tree_get_mutatable_nodes(tree)
    node_id = np.random.choice(node_ids)
    label, arity, xtype = tree_node_get_lax(tree, node_id, variables_dict)

    if arity > 0:
        new_label, new_arity = xtype_choose_func(func_array, xtype=xtype, arity=arity)  # Function is same type, same arity
        tree = tree_node_set_label(tree, node_id, new_label)
    else:  # arity == 0:  # aka a terminal
        new_label = xtype_choose_term_v2(xtype, variables_dict)  # 3 -> '2f' -> 5
        tree = tree_node_set_label(tree, node_id, new_label)

    return tree  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping


def tree_evolve_reduce_parts(tree, completely=True):

    """
    Mutate a single mutatable point in any Tree.
    """
    if completely:  # reduce the complete tree
        nodes_lv0 = tree_get_mutatable_layer_lv0(tree)
        for node_id in nodes_lv0:
            tree = treegp_reduce_branch(tree, node_id, karoo=True)
    else:  # only choose one node to be reduced
        node_ids = tree_get_mutatable_nodes(tree)
        func_ids = [x for x in node_ids if tree_node_get_arity(tree, x) > 0]
        if len(func_ids) > 0:
            node_id = np.random.choice(node_ids)  # choose
            tree = treegp_reduce_branch(tree, node_id, karoo=True)
    return tree


def tree_get_mutatable_nodes(tree, no_root=False, karoo=True):
    """
    Returns a list with mutatable ids
    """

    node_ids = []
    for node_id in tree_get_ids(tree, karoo=karoo):
        if node_id == 'node_modify':
            continue
        if tree[N_modify][node_id] == '1':
            node_ids.append(int(node_id))

    if no_root and root_id in node_ids:
        node_ids.remove(root_id)

    return node_ids


def tree_get_fix_nodes(tree, karoo=True):
    """
    Returns a list with mutatable ids
    """

    node_ids = []

    for node_id in tree_get_ids(tree, karoo=karoo):
        if not tree_node_is_modifiable(tree, node_id):
            node_ids.append(int(node_id))

    return node_ids


def labels_get_aritys_list(label_list, karoo=False):
    """
    returns an arity list for a label list
    """

    arity_list = [label_get_arity(x) for x in label_list]

    if karoo:
        arity_list.pop(0)
    return arity_list


def tree_get_branch(tree, node, karoo=False):
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


def tree_get_branch_lax(tree, node_id, karoo=True):
    """
    returns all ids, labels and arities for a node in a tree
    """
    ids = tree_get_branch(tree, node_id, karoo=karoo)
    labels = [tree[N_label][i] for i in ids]
    aritys = [tree[N_arity][i] for i in ids]
    return ids, labels, aritys


def tree_pretty_print(tree, karoo=False):
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    depth = 0
    layer_labels = []
    for i, n_depth in enumerate(tree[N_depth]):
        label = tree_node_get_label(tree, i)
        if int(n_depth) == depth:
            layer_labels.append(label)
        else:
            layer_labels = [label]
            depth += 1
    else:
        print(layer_labels)

    return


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


def tree_check_child_xtype(tree, variables_dict, karoo=True):
    """
    A method to check if a tree is plausible. aka:
    - do the values in c1, c2, c3 link to correct
    """
    if not karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    for node_id in range(1, len(tree[3])):
        label = tree_node_get_label(tree, node_id)
        xtype = xtype_get(label, variables_dict)
        arity = label_get_arity(label)
        test_xtype = xtype_get_child_todos(label, arity, variables_dict)

        for c in range(0, 3):
            if tree[N_c1 + c][node_id] != '':
                c_node_id = tree[N_c1 + c][node_id]
                c_label = tree_node_get_label(tree, c_node_id)
                c_xtype = xtype_get(c_label, variables_dict)
                # if c_xtype != test_xtype[c]:
                if not xtype_equi_outcome(c_xtype, test_xtype[c]):
                    print_blue('Label {}, child {} with c_label {} does not match xtype {}. It is c_xtype {}.\ntree labels: {}'.format(label, c, c_label, xtype, c_xtype, tree[N_label]))
                    return False

    return True


def tree_delete_nodes(tree, node_list):
    tree = np.delete(tree, node_list, axis=1)  # delete all branches below
    return tree


def tree_check_expression(tree, karoo=True):
    """
    Check if a valid tree can be built from the expression
    """

    label_list = tree[N_label]
    arity_list = tree[N_arity]

    if karoo:
        label_list = label_list[1:]
        arity_list = arity_list[1:]

    try:
        core = core_from_labels(label_list, arity_list)
        if core:
            return True
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


def tree_evolve_node_insert(tree, variables_dict):
    """
    Inserts a (arity-1) node into a tree
    Especially useful when a ** power shall be normalized
    """

    node_ids = tree_get_mutatable_nodes(tree)
    insert_id = None
    for node_id in node_ids:
        label = tree_node_get_label(tree, node_id)
        xtype = xtype_get(label, variables_dict)  # '>' -> 'f2b'
        if label == '**' and tree_node_get_child(tree, node_id, 1) != 'Power':  # todo
            insert_id = node_id
            break

    if insert_id:
        old_ids, old_labels, old_aritys = tree_get_branch_lax(tree, insert_id)
        new_labels = ['Power'] + old_labels
        new_aritys = [1] + old_aritys
        insert_core = core_from_labels(new_labels, new_aritys)
        tree_insert_subtree(tree, insert_core, old_ids, karoo=False)

    return tree


def tree_evolve_complexify(tree, same_arity=True):
    """
    todo
    a function that inserts certain functions that hopefully give good opportunities for next generations
    eg: in old_node '+', inserting Ifte(True, '+', 1.23) or so...
    """
    pass


def gp_mutate_constants(constant, term_type=None, filter_type='gaussian_filter'):
    """
    When this happens, constants get a a small variance
    """

    if term_type == 'float':
        if filter_type == 'gaussian_filter':
            constant = np.random.normal(constant, 0.1)
        else:
            print_warning('w', 'Warning: Filter  not specified. Please specify a filter_type.')
            constant = np.random.normal(constant, 0.1)

    if term_type == 'int':
        constant = int(np.random.normal(constant, 2))

    if term_type == 'bool':
        constant = not constant
        # random by 50:50?

    return constant

def tree_evolve_mutate_filter_one(tree):
    """
    Mutates one float terminal of a tree
    """
    # 1. choose a node
    node_ids = tree_get_mutatable_nodes(tree)
    float_nodes = []
    for node_id in node_ids:
        label = tree_node_get_label(tree, node_id)
        if xtype_get_constant(label) == '2f':
            float_nodes.append(node_id)
    if float_nodes:
        # todo modify multiple float nodes at once?
        float_id = np.random.choice(float_nodes)
        val = float(tree_node_get_label(tree, float_id))
        new_value = gp_mutate_constants(val, term_type='float', filter_type='gaussian_filter')
        tree = tree_node_set_label(tree, float_id, new_value)
        return tree
    else:
        raise Exception('No mutatable node found!')
        # return None

def tree_evolve_tree_prune(tree, max_depth, variables_dict):
    """
    reduces the depth of a Tree (in case it is too deep).
    Arguments required: tree, depth
    """

    nodes = []

    for node_id in range(root_id, len(tree[3])):

        node_depth = tree_node_get_label(tree, node_id)
        node_arity = tree_node_get_arity(tree, node_id)
        if node_depth == max_depth and node_arity > 0:  # replace this node with terminal
            label = tree_node_get_label(tree, node_id)
            node_xtype = xtype_get(label, variables_dict)
            tree = tree_node_set_arity(tree, node_id, 0)
            new_term = xtype_choose_term_v2(node_xtype, variables_dict)  # replace label
            tree = tree_node_set_label(tree, node_id, new_term)

        elif tree_node_get_depth(tree, node_id) > max_depth:  # record nodes deeper than the maximum allowed Tree depth
            nodes.append(node_id)

    tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
    tree = evolve_node_arity_fix(tree)  # fix all node arities

    return tree


def tree_get_ids(tree, skip_nodes=0, karoo=True):
    if karoo:
        start = 1 + skip_nodes
    else:
        start = 0 + skip_nodes
    node_id_list = [int(node_id) for node_id in tree[N_id][start:]]
    return node_id_list


def tree_iterate_range(tree, karoo=True):
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


def tree_normalize_exponentiation(tree):

    # 1. ** should have an int as second number
    for node_id in tree_get_ids(tree, karoo=True):
        if tree_node_get_label(tree, node_id) == '**':
            child_id = tree_node_get_child(tree, node_id, 1)  # get second argument
            old_power = tree_node_get_label(tree, child_id)
            try:
                new_power = float(int(float(old_power)))
                tree = tree_node_set_label(tree, child_id, new_power)
            except ValueError:
                pass  # sfeh: This may actually take some time. Every tree gets checked any many have '**'.
    return tree


def tree_get_mutatable_layer_lv0(tree):
    """
    Returns a list with mutatable ids on the outside
    """

    node_ids = []
    fix_ids = tree_get_fix_nodes(tree)
    if len(fix_ids) == 0:
        node_ids = [root_id]
    else:
        for node_id in fix_ids:

            child_ids = tree_node_get_childs(tree, node_id)
            for child_id in child_ids:
                if tree_node_is_modifiable(tree, child_id):
                    node_ids.append(child_id)

    return node_ids


def tree_get_mutatable_extendables(tree):
    """
    Returns a list with mutatable ids on the outside
    """
    fix_ids = tree_get_fix_nodes(tree)
    leaf_ids = []
    for node_id in fix_ids:

        arity = tree_node_get_arity(tree, node_id)
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


def tree_get_mutatable_layer(tree, lvl_goal, sum_layers=False, get_closest=False):
    """
    Returns a list with mutatable ids which are *lvl_goal* layers away from non modifiable nodes
    last_leaves: if you want so save all leave nodes aswell
    """

    lvl_count = 0
    layer_lists = [tree_get_mutatable_layer_lv0(tree)]

    while len(layer_lists[lvl_count]) > 0:

        next_ids = []
        for layer_id in layer_lists[lvl_count]:
            next_ids.extend(tree_node_get_childs(tree, layer_id))

        if next_ids:
            layer_lists.append(next_ids)
        else:
            break
        lvl_count += 1

    if get_closest:
        lvl_best = min(lvl_count, lvl_goal)
    else:
        lvl_best = lvl_goal

    if sum_layers:
        result_ids = sum(layer_lists[:lvl_best+1], [])
    else:
        result_ids = layer_lists[lvl_best]

    return result_ids


