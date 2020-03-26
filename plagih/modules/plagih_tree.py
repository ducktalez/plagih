import os
import numpy as np
from plagih.modules.plagih_sympy_extras import plagih_sympify
from plagih.tree_distances.tree_edit_distance import apted_distance
from plagih.modules.plagih_types import *
from plagih.modules.plagih_eval import *
import csv
from plagih.modules.viz_with_latex import *
from sympy import sympify
import copy

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees

N_label = 6
N_id = 3
N_depth = 4
N_type = 5
N_parent = 7
N_arity = 8
N_c1 = 9
N_c2 = 10
N_c3 = 11
N_modify = 13

TR_ID = 0  # todo I think the id is irrelevant
TR_type = 1  # todo I think the type is irrelevant
T_fitness = 12
T_parsimony = 14

T_num_lines = 15
root_id = 1

node_is_modifiable = 1


class Plagih_Tree():
    """
    Plagih trees are computational trees that hold the genetic programs.
    # todo the same expr_raw can originate from many trees. not good for the meta dict

    What is a trees primary identificable?
    - the alignment of labels: [+, a, b]

    What is additonal node-info we need?
    - Node positioning info:
    - modifiable nodes

    What is the trees meta data?:
    - fitness
    - parsimony
    - expr_raw
    - expr_sym
    (- last modifications)
    (- last parsimony)
    (- other complexity measurement?)
    ((- its last versions tree data))  # -> needs too much memory?
    ((- its last fitness, parsimony))

    What run-specific data is irrelevant?
    - pop_id (NO!)
    """

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
        arity_list = [label_get_arity(label) for label in label_list]  # ~- problem: fine. [-, 1, 2] vs [*, 1, -2]
        core = core_from_labels(label_list, arity_list)
        if modify_list:
            for i, val in enumerate(modify_list):
                core[N_modify][i] = val
        else:  # all can be modified
            for i, val in enumerate(label_list):
                core[N_modify][i] = 1
        self.tree = tree_convert_plagih_to_karoo(core)

        return

    def get_uninstanced_tree(self):
        return self.tree

    def write_to_file(self, path):
        pass


class Plagih_node():

    def __init__(self, n_id, depth, n_type, label, parent, arity, c1, c2, c3):
        return


def karoo_tree_from_labellist(label_list, modify_list=None):
    """
    returns: tree, from label_list (newest version)
    """
    p_tree = Plagih_Tree(label_list, modify_list=modify_list)
    tree = p_tree.get_uninstanced_tree()
    return tree


def karoo_tree_from_expr(expr, modify_list=None):
    """
    DELETE later sfeh
    Generate tree from a raw or sympified expression
    """
    label_list = ast_convert_from_expr(expr, build=True)
    p_tree = Plagih_Tree(label_list, modify_list=modify_list)
    tree = p_tree.get_uninstanced_tree()
    return tree


def tree_from_load_numpycsv(origin_tree_file_path=None):
    """
    returns: tree, old karoo version
    """

    if origin_tree_file_path:
        tree = tree_single_from_csv(origin_tree_file_path)
    else:
        print_warning('ww', 'No origin provided. Starting from scratch with random generation.')
        tree = None

    return tree


def tree_save_csv(tree, path_csv):
    """
    Writing one tree to a .csv file. As it is appended, many can be added.
    """
    with Path.open(path_csv, 'a', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
        target = csv.writer(csv_file, delimiter=',')

        target.writerows([''])  # empty row before each Tree
        for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
            target.writerows([tree][row])


def tree_check_expr(tree):
    """
    todo make this look better later...
    """
    expr_raw = tree_get_expr_raw(tree, node_id=root_id)

    try:
        expr_sym = expr_sympify(expr_raw=expr_raw)
    except:
        raise Exception('Your tree\'s algorithm could not be sympified. Aborting.')

    return


def load_pop_from_csv(pop_csv):
    """
    This method is used to load a saved population of Trees, as invoked through the (pause) menu where population_r
    replaces population_a in the karoo_gp/runs/[date-time]/ directory.
    """

    with Path.open(pop_csv, 'r') as csv_file:
        target = csv.reader(csv_file, delimiter=',')
        n = 0  # track row count

        for row in target:
            n = n + 1
            if n == 1:
                pass  # skip first empty row
            elif n == 2:
                tree_list = [row]  # write header to population_a
            else:
                if not row:
                    tree = np.array([[]])  # initialise Tree array
                else:
                    if tree.shape[1] == 0:
                        tree = np.append(tree, [row], axis=1)  # append first row to Tree
                    else:
                        tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree

                if tree.shape[0] == T_num_lines:
                    tree_list.append(tree)  # append complete Tree to population list

    return tree_list


def tree_get_labellist(tree):
    """
    Returns all tree labels in order
    these identify a tree completely (if the tree is fully functioning)
    """
    label_list = tree[N_label]
    return label_list


def tree_get_size(tree, karoo=True):
    if karoo:
        size = len(tree[0])
        return size
    else:
        return 0


def tree_get_history(tree):
    return tree[TR_type][1]


def tree_set_id(tree, tree_id):
    """
    Set the tree's id, aka the number in the population.
    But we could also enumerate over  the population. not needed.
    sfeh delete this?
    """
    # tree[TR_ID][1] = tree_id
    print_warning('w', 'This function is not in use!')
    return tree


def tree_set_last_evolution(tree, last_modification):
    tree[TR_type][1] = last_modification
    return tree


def tree_check_xtypes(tree):
    for node in tree_iterate_range(tree):
        if tree[N_type][node] == '':  # are xtypes set?
            return False
    return True


def tree_set_xtypes(tree, variables_dict):
    """
    Set xtype for all nodes in the tree.
    Faster than 'looking up' the xtype every time with xtype_get_from_label which needs extra dicts
    :param tree:
    :param variables_dict:
    :return:
    """
    for node_id in tree_nodes_get_ids(tree):
        label = tree_node_get_label(tree, node_id)
        xtype = xtype_get_from_label(label, variables_dict)
        tree = tree_node_set_xtype(tree, node_id, xtype)
    return tree


def tree_set_fitness(tree, fitness, precision=6):
    """
    Store the fitness within the tree np-array

    """
    if fitness != '':
        fitness = float(fitness)
        fitness = round(fitness, precision)

    tree[T_fitness][1] = fitness  # store the fitness with each tree

    return tree


def tree_set_parsimony(tree, parsimony):
    """
    Store the parsimony within the tree np-array
    """
    if parsimony == '':
        pass  # This is actually done when 'wiping' the tree's data
    elif parsimony < 0:
        print_warning('w', 'Warning: Parsimony is: {}'.format(parsimony))

    tree[T_parsimony][1] = parsimony
    return tree


def tree_set_modifyable_nodes_true(tree, karoo=True):
    """

    """

    for node_id in tree_nodes_get_ids(tree, karoo=karoo):
        tree[N_modify][node_id] = 1
    return tree


def tree_set_meta(tree, tree_meta):
    """
    When having the meta data, save it in the tree.
    """
    parsimony = tree_meta['parsimony']
    fitness_train = tree_meta['fitness_train']
    expr_sym = tree_meta['expr_sym']
    expr_raw = tree_meta['expr_raw']

    tree = tree_set_parsimony(tree, parsimony)
    tree = tree_set_fitness(tree, fitness_train)
    # tree = tree_set_expr_sym(tree, expr_sym) # todo, also at get method
    # tree = tree_set_expr_raw(tree, expr_raw) # todo
    return tree


def tree_node_set_xtype(tree, node_id, xtype):
    tree[N_type][node_id] = xtype
    return tree


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

    if karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    return tree


def tree_node_set_label(tree, node_id, label):
    tree[N_label][int(node_id)] = label
    return tree


def tree_node_set_arity(tree, node_id, arity):
    tree[N_arity][int(node_id)] = int(arity)
    return tree


def tree_node_set_modify(tree, node_id, value):
    """

    """
    tree[N_modify][node_id] = value

    return tree


def tree_nodes_get_ids_string(tree, node_id):
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
        return '{}, {}'.format(tree[3, node_id], tree_nodes_get_ids_string(tree, tree[9, node_id]))

    elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[node_id], [node_id], [node_id]'
        return '{}, {}, {}'.format(
            tree[3, node_id],
            tree_nodes_get_ids_string(tree, tree[9, node_id]),
            tree_nodes_get_ids_string(tree, tree[10, node_id]))

    elif tree[N_arity, node_id] == '3':  # arity of 3 for the pattern '[node_id], [node_id], [node_id], [node_id]'
        return '{}, {}, {}, {}'.format(
            tree[3, node_id],
            tree_nodes_get_ids_string(tree, tree[9, node_id]),
            tree_nodes_get_ids_string(tree, tree[10, node_id]),
            tree_nodes_get_ids_string(tree, tree[11, node_id]))


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
    special node-type 'nodekind'
    'func', 'term-variable', 'term-float', 'term-bool'
    """
    arity = tree_node_get_arity(tree, node)
    if arity > 0:
        nodekind = 'func'
    else:
        label = tree[N_label][node]
        if name_observation in label:  # 'observation'
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


def tree_node_get_lax(tree, node_id):
    label = tree_node_get_label(tree, node_id)
    arity = tree_node_get_arity(tree, node_id)
    # xtype = xtype_get_from_label(label, variables_dict)
    xtype = tree_node_get_xtype(tree, node_id)
    return label, arity, xtype


def tree_node_get_lax_v3(tree, node_id):
    """
    no need for variables dict!
    """
    label = tree_node_get_label(tree, node_id)
    arity = tree_node_get_arity(tree, node_id)
    xtype = tree_node_get_xtype(tree, node_id)
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
    """

    """
    modify = tree[N_modify][node_id]
    if modify == '':
        modify = 1
    else:
        modify = int(modify)
    return modify


def tree_node_all_info(tree, node_id):
    """
    All info in the column of a node
    """
    node_info = {'node_id': node_id,
                 'label': tree_node_get_label(tree, node_id),
                 'parent': tree_node_get_parent(tree, node_id),
                 'childs': tree_node_get_childs(tree, node_id),
                 'modify': tree_node_get_modify(tree, node_id),
                 'xtype': tree_node_get_xtype(tree, node_id),
                 'arity': tree_node_get_arity(tree, node_id),
                 'depth': tree_node_get_depth(tree, node_id),
                 'kind': tree_node_get_nodekind(tree, node_id)}

    return node_info


def tree_node_debug_print(tree, node_id):
    """
    print some node-info, maybe also tree info
    """
    node_parent = tree_node_get_parent(tree, node_id)
    # parent_info = tree_node_all_info(tree, node_id)
    debug_print = 'Tree node_id {}: \n' \
                  'Node-info: {}\n' \
                  'Tree_labels: {}\n' \
                  'Tree-modify:'.format(node_id, tree_node_all_info(tree, node_id), node_parent, tree_get_labellist(tree), tree[N_modify])
    return debug_print


def tree_set_modifyable_nodes(tree, origin_tree=None):
    """
    Sets all the origin_meta core nodes back to non-modifyable
    """

    tree = tree_set_modifyable_nodes_true(tree)

    if origin_tree is not None:  # todo loop is unneccessary if there are no set fix nodes
        non_modifiable_nodes = []
        if tree_node_get_modify(origin_tree, root_id) == 0:  # check if modifiable nodes are specified
            non_modifiable_nodes.extend(tree_permanent_nodes_get(1, tree, 1, origin_tree))

        for non_modifiable in non_modifiable_nodes:
            tree = tree_node_set_modify(tree, non_modifiable, 0)

    return tree


def tree_permanent_nodes_get(origin_node, chosen_tree, chosen_node, origin_tree):
    """
    Returns a list of nodes that are not supposed to be modified
    """

    if tree_node_get_modify(origin_tree, origin_node) == 0:

        permanent_nodes = [chosen_node]
        for c in [N_c1, N_c2, N_c3]:
            if origin_tree[c][origin_node] != '':  # aka a child exists
                next_origin_node = int(origin_tree[c][origin_node])
                next_chosen_node = int(chosen_tree[c][chosen_node])
                tmp = tree_permanent_nodes_get(next_origin_node, chosen_tree, next_chosen_node, origin_tree)
                if tmp is not None:
                    permanent_nodes.extend(tmp)
        return permanent_nodes
    else:
        return


def tree_node_is_variable(tree, node_id):
    label = tree_node_get_label(tree, node_id)
    return name_observation in label


def tree_node_is_modifiable(tree, node_id):
    """
    returns True if modifiable is 1
    """
    modify = tree_node_get_modify(tree, node_id)
    return modify == 1


def tree_node_get_parent_functype(tree, node_id, variables_dict):
    """

    """
    parent_id = tree[N_parent][node_id]
    if tree_node_get_arity(tree, parent_id) > 0:
        parent_label = tree_node_get_label(tree, parent_id)
        fun_type = xtype_get_from_label(parent_label, variables_dict)
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
    return np.random.randint(FIRST_TREE, len(population))  # 1-len is correct. Tested it several times now.


def tree_init_core(node_amount):
    """
    returns an empty tree with an amount of nodes, auto fills
    """
    tree = np.zeros((T_num_lines, node_amount), dtype=np.dtype('U12'))  # U12: longest is observation1

    return tree


def insert_function_or_term(depth, depth_goal):
    """
    on every tree depth
    """
    if np.random.choice(['50', 'larger', 'larger', 'larger']) == 'larger':  # todo
        if np.random.uniform(0, depth_goal) > min(depth, depth_goal / 2):
            decision = 'func'
        else:
            decision = 'term'
    else:
        decision = np.random.choice(['term', 'func'])

    return decision


def invent_label_list_depth_random(xtype_root, depth_goal, variables_dict, func_array, min_depth=0, build_mode='grow'):
    """
    build a random, but within itself consistent label list
    Also, return the arities aswell (they are searched anyways)
    """
    tbdo_xtypes = [xtype_root]
    result_label_list = []
    result_arity_list = []

    # Build a list with labels in row, and a list with their arities
    for depth in range(min_depth, depth_goal):
        next_xtype_list = []

        if depth < depth_goal - 1:
            for xtype in tbdo_xtypes:
                if build_mode == 'grow':
                    if insert_function_or_term(depth, depth_goal) == 'term' and depth >= min_depth:
                        label, arity = xtype_choose_term_v2(xtype, variables_dict), 0
                    else:
                        label, arity = xtype_choose_func(func_array, xtype=xtype, arity=None)
                elif build_mode == 'full':
                    label, arity = xtype_choose_func(func_array, xtype=xtype, arity=None)
                else:
                    raise

                # xtype-'To-do' list for the next depth to give values to these functions
                if label == 'Ifte':
                    next_xtype_list.extend(['2b', '2f', '2f'])
                else:
                    tmp_xtype = xtype_get_from_label(label, variables_dict)
                    child_type = tmp_xtype[:2][::-1]  # the input of our function "reverted" is the xtype
                    for _ in range(0, arity):  # when arity==2, add 2 times
                        next_xtype_list.append(child_type)

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)
        else:  # now, we are on the lowest dim_y.

            for xtype in tbdo_xtypes:  # Build terminals now.
                label, arity = xtype_choose_term_v2(xtype, variables_dict), 0

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)

        # Finally, update the list for the next round
        tbdo_xtypes = next_xtype_list[:]

    return result_label_list, result_arity_list


def tree_evolve_insert_branch_v1(tree, branch_ids, variables_dict, func_array, depth_max=None, depth_min=None, depth_goal=None):
    """
    # The old depth based version
    # Not used anymore, as the amount of nodes is much more useful
    Given: Tree and a list of node ids
    - checks how far to build down
    - checks the old nodes xtype, etc.
    - checks if we are not too far down the tree
    -

    """

    # Get information about the top-node we have to replace
    old_label = tree_node_get_label(tree, branch_ids[0])
    old_xtype = xtype_get_from_label(old_label, variables_dict)

    # calculate depth restriction
    depth_upper_bound = depth_max - tree_node_get_depth(tree, branch_ids[0])
    depth_goal = min(depth_goal, depth_upper_bound)

    build_mode = np.random.choice(['full', 'grow'])  # todo test full method
    # Build a new tree
    label_list, arity_list = invent_label_list_depth_random(old_xtype, depth_goal, variables_dict, func_array, min_depth=depth_min, build_mode=build_mode)

    if label_list:
        core_insert = core_from_labels(label_list, arity_list)
        tree = tree_insert_subtree(tree, core_insert, branch_ids, karoo=True)

    return tree


def randomly_split_range(range_max, num_splits):
    """
    split a integer range randomly into parts
    [1..100] -> [33, 15, 52] (0 is allowed)
    """

    # tmp_distributions = random.sample(range(1, range_max), num_splits)
    # d_sum = sum(tmp_distributions)
    # d_list = [int(round(range_max*(x/d_sum), 0)) for x in tmp_distributions]
    sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
    d_sum = sum(sample_dist)  # 1.5
    sample_dist = [x / d_sum for x in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
    sample_dist = [x * range_max for x in sample_dist]  # [12, 60, 28] -> for 100 nodes
    sample_dist = [int(round(x, 0)) for x in sample_dist]  # make them useable ints

    # sfeh workaround, this makes exactly the correct range by changing the most extreme entry
    helper_diff = range_max - sum(sample_dist)
    if sum(sample_dist) < range_max:
        smallest = sample_dist.index(min(sample_dist))
        sample_dist[smallest] += helper_diff

    if sum(sample_dist) > range_max:
        greatest = sample_dist.index(max(sample_dist))
        sample_dist[greatest] += helper_diff

    return sample_dist


def tree_evolve_branch_multiple(tree, goal_nodes, variables_dict, func_array):
    """
    insert a (random) number of branches at the first possible "layer"
    (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
    - get these nodes, randomly choose a subset of those
    - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
    - split the amount of nodes up (randomly) and add these new branches to the tree
    todo fix min and max border
    """

    tree_base = tree.copy()
    layer0_ids = tree_get_mutatable_layer(tree, 0)  # ('We are about to create new branches randomly at nodes {}.'.format(layer0_ids))
    nodes_left = goal_nodes  # sfeh - max_nodes-tree_get_size(tree, karoo=True))  # ('Which lets us replace {} amount of old nodes'.format(nodes_left))

    num_nodes_split = randomly_split_range(nodes_left, len(layer0_ids))

    for i in range(len(layer0_ids)):  # finally, insert branches. need to get layer every time as node ids might have changed.
        layer0_ids = tree_get_mutatable_layer_lv0(tree)
        node_id = layer0_ids[i]
        old_branch = tree_get_branch(tree, node_id, karoo=True)
        tree = tree_insert_branch_v2(tree_base, old_branch, variables_dict, func_array, goal_nodes=num_nodes_split[i])  # tree with new branch
    return tree


def raise_if_empty(name, val):
    if val == '' or val is None:
        print('This variable did not work'.format(name))
        raise


def tree_insert_branch_v2(tree, branch_ids, variables_dict, func_array, goal_nodes):
    """
    replaces the branch_ids in a tree with a new branch

    returns: new tree

    We allow a certain amount of new nodes instead tree depth.
    This could be calculated respectively to the parsimony dim_y
    which the tree might have up his sleeve
    """

    # Get information about the top-node we have to replace
    # old_label = tree_node_get_label(tree, branch_ids[0])
    old_xtype = tree_node_get_xtype(tree, branch_ids[0])
    raise_if_empty('old_xtype', old_xtype)
    # old_xtype = xtype_get_from_label(old_label, variables_dict)

    label_list, arity_list = invent_label_list_nodes_grow(old_xtype, goal_nodes, variables_dict, func_array, build_type='grow')

    if label_list:
        core_insert = core_from_labels(label_list, arity_list)
        tree = tree_insert_subtree(tree, core_insert, branch_ids, karoo=True)

    return tree


def invent_label_list_nodes_grow(xtype, goal_max_nodes, variables_dict, func_array, build_type='grow'):
    """
    build a random function (as label list)
    -> labels, arities: ['+', '1.23', '2.34'], [2, 0, 0]
    E. g.: 'float', 5 nodes, min_nodes = 2
    - tbd list: ['2b', '2f']
    - random term_fun_list: ['func', 'term']


    How this works:
    - Building until we have achieved goal_max_nodes
    - xtype is the root node
    -> functerm_list: ['func', 'term', ...] for the current depth.
       shuffled for complete randomness
    ->
    """
    tbdo_xtypes = [xtype]
    num_inserts = 1
    result_label_list = []
    result_arity_list = []
    done = False

    while not done:

        functerm_list = ['func']
        for _ in range(num_inserts - 1):  # 1 -> at least one function
            if build_type == 'grow':
                functerm_list.append(np.random.choice(['func', 'term']))  # sfeh choice
            elif build_type == 'full':
                functerm_list.append('func')
            else:
                raise
        np.random.shuffle(functerm_list)  # ['term', 'func', 'term', ...]

        tmp_label_list = ['dummy'] * num_inserts
        tmp_arity_list = [42] * num_inserts

        func_indices = [i for i, x in enumerate(functerm_list) if x == 'func']
        term_indices = [i for i, x in enumerate(functerm_list) if x == 'term']
        np.random.shuffle(func_indices)

        for enum, index in enumerate(func_indices):  #
            xtype = tbdo_xtypes[index]

            label, arity = xtype_choose_func(func_array, xtype=xtype)
            # ('GG', result_label_list, tmp_label_list, '(', len(result_label_list), num_inserts, '>', arity, ')', (len(result_label_list) + num_inserts + arity), goal_max_nodes)
            if goal_max_nodes > (len(result_label_list) + num_inserts) + arity + 1:  # +1 = the start node which we must not forget
                tmp_label_list[index] = label
                tmp_arity_list[index] = arity
                num_inserts += arity - 1
            else:
                term_indices.extend(func_indices[enum:])
                done = True
                break

        for index in term_indices:
            label, arity = xtype_choose_term_v2(tbdo_xtypes[index], variables_dict), 0
            tmp_label_list[index] = label
            tmp_arity_list[index] = arity
            num_inserts -= 1

        # prepare next loop
        tbdo_xtypes = []
        for index, label in enumerate(tmp_label_list):
            if label == 'Ifte':
                tbdo_xtypes.extend(['2b', '2f', '2f'])
            else:
                xtype = xtype_get_from_label(label, variables_dict)
                child_type = xtype[:2][::-1]  # e. g. 'f2b' requires '2f' input
                arity = tmp_arity_list[index]
                tbdo_xtypes.extend([child_type] * arity)

        result_label_list.extend(tmp_label_list)
        result_arity_list.extend(tmp_arity_list)

    else:
        # Fix the last leftover nodes
        for xtype in tbdo_xtypes:
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
    # Load origin_meta from file
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


def tree_get_ids_depthfirst(tree, node_id=root_id):
    """
    returns tree ids depth-first wise.
    """
    result = [node_id]
    child_ids = tree_node_get_childs(tree, node_id)
    for child_id in child_ids:
        result.extend(tree_get_ids_depthfirst(tree, child_id))
    return result


def tree_get_fitness(tree, precision=None, karoo=True):
    """
    Get the fitness of a tree
    """
    if not karoo:
        raise

    fitness = tree[T_fitness][1]
    if fitness != '':
        fitness = round(float(fitness), precision)
    else:
        raise Exception('This tree does not contain float fitness: {}.'.format(fitness))
    return fitness


def tree_get_ident(tree):
    """
    What is used as identificator for a tree...
    - hash(expr_raw)
    """
    expr_raw = tree_get_expr_raw(tree, node_id=root_id)
    tree_ident = hash(expr_raw)
    return tree_ident


def tree_get_last_evolution(tree):
    """
    return a tree's last genetic modification
    """
    last_modi = tree[TR_type][1]
    return last_modi


def tree_get_parsimony(tree):
    """
    Get parsimony from value in tree
    """
    parsimony = tree[T_parsimony][root_id]
    if parsimony != '':
        parsimony = float(parsimony)
    return parsimony


def tree_get_meta(tree):
    """
    Get the meta information from a tree
    ! This does not evaluate fitness or parsimony !
    """
    tree_meta = {}
    parsimony = tree_get_parsimony(tree)
    fitness_train = tree_get_fitness(tree)
    expr_raw = tree_get_expr_raw(tree, node_id=root_id)  # sfeh store algo raw?
    expr_sym = expr_sympify(expr_raw=expr_raw)  # sfeh store algo sym?

    tree_meta['parsimony'] = parsimony
    tree_meta['fitness_train'] = fitness_train
    tree_meta['expr_raw'] = expr_raw
    tree_meta['expr_sym'] = expr_sym
    return tree_meta


def tree_get_expr_raw(tree, node_id):
    """
    Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').
    The large amount of () is required doe to some sympify errors. But feel free to reduce them.
    """
    node_id = int(node_id)
    arity = tree[N_arity, node_id]
    label = tree[N_label, node_id]
    if arity == '0':  # arity of 0 for the pattern '[term]'
        return '({})'.format(label)  # 'node_label' (function or terminal)

    elif arity == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        fun = label
        if fun == '~':  # ~- workaround
            return '(-({}))'.format(tree_get_expr_raw(tree, tree[9, node_id]))
        else:
            return '({}{})'.format(fun, tree_get_expr_raw(tree, tree[9, node_id]))

    elif arity == '2':
        if label not in functions_infix_dict:
            return '({}({}, {}))'.format(label, tree_get_expr_raw(tree, tree[9, node_id]), tree_get_expr_raw(tree, tree[10, node_id]))
        else:
            return '({}{}{})'.format(tree_get_expr_raw(tree, tree[9, node_id]), label, tree_get_expr_raw(tree, tree[10, node_id]))

    elif arity == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return '(Ifte({}, {}, {}))'.format(tree_get_expr_raw(tree, tree[9, node_id]), tree_get_expr_raw(tree, tree[10, node_id]), tree_get_expr_raw(tree, tree[11, node_id]))


def tree_get_pycode(tree, node_id=root_id):
    """
    returns python (one-lined) code from a tree
    """
    node_id = int(node_id)
    arity = tree[N_arity, node_id]
    label = tree_node_get_label(tree, node_id)

    if arity == '0':
        return '{}'.format(label)
    else:
        childs = tree_node_get_childs(tree, node_id)
        results = []
        for child in childs:
            results.append(tree_get_pycode(tree, node_id=child))  # = tree_node_get_label(tree, int(child))
        return op[label]['pycode'](*results)  # abs -> lambda a: 'abs({})'.format(a) (result1)


def tree_raw_depth_prefix(tree, node_id):
    """
    Does the same as tree_expr_raw, but evaluates infix functions in prefix notation (functional form)

    """

    node_id = int(node_id)
    arity = tree[N_arity, node_id]
    label = tree[N_label, node_id]

    if arity == '0':  # arity of 0 for the pattern '[term]'
        return '{{{}}}'.format(label)  # '{{{}}}'

    elif arity == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        return '{{{}{}}}'.format(label, tree_raw_depth_prefix(tree, tree[9, node_id]))

    elif arity == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
        return '{{{}{}{}}}'.format(label, tree_raw_depth_prefix(tree, tree[9, node_id]), tree_raw_depth_prefix(tree, tree[10, node_id]))

    elif arity == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return '{{Ifte{}{}{}}}'.format(tree_raw_depth_prefix(tree, tree[9, node_id]), tree_raw_depth_prefix(tree, tree[10, node_id]), tree_raw_depth_prefix(tree, tree[11, node_id]))


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


def tree_get_mutatable_nodes(tree, no_root=False, karoo=True):
    """
    Returns a list with mutatable ids
    """

    node_ids = []
    for node_id in tree_nodes_get_ids(tree, karoo=karoo):
        if node_id == 'node_modify':
            continue
        if float(tree[N_modify][node_id]) == node_is_modifiable:
            node_ids.append(int(node_id))

    if no_root and root_id in node_ids:
        node_ids.remove(root_id)

    return node_ids


def tree_get_fix_nodes(tree, karoo=True):
    """
    Returns a list with mutatable ids
    """

    node_ids = []

    for node_id in tree_nodes_get_ids(tree, karoo=karoo):
        if not tree_node_is_modifiable(tree, node_id):
            node_ids.append(int(node_id))

    return node_ids


def tree_get_branch(tree, node, karoo=False):
    """
    return all child-nodes as list
    """
    if not karoo:
        raise Exception

    branch = np.array([])  # the array is necessary in order to len(branch) when 'branch' has only one element

    # 2. Also return all child nodes
    branch_eval = tree_nodes_get_ids_string(tree, node)  # generate tuple of 'branch_top' and subsequent nodes
    branch_symp = sympify(branch_eval)  # convert string into something useful

    branch = np.append(branch, branch_symp)
    branch = np.sort(branch)

    return branch


def tree_get_branch_lax(tree, node_id, karoo=True):
    """
    returns all ids, labels and arities for a node in a tree
    """
    ids = tree_get_branch(tree, node_id, karoo=karoo)
    labels = [tree[N_label][i] for i in ids]
    aritys = [tree[N_arity][i] for i in ids]
    return ids, labels, aritys


def tree_get_layer_fix(tree, get_all_leaves=False):
    """
    Returns the last layer with fix nodes that have children which are modifiable

    """

    node_ids = []
    fix_ids = tree_get_fix_nodes(tree)

    if len(fix_ids) == 0:
        node_ids = []
    else:
        for node_id in fix_ids:

            only_fix_childs = True  # we assume this
            child_ids = tree_node_get_childs(tree, node_id)
            for child_id in child_ids:
                if tree_node_is_modifiable(tree, child_id):
                    only_fix_childs = False

            if get_all_leaves and len(child_ids) == 0:  # e. g. fix constants
                node_ids.append(node_id)

            if not only_fix_childs:  #
                node_ids.append(node_id)

    return node_ids


def tree_get_mutatable_layer_lv0(tree):
    """
    Returns a list with mutatable ids on layer 0
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
            child_id = int(tree[N_c1 + c][node_id])
            # if tree_node_modifiable(tree, node_id):
            if int(tree[N_modify][child_id]) == 1:
                leaf_ids.append(int(tree[N_c1 + c][node_id]))

    core_ids = []
    core_ids.extend(fix_ids)
    core_ids.extend(leaf_ids)
    core_ids.sort()

    return core_ids


def tree_get_mutatable_layer(tree, lvl_goal, sum_layers=False, get_closest=True, return_all_layers=False):
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

    if return_all_layers:
        return layer_lists

    if get_closest:
        lvl_best = min(lvl_count, lvl_goal)
    elif lvl_count > lvl_goal:  # really want to get nodes on layer 20? no matter what?
        return []  # Now you do not have any nodes.
    else:
        lvl_best = lvl_goal

    if sum_layers:
        result_ids = sum(layer_lists[:lvl_best + 1], [])
    else:
        result_ids = layer_lists[lvl_best]
    return result_ids


def tree_get_depth(tree):
    """
    Return the depth of the last node
    """
    max_depth = tree_node_get_depth(tree, -1)
    return max_depth


def tree_get_depth_ids(tree):
    """
    [[1],[2,3,4],[5,6]]
    """
    depth_id_list = [[]]
    depth = 0
    for node_id in tree_nodes_get_ids(tree):
        if tree_node_get_depth(tree, node_id) == depth:
            depth_id_list[depth].append(node_id)
        else:
            depth += 1
            depth_id_list.append([node_id])
    return depth_id_list


def tree_parsimony_ted(tree1, tree2):
    """
    The Tree Edit distance (TED) ('coolest' distance)
    - the amount of changes that have to be applied to the origin_meta to equality are counted
    """
    apted_tree1 = tree_raw_depth_prefix(tree1, 1)
    apted_tree2 = tree_raw_depth_prefix(tree2, 1)
    distance, mapping = apted_distance(apted_tree1, apted_tree2)
    # sfeh the mapping could be handy somewhere
    return distance, mapping


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


def expr_sympify(expr_raw):
    """
    Returns a simplified expression using sympify.
    - sympify the expression
    - If sympify evaluates to one of these errors: 'zoo', 'inf', '*I', 'nan', stop evaluation

    Sympify is a python core module which reduced mathematical expressions.
    Example: sympify('a+a+a+a') -> a*4
    Note that the sympify was extended in plagih_sympify_extras.py with extra functions

    Sympify fails: The results are, or contain, expressions that should/can not be evaluated
    'zoo': (Complex infinity) E.g. when a int-number is divided by zero
    'inf': (Regular infinity) E.g. when a float-number is divided by zero (...i know, why are there two infinities?)
    '*I': (Complex number) E.g. when putting a number to the power of negative fractals, 1**(-0.5)
    'nan': (Not a number) when Evaluation fails, E.g. types contradict, expression is empty, 'Mini(a, zoo' ...
    """

    try:
        expr_sym = str(plagih_sympify(expr_raw))
    except Exception as ex:
        raise Exception('Sympify: Fail caused by this raw algorithm: {}. Ex: {}'.format(expr_raw, ex))

    for fail_reason in ['zoo', 'inf', '*I', 'nan']:
        if fail_reason in expr_sym:
            raise Exception('Sympify: Failed due to a fail reason: {}.'.format(fail_reason))

    return expr_sym


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
            tree[N_modify][n] = node_is_modifiable

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

    if arity_list is not None:
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

    if not tree_check_children(tree, karoo=False):
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
        tree = tree_node_set_childs_ids(tree, node, c_buffer)  # update child links for each node # todo karoo=True??

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
        elif name_observation in label or name_action in label:
            pass
        else:
            # now it MUST be float
            const_xtype = '2f'

    return const_xtype


def treegp_reduce_branch(tree, node_id, karoo=False):
    """
    Reduce a branch of a tree with sympify
    """
    delete_ids = tree_get_branch(tree, node_id, karoo=karoo)
    expr_raw = tree_get_expr_raw(tree, node_id=node_id)
    try:
        expr_sym = expr_sympify(expr_raw=expr_raw)
        label_list = ast_convert_from_expr(expr_sym, build=True)
        arity_list = [label_get_arity(label) for label in label_list]
        core = core_from_labels(label_list, arity_list)
        tree_sympified = tree_insert_subtree(tree, core, delete_ids, karoo=karoo)

        return tree_sympified
    except Exception as ex:
        raise Exception('Reducing branch failed! Ex: {}'.format(ex))


def tree_check_meta_exists(tree):
    """

    """
    cond1 = str(tree_get_fitness(tree)) == ''
    cond2 = str(tree_get_parsimony(tree)) == ''
    # cond3 = str(tree_get_id(tree)) == ''
    if cond1 or cond2:
        return False
    else:
        return True


def tree_evolve_mutate_point(tree, func_array, variables_dict):
    """
    Mutate a single mutatable point in any Tree.
    """

    # 1. choose a node
    node_ids = tree_get_mutatable_nodes(tree)
    node_id = np.random.choice(node_ids)
    label, arity, xtype = tree_node_get_lax_v3(tree, node_id)

    if arity > 0:
        new_label, new_arity = xtype_choose_func(func_array, xtype=xtype, arity=arity)  # Function is same type, same arity
        tree = tree_node_set_label(tree, node_id, new_label)
    else:
        new_label = xtype_choose_term_v2(xtype, variables_dict)  # 3 -> '2f' -> 5
        tree = tree_node_set_label(tree, node_id, new_label)

    # All node info should stay the same. xtype, arity

    return tree  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping


def tree_evolve_reduce(tree, completely=True):
    """
    Reducing a tree to its most basic form with sympify.
    (completely = False: reduce just one branch. if you wanted to have more complexity)
    """
    try:
        if completely:  # reduce the complete tree
            nodes_lv0 = tree_get_mutatable_layer(tree, 0)
            for i in range(len(nodes_lv0)):
                nodes_lv0 = tree_get_mutatable_layer(tree, 0)
                node_id = nodes_lv0[i]
                tree = treegp_reduce_branch(tree, node_id, karoo=True)
        else:  # only choose one node to be reduced
            node_ids = tree_get_mutatable_nodes(tree)
            func_ids = [x for x in node_ids if tree_node_get_arity(tree, x) > 0]
            if len(func_ids) > 0:
                node_id = np.random.choice(node_ids)
                tree = treegp_reduce_branch(tree, node_id, karoo=True)
        return tree
    except Exception as ex:
        print_warning('ww', 'Could not reduce tree/branch due to Exception: {}'.format(ex))
        raise
        return


def labels_get_aritys_list(label_list, karoo=False):
    """
    returns an arity list for a label list
    """

    arity_list = [label_get_arity(x) for x in label_list]

    if karoo:
        arity_list.pop(0)
    return arity_list


def tree_pretty_print(tree, karoo=False):
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    depth = 0
    layer_labels = []
    print_style = 'Depth{:>3}: {}'  # {:>3} always print 3 letters at least
    node_depth = '-1'

    for node_id, node_depth in enumerate(tree[N_depth]):
        label = tree_node_get_label(tree, node_id)
        if int(node_depth) == depth:
            layer_labels.append(label)
        else:
            print(print_style.format(node_depth, layer_labels))
            layer_labels = [label]
            depth += 1
    else:
        print(print_style.format(node_depth, layer_labels))

    return


def tree_labels(tree):
    """
    Just helps printing trees better
    """
    label_list = tree[N_label]
    return label_list


def tree_delete_nodes(tree, node_list):
    tree = np.delete(tree, node_list, axis=1)  # delete all branches below
    return tree


def tree_check_children(tree, karoo=True):
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


def tree_check_rebuild(tree, karoo=True):
    """
    Check if a valid tree can be rebuilt from its expression
    sfeh: the expression must currently not be equal.
    The expression can include separate '~' (usub) nodes, which makes expressions not completely equal
    """

    label_list = tree[N_label]
    arity_list = tree[N_arity]

    if karoo:
        label_list = label_list[1:]
        arity_list = arity_list[1:]

    try:
        core = core_from_labels(label_list, arity_list)
        if core:
            tree_works = True
        else:
            tree_works = False
    except:
        tree_works = False

    return tree_works


def tree_check_typed(tree, variables_dict, karoo=True):
    """
    A method to check if a tree is type consistant:
    - do the values in c1, c2, c3 link to its parent?
    """
    if not karoo:
        tree = tree_convert_plagih_to_karoo(tree)

    for node_id in range(1, len(tree[3])):
        label, arity, xtype = tree_node_get_lax(tree, node_id)

        children_xtypes = xtype_label_get_child_xtypes(label, arity, variables_dict)

        for c in range(0, 3):  # children 0, 1, 2
            if tree[N_c1 + c][node_id] != '':  # if child exists
                c_node_id = tree[N_c1 + c][node_id]
                c_label = tree_node_get_label(tree, c_node_id)
                c_xtype = xtype_get_from_label(c_label, variables_dict)
                # if c_xtype != test_xtype[c]:
                if not xtype_equi_outcome(c_xtype, children_xtypes[c]):
                    print_e('Label ({}), child ({}) with c_label ({}) does not match xtype ({}). It is c_xtype ({}).\ntree labels: ({})'.format(label, c, c_label, xtype, c_xtype, tree[N_label]))
                    print_e('Last tree modification was: {}'.format(tree_get_history(tree)))
                    return False

    return True


def tree_check_reproduce_loop(tree, karoo=True):
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
    tree_check_reproduce_loop(tree, karoo=karoo)
    result = tree_check_children(tree, karoo=karoo)

    return result


def tree_evolve_node_insert(tree, variables_dict):
    """
    Inserts a (arity-1) node into a tree
    Especially useful when a ** power shall be normalized
    """

    node_ids = tree_get_mutatable_nodes(tree)
    insert_id = None
    for node_id in node_ids:
        label = tree_node_get_label(tree, node_id)
        xtype = xtype_get_from_label(label, variables_dict)  # '>' -> 'f2b'
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
    Mutates a number of float terminal of a tree
    """
    # 1. choose a node
    node_ids = tree_get_mutatable_nodes(tree)

    float_nodes = []
    for node_id in node_ids:
        label = tree_node_get_label(tree, node_id)
        if xtype_get_constant(label) == '2f':
            float_nodes.append(node_id)
    if float_nodes:
        float_id = np.random.choice(float_nodes)
        val = float(tree_node_get_label(tree, float_id))
        new_value = gp_mutate_constants(val, term_type='float', filter_type='gaussian_filter')
        tree = tree_node_set_label(tree, float_id, new_value)
        return tree
    else:
        raise Exception('No mutatable node found!')
        # return None


def tree_prune_depth(tree, max_depth, variables_dict):
    """
    reduces the depth of a Tree (in case it is too deep).
    Arguments required: tree, depth
    """

    nodes = []

    for node_id in range(root_id, len(tree[3])):

        node_depth = tree_node_get_depth(tree, node_id)
        node_arity = tree_node_get_arity(tree, node_id)
        if node_depth == max_depth and node_arity > 0:  # replace this node with terminal
            label = tree_node_get_label(tree, node_id)
            node_xtype = xtype_get_from_label(label, variables_dict)
            tree = tree_node_set_arity(tree, node_id, 0)
            new_term = xtype_choose_term_v2(node_xtype, variables_dict)  # replace label
            tree = tree_node_set_label(tree, node_id, new_term)

        elif tree_node_get_depth(tree, node_id) > max_depth:  # record nodes deeper than the maximum allowed Tree depth
            nodes.append(node_id)

    tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
    tree = evolve_node_arity_fix(tree)  # fix all node arities

    return tree


def tree_nodes_get_ids(tree, skip_nodes=0, karoo=True):
    """
    returns all node ids as list
    skip_nodes: extra parameter which (was) used, i dont remember why. maybe useful in the future
    """
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
    for node_id in tree_nodes_get_ids(tree, karoo=True):
        if tree_node_get_label(tree, node_id) == '**':
            child_id = tree_node_get_child(tree, node_id, 1)  # get second argument
            old_power = tree_node_get_label(tree, child_id)
            try:
                new_power = float(int(float(old_power)))
                tree = tree_node_set_label(tree, child_id, new_power)
            except ValueError:
                pass  # sfeh: This may actually take some time. Every tree gets checked any many have '**'.
    return tree


def tree_set_meta_wipe(tree):
    """
    Wipes all tree meta data, e.g.
    todo save history of last values
    What should be deleted?
    - fitness
    - parsimony
    - tree_id
    - tree_type

    What should not be deleted?
    - modifiable nodes
    """
    # tree = tree_set_fitness(tree, '')  # todo just randomly kick this out aswell
    # tree = tree_set_parsimony(tree, '')  # todo if we wipe this, parsimony can not be checked anymore
    # tree = tree_set_id(tree, 'tourn win')
    return tree


def tree_eval_parsimony(tree, parsimony_distance, origin_tree=None, weights=None):
    """
    parsimony_distance: compute the chosen distance by the user.

    """

    if parsimony_distance == 'total_count_nodes':  # number of nodes
        return tree_get_last_nodeid(tree)  # returns the number of nodes
    elif parsimony_distance == 'total_tree_depth':
        return 0
    if parsimony_distance == 'ted':  # tree edit distance, tree-edit-distance
        distance, mapping = tree_parsimony_ted(tree, origin_tree)
        if weights is None:
            return distance
        else:
            raise
            # TODO weights
    elif parsimony_distance == 'rel_ari_1':  # Does this work?
        return tree_parsimony_relari(tree, origin_tree)
    else:
        print_e('Complexity measurement not available: {}'.format(parsimony_distance))
        raise


def tree_check_is_sympified(tree):
    """
    Label list from expression
    """
    tree_raw = copy.deepcopy(tree)
    tree_sym = tree_evolve_reduce(tree, completely=True)

    labellist_raw = tree_get_labellist(tree_raw)
    labellist_sym = tree_get_labellist(tree_sym)
    if list(labellist_raw) == list(labellist_sym):
        return True
    else:
        return False


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


def tree_viz_get_forest(tree, node_id=root_id):
    """
    creates a tex file with a tikz figure of a tree.

    Labeling edges: , edge label = {node[midway, font =\scriptsize]{If...}}
    """
    extras = ''
    label, arity, xtype = tree_node_get_lax_v3(tree, node_id)
    latex_label = None

    # Get the best math-like representation
    if label in op:
        latex_label = op[label]['latex']
    if latex_label is None:
        latex_label = label

    latex_label = '{{{}}}'.format(latex_label)  # {12} because

    # custom node design
    if arity > 0:
        extras += ',nonterminal'
    else:
        extras += ',terminal'

        if tree_node_is_variable(tree, node_id):
            extras += ',variable'
        else:
            extras += ',constant'

    if not tree_node_is_modifiable(tree, node_id):
        extras += ',fixnode'

    latex_label += extras

    child_ids = tree_node_get_childs(tree, node_id)
    for child_id in child_ids:
        latex_label += (tree_viz_get_forest(tree, child_id))
    else:
        latex_label = '[{}]'.format(latex_label)

    return latex_label


def tree_get_latex_forest(tree):
    """
    Creates forest tree representation (based on tikz) for LaTeX.
    The file can easily ne included in a .tex file with '\input{file_name}'
    optional: stand_alone = True for a complete latex file
    """

    bracket_tree = tree_viz_get_forest(tree)
    forest_viz = latex_wrap_forest(bracket_tree)

    return forest_viz


def tree_remove_minus_workaround(tree):
    """
    The ~ operator should be removed and replaced with a
    """


def tree_group_branch_expressions(tree):
    """
    E.g. combine a mathematical expression
    - from leaf to root: give info whether you are a inline math-op
    """

