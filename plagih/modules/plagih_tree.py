import os

from plagih.modules.plagih_data import obs_get_timedelta, observation_get_family_and_time
from plagih.modules.plagih_sympy_extras import plagih_sympify
from plagih.tree_distances.tree_edit_distance import apted_distance
from plagih.modules.plagih_types import *
from plagih.modules.plagih_eval import *
import csv
# from plagih.modules.viz_with_latex import *
from sympy import sympify
import copy
from pathlib import Path as Path

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


N_id = 3
N_depth = 4
N_xtype = 5
N_label = 6
N_parent = 7
N_arity = 8
N_c1 = 9
N_c2 = 10
N_c3 = 11
N_modify = 13

TR_lastEvolve = 1
T_fitness = 12
T_parsimony = 14

T_deleteable = [0, 2]  # TR_ID = 0, tree_depth_base = 2

T_num_lines = 15
root_id = 1

node_is_modifiable = 1


class Core_From_Labels():

    def __init__(self, label_list, arity_list, xtype_list, force_np_dtype='U15'):
        """
        Given the labels (and label infos) as list
        this function builds the core of a tree (no node_modify)
        """

        if len(label_list) == 0:
            print_warning('w', 'label list is empty')
            raise

        # if arity_list is None:
        #     arity_list = [label_get_arity(label) for label in label_list]
        #
        # if xtype_list is None:

        size = len(label_list)
        core_tree = tree_init_core(size, force_np_dtype)

        # set all the rows that are super easy
        core_tree = tree_core_init_row(core_tree, N_id, [x for x in range(0, size)])
        core_tree = tree_core_init_row(core_tree, N_label, label_list)
        core_tree = tree_core_init_row(core_tree, N_arity, arity_list)
        core_tree = tree_core_init_row(core_tree, N_xtype, xtype_list)

        # and also, fill all the leftover rows
        parent_list = parents_from_arities(arity_list)
        core_tree = tree_core_init_row(core_tree, N_parent, parent_list)
        core_tree = tree_core_build_childs(core_tree)
        core_tree = tree_core_init_depth(core_tree, parent_list)

        self.core_tree = core_tree
        return

    def get_uninstanced_core(self):
        return self.core_tree


def core_from_labels(label_list, arity_list, xtype_list, force_np_dtype='U15'):
    """
    Given the labels (and label infos) as list
    this function builds the core of a tree (no node_modify)
    """

    tree = Core_From_Labels(label_list, arity_list, xtype_list).get_uninstanced_core()

    return tree


class Ptree_karoo():
    """
    Plagih trees are computational trees that hold the genetic programs.

    What is a trees primary identificable?
    - the alignment of labels: [+, a, b]

    What is additonal node-info we need?
    - Node positioning info:
    - modifiable nodes

    What is the trees meta data?:
    - fitness
    - parsimony
    - last modification

    evolve-based:
    - last evolve time

    - expr_raw
    - expr_sym
    (- last modifications)
    (- last parsimony)
    (- other complexity measurement?)
    ((- its last versions tree data))  # -> needs too much memory?
    ((- its last fitness, parsimony))

    What run-specific data is irrelevant?
    - pop_id (NO!)

    node:
    branch depth
    """

    #
    # def __init__(self, expr=None):
    #     self.fitness = None
    #     self.parsimony = None
    #     self.expr = expr
    #     self.numpy_nodes = None

    def __init__(self, label_list, xtype_list, modify_list=None, arity_list=None):
        """
        create a tree from user input
        """
        if not arity_list:
            arity_list = [label_get_arity(label) for label in label_list]  # ~- problem: fine. [-, 1, 2] vs [*, 1, -2]

        core = Core_From_Labels(label_list, arity_list, xtype_list).get_uninstanced_core()

        if modify_list:
            for i, val in enumerate(modify_list):
                core[N_modify][i] = val
        else:  # all can be modified
            for i, val in enumerate(label_list):
                core[N_modify][i] = 1

        self.tree_nodes = tree_convert_pcore_to_karoo(core)

        return

    def get_uninstanced_tree(self):
        return self.tree_nodes

    def write_to_file(self, path):
        pass


def TEST_karoo_tree_from_labellist(label_list, obs_krazy, modify_list=None, arity_list=None):
    """
    returns: tree, from label_list (newest version)
    """

    xtype_list = xtypes_from_labels(label_list, obs_krazy)
    p_tree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list, arity_list=arity_list)
    tree = p_tree.get_uninstanced_tree()

    return tree


def karoo_tree_from_expr(expr, obs_krazy):
    """
    DELETE later sfeh
    Generate tree from a raw or sympified expression
    # label_list = workaround_remove_tilde_operator(label_list)
    """
    label_list = ast_convert_from_expr(expr, build=True)
    xtype_list = xtypes_from_labels(label_list, obs_krazy)
    p_tree = Ptree_karoo(label_list, xtype_list, modify_list=None)
    tree = p_tree.get_uninstanced_tree()
    return tree


def tree_save_csv(tree, path_csv):
    """
    Writing one tree to a .csv file. As it is appended, many can be added.
    """
    with Path.open(path_csv, 'a', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
        target = csv.writer(csv_file, delimiter=',')

        target.writerows([''])  # empty row before each Tree
        # tree = tree_remove_minus_workaround(tree)
        for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
            target.writerows([tree][row])


def load_pop_from_csv(pop_csv):
    """
    This method is used to load a saved population of Trees, as invoked through the (pause) menu where population_r
    replaces population_a in the karoo_gp/runs/[date-time]/ directory.
    sfeh... delete if not needed?
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


def tree_get_labellist(tree, karoo=True):
    """
    Returns all tree labels in order
    these identify a tree completely (if the tree is fully functioning)
    """
    label_list = tree[N_label]
    if karoo:
        label_list = label_list[1:]
    return label_list


def tree_get_size(tree, karoo=True):
    size = len(tree[0])
    if karoo:
        return size
    else:
        raise


def tree_get_history(tree):
    return tree[TR_lastEvolve][1]


def tree_get_last_evolution(tree):
    """
    return a tree's last genetic modification
    """
    last_modi = tree[TR_lastEvolve][root_id]
    return last_modi


def tree_set_last_evolution(tree, last_modification):
    """
    the last evolution (or evolutions)
    """
    tree[TR_lastEvolve][root_id] = last_modification
    return tree


def tree_check_quick(tree, karoo=True, print_type=None, allow_root_only=True):
    """
    without some of the heavy tests
    """
    if tree is None:
        return False

    if not tree_check_children(tree, karoo=karoo):
        tree_works = False
    elif not tree_check_node_label_info(tree):
        tree_works = False
    elif not tree_check_types(tree):
        tree_works = False
    else:
        tree_works = True

    if tree_node_get_arity(tree, root_id) == 0:
        print_warning('wwww', 'Tree is only a root node. Might occur after a simplification.', print_type=print_type)
        tree_works = allow_root_only

    return tree_works


def tree_check_deep(tree, print_type=None):
    """
    Performs all checks that we currently have
    # sfeh do not use this if trees are safely generated
    # sfeh check meta values in separate method? update those aswell?
    """

    if not tree_check_quick(tree):
        tree_works = False
    elif not tree_check_rebuild(tree):
        tree_works = False
    else:
        tree_works = True

    return tree_works


def tree_check_xtypes(tree):
    for node in tree_iterate_range(tree):
        if tree[N_xtype][node] == '':  # are xtypes set?
            # print_warning('ww', 'xtypes in tree were not set correctly.\n{}'.forsmat(tree))
            return False
    return True


def tree_set_fitness(tree, fitness, precision=6):
    """
    Store the fitness within the tree np-array

    """
    fitness = '' if fitness is None else fitness
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
        print_warning('w', f'Warning: Parsimony is: {parsimony}')

    tree[T_parsimony][1] = parsimony
    return tree


def tree_set_modifyable_nodes_true(tree, karoo=True):
    """

    """

    for node_id in tree_nodes_get_ids(tree, karoo=karoo):
        tree[N_modify][node_id] = 1
    return tree


def tree_node_set_xtype(tree, node_id, xtype):
    tree[N_xtype][node_id] = xtype
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

    arity = tree_node_get_arity(tree, node_id, empty_is_zero=True)

    for i in range(arity):
        tree[N_c1 + i][node_id] = c_buffer + i

    for i in range(arity, 3):
        tree[N_c1 + i][node_id] = ''

    if karoo:
        tree = tree_convert_pcore_to_karoo(tree)

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


def tree_node_get_xtype(tree, node_id):
    return tree[N_xtype][node_id]


def tree_node_get_arity(tree, node_id, empty_is_zero=False):
    arity = tree[N_arity][int(node_id)]
    if arity == '':
        if empty_is_zero:
            return empty_is_zero
        else:
            print_warning('ww', 'Arity was not set!')
            raise
    else:
        arity = int(arity)

    return arity


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
    if child_id != '':  # when checking for children that do not exist
        child_id = int(child_id)

    return child_id


def tree_node_get_childs(tree, node_id):
    """

    """
    child_list = []
    arity = tree_node_get_arity(tree, node_id)
    for child in range(arity):
        child_list.append(tree_node_get_child(tree, node_id, child))
    return child_list


def tree_node_get_parent(tree, node_id):
    """

    """
    parent_id = tree[N_parent][node_id]
    if parent_id == '':
        return parent_id
    else:
        return int(parent_id)


def tree_node_get_parents(tree, node_id):
    node_list = [node_id]
    while node_id > root_id:
        node_id = tree_node_get_parent(tree, node_id)
        if len(node_list) > 200:
            raise  # tree has intrinsic parent node recursion (node is its own parent?)
        node_list.append(node_id)
    return node_list


def tree_node_get_modify(tree, node_id):
    """

    """
    modify = tree[N_modify][node_id]
    if modify == '':
        modify = 1
    else:
        modify = int(modify)
    return modify


def tree_set_modifyable_nodes(tree, origin_tree=None):
    """
    Sets all the origin_meta core nodes back to non-modifyable
      # sfeh: somewhere else
    """

    tree = tree_set_modifyable_nodes_true(tree)
    if origin_tree is not None:
        if not tree_node_is_modifiable(origin_tree, root_id):
            non_modifiable_nodes = []
            if tree_node_get_modify(origin_tree, root_id) == 0:  # check if modifiable nodes are specified
                permanent_nodes = tree_permanent_nodes_get(1, tree, 1, origin_tree)
                non_modifiable_nodes.extend(permanent_nodes)

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
                next_chosen_node = int(chosen_tree[c][chosen_node])  # error? probably nodes that should not be changed were changed earlier. correct random function?

                tmp = tree_permanent_nodes_get(next_origin_node, chosen_tree, next_chosen_node, origin_tree)
                if tmp is not None:
                    permanent_nodes.extend(tmp)
        return permanent_nodes
    else:
        return


def terminal_label_is_observation(label):
    """
    sfeh if contains pi/e/... in future, change this
    """
    try:
        float(label)
        return False
    except:
        if label in ['True', 'False']:
            return False
    return True


def tree_node_is_modifiable(tree, node_id):
    """
    returns True if modifiable is 1
    """
    modify = tree_node_get_modify(tree, node_id)
    return modify == 1


def tree_init_core(node_amount, np_dtype):
    """
    returns an empty tree with an amount of nodes, auto fills
    """

    tree = np.zeros((T_num_lines, node_amount), dtype=np.dtype(np_dtype))  # U12: longest is observation1
    return tree


def tree_try_get_swapids(a_tree, b_tree, version='default'):
    """
    Try to return two branches (aka ids) [for crossover] that can be crossed

    """
    if version == 'default':
        # choose a node from parent a
        a_ids = tree_get_mutatable_nodes(a_tree, no_root=True)
        # a_ids = tree_get_mutatable_layer_lv0(a_tree)  # todo
        a_id = random.choice(a_ids)
        a_label, _, a_xtype = tree_node_get_lax_v3(a_tree, a_id)

        # create a list from parent b with same xtype
        b_node_ids = tree_get_mutatable_nodes(b_tree, no_root=True)
        b_sametype_ids = b_node_ids[:]
        for b_id in b_node_ids:
            b_label, _, b_xtype = tree_node_get_lax_v3(b_tree, b_id)
            if not xtype_equi_outcome(b_xtype, a_xtype):
                b_sametype_ids.remove(b_id)  # remove one-by-one false partner nodes.

        if b_sametype_ids:  # if entries were found, choose one. we are custom_done
            b_id = random.choice(b_sametype_ids)
            success = True
        else:
            b_id = random.choice(b_node_ids)
            success = False

        return a_id, b_id, success
    else:
        raise


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


def raise_if_empty(name, val):
    if val == '' or val is None:
        print(f'This variable did not work {name}')
        raise


def invent_label_list_depth(xtype_root, depth_goal, float_decimals, choose_obs, obs_krazy, choose_oparray2, choose_distributions, min_depth=0, full_or_grow=None):
    """
    build a random, but within itself consistent label list
    Also, return the arities aswell (they are searched anyways)
    """

    tbdo_xtypes = [xtype_root]
    result_label_list = []
    result_arity_list = []
    result_xtype_list = []

    # Build a list with labels in row, and a list with their arities
    for depth in range(min_depth, depth_goal):
        next_xtype_list = []

        if depth < depth_goal - 1:

            functerm_list = ['func']
            for _ in range(len(tbdo_xtypes) - 1):  # 1 -> at least one function
                if full_or_grow == 'grow' and depth >= min_depth:
                    functerm_list.append(random.choice(['func', 'term']))  # sfeh choice always 50:50? terminal-factor?
                elif full_or_grow == 'full':
                    functerm_list.append('func')
                else:
                    raise
            np.random.shuffle(functerm_list)  # ['term', 'func', 'term', ...]

            for ii, xtype in enumerate(tbdo_xtypes):
                if functerm_list[ii] == 'term':
                    label = choose_term(xtype[-2:], choose_obs, choose_distributions, float_decimals)
                    # xtype stays the same 'arity-2' version
                    arity = 0
                elif functerm_list[ii] == 'func':
                    label = choose_operator(xtype[-2:], choose_oparray2, arity=None)
                    arity = label_get_arity(label)
                    xtype = op[label]['xtype']
                else:
                    raise

                # xtype-'To-do' list for the next depth to give values to these functions
                if label == 'Ifte':
                    next_xtype_list.extend(['2b', '2f', '2f'])
                else:
                    tmp_xtype = xtype_get_from_label(label, obs_krazy)
                    child_type = tmp_xtype[:2][::-1]  # the input of our function "reverted" is the xtype
                    for _ in range(0, arity):  # when arity==2, add 2 times
                        next_xtype_list.append(child_type)

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)
                result_xtype_list.append(xtype)
        else:  # now, we are on the lowest dim_y.

            for xtype in tbdo_xtypes:  # Build terminals now.
                label, arity = choose_term(xtype[-2:], choose_obs, choose_distributions, float_decimals), 0

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)
                result_xtype_list.append(xtype)

        # Finally, update the list for the next round
        tbdo_xtypes = next_xtype_list[:]

    return result_label_list, result_arity_list, result_xtype_list


def invent_label_list_nodes(t_xtype, goal_max_nodes, float_decimals, choose_obs, obs_krazy, choose_oparray2, choose_distributions, full_or_grow='grow'):
    """
    build a random function (as label list)
    -> labels, arities: ['+', '1.23', '2.34'], [2, 0, 0]
    E. g.: 2f, 5 nodes, min_nodes = 2
    - tbd list: ['2b', '2f']
    - random term_fun_list: ['func', 'term']


    How this works:
    - Building until we have achieved goal_max_nodes
    - t_xtype is the root node
    -> functerm_list: ['func', 'term', ...] for the current depth.
       shuffled for complete randomness
    ->
    # sfeh t_xtype here is a filler, e.g. 2f -> + -> f2f
    """
    tbdo_xtypes = [t_xtype]
    num_inserts = 1
    result_label_list = []
    result_arity_list = []
    result_xtype_list = []
    done = False

    while not done:

        functerm_list = ['func']
        for _ in range(num_inserts - 1):  # 1 -> at least one function
            if full_or_grow == 'grow':
                functerm_list.append(random.choice(['func', 'term']))  # sfeh choice
            elif full_or_grow == 'full':
                functerm_list.append('func')
            else:
                raise
        np.random.shuffle(functerm_list)  # ['term', 'func', 'term', ...]

        # tmp_node_lax = ('dummy', 42, 'xdummy') * num_inserts
        tmp_label_list = ['dummy'] * num_inserts
        tmp_arity_list = [42] * num_inserts
        tmp_xtype_list = ['xdummy'] * num_inserts

        func_at = [i for i, x in enumerate(functerm_list) if x == 'func']
        term_at = [i for i, x in enumerate(functerm_list) if x == 'term']
        np.random.shuffle(func_at)

        for enum, index in enumerate(func_at):  #
            t_xtype = tbdo_xtypes[index]

            label = choose_operator(t_xtype, choose_oparray2, arity=None)
            arity = label_get_arity(label)
            label_xtype = op[label]['xtype']
            # ('GG', result_label_list, tmp_label_list, '(', len(result_label_list), num_inserts, '>', arity, ')', (len(result_label_list) + num_inserts + arity), goal_max_nodes)
            if goal_max_nodes > (len(result_label_list) + num_inserts) + arity + 1:  # +1 = the start node which we must not forget

                tmp_label_list[index] = label
                tmp_arity_list[index] = arity
                tmp_xtype_list[index] = label_xtype
                num_inserts += arity - 1
            else:
                term_at.extend(func_at[enum:])
                done = True
                break

        for index in term_at:
            t_xtype = tbdo_xtypes[index]
            label, arity = choose_term(t_xtype[-2:], choose_obs, choose_distributions, float_decimals), 0
            label_xtype = xtype_get_from_label(label, obs_krazy)
            tmp_label_list[index] = label
            tmp_arity_list[index] = arity
            tmp_xtype_list[index] = label_xtype
            num_inserts -= 1

        # prepare next loop
        tbdo_xtypes = []
        for index, label in enumerate(tmp_label_list):
            if label == 'Ifte':
                tbdo_xtypes.extend(['2b', '2f', '2f'])
            else:
                t_xtype = xtype_get_from_label(label, obs_krazy)
                child_type = t_xtype[:2][::-1]  # e. g. 'f2b' requires '2f' input
                arity = tmp_arity_list[index]
                tbdo_xtypes.extend([child_type] * arity)

        result_label_list.extend(tmp_label_list)
        result_arity_list.extend(tmp_arity_list)
        result_xtype_list.extend(tmp_xtype_list)

    else:
        # Fix the last leftover nodes
        for t_xtype in tbdo_xtypes:
            label, arity = choose_term(t_xtype[-2:], choose_obs, choose_distributions, float_decimals), 0
            label_xtype = xtype_get_from_label(label, obs_krazy)
            result_label_list.append(label)
            result_arity_list.append(arity)
            result_xtype_list.append(label_xtype)

    return result_label_list, result_arity_list, result_xtype_list


def round_constant(constant, float_decimals):
    """
    Rounding float distributions_file
    """

    try:
        constant = float(constant)
    except Exception:
        return constant

    if constant == 0:
        return constant

    new_const = round(constant, float_decimals)
    if new_const == 0:
        if constant > 0:
            new_const = 1 / 10**float_decimals
        else:
            new_const = -1 / 10**float_decimals
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


def tree_get_ids_depthfirst(tree, node_id=root_id):
    """
    returns tree ids depth-first wise.
    """
    result = [node_id]
    child_ids = tree_node_get_childs(tree, node_id)
    for child_id in child_ids:
        result.extend(tree_get_ids_depthfirst(tree, child_id))
    return result


def tree_get_fitness(tree, karoo=True):
    """
    Get the fitness that a tree holds.
    For evaluation the fitness, use ? plagih_eval -> tree_eval_fitness_train()
    """
    if not karoo:
        raise

    fitness = tree[T_fitness][1]
    if fitness != '':
        fitness = float(fitness)
    else:
        # raise Exception(f'This tree does not contain float fitness: {fitness}.')
        fitness = None
    return fitness


def tree_hash(tree):
    """
    What is used as identificator for a tree...
    The tree is identifiable by the node-structure it holds, aka the labels in order.
    hash(label_list)
    old version: hash(expr_raw) deprecated.
    trees can hold the same expr_raw with different label-lists and thus different parsimonies
    """
    label_list = tree_get_labellist(tree)
    tree_ident = hash(','.join(np.array(label_list)))
    return tree_ident


def tree_get_parsimony(tree):
    """
    Get parsimony from value in tree
    """
    parsimony = tree[T_parsimony][root_id]
    if parsimony != '':
        parsimony = float(parsimony)
    return parsimony


def tree_get_expr_raw(tree, node_id=root_id):
    """
    Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').
    The large amount of () is required doe to some sympify errors. But feel free to reduce them.
    sfeh update this crapshit
    """
    node_id = int(node_id)
    arity = tree[N_arity, node_id]
    label = tree[N_label, node_id]
    if arity == '0':  # arity of 0 for the pattern '[term]'
        return f'{label}'  # 'node_label' (function or terminal)

    elif arity == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        fun = label
        if fun == '~':  # ~- workaround
            return '-({})'.format(tree_get_expr_raw(tree, tree[9, node_id]))
        else:
            return '{}({})'.format(fun, tree_get_expr_raw(tree, tree[9, node_id]))

    elif arity == '2':
        if label not in expr_raw_infix:
            return '{}({}, {})'.format(label, tree_get_expr_raw(tree, tree[9, node_id]), tree_get_expr_raw(tree, tree[10, node_id]))
        else:
            return '({}){}({})'.format(tree_get_expr_raw(tree, tree[9, node_id]), label, tree_get_expr_raw(tree, tree[10, node_id]))

    elif arity == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return 'Ifte(({}), ({}), ({}))'.format(tree_get_expr_raw(tree, tree[9, node_id]), tree_get_expr_raw(tree, tree[10, node_id]), tree_get_expr_raw(tree, tree[11, node_id]))


def tree_get_pycode(tree, node_id=root_id):
    """
    returns python (inline-) code from a tree
    """
    node_id = int(node_id)
    arity = tree_node_get_arity(tree, node_id)
    label = tree_node_get_label(tree, node_id)

    if arity == 0:
        if terminal_label_is_observation(label):
            ib_sfeh_dict = {'p': 'SetPoint',
                            'v': 'Velocity',
                            'g': 'Gain',
                            'h': 'Shift',
                            'f': 'Fatigue',
                            'c': 'Consumption'}
            ib_sfeh_rev = {v: k for k, v in ib_sfeh_dict.items()}
            obs_family, obs_time = observation_get_family_and_time(label, none_return=None)
            if obs_time is None:
                pass
            else:
                geth_name = ib_sfeh_rev[obs_family]
                return f"self.get_h('{geth_name}', {obs_time})"

        return f'{label}'
    else:
        childs = tree_node_get_childs(tree, node_id)
        results = []
        for child in childs:
            results.append(tree_get_pycode(tree, node_id=child))  # = tree_node_get_label(tree, int(child))
        return op[label]['pycode'](*results)  # abs -> lambda a: 'abs({})'.formadt(a) (result1)


def tree_raw_depth_prefix(tree, node_id):
    """
    Does the same as tree_expr_raw, but evaluates infix functions in prefix notation (functional form)
    input: +, Ifte, 3, True, 1, 2
    result: {+{Ifte{True}{1}{2}}{3}}
    """

    node_id = int(node_id)
    arity = tree[N_arity, node_id]
    label = tree[N_label, node_id]

    if arity == '0':  # arity of 0 for the pattern '[term]'
        return '{{{}}}'.format(label)  # '{{{}}}'.format('test') -> {test}

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
    for node_id in tree_nodes_get_ids(tree, karoo=karoo):
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
        if len(tree_get_labellist(tree)) > 1:  # sfeh dummy if the tree is only one node # sfeh no_root?
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


def tree_node_get_branch(tree, node_id, karoo=True):
    """
    return all child-nodes as list
    """
    if not karoo:
        raise Exception

    branch = [node_id]

    child_ids = tree_node_get_childs(tree, node_id)
    for child in child_ids:
        branch.extend(tree_node_get_branch(tree, child))

    branch = np.sort(branch)  # may be required for getting numpy array

    return branch


def tree_get_branch_ilax(tree, node_id, karoo=True):
    """
    returns all ids, labels and arities for a node in a tree
    """
    ids = tree_node_get_branch(tree, node_id, karoo=karoo)
    labels = [tree_node_get_label(tree, i) for i in ids]
    aritys = [tree_node_get_arity(tree, i) for i in ids]
    xtypes = [tree_node_get_xtype(tree, i) for i in ids]
    return ids, labels, aritys, xtypes


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

            if get_all_leaves and len(child_ids) == 0:  # e. g. fix distributions_file
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
        for cc in range(0, arity):
            child_id = int(tree[N_c1 + cc][node_id])
            # if tree_node_modifiable(tree, node_id):
            if int(tree[N_modify][child_id]) == 1:
                leaf_ids.append(int(tree[N_c1 + cc][node_id]))

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


def tree_parsimony_ted(cooltree1, cooltree2):
    """
    The Tree Edit distance (TED) ('coolest' distance)
    - the amount of changes that have to be applied to the origin_meta to equality are counted
    """
    # apted_tree1 = tree_raw_depth_prefix(tree1, root_id)
    # apted_tree2 = tree_raw_depth_prefix(tree2, root_id)
    apted_tree1 = cooltree1.get_apted_notation()
    apted_tree2 = cooltree2.get_apted_notation()
    distance, mapping = apted_distance(apted_tree1, apted_tree2)  # sfeh the mapping could be handy somewhere

    return distance, mapping


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
        raise Exception(f'sympify_1: {expr_raw} reason: ({ex})')

    for fail_reason in ['zoo', 'inf', '*I', 'nan']:
        if fail_reason in expr_sym:
            raise Exception(f'sympifail: {fail_reason}')

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
        # # sfeh root-only problem, had to comment the following lines. They never occured otherwise
        # if len(tree[N_id]) <= 2:
        #     print_warning('ww', 'Tree has <=2 nodes. Change configuration. {}'.format(tree))
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
        for cc in [c1, c2, c3]:
            if cc != '':
                tree[N_depth][int(cc)] = child_depth
    return tree


def parents_from_arities(arity_lst):
    """
    working on the tree 'core'
    Automatically fill the node_parent slots of a tree
    - arity list in tree or
    """

    parent_list = [-1]  # -1 is the root node in the non-karoo tree version
    for i, arity in enumerate(arity_lst):
        parent_list.extend([i] * arity)

    return parent_list


def tree_core_build_childs(tree, parent_list=None):
    """
    automaticalls fills c1, c2, c3 for each node
    Needed: node_parent
    """
    if not parent_list:
        parent_list = tree_row_int(tree, N_parent)

    c_iter = 0
    last_parent = -1

    # parent_list [-1, 0, 0, 0, 1, 1]
    for i, parent_id in enumerate(parent_list):
        if parent_id >= 0:

            if parent_id == last_parent:
                c_iter += 1
            else:
                last_parent = parent_id
                c_iter = 0
            tree[N_c1 + c_iter][parent_id] = i
    return tree


def tree_core_childs(tree, parent_list=None):
    """
    automaticalls fills c1, c2, c3 for each node
    Needed: node_parent
    """
    if not parent_list:
        parent_list = tree_row_int(tree, N_parent)

    c_iter = 0
    last_parent = -1

    # parent_list [-1, 0, 0, 0, 1, 1]
    for i, parent_id in enumerate(parent_list):
        if parent_id >= 0:

            if parent_id == last_parent:
                c_iter += 1
            else:
                last_parent = parent_id
                c_iter = 0
            tree[N_c1 + c_iter][parent_id] = i
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


def core_from_expr(expr, obs_krazy):
    """
    Creating a karoo tree from a raw expression
    """

    label_list = ast_convert_from_expr(expr, build=True)
    # label_list = workaround_remove_tilde_operator(label_list)
    arity_list = [label_get_arity(label) for label in label_list]
    xtype_list = xtypes_from_labels(label_list, obs_krazy)
    core = Core_From_Labels(label_list, arity_list, xtype_list).get_uninstanced_core()
    return core


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


def evolve_c_buffer(ztree, node_id, karoo=False):
    """
    Generates the c_buffer for a node_id of a ztree
    The c_buffer is:
    """
    if karoo:
        ztree = tree_convert_karoo_to_plagih(ztree)
        node_id -= 1

    parent_arity_sum = 0
    prior_sibling_arity_sum = 0
    prior_siblings = 0

    if node_id == 0:
        return 1

    for n in tree_iterate_range(ztree, karoo=False):  # increment through all nodes in array 'ztree'

        # sum up all arities of the parent dim_y
        if int(ztree[N_depth][n]) == int(ztree[N_depth][node_id]) - 1:  # find parent nodes at the prior depth
            if ztree[N_arity][n] != '':
                parent_arity_sum += int(ztree[N_arity][n])  # sum arities of all parent nodes at the prior depth

        # add the arities of nodes on the left (siblings)
        elif int(ztree[N_depth][n]) == int(ztree[N_depth][node_id]) and int(ztree[N_id][n]) < int(ztree[N_id][node_id]):  # find prior siblings at the current depth
            if ztree[N_arity][n] != '':
                prior_sibling_arity_sum += int(ztree[N_arity][n])  # sum prior sibling arity
            prior_siblings += 1  # sum quantity of prior siblings

    # node_id = the position from where we start counting
    # (parent_arity_sum - prior_siblings - 1) = the amount of nodes on our level after our node
    # prior_sibling_arity_sum = the amount of children before our children
    # + 1 = our first child node, not the last child of the prior sibling
    c_buffer = node_id + (parent_arity_sum - prior_siblings - 1) + prior_sibling_arity_sum + 1  # see above

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


def tree_convert_pcore_to_karoo(plagih_tree):
    """
    tests has a first row with nonsense and nodes start with 1
    plagih has no first row and nodes start with 0
    """
    first_col = tree_init_first_column()
    tree_withfirst = np.concatenate((first_col, plagih_tree), axis=1)
    tree_karoo = tree_convert_plusnode(tree_withfirst, add_or_sub=1, firstrow=1)
    tree_karoo[N_parent][1] = ''
    return tree_karoo


def tree_insert_subtree(ztree, insert_core, delete_ids, karoo=False):
    """
    insert a prepared subtree in a node-spot
    """
    if karoo:
        ztree = tree_convert_karoo_to_plagih(ztree)
        for i, val in enumerate(delete_ids):
            delete_ids[i] -= 1

    # 1. insert the top node
    top_node_id = int(delete_ids[0])

    ztree[N_label][top_node_id] = insert_core[N_label][0]  # --label
    ztree[N_arity][top_node_id] = insert_core[N_arity][0]  # --arity
    ztree[N_xtype][top_node_id] = insert_core[N_xtype][0]

    ztree = np.delete(ztree, delete_ids[1:], axis=1)  # delete all branches below

    c_buffer = evolve_c_buffer(ztree, top_node_id)  # child nr.1 at c_buffer
    ztree = tree_unsafe_insert_node_child_dummies(ztree, top_node_id, c_buffer)  # --child: id, depth, parent
    # if not tree_check_arity_exists(ztree):
    #     raise

    ztree = evolve_node_renum(ztree)  # --all: ids
    # ztree = tree_fix_arity(ztree)
    ztree = tree_fix_link_child(ztree)

    # 2. insert all following nodes
    insert_count = 1  # set node count to +1 as the new root has already replaced 'branch_top' (above)

    while insert_count < len(insert_core[3]):  # increment through all nodes in the new Tree, leaving out the root... +1??

        for j in tree_iterate_range(ztree, karoo=False):  # increment through all nodes in og ztree ('ztree')  # range(0, len(ztree[N_id]))

            if ztree[N_label][j] == '':  # aka: is this a dummy?
                ztree[N_label][j] = insert_core[N_label][insert_count]  # --label
                ztree[N_arity][j] = insert_core[N_arity][insert_count]  # --arity
                ztree[N_xtype][j] = insert_core[N_xtype][insert_count]

                if int(ztree[N_arity][j]) == 0:
                    ztree = tree_fix_link_child(ztree)  # fix all child links
                    ztree = evolve_node_renum(ztree)  # renumber all 'NODE_ID's

                elif int(ztree[N_arity][j]) > 0:
                    c_buffer = evolve_c_buffer(ztree, j)  # generate 'c_buffer' for point of mutation ('branch_top')
                    ztree = tree_unsafe_insert_node_child_dummies(ztree, j, c_buffer)  # insert new nodes
                    ztree = tree_fix_link_child(ztree)  # fix all child links
                    ztree = evolve_node_renum(ztree)  # renumber all 'NODE_ID's

                insert_count = insert_count + 1  # exit loop when 'node_count' reaches the number of columns in the array

    if karoo:
        ztree = tree_convert_pcore_to_karoo(ztree)

    return ztree


def tree_fix_link_child(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'config.tree' has been modified, as with both Grow and Full mutation.

    """

    for node in range(0, len(tree[N_id])):
        c_buffer = evolve_c_buffer(tree, node)  # generate c_buffer for each node
        tree = tree_node_set_childs_ids(tree, node, c_buffer, karoo=False)  # update child links for each node

    return tree


def tree_unsafe_insert_node_child_dummies(tree, node_id, c_buffer, karoo=False):
    """
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
    for c in range(0, tree_node_get_arity(tree, node_id)):  # 0 to 3
        tree = np.insert(tree, c_buffer + c, '', axis=1)  # insert node_id for 'node_c1'
        tree[N_id][c_buffer + c] = c_buffer + c  # node_id ID
        tree[N_depth][c_buffer + c] = int(tree[N_depth][node_id]) + 1  # node_depth
        tree[N_parent][c_buffer + c] = int(tree[N_id][node_id])  # parent ID

    if karoo:
        tree = tree_convert_pcore_to_karoo(tree)

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


def treegp_reduce_branch(tree, node_id, env_vars, karoo=True):
    """
    Reduce a branch of a tree with sympify
    """
    delete_ids = tree_node_get_branch(tree, node_id, karoo=karoo)
    expr_raw = tree_get_expr_raw(tree, node_id=node_id)
    expr_sym = expr_sympify(expr_raw)
    core = core_from_expr(expr_sym, env_vars)
    tree_sym = tree_insert_subtree(tree, core, delete_ids, karoo=karoo)
    tree_sym_tildefree = tree_remove_tilde(tree_sym)

    return tree_sym_tildefree


# def tree_evolve_mutate_point(tree, float_decimals, choose_oparray2, random_obs, choose_distributions):
#     """
#     Mutate a single mutatable point in any Tree.
#     """
#
#     # 1. choose a node
#     node_ids = tree_get_mutatable_nodes(tree)
#     node_id = random.choice(node_ids)
#     label, arity, xtype = tree_node_get_lax_v3(tree, node_id)
#
#     if arity > 0:
#         new_label, new_arity, new_xtype = choose_operator(xtype, choose_oparray2=choose_oparray2, arity=arity)  # Function is same type, same arity
#         tree = tree_node_set_label(tree, node_id, new_label)
#     else:
#         new_label = choose_term(xtype[-2:], random_obs, choose_distributions, float_decimals)  # 3 -> '2f' -> 5
#         tree = tree_node_set_label(tree, node_id, new_label)
#
#     # All node info should stay the same. xtype, arity
#
#     return tree  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping


def tree_remove_tilde(tree):
    """
    ~- workaround
    todo still needed?
    """
    while True:
        for node_id in tree_iterate_range(tree, karoo=True):
            if tree_node_get_label(tree, node_id) == '~':
                try:
                    c_id = tree_node_get_child(tree, node_id, 0)
                except IndexError:
                    return tree
                clabel, carity, cxtype = tree_node_get_lax_v3(tree, c_id)
                if carity == 0:
                    tree = tree_node_set_label(tree, c_id, f'-{clabel}')
                    tree = tree_remove_node_with_child0(tree, node_id)
                    break
            else:
                pass
        else:
            break
    return tree


def tree_remove_node_with_child0(tree, node_id):
    child0 = tree_node_get_child(tree, node_id, 0)
    if child0 != '':
        ids, labels, aritys, xtypes = tree_get_branch_ilax(tree, child0)
        core = Core_From_Labels(labels, aritys, xtypes).get_uninstanced_core()
        delete_ids = tree_node_get_branch(tree, node_id)
        tree = tree_insert_subtree(tree, core, delete_ids, karoo=True)
    else:
        raise  # should only call this when secure
    return tree


def tree_evolve_reduce(tree, env_vars, completely=True):
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
                tree = treegp_reduce_branch(tree, node_id, env_vars, karoo=True)
        else:
            node_ids = tree_get_mutatable_nodes(tree)
            func_ids = [x for x in node_ids if tree_node_get_arity(tree, x) > 0]
            if len(func_ids) > 0:
                node_id = random.choice(node_ids)
                try:
                    tree = treegp_reduce_branch(tree, node_id, env_vars, karoo=True)
                except Exception as ex:
                    print_e(f'This failed during reduce process: ex: {ex}\nTree labels:\n{tree_get_labellist(tree)}')
                    # tree = treegp_reduce_branch(tree, node_id, env_vars, karoo=True)  # debug
                    pass  # This might occur when a tree is sympified (?)
        return tree
    except Exception as ex:
        print_warning('ww', f'Could not reduce tree/branch due to Exception: {ex}')
        raise


def tree_pretty_print(tree, karoo=True):
    """
    prints a tree, each line ia a layer.
    looks a little bit better than printing the whole array
    """
    if karoo:
        tree = tree_convert_karoo_to_plagih(tree)

    depth = 0
    layer_labels = []
    print_style = 'Depth{:>3}: {}\n'  # {:>3} always print 3 letters at least
    node_depth = '-1'

    print_str = ''

    for node_id, node_depth in enumerate(tree[N_depth]):
        label = tree_node_get_label(tree, node_id)
        if int(node_depth) == depth:
            layer_labels.append(label)
        else:
            # print(print_style.formjat(node_depth, layer_labels))
            print_str += print_style.format(node_depth, layer_labels)
            layer_labels = [label]
            depth += 1
    else:
        # print(print_style.format(node_depth, layer_labels))
        print_str += (print_style.format(node_depth, layer_labels))

    return print_str


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
        tree = tree_convert_pcore_to_karoo(tree)

    id_list = []
    c_list = []
    for n in range(1, len(tree[3])):
        for child in range(0, 3):
            if tree[N_c1 + child][n] != '':
                c_list.append(int(tree[N_c1 + child][n]))
        id_list.append(int(tree[N_id][n]))
    if sum(c_list) == sum(id_list) - 1:
        return True
    else:
        return False


def tree_check_rebuild(tree):
    """
    Check if a valid tree can be rebuilt from its expression
    sfeh: the expression must currently not be equal.
    The expression can include separate '~' (usub) nodes, which makes expressions not completely equal
    """

    label_list = [tree_node_get_label(tree, ii) for ii in tree_iterate_range(tree)]
    arity_list = [tree_node_get_arity(tree, ii) for ii in tree_iterate_range(tree)]
    xtype_list = [tree_node_get_xtype(tree, ii) for ii in tree_iterate_range(tree)]

    try:
        core = Core_From_Labels(label_list, arity_list, xtype_list).get_uninstanced_core()
    except:
        return False
    return True


def tree_check_node_label_info(tree, obs_krazy=None, karoo=True):
    """
    A method to check if a tree is type consistant:
    - do the values in c1, c2, c3 link to its parent?
    """
    if not karoo:
        tree = tree_convert_pcore_to_karoo(tree)

    for node_id in tree_iterate_range(tree):
        label, arity, xtype = tree_node_get_lax_v3(tree, node_id)
        label_arity = label_get_arity(label)
        label_xtype = xtype_get_from_label(label)
        if arity != label_arity or xtype != label_xtype:
            print_e('Tree node info differs from label-version: arity: {}, {} xtype: {}, {}')
            return False
    return True


def tree_check_types(tree, karoo=True):
    """
    A method to check if a tree is type consistant:
    - do the values in c1, c2, c3 link to its parent?
    """
    if not karoo:
        tree = tree_convert_pcore_to_karoo(tree.copy())

    for node_id in tree_iterate_range(tree):
        label, arity, xtype = tree_node_get_lax_v3(tree, node_id)

        if xtype == 'b2f2f':
            xtypes_required = ['2b', '2f', '2f']
        else:
            xtypes_required = [xtype[:2][::-1]] * arity + [''] * (3-arity)  # ['2f', '2f', '']

        # children_xtypes = xtype_label_get_child_xtypes(label, arity, env_vars)
        # print('asd tree check types\n',
        #       tree_get_labellist(tree), '\n',
        #       [tree_node_get_xtype(tree, x) for x in tree_iterate_range(tree)])
        for ii, c_id in enumerate(tree_node_get_childs(tree, node_id)):
            c_label = tree_node_get_label(tree, c_id)
            c_xtype = tree_node_get_xtype(tree, c_id)
            if not xtypes_required[ii] == c_xtype[-2:]:
                print_e(f'Tree check failed. Node ({node_id}, {label}, {xtype}) at child {ii} reqquires {xtypes_required[ii]}, but node is ({c_id}, {c_label}, {c_xtype}).\n'
                        f'tree (pretty print):\n{tree_pretty_print(tree, karoo=True)}\n'
                        f'xtype_list: {[tree_node_get_xtype(tree, x) for x in tree_iterate_range(tree)]}\n'
                        f'Last modification was: {tree_get_history(tree)}')
                return False

    return True


def tree_evolve_node_insert(tree, env_vars):
    """
    Inserts a (arity-1) node into a tree
    Especially useful when a ** power shall be normalized
    """

    node_ids = tree_get_mutatable_nodes(tree)
    insert_id = None
    for node_id in node_ids:
        label = tree_node_get_label(tree, node_id)
        if label == '**' and tree_node_get_child(tree, node_id, 1) != 'Power':  # sfeh
            insert_id = node_id
            break

    if insert_id:
        old_ids, old_labels, old_aritys, old_xtypes = tree_get_branch_ilax(tree, insert_id)
        new_labels = ['DUMMY'] + old_labels
        new_aritys = [1] + old_aritys
        new_xtypes = ['f2f'] + old_xtypes
        insert_core = Core_From_Labels(new_labels, new_aritys, new_xtypes).get_uninstanced_core()
        tree_insert_subtree(tree, insert_core, old_ids, karoo=False)

    return tree


def label_constant_mutate(constant, term_type=float, filter_type='gaussian_filter', float_decimals=6):
    """
    When this happens, distributions_file get a a small variance
    """

    if term_type == float:
        if filter_type == 'gaussian_filter':
            if random.choice(['v1', 'v2']) == 'v1' or constant == 0:
                constant += np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(constant, 0.1)  # sfeh better adjustments?
        else:
            raise Exception('w', 'Warning: Filter  not specified. Please specify a filter_type.')
        constant = round_constant(constant, float_decimals)

    elif term_type == int:
        if random.choice(['v1', 'v2']) == 'v1' or constant == 0:
            term_filter = np.random.normal(0, 1)  # sfeh better adjustments?
            constant += term_filter
        else:
            constant = np.random.normal(constant, 1)  # sfeh
        constant = int(round(constant))

    elif term_type == bool:
        constant = random.choice([True, False])
        # random by 50:50?

    return constant


def tree_prune_depth(tree, max_depth, obs_krazy, choose_obs, choose_distributions, float_decimals):
    """
    reduces the depth of a Tree (in case it is too deep).
    Arguments required: tree, depth
    # sfeh prune node_count?
    """

    nodes = []

    for node_id in range(root_id, len(tree[3])):

        node_depth = tree_node_get_depth(tree, node_id)
        node_arity = tree_node_get_arity(tree, node_id)
        if node_depth == max_depth and node_arity > 0:  # replace this node with terminal
            label = tree_node_get_label(tree, node_id)
            xtype = xtype_get_from_label(label, obs_krazy)
            tree = tree_node_set_arity(tree, node_id, 0)
            new_term = choose_term(xtype[-2:], choose_obs, choose_distributions, float_decimals)  # replace label
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


def tree_eval_parsimony(cooltree, parsimony_distance, origin_cooltree=None, weights=None):
    """
    parsimony_distance: compute the chosen distance by the user.
    #     'tree_node_count': tree_get_size,
    #     'tree_depth': tree_get_depth,
    #     'tree_edit_distance': tree_parsimony_ted,
    """

    if parsimony_distance == 'tree_node_count':  # number of nodes
        return len(cooltree)  # returns the number of nodes
    if parsimony_distance == 'tree_edit_distance':  # tree_edit_distance, tree-edit-distance
        distance, mapping = tree_parsimony_ted(cooltree, origin_cooltree)
        if weights is None:
            return distance
        else:
            raise
    else:
        print_e(f'Complexity measurement not available: {parsimony_distance}')
        raise


def tree_check_is_sympified(tree):
    """
    Label list from expression
    """
    tree_raw = copy.deepcopy(tree)
    env_vars = 'ö'
    tree_sym = tree_evolve_reduce(tree, env_vars, completely=True)

    labellist_raw = tree_get_labellist(tree_raw)
    labellist_sym = tree_get_labellist(tree_sym)
    if list(labellist_raw) == list(labellist_sym):
        return True
    else:
        return False
