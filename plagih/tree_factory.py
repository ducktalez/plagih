"""
The factory to create trees
"""
import copy
import random

import numpy as np
import logging
from pathlib import Path

from plagih.file_interaction import yaml_load
from plagih.plagih_tree import *


class Selectable:
    """

    """

    def select(self, xtype):
        return


class ChooseOperators(Selectable):
    """

    """

    def __init__(self, operator_pool=None):
        """

        """

        def check_operator_pool(operator_pool):
            """
            Check if the user-specified loaded operators allow closure
            (either float-only/bool only or all 4 types of operators)
            @:param operator_pool: list with operators and their weight of being selected
            """
            # sfeh dunno if that works... 2f not in x
            opxtypes = [oper.xtype for oper in operator_pool.keys()]
            has_2f = any([float == x[1] for x in opxtypes])
            has_2b = any([bool == x[1] for x in opxtypes])
            has_f2b = any([float in x[0] and bool == x[1] for x in opxtypes])
            has_b2f = any([bool in x[0] and float == x[1] for x in opxtypes])
            if not all([has_2f, has_2b, has_f2b, has_b2f]):
                logging.warning(f'Loaded operators do not feature both numeric (float) and Boolean type.')
            if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
                raise Exception(f'Loaded operators do not allow closure!')

        if operator_pool is None:  # quick developer adjustments
            operator_pool = [['+', 2],
                             ['-', 1], ['Usub', 1],
                             ['*', 2], ['/', 1],
                             ['Square', 0.75], ['**', 0.25],
                             ['Abs', 0.5], ['sign', 0.5], ['Round', 0.5],  # sfeh stop chain of arity-1 op in buid method?
                             ['sqrt', 0.25],
                             # ['log', 0.1], ['log1p', 0.1],
                             ['sin', 0.5],  # ['tan', 0.1], ['cos', 0.33], ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
                             ['tanh', 0.2],
                             ['Andb', 1], ['Orb', 1], ['Notb', 0.5], ['Xor', 1],
                             ['==', 1], ['!=', 0.5],
                             ['<', 0.5], ['<=', 0.5], ['>', 0.1], ['>=', 0.1],
                             ['Ifte', 2],
                             ['Mini', 1], ['Maxi', 1]]
            operator_pool = {op[x[0]]: x[1] for x in operator_pool}  # sfeh this maps the actual class to the label

        # if sfeh_no_crazyops:
        #     del operator_pool['**']
        #     # workaround sfeh (delete this)

        check_operator_pool(operator_pool)

        choose_oparray = {
            # all operator_pool with a certain xtype-result
            # None: [[], []],
            float: [[], []],  # 2f
            bool: [[], []],  # 2b
            (tuple([]), float): [[], []],  # todo?  replacing float
            (tuple([]), float): [[], []],  # todo?
            (tuple([float]), float): [[], []],  # x**2, sqrt, log, sin, ...
            (tuple([float, float]), float): [[], []],  # +, -, *, /, **, ...
            (tuple([bool, float, float]), float): [[], []],  # Ifte
            (tuple([float, float]), bool): [[], []],  # <, >, =, >=
            (tuple([bool]), bool): [[], []],  # not
            (tuple([float]), bool): [[], []],  # dummy, currently no such operator
            (tuple([bool, bool]), bool): [[], []],  # and, or, xor, ...
        }
        for label, prob in operator_pool.items():
            # tuple-xtype (point mutations)
            choose_oparray[label.xtype][0].append(label)
            choose_oparray[label.xtype][1].append(prob)
            # float/bool (construction of trees)
            choose_oparray[label.xtype[1]][0].append(label)
            choose_oparray[label.xtype[1]][1].append(prob)

        for o, p in choose_oparray.items():
            # normalizing the probabilities in every case to a sum of 1 (100%)
            # (saving some very little time...)
            if p[0]:
                choose_oparray[o][1] = [x / sum(p[1]) for x in p[1]]
            else:
                pass  # todo delete line?

        self.choose_oparray = {}
        for xtype, x in choose_oparray.items():
            # self.choose_oparray[xtype] = lambda: np.random.choice(x[0], p=x[1])  # "seloplam" faster, but less readable version
            self.choose_oparray[xtype] = (x[0], x[1])

    def select(self, xtype):
        """

        """
        return np.random.choice(self.choose_oparray[xtype][0], p=self.choose_oparray[xtype][1])
        # return self.choose_oparray[xtype]()  # "seloplam"


class ChooseConstants(Selectable):
    """

    """
    # sfeh random with numpy?
    distributions = {float: [lambda: random.normalvariate(0, 1),
                             lambda: random.normalvariate(1, 1),
                             lambda: random.normalvariate(10, 5),
                             lambda: random.randint(1, 20)],  # 0 has actually no purpose (except as being an action)
                     bool: [lambda: random.choice([True, False])]}

    def __init__(self, float_decimals=6, path_distrib=Path.cwd(), data_train=None, n_samples=100):
        """
        todo path.cwd() is no good input
        """
        self.float_decimals = float_decimals
        try:
            lambdadist_as_string = yaml_load(path_distrib)
            # todo how should distributions be loaded?
            # e.g. sample_amount = lambdadist_as_string.get('observed_floats')
            self.terminal_distributions = {float: [], bool: []}
            self.terminal_distributions[float].extend([eval(x) for x in lambdadist_as_string[float]]),
            self.terminal_distributions[bool].extend([eval(x) for x in lambdadist_as_string[bool]])

            # self.sample_floats_from_data(obs_infos, data_train, n_samples=n_samples)  # todo
        except Exception:
            logging.info('Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.')

    def sample_floats_from_data(self, obs_infos, data_train, n_samples=100):
        """
        ONLY floats, because ...do you really want to load Boolean True/False samples??
        (okay, it might make sense as it better represents the actual distribution- NO FUCK IT.)
        """
        if obs_infos is not None:
            obsnames = obs_infos.observables[float].keys()
            obs_samples = data_train[obsnames].to_numpy().flatten()
            obs_samples = np.random.choice(obs_samples, size=n_samples)
            self.terminal_distributions[float].extend([lambda: random.choice(obs_samples)]),  # take one

    def select(self, xtype):
        """

        """
        value = random.choice(self.distributions[xtype])()
        if xtype == float:  # sfeh int aswell?
            value = float(round(value, self.float_decimals))
            return FloatConstant(value)
        elif xtype == bool:
            return BoolConstant(value)


class ChooseObservation(Selectable):
    """
    func_list, probability_list = self.operators[xtype]
    return np.random.choice(func_list, p_op=probability_list)
    """

    def select(self, xtype):
        """
        Randomly choosing an operator-label for a given xtype.
        choose_oparray3 must be given, as they are different between runs.
        arity can also be set optionally, e.g. for point mutation
        todo DOUBLE-check if this xtype is chosen correctly... better: replace it
        """
        return self.observables[xtype]()

    def __init__(self, observations_list, todo=False):
        """
        :param observations_list: list of all observation names (e.g. ['cartVel', 'cartPos'])
        """

        def observation_select_index(obs_list, max_hist=10):
            """
            chooses variables but weighting how old they are.
            obs_list = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4'] -> [0.28, 0.23, 0.19, 0.16, 0.13]
            sfeh: what about larger steps?
            e.g. [0, 1, 2, 3] is good, but [0, 5, 10, 15] is baaaad
            what if variables are not all of same diff?
            """
            obs_list = np.delete(obs_list, np.s_[max_hist:])
            x = len(obs_list)
            fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
            p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
            p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
            return np.random.choice(obs_list, p=p)  # returning a function this time

        obs_prop = []
        obs_info = {}

        if todo:
            for fam in list(set(observation_get_family_and_time(x)[0] for x in observations_list)):
                fam_members = sorted([x for x in observations_list if x.fam == fam], key=lambda o: o.obs_index)
                if len(fam_members) > 1:
                    observations_list.extend([x for x in fam_members])
                    obs_prop.extend(list(observation_select_index(fam_members)))
                    index_minmax = (fam_members[0].obs_index, fam_members[-1].obs_index)
                    for obs in fam_members:
                        obs.index_minmax = index_minmax
                        # environment.obs_infos[obs.name] = obs  # todo okay do we need this? :s guess we will find out x.D haha
                        obs_info[obs.name] = obs
                else:
                    obs = fam_members[0]
                    obs_info[obs.name] = obs
                    observations_list.append(obs)
                    obs_prop.append(1)  # just one value
                    # todo

        self.observables = {float: lambda: np.random.choice(observations_list, p=obs_prop),
                            bool: None}  # sfeh None? no  lambda? yeah, not important but still...


class TreeBuilder:
    # class Choosing(Selectable):  # sfeh was
    """
    Just a class to prevent referencing all the separate shizzle everytime
    todo float_decimals?

    ===
    this was:
    def choose_op():
    choose_oparray3 -> operator
    env_vars.choose_obs -> observation
    choose_distributions -> constant
    """

    def __init__(self, operators: ChooseOperators, observations: ChooseObservation, constants: ChooseConstants, root_xtype):
        self.operators = operators
        self.observations = observations
        self.constants = constants

        self.root_xtype = root_xtype

    def choose_any(self, xtype, p_op):
        """

        """
        if random.random() < p_op:
            return self.operators.select(xtype)
        else:
            # sfeh add p_term? 0.5?
            return self.choose_term(xtype)

    def choose_term(self, xtype, p_observation=0.5):
        """
        sfeh: float_decimals not required?
        # sfeh 50% chance observatio    n/value
        """
        if random.random() < p_observation:
            try:
                return self.observations.select(xtype)
            except Exception:
                pass

        return self.constants.select(xtype)

    def choose_op(self, any_xtype):
        """
        any_xtype can be a tuple, single type or even None
        """
        return self.operators.select(any_xtype)

    def invent_core_depth(self, xtype, depth_max, depth=0, p_op=1):  # todo grow method
        """
        # set path/id? todo
        # set depth? todo
        """
        if depth < depth_max:
            label = self.choose_any(xtype, p_op)
            childs = [self.invent_core_depth(xt, depth_max, depth=depth+1, p_op=p_op) for xt in label.xtype[0]]
            node = Node(label=label, childs=childs)  # , depth=depth sfeh no depth?
        else:
            label = self.choose_term(xtype)
            node = Node(label=label)
        return node

    def evolve_mutate_point(self, tree: Node):
        """
        Mutate a single mutatable point in any Tree.
        sfeh is the tree a tree copy or the same tree?
        """
        etree = copy.deepcopy(tree)  # todo ==>state

        node = np.random.choice(etree.eval_mutatable_nodes())
        xtype = node.get_xtype()

        if node.get_arity() > 0:
            node.set_label(tb.choose_op(xtype))  # Function is same type, same arity
        else:
            node.set_label(tb.choose_term(xtype[1]))  # 3 -> '2f' -> 5

        etree.finalize()  # todo ==>state
        return etree

    def evolve_mutate_branch_depth(self, tree, depth_max):
        """
        todo ==>depth only
        currently only one branch
        """
        etree = copy.deepcopy(tree)  # todo ==>state

        node = np.random.choice(etree.eval_mutatable_nodes())
        xtype = node.get_xtype()[1]  # todo todotodo

        branch = self.invent_core_depth(xtype, 3, depth=0, p_op=1)  # todo ==>dummies
        node.replace_with_branch(branch)
        # if node.depth == depth_max:
        #     node.set_label(tb.choose_term(xtype[1]))  # sfeh update node plabel
        # else:
        #     node.childs = [Node(tb.choose_any(xt, p=1)) for xt in node.get_xtype()[0]]  # todo ==>

        etree.finalize()  # todo ==>state
        return etree

    def evolve_crossover(self, tree1: Node, tree2: Node):
        """
        todo ==>depth only
        currently only one branch
        """
        atree = copy.deepcopy(tree1)  # todo ==>state
        btree = copy.deepcopy(tree2)  # todo ==>state

        try:
            anodes = atree.eval_mutatable_nodes(allow_root=False)
            anode = np.random.choice(anodes)
            xtype = anode.get_xtype_out()
            bnodes = btree.eval_mutatable_nodes(xtype_out=xtype)
            if bnodes:
                bnode = np.random.choice(bnodes)
            else:
                xtype = float if xtype == bool else bool  # the other swap type now
                bnodes = btree.eval_mutatable_nodes(allow_root=False, xtype_out=xtype)
                bnode = np.random.choice(bnodes)
                anode = atree.eval_mutatable_nodes(xtype_out=xtype)
        except:
            raise Exception

        # todo deepcopy required??
        anode_copy = copy.deepcopy(anode)  # todo ==>state
        # bnode = copy.deepcopy(bnode)  # todo ==>state

        anode.replace_with_branch(bnode)
        bnode.replace_with_branch(anode_copy)

        # atree.meta.last_evolution = tag  # todo ==>tree tag
        # btree.meta.last_evolution = tag
        # self.pop_append(left_parent)
        # self.pop_append(right_parent)

        return atree, btree

    def pop_random(self, call_params, from_origin=False):
        """
        Creates random trees for the population
        """
        build_spec, size_mode, mean_min_max_var, full_or_grow = helper_evolve_params_branch(call_params)

        if from_origin:
            """
            insert a (random) number of branches at the first possible "layer"
            (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
            - get these nodes, randomly choose a subset of those
            - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
            - split the amount of nodes up (randomly) and add these new branches to the tree
            """

            # layer0_ids = tree_get_mutatable_layer(from_origin, 0)
            layer0_ids = [1, 2, 3]

            # build_split = []
            # if 'depth' in size_mode:
            #     for ii in range(len(layer0_ids)):
            #         build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')
            #         build_split.append(build_size)
            #
            # elif 'nodes' in size_mode:
            #     build_nodes = choose_build_size(size_mode, mean_min_max_var, force='branch')
            #     build_split = randomly_split_range(build_nodes, len(layer0_ids))
            # else:
            #     raise
            #
            # tree = copy.deepcopy(from_origin)
            # for i in range(len(layer0_ids)):  # insert branches! get layer every time (node ids might have changed)
            #     layer0_ids = tree_get_mutatable_layer_lv0(tree)
            #     node_id = layer0_ids[i]
            #     first_xtype = float  # tree_node_get_xtype(tree, node_id)  # todo
            #     old_branch = tree_node_get_branch(tree, node_id, karoo=True)
            #     build_size = build_split[i]
            #
            #     # tree = tree(BuildDummy(float))   # todo deprecated
            #     core = tree.invent_core(size_mode, first_xtype, build_size, full_or_grow)
            #     tree = tree_insert_subtree(tree, core, old_branch, karoo=True)
        else:
            build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')  # depth, in this case
            # todo
            # coolcore.evolve_mutate_branch_depth(build_size, choose_oparray3, env_vars.choose_obs,
            #                                      choose_distributions, float_decimals, size_mode=size_mode, full_or_grow=full_or_grow)
            # coolcore.evolve_random_tree_depth(size_mode, xtype_root, build_size, full_or_grow)

        # return coolcore


# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['label'], 1/v['xtype': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


# def choose_term(xtype_out, choose_obs, choose_distributions, float_decimals):
#     """
#
#     """
#
#     # sfeh 50% chance observation/value
#     if random.choice(['obs', 'distrib']) == 'obs' and choose_obs[xtype_out]:
#         obs = choose_obs[xtype_out]()
#         # print('SAME???', obs.name, obs.label)  # sfeh
#         return obs
#     else:
#         dist_fun = random.choice(choose_distributions[xtype_out])
#         value = dist_fun()
#         if xtype_out == float:  # sfeh int aswell?
#             value = float(round(value, float_decimals))
#             const = FloatConstant(value)
#         elif xtype_out == bool:
#             const = BoolConstant(value)
#         else:
#             raise Exception('ASDASD NOOO WHYY')
#         return const


def helper_evolve_params_branch(call_params, tree_depth_max=10, parsimony_max=30):
    """
    The call parameters in the evolution file need to be adjusted
    delete if possible
    """
    build_spec = call_params.get('build_max')

    size_mode = build_spec['size_mode']

    mean_min_max_var = build_spec.get('mean_min_max_var')  # (base, min, max, normal_distrib)
    mean_min_max_var = list(mean_min_max_var)
    if 'depth' in size_mode:
        max_dummy = tree_depth_max
    elif 'nodes' in size_mode:
        max_dummy = parsimony_max
    else:
        raise

    if mean_min_max_var[2] is None:
        mean_min_max_var[2] = max_dummy
    else:
        mean_min_max_var[2] = min(mean_min_max_var[2], parsimony_max)
    mean_min_max_var = tuple(mean_min_max_var)

    full_or_grow = build_spec['full_or_grow']

    return build_spec, size_mode, mean_min_max_var, full_or_grow


def randomly_split_range(range_max, num_splits):
    """
    todo reuse this
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


def shuffle_tree_construction(size):
    """
    todo reuse this
    """
    terms = random.randint(0, size - 1)  # -1 because at least one 'func'
    cons_buf = ['term'] * terms + ['func'] * (size - terms)
    np.random.shuffle(cons_buf)
    return cons_buf


def choose_build_size(size_mode, mean_min_max_var, tree=None, nodepath=None, force=None):
    """
    delete this?
    Very unified utility function that returns the required tree size from the following parameters
    # branch_nodes, branch_depth, tree_depth, tree_nodes

    It can either return a tree depth or an amount of tree nodes
    """
    mean, size_min, size_max, size_variance = mean_min_max_var
    if size_mode == 'branch_nodes' or size_mode == 'branch_depth' or force == 'branch':
        relative_size = 0
    else:
        if tree and nodepath:
            pass
        else:
            raise Exception('No tree or node is given for computing the relative size')

        if size_mode == 'tree_depth':
            tree_size = tree.core.childs_depth_max
            print('tree_size = tree.core.childs_depth_max:', tree_size)
            if tree_size is None and 'delete_this':
                raise Exception('ASDASDASD')
            node_size = len(nodepath)
        elif size_mode == 'tree_nodes':
            tree_size = len(tree)
            print('len(tree)?:', len(tree))
            node_size = len(tree.get_nodepath(nodepath))
            print('tree.get_nodepath(nodepath)?:', tree.get_nodepath(nodepath))
        else:
            raise Exception('Sizemode not known?')

        relative_size = tree_size - node_size
        print('asdasd', relative_size)

    build_size = int(random.normalvariate(mean, size_variance))
    if size_max is not None:
        build_size = min(size_max - relative_size, build_size)
    build_size = max(size_min, build_size)

    return int(build_size)


if __name__ == '__main__':
    """
    Alpha tests
    """
    ops = ChooseOperators()
    inputs = ChooseObservation(['a', 'b'])
    consts = ChooseConstants()
    tb = TreeBuilder(ops, inputs, consts, float)
    t1 = tb.invent_core_depth(float, 3, p_op=0.5)
    tree2 = tb.evolve_mutate_point(t1)
    tree3 = tb.evolve_mutate_branch_depth(tree2, depth_max=4)
    tree4 = tb.evolve_mutate_branch_depth(tree3, depth_max=4)
    tree5, tree6 = tb.evolve_crossover(tree3, tree4)
    print('crossover')
    t3 = tb.invent_core_depth(float, 4, p_op=0.5)
    t4 = tb.invent_core_depth(float, 3, p_op=0.6)
    print('tree3', tree3)
    print('tree4', tree4)
    print('tree5', tree5)
    print('tree6', tree6)
