"""
The factory to create trees
"""

from plagih.plagih_tree import *
from plagih.node_labels import *
from plagih.plagih_tree import Node
from plagih.sympy_extras import expr_sympify
from plagih.util import *

import copy
import random
from collections import deque
import numpy as np
import logging
from pathlib import Path


def randomly_split_range(range_max, num_splits):
    """
    split a integer range randomly into parts
    [1..100] -> [33, 15, 52]
    0 is allowed! (ends a branch with a terminal node)
    """

    # tmp_distributions = random.sample(range(1, range_max), num_splits)
    # d_sum = sum(tmp_distributions)
    # d_list = [int(round(range_max*(x/d_sum), 0)) for x in tmp_distributions]
    # if num_splits == 0:
    try:
        sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
        d_sum = sum(sample_dist)  # 1.5
        sample_dist = [x / d_sum for x in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
        sample_dist = [x * range_max for x in sample_dist]  # [12, 60, 28] -> for 100 nodes
        sample_dist = [int(round(x, 0)) for x in sample_dist]  # make them useable ints

        # sfeh workaround, this makes exactly the correct range by changing the most extreme entry
        imprecise_diff = range_max - sum(sample_dist)  # sfeh: this can be [0, 0, 0], which assigns to the 0th bin...
        # sfeh:discussion: maybe this difference is 2 or larger more often than 1 (->rounding), so maybe while-loop (just check if it happens?)
        if imprecise_diff != 0:
            if sum(sample_dist) < range_max:
                # sfeh:minor mistake: if relatively empty, this appends to the first bin
                sample_dist[sample_dist.index(min(sample_dist))] += imprecise_diff  # extreme_bin = smallest
            elif sum(sample_dist) > range_max:
                sample_dist[sample_dist.index(max(sample_dist))] += imprecise_diff  # extreme_bin = greatest
            else:
                raise
    except Exception as ex:
        sample_dist = [range_max]

    return sample_dist


def choose_build_size(size_mode, mean_min_max_var, tree=None, nodepath=None, force=None):
    """
    delete this?
    Very unified utility function that returns the required fintree size from the following parameters
    # branch_nodes, branch_depth, tree_depth, tree_nodes

    It can either return a fintree depth or an amount of fintree nodes
    """
    mean, size_min, size_max, size_variance = mean_min_max_var
    if size_mode == 'branch_nodes' or size_mode == 'branch_depth' or force == 'branch':
        relative_size = 0
    else:
        if tree and nodepath:
            pass
        else:
            raise Exception('No fintree or node is given for computing the relative size')

        if size_mode == 'tree_depth':
            tree_size = tree.core.childs_depth_max
            print('tree_size = fintree.core.childs_depth_max:', tree_size)

            node_size = len(nodepath)
        elif size_mode == 'tree_nodes':
            tree_size = len(tree)
            print('len(fintree)?:', len(tree))
            node_size = len(tree.get_nodepath(nodepath))
            print('fintree.get_nodepath(nodepath)?:', tree.get_nodepath(nodepath))
        else:
            raise Exception('Sizemode not known?')

        relative_size = tree_size - node_size
        print('scheduled_io', relative_size)

    build_size = int(random.normalvariate(mean, size_variance))
    if size_max is not None:
        build_size = min(size_max - relative_size, build_size)
    build_size = max(size_min, build_size)

    return int(build_size)


class TreeBuilder:
    # class Choosing(Selectable):  # sfeh was
    """
    Just a class to prevent referencing all the separate shizzle everytime

    func_list, probability_list = self.operators[xtype_out]
    return np.random.choice(func_list, p_full=probability_list)
    """

    # sfeh random with numpy?
    distributions = {float: [lambda: random.normalvariate(0, 1),
                             lambda: random.normalvariate(1, 1),
                             lambda: random.normalvariate(10, 5),
                             lambda: random.randint(1, 20)],  # 0 has actually no purpose (except as being an action)
                     bool: [lambda: random.choice([True, False])]}  # sfeh:discussion

    def __init__(self, obs_names, conf, operator_pool=None, root_xtype=float, csv_data_samples=None, precision=6):
        self.operators_add(operator_pool)
        self.constants_add()
        self.observations_add(obs_names)
        self.root_xtype = root_xtype
        self.precision = precision
        if conf:
            self.tree_depth_max = conf.tree_depth_max
            self.parsimony_max = conf.parsimony_max
            self.print_type = conf.print_type
        else:
            # Loading some random default options for quick debugging, without loading a config
            self.tree_depth_max = 10
            self.parsimony_max = 50
            self.print_type = None

            # class ChooseOperators(Selectable):

    def operators_add(self, operator_pool=None):
        """

        """

        def operator_pool_check(operator_pool):
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
                             ['Square', 0.75],
                             # ['**', 0.25],  # sfeh:open
                             ['abs', 0.5], ['sign', 0.5],  # sfeh stop chain of arity-1 op_dict in buid method?
                             # ['sqrt', 0.25],  # sfeh debug this
                             # ['log', 0.1], ['log1p', 0.1],
                             ['sin', 0.5],
                             # ['tan', 0.1], ['cos', 0.33], ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
                             ['tanh', 0.2],
                             # ['xor', 1],  # sfeh
                             # sympy extra classes (Capitalized)
                             ['Round', 0.5],
                             ['Andb', 1], ['Orb', 1], ['Notb', 0.5],
                             ['==', 1], ['!=', 0.5],
                             ['<', 0.5], ['<=', 0.5], ['>', 0.1], ['>=', 0.1],
                             ['Ifte', 2],
                             ['Mini', 1], ['Maxi', 1]]
            operator_pool = {op_dict[x[0]]: x[1] for x in operator_pool}  # sfeh this maps the actual class to the label

        # if no_crazyops:
        #     del operator_pool['**']
        #     # workaround sfeh (delete this)

        operator_pool_check(operator_pool)

        choose_oparray = {
            # all operator_pool with a certain xtype_out-result
            # None: [[], []],
            float: [[], []],  # 2f
            bool: [[], []],  # 2b
            (tuple([]), float): [[], []],  # sfeh?  replacing float
            (tuple([]), bool): [[], []],  # sfeh? not required? empty
            (tuple([float]), float): [[], []],  # x**2, sqrt, log, sin, ...
            (tuple([float, float]), float): [[], []],  # +, -, *, /, **, ...
            (tuple([bool, float, float]), float): [[], []],  # Ifte
            (tuple([float, float]), bool): [[], []],  # <, >, =, >=
            (tuple([bool]), bool): [[], []],  # not
            (tuple([float]), bool): [[], []],  # dummy, currently no such operator
            (tuple([bool, bool]), bool): [[], []],  # and, or, xor, ...
        }
        for label, prob in operator_pool.items():
            # tuple-xtype_out (point mutations)
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
                pass  # sfeh debug delete the entry?

        self.operators = {}
        for xtype, x in choose_oparray.items():
            # self.operators[xtype_out] = lambda: np.random.choice(x[0], p=x[1])  # "seloplam" faster, but less readable version
            self.operators[xtype] = (x[0], x[1])

    def choose_op(self, any_xtype):
        """
        any_xtype can be a tuple, single type or even None
        """
        # return self.operators[xtype_out]()  # "seloplam"
        return np.random.choice(self.operators[any_xtype][0], p=self.operators[any_xtype][1])

    # def workaround_remove_tilde(self, tree):
    #     """
    #     sfeh
    #     What is the role of the tilde in a label?
    #     I think it came from sympy... or i introduced it to trick sympy in some way
    #     I guess it is something like -
    #     """
    #     if isinstance(tree.get_label(), Usub):  # tilde '~'
    #         new_core = tree.childs[0]
    #         tree.replace_with_branch(new_core)
    #
    #     for cc in tree.childs:
    #         cc.workaround_remove_tilde()
    #
    #     return tree  # sfeh:delete_this debug/check if this changes the og fintree check

    def constants_add(self, path_distrib=Path.cwd(), data_train=None, n_samples=100):
        """
        sfeh:open path.cwd() is no good input
        """
        try:
            lambdadist_as_string = yaml_load(path_distrib)
            # sfeh:discussion how should distributions be loaded?
            # e.g. sample_amount = lambdadist_as_string.get('observed_floats')
            self.distributions = {float: [], bool: []}
            self.distributions[float].extend([eval(x) for x in lambdadist_as_string[float]]),
            self.distributions[bool].extend([eval(x) for x in lambdadist_as_string[bool]])

            # self.constants_add_data_samples(obs_infos, data_train, n_samples=n_samples)  # sfeh:open
        except Exception as ex:
            logging.info(
                'Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.')

    def constants_add_data_samples(self, obs_infos, data_train, n_samples=100):
        """
        ONLY floats, because ...do you really want to load Boolean True/False samples??
        (okay, it might make sense as it better represents the actual distribution- NO FUCK IT.)
        """
        if obs_infos is not None:
            obsnames = obs_infos.observables[float].keys()
            obs_samples = data_train[obsnames].to_numpy().flatten()
            obs_samples = np.random.choice(obs_samples, size=n_samples)
            self.distributions[float].extend([lambda: random.choice(obs_samples)]),  # take one

    def observations_add(self, obs_names):
        """
        :param obs_names: list of all observation names (e.g. ['cartVel', 'cartPos'])
        """
        # def observation_select_index(observations, max_hist=10):
        #     """
        #     chooses variables but weighting how old they are.
        #     observations = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4'] -> [0.28, 0.23, 0.19, 0.16, 0.13]
        #     sfeh: what about larger steps?
        #     e.g. [0, 1, 2, 3] is good, but [0, 5, 10, 15] is baad
        #     what if variables are not all of same diff?
        #     """
        #     observations = np.delete(observations, np.s_[max_hist:])
        #     x = len(observations)
        #     fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
        #     p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
        #     p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
        #     return np.random.choice(observations, p=p)  # returning a function this time
        #
        # obs_prop = []
        # obs_info = {}
        #
        # for fam in list(set(observation_get_family_and_time(x)[0] for x in obs_names)):
        #     fam_members = sorted([x for x in obs_names if x.fam == fam], key=lambda o: o.timeindex)
        #     if len(fam_members) > 1:
        #         obs_names.extend([x for x in fam_members])
        #         obs_prop.extend(list(observation_select_index(fam_members)))
        #         index_minmax = (fam_members[0].timeindex, fam_members[-1].timeindex)
        #         for obs in fam_members:
        #             obs.index_minmax = index_minmax
        #             # environment.obs_infos[obs.name] = obs  # sfeh:open okay do we need this? :s guess we will find out x.D haha
        #             obs_info[obs.name] = obs
        #     else:
        #         obs = fam_members[0]
        #         obs_info[obs.name] = obs
        #         obs_names.pop_append_evotree(obs)
        #         obs_prop.pop_append_evotree(1)  # just one value

        obs_list = [Observation(x) for x in obs_names]

        self.observables = {float: lambda: np.random.choice(obs_list),  # , p=obs_prop
                            bool: None}  # sfeh:discussion None? no  lambda? yeah, not important but still...

    def choose_obs(self, xtype):
        """
        Randomly choosing an operator-label for a given xtype_out.
        choose_oparray3 must be given, as they are different between runs.
        arity can also be set optionally, e.g. for point mutation
        sfeh:open DOUBLE-check if this xtype_out is chosen correctly... better: replace it
        """
        return self.observables[xtype]()

    def choose_const(self, xtype):
        """

        """
        value = random.choice(self.distributions[xtype])()
        if xtype == float:
            value = float(round(value, self.precision))
            return FloatConstant(value)
        elif xtype == bool:
            return BoolConstant(value)

    def choose_term(self, xtype, p_observation=0.5):
        """
        sfeh: precision not required?
        # sfeh 50% chance observatio    n/value
        """
        if random.random() < p_observation:
            try:
                return self.choose_obs(xtype)
            except Exception:
                pass  # just return a constant now, e.g. because there are no boolean observations

        return self.choose_const(xtype)

    def choose_any(self, xtype, p_full):
        """

        """
        if random.random() < p_full:
            return self.choose_op(xtype)
        else:
            # sfeh add p_term? 0.5?
            return self.choose_term(xtype)

    def invent_core_nodeops(self, xtype_out, nodeops_max, p_full, depth):
        """
        This version counts the amount of operators as construction limit!
        sfeh:idea nodes are now about being operators...
        """
        childs = []
        label = self.choose_op(xtype_out)

        if nodeops_max > 0:
            nodeops_max -= 1

            nodeops_split = randomly_split_range(nodeops_max, label.arity)

            childs = [self.invent_core_nodeops(xt_out, nodeops_split[ii], p_full, depth + 1) for ii, xt_out in
                      enumerate(label.xtype[0])]

        else:
            label = self.choose_term(xtype_out)

        node = Node(label=label, childs=childs, depth=depth)  # , depth=depth sfeh no depth?

        return node

    def invent_core_depth(self, xtype, depth_max, p_full=1.0, depth=0):  # sfeh:check grow method
        """
        # sfeh:discussion set path/id?
        """
        if depth < depth_max:
            label = self.choose_any(xtype, p_full)
            childs = [self.invent_core_depth(xt, depth_max, p_full, depth=depth + 1) for xt in label.xtype[0]]
            node = Node(label=label, childs=childs, depth=depth)  # , depth=depth sfeh no depth?
        else:
            label = self.choose_term(xtype)
            node = Node(label=label, depth=depth)
        return node

    def evotree_deepcopy(self, tree: Node):
        """
        sfeh:==>stuff
        """
        evotree = copy.deepcopy(tree)  # sfeh==>state
        return evotree

    def evolve_mutate_filter_random(self, evotree, custom_params):
        """
        Mutates a number of float terminal of a fintree
        - sfeh:==>ROOT
        - filter point/branch/all, branch can also affect a point only aswell as all nodes
        - filter observations?
        - filter terminals
        - filter with which filter?
        """

        # filter_mode = custom_params['filter_mode']
        # filter_observations = custom_params['filter_observations']
        # mutate_filter = 'gaussian_filter'  # sfeh:future

        node = np.random.choice(evotree.eval_mutatable_nodes())
        node.evolve_mutate_filter_branch(precision=self.precision)

        # sfeh ==>state
        return evotree

    def evolve_mutate_point(self, tree: Node):
        """
        Mutate a single mutatable point in any Tree.
        sfeh is the fintree a fintree copy or the same fintree?
        """
        evotree = self.evotree_deepcopy(tree)  # ==>state

        node = np.random.choice(evotree.eval_mutatable_nodes())
        xtype = node.get_xtype()

        if node.get_arity() > 0:
            node.set_label(self.choose_op(xtype))  # Function is same type, same arity
        else:
            node.set_label(self.choose_term(xtype[1]))  # 3 -> '2f' -> 5

        # sfeh==>state
        return evotree

    def evolve_mutate_branch_depth(self, evotree, depth_goal, p_full=1.0):
        """
        evotree, cool_build_size, p_full=p_full

        sfeh ==>depth only
        currently only one branch
        """
        node = np.random.choice(evotree.eval_mutatable_nodes())
        xtype_out = node.get_xtype_out()

        branch = self.invent_core_depth(xtype_out, depth_goal, p_full, depth=0)  # sfeh ==>dummies
        node.replace_with_branch(branch)
        # if node.depth == depth_goal:
        #     node.set_label(tb.choose_term(xtype_out))  # sfeh update node nlabel
        # else:
        #     node.childs = [Node(tb.choose_any(xt, p=1)) for xt in node.get_xtype()[0]]  # sfeh ==>

        # etree.finalize()  # sfeh ==>state
        return evotree

    def evolve_mutate_branch_nodes(self, evotree, nodes_goal, p_full=1.0):
        """
        evotree, cool_build_size, p_full=p_full

        sfeh ==>depth only
        currently only one branch
        """
        node = np.random.choice(evotree.eval_mutatable_nodes())
        xtype_out = node.get_xtype_out()
        branch = self.invent_core_nodeops(xtype_out, nodes_goal, p_full, depth=0)  # sfeh ==>dummies
        node.replace_with_branch(branch)
        return evotree

    def evolve_crossover(self, tree1: Node, tree2: Node):
        """
        currently only one branch
        """
        atree = self.evotree_deepcopy(tree1)  # ==>state
        btree = self.evotree_deepcopy(tree2)  # ==>state, was: btree = copy.deepcopy(tree2)

        anodes = atree.eval_mutatable_nodes(allow_root=False)
        anode = np.random.choice(anodes)

        xtype_out = anode.get_xtype_out()
        bnodes = btree.eval_mutatable_nodes(xtype_out=xtype_out)
        if len(bnodes) > 0:
            bnode = np.random.choice(bnodes)
        else:
            xtype_out = float if xtype_out == bool else bool  # the other swap type now
            bnodes = btree.eval_mutatable_nodes(xtype_out=xtype_out)
            bnode = np.random.choice(bnodes)
            anodes = atree.eval_mutatable_nodes(allow_root=False, xtype_out=xtype_out)
            anode = np.random.choice(anodes)

        # sfeh deepcopy required??
        anode_copy = copy.deepcopy(anode)  # sfeh ==>state

        anode.replace_with_branch(bnode)
        bnode.replace_with_branch(anode_copy)

        # todo set depth correctly instead of repair!
        atree.repair_depth(depth=0)
        btree.repair_depth(depth=0)

        atree = self.evolve_prune(evotree=atree)
        btree = self.evolve_prune(evotree=btree)

        # atree.meta.last_evolution = tag  # sfeh ==>fintree tag
        # btree.meta.last_evolution = tag

        return atree, btree

    def printpl(self, message_type, message_str):
        """
        clone of the popular function
        """
        printez(message_type, message_str, print_type=self.print_type)
        return

    def evolve_prune(self, evotree: Node):
        """
        prune depth
        -> prune everything below a certain level... (should not happen in the first place)
        prune nodes
        -> get node difference, get nodelist, untill small enough: split the difference, prune nodes until

        sfeh:discussion there is a difference between parsimony and complexity...
        sfeh:discuss analyze the amount of trees that have to be pruned?
        """

        nodelist = evotree.eval_mutatable_nodes()
        for dnode in nodelist:
            if dnode.depth == self.parsimony_max and dnode.get_arity() > 0:
                print_warning('wwww', f'Node in fintree is too deep: {dnode.depth}', print_type=self.print_type)
                new_node = Node(label=self.choose_term(dnode.get_xtype_out()), depth=dnode.depth)
                dnode.replace_with_branch(new_node)
                # sfeh:debug did this work?

        prune_amount = len(evotree) - self.parsimony_max
        while prune_amount > 0:
            print_warning('wwww',
                          f'Tree is too complex: {len(evotree)} > {self.parsimony_max}, pruning {prune_amount} nodes.',
                          print_type=self.print_type)
            nodelist = evotree.eval_mutatable_nodes()
            prune_now = 1 + np.random.randint(prune_amount)  # 19 -> prune branch with 1 to max. 19 nodes

            nodelist = [x for x in nodelist if len(x) >= prune_now]  # only (operator-) nodes
            node = np.random.choice(nodelist)
            new_node = Node(label=self.choose_term(node.get_xtype_out()), depth=node.depth)
            node.replace_with_branch(new_node)
            prune_amount = len(evotree) - self.parsimony_max
        return evotree

    def pop_random(self, custom_params, origin: 'OriginTree' = None):
        """
        Creates random trees for the population
        """
        _, size_mode, mean_min_max_var, p_full = helper_evolve_params_branch(custom_params,
                                                                             tree_depth_max=self.tree_depth_max,
                                                                             parsimony_max=self.parsimony_max)

        if origin:  # .exists():
            """
            pareto_insert a (random) number of branches at the first possible "layer"
            (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
            - get these nodes, randomly choose a subset of those
            - get the amount of nodes we are allowed to add. (max nodes without the core-fintree and the nodes we are about to delete)
            - split the amount of nodes up (randomly) and add these new branches to the fintree
            todo ...and if origin is fix?
            sfeh:idea mutate only the childs of a node! The label stays the same
            """
            evotree = origin.origin_tree_copy()

            layer0_nodes = evotree.get_nodes_at_depth(0, allow_fixed=False, expand_depth=True)

            if '_depth' in size_mode:  # "tree_depth"
                build_depth = choose_build_size(size_mode, mean_min_max_var, force='branch')
                for ii, node0 in enumerate(
                        layer0_nodes):  # pareto_insert branches! get layer every time (node ids might have changed)
                    todo = node0.eval_mutatable_nodes()
                    lvl0_node = np.random.choice(todo)  # layer0_branch =
                    # branch_size = layer0_nodes[ii]  # sfeh:idea + len(lvl0_node)
                    new_subbranch = self.invent_core_depth(lvl0_node.get_xtype_out(), build_depth, p_full,
                                                           depth=lvl0_node.depth)
                    lvl0_node.replace_with_branch(new_subbranch)

            elif '_nodes' in size_mode:  # "tree_nodes"
                build_amount = choose_build_size(size_mode, mean_min_max_var, force='branch')
                layer0_splits = randomly_split_range(build_amount, len(layer0_nodes))

                for ii, node0 in enumerate(
                        layer0_nodes):  # pareto_insert branches! get layer every time (node ids might have changed)
                    lvl0_node = np.random.choice(node0.eval_mutatable_nodes())  # layer0_branch =
                    # branch_size = layer0_nodes[ii]  # sfeh:idea + len(lvl0_node)
                    new_subbranch = self.invent_core_nodeops(lvl0_node.get_xtype_out(), layer0_splits[ii], p_full,
                                                             depth=lvl0_node.depth)
                    lvl0_node.replace_with_branch(new_subbranch)
            else:
                raise

        else:
            build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')  # depth, in this case
            if size_mode == 'tree_depth':
                evotree = self.invent_core_depth(self.root_xtype, build_size, p_full, depth=0)
            elif size_mode == 'tree_nodes':
                evotree = self.invent_core_nodeops(self.root_xtype, build_size, p_full, depth=0)  # more debugging?
            else:
                raise

        return evotree


def helper_evolve_params_branch(custom_params, tree_depth_max=10, parsimony_max=50):
    """
    tree_depth_max=10, parsimony_max=30, build_spec has no real function? ...
    sfeh:discussion difference between parsimony and complexity or tree_size/nodecount
    The call parameters in the evolution file need to be adjusted
    sfeh:delete if possible
    """
    build_spec = custom_params.get('build_spec')

    size_mode = build_spec['size_mode']

    mean_min_max_var = list(build_spec.get('mean_min_max_var'))  # (base, min, max, normal_distrib)->list
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

    p_full = build_spec['p_full']

    return build_spec, size_mode, mean_min_max_var, p_full


class TreeMeta:
    # fitness: float = None
    # parsimony: float = None
    # expr_raw: str = None
    # expr_sym: str = None
    # state: int = None
    # last_evolution: list[str] = dataclasses.field(default_factory=list)
    def __init__(self, fitness=None, parsimony=None, expr_raw=None, expr_sym=None):
        # def __init__(self, fitness, parsimony, expr_raw, expr_sym, state):
        self.fitness = fitness
        self.parsimony = parsimony
        self.expr_raw = expr_raw
        self.expr_sym = expr_sym
        self.last_evolution = deque([], maxlen=10)  # sfeh:open

    def append_tag(self, tag):
        self.last_evolution.append(tag)

    def reset(self):
        self.fitness = None
        self.parsimony = None
        self.expr_raw = None
        self.expr_sym = None
        # self.last_evolution = deque([], maxlen=10)


class Tree:
    # sfeh this only holds "fintree" right now.

    def __init__(self, tree, meta):
        self.tree = tree
        self.meta = meta


class FinalizedTree(Tree):
    # class FinalizedTree(Node):
    # ==>fintree status sfeh to evaluated...
    # ->rootnode

    def __init__(self, tree: Node, meta: TreeMeta):
        super().__init__(tree, meta)
        self.tree = tree
        self.meta = meta

    def get_evotree(self):
        return self.tree

    def append_tag(self, tag):
        """

        """
        self.meta.append_tag(tag)

    def get_fitness(self):
        return self.meta.fitness

    def get_parsimony(self):
        return self.meta.parsimony
        # sfeh hmmm
        # if self.meta.state == STATE_EVALUATED:
        #     pass
        # else:
        #     raise Exception('This fintree was not yet evaluated!')

    def set_fitness(self, fitness):
        self.meta.fitness = fitness

    def set_parsimony(self, parsimony):
        self.meta.parsimony = parsimony

    def get_last_evolution(self):
        return self.meta.last_evolution[-1]  # sfeh not even sure


class OriginTree:
    """
    The origin fintree (which was already loaded) gets activated for its use in the GP-process
    """

    def __init__(self, kernel, path_origin=None):
        if path_origin:

            with Path.open(path_origin, newline='') as file:
                nested_expr = file.read()

            tree = tree_from_nested_string(nested_expr)
            expr_raw = tree.eval_expr()
            try:
                expr_sym = expr_sympify(expr_raw)
            except Exception as sympex:
                raise Exception(f'Loaded origin_tree expression could not be mathematically simplified: {sympex}')

            # sfeh, this does not work
            # if not tree_check_is_sympified(fintree):
            #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
            #                          ''.format(expr_raw, expr_sym))

            used_observations = tree.get_observation_list()
            tf_origin_results = kernel.eval_tf(expr_sym, used_observations)
            fitness_train = round(float(tf_origin_results['mean_error']), kernel.precision)
            if kernel.exploration_risk:
                kernel.origin_results = tf_origin_results[
                    'results_kernel']  # after getting the origin-results, these informations can be updated

            meta = TreeMeta(fitness=fitness_train, parsimony=0, expr_raw=expr_raw, expr_sym=expr_sym)
            meta.append_tag('origin')
            self.fintree = FinalizedTree(tree, meta)
            self.origin_is_fix = self.fintree.tree.is_fix
            self.existing = True
            # self.printpl('gg', f'Loading origin fintree, regr. error {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')
        else:
            self.existing = False
            self.fintree = None  # sfeh probably the 'existing' above is deprecated
            self.origin_is_fix = False  # ...if non-existent, it is also not fix

    def origin_tree_copy(self):
        return copy.deepcopy(self.fintree.tree)

    def exists(self):
        return self.existing


def rec_build_tree(lst, depth=0, obs_list=None):
    """
    [rec]ursive [i]dk what the second thing was :D
    recursively loads a nested list into a evotree structure
    nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    nstr = '[+,[-,[Ifte,[True],[sin,[2]],[/,[2.043],[4]]],[cartVel]],[-1.3]]'
    """
    strlabel = str(lst[0])
    if ':fix' in strlabel:
        strlabel = strlabel.replace(':fix', '')
        print(f'Label {strlabel} is fixed!')
        is_fix = True
    else:
        is_fix = False

    if strlabel in ['True', 'False']:
        print(f'Label {strlabel} is boolean.')
        label = BoolConstant(strlabel)
    else:
        try:
            strlabel = float(strlabel)
            label = FloatConstant(strlabel)
            print(f'Label {strlabel} is float.')
        except Exception as ex:

            if strlabel in op_dict:
                print(f'Label {strlabel} is an operator.')
                label = op_dict[strlabel]
            else:
                if obs_list:
                    if strlabel in obs_list:
                        print(f'Label {strlabel} is an observation.')
                        label = Observation(strlabel)
                    else:
                        raise Exception(f'Label "{strlabel}" can not be assigned to a node-label!')
                else:
                    print(f'Label {strlabel} is assumed to be an observation.')
                    label = Observation(strlabel)

    node = Node(label=label, depth=depth, is_fix=is_fix)

    if len(lst[1:]) == node.get_arity():
        childs = [rec_build_tree(x, depth=depth + 1) for x in lst[1:]]
        node.set_childs(childs)

    else:
        raise Exception(f'Tree-building list length {len(lst[1:])} does not match the nodes arity {node.get_arity()}.')

    return node


def tree_from_nested_string(nested_str):
    """
    optional: op_dict + labels not in '' can be used to load the operators directly
    all_input_options = ['1', '0', '-1.132', 'True', 'False', 'vel', 'Ifte', 'max', 'Maxi', '-vel']
    nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    """

    evaled_expr = eval(nested_str)  # , op_dict)

    tree = rec_build_tree(evaled_expr, 0)
    return tree
