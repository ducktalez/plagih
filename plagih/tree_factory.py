"""
The factory to create trees
"""
from plagih.nested_structure import Nested, sympy_to_nsted
from plagih.plagih_tree import *
from plagih.util import *
from plagih.tree_complexity.tree_edit_distance import apted_distance

import random
from collections import deque
import logging
import copy
import numpy as np


def eval_parsimony(tree: Nested, complexity_measure, origin_tree=None):
    """
    complexity_measure: compute the chosen distance by the user.
    #     'tree_node_count': tree_get_size,
    #     'tree_depth': tree_get_depth,
    #     'tree_edit_distance': tree_parsimony_ted,

    sfeh open: weights
    """
    if complexity_measure == 'tree_node_count':  # number of nodes
        return len(tree)  # returns the number of nodes  # sfeh weights
    elif complexity_measure == 'tree_edit_distance':  # tree_edit_distance, fintree-edit-distance
        apted1 = tree.eval_apted_notation()
        apted2 = origin_tree.eval_apted_notation()
        distance, mapping = apted_distance(apted1, apted2)  # sfeh the mapping could be useful somewhere
        return distance
    else:
        raise Exception(f'Complexity measurement not available: {complexity_measure}')


def randomly_split_range(range_max, num_splits):
    """
    split a integer range randomly into parts
    [1..100] -> [33, 15, 52]
    used for building trees
    0 is allowed! (ends a branch with a terminal node)
    """
    # tmp_distributions = random.sample(range(1, range_max), num_splits)
    # d_sum = sum(tmp_distributions)
    # d_list = [int(round(range_max*(x/d_sum), 0)) for x in tmp_distributions]
    # if num_splits == 0:
    # sfeh:discuss create 2 more random split values and remove largest and smallest entry. (better distribution?)
    # No. Also, allow 0 to occur.
    sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
    d_sum = sum(sample_dist)  # 1.5
    sample_dist = [i / d_sum for i in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
    sample_dist = [i * range_max for i in sample_dist]  # [12, 60, 28] -> for 100 nodes
    sample_dist = [int(round(i, 0)) for i in sample_dist]  # convert to usable ints

    # sfeh workaround, this makes exactly the correct range by changing the most extreme entry
    imprecise_diff = range_max - sum(sample_dist)  # sfeh: this can be [0, 0, 0], which assigns to the 0th bin...
    # sfeh:discussion: maybe this difference is 2 or larger more often than 1 (->rounding),
    # so maybe while-loop (just check if it happens?)
    if imprecise_diff != 0:
        if sum(sample_dist) < range_max:
            # sfeh:minor mistake: if relatively empty, this appends to the first bin
            sample_dist[sample_dist.index(min(sample_dist))] += imprecise_diff  # extreme_bin = smallest
        elif sum(sample_dist) > range_max:
            sample_dist[sample_dist.index(max(sample_dist))] += imprecise_diff  # extreme_bin = greatest
        else:
            raise

    return sample_dist


def node_simplification(nsted):
    """
    # sfeh sympy-reconstruct patterns
    #   map symoy-sign to a sum
    #   map piecewise to if-then-else
    #   map power fractal - to sqrt?
    (Tries to) simplify/reduce a tree. It is quite experimental

    SFEH Discussion
        # example: Tree sympification did not work: Reduced core is even more complex than before.
        # expr_raw: sign(Min(((Velocity_2 * -0.790706) - sqrt(Gain_0)), (-0.569271 - Velocity_9)))
        # old_core:[sign, [Min, [-, [*, Velocity_2, -0.790706], [sqrt, Gain_0]], [-, -0.569271, Velocity_9]]]
        # new_node: [sign, [Min, [-, [Usub, [sqrt, Gain_0]], [*, 0.790706, Velocity_2]], [-, -Velocity_9, 0.56921]]]
    """
    expr_sym = nsted.get_sympy_expr()
    nsted_rebuilt = sympy_to_nsted(expr_sym)
    # node_rebuilt = node_rebuilt.update_fixed_nodes(node)  # this is not our problem

    return nsted_rebuilt


def evolve_reduce_simplify(nstruc: Nested, completely=True, force=False):
    """
    # sfeh:open this function does currently not work
    Reducing a fintree to its most basic form with sympify.
    (completely = False: reduce just one branch. if you wanted to have more complexity)

    """
    tree_copy = copy.deepcopy(nstruc)
    if completely:  # reduce the complete fintree
        nodes_lv0 = nstruc.get_nodes_at_depth(0, allow_fixed=False)  # only required for fixed-core trees
        for cc in nodes_lv0:
            cc.set_new_nested(node_simplification(cc))
    else:
        # # this was implemented for runtime, to prevent simplifing leaf nodes
        # functions = [x for x in nodes if x.get_arity() > 0]
        # if functions:
        # nd = np.random.choice(functions)
        #   ...
        nd_list = nstruc.eval_mutable_nodes()
        nd_list = [x for x in nd_list if x.get_arity() > 0]  # ignoring leaf nodes
        nd = np.random.choice(nd_list)
        nd.set_new_nested(node_simplification(nd))  # sfeh chosen must be set again? or not? test it at least.
    if force:
        return nstruc
    else:
        if len(tree_copy) < len(nstruc):
            print_warning('w', f'sfeh Trees get larger during simplification? {nstruc.__class__.__name__}')
            return tree_copy
        else:
            return nstruc


def nsted_deepcopy(nsted: Nested):
    _cpy = copy.deepcopy(nsted)
    return _cpy


class TreeBuilder:
    """
    Just a class to prevent referencing all the separate shizzle everytime

    func_list, probability_list = self. Operators[xtype_out]
    return np.random.choice(func_list, p_full=probability_list)
    """

    distributions = {float: [lambda: random.normalvariate(0, 1),
                             lambda: random.normalvariate(1, 1),
                             lambda: random.normalvariate(10, 5),
                             lambda: random.randint(1, 20)],  # 0 has actually no purpose (except as being an action)
                     bool: [lambda: random.choice([True, False])]}  # sfeh:discussion

    def __init__(self, obs_names, depth_max, nodes_max, root_xtype, operator_pool=None, origin=None):
        self.observations_add(obs_names)
        self.root_xtype = root_xtype
        self.depth_max = depth_max
        self.nodeamount_max = nodes_max
        self.origin = origin

        def operator_pool_check(operatorPool):
            """
            Check if the user-specified loaded operators allow closure
            (either float-only/bool only or all 4 types of operators)
            @:param operator_pool: list with operators and their weight of being selected
            """
            # sfeh dunno if that works... 2f not in x
            opxtypes = [oper.xtype for oper in operatorPool.keys()]
            has_2f = any([float == x[1] for x in opxtypes])
            has_2b = any([bool == x[1] for x in opxtypes])
            has_f2b = any([float in x[0] and bool == x[1] for x in opxtypes])
            has_b2f = any([bool in x[0] and float == x[1] for x in opxtypes])
            if not all([has_2f, has_2b, has_f2b, has_b2f]):
                logging.warning(f'Loaded operators do not feature both numeric (float) and bool type.')
            if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
                raise Exception(f'Loaded operators do not allow closure!')

        if operator_pool is None:  # quick developer adjustments

            # sfeh same prob for all for testing
            operator_pool = {Add: 2, Sub: 1,
                             Mul: 2, Div: 1,
                             # Usub: 1,  # sfeh
                             Square: 0.75,
                             Powrounded: 0.1,
                             Abs: 0.5, sign: 0.5,  # sfeh stop chain of arity-1 op_dict in buid method?
                             Sqrt: 0.1,  # 0.25,  # sfeh debug this
                             log: 0.1,  # Log1p: 0.1,
                             sin: 0.5, tan: 0.1, cos: 0.33,
                             # Acos: 0.33, Asin: 0.33, Atan: 0.33, Tanh: 0.5,
                             Xor: 1,  # sfeh
                             # sympy extra classes (Capitalized)
                             Round: 0.5,
                             And: 1, Or: 1, Not: 0.5,
                             # Eq: 1,  # Ne: 0.5,
                             Lt: 0.5, Le: 0.5, Gt: 0.1, Ge: 0.1,
                             Ifte: 2,
                             Min: 1, Max: 1}

        operator_pool_check(operator_pool)

        choose_oparray = {
            # all operator_pool with a certain xtype_out-result
            # None: [[], []],
            float: [[], []],  # to float
            bool: [[], []],  # to bool
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
            if p[0]:  # if operators for this xtype/arity combination
                choose_oparray[o][1] = [x / sum(p[1]) for x in p[1]]

        self.operators = {}
        for xtype, _opx in choose_oparray.items():
            # self.operators[xtype_out] = lambda: np.random.choice(x[0], p=x[1])  # "seloplam" faster, but less readable
            self.operators[xtype] = (_opx[0], _opx[1])

        obs_list = [x for x in obs_names]

        self.symbols_xtdc = {float: lambda: np.random.choice(obs_list),  # , p=obs_prop
                             bool: None}  # sfeh:discussion None? no  lambda? yeah, not important but still...

    def choose_op(self, any_xtype):
        """
        any_xtype can be a tuple, single type or even None
        """
        # return self.operators[xtype_out]()  # "seloplam"
        return np.random.choice(self.operators[any_xtype][0], p=self.operators[any_xtype][1])

    def constants_add_data_samples(self, obs_infos, data_train, n_samples=100):
        """
        ONLY floats, because ...why would you want to load True/False samples.
        (okay, it might make sense as it better represents the actual distribution- NO FUCK IT.)
        """
        if obs_infos is not None:
            obsnames = obs_infos.symbols_xtdc[float].keys()
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
        #     fam_members = sorted([x for x in obs_names if x.fam == fam], key=lambda o: o.time_index)
        #     if len(fam_members) > 1:
        #         obs_names.extend([x for x in fam_members])
        #         obs_prop.extend(list(observation_select_index(fam_members)))
        #         index_minmax = (fam_members[0].time_index, fam_members[-1].time_index)
        #         for obs in fam_members:
        #             obs.index_minmax = index_minmax
        #             # environment.obs_infos[obs.name] = obs  # sfeh:open okay do we need this? :s guess we will find out x.D haha
        #             obs_info[obs.name] = obs
        #     else:
        #         obs = fam_members[0]
        #         obs_info[obs.name] = obs
        #         obs_names.pop_append_evotree(obs)
        #         obs_prop.pop_append_evotree(1)  # just one value
        pass

    def choose_symbol(self, xtype):
        """
        Randomly choosing an operator-label for a given xtype_out.
        choose_oparray3 must be given, as they are different between runs.
        arity can also be set optionally, e.g. for point mutation
        sfeh:open DOUBLE-check if this xtype_out is chosen correctly... better: replace it
        """
        _sym = self.symbols_xtdc[xtype]()
        return Symbol(_sym)

    def choose_value(self, xt_out):
        value = np.random.choice(self.distributions[xt_out])()
        if xt_out == float:
            return Float(round(value, PRECISION))
        elif xt_out == bool:
            return Boolean(value)

    def choose_term(self, xt_out, p_observation=0.5):
        if random.random() < p_observation:
            try:
                return self.choose_symbol(xt_out)
            except TypeError:
                pass  # return a constant (E.g. because there are no boolean observations)

        return self.choose_value(xt_out)

    def invent_core_depth(self, xt_out, depth_goal, depth=0, p_full=1.0):
        if depth == self.depth_max or depth == depth_goal or random.random() > p_full:
            label = self.choose_term(xt_out)
            nsted = Nested(label, [], depth=depth)
        else:
            label = self.choose_op(xt_out)  # self.choose_any(xtype, p_full)
            childs = [self.invent_core_depth(xt, depth_goal, depth=depth + 1, p_full=p_full) for xt in label.xtype[0]]
            nsted = Nested(label, childs, depth=depth)  # , depth=depth sfeh no depth?

        return nsted

    def invent_core_operatoramount(self, xt_out, ops_left, depth=0, p_full=1.0):
        """
        This version counts the amount of operators as construction limit!
        sfeh:idea nodes are now about being operators...
        '+': xtype = (tuple([float, float]), float)
        sfeh:pfull?
        """
        childs = []

        if depth == self.depth_max or ops_left == 0:
            label = self.choose_term(xt_out)

        else:  # nodeops_max > 0:
            label = self.choose_op(xt_out)  # sfeh:xxx

            ops_left -= 1
            nodeops_split = randomly_split_range(ops_left, len(label.xtype[0]))

            for ii, xt_child in enumerate(label.xtype[0]):
                childs.append(self.invent_core_operatoramount(xt_child, nodeops_split[ii], depth=depth+1))

        nsted = Nested(label, childs, depth=depth)  # , depth=depth sfeh no depth?
        return nsted

    def evolve_mutate_filter_random(self, nsted):
        """
        Mutates a number of float terminal of a fintree
        - filter point/branch/all, branch can also affect a point only aswell as all nodes
        - filter observations?
        - filter terminals
        - filter with which filter?
        """

        _nd = np.random.choice(nsted.eval_mutable_nodes())
        _nd.evolve_mutate_filter_branch()

        return nsted

    def evolve_mutate_point(self, nsted):
        """
        Mutate a single mutable point in any Tree.
        sfeh is the fintree a fintree copy or the same fintree?
        """
        evostruc = nsted_deepcopy(nsted)

        # todo:chain only mutable nodes that are not operators in chained-mode
        _nd = np.random.choice(evostruc.eval_mutable_nodes())
        xtype = _nd.get_xtype()

        if _nd.get_arity() > 0:
            new_label = self.choose_op(xtype)  # Function is same type, same arity
        else:
            new_label = self.choose_term(xtype[1])  # 3 -> '2f' -> 5
        _nd.set_label(new_label)
        return evostruc

    def evolve_prune(self, nsted: Nested):
        """
        prune depth
        -> prune everything below a certain level... (should not happen in the first place)
        prune nsteds
        -> get nsted difference, get nstedlist, untill small enough: split the difference, prune nsteds until

        sfeh:discussion there is a difference between parsimony and complexity...
        sfeh:discuss analyze the amount of trees that have to be pruned?
        sfeh:open add labelweight_max to
        """
        nodelist = nsted.eval_mutable_nodes()
        for dnode in nodelist:
            if dnode.depth == self.nodeamount_max and dnode.get_arity() > 0:
                print_warning('wwww', f'Node in fintree is too deep: {dnode.depth}')
                new_node = Nested(self.choose_term(dnode.get_xtype_out()), childs=[], depth=dnode.depth)
                dnode.set_new_nested(new_node)
                # sfeh:debug did this work?

        prune_amount = len(nsted) - self.nodeamount_max
        while prune_amount > 0:
            print_warning('wwww', f'Tree too complex: {len(nsted)} > {self.nodeamount_max}, pruning {prune_amount}.')
            nodelist = nsted.eval_mutable_nodes()
            prune_now = 1 + np.random.randint(prune_amount)  # 19 -> prune branch with 1 to max. 19 nodes

            nodelist = [x for x in nodelist if len(x) >= prune_now]  # only (operator-) nodes
            node = np.random.choice(nodelist)
            new_node = Nested(self.choose_term(node.get_xtype_out()), childs=[], depth=node.depth)
            node.set_new_nested(new_node)
            prune_amount = len(nsted) - self.nodeamount_max
        return nsted

    def evolve_mutate_branch_depth(self, nsted, depth_goal, p_full=1.0):
        """
        nsted, cool_build_size, p_full=p_full

        sfeh ==>depth only
        currently only one branch
        """
        _nd = np.random.choice(nsted.eval_mutable_nodes())
        xtype_out = _nd.get_xtype_out()
        branch = self.invent_core_depth(xtype_out, depth_goal, depth=0, p_full=p_full)  # sfeh ==>dummies
        _nd.set_new_nested(branch)
        # if _nd.depth == depth_goal:
        #     _nd.set_label(tb.choose_term(xtype_out))  # sfeh update _nd nlabel
        # else:
        #     _nd.childs = [Node(tb.choose_any(xt, p=1)) for xt in _nd.get_xtype()[0]

        # etree.finalize()  # sfeh ==>state
        return nsted

    def evolve_mutate_branch_nodes(self, nsted, nodes_goal, p_full=1.0):
        """
        nsted, cool_build_size, p_full=p_full

        sfeh ==>depth only
        currently only one branch
        """
        if nsted is None:
            raise NotImplementedError('SFEH:open Implement standard selection mechanism')
        _nd = np.random.choice(nsted.eval_mutable_nodes())
        xtype_out = _nd.get_xtype_out()
        branch = self.invent_core_operatoramount(xtype_out, nodes_goal, depth=_nd.depth, p_full=p_full)
        _nd.set_new_nested(branch)
        return nsted

    def evolve_crossover(self, tree1: Nested, tree2: Nested):
        """
        Evolution with crossover of branches with two trees
        currently only one branch

        swap branches of two trees
        - select parent a and b
        - select swappable branche for a_parent from b_parent
            - select a node in a (and crossover here, no matter what)
        - delete a_parent branch and pareto_insert b_parent branch (which tactic?)
        sfeh:idea into main fintree?
        """
        _a = nsted_deepcopy(tree1)
        _b = nsted_deepcopy(tree2)

        _a_nds = _a.eval_mutable_nodes(allow_root=False)
        _a_nd = np.random.choice(_a_nds)

        xtype_out = _a_nd.get_xtype_out()
        _b_nds = _b.eval_mutable_nodes(xtype_out=xtype_out)
        if len(_b_nds) > 0:
            _b_nd = np.random.choice(_b_nds)
        else:
            xtype_out = float if xtype_out == bool else bool  # the other swap type now
            _b_nds = _b.eval_mutable_nodes(xtype_out=xtype_out)
            _b_nd = np.random.choice(_b_nds)
            _a_nds = _a.eval_mutable_nodes(allow_root=False, xtype_out=xtype_out)
            _a_nd = np.random.choice(_a_nds)

        ansted_copy = copy.deepcopy(_a_nd)  # sfeh deepcopy required??

        _a_nd.set_new_nested(_b_nd)
        _b_nd.set_new_nested(ansted_copy)

        _a = self.evolve_prune(nsted=_a)
        _b = self.evolve_prune(nsted=_b)

        return _a, _b

    def pop_random_depth(self, depth_goal, xt_out=None, p_full=1.0):

        xt_out = xt_out or self.root_xtype

        if self.origin is not None:

            evonsted = self.origin.origin_tree_copy()
            layer0_nsteds = evonsted.get_nsteds_at_depth(0, allow_fixed=False, expand_depth=True)

            for ii, nsted0 in enumerate(layer0_nsteds):  # -> get layer every time (nsted ids might have changed)
                nd_list = nsted0.eval_mutable_nsteds()
                lvl0_nsted = np.random.choice(nd_list)
                new_subbranch = self.invent_core_depth(lvl0_nsted.get_xtype_out(), depth_goal, depth=lvl0_nsted.depth, p_full=p_full)
                lvl0_nsted.set_new_nested(new_subbranch)

        else:
            evonsted = self.invent_core_depth(xt_out, depth_goal)

        return evonsted

    def pop_random_nodes(self, nodeamount, p_full, xtype=None):

        xtype = xtype or self.root_xtype

        if self.origin is not None:
            """
            pareto_insert a (random) number of branches at the first possible "layer"
            (If all nodes are modifiable, it is the root node. Otherwise, it is the first modifiable nodes
            - get these nodes, randomly choose a subset of those
            - get the amount of nodes allowed to add. (max nodes without the core-fintree + the nodes about to delete)
            - split the amount of nodes up (randomly) and add these new branches to the fintree
            sfeh:idea mtate only the childs of a node! The label stays the same
            """
            evonsted = nsted_deepcopy(self.origin)
            layer0_nodes = evonsted.get_nodes_at_depth(0, allow_fixed=False, expand_depth=True)

            layer0_splits = randomly_split_range(nodeamount, len(layer0_nodes))

            for ii, node0 in enumerate(
                    layer0_nodes):  # pareto_insert branches! get layer every time (node ids might have changed)
                lvl0_node = np.random.choice(node0.eval_mutable_nodes())  # layer0_branch =
                # branch_size = layer0_nodes[ii]  # sfeh:idea + len(lvl0_node)
                new_subbranch = self.invent_core_operatoramount(lvl0_node.get_xtype_out(), layer0_splits[ii],
                                                                depth=lvl0_node.depth)
                lvl0_node.set_new_nested(new_subbranch)

        else:

            evonsted = self.invent_core_operatoramount(xtype, nodeamount, depth=0)  # more debugging?

        return evonsted


class TreeMeta:

    def __init__(self, fitness, parsimony, expr_sym):
        self.fitness = fitness
        self.parsimony = parsimony
        self.expr_sym = expr_sym
        self.last_evolution = deque([], maxlen=10)  # sfeh:open

    def append_tag(self, tag):
        self.last_evolution.append(tag)

    def get_last_tag(self):
        return self.last_evolution[-1]

    def reset(self):
        self.fitness = None
        self.parsimony = None
        self.expr_sym = None
        # self.last_evolution = deque([], maxlen=10)

    def get_fitness(self):
        return self.fitness

    def get_parsimony(self):
        return self.parsimony

    # ...should this mean the size or fitness? not clear at all
    # def __lt__(self, other):
    #     return self.get_fitness() < other.get_fitness()
    #
    # def __eq__(self, other):
    #     return self.get_fitness() <= other.get_fitness()


class FinalizedTree:
    """An actual individual (Tree + meta-infos/phenotypes)"""

    def __init__(self, tree: Nested, meta: TreeMeta):
        self.tree = tree
        self.meta = meta

    def __str__(self):
        """Show the Fitness and Parsimony of a tree"""
        return f'[{self.get_parsimony():2.1f}: fit {self.get_fitness():4.2f}]'

    def get_evotree(self):
        return self.tree

    def append_tag(self, tag):
        self.meta.append_tag(tag)

    def get_fitness(self):
        return self.meta.fitness

    def get_parsimony(self):
        return self.meta.parsimony

    def set_fitness(self, fitness):
        self.meta.fitness = fitness

    def set_parsimony(self, parsimony):
        self.meta.parsimony = parsimony

    def get_last_evolution(self):
        return self.meta.get_last_tag()  # sfeh same name?


# class OriginTree(FinalizedTree):
#     """
#     The origin fintree (which was already loaded) gets activated for its use in the GP-process
#     sfeh: This class could be a subclass of FinalizedTree, but only if it is used only when an origin exists
#     """
#
#     def __init__(self, tree, meta):
#         super().__init__(tree, meta)
#         if tree:
#             meta.append_tag('origin')  # sfeh:discuss
#             self.existing = True
#             # self.printpl('gg', f'Loading origin fintree, regr. error {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')
#         else:
#             self.existing = False
#             self.fintree = None  # sfeh probably the 'existing' above is deprecated
#
#     def origin_is_fix(self):
#         return self.tree.is_fix
#
#     def origin_tree_copy(self):
#         return copy.deepcopy(self.fintree.tree)


# def rec_build_tree(lst, obs_list=None, depth=0):
#     """
#     [rec]ursive building of a tree
#     recursively loads a nested list into a evonsted structure
#     nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
#     nstr = '[+,[-,[Ifte,[True],[sin,[2]],[/,[2.043],[4]]],[cartVel]],[-1.3]]'
#     """
#
#     strlabel = str(lst[0])
#     if ':fix' in strlabel:
#         strlabel = strlabel.replace(':fix', '')
#         is_fix = True
#     else:
#         is_fix = False
#
#     if strlabel in ['True', 'False']:
#         node = Bool(strlabel)
#     else:
#         try:
#             strlabel = float(strlabel)
#             node = Float(strlabel)
#         except ValueError:
#             if strlabel in loadable_ops_dict:
#                 node = loadable_ops_dict[strlabel]
#             else:
#                 if obs_list:
#                     if strlabel in obs_list:
#                         node = Symbol(strlabel)
#                     else:
#                         raise Exception(f'Label "{strlabel}" can not be assigned to a node-label!')
#                 else:
#                     node = Symbol(strlabel)
#
#     # node = Nested(label, depth=depth, is_fix=is_fix)
#
#     if len(lst[1:]) == node.get_arity():
#         childs = [rec_build_tree(x, depth=depth + 1, obs_list=obs_list) for x in lst[1:]]
#         node.set_childs(childs)
#
#     else:
#         # childs = [rec_build_tree(x, depth=depth + 1, obs_list=obs_list) for x in lst[1:]]
#         # node.set_childs(childs)  # sfeh delete
#         raise Exception(f'Tree-building list length {len(lst[1:])} does not match the nodes arity {node.get_arity()}.')
#
#     return node


# def check_tree_loadable_reconstruction(tree: Nested):
#     """
#     Extracts a tree expression and rebuilds the tree
#     The trees must be identical, as it only rebuilt itself
#     :return:
#     """
#     tree_0 = copy.deepcopy(tree)
#     _nested = tree.eval_expr_str()
#     tree_1 = evonsted_from_nested_labels(_nested)
#     tree_1.update_fixed_nsteds(tree_0)
#
#     a = repr(tree_0)
#     b = repr(tree_1)
#
#     return a == b


# def evonsted_from_nested_labels(nested_str, obs_list=None):
#     """
#     optional: op_dict + labels not in '' can be used to load the operators directly
#     all_input_options = ['1', '0', '-1.132', 'True', 'False', 'vel', 'Ifte', 'max', 'Max', '-vel']
#     nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
#     """
#     evaled_expr = eval(nested_str)  # sfeh:discuss -> sympify? <- no
#     tree = rec_build_tree(evaled_expr, depth=0, obs_list=obs_list)
#     tree.finalize_set_depth()
#
#     return tree


def selection_tournament(individuals, tournsize=3):
    """
    SFEH's tournament selection
    sfeh: discuss extracting & deepcopying the inner tree
    """
    tree_list = [np.random.choice(individuals) for _ in range(tournsize)]
    fintree: 'FinalizedTree' = min(tree_list, key=lambda tree: tree.get_fitness())
    evonsted = fintree.get_evotree()
    evonsted = copy.deepcopy(evonsted)
    return evonsted


if __name__ == '__main__':
    _test_open = '[Ifte, [Or, [b < -1], [And, [b < 0.1], [a < -0.05]]], 2, [Ifte, [And, [And, ' \
                 '[b > -0.45], [b < -0.05]], [a < -0.5]], 0, [Ifte, [a < 0], 0, 2]]]',

    _test_loadabls = ["['+',['-',['Ifte',['True'],['sign',['cartVel']],['/',[2.3],[4]]],['cartVel']],[-1.3]]",
                      '["Ifte:fix",["<",["cartVel"],[0]],["0:fix"],["2:fix"]]',
                      '["Ifte", ["Not", [False]], [0.0], [2.0]]']

    tb = TreeBuilder(['a', 'b'], 10, 30, float)
    tr = Nested(Add, [Nested(Symbol('a'), []), Nested(Float(1.23), [])])
    tr = Nested(Ifte, [Nested(Gt, [Nested(Symbol('a'), []), Nested(Float(1.2), [])]), Nested(Float(1.23), []), Nested(Float(2.3), [])])
    tr = Nested(Max, [Nested(Symbol('a'), []), Nested(Float(1.2), [])])
    x = tr.get_sympy_expr()
    tr2 = sympy_to_nsted(x)
    print(tr, tr2)
    for _ in range(10):
        tr = tb.pop_random_depth(3, float, p_full=0.7)
        x = tr.get_sympy_expr()
        print('First sym success')
        tr_new = sympy_to_nsted(x)
        x2 = tr_new.get_sympy_expr()
        print(tr)
        print(tr_new)
        if str(x) != str(x2):
            print()
            raise Exception(f'sympy process failed {x}, <-->, {x2}')

