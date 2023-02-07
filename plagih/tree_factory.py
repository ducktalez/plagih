"""
The factory to create trees
"""
from plagih.tree import *
from plagih.plagih_tree import *
from plagih.util import *
from plagih.tree_complexity.tree_edit_distance import apted_distance

import random
from collections import deque
import copy
import numpy as np


def eval_parsimony(tree: Node, complexity_measure, origin_tree=None):
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
    """split integer range randomly into num_splits parts
    [1..100] -> [33, 15, 52]
    used for building trees
    0 is allowed! (ends a branch with a terminal node)
    sfeh:discuss create 2 more random split values and remove largest and smallest entry. (better distribution?)
      -> No. Also, allow 0 nodes."""
    sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
    d_sum = sum(sample_dist)  # 1.5
    sample_dist = [i / d_sum for i in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
    sample_dist = [i * range_max for i in sample_dist]  # [12, 60, 28] -> for 100 nodes
    sample_dist = [int(round(i, 0)) for i in sample_dist]  # int required

    # sfeh workaround, this makes exactly the correct range by changing the most "extreme" entry
    imprecise_diff = range_max - sum(sample_dist)  # sfeh: this can be [0, 0, 0], which assigns to the 0th bin...
    # sfeh:discuss: maybe this difference is 2 or larger more often than 1 (->rounding),
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


def tree_simplification(tree):
    """
    (Tries to) simplify/mathematically-reduce a tree. It is quite experimental
    # sfeh sympy-reconstruct patterns
    #   map symoy-sign to a sum
    #   map piecewise to if-then-else
    #   map power fractal - to sqrt?
    """
    expr_sym = tree.get_sympy_expr()
    nsted_rebuilt = sympy_to_tree(expr_sym)

    return nsted_rebuilt


def evolve_reduce_simplify(tree: Node, completely=True, force=False):
    """Reducing a fintree to its most basic form with sympify.
    (completely = False: reduce just one branch. if you wanted to have more complexity)"""
    tree_copy = copy.deepcopy(tree)
    if completely:  # reduce the complete tree
        nodes_lv0 = tree.get_nodes_at_depth(0, allow_fixed=False)  # only required for fixed-core trees
        for cc in nodes_lv0:
            cc.set_new_node(tree_simplification(cc))
    else:
        node_list = [n for n in tree.eval_mutable_nodes() if issubclass(n.label, Operator)]  # ignoring leaf nodes...
        if len(node_list) == 0:
            print_warning('wwww', f'Tree for simplification does not provide operators: {tree}')
            return tree
        node = np.random.choice(node_list)
        node.set_new_node(tree_simplification(node))  # sfeh chosen must be set again? or not? test it at least.
    if force:
        return tree
    else:
        if len(tree_copy) < len(tree):
            print_warning('w', f'Tree grew larger during simplification: {tree_copy} < {tree}')
            # [Square, [Powrounded, [-0.91], [cartPos]]] < [Pow, [-0.91], [Mul, [2], [Round, [cartPos]]]]
            return tree_copy
        else:
            return tree


def node_deepcopy(tree: Node):
    _cpy = copy.deepcopy(tree)
    return _cpy


class TreeBuildRestrictions:
    """functions to build trees, with the advantage of being able to use general build restrictions."""

    def __init__(self, root_xt_out, nc, build_restrictions, complexity_metric, origin_tree=None):
        self.root_xt_out = root_xt_out
        self.nc = nc

        self.complexity_metric = complexity_metric
        self.origin_tree = origin_tree

        self.depth_max = build_restrictions.get('depth_max', 10)
        self.nodes_max = build_restrictions.get('nodes_max', 100)

    # def observations_add(self, obs_names):
    #     """
    #     :param obs_names: list of all observation names (e.g. ['cartVel', 'cartPos'])
    #     """
    #     # def observation_select_index(observations, max_hist=10):
    #     #     """
    #     #     chooses variables but weighting how old they are.
    #     #     observations = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4'] -> [0.28, 0.23, 0.19, 0.16, 0.13]
    #     #     sfeh: what about larger steps?
    #     #     e.g. [0, 1, 2, 3] is good, but [0, 5, 10, 15] is baad
    #     #     what if variables are not all of same diff?
    #     #     """
    #     #     observations = np.delete(observations, np.s_[max_hist:])
    #     #     x = len(observations)
    #     #     fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
    #     #     p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
    #     #     p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
    #     #     return np.random.choice(observations, p=p)  # returning a function this time
    #     #
    #     # obs_prop = []
    #     # obs_info = {}
    #     #
    #     # for fam in list(set(observation_get_family_and_time(x)[0] for x in obs_names)):
    #     #     fam_members = sorted([x for x in obs_names if x.fam == fam], key=lambda o: o.time_index)
    #     #     if len(fam_members) > 1:
    #     #         obs_names.extend([x for x in fam_members])
    #     #         obs_prop.extend(list(observation_select_index(fam_members)))
    #     #         index_minmax = (fam_members[0].time_index, fam_members[-1].time_index)
    #     #         for obs in fam_members:
    #     #             obs.index_minmax = index_minmax
    #     #             obs_info[obs.name] = obs
    #     #     else:
    #     #         obs = fam_members[0]
    #     #         obs_info[obs.name] = obs
    #     #         obs_names.pop_append_evotree(obs)
    #     #         obs_prop.pop_append_evotree(1)  # just one value
    #     pass

    def evolve_new_tree_depth(self, depth_goal, xt_out=None, p_term=0.0):
        # todo non-recursive version
        xt_out = xt_out or self.root_xt_out

        if self.origin_tree is not None:

            evotree = self.origin_tree.origin_tree_copy()
            layer0 = evotree.get_nodes_at_depth(0, allow_fixed=False, earliest_nonfix=True)

            for ii, nodes0 in enumerate(layer0):  # -> get layer every time (nsted ids might have changed)
                nd_list = nodes0.eval_mutable_nsteds()
                lvl0_nodes = np.random.choice(nd_list)
                new_subbranch = self.evolve_create_random(lvl0_nodes.get_xtype_self(), depth_goal, num_rest=-1,
                                                          depth=lvl0_nodes.depth, p_term=p_term)
                lvl0_nodes.set_new_node(new_subbranch)

        else:
            evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    def new_tree_nodes(self, nodeamount, p_term, xtype=None):

        xtype = xtype or self.root_xt_out

        if self.origin_tree is not None:
            """pareto_insert a (random) number of branches at the first possible "layer"
            (If all nodes are modifiable, it is the root node. Otherwise, it is the first modifiable nodes
            - get these nodes, randomly choose a subset of those
            - get the amount of nodes allowed to add. (max nodes without the core-fintree + the nodes about to delete)
            - split the amount of nodes up (randomly) and add these new branches to the fintree
            sfeh:idea mutate only the childs of a node! The label stays the same"""
            evotree = node_deepcopy(self.origin_tree)
            layer0_nodes = evotree.get_nodes_at_depth(0, allow_fixed=False, earliest_nonfix=True)

            layer0_splits = randomly_split_range(nodeamount, len(layer0_nodes))

            for ii, node0 in enumerate(
                    layer0_nodes):  # pareto_insert branches! get layer every time (node ids might have changed)
                lvl0_node = np.random.choice(node0.eval_mutable_nodes())  # layer0_branch =
                # branch_size = layer0_nodes[ii]  # sfeh:idea + len(lvl0_node)
                new_subbranch = self.evolve_create_random(lvl0_node.get_xtype_self(), -1, depth=lvl0_node.depth,
                                                          num_rest=layer0_splits[ii], p_term=p_term)
                lvl0_node.set_new_node(new_subbranch)

        else:
            evotree = self.evolve_create_random(xtype, -1, num_rest=nodeamount, depth=0, p_term=p_term)

        return evotree

    def evolve_create_random(self, xt_out, depth_goal, num_rest=-1, depth=0, p_term=0.0):
        """num_rest: -1 ignores the node number restriction
        sfeh:open make depth_goal -> depth_rest"""

        if depth == self.depth_max or depth == depth_goal or num_rest == 0 or random.random() < p_term:
            node = self.nc.choose_terminal(xt_out, as_node=True)
            node.depth = depth

        else:
            # todo allow_chain
            label = self.nc.choose_operator(xt_out)
            child_xts = label.get_child_xts()
            num = len(child_xts)
            childs = []

            if num_rest > 0:
                nums = randomly_split_range(num_rest-1, num)
                for ii, xt in enumerate(child_xts):
                    cc = self.evolve_create_random(xt, depth_goal, num_rest=nums[ii], depth=depth + 1, p_term=p_term)
                    childs.append(cc)
            else:
                for xt in child_xts:
                    cc = self.evolve_create_random(xt, depth_goal, num_rest=-1, depth=depth + 1, p_term=p_term)
                    childs.append(cc)

            node = Node(label, childs, depth=depth)
        return node

    def evolve_new(self, xt_out, depth_goal, num_rest=-1, depth=0, p_term=0.0):
        # todo todotodo

        if self.origin_tree is not None:

            evotree = self.origin_tree.origin_tree_copy()
            layer0 = evotree.get_nodes_at_depth(0, allow_fixed=False, earliest_nonfix=True)

            for ii, nodes0 in enumerate(layer0):  # -> get layer every time (nsted ids might have changed)
                nd_list = nodes0.eval_mutable_nsteds()
                lvl0_nodes = np.random.choice(nd_list)
                new_subbranch = self.evolve_new(lvl0_nodes.get_xtype_self(), depth_goal, num_rest=-1,
                                                depth=lvl0_nodes.depth, p_term=p_term)
                lvl0_nodes.set_new_node(new_subbranch)

        else:
            xt_out = xt_out or self.root_xt_out
            evotree = self.evolve_new(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        if depth == self.depth_max or depth == depth_goal or num_rest == 0 or random.random() < p_term:
            node = self.nc.choose_terminal(xt_out, as_node=True)
            node.depth = depth

        else:
            label = self.nc.choose_operator(xt_out)
            child_xts = label.get_child_xts()
            n_ch = len(child_xts)
            childs = []

            if num_rest > 0:
                nums = randomly_split_range(num_rest-1, n_ch)
                for ii, xt in enumerate(child_xts):
                    cc = self.evolve_new(xt, depth_goal, num_rest=nums[ii], depth=depth + 1, p_term=p_term)
                    childs.append(cc)
            else:
                for xt in child_xts:
                    cc = self.evolve_new(xt, depth_goal, num_rest=-1, depth=depth + 1, p_term=p_term)
                    childs.append(cc)

            node = Node(label, childs, depth=depth)
        return node

    def evolve_mutate_filter(self, tree):
        """Mutates a number of float terminal of a fintree
        - filter point/branch/all, branch can also affect a point only aswell as all nodes
        - filter observations?
        - filter terminals
        - filter with which filter?"""

        _nd = np.random.choice(tree.eval_mutable_nodes())
        # sfeh: does nothing if no float values are in this tree
        _nd.evolve_mutate_filter_gauss()

        return tree

    def evolve_mutate_point(self, tree):
        # todo mutate a child?
        """Mutate a single mutable point in any Tree.
        sfeh:debug is the fintree a fintree copy or the same fintree?"""
        evotree = copy.deepcopy(tree)

        _nd = rnd_choice(evotree.eval_mutable_nodes(ignore_chain=True))  # debug if ignores chains
        xtype = _nd.get_xtype()

        if _nd.is_operator():
            # todo allow_chain
            # sfeh:what if its the same function?
            new_label = self.nc.choose_operator_match(xtype)  # Function is same type, same arity
            _nd.set_label(new_label)
        elif _nd.is_term:
            new_node = self.nc.choose_terminal(xt_self(xtype), as_node=True)
            _nd.set_new_node(new_node)
        else:
            raise NotImplementedError

        return evotree

    def evolve_mutate_branch_depth(self, tree, depth_goal, p_term=0.0):
        _nodes_init = len(tree)
        _node = np.random.choice(tree.eval_mutable_nodes())
        xtype_out = _node.get_xtype_self()
        branch = self.evolve_create_random(xtype_out, depth_goal, num_rest=self.nodes_max - _nodes_init, depth=0,
                                           p_term=p_term)
        _node.set_new_node(branch)

        # etree.finalize()
        return tree

    def evolve_mutate_branch_nodes(self, tree, nodes_goal, p_term=0.0):
        """currently only one branch"""
        _nodes_init = len(tree)
        if tree is None:
            raise NotImplementedError('SFEH:open Implement standard selection mechanism')
        nd = np.random.choice(tree.eval_mutable_nodes())
        xt_out = nd.get_xtype_self()
        nodes_goal = min(self.nodes_max - (_nodes_init - len(nd)), nodes_goal)
        branch = self.evolve_create_random(xt_out, -1, num_rest=nodes_goal, depth=nd.depth, p_term=p_term)
        nd.set_new_node(branch)
        return tree

    def evolve_crossover(self, tree1: Node, tree2: Node):
        """Evolution with crossover of branches between two trees
        currently only one branch

        swap branches of two trees
        - select parent aa and bb
        - select swappable branche for a_parent from b_parent
            - select aa node in aa (and crossover here, no matter what)
        - delete a_parent branch and pareto_insert b_parent branch (which tactic?)
        sfeh:idea into main fintree?"""

        aa = node_deepcopy(tree1)
        bb = node_deepcopy(tree2)

        a_nds = aa.eval_mutable_nodes(ignore_first=True)  # why actually ignore root node
        if len(a_nds) == 0:
            raise ValueError(f'Crossover tree 1 has no mutable nodes')

        a_nd = np.random.choice(a_nds)
        xt_out = a_nd.get_xtype_self()

        b_nds = bb.eval_mutable_nodes(match_xt=xt_out)

        if len(b_nds) > 0:
            b_nd = np.random.choice(b_nds)
        else:
            xt_out = float if xt_out == bool else bool  # the other swap type now sfeh:open
            b_nds = bb.eval_mutable_nodes(match_xt=xt_out)
            b_nd = np.random.choice(b_nds)
            a_nds = aa.eval_mutable_nodes(ignore_first=True, match_xt=xt_out)
            if len(a_nds) == 0:
                raise ValueError(f'Crossover cant find matching nodes')
            a_nd = np.random.choice(a_nds)

        # try:
        #     b_nd = np.random.choice(b_nds)
        # except ValueError:
        #
        #     # if len(b_nds) > 0:
        #     # else:
        #     xt_out = float if xt_out == bool else bool  # the other swap type now sfeh:open
        #     b_nds = bb.eval_mutable_nodes(match_xt=xt_out)
        #     b_nd = np.random.choice(b_nds)
        #     a_nds = aa.eval_mutable_nodes(ignore_first=True, match_xt=xt_out)
        #     a_nd = np.random.choice(a_nds)

        cpy = copy.deepcopy(a_nd)  # sfeh deepcopy required??

        a_nd.set_new_node(b_nd)
        b_nd.set_new_node(cpy)

        aa = self.evolve_prune_tree(tree=aa)
        bb = self.evolve_prune_tree(tree=bb)

        return aa, bb

    def evolve_prune_tree(self, tree: Node):
        """prune depth
        -> prune everything below a certain level... (should not happen in the first place)
        prune nodes
        -> get node difference, get nodelist, untill small enough: split the difference, prune nodes until

        sfeh:discussion there is a difference between parsimony and complexity...
        sfeh:discuss analyze the amount of trees that have to be pruned?
        sfeh:open add labelweight_max to"""
        nodelist = tree.eval_mutable_nodes()
        for dnode in nodelist:
            if dnode.depth == self.nodes_max and dnode.get_arity() > 0:
                print_warning('wwww', f'Node in fintree is too deep: {dnode.depth}')
                new_node = self.nc.choose_terminal(dnode.get_xtype_self(), as_node=True)
                new_node.depth = dnode.depth
                dnode.set_new_node(new_node)

        prune_amount = len(tree) - self.nodes_max
        while prune_amount > 0:
            print_warning('wwww', f'Tree too complex: {len(tree)} > {self.nodes_max}, pruning {prune_amount}.')
            nodelist = tree.eval_mutable_nodes()
            prune_now = 1 + np.random.randint(prune_amount)  # 19 -> prune branch with 1 to max. 19 nodes

            nodelist = [x for x in nodelist if len(x) >= prune_now]  # only (operator-) nodes
            tree = np.random.choice(nodelist)
            new_node = self.nc.choose_terminal(tree.get_xtype_self(), as_node=True)
            new_node.depth = tree.depth
            tree.set_new_node(new_node)
            prune_amount = len(tree) - self.nodes_max
        return tree

    def finalize_tree(self, tree):
        """When an evolution is done, this function...:
        - inserts node with input data, if tree has none yet
        - prunes tree (...should be handled in the respected evolution, as the pruning will affect random nodes)
        - sets depth in all nodes correctly
        - (currently) does not perform any checks (depth set correctly? )"""
        pass


class TreeMeta:

    def __init__(self, fitness, parsimony, expr_sym, tag):
        self.fitness = fitness
        self.parsimony = parsimony
        self.expr_sym = expr_sym
        self.last_evolution = deque([tag], maxlen=10)  # sfeh:open

    def append_tag(self, tag):
        self.last_evolution.append(tag)

    def get_last_tag(self):
        return self.last_evolution[-1]

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

    def __init__(self, tree: Node, meta: TreeMeta):
        self.tree = tree
        self.meta = meta

    def __str__(self):
        """Show the Parsimony and Fitness of a tree"""
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
#             # self.printpl('gg', f'Loading origin fintree, regr. error {fitness_train}.
#             Time: {time.perf_counter() - self.time_start:4.2f}s')
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
#     recursively loads a nested list into a evotree structure
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
#         raise Exception(f'Tree-building list length {len(lst[1:])} does not match arity {node.get_arity()}.')
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
#     tree_1 = evotree_from_nested_labels(_nested)
#     tree_1.update_fixed_nsteds(tree_0)
#
#     a = repr(tree_0)
#     b = repr(tree_1)
#
#     return a == b


# def evotree_from_nested_labels(nested_str, obs_list=None):
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
    evotree = fintree.get_evotree()
    evotree = copy.deepcopy(evotree)
    return evotree


if __name__ == '__main__':
    _test_open = '[Ifte, [Or, [b < -1], [And, [b < 0.1], [a < -0.05]]], 2, [Ifte, [And, [And, ' \
                 '[b > -0.45], [b < -0.05]], [a < -0.5]], 0, [Ifte, [a < 0], 0, 2]]]',
    _test_loadabls = ["['+',['-',['Ifte',['True'],['sign',['cartVel']],['/',[2.3],[4]]],['cartVel']],[-1.3]]",
                      '["Ifte:fix",["<",["cartVel"],[0]],["0:fix"],["2:fix"]]',
                      '["Ifte", ["Not", [False]], [0.0], [2.0]]']

    tb = TreeBuildRestrictions(['a', 'b'], 10, 30, float, origin_tree=None)
    tr = Node(Add, [Node(Symbol('a'), []), Node(Float(1.23), [])])
    tr = Node(Ifte, [Node(Gt, [Node(Symbol('a'), []), Node(Float(1.2), [])]), Node(Float(3.), []), Node(Float(2.), [])])
    tr = Node(Max, [Node(Symbol('a'), []), Node(Float(1.2), [])])
    x = tr.get_sympy_expr()
    tr2 = sympy_to_tree(x)
    print(tr, tr2)
    for _ in range(10):
        tr = tb.evolve_new_tree_depth(3, float, p_term=0.3)
        x = tr.get_sympy_expr()
        print('First sym success')
        tr_new = sympy_to_tree(x)
        x2 = tr_new.get_sympy_expr()
        print(tr)
        print(tr_new)
        if str(x) != str(x2):
            print()
            raise Exception(f'sympy process failed {x}, <-->, {x2}')
