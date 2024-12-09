
from random import Random

from plagih.trees import *
from plagih.util import *


def check_operator_pool(ops: iter):
    """Check if the user-specified loaded operators allow closure
    (either float-only/bool only or all 4 types of operators)
    @:param operator_pool: list with operators and their weight of being selected

    Example, only works for numbers:
    dict_operator_pool = {Add: 2, Sub: 1, Mul: 2, Div: 1}
    """

    opxtypes = [oper.xtype for oper in ops.keys()]
    has_2f = any([float == i[1] for i in opxtypes])
    has_2b = any([bool == i[1] for i in opxtypes])
    has_f2b = any([float in i[0] and bool == i[1] for i in opxtypes])
    has_b2f = any([bool in i[0] and float == i[1] for i in opxtypes])
    if not all([has_2f, has_2b, has_f2b, has_b2f]):
        print_warning('w', f'Loaded operators do not feature both numeric (float) and bool type.')
    if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
        raise Exception(f'Loaded operators do not allow closure!')


def norm_choices(val_p_tuples: (any, float)) -> [any, float]:
    """make a tuple-list callable for weighted numpy choice
    [['a', 1], ['b', 2]] -> [('a', 'b'), (0.333, 666)]"""
    xx = list(zip(*val_p_tuples))
    # normalizing the probabilities in every case to a sum of 1 (100%)
    psum = sum(xx[1])
    xx[1] = [i / psum for i in xx[1]]
    # lambda: np.random.choice(xx[0], p=xx[1])
    return xx


def operatorpool_to_picks(d_operator_pool):
    check_operator_pool(d_operator_pool)
    pick_op = {float: [], bool: []}
    pick_op_match = {}
    for _cls, _p in d_operator_pool.items():
        xt = _cls.xtype
        pick_op[xt[1]].append([_cls, _p])
        if pick_op_match.get(xt, None) is None:
            pick_op_match[xt] = []
        pick_op_match[xt].append([_cls, _p])

    pick_op = {float: norm_choices(pick_op[float]),
               bool: norm_choices(pick_op[bool])}
    for k_xt in pick_op_match.keys():
        pick_op_match[k_xt] = norm_choices(pick_op_match[k_xt])
    return pick_op, pick_op_match


# class NodeCreatorBase(ABC):
#
#     @abstractmethod
#     def choose_operator(self, xt):
#         pass
#
#     @abstractmethod
#     def choose_operator_match(self, xtype):
#         pass
#
#     @abstractmethod
#     def choose_terminal(self, xt):
#         pass
#
#     @abstractmethod
#     def choose_constant(self, xt):
#         pass
#
#     @abstractmethod
#     def choose_symbol(self, xt):
#         pass


class NodeSelect:

    def __init__(self, operators: dict, symbol_list: [sympy.Symbol]):
        """make all probabilities sum to 1 for each categoray (Add: 2, Mul: 1, Tan: 0.5) in

        sfeh: replace operators-"dict" with a cost-value in the operators class that can be set and is considered
            in the random choose-function?
        """

        self.pick_op, self.pick_op_match = operatorpool_to_picks(operators)
        # sfeh: Acos: 0.33, Asin: 0.33, Atan: 0.33, Tanh: 0.5, Usub: 1, Xor: 1
        # Round: 0.5, Eq: 1,  # Ne: 0.5, #  # Log1p: 0.1, Gt: 0.1, Ge: 0.1,, Tan: 0.1, Sub: 1, Cos: 0.33
        # Powrounded: 0.5

        self.pick_symbol = {
            # float: norm_choices([[symbols_lambda(ii), 1] for ii in symbols]),
            float: norm_choices([[ii, 1] for ii in symbol_list]),
            bool: []}  # NotImplementedError

        # -> Choosing 50 random numeric values from the dataset for building trees ...just not zeros)
        # samples = [ii for ii in itertools.chain.from_iterable(df[build_variables_list].sample(n=50).values) if ii != 0]
        self.pick_constant = {float: norm_choices([
            [lambda: round(random.normalvariate(1, 1), FLOAT_PRECISION), 0.1],
            [lambda: round(random.randint(1, 20), FLOAT_PRECISION), 0.1],
            # [lambda: round(random.choice(samples), FLOAT_PRECISION), 0.5]
        ]),
            bool: norm_choices([[lambda: random.choice((True, False)), 1]])}

    def choose_operator(self, xt) -> type(BaseOperator):
        op = np.random.choice(self.pick_op[xt][0], p=self.pick_op[xt][1])  # no (), which would evaluate the op
        return op

    def choose_operator_match(self, xtype):
        if CHAIN_implement:
            pass
        op = np.random.choice(self.pick_op_match[xtype][0], p=self.pick_op_match[xtype][1])
        return op

    def choose_terminal_node(self, xt, p_observation=0.5):
        """
        # sfeh expected str|int|long|float|Decimal|Number object but got 'Node'
        """
        if np.random.random() > p_observation:
            try:
                _v = self.choose_symbol_node(xt)
                return _v  # MUST STAY HERE
            except (TypeError, IndexError):
                # return a constant (E.g. because there are no boolean observations)
                pass

        _v = self.choose_constant_node(xt)

        return _v

    def choose_constant_node(self, xt):
        _v = np.random.choice(self.pick_constant[xt][0], p=self.pick_constant[xt][1])()  # just dist. must be ()
        if xt == float:
            _v = sympy.Float(_v)  # sfeh:discuss allow "rational" inputs? 1/3, 3/4, ...
            # _v = sympy.Rational(_v)  # sfeh:discuss allow "rational" inputs? 1/3, 3/4, ...
            # return nd(Number, [_v])  # sfeh: check all "Was here"; round FLOAT_PRECISION was here
            return nd(Number, _v)  # round FLOAT_PRECISION was here
        else:
            # _v = sympy.logic.boolalg.BooleanAtom(_v)  # sfeh:discuss: vs. Boolean
            # -> sympy.sympify('And(True, BooleanAtom(False))')
            _v = _v  # BooleanAtom was here - why? Any purpose?
            return nd(Boolean, _v)

    def choose_symbol_node(self, xt) -> Symbol:
        """similar to choose_terminal_node()
        sfeh: delete?"""
        _v = np.random.choice(self.pick_symbol[xt][0], p=self.pick_symbol[xt][1])
        return nd(Symbol, _v)


class Evolution:
    """
    was "TreeBuildRestrictions"
    functions to build trees, with the advantage of being able to use general build restrictions.

    sfeh: all Symbol-inputs are chosen from a list with equal probability.
        -> don't overcomplicate this process.
        -> provide more options when asked for, like giving random()-probabilities
    """

    operator_presets = {'math_simple':
                        {Add: 2, Mul: 2, Div: 1, Square: 0.75, Abs: 0.5, Sign: 0.5, Sqrt: 0.1, Log: 0.1,
                         Sin: 0.5, Not: 0.5, Lt: 0.5, Le: 0.5, And: 1, Or: 1, Min: 1, Max: 1}}

    def __init__(self, symbol_list=None, origin_xtype=float, operators=None, origin_tree=None,
                 depth_max=10, nodes_max=100, complexity_metric='tree_node_count_fair', allow_chain=None):
        """
        origin_tree: A tree, which
        sfeh:warning if options are left empty?
        """
        self.origin_xtype = origin_xtype
        self.origin_tree = origin_tree

        # operators -> {Add: 1}
        if operators is None:
            operators = self.operator_presets['math_simple']
        elif isinstance(operators, str):
            operators = self.operator_presets[operators]
        elif isinstance(operators, list):
            operators = {e: 1 for e in list(operators)}
        elif isinstance(operators, dict):
            pass
        else:
            raise NotImplementedError

        if symbol_list is None:
            symbol_list = sympy.symbols('a b')  # sfeh:sympy symbols options

        self.node_selector = NodeSelect(operators, symbol_list)

        self.complexity_metric = complexity_metric

        self.depth_max = depth_max
        self.nodes_max = nodes_max

        self.allow_a_chain = allow_chain

    def evolve_prune_tree(self, tree: Node, allow_chain):
        """prune depth
        -> prune everything below a certain level... (should not happen in the first place)
        prune nodes
        -> get node difference, get nodelist, untill small enough: split the difference, prune nodes until

        sfeh:discussion there is a difference between parsimony and complexity...
        sfeh:discuss analyze the amount of trees that have to be pruned?
        sfeh:open add labelweight_max to"""
        nodelist = tree.list_mutable_nodes(allow_chain=allow_chain)
        for dnode in nodelist:
            if dnode.depth == self.nodes_max and dnode.get_arity() > 0:
                print_warning('wwww', f'Node in fintree is too deep: {dnode.depth}')
                new_node = self.node_selector.choose_terminal_node(dnode.get_xtype_self())
                new_node.depth = dnode.depth
                dnode.set_new_node(new_node)

        prune_amount = len(tree) - self.nodes_max
        while prune_amount > 0:
            print_warning('wwww', f'Tree too complex: {len(tree)} > {self.nodes_max}, pruning {prune_amount}.')
            nodelist = tree.list_mutable_nodes(allow_chain=allow_chain)
            prune_now = 1 + np.random.randint(prune_amount)  # 19 -> prune branch with 1 to max. 19 nodes

            nodelist = [x for x in nodelist if len(x) >= prune_now]  # only (operator-) nodes
            tree = np.random.choice(nodelist)
            new_node = self.node_selector.choose_terminal_node(tree.get_xtype_self())
            new_node.depth = tree.depth
            tree.set_new_node(new_node)
            prune_amount = len(tree) - self.nodes_max

        return tree

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

    def evolve_new_tree_depth(self, depth_goal, xt_out, p_term=0.0) -> Node:

        if self.origin_tree is not None:

            evotree = copy.deepcopy(self.origin_tree)
            layer0 = evotree.get_mutable_rootnodes(extend_lvls=0)
            # sfeh:debug more, also... takes time, just define the nodes in tree for mutation once?

            for ii, nd in enumerate(layer0):  # -> get layer every time (nsted ids might have changed)
                new_subbranch = self.evolve_create_random(nd.get_xtype_self(), depth_goal, num_rest=-1,
                                                          depth=nd.depth, p_term=p_term)
                nd.set_new_node(new_subbranch)

        else:
            evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    def evolve_chained_new_tree_depth(self, depth_goal, xt_out, p_term=0.0) -> Node:

        evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    # def new_tree_nodes(self, nn, allow_chain, p_term=0):
    #     """insert a (random) number of branches at the first possible "layer" (not necessarily depth)
    #     (If all nodes are modifiable, it is the root node. Otherwise, it is the first layer of modifiable nodes
    #     - get these nodes, randomly choose a subset of those
    #     - get the amount of nodes allowed to add. (max_nodes, without the core, + the nodes about to delete)
    #     - split the amount of nodes up (randomly) and add these new branches to the fintree
    #     sfeh:idea mutate only the childs of a node! The label stays the same"""
    #
    #     # if not isinstance(self.origin_tree, RootNode_Dummy):
    #     #
    #     # else:
    #     # layer0_nodes = self.origin_root
    #     # evotree = self.evolve_create_random(xtype, -1, num_rest=nodeamount, depth=0, p_term=p_term)
    #
    #     evotree = node_deepcopy(self.origin_tree)
    #     layer0_nodes = evotree.get_mutable_rootnodes()
    #     layer0_splits = randomly_split_range(nn, len(layer0_nodes))
    #
    #     for ii, node0 in enumerate(layer0_nodes):  # pareto_insert branches! get layer always, node ids might change
    #         lvl0_node = np.random.choice(node0.list_mutable_nodes(allow_chain=allow_chain))  # layer0_branch =
    #         # branch_size = layer0_nodes[ii]  # sfeh:idea + len(lvl0_node)
    #         new_subbranch = self.new_tree_nodes(lvl0_node.get_xtype_self(), allow_chain, p_term=p_term)
    #         lvl0_node.set_new_node(new_subbranch)
    #
    #     return evotree

    def evolve_create_random(self, xt_out, depth_max_local, num_rest=-1, depth=0, p_term=0.0) -> Typus:
        """
        sfeh: just use depth_rest and calculate it earlier with depth_max_local and self.depth_max
        sfeh: make this specific to tree complexity measure?
        num_rest: -1 ignores the node number restriction
        depth_max_local: can be set lower than self.depth_max
        sfeh:open make depth_goal -> depth_rest"""

        # setting a terminal-node if it is required OR p_term is met
        if depth >= min(self.depth_max, depth_max_local) or num_rest == 0 or random.random() < p_term:
            node = self.node_selector.choose_terminal_node(xt_out)
        else:

            node_cls = self.node_selector.choose_operator(xt_out)
            child_xts = node_cls.get_child_xts()
            childs = []

            if CHAIN_implement:  # sfeh:open
                pass  # optional; just add more node here already

            nums = randomly_split_range(num_rest - 1, len(child_xts))  # sfeh len childlist is weak. chain, also.

            for ii, xt in enumerate(child_xts):
                cc = self.evolve_create_random(xt, depth_max_local, num_rest=nums[ii], depth=depth+1, p_term=p_term)
                childs.append(cc)
            # else:
            #     for xt in child_xts:
            #         cc = self.evolve_create_random(xt, depth_max_local, num_rest=-1, depth=depth+1, p_term=p_term)
            #         childs.append(cc)
            node = node_cls(*childs)

        node.depth = depth
        # node = Node(label, childs, depth=depth)

        return node

    def evolve_new_endrecursive(self, depth_goal, num_rest=-1, depth=0, p_term=0):
        """Evolve, creating a new branch in this node
        """
        # sfeh:open This is currently unused
        num_rest -= 1  # sfeh i guess

        if self.origin_tree is not None:
            evotree = copy.deepcopy(self.origin_tree)
            layer0 = evotree.get_mutable_rootnodes()

            for ii, nodes0 in enumerate(layer0):  # -> get layer every time (nsted ids might have changed)
                nd_list = nodes0.eval_mutable_nsteds()
                lvl0_nodes = np.random.choice(nd_list)
                new_subbranch = self.evolve_new_endrecursive(lvl0_nodes.get_xtype_self(), depth_goal,
                                                             depth=lvl0_nodes.depth, p_term=p_term)
                lvl0_nodes.set_new_node(new_subbranch)

        else:
            evotree = self.evolve_new_endrecursive(self.origin_xtype, depth_goal, depth=depth,
                                                   p_term=p_term)
        return evotree

    def evolve_mutate_filter(self, tree, allow_chain):
        """Mutates a number of float terminal of a fintree
        - filter point/branch/all, branch can also affect a point only as well as all nodes
        - filter observations?
        - filter terminals
        - filter with which filter?"""

        _nd = np.random.choice(tree.list_mutable_nodes(allow_chain=allow_chain))
        _nd.evolve_mutate_filter_gauss()

        return tree

    def evolve_mutate_point(self, tree: Node, allow_chain):
        """Mutate a single mutable point in any Tree.
        sfeh:debug is the fintree a fintree copy or the same fintree?"""
        evotree = copy.deepcopy(tree)

        node = rnd_choice(evotree.list_mutable_nodes(allow_chain=allow_chain))  # debug if ignores chains
        xtype = node.get_xtype_tuple()

        if node.is_operator():
            # sfeh:allow_chain
            # sfeh:what if its the same function?
            new_label = self.node_selector.choose_operator_match(xtype)  # Function is same type, same arity
            node.set_typus(new_label)
        elif node.is_term:
            new_node = self.node_selector.choose_terminal_node(xt_self(xtype))
            node.set_new_node(new_node)
        else:
            raise NotImplementedError

        return evotree

    def evolve_mutate_branch_depth(self, tree: Node, depth_goal, allow_chain, p_term=0.0):
        """"""
        n_init = len(tree)
        node_list = tree.list_mutable_nodes(allow_chain=allow_chain)
        node = np.random.choice(node_list)
        xtype_out = node.get_xtype_self()  # ValueError: 'a' cannot be empty unless no samples are taken
        branch = self.evolve_create_random(xtype_out, depth_goal, num_rest=self.nodes_max - n_init, depth=0,
                                           p_term=p_term)
        node.set_new_node(branch)

        return tree

    def evolve_mutate_branch_nodes(self, tree: Node, nodes_goal, p_term=0.0):
        """currently only one branch
        p_term: probability terminating the tree in a node
        """
        nodes_init = len(tree)
        if tree is None:
            raise NotImplementedError('SFEH:open Implement standard selection mechanism')
        nd = tree.list_mutable_nodes(allow_chain=self.allow_a_chain)
        nd = rnd_choice(nd)
        xt_out = nd.get_xtype_self()
        nodes_goal = min(self.nodes_max - (nodes_init - len(nd)), nodes_goal)

        branch = self.evolve_create_random(xt_out, -1, num_rest=nodes_goal, depth=nd.depth, p_term=p_term)
        nd.set_new_node(branch)
        return tree

    def evolve_crossover(self, aa: Node, bb: Node):
        """Evolution with crossover of branches between two trees
        currently only one branch

        swap branches of two trees
        - select parent aa and bb
        - select swappable branche for a_parent from b_parent
            - select aa node in aa (and crossover here, no matter what)
        - delete a_parent branch and pareto_insert b_parent branch (which tactic?)
        sfeh:idea into main fintree?"""

        # aa = node_deepcopy(tree1)
        # bb = node_deepcopy(tree2)

        a_nds = aa.list_mutable_nodes(skip_first=True)  # sfeh ...why actually ignore root node?
        # a_nds = [x for x in a_nds if len(x) > 1]  # ignore terminal nodes

        if len(a_nds) == 0:
            raise ValueError(f'Crossover tree 1 has no mutable nodes!')

        a_nd = np.random.choice(a_nds)
        xt_out = a_nd.get_xtype_self()
        b_nds = bb.list_mutable_nodes(xtype=xt_out)

        if len(b_nds) > 0:
            b_nd = np.random.choice(b_nds)
        else:
            xt_out = float if xt_out == bool else bool  # switching to the other swap type
            b_nds = bb.list_mutable_nodes(xtype=xt_out)
            b_nd = np.random.choice(b_nds)
            a_nds = [x for x in a_nds if x.get_xtype_self() == xt_out]
            if len(a_nds) == 0:
                raise ValueError(f'Crossover cant find matching nodes. This Should always be possible.')
            a_nd = np.random.choice(a_nds)

        cpy = copy.deepcopy(a_nd)  # sfeh deepcopy required??

        a_nd.set_new_node(b_nd)
        b_nd.set_new_node(cpy)

        aa = self.evolve_prune_tree(tree=aa, allow_chain=True)
        bb = self.evolve_prune_tree(tree=bb, allow_chain=True)

        return aa, bb

    def finalize_tree(self, tree):
        """When an evolution is done, this function...:
        - inserts node with input data, if tree has none yet
        - prunes tree (...should be handled in the respected evolution, as the pruning will affect random nodes)
        - sets depth in all nodes correctly
        - (currently) does not perform any checks (depth set correctly? )"""
        # sfeh:open
        pass


def randomly_split_range(range_max: int, num_splits: int) -> list[int]:
    """split integer range randomly into num_splits parts
    [1..100] -> [33, 15, 52]
    used for building trees
    0 is allowed! (ends a branch with a terminal node)
    sfeh:discuss create 2 more random split values and remove largest and smallest entry. (better distribution?)
      -> No. Also, allow 0 nodes."""

    if range_max < 0:
        return [-1 for _ in range(num_splits)]

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


if __name__ == "__main__":
    evo = Evolution()
    for _ in range(10):
        tree = evo.evolve_create_random(float, 4)
        print(tree)