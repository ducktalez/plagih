"""
plagih_tree contain a new implementation of trees that we use in genetic programming to display a program.
The old karoo "fintree" is replaced with, for now, "treer" in the code.
not all functions can use fintree for now and some fintree-functions require the old "fintree"
fintree splits the karoo fintree into the
- meta-info (fitness_train, parsimony, fintree-id, ...) and the
The core of the fintree, which "is" the fintree, is stored recursively
Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)

sfeh: write test that checks all operators for sympificytion (...+branch-combinations, and more?)
sfeh: use function-types (-> 'commutative'?)

"""

from plagih.sympy_extras import *
from plagih.tree_distances.tree_edit_distance import apted_distance

from dataclasses import dataclass
import itertools


# logging.basicConfig(filename='example.log', filemode='a', level=logging.DEBUG)  # sfeh encoding='utf-8' maybe in the future

# lol, lol. https://github.com/tensorflow/tensorflow/issues/27023 these messages are tingeling


@dataclass
class Node:
    """
    The core is the structure of a plagih gp-fintree.
    It recursively holds the nodes of a fintree; every fintree has a list of potential children.
    Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)

    states? sfeh:discuss
    [None]: not set
    [0]:    evolution/construction/build mode (potentially missing leaf nodes)
    [1]:    structurally complete/finalized branch (node_depths correct, node_id set, ...)
    [2]:    root-correct structure
    [3]:    including meta-data (fitness_train, complexity)
    """

    meta = None

    def __init__(self, label: 'NodeLabel' = None, depth=None, is_fix=False, childs=None, status=0):
        self.label = label
        self.is_fix = is_fix
        self.childs = childs or []
        self.depth = depth
        self.status = status  # 0 =

    def __hash__(self):
        """
        CAUTION: Do not use this function and do not delete this function
        CAUTION: This hash function has currently no use.
        The hash-value of a fintree was used as key for the LUT.
        However, the python hash-function has a run-specific salt for security reasons,
        making it impossible to load the LUT table between runs, so just use the str as key.
        """
        return hash(repr(self))  # sfeh

    def __str__(self):
        """
        Printing the nodes as nested array structure, easy to read.
        Also, have a look at __repr__(self) for a more detailed result
        """
        label_str = self.get_nlabel()  # sfeh or: return the label __str__

        if self.childs:
            childstr = ', '.join([str(x) for x in self.childs])
            label_str = f"{label_str}, {childstr}"

        return f"[{label_str}]"

    def __repr__(self):
        """
        Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to __str__(), but adds the following information:
        - ":fix", when nodes are fixed
        """
        label_str = self.get_nlabel()
        label_str = str(label_str)

        if self.is_fix:
            label_str += ':fix'

        if self.childs:
            childstr = ', '.join([repr(x) for x in self.childs])
            label_str = f"{label_str}, {childstr}"
        # elif self.is_root():
        #         label_str = f"[{label_str}]"  # another version
        return f"[{label_str}]"

    # def choose_term(xtype_out, choose_obs, choose_distributions, precision):
    #
    #     # sfeh 50% chance observation/value
    #     if random.choice(['obs', 'distrib']) == 'obs' and choose_obs[xtype_out]:
    #         obs = choose_obs[xtype_out]()
    #         # prsint('SAME???', obs.name, obs.label)  # sfeh
    #         return obs
    #     else:
    #         dist_fun = random.choice(choose_distributions[xtype_out])
    #         value = dist_fun()
    #         if xtype_out == float:  # sfeh int aswell?
    #             value = float(round(value, precision))
    #             const = FloatConstant(value)
    #         elif xtype_out == bool:
    #             const = BoolConstant(value)
    #         else:
    #             raise Exception('ASDASD NOOO WHYY')
    #         return const

    def __len__(self):
        """
        counting the amount of nodes recursively
        """
        return 1 + sum([len(cc) for cc in self.childs])

    def get_label(self):
        return self.label

    def get_nlabel(self):
        """

        """
        return self.label.nlabel

    def get_arity(self):
        return self.label.arity

    def get_xtype(self):
        return self.label.xtype

    def get_xtype_in(self):
        return self.label.xtype[0]

    def get_xtype_out(self):
        """type (float or bool), which is the nodes output"""
        return self.label.xtype[1]

    def is_root(self):
        """
        does this work while building a fintree?
        """
        return self.depth == 0

    def get_observation_list(self):
        """
        these are required for the evaluation (are loaded by Tensorflow)
        todo returns [cartVel, -cartVel], should not ever happen?
        """
        obslist = []
        if self.get_arity() > 0:
            obslist.extend(list(itertools.chain(*[cc.get_observation_list() for cc in self.childs])))
        elif isinstance(self.label, Observation):
            obslist.extend([self.get_nlabel()])

        return list(set(obslist))

    def set_label(self, label: 'NodeLabel'):
        """
        all other values are automatically set by assigning the respected node
        """
        self.label = label

    def set_childs(self, childs):
        """
        Sets the self.childs variable, which must be nodes or None
        """
        if len(childs) == self.get_arity():
            self.childs = childs
        return  # ==>STATE?

    def update_fixed_nodes(self, origin: 'Node'):
        """
        Updating the fixed nodes in a tree where they were lost for some reason.
        This should never be the case! But it happened during development of recreating a tree from expression.
        This might also be useful in tree checks
        """
        if origin.is_fix:
            if self.label.nlabel != origin.label.nlabel:
                raise
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.update_fixed_nodes(origin.childs[ii])

    def get_nodes_to_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """
        sum_layers=False, get_closest=True, return_all_layers=False
        """
        child_results = []
        if self.depth < goal_depth:
            child_results = sum(
                [child.get_nodes_to_depth(goal_depth, only_mutable=only_mutable, force_depth=get_closest_depth) for
                 child in self.childs], [])

        if only_mutable and self.is_fix or \
                get_closest_depth and self.depth != goal_depth:
            my_result = []
        else:
            my_result = [self]

        return my_result + child_results

    def get_labellist_breath(self):
        """
        Returns all labels in a core node
        Breitensuche im Baum
        """
        label_list = []
        max_depth = self.childs_depth_max
        for depth in range(0, max_depth + 1):
            labels_at_depth = [x.label for x in self.get_nodes_at_depth(depth)]
            label_list.extend(labels_at_depth)

        return label_list

    def get_all_nodes(self):
        if len(self.childs) == 0:
            return [self]
        else:
            return [self] + [cc.get_all_nodes() for cc in self.childs]

    def get_nodes_at_depth(self, goal_depth, allow_fixed=False, expand_depth=False):
        """
        Returns a list with mutable ids which are *goal_depth* layers away from non-modifiable nodes
        last_leaves: if you want so save all leave nodes aswell

        sum_layers=False, get_closest=True, return_all_layers=False

        """
        nodes = []
        if (not self.is_fix or allow_fixed) and (
                self.depth == goal_depth or (self.depth > goal_depth and expand_depth)):
            return [self]
        elif self.depth <= goal_depth or expand_depth:
            nodes.extend(list(itertools.chain(
                *[cc.get_nodes_at_depth(goal_depth, allow_fixed=allow_fixed, expand_depth=expand_depth) for cc in
                  self.childs])))
            return nodes
        else:
            return []

    def eval_expr_todo(self):
        """
        """

        expr_str = self.label.nlabel
        expr_str = f'"{expr_str}"'  # todo

        if self.childs:
            cc_expr = [cc.eval_expr_todo() for cc in self.childs]
            cc_expr = ', '.join(cc_expr)

            expr_str = f'{expr_str}, {cc_expr}'

        return f'[{expr_str}]'

    def eval_nested(self):
        """

        """
        expr_str = self.label.nlabel
        expr_str = f'"{expr_str}"'  # todo

        if self.childs:
            cc_expr = [cc.eval_nested() for cc in self.childs]
            cc_expr = ', '.join(cc_expr)

            expr_str = f'{expr_str}, {cc_expr}'

        return f'{expr_str}'

    def eval_expr_str(self):
        """
        Accumulate and return the complete expression the fintree holds recursively
        todo directly to sympy in a different method
        """

        _expr = f'{self.label.nlabel}'

        if self.childs:
            _cc_expr = ', '.join([cc.eval_expr_str() for cc in self.childs])

            _expr = f'{_expr}({_cc_expr})'

        return _expr

    # def eval_expr_todo_old(self):
    #     """
    #     """
    #     if self.get_arity() > 0:
    #         child_expr_list = [cc.eval_expr_todo_old() for cc in self.childs]
    #         return self.label.expr_sym.format(*child_expr_list)  # *list makes the list args :D f'cos({})'([33]) does not work.
    #     else:
    #         return self.label.expr_sym

    def eval_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
        e.g. {+{Ifte{True}{1}{2}}{3}}
        """
        # sfEh check if this still works as one-liner
        return f"{{{self.get_nlabel()}{''.join([cc.eval_apted_notation() for cc in self.childs])}}}"

    def get_max_depth(self, depth=0):
        """
        Go through all nodes, save depth
        """
        if len(self.childs) == 0:
            return depth
        else:
            return max(cc.get_max_depth(depth=depth + 1) for cc in self.childs)

    def repair_depth(self, depth=0):
        """
        aka set_depth recursively for all nodes in a branch
        mainly used in branch

        The depth is written inevery node (for whatever reason), and instead of having to propagate
        the depth through every crossover/branch mutation function, instead, we call it when replacing nodes
        """
        self.depth = depth
        for cc in self.childs:
            cc.repair_depth(depth=depth + 1)

    def set_new_node(self, new_node: 'Node'):
        """
        was: new_core
        """
        self.set_label(new_node.get_label())
        self.childs = new_node.childs or []  # maybe must be updated recursively

        self.repair_depth(self.depth)  # Especially required for crossover or branches

    def eval_mutable_nodes(self, xtype_out=None, allow_root=True):
        """
        return all nodes that are mutable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!
        """
        node_list = []
        if not self.is_fix:  # requirement for mutability
            # crossover requires excluding types that are not matching, and excludes the root node
            if (xtype_out is None or xtype_out == self.get_xtype_out()) and (allow_root or not self.is_root()):
                node_list.append(self)

        for cc in self.childs:
            node_list.extend(cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root))
        # deprecated:
        # node_list.extend(list(itertools.chain(
        #     *[cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root) for cc in self.childs])))
        return node_list

    def evolve_mutate_filter_branch(self):
        """
        Recursively filter the nodes in the branch of fintree
        sfeh:   random filter all terminal nodes /
                single node /
                nodes in a branch /
                random nodes in a branch /
                intelligent filtering
        """
        # self.state = STATE_BUILDING  #  ==>state
        if self.get_arity() > 0:
            for cc in self.childs:
                cc.evolve_mutate_filter_branch()
        else:
            self.label.mutate_self_filter(filter_type='gaussian_filter')

    # def finalize_set_nodepath(self, nodepath):
    #     """
    #     [0,2,1,0,0]
    #     ==>ROOT
    #     """
    #     self.nodepath = nodepath
    #     for ii, child in enumerate(self.childs):
    #         nodepath_child = nodepath + [ii]
    #         child.finalize_set_nodepath(nodepath_child)

    def finalize_set_depth(self, depth=0, recursive=True):
        """
        depth=0 is the root node
        """
        self.depth = depth
        max_depth = depth
        if recursive:
            for cc in self.childs:
                cc_depth = cc.finalize_set_depth(depth=depth + 1)
                max_depth = max(cc_depth, max_depth)

        self.childs_depth_max = max_depth

        return max_depth

    def check_typing(self, xtype_parent, fatal=True):  # sfeh
        """
        Checks, if all child nodes match the parents nodes types
        """
        result = [self.get_xtype_out() == xtype_parent,
                  self.get_xtype_in() == tuple([cc.get_xtype_out() for cc in self.childs]),
                  len(self.childs) == self.get_arity()]

        if sum(result) < len(result) and fatal:
            raise  # sfeh

        for ii, cc in enumerate(self.childs):
            cc.check_typing(self.get_xtype_in()[ii], fatal=fatal)

        return True

    def check_depth_infos(self, depth=0, fatal=None):
        """
        Checks, whether the depth of all nodes in the tree are set correctly.
        Should not be necessary if all evolutions (branch-mutation, crossover) work fine
        - starts at depth 0, increase at every level
        - compare with the written value
        :return: boolean
        """
        if depth != self.depth:
            if fatal:
                raise
            return False
        return all([cc.check_depth_infos(depth=depth + 1, fatal=fatal) for cc in self.childs])

    def selfcheck(self, check_depth=True, fatal=None):
        """
        Tree Self-check for its structure
        """
        results = [self.check_depth_infos(fatal=fatal) if check_depth else 0]
        return sum(results)


def eval_parsimony(tree: Node, complexity_measure, origin_tree: Node = None):
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
    else:
        raise Exception(f'Complexity measurement not available: {complexity_measure}')

# sfeh:discussion especially with mc: there can be more than one pareto entry with the same parsimony/fitness!


# class ObservationIndex(Observation):
#     """
#       sfeh:open
#     """
#
#     def __init__(self, nlabel, xtype_out=float, obs_indizes=None):
#         # super().__init__(nlabel, xtype_out)
#         self.obs_indizes = obs_indizes
#         latex = f'\\text{{{self.fam}}}_{{{self.timeindex}}}'  # remove this {self.preexpr}
#         self.latex = (latex, latex)  # remove this {self.preexpr}
#
#     def mutate_self_filter(self):
#         new_index = int(max(min(round(random.gauss(self.timeindex, 1)), self.index_minmax[1]), 0))
#         self.timeindex = new_index
#         self.name = f'{self.fam}_{new_index}'
