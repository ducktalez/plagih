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


def __hash__(self):
    CAUTION: Do not use this function and do not delete this function
    CAUTION: This hash function has currently no use.
    The hash-value of a fintree was used as key for the LUT.
    However, the python hash-function has a run-specific salt for security reasons,
    making it impossible to load the LUT table between runs, so just use the str as key.
    return hash(repr(self))


Enriching the python-core 'sympy'. Sympy is used to unify and reduce functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.

- implementing missing functions in sympify, e. g. 'if a then b else c'.
- All number-related functions must have set
    is_real = True
    otherwise: '1 < BinaryMax(2, Ifte(1 < a, 1, 1))' will crash. (< operators only work on non-complex - aka real numbers)
    check for is_number if required.

- Classes must currently have the exact same name as their occurance (Ifte -> Ifte, not ifte or so)
    This is because when None is returned, the class name gets replaced at the function. could be solved, but why though :P
The following line is an honorable mention for myself; it was required for
## if ((a == True or a == False) and (b == True or b == False)) == (sympify(a).is_Boolean and sympify(b).is_Boolean):

Useful information:
- These variables are set for every sympy object and thus can be tested, e.g. a.is_Boolean
    # To be overridden with True in the appropriate subclasses

    #sfeh:open combine the nodes with the sympy shizzle
    sfeh xxx input variables as locals? Provide information such as real, integer, positive, range/interval?
    sfeh:open this is probably the reason for the capitalized class names in sympy: return eval(self, a)
    sfeh: I think we should get rid of sympy in the long term. A lot of problems are related to sympy.

    sfeh:sypyunification errors:
        - 'a and b', 'b and a'
        - 'And(a<2, a < 5)'
        - sympy.simplify('sign(-a)') -> -sign(a)

    sfeh:xxx sympy facttor (up/downfactor), so it adds stuff together
    sfeh:discus simplify/unify
"""
import copy
import os
import random

import numpy as np

from plagih.util import get_subclasses, PRECISION

os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import sympy.functions.elementary.piecewise  # sfeh: needs separate import?
import sympy
from dataclasses import dataclass

import itertools
import re


# lol, lol. https://github.com/tensorflow/tensorflow/issues/27023 these messages are tingeling


@dataclass
class NestedStruc:
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

    def __init__(self, anylabel, depth=None, is_fix=False, childs=None):
        try:
            anylabel.args = None  # sfeh: or remove childs, isfix
            anylabel.is_fix = None
            self.label: 'BaseTree' = anylabel
        except AttributeError:
            self.label = anylabel

        self.is_fix = is_fix
        self.childs = childs or []
        self.depth = depth

    def __str__(self):
        """
        Printing the nodes as nested array structure, easy to read.
        Also, have a look at __repr__(self) for a more detailed result
        """
        label_str = self.get_nlabel()

        if self.childs:
            childstr = ', '.join([str(x) for x in self.childs])
            label_str = f"{label_str}, {childstr}"

        return f"[{label_str}]"

    def eval_str(self):

        return self.get_nlabel()  # sfeh open

    def __repr__(self):
        """
        Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to str(), but adds the following information:
        - ":fix", when nodes are fixed
        """
        label_str = self.get_nlabel()
        label_str = str(label_str)

        if self.is_fix:
            label_str += ':fix'

        if self.childs:
            childstr = ', '.join([repr(x) for x in self.childs])
            label_str = f"{label_str}, {childstr}"
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
        return self.label

    def get_arity(self):
        return len(self.label.xtype[0])

    def get_xtype(self):
        return self.label.xtype

    def get_xtype_in(self):
        return self.label.xtype[0]

    def get_xtype_out(self):
        return self.label.xtype[1]

    def is_root(self):
        """
        does this work while building a fintree?
        """
        return self.depth == 0

    def set_label(self, label: 'NestedStruc'):
        """
        all other values are automatically set by assigning the respected node
        """
        self.label = label

    def update_fixed_nodes(self, origin: 'NestedStruc'):
        """
        Updating the fixed nodes in a tree where they were lost for some reason.
        This should never be the case! But it happened during development of recreating a tree from expression.
        This might also be useful in tree checks
        """
        if origin.is_fix:
            if str(self.label) != str(origin.label):
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

    def eval_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
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

    def set_new_node(self, new_node: 'NestedStruc'):
        self.set_label(new_node)  # sfeh remove childs, is_fix...
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
            # sfeh:open in anderer klasse
            if (xtype_out is None or xtype_out == self.get_xtype_out()) and (allow_root or not self.is_root()):
                node_list.append(self)

        for cc in self.childs:
            node_list.extend(cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root))
        # deprecated:
        # node_list.extend(list(itertools.chain(
        #     *[cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root) for cc in self.childs])))
        return node_list

    def evolve_mutate_filter_branch(self, precision=6):
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
                cc.evolve_mutate_filter_branch(precision=precision)
        else:
            # self.label.mutate_self_filter(filter_type='gaussian_filter', precision=precision)
            # sfeh:xxx
            pass

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


class BaseTree:
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

    # sfeh:discuss set tree depth at the end or so
    """
    insym = None
    is_fix = False  # sfeh:xx here?
    args = []
    xtype = None  # sfeh: will this b deprecated?

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args = [x for x in args]
        obj.is_fix = kwargs.get('is_fix', False)
        return obj

    def __str__(self):
        return self.__class__.__name__

    def _sympy_(self):  # -> sympy.Basic:
        _sym = self.insym

        if self.args:
            childstr = [sympy.sympify(cc) for cc in self.args]
            _sym = _sym(*childstr)

        return _sym

    def get_str(self):
        return self.__class__.__name__

    # def eval_expr_str(self):
    #     """
    #     Accumulate and return the complete expression the fintree holds recursively
    #     """
    #     _expr = self.get_nclass()
    #
    #     if self.args:
    #         _cc_expr = ', '.join([cc.eval_expr_str() for cc in self.args])
    #         _expr = f'{_expr}({_cc_expr})'
    #
    #     return _expr

    # def get_tfgraph(self):
    #     _graph = self.tflow
    #
    #     if self.args:
    #         _graph.args = [cc.eval_expr_str() for cc in self.args]

    def __repr__(self):
        return self.__class__.__name__

    def __len__(self):
        return 1 + sum([len(cc) for cc in self.args])

    # def get_xtype_out = sfeh xtype[1]

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

    def get_nclass(self):
        return self.__class__.__name__

    # def get_xtype(self):
    #     return self.xtype
    #
    # def get_xtype_in(self):
    #     return self.xtype[0]

    # def get_depth(self):
    #     return self.depth
    #
    # def set_depth(self, depth):
    #     self.depth = depth

    def is_root(self):
        """
        does this work while building a fintree?
        """
        return self.depth == 0

    # sfeh:delete, deprecated, but bodymight be useful still
    # def get_observation_list(self):
    #     """
    #     these are required for the evaluation (are loaded by Tensorflow)
    #     sfeh:bug returns [cartVel, -cartVel], should not ever happen?
    #     -> SFEH: But it is also not used anymore
    #     """
    #     obslist = []
    #     if self.garitgetya() > 0:
    #         obslist.extend(list(itertools.chain(*[cc.get_observation_list() for cc in self.args])))
    #     elif isinstance(self.label, Observation):
    #         obslist.extend([self.get_nlabel()])
    #
    #     return list(set(obslist))

    # def set_label(self, label: 'NodeLabel'):
    #     """
    #     all other values are automatically set by assigning the respected node
    #     """
    #     self.label = label

    def set_childs(self, childs):
        """
        Sets the self.args variable, which must be nodes or None
        """
        self.args = childs

    def set_child_n(self, n, new_node):
        """
        Sets the self.args variable, which must be nodes or None
        """
        self.args[n] = new_node

    def update_fixed_nodes(self, other: 'BaseTree'):
        """
        Updating the fixed nodes in a tree where they were lost for some reason.
        This should only occur during the reconstructing test of a tree from expression.
        """
        if other.is_fix:
            if self.get_nclass() != other.get_nclass():
                raise
            self.is_fix = True
            for ii, cc in enumerate(self.args):
                cc.update_fixed_nodes(other.args[ii])

    # def get_nodes_to_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
    #     """
    #     sum_layers=False, get_closest=True, return_all_layers=False
    #     """
    #     child_results = []
    #     if self.depth < goal_depth:
    #         child_results = sum(
    #             [child.get_nodes_to_depth(goal_depth, only_mutable=only_mutable, force_depth=get_closest_depth) for
    #              child in self.args], [])
    #
    #     if only_mutable and self.is_fix or \
    #             get_closest_depth and self.depth != goal_depth:
    #         my_result = []
    #     else:
    #         my_result = [self]
    #
    #     return my_result + child_results
    #
    # def get_labellist_breath(self):
    #     """
    #     Returns all labels in a core node
    #     Breitensuche im Baum
    #     """
    #     label_list = []
    #     max_depth = self.args_depth_max
    #     for depth in range(0, max_depth + 1):
    #         labels_at_depth = [x.label for x in self.get_nodes_at_depth(depth)]
    #         label_list.extend(labels_at_depth)
    #
    #     return label_list
    #
    # def get_all_nodes(self):
    #     if len(self.args) == 0:
    #         return [self]
    #     else:
    #         return [self] + [cc.get_all_nodes() for cc in self.args]

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
                  self.args])))
            return nodes
        else:
            return []

    # def eval_nested(self):
    #     """
    #
    #     """
    #     expr_str = self.label.nlabel
    #     expr_str = f'"{expr_str}"'  # sfeh: Also not used anymore? to-do was here
    #
    #     if self.args:
    #         cc_expr = [cc.eval_nested() for cc in self.args]
    #         cc_expr = ', '.join(cc_expr)
    #
    #         expr_str = f'{expr_str}, {cc_expr}'
    #
    #     return f'{expr_str}'

    def eval_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
        e.g. {+{Ifthe{True}{1}{2}}{3}}
        """
        # sfEh check if this still works as one-liner
        return f"{{{self.get_nclass()}{''.join([cc.eval_apted_notation() for cc in self.args])}}}"

    def get_max_depth(self, depth=0):
        """
        Go through all nodes, save depth
        """
        if len(self.args) == 0:
            return depth
        else:
            return max(cc.get_max_depth(depth=depth + 1) for cc in self.args)

    def repair_depth(self, depth=0):
        """
        aka set_depth recursively for all nodes in a branch
        mainly used in branch

        The depth is written inevery node (for whatever reason), and instead of having to propagate
        the depth through every crossover/branch mutation function, instead, we call it when replacing nodes
        """
        self.depth = depth
        for cc in self.args:
            cc.repair_depth(depth=depth + 1)

    def set_new_node(self, new_node: 'BaseTree'):
        """
        was: new_core
        """
        self.__class__ = new_node
        self.args = new_node.args or []  # maybe must be updated recursively
        #
        # self.repair_depth(self.depth)  # Especially required for crossover or branches

    def eval_mutable_nodes(self, xtype_out=None, allow_root=True):
        """
        return all nodes that are mutable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!
        """
        node_list = []
        if not self.is_fix:  # requirement for mutability
            # crossover requires excluding types that are not matching, and excludes the root node
            if (xtype_out is None) and (allow_root or not self.is_root()):
                node_list.append(self)

        for cc in self.args:
            node_list.extend(cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root))
        # deprecated:
        # node_list.extend(list(itertools.chain(
        #     *[cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root) for cc in self.args])))
        return node_list

    # sfeh:xxx
    # def evolve_mutate_filter_branch(self):
    #     """
    #     Recursively filter the nodes in the branch of fintree
    #     sfeh:   random filter all terminal nodes /
    #             single node /
    #             nodes in a branch /
    #             random nodes in a branch /
    #             intelligent filtering
    #     """
    #     # self.state = STATE_BUILDING  #  ==>state
    #         for cc in self.args:
    #             cc.evolve_mutate_filter_branch()
    #     else:
    #         self.mutate_self_filter(filter_type='gaussian_filter')

    # def finalize_set_nodepath(self, nodepath):
    #     """
    #     [0,2,1,0,0]
    #     ==>ROOT
    #     """
    #     self.nodepath = nodepath
    #     for ii, child in enumerate(self.args):
    #         nodepath_child = nodepath + [ii]
    #         child.finalize_set_nodepath(nodepath_child)

    def finalize_set_depth(self, depth=0, recursive=True):
        """
        depth=0 is the root node
        """
        self.depth = depth
        max_depth = depth
        if recursive:
            for cc in self.args:
                cc_depth = cc.finalize_set_depth(depth=depth + 1)
                max_depth = max(cc_depth, max_depth)

        return max_depth

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
        return all([cc.check_depth_infos(depth=depth + 1, fatal=fatal) for cc in self.args])

    def selfcheck(self, check_depth=True, fatal=None):
        """
        Tree Self-check for its structure
        """
        results = [self.check_depth_infos(fatal=fatal) if check_depth else 0]
        return sum(results)


class Operator(BaseTree):  # sfeh:xxx sympy.Function was here
    # is_Function = True

    def __new__(cls, *args, **kwargs):
        obj = BaseTree.__new__(cls, *args, **kwargs)
        return obj

    def __str__(self):
        _str = self.__class__.__name__
        if self.args:
            childstr = ', '.join([str(x) for x in self.args])
            _str = f"{_str}({childstr})"
        return _str

    def __repr__(self):
        _rpr = super().__repr__()
        childstr = ', '.join([repr(x) for x in self.args])
        _isfix = ', is_fix=True' if self.is_fix else ''
        _rpr = f"{_rpr}({childstr}{_isfix})"
        return _rpr

    def eval(self, *args, **kwargs):
        return self.insym  # eval(*args)


class ChainOperator(BaseTree):
    """
    sfeh:diskuss: Abstract class for the elements in chain operators?
    Add, Mult, Min, Max
    And, Or
    Actually not:
    """
    pass


class MathOperator(Operator):
    # is_real = True
    # is_Boolean = False
    pass


class logicOperator(Operator):
    # And, Or, Xor, Not
    # is_real = False
    # is_Boolean = True
    pass


class RelationalOperator(Operator):
    pass


class AngleOperator(Operator):
    pass


class MinMaxBase(Operator):  # sfeh chainableOperator
    pass


class NoSymCapitalized:
    pass


class TerminalNode(BaseTree):  # sfeh sympy.Atom
    """
    Terminal nodes are leaf nodes which can not have children. e.g.:
    - constants (e.g. 2.3)
    - observations (e.g. b, aka data input)
    - user-functions (sfeh:open)
    """
    nlabel = None

    def __new__(cls, nlabel, *args, **kwargs):
        obj = BaseTree.__new__(cls, *args, **kwargs)
        obj.nlabel = nlabel
        return obj

    def __str__(self):
        return f'{self.nlabel}'

    def __repr__(self):
        _rpr = super().__repr__()

        # this is an operator - there are definitely child-nodes
        _isfix = ', is_fix=True' if self.is_fix else ''
        _rpr = f"{_rpr}({self.nlabel}{_isfix})"
        return _rpr

    def _sympy_(self):
        return self.insym(self.nlabel)


class ConstantNode(TerminalNode):

    def mutate_self_filter(self, *args, **kwargs):
        pass


class Boolean(ConstantNode):
    xtype = (None, bool)
    insym = sympy.logic.boolalg.Boolean


class Float(ConstantNode):
    xtype = (None, bool)
    insym = sympy.Float

    def mutate_self_filter(self, filter_type='gaussian_filter', *args, **kwargs):  # sfeh:open
        """
        sfeh:open, gaussian_filter,
        sfeh: only one random, either numpy <- (or random)
        """
        if filter_type == 'gaussian_filter':
            if np.random.choice(['v1', 'v2']) == 'v1' or self.nlabel == 0:
                constant = self.nlabel + np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.nlabel, 0.1)  # sfeh better adjustments?
            self.nlabel = round(constant, PRECISION)  # sfeh:discussion be careful, might create zero sometimes
        pass


def observation_get_family_and_time(name, re_pattern='_\\d+$', none_return=None):
    """
    When an observation is known, return the family, the time and the SIGN!!
    sfeh:move this function somewhere else
    """
    core_expr = re.split(re_pattern, name)[0]
    if core_expr[0] == '-':
        core_expr = core_expr[1::]
        preexpr = '-'
    else:
        preexpr = ''
    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception as ex:
        temp_diff = none_return
    return core_expr, temp_diff, preexpr


class Symbol(TerminalNode):
    """
    sfeh:discuss:
        labels should not have a sign (-pos)
        y though? -> just use a '-'-operator in an additional node.
        also: When reconstructing trees, the sign can appear in observations
    This was used to deal with negative labels
        self.name = nlabel if nlabel[0] != '-' else nlabel[1:]
        sfeh:xxx option here for type float/bool
    """
    insym = lambda x: sympy.Symbol(x, real=True, imaginary=False)  # sfeh: idea making some things faster
    xtype = (tuple([]), float)

    def __new__(cls, nlabel, *args, **kwargs):
        obj = BaseTree.__new__(cls, *args, **kwargs)
        obj.nlabel = nlabel

        obj.fam, obj.time_index, _ = observation_get_family_and_time(nlabel, none_return=None)
        obj.index_minmax = None

        # obj = object.__new__(cls)
        return obj


# class NodeType:
#     xtype = None
#
#
# class FloatType(NodeType, float):
#     pass
#
#
# class BoolType(NodeType):
#     pass


class BaseType:
    pass


class ExprCondPair(ChainOperator):
    insym = sympy.functions.elementary.piecewise.ExprCondPair
    tflow = tf.where  # sfeh:open tf.cond https://stackoverflow.com/questions/45517940

    is_real = True


class Add(MathOperator):  # expr_sym = '({} + {})'

    insym = sympy.Add
    tflow = tf.add
    xtype = (tuple([float, float]), float)
    # sfeh:discuss in sympy, this is a "FiniteSet"? whats going on...
    is_real = True

    def backprop(self):
        """
        #:      count >, count <, count =
        #me:    wenn Abweichung am höchsten von allen sum-knoten
        c:      sum(alle Abweichungen) -> schlimmster +-child
        propagate-down: y - ^y, wenn 5 zu hoch -> -5 nach unten
        erst normieren? Also, die avg. Abweichung abziehen?
        """
        pass


class InverseFraction(Operator):
    # tflow = tf.pow  # sfeh:open
    xtype = (tuple([float, float]), float)

    def _sympy_(self):
        return sympy.Pow(self.args[0], -1)


class Pow(MathOperator):  # nlabel = 'Pow'  # expr_sym = '({} ** {})'  # **
    """ALERT: Power can create complex numbers, maybe you should use Powerounded"""
    insym = sympy.Pow
    tflow = tf.pow
    xtype = (tuple([float, float]), float)

    def backprop(self):
        """
        # KILL, wenn Lösung gar nicht erreicht werden kenn (wegen Definitionsbereich). Es ist dann halt einfach so.
        Idee: Ergebnis Differenz ist vermutlich manchmal sehr hoch.
            # Zähle, wie oft Basis & Exponent jeweils drüber/drunter liegen
            -> Nur den Exponenten beachten. Anzahl drüber/drunter im Vergleich zum besten Exponenten (der Ziel erreicht)
        """
        pass


class Abs(MathOperator):  # nlabel = 'Abs'  # expr_sym = 'Abs({})'
    insym = sympy.Abs
    tflow = tf.abs
    xtype = (tuple([float]), float)


class sign(MathOperator, NoSymCapitalized):  # nlabel = 'sign',  # expr_sym = 'sign({})'
    # does not work in string, but irrelevant. sympy.simplify('sign(-a)') -> -sign(a)
    insym = sympy.sign
    tflow = tf.sign
    xtype = (tuple([float]), float)


class log(MathOperator, NoSymCapitalized):  # nlabel = 'log'  # expr_sym = 'log({})'
    """sfeh: Log isactually Ln (base e)"""
    insym = sympy.log
    tflow = tf.math.log
    xtype = (tuple([float]), float)


class cos(AngleOperator, NoSymCapitalized):  # nlabel = 'cos' # expr_sym = 'cos({})'
    insym = sympy.cos
    tflow = tf.cos
    xtype = (tuple([float]), float)


class sin(AngleOperator, NoSymCapitalized):  # expr_sym = 'sin({})'
    insym = sympy.sin
    tflow = tf.sin
    xtype = (tuple([float]), float)


# sin = Sin  # sfeh:discuss: i think this wont work, cause __name__


class tan(AngleOperator, NoSymCapitalized):
    insym = sympy.tan
    tflow = tf.tan
    xtype = (tuple([float]), float)


class acos(AngleOperator, NoSymCapitalized):  # nlabel = 'acos'  # expr_sym = 'acos({})'
    insym = sympy.acos
    tflow = tf.acos
    xtype = (tuple([float]), float)


class asin(AngleOperator, NoSymCapitalized):  # nlabel = 'asin'
    insym = sympy.asin
    tflow = tf.asin
    xtype = (tuple([float]), float)


class atan(AngleOperator, NoSymCapitalized):  # nlabel = 'atan'
    insym = sympy.atan
    tflow = tf.atan
    xtype = (tuple([float]), float)


class tanh(AngleOperator, NoSymCapitalized):  # nlabel = 'tanh'
    insym = sympy.tanh
    tflow = tf.tanh
    xtype = (tuple([float]), float)


class sinh(AngleOperator, NoSymCapitalized):  # nlabel = 'sinh'
    insym = sympy.sinh
    tflow = tf.sinh  # sfeh sinh, asinh
    xtype = (tuple([float]), float)


class cosh(AngleOperator, NoSymCapitalized):  # nlabel = 'cosh'
    insym = sympy.cosh
    tflow = tf.cosh  # sfeh acosh came up...
    xtype = (tuple([float]), float)


class Xor(logicOperator, NoSymCapitalized):  # nlabel = 'Xor'
    insym = sympy.Xor
    tflow = tf.math.logical_xor
    xtype = (tuple([bool, bool]), bool)


class Not(logicOperator):  # nlabel = 'Not'  # expr_sym = '~({})'
    """
    Problem was:
    - Not(a) evaluates to ~a
    - not(a<2) evaluates to nan
    """
    insym = sympy.Not
    tflow = tf.logical_not
    xtype = (tuple([bool]), bool)


class Eq(logicOperator):  # nlabel = 'Eq'  # expr_sym = '({} == {})'
    insym = sympy.Eq
    tflow = tf.equal
    xtype = (tuple([float, float]), bool)


class Mul(MathOperator):  # nlabel = 'Mul'  # expr_sym = '({} * {})'
    """
    sfeh:reduceoperator
    """
    insym = sympy.Mul
    tflow = tf.multiply
    xtype = (tuple([float, float]), float)

    is_real = True

    def propagate_down(self):
        """
        Idee: Ein weiterer Faktor bestimmt, wie weit man am Ziel vorbei ist.
            Beispielsweise Ziel: 20, aber Ergebnis war 10 -> Lösungsfaktor ist 20/10=2.
            Nun gibt man den eigenen Wert*2 nach unten als Ziel zurück.
            Falls selbst=0, gib differenz zwischen Ziel und 0 nach unten (also: Ziel)

        wenn 0: immer selbst runterpropagieren, wenn anderer auch 0: beide 0.5
        defprop:
        # propagation:
        propagate-down error: (y-^Y)-
        propagate-down itsme: (y-^Y)-
        """
        pass


class And(logicOperator):  # nlabel = 'And'  # expr_sym = '({} & {})'
    insym = sympy.And
    tflow = tf.logical_and
    xtype = (tuple([bool, bool]), bool)


class Piecewise(ChainOperator):  # nlabel = 'Piecewise'
    # MapxPiecewise was here
    # sfeh:xxx all the function assumptions that sympy has
    # sfeh:xxx must have a True-case
    insym = sympy.Piecewise
    tflow = tf.where  # sfeh:open tf.cond https://stackoverflow.com/questions/45517940

    is_real = True

    def __add__(self, other):
        pass  # sfeh:idea? probable no reason for


class Min(MinMaxBase):  # nlabel = 'Min'  # expr_sym = 'Min({}, {})'
    insym = sympy.Min
    tflow = tf.minimum
    xtype = (tuple([float, float]), float)

    is_real = True


class Max(MinMaxBase):  # nlabel = 'Max'  # expr_sym = 'BinaryMax({}, {})'
    insym = sympy.Max
    tflow = tf.maximum
    xtype = (tuple([float, float]), float)


class Or(logicOperator):  # nlabel = 'Or'  # expr_sym = '({} | {})'
    insym = sympy.Or
    tflow = tf.logical_or
    xtype = (tuple([bool, bool]), bool)


class Ne(RelationalOperator):  # nlabel = 'Ne'  # expr_sym = '({} != {})'
    insym = sympy.Ne  # sympy.Unequality
    tflow = tf.not_equal
    xtype = (tuple([bool, bool]), bool)


class Lt(RelationalOperator):  # nlabel = 'Lt'  # expr_sym = '({} < {})'
    insym = sympy.Lt  # sympy.StrictLessThan
    tflow = tf.less
    xtype = (tuple([float, float]), bool)


class Le(RelationalOperator):  # nlabel = 'Le'  # expr_sym = '({} <= {})'
    insym = sympy.Le
    tflow = tf.less_equal
    xtype = (tuple([float, float]), bool)


class Gt(RelationalOperator):  # nlabel = 'Gt'  # expr_sym = '({} > {})'
    insym = sympy.Gt
    tflow = tf.greater
    xtype = (tuple([float, float]), bool)


class Ge(RelationalOperator):  # nlabel = 'Ge'  # expr_sym = '({} >= {})'
    insym = sympy.Ge
    tflow = tf.greater_equal
    xtype = (tuple([float, float]), bool)


# def sympy_symbol_defaults(name_list):
#     """
# sfeh:idea setting real=true is still a good idea
#     sfeh workaround.
#     sympy expressions like 'sign(((a * b) ** 151))' take forever.
#     ignoring complex numbers with this trick (use this as locals)
#
#     'sym_reduce': '({} ** {})'
#     'sym_reduce': 'sign(re({}))'
#     """
#     return {str(x): sympy.symbols(str(x), real=True, imaginary=False) for x in name_list}


# attention: exactly same capitals/letters! (gets replaced)
# sfeh:xxx potential names
# Folding, collective, reduce, chained, fold, map


def expr_sympify(expr):
    """
    Returns a simplified expression using sympify.
    - sympify the expression
    - If sympify evaluates to one of these errors: 'zoo', 'inf', '*I', 'nan', stop evaluation

    Sympify is a python core module which reduced mathematical expressions.
    Example: sympify('a+a+a+a') -> a*4
    Note that the sympify was extended in plagih_sympify_extras.py with extra functions

    Sympify fails: The results are, or contain, expressions that should/can not be evaluated
    'zoo': (Complex infinity) E.g. when an int-number is divided by zero
    'inf': (Regular infinity) E.g. when a float-number is divided by zero (...i know, why are there two infinities?)
    '*I': (Complex number) E.g. when putting a number to the power of negative fractals, 1**(-0.5)
    'nan': (Not a number) when Evaluation fails, E.g. types contradict, expression is empty, 'BinaryMin(a, zoo' ...

    Sympy bug #1:
    It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
    Or this issue: https://github.com/sympy/sympy/issues/17785

    Sympy bug #2:
    print(plagih_sympify('a<zoo'))
    throws an exception.
    -> Try-except block for this case

    Lastly, it is recommended that you not use I, E, S, N, C, O, or Q

    sfeh: more sympy bugs
    sympify option evaluate=None does not work with custom functions
    """

    # loadable_ops_dict.update(eval_locals or {})  # sfeh:delete? irrelevant, cause every class defines the eval method?

    try:
        expr_sym = sympy.sympify(expr)
        # try:  # sfeh:xxx
        #     # expr_sym2 = # expr_sym.factor()
        #     if # expr_sym != # expr_sym2:
        #         print(f'COMPARE FACTOR:\n{# expr_sym}  len {len(# expr_sym)}\n{# expr_sym2}  len {len(# expr_sym2)}')
        # except Exception as ex:
        #     print('fdsfdsahds')
        if expr_sym.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I):
            raise ArithmeticError(f'Simplification failed for expression: {expr_sym}')
        return expr_sym

    except ValueError as ex:
        # return 'nan'  # 'nan' always evaluates to nan. ALl nan bugs should be solved.
        raise ValueError(f'NaN in {ex}')
    except AttributeError as ex:
        # print(f'sfeh: This sympy bug happens, when sympifying "True": {ex}')
        return sympy.true if expr else sympy.false
    # except Exception as ex:
    #     raise Exception(f'sympify_1: {expr} reason: ({ex})')


# sympy_constants = {
#     sympy.numbers.Zero: 0,
#     sympy.numbers.Half: 0.5,
#     sympy.numbers.One: 1,
#     sympy.numbers.NegativeOne: -1,
#     sympy.numbers.Exp1: 2.71828182845904,  # sympy.numbers.Exp1().evalf(16)
#     sympy.numbers.Pi: 3.1415926535897932,  # sympy.numbers.Pi().evalf(17)
#     sympy.numbers.GoldenRatio: 1.61803398874989,  # sympy.numbers.GoldenRatio().evalf(16)
#     sympy.numbers.TribonacciConstant: 1.83928675521416,  # sympy.numbers.TribonacciConstant().evalf(16)
#     sympy.numbers.EulerGamma: 0.577215664901532,  # sympy.numbers.EulerGamma().evalf(16)
# }
# sfeh
#  sympy.numbers.Infinity: tensorflow.constant(np.Infinity),
#  sympy.numbers.NegativeInfinity: tensorflow.constant(-np.Infinity),
#  sympy.numbers.ComplexInfinity: tensorflow.complex(0, np.Infinity),sympy.numbers.ImaginaryUnit,
#  sympy.numbers.NumberSymbol
#  sympy.numbers.Catalan: tensorflow.constant(sympy.numbers.Catalan),
#  sympy.numbers.NaN: tensorflow.constant(sympy.numbers.NaN),


def labels_from_nestedexpr(labels_nested_list, result_accum):
    """
    Returns a label list from the nested list which ast_expr_to() created
    [+, [a], [/, [b, c]]]]  -> [+, a, /, b, c]
    """

    for x in labels_nested_list:  # all elements, that are not lists themselves
        if type(x) is not list:
            x = str(x)  # labels must be string!
            result_accum.append(x)

    only_lists = [x for x in labels_nested_list if (type(x) == list)]
    if only_lists:
        from itertools import chain
        lists_removed = list(chain(*only_lists))
        result_accum = labels_from_nestedexpr(lists_removed, result_accum)

    return result_accum


totf = {
    # sympy.Symbol: 'tf': lambda x: tf.cons
    sympy.Min: tf.minimum,
    sympy.Max: tf.maximum,
    sympy.Add: tf.add,
    sympy.Mul: tf.multiply,
    sympy.Pow: tf.pow,
    sympy.Abs: tf.abs,

    sympy.Not: tf.logical_not,
    sympy.And: tf.logical_and,
    sympy.Or: tf.logical_or,
    sympy.Xor: tf.math.logical_xor,

    sympy.Equality: tf.equal,
    sympy.Unequality: tf.not_equal,
    sympy.GreaterThan: tf.greater_equal,
    sympy.StrictGreaterThan: tf.greater,
    sympy.LessThan: tf.less_equal,
    sympy.StrictLessThan: tf.less,

    sympy.N: tf.math.round,

    sympy.log: tf.math.log,
    sympy.cos: tf.cos,
    sympy.sin: tf.sin,
    sympy.tan: tf.tan,
    sympy.acos: tf.acos,
    sympy.asin: tf.asin,
    sympy.atan: tf.atan,
    sympy.tanh: tf.tanh,

    sympy.sign: tf.sign,

    sympy.ITE: tf.where,  # sfeh:test this
    sympy.re: lambda x: tf.convert_to_tensor(x, dtype=tf.dtypes.float32),  # gotcha, comes up rndomly
}


class CustomSympyFunction:
    pass


class Square(MathOperator, CustomSympyFunction):  # nlabel = 'Square'  # expr_sym = 'Square({})'
    tflow = tf.square
    xtype = (tuple([float]), float)

    is_real = True

    @classmethod
    def eval(cls, a):
        return sympy.Pow(a, 2)


class Sub(MathOperator, CustomSympyFunction):
    # sfeh:discuss remove for add? Sub is subclass of add?  # nlabel = 'Sub', expr_sym = '({}-{})'
    tflow = tf.subtract
    xtype = (tuple([float, float]), float)
    insym = lambda a, b: sympy.Add(a, -b)

    @classmethod
    def eval(cls, a, b):
        return a - b  # sfeh check if its sympied


class Ifte(Operator, CustomSympyFunction):  # nlabel = 'Ifte'  # expr_sym = 'Ifte({}, {}, {})'
    """
    self-expert: opportunity_cost = best_vals - chosen_vals
    self-childs:
    condition: opportunity_cost = chosen_vals - best_vals
    a        : opportunity_cost = when_chosen(chosen_vals - best)
    b        : opportunity_cost = when_chosen(chosen_vals - best)
    #self.args:
    condition: #correct_decisions
    a        : cond->a, #correct_decisions - #false_decisions
    b        : cond->b, #correct_decisions - #false_decisions
    --> If a node
    """
    tflow = tf.where
    xtype = (tuple([bool, float, float]), float)
    insym = sympy.Piecewise
    is_real = True

    # @classmethod
    # def eval(cls, a, b, c):
    #     # if a.is_Boolean:
    #     #     return b if a else c  # search for 'gotcha' in https://docs.sympy.org/latest/_modules/sympy/core/relational.html
    #     # else:
    #     #     return None  # returns None, is not further evaluated
    #     return sympy.Piecewise((b, a), (c, True))  # also available: sympy.piecewise_fold

    def _sympy_(self):
        a = bool(self.args[0])  # sympy.logic.boolalg.Boolean(True) not working, sfeh: piecewise quecks the datatype
        b = self.args[1]
        c = self.args[2]
        return sympy.Piecewise((b, a), (c, sympy.S.true))


class Round(MathOperator, CustomSympyFunction):  # nlabel = 'Round'
    """
    sfeh:discussion this does only round to full numbers
    """

    tflow = lambda a: tf.math.round(a, 1)
    xtype = (tuple([float]), float)

    is_real = True

    @classmethod
    def eval(cls, a):
        return sympy.N(a, 0)


class Log1p(MathOperator, CustomSympyFunction):
    nlabel = 'log1p'
    tflow = tf.math.log1p
    # expr_sym = 'log1p({})'
    xtype = (tuple([float]), float)

    @classmethod
    def eval(cls, a):
        return sympy.log(a + 1)  # just if no eval is implemented


class Div(MathOperator, CustomSympyFunction):  # nlabel = 'Div'  # # expr_sym = '({} / {})'
    """sfeh:xxx make this available, make a correct version of "Divide_no_nan" """
    tflow = tf.math.divide
    xtype = (tuple([float, float]), float)

    @classmethod
    def eval(cls, a, b):
        return a / b  # is a sympy thing returned?

    def backprop(self):
        """
        Wie bei mult: Ziel wird ausgerechnet, Faktor für sich selbst wird ausgerechnet. Nach unten propagieren.
        """
        pass


class Sqrt(MathOperator, CustomSympyFunction):
    # sfeh:xxx sqrt is actually not a simplified Version
    # sfeh sympy.sqrt is a function, not a class
    insym = sympy.sqrt
    tflow = tf.sqrt
    xtype = (tuple([float]), float)


# class Divide_no_nan(Operator):
#     """
#     # Division: SAFE division by zero!
#     -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented
#     sfeh: is it okay to display this as '/'?
#     xxxx optional to use high value instead of tf-eval to 1
#     """
#     nlabel = 'Divide_no_nan'
#     # classname = 'Divide_no_nan'  # sfeh??
#     tflow = tf.math.divide_no_nan
#     # expr_sym = 'Div_no_nan({}, {})'
#     insym = None
#     xtype = (tuple([float, float]), float)
#     @classmethod
#     def eval(self, a, b):
#         return a / b


# class Usub(Operator, sympy.Function):
#     """
#     sfeh:idea introduce negative labels as input?
#     """
#     nlabel = 'Usub'
#     tflow = tf.negative
#     # expr_sym = '(-{})'
#     insym = None
#     xtype = (tuple([float]), float)
#
#     is_real = True
#
#     @classmethod
#     def eval(cls, a):
#         return -a  # sfeh
#
#     def _sympy_(self, *args, **kwargs):
#         return self.eval(*args, **kwargs)


# class Powrounded(Operator):
#     """
#     ALERT: This Power Version does not round the results
#     inline-available
#     sfeh:xxx not yet used, make this available and rewrite Power above
#     """
#     nlabel = 'Powrounded'
#     tflow = tf.pow
#     # expr_sym = '({}**Round({}))'
#     insym = None
#     xtype = (tuple([float, float]), float)
#
#     @classmethod
#     def eval(cls, a, b):
#         return sympy.Pow(a, sympy.N(b, 0))


# print(Max(3, 4), sympy.Max(3, 4))

# x = sympy.sympify('Max(a, 4)', locals={'Max': Max.__base__, 'Integer': Integer.__base__})
# x = sympy.sympify('Max(a, 4) + b ** 2')


# class PowRounded(Operator):
# sfeh open
#     nlabel = 'Pown'
#     tflow = tf.pow
# #     # expr_sym = 'BinaryMax({}, {})'
# #     xtype = (tuple([float, float]), float)

# loadable_ops_dict = {'BinaryAdd': Add, 'BinaryMultiply': Mul, 'Sub': Sub,
#                      # 'Divide_no_nan': Divide_no_nan, 'Usub': Usub,
#                      'Div': Div, 'Pow': Pow, 'Abs': Abs, 'sign': sign, 'Square': Square,
#                      'sqrt': sqrt, 'log': log, 'log1p': Log1p, 'cos': cos, 'sin': sin, 'tan': tan, 'acos': acos,
#                      'asin': asin, 'atan': atan, 'tanh': tanh, 'sinh': sinh, 'cosh': cosh, 'Xor': Xor,
#                      'BinaryNot': Not,
#                      'Ne': Ne, 'Lt': Lt, 'Le': Le, 'Gt': Gt, 'Ge': Ge, 'Ifte': Ifte,
#                      'BinaryMin': Min, 'Max': Max, 'BinaryAnd': And, 'Or': Or,
#                      'Round': Round}
# # _LOADABLE_LATER = { 'Multiply': Multiply, 'And': And, 'Or': Or, 'Eq': Eq, 'Max': Max, 'Min': Min}
#
# loadable_inline_operator_dict = {'+': Add, '-': Sub, '*': Mul, '/': Div, '**': Pow,
#                                  '==': Eq, '!=': Ne, '<': Lt, '<=': Le, '>': Gt, '>=': Ge,
#                                  '&': And, '|': Or}
# loadable_ops_dict.update(loadable_inline_operator_dict)


def sympy_to_tensorflow(expr, pandas_df):
    """
    - check terminal-node
    -- check symbol
    --

    Bugs/gotchas:

    sympify('True')             -> True
    sympify('1')==True          ->
    sympify('~(True)')          -> -2
    sympify('~(False)')         -> -1
    sympify('a <= Min(a, b)')   ->

    sympy.logic.boolalg.ITE has only boolean inputs
    evaluate=None/false

    sympy.logic.boolalg.ITE is not If-then-else
    sympy.cosh can emerge out of sin-stuff
    sympy.sympify('True')->True is no sympy expression anymore
    # sfeh:bug gotcha sympy.re comes up randomly
    """
    # shape = tensors[list(tensors.keys())[0]].get_shape()  # sfeh:open:workaround:

    # ==Bug-handling==  sympy.sympify('True')->True is no sympy expression anymore
    if isinstance(expr, bool):  # e.g. '1'
        return tf.constant(expr, dtype=tf.dtypes.bool)
    if isinstance(expr, (bool, sympy.logic.boolalg.BooleanAtom)):
        expr = True if isinstance(expr, sympy.logic.boolalg.BooleanTrue) else False  # sfeh:collect sympy bug gotcha bug
        return tf.constant(expr, dtype=tf.dtypes.bool)

    # the following lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    # ==Terminal nodes==
    elif expr.is_Atom:
        if expr.is_Symbol:
            result = pandas_df[expr.name]  # sfeh:discuss placeholder
            return result

        else:
            expr_eval = expr.evalf()  # standard 15 digits
            if expr.is_Boolean:
                return tf.constant(expr_eval, dtype=tf.dtypes.bool)
            else:  # float
                return tf.constant(float(expr_eval), dtype=tf.dtypes.float32)

    else:  # Operator # len(expr.args) > 0:  # sfeh: line can be removed or replaced
        if isinstance(expr, sympy.Piecewise):
            _revlist = list(expr.args[::-1])  # tuples to list, reverse: last tuple must be nested the deepest
            _revlist = [[sympy_to_tensorflow(xx, pandas_df) for xx in list(_ls)] for _ls in _revlist]
            otherwise = _revlist[0][0]  # the last "True" condition
            for x in _revlist[1:]:
                otherwise = tf.where(x[1], x[0], otherwise)
            return otherwise
        try:
            tf_fun = totf[type(expr)]
        except KeyError:
            try:
                tf_fun = type(expr).tflow  # sfeh
            except Exception as ex:
                # print('aaa', type(expr), expr, type(expr) in sympy_to_node)
                # ignore:
                # -> sympy.conjugate
                tf_fun = expr.tflow  # sfeh:delete? delete case above? ### binarymax,

        tf_args = [sympy_to_tensorflow(x, pandas_df) for x in expr.args]
        try:
            result = tf_fun(*tf_args)  # fits, if the arguments match the expected arguments exactly Add(a, b)
        except TypeError:
            result = tf_args.pop()  # only commutative arity-2 functions here (Add, Mul, Max, Min)
            while tf_args:
                # sfeh:optimization
                result = tf_fun(result, tf_args.pop())
        return result


if __name__ == '__main__':

    ns = {
        'a': sympy.Symbol('a', real=True),
        'b': sympy.Symbol('b', real=True),
        'c': sympy.Symbol('c', bool=True),
        'd': sympy.Symbol('d', bool=True),
    }

    tensors = {
        'a': tf.constant([1.0, 2, 3, 4, 5, 6], dtype=tf.dtypes.float32),
        'b': tf.constant([-1.0, -2, -3, -4, -5, -6], dtype=tf.float32),
        'c': tf.constant([True, False, True, False, True, False], dtype=tf.dtypes.bool),
        'd': tf.constant([True, True, True, True, True, True], dtype=tf.dtypes.bool)
    }
    tst = [
        '5', '1', '0', '0.5', '-1', 'True', 'False',
        'c & True', 'c | False', '~c',
        'a<1', 'a<b', 'a<=b', 'a>=b', 'a>b', 'a==b', 'a!=b', 'a',
        # 'oo', 'zoo', 'I',
        'a + 1', 'a + 2', 'a*2', 'a - 2', 'a/2', 'a < 2', 'a**2', '2/a',
        'a*b*2', 'a+b+a+2+4', 'Min(a, b, 3)', 'Max(a, b, 4, a**2, a+b)', 'a<3',
        'Piecewise((a, c), (b, d), (a+b, True))',
        'Eq(4, 4.0)',
        # 'Square((BinaryMin(-2.176629, b) - Abs(a)))', 'Round(-123.333334234) + Round(b)',
        # 'Ifte(c, 1, 2)', '1 < BinaryMax(2, Ifte(1 < a, 1, 1))', 'BinaryMax(a+1, 2**(5-b))'
    ]
    tst_custom = [
        'Ifte(a, b, c)',
        '(((0.326675 * b_2) - c_9) + (Ifte((-c_9 < b_5), c_7, Ifte((Square(Gain_6) < BinaryMax(a_2, Ifte((c_9 < '
        'c_4), -Gain_3, Gain_5))), c_9, c_4))))',
        'BinaryMax(a, 2)',
        'BinaryMin(b, b)',
        'Ifte(True, a, 3)',
        'sin(asin(b))',
        'And(False, True)',
        'BinaryAdd(-1.490149, 14.0)',
        'Ifte(False, (3+a), 3)',
        'Eq(4, 4.0)',
        'Lt(a, a)',
        'BinaryOr(Ne(False, False), False)',
        'sqrt(5 * a)',
        'BinaryMultiply(log(acos(-0.212976)), asin(2))',
        'BinaryNot(False)',
        'acos(0.5)',
        'Round(1.2345)',
        'BinaryMin(Ifte(True, BinaryMultiply(a, 20.0), acos(-0.5)), 1)',
        'Div(BinaryAdd(13.159398, 19.284178), 1)',
        'Pow(a, b)',
        'BinaryAdd(-2, BinaryMin(Ifte(True, 1, b), 8))',
        '(BinaryOr(True, True) & c)',
        'Ifte(BinaryAnd(False, True), b, 0.046948)',
        'Ifte(Ne(True, Lt(Sub(a, 2), 1)), 1, 2)',
        'cos(tan(Square(BinaryMultiply(BinaryAdd(Round(Ifte(Ne(Ge(b, 15), True), 7, Sub(a, 16.5))), 5), 4))))'
    ]
    xxx_problems = ['Ifte(Lt(Ifte(Eq(Min(b, 1), 3), Max(a, b), b), 0), 0, 2)']


    def test_basic_tfconversion():
        # sfeh:open tests

        for t in tst:
            x = sympy.sympify(t, locals=ns)
            x = sympy_to_tensorflow(x, tensors)

            print(f'{t} \t{x}')


    def test_sympify():
        print('Running sympify example')

        # expr = '-b_0*sign(re(asdW**2)) - 0.004073'
        # expr = 'BinaryMin(-1 - 1 + sqrt(1)'
        # expr = 'BinaryMax(2.202197, (Abs(b) - sqrt(b)))'
        # expr = '(vel + vel)'
        # expr = 'a - 0.4375'

        # obs = ['a', 'b']
        # symloc = {x: sympy.symbols(x, real=True, imaginary=False) for x in obs}
        # sympy_symbol_dict = {'a': sympy.symbols('a', real=True, imaginary=False),
        #                      'b': sympy.symbols('b', real=True, imaginary=False)}
        # sympify('sign(((a * b) ** 151))', symloc)

        # obs = {'b': 0.5, 'a': -0.8}
        # sympex = plagih_sympify(expr, eval_locals=obs)
        for x in tst + tst_custom:
            sx = expr_sympify(x)
            print(sx)


    def test_this():
        x = expr_sympify(
            '(((0.326675 * b_2) - c_9) + (Ifte((-c_9 < b_5), c_7, Ifte((Square(Gain_6) < BinaryMax(a_2, Ifte((c_9 < c_4), -Gain_3, Gain_5))), c_9, c_4))))')
        print(x)


    def print_relevant_subclasses():

        l = [x.get_nclass() for x in get_subclasses(Operator)]
        print(f'loadable_strings = {l}')
        c = [x.__name__ for x in get_subclasses(Operator)]
        print(f'operator_classes = {c}')
        d = dict(zip(l, c))
        d = ', '.join([f"'{k}': {v}" for k, v in d.items()])
        print(f'loadable_ops_dict = {{{d}}}')
        # sfeh: matches sympy/tenorflow
        st = {}
        for x in get_subclasses(Operator):
            try:
                st[f'sympy.{x.insym.__name__}'] = x.__name__  # x.tflow.__name__
            except AttributeError as ex:
                print(f'Could not get {x}: {ex}')
                st[x.__name__] = x.__name__
        st = ', '.join([f"{k}: {v}" for k, v in st.items()])
        print(f'sympy_to_node = {{{st}}}')


    # # test_this()
    # test_basic_tfconversion()  # sfeh all tests
    # test_sympify()
    # # x = expr_sympify('Ifte(True, 1, 2)')
    # # print(x)

    # print(list(get_subclasses(Operator)))

    # sfeh:xxx why are node classes all in memory, is that bad? use__neew__()?

    # x = Add(childs=[Symbol('a'), Mul(childs=[2, 3])])
    n1 = Float(1.23)
    n2 = Symbol('b')
    n3 = Boolean('True', is_fix=True)
    n4 = Add(n1, n2)
    print(n1, n2, n3, n4)
    tr = Add(n1, n2)
    tr = Ifte(Boolean(True, is_fix=True), Mul(sin(Add(n1, n1)), n2), n1, is_fix=True)
    xx = Add()
    xx.args = [n1, n2]
    tr = Pow(Symbol('a'))
    tr = sin(Symbol('a'))
    print(tr)
    tr = sin(sin(Symbol('a')))


    def get_ev_childs(node):
        _childs = []
        for ii, cc in enumerate(node.args):
            _childs.append(lambda x: node.set_child_n(ii, x))
        _childs.extend(list(itertools.chain(*[get_ev_childs(cc) for cc in node.args])))
        return _childs


    # lel = get_ev_childs(tr)
    # print(lel)
    # lul = random.choice(lel)
    # lul(Float(5))
    # print(tr)

    print(tr)

    # for _subc in get_subclasses(BaseTree):
    #     if  in _subc.__bases__:
    #         # print(f'ignoring {_subc}')
    #         pass
    #     else:
    #         print(f'{_subc.__name__}')
    #
    # print(type(RelationalOperator))
