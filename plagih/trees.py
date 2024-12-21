"""
The main class of a gp run. It holds the following functionalities
- run information (config, evolution-specifications for the loop, success monitoring])
- population (pop_base, pop_next)
-
"""
# import os
# os.environ["KMP_WARNINGS"] = "FALSE"
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()  # sfeh possibly faster with disable
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # https://github.com/tensorflow/tensorflow/issues/27023
# import tensorflow as tf  # noqa check if ignoring warnings still required (tensorflow sends endless warnings)

from typing import Callable, Union, Optional
from plagih.util import *
import copy
import random
from dataclasses import dataclass
from plagih.tree_complexity.tree_edit_distance import apted_distance
import numpy as np
import pandas as pd
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair

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
    otherwise: '1 < Max(2, Ifte(1 < a, 1, 1))' will crash. (< operators only work on non-complex - aka real numbers)
    check for is_number if required.

The following line is an honorable mention for myself; it was required for
## if ((a == True or a == False) and (b == True or b == False)) == (sympify(a).is_Boolean and sympify(b).is_Boolean):

Useful information:
- These variables are set for every sympy object and thus can be tested, e.g. a.is_Boolean
    # To be overridden with True in the appropriate subclasses

Custom Operators /Functions/Nodes/Terminals/Nested:
    Any custom node must have a typus subclassing NodeBase
    Also, make a case in sympy_to_nested to reconstruct trees from sympy expressions.
"""

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


@dataclass
class NodeStructure:
    """
    Recursively holds the nodes of a tree
    parent: pointer to parent node, 'None' implies root-node
        - parent-link added for pseudo-backprop
    """
    childs: list
    is_fix: bool
    depth: int
    root_node: Optional['Node']
    parent_node: Optional['Node']
    depth: Optional[int]

    def __init__(self, *args: iter, depth=None, is_fix=None, set_class=None):

        self.childs = list(args)  # ...usually a list, (??) but can also be 'None' (<??)
        self.is_fix = is_fix  # only for trees with a "fixed" root structure
        self.depth = depth  # requires repair after changes
        self.root_node = None  # pointer to parent, requires repair after changes
        self.parent_node = None  # pointer to parent, requires repair after changes
        self.depth = None  # pointer to the one root-node, requires repair after changes

    # def represent_str(self, *args):
    #     # sfeh: not here?
    #     pass

    # def set_typus(self, t):  # : Type[Typus]
    #     """all other values are automatically set by assigning the respected node"""
    #     self.typus = t

    def set_parent(self, n):
        self.parent_node = n

    def set_root(self, n: 'Node'):
        self.root_node = n

    def get_childs(self):
        return self.childs  # sfeh:open

    def set_childs(self, child_list: (list, tuple)):
        if isinstance(child_list, (list, tuple)):
            ccs = [cast_input(x) for x in child_list]
            self.childs = ccs
            if self.has_childs():
                for cc in self.get_childs():
                    cc.set_parent(self)  # set pointer in child-nodes
                    cc.set_root(self)
        else:
            raise TypeError(f'childs must be set as list, not {type(child_list)}: {child_list}')

    def get_mutable_rootnodes(self, extend_lvls=2) -> Optional[list['Node']]:
        """Returns the list of first mutable nodes
        last_leaves: if you want so save all leave nodes aswell
        sum_layers=False, get_closest=True, return_all_layers=False
        sfeh: option
        """
        n = []
        if not self.is_fix:
            n = [self]
            extend_lvls -= 1

        if extend_lvls >= 0:
            if self.has_childs():
                for cc in self.get_childs():
                    n.extend(cc.get_mutable_rootnodes(extend_lvls=extend_lvls))

        return n

    def get_max_depth(self, depth=0):
        """Go through all nodes, save depth
        sfeh: this computes the depth and does not take advantage of saved depths"""
        if len(self.childs) <= 1:
            return depth
        else:
            return max(cc.get_max_depth(depth=depth + 1) for cc in self.get_childs())

    def is_operator_arity(self):
        """Check, if node is an operator with a fixed arity
        Nodes that are notchained-nodes"""
        return issubclass(type(self), OperatorArity)  # sfeh check? Terminal?

    def is_operator(self):
        return issubclass(type(self), BaseOperator)

    def is_term_and_symbol(self):
        if self.is_term():
            a = issubclass(type(self), Symbol)
            if a:
                return True
        return False

    def is_term(self):
        # sfeh:discuss is_atom, rename all to atom?
        return issubclass(self.get_typus(), Terminal)

    def get_typus(self):
        t = self.__class__
        return t

    def has_childs(self):
        # better to check for recursive use, as e.g. ExprCondPair is not a regular operator
        return not self.is_term()

    def repair_depth(self, depth=0):
        """
        aka set_depth recursively for all nodes in a branch
        mainly used in branch
        The depth is written inevery node (for whatever reason), and instead of having to propagate
        the depth through every crossover/branch mutation function, instead, we call it when replacing nodes
        """
        depth = depth or 0  # sfeh: "None" was set as depth somewhere. Could not find it.
        self.depth = depth
        if self.has_childs():
            for cc in self.get_childs():
                cc.repair_depth(depth=depth + 1)

        return

    # sfeh:Do links to parents lead to problems when crossover/etc happens?

    def repair_backlink(self, parent: 'Node', root: 'Node'):
        """backlink was introduced on 23.04.2024,
        linking the root and parent nodes"""
        self.root_node = root
        self.parent_node = parent
        for cc in self.get_childs():
            cc.repair_backlink(self, root)
        # self.repair_depth()  # sfeh:random powrounded replace is always when fitting? should maybe not?


class Node(NodeStructure):
    """
    was Typus, was Label
    The expr-content of a node
        - most functions in here are not tested, but also never required
        - str() function is used only for showing infos in debugger
    """
    symfun: Callable[..., sympy.Basic]
    np_fun: Callable[..., np.ndarray]
    showme: str
    sy_str: str  # 'sympy-fun'
    formulae_str: str  # 'sympy-expr'
    repr_str: str  # 'repr-format-{}-options'
    xtype: tuple
    xtype_chain: Union[bool, float]

    def __init__(self, *args: iter):
        super().__init__(*args)

    @staticmethod
    def np_hit(self, *args):
        return self.np_fun(list(args))

    def eval_np_debug(self, df):
        if self.is_term():
            my_np = self.eval_np()
            res = my_np(df)
        elif self.is_operator():
            child_values = [ccl.eval_np_debug(df) for ccl in self.get_childs()]
            my_np = self.eval_np(*child_values)
            res = my_np(df)
        else:
            raise NotImplementedError

        return res

    # def eval_np(self, *args):
    #     """
    #     Evaluate the node using NumPy.
    #     Recursively applies `eval_np` on all children and uses `self.np_fun` for evaluation.
    #     """
    #     # Generate lambdas for children
    #     child_lambdas = [c.eval_np(*args) for c in self.childs]
    #
    #     # Return a lambda that applies the numpy function to evaluated children
    #     def node_lambda(df):
    #         # Evaluate child lambdas with the DataFrame
    #         child_values = [child_lambda(df) for child_lambda in child_lambdas]
    #         # Apply the numpy function
    #         return self.np_fun(*child_values)
    #
    #     return node_lambda
    #
    # # @classmethod
    # def eval_np(self, *args):
    #     cc = [lambda df: c.eval_np(*args)(df) for c in self.get_childs()]
    #     fun = lambda df: type(self).np_fun(*cc)(df)
    #     return fun

    def set_new_node(self, nd_new: 'Node'):
        """Replacing oneself with another node"""
        # self_copy = copy.deepcopy(self)
        self.__class__ = nd_new.__class__
        self.__dict__.update(nd_new.__dict__)
        # self.xtype_chain = nd_new.xtype_chain
        # self.childs = nd_new.childs
        # self.formulae_str = nd_new.formulae_str
        # self.repr_str = nd_new.repr_str
        # self.showme = nd_new.showme
        # self.depth = sfeh?
        self.parent_node = None
        self.root_node = None
        self.depth = None  # todo?
        # sfeh all BaseNodeStructure infos should not be updated

        # self.set_typus(nd_new.typus)  # sfeh remove childs, is_fix...
        # self.set_childs(nd_new.childs)  # sfeh maybe must be updated recursively
        # self.repair_depth(depth=self.depth)  # Especially required for crossover or branchesnd_new
        # # sfeh: depth is repaired at the end, as some bug leads to wrong depths somewhere (depth=None)
        # # sfeh check fixed or if type matches?
        pass

    def replace_with(self, typus, childs):
        new_node = typus(*childs)
        self.set_new_node(new_node)
        pass

    def get_sympy_expr(self, simplimore=False) -> sympy.Basic:
        """Converts directly into a sympy expr
        from node-class, go to a sympy expression
        """

        _sym = type(self).symfun
        _cs = self.get_childs()

        if self.is_term():
            _r = _sym(_cs)

        elif isinstance(self, Piecewise):  # sfeh:open delete ONE of them?
            # _sym = sympy.Piecewise
            _cs = [(cc.get_childs()[0], cc.get_childs()[1]) for cc in _cs]
            _cs = [(cc[0].get_sympy_expr(simplimore=simplimore), cc[1].get_sympy_expr(simplimore=simplimore)) for cc in _cs]
            _cs = [ExprCondPair(cc[0], cc[1]) for cc in _cs]
            _r = _sym(*_cs)
        elif self.is_operator():
            _cs = [cc.get_sympy_expr(simplimore=simplimore) for cc in _cs]

            try:
                _r = _sym(_cs)  # noqa (_sym is definitely assigned)
            except (ValueError, TypeError) as ex:
                raise SympySimplificationError(f'getsympyexpr-err| {ex}')
        else:
            raise NotImplementedError

        sympy_expression_check_raise(_r)

        # _r_td = sympy.simplify(_r)
        # _r2 = sympy.S(_r)
        # _r3 = sympy.S(_r_td)
        # if _r != _r_td:
        #     # this may make the expression bigger??
        #     # _r2 = 64.0*cartPos**2*(cartPos + Abs(cartVel) + 1.15)
        #     # _r3 = cartPos**2*(64.0*cartPos + 64.0*Abs(cartVel) + 73.5)
        #     print(f'SFEH:XXX! {_r_td}')

        return _r

    def list_terminal_nodes(self):
        base = self.list_mutable_nodes()
        base = [x for x in base if x.is_term()]
        return base

    def force_input_node(self, tb):
        """Some trees have only constants as terminals.
        This function replaces a terminal node with an input, if the tree only has constants"""
        node_list = self.list_terminal_nodes()
        a = [x.is_term_and_symbol() for x in node_list]
        if any(a):
            return
        else:
            node_list = [x for x in node_list if type(x) == Number]  # sfeh only replaces numbers. ok for now.
            # sfeh: do this in create new tree?
            try:
                node = rnd_choice(node_list)  # debug if ignores chains
            except ValueError as ex:
                raise ValueError(f'Single-node tree with non-matching xtype: {self}')


            xtype = xt_self(node.get_xtype_tuple())
            new_node = tb.node_selector.choose_symbol_node(xtype)
            # node.set_new_node(new_node)
            node.set_new_node(new_node, debug=False)

    def is_chain(self):
        """
        is the node "in chain mode"?
        -> when there are more childs than input-xtypes
        ...if there are less. its weird, maybe node in construction?"""
        a = issubclass(type(self), OperatorChained)
        # b = len(self.get_xtype_childs()) < len(self.get_childs())
        # if a != b:
        #     raise NotImplementedError  # sfeh only debug
        # if a or b:
        #     return True
        # else:
        #     return False
        return a

    def is_number(self):
        """sfeh's check"""
        x = issubclass(type(self), Number)
        return x

    # def repr_as_list(self):
    #     typus_str = self.typus.__name__  # Node-name (Mul, Symbol)
    #
    #     if self.is_term():
    #         try:
    #             typus_str = self.childs[0].evalf()
    #         except AttributeError as ex:
    #             typus_str = self.childs[0]  # AttributeError("'bool' object has no attribute 'evalf'")
    #     else:
    #         childstr = ', '.join([cc.repr_as_list() for cc in self.get_childs()])
    #         typus_str = f'{typus_str}, {childstr}'
    #
    #     return f"[{typus_str}]"
        

    def get_ma_name_sfeh(self):
        s = self.showme
        return s

    def str_as_list(self, cut_terms=False):
        # typus_str = self.typus.__name__  # sfeh: can str(typus) work? -> str with args recursively?
        # sfeh:delete_me if no error after 27-11-2023
        # typus_str = self.typus.__name__  # sfeh: can str(typus) work? -> str with args recursively?
        typus_str = self.get_ma_name_sfeh()  # sfeh: can str(typus) work? -> str with args recursively?

        if self.get_childs():
            if issubclass(type(self), BaseOperator):
                childstr = ', '.join([cc.str_as_list(cut_terms=cut_terms) for cc in self.get_childs()])
                typus_str = f'{typus_str}, {childstr}'
            else:
                # terminal nodes
                v = self.get_childs()[0]
                try:
                    if self.is_number():
                        # only show decimals for very small numbers: e.g. 0.00002, but 0.123456 -> 0.123
                        typus_str = term_format(f'{v}', cut=cut_terms)

                        # sfeh:discuss: now, small values are displayed as 0
                        # sfeh:discuss this is removed for mow
                        #   typus_str = f'{v:.3g}'  # '.5g'->5 decimals, trailing zeros, but rare (ugly) "E+04"
                        #   TypeError('unsupported format string passed to One.__format__')
                    else:
                        typus_str = f'{v}'
                except TypeError as ex:  # noqa  # sfeh
                    v_eval = v.evalf()
                    typus_str = term_format(v_eval, cut=cut_terms)
                    # sympy.ONE -> 1.0000000...
                    # sfeh:open int, non-floats are handled badly
                except Exception as ex:
                    print(f'SUCCESS sfeh:debug, delete?2 KEEP? {ex}')

        return f"[{typus_str}]"

    def __repr__(self):
        """
        sfeh:WRONG! Do NOT use __str__!
        This is acceptable, as it is never used anyways.
        only fixed nodes are missing
        """
        # raise NotImplementedError
        return self.represent_str(show_all=False)  # sfeh yeah, prints trees in debugger

    def get_lut_id(self):
        """
        # this whole function WAS replaced with repr_as_list()

        sfeh: Do NOT use str() or str_as_list(), as values get rounded
            Do NOT return a node-list and convert them to strings
            -> use repr()
        Unique+simple representation of a tree (to check in a lut if it was calculated already)
        regular print/string option should look better, this is just for getting a unique identifier
        returns string Identificator
        sfeh:discuss is this just repr?
        ID=Identificator, which"""

        # s = self.repr_as_list()
        s = self.represent_str(show_all=False)
        return s

    def str_as_expr(self):
        s = self.get_sympy_expr()
        return s

    # def get_expr_raw_fstring(self):
    #     """Add (1, a)
    #     self.typus.__name__ gets the class name
    #     """
    #     if self.is_term():
    #         fex = f'{self.get_childs()[0]}'
    #         fex = string_remove_trailing_zeroes(fex)
    #         return fex
    #     else:
    #         fex = [cc.get_expr_raw_fstring() for cc in self.get_childs()]
    #         fex = ', '.join(fex)
    #         if issubclass(type(self), OperatorArity):
    #             fex = f'{self.typus.showme}({fex})'
    #         elif self.is_ExprCdPair():
    #             fex = f'({fex})'
    #         else:
    #             fex = f'{self.typus.showme}({fex})'
    #     return f'{fex}'

    def get_expr_symlike(self, try_sympify=False, cut_terms=False):
        """(1 + a)
        each step returns like ({} + {})
        and then, the inputs are filled in the gaps
        """
        if self.is_term():
            expr = f'{self.get_childs()[0]}'
            expr = term_format(expr, cut=cut_terms)
            return expr
        else:
            expr = [cc.get_expr_symlike(try_sympify=try_sympify, cut_terms=cut_terms) for cc in self.get_childs()]
            # if issubclass(type(self), (ExprCondPair)):
            #     print('kjh')
            if self.is_chain():
                if isinstance(self, (AddChain, MulChain, AndChain, OrChain, XorChain)):
                    expr = self.inline_sep.join(expr)
                    expr = f'({expr})'
                else:
                    expr = ', '.join(expr)
                    expr = self.sy_str.format(expr)
            else:
                expr = self.sy_str.format(*expr)

        if try_sympify:
            expr_sy = sympy.sympify(expr)  # sfeh No local dict required?
            return expr_sy

        return f'{expr}'

    def list_mutable_nodes(self, xtype=None, allow_chain=True) -> ['NodeStructure']:
        """return all nodes that are mutable, aka suite for point- or branchmutation
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!

        === ValueError: 'a' cannot be empty unless no samples are taken
        ===> probably, no nodes were there to bemutated
        """

        # -> Check, if this node should be added
        if self.is_fix:
            node_list = []
        elif self.is_ExprCdPair():  # It is just a dummy holding a tuple
            node_list = []
        elif self.is_typus(Piecewise):
            node_list = []  # Piecewise has ambiguous xtype that can not be checked
        else:

            if xtype is None or xtype == self.get_xtype_self():
                # if not allow_chain or (allow_chain and self.is_chain()):
                if allow_chain or (not self.is_chain()):  # if it is chain -> check if allowed
                    node_list = [self]
                else:
                    node_list = []
            else:
                node_list = []

        # recursively add the other nodes
        if self.has_childs():  # sfeh:chain-operators discuss
            for cc in self.get_childs():
                a = cc.list_mutable_nodes(xtype=xtype, allow_chain=allow_chain)
                node_list.extend(a)

        return node_list

    def get_all_nodes_visualize(self, setid: str):
        """returns all nodes in a tree as list
        a+1 -> [+a1, a, 1]"""
        showme = f'{self.childs[0]}' if self.is_term() else f'{self.showme}'

        res = {setid: {'node': self,
                       'showme': showme}}
        edges = []

        if self.is_term():  # sfeh is_term instead of operator due to fuckin exprCondPairs
            pass
        else:
            for ii, cc in enumerate(self.childs):
                cid = f'{setid}-{ii}'
                cr, ce = cc.get_all_nodes_visualize(cid)
                res.update(cr)
                edges.append((setid, cid))
                edges.extend(ce)

        return res, edges

    def get_apted_notation(self):
        """Calculating the TED requires this (weird) representation"""
        return f"{{{self.get_typus()}{''.join([cc.get_apted_notation() for cc in self.get_childs()])}}}"

    def evolve_mutate_filter_gauss(self):
        """Recursively filter the nodes in the branch of fintree
        sfeh:   random filter all terminal nodes /
                single node /
                nodes in a branch /
                random nodes in a branch /
                intelligent filtering
        """
        if self.has_childs():
            for cc in self.get_childs():
                cc.evolve_mutate_filter_gauss()

        else:
            if self.is_number():
                self.childs[0] = round(random.gauss(self.childs[0], 0.1), FLOAT_PRECISION)  # sfeh: -> no symbols -> userspecific

        return

    def tree_node_grouping(self, tolerance=0):
        """
        If possible, this groups nodes to a simpler expression (if possible).
        E.g. a ** 2 -> square(a)
             a + b + c -> sum(a, b, c)  (chaining)

        # sfeh:idea Heavyside function. input a val, input b threshold
        # sfeh: sub, usub replace?
        sfeh:idea tolerance for grouping?
        """

        if self.is_term():  # good for runtime
            # sfeh:xxx sfeh:open do this in mutation?
            if self.is_number() and tolerance > 0:
                val = self.childs[0]
                # from sympy.physics.units import speed_of_light, meter, second more ideas!
                # sfeh also find more math building blocks, typical formulae
                # sympy.pi,         sympy.GoldenRatio,  sympy.Catalan,      sympy.EulerGamma,   sympy.TribonacciConstant
                # 3.14159265358979, 1.61803398874989,   0.915965594177219,  0.577215664901533,  1.83928675521416
                for const in [sympy.pi, sympy.GoldenRatio, sympy.Catalan, sympy.EulerGamma, sympy.TribonacciConstant]:
                    if (const-val) < tolerance:
                        self.childs[0] = const
                        return
                # sfeh: [sympy.S.One, sympy.S.Half, sympy.S.NegativeOne, sympy.S.NegativeHalf]:
                # sfeh: ALso take care of sqrt(2)
                if val - sympy.nsimplify(val, tolerance=tolerance, rational=True) < tolerance:
                    val_new = sympy.nsimplify(val, tolerance=tolerance, rational=True)
                    self.set_value(val_new)
                # sfeh:idea sympy.nsimplify('3.333333*x+0.522', tolerance=0.1, rational=True) for
                #   - Terminals
                #   - Even whole formulae!

                # sfeh:VERY USEFUL: strg+TribonacciConstant to go to init with useful info
            return
        else:

            mychlds = self.get_childs()

            for cc in mychlds:
                cc.tree_node_grouping(tolerance=tolerance)

            if isinstance(self, (Pow, PowRounded)):
                n_exp = mychlds[1].get_childs()[0]  # must exist
                if n_exp == -1:
                    self.replace_with(DivFraction, childs=[mychlds[0]])
                # sfeh:discuss (-2), respective negative exponent in general
                elif n_exp == 2:
                    self.replace_with(Square, childs=[mychlds[0]])
                elif n_exp == 0.5:
                    self.replace_with(Sqrt, childs=[mychlds[0]])
                    raise   # never hits!
                elif n_exp == sympy.S.Half:
                    self.replace_with(Sqrt, childs=[mychlds[0]])
                elif n_exp == 0.5 or n_exp == sympy.S.Half:
                    self.replace_with(Sqrt, childs=[mychlds[0]])
                elif isinstance(mychlds[1], Round):
                    self.replace_with(PowRounded, childs=[mychlds[0], n_exp])
                elif isinstance(mychlds[1], Number) and n_exp % 1 == 0:
                    self.replace_with(PowRounded, childs=mychlds)

            elif isinstance(self, (Mul, MulChain)):
                if self.is_chain():  # div only for
                    for cc in mychlds:
                        if isinstance(cc, DivFraction):
                            div_by = cc.get_childs()[0]
                            mychlds.remove(cc)
                            if len(mychlds) <= 1:
                                self.replace_with(Div, childs=[mychlds, div_by])
                            else:
                                self.replace_with(Div, childs=[AddChain(mychlds), div_by])
                        elif isinstance(cc, Number):
                            mul1 = cc.get_value()
                            if mul1 == sympy.S.NegativeOne:  # sfeh aka sympy.S.NegativeOne -1, was -1 before

                                self.replace_with(Usub, childs=[mychlds[1]])
                            elif 0 < mul1 < 1:
                                _n = Number((1 / mul1))  # sfeh: discuss; not always the best choice? maybe never? :P
                                self.replace_with(Div, childs=[mychlds[1], _n])

                else:

                    if isinstance(mychlds[0], DivFraction):
                        self.replace_with(Div, childs=[mychlds[1], mychlds[0].get_childs()[0]])
                    elif isinstance(mychlds[1], DivFraction):
                        self.replace_with(Div, childs=[mychlds[0], mychlds[1].get_childs()[0]])

                    elif mychlds[0].get_typus() == Number:
                        mul1 = mychlds[0].get_value()  # sfeh
                        if mul1 == sympy.S.NegativeOne:  # sfeh aka sympy.S.NegativeOne -1, was -1 before

                            self.replace_with(Usub, childs=[mychlds[1]])  # sfeh Usub ONLY option, ignore usub tree len
                        elif 0 < mul1 < 1:
                            _n = Number((1 / mul1))  # sfeh: discuss; not always the best choice? maybe never? :P
                            self.replace_with(Div, childs=[mychlds[1], _n])

                    elif mychlds[1].get_typus() == Number:
                        # sfeh this code can be simplified?
                        # mul1 = mychlds[1].get_childs()[0]
                        # if mul1 == -1:  # aka sympy.S.NegativeOne -1, was -1 before
                        #     self.replace_with(Usub, childs=[mychlds[0]])  # sfeh Usub ONLY option, ignore usub tree len
                        # elif 0 < mul1 < 1:
                        #     self.replace_with(Div, childs=[mychlds[0], 1 / mul1])
                        #     sfeh_div_by_test = 1 / mul1  # sfeh keep "div" as option
                        #     print(f'sfeh:test this needs testing when reached in future {sfeh_div_by_test}')
                        raise NotImplementedError  # commented block above
        return  # two indents

    def len_nodecount_raw(self):
        """counting the amount of nodes recursively"""
        if self.has_childs():
            return 1 + sum([cc.len_nodecount_raw() for cc in self.get_childs()])
        else:
            return 1  # childs can currently be floats

    def is_typus(self, t):
        r = issubclass(type(self), t)
        return r

    def is_ExprCdPair(self):  # noqa
        # sfeh: Sometimes, it hits the Dummy-class, sometimes (chained?) the sympy class. Not sure, why.
        r = issubclass(type(self), (ExprCondPair, ExprCondPair_Dummy))
        return r

    def len_nodecount_fair(self):
        """counting the amount of nodes, but
            - ignoring "Usub"!"
            - only biggest branch of Piecewise (aka If-then-else)
        """
        if self.is_term():
            n = 1
        else:
            cc_list = [cc.len_nodecount_fair() for cc in self.get_childs()]

            if self.is_typus(Usub):
                n = sum(cc_list)
            else:
                n = 1 + sum(cc_list)

        return n

    def __len__(self):
        return self.len_nodecount_fair()

    def get_arity(self):
        return len(self.get_typus().get_child_xts())

    def get_xtype_tuple(self):
        return self.xtype

    def get_xtype_self(self):
        return self.xtype[1]

    def represent_str(self, show_all=True, cut_terms=False):
        """
        sfeh:open
            - represent does not work and needs a lot of testing
            - Is not required for now!
        Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to str(), but adds the following information:
        - ":fix", when nodes are fixed"""

        s = self.showme  # class name

        if self.is_fix and show_all:
            s += ':fix'

        if self.is_term():
            cs = f'{self.get_childs()[0]}'
            if cut_terms:
                cs = term_format(cs, cut=cut_terms)
            else:
                cs = remove_trailing_zeroes(cs)
            if show_all:
                raise NotImplementedError
            else:
                s = f'{cs}'

        else:
            cs = [cc.represent_str(show_all=show_all, cut_terms=cut_terms) for cc in self.get_childs()]
            cs = ', '.join(cs)
            s = f"{s}({cs})"

        s = f"{s}"  # v1

        return s

    def __str__(self):
        # sfeh which style for trees?
        """
        Those string can be used:
        [Abs, [Abs, [Square, [cartPos]]]]
        [Abs, [Abs, [Square, [Symbol(cartPos)]]]]
        Abs(Abs((cartPos)**2))
        Abs(cartPos**2)
        cartPos**2
        Abs(Abs(Square(cartPos)))
        [Abs, [Abs, [Square, [cartPos]]]]
        """
        s = self.represent_str(show_all=False, cut_terms=True)
        # s1 = self.represent_str()
        # s2 = self.get_expr_symlike()
        # s3 = self.get_expr_symlike(try_sympify=True)
        # s4 = self.get_sympy_expr()
        # s5 = self.get_expr_raw_fstring()
        # s6 = self.str_as_list()
        # s_export = self.get_tree_export()
        # print(f'{s}\n{s1}\n{s2}\n{s3}\n{s4}\n{s5}\n{s6}\n{s_export}')
        return s

    def _sympy_(self, *args):  # -> sympy.Basic:
        _sym = self.symfun
        # _sym = _sym(*args)
        # childstr = [sympy.sympify(cc) for cc in args]
        # _sym = _sym(*childstr)

        # if isinstance(self, Operator):
        #     childstr = [sympy.sympify(cc) for cc in args]
        #     _sym = _sym(*childstr)
        #
        # elif isinstance(self, TerminalNode):
        #     return self.symfun(self.args[0])
        #
        # else:
        #     raise NotImplementedError(f'sfeh:Specify exception. Class-type {type(self)}')

        return _sym

    # @classmethod
    # def get_sym(cls):
    #     _sym = cls.symfun
    #     return _sym

    def get_symstr(self):
        return self.symfun.__name__

    @classmethod
    def get_child_xts(cls):
        return cls.xtype[0]

    def eval_np(self, *args)  -> Callable[[np.ndarray], np.ndarray]:
        ...


def eval_parsimony(tree: Node, complexity_measure, origin_tree=None):
    """
    sfeh: very inefficient to always get origin-tree apted
        -> introduce own complexity measure, do not import...
    """
    if complexity_measure == 'tree_node_count_raw':  # number of nodes
        return tree.len_nodecount_raw()  # returns the number of nodes  # sfeh weights
    elif complexity_measure == 'tree_node_count_fair':
        return tree.len_nodecount_fair()  # returns the number of nodes  # sfehxx weights
    elif complexity_measure == 'tree_edit_distance':  # tree_edit_distance, fintree-edit-distance
        apted1 = tree.get_apted_notation()
        apted2 = origin_tree.get_apted_notation()
        distance, mapping = apted_distance(apted1, apted2)  # sfeh the mapping could be useful somewhere
        return distance
    else:
        raise Exception(f'Complexity measurement not available: {complexity_measure}')


# class RootNode_Dummy(Node):
#     """Sfeh:discuss
#     this node can be used as dummy and is used to mimic a root type"""
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


def sympy_to_tree(s_expr: sympy.Basic, allow_chain) -> Node:
    """
    tree_from_expr, tree_from_sympy
    Important: start with the most specific rule
    # sfeh:discuss computational improvement when option to ignore args? do "raises" in args save time?
    # check is expr is an accepted operator, otherwise reconstruction probably fails"""
    if isinstance(s_expr, bool):
        return Boolean(s_expr)

    elif isinstance(s_expr, sympy.logic.boolalg.BooleanAtom):
        s_expr = True if isinstance(s_expr, (bool, sympy.logic.boolalg.BooleanTrue)) else False
        return Boolean(s_expr)

    # the following two lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    elif s_expr.is_Atom:
        if s_expr.is_Symbol:
            _n = Symbol(s_expr)
            return _n
        else:
            # expr_eval = s_expr.evalf(FLOAT_PRECISION)  # sfeh
            # if abs(s_expr - expr_eval) > 0.001:  # sfeh use float recision here 0.1**FLOAT_PRECISION
            #     expr_eval = s_expr
            if s_expr.is_Boolean:
                _n = Boolean(s_expr)
                return _n
            elif s_expr.is_number:  # is_float does not match int
                _n = Number(s_expr)
                return _n
                # "TypeError: Cannot convert complex to float" -> ignore the whole expression, let it fail
            else:
                raise NotImplementedError(f'What happened here? {s_expr}')

    else:  # **Operators**

        cc_nodes = []
        for arg in s_expr.args:
            cc_nodes.append(sympy_to_tree(arg, allow_chain=allow_chain))

        if isinstance(s_expr, Round_Dummy):
            _n = Round(cc_nodes[0])
            return _n

        elif allow_chain:  # sfeh:xxx do this one level above? ignore all allow_chains in here?
            op = d_sym2node_chain[type(s_expr)]
            _n = op(*cc_nodes)
            return _n

        elif isinstance(s_expr, ExprCondPair):
            return ExprCondPair_Dummy(cc_nodes)  # sfeh:debug/test

        elif isinstance(s_expr, sympy.Piecewise):
            # "Chained_VERSION" version is handled before
            reversed_pairs = list(s_expr.args[::-1])  # tuples to list, reversed: tuple must be nested the deepest
            reversed_pairs = [[sympy_to_tree(xx, allow_chain=allow_chain) for xx in list(i)] for i in reversed_pairs]
            otherwise = reversed_pairs[0][0]  # the last "True" condition
            for pairs in reversed_pairs[1:]:
                otherwise = Ifte(pairs[1], pairs[0], otherwise)
            return otherwise

        # sfehx include Usub, ignore usub in tree len()

        # elif isinstance(s_expr, Round_Dummy):
        #     return nd2(Round, [cc_nodes[0]])

        # elif isinstance(s_expr, Mul):
        #     # if s_expr.args[0].is_Rational:
        #     if s_expr.args[0].is_Rational:
        #         div_by = 1 / s_expr  # noqa
        #         print(f'sfeh:open div by {div_by}')

        elif isinstance(s_expr, tuple(d_sym2node)):
            clss = d_sym2node[type(s_expr)]
            if len(s_expr.args) > len(clss.get_child_xts()):
                _cc = cc_nodes[0]
                for _c2 in cc_nodes[1:]:  # sfeh check, if the order is like sympy order
                    _cc = clss(_cc, _c2)
                return _cc
                # raise TypeError(f"{clss} takes exactly {len(clss.get_child_xts())} args ({len(expr.args)} given)")
            else:
                n = clss(*cc_nodes)  # Max, Min, sign, add, mul, pow
                return n

    # sfeh:discuss
    # NotImplementedError: Expr missing: ITE(p > 13, tan(p - v) >= 2.578643, tan(p - v) >= 1)
    # this should not have occured, because it evaluates to bool, not to float
        sympy_expression_check_raise(s_expr)
    raise NotImplementedError(f'Expr missing: {s_expr}')


def tree_simplification(tree: Node, allow_chain) -> Node:
    """
    (Tries to) simplify/mathematically-reduce a tree. It is quite experimental
    # sfeh sympy-reconstruct patterns
    #   map symoy-sign to a sum
    #   map piecewise to if-then-else
    #   map power fractal - to sqrt?
    """
    tree_copy = copy.deepcopy(tree)
    expr_sym = tree.get_sympy_expr()
    # expr_sym2 = sympy.simplify(expr_sym)
    # if str(expr_sym) != str(expr_sym2):
    #     print(f'sfeh: {expr_sym} // {expr_sym2}')
    # expr_sym3  = tree.get_sympy_expr(simplimore=True)
    tree = sympy_to_tree(expr_sym, allow_chain=allow_chain)
    # if not allow_chain:
    # print(f'Copy : {len(tree_copy)}\t{tree_copy}')
    # print(f'Start: {len(tree)}\t{tree}')
    tree.tree_node_grouping(tolerance=0)
    # print(f'End:   {len(tree)}\t{tree}')
    if len(tree_copy) < len(tree):
        astr = string_remove_trailing_zeroes(str(tree_copy.get_sympy_expr()))
        bstr = string_remove_trailing_zeroes(str(tree.get_sympy_expr()))
        print(f'WHATHAPPENED SFEH\t{astr}'
              f'\n\told: {tree_copy.str_as_list()}'
              f'\n\tsym: {tree.str_as_list()}')
              # f'\n\told: {tree_copy.get_expr_symlike()}'
              # f'\n\tsym: {tree.get_expr_symlike()}'
              # f'\n\tsym: {tree_copy.get_tree_export()}')
        if astr != bstr:  # sfeh str() should not be required
            # sfeh 'a**0.5' does not become 'sqrt(a)'! use rational=True or sympy.S.Half
            # sfeh: rational can be a valid improvement, creating unified trees
            print_warning('w', f'Diff in sympy expression?\n\t{astr}\n\t{bstr}')
    return tree

def evolve_reduce_simplicate(tree: Node, allow_chain, completely=True, force=False) -> Node:
    """Reducing a fintree to its most basic form with sympify.
    (completely = False: reduce just one branch. if you wanted to have more complexity)"""
    tree_copy = copy.deepcopy(tree)
    if completely:  # reduce the complete tree
        nodes_lv0 = tree.get_mutable_rootnodes(extend_lvls=0)  # only required for fixed-core trees
        for cc in nodes_lv0:
            cc2 = tree_simplification(cc, allow_chain)
            cc.set_new_node(cc2)
    else:
        node_list = [n for n in tree.list_mutable_nodes(allow_chain=allow_chain) if
                     issubclass(n.get_typus(), OperatorArity)]  # ignoring leaf nodes...
        if len(node_list) == 0:
            print_warning('wwww', f'Tree for simplification does not provide operators: {tree}')
            return tree
        node = np.random.choice(node_list)
        node2 = tree_simplification(node, allow_chain)
        node.set_new_node(node2)  # sfeh chosen must be set again? or not? test it at least.
    if force:
        return tree
    else:
        if len(tree_copy) < len(tree):
            print_warning('w',
                          f'Tree grew larger during simplification:\n\t{tree_copy.str_as_list()}\n\t{tree.str_as_list()}')
            print_warning('ww', f'tree_copy: {len(tree_copy)} vs. {len(tree)}')
            return tree_copy
        else:
            return tree


def node_deepcopy(tree: Node) -> Node:
    _cpy = copy.deepcopy(tree)
    return _cpy


# class TreeMeta:
#
#     def __init__(self, fitness, parsimony):
#         self.fitness = fitness
#         self.parsimony = parsimony
#
#     def get_fitness(self):
#         return self.fitness
#
#     def get_parsimony(self):
#         return self.parsimony
#
#     # ...should this mean the size or fitness? not clear at all
#     # def __lt__(self, other):
#     #     return self.get_fitness() < other.get_fitness()
#     #
#     # def __eq__(self, other):
#     #     return self.get_fitness() <= other.get_fitness()


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



class CustomOperator:
    pass


class Node_Dummy(Node):
    """Terminal_Dummy, Function_Dummy now both in here"""
    # @classmethod
    # def get_child_xts(cls):
    #     return cls.xtype[0]
    pass


class PleaseUsePartnerOp(Node_Dummy):
    """Those operator-classes should be withdrawn, as they do not add further information
    E. g. Ge, Gt,
    such that only <, <=, ==, != exist and >, >= are not actually used
    """
    pass


class BaseOperator(Node):

    def __init__(self, *args):
        super().__init__()
        self.set_childs(args)

    def get_np_child_lambdas(self, *args):
        ccl = [cc.eval_np(*args) for cc in self.get_childs()]
        return ccl

    # Add debug prints
    def eval_np(self, *args):
        # Debug: Print the node type
        # print(f"Evaluating Node: {self}")

        def node_lambda(df):
            child_values = [ccl(df) for ccl in self.get_np_child_lambdas()]
            # print(f"Child Values for {self}: {child_values}")
            return self.np_fun(*child_values)

        return node_lambda


class OperatorArity(BaseOperator):

    arity = None


class OperatorChained(BaseOperator):
    # no xtype, only input type
    # no tflow, separate handling in totf-function
    # Piecewise, AddChain, MulChain, MinChain, MaxChain, AndChain, OrChain
    # childs_min_max = [1, 5]

    showme = 'OperatorChained'
    sy_str = lambda self, s: None
    repr_str = None
    pass

    @staticmethod
    def np_hit(self, *args):
        return self.np_fun(*args)


class ChainableOp:
    """(Abstract) class for operators, that allow flexible arity (1-n args).
    Used e.g. while reconstructing trees from sympy expressions,
    to check whether it is possible to put more childs than planned into the node.

    The respected Operators are
    Add, Mul, Min, Max
    And, Or
    Piecewise/Ifte

    sympy-equivalent: class LatticeOp
    """
    xtype_chain = None


class MathOperator(OperatorArity):
    # is_real = True
    # is_Boolean = False
    pass


class LogicOperator(OperatorArity):
    # And, Or, Xor, Not
    # is_real/real = False
    # is_Boolean/bool = True
    pass


class RelationalOperator(OperatorArity):
    pass


class Trigonometry(MathOperator):
    pass


class MinMaxBase(MathOperator):
    pass


class NoSymCapitalized:
    """Does nothing, but maybe its good to know, which OG-sympy classes are lower case
    E. g. Sin() is sin() in sympy (?)
    """
    pass


class Terminal(Node):  # sfeh sympy.Atom
    """Terminal nodes are leaf nodes which can not have children. e.g.:
    - constants (e.g. 2.3)
    - observations (e.g. b, aka data input)
    """
    # value = None
    def __len__(self):
        return 1

    def get_value(self):
        return self.get_childs()[0]

    def set_value(self, val):
        self.set_childs([val])


class Boolean(Terminal):
    # sfeh:discuss just for True/False?
    xtype = ((), bool)  # sfeh ((None,), bool)?
    symfun = lambda a: sympy.S.true if a[0] else ~sympy.S.true  # sympy.logic.boolalg.Boolean  # sfeh:discuss
    np_fun = np.array
    showme = 'Boolean'
    # tflow = lambda arg: tf.constant(arg, dtype=tf.bool)

    # def __init__(self, value):
    #     self.value = sympy.S.true if value else ~sympy.S.true

    def eval_np(self, *args):
        return lambda df: np.full(df.shape[0], self.get_value(), dtype=bool)


class Number(Terminal):
    xtype = ((), float)
    symfun = lambda a: sympy.Float(float(a[0]), FLOAT_PRECISION)
    np_fun = np.array
    # sympy.Rational(0.1) -> 3602879701896397/36028797018963968
    # sympy.Rational('0.1') -> 1/10
    showme = 'Number'
    # sfeh: problem with rational: Sqrt(8.0) -> 2*sqrt(6)/3. sfeh: actually is_atomic?
    # tflow = lambda a: tf.constant(a, dtype=tf.float32)

    def eval_np(self, *args):  # sfeh: required to make np.array()?
        return lambda df: np.full(df.shape[0], self.get_value(), dtype=float)


class Symbol(Terminal):
    """
    The Symbol is set with sympy.Symbol() on creation in child[0]
    sfeh:discuss: should typus have a sign (-pos); can appear in observations
    This was used to deal with negative values
        self.name = nlabl if nlabl[0] != '-' else nlabl[1:]
    """
    symfun = lambda a: a[0]  # sfeh, no symbol-conversion; no reason for.
    np_fun = None
    # np_fun =
    xtype = ((), float)
    showme = 'Symbol'

    def eval_np(self, *args):
        return lambda df: df[str(self.get_value())].to_numpy()


def cast_input(ii):
    if isinstance(ii, NodeStructure):
        return ii
    else:
        # For "human" inputs like Add(1, 'var')
        if isinstance(ii, bool):
            return Boolean(ii)
        elif isinstance(ii, (float, int)):
            return Number(ii)
        elif isinstance(ii, str):
            return Symbol(ii)  # sfeh symbol opts
        else:
            raise NotImplementedError


class Add(MathOperator, ChainableOp):
    symfun = lambda a: sympy.Add(a[0], a[1])
    np_fun = np.add
    showme = 'Add'
    sy_str = '({0} + {1})'
    formulae_str = '({} + {})'
    repr_str = 'Add{},[{},{}]'
    xtype = ((float, float), float)
    xtype_chain = float


class DivFraction(MathOperator):
    """x**-1
    aka InverseFraction aka DivFraction aka Reciprocal"""
    xtype = ((float,), float)
    symfun = lambda a: sympy.Pow(a[0], sympy.S.NegativeOne)
    np_fun = lambda *a: np.reciprocal(float(*a))
    # sfeh np.reciprocal(2) -> 0 np.reciprocal(float(2)) -> 0.5
    # -> it is okay, 1/(int) is just always zero
    # np_fun = lambda *a: np.reciprocal(float(*a))  # sfeh rename class?
    showme = 'DivFraction'
    sy_str = '1/({})'
    repr_str = 'DivFraction{},[{}]'


class Pow(MathOperator):
    symfun = lambda a: sympy.Pow(a[0], a[1])
    np_fun = np.power
    showme = 'Pow'
    sy_str = '({0})**({1})'
    repr_str = 'Pow{},[{},{}]'
    xtype = ((float, float), float)


class Abs(MathOperator):
    symfun = lambda a: sympy.Abs(a[0])
    np_fun = np.absolute  # np.fabs works only for non-complex numbers
    showme = 'Abs'
    sy_str = 'Abs({})'
    repr_str = 'Abs{},[{}]'
    xtype = ((float,), float)


class Sign(MathOperator, NoSymCapitalized):
    # does not work in string, but irrelevant. sympy.simplify('sign(-a)') -> -sign(a)
    symfun = lambda a: sympy.sign(a[0])
    np_fun = np.sign
    showme = 'Sign'
    sy_str = 'sign({})'
    repr_str = 'Sign{},[{}]'
    xtype = ((float,), float)


class Log(MathOperator, NoSymCapitalized):
    # Log isactually Ln (base e). Log/Ln is the same, idk fuck Log10
    symfun = lambda a: sympy.log(a[0])
    np_fun = np.log
    showme = 'Log'
    sy_str = 'log({})'
    repr_str = 'Log{},[{}]'
    xtype = ((float,), float)


class Cos(Trigonometry, NoSymCapitalized):
    symfun = lambda a: sympy.cos(a[0])
    np_fun = np.cos
    showme = 'Cos'
    sy_str = 'cos({})'
    repr_str = 'Cos{},[{}]'
    xtype = ((float,), float)


class Sin(Trigonometry, NoSymCapitalized):
    symfun = lambda a: sympy.sin(a[0])
    np_fun = np.sin
    showme = 'Sin'
    sy_str = 'sin({})'
    repr_str = 'Sin{},[{}]'
    xtype = ((float,), float)


class Tan(Trigonometry, NoSymCapitalized):
    # sfeh:discuss actually rename classes.
    # they do not have to match sympy expressions/classes
    symfun = lambda a: sympy.tan(a[0])
    np_fun = np.tan
    showme = 'Tan'
    sy_str = 'tan({})'
    repr_str = 'Tan{},[{}]'
    xtype = ((float,), float)


class Acos(Trigonometry, NoSymCapitalized):
    symfun = lambda a: sympy.acos(a[0])
    np_fun = np.arccos  # arccosh
    showme = 'Acos'
    sy_str = 'acos({})'
    repr_str = 'Acos{},[{}]'
    xtype = ((float,), float)


class Asin(Trigonometry, NoSymCapitalized):
    """"""
    symfun = lambda a: sympy.asin(a[0])
    np_fun = np.arcsin
    showme = 'Asin'
    sy_str = 'asin({})'
    repr_str = 'Asin{},[{}]'
    xtype = ((float,), float)


class Atan(Trigonometry, NoSymCapitalized):
    symfun = lambda a: sympy.atan(a[0])
    np_fun = np.arctan
    showme = 'Atan'
    sy_str = 'atan({})'
    repr_str = 'Atan{},[{}]'
    xtype = ((float,), float)


class Tanh(Trigonometry, NoSymCapitalized):
    symfun = lambda a: sympy.tanh(a[0])
    np_fun = np.tanh
    showme = 'Tanh'
    sy_str = 'tanh({})'
    repr_str = 'Tanh{},[{}]'
    xtype = ((float,), float)


class Sinh(Trigonometry, NoSymCapitalized):
    symfun = lambda a: sympy.sinh(a[0])
    np_fun = np.sinh
    showme = 'Sinh'
    sy_str = 'sinh({})'
    repr_str = 'Sinh{},[{}]'
    xtype = ((float,), float)


class Cosh(Trigonometry, NoSymCapitalized):
    symfun = lambda a: sympy.cosh(a[0])
    np_fun = np.cosh
    showme = 'Cosh'
    sy_str = 'cosh({})'
    repr_str = 'Cosh{},[{}, {}]'
    xtype = ((float,), float)


class Not(LogicOperator):
    """not"""
    symfun = lambda a: sympy.Not(a[0])
    np_fun = np.logical_not
    showme = 'Not'
    sy_str = '~({})'
    repr_str = 'Not{},[{}]'
    xtype = ((bool,), bool)


class Eq(RelationalOperator):
    """a == b"""
    # sfeh:debug Eq and Ne (), which also work for boolean inputs in sympy
    symfun = lambda a: sympy.Eq(a[0], a[1])
    np_fun = np.equal
    showme = 'Eq'  # '==' not working in sympy!
    sy_str = 'Eq({0}, {1})'
    repr_str = 'Eq{},[{}, {}]'
    xtype = ((float, float), bool)


class Ne(RelationalOperator):
    """a != b"""
    symfun = lambda a: sympy.Ne(a[0], a[1])
    np_fun = np.not_equal
    showme = 'Ne'  # != not working in sympy
    sy_str = 'Ne({0}, {1})'
    repr_str = 'Ne{},[{}, {}]'
    xtype = ((float, float), bool)


class Mul(MathOperator, ChainableOp):
    symfun = lambda a: sympy.Mul(a[0], a[1])
    np_fun = np.multiply
    showme = 'Mul'  #
    sy_str = '({0} * {1})'
    repr_str = 'Mul{},[{}, {}]'
    xtype = ((float, float), float)
    xtype_chain = float


class And(LogicOperator, ChainableOp):
    symfun = lambda a: sympy.And(a[0], a[1])
    np_fun = np.logical_and
    showme = 'And'
    sy_str = '({0} & {1})'
    repr_str = 'And{},[{}, {}]'
    xtype = ((bool, bool), bool)
    xtype_chain = bool


class Or(LogicOperator, ChainableOp):
    symfun = lambda a: sympy.Or(a[0], a[1])
    np_fun = np.logical_or
    showme = 'Or'
    sy_str = '({0}|{1})'
    repr_str = 'Or{},[{}, {}]'
    xtype = ((bool, bool), bool)
    xtype_chain = bool


class Xor(LogicOperator, NoSymCapitalized, ChainableOp):
    """
    Caution: loading '(a ^ b)', the sympy-Xor-representation, is interpreted as a**b"""
    symfun = lambda a: sympy.Xor(a[0], a[1])
    np_fun = np.logical_xor
    showme = 'Xor'
    sy_str = 'Xor({}, {})'  # 'a ^ b'
    repr_str = 'Xor{},[{}, {}]'
    xtype = ((bool, bool), bool)


class ITE(LogicOperator):
    """sfeh:is this really required? currently not in use"""
    symfun = lambda a: sympy.ITE(a[0], a[1], a[2])
    # np_fun = lambda *a: np.logical_or(np.logical_and(*a[0], *a[1]), np.logical_and(np.logical_not(*a[0]), *a[2]))
    # np_fun = lambda a: np.logical_or(np.logical_and(a[0], a[1]), np.logical_and(np.logical_not(a[0]), a[2]))
    # np_fun = lambda a, b, c: np.logical_or(np.logical_and(a, b), np.logical_and(np.logical_not(a), c))
    np_fun = lambda a, b, c: ((a & b) | (not a & c))
    showme = 'ITE'
    sy_str = 'ITE({0}, {1}, {2})'
    repr_str = 'ITE{},[{}, {}, {}]'
    xtype = ((bool, bool, bool), bool)
    # tflow = lambda *args: tf.cond(args[0], true_fn=args[1], false_fn=args[2])


class Min(MinMaxBase, ChainableOp):
    symfun = lambda a: sympy.Min(a[0], a[1])
    np_fun = np.minimum  # sfeh max, maximum, maximum.reduce
    showme = 'Min'
    sy_str = 'Min({0},{1})'
    repr_str = 'Min{},[{}, {}]'
    xtype = ((float, float), float)
    xtype_chain = float

    # def __call__(self, a):
    #     # Handle numerical evaluation (for lambdify or direct calls)
    #     if isinstance(a, (int, float, np.ndarray, pd.DataFrame)):
    #         return np.minimum.reduce(a)
    #     raise TypeError("Unsupported type for numerical evaluation in Min(MinMaxBase, ChainableOp)")


class Max(MinMaxBase, ChainableOp):
    symfun = lambda a: sympy.Max(a[0], a[1])
    np_fun = np.maximum  # sfeh max, maximum, maximum.reduce
    showme = 'Max'
    sy_str = 'Max({0}, {1})'
    repr_str = 'Max{},[{}, {}]'
    xtype = ((float, float), float)
    xtype_chain = float

    def __call__(self, a):
        # Handle numerical evaluation (for lambdify or direct calls)
        if isinstance(a, (int, float, np.ndarray, pd.DataFrame)):
            return np.maximum.reduce(a)
        raise TypeError("Unsupported type for numerical evaluation in Min(MinMaxBase, ChainableOp)")


class Lt(RelationalOperator):
    symfun = lambda a: sympy.Lt(a[0], a[1])
    np_fun = np.less
    showme = 'Lt'
    sy_str = '({0} < {1})'
    repr_str = 'Lt{},[{}, {}]'
    xtype = ((float, float), bool)


class Le(RelationalOperator):
    symfun = lambda a: sympy.Le(a[0], a[1])
    np_fun = np.less_equal
    showme = 'Le'
    sy_str = '({0} <= {1})'
    repr_str = 'Le{},[{}, {}]'
    xtype = ((float, float), bool)


class Gt(RelationalOperator, PleaseUsePartnerOp):
    symfun = lambda a: sympy.Gt(a[0], a[1])
    np_fun = np.greater
    showme = 'Gt'
    sy_str = '({0} > {1})'
    repr_str = 'Gt{},[{}, {}]'
    xtype = ((float, float), bool)


class Ge(RelationalOperator, PleaseUsePartnerOp):
    xtype = ((float, float), bool)
    symfun = lambda a: sympy.Ge(a[0], a[1])
    np_fun = np.greater_equal
    showme = 'Ge'
    sy_str = '({0} >= {1})'
    repr_str = 'Ge{},[{}, {}]'


class Square(MathOperator):
    symfun = lambda a: sympy.Pow(a[0], 2)
    np_fun = np.square
    xtype = ((float,), float)
    showme = 'Square'
    sy_str = '({})**2'
    repr_str = 'Square{},[{}]'


class Exp(MathOperator):
    symfun = lambda a: sympy.exp(a[0])
    np_fun = np.exp
    showme = 'Exp'
    sy_str = '{}**E'
    repr_str = 'Exp{},[{}, {}]'
    xtype = ((float,), float)


class Exp2(MathOperator):
    symfun = lambda a: sympy.Pow(2, a[0])
    np_fun = np.exp2
    xtype = ((float,), float)
    showme = 'Exp2'
    sy_str = '2**({})'
    repr_str = 'Exp2{},[{}]'


class Sub(MathOperator):
    xtype = ((float, float), float)
    symfun = lambda a: sympy.Add(a[0], -a[1])
    np_fun = np.subtract
    showme = 'Sub'
    sy_str = '({0} - {1})'
    repr_str = 'Sub{},[{}, {}]'


class Ifte(OperatorArity):  # sfeh:Discuss: ChainableOp
    """Also class Piecewise"""
    xtype = ((bool, float, float), float)
    symfun = lambda a: sympy.Piecewise((a[1], a[0]), (a[2], True))
    np_fun = np.where
    showme = 'Ifte'
    sy_str = 'Ifte({0},{1},{2})'
    repr_str = 'Ifte{},[{}, {}, {}]'
    expr_dummy = 'Ifte'
    xtype_chain = (float, bool)

    def eval_np(self, *args):
        # cond, if_true, if_false = [c.eval_np(*args) for c in self.childs]
        cond, if_true, if_false = self.get_np_child_lambdas()
        fun = lambda df: self.np_fun(cond(df), if_true(df), if_false(df))
        return fun


class Round(MathOperator):
    """

    """
    xtype = ((float,), float)
    # symfun = lambda a: a.round(0) if a.is_number else Round_Dummy(a)
    # symfun: Callable[[sympy.Expr], sympy.Expr] = lambda a: a.round(0) if a.is_number else Round(a)  # sfeh (next line)
    # this is here to hint the type, as sympy will throw a warning otherwise, leading to this
    symfun = lambda a: Round_Dummy(a[0])
    np_fun = np.round
    showme = 'Round'
    sy_str = 'Round_Dummy({},1)'
    repr_str = 'Round_Dummy{},[{}]'


class Round_Dummy(sympy.Function):  # Not a Math-operator
    """
    Workaround for rounding exponents
    For more details, look at: plagih/discoveries/rounding_exponents.py
    """
    @classmethod
    def eval(cls, a):
        if a.is_symbol:
            return
        elif a.is_number:
            return sympy.Integer(round(a))  # Ensure it's a SymPy Integer

    def _sympy_(self, a):
        return eval(self, a)  # sfeh: is this required?

    def __call__(self, a):
        # Handle numerical evaluation (for lambdify or direct calls)
        if isinstance(a, (int, float, np.ndarray)):
            return np.round(a)
        raise TypeError("Unsupported type for numerical evaluation in Round_Dummy")

class PowRounded(MathOperator):
    """Requires class Round_Dummy!
    Rounds the exponent; sfeh:idea clip exponent?"""
    symfun = lambda a: sympy.Pow(a[0], Round_Dummy(a[1]))
    np_fun = staticmethod(lambda base, exponent: np.power(base, np.round(exponent)))
    # np_fun = lambda base, exponent: np.power(base, np.round(exponent))
    # np_fun = lambda *a: np.power(*a[0], np.round(*a[1]))
    # np_fun = None
    showme = 'PowRounded'
    sy_str = '{0})**Round_Dummy({1})'
    repr_str = 'PowRounded{},[{}, {}]'
    xtype = ((float, float), float)

    def eval_np(self, *args):
        return lambda df: self.np_fun(*[ccl(df) for ccl in self.get_np_child_lambdas()])

# sfeh:open
# class Log1p(MathOperator):
#     # https://docs.sympy.org/latest/modules/codegen.html#sympy.codegen.cfunctions.log1p
#     xtype = ((float,), float)
#     symfun = lambda a: sympy.log(a + 1)
#     showme = 'log1p'


class Div(MathOperator):
    symfun = lambda a: sympy.Mul(a[0], 1 / a[1])
    np_fun = np.divide
    showme = 'Div'
    sy_str = '({0}/{1})'
    repr_str = 'Div{},[{}, {}]'
    xtype = ((float, float), float)


class Sqrt(MathOperator):
    """Capitalized class name, even though its a sympy function
    In SymPy, sqrt(x) is just a shortcut to x**Rational(1, 2)"""
    xtype = ((float,), float)
    symfun = lambda a: sympy.sqrt(a[0])  # same as: lambda a: sympy.Pow(a, sympy.S.Half)
    np_fun = np.sqrt
    showme = 'Sqrt'
    sy_str = 'sqrt({})'
    repr_str = 'Sqrt{},[{}, {}]'


# class Divide_no_nan(Operator):
#     # class-name = 'Divide_no_nan'  # sfeh??
#     tflow = tf.math.divide_no_nan
#     symfun = lambda a, b: sympy.Mul(a, )
#     xtype = ((float, float), float)


class Usub(MathOperator):
    xtype = ((float,), float)
    symfun = lambda a: sympy.Mul(a[0], -1)
    np_fun = np.negative
    # tf_fun = tf.negative
    showme = 'Usub'  # sfeh
    sy_str = '(-{})'
    repr_str = 'Usub{},[{}]'


class Clip(MinMaxBase, CustomOperator):
    # sfeh:open use this
    symfun = lambda a: sympy.Min(sympy.Max(a[0], a[1]), a[2])
    np_fun = np.clip  # lambda a, b, c: np.clip(a, b, c)
    # tf_fun = lambda a, b, c: tf.clip_by_value(a, b, c)
    showme = 'Clip'
    sy_str = '(sympy.Min(sympy.Max({0}, {1}), {2}))'
    repr_str = 'Clip{},[{}, {}]'
    xtype = ((float, float, float), float)
    # sfeh: param 1 and 2 should be set according to the min and max of available param?


class ExprCondPair_Dummy(Node_Dummy):  # noqa
    """
    Named like this to differ from the sympy original (ExprCondPair)
    sfeh:discuss
    The only purpose is to wrap the results for a Node-structure, where every Node has childs with other nodes"""
    symfun = lambda a: ExprCondPair(a[0], a[1])
    np_fun = None
    showme = 'ExprCondPair_Dummy'  # sfeh... mmake this a tuple?
    sy_str = 'ExprCondPair({0}, {1})'
    repr_str = 'ExprCondPair_Dummy{},[{}, {}]'
    xtype = ([(float, bool)], float)
    expr_dmy = 'ExprCondPair_Dummy'


class AddChain(OperatorChained):
    """It is Sum, but we call it chain, as vector also is taken"""
    symfun = lambda a: sympy.Add(*a)
    np_fun = lambda *a: np.sum(a)  # n
    showme = 'Add'
    sy_str = 'Add({})'
    formulae_str = 'Add({})'
    repr_str = 'AddChain{},[{}]'
    inline_sep = ' + '
    xtype = ([(float,)], float)
    xtype_chain = float


class MulChain(OperatorChained):
    showme = 'Mul'
    symfun = lambda a: sympy.Mul(*a)
    np_fun = lambda *a: np.prod(a)
    sy_str = 'Mul({})'
    formulae_str = 'Mul({})'
    repr_str = 'MulChain{},[{}]'
    inline_sep = ' * '
    xtype = ([(float,)], float)
    xtype_chain = float


class MinMaxChainBase(OperatorChained):

    # def eval_np(self, *args):
    #     return lambda df: self.np_fun([ccl(df) for ccl in self.get_np_child_lambdas()])
    pass


class MinChain(MinMaxChainBase):
    symfun = lambda a: sympy.Min(*a)
    # np_fun = lambda *x: np.min(*x, axis=1)  # np.vstack(x)
    np_fun = np.minimum
    # np_fun = lambda x: np.minimum  # sfeh open
    showme = 'Min'
    sy_str = 'Min({})'
    formulae_str = 'Min({})'
    repr_str = 'MinChain{},[{}]'
    xtype = ([(float,)], float)
    xtype_chain = float  # sfeh xtype_chain...


class MaxChain(MinMaxChainBase):
    symfun = lambda a: sympy.Max(*a)
    np_fun = np.maximum  # np.vstack(x)
    showme = 'Max'
    sy_str = 'Max({})'
    formulae_str = 'Max({})'
    repr_str = 'MaxChain{},[{}]'
    xtype = ([(float,)], float)
    xtype_chain = float

    # def eval_np(self, args):
    #     _r = np.max(np.vstack(args), axis=0)
    #     return _r


# class OrderedSelector(ChainOp):
#     """sfeh:Orders Elements with < and picks the (1, -1 or even -2, 'median')-element?
#     sfeh:idea using a function for selecting the n-th is an option!"""
#     xtype = ((float,), float)
#     xtype_chain = float
#     symfun = lambda *a: sympy.Order(a)
#     np_fun = None


class Piecewise(OperatorChained):
    """sfeh:discuss: the only Operator, which has tuples as input

    sfeh:open force a (foo, True) option, restrict mutating "True"
            CAUTION! Notimplemented?
        A method to determine whether a multivariate conditional is consistent
        with a complete coverage of all variables has not been implemented so
        the rewrite is being stopped after encountering `cartPos >=
        Max(cartPos, 10.4*cartPos*cartVel)`. This error would not occur if a
        default expression like `(foo, True)` were given.
"""
    # ogclass = Ifte
    # xtype = ((float, bool), float)
    symfun = lambda a: sympy.Piecewise(*a)
    np_fun = None
    showme = 'Piecewise'
    sy_str = 'Piecewise({})'
    formulae_str = 'Piecewise({})'
    repr_str = 'Piecewise{},[{}]'
    # these must be handeled differently, so commented out
    xtype = ([(ExprCondPair,)], float)
    xtype_chain = ExprCondPair  # discuss (float, bool)

    def eval_np(self, *args):
        pairs = [(c.childs[0].eval_np(*args), c.childs[1].eval_np(*args)) for c in self.get_childs()]

        def piecewise_lambda(df):
            result = pairs[-1][1](df)  # Default case
            for cond, expr in reversed(pairs[:-1]):
                result = np.where(cond(df), expr(df), result)
            return result

        return piecewise_lambda


class AndChain(OperatorChained):
    expr_dmy = 'And'
    symfun = lambda a: sympy.And(*a)
    np_fun = lambda *a: np.logical_and(*a)
    showme = 'AndChain'
    sy_str = 'And({})'
    formulae_str = 'And({})'
    repr_str = 'And{},[{}]'
    inline_sep = ' & '
    xtype = ([(bool,)], bool)
    xtype_chain = bool


class OrChain(OperatorChained):
    xtype = ([(bool,)], bool)
    xtype_chain = bool
    symfun = lambda a: sympy.Or(*a)
    np_fun = lambda *a: np.logical_or(*a)
    showme = 'Or'
    sy_str = 'Or({})'
    formulae_str = 'Or({})'
    repr_str = 'OrChain{},[{}]'
    inline_sep = ' | '


class XorChain(OperatorChained):
    """
    Caution: loading '(a ^ b)', the sympy-Xor-representation, is interpreted as a**b"""
    symfun = lambda a: sympy.Xor(*a)
    np_fun = lambda *a: np.logical_xor(*a)
    showme = 'Xor'
    sy_str = 'Xor({})'  # 'a ^ b'
    formulae_str = 'Xor({})'
    inline_sep = ' ^ '
    repr_str = 'XorChained{},[{}, {}]'
    xtype = ([(bool,)], bool)
    xtype_chain = bool


sym2node = {sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos, sympy.asin: Asin, sympy.atan: Atan,
            sympy.tanh: Tanh, sympy.sinh: Sinh, sympy.cosh: Cosh, sympy.Min: Min, sympy.Max: Max, sympy.Add: Add,
            sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul, sympy.sqrt: Sqrt,
            sympy.exp: Exp, sympy.Xor: Xor, sympy.Not: Not, sympy.Equality: Eq, sympy.Ne: Ne, sympy.And: And,
            sympy.Or: Or, sympy.ITE: ITE, sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.Gt: Gt,
            sympy.GreaterThan: Ge}


d_sym2node = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul,
              sympy.Xor: Xor, sympy.Not: Not, sympy.And: And, sympy.Or: Or, sympy.StrictLessThan: Lt, sympy.LessThan: Le,
              sympy.StrictGreaterThan: Gt, sympy.GreaterThan: Ge, sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan,
              sympy.acos: Acos, sympy.asin: Asin, sympy.atan: Atan, sympy.tanh: Tanh, sympy.sinh: Sinh, sympy.cosh: Cosh,
              sympy.Min: Min, sympy.Max: Max, sympy.ITE: ITE, sympy.exp: Exp}
# The chained version is the regular version updated with the following operators
d_sym2node_chain = d_sym2node | {sympy.Add: AddChain, sympy.Mul: MulChain, sympy.Min: MinChain, sympy.Max: MaxChain,
                                 sympy.And: AndChain, sympy.Or: OrChain,  sympy.Xor: XorChain, sympy.Piecewise: Piecewise,
                                 ExprCondPair: ExprCondPair_Dummy}


def sympy_expression_check_raise(expr_sym):

    if expr_sym.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I, sympy.im, sympy.re):
        # sfeh:discuss sympy.re: real part -> don't ignore; if there is a real part, there is a imaginary part.
        raise SympySimplificationError(f'Simplification failed: {expr_sym}')
    return expr_sym


def expr_sympify(expr):
    """
    Returns a simplified expression using sympify.
    - If sympify evaluates to one of these errors: 'zoo', 'inf', '*I', 'nan', stop evaluation

    Sympify is a python core module which reduced mathematical expressions.
    Example: sympify('a+a+a+a') -> a*4
    Note that the sympify was extended in plagih_sympify_extras.py with extra functions

    Sympify fails: The results are, or contain, expressions that should/can not be evaluated
    'zoo': (Complex infinity) E.g. when an int-number is divided by zero
    'inf': (Regular infinity) E.g. when a float-number is divided by zero (...i know, why are there two infinities?)
    '*I': (Complex number) E.g. when putting a number to the power of negative fractals, 1**(-0.5)
    'nan': (Not a number) when Evaluation fails, E.g. types contradict, expression is empty, 'Min(a, zoo' ...

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

    try:
        expr_sym = sympy.sympify(expr)
        sympy_expression_check_raise(expr_sym)
        return expr_sym

    except ValueError as ex:
        raise ValueError(f'NaN in {ex}')
    except AttributeError as ex:
        # print(f'sfeh: This sympy bug happens, when sympifying "True": {ex}')
        raise
        # return sympy.true if expr else sympy.false

# sfeh:RANDOM IDEA when loading the data, check for every possible assupmtion. Better: load them manually.

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

# totf = {
#     # sympy.Symbol: 'tf': lambda x: tf.cons
#     sympy.Min: tf.minimum,
#     sympy.Max: tf.maximum,
#     sympy.Add: tf.add,
#     sympy.Mul: tf.multiply,
#     sympy.Pow: tf.pow,
#     sympy.Abs: tf.abs,
#
#     sympy.Not: tf.logical_not,
#     sympy.And: tf.logical_and,
#     sympy.Or: tf.logical_or,
#     sympy.Xor: tf.math.logical_xor,
#
#     sympy.Equality: tf.equal,
#     sympy.Unequality: tf.not_equal,
#     sympy.GreaterThan: tf.greater_equal,
#     sympy.StrictGreaterThan: tf.greater,
#     sympy.LessThan: tf.less_equal,
#     sympy.StrictLessThan: tf.less,
#     # sympy.N: tf.math.round,  # incorrect, sympy.N(x, 1) is right
#     sympy.log: tf.math.log,
#     sympy.cos: tf.cos,
#     sympy.cosh: tf.cosh,
#     sympy.sin: tf.sin,
#     sympy.sinh: tf.sinh,
#     sympy.tan: tf.tan,
#     sympy.tanh: tf.tanh,
#     sympy.acos: tf.acos,
#     sympy.asin: tf.asin,
#     sympy.atan: tf.atan,
#     sympy.sign: tf.sign,
#     # The real Part
#     sympy.re: lambda a: tf.convert_to_tensor(a, dtype=tf.dtypes.float32),  # sfeh sympy-gotcha, comes up randomly
#     # Round_Dummy: tf.round,
#     sympy.exp: tf.exp,  # sfeh this occurs randomly...
#     sympy.ITE: tf.cond
# }

# def sympy_to_tensorflow(expr_sy, d_tensors):
#     """
#     - check terminal-node
#     -- check symbol
#     --
#
#     Bugs/gotchas:
#
#     sympify('True')             -> True
#     sympify('1')==True          ->
#     sympify('~(True)')          -> -2
#     sympify('~(False)')         -> -1
#     sympify('a <= Min(a, b)')   ->
#
#     sympy.logic.boolalg.ITE has only boolean inputs
#     evaluate=None/false
#
#     sympy.logic.boolalg.ITE is not If-then-else
#     sympy.cosh can emerge out of sin-stuff
#     sympy.sympify('True')->True is no sympy expression anymore
#     # sfeh:bug gotcha sympy.re comes up randomly
#     """
#     # shape = tensors[list(tensors.keys())[0]].get_shape()  # sfeh:open:workaround:
#
#     # ==Bug-handling==  sympy.sympify('True')->True is no sympy expression anymore
#     if isinstance(expr_sy, bool):  # e.g. '1'
#         return tf.constant(expr_sy, dtype=tf.dtypes.bool)
#     if isinstance(expr_sy, (bool, sympy.logic.boolalg.BooleanAtom)):
#         expr_sy = True if isinstance(expr_sy,
#                                      sympy.logic.boolalg.BooleanTrue) else False  # sfeh:collect sympy bug gotcha bug
#         return tf.constant(expr_sy, dtype=tf.dtypes.bool)
#
#     # the following lines are not required, if sympy filters for bad expressions earlier
#     if expr_sy.is_imaginary or expr_sy.is_infinite:
#         raise ValueError(f'Cannot convert this to Tensorflow: {expr_sy}')
#
#     # ==Terminal nodes==
#     elif expr_sy.is_Atom:
#         if expr_sy.is_Symbol:
#             # result = feed_dict[expr.name]  # sfeh:discuss placeholder
#
#             # result = tf.compat.v1.placeholder(tf.bool, name=str(expr))
#             # sfeh:runtime?
#             result = tf.constant(d_tensors[str(expr_sy)],
#                                  dtype=tf.bool if expr_sy.assumptions0.get('bool') else tf.float32)
#             return result
#
#         else:
#             expr_eval = expr_sy.evalf()  # standard 15 digits
#             if expr_sy.is_Boolean:
#                 return tf.constant(bool(expr_eval), dtype=tf.dtypes.bool)
#             elif expr_sy.is_number:  # is_float does not match int
#                 return tf.constant(float(expr_eval), dtype=tf.dtypes.float32)
#             else:
#                 raise NotImplementedError(f'Atom, but no bool or number? {expr_sy}')
#
#     else:  # Operator # len(expr.args) > 0:  # sfeh: line can be removed or replaced
#         if isinstance(expr_sy, sympy.Piecewise):
#             args_reversed = list(expr_sy.args[::-1])  # tuples to list
#             # whY WOULD ONE    reverse the args? --> NOT REVERSER
#             # reverse: most specific (/last) to lowest,  tuple must be nested the deepest node:
#             try:
#                 args_reversed = [[sympy_to_tensorflow(xx, d_tensors) for xx in list(ii)] for ii in args_reversed]
#             except TypeError as ex:
#                 print(f'lolololol {ex}')
#                 # [(2.07, True), (8.0, ITE(cartVel <= 0.34, 0.755*cartPos > 2.0, cartPos/(cartVel**2*sign(cartVel) + 0.866) > 2.0))]
#                 # (8.0, ITE(cartVel <= 0.34, 0.755*cartPos > 2.0, cartPos/(cartVel**2*sign(cartVel) + 0.866) > 2.0))
#                 # args_reversed = [[sympy_to_tensorflow(xx, d_tensors) for xx in list(ii)] for ii in args_reversed]
#                 # args_reversed = [[sympy_to_tensorflow(xx, d_tensors) for xx in list(ii)] for ii in args_reversed]
#                 raise
#
#             otherwise = args_reversed[0][0]  # the last "True" condition
#             for cet in args_reversed[1:]:
#                 otherwise = tf.where(cet[1], cet[0], otherwise)
#             return otherwise
#
#         elif isinstance(expr_sy, sympy.ITE):
#             raise NotImplementedError(f'Sfeh: sympy.ITE not yet implemented, occurs as side-effect of boolean logic')
#
#         tf_fun = totf[type(expr_sy)]
#         # try:
#         #     tf_fun = totf[type(expr)]
#         # except KeyError:
#         #     # sfeh:debug can this work??
#         #     #   - im(Round_Dummy(cartVel))
#         #     tf_fun = type(expr).tflow  # sfeh:debug-01.02 why does im come up here? (mut_br) im(Round_Dummy(cartPos))
#         #     # sfeh:idea exception, try to map sympy to tf function with same name (sympy.cos -> tf.cos)
#
#         tf_args = [sympy_to_tensorflow(a, d_tensors) for a in expr_sy.args]
#         # SFEH:Missing and Problems:
#         #   - Exception: eval-ex: type object 'cosh' has no attribute 'tflow'
#         #   - AttributeError: type object 'Round_Dummy' has no attribute 'tflow'
#         try:
#             result = tf_fun(*tf_args)  # fits, if the arguments match the expected arguments exactly Add(a, b)
#         except TypeError:
#             result = tf_args.pop()  # only commutative arity-2 functions here (Add, Mul, Max, Min)
#             while tf_args:
#                 # sfeh:optimization
#                 try:
#                     result = tf_fun(result, tf_args.pop())
#                 except Exception as ex:
#                     result = tf_fun(result, tf_args.pop())
#         return result
#     raise NotImplementedError(f'Cannot convert {expr_sy}')  # noqa: unreachable code, but was reached often while dev


# class ExperimentalBaseTree(NodeBase):
#     """
#     Is currently not in use and Experimental
#     states? sfeh:discuss
#     [None]: not set
#     [0]:    evolution/construction/build mode (potentially missing leaf nodes)
#     [1]:    structurally complete/finalized branch (node_depths correct, node_id set, ...)
#     [2]:    root-correct structure
#     [3]:    including meta-data (fitness_train, complexity)
#
#     # sfeh:discuss set tree depth at the end or so
#     # sfeh: set NodeBase as metatype and make this class (ExpressionTree, or so...) a new thing
#     """
#
#     args = []
#     is_fix = False  # sfeh:x here?
#
#     def __new__(cls, subcls, *args, **kwargs):
#         """
#         Why new instead of init?
#         -> https://docs.sympy.org/latest/tutorials/intro-tutorial/manipulation.html
#         " SymPy classes make heavy use of the __new__ class constructor, which, unlike __init__,
#         allows a different class to be returned from the constructor."
#         """
#         # if isinstance(cls, Operator):
#         #     pass
#         # elif isinstance(cls, TerminalNode):
#         #     if isinstance(cls, Symbol):
#         #         cls.fam, cls.time_index, _ = observation_get_family_and_time(args, none_return=None)
#         #         cls.index_minmax = None
#
#         obj = object.__new__(subcls)
#         obj.args = [arg for arg in args]
#         return obj
#
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls.args = kwargs.get('args', [])
#         cls.is_fix = kwargs.get('is_fix', [])
#
#     def __str__(self):
#         return self.get_str_recursive()
#
#     def get_str_recursive(self):
#         if isinstance(self, Operator):
#             childstr = ', '.join([str(x) for x in self.args])
#             _str = f"{self}({childstr})"
#         elif isinstance(self, TerminalNode):
#             _str = f'{self.value[0]}'
#         else:
#             raise NotImplementedError(f'sfeh:Specify exception. Class-type {type(self)}')
#
#         return _str
#
#     def __repr__(self):
#         return self.get_repr_recursive()
#
#     def get_repr_recursive(self):
#
#         _isfix = ', is_fix=True' if self.is_fix else ''
#         if isinstance(self, Operator):
#             childstr = ', '.join([repr(x) for x in self.args])
#         elif isinstance(self, TerminalNode):
#             childstr = self.args[0]
#         else:
#             raise NotImplementedError(f'sfeh:Specify exception. Class-type {type(self)}')
#
#         return f"{self.__class__.__name__}({childstr}{_isfix})"
#
#     def __len__(self):
#         return 1 + sum([len(cc) for cc in self.args])
#
#     def get_nclass(self):
#         return self.__class__.__name__


#######################################
# sfeh:idea check these options
#     import sympy
#     a, b = sp.symbols('a b')
#     expr = b - a**2 * a**3 + 2 + 3 * a * b**(-2)
#     [expr,
#      expr.as_expr(),
#      expr.as_poly(),
#      expr.as_base_exp(),
#      expr.as_coeff_add(),
#      expr.as_coeff_Add(),
#      expr.as_coeff_mul(),
#      expr.as_coeff_Mul(),
#      # expr.as_coeff_exponent(),
#      # expr.leadterm(),  # not working
#      # expr.subs(),
#      expr.as_coefficients_dict(),
#      expr.as_content_primitive(),
#      expr.as_dummy(),
#      expr.as_expr(),
#      expr.as_leading_term(),
#      expr.as_numer_denom(),
#      expr.as_two_terms(),
#      expr.as_independent(),
#      expr.expand(),
#      expr.factor(),
#      expr.assumptions0,
#      expr.normal(),
#      expr.nsimplify(),
#      # expr.extract_multiplicatively(),
#      # USEFUL
#      expr.atoms(),  # for leaf nodes
#      ]
###########################



if __name__ == '__main__':

    ns = {
        'a': sympy.Symbol('a', real=True),
        'b': sympy.Symbol('b', real=True),
        'c': sympy.Symbol('c', bool=True),
        'd': sympy.Symbol('d', bool=True),
    }

    # tensors = {
    #     'a': tf.constant([1.0, 2, 3, 4, 5, 6], dtype=tf.dtypes.float32),
    #     'b': tf.constant([-1.0, -2, -3, -4, -5, -6], dtype=tf.float32),
    #     'c': tf.constant([True, False, True, False, True, False], dtype=tf.dtypes.bool),
    #     'd': tf.constant([True, True, True, True, True, True], dtype=tf.dtypes.bool)
    # }` # sfeh are tensors actually better??

    tensors = {
        'a': [1.0, 2, 3, 4, 5, 6],
        'b': [-1.0, -2, -3, -4, -5, -6],
        'c': [True, False, True, False, True, False],
        'd': [True, True, True, True, True, True]}

    tst = [
        '5', '1', '0', '0.5', '-1', 'True', 'False',
        'c & True', 'c | False', '~c',
        'a<1', 'a<b', 'a<=b', 'a>=b', 'a>b', 'a==b', 'a!=b', 'a',
        # 'oo', 'zoo', 'I',
        'a + 1', 'a + 2', 'a*2', 'a - 2', 'a/2', 'a < 2', 'a**2', '2/a',
        'a*b*2', 'a+b+a+2+4', 'Min(a, b, 3)', 'Max(a, b, 4, a**2, a+b)', 'a<3',
        'Piecewise((a, c), (b, d), (a+b, True))',
        'Eq(4, 4.0)',
        # 'Square((Min(-2.176629, b) - Abs(a)))', 'Round(-123.333334234) + Round(b)',
        # 'Ifte(c, 1, 2)', '1 < Max(2, Ifte(1 < a, 1, 1))', 'Max(a+1, 2**(5-b))'
    ]
    tst_custom = [
        'Ifte(a, b, c)',
        '(((0.326675 * b_2) - c_9) + (Ifte((-c_9 < b_5), c_7, Ifte((Square(Gain_6) < Max(a_2, Ifte((c_9 < '
        'c_4), -Gain_3, Gain_5))), c_9, c_4))))',
        'Max(a, 2)',
        'Min(b, b)',
        'Ifte(True, a, 3)',
        'sin(asin(b))',
        'And(False, True)',
        'Add(-1.490149, 14.0)',
        'Ifte(False, (3+a), 3)',
        'Eq(4, 4.0)',
        'Lt(a, a)',
        'Or(Ne(False, False), False)',
        'sqrt(5 * a)',
        'Multiply(log(acos(-0.212976)), asin(2))',
        'Not(False)',
        'acos(0.5)',
        # 'Round(1.2345)',
        'Min(Ifte(True, Multiply(a, 20.0), acos(-0.5)), 1)',
        'Div(Add(13.159398, 19.284178), 1)',
        'Pow(a, b)',
        'Add(-2, Min(Ifte(True, 1, b), 8))',
        '(Or(True, True) & c)',
        'Ifte(And(False, True), b, 0.046948)',
        'Ifte(Ne(True, Lt(Sub(a, 2), 1)), 1, 2)',
        'cos(tan(Square(Multiply(Add(Round(Ifte(Ne(Ge(b, 15), True), 7, Sub(a, 16.5))), 5), 4))))',
        'Ifte(Lt(Ifte(Eq(Min(b, 1), 3), Max(a, b), b), 0), 0, 2)'
    ]

    def test_sympify():
        print('Running sympify example')

        for x in tst + tst_custom:
            sx = expr_sympify(x)
            print(sx)


    def print_relevant_subclasses():

        st = {}
        for x in get_subclasses(OperatorArity):
            try:
                st[f'sympy.{x.symfun.__name__}'] = x.__name__  # x.tflow.__name__
            except AttributeError as ex:
                print(f'Could not get {x}: {ex}')
                # st[x.__name__] = x.__name__
                pass
        st = ', '.join([f"{k}: {v}" for k, v in st.items()])
        print(f'sym2node = {{{st}}}')


    def all_typus_subclasses(cls=Node):
        sub = []
        for x in get_subclasses(cls):
            if len(x.__subclasses__()) > 0:
                pass
            else:
                # sub.append(x.__name__)
                sub.append(x)
        return sub

    # # print(all_typus_subclasses())
    # # print('===================')
    # # print_relevant_subclasses()
    #
    # # test_basic_tfconversion()  # sfeh all tests
    # # test_sympify()
    #
    # # x = Add(childs=[Symbol('a'), Mul(childs=[2, 3])])
    # # n1 = Float
    # # n2 = Symbol
    # # n3 = Boolean
    # # n4 = Add()
    # # # print(n1, n2, n3, n4)
    # # print(len(n1), len(n4))
    #
    # # for _subc in get_subclasses(BaseTree):
    # #     if  in _subc.__bases__:
    # #         # print(f'ignoring {_subc}')
    # #         pass
    # #     else:
    # #         print(f'{_subc.__name__}')
    # #
    # # print(type(RelationalOperator))
    # # print_relevant_subclasses()
    #
    # """
    # Alpha tests
    # """
    # # t1 = tb.invent_core_depth(float, 3, p_term=0.5)
    # # tree2 = tb.evolve_mutate_point(t1)
    # # t1 = tb.invent_core_depth(float, 3, p_term=0.1)
    # # t2 = tb.invent_core_depth(float, 3, p_term=0.1)
    # # t1, t2 = tb.evolve_crossover(t1, t2)
    # # for _ in range(5):
    # #     print('x.D', t1, '===', t2)
    # #     t1, t2 = tb.evolve_crossover(t1, t2)
    # #     print('x~D', t1, '===', t2)
    #
    # # nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    # # nstr = '["+:fix",["-:fix",["Ifte",["True"],["sin",["2"]],["/",["2.043"],["4"]]],["cartVel"]],["-1.3"]]'
    # # expr = 'Mul(2.5, cartVel)'
    # # expr = 'Ifte(Or(Lt(cartPos, cartPos), True), (Abs(cartVel) + cartVel/2), (Add(11.0, cartVel) / Sign(16.0)))'
    # # expr_sy = plagih_sympify(expr, eval_locals={'cartVel': sympy.Symbol('cartVel'), 'cartPos': sympy.Symbol('cartPos')})
    # # tree = sympy_to_tree(expr_sy, allow_chain=False)
    # tree = Ifte(False, 1, Add(Pow(1, 2), 'vel'))
    # print(tree)

    # sfeh: staticmethod for symfun?

    ndclasses = all_typus_subclasses()
    for c in ndclasses:
        if c in [ExprCondPair_Dummy]:
            continue

        if c in [DivFraction]:
            pass

        d = {float: lambda: np.random.random(),
             bool: lambda: np.random.choice([True, False])}
        try:
            if issubclass(c, OperatorChained):
                xtype_childs = c.xtype_chain
                xtype_me = c.xtype_chain
                inputs_sy = [d.get(xtype_childs)() for _ in range(np.random.choice([2, 3]))]
                inputs_np = np.array([[x] for x in inputs_sy])
            else:
                xtype_childs = c.xtype[0]
                xtype_me = c.xtype[1]
                inputs_sy = [d.get(x)() for x in xtype_childs]
                inputs_np = [np.array(x) for x in inputs_sy]
            symfun = c.symfun
            np_fun = c.np_fun
            res_sy = symfun(inputs_sy)  # symfun(inputs_sy), np_fun(*inputs_np)
            res_np = np_fun(*inputs_np)
            res_sy = xtype_me(res_sy)
            res_np = xtype_me(res_np)
            if abs(res_sy-res_np) < 0.0001:
                print(c.__name__, res_sy, res_np)
            else:
                print('FAILED!', c.__name__, res_sy, res_np, (res_sy-res_np), inputs_sy)
        except Exception as ex:
            if c in [ExprCondPair_Dummy, Piecewise, Boolean, Number, Symbol]:  # sfeh
                pass
            else:
                raise(c.__name__, 'Exception!', ex)
    # Example pandas DataFrame
    data = {
        'cartPos': [0, np.pi / 4, np.pi / 2, np.pi],
        'cartVel': [0.1, 0.2, 0.3, 0.4]
    }
    df = pd.DataFrame(data)
    tree = PowRounded(Number(3), Number(2))
    npex = tree.eval_np_debug(df)
    print(f'res: {npex}')
