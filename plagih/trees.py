"""
plagih_tree contain a new implementation of trees that we use in genetic programming to display a program.


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
    Also, make a case in sympy_to_tree to reconstruct trees from sympy expressions.
"""


import copy
import random
import warnings
from abc import ABC
from collections import deque

from matplotlib.patches import Patch

import pandas as pd
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.utilities.exceptions import ignore_warnings

from matplotlib.ticker import StrMethodFormatter

from plagih.paretofront import *
from plagih.tree_complexity.tree_edit_distance import *
from plagih.util import *

from typing import Optional, List, Union, Callable, Type, TypeGuard, Any, Dict, Tuple

from dataclasses import dataclass, field


np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


class RoundDummy(sympy.Function):  # Not a Math-operator
    """
    Workaround for rounding exponents
    For more details, look at: plagih/discoveries/rounding_exponents.py
    """
    @classmethod
    def eval(cls, a):
        try:
            if not isinstance(a, sympy.Basic):
                return sympy.Integer(round(a.evalf()))
                # return sympy.Integer(round(a))
            elif a.is_symbol:
                return  None
            elif a.is_number:
                return sympy.Integer(round(a.evalf()))  # type: ignore[attr-defined]
            else:
                return None
        except (TypeError, ZeroDivisionError) as imex:
            # TypeError('Argument of Integer should be of numeric type, got 2 - I.')
            #   -> 'asin(tan(1))' has imaginary part. sfeh: open
            raise SympyImaginaryNumber(imex)
        except Exception:
            raise CuriosityError

    def __call__(self, a):
        # if isinstance(a, (int, float, np.ndarray)):
        #     return np.round(a).astype(np.int64)
        if isinstance(a, (int, float)):
            return int(round(a))
        elif isinstance(a, np.ndarray):
            return np.vectorize(lambda v: int(round(float(v))))(a)
        raise TypeError("Unsupported type for numerical evaluation in RoundDummy")

    @staticmethod
    def np_round_dummy(x):
        """Exact numpy-equivalent of RoundDummy logic"""
        # return np.vectorize(lambda v: int(round(float(v))))(x)

        x = np.asarray(x, dtype=np.float64)
        # Banker's Rounding wie Python round/NumPy rint
        return np.rint(x)


@dataclass
class Node(ABC):
    """
    Subclasses: BaseOperator, Terminal, NodeDummy
    Recursively holds the nodes of a tree.
    Represents a node in a computation tree.
    Each node can evaluate its expression via sympy or NumPy and supports
    simplification and replacement operations.
    - `parent_node`: Pointer to the parent node. None implies this is the root node.
    - `childs`: List of child nodes, default is an empty list.
    - `is_fix`: Flag indicating whether the node structure is fixed (e.g., immovable in evolution).
    - `depth`: Current depth of the node. This should be updated if the structure changes.
    - `root_node`: Pointer to the root node for easy access throughout the tree.
    - `_loc`: (also _iloc/_xloc) for quickinfo about the coordination in the tree
    """
    # Tree-content
    symfun: Optional[Callable[..., sympy.Basic]]
    np_fun: Optional[Callable[..., np.ndarray]]
    showme: str = ""
    sy_str: str = ""  # String representation of the symbolic function.
    formulae_str: str = ""  # String representation of the formula.
    repr_str: str = ""  # Representation format string.
    xtype: tuple = ()
    xtype_chain: Union[bool, float] = False

    # Tree-structure
    childs: List[Union['Node', Any]] = field(default_factory=list)  # Terminal-Nodes speichern Werte statt Nodes
    is_fix: bool = False  # Whether the node is fixed in the structure.
    depth: Optional[int] = None
    # unused for now, preparation for backlinks
    root_node: Optional['Node'] = None
    parent_node: Optional['Node'] = None

    is_Atom = ...  # sympy-check

    def __init__(self, *args, **kwargs):
        self.childs = list(args)
        self.is_fix = kwargs.get('is_fix', False)
        self.depth = kwargs.get('depth', None)
        self.root_node = kwargs.get('root_node', None)
        self.parent_node = kwargs.get('parent_node', None)

    def __repr__(self):
        """
        Do NOT use __str__!
        This is acceptable, as it is never used anyways.
        only fixed nodes are missing
        """
        return self.represent_str(show_all=False)

    def __len__(self):
        return self.len_nodecount_fair()

    def __str__(self):
        # discuss which style for trees?
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
        # print(f'Compare prints:\n{s}\n{s1}\n{s2}')
        # s3 = self.get_expr_symlike(try_sympify=True)
        # s4 = self.get_sympy_expr()
        # s5 = self.get_expr_raw_fstring()
        # s6 = self.str_as_list()
        # s_export = self.get_tree_export()
        # print(f'{s}\n{s1}\n{s2}\n{s3}\n{s4}\n{s5}\n{s6}\n{s_export}')
        return s

    def add_child(self, child: 'Node') -> None:
        """Adds a child node and updates its parent_node reference."""
        self.childs.append(child)
        child.parent_node = self

    def set_depth(self, depth) -> None:
        """Updates the depth of the current node and propagates to child nodes."""
        self.depth = depth
        for child in self.childs:
            child.set_depth(depth + 1)

    def is_root(self) -> bool:
        """Checks if the current node is the root node."""
        return self.parent_node is None

    def set_parent(self, n):
        self.parent_node = n

    def set_root(self, n: 'Node'):
        self.root_node = n

    def get_childs(self) -> List['Node']:
        return self.childs

    def set_childs(self, child_list: Union[list, tuple]):
        if isinstance(child_list, (list, tuple)):
            # ccs = [cast_input(x) for x in child_list]
            # self.childs = ccs
            self.childs = list(child_list)
            if self.has_childs():
                for cc in self.get_childs():
                    cc.set_root(self)
                    cc.set_parent(self)  # set pointer in child-nodes
        else:
            raise TypeError(f'childs must be set as list, not {type(child_list)}: {child_list}')

    def repair_all(self, parent: 'Node'=None, root: 'Node'=None, depth:int=0):
        """backlink was introduced on 23.04.2024,
        linking the root and parent nodes"""
        self.root_node = root
        self.parent_node = parent
        self.depth=depth

        for ii, cc in enumerate(self.get_childs()):
            cc.repair_all(parent=self, depth=depth+1)

    def get_mutable_rootnodes(self, extend_lvls=2) -> Optional[list['Node']]:
        """Returns the list of first mutable nodes
        last_leaves: if you want so save all leave nodes aswell
        sum_layers=False, get_closest=True, return_all_layers=False

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
        this computes the depth and does not take advantage of saved depths"""

        if self.has_childs():
            max_depth = max(cc.get_max_depth(depth=depth + 1) for cc in self.get_childs())
        else:
            max_depth = depth

        return max_depth

    def is_operator(self) -> bool:
        return issubclass(type(self), BaseOperator)

    def is_term_and_symbol(self) -> bool:
        if self.is_term():
            a = issubclass(type(self), Symbol)
            if a:
                return True
        return False

    def is_term(self) -> bool:
        """is_term instead of operator due to fuckin exprCondPairs"""
        result = issubclass(self.__class__, Terminal)
        return result

    def get_typus(self) -> Type['Node']:
        nt = self.__class__
        return nt

    def has_childs(self) -> bool:
        """Explicitly"""
        # better to check for recursive use, as e.g. ExprCondPair is not a regular operator
        return not self.is_term()

    def repair_depth(self, depth: Optional[int] = None) -> None:
        """
        aka set_depth recursively for all nodes in a branch
        mainly used in branch
        The depth is written inevery node (for whatever reason), and instead of having to propagate
        the depth through every crossover/branch mutation function, instead, we call it when replacing nodes

        Can repair the depth starting from a random node!!
        """
        depth = depth or self.depth or 0
        self.depth = depth
        if self.has_childs():
            for cc in self.get_childs():
                cc.repair_depth(depth=depth + 1)

        return

    # def import_tree(self, input):
    #
    #     ...
    #     return

    def export_tree(self, cut_terms: bool = False) -> str:
        """
        Gibt den Baum als direkt ausführbare Konstruktor-Syntax aus:
          - Operator: Klassename(child1, child2, ...)
          - Terminals:
              Number(2.5), Boolean(True/False), Symbol(cartVel)
          - Fix-Flag: Klassename(..., is_fix=True)
        Die Ausgabe kann in Python eingefügt werden, sofern alle Klassen verfügbar sind.
        """

        def format_terminal(node_term: 'Terminal') -> str:
            """Best-practice for readability, but non-true representation
            - Mathematical expression-style (Max(1.49, 0.1*cartPos))
            - big values without decimals
            - small values with relevant decimals
            - Very small values as 0.001
            - booleans as True/False"""
            val = node_term.get_value()
            # Booleans
            if isinstance(node_term, Boolean):
                v = bool(val) if isinstance(val, (bool, sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)) else bool(
                    sympy.sympify(val))
                return f"Boolean({str(v)})"
            # Numbers
            if isinstance(node_term, Number):
                try:
                    v = float(val)  # noqa child[0] is the value in a Number-node
                except TypeError:
                    v = float(sympy.sympify(val).evalf())
                except Exception:
                    raise CuriosityError
                s = f"{v}"
                if cut_terms:
                    s = remove_trailing_zeroes(s)
                return f"Number({s})"
            # Symbols: Name ohne Quotes
            if isinstance(node_term, Symbol):
                # Wert kann SymPy-Symbol oder String sein
                name = str(val)
                return f"Symbol({name})"
            # Fallback: generisch
            return f"{type(node_term).__name__}({val})"

        def walk(node: 'Node') -> str:
            cls_name = type(node).__name__

            if is_terminal(node):
                out = format_terminal(node)
                # Fix-Flag für Terminals
                if node.is_fix:
                    if out.endswith(')'):
                        out = out[:-1] + ", is_fix=True)"
                return out

            # Kinder serialisieren
            children_str = ", ".join(walk(cc) for cc in node.get_childs())
            # Fix-Flag für Operatoren
            if node.is_fix:
                if children_str:
                    return f"{cls_name}({children_str}, is_fix=True)"
                else:
                    return f"{cls_name}(is_fix=True)"
            else:
                return f"{cls_name}({children_str})"

        return walk(self)

    def revoke_useless_nodes(self) -> None:
        """Mandatory Part of the basic simplification-process.
        E. g. when
            - chainable operators have too few operators: Mul(a) -> a
            - One child is the neutral element: Add(a, 0)
        """
        # Recursively simplify children
        # sfeh:if this only does things with multiplication, move this to multiplication
        if self.is_term():
            pass
        else:
            if isinstance(self, (Add, Mul)):
                if isinstance(self, Add):
                    for cc in self.get_childs():
                        if is_terminal(cc):
                            if cc.get_value() in [0, sympy.S.Zero, 0.0]:
                                childs_new = self.get_childs()
                                childs_new.remove(cc)
                                self.set_childs(childs_new)
                                raise CuriosityError
                elif isinstance(self, Mul):
                    for cc in self.get_childs():
                        if is_terminal(cc):
                            if cc.get_value() in [1, sympy.S.One, 1.0]:
                                childs_new = self.get_childs()
                                childs_new.remove(cc)
                                self.set_childs(childs_new)
                                # raise CuriosityError
                                # happened in crossover

            for child in self.get_childs():
                child.revoke_useless_nodes()

            arity = self.get_arity()
            num_childs = len(self.get_childs())

            # Check for simplifications based on arity
            if num_childs < arity:
                # e.g. Mul(a) -> a
                cc = self.get_childs()[0]
                self.set_new_node(cc)

            elif num_childs == arity:
                ...  # no action needed

            elif num_childs > arity:
                ... # no action needed for now, e.g., chained operators

        return


    def set_new_node(self, nd_new: 'Node', repair: bool = False, clean_chain: bool = True) -> None:
        """
        Replaces self with a new node/branch (including child nodes).

        Args:
            nd_new (-Node): The node to replace self with.
            repair (bool): Whether to repair depth and parent relationships.
            clean_chain (bool): Whether to remove unnecessary chain operators.
        """
        # Backup of old node, required for repair
        self_copy = copy.deepcopy(self)

        # Updating everything in the Node-class
        self.__class__ = nd_new.__class__
        self.__dict__.update(nd_new.__dict__)

        # debug_me = copy.deepcopy(self)

        if clean_chain:
            self.revoke_useless_nodes()

        # if len(debug_me) != len(self):
        #     pass  # this thing is useful, removing multiplications

        # Updating the structural Infos that have been updated with false Informations
        if repair:
            self.parent_node = self_copy.parent_node  # debug if parent are linked correctly
            self.root_node = self_copy.root_node  # sfeh:open recursively update
            self.depth = self_copy.depth

            self.repair_all()

        else:
            self.parent_node = None
            self.root_node = None
            self.depth = None

        # all BaseNode infos should not be updated
        pass

    def replace_with(self, new_class: Type['Node'], new_args: list) -> None:
        """Replace the current node with a simpler equivalent."""
        new_node = new_class(*new_args)  # Create new instance
        new_node.parent_node = self.parent_node  # Preserve parent reference
        if self.parent_node:
            self.parent_node.childs = [new_node if child is self else child for child in self.parent_node.childs]
        self.set_new_node(new_node)

    def replace_with_node(self, new_node: 'Node') -> None:

        new_node.revoke_useless_nodes()
        self.set_new_node(new_node)

        return

    def get_sympy_expr(self, simplimore: bool = False) -> sympy.Basic:
        """Converts directly into a sympy expr
        from node-class, go to a sympy expression
        """
        _sym = type(self).symfun
        _cs = self.get_childs()

        if self.is_term():
            _r = _sym(*_cs)

        elif isinstance(self, Piecewise):
            # _sym = sympy.Piecewise
            _cs = [(cc.get_childs()[0], cc.get_childs()[1]) for cc in _cs]
            _cs = [(cc[0].get_sympy_expr(simplimore=simplimore), cc[1].get_sympy_expr(simplimore=simplimore)) for cc in _cs]
            _cs = [ExprCondPair(cc[0], cc[1]) for cc in _cs]
            _r = _sym(*_cs)
        elif self.is_operator():
            _cs = [cc.get_sympy_expr(simplimore=simplimore) for cc in _cs]

            _r = _sym(*_cs)  # noqa (_sym is definitely assigned)
            # -> AttributeError: 'Xor' object has no attribute '_eval_as_set'
            # -> TypeError: Invalid comparison of non-real asin(15)
            try:
                _r = _sym(*_cs)  # noqa (_sym is definitely assigned)
                # -> AttributeError: 'Xor' object has no attribute '_eval_as_set'
                # -> TypeError: Invalid comparison of non-real asin(15)
            except TypeError as imex:
                raise SympyImaginaryNumber(imex)

        else:
            raise NotImplementedError

        sympy_expression_check_raise(_r)

        return _r

    def list_terminal_nodes(self) -> List['Node']:
        base = self.list_mutable_nodes()
        base = [x for x in base if x.is_term()]
        return base

    def force_input_node(self, ev: 'Evolution') -> None:
        """Some trees have only constants as terminals.
        This function replaces a terminal node with an input, if the tree only has constants"""
        node_list = self.list_terminal_nodes()
        a = [x.is_term_and_symbol() for x in node_list]
        if any(a):
            return
        else:
            node_list = [x for x in node_list if isinstance(x, Number)]  # only replaces numbers. ok for now.
            if not node_list:
                raise TreeError(f'No terminal nodes found to replace in tree: {self}')

            try:
                node = rnd_choice(node_list)  # debug if ignores chains
                xtype = xt_self(node.get_xtype_tuple())
                new_node = ev.node_selector.choose_symbol_node(xtype)
            except (ValueError, IndexError) as e:
                raise TreeError(f'Single-node tree,  no matching input found (probably boolean - ex: {e}): {self}')

            # node.set_new_node(new_node)
            node.set_new_node(new_node)

    def is_number(self) -> bool:
        return issubclass(type(self), Number)

    def str_as_list(self, cut_terms: bool = False) -> str:
        """Breath-first representation, I guess?: [Add, [Sin, [5]], [vel]]
        Useful to compare trees with prints"""
        typus_str = self.showme

        if self.get_childs():
            if issubclass(type(self), BaseOperator):
                childstr = ', '.join([cc.str_as_list(cut_terms=cut_terms) for cc in self.get_childs()])
                typus_str = f'{typus_str}, {childstr}'
            else:
                # terminal nodes
                v = self.get_childs()[0]
                try:
                    if self.is_number():
                        typus_str = term_format(f'{v}', cut=cut_terms)
                    else:
                        typus_str = f'{v}'
                except TypeError as ex:  # noqa  # sfeh delete 15.01.2026
                    v_eval = v.evalf()  # noqa
                    typus_str = term_format(v_eval, cut=cut_terms)
        else:
            raise CuriosityError('Holla sfeh')

        return f"[{typus_str}]"

    def get_lut_id(self) -> str:
        """
        Do NOT use str() or str_as_list(), as values get rounded
        Do NOT return a node-list and convert them to strings

        Unique+simple representation of a tree (to check in a lut if it was calculated already)
        regular print/string option should look better, this is just for getting a unique identifier
        returns string Identificator
        """

        s = self.represent_str(show_all=False)
        return s

    def str_as_expr(self) -> sympy.Basic:
        s = self.get_sympy_expr()
        return s

    def get_expr_symlike(self, try_sympify: bool = False, cut_terms: bool = False) -> str:
        """(1 + a)
        each step returns like ({} + {})
        and then, the inputs are filled in the gaps
        """
        if is_terminal(self):
            _expr = f'{self.get_childs()[0]}'
            _expr = term_format(_expr, cut=cut_terms)
            return f'{_expr}'
        else:
            childs = [cc.get_expr_symlike(try_sympify=try_sympify, cut_terms=cut_terms) for cc in self.get_childs()]
            # if issubclass(type(self), (ExprCondPair)):

            if isinstance(self, (Add, Mul, And, Or, Xor)):
                _expr = self.inline_sep.join(childs)
                _expr = f'({_expr})'

            else:
                _expr = self.sy_str.format(*childs)

        # Not working, requires handlich exprCondPair
        # if try_sympify:
        #     try:
        #         expr_sy = sympy.sympify(_expr)  # local dict required?
        #         return f'{expr_sy}'
        #     except Exception as ex:
        #         return f'{_expr}'

        return f'{_expr}'

    def list_mutable_nodes(self, xtype=None) -> List['Node']:
        """return all nodes that are mutable, aka suite for point- or branchmutation
        === ValueError: 'a' cannot be empty unless no samples are taken
        ===> probably, no nodes exist to be mutated
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
                node_list = [self]
            else:
                node_list = []

        # recursively add the other nodes
        if self.has_childs():
            for cc in self.get_childs():
                a = cc.list_mutable_nodes(xtype=xtype)
                node_list.extend(a)

        return node_list

    def get_all_nodes_visualize(self, setid: str):
        """returns all nodes in a tree as list
        a+1 -> [+a1, a, 1]"""
        showme = f'{self.childs[0]}' if self.is_term() else f'{self.showme}'

        res = {setid: {'node': self,
                       'showme': showme}}
        edges = []

        if self.is_term():
            pass
        else:
            for ii, cc in enumerate(self.childs):
                cid = f'{setid}-{ii}'
                cr, ce = cc.get_all_nodes_visualize(cid)
                res.update(cr)
                edges.append((setid, cid))
                edges.extend(ce)

        return res, edges

    def get_apted_notation(self) -> str:
        """Calculating the TED requires this (weird) representation"""
        return f"{{{self.get_typus()}{''.join([cc.get_apted_notation() for cc in self.get_childs()])}}}"

    def evolve_mutate_filter_gauss(self) -> None:
        """Recursively filter the nodes in the branch of fintree
        random filter all terminal nodes /
                single node /
                nodes in a branch /
                random nodes in a branch /
                intelligent filtering

        """
        if self.has_childs():
            for cc in self.get_childs():
                cc.evolve_mutate_filter_gauss()

        else:
            if is_number(self):
                val = (round(random.gauss(self.get_value(), 0.1), FLOAT_PRECISION))

               #  self.childs[0] = val only this used to work
                self.set_value(val)

        return

    def tree_node_grouping(self, tolerance: float = 0) -> None:
        """
        If possible, this groups nodes to a simpler expression (if possible).
        E.g. a ** 2 -> square(a)
             a + b + c -> sum(a, b, c)  (chaining)

        # sfeh:idea Heavyside function. input a val, input b threshold


        sfeh:idea expr = b + a + a
            terms = expr.as_ordered_terms()
        """

        if issubclass(self.__class__, Terminal):  # good for runtime

            if is_number(self) and tolerance > 0:
                val = self.get_value()
                # from sympy.physics.units import speed_of_light, meter, second more ideas!

                # VERY USEFUL: strg+TribonacciConstant to go to init with useful info
                # sympy.pi,         sympy.GoldenRatio,  sympy.Catalan,      sympy.EulerGamma,   sympy.TribonacciConstant
                # 3.14159265358979, 1.61803398874989,   0.915965594177219,  0.577215664901533,  1.83928675521416
                # idea sympy.nsimplify('3.333333*x+0.522', tolerance=0.1, rational=True) for
                for const in [sympy.pi, sympy.GoldenRatio, sympy.Catalan, sympy.EulerGamma, sympy.TribonacciConstant]:
                    if (const-val) < tolerance:
                        self.childs[0] = const
                        return
                # [sympy.S.One, sympy.S.Half, sympy.S.NegativeOne, sympy.S.NegativeHalf]:
                # sfeh: ALso take care of sqrt(2)
                # -> rescale of all input variables (no, also dont make them too big)
                if val - sympy.nsimplify(val, tolerance=tolerance, rational=True) < tolerance:
                    val_new = sympy.nsimplify(val, tolerance=tolerance, rational=True)
                    self.set_value(val_new)

            return

        elif self.is_operator():

            mychlds = self.get_childs()

            if isinstance(self, Usub) and len(mychlds) > 1:
                print(f"ERROR: Usub should only have one child, but got {len(mychlds)}")

            for cc in mychlds:
                cc.tree_node_grouping(tolerance=tolerance)

            """
            Save replacements
            """
            if isinstance(self, (Pow, PowRounded)):
                p_base = mychlds[0]
                p_exp = mychlds[1]

                if isinstance(p_exp, Number):
                    p_exp_v = p_exp.get_value()

                    if p_exp_v in [1, sympy.S.One]:  # x**1 → x
                        self.replace_with_node(p_base)

                    elif p_exp_v in [-1, sympy.S.NegativeOne]:  # x**-1 → 1/x
                        self.replace_with(DivFraction, [p_base])

                    elif p_exp_v == 2:  # x**2 → Square(x)
                        self.replace_with(Square, [p_base])

                    elif p_exp_v == -2:  # x**-2 → 1/(x**2)
                        self.replace_with(DivFraction, [Square(p_base)])

                    elif p_exp_v in [0.5, sympy.S.Half]:  # x**0.5 → sqrt(x)
                        self.replace_with(Sqrt, [p_base])

                    elif p_exp_v < 0 and p_exp_v % 1 == 0:
                        # x**-n → 1/(x**n) für ganzzahliges negatives n
                        self.replace_with(DivFraction, [PowRounded(p_base, Number(abs(p_exp_v)))])

                    elif p_exp_v > 0 and p_exp_v % 1 == 0:
                        # x**n bleibt gleich, aber wird in PowRounded umgewandelt
                        self.replace_with(PowRounded, [p_base, p_exp])

                    # elif p_exp_v > 0 and (1 / p_exp_v) % 1 == 0:
                    #     # x**(1/n) → n-te Wurzel(x) (z. B. x**(1/3) → Kubikwurzel(x))
                    #     root_degree = int(1 / p_exp_v)
                    #     self.replace_with(NthRoot, [p_base, Number(root_degree)])
                    #
                    # elif p_exp_v < 0 and (1 / abs(p_exp_v)) % 1 == 0:
                    #     # x**(-1/n) → 1/(n-te Wurzel(x))
                    #     root_degree = int(1 / abs(p_exp_v))
                    #     node_sub = NthRoot(p_base, Number(root_degree))
                    #     self.replace_with(DivFraction, [node_sub])

                    elif p_exp_v > 0 and (1 / p_exp_v) % 1 == 0:
                        # x**(1/n) → x**(1/n) bleibt als Potenz stehen (z. B. x**(1/3))
                        root_degree = int(1 / p_exp_v)
                        self.replace_with(PowRounded, [p_base, Number(sympy.Rational(1, root_degree))])

                    elif p_exp_v < 0 and (1 / abs(p_exp_v)) % 1 == 0:
                        # x**(-1/n) → 1/(x**(1/n))
                        root_degree = int(1 / abs(p_exp_v))
                        node_sub = PowRounded(p_base, Number(sympy.Rational(1, root_degree)))
                        self.replace_with(DivFraction, [node_sub])

                elif isinstance(p_exp, Round):  # Wenn der Exponent gerundet wurde
                    deep_child = p_exp.childs[0]
                    self.replace_with(PowRounded, [p_base, deep_child])

            elif isinstance(self, Mul):  # MulChain

                denominators = []
                for cc in mychlds:

                    # Removes the one factor from the childs, that is matched
                    mychlds_remove = lambda el: [x for x in mychlds if x != el]

                    # if isinstance(cc, DivFraction):
                    #     has_div_frac = [isinstance(ix, DivFraction) for ix in mychlds]
                    #     if sum(has_div_frac) > 1 or len(mychlds) > 2:  # this makes chained mul to 1/(mul( )) -> check if more than 2 inputs
                    #         continue  # leave them alone
                    #     denominators.append(cc.get_childs()[0])
                    #     # e. g.: "a * 1/3" -> "3/a"
                    #     div_by = cc.get_childs()[0]
                    #     node_sub = Mul(*mychlds_remove(cc))
                    #     self.replace_with(Div, [node_sub, div_by])
                    # el
                    if isinstance(cc, Number):
                        mul1 = cc.get_value()
                        if mul1 in (1, sympy.S.One):
                            raise CuriosityError  # "revoke_useless_nodes" should remove

                        elif mul1 in (-1, sympy.S.NegativeOne):
                            self.replace_with(Usub, [Mul(*mychlds_remove(cc))])
                        elif 0 < mul1 < 1:
                            if (1 / mul1) % 1  == 0:  # check if the result is a natural number
                                node_sub = Mul(*mychlds_remove(cc))
                                new_num = (1 / mul1)
                                self.replace_with(Div, [node_sub, Number(new_num)])
                        else:
                            # ScaleNode-idea here
                            pass
                    elif isinstance(cc, DivFraction):
                        has_div_frac = [isinstance(ix, DivFraction) for ix in mychlds]
                        if sum(has_div_frac) > 1 or len(mychlds) > 2:  # this makes chained mul to 1/(mul( )) -> check if more than 2 inputs
                            continue  # leave them alone
                        denominators.append(cc.get_childs()[0])
                        # e. g.: "a * 1/3" -> "3/a"
                        div_by = cc.get_childs()[0]
                        node_sub = Mul(*mychlds_remove(cc))
                        self.replace_with(Div, [node_sub, div_by])
                    else:
                        continue  # make sure to skip the following return statement

                    # self.revoke_useless_nodes()  # clean Mul-junk here
                    # IMPORTANT! EXIT this loop, if one factor was found
                    return

        elif issubclass(type(self), ExprCondPair_Dummy):
            pass

        else:
            raise NotImplementedError
        # # Wenn Divisionen erkannt wurden → baue Bruchstruktur
        # if denominators:
        #     numerator_expr = Mul(*numerators, evaluate=False) if numerators else Number(1)
        #     denominator_expr = Mul(*denominators, evaluate=False) if len(denominators) > 1 else denominators[0]
        #     self.replace_with(Div, [numerator_expr, denominator_expr])
        #     return
        #
        # # Falls nach Entfernung von `1` nur noch ein Faktor bleibt
        # remaining = numerators
        # if len(remaining) == 1:
        #     self.replace_with_node(remaining[0])
        return  # two indents

    def len_nodecount_raw(self) -> int:
        """counting the amount of nodes recursively"""
        if self.has_childs():
            return 1 + sum([cc.len_nodecount_raw() for cc in self.get_childs()])
        else:
            return 1  # childs can currently be floats

    def is_typus(self, nt: Type['Node']) -> bool:
        """Check Node-type"""
        r = issubclass(type(self), nt)
        return r

    def is_ExprCdPair(self) -> bool:  # noqa
        #  Sometimes, it hits the Dummy-class, sometimes (chained?) the sympy class. Not sure, why.
        r = issubclass(type(self), (ExprCondPair, ExprCondPair_Dummy))
        return r

    def len_nodecount_fair(self) -> int:
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

    def get_arity(self) -> int:
        return len(self.get_typus().get_child_xts())

    def get_xtype_tuple(self) -> tuple:
        return self.xtype

    def get_xtype_self(self) -> Union[Type[float], Type[bool]]:
        return self.xtype[1]

    def represent_str(self, show_all: bool = True, cut_terms: bool = False) -> str:
        """Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to str(), but adds the following information:
        - ":fix", when nodes are fixed"""

        s = self.showme  # class name

        if self.is_term():
            cs = f'{self.get_childs()[0]}'
            if cut_terms:
                cs = term_format(cs, cut=cut_terms)
            else:
                cs = remove_trailing_zeroes(cs)

            s = f'{cs}'

            if self.is_fix and show_all:
                s += ':fix'  # discuss: there must be a more natural way to show that...

        else:
            cs = [cc.represent_str(show_all=show_all, cut_terms=cut_terms) for cc in self.get_childs()]
            cs = ', '.join(cs)
            s = f"{s}({cs})"

        s = f"{s}"  # v1

        return s

    def get_symstr(self) -> str:
        return self.symfun.__name__

    @classmethod
    def get_child_xts(cls) -> tuple:
        return cls.xtype[0]

    def eval_np_lambdas(self, *args) -> Callable[[pd.DataFrame], np.ndarray]:
        """Eager evaluation with NumPy arrays
            - Building a graph without evaluation
            - Does not calculate useless stuff
            But: harder for debugging!
        """
        raise NotImplementedError(f"eval_np_lambdas not implemented for {type(self).__name__}")

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        """Evaluates children first, then applies own numpy function
        aka Debugging-version of eval_np"""
        raise NotImplementedError(f"eval_predict_numpy_now not implemented for {type(self).__name__}")


def eval_parsimony(tree: Node, complexity_measure: str, origin_tree: Optional[Node] = None) -> int:
    """
    very inefficient to always get origin-tree apted
        -> introduce own complexity measure, do not import...
    """
    if complexity_measure == 'tree_node_count_raw':  # number of nodes
        return tree.len_nodecount_raw()
    elif complexity_measure == 'tree_node_count_fair':
        return tree.len_nodecount_fair()
    elif complexity_measure == 'tree_edit_distance':  # tree_edit_distance, fintree-edit-distance
        apted1 = tree.get_apted_notation()
        apted2 = origin_tree.get_apted_notation()
        distance, mapping = apted_distance(apted1, apted2)
        return distance
    else:
        raise Exception(f'Complexity measurement not available: {complexity_measure}')


def sympy_to_tree(s_expr: sympy.Basic, allow_chain: bool) -> Node:
    """
    tree_from_expr, tree_from_sympy
    Important: start with the most specific rule

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

        if isinstance(s_expr, RoundDummy):
            _n = Round(cc_nodes[0])
            return _n

        elif allow_chain:
            op = d_sym2node_chain[type(s_expr)]
            _n = op(*cc_nodes)
            return _n

        elif isinstance(s_expr, ExprCondPair):
            return ExprCondPair_Dummy(cc_nodes)

        elif isinstance(s_expr, sympy.Piecewise):
            # "Chained_VERSION" version is handled before
            # Baue Piecewise von hinten nach vorne auf (letztes Paar wird innerster Else-Zweig)
            pairs = [(sympy_to_tree(cond, allow_chain=allow_chain),
                      sympy_to_tree(ceex, allow_chain=allow_chain))
                     for ceex, cond in s_expr.args]

            otherwise = pairs[-1][1]  # Last pair: (True, expr) → expr is else
            for cond, expre in reversed(pairs[:-1]):
                otherwise = Ifte(cond, expre, otherwise)
            return otherwise

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

        sympy_expression_check_raise(s_expr)  # not required anymore? checks earlier
    raise NotImplementedError(f'Expr missing: {s_expr}')


def tree_simplification(tree: Node, allow_chain: bool) -> Node:
    """
    (Tries to) simplify/sympify/unify/mathematically-reduce a tree. It is quite experimental
    # sfeh sympy-reconstruct patterns
    #   map symoy-sign to a sum
    #   map piecewise to if-then-else
    #   map power fractal - to sqrt?
    """
    tree_history = [copy.deepcopy(tree)]
    expr_sym = tree.get_sympy_expr()
    # expr_sym2 = sympy.simplify(expr_sym)
    # if str(expr_sym) != str(expr_sym2):
    #     print(f'okke: {expr_sym} // {expr_sym2}')
    # expr_sym3  = tree.get_sympy_expr(simplimore=True)
    tree = sympy_to_tree(expr_sym, allow_chain=allow_chain)
    for _ in range(10):
        tree_history.append(copy.deepcopy(tree))
        tree.tree_node_grouping(tolerance=0)
        if _ == 6:
            print(tree_history)
            raise CuriosityError
        if str(tree) == str(tree_history[-1]):
            break

    # print(f'Tree updates\n'
    #       f'\t{tree_copy.represent_str(show_all=False)}\n'
    #       f'a\t{tree_a.represent_str(show_all=False)}\n'
    #       f'b\t{tree_b.represent_str(show_all=False)}\n'
    #       f'c\t{tree_c.represent_str(show_all=False)}')
    # if tree_b.represent_str(show_all=False) != tree_c.represent_str(show_all=False):
    #     raise CuriosityError
    # sfeh:discuss
    # print(f'After simplification:  {len(tree)}\t{tree}')

    if len(tree_history[0]) < len(tree):
        astr = string_remove_trailing_zeroes(str(tree_history[0].get_sympy_expr()))
        bstr = string_remove_trailing_zeroes(str(tree.get_sympy_expr()))
        print(f'WHATHAPPENED SFEH\t{astr}')
        for ii, tt in enumerate(tree_history):
            print(f'\t{ii}:\t{tt.str_as_list()}')

        if astr != bstr:
            # sfeh 'a**0.5' does not become 'sqrt(a)'! use rational=True or sympy.S.Half
            print_warning('w', f'Diff in sympy expression?\n\t{astr}\n\t{bstr}')  # raise ex? does not occur after grouping?
            # sfeh Example 03.05.2025
            # 	sin(cartVel**(6450000000*RoundDummy(cartVel))/(cartPos**6450000000*cartVel**6450000000))
            # 	sin(cartVel**(6.45e+9*RoundDummy(cartVel))/(cartPos**6450000000*cartVel**6450000000))
    return tree

def evolve_reduce_simplicate(tree: Node, allow_chain: bool, completely: bool = True, force: bool = False) -> Node:
    """Reducing a fintree to its most basic form with sympify.
    (completely = False: reduce just one branch. if you wanted to have more complexity)"""
    tree_copy = copy.deepcopy(tree)
    if completely:  # reduce the complete tree
        nodes_lv0 = tree.get_mutable_rootnodes(extend_lvls=0)  # only required for fixed-core trees
        for cc in nodes_lv0:
            cc2 = tree_simplification(cc, allow_chain)
            cc.set_new_node(cc2)
    else:
        node_list = [n for n in tree.list_mutable_nodes() if issubclass(n.get_typus(), OperatorArity)]  # ignoring leaf nodes...
        if len(node_list) == 0:
            print_warning('wwww', f'Tree for simplification does not provide operators: {tree}')
            return tree
        node = random.choice(node_list)
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


class CustomOperator:
    pass


class NodeWithChilds(Node):
    ...


class NodeDummy(NodeWithChilds):
    """Terminal_Dummy, Function_Dummy now both in here"""
    # @classmethod
    # def get_child_xts(cls):
    #     return cls.xtype[0]
    pass


class PleaseUsePartnerOp(NodeDummy):
    """Those operator-classes should be withdrawn, as they do not add further information
    E. g. Ge, Gt,
    such that only <, <=, ==, != exist and >, >= are not actually used
    """
    pass


class BaseOperator(NodeWithChilds):

    chain: Optional[bool] = None
    is_Atom = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain = kwargs.get('chain')

    def set_chain(self, param: bool) -> None:
        self.chain = param


    def get_np_child_now(self, _df: pd.DataFrame, *args) -> List[np.ndarray]:
        # Kinder auswerten und als ndarray zurückgeben (keine Konvertierung/Reshape)
        values = [cc.eval_predict_numpy_now(_df, *args) for cc in self.get_childs()]

        if any((not isinstance(v, np.ndarray)) or (getattr(v, "ndim", 0) == 0) for v in values):
            values = [np.atleast_1d(np.asarray(v)) for v in values]  # debug why

        return values

    def get_np_child_lambdas(self, *args) -> List[Callable[[pd.DataFrame], np.ndarray]]:
        """Returning list of callables for each child node"""
        ccl = [cc.eval_np_lambdas(*args) for cc in self.get_childs()]
        
        return ccl

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        children = self.get_np_child_now(_df, *args)
        try:
            result = self.np_fun(*children)
        except TypeError as sfeh:
            if 'loop of ufunc does not support argument' in str(sfeh):
                # TypeError: loop of ufunc does not support argument 0 of type int which has no callable sin method
                print('ASDASDASDASD TODO')
            raise TypeError(ex)

        return result

    def eval_np_lambdas(self, *args):
        """
        Evaluate the node using NumPy and return a callable function.
        Args:
            *args: Additional arguments that can be used in evaluation.
        """

        child_lambdas = self.get_np_child_lambdas(*args)

        def node_lambda(_df: pd.DataFrame) -> np.ndarray:
            """
            The function that will be returned by eval_np().
            It applies np_fun to evaluated children given an input NumPy array.
            """
            child_values = [func(_df) for func in child_lambdas]
            for v in child_values:
                if not isinstance(v, np.ndarray):
                    raise

            result = self.np_fun(*child_values)
            return result

        return node_lambda

class OperatorArity(BaseOperator):

    arity = None


# class OperatorChained(BaseOperator):
#     # no xtype, only input type
#     # no tflow, separate handling in totf-function
#     # Piecewise, AddChain, MulChain, MinChain, MaxChain, AndChain, OrChain
#     # childs_min_max = [1, 5]
#
#     # def get_arity_class(self):
#     #     dct = {AddChain: Add,
#     #            MulChain: Mul,
#     #            MinChain: Min,
#     #            MaxChain: Max,
#     #            Piecewise: Ifte,
#     #            AndChain: And,
#     #            OrChain: Or,
#     #            XorChain: Xor}
#     #     p_op = dct[type(self)]
#     #     return p_op
#     ...


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
    xtype_input = None


class MathOperator(BaseOperator):
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


class BaseMinMax(MathOperator):
    pass


class NoSymCapitalized:
    """Does nothing, but maybe its good to know, which OG-sympy classes are lower case
    E. g. Sin() is sin() in sympy (?)
    """
    pass


class Terminal(Node):
    """Sympy.Atom equivalent
    Terminal nodes are leaf nodes which can not have children. e.g.:
    - constants (e.g. 2.3)
    - observations (e.g. b, aka data input)

    _np_lambda_cache: Currently NOT used and not working
      -> Problem: eval_np_lambdas() was called twice of eval_predict_numpy_now()
      -> Solution: Cache only within one of those calls?
      -> Performance improvement not very significant anyways
      -> Caching (as presented) cached wrong values
    """
    is_Atom = True
    _value: Union[sympy.AtomicExpr, float, bool]

    def __len__(self):
        return 1

    def get_value(self) -> Union[sympy.AtomicExpr, float, bool]:
        _val = self.get_childs()[0]
        return _val  # type: ignore[return-value]

    def set_value(self, val: Union[sympy.AtomicExpr, float, bool]) -> None:
        self._value = val
        self.childs = [val]  # Terminal-Nodes haben keine Node-Kinder, sondern Werte

class Boolean(Terminal):
    xtype = ((), bool)
    symfun = staticmethod(lambda *a: sympy.S.true if a[0] else ~sympy.S.true)  # sympy.logic.boolalg.Boolean
    np_fun = staticmethod(lambda x: bool(x))
    showme = 'Boolean'

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        val = bool(self.get_value())
        return np.full(len(_df), val, dtype=np.bool_)

    def eval_np_lambdas(self, *args):
        val = bool(self.get_value())
        return lambda _df: np.full(len(_df), val, dtype=np.bool_)  # noqa: PLW0640


class Number(Terminal):
    xtype = ((), float)
    symfun = staticmethod(lambda *a: sympy.Float(float(a[0]), FLOAT_PRECISION))
    # np_fun = np.array
    np_fun = staticmethod(lambda x: float(x))
    # sympy.Rational(0.1) -> 3602879701896397/36028797018963968
    # sympy.Rational('0.1') -> 1/10
    showme = 'Number'
    # sfeh: problem with rational: Sqrt(8.0) -> 2*sqrt(6)/3. : actually is_atomic?
    # tflow = lambda a: tf.constant(a, dtype=tf.float32)
    # def eval_predict_numpy_now(self, df, *args) -> np.ndarray:
    #     return np.full(df.shape[0], float(self.get_value()), dtype=np.float64)
    #

    # def get_value(self) -> Union[sympy.AtomicExpr, float]:
    #     return self._value

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        val = float(self.get_value())
        return np.full(len(_df), val, dtype=np.float64)

    def eval_np_lambdas(self, *args):
        return lambda _df: np.full(_df.shape[0], float(self.get_value()), dtype=np.float64)

    def __float__(self):
        return float(self.get_value())


class Symbol(Terminal):
    """
    The Symbol is set with sympy.Symbol() on creation in child[0]
    This was used to deal with negative values
        self.name = nlabl if nlabl[0] != '-' else nlabl[1:]
    """
    symfun = staticmethod(lambda *a: a[0])  # no symbol-conversion; no reason for.
    np_fun = None
    xtype = ((), float)
    showme = 'Symbol'

    # def get_value(self) -> Union[sympy.AtomicExpr]:
    #     return self._value

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        return _df[str(self.get_value())].to_numpy(dtype=np.float64)

    def eval_np_lambdas(self, *args):
        name = str(self.get_value())
        return lambda _df: _df[name].to_numpy(dtype=np.float64)


# def cast_input(value: Any):
#     if isinstance(value, Node):
#         return value
#     else:
#         # # For "human" inputs like Add(1, 'var')
#         # if isinstance(value, (sympy.logic.boolalg.Boolean, bool)):
#         #     return Boolean(value)
#         # elif isinstance(value, (sympy.Number, float, int)):
#         #     return Number(value)
#         # elif isinstance(value, str):
#         #     return Symbol(value)
#         # elif isinstance(value, sympy.Symbol):
#         #     # return value
#         #     raise NotImplementedError
#         # else:
#         #     raise NotImplementedError
#         return value


class Add(MathOperator, ChainableOp):
    """Addition operator for two or more operands."""
    symfun = staticmethod(lambda *a: sympy.Add(*a))
    np_fun = staticmethod(lambda *a: np.sum(np.stack(a), axis=0))
    showme = 'Add'
    sy_str = '({0} + {1})'
    formulae_str = '({} + {})'
    repr_str = 'Add{},[{},{}]'
    xtype = ((float, float), float)
    symfun_chain = lambda a: sympy.Add(*a)
    np_fun_chain = lambda *a: np.sum(*a)  # n
    sy_str_chain = 'Add({})'
    formulae_str_chain = 'Add({})'
    repr_str_chain = 'AddChain{},[{}]'
    inline_sep = ' + '
    xtype_chain = ([(float,)], float)
    xtype_input = float


class Mul(MathOperator, ChainableOp):
    """np.multiply ONLY for pairwise multiplication!"""
    symfun = staticmethod(lambda *args: sympy.Mul(*args))
    np_fun = staticmethod(lambda *a: np.prod(np.stack(a), axis=0))
    showme = 'Mul'  #
    sy_str = '({0} * {1})'
    repr_str = 'Mul{},[{}, {}]'
    xtype = ((float, float), float)
    symfun_chain = lambda a: sympy.Mul(*a)
    sy_str_chain = 'Mul({})'
    repr_str_chain = 'MulChain{},[{}]'
    inline_sep = ' * '
    xtype_chain = ([(float,)], float)
    xtype_input = float


class DivFraction(MathOperator):
    """x**-1
    aka InverseFraction aka DivFraction aka Reciprocal

    From https://numpy.org/doc/2.1/reference/generated/numpy.reciprocal.html
        ~ "This function is not designed to work with integers."

        np.reciprocal(2)  # Output: 0
        np.reciprocal(2.0)  # Output: 0.5
    """
    xtype = ((float,), float)
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], sympy.S.NegativeOne))
    np_fun = staticmethod(lambda a: np.reciprocal(a))
    showme = 'DivFraction'
    sy_str = '1/({})'
    repr_str = 'DivFraction{},[{}]'


class NthRoot(MathOperator):
    """Repräsentiert die n-te Wurzel: NthRoot(x, n) → x**(1/n)
    currently untested"""

    xtype = ((float, float), float)  # (Basis, Wurzelgrad) → float
    symfun = staticmethod(lambda *a: sympy.root(a[0], a[1]))  # SymPy Wurzel-Funktion
    np_fun = staticmethod(lambda base, n: np.power(base, 1 / n))  # NumPy Power-Funktion
    showme = 'NthRoot'
    sy_str = 'root({}, {})'
    repr_str = 'NthRoot{},[{}, {}]'


class Pow(MathOperator):
    # symfun = lambda *a: sympy.Pow(a[0], a[1])
    # np_fun = np.power
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], a[1]) if len(a) == 2 else None)
    np_fun = staticmethod(lambda base, exp: np.power(base, exp))
    showme = 'Pow'
    sy_str = '({0})**({1})'
    repr_str = 'Pow{},[{},{}]'
    xtype = ((float, float), float)


class Abs(MathOperator):
    symfun = staticmethod(lambda *a: sympy.Abs(a[0]))
    np_fun = np.absolute  # np.fabs works only for non-complex numbers
    showme = 'Abs'
    sy_str = 'Abs({})'
    repr_str = 'Abs{},[{}]'
    xtype = ((float,), float)


class Sign(MathOperator, NoSymCapitalized):
    # does not work in string, but irrelevant. sympy.simplify('sign(-a)') -> -sign(a)
    symfun = staticmethod(lambda *a: sympy.sign(a[0]))
    np_fun = np.sign
    showme = 'Sign'
    sy_str = 'sign({})'
    repr_str = 'Sign{},[{}]'
    xtype = ((float,), float)


class Log(MathOperator, NoSymCapitalized):
    # Log isactually Ln (base e). Log/Ln is the same, idk fuck Log10
    # discuss: Log-operator + abs/max so inputs are >0
    symfun = staticmethod(lambda *a: sympy.log(a[0]))
    np_fun = np.log
    showme = 'Log'
    sy_str = 'log({})'
    repr_str = 'Log{},[{}]'
    xtype = ((float,), float)


class Cos(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.cos(a[0]))
    np_fun = np.cos
    showme = 'Cos'
    sy_str = 'cos({})'
    repr_str = 'Cos{},[{}]'
    xtype = ((float,), float)


class Sin(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.sin(a[0]))
    np_fun = np.sin
    showme = 'Sin'
    sy_str = 'sin({})'
    repr_str = 'Sin{},[{}]'
    xtype = ((float,), float)


class Tan(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.tan(a[0]))
    np_fun = np.tan
    showme = 'Tan'
    sy_str = 'tan({})'
    repr_str = 'Tan{},[{}]'
    xtype = ((float,), float)


class Acos(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.acos(a[0]))
    np_fun = np.arccos  # arccosh
    showme = 'Acos'
    sy_str = 'acos({})'
    repr_str = 'Acos{},[{}]'
    xtype = ((float,), float)


class Asin(Trigonometry, NoSymCapitalized):
    """"""
    symfun = staticmethod(lambda *a: sympy.asin(a[0]))
    np_fun = np.arcsin
    showme = 'Asin'
    sy_str = 'asin({})'
    repr_str = 'Asin{},[{}]'
    xtype = ((float,), float)


class Atan(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.atan(a[0]))
    np_fun = np.arctan
    showme = 'Atan'
    sy_str = 'atan({})'
    repr_str = 'Atan{},[{}]'
    xtype = ((float,), float)


class Tanh(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.tanh(a[0]))
    np_fun = np.tanh
    showme = 'Tanh'
    sy_str = 'tanh({})'
    repr_str = 'Tanh{},[{}]'
    xtype = ((float,), float)


class Sinh(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.sinh(a[0]))
    np_fun = np.sinh
    showme = 'Sinh'
    sy_str = 'sinh({})'
    repr_str = 'Sinh{},[{}]'
    xtype = ((float,), float)


class Cosh(Trigonometry, NoSymCapitalized):
    symfun = staticmethod(lambda *a: sympy.cosh(a[0]))
    np_fun = np.cosh
    showme = 'Cosh'
    sy_str = 'cosh({})'
    repr_str = 'Cosh{},[{}, {}]'
    xtype = ((float,), float)


class Not(LogicOperator):
    """not"""
    symfun = staticmethod(lambda *a: sympy.Not(a[0]))
    np_fun = np.logical_not
    showme = 'Not'
    sy_str = '~({})'
    repr_str = 'Not{},[{}]'
    xtype = ((bool,), bool)


class Eq(RelationalOperator):
    """a == b"""
    symfun = staticmethod(lambda *a: sympy.Eq(a[0], a[1]))
    np_fun = np.equal
    showme = 'Eq'  # '==' not working in sympy!
    sy_str = 'Eq({0}, {1})'
    repr_str = 'Eq{},[{}, {}]'
    xtype = ((float, float), bool)


class Ne(RelationalOperator):
    """a != b"""
    symfun = staticmethod(lambda *a: sympy.Ne(a[0], a[1]))
    np_fun = np.not_equal
    showme = 'Ne'  # != not working in sympy
    sy_str = 'Ne({0}, {1})'
    repr_str = 'Ne{},[{}, {}]'
    xtype = ((float, float), bool)


class And(LogicOperator, ChainableOp):
    """ Logisches UND für zwei oder mehr Eingaben """
    symfun = staticmethod(lambda *a: sympy.And(*a))
    np_fun = staticmethod(lambda *a: np.logical_and.reduce(a))
    showme = 'And'
    sy_str = '({0} & {1})'  # Arity-2 Formatierung
    repr_str = 'And{},[{}, {}]'
    xtype = ((bool, bool), bool)
    xtype_input = bool
    expr_dmy = 'And'
    symfun_chain = staticmethod(lambda a: sympy.And(*a))
    np_fun_chain = staticmethod(lambda *a: np.logical_and.reduce(a))

    showme_chain = 'AndChain'
    sy_str_chain = 'And({})'  # Variadische Notation
    repr_str_chain = 'And{},[{}]'
    inline_sep = ' & '
    xtype_chain = ([(bool,)], bool)


class Or(LogicOperator, ChainableOp):
    """np.logical_or only for arity-2"""
    symfun = staticmethod(lambda *a: sympy.Or(a[0], a[1]))
    np_fun = staticmethod(lambda *a: np.any(a, axis=0))
    showme = 'Or'
    sy_str = '({0}|{1})'
    repr_str = 'Or{},[{}, {}]'
    xtype = ((bool, bool), bool)
    xtype_chain = ([(bool,)], bool)
    xtype_input = bool
    symfun_chain = lambda a: sympy.Or(*a)
    np_fun_chain = lambda *a: np.logical_or(*a)
    sy_str_chain = 'Or({})'
    repr_str_chain = 'OrChain{},[{}]'
    inline_sep = ' | '


class Xor(LogicOperator, NoSymCapitalized, ChainableOp):
    """Caution: loading '(a ^ b)', the sympy-Xor-representation, is interpreted as a**b"""
    symfun = staticmethod(lambda *a: sympy.Xor(*a))
    # np_fun = staticmethod(lambda *a: np.logical_xor.reduce(*a, axis=0))
    np_fun = staticmethod(lambda *a: np.logical_xor.reduce(a)) # np.logical_and.reduce())
    showme = 'Xor'
    sy_str = 'Xor({}, {})'  # 'a ^ b'
    repr_str = 'Xor{},[{}, {}]'
    xtype = ((bool, bool), bool)
    symfun_chain = lambda a: sympy.Xor(*a)
    np_fun_chain = lambda *a: np.logical_xor(*a)
    sy_str_chain = 'Xor({})'  # 'a ^ b'
    inline_sep = ' ^ '

    xtype_chain = ([(bool,)], bool)
    xtype_input = bool


class ITE(LogicOperator):
    """only logical if-then-else"""
    symfun = staticmethod(lambda *a: sympy.ITE(a[0], a[1], a[2]))
    np_fun = staticmethod(lambda a, b, c: ((a & b) | (not a) & c))  # this fucked me ((a & b) | (not a & c))
    showme = 'ITE'
    sy_str = 'ITE({0}, {1}, {2})'
    repr_str = 'ITE{},[{}, {}, {}]'
    xtype = ((bool, bool, bool), bool)
    # tflow = lambda *args: tf.cond(args[0], true_fn=args[1], false_fn=args[2])


class Min(BaseMinMax, ChainableOp):
    symfun = staticmethod(lambda *a: sympy.Min(*a))
    np_fun = staticmethod(lambda *a: np.minimum.reduce(np.vstack(a), axis=0))
    showme = 'Min'
    sy_str = 'Min({0},{1})'
    repr_str = 'Min{},[{}, {}]'
    xtype = ((float, float), float)
    xtype_input = float
    symfun_chain = lambda a: sympy.Min(*a)
    np_fun_chain = np.minimum
    sy_str_chain = 'Min({})'
    repr_str_chain = 'MinChain{},[{}]'
    xtype_chain = ([(float,)], float)


class Max(BaseMinMax, ChainableOp):
    symfun = staticmethod(lambda *a: sympy.Max(*a))
    np_fun = staticmethod(lambda *a: np.maximum.reduce(np.vstack(a), axis=0))
    showme = 'Max'
    sy_str = 'Max({0}, {1})'
    repr_str = 'Max{},[{}, {}]'
    xtype = ((float, float), float)
    xtype_input = float
    symfun_chain = lambda a: sympy.Max(*a)
    np_fun_chain = np.maximum  # np.vstack(x)
    sy_str_chain = 'Max({})'
    xtype_chain = ([(float,)], float)

    def __call__(self, a):
        # Handle numerical evaluation (for lambdify or direct calls)
        if isinstance(a, (int, float, np.ndarray, pd.DataFrame)):
            return np.maximum.reduce(a)
        raise TypeError("Unsupported type for numerical evaluation in Min(MinMaxBase, ChainableOp)")


class Lt(RelationalOperator):
    symfun = staticmethod(lambda *a: sympy.Lt(a[0], a[1]))
    np_fun = np.less
    showme = 'Lt'
    sy_str = '({0} < {1})'
    repr_str = 'Lt{},[{}, {}]'
    xtype = ((float, float), bool)


class Le(RelationalOperator):
    symfun = staticmethod(lambda *a: sympy.Le(a[0], a[1]))
    np_fun = np.less_equal
    showme = 'Le'
    sy_str = '({0} <= {1})'
    repr_str = 'Le{},[{}, {}]'
    xtype = ((float, float), bool)


class Gt(RelationalOperator, PleaseUsePartnerOp):
    symfun = staticmethod(lambda *a: sympy.Gt(a[0], a[1]))
    np_fun = np.greater
    showme = 'Gt'
    sy_str = '({0} > {1})'
    repr_str = 'Gt{},[{}, {}]'
    xtype = ((float, float), bool)


class Ge(RelationalOperator, PleaseUsePartnerOp):
    xtype = ((float, float), bool)
    symfun = staticmethod(lambda *a: sympy.Ge(a[0], a[1]))
    np_fun = np.greater_equal
    showme = 'Ge'
    sy_str = '({0} >= {1})'
    repr_str = 'Ge{},[{}, {}]'


class Square(MathOperator):
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], 2))
    np_fun = np.square
    xtype = ((float,), float)
    showme = 'Square'
    sy_str = '({})**2'
    repr_str = 'Square{},[{}]'


class Exp(MathOperator):
    symfun = staticmethod(lambda *a: sympy.exp(a[0]))
    np_fun = np.exp
    showme = 'Exp'
    sy_str = '{}**E'
    repr_str = 'Exp{},[{}, {}]'
    xtype = ((float,), float)


class Exp2(MathOperator):
    symfun = staticmethod(lambda *a: sympy.Pow(2, a[0]))
    np_fun = np.exp2
    xtype = ((float,), float)
    showme = 'Exp2'
    sy_str = '2**({})'
    repr_str = 'Exp2{},[{}]'


class Sub(MathOperator):
    xtype = ((float, float), float)
    symfun = staticmethod(lambda *a: sympy.Add(a[0], -a[1]))
    np_fun = np.subtract
    showme = 'Sub'
    sy_str = '({0} - {1})'
    repr_str = 'Sub{},[{}, {}]'


class Ifte(OperatorArity):
    """Also Piecewise"""
    xtype = ((bool, float, float), float)
    symfun = staticmethod(lambda *a: sympy.Piecewise((a[1], a[0]), (a[2], True)))  # sfeh: exprcondpair hier um die tupel rum?
    # np_fun = np.where
    np_fun = staticmethod(lambda cond, if_true, if_false: np.where(cond, if_true, if_false))
    showme = 'Ifte'
    sy_str = 'Ifte({0},{1},{2})'
    repr_str = 'Ifte{},[{}, {}, {}]'
    expr_dummy = 'Ifte'
    xtype_chain = (float, bool)

    # def eval_np_lambdas(self, *args):
    #    no _now version required?
    #     cond, if_true, if_false = self.get_np_child_lambdas(*args)
    #     fun = lambda df: self.np_fun(cond(df), if_true(df), if_false(df))
    #     return fun


class Piecewise(BaseOperator, ChainableOp):
    """ogclass = Ifte"""
    symfun = staticmethod(lambda *a: sympy.Piecewise(*a))
    np_fun = None
    showme = 'Piecewise'
    sy_str = 'Piecewise({})'
    formulae_str = 'Piecewise({})'
    repr_str = 'Piecewise{},[{}]'
    # these must be handeled differently, so commented out
    xtype = ([(ExprCondPair,)], float)
    xtype_chain = ExprCondPair  # discuss (float, bool)
    xtype_input = ExprCondPair

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        # discuss: boolean-results can return the coordinate of the resulting array!
        # sfeh merge eval_np to this
        pairs = [
            (c.childs[0].eval_predict_numpy_now(_df, *args),
             c.childs[1].eval_predict_numpy_now(_df, *args))
            for c in self.get_childs()]

        # Default values are in last position
        result = pairs[-1][0].astype(np.float64, copy=False)

        # from default to first condition
        for _e, _c in reversed(pairs[:-1]):
            result = np.where(_c, _e, result)
        return result

    def eval_np_lambdas(self, *args):
        pairs = [(c.childs[0].eval_np_lambdas(*args),
                  c.childs[1].eval_np_lambdas(*args))
                 for c in self.get_childs()]

        def piecewise_lambda(_df):
            result = pairs[-1][0](df)  # Default case
            for _expr, _cond in reversed(pairs[:-1]):
                result = np.where(_cond(_df), _expr(_df), result)
            return result

        return piecewise_lambda


class Round(MathOperator):
    """

    """
    xtype = ((float,), float)
    symfun = staticmethod(lambda *a: RoundDummy(a[0]))
    np_fun = staticmethod(lambda x: np.vectorize(lambda v: int(round(float(v))))(x))
    showme = 'Round'
    sy_str = 'RoundDummy({},1)'
    repr_str = 'RoundDummy{},[{}]'

    # def eval_predict_numpy_now(self, df, *args) -> np.ndarray:
    #     child_values = self.get_np_child_now(df, *args)
    #     arr = np.asarray(child_values[0], dtype=np.float64)
    #     res = self.np_fun(arr)
    #     return res
    #
    # def eval_np_lambdas(self, *args: Any):
    #     child_lambda = self.get_np_child_lambdas(*args)[0]
    #
    #     def node_lambda(df: np.ndarray) -> np.ndarray:
    #         x = child_lambda(df)
    #         result = self.np_fun(x)
    #         return result.astype(np.int64, copy=False)
    #
    #     return node_lambda


class PowRounded(MathOperator):
    """Requires class RoundDummy!
    Roundind the exponent
    idea: clip the exponent to integer values (e.g. -2-6)"""
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], RoundDummy(a[1])))
    np_fun = staticmethod(lambda base, exponent: np.power(base, np.vectorize(lambda x: int(round(float(x))))(exponent)))
    showme = 'PowRounded'
    sy_str = '({0})**RoundDummy({1})'
    repr_str = 'PowRounded{},[{}, {}]'
    xtype = ((float, float), float)


# sfeh:open
# class Log1p(MathOperator):
#     # https://docs.sympy.org/latest/modules/codegen.html#sympy.codegen.cfunctions.log1p
#     xtype = ((float,), float)
#     symfun = lambda *a: sympy.log(a + 1)
#     showme = 'log1p'


class Div(MathOperator):
    """
    sympy.div() doesn't work for non-polynomials
    """
    symfun = staticmethod(lambda *a: sympy.Mul(a[0], sympy.Pow(a[1], -1)))
    np_fun = staticmethod(np.divide)
    showme = 'Div'
    sy_str = '({0}/{1})'
    repr_str = 'Div{},[{}, {}]'
    xtype = ((float, float), float)


class Sqrt(MathOperator):
    """Capitalized class name, even though its a sympy function
    In SymPy, sqrt(x) is just a shortcut to x**Rational(1, 2)"""
    xtype = ((float,), float)
    symfun = staticmethod(lambda *a: sympy.sqrt(a[0]))  # same as: lambda a: sympy.Pow(a, sympy.S.Half)
    np_fun = staticmethod(np.sqrt)
    showme = 'Sqrt'
    sy_str = 'sqrt({})'
    repr_str = 'Sqrt{},[{}, {}]'


# class Divide_no_nan(Operator):
#     # class-name = 'Divide_no_nan'
#     tflow = tf.math.divide_no_nan
#     symfun = lambda *a, b: sympy.Mul(a, )
#     xtype = ((float, float), float)


# class Nan_replace(Operator):
# """Replaces NaN with a given value"""
#     tflow = tf.where(tf.math.is_nan(a), b, a)
#     symfun = lambda *a: sympy.Piecewise((b, sympy.Eq(a, sympy.nan)), (a, True))
#     xtype = ((float, float), float)
#    showme = 'Nan_replace'
#    sy_str = 'Piecewise(({1}, Eq({0}, nan)), ({0}, True))'
#    repr_str = 'Nan_replace{},[{}, {}]'
#    np_fun = lambda a, b: np.where(np.isnan(a), b, a)


class Usub(MathOperator):
    xtype = ((float,), float)
    # symfun: Callable[[Tuple[float]], float] = staticmethod(lambda a: -a)
    symfun = staticmethod(lambda a, *_: sympy.Mul(-1, a))
    np_fun = staticmethod(lambda x: np.negative(x))
    showme = 'Usub'
    sy_str = '(-{})'
    repr_str = 'Usub{},[{}]'


class Clip(BaseMinMax, CustomOperator):
    # sfeh:open use this
    symfun = staticmethod(lambda *a: sympy.Min(sympy.Max(a[0], a[1]), a[2]))
    np_fun = np.clip  # lambda a, b, c: np.clip(a, b, c)
    showme = 'Clip'
    sy_str = '(sympy.Min(sympy.Max({0}, {1}), {2}))'
    repr_str = 'Clip{},[{}, {}]'
    xtype = ((float, float, float), float)


class ExprCondPair_Dummy(NodeDummy):  # noqa
    """
    Named like this to differ from the sympy original (ExprCondPair)
    The only purpose is to wrap the results for a Node-structure, where every Node has childs with other nodes

    Currently just an idea for setting a default-value
        class ExprCondPair_Default(ExprCondPair):
            ...
    """
    arity = 2
    symfun = staticmethod(lambda *a: ExprCondPair(a[0], a[1]))
    np_fun = None  # discuss
    showme = 'ExprCondPair_Dummy'
    sy_str = 'ExprCondPair({0}, {1})'
    repr_str = 'ExprCondPair_Dummy{},[{}, {}]'
    xtype = ([(float, bool)], float)
    expr_dmy = 'ExprCondPair_Dummy'


# Mapping from sympy classes to plagih Node-classes
# The non-chained version
d_sym2node = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul,
              sympy.Xor: Xor, sympy.Not: Not, sympy.Equality: Eq,  sympy.Unequality: Ne, sympy.And: And, sympy.Or: Or, sympy.StrictLessThan: Lt, sympy.LessThan: Le,
              sympy.StrictGreaterThan: Gt, sympy.GreaterThan: Ge,
              sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos, sympy.asin: Asin, sympy.atan: Atan, sympy.tanh: Tanh, sympy.sinh: Sinh, sympy.cosh: Cosh,
              sympy.Min: Min, sympy.Max: Max, sympy.ITE: ITE, sympy.exp: Exp,}
# sympy.sqrt: Sqrt,  sympy.root: NthRoot not required, as Pow can handle it

# The chained version is the regular version updated with the following operators
d_sym2node_chain = d_sym2node | {sympy.Piecewise: Piecewise, ExprCondPair: ExprCondPair_Dummy}
# sympy.Add: AddChain, sympy.Mul: MulChain, sympy.Min: MinChain, sympy.Max: MaxChain,
#                                  sympy.And: AndChain, sympy.Or: OrChain,  sympy.Xor: XorChain, }


def sympy_expression_check_raise(expr_sym: sympy.Basic) -> sympy.Basic:
    """
    old/analog Version: re.search(r'zoo|inf|nan|I[^f]|\\*I|re\\(', str(expr_sym))"""
    if expr_sym.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I, sympy.im, sympy.re):
        # sympy.re: real part -> don't ignore; if there is a real part, there is an imaginary part.
        raise SympyError(f'Simplification failed: {expr_sym}')
    return expr_sym


def expr_sympify(_expr):
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

    """

    try:
        expr_sym = sympy.sympify(_expr)
        sympy_expression_check_raise(expr_sym)
        return expr_sym

    except ValueError as e:
        raise ValueError(f'NaN in {e}')
    except AttributeError as e:
        # print(f'This sympy bug happens, when sympifying "True": {e}')
        # return sympy.true if expr else sympy.false
        raise e


class Candidate:
    """
    WAS: class FinalizedTree
    An actual individual (Tree + meta-infos/phenotypes)"""

    def __init__(self, tree: Node, fitness, parsimony, tag: str):
        self.tree = tree
        self.fitness = fitness
        self.parsimony = parsimony
        self.tag = deque([tag], maxlen=10)  # Track which evolution created this candidate

    def append_tag(self, tag: str) -> None:
        self.tag.append(tag)

    def get_tag(self, i_evo: int = -1) -> str:
        # i_evo: -1 is last, -2 is second last, ...
        return self.tag[i_evo]

    def __str__(self):
        """Show the Parsimony and Fitness of a tree"""
        return f'[{self.get_parsim():2.0f}: fit {self.get_fitness():4.2f} ({self.tree.__str__()})]'

    def full_string(self) -> str:
        # Paretofront: Removing obsol ... ))]: \x1b[1msign(Max(c
        # https://stackoverflow.com/questions/62213322/python-3-bug-print-background-color-issue
        return f'{self.__str__()}: {BColors.BOLD}{self.get_evotree().get_sympy_expr()}{BColors.RESET}'

    def get_evotree(self) -> Node:
        return self.tree

    def get_fitness(self) -> float:
        # return self.meta.fitness
        return self.fitness

    def get_parsim(self) -> int:
        return self.parsimony


def check_operator_pool(ops: Dict[Type[BaseOperator], float]) -> None:
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


def norm_choices(val_p_tuples: list) -> list:
    """make a tuple-list callable for weighted numpy choice
    [['a', 1], ['b', 2]] -> [('a', 'b'), (0.333, 666)]"""
    xx = list(zip(*val_p_tuples))
    # normalizing the probabilities in every case to a sum of 1 (100%)
    psum = sum(xx[1])
    xx[1] = [i / psum for i in xx[1]]
    return xx


def operatorpool_to_picks(d_operator_pool: Dict[Type[BaseOperator], float]) -> Tuple[dict, dict]:
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


class NodeSelect:

    def __init__(self, operators: dict, symbol_list: List[sympy.Symbol]):
        """make all probabilities sum to 1 for each categoray (Add: 2, Mul: 1, Tan: 0.5) in

        discuss: replace operators-"dict" with a cost-value in the operators class that can be set and is considered
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

    def choose_operator_class(self, xt: Union[Type[float], Type[bool]]) -> Type[BaseOperator]:
        op = np.random.choice(self.pick_op[xt][0], p=self.pick_op[xt][1])  # no (), which would evaluate the op
        return op

    def choose_operator_class_match(self, xtype: tuple) -> Type[BaseOperator]:
        if CHAIN_implement:
            pass
        op = np.random.choice(self.pick_op_match[xtype][0], p=self.pick_op_match[xtype][1])
        return op

    def choose_terminal_node(self, xt: Union[Type[float], Type[bool]], p_observation: float = 0.5) -> Terminal:
        """
        # solved bug: expected str|int|long|float|Decimal|Number object but got 'Node'
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

    def choose_constant_node(self, xt: Union[Type[float], Type[bool]]) -> Terminal:
        _v = np.random.choice(self.pick_constant[xt][0], p=self.pick_constant[xt][1])()  # just dist. must be ()
        if xt == float:
            _v = sympy.Float(_v, FLOAT_PRECISION)  #  discuss allow "rational" inputs? 1/3, 3/4, ...
            return Number(_v)  # round FLOAT_PRECISION was here
        else:
            # _v = sympy.logic.boolalg.BooleanAtom(_v)  # discuss: vs. Boolean
            # -> sympy.sympify('And(True, BooleanAtom(False))')
            return Boolean(_v)

    def choose_symbol_node(self, xt: Union[Type[float], Type[bool]]) -> Symbol:
        """similar to choose_terminal_node()"""
        _v = np.random.choice(self.pick_symbol[xt][0], p=self.pick_symbol[xt][1])
        n = Symbol(_v)
        return n


class Evolution:
    """
    was "TreeBuildRestrictions"
    functions to build trees, with the advantage of being able to use general build restrictions.

    all Symbol-inputs are chosen from a list with equal probability.
        -> don't overcomplicate this process.
        -> provide more options when asked for, like giving random()-probabilities

    Evolution discussion: Random tree creation strategies
    - Size measure: depth, node count, weighted node count (parsimony)
    - Architectures: Full, Grow, Ramped Half-and-Half
    - Node selection: Random, weighted random


    """

    operator_presets = {'math_simple':
                        {Add: 2, Mul: 2, Div: 1, Square: 0.75, Abs: 0.5, Sign: 0.5, Sqrt: 0.1, Log: 0.1,
                         Sin: 0.5, Not: 0.5, Lt: 0.5, Le: 0.5, And: 1, Or: 1, Min: 1, Max: 1}}

    def __init__(self, symbol_list=None, origin_xtype=float, operators=None, origin_tree=None,
                 depth_max=10, nodes_max=100, complexity_metric='tree_node_count_fair', allow_chain=None):
        """
        origin_tree: A tree, which

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
            symbol_list = sympy.symbols('a b', real=True, imaginary=False)  # sympy symbols options
        else:
            symbol_list = [sympy.Symbol(s) if isinstance(s, str) else s for s in symbol_list]
            symbol_list = sorted(symbol_list, key=lambda x: str(x))
        self.symbol_list = symbol_list
        self.symbol_list_str = [str(s) for s in symbol_list]  # -> for df-evaluation (string-keys are expected...)
        self.node_selector = NodeSelect(operators, symbol_list)

        self.complexity_metric = complexity_metric

        self.depth_max = depth_max
        self.nodes_max = nodes_max

        self.allow_a_chain = allow_chain

    def evolve_prune_tree(self, tree: Node) -> Node:
        """
        prune depth
        -> prune everything below a certain level... (should not happen in the first place)
        prune nodes
        -> get node difference, get nodelist, untill small enough: split the difference, prune nodes until
        """
        nodelist = tree.list_mutable_nodes()
        for dnode in nodelist:
            if dnode.depth == self.nodes_max and dnode.get_arity() > 0:
                print_warning('wwww', f'Node in fintree is too deep: {dnode.depth}')
                new_node = self.node_selector.choose_terminal_node(dnode.get_xtype_self())
                new_node.depth = dnode.depth
                dnode.set_new_node(new_node)

        # sfeh not as trivial as pruning the max. tree depth: Which nodes to prune randomly?
        #   This strongly affects the tree structure and should thus be decided in the creation process
        #   Pruning strategies:
        #   - Randomly prune nodes until complexity is met
        #   - Prune the deepest nodes first, every depth level completely
        #   - check crossover
        prune_amount = len(tree) - self.nodes_max
        while prune_amount > 0:
            print_warning('wwww', f'Tree too complex: {len(tree)} > {self.nodes_max}, pruning {prune_amount}.')
            nodelist = tree.list_mutable_nodes()
            prune_now = 1 + np.random.randint(prune_amount)  # 19 -> prune branch with 1 to max. 19 nodes

            nodelist = [x for x in nodelist if len(x) >= prune_now]  # only (operator-) nodes
            tree = random.choice(nodelist)
            new_node = self.node_selector.choose_terminal_node(tree.get_xtype_self())
            new_node.depth = tree.depth
            tree.set_new_node(new_node)
            prune_amount = len(tree) - self.nodes_max

        return tree

    def evolve_new_tree_depth(self, xt_out: Union[Type[float], Type[bool]], depth_goal: int, p_term: float = 0.0) -> Node:

        if self.origin_tree is not None:

            evotree = copy.deepcopy(self.origin_tree)
            layer0 = evotree.get_mutable_rootnodes(extend_lvls=0)

            for ii, nd in enumerate(layer0):  # -> get layer every time (nsted ids might have changed)
                new_subbranch = self.evolve_create_random(nd.get_xtype_self(), depth_goal, num_rest=-1,
                                                          depth=nd.depth, p_term=p_term)
                nd.set_new_node(new_subbranch)

        else:
            evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    def evolve_chained_new_tree_depth(self, depth_goal: int, xt_out: Union[Type[float], Type[bool]], p_term: float = 0.0) -> Node:

        evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    def evolve_create_random(self, xt_out: Union[Type[float], Type[bool]], depth_max_local: int, num_rest: int = -1, depth: int = 0, p_term: float = 0.0) -> Node:
        """

        discuss: number of leftover nodes is not a good threshold, as it limits depth-spreading branches in growing.
                    -> Prune the tree at the end and allow any growth in the beginning
                    -> Tree depth
        num_rest: -1 ignores the node number restriction
        depth_max_local: can be set lower than self.depth_max"""

        # setting a terminal-node if it is required OR p_term is met
        if depth >= min(self.depth_max, depth_max_local) or num_rest == 0 or random.random() < p_term:
            node = self.node_selector.choose_terminal_node(xt_out)
        else:

            node_cls = self.node_selector.choose_operator_class(xt_out)
            child_xts = node_cls.get_child_xts()
            childs = []

            if CHAIN_implement:
                pass  # optional; just add more node here already

            nums = randomly_split_range(num_rest - 1, len(child_xts))

            for ii, xt in enumerate(child_xts):
                cc = self.evolve_create_random(xt, depth_max_local, num_rest=nums[ii], depth=depth+1, p_term=p_term)
                childs.append(cc)

            node = node_cls(*childs)

        node.depth = depth

        return node

    def evolve_mutate_filter(self, tree: Node) -> Node:
        """Mutates a number of float terminal of a fintree
        - filter point/branch/all, branch can also affect a point only as well as all nodes
        - filter observations?
        - filter terminals
        - filter with which filter?"""

        _nd = random.choice(tree.list_mutable_nodes())
        _nd.evolve_mutate_filter_gauss()

        return _nd

    def evolve_mutate_point(self, tree: Node) -> Node:
        """Mutate a single mutable point in any Tree."""
        evotree = copy.deepcopy(tree)

        node = rnd_choice(evotree.list_mutable_nodes())  # debug if ignores chains
        xtype = node.get_xtype_tuple()

        if node.is_operator():
            # allow_chain-option
            new_label = self.node_selector.choose_operator_class_match(xtype)  # Function is same type, same arity
            node = new_label(*node.childs)  # noqa debug_me not tested, not used

        elif is_terminal(node):
            new_node = self.node_selector.choose_terminal_node(xt_self(xtype))
            node.set_new_node(new_node)
        else:
            raise NotImplementedError

        return evotree

    def evolve_mutate_branch_depth(self, tree: Node, depth_goal: int, allow_chain=False, p_term: float = 0.0) -> Node:
        """"""
        n_init = len(tree)
        node_list = tree.list_mutable_nodes()
        node = random.choice(node_list)
        xtype_out = node.get_xtype_self()  # ValueError: 'a' cannot be empty unless no samples are taken
        branch = self.evolve_create_random(xtype_out, depth_goal, num_rest=self.nodes_max - n_init, depth=0,
                                           p_term=p_term)
        node.set_new_node(branch)

        return tree

    def evolve_mutate_branch_nodes(self, tree: Node, nodes_goal, p_term=0.0) -> Node:
        """currently only one branch
        p_term: probability terminating the tree in a node
        """
        nodes_init = len(tree)
        if tree is None:
            raise NotImplementedError('Implement standard selection mechanism')
        nd = tree.list_mutable_nodes()
        nd = rnd_choice(nd)
        xt_out = nd.get_xtype_self()
        nodes_goal = min(self.nodes_max - (nodes_init - len(nd)), nodes_goal)

        branch = self.evolve_create_random(xt_out, -1, num_rest=nodes_goal, depth=nd.depth, p_term=p_term)
        nd.set_new_node(branch)
        mutated_tree = tree
        return mutated_tree

    def evolve_crossover(self, aa: Node, bb: Node):
        """Evolution with crossover of branches between two trees
        currently only one branch

        swap branches of two trees
        - select parent aa and bb
        - select swappable branche for a_parent from b_parent
            - select aa node in aa (and crossover here, no matter what)
        - delete a_parent branch and pareto_insert b_parent branch (which tactic?)"""

        a_nds = aa.list_mutable_nodes()
        a_nds = a_nds[1:]  # skip_first ...why actually ignore root node?
        #   -> this shall prevent two trees from just "swapping place" (aka only root nodes are exchanged)
        #   -> this can actually happen quite often, when trees have low complexity

        if len(a_nds) == 0:
            raise TreeError(f'Crossover tree 1 has no mutable nodes!')

        a_nd = random.choice(a_nds)
        xt_out = a_nd.get_xtype_self()
        b_nds = bb.list_mutable_nodes(xtype=xt_out)

        if len(b_nds) > 0:
            b_nd = random.choice(b_nds)

        else:
            xt_out = float if xt_out == bool else bool  # switching to the other swap type
            b_nds = bb.list_mutable_nodes(xtype=xt_out)
            b_nd = random.choice(b_nds)
            a_nds = [x for x in a_nds if x.get_xtype_self() == xt_out]
            if len(a_nds) == 0:
                raise ValueError(f'Crossover cant find matching nodes. This Should always be possible.')
            a_nd = random.choice(a_nds)

        cpy = copy.deepcopy(a_nd)  # deepcopy required??

        a_nd.set_new_node(b_nd)
        b_nd.set_new_node(cpy)

        # only required, if pruning is not done in finalize_tree()
        aa = self.evolve_prune_tree(tree=aa)
        bb = self.evolve_prune_tree(tree=bb)

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
    0 is allowed! (ends a branch with a terminal node)"""

    if range_max < 0:
        return [-1 for _ in range(num_splits)]

    sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
    d_sum = sum(sample_dist)  # 1.5
    sample_dist = [i / d_sum for i in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
    sample_dist = [i * range_max for i in sample_dist]  # [12, 60, 28] -> for 100 nodes
    sample_dist = [int(round(i, 0)) for i in sample_dist]  # int required

    # workaround, this makes exactly the correct range by changing the most "extreme" entry
    imprecise_diff = range_max - sum(sample_dist)  # sfeh: this can be [0, 0, 0], which assigns to the 0th bin...
    # sfeh:discuss: maybe this difference is 2 or larger more often than 1 (->rounding),
    # so maybe while-loop (just check if it happens?)
    if imprecise_diff != 0:
        if sum(sample_dist) < range_max:
            # if relatively empty, this appends to the first bin
            sample_dist[sample_dist.index(min(sample_dist))] += imprecise_diff  # extreme_bin = smallest
        elif sum(sample_dist) > range_max:
            sample_dist[sample_dist.index(max(sample_dist))] += imprecise_diff  # extreme_bin = greatest
        else:
            raise

    return sample_dist


def print_pop(pop):
    """
    Print the expressions of all trees in a population
    pop_print
    """
    n = [f'{k.full_string()}' for k in pop]
    n = [f'{BColors.BLUE}{x}' if ii % 2 == 0 else f'{BColors.YELLOW}{x}' for ii, x in enumerate(n)]
    n = [f'{k}\n' if ii % 10 == 9 else f'{k}\t' for ii, k in enumerate(n)]  # stop \n in line 0
    n = ''.join(n)
    n = re.sub(r'\n$', '', n)  # remove trailing \n (\t irrelevant)
    n = f'{n}{BColors.RESET_COLOR}'
    print(n)


class ExplainableGP:
    """

    """
    def __init__(self, evolve: Evolution, df_train, rootdir: Path, pop_max_size = 100, gen_end=100, eval_autocast=np.array, allow_chain=False, eval_error_metric=None):
        self.time_start = time.perf_counter()

        self.rootdir = rootdir
        self.rootdir.mkdir(parents=True, exist_ok=True)

        self.df_train = df_train

        self.evolve = evolve
        self.gen_end = gen_end
        self.pop_max_size = pop_max_size
        self.gen_id: int = 0
        self.eval_autocast = eval_autocast
        self.eval_error_metric = eval_error_metric

        # sfeh only used implicitly ( different evolution for chain vs. non-chain)
        # chained-versions are only created after sympy. no evolution atm
        self.allow_chain = allow_chain

        print(f'\n'
              f'\tInitializing Plagih.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET_COLOR}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        self.paretofront = []  # not a separate class; requires too much information
        self.pop_genepool = []
        self.pop_next = []

        self.lut_tree_infos = {}
        self.lut_symex_fitness = {}  # Lookup-table for tree(-expressions) and its fitness/parsimony. Improving runtime a lot!

        # monitoring
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0
        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'lut_symex_fitness-len',
                                                'time',
                                                'fit_avg', 'fit_var',
                                                'fit_quantile_25', 'fit_quantile_50', 'fit_quantile_75', 'fit_best',
                                                'parsim_avg', 'parsim_var', 'parsim_quantile_25', 'parsim_quantile_50',
                                                'parsim_quantile_75', 'parsim_best',
                                                'gens_since_last_pareto'])

    def get_name(self):
        if isinstance(self.rootdir, Path):
            s = self.rootdir.name
        else:
            s = None
        return s

    def run_update_paretofront(self, pop):
        """
        Aktualisiert self.paretofront mit nicht-dominierten Kandidaten aus 'pop'.
        Minimiert sowohl Fitness als auch Parsimony. Gibt True zurück, wenn sich die Front geändert hat.
        """
        new_cands = pareto_from_pop(pop) or []
        if not new_cands:
            return False

        old_front = list(self.paretofront) if self.paretofront else []
        old_best_par = min((c.get_parsim() for c in old_front), default=float("inf"))
        old_best_fit = min((c.get_fitness() for c in old_front), default=float("inf"))

        combined = old_front + new_cands

        def dominated(a, b):
            pa, fa = a.get_parsim(), a.get_fitness()
            pb, fb = b.get_parsim(), b.get_fitness()
            return (pb <= pa and fb <= fa) and (pb < pa or fb < fa)

        new_front = []
        for c in combined:
            if any(dominated(c, o) for o in combined if o is not c):
                continue
            # Duplikate (gleiche Metriken) vermeiden
            if not any((c.get_parsim() == e.get_parsim() and c.get_fitness() == e.get_fitness()) for e in new_front):
                new_front.append(c)

        # Sortierung für stabile Ausgabe/Weiterverarbeitung
        new_front.sort(key=lambda x: (x.get_parsim(), x.get_fitness()))

        # Änderung erkennen
        old_keys = {(cand.get_parsim(), cand.get_fitness(), cand.full_string()) for cand in old_front}
        new_keys = {(cand.get_parsim(), cand.get_fitness(), cand.full_string()) for cand in new_front}
        changed = new_keys != old_keys

        if changed:
            self.paretofront = new_front
            new_best_par = min(c.get_parsim() for c in new_front)
            new_best_fit = min(c.get_fitness() for c in new_front)

            if old_front and new_best_par < old_best_par:
                printez('a', f'Paretofront: Neuer simpelster Eintrag. parsimony: {new_best_par} '
                             f'old simplest had {old_best_par}')
            if old_front and new_best_fit < old_best_fit:
                printez('a', f'Paretofront: Neuer fittester Eintrag. fitness: {new_best_fit:6.4f} '
                             f'old best had {old_best_fit:6.4f}')
            if not old_front:
                printez('a', f'Paretofront initialisiert mit {len(new_front)} Kandidaten.')

            self.gens_since_last_pareto = 0

        return changed

    def end_generation(self):
        # sfeh:open end generation in every generation
        if self.run_update_paretofront(self.pop_next):
            self.gens_since_last_pareto = 0

        self.pop_genepool = self.pop_next[:]
        print_pop(self.pop_next)
        self.pop_next = []
        self.analyze_generation()
        self.gen_id += 1

        self.time_genstart = time.perf_counter()

    def gen_create_initial(self, origin_tree=None):
        """

        """
        printpl('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')

        if origin_tree is not None:
            cand_origin = self.tree_to_candidate(origin_tree, raise_if_useless=False, tag='origin')
            self.pop_next_append(cand_origin)
        else:
            if self.allow_chain:
                @self.create_trees(rate=0.5)
                def init_rand1():
                    n = np.clip(int(random.normalvariate(4.0, 1.0)), 4, 6)
                    tree = self.evolve.evolve_new_tree_depth(float, n, p_term=0)
                    tree = tree_simplification(tree, allow_chain=self.allow_chain)

                    if tree.get_max_depth() == 0:
                        raise TreeSizeError(f'Tree did not get complex enough (only root node).')
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2():
                    n = np.clip(int(random.normalvariate(4.5, 1.0)), 3, self.evolve.depth_max)
                    tree = self.evolve.evolve_new_tree_depth(float, n, p_term=0)
                    tree = tree_simplification(tree, allow_chain=self.allow_chain)
                    return tree
            else:
                @self.create_trees(rate=0.5)
                def init_rand1a():
                    n = np.clip(int(random.normalvariate(4.0, 1.0)), 3, 5)
                    tree = self.evolve.evolve_new_tree_depth(float, n, p_term=0)
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2a():
                    n = np.clip(int(random.normalvariate(3.5, 1.0)), 3, self.evolve.depth_max)
                    return self.evolve.evolve_new_tree_depth(float, n, p_term=0)

        self.paretofront = pareto_from_pop(self.pop_next)
        self.pop_genepool = self.pop_next[:]
        self.pop_next = []
        self.analyze_generation()
        self.gen_id += 1
        return self.pop_genepool

    def pop_next_append(self, ct: Candidate, force=False):
        evotree = ct.get_evotree()
        # from visualization.pygraphviz import render_pygraphviz
        if force and ct.get_parsim() < TREE_MIN_PARSIMONY:
            # raise ValueError(f'Tree not complex enough for population, sfeh')
            return
        printpl('gggg', f'|->{evotree.len_nodecount_fair():2.0f}: {evotree.str_as_expr()}')
        self.pop_next.append(ct)

    def create_trees(self, rate=0.0, crossover=False, simplicate=False, allow_chain=False):
        """Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the final tree (candidate_tree) is refurbished."""

        def loop(create_tree_f):
            n = int(rate * self.pop_max_size)
            n_success = 0
            fails_list = []
            tag = create_tree_f.__name__
            printpl('ggg', f'->Evolving {n}x \'{tag}\'...')

            while n_success < n:
                try:
                    if crossover:
                        t1, t2 = create_tree_f()
                        if simplicate:
                            t1 = tree_simplification(t1, allow_chain=self.allow_chain)
                        ctree1 = self.tree_to_candidate(t1, tag=tag)
                        self.pop_next_append(ctree1)
                        n_success += 1
                        if simplicate:
                            t2 = tree_simplification(t2, allow_chain=self.allow_chain)
                        ctree2 = self.tree_to_candidate(t2, tag=tag)
                        self.pop_next_append(ctree2)
                        n_success += 1
                    else:
                        evotree = create_tree_f()
                        if simplicate:
                            evotree = tree_simplification(evotree, allow_chain=self.allow_chain)
                        ctree = self.tree_to_candidate(evotree, tag=tag)
                        self.pop_next_append(ctree)
                        n_success += 1

                except (TreeError, TreeSizeError, SympyError) as e:

                    fails_list.append(e)
                    print_warning('www', f'Failed evolution tag \'{tag}\': {e}')
                    if len(fails_list) > 2 * n_success + 5:  # allow more fails: fails_list > n
                        print_caution(f'Evolution fails too often: {tag}, failed: {len(fails_list)}x. ({n_success} ok).'
                                      f'\n{fails_list}')
                        return  # optional raise

                except (ValueError, ArithmeticError) as e:
                    # if 'Crossover tree 1 has no mutable nodes!' in str(ex):
                    if ("'a' cannot be empty unless no samples are taken" in str(e)
                            or "The argument 'zoo' is not comparable" in str(e)):
                        print_warning('ww', f'OnlyPrintException: {e}')

                except KeyError as e:
                    # KeyError(re) -> okay?, real part implies complex numbers, ignoring is okay
                    # (probably sympy.lambdify expression not evaluable)
                    print(f'OnlyPrintException: Keyerror?: {e}')
                except RecursionError as e:
                    print(f'OnlyPrintException: RecursionError (probably Piecewise/relational combination?): {e}')
                # except NotImplementedError as nie:
                #     print_caution(f'Notimplemented? {nie}')
                # except Exception as ex:
                #     print(f'OnlyPrintException: Why are we not here??? {ex}')
        return loop

    def tree_to_candidate(self, evotree: Node, origin_tree=None, tag=None, raise_if_useless=True, compare_with_sympy=DEBUG_DUMMY):  # DEBUG
        """the "fixed" node information is not relevant

        Tree MUST NOT be altered from here!
        raise_if_useless is here in order to show, where the maximum nodes is exceeded!
        """

        # Make this tree usable for evaluation
        evotree.force_input_node(self.evolve)
        evotree = self.evolve.evolve_prune_tree(evotree)
        evotree.repair_depth()

        tree_id = evotree.get_lut_id()

        if tree_id in self.lut_tree_infos:
            sy_expr = self.lut_tree_infos[tree_id].get('sy_expr')
            parsimony = self.lut_tree_infos[tree_id].get('parsimony')
            fitness = self.lut_tree_infos[tree_id].get('fitness')
            if not all([sy_expr, parsimony, fitness]):
                # print_warning('ww', f'Could not evaluate fitness for tree {sy_expr}: {ex}')
                asd = self.lut_tree_infos[tree_id].get('error')
                if asd is None:
                    pass
                raise TreeLutError(f'Tree LUT Entry implies Problem: {asd}')
        else:
            # requires: valid, sympy expr, parsimony, fitness
            self.lut_tree_infos[tree_id] = {}  # empty placeholder, if correctly filled later

            parsimony = eval_parsimony(evotree, self.evolve.complexity_metric, origin_tree=origin_tree)
            if raise_if_useless and parsimony > self.evolve.nodes_max:
                err_txt = f'Tree too complex: {parsimony} > {self.evolve.nodes_max}'
                self.lut_tree_infos[tree_id]['error'] = err_txt
                raise TreeSizeError(err_txt)
            try:
                sy_expr = evotree.get_sympy_expr()
            except SympyError as e:
                print_warning('www', f'Could not create sympy expression for tree: {e}')
                self.lut_tree_infos[tree_id]['error'] = str(e)
                raise

            if sy_expr in self.lut_symex_fitness:
                # other tree might have same expression -> lookup fitness
                fitness = self.lut_symex_fitness[sy_expr]
            else:
                """Numpy eval"""
                perf_t = {0: time.perf_counter()}
                true_values = self.df_train['action'].to_numpy()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)  # sfeh:discuss
                        np_results_raw = evotree.eval_predict_numpy_now(self.df_train)  # exception? -> check np.isnan(sym_results).any()
                        np_results = self.eval_autocast(np_results_raw)
                        np_fitness = self.eval_error_metric(np_results, true_values)
                        np_fitness = round(np_fitness, FLOAT_PRECISION)

                        if 'nan' in str(np_fitness) or np_fitness == np.nan or np_fitness == np.inf:  # sfeh:code not so good looking
                            err_txt = f'NaN in results'
                            self.lut_tree_infos[tree_id]['error'] = err_txt
                            raise TreeError(f'{err_txt}')

                        perf_t[1] = time.perf_counter()

                        if compare_with_sympy:

                            """Numpy eager eval"""
                            # sfeh _lambda verion comparisson, functionality-wise and time-wise, eval sympy first
                            try:
                                nplmd_results_raw = evotree.eval_np_lambdas()
                                nplmd_results_raw = nplmd_results_raw(self.df_train)
                            except Exception as e:
                                nplmd_results_raw = evotree.eval_np_lambdas()
                                nplmd_results_raw = nplmd_results_raw(self.df_train)

                            perf_t[2] = time.perf_counter()

                            """Sympy lambdify"""
                            sym_results_raw = eval_predict_sympyBatch(sy_expr, self.df_train, self.evolve.symbol_list)
                            sym_results = self.eval_autocast(sym_results_raw)
                            sym_fitness = self.eval_error_metric(sym_results, self.df_train['action'])
                            sym_fitness = round(sym_fitness, FLOAT_PRECISION)

                            perf_t[3] = time.perf_counter()

                            printpl('pp', f'NP: {perf_t[1]-perf_t[0]:4.4f}s, NE: {perf_t[2]-perf_t[1]:4.4f}s, SY: {perf_t[3]-perf_t[2]:4.4f}s. '
                                          f'Fitness NP: {np_fitness}, SY: {sym_fitness} ({sy_expr}), eval tree id {tree_id}')

                            try:
                                sum(nplmd_results_raw - np_results_raw)
                            except Exception as e:
                                raise TreeError('SFEH THESE ARE [True] trees')

                            if sum(nplmd_results_raw - np_results_raw) > 0.001:
                                diffs = np.abs(nplmd_results_raw - np_results_raw)
                                mask = diffs > 0.001
                                if np.any(mask):
                                    indices = np.where(mask)[0]
                                    print_warning('w', f'{len(indices)} differences found above tolerance 0.001: (NP VERSION)')
                                print(f'Different in (NP VERSION): {sum(nplmd_results_raw - np_results)} ({sy_expr})')

                            if np.sum(np.abs(nplmd_results_raw - np_results_raw)) > 0.001:
                                sym_results_raw_np = sym_results_raw.to_numpy()
                                diffs = np.abs(sym_results_raw_np - np_results_raw)
                                mask = diffs > 0.001
                                if np.any(mask):
                                    indices = np.where(mask)[0]
                                    print_warning('w', f'{len(indices)} differences found above tolerance 0.001:')
                                # results_syraw_df = eval_predict_df_sympy_only(sy_expr, self.df_train)  #  takes forever
                                result_diffs = sum(sym_results_raw - np_results_raw)
                                print(f'Different results in evaluation: {result_diffs} sy-expr: ({sy_expr})')

                            fitness = sym_fitness
                            self.lut_tree_infos[tree_id]['fitness-sympy'] = sym_fitness
                except (SympyError, TreeError, ValueError) as e:
                    print_warning('wwww', f'Could not evaluate fitness for tree {sy_expr}: {e}')
                    self.lut_tree_infos[tree_id]['error'] = str(e)
                    raise

                fitness = np_fitness

                self.lut_symex_fitness[sy_expr] = fitness

            self.lut_tree_infos[tree_id]['sy_expr'] = sy_expr
            self.lut_tree_infos[tree_id]['parsimony'] = parsimony
            self.lut_tree_infos[tree_id]['fitness'] = fitness

        candidate = Candidate(evotree, fitness=fitness, parsimony=parsimony, tag=tag)
        return candidate

    def evoloop_monitoring_plots(self):
        """
        Create all run-related analysis plots in the root directory
        """
        plot_performance(self.monitor_df, self.rootdir / 'monitoring.png')
        plot_paretofront(self.paretofront, self.rootdir, self.evolve.nodes_max)

        from visualization.visualize_trees import visualize_paretofront
        visualize_paretofront(self.paretofront, filename="paretofront_trees", output_dir=self.rootdir)

        if self.gen_id <= 20:
            gen_filename = f'monitoring_parsimony_histogram_{self.gen_id:03d}.png'
            # Use pop_size as max_population and nodes_max as max_parsimony for fixed scaling
            plot_parsimony_histogram(self.pop_genepool, self.rootdir / gen_filename,
                                     max_population=self.pop_max_size,
                                     max_parsimony=self.evolve.nodes_max)

    def backup_save(self, opt_path_backup=None):
        """
        Load/safe backup of a run
        """

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
        run_backup_data = {}, self.gen_id, self.pop_genepool, self.paretofront, self.monitor_df
        path_backup = path_make_dir(path_backup)
        pickle_dump(path_backup, run_backup_data)

    def backup_load(self, opt_path_backup=None):
        """Load/safe backup of a run"""

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        if Path.is_file(path_backup):
            printpl('g', f'Loading data from backup-file {path_backup}')
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)
            except EOFError as e:
                raise Exception(f'EOFError: \n{e}')

            help_dict, self.gen_id, self.pop_genepool, self.paretofront, self.monitor_df = run_data
            self.backup_save(opt_path_backup=self.rootdir / f'backup/backup-{self.gen_id}.pkl')
            printpl('g', f'Successfully loaded backup file. Generation: {self.gen_id}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}')

    def analyze_generation(self):
        gen_time = time.perf_counter() - self.time_genstart
        tmp_dict = pop_analyze(self.pop_genepool, gen_time, self.gens_since_last_pareto, self.lut_symex_fitness)
        self.monitor_df.loc[self.gen_id] = pd.Series(tmp_dict)
        printpl('gg',
                f"Created {len(self.pop_genepool)}/{self.gen_end} ({tmp_dict['pop_unique']} unique) in generation {self.gen_id}. "
                f"Trees in LUT: {len(self.lut_symex_fitness)} Generation took {gen_time:4.2f}s")

        printpl('ggg', f'--- Generation {self.gen_id} took: {time.perf_counter() - self.time_genstart:4.2f}. ---')

        # def monitoring_scheduled_io(self, gen_id, plots_interval=10, backup_interval=10):
        """
        Every x generations, save a backup and/or save plots
        """
        if self.gen_id >= PLOTS_INTERVAL and self.gen_id % PLOTS_INTERVAL == 0:
            self.evoloop_monitoring_plots()

        if self.gen_id >= BACKUP_INTERVAL and self.gen_id % BACKUP_INTERVAL == 0 or self.gen_id == 10:
            self.backup_save()

    def run_custom_exit_condition(self):
        """
        Special condition to exit the evolve-loop
        1. when >100 generations, no new paretofront were found
        """
        if self.gens_since_last_pareto > 100:
            printpl('i', 'Custom Condition made your program exit! (No new pareto entries in 100 generations)')
            return True
        else:
            return False

def is_terminal(node: Node) -> TypeGuard[Terminal]:
    return issubclass(node.__class__, Terminal)

def is_number(node: Node) -> TypeGuard[Number]:
    return issubclass(node.__class__, Number)

def plot_performance(monitor_df, path_monitoring: Path):
    """
    All monitoring infos
    # fit_best is not necessary
    """
    with plt.rc_context(rc={'axes.grid': True}):
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]},
                                sharex='all')
        plt.subplots_adjust(wspace=0, hspace=0.1)  # left=0, bottom=0, right=1, top=1
        xx = list(monitor_df.index)

        axs0 = axs[0]
        axs0.plot(monitor_df['fit_avg'], marker='', label='regression error (average)')
        avg = monitor_df['fit_avg']
        std = monitor_df['fit_var']
        fit_quantile_25 = monitor_df['fit_quantile_25']
        fit_quantile_50 = monitor_df['fit_quantile_50']
        fit_quantile_75 = monitor_df['fit_quantile_75']
        parsim_avg = monitor_df['parsim_avg']
        parsim_var = monitor_df['parsim_var']
        parsim_best = monitor_df['parsim_best']
        parsim_quantile_50 = monitor_df['parsim_quantile_50']
        parsim_quantile_25 = monitor_df['parsim_quantile_25']
        parsim_quantile_75 = monitor_df['parsim_quantile_75']

        axs0.fill_between(xx, avg - std, avg + std, alpha=0.2)  # do not use avg in both directions...
        axs0.fill_between(xx, fit_quantile_25, fit_quantile_75, color='b', alpha=0.2)
        # axs0.set_title('regression Error (average)')  # sfeh not stderr... upper/lower bound?
        axs0.step(x=xx, y=monitor_df['fit_best'], linestyle='dashed', marker='', where='post', color='g',
                  label='Best candidate in pop')  # , label=ax_label
        # axs0.step(x=xx, y=fit_quantile_50, linestyle='dashed', marker='', where='post', color='b',
        #           label='Best candidate')
        axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

        axs0_twin = axs0.twinx()
        axs0_twin.plot(xx, monitor_df['gens_since_last_pareto'], color='tab:gray',
                       label='Gen since last pareto entry', linestyle='dashed',
                       marker='')  # linestyle='None'
        axs0_twin.tick_params(axis='y', labelcolor='tab:gray')
        axs0_twin.set_ylim(ymin=0, ymax=max(monitor_df['gens_since_last_pareto'].max() or 1, 50))
        axs0_twin.legend(loc='lower right')
        axs1 = axs[1]
        axs1.plot(monitor_df['parsim_avg'], label='Complexity (average)')

        p_avg = monitor_df['parsim_avg']
        p_var = monitor_df['parsim_var']
        axs1.fill_between(xx, p_avg - p_var, p_avg + p_var, alpha=0.2)  # axs1.set_title('TED (average)')
        axs1.set_ylim(ymin=0), axs1.legend(loc='lower left')

        axs2 = axs[2]
        axs2.plot(monitor_df['pop_len'], label='pop_list size')
        axs2.plot(monitor_df['pop_unique'], label='unique')
        axs2.margins(y=0.25), axs2.set_ylim(ymin=0), axs2.legend(loc='lower left')

        # optional: plot lut_symex_fitness-len

        axs3 = axs[3]
        # # Remove extreme time outliers for better visibility
        #   e.g. debugging, reloading
        # between_outliers = monitor_df['time'].between(0, 10 * monitor_df['time'].mean())
        # axs3.plot(monitor_df['time'][between_outliers], label='time (s)')
        axs3.plot(monitor_df['time'], label='time (s)')
        axs3.set_ylim(ymin=0), axs3.legend(loc='lower left')

        # Top level style
        axs3.set_xlim(xmin=0, xmax=max(xx)), axs3.set_xlabel('generation')
        axs3.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axs0.set_title(f'Monitoring GP Generations {path_monitoring.name}')
        fig.tight_layout()
        fig.savefig(path_monitoring)
        plt.close('all')


def plot_parsimony_histogram(population, path_out: Path, max_population: int, max_parsimony: int):
    """Plottet die Parsimony/Komplexität einer Population als Histogramm.

    Ziel: schnell sehen, welche Tree-Größen in der Population vorkommen.

    Eigenschaften:
    - Bins sind ganzzahlige Parsimony-Werte (ein Bin pro Wert).
    - Balken sind nach Evolution (tag) gruppiert und farbcodiert
    - Keine Abstände zwischen den Balken
    - Feste Skalierung für Vergleichbarkeit über Generationen

    Erwartete Populationseinträge:
    - `Candidate`-Objekte (haben `.parsimony` oder `.get_parsim()` und `.tag`)
    - oder Nodes/Trees, wenn sie ein Attribut `.parsimony` haben (Fallback)

    `path_out` sollte ein `pathlib.Path` sein (z.B. `self.rootdir / 'monitoring_parsimony_histogram.png'`).
    `max_population`: Maximale Populationsgröße für feste X-Achsen-Skalierung
    `max_parsimony`: Maximale Parsimony für feste Y-Achsen-Skalierung
    """
    tags = [_tree.get_tag() for _tree in population]

    unique_tags = list(dict.fromkeys(tags))  # stabile Reihenfolge
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, max(1, len(unique_tags))))
    tag_colors = {_t: colors[i] for i, _t in enumerate(unique_tags)}

    with plt.rc_context(rc={'axes.grid': True}):
        fig, ax = plt.subplots(figsize=(16, 6))

        xticks_positions = []
        xticks_labels = []

        for current_position, ctree in enumerate(population):
            pa = ctree.get_parsim()
            depth = ctree.get_evotree().get_max_depth()
            tag = ctree.get_tag()
            ax.bar(current_position, pa, width=1.0, align='edge', color=tag_colors[tag], linewidth=0.5)
            ax.bar(current_position, depth, width=1.0, align='edge', color=tag_colors[tag]*0.3,  linewidth=0.5)

        ax.set_xlabel('Evolution')
        ax.set_ylabel('count')
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.set_xlim(0, max_population)
        ax.set_ylim(0, max_parsimony)

        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticks_labels, rotation=45, ha='right')
        ax.set_title(f'Parsimony histogram ({path_out.name})')

        # Eindeutige Tag-Legende über Proxy-Patches
        legend_handles = [Patch(facecolor=tag_colors[t], edgecolor='none', label=t) for t in unique_tags]
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, title='Tags')

        fig.tight_layout()
        fig.savefig(path_out)
        plt.close('all')


def pop_analyze(popul, gen_time, gens_since_last_pareto, lut_symex_fitness) -> dict:
    """Analysing the population (in each generation)
    - amount of trees
    - fittest tree
    - average fitness
    - average tree parsimony"""

    if len(popul) == 0:
        raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

    pop_fitness = [tree.get_fitness() for tree in popul]
    pop_parsim = [tree.get_parsim() for tree in popul]
    pop_treelen = [len(candidate_tree.tree) for candidate_tree in popul]
    pop_fitness_best = np.min(pop_fitness)
    pop_unique = len(set([str(x.tree) for x in popul]))

    result = {'pop_len': len(popul),
              'pop_unique': pop_unique,
              'lut_symex_fitness-len': len(lut_symex_fitness),
              'time': gen_time,
              'gens_since_last_pareto': gens_since_last_pareto,
              'fit_avg': np.average(pop_fitness),
              'fit_var': np.std(pop_fitness),
              'fit_best': pop_fitness_best,
              'fit_quantile_50': np.quantile(pop_fitness, 0.5),
              'fit_quantile_25': np.quantile(pop_fitness, 0.25),
              'fit_quantile_75': np.quantile(pop_fitness, 0.75),
              'parsim_avg': np.average(pop_parsim),
              'parsim_var': np.std(pop_parsim),
              'parsim_best': np.min(pop_treelen),
              'parsim_quantile_50': np.quantile(pop_parsim, 0.5),
              'parsim_quantile_25': np.quantile(pop_parsim, 0.25),
              'parsim_quantile_75': np.quantile(pop_parsim, 0.75)
              }
    return result


def selection_tournament(pop, n=3):
    """
    Survival of the fittest
    Returns the fittest from n random trees of the last population
    """
    tree_list = [np.random.choice(pop) for _ in range(n)]
    fintree: 'Candidate' = min(tree_list, key=lambda tree: tree.get_fitness())
    evotree = fintree.get_evotree()
    evotree = copy.deepcopy(evotree)
    return evotree

def eval_predict_sympyBatch(sy_expr: sympy.Basic, _df: pd.DataFrame, symbol_list) -> pd.Series:
    """
    Evaluation with Sympy
    """

    symbol_list_str = [str(s) for s in symbol_list]

    # Required functions for lambdify (poor native handling ofdimensionslity, ...)
    sy_np_handling = {'Abs': Abs.np_fun, 'RoundDummy': RoundDummy.np_round_dummy,
                 'Min': Min.np_fun, 'Max': Max.np_fun}
    func = sympy.lambdify(symbol_list, sy_expr, modules=[sy_np_handling, 'numpy'])

    with warnings.catch_warnings():
        with ignore_warnings(RuntimeWarning):  # often in ITE-terms? When math errors occur
            with ignore_warnings(DeprecationWarning):  # something 'like use "**" instead of "Pow"'
                df_results = _df.apply(lambda row: func(*[row[s] for s in symbol_list_str]), axis=1)

    return df_results


if __name__ == '__main__':

    ns = {
        'a': sympy.Symbol('a', real=True),
        'b': sympy.Symbol('b', real=True),
        'c': sympy.Symbol('c', bool=True),
        'd': sympy.Symbol('d', bool=True),
    }

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
            except AttributeError as e:
                print(f'Could not get {x}: {e}')
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

    ndclasses = all_typus_subclasses()
    for ncls in ndclasses:
        if ncls in [ExprCondPair_Dummy]:
            continue

        if ncls in [DivFraction]:
            pass

        d = {float: lambda: np.random.random(),
             bool: lambda: np.random.choice([True, False])}
        try:
            xtype_me = ncls.xtype[1]
            if issubclass(ncls, ChainableOp):
                xtype_childs = ncls.xtype_input
                inputs_sy = [d.get(xtype_childs)() for _ in range(4)]
            else:
                xtype_childs = ncls.xtype[0]
                inputs_sy = [d.get(x)() for x in xtype_childs]
            inputs_np = np.array([[x] for x in inputs_sy])
            symf = ncls.symfun
            np_fun = ncls.np_fun
            res_sy = symf(*inputs_sy)  # symfun(*inputs_sy), np_fun(*inputs_np)
            res_np = np_fun(*inputs_np)
            res_sy = xtype_me(res_sy)
            res_np = xtype_me(res_np)
            if abs(res_sy-res_np) < 0.0001:
                print(ncls.__name__, res_sy, res_np)
            else:
                print('FAILED!', ncls.__name__, res_sy, res_np, (res_sy - res_np), inputs_sy)
                pass
        except Exception as ex:
            if ncls in [ExprCondPair_Dummy, Piecewise, Boolean, Number, Symbol]:
                pass
            else:
                raise Exception(ncls.__name__, 'Exception!', ex)
    # Example pandas DataFrame
    data = {
        'cartPos': [0, np.pi / 4, np.pi / 2, np.pi],
        'cartVel': [0.1, 0.2, 0.3, 0.4]
    }
    df = pd.DataFrame(data)
    test_tree = PowRounded(Number(3), Number(2))
    # sfeh test hieraus machen, soll am ende von einem testlauf durchlaufen

    expr = 'Mul(289, cartVel, Add(cartPos, Mul(2.27, cartVel)), Sin(PowRounded(12, cartPos)))'
    tr = PowRounded(Number(12), Symbol(sympy.S('cartPos')))
    ex = 'Mul(12, cartPos)'
    ta = PowRounded(Number(12), Symbol(sympy.Symbol('cartPos')))
    tb = Mul(Number(289), Symbol(sympy.Symbol('cartVel')),
            Add( Symbol(sympy.Symbol('cartPos')),
                 Mul(Number(2.27), Symbol(sympy.Symbol('cartVel')))),
            Sin(PowRounded(Number(12), Symbol(sympy.Symbol('cartPos')))))
    print(tb)
    print(tb.get_sympy_expr())

    print(RoundDummy(sympy.Float(-0.1)))       # sollte -0
    print(RoundDummy(sympy.Float(0.49)))       # sollte 0
    print(RoundDummy(sympy.Symbol('x') + 1.13))  # sollte eine Zahl liefern, wenn x ersetzt wird

    # sympy problems:
    #     # this may make the expression bigger??
    #     # _r2 = 64.0*cartPos**2*(cartPos + Abs(cartVel) + 1.15)
    #     # _r3 = cartPos**2*(64.0*cartPos + 64.0*Abs(cartVel) + 73.5)
