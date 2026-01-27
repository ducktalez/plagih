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
import os
from abc import ABC
from collections import deque

import pandas as pd
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.utilities.exceptions import ignore_warnings

from plagih.paretofront import *
from plagih.tree_complexity.tree_edit_distance import *
from plagih.util import *
from plagih.monitoring import GPMonitor

from typing import Optional, List, Union, Callable, Type, Any, Dict, Tuple, TypeGuard

from dataclasses import dataclass, field


np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


# =============================================================================
# Helper functions for type checking (forward references resolved at runtime)
# =============================================================================

def is_terminal(node: 'Node') -> TypeGuard['Terminal']:
    """Checks if a node is a Terminal (leaf) node.

    Standalone function for use for type-hinting terminal-node functions
    correctly..
    """
    return node.is_term()


def is_number(node: 'Node') -> TypeGuard['Number']:
    """Checks if a node is a Number terminal.

    Standalone function for use for type-hinting number-node functions
    correctly.
    """
    return node.is_number()


class RoundDummy(sympy.Function):  # Not a Math-operator
    """Sympy-compatible rounding function for use in symbolic expressions.

    Workaround for rounding exponents in symbolic computations.
    Evaluates to an integer when the argument is numeric, otherwise
    remains unevaluated for symbolic arguments.

    For implementation details, see: plagih/discoveries/rounding_exponents.py
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
        """Returns a detailed string representation of the node.

        Do NOT use __str__ for debugging - use __repr__ instead.
        Shows fixed node hints and full precision numbers.
        """
        return self.represent_str(show_fixed_hint=True, cut_terms=False)

    def __len__(self):
        """Returns the fair node count of this tree/subtree."""
        return self.len_nodecount_fair()

    def __str__(self):
        """Returns a human-readable string representation of the tree.

        Uses condensed format with cut terms for readability.

        Alternative representation styles (for reference):
        - [Abs, [Abs, [Square, [cartPos]]]]   - nested list
        - Abs(Abs((cartPos)**2))              - function notation
        - Abs(cartPos**2)                     - simplified notation
        - cartPos**2                          - sympy expression
        """
        s = self.represent_str(show_fixed_hint=False, cut_terms=True)
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
        """Returns the list of child nodes.

        For operators, these are Node instances.
        For terminals, childs[0] contains the value (not a Node).
        """
        return self.childs

    def set_childs(self, child_list: Union[list, tuple]):
        """Sets the child nodes and updates their parent/root references.

        Args:
            child_list: List or tuple of Node instances to set as children.

        Raises:
            TypeError: If child_list is not a list or tuple.
        """
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
        """Repairs all backlinks (parent, root, depth) in the subtree.

        Introduced on 23.04.2024 to maintain tree structure integrity.
        Recursively updates all nodes starting from this node.

        Args:
            parent: The parent node reference to set.
            root: The root node reference to set.
            depth: The depth level of this node.
        """
        self.root_node = root
        self.parent_node = parent
        self.depth=depth

        # Only recurse on operator nodes (terminals have values, not Node children)
        if self.has_childs():
            for ii, cc in enumerate(self.get_childs()):
                cc.repair_all(parent=self, root=root or self, depth=depth+1)

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
        """Checks if this node is an operator (non-terminal) node."""
        return issubclass(type(self), BaseOperator)

    def is_term_and_symbol(self) -> bool:
        """Checks if this node is a terminal Symbol node (input variable)."""
        if self.is_term():
            a = issubclass(type(self), Symbol)
            if a:
                return True
        return False

    def is_term(self) -> bool:
        """Checks if this node is a terminal (leaf) node.

        Named 'is_term' instead of 'is_terminal' to distinguish from ExprCondPairs
        which are neither operators nor true terminals.
        """
        result = issubclass(self.__class__, Terminal)
        return result

    def get_typus(self) -> Type['Node']:
        """Returns the class type of this node."""
        nt = self.__class__
        return nt

    def has_childs(self) -> bool:
        """Checks if this node has child nodes (i.e., is not a terminal).

        Explicitly checks node type rather than childs list,
        since ExprCondPair is not a regular operator.
        """
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
        """Exports the tree as executable Python constructor syntax.

        Output format:
        - Operators: ClassName(child1, child2, ...)
        - Terminals: Number(2.5), Boolean(True/False), Symbol(cartVel)
        - Fixed nodes: ClassName(..., is_fix=True)

        The output can be executed in Python if all classes are imported.

        Args:
            cut_terms: If True, truncates numbers for readability.

        Returns:
            String representation that can be eval'd to recreate the tree.
        """

        def format_terminal(node_term: 'Terminal') -> str:
            """Formats a terminal node for readability.

            - Mathematical expression-style (Max(1.49, 0.1*cartPos))
            - Big values without decimals
            - Small values with relevant decimals
            - Very small values as 0.001
            - Booleans as True/False
            """
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
        """Converts this node tree into a SymPy expression.

        Recursively traverses the tree and builds the equivalent SymPy
        expression using each node's symfun.

        Args:
            simplimore: Reserved for future simplification options.

        Returns:
            SymPy expression representing this computation tree.

        Raises:
            SympyImaginaryNumber: If comparison involves complex numbers.
            NotImplementedError: If node type is not supported.
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

            try:
                _r = _sym(*_cs)  # noqa (_sym is definitely assigned)
                # -> AttributeError: 'Xor' object has no attribute '_eval_as_set'
                # -> TypeError: Invalid comparison of non-real asin(15)
            except TypeError as e:
                raise SympyImaginaryNumber(e)

        else:
            raise NotImplementedError

        sympy_expression_check_raise(_r)

        return _r

    def list_terminal_nodes(self) -> List['Node']:
        """Returns a list of all terminal (leaf) nodes in this subtree."""
        base = self.list_mutable_nodes()
        base = [x for x in base if x.is_term()]
        return base

    def force_input_node(self, ev: 'Evolution') -> None:
        """Ensures the tree has at least one input Symbol node.

        If the tree only has constants as terminals, replaces one with
        an input variable to ensure the tree actually uses input data.

        Args:
            ev: Evolution instance providing available symbols.

        Raises:
            TreeError: If no suitable terminal node can be replaced.
        """
        node_list = self.list_terminal_nodes()
        a = [_x.is_term_and_symbol() for _x in node_list]
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
        """Checks if this node is a numeric constant (Number terminal)."""
        return issubclass(type(self), Number)

    def str_as_list(self, cut_terms: bool = False) -> str:
        """Returns a breadth-first nested list representation.

        Example: a+1 -> [Add, [a], [1]]
        Useful for comparing tree structures in debug output.

        Args:
            cut_terms: If True, truncates numbers for readability.
        """
        typus_str = self.showme

        if self.get_childs():
            if issubclass(type(self), BaseOperator):
                childstr = ', '.join([cc.str_as_list(cut_terms=cut_terms) for cc in self.get_childs()])
                typus_str = f'{typus_str}, {childstr}'
            else:
                # terminal nodes
                v = self.get_childs()[0]
                if self.is_number():
                    typus_str = term_format(f'{v}', cut=cut_terms)
                else:
                    typus_str = f'{v}'
        else:
            raise CuriosityError('Holla sfeh')

        return f"[{typus_str}]"

    def get_lut_id(self) -> str:
        """Returns a unique string identifier for lookup table (LUT) caching.

        Used to check if a tree's fitness was already computed.

        Important:
        - Do NOT use str() or str_as_list() as values get rounded
        - Do NOT convert node-lists to strings

        Returns:
            Unique string identifier for this tree structure.
        """

        s = self.represent_str(show_fixed_hint=False)
        return s

    def str_as_expr(self) -> sympy.Basic:
        """Returns the SymPy expression for this tree (alias for get_sympy_expr)."""
        s = self.get_sympy_expr()
        return s

    def get_expr_symlike(self, try_sympify: bool = False, cut_terms: bool = False) -> str:
        """Returns a SymPy-like string representation with infix notation.

        Example: (1 + a) - each step returns a format template like ({} + {})
        and inputs are filled into the placeholders.

        Args:
            try_sympify: Reserved for future sympify attempt.
            cut_terms: If True, truncates numbers for readability.
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
        """Returns all nodes that can be mutated (point or branch mutation).

        Excludes fixed nodes, ExprCondPairs, and Piecewise nodes.

        Args:
            xtype: If provided, only returns nodes with matching output type.

        Returns:
            List of mutable Node instances.

        Note:
            ValueError "'a' cannot be empty unless no samples are taken"
            indicates no mutable nodes exist.
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
        """Returns all nodes and edges for tree visualization.

        Example: a+1 -> nodes: {'+', 'a', '1'}, edges: [('+','a'), ('+','1')]

        Args:
            setid: Unique identifier prefix for node IDs.

        Returns:
            Tuple of (nodes_dict, edges_list) for graph rendering.
        """
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
        """Returns the tree in APTED bracket notation for tree edit distance.

        APTED (All Path Tree Edit Distance) requires this specific format.
        Example: Add(a, 1) -> {Add{Symbol}{Number}}
        """
        return f"{{{self.get_typus()}{''.join([cc.get_apted_notation() for cc in self.get_childs()])}}}"

    def evolve_mutate_filter_gauss(self) -> None:
        """Applies Gaussian noise mutation to numeric terminal nodes.

        Recursively traverses the subtree and adds Gaussian noise (σ=0.1)
        to all Number terminals. Used for fine-tuning constants.

        Mutation strategies (for reference):
        - Single node, all terminals, random branch, intelligent filtering
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
        """Simplifies nodes by grouping/replacing with simpler equivalents.

        Transformations include:
        - a ** 2 -> Square(a)
        - a + b + c -> chained Add
        - x ** 1 -> x
        - x ** -1 -> 1/x
        - Constants near π, φ, etc. replaced with exact values

        Args:
            tolerance: Threshold for matching mathematical constants.

        Ideas for future:
        - Heaviside function detection
        - expr = b + a + a -> terms grouping via as_ordered_terms()
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
        """Returns the raw count of all nodes in this subtree.

        Simple recursive counting without any special handling.
        """
        if self.has_childs():
            return 1 + sum([cc.len_nodecount_raw() for cc in self.get_childs()])
        else:
            return 1  # childs can currently be floats

    def is_typus(self, nt: Type['Node']) -> bool:
        """Checks if this node is an instance of the given Node type.

        Args:
            nt: The Node class to check against.
        """
        r = issubclass(type(self), nt)
        return r

    def is_ExprCdPair(self) -> bool:  # noqa
        """Checks if this node is an expression-condition pair for Piecewise.

        Note: Sometimes matches Dummy-class, sometimes sympy class (chained?).
        """
        r = issubclass(type(self), (ExprCondPair, ExprCondPair_Dummy))
        return r

    def len_nodecount_fair(self) -> int:
        """Returns a fair node count for complexity measurement.

        Adjustments:
        - Usub (unary minus) is not counted as it adds no real complexity
        - For Piecewise, only the largest branch is counted
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
        """Returns the arity (number of expected children) for this node type."""
        return len(self.get_typus().get_child_xts())

    def get_xtype_tuple(self) -> tuple:
        """Returns the full type signature tuple ((input_types), output_type)."""
        return self.xtype

    def get_xtype_self(self) -> Union[Type[float], Type[bool]]:
        """Returns the output type of this node (float or bool)."""
        return self.xtype[1]

    def represent_str(self, show_fixed_hint: bool = True, cut_terms: bool = False) -> str:
        """Returns a string representation suitable for saving/loading trees.

        Similar to str() but with additional information:
        - ':fix' suffix when nodes are fixed (immutable)
        - Full precision or cut terms based on parameters

        Args:
            show_fixed_hint: If True, appends ':fix' to fixed nodes.
            cut_terms: If True, truncates numbers for readability.
        """

        s = self.showme  # class name

        if self.is_term():
            cs = f'{self.get_childs()[0]}'
            if cut_terms:
                cs = term_format(cs, cut=cut_terms)
            else:
                cs = remove_trailing_zeroes(cs)

            s = f'{cs}'

            if self.is_fix and show_fixed_hint:
                s += ':fix'  # discuss: there must be a more natural way to show that...

        else:
            cs = [cc.represent_str(show_fixed_hint=show_fixed_hint, cut_terms=cut_terms) for cc in self.get_childs()]
            cs = ', '.join(cs)
            s = f"{s}({cs})"

        s = f"{s}"  # v1

        return s

    def get_symstr(self) -> str:
        """Returns the name of the associated SymPy function."""
        return self.symfun.__name__

    @classmethod
    def get_child_xts(cls) -> tuple:
        """Returns the expected input types for child nodes."""
        return cls.xtype[0]

    def eval_np_lambdas(self, *args) -> Callable[[pd.DataFrame], np.ndarray]:
        """Returns a lazy-evaluation lambda for NumPy array computation.

        Builds a computation graph without immediate evaluation.
        Advantages: Does not compute unused branches.
        Disadvantages: Harder to debug.

        Args:
            *args: Additional arguments passed to child evaluations.

        Returns:
            Callable that takes a DataFrame and returns computed results.
        """
        raise NotImplementedError(f"eval_np_lambdas not implemented for {type(self).__name__}")

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        """Eagerly evaluates this node using NumPy (debugging-friendly).

        Evaluates all children first, then applies this node's np_fun.
        Easier to debug than lazy evaluation but may compute unused branches.

        Args:
            _df: Input DataFrame with feature columns.
            *args: Additional arguments for evaluation.

        Returns:
            NumPy array with computed results.
        """
        raise NotImplementedError(f"eval_predict_numpy_now not implemented for {type(self).__name__}")


def eval_parsimony(_tree: Node, complexity_measure: str, origin_tree: Optional[Node] = None) -> int:
    """Evaluates the complexity/parsimony of a tree.

    Args:
        _tree: The tree to measure.
        complexity_measure: One of:
            - 'tree_node_count_raw': Simple node count
            - 'tree_node_count_fair': Adjusted count (ignores Usub, etc.)
            - 'tree_edit_distance': Distance from origin_tree
        origin_tree: Reference tree for edit distance calculation.

    Returns:
        Integer complexity score (lower is simpler).

    Note:
        Tree edit distance is inefficient as origin_tree's APTED notation
        is computed on every call.
    """
    if complexity_measure == 'tree_node_count_raw':  # number of nodes
        return _tree.len_nodecount_raw()
    elif complexity_measure == 'tree_node_count_fair':
        return _tree.len_nodecount_fair()
    elif complexity_measure == 'tree_edit_distance':  # tree_edit_distance, fintree-edit-distance
        apted1 = _tree.get_apted_notation()
        apted2 = origin_tree.get_apted_notation()
        distance, mapping = apted_distance(apted1, apted2)
        return distance
    else:
        raise Exception(f'Complexity measurement not available: {complexity_measure}')


def sympy_to_tree(s_expr: sympy.Basic, allow_chain: bool) -> Node:
    """Converts a SymPy expression to a plagih Node tree.

    Recursively builds a tree from sympy expressions using the d_sym2node mapping.
    Important: Handle most specific rules first in the conversion logic.

    Args:
        s_expr: SymPy expression to convert.
        allow_chain: Whether to allow chained operators (Add with >2 args).

    Returns:
        Root Node of the converted tree.

    Raises:
        NotImplementedError: If expression type is not supported.
    """
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


def tree_simplification(_tree: Node, allow_chain: bool) -> Node:
    """Simplifies a tree using SymPy and node grouping.

    Process:
    1. Convert tree to SymPy expression
    2. Convert back to tree (SymPy may simplify)
    3. Apply tree_node_grouping iteratively

    Known patterns handled:
    - sympy.sign -> sign node
    - Piecewise -> if-then-else
    - Power fractions -> sqrt

    Args:
        _tree: The tree to simplify.
        allow_chain: Whether to allow chained operators.

    Returns:
        Simplified tree (may be same object if no changes).
    """
    tree_history = [copy.deepcopy(_tree)]
    expr_sym = _tree.get_sympy_expr()
    # expr_sym2 = sympy.simplify(expr_sym)
    # if str(expr_sym) != str(expr_sym2):
    #     print(f'okke: {expr_sym} // {expr_sym2}')
    # expr_sym3  = tree.get_sympy_expr(simplimore=True)
    # s4 = self.get_sympy_expr()
    # s5 = self.get_expr_raw_fstring()
    # s6 = self.str_as_list()
    # s_export = self.get_tree_export()
    # print(f'{s}\n{s1}\n{s2}\n{s3}\n{s4}\n{s5}\n{s6}\n{s_export}')
    _tree = sympy_to_tree(expr_sym, allow_chain=allow_chain)
    for _ in range(10):
        tree_history.append(copy.deepcopy(_tree))
        _tree.tree_node_grouping(tolerance=0)
        if _ == 6:
            print(tree_history)
            raise CuriosityError
        if str(_tree) == str(tree_history[-1]):
            break

    # print(f'Tree updates\n'
    #       f'\t{tree_copy.represent_str(show_all=False)}\n'
    #       f'a\t{tree_a.represent_str(show_all=False)}\n'
    #       f'b\t{tree_b.represent_str(show_all=False)}\n'
    #       f'c\t{tree_c.represent_str(show_all=False)}')
    # if tree_b.represent_str(show_all=False) != tree_c.represent_str(show_all=False):
    #     raise CuriosityError
    # sfeh:discuss
    # print(f'After simplification:  {len(tree_copy)}\t{tree}')

    if len(tree_history[0]) < len(_tree):
        astr = string_remove_trailing_zeroes(str(tree_history[0].get_sympy_expr()))
        bstr = string_remove_trailing_zeroes(str(_tree.get_sympy_expr()))
        print(f'WHATHAPPENED SFEH\t{astr}')
        for ii, tt in enumerate(tree_history):
            print(f'\t{ii}:\t{tt.str_as_list()}')

        if astr != bstr:
            # sfeh 'a**0.5' does not become 'sqrt(a)'! use rational=True or sympy.S.Half
            print_warning('w', f'Diff in sympy expression?\n\t{astr}\n\t{bstr}')  # raise ex? does not occur after grouping?
            # sfeh Example 03.05.2025
            # 	sin(cartVel**(6450000000*RoundDummy(cartVel))/(cartPos**6450000000*cartVel**6450000000))
            # 	sin(cartVel**(6.45e+9*RoundDummy(cartVel))/(cartPos**6450000000*cartVel**6450000000))
    return _tree

def evolve_reduce_simplicate(_tree: Node, allow_chain: bool, completely: bool = True, force: bool = False) -> Node:
    """Reduces a tree to its simplest form using SymPy simplification.

    Args:
        _tree: The tree to simplify.
        allow_chain: Whether to allow chained operators.
        completely: If True, simplify entire tree. If False, only one random branch.
        force: If True, return simplified tree even if it grew larger.

    Returns:
        Simplified tree, or original if simplification increased size (unless force=True).
    """
    tree_copy = copy.deepcopy(_tree)
    if completely:  # reduce the complete tree
        nodes_lv0 = _tree.get_mutable_rootnodes(extend_lvls=0)  # only required for fixed-core trees
        for cc in nodes_lv0:
            cc2 = tree_simplification(cc, allow_chain)
            cc.set_new_node(cc2)
    else:
        node_list = [n for n in _tree.list_mutable_nodes() if issubclass(n.get_typus(), OperatorArity)]  # ignoring leaf nodes...
        if len(node_list) == 0:
            print_warning('wwww', f'Tree for simplification does not provide operators: {_tree}')
            return _tree
        node = random.choice(node_list)
        node2 = tree_simplification(node, allow_chain)
        node.set_new_node(node2)  # sfeh chosen must be set again? or not? test it at least.
    if force:
        return _tree
    else:
        if len(tree_copy) < len(_tree):
            print_warning('w',
                          f'Tree grew larger during simplification:\n\t{tree_copy.str_as_list()}\n\t{_tree.str_as_list()}')
            print_warning('ww', f'tree_copy: {len(tree_copy)} vs. {len(_tree)}')
            return tree_copy
        else:
            return _tree


def node_deepcopy(_tree: Node) -> Node:
    """Creates a deep copy of a tree.

    Convenience wrapper around copy.deepcopy for Node trees.
    """
    _cpy = copy.deepcopy(_tree)
    return _cpy


class CustomOperator:
    """Marker class for custom operators that are not part of standard sympy."""
    pass


class NodeWithChilds(Node):
    """Base class for nodes that can have child nodes (operators and dummies)."""
    ...


class NodeDummy(NodeWithChilds):
    """Placeholder node class for special cases like Terminal_Dummy and Function_Dummy.

    Used as a structural wrapper without actual computation logic.
    """
    pass


class PleaseUsePartnerOp(NodeDummy):
    """Marker class for operators that should be replaced by their partner operators.

    For example, Ge (>=) and Gt (>) should be rewritten using Lt (<) and Le (<=)
    to reduce operator redundancy in the system.
    """
    pass


class BaseOperator(NodeWithChilds):
    """Base class for all operator nodes in the computation tree.

    Operators are internal nodes that combine their children's values
    using mathematical or logical operations. Each operator defines:
    - symfun: SymPy function for symbolic computation
    - np_fun: NumPy function for numerical evaluation
    - xtype: Input/output type signature ((input_types), output_type)

    Attributes:
        chain: Optional flag indicating if this operator supports chaining.
        is_Atom: Always False for operators (they have children).
    """

    chain: Optional[bool] = None
    is_Atom = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain = kwargs.get('chain')

    def set_chain(self, param: bool) -> None:
        """Sets whether this operator instance uses chained mode."""
        self.chain = param


    def get_np_child_now(self, _df: pd.DataFrame, *args) -> List[np.ndarray]:
        """Evaluates all children and returns results as NumPy arrays.

        Ensures all results are at least 1D arrays for consistent processing.
        """
        # Evaluate children and return as ndarray (no conversion/reshape)
        values = [cc.eval_predict_numpy_now(_df, *args) for cc in self.get_childs()]

        if any((not isinstance(v, np.ndarray)) or (getattr(v, "ndim", 0) == 0) for v in values):
            values = [np.atleast_1d(np.asarray(v)) for v in values]  # debug why

        return values

    def get_np_child_lambdas(self, *args) -> List[Callable[[pd.DataFrame], np.ndarray]]:
        """Returns lazy-evaluation lambdas for all child nodes.

        Each lambda takes a DataFrame and returns computed results.
        """
        ccl = [cc.eval_np_lambdas(*args) for cc in self.get_childs()]
        
        return ccl

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        children = self.get_np_child_now(_df, *args)
        try:
            result = self.np_fun(*children)
        except TypeError as sfeh:
            if 'loop of ufunc does not support argument' in str(sfeh):
                # TypeError: loop of ufunc does not support argument 0 of type int which has no callable sin method
                print('ASDASDASDASD TODO')  # Debug breakpoint
            raise TypeError(sfeh)

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
    """Base class for operators with a fixed arity (number of arguments).

    Most standard operators (Add, Mul, Sin, etc.) inherit from this class.
    The arity is determined by the length of the input types in xtype[0].
    """
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
    """Mixin class for operators that support variable arity (1 to n arguments).

    Chainable operators can accept more children than their base arity.
    For example, Add can chain: Add(a, b, c, d) instead of Add(Add(Add(a, b), c), d).

    Supported operators: Add, Mul, Min, Max, And, Or, Piecewise/Ifte.

    Equivalent to sympy's LatticeOp concept.

    Attributes:
        xtype_chain: Type signature for chained operation.
        xtype_input: Expected input type for each chained argument.
    """
    xtype_chain = None
    xtype_input = None


class MathOperator(BaseOperator):
    """Base class for mathematical operators that produce numeric (float) results.

    Includes arithmetic operations (Add, Mul, Div), power functions (Pow, Sqrt),
    trigonometric functions (Sin, Cos), and other numeric transformations.
    """
    pass


class LogicOperator(OperatorArity):
    """Base class for logical operators that produce boolean results.

    Includes And, Or, Xor, Not operations.
    """
    pass


class RelationalOperator(OperatorArity):
    """Base class for relational/comparison operators.

    Includes Lt (<), Le (<=), Gt (>), Ge (>=), Eq (==), Ne (!=).
    These operators compare numeric values and return boolean results.
    """
    pass


class Trigonometry(MathOperator):
    """Base class for trigonometric functions.

    Includes Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh.
    """
    pass


class BaseMinMax(MathOperator):
    """Base class for Min and Max operators.

    These operators select the minimum or maximum value from their arguments.
    """
    pass


class NoSymCapitalized:
    """Marker class for operators whose sympy equivalents are lowercase.

    In SymPy, some functions like sin(), cos(), log() are lowercase,
    while our class names are capitalized (Sin, Cos, Log).
    This marker helps identify such cases.
    """
    pass


class Terminal(Node):
    """Base class for terminal (leaf) nodes in the computation tree.

    Terminal nodes have no children and represent atomic values:
    - Number: Numeric constants (e.g., 2.3, -1.5)
    - Symbol: Variable references to input data columns
    - Boolean: Boolean constants (True/False)

    Equivalent to sympy.Atom.

    Attributes:
        is_Atom: Always True for terminals.
        _value: The stored value (sympy.AtomicExpr, float, or bool).

    Note:
        Unlike operators, terminal nodes store their value in childs[0]
        rather than having Node children.
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
    """Terminal node representing a boolean constant (True or False).

    Used for logical operations and conditions in Piecewise/Ifte expressions.

    Attributes:
        xtype: ((), bool) - no inputs, outputs boolean.
    """
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
    """Terminal node representing a numeric constant.

    Stores floating-point values with configurable precision (FLOAT_PRECISION).
    When evaluated, returns an array filled with the constant value.

    Attributes:
        xtype: ((), float) - no inputs, outputs float.

    Note:
        Rational numbers are converted to floats to avoid precision issues
        in sympy simplification (e.g., Sqrt(8.0) -> 2*sqrt(6)/3).
    """
    xtype = ((), float)
    symfun = staticmethod(lambda *a: sympy.Float(float(a[0]), FLOAT_PRECISION))
    np_fun = staticmethod(lambda _x: float(_x))
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
    """Terminal node representing a variable/input reference.

    Symbols reference columns in the input DataFrame by name.
    When evaluated, they return the corresponding column values.

    Attributes:
        xtype: ((), float) - no inputs, outputs float (column values).

    Example:
        Symbol('cartPos') references the 'cartPos' column in the DataFrame.
    """
    symfun = staticmethod(lambda *a: a[0])  # no symbol-conversion; no reason for.
    np_fun = None
    xtype = ((), float)
    showme = 'Symbol'


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
    """Addition operator for two or more operands.

    Computes the sum of all child values. Supports chaining for
    more than two operands: Add(a, b, c) = a + b + c.
    """
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
    """Multiplication operator for two or more operands.

    Computes the product of all child values. Supports chaining for
    more than two operands: Mul(a, b, c) = a * b * c.

    Note: np.multiply only works for pairwise multiplication,
    so np.prod with stacking is used instead.
    """
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
    """Reciprocal operator computing 1/x (x^-1).

    Also known as InverseFraction or Reciprocal.

    Warning: np.reciprocal does not work correctly with integers.
    np.reciprocal(2) returns 0, np.reciprocal(2.0) returns 0.5.
    """
    xtype = ((float,), float)
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], sympy.S.NegativeOne))
    np_fun = staticmethod(lambda a: np.reciprocal(a))
    showme = 'DivFraction'
    sy_str = '1/({})'
    repr_str = 'DivFraction{},[{}]'


class NthRoot(MathOperator):
    """N-th root operator: NthRoot(x, n) computes x^(1/n).

    Currently untested. Use with caution.
    """
    xtype = ((float, float), float)
    symfun = staticmethod(lambda *a: sympy.root(a[0], a[1]))
    np_fun = staticmethod(lambda base, n: np.power(base, 1 / n))
    showme = 'NthRoot'
    sy_str = 'root({}, {})'
    repr_str = 'NthRoot{},[{}, {}]'


class Pow(MathOperator):
    """Power operator: Pow(base, exp) computes base^exp."""
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], a[1]) if len(a) == 2 else None)
    np_fun = staticmethod(lambda base, exp: np.power(base, exp))
    showme = 'Pow'
    sy_str = '({0})**({1})'
    repr_str = 'Pow{},[{},{}]'
    xtype = ((float, float), float)


class Abs(MathOperator):
    """Absolute value operator: Abs(x) returns |x|."""
    symfun = staticmethod(lambda *a: sympy.Abs(a[0]))
    np_fun = np.absolute  # np.fabs works only for non-complex numbers
    showme = 'Abs'
    sy_str = 'Abs({})'
    repr_str = 'Abs{},[{}]'
    xtype = ((float,), float)


class Sign(MathOperator, NoSymCapitalized):
    """Sign function: returns -1, 0, or 1 depending on the sign of the input."""
    symfun = staticmethod(lambda *a: sympy.sign(a[0]))
    np_fun = np.sign
    showme = 'Sign'
    sy_str = 'sign({})'
    repr_str = 'Sign{},[{}]'
    xtype = ((float,), float)


class Log(MathOperator, NoSymCapitalized):
    """Natural logarithm (base e): Log(x) computes ln(x).

    Note: Input should be positive to avoid undefined results.
    """
    symfun = staticmethod(lambda *a: sympy.log(a[0]))
    np_fun = np.log
    showme = 'Log'
    sy_str = 'log({})'
    repr_str = 'Log{},[{}]'
    xtype = ((float,), float)


class Cos(Trigonometry, NoSymCapitalized):
    """Cosine function: Cos(x) computes cos(x) in radians."""
    symfun = staticmethod(lambda *a: sympy.cos(a[0]))
    np_fun = np.cos
    showme = 'Cos'
    sy_str = 'cos({})'
    repr_str = 'Cos{},[{}]'
    xtype = ((float,), float)


class Sin(Trigonometry, NoSymCapitalized):
    """Sine function: Sin(x) computes sin(x) in radians."""
    symfun = staticmethod(lambda *a: sympy.sin(a[0]))
    np_fun = np.sin
    showme = 'Sin'
    sy_str = 'sin({})'
    repr_str = 'Sin{},[{}]'
    xtype = ((float,), float)


class Tan(Trigonometry, NoSymCapitalized):
    """Tangent function: Tan(x) computes tan(x) in radians."""
    symfun = staticmethod(lambda *a: sympy.tan(a[0]))
    np_fun = np.tan
    showme = 'Tan'
    sy_str = 'tan({})'
    repr_str = 'Tan{},[{}]'
    xtype = ((float,), float)


class Acos(Trigonometry, NoSymCapitalized):
    """Inverse cosine (arccosine): Acos(x) returns arccos(x) in radians."""
    symfun = staticmethod(lambda *a: sympy.acos(a[0]))
    np_fun = np.arccos  # arccosh
    showme = 'Acos'
    sy_str = 'acos({})'
    repr_str = 'Acos{},[{}]'
    xtype = ((float,), float)


class Asin(Trigonometry, NoSymCapitalized):
    """Inverse sine (arcsine): Asin(x) returns arcsin(x) in radians."""
    symfun = staticmethod(lambda *a: sympy.asin(a[0]))
    np_fun = np.arcsin
    showme = 'Asin'
    sy_str = 'asin({})'
    repr_str = 'Asin{},[{}]'
    xtype = ((float,), float)


class Atan(Trigonometry, NoSymCapitalized):
    """Inverse tangent (arctangent): Atan(x) returns arctan(x) in radians."""
    symfun = staticmethod(lambda *a: sympy.atan(a[0]))
    np_fun = np.arctan
    showme = 'Atan'
    sy_str = 'atan({})'
    repr_str = 'Atan{},[{}]'
    xtype = ((float,), float)


class Tanh(Trigonometry, NoSymCapitalized):
    """Hyperbolic tangent: Tanh(x) returns tanh(x)."""
    symfun = staticmethod(lambda *a: sympy.tanh(a[0]))
    np_fun = np.tanh
    showme = 'Tanh'
    sy_str = 'tanh({})'
    repr_str = 'Tanh{},[{}]'
    xtype = ((float,), float)


class Sinh(Trigonometry, NoSymCapitalized):
    """Hyperbolic sine: Sinh(x) returns sinh(x)."""
    symfun = staticmethod(lambda *a: sympy.sinh(a[0]))
    np_fun = np.sinh
    showme = 'Sinh'
    sy_str = 'sinh({})'
    repr_str = 'Sinh{},[{}]'
    xtype = ((float,), float)


class Cosh(Trigonometry, NoSymCapitalized):
    """Hyperbolic cosine: Cosh(x) returns cosh(x)."""
    symfun = staticmethod(lambda *a: sympy.cosh(a[0]))
    np_fun = np.cosh
    showme = 'Cosh'
    sy_str = 'cosh({})'
    repr_str = 'Cosh{},[{}, {}]'
    xtype = ((float,), float)


class Not(LogicOperator):
    """Logical NOT operator: Not(a) returns ~a (negation)."""
    symfun = staticmethod(lambda *a: sympy.Not(a[0]))
    np_fun = np.logical_not
    showme = 'Not'
    sy_str = '~({})'
    repr_str = 'Not{},[{}]'
    xtype = ((bool,), bool)


class Eq(RelationalOperator):
    """Equality comparison: Eq(a, b) returns True if a == b."""
    symfun = staticmethod(lambda *a: sympy.Eq(a[0], a[1]))
    np_fun = np.equal
    showme = 'Eq'  # '==' not working in sympy!
    sy_str = 'Eq({0}, {1})'
    repr_str = 'Eq{},[{}, {}]'
    xtype = ((float, float), bool)


class Ne(RelationalOperator):
    """Inequality comparison: Ne(a, b) returns True if a != b."""
    symfun = staticmethod(lambda *a: sympy.Ne(a[0], a[1]))
    np_fun = np.not_equal
    showme = 'Ne'  # != not working in sympy
    sy_str = 'Ne({0}, {1})'
    repr_str = 'Ne{},[{}, {}]'
    xtype = ((float, float), bool)


class And(LogicOperator, ChainableOp):
    """Logical AND operator for two or more boolean inputs.

    Supports chaining: And(a, b, c) = a & b & c.
    """
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
    """Logical OR operator for two or more boolean inputs.

    Supports chaining: Or(a, b, c) = a | b | c.
    """
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
    """Logical XOR (exclusive or) operator.

    Caution: The sympy representation '(a ^ b)' is interpreted as a**b in Python.
    Supports chaining for multiple operands.
    """
    symfun = staticmethod(lambda *a: sympy.Xor(*a))
    np_fun = staticmethod(lambda *a: np.logical_xor.reduce(a))
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
    """Logical if-then-else for boolean results only.

    ITE(condition, then_value, else_value) returns then_value if condition
    is True, otherwise else_value. All arguments must be boolean.

    For numeric results, use Ifte instead.
    """
    symfun = staticmethod(lambda *a: sympy.ITE(a[0], a[1], a[2]))
    np_fun = staticmethod(lambda a, b, c: ((a & b) | (not a) & c))
    showme = 'ITE'
    sy_str = 'ITE({0}, {1}, {2})'
    repr_str = 'ITE{},[{}, {}, {}]'
    xtype = ((bool, bool, bool), bool)


class Min(BaseMinMax, ChainableOp):
    """Minimum operator: returns the smallest value among inputs.

    Supports chaining: Min(a, b, c) returns min(a, b, c).
    """
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
    """Maximum operator: returns the largest value among inputs.

    Supports chaining: Max(a, b, c) returns max(a, b, c).
    """
    symfun = staticmethod(lambda *a: sympy.Max(*a))
    np_fun = staticmethod(lambda *a: np.maximum.reduce(np.vstack(a), axis=0))
    showme = 'Max'
    sy_str = 'Max({0}, {1})'
    repr_str = 'Max{},[{}, {}]'
    xtype = ((float, float), float)
    xtype_input = float
    symfun_chain = lambda a: sympy.Max(*a)
    np_fun_chain = np.maximum
    sy_str_chain = 'Max({})'
    xtype_chain = ([(float,)], float)

    def __call__(self, a):
        # Handle numerical evaluation (for lambdify or direct calls)
        if isinstance(a, (int, float, np.ndarray, pd.DataFrame)):
            return np.maximum.reduce(a)
        raise TypeError("Unsupported type for numerical evaluation in Min(MinMaxBase, ChainableOp)")


class Lt(RelationalOperator):
    """Less-than comparison: Lt(a, b) returns True if a < b."""
    symfun = staticmethod(lambda *a: sympy.Lt(a[0], a[1]))
    np_fun = np.less
    showme = 'Lt'
    sy_str = '({0} < {1})'
    repr_str = 'Lt{},[{}, {}]'
    xtype = ((float, float), bool)


class Le(RelationalOperator):
    """Less-than-or-equal comparison: Le(a, b) returns True if a <= b."""
    symfun = staticmethod(lambda *a: sympy.Le(a[0], a[1]))
    np_fun = np.less_equal
    showme = 'Le'
    sy_str = '({0} <= {1})'
    repr_str = 'Le{},[{}, {}]'
    xtype = ((float, float), bool)


class Gt(RelationalOperator, PleaseUsePartnerOp):
    """Greater-than comparison: Gt(a, b) returns True if a > b.

    Note: Consider using Lt(b, a) instead to reduce operator redundancy.
    """
    symfun = staticmethod(lambda *a: sympy.Gt(a[0], a[1]))
    np_fun = np.greater
    showme = 'Gt'
    sy_str = '({0} > {1})'
    repr_str = 'Gt{},[{}, {}]'
    xtype = ((float, float), bool)


class Ge(RelationalOperator, PleaseUsePartnerOp):
    """Greater-than-or-equal comparison: Ge(a, b) returns True if a >= b.

    Note: Consider using Le(b, a) instead to reduce operator redundancy.
    """
    xtype = ((float, float), bool)
    symfun = staticmethod(lambda *a: sympy.Ge(a[0], a[1]))
    np_fun = np.greater_equal
    showme = 'Ge'
    sy_str = '({0} >= {1})'
    repr_str = 'Ge{},[{}, {}]'


class Square(MathOperator):
    """Square operator: Square(x) computes x^2."""
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], 2))
    np_fun = np.square
    xtype = ((float,), float)
    showme = 'Square'
    sy_str = '({})**2'
    repr_str = 'Square{},[{}]'


class Exp(MathOperator):
    """Exponential function: Exp(x) computes e^x."""
    symfun = staticmethod(lambda *a: sympy.exp(a[0]))
    np_fun = np.exp
    showme = 'Exp'
    sy_str = '{}**E'
    repr_str = 'Exp{},[{}, {}]'
    xtype = ((float,), float)


class Exp2(MathOperator):
    """Base-2 exponential: Exp2(x) computes 2^x."""
    symfun = staticmethod(lambda *a: sympy.Pow(2, a[0]))
    np_fun = np.exp2
    xtype = ((float,), float)
    showme = 'Exp2'
    sy_str = '2**({})'
    repr_str = 'Exp2{},[{}]'


class Sub(MathOperator):
    """Subtraction operator: Sub(a, b) computes a - b."""
    xtype = ((float, float), float)
    symfun = staticmethod(lambda *a: sympy.Add(a[0], -a[1]))
    np_fun = np.subtract
    showme = 'Sub'
    sy_str = '({0} - {1})'
    repr_str = 'Sub{},[{}, {}]'


class Ifte(OperatorArity):
    """Numeric if-then-else operator (Piecewise).

    Ifte(condition, then_value, else_value) returns then_value if condition
    is True, otherwise else_value. Returns numeric (float) results.

    For boolean results, use ITE instead.
    """
    xtype = ((bool, float, float), float)
    symfun = staticmethod(lambda *a: sympy.Piecewise((a[1], a[0]), (a[2], True)))
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
    """Piecewise function for multiple condition-expression pairs.

    Evaluates conditions in order and returns the expression corresponding
    to the first True condition. The last pair should have True as condition
    to serve as the default/else case.

    Internal representation uses ExprCondPair_Dummy for each (expression, condition) pair.
    """
    symfun = staticmethod(lambda *a: sympy.Piecewise(*a))
    np_fun = None
    showme = 'Piecewise'
    sy_str = 'Piecewise({})'
    formulae_str = 'Piecewise({})'
    repr_str = 'Piecewise{},[{}]'
    xtype = ([(ExprCondPair,)], float)
    xtype_chain = ExprCondPair
    xtype_input = ExprCondPair

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        # ...existing code...
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
            result = pairs[-1][0](_df)  # Default case
            for _expr, _cond in reversed(pairs[:-1]):
                result = np.where(_cond(_df), _expr(_df), result)
            return result

        return piecewise_lambda


class Round(MathOperator):
    """Rounding operator: Round(x) rounds x to the nearest integer.

    Uses banker's rounding (round half to even) via RoundDummy for sympy compatibility.
    """
    xtype = ((float,), float)
    symfun = staticmethod(lambda *a: RoundDummy(a[0]))
    np_fun = staticmethod(lambda x: np.vectorize(lambda v: int(round(float(v))))(x))
    showme = 'Round'
    sy_str = 'RoundDummy({},1)'
    repr_str = 'RoundDummy{},[{}]'


class PowRounded(MathOperator):
    """Power operator with rounded exponent: PowRounded(base, exp) computes base^round(exp).

    Useful to constrain exponents to integer values during evolution,
    avoiding fractional powers that can produce complex numbers.
    """
    symfun = staticmethod(lambda *a: sympy.Pow(a[0], RoundDummy(a[1])))
    np_fun = staticmethod(lambda base, exponent: np.power(base, np.vectorize(lambda _x: int(round(float(_x))))(exponent)))
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
    """Division operator: Div(a, b) computes a / b.

    Implemented as a * b^(-1) since sympy.div() only works for polynomials.
    """
    symfun = staticmethod(lambda *a: sympy.Mul(a[0], sympy.Pow(a[1], -1)))
    np_fun = staticmethod(np.divide)
    showme = 'Div'
    sy_str = '({0}/{1})'
    repr_str = 'Div{},[{}, {}]'
    xtype = ((float, float), float)


class Sqrt(MathOperator):
    """Square root operator: Sqrt(x) computes x^0.5.

    In SymPy, sqrt(x) is equivalent to x**Rational(1, 2).
    """
    xtype = ((float,), float)
    symfun = staticmethod(lambda *a: sympy.sqrt(a[0]))
    np_fun = staticmethod(np.sqrt)
    showme = 'Sqrt'
    sy_str = 'sqrt({})'
    repr_str = 'Sqrt{},[{}, {}]'


class Usub(MathOperator):
    """Unary negation operator: Usub(x) computes -x.

    Note: This operator is typically not counted in parsimony/complexity
    measures as it doesn't add significant complexity.
    """
    xtype = ((float,), float)
    symfun = staticmethod(lambda a, *_: sympy.Mul(-1, a))
    np_fun = staticmethod(lambda x: np.negative(x))
    showme = 'Usub'
    sy_str = '(-{})'
    repr_str = 'Usub{},[{}]'


class Clip(BaseMinMax, CustomOperator):
    """Clipping operator: Clip(x, min, max) constrains x to [min, max].

    Equivalent to Min(Max(x, min), max).
    """
    symfun = staticmethod(lambda *a: sympy.Min(sympy.Max(a[0], a[1]), a[2]))
    np_fun = np.clip
    showme = 'Clip'
    sy_str = '(sympy.Min(sympy.Max({0}, {1}), {2}))'
    repr_str = 'Clip{},[{}, {}]'
    xtype = ((float, float, float), float)


class ExprCondPair_Dummy(NodeDummy):  # noqa
    """Wrapper for expression-condition pairs in Piecewise expressions.

    Each pair consists of (expression, condition) where the expression
    is returned if the condition evaluates to True.

    Named differently from sympy.ExprCondPair to avoid confusion.
    """
    arity = 2
    symfun = staticmethod(lambda *a: ExprCondPair(a[0], a[1]))
    np_fun = None
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
    # Handle Python booleans - sympify('True') returns Python True, not sympy.true
    if isinstance(expr_sym, bool):
        return sympy.true if expr_sym else sympy.false
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

        # Handle Python booleans - sympify('True') returns Python True, not sympy.true
        if isinstance(expr_sym, bool):
            return sympy.true if expr_sym else sympy.false

        sympy_expression_check_raise(expr_sym)
        return expr_sym

    except ValueError as e:
        raise ValueError(f'NaN in {e}')
    except AttributeError as e:
        # This can happen when sympifying expressions that return non-sympy types
        if isinstance(_expr, bool):
            return sympy.true if _expr else sympy.false
        raise e


class Candidate:
    """A finalized individual in the genetic programming population.

    Combines a computation tree with its evaluated fitness and complexity metrics.
    Tracks the evolutionary history through tags indicating which operations
    created or modified this candidate.

    Attributes:
        tree: The computation tree (Node) representing the symbolic expression.
        fitness: The evaluated fitness score (lower is better).
        parsimony: The complexity/size measure of the tree.
        tag: Deque tracking the evolution history (max 10 entries).
    """

    def __init__(self, _tree: Node, fitness, parsimony, tag: str):
        self.tree = _tree
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


def selection_tournament(population: List[Candidate], n: int = 3) -> Node:
    """Selects an individual from population using tournament selection.

    Randomly samples n individuals and returns the fittest one's tree.
    Lower fitness is better (minimization).

    Args:
        population: List of Candidate objects to select from.
        n: Tournament size (number of individuals to compare).

    Returns:
        Deep copy of the winning candidate's tree.

    Raises:
        ValueError: If population is empty.
    """
    if not population:
        raise ValueError("Cannot select from empty population")

    # Sample n candidates (with replacement if n > len)
    tournament = random.choices(population, k=min(n, len(population)))

    # Select best (lowest fitness)
    winner = min(tournament, key=lambda c: c.get_fitness())

    # Return deep copy of tree to avoid modifying original
    return copy.deepcopy(winner.get_evotree())


def eval_predict_sympyBatch(sy_expr: sympy.Basic, _df: pd.DataFrame, symbol_list) -> pd.Series:
    """Evaluates a SymPy expression on a DataFrame using lambdify.

    Uses sympy.lambdify for vectorized evaluation with custom NumPy handlers
    for functions like Abs, RoundDummy, Min, Max that need special treatment.

    Args:
        sy_expr: SymPy expression to evaluate.
        _df: DataFrame with input columns.
        symbol_list: List of symbols matching DataFrame columns.

    Returns:
        Series with evaluated results for each row.
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


def check_operator_pool(ops: Dict[Type[BaseOperator], float]) -> None:
    """Validates that the operator pool allows type closure.

    Closure means the system can generate any required type:
    - Operators producing float from float
    - Operators producing bool from bool
    - Operators converting between types (float->bool, bool->float)

    Args:
        ops: Dict mapping operator classes to selection weights.

    Raises:
        Exception: If operators don't allow closure.

    Example (float-only, works):
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
    """Normalizes a weighted choice list for numpy.random.choice.

    Transforms [['a', 1], ['b', 2]] -> [('a', 'b'), (0.333, 0.666)]
    Probabilities are normalized to sum to 1.

    Args:
        val_p_tuples: List of [value, weight] pairs.

    Returns:
        [values_tuple, probabilities_tuple] for np.random.choice.
    """
    xx = list(zip(*val_p_tuples))
    # normalizing the probabilities in every case to a sum of 1 (100%)
    psum = sum(xx[1])
    xx[1] = [i / psum for i in xx[1]]
    return xx


def operatorpool_to_picks(d_operator_pool: Dict[Type[BaseOperator], float]) -> Tuple[dict, dict]:
    """Converts operator pool to selection dictionaries.

    Creates two lookup structures:
    - pick_op: Operators grouped by output type (float/bool)
    - pick_op_match: Operators grouped by full xtype signature

    Args:
        d_operator_pool: Dict mapping operator classes to weights.

    Returns:
        Tuple of (pick_op, pick_op_match) dictionaries.
    """
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
    """Node selection utility for random tree generation.

    Manages probability distributions for selecting operators, terminals,
    and constants during tree creation and mutation.

    Attributes:
        pick_op: Probability distributions for operators by output type (float/bool).
        pick_op_match: Probability distributions for operators by full xtype signature.
        pick_symbol: Probability distributions for symbol selection.
        pick_constant: Probability distributions for constant generation.
    """

    def __init__(self, operators: dict, symbol_list: List[sympy.Symbol]):
        """Initialize node selector with operator pool and available symbols.

        Args:
            operators: Dict mapping operator classes to their selection weights.
            symbol_list: List of sympy symbols available as input variables.
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
        """Randomly selects an operator class that produces the given output type.

        Args:
            xt: The required output type (float or bool).

        Returns:
            An operator class (not instance) matching the output type.
        """
        op = np.random.choice(self.pick_op[xt][0], p=self.pick_op[xt][1])  # no (), which would evaluate the op
        return op

    def choose_operator_class_match(self, xtype: tuple) -> Type[BaseOperator]:
        """Selects an operator class matching the exact type signature.

        Args:
            xtype: Full type signature ((input_types), output_type).

        Returns:
            An operator class with matching xtype.
        """
        if CHAIN_implement:
            pass
        op = np.random.choice(self.pick_op_match[xtype][0], p=self.pick_op_match[xtype][1])
        return op

    def choose_terminal_node(self, xt: Union[Type[float], Type[bool]], p_observation: float = 0.5) -> Terminal:
        """Randomly selects a terminal node (Symbol or constant).

        With probability p_observation, tries to select a Symbol (input variable).
        Falls back to constant if no Symbol available for the type.

        Args:
            xt: The required output type (float or bool).
            p_observation: Probability of choosing a Symbol over a constant.

        Returns:
            A Terminal node (Symbol, Number, or Boolean).

        Note:
            Bug fixed: 'expected str|int|long|float|Decimal|Number object but got Node'
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
        """Randomly generates a constant terminal node.

        For float: Samples from normal distribution or random integers.
        For bool: Random True/False.

        Args:
            xt: The required output type (float or bool).

        Returns:
            A Number or Boolean terminal node.
        """
        _v = np.random.choice(self.pick_constant[xt][0], p=self.pick_constant[xt][1])()  # just dist. must be ()
        if xt == float:
            _v = sympy.Float(_v, FLOAT_PRECISION)  #  discuss allow "rational" inputs? 1/3, 3/4, ...
            return Number(_v)  # round FLOAT_PRECISION was here
        else:
            # _v = sympy.logic.boolalg.BooleanAtom(_v)  # discuss: vs. Boolean
            # -> sympy.sympify('And(True, BooleanAtom(False))')
            return Boolean(_v)

    def choose_symbol_node(self, xt: Union[Type[float], Type[bool]]) -> Symbol:
        """Randomly selects a Symbol node from available input variables.

        Similar to choose_terminal_node but always returns a Symbol.

        Args:
            xt: The required output type (float or bool).

        Returns:
            A Symbol terminal node referencing an input column.
        """
        _v = np.random.choice(self.pick_symbol[xt][0], p=self.pick_symbol[xt][1])
        n = Symbol(_v)
        return n


class Evolution:
    """Tree evolution operations for genetic programming.

    Provides methods for creating, mutating, and crossing over computation trees.
    Manages constraints like maximum depth and node count.

    Tree creation strategies:
    - Size measure: depth, node count, weighted node count (parsimony)
    - Architectures: Full, Grow, Ramped Half-and-Half
    - Node selection: Random, weighted random

    Attributes:
        origin_xtype: Output type of the root node (float or bool).
        origin_tree: Optional template tree with fixed structure.
        symbol_list: Available input variables as sympy symbols.
        node_selector: NodeSelect instance for random node generation.
        depth_max: Maximum allowed tree depth.
        nodes_max: Maximum allowed node count.
        complexity_metric: Method for measuring tree complexity.
    """

    operator_presets = {'math_simple':
                        {Add: 2, Mul: 2, Div: 1, Square: 0.75, Abs: 0.5, Sign: 0.5, Sqrt: 0.1, Log: 0.1,
                         Sin: 0.5, Not: 0.5, Lt: 0.5, Le: 0.5, And: 1, Or: 1, Min: 1, Max: 1}}

    def __init__(self, symbol_list=None, origin_xtype=float, operators=None, origin_tree=None,
                 depth_max=10, nodes_max=100, complexity_metric='tree_node_count_fair', allow_chain=None):
        """Initialize evolution with operator pool and constraints.

        Args:
            symbol_list: List of input variable names or sympy symbols.
            origin_xtype: Expected output type (float or bool).
            operators: Dict of operators with weights, or preset name, or list.
            origin_tree: Optional template tree with fixed nodes.
            depth_max: Maximum tree depth.
            nodes_max: Maximum node count.
            complexity_metric: 'tree_node_count_raw', 'tree_node_count_fair', or 'tree_edit_distance'.
            allow_chain: Whether to allow chained operators.
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

    def evolve_prune_tree(self, _tree: Node) -> Node:
        """Prunes a tree to meet depth and node count constraints.

        Strategies:
        - Depth pruning: Replaces nodes exceeding max depth with terminals
        - Node pruning: Randomly replaces branches to reduce total count

        Note: Pruning should ideally be handled during creation, as it
        strongly affects tree structure and randomly removes nodes.

        Args:
            _tree: The tree to prune.

        Returns:
            The pruned tree (modified in place).
        """
        nodelist = _tree.list_mutable_nodes()
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
        prune_amount = len(_tree) - self.nodes_max
        while prune_amount > 0:
            print_warning('wwww', f'Tree too complex: {len(_tree)} > {self.nodes_max}, pruning {prune_amount}.')
            nodelist = _tree.list_mutable_nodes()
            prune_now = 1 + np.random.randint(prune_amount)  # 19 -> prune branch with 1 to max. 19 nodes

            nodelist = [x for x in nodelist if len(x) >= prune_now]  # only (operator-) nodes
            _tree = random.choice(nodelist)
            new_node = self.node_selector.choose_terminal_node(_tree.get_xtype_self())
            new_node.depth = _tree.depth
            _tree.set_new_node(new_node)
            prune_amount = len(_tree) - self.nodes_max

        return _tree

    def evolve_new_tree_depth(self, xt_out: Union[Type[float], Type[bool]], depth_goal: int, p_term: float = 0.0) -> Node:
        """Creates a new random tree with target depth.

        If an origin_tree is set, fills its mutable slots with random branches.
        Otherwise creates a completely random tree.

        Args:
            xt_out: Output type for the root node (float or bool).
            depth_goal: Target maximum depth.
            p_term: Probability of terminating at each node with a terminal.

        Returns:
            A new random tree.
        """

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
        """Creates a new random tree with chained operators allowed.

        Args:
            depth_goal: Target maximum depth.
            xt_out: Output type for the root node.
            p_term: Probability of terminating branches early.

        Returns:
            A new random tree potentially using chained operators.
        """

        evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    def evolve_create_random(self, xt_out: Union[Type[float], Type[bool]], depth_max_local: int, num_rest: int = -1, depth: int = 0, p_term: float = 0.0) -> Node:
        """Recursively creates a random tree/subtree.

        Args:
            xt_out: Required output type for this node.
            depth_max_local: Maximum depth (can be less than self.depth_max).
            num_rest: Remaining node budget (-1 ignores limit).
            depth: Current depth level.
            p_term: Probability of placing a terminal instead of operator.

        Returns:
            A randomly generated subtree.

        Note:
            Node count is not an ideal threshold as it limits depth-spreading.
            Consider pruning at the end and allowing any growth initially.
        """

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

    def evolve_mutate_filter(self, _tree: Node) -> Node:
        """Applies Gaussian mutation to a random subtree's numeric terminals.

        Selects a random mutable node and applies evolve_mutate_filter_gauss
        to all Number terminals in that subtree.

        Args:
            _tree: The tree to mutate.

        Returns:
            The mutated subtree node.
        """

        _nd = random.choice(_tree.list_mutable_nodes())
        _nd.evolve_mutate_filter_gauss()

        return _nd

    def evolve_mutate_point(self, _tree: Node) -> Node:
        """Mutates a single random node while preserving type signature.

        For operators: Replaces with another operator of same arity/type.
        For terminals: Replaces with another terminal of same type.

        Args:
            _tree: The tree to mutate.

        Returns:
            A deep copy of the tree with one node mutated.
        """
        evotree = copy.deepcopy(_tree)

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
        """Replaces a random subtree with a new random branch.

        Args:
            tree: The tree to mutate.
            depth_goal: Target depth for the new branch.
            allow_chain: Whether to allow chained operators.
            p_term: Probability of terminating nodes early.

        Returns:
            The mutated tree (modified in place).
        """
        n_init = len(tree)
        node_list = tree.list_mutable_nodes()
        node = random.choice(node_list)
        xtype_out = node.get_xtype_self()  # ValueError: 'a' cannot be empty unless no samples are taken
        branch = self.evolve_create_random(xtype_out, depth_goal, num_rest=self.nodes_max - n_init, depth=0,
                                           p_term=p_term)
        node.set_new_node(branch)

        return tree

    def evolve_mutate_branch_nodes(self, _tree: Node, nodes_goal, p_term=0.0) -> Node:
        """Replaces a random subtree with a new branch of target node count.

        Args:
            _tree: The tree to mutate.
            nodes_goal: Target number of nodes for the new branch.
            p_term: Probability of terminating the tree at each node.

        Returns:
            The mutated tree.

        Raises:
            NotImplementedError: If tree is None (selection mechanism needed).
        """
        nodes_init = len(_tree)
        if _tree is None:
            raise NotImplementedError('Implement standard selection mechanism')
        nd = _tree.list_mutable_nodes()
        nd = rnd_choice(nd)
        xt_out = nd.get_xtype_self()
        nodes_goal = min(self.nodes_max - (nodes_init - len(nd)), nodes_goal)

        branch = self.evolve_create_random(xt_out, -1, num_rest=nodes_goal, depth=nd.depth, p_term=p_term)
        nd.set_new_node(branch)
        mutated_tree = _tree
        return mutated_tree

    def evolve_crossover(self, aa: Node, bb: Node):
        """Performs subtree crossover between two trees.

        Swaps compatible subtrees between parent trees:
        1. Select a random node in tree aa
        2. Find a compatible node (same output type) in tree bb
        3. Swap the subtrees

        Args:
            aa: First parent tree.
            bb: Second parent tree.

        Returns:
            Tuple (aa, bb) with swapped subtrees, pruned if necessary.

        Raises:
            TreeError: If tree aa has no mutable nodes.
            ValueError: If no compatible nodes can be found.
        """

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
        aa = self.evolve_prune_tree(_tree=aa)
        bb = self.evolve_prune_tree(_tree=bb)

        return aa, bb

    def finalize_tree(self, tree):
        """Finalizes a tree after evolution operations.

        Performs:
        - Ensures tree has at least one input variable
        - Prunes if necessary (should be handled in evolution)
        - Repairs depth values in all nodes

        Currently a no-op placeholder for future validation.

        Args:
            tree: The tree to finalize.
        """
        # sfeh:open
        pass


def randomly_split_range(range_max: int, num_splits: int) -> list[int]:
    """Randomly splits an integer range into parts that sum to range_max.

    Example: split_range(100, 3) -> [33, 15, 52]

    Used for distributing node budget among child branches during tree building.
    Zero is allowed as it terminates a branch with a terminal node.

    Args:
        range_max: Total to split (or -1 to ignore limits).
        num_splits: Number of parts to create.

    Returns:
        List of integers summing to range_max.
    """

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
    """Prints all candidates in a population with colored formatting.

    Alternates between blue and yellow for readability.
    Shows parsimony, fitness, and sympy expression for each tree.

    Args:
        pop: List of Candidate objects to print.
    """
    n = [f'{k.full_string()}' for k in pop]
    n = [f'{BColors.BLUE}{x}' if ii % 2 == 0 else f'{BColors.YELLOW}{x}' for ii, x in enumerate(n)]
    n = [f'{k}\n' if ii % 10 == 9 else f'{k}\t' for ii, k in enumerate(n)]  # stop \n in line 0
    n = ''.join(n)
    n = re.sub(r'\n$', '', n)  # remove trailing \n (\t irrelevant)
    n = f'{n}{BColors.RESET_COLOR}'
    print(n)


def pop_analyze(population: List[Candidate], gen_time: float, gens_since_pareto: int, lut_symex: dict) -> dict:
    """Analyzes population statistics for monitoring.

    Computes fitness and parsimony statistics for the current generation.

    Args:
        population: List of Candidate objects.
        gen_time: Time taken for this generation in seconds.
        gens_since_pareto: Generations since last Pareto front update.
        lut_symex: Lookup table for sympy expressions to fitness.

    Returns:
        Dictionary with population statistics.
    """
    if not population:
        return {
            'pop_len': 0, 'pop_unique': 0, 'lut_symex_fitness-len': len(lut_symex),
            'time': gen_time,
            'fit_avg': np.nan, 'fit_var': np.nan,
            'fit_quantile_25': np.nan, 'fit_quantile_50': np.nan, 'fit_quantile_75': np.nan, 'fit_best': np.nan,
            'parsim_avg': np.nan, 'parsim_var': np.nan,
            'parsim_quantile_25': np.nan, 'parsim_quantile_50': np.nan, 'parsim_quantile_75': np.nan, 'parsim_best': np.nan,
            'gens_since_last_pareto': gens_since_pareto
        }

    fitnesses = np.array([c.get_fitness() for c in population])
    parsimony = np.array([c.get_parsim() for c in population])

    # Count unique expressions
    unique_exprs = set(str(c.tree.get_sympy_expr()) for c in population)

    return {
        'pop_len': len(population),
        'pop_unique': len(unique_exprs),
        'lut_symex_fitness-len': len(lut_symex),
        'time': gen_time,
        'fit_avg': np.mean(fitnesses),
        'fit_var': np.var(fitnesses),
        'fit_quantile_25': np.percentile(fitnesses, 25),
        'fit_quantile_50': np.percentile(fitnesses, 50),
        'fit_quantile_75': np.percentile(fitnesses, 75),
        'fit_best': np.min(fitnesses),
        'parsim_avg': np.mean(parsimony),
        'parsim_var': np.var(parsimony),
        'parsim_quantile_25': np.percentile(parsimony, 25),
        'parsim_quantile_50': np.percentile(parsimony, 50),
        'parsim_quantile_75': np.percentile(parsimony, 75),
        'parsim_best': np.min(parsimony),
        'gens_since_last_pareto': gens_since_pareto
    }


class ExplainableGP:
    """Main class for explainable genetic programming.

    Manages the complete GP workflow including population evolution,
    fitness evaluation, Pareto front maintenance, and monitoring.

    Attributes:
        evolve: Evolution instance for tree operations.
        df_train: Training DataFrame with input features and target.
        rootdir: Output directory for logs, plots, and backups.
        pop_max_size: Maximum population size per generation.
        gen_end: Target number of generations.
        paretofront: List of non-dominated Candidates.
        pop_genepool: Current generation's population.
        lut_tree_infos: Cache for tree metadata (sympy expr, fitness, parsimony).
        lut_symex_fitness: Cache mapping sympy expressions to fitness values.
        monitor_df: DataFrame tracking generation statistics.

    Example:
        # Simple usage with defaults
        gp = ExplainableGP.create(
            symbols=['x', 'y'],
            df_train=my_data,
            rootdir='./results'
        )

        # Or with custom Evolution
        gp = ExplainableGP(
            evolve=my_evolution,
            df_train=my_data,
            rootdir=Path('./results'),
            pop_max_size=50,
            gen_end=20
        )
    """

    # Default error metric: RMSE
    DEFAULT_ERROR_METRIC = staticmethod(lambda pred, true: np.sqrt(np.mean((pred - true) ** 2)))

    # Default autocast: identity (np.array)
    DEFAULT_AUTOCAST = staticmethod(lambda _x: np.asarray(_x, dtype=np.float64))

    def __init__(
        self,
        evolve: Evolution,
        df_train: pd.DataFrame,
        rootdir: Union[Path, str],
        *,  # Force keyword-only args after this
        pop_max_size: int = 100,
        gen_end: int = 100,
        eval_autocast: Optional[Callable] = None,
        eval_error_metric: Optional[Callable] = None,
        allow_chain: bool = False,
        target_column: str = 'action',
        verbose: bool = True
    ):
        """Initialize the GP system.

        Args:
            evolve: Evolution instance with operator pool and constraints.
            df_train: Training data with target column.
            rootdir: Path for output files (str or Path).
            pop_max_size: Maximum individuals per generation. Default: 100.
            gen_end: Number of generations to run. Default: 100.
            eval_autocast: Function to cast predictions. Default: np.array.
            eval_error_metric: Error function(pred, true) -> float. Default: RMSE.
            allow_chain: Whether to allow chained operators. Default: False.
            target_column: Name of target column in df_train. Default: 'action'.
            verbose: Print initialization info. Default: True.
        """
        self.time_start = time.perf_counter()

        # Handle rootdir as str or Path
        self.rootdir = Path(rootdir) if isinstance(rootdir, str) else rootdir
        self.rootdir.mkdir(parents=True, exist_ok=True)

        self.df_train = df_train
        self.target_column = target_column

        self.evolve = evolve
        self.gen_end = gen_end
        self.pop_max_size = pop_max_size
        self.gen_id: int = 0

        # Use defaults if not provided
        self.eval_autocast = eval_autocast or self.DEFAULT_AUTOCAST
        self.eval_error_metric = eval_error_metric or self.DEFAULT_ERROR_METRIC

        self.allow_chain = allow_chain

        if verbose:
            print(f'\n'
                  f'\tInitializing Plagih.\n'
                  f'\tName: {BColors.CYAN}{self.rootdir.name}{BColors.RESET_COLOR}.\n'
                  f'\tLocated in: \n'
                  f'\t{self.rootdir}\n')

        self.paretofront = []
        self.pop_genepool = []
        self.pop_next = []

        self.lut_tree_infos = {}
        self.lut_symex_fitness = {}

        # monitoring
        self.time_genstart = time.perf_counter()
        self.monitor = GPMonitor()

    # =========================================================================
    # Backwards Compatibility Properties
    # =========================================================================

    @property
    def monitor_df(self):
        """Backwards compatible access to monitoring DataFrame.

        Returns the monitor data as a pandas DataFrame with old column names.
        """
        return self.monitor.to_dataframe()

    @property
    def gens_since_last_pareto(self):
        """Backwards compatible access to generations since last Pareto update."""
        return self.monitor.gens_since_last_pareto

    @classmethod
    def create(
        cls,
        symbols: List[Union[str, sympy.Symbol]],
        df_train: pd.DataFrame,
        rootdir: Union[Path, str],
        *,
        operators: Optional[Dict] = None,
        preset: str = 'math_full',
        depth_max: int = 7,
        nodes_max: int = 40,
        pop_max_size: int = 100,
        gen_end: int = 50,
        clip_range: Optional[Tuple[float, float]] = None,
        error_metric: str = 'rmse',
        allow_chain: bool = False,
        target_column: str = 'action',
        verbose: bool = True
    ) -> 'ExplainableGP':
        """Factory method for easy GP creation with sensible defaults.

        Args:
            symbols: List of input variable names or sympy Symbols.
            df_train: Training DataFrame with features and target.
            rootdir: Output directory path.
            operators: Custom operator dict. If None, uses preset.
            preset: Operator preset name ('math_simple', 'math_full', 'with_logic').
            depth_max: Maximum tree depth. Default: 7.
            nodes_max: Maximum nodes per tree. Default: 40.
            pop_max_size: Population size. Default: 100.
            gen_end: Number of generations. Default: 50.
            clip_range: Optional (min, max) to clip predictions.
            error_metric: 'rmse', 'mse', 'mae', or custom callable.
            allow_chain: Allow chained operators. Default: False.
            target_column: Target column name. Default: 'action'.
            verbose: Print info. Default: True.

        Returns:
            Configured ExplainableGP instance.

        Example:
            gp = ExplainableGP.create(
                symbols=['x', 'y'],
                df_train=data,
                rootdir='./run_001',
                preset='math_simple',
                pop_max_size=50,
                gen_end=20
            )
        """
        # Create Evolution with preset or custom operators
        # Evolution accepts: operators as dict, string (preset name), or list
        ops = operators if operators is not None else preset
        evolve = Evolution(
            symbol_list=symbols,
            operators=ops,
            depth_max=depth_max,
            nodes_max=nodes_max,
            allow_chain=allow_chain
        )

        # Setup autocast
        if clip_range:
            def _clip_autocast(x):
                return np.clip(np.asarray(x, dtype=np.float64), clip_range[0], clip_range[1])
            eval_autocast = _clip_autocast
        else:
            eval_autocast = None  # Will use default

        # Setup error metric
        if callable(error_metric):
            eval_error_metric = error_metric
        elif error_metric == 'rmse':
            eval_error_metric = lambda pred, true: np.sqrt(np.mean((pred - true) ** 2))
        elif error_metric == 'mse':
            eval_error_metric = lambda pred, true: np.mean((pred - true) ** 2)
        elif error_metric == 'mae':
            eval_error_metric = lambda pred, true: np.mean(np.abs(pred - true))
        else:
            raise ValueError(f"Unknown error_metric: {error_metric}. Use 'rmse', 'mse', 'mae', or callable.")

        return cls(
            evolve=evolve,
            df_train=df_train,
            rootdir=rootdir,
            pop_max_size=pop_max_size,
            gen_end=gen_end,
            eval_autocast=eval_autocast,
            eval_error_metric=eval_error_metric,
            allow_chain=allow_chain,
            target_column=target_column,
            verbose=verbose
        )

    @classmethod
    def from_config(cls, config: dict, df_train: pd.DataFrame) -> 'ExplainableGP':
        """Create GP instance from configuration dictionary.

        Args:
            config: Dictionary with GP configuration.
            df_train: Training DataFrame.

        Returns:
            Configured ExplainableGP instance.

        Example:
            config = {
                'symbols': ['x', 'y'],
                'rootdir': './results',
                'preset': 'math_simple',
                'pop_max_size': 50,
                'gen_end': 20
            }
            gp = ExplainableGP.from_config(config, my_data)
        """
        return cls.create(df_train=df_train, **config)

    def get_name(self):
        """Returns the name of this GP run (derived from rootdir)."""
        if isinstance(self.rootdir, Path):
            s = self.rootdir.name
        else:
            s = None
        return s

    def run_update_paretofront(self, pop):
        """Updates the Pareto front with non-dominated candidates from pop.

        Minimizes both fitness and parsimony. A candidate dominates another
        if it is better in at least one objective and not worse in any.

        Args:
            pop: Population to extract Pareto-optimal candidates from.

        Returns:
            True if the Pareto front changed, False otherwise.
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

            # Note: gens_since_last_pareto is now tracked by self.monitor

        return changed

    def end_generation(self):
        """Finalizes the current generation and prepares for the next.

        Actions:
        - Updates Pareto front with new candidates
        - Moves pop_next to pop_genepool
        - Prints population summary
        - Runs analysis and monitoring
        - Increments generation counter
        """
        # sfeh:open end generation in every generation
        pareto_updated = self.run_update_paretofront(self.pop_next)
        # Note: gens_since_last_pareto is now tracked by self.monitor

        self.pop_genepool = self.pop_next[:]
        print_pop(self.pop_next)
        self.pop_next = []
        self.analyze_generation(pareto_updated=pareto_updated)
        self.gen_id += 1

        self.time_genstart = time.perf_counter()

    def gen_create_initial(self, origin_tree=None):
        """Creates the initial population (generation 0).

        If an origin_tree is provided, adds it as a candidate.
        Otherwise generates random trees with varying depths.

        Args:
            origin_tree: Optional seed _tree to include in initial population.

        Returns:
            The initial population (pop_genepool).
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
                    _tree = self.evolve.evolve_new_tree_depth(float, n, p_term=0)
                    return _tree

                @self.create_trees(rate=0.5)
                def init_rand2a():
                    n = np.clip(int(random.normalvariate(3.5, 1.0)), 3, self.evolve.depth_max)
                    return self.evolve.evolve_new_tree_depth(float, n, p_term=0)

        self.paretofront = pareto_from_pop(self.pop_next)
        self.pop_genepool = self.pop_next[:]
        self.pop_next = []
        self.analyze_generation(pareto_updated=True)  # Initial population always updates Pareto
        self.gen_id += 1
        return self.pop_genepool

    def pop_next_append(self, ct: Candidate, force=False):
        """Appends a candidate to the next generation's population.

        Logs the tree expression and adds it to pop_next.

        Args:
            ct: The Candidate to add.
            force: If True, skips minimum parsimony check.
        """
        evotree = ct.get_evotree()
        # from visualization.pygraphviz import render_pygraphviz
        if force and ct.get_parsim() < TREE_MIN_PARSIMONY:
            # raise ValueError(f'Tree not complex enough for population, sfeh')
            return
        printpl('gggg', f'|->{evotree.len_nodecount_fair():2.0f}: {evotree.str_as_expr()}')
        self.pop_next.append(ct)

    def create_trees(self, rate=0.0, crossover=False, simplicate=False, allow_chain=False):
        """Decorator factory for safely creating and adding trees to the population.

        Wraps a tree creation function to handle errors, apply simplification,
        and convert trees to Candidates.

        Args:
            rate: Fraction of pop_max_size to create (0.0 to 1.0).
            crossover: If True, expects function to return two trees.
            simplicate: If True, applies tree_simplification before evaluation.
            allow_chain: Whether to allow chained operators in simplification.

        Returns:
            Decorator function that wraps tree creation logic.
        """

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
        """Converts a tree to a fully evaluated Candidate.

        Process:
        1. Ensures tree has input variables
        2. Prunes if necessary
        3. Computes sympy expression
        4. Evaluates fitness using NumPy
        5. Computes parsimony

        Uses lookup tables to avoid redundant computation.

        Args:
            evotree: The tree to convert.
            origin_tree: Reference tree for edit distance (if used).
            tag: Label indicating which evolution created this tree.
            raise_if_useless: If True, raises error for oversized trees.
            compare_with_sympy: If True, validates NumPy results against SymPy.

        Returns:
            A Candidate object with tree, fitness, and parsimony.

        Raises:
            TreeLutError: If cached tree had errors.
            TreeSizeError: If tree exceeds max nodes.
            SympyError: If sympy expression cannot be created.
        """

        # Make this tree usable for evaluation
        evotree.force_input_node(self.evolve)
        evotree = self.evolve.evolve_prune_tree(evotree)
        evotree.repair_depth()

        tree_id = evotree.get_lut_id()

        if tree_id in self.lut_tree_infos:
            sy_expr = self.lut_tree_infos[tree_id].get('sy_expr')  # Attention: can be "False"
            parsimony = self.lut_tree_infos[tree_id].get('parsimony')
            fitness = self.lut_tree_infos[tree_id].get('fitness')
            if any(v is None for v in [sy_expr, parsimony, fitness]):
                _err = self.lut_tree_infos[tree_id].get('error')
                raise TreeLutError(f'Tree LUT Entry implies Problem: {_err}')
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

                            # =========================================================
                            # BENCHMARK: New EvaluationContext System vs. Old Methods
                            # =========================================================
                            from plagih.evaluation_context import EvaluationContext, create_context
                            import time as bench_time

                            bench_results = {}

                            # 1. NumPy Eager (new context)
                            t0 = bench_time.perf_counter()
                            ctx_np = create_context('numpy_eager', use_lut=False)
                            result_np = ctx_np.evaluate(evotree, self.df_train)
                            bench_results['1_numpy_ctx'] = bench_time.perf_counter() - t0

                            # 2. NumPy Lambda (new context)
                            t0 = bench_time.perf_counter()
                            ctx_lambda = create_context('numpy_lambda', use_lut=False)
                            result_lambda_fn = ctx_lambda.evaluate(evotree)
                            result_lambda = result_lambda_fn(self.df_train)
                            bench_results['2_lambda_ctx'] = bench_time.perf_counter() - t0

                            # 3. SymPy (new context)
                            t0 = bench_time.perf_counter()
                            ctx_sympy = create_context('sympy', use_lut=False)
                            result_sympy = ctx_sympy.evaluate(evotree)
                            bench_results['3_sympy_ctx'] = bench_time.perf_counter() - t0

                            # 4. All together without LUT
                            t0 = bench_time.perf_counter()
                            ctx_all = EvaluationContext(
                                modes=['numpy_eager', 'numpy_lambda', 'sympy'],
                                use_lut=False
                            )
                            results_all = ctx_all.evaluate(evotree, self.df_train)
                            bench_results['4_all_no_lut'] = bench_time.perf_counter() - t0

                            # 5. All together WITH LUT (second call should be faster)
                            t0 = bench_time.perf_counter()
                            ctx_lut = EvaluationContext(
                                modes=['numpy_eager', 'numpy_lambda', 'sympy'],
                                use_lut=True
                            )
                            results_lut1 = ctx_lut.evaluate(evotree, self.df_train)  # First call (cache miss)
                            results_lut2 = ctx_lut.evaluate(evotree, self.df_train)  # Second call (cache hit!)
                            bench_results['5_all_with_lut'] = bench_time.perf_counter() - t0

                            # Compare with OLD methods timing
                            t0 = bench_time.perf_counter()
                            old_np = evotree.eval_predict_numpy_now(self.df_train)
                            bench_results['OLD_numpy'] = bench_time.perf_counter() - t0

                            t0 = bench_time.perf_counter()
                            old_lambda = evotree.eval_np_lambdas()(self.df_train)
                            bench_results['OLD_lambda'] = bench_time.perf_counter() - t0

                            t0 = bench_time.perf_counter()
                            old_sympy = evotree.get_sympy_expr()
                            bench_results['OLD_sympy'] = bench_time.perf_counter() - t0

                            # Print benchmark results
                            printpl('pp', f'=== EvaluationContext Benchmark ===')
                            printpl('pp', f'  1. NumPy (ctx):     {bench_results["1_numpy_ctx"]*1000:6.2f}ms | OLD: {bench_results["OLD_numpy"]*1000:6.2f}ms')
                            printpl('pp', f'  2. Lambda (ctx):    {bench_results["2_lambda_ctx"]*1000:6.2f}ms | OLD: {bench_results["OLD_lambda"]*1000:6.2f}ms')
                            printpl('pp', f'  3. SymPy (ctx):     {bench_results["3_sympy_ctx"]*1000:6.2f}ms | OLD: {bench_results["OLD_sympy"]*1000:6.2f}ms')
                            printpl('pp', f'  4. All (no LUT):    {bench_results["4_all_no_lut"]*1000:6.2f}ms')
                            printpl('pp', f'  5. All (with LUT):  {bench_results["5_all_with_lut"]*1000:6.2f}ms (2 evals)')
                            printpl('pp', f'  LUT Stats: {ctx_lut.get_cache_size()} entries, hit-rate: {ctx_lut.get_cache_hit_rate("numpy_eager"):.0%}')

                            # Verify results match
                            np.testing.assert_array_almost_equal(result_np, old_np, decimal=6,
                                err_msg="NumPy context result doesn't match old method!")
                            np.testing.assert_array_almost_equal(result_lambda, old_lambda, decimal=6,
                                err_msg="Lambda context result doesn't match old method!")
                            # =========================================================

                            """Numpy eager eval"""
                            # sfeh _lambda verion comparisson, functionality-wise and time-wise, eval sympy first
                            nplambda_results_raw = evotree.eval_np_lambdas()
                            nplambda_results_raw = nplambda_results_raw(self.df_train)

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
                                sum(nplambda_results_raw - np_results_raw)
                            except Exception as e:
                                raise TreeError('SFEH THESE ARE [True] trees')

                            if sum(nplambda_results_raw - np_results_raw) > 0.001:
                                diffs = np.abs(nplambda_results_raw - np_results_raw)
                                mask = diffs > 0.001
                                if np.any(mask):
                                    indices = np.where(mask)[0]
                                    print_warning('w', f'{len(indices)} differences found above tolerance 0.001: (NP VERSION)')
                                print(f'Different in (NP VERSION): {sum(nplambda_results_raw - np_results)} ({sy_expr})')

                            if np.sum(np.abs(nplambda_results_raw - np_results_raw)) > 0.001:
                                sym_results_raw_np = sym_results_raw.to_numpy()
                                diffs = np.abs(sym_results_raw_np - np_results_raw)
                                mask = diffs > 0.001
                                if np.any(mask):
                                    indices = np.where(mask)[0]
                                    print_warning('w', f'{len(indices)} differences found above tolerance 0.001:')
                                # results_syraw_df = eval_predict_df_sympy_only(sy_expr, self.df_train)  #  takes forever
                                result_diffs = sum(sym_results_raw - np_results_raw)
                                print(f'Different results in evaluation: {result_diffs} sy-expr: ({sy_expr})')

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

    def _save_merged_population_tree(self):
        """Creates and saves a merged population tree visualization.

        Merges all trees in the current population into a single DAG
        and saves a PNG visualization using graphviz or matplotlib fallback.

        Output files:
        - population_merged/Population-merged-gen-XXX.png (visualization)

        Note: Intermediate dot files are cleaned up to keep only PNG output.
        """
        from plagih.population_merge import build_one_evaluation_tree

        try:
            # Build merged graph from current population
            graph = build_one_evaluation_tree(self.pop_genepool)

            # Create output directory
            merged_dir = self.rootdir / 'population_merged'
            merged_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with generation number
            base_filename = f'Population-merged-gen-{self.gen_id:03d}'

            #
            # # Always save DOT file (useful for manual rendering or debugging)
            # dot_path = merged_dir / f'{base_filename}.dot'
            dot_source = graph.to_graphviz_dot()
            # with open(dot_path, 'w', encoding='utf-8') as f:
            #     f.write(dot_source)

            # Try to render PNG with graphviz
            png_created = False
            try:
                from graphviz import Source
                src = Source(dot_source)
                png_path = merged_dir / base_filename
                # Use a timeout-safe approach: render to file
                # cleanup=True removes the .dot file after rendering
                src.render(str(png_path), format='png', cleanup=True)

                # Rename the rendered file from base_filename to base_filename.png
                # (graphviz.render creates filename.png but also leaves filename without extension)
                temp_file = str(png_path)  # This might exist without .png extension
                final_png = str(png_path) + '.png'

                # Clean up any files without extension
                if os.path.exists(temp_file) and temp_file != final_png:
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                png_created = True
                printpl('ggg', f'Saved merged population tree: {base_filename}.png')
            except ImportError:
                printpl('ggg', f'Graphviz not installed. Trying matplotlib fallback...')
                png_created = False  # Fall through to matplotlib
            except Exception as viz_err:
                # Graphviz crashed (common with large graphs)
                printpl('ggg', f'Graphviz render failed: {viz_err}. Trying matplotlib fallback...')
                png_created = False  # Fall through to matplotlib

            # Try alternative: matplotlib-based visualization
            if not png_created:
                try:
                    self._save_merged_tree_matplotlib(graph, merged_dir / f'{base_filename}.png')
                    printpl('ggg', f'Saved merged tree (matplotlib): {base_filename}.png')
                except Exception as mpl_err:
                    printpl('w', f'Could not save merged tree visualization: {mpl_err}')

        except Exception as e:
            printpl('w', f'Could not create merged population tree: {e}')

    def _save_merged_tree_matplotlib(self, graph, output_path):
        """Fallback visualization using matplotlib when graphviz fails.

        Creates a hierarchical layout of the merged graph with:
        - Root nodes at the top, terminals at the bottom
        - Rounded rectangles for nodes (auto-sized to content)
        - No overlapping nodes

        Args:
            graph: MergedEvaluationGraph instance
            output_path: Path to save the PNG file
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.lines import Line2D
        import textwrap

        # Get statistics for title
        stats = graph.get_statistics()

        # Group nodes by depth
        max_depth = max((n.depth for n in graph.nodes.values()), default=0)
        layers = {d: [] for d in range(max_depth + 1)}
        for node in graph.nodes.values():
            layers[node.depth].append(node)

        # Configuration
        base_font_size = 7
        max_label_width = 16  # characters per line before wrapping
        min_x_gap = 0.3  # minimum horizontal gap between node EDGES (not centers)
        min_y_gap = 0.4  # minimum vertical gap between node EDGES
        box_padding_x = 0.08  # padding inside box (horizontal)
        box_padding_y = 0.06  # padding inside box (vertical)

        # Create a temporary figure to measure text sizes accurately
        temp_fig, temp_ax = plt.subplots(figsize=(10, 10))
        temp_fig.canvas.draw()
        renderer = temp_fig.canvas.get_renderer()
        dpi = temp_fig.dpi

        # Prepare labels and measure actual text sizes
        node_labels = {}
        node_sizes = {}  # (width, height) for each node

        for node in graph.nodes.values():
            # Create label
            label = str(node.sympy_expr)

            # Truncate very long expressions
            if len(label) > 35:
                label = label[:32] + '...'

            # Add usage count
            usage = len(node.original_nodes)
            if usage > 1:
                label += f' ({usage}x)'

            # Add root marker
            if node.is_root:
                label = '[R] ' + label

            # Wrap long labels to multiple lines
            if len(label) > max_label_width:
                wrapped = textwrap.wrap(label, width=max_label_width)
                label = '\n'.join(wrapped)

            node_labels[node.node_id] = label

            # Measure actual text size
            txt = temp_ax.text(0, 0, label, fontsize=base_font_size,
                              fontfamily='monospace', linespacing=1.1)
            bbox = txt.get_window_extent(renderer=renderer)
            txt.remove()

            # Convert from pixels to data coordinates
            width_inches = bbox.width / dpi
            height_inches = bbox.height / dpi

            # Scale to data coordinates - use smaller scale
            scale = 6.0
            width = width_inches * scale + 2 * box_padding_x
            height = height_inches * scale + 2 * box_padding_y

            # Ensure minimum size
            width = max(0.4, width)
            height = max(0.25, height)

            node_sizes[node.node_id] = (width, height)

        plt.close(temp_fig)

        # Calculate positions with proper spacing to avoid overlap
        positions = {}

        # Calculate max height per layer for consistent y-spacing
        layer_max_heights = {}
        for depth in range(max_depth + 1):
            nodes = layers[depth]
            if nodes:
                layer_max_heights[depth] = max(node_sizes[n.node_id][1] for n in nodes)
            else:
                layer_max_heights[depth] = 0.5

        # Calculate x positions for each layer
        for depth in range(max_depth + 1):
            nodes = sorted(layers[depth], key=lambda n: n.node_id)
            if not nodes:
                continue

            # Calculate total width needed for this layer (node widths + gaps)
            layer_widths = [node_sizes[n.node_id][0] for n in nodes]
            total_width = sum(layer_widths) + min_x_gap * (len(nodes) - 1)

            # Calculate y position based on cumulative heights of lower layers
            y = 0
            for d in range(depth):
                y += layer_max_heights.get(d, 0.5) + min_y_gap

            # Position nodes centered around 0
            current_x = -total_width / 2
            for node in nodes:
                w, h = node_sizes[node.node_id]
                x = current_x + w / 2  # center of the box
                positions[node.node_id] = (x, y)
                current_x += w + min_x_gap

        # Determine figure size based on graph extent
        all_x = []
        all_y = []
        if positions and any(p[0] is not None for p in positions.values()):
            # Include box extents in calculations
            all_x_min = [p[0] - node_sizes[nid][0]/2 for nid, p in positions.items() if p[0] is not None]
            all_x_max = [p[0] + node_sizes[nid][0]/2 for nid, p in positions.items() if p[0] is not None]
            all_y_min = [p[1] - node_sizes[nid][1]/2 for nid, p in positions.items() if p[0] is not None]
            all_y_max = [p[1] + node_sizes[nid][1]/2 for nid, p in positions.items() if p[0] is not None]

            x_range = max(all_x_max) - min(all_x_min) + 2
            y_range = max(all_y_max) - min(all_y_min) + 2

            # Dynamic figure size based on content
            fig_width = max(14, min(50, x_range * 1.0))
            fig_height = max(8, min(35, y_range * 1.0))
        else:
            fig_width, fig_height = 14, 10

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        # Draw edges first (so they're behind nodes)
        for node in graph.nodes.values():
            if node.node_id not in positions:
                continue
            x1, y1 = positions[node.node_id]
            h1 = node_sizes[node.node_id][1]

            for child_id in node.child_ids:
                if child_id not in positions:
                    continue
                x2, y2 = positions[child_id]
                h2 = node_sizes[child_id][1]

                # Draw line from bottom of parent to top of child
                y1_bottom = y1 - h1 / 2
                y2_top = y2 + h2 / 2

                ax.plot([x1, x2], [y1_bottom, y2_top],
                       color='#888888', linewidth=1.2, zorder=1)

        # Draw nodes
        for node in graph.nodes.values():
            if node.node_id not in positions:
                continue
            x, y = positions[node.node_id]
            w, h = node_sizes[node.node_id]
            label = node_labels[node.node_id]

            # Color based on type
            if node.is_root:
                facecolor = '#FFB74D'  # Orange for roots
                edgecolor = '#E65100'
            elif node.node_type == 'terminal':
                facecolor = '#A5D6A7'  # Green for terminals
                edgecolor = '#2E7D32'
            else:
                facecolor = '#90CAF9'  # Blue for operators
                edgecolor = '#1565C0'

            # Draw rounded rectangle
            box = FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=2.0,
                zorder=2
            )
            ax.add_patch(box)

            # Draw label
            ax.text(x, y, label,
                   ha='center', va='center',
                   fontsize=base_font_size,
                   fontfamily='monospace',
                   linespacing=1.1,
                   zorder=3)

        # Title with statistics
        title = (f"Merged Population Tree - Generation {self.gen_id}\n"
                 f"Trees: {stats['tree_count']} | Nodes: {stats['total_nodes']} | "
                 f"Shared: {stats['shared_nodes']} | Savings: {stats['savings_percent']:.1f}%")
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFB74D',
                   markersize=12, label='Root (output)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#90CAF9',
                   markersize=12, label='Operator'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#A5D6A7',
                   markersize=12, label='Terminal (input)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Add depth labels on the left (only if we have positions)
        if positions and all_x:
            x_min = min(all_x) - max(node_sizes[nid][0]/2 for nid in positions.keys())
            for depth in range(max_depth + 1):
                if layers[depth]:
                    # Calculate y position same way as nodes
                    y = 0
                    for d in range(depth):
                        y += layer_max_heights.get(d, 0.5) + min_y_gap
                    ax.text(x_min - 0.5, y, f'Depth {depth}',
                           ha='right', va='center', fontsize=8, color='#555555',
                           fontweight='bold')

        ax.set_aspect('equal')
        ax.axis('off')
        ax.autoscale()

        # Add margins
        ax.margins(0.05)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    def evoloop_monitoring_plots(self):
        """Creates all monitoring visualizations for the GP run.

        Generates:
        - Performance plot (fitness/parsimony over generations)
        - Pareto front plot
        - Pareto front tree visualization
        - Parsimony histogram (for first 20 generations)
        """
        # Use monitor's built-in plotting or convert to DataFrame for compatibility
        self.monitor.plot_performance(self.rootdir / 'monitoring.png')
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
        """Saves a pickle backup of the current GP run state.

        Saves: generation ID, population, Pareto front, and monitoring data.

        Args:
            opt_path_backup: Optional custom path. Defaults to rootdir/backup/backup.pkl.
        """

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
        # Save monitor as DataFrame for backwards compatibility
        monitor_df = self.monitor.to_dataframe()
        run_backup_data = {}, self.gen_id, self.pop_genepool, self.paretofront, monitor_df
        path_backup = path_make_dir(path_backup)
        pickle_dump(path_backup, run_backup_data)

    def backup_load(self, opt_path_backup=None):
        """Loads a pickle backup of a previous GP run.

        Restores: generation ID, population, Pareto front, and monitoring data.
        Also creates a timestamped copy of the backup.

        Args:
            opt_path_backup: Optional custom path. Defaults to rootdir/backup/backup.pkl.

        Raises:
            FileNotFoundError: If backup file doesn't exist.
            Exception: If backup file is corrupted (EOFError).
        """

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        if Path.is_file(path_backup):
            printpl('g', f'Loading data from backup-file {path_backup}')
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)
            except EOFError as e:
                raise Exception(f'EOFError: \n{e}')

            help_dict, self.gen_id, self.pop_genepool, self.paretofront, loaded_monitor_df = run_data
            # Recreate monitor from loaded DataFrame (for backwards compatibility)
            # The monitor will be repopulated if evolution continues
            self.monitor = GPMonitor()
            self.backup_save(opt_path_backup=self.rootdir / f'backup/backup-{self.gen_id}.pkl')
            printpl('g', f'Successfully loaded backup file. Generation: {self.gen_id}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}')

    def analyze_generation(self, pareto_updated: bool = False):
        """Analyzes and logs statistics for the current generation.

        Computes population metrics and stores them in monitor.
        Triggers scheduled IO operations (plots, backups) based on intervals.

        Args:
            pareto_updated: Whether the Pareto front was updated this generation.
        """
        gen_time = time.perf_counter() - self.time_genstart

        # Record generation metrics using GPMonitor
        self.monitor.record_generation(
            gen_id=self.gen_id,
            population=self.pop_genepool,
            gen_time=gen_time,
            pareto_updated=pareto_updated,
            lut_size=len(self.lut_symex_fitness)
        )

        # Get latest metrics for logging
        latest = self.monitor.latest
        printpl('gg',
                f"Created {latest.get('pop_size', 0)}/{self.gen_end} ({latest.get('pop_unique', 0)} unique) in generation {self.gen_id}. "
                f"Trees in LUT: {len(self.lut_symex_fitness)} Generation took {gen_time:4.2f}s")

        # Generate merged population tree visualization
        self._save_merged_population_tree()

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
        """Checks for early termination conditions.

        Currently checks: No new Pareto entries in 100 generations.

        Returns:
            True if evolution should stop early, False otherwise.
        """
        if self.monitor.gens_since_last_pareto > 100:
            printpl('i', 'Custom Condition made your program exit! (No new pareto entries in 100 generations)')
            return True
        else:
            return False

if __name__ == '__main__':
    """
    Tests have been moved to plagih/test/ directory.
    
    Run tests with:
        pytest plagih/test/ -v
    
    Or run a quick sanity check:
        python -c "from plagih.trees import *; print('Import successful')"
    """
    print("Tests have been moved to plagih/test/")
    print("Run: pytest plagih/test/ -v")
    print()
    print("Quick sanity check:")

    # Quick sanity check
    tree = Add(Symbol(sympy.Symbol('a')), Number(1.0))
    expr = tree.get_sympy_expr()
    print(f"  Tree: {tree}")
    print(f"  SymPy: {expr}")
    print(f"  Length: {len(tree)}")
    print()
    print("All basic imports and operations work correctly.")

