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
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Type, TypeGuard, Union

import pandas as pd
import sympy
from sympy.functions.elementary.piecewise import ExprCondPair

from plagih.config import cfg as _cfg
from plagih.tree_complexity.python_bytecode_complexity import *
from plagih.tree_complexity.tree_edit_distance import *
from plagih.util import *

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


# =============================================================================
# Helper functions for type checking (forward references resolved at runtime)
# =============================================================================


def is_terminal(node: "Node") -> TypeGuard["Terminal"]:
    """Checks if a node is a Terminal (leaf) node.

    Standalone function for use for type-hinting terminal-node functions
    correctly..
    """
    return node.is_term()


def is_number(node: "Node") -> TypeGuard["Number"]:
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

    For implementation details, see: docs/PITFALLS.md → P17
    """

    @classmethod
    def eval(cls, a):
        try:
            if not isinstance(a, sympy.Basic):
                return sympy.Integer(round(a.evalf()))
                # return sympy.Integer(round(a))
            elif a.is_symbol:
                return None
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
            return round(a)
        elif isinstance(a, np.ndarray):
            return np.vectorize(lambda v: round(float(v)))(a)
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

    # Commutativity marker — set to True on operators whose child order
    # does not affect the result (e.g. Add, Mul, Min, Max, And, Or, Xor).
    # Used by canonicalize_children() to sort children deterministically.
    is_commutative: bool = False

    # Visualization defaults (overridden by subclass hierarchy)
    _viz_color, _viz_border, _viz_text = "#ECEFF1", "#607D8B", "#263238"
    _viz_shape = "rounded"  # ellipse | rounded | diamond

    # Tree-structure
    childs: List[Union["Node", Any]] = field(default_factory=list)  # Terminal-Nodes speichern Werte statt Nodes
    is_fix: bool = False  # Whether the node is fixed in the structure.
    depth: Optional[int] = None
    # unused for now, preparation for backlinks
    root_node: Optional["Node"] = None
    parent_node: Optional["Node"] = None

    is_Atom = ...  # sympy-check

    def __init__(self, *args, **kwargs):
        self.childs = list(args)
        self.is_fix = kwargs.get("is_fix", False)
        self.depth = kwargs.get("depth")
        self.root_node = kwargs.get("root_node")
        self.parent_node = kwargs.get("parent_node")

    # ------------------------------------------------------------------
    # Pickle optimization: exclude circular back-references
    # parent_node and root_node create circular reference chains that
    # cause pickle to serialize the entire tree multiple times.
    # Excluding them reduces pickle size ~5-10x and speeds up
    # ProcessPoolExecutor IPC dramatically.
    # After unpickling, call repair_all() to restore back-references.
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("parent_node", None)
        state.pop("root_node", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.parent_node = None
        self.root_node = None

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

    def add_child(self, child: "Node") -> None:
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

    def set_root(self, n: "Node"):
        self.root_node = n

    def get_childs(self) -> List["Node"]:
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
            raise TypeError(f"childs must be set as list, not {type(child_list)}: {child_list}")

    def repair_all(self, parent: "Node" = None, root: "Node" = None, depth: int = 0):
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
        self.depth = depth

        # Only recurse on operator nodes (terminals have values, not Node children)
        if self.has_childs():
            for ii, cc in enumerate(self.get_childs()):
                cc.repair_all(parent=self, root=root or self, depth=depth + 1)

    def get_mutable_rootnodes(self, extend_lvls=2) -> Optional[list["Node"]]:
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

    def get_typus(self) -> Type["Node"]:
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

        def format_terminal(node_term: "Terminal") -> str:
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
                v = (
                    bool(val)
                    if isinstance(val, (bool, sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse))
                    else bool(sympy.sympify(val))
                )
                return f"Boolean({v!s})"
            # Numbers
            if isinstance(node_term, Number):
                try:
                    v = float(val)
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

        def walk(node: "Node") -> str:
            cls_name = type(node).__name__

            if is_terminal(node):
                out = format_terminal(node)
                # Fix-Flag für Terminals
                if node.is_fix:
                    if out.endswith(")"):
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
                ...  # no action needed for now, e.g., chained operators

        return

    def set_new_node(self, nd_new: "Node", repair: bool = False, clean_chain: bool = True) -> None:
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

    def replace_with(self, new_class: Type["Node"], new_args: list) -> None:
        """Replace the current node with a simpler equivalent."""
        new_node = new_class(*new_args)  # Create new instance
        new_node.parent_node = self.parent_node  # Preserve parent reference
        if self.parent_node:
            self.parent_node.childs = [new_node if child is self else child for child in self.parent_node.childs]
        self.set_new_node(new_node)

    def replace_with_node(self, new_node: "Node") -> None:
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
        try:
            _sym = type(self).symfun
            _cs = self.get_childs()

            node_summary = f"node={type(self).__name__}, child_types={[type(child).__name__ for child in _cs[:4]]}"

            def _contains_piecewise_like(node: "Node") -> bool:
                if isinstance(node, (Ifte, Piecewise)):
                    return True
                if node.has_childs():
                    return any(_contains_piecewise_like(child) for child in node.get_childs())
                return False

            if self.is_term():
                _r = _sym(*_cs)

            elif isinstance(self, Piecewise):
                # _sym = sympy.Piecewise
                _cs = [(cc.get_childs()[0], cc.get_childs()[1]) for cc in _cs]
                _cs = [
                    (cc[0].get_sympy_expr(simplimore=simplimore), cc[1].get_sympy_expr(simplimore=simplimore))
                    for cc in _cs
                ]
                _cs = [ExprCondPair(cc[0], cc[1]) for cc in _cs]
                _r = _sym(*_cs)
            elif self.is_operator():
                if isinstance(self, RelationalOperator) and any(_contains_piecewise_like(child) for child in _cs):
                    raise SympyError(
                        "Relational operator on Ifte/Piecewise subtree is not supported for SymPy conversion "
                        f"({node_summary}). This combination is known to trigger SymPy recursion/hangs."
                    )

                _cs = [cc.get_sympy_expr(simplimore=simplimore) for cc in _cs]

                try:
                    _r = _sym(*_cs)
                    # -> AttributeError: 'Xor' object has no attribute '_eval_as_set'
                    # -> TypeError: Invalid comparison of non-real asin(15)
                except TypeError as e:
                    raise SympyImaginaryNumber(e)

            else:
                raise NotImplementedError

            sympy_expression_check_raise(_r)
            return _r

        except RecursionError as e:
            raise SympyError(
                f"RecursionError while building sympy expression for {type(self).__name__} ({node_summary}): {e}"
            ) from e

    def list_terminal_nodes(self) -> List["Node"]:
        """Returns a list of all terminal (leaf) nodes in this subtree."""
        base = self.list_mutable_nodes()
        base = [x for x in base if x.is_term()]
        return base

    def force_input_node(self, ev: "Evolution") -> None:
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
                raise TreeError(f"No terminal nodes found to replace in tree: {self}")

            try:
                node = rnd_choice(node_list)  # debug if ignores chains
                xtype = xt_self(node.get_xtype_tuple())
                new_node = ev.node_selector.choose_symbol_node(xtype)
            except (ValueError, IndexError) as e:
                raise TreeError(f"Single-node tree,  no matching input found (probably boolean - ex: {e}): {self}")

            # node.set_new_node(new_node)
            node.set_new_node(new_node)

    def is_number(self) -> bool:
        """Checks if this node is a numeric constant (Number terminal)."""
        return issubclass(type(self), Number)

    def str_as_list(self, cut_terms: bool = False, order: str = "pre") -> str:
        """Returns a nested list representation of the tree.

        Args:
            cut_terms: If True, truncates numbers for readability.
            order: Traversal order for the output.
                ``"pre"``  (default) – operator first, then children:
                ``Add(a, 1) → [Add, [a], [1]]``
                ``"post"`` – children first, then operator:
                ``Add(a, 1) → [[a], [1], Add]``

        Returns:
            Bracket-formatted string representation.
        """
        typus_str = self.showme

        if self.get_childs():
            if issubclass(type(self), BaseOperator):
                childstr = ", ".join([cc.str_as_list(cut_terms=cut_terms, order=order) for cc in self.get_childs()])
                if order == "post":
                    typus_str = f"{childstr}, {typus_str}"
                else:
                    typus_str = f"{typus_str}, {childstr}"
            else:
                # terminal nodes
                v = self.get_childs()[0]
                if self.is_number():
                    typus_str = term_format(f"{v}", cut=cut_terms)
                else:
                    typus_str = f"{v}"
        else:
            raise CuriosityError("Holla sfeh")

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
            _expr = f"{self.get_childs()[0]}"
            _expr = term_format(_expr, cut=cut_terms)
            return f"{_expr}"
        else:
            childs = [cc.get_expr_symlike(try_sympify=try_sympify, cut_terms=cut_terms) for cc in self.get_childs()]
            # if issubclass(type(self), (ExprCondPair)):

            if isinstance(self, (Add, Mul, And, Or, Xor)):
                _expr = self.inline_sep.join(childs)
                _expr = f"({_expr})"

            else:
                _expr = self.sy_str.format(*childs)

        # Not working, requires handlich exprCondPair
        # if try_sympify:
        #     try:
        #         expr_sy = sympy.sympify(_expr)  # local dict required?
        #         return f'{expr_sy}'
        #     except Exception as ex:
        #         return f'{_expr}'

        return f"{_expr}"

    def list_mutable_nodes(self, xtype=None) -> List["Node"]:
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
        if self.is_fix or self.is_ExprCdPair():
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
        showme = f"{self.childs[0]}" if self.is_term() else f"{self.showme}"

        res = {setid: {"node": self, "showme": showme}}
        edges = []

        if self.is_term():
            pass
        else:
            for ii, cc in enumerate(self.childs):
                cid = f"{setid}-{ii}"
                cr, ce = cc.get_all_nodes_visualize(cid)
                res.update(cr)
                edges.append((setid, cid))
                edges.extend(ce)

        return res, edges

    def get_apted_notation(self) -> str:
        """Returns the tree in APTED bracket notation for tree edit distance.

        .. deprecated::
            Use :meth:`compute_ted` instead, which operates directly on Node
            objects without requiring string conversion.

        APTED (All Path Tree Edit Distance) requires this specific format.
        Example: Add(a, 1) -> {Add{Symbol}{Number}}
        """
        warnings.warn(
            "get_apted_notation() is deprecated.  Use Node.compute_ted() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return f"{{{self.get_typus()}{''.join([cc.get_apted_notation() for cc in self.get_childs()])}}}"

    # ------------------------------------------------------------------
    # Tree Edit Distance (Zhang-Shasha, directly on Node objects)
    # ------------------------------------------------------------------

    def compute_ted(self, other: "Node", config: Optional[TedConfig] = None) -> TedResult:
        """Compute tree edit distance between this tree and *other*.

        The Zhang-Shasha algorithm (in ``tree_complexity/tree_edit_distance.py``)
        traverses the Node tree directly via :meth:`get_childs` /
        :meth:`has_childs` and compares labels via :func:`type` /
        :meth:`get_value`.  No string conversion or external library required.

        Args:
            other: The target tree to compare against.
            config: Distance configuration (mode, costs).
                Defaults to ``TedConfig(mode="full")``.

        Returns:
            :class:`TedResult` with ``distance``, ``mapping`` (referencing
            the actual Node objects from both trees), and optional
            ``leaf_diff_count``.

        Complexity:
            Time  O(n² · m²),  Space O(n · m)
        """
        from plagih.tree_complexity.tree_edit_distance import zhang_shasha_ted

        return zhang_shasha_ted(self, other, config)

    # ------------------------------------------------------------------
    # Tree traversal orders
    # ------------------------------------------------------------------

    def to_traversal(self, order: str = "pre") -> List["Node"]:
        """Returns a flat list of all nodes in the given traversal order.

        Args:
            order: ``"pre"`` (default) – root first, then children (preorder).
                   ``"post"`` – children first, then root (postorder).

        Examples::

            Add(Mul(x, y), z)
              preorder  → [Add, Mul, x, y, z]
              postorder → [x, y, Mul, z, Add]
        """
        pre = order == "pre"
        result: List[Node] = []

        def _visit(node: "Node") -> None:
            if pre:
                result.append(node)
            if node.has_childs():
                for child in node.get_childs():
                    _visit(child)
            if not pre:
                result.append(node)

        _visit(self)
        return result

    def to_postorder(self) -> List["Node"]:
        """Shorthand for ``to_traversal("post")``."""
        return self.to_traversal("post")

    def to_preorder(self) -> List["Node"]:
        """Shorthand for ``to_traversal("pre")``."""
        return self.to_traversal("pre")

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
                val = round(random.gauss(self.get_value(), 0.1), _cfg.float_precision)

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
                    if (const - val) < tolerance:
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
                log_error(f"ERROR: Usub should only have one child, but got {len(mychlds)}")

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
                            if (1 / mul1) % 1 == 0:  # check if the result is a natural number
                                node_sub = Mul(*mychlds_remove(cc))
                                new_num = 1 / mul1
                                self.replace_with(Div, [node_sub, Number(new_num)])
                        else:
                            # Scale grouping: Mul(c, expr) → Scale(c, expr)
                            # where c is a Number and expr is a non-Number expression.
                            rest = mychlds_remove(cc)
                            has_other_numbers = any(isinstance(r, Number) for r in rest)
                            if not has_other_numbers and len(rest) >= 1:
                                if len(rest) == 1:
                                    self.replace_with(Scale, [cc, rest[0]])
                                else:
                                    # Mul(c, a, b) → Scale(c, Mul(a, b))
                                    self.replace_with(Scale, [cc, Mul(*rest)])
                    elif isinstance(cc, DivFraction):
                        has_div_frac = [isinstance(ix, DivFraction) for ix in mychlds]
                        if (
                            sum(has_div_frac) > 1 or len(mychlds) > 2
                        ):  # this makes chained mul to 1/(mul( )) -> check if more than 2 inputs
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
            # No specific grouping rule for this operator type (e.g. Scale, Add,
            # Div, trigonometry, relational operators). Children were already
            # recursively grouped above.  (See M4 in IMPLEMENTATION_PLAN.md.)
            pass
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

    def canonicalize_children(self) -> None:
        """Sorts children of commutative operators for a canonical representation.

        Recursively traverses the tree and sorts children of commutative operators
        (Add, Mul, Min, Max, And, Or, Xor) by their string representation.
        This is a **post-processing** step — call it after tree construction
        is complete (e.g. in tree_node_grouping or tree_simplification).

        This does NOT alter semantics (commutativity guarantees identical results)
        but normalises trees so that structurally equivalent expressions have
        identical string representations and LUT keys.

        # TODO(discuss): Open design concerns for canonicalize_children
        #
        # 1. **Performance**: This is a recursive function that traverses every
        #    node and calls represent_str() as sort key. On large trees this may
        #    become a measurable overhead, especially since it is called in
        #    tree_to_candidate() (hot path). Benchmark before optimising.
        #
        # 2. **Sort key quality**: The current sort key is represent_str() which
        #    is a simple string comparison. SymPy uses a much more sophisticated
        #    ordering (sympy.core.sorting.default_sort_key) that considers type
        #    priority, complexity, and algebraic properties. A SymPy-equivalent
        #    sort would produce better canonical forms but would be significantly
        #    more expensive to evaluate. An alternative middle-ground: sort by
        #    subtree size (len(child)) — cheap, stable, and groups simple branches
        #    before complex ones.
        #
        # 3. **Invalidation after mutation**: When a mutation changes a child
        #    node deep in the tree, the canonical order of ALL ancestor
        #    commutative nodes may become stale. This means canonicalize_children()
        #    must be re-run after any mutation — or the mutation itself must
        #    propagate re-sorting upwards. Currently handled by calling
        #    canonicalize_children() in tree_to_candidate(), but if it were
        #    moved earlier in the pipeline, mutation invalidation would become
        #    a pitfall. See docs/PITFALLS.md P10.
        #
        # These trade-offs should be revisited once there is benchmark data on
        # the actual overhead and LUT hit-rate improvement.
        """
        if self.is_term():
            return

        # Recurse first so children are canonical before sorting
        for cc in self.get_childs():
            cc.canonicalize_children()

        if getattr(self, "is_commutative", False):
            # Sort by string representation — available after tree is fully built.
            # Terminals sort lexicographically (symbols by name, numbers by value).
            # Operators sort by their showme + children recursively.
            # TODO(discuss): Consider alternative sort keys — see docstring above.
            self.childs.sort(key=lambda c: c.represent_str(show_fixed_hint=False, cut_terms=False))

    def len_nodecount_raw(self) -> int:
        """Returns the raw count of all nodes in this subtree.

        Simple recursive counting without any special handling.
        """
        if self.has_childs():
            return 1 + sum([cc.len_nodecount_raw() for cc in self.get_childs()])
        else:
            return 1  # childs can currently be floats

    def is_typus(self, nt: Type["Node"]) -> bool:
        """Checks if this node is an instance of the given Node type.

        Args:
            nt: The Node class to check against.
        """
        r = issubclass(type(self), nt)
        return r

    def is_ExprCdPair(self) -> bool:
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
            cs = f"{self.get_childs()[0]}"
            if cut_terms:
                cs = term_format(cs, cut=cut_terms)
            else:
                cs = remove_trailing_zeroes(cs)

            s = f"{cs}"

            if self.is_fix and show_fixed_hint:
                s += ":fix"  # discuss: there must be a more natural way to show that...

        else:
            cs = [cc.represent_str(show_fixed_hint=show_fixed_hint, cut_terms=cut_terms) for cc in self.get_childs()]
            cs = ", ".join(cs)
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


# Backward-compatible free function (delegates to Node.compute_ted)
def compute_ted(tree1: Node, tree2: Node, config: Optional[TedConfig] = None) -> TedResult:
    """Convenience wrapper — equivalent to ``tree1.compute_ted(tree2, config)``."""
    return tree1.compute_ted(tree2, config)


def pairwise_ted_matrix(
    trees: List[Node],
    config: Optional[TedConfig] = None,
) -> np.ndarray:
    """Compute a symmetric pairwise TED matrix for a list of trees.

    Useful as foundation for diversity measurement and clustering.

    Returns:
        Symmetric ``(k, k)`` NumPy array where ``[i, j]`` is the tree
        edit distance between ``trees[i]`` and ``trees[j]``.
    """
    if config is None:
        config = TedConfig()
    k = len(trees)
    matrix = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            result = trees[i].compute_ted(trees[j], config)
            matrix[i, j] = result.distance
            matrix[j, i] = result.distance
    return matrix


def eval_parsimony(_tree: Node, complexity_measure: str, origin_tree: Optional[Node] = None) -> int:
    """Evaluates the complexity/parsimony of a tree.

    Args:
        _tree: The tree to measure.
        complexity_measure: One of:
            - 'tree_node_count_raw': Simple node count
            - 'tree_node_count_fair': Adjusted count (ignores Usub, etc.)
            - 'tree_edit_distance': Distance from origin_tree
            - 'tree_python_bytecode_count': CPython bytecode instruction count
            - 'tree_python_bytecode_weighted_count': Weighted CPython bytecode instruction count
            - 'tree_cpu_cost_proxy': Heuristic CPU-cost proxy
            - 'tree_flops_proxy': Heuristic FLOPs proxy
        origin_tree: Reference tree for edit distance calculation.

    Returns:
        Integer complexity score (lower is simpler).
    """
    if complexity_measure == "tree_node_count_raw":  # number of nodes
        return _tree.len_nodecount_raw()
    elif complexity_measure == "tree_node_count_fair":
        return _tree.len_nodecount_fair()
    elif complexity_measure == "tree_edit_distance":  # tree_edit_distance, fintree-edit-distance
        return int(_tree.compute_ted(origin_tree, TedConfig(mode="structural")).distance)
    elif complexity_measure == "tree_python_bytecode_count":
        return count_python_bytecode_instructions(
            _tree,
            BytecodeComplexityConfig(method="instruction_count"),
        )
    elif complexity_measure == "tree_python_bytecode_weighted_count":
        return count_python_bytecode_instructions(
            _tree,
            BytecodeComplexityConfig(method="weighted_instruction_count"),
        )
    elif complexity_measure == "tree_cpu_cost_proxy":
        return count_python_bytecode_instructions(
            _tree,
            BytecodeComplexityConfig(method="cpu_cost_proxy"),
        )
    elif complexity_measure == "tree_flops_proxy":
        return count_python_bytecode_instructions(
            _tree,
            BytecodeComplexityConfig(method="flops_proxy"),
        )
    else:
        raise Exception(f"Complexity measurement not available: {complexity_measure}")


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
                raise NotImplementedError(f"What happened here? {s_expr}")

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
            pairs = [
                (sympy_to_tree(cond, allow_chain=allow_chain), sympy_to_tree(ceex, allow_chain=allow_chain))
                for ceex, cond in s_expr.args
            ]

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
    raise NotImplementedError(f"Expr missing: {s_expr}")


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

    # Canonical child ordering for commutative operators (post-processing).
    # This normalises e.g. Add(b, a) → Add(a, b) so that structurally equivalent
    # expressions produce identical string representations and LUT keys.
    _tree.canonicalize_children()

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
        # TODO(sfeh): Investigate why simplification can *grow* a tree.
        #  Possible causes: sympy expanding terms, or tree-rebuild adding wrapper nodes.
        #  Keep this print until root cause is understood.
        astr = string_remove_trailing_zeroes(str(tree_history[0].get_sympy_expr()))
        bstr = string_remove_trailing_zeroes(str(_tree.get_sympy_expr()))
        print(f"WHATHAPPENED SFEH\t{astr}")
        for ii, tt in enumerate(tree_history):
            print(f"\t{ii}:\t{tt.str_as_list()}")

        if astr != bstr:
            # sfeh 'a**0.5' does not become 'sqrt(a)'! use rational=True or sympy.S.Half
            log("w", f"Diff in sympy expression?\n\t{astr}\n\t{bstr}")  # raise ex? does not occur after grouping?
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
        node_list = [
            n for n in _tree.list_mutable_nodes() if issubclass(n.get_typus(), BaseOperator)
        ]  # ignoring leaf nodes...
        if len(node_list) == 0:
            log("wwww", f"Tree for simplification does not provide operators: {_tree}")
            return _tree
        node = random.choice(node_list)
        node2 = tree_simplification(node, allow_chain)
        node.set_new_node(node2)  # sfeh chosen must be set again? or not? test it at least.
    if force:
        return _tree
    else:
        if len(tree_copy) < len(_tree):
            log("w", f"Tree grew larger during simplification:\n\t{tree_copy.str_as_list()}\n\t{_tree.str_as_list()}")
            log("ww", f"tree_copy: {len(tree_copy)} vs. {len(_tree)}")
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
        self.chain = kwargs.get("chain")

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
            if "loop of ufunc does not support argument" in str(sfeh):
                # TypeError: loop of ufunc does not support argument 0 of type int which has no callable sin method
                print("ASDASDASDASD TODO")  # Debug breakpoint
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

    _viz_color, _viz_border, _viz_text = "#FCE4EC", "#E91E63", "#880E4F"


class LogicOperator(BaseOperator):
    """Base class for logical operators that produce boolean results.

    Includes And, Or, Xor, Not operations.
    """

    _viz_color, _viz_border, _viz_text = "#F3E5F5", "#9C27B0", "#4A148C"
    _viz_shape = "diamond"


class RelationalOperator(BaseOperator):
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
    showme = "Boolean"
    _viz_color, _viz_border, _viz_text = "#FFF3E0", "#FF9800", "#E65100"
    _viz_shape = "ellipse"

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        val = bool(self.get_value())
        return np.full(len(_df), val, dtype=np.bool_)

    def eval_np_lambdas(self, *args):
        val = bool(self.get_value())
        return lambda _df: np.full(len(_df), val, dtype=np.bool_)


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
    symfun = staticmethod(lambda *a: sympy.Float(float(a[0]), _cfg.float_precision))
    np_fun = staticmethod(lambda _x: float(_x))
    # sympy.Rational(0.1) -> 3602879701896397/36028797018963968
    # sympy.Rational('0.1') -> 1/10
    showme = "Number"
    _viz_color, _viz_border, _viz_text = "#E8F5E9", "#4CAF50", "#1B5E20"
    _viz_shape = "ellipse"
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
    showme = "Symbol"
    _viz_color, _viz_border, _viz_text = "#E3F2FD", "#2196F3", "#0D47A1"
    _viz_shape = "ellipse"

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

    is_commutative = True

    symfun = staticmethod(lambda *a: sympy.Add(*a))
    np_fun = staticmethod(lambda *a: np.sum(np.stack(a), axis=0))
    showme = "Add"
    sy_str = "({0} + {1})"
    formulae_str = "({} + {})"
    repr_str = "Add{},[{},{}]"
    latex_inline = " + "
    xtype = ((float, float), float)
    symfun_chain = lambda a: sympy.Add(*a)
    np_fun_chain = lambda *a: np.sum(*a)  # n
    sy_str_chain = "Add({})"
    formulae_str_chain = "Add({})"
    repr_str_chain = "AddChain{},[{}]"
    inline_sep = " + "
    xtype_chain = ([(float,)], float)
    xtype_input = float


class Mul(MathOperator, ChainableOp):
    """Multiplication operator for two or more operands.

    Computes the product of all child values. Supports chaining for
    more than two operands: Mul(a, b, c) = a * b * c.

    Note: np.multiply only works for pairwise multiplication,
    so np.prod with stacking is used instead.
    """

    is_commutative = True

    symfun = staticmethod(lambda *args: sympy.Mul(*args))
    np_fun = staticmethod(lambda *a: np.prod(np.stack(a), axis=0))
    showme = "Mul"  #
    sy_str = "({0} * {1})"
    repr_str = "Mul{},[{}, {}]"
    latex_inline = r" \cdot "
    xtype = ((float, float), float)
    symfun_chain = lambda a: sympy.Mul(*a)
    sy_str_chain = "Mul({})"
    repr_str_chain = "MulChain{},[{}]"
    inline_sep = " * "
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
    showme = "DivFraction"
    sy_str = "1/({})"
    repr_str = "DivFraction{},[{}]"


class NthRoot(MathOperator):
    """N-th root operator: NthRoot(x, n) computes x^(1/n).

    Currently untested. Use with caution.
    """

    xtype = ((float, float), float)
    symfun = staticmethod(lambda *a: sympy.root(a[0], a[1]))
    np_fun = staticmethod(lambda base, n: np.power(base, 1 / n))
    showme = "NthRoot"
    sy_str = "root({}, {})"
    repr_str = "NthRoot{},[{}, {}]"


class Pow(MathOperator):
    """Power operator: Pow(base, exp) computes base^exp."""

    symfun = staticmethod(lambda *a: sympy.Pow(a[0], a[1]) if len(a) == 2 else None)
    np_fun = staticmethod(lambda base, exp: np.power(base, exp))
    showme = "Pow"
    sy_str = "({0})**({1})"
    repr_str = "Pow{},[{},{}]"
    latex_fmt = r"{{{}}}^{{{}}}"
    xtype = ((float, float), float)


class Abs(MathOperator):
    """Absolute value operator: Abs(x) returns |x|."""

    symfun = staticmethod(lambda *a: sympy.Abs(a[0]))
    np_fun = np.absolute  # np.fabs works only for non-complex numbers
    showme = "Abs"
    sy_str = "Abs({})"
    repr_str = "Abs{},[{}]"
    latex_fmt = r"\left|{} \right|"
    xtype = ((float,), float)


class Sign(MathOperator, NoSymCapitalized):
    """Sign function: returns -1, 0, or 1 depending on the sign of the input."""

    symfun = staticmethod(lambda *a: sympy.sign(a[0]))
    np_fun = np.sign
    showme = "Sign"
    sy_str = "sign({})"
    repr_str = "Sign{},[{}]"
    xtype = ((float,), float)


class Log(MathOperator, NoSymCapitalized):
    """Natural logarithm (base e): Log(x) computes ln(x).

    Note: Input should be positive to avoid undefined results.
    """

    symfun = staticmethod(lambda *a: sympy.log(a[0]))
    np_fun = np.log
    showme = "Log"
    sy_str = "log({})"
    repr_str = "Log{},[{}]"
    xtype = ((float,), float)


class Cos(Trigonometry, NoSymCapitalized):
    """Cosine function: Cos(x) computes cos(x) in radians."""

    symfun = staticmethod(lambda *a: sympy.cos(a[0]))
    np_fun = np.cos
    showme = "Cos"
    sy_str = "cos({})"
    repr_str = "Cos{},[{}]"
    xtype = ((float,), float)


class Sin(Trigonometry, NoSymCapitalized):
    """Sine function: Sin(x) computes sin(x) in radians."""

    symfun = staticmethod(lambda *a: sympy.sin(a[0]))
    np_fun = np.sin
    showme = "Sin"
    sy_str = "sin({})"
    repr_str = "Sin{},[{}]"
    xtype = ((float,), float)


class Tan(Trigonometry, NoSymCapitalized):
    """Tangent function: Tan(x) computes tan(x) in radians."""

    symfun = staticmethod(lambda *a: sympy.tan(a[0]))
    np_fun = np.tan
    showme = "Tan"
    sy_str = "tan({})"
    repr_str = "Tan{},[{}]"
    xtype = ((float,), float)


class Acos(Trigonometry, NoSymCapitalized):
    """Inverse cosine (arccosine): Acos(x) returns arccos(x) in radians."""

    symfun = staticmethod(lambda *a: sympy.acos(a[0]))
    np_fun = np.arccos  # arccosh
    showme = "Acos"
    sy_str = "acos({})"
    repr_str = "Acos{},[{}]"
    xtype = ((float,), float)


class Asin(Trigonometry, NoSymCapitalized):
    """Inverse sine (arcsine): Asin(x) returns arcsin(x) in radians."""

    symfun = staticmethod(lambda *a: sympy.asin(a[0]))
    np_fun = np.arcsin
    showme = "Asin"
    sy_str = "asin({})"
    repr_str = "Asin{},[{}]"
    xtype = ((float,), float)


class Atan(Trigonometry, NoSymCapitalized):
    """Inverse tangent (arctangent): Atan(x) returns arctan(x) in radians."""

    symfun = staticmethod(lambda *a: sympy.atan(a[0]))
    np_fun = np.arctan
    showme = "Atan"
    sy_str = "atan({})"
    repr_str = "Atan{},[{}]"
    xtype = ((float,), float)


class Tanh(Trigonometry, NoSymCapitalized):
    """Hyperbolic tangent: Tanh(x) returns tanh(x)."""

    symfun = staticmethod(lambda *a: sympy.tanh(a[0]))
    np_fun = np.tanh
    showme = "Tanh"
    sy_str = "tanh({})"
    repr_str = "Tanh{},[{}]"
    xtype = ((float,), float)


class Sinh(Trigonometry, NoSymCapitalized):
    """Hyperbolic sine: Sinh(x) returns sinh(x)."""

    symfun = staticmethod(lambda *a: sympy.sinh(a[0]))
    np_fun = np.sinh
    showme = "Sinh"
    sy_str = "sinh({})"
    repr_str = "Sinh{},[{}]"
    xtype = ((float,), float)


class Cosh(Trigonometry, NoSymCapitalized):
    """Hyperbolic cosine: Cosh(x) returns cosh(x)."""

    symfun = staticmethod(lambda *a: sympy.cosh(a[0]))
    np_fun = np.cosh
    showme = "Cosh"
    sy_str = "cosh({})"
    repr_str = "Cosh{},[{}, {}]"
    xtype = ((float,), float)


class Not(LogicOperator):
    """Logical NOT operator: Not(a) returns ~a (negation)."""

    symfun = staticmethod(lambda *a: sympy.Not(a[0]))
    np_fun = np.logical_not
    showme = "Not"
    sy_str = "~({})"
    repr_str = "Not{},[{}]"
    xtype = ((bool,), bool)


class Eq(RelationalOperator):
    """Equality comparison: Eq(a, b) returns True if a == b."""

    symfun = staticmethod(lambda *a: sympy.Eq(a[0], a[1]))
    np_fun = np.equal
    showme = "Eq"  # '==' not working in sympy!
    sy_str = "Eq({0}, {1})"
    repr_str = "Eq{},[{}, {}]"
    xtype = ((float, float), bool)


class Ne(RelationalOperator):
    """Inequality comparison: Ne(a, b) returns True if a != b."""

    symfun = staticmethod(lambda *a: sympy.Ne(a[0], a[1]))
    np_fun = np.not_equal
    showme = "Ne"  # != not working in sympy
    sy_str = "Ne({0}, {1})"
    repr_str = "Ne{},[{}, {}]"
    xtype = ((float, float), bool)


class And(LogicOperator, ChainableOp):
    """Logical AND operator for two or more boolean inputs.

    Supports chaining: And(a, b, c) = a & b & c.
    """

    is_commutative = True

    symfun = staticmethod(lambda *a: sympy.And(*a))
    np_fun = staticmethod(lambda *a: np.logical_and.reduce(a))
    showme = "And"
    sy_str = "({0} & {1})"  # Arity-2 Formatierung
    repr_str = "And{},[{}, {}]"
    latex_inline = r" \wedge "
    xtype = ((bool, bool), bool)
    xtype_input = bool
    expr_dmy = "And"
    symfun_chain = staticmethod(lambda a: sympy.And(*a))
    np_fun_chain = staticmethod(lambda *a: np.logical_and.reduce(a))

    showme_chain = "AndChain"
    sy_str_chain = "And({})"  # Variadische Notation
    repr_str_chain = "And{},[{}]"
    inline_sep = " & "
    xtype_chain = ([(bool,)], bool)


class Or(LogicOperator, ChainableOp):
    """Logical OR operator for two or more boolean inputs.

    Supports chaining: Or(a, b, c) = a | b | c.
    """

    is_commutative = True

    symfun = staticmethod(lambda *a: sympy.Or(a[0], a[1]))
    np_fun = staticmethod(lambda *a: np.any(a, axis=0))
    showme = "Or"
    sy_str = "({0}|{1})"
    repr_str = "Or{},[{}, {}]"
    latex_inline = r" \vee "
    xtype = ((bool, bool), bool)
    xtype_chain = ([(bool,)], bool)
    xtype_input = bool
    symfun_chain = lambda a: sympy.Or(*a)
    np_fun_chain = lambda *a: np.logical_or(*a)
    sy_str_chain = "Or({})"
    repr_str_chain = "OrChain{},[{}]"
    inline_sep = " | "


class Xor(LogicOperator, NoSymCapitalized, ChainableOp):
    """Logical XOR (exclusive or) operator.

    Caution: The sympy representation '(a ^ b)' is interpreted as a**b in Python.
    Supports chaining for multiple operands.
    """

    is_commutative = True

    symfun = staticmethod(lambda *a: sympy.Xor(*a))
    np_fun = staticmethod(lambda *a: np.logical_xor.reduce(a))
    showme = "Xor"
    sy_str = "Xor({}, {})"  # 'a ^ b'
    repr_str = "Xor{},[{}, {}]"
    latex_inline = r" \oplus "
    xtype = ((bool, bool), bool)
    symfun_chain = lambda a: sympy.Xor(*a)
    np_fun_chain = lambda *a: np.logical_xor(*a)
    sy_str_chain = "Xor({})"  # 'a ^ b'
    inline_sep = " ^ "

    xtype_chain = ([(bool,)], bool)
    xtype_input = bool


class ITE(LogicOperator):
    """Logical if-then-else for boolean results only.

    ITE(condition, then_value, else_value) returns then_value if condition
    is True, otherwise else_value. All arguments must be boolean.

    For numeric results, use Ifte instead.
    """

    symfun = staticmethod(lambda *a: sympy.ITE(a[0], a[1], a[2]))
    np_fun = staticmethod(lambda a, b, c: (a & b) | (not a) & c)
    showme = "ITE"
    sy_str = "ITE({0}, {1}, {2})"
    repr_str = "ITE{},[{}, {}, {}]"
    xtype = ((bool, bool, bool), bool)


class Min(BaseMinMax, ChainableOp):
    """Minimum operator: returns the smallest value among inputs.

    Supports chaining: Min(a, b, c) returns min(a, b, c).
    """

    is_commutative = True

    symfun = staticmethod(lambda *a: sympy.Min(*a))
    np_fun = staticmethod(lambda *a: np.minimum.reduce(np.vstack(a), axis=0))
    showme = "Min"
    sy_str = "Min({0},{1})"
    repr_str = "Min{},[{}, {}]"
    latex_fmt = r"\min\left({}\right)"
    xtype = ((float, float), float)
    xtype_input = float
    symfun_chain = lambda a: sympy.Min(*a)
    np_fun_chain = np.minimum
    sy_str_chain = "Min({})"
    repr_str_chain = "MinChain{},[{}]"
    xtype_chain = ([(float,)], float)


class Max(BaseMinMax, ChainableOp):
    """Maximum operator: returns the largest value among inputs.

    Supports chaining: Max(a, b, c) returns max(a, b, c).
    """

    is_commutative = True

    symfun = staticmethod(lambda *a: sympy.Max(*a))
    np_fun = staticmethod(lambda *a: np.maximum.reduce(np.vstack(a), axis=0))
    showme = "Max"
    sy_str = "Max({0}, {1})"
    repr_str = "Max{},[{}, {}]"
    latex_fmt = r"\max\left({}\right)"
    xtype = ((float, float), float)
    xtype_input = float
    symfun_chain = lambda a: sympy.Max(*a)
    np_fun_chain = np.maximum
    sy_str_chain = "Max({})"
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
    showme = "Lt"
    sy_str = "({0} < {1})"
    repr_str = "Lt{},[{}, {}]"
    xtype = ((float, float), bool)


class Le(RelationalOperator):
    """Less-than-or-equal comparison: Le(a, b) returns True if a <= b."""

    symfun = staticmethod(lambda *a: sympy.Le(a[0], a[1]))
    np_fun = np.less_equal
    showme = "Le"
    sy_str = "({0} <= {1})"
    repr_str = "Le{},[{}, {}]"
    xtype = ((float, float), bool)


class Gt(RelationalOperator, PleaseUsePartnerOp):
    """Greater-than comparison: Gt(a, b) returns True if a > b.

    Note: Consider using Lt(b, a) instead to reduce operator redundancy.
    """

    symfun = staticmethod(lambda *a: sympy.Gt(a[0], a[1]))
    np_fun = np.greater
    showme = "Gt"
    sy_str = "({0} > {1})"
    repr_str = "Gt{},[{}, {}]"
    xtype = ((float, float), bool)


class Ge(RelationalOperator, PleaseUsePartnerOp):
    """Greater-than-or-equal comparison: Ge(a, b) returns True if a >= b.

    Note: Consider using Le(b, a) instead to reduce operator redundancy.
    """

    xtype = ((float, float), bool)
    symfun = staticmethod(lambda *a: sympy.Ge(a[0], a[1]))
    np_fun = np.greater_equal
    showme = "Ge"
    sy_str = "({0} >= {1})"
    repr_str = "Ge{},[{}, {}]"


class Square(MathOperator):
    """Square operator: Square(x) computes x^2."""

    symfun = staticmethod(lambda *a: sympy.Pow(a[0], 2))
    np_fun = np.square
    xtype = ((float,), float)
    showme = "Square"
    sy_str = "({})**2"
    repr_str = "Square{},[{}]"


class Exp(MathOperator):
    """Exponential function: Exp(x) computes e^x."""

    symfun = staticmethod(lambda *a: sympy.exp(a[0]))
    np_fun = np.exp
    showme = "Exp"
    sy_str = "{}**E"
    repr_str = "Exp{},[{}, {}]"
    xtype = ((float,), float)


class Exp2(MathOperator):
    """Base-2 exponential: Exp2(x) computes 2^x."""

    symfun = staticmethod(lambda *a: sympy.Pow(2, a[0]))
    np_fun = np.exp2
    xtype = ((float,), float)
    showme = "Exp2"
    sy_str = "2**({})"
    repr_str = "Exp2{},[{}]"


class Sub(MathOperator):
    """Subtraction operator: Sub(a, b) computes a - b."""

    xtype = ((float, float), float)
    symfun = staticmethod(lambda *a: sympy.Add(a[0], -a[1]))
    np_fun = np.subtract
    showme = "Sub"
    sy_str = "({0} - {1})"
    repr_str = "Sub{},[{}, {}]"
    latex_inline = " - "


class Ifte(BaseOperator):
    """Numeric if-then-else operator (Piecewise).

    Ifte(condition, then_value, else_value) returns then_value if condition
    is True, otherwise else_value. Returns numeric (float) results.

    For boolean results, use ITE instead.
    """

    xtype = ((bool, float, float), float)
    symfun = staticmethod(lambda *a: sympy.Piecewise((a[1], a[0]), (a[2], True)))
    np_fun = staticmethod(lambda cond, if_true, if_false: np.where(cond, if_true, if_false))
    showme = "Ifte"
    sy_str = "Ifte({0},{1},{2})"
    repr_str = "Ifte{},[{}, {}, {}]"
    expr_dummy = "Ifte"
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
    showme = "Piecewise"
    sy_str = "Piecewise({})"
    formulae_str = "Piecewise({})"
    repr_str = "Piecewise{},[{}]"
    xtype = ([(ExprCondPair,)], float)
    xtype_chain = ExprCondPair
    xtype_input = ExprCondPair

    def eval_predict_numpy_now(self, _df: pd.DataFrame, *args) -> np.ndarray:
        # ...existing code...
        pairs = [
            (c.childs[0].eval_predict_numpy_now(_df, *args), c.childs[1].eval_predict_numpy_now(_df, *args))
            for c in self.get_childs()
        ]

        # Default values are in last position
        result = pairs[-1][0].astype(np.float64, copy=False)

        # from default to first condition
        for _e, _c in reversed(pairs[:-1]):
            result = np.where(_c, _e, result)
        return result

    def eval_np_lambdas(self, *args):
        pairs = [(c.childs[0].eval_np_lambdas(*args), c.childs[1].eval_np_lambdas(*args)) for c in self.get_childs()]

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
    np_fun = staticmethod(lambda x: np.vectorize(lambda v: round(float(v)))(x))
    showme = "Round"
    sy_str = "RoundDummy({},1)"
    repr_str = "RoundDummy{},[{}]"


class PowRounded(MathOperator):
    """Power operator with rounded exponent: PowRounded(base, exp) computes base^round(exp).

    Useful to constrain exponents to integer values during evolution,
    avoiding fractional powers that can produce complex numbers.
    """

    symfun = staticmethod(lambda *a: sympy.Pow(a[0], RoundDummy(a[1])))
    np_fun = staticmethod(lambda base, exponent: np.power(base, np.vectorize(lambda _x: round(float(_x)))(exponent)))
    showme = "PowRounded"
    sy_str = "({0})**RoundDummy({1})"
    repr_str = "PowRounded{},[{}, {}]"
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
    showme = "Div"
    sy_str = "({0}/{1})"
    repr_str = "Div{},[{}, {}]"
    latex_inline = " / "
    xtype = ((float, float), float)


class Sqrt(MathOperator):
    """Square root operator: Sqrt(x) computes x^0.5.

    In SymPy, sqrt(x) is equivalent to x**Rational(1, 2).
    """

    xtype = ((float,), float)
    symfun = staticmethod(lambda *a: sympy.sqrt(a[0]))
    np_fun = staticmethod(np.sqrt)
    showme = "Sqrt"
    sy_str = "sqrt({})"
    repr_str = "Sqrt{},[{}, {}]"
    latex_fmt = r"\sqrt{{{}}}"


class Usub(MathOperator):
    """Unary negation operator: Usub(x) computes -x.

    Note: This operator is typically not counted in parsimony/complexity
    measures as it doesn't add significant complexity.
    """

    xtype = ((float,), float)
    symfun = staticmethod(lambda a, *_: sympy.Mul(-1, a))
    np_fun = staticmethod(lambda x: np.negative(x))
    showme = "Usub"
    sy_str = "(-{})"
    repr_str = "Usub{},[{}]"


class Scale(MathOperator):
    """Scaling operator: Scale(factor, expression) computes factor * expression.

    A semantically specialised Multiplication where the first operand is always
    a numeric constant (Number terminal) and the second operand is a
    float-producing expression (operator or symbol, but NOT a Number).

    Purpose:
    - Makes the scaling factor explicit in the tree structure.
    - Ideal target for Gaussian filter mutation (tune the constant).
    - Created primarily via tree_node_grouping from Mul(Number, expr).
    - Can also be created directly in evolve_create_random.

    Invariant:
        childs[0] must be a Number terminal.
        childs[1] must be a float-producing node that is NOT a Number terminal.

    Note:
        SymPy does not have a dedicated Scale type — this maps to sympy.Mul.
        Therefore Scale is NOT registered in sym2node/d_sym2node; it is
        produced only by grouping or direct creation.

    # TODO: Discuss whether a chainable ScaleMul (Scale with multiple
    #   non-Number factors) is worthwhile. Currently kept at arity-2
    #   for simplicity; chained Mul(Number, a, b) is grouped as
    #   Scale(Number, Mul(a, b)).
    """

    xtype = ((float, float), float)
    symfun = staticmethod(lambda *a: sympy.Mul(a[0], a[1]))
    np_fun = staticmethod(np.multiply)
    showme = "Scale"
    sy_str = "({0} · {1})"
    repr_str = "Scale{},[{}, {}]"
    latex_inline = r" \cdot "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Soft validation: warn if invariant is violated (can happen during
        # intermediate tree construction or deserialization).
        if len(self.childs) >= 2:
            if not isinstance(self.childs[0], Number):
                log("w", f"Scale: first child should be Number, got {type(self.childs[0]).__name__}")
            if isinstance(self.childs[1], Number):
                log("w", f"Scale: second child should not be Number, got Number({self.childs[1]})")


class Clip(BaseMinMax, CustomOperator):
    """Clipping operator: Clip(x, min, max) constrains x to [min, max].

    Equivalent to Min(Max(x, min), max).
    """

    symfun = staticmethod(lambda *a: sympy.Min(sympy.Max(a[0], a[1]), a[2]))
    np_fun = np.clip
    showme = "Clip"
    sy_str = "(sympy.Min(sympy.Max({0}, {1}), {2}))"
    repr_str = "Clip{},[{}, {}]"
    xtype = ((float, float, float), float)


class ExprCondPair_Dummy(NodeDummy):
    """Wrapper for expression-condition pairs in Piecewise expressions.

    Each pair consists of (expression, condition) where the expression
    is returned if the condition evaluates to True.

    Named differently from sympy.ExprCondPair to avoid confusion.
    """

    arity = 2
    symfun = staticmethod(lambda *a: ExprCondPair(a[0], a[1]))
    np_fun = None
    showme = "ExprCondPair_Dummy"
    sy_str = "ExprCondPair({0}, {1})"
    repr_str = "ExprCondPair_Dummy{},[{}, {}]"
    xtype = ([(float, bool)], float)
    expr_dmy = "ExprCondPair_Dummy"


# Mapping from sympy classes to plagih Node-classes
# The non-chained version
d_sym2node = {
    sympy.Add: Add,
    sympy.Pow: Pow,
    sympy.Abs: Abs,
    sympy.sign: Sign,
    sympy.log: Log,
    sympy.Mul: Mul,
    sympy.Xor: Xor,
    sympy.Not: Not,
    sympy.Equality: Eq,
    sympy.Unequality: Ne,
    sympy.And: And,
    sympy.Or: Or,
    sympy.StrictLessThan: Lt,
    sympy.LessThan: Le,
    sympy.StrictGreaterThan: Gt,
    sympy.GreaterThan: Ge,
    sympy.cos: Cos,
    sympy.sin: Sin,
    sympy.tan: Tan,
    sympy.acos: Acos,
    sympy.asin: Asin,
    sympy.atan: Atan,
    sympy.tanh: Tanh,
    sympy.sinh: Sinh,
    sympy.cosh: Cosh,
    sympy.Min: Min,
    sympy.Max: Max,
    sympy.ITE: ITE,
    sympy.exp: Exp,
}
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
    try:
        has_bad_atoms = expr_sym.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I, sympy.im, sympy.re)
    except RecursionError as e:
        raise SympyError(f"RecursionError while validating sympy expression: {e}") from e

    if has_bad_atoms:
        # sympy.re: real part -> don't ignore; if there is a real part, there is an imaginary part.
        raise SympyError(f"Simplification failed: {expr_sym}")
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
        raise ValueError(f"NaN in {e}")
    except AttributeError as e:
        # This can happen when sympifying expressions that return non-sympy types
        if isinstance(_expr, bool):
            return sympy.true if _expr else sympy.false
        raise e
