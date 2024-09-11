"""
The main class of a gp run. It holds the following functionalities
- run information (config, evolution-specifications for the loop, success monitoring])
- population (pop_base, pop_next)
-
"""
from collections import deque

from plagih.fitness_kernel import eval_regression_sym_experimental, regression_error
from plagih.monitoring import plot_performance
from plagih.paretofront import *
import pandas as pd

from plagih.util import *
from plagih.random_nodes_generator import norm_choices, operatorpool_to_picks
import copy
import numpy as np
import random
from dataclasses import dataclass
from typing import Type
import sympy.core.numbers
from plagih.tree_labels import *
from plagih.tree_labels_chained import *
from plagih.tree_complexity.tree_edit_distance import apted_distance
from plagih.util import FLOAT_PRECISION, CHAINED_VERION, string_remove_trailing_zeroes, rnd_choice, xt_self

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


def printpl(msg_t, message_str):
    """Lightweight print function.
    Instead of checking if you should print every time, this is done here.
    message_type options can be found in config
    """
    printez(msg_t, message_str)
    return """
The factory to create trees
"""


d_sym2node = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul,
              sympy.Xor: Xor, sympy.Not: Not, sympy.And: And, sympy.Or: Or,sympy.StrictLessThan: Lt, sympy.LessThan: Le,
              sympy.StrictGreaterThan: Gt,sympy.GreaterThan: Ge, sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan,
              sympy.acos: Acos,sympy.asin: Asin, sympy.atan: Atan, sympy.tanh: tanh, sympy.sinh: Sinh, sympy.cosh: Cosh,
              sympy.Min: Min, sympy.Max: Max, sympy.ITE: ITE, sympy.exp: exp}
# The chained version is the regular version updated with the following operators
d_sym2node_chain = d_sym2node | {sympy.Add: AddChain, sympy.Mul: MulChain, sympy.Min: MinChained, sympy.Max: MaxChained,
                                 sympy.And: AndChained, sympy.Or: OrChained, sympy.Piecewise: Piecewise,
                                 sympy.functions.elementary.piecewise.ExprCondPair: ExprCondPair}


@dataclass
class Node:
    """
    Recursively holds the nodes of a tree
    parent: pointer to parent node, 'None' implies root-node
        parent-parameter added for pseudo-backprop
    """

    def __init__(self, typus: Type[Typus], childs: iter, depth=None, is_fix=False, is_chain=False):
        self.typus = typus
        self.childs = childs[:]  # ...usually a list, (??>) but can also be 'None' (<??)
        self.is_fix = is_fix  # only for trees with a "fixed" root structure
        self.depth = depth  # requires repair after changes
        self.parent_node = None  # pointer to parent, requires repair after changes
        self.root_node = None  # pointer to the one root-node, requires repair after changes
        # self.is_chain = is_chain  # sfeh:xxx check update required

    def is_chain(self):
        """
        is the node "in chain mode"?
        -> when there are more childs than input-xtypes
        ...if there are less. its weird, maybe node in construction?"""
        a = issubclass(self.typus, OperatorChained)
        b = len(self.typus.xtype[0]) < len(self.childs)
        if (a or b):
            return True
        else:
            return False

    def is_number(self):
        """sfeh's check"""
        x = issubclass(self.typus, Number)
        return x

    def repr_as_list(self):
        typus_str = self.typus.__name__  # Node-name (Mul, Symbol)

        if self.is_term():
            try:
                typus_str = self.childs[0].evalf()
            except AttributeError as ex:
                typus_str = self.childs[0]  # AttributeError("'bool' object has no attribute 'evalf'")
        else:
            childstr = ', '.join([cc.repr_as_list() for cc in self.childs])
            typus_str = f'{typus_str}, {childstr}'

        return f"[{typus_str}]"

    def str_as_list(self):
        # typus_str = self.typus.__name__  # sfeh: can str(typus) work? -> str with args recursively?
        # sfeh:delete_me if no error after 27-11-2023
        typus_str = self.typus.__name__  # sfeh: can str(typus) work? -> str with args recursively?
        # try:
        # except AttributeError as ex:
        #     print(f'XXXxxx DELETE ME if i do not come up')
        #     typus_str = self.typus  # because Terminals are obj -> 'Symbol' obj has no attr __name__

        if self.childs:
            if issubclass(self.typus, BaseOperator):
                childstr = ', '.join([cc.str_as_list() for cc in self.childs])
                typus_str = f'{typus_str}, {childstr}'
            else:
                try:
                    if self.is_number():
                        typus_str = f'{self.childs[0]:.3g}'  # '.5g'->5 decimals, trailing zeros, but rare (ugly) "E+04"
                    else:
                        typus_str = f'{self.childs[0]}'
                except TypeError as ex:
                    typus_str = str(self.childs[0].evalf())
                    typus_str = string_remove_trailing_zeroes(typus_str)
                    # sympy.ONE -> 1.0000000...
                    # sfeh:open int, non-floats are handled badly
                except Exception as ex:
                    print(f'SUCCESS sfeh:debug, delete?2 KEEP? {ex}')

        return f"[{typus_str}]"

    def get_lut_id(self):
        # """
        # sfeh: Do NOT use str() or str_as_list(), as values get rounded
        #     Do NOT return a node-list and convert them to strings
        #     -> use repr()
        # Unique+simple representation of a tree (to check in a lut if it was calculated already)
        # regular print/string option should look better, this is just for getting a unique identifier
        # returns string Identificator
        # sfeh:discuss is this just repr?
        # ID=Identificator, which"""
        #
        # # todo this whole function can be replaced with repr_as_list()
        # try:
        #     typus_str = self.typus.__name__  # sfeh: can str(typus) work? -> str with args recursively?
        # except AttributeError as ex:
        #     print('XXX DEBUG delete me if you do not remember me :)')
        #     # sfeh debug me
        #     typus_str = self.typus  # because Terminals are obj -> 'Symbol' obj has no attr __name__
        #
        # if self.childs:
        #     # if self.is_ExprCdPair():
        #     #     # elif issubclass(self.typus, TerminalDummy):
        #     #     childstr = ', '.join([cc.repr_as_list() for cc in self.childs])  # those are actual childs
        #     #     typus_str = f'({childstr})'
        #     # el
        #     if self.has_childs():  # self.is_arity_operator() n
        #         childstr = [cc.repr_as_list() for cc in self.childs]
        #         childstr = ', '.join(childstr)
        #         typus_str = f'{typus_str}, {childstr}'
        #     else:
        #         try:
        #             typus_str = f'{self.childs[0]}'
        #         except TypeError as ex:
        #             typus_str = str(self.childs[0].evalf())  # sfeh:open int? rational?, non-floats are handled badly
        #         except Exception as ex:
        #             print(f'sfeh:debug, delete? KEEP? {ex}')  # InvalidOperation([<class 'decimal.InvalidOperation'>])
        #
        # return f"[{typus_str}]"
        return self.repr_as_list()

    def str_as_expr(self):
        s = self.get_sympy_expr()
        return s

    def __str__(self):
        return self.get_expr_raw_fstring()

    def get_sympy_expr(self) -> sympy.Basic:
        """Converts directly into a sympy expr
        from node-class, go to a sympy expression
        """
        if issubclass(self.typus, Terminal):
            _sym = self.typus.get_sym()  # _sym = self.typus.symfun
            _cs = self.childs[0]
            return _sym(_cs)
        elif self.typus in (Piecewise, sympy.Piecewise):
            _sym = sympy.Piecewise
            _cs = self.childs
            _cs = [(cc.childs[0], cc.childs[1]) for cc in _cs]
            _cs = [(cc[0].get_sympy_expr(), cc[1].get_sympy_expr()) for cc in _cs]
            _cs = [sympy.functions.elementary.piecewise.ExprCondPair(cc[0], cc[1]) for cc in _cs]
            _cs = _sym(*_cs)
            return _cs
        else:
            _cs = [cc.get_sympy_expr() for cc in self.childs]
            _sym = self.typus.get_sym()
            # # sfeh: 25.08.2024 was used for debugging
            # if self.is_operator():
            #     _sym = self.typus.get_sym()
            # elif self.is_ExprCdPair():
            #     _sym = self.typus.get_sym()
            # elif CHAINED_VERION:
            #     _sym = self.typus.get_sym()
            # else:
            #     raise NotImplementedError(f'get_sympy_expr no match for {self}, {type(self.typus)}')

            try:
                r = _sym(*_cs)  # noqa (_sym is definitely assigned)
                # NO sym.check here. Just return the final result
                return r
            # except RecursionError as ex:
            #     print(f'sfeh:RecursionError, maybe Piecewise?: {self.typus}, {self.childs}, {ex}')
            #     raise RecursionError
            # except AttributeError as sfeh:
            #     return _sym(*_cs)
            # except TypeError as ex:
            #     # print(f'sfeh:TypeError?: {self.typus}, {self.childs}: {ex}')
            #     # TypeError: Argument must be a Basic object, not `Node`
            #     # !! TypeError('expecting bool or Boolean, not `(cartPos > 18.0, cartPos < 14.0)`.')
            #     return _sym(*_cs)
            except sympy.polys.polyerrors.CoercionFailed as ex:
                # sympy.polys.polyerrors.CoercionFailed: expected an integer, got 0.000564
                print(f'XXX Sympy error? Changing to TypError: {ex}')
                raise sympy.polys.polyerrors.CoercionFailed(ex)
            except IndexError as ex:
                print(f'sfeh:asddsaasd. {ex}')  # What is this kind of error?
                raise ex
            except (ValueError, TypeError) as ex:
                # if complex number; sfeh:discuss keep real part with sympy.re?
                raise ex
            except Exception as ex:
                raise ex

    def get_expr_raw_fstring(self):
        """Add (1 + a)"""
        if self.is_term():
            expr = f'{self.childs[0]}'
            expr = string_remove_trailing_zeroes(expr)
        else:
            expr = [cc.get_expr_raw_fstring() for cc in self.childs]
            expr = ', '.join(expr)
            if issubclass(self.typus, OperatorArity):
                expr = f'{self.typus.__name__}({expr})'
            elif self.is_ExprCdPair():
                expr = f'({expr})'
            elif CHAINED_VERION:
                try:
                    expr = f'{self.typus.expr_dmy}({expr})'
                except Exception as ex:
                    raise ex
                    # sfeh:debug "AttributeError("type object 'ExprCondPair' has no attribute 'expr_dmy'")"

        # Sfeh:notimplementederror here?
        return expr

    # def get_tf_expr(self):
    #     if self.childs and issubclass(self.typus, OperatorArity):
    #         _tf = self.typus.tflow
    #         _cs = [cc.get_tf_expr() for cc in self.childs]
    #         return _tf(_cs)
    #     elif self.childs and issubclass(self.typus, Terminal):
    #         _tf = self.typus.tflow
    #         _cs = self.childs[0]
    #         return _tf(_cs)
    #     raise NotImplementedError(f'get_tf_expr no match for {self}, {type(self.typus)}')

    def represent(self):
        """
        Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to str(), but adds the following information:
        - ":fix", when nodes are fixed"""
        s = self.typus

        if self.is_fix:
            s += ':fix'

        if self.has_childs():
            cs = [self.represent(cc) for cc in self.childs]
            childstr = ', '.join(cs)
            s = f"{s}, {childstr}"
        return f"[{s}]"

    def __repr__(self):
        """sfeh:WRONG! Do NOT use __str__!"""
        raise NotImplementedError

    def len_nodecount_raw(self):
        """counting the amount of nodes recursively"""
        if self.has_childs():
            return 1 + sum([cc.len_nodecount_raw() for cc in self.childs])
        else:
            return 1  # childs can currently be floats

    def is_typus(self, t: Type[Typus]):
        r = issubclass(self.typus, t)
        return r

    def is_ExprCdPair(self):
        # sfeh: Sometimes, it hits the Dummy-class, sometimes (chained?) the sympy class. Not sure, why.

        a = issubclass(self.typus, ExprCondPair_Dummy)
        b = issubclass(self.typus, sympy.functions.elementary.piecewise.ExprCondPair)
        r = issubclass(self.typus, (sympy.functions.elementary.piecewise.ExprCondPair, ExprCondPair_Dummy))
        if a:
            print('SFEH:XXX 123')
        return r

    def len_nodecount_fair(self):
        """counting the amount of nodes, but
            - ignoring "Usub"!"
            - only biggest branch of Piecewise (aka If-then-else)
        """
        if self.is_term():
            n = 1
        else:
            cc_list = [cc.len_nodecount_fair() for cc in self.childs]

            if self.is_typus(Usub):
                n = sum(cc_list)
            # elif self.is_ExprCdPair():
            #     n = 1 + max(cc_list)  # sfeh:discuss
            else:
                n = 1 + sum(cc_list)

        return n

    def __len__(self):
        return self.len_nodecount_fair()

    def get_typus(self):
        return self.typus

    def get_arity(self):
        return len(self.typus.get_child_xts())

    def get_xtype_tuple(self):
        return self.typus.xtype

    def get_xtype_childs(self):
        return self.typus.xtype[0]

    def get_xtype_self(self):
        return self.typus.xtype[1]

    def set_typus(self, t: Type[Typus]):
        """all other values are automatically set by assigning the respected node"""
        self.typus = t

    def set_parent(self, n: 'Node'):
        self.parent_node = n

    def set_root(self, n: 'Node'):
        self.root_node = n

    def set_childs(self, child_list):
        if isinstance(child_list, (list, tuple)):
            self.childs = child_list
            if self.has_childs():
                for cc in self.childs:
                    cc.set_parent(self)  # set pointer in child-nodes
                    cc.set_root(self)
        else:
            raise TypeError(f'childs must be set as list, not {type(child_list)}: {child_list}')

    def update_fixed_nodes(self, origin: 'Node'):
        """Updating the fixed nodes in a tree where they were lost for some reason.
        This should never be the case! But it happened during development of recreating a tree from expression.
        This might also be useful in tree checks"""
        if origin.is_fix:
            if str(self.typus) != str(origin.typus):
                raise
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.update_fixed_nodes(origin.childs[ii])

    def get_all_nodes_visualize(self, setid: str):
        """returns all nodes in a tree as list
        a+1 -> [+a1, a, 1]"""
        # if len(self.childs) == 0:
        showme = f'{self.childs[0]}' if self.is_term() else f'{self.get_typus().showme}'

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

    # def get_enum(self, parent_id='root'):
    #     res = parent_id
    #
    #     if self.is_operator():
    #         for cc in self.childs:
    #             res.extend(cc.get_all_nodes())
    #
    #     return res

    def get_mutable_rootnodes(self, extend_lvls=2):
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
                for cc in self.childs:
                    n.extend(cc.get_mutable_rootnodes(extend_lvls=extend_lvls))

        return n

    def get_apted_notation(self):
        """Calculating the TED requires this (weird) representation"""
        return f"{{{self.get_typus()}{''.join([cc.get_apted_notation() for cc in self.childs])}}}"

    def get_max_depth(self, depth=0):
        """Go through all nodes, save depth
        sfeh: this computes the depth and does not take advantage of saved depths"""
        if len(self.childs) <= 1:
            return depth
        else:
            return max(cc.get_max_depth(depth=depth + 1) for cc in self.childs)

    def is_arity_operator(self):
        # Nodes that are notchained-nodes
        return issubclass(self.typus, OperatorArity)  # sfeh check? Terminal?

    def is_operator(self):
        return issubclass(self.typus, BaseOperator)

    def is_term_and_symbol(self):
        if self.is_term():
            a = issubclass(self.typus, Symbol)
            # b = issubclass(self.typus, Number)
            # c = issubclass(self.typus, Boolean)
            if a:
                return True
        return False

    def is_term(self):
        # sfeh:discuss is_atom, rename all to atom?
        return issubclass(self.typus, Terminal)

    def has_childs(self):
        # better to check for recursive use, as e.g. ExprCondPair is not a regular operator
        return not self.is_term()

    def force_input_node(self, tb):
        """Some trees have only constants as terminals.
        This function replaces a terminal node with an input, if the tree only has constants"""
        node_list = self.list_terminal_nodes()
        a = [x.is_term_and_symbol() for x in node_list]
        if any(a):
            return
        else:
            node_list = [x for x in node_list if x.typus == Number]
            # sfeh: do this in create new tree?
            node = rnd_choice(node_list)  # debug if ignores chains
            xtype = xt_self(node.get_xtype_tuple())
            new_node = tb.node_selector.choose_symbol_node(xtype)
            node.set_new_node(new_node)

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
            for cc in self.childs:
                cc.repair_depth(depth=depth + 1)

        return

    def set_new_node(self, nd_new: 'Node'):
        """Replacing oneself with another node"""
        self.set_typus(nd_new.typus)  # sfeh remove childs, is_fix...
        self.set_childs(nd_new.childs)  # sfeh maybe must be updated recursively
        self.repair_depth(depth=self.depth)  # Especially required for crossover or branchesnd_new
        # sfeh: depth is repaired at the end, as some bug leads to wrong depths somewhere (depth=None)
        # sfeh check fixed or if type matches?

    # sfeh:Do links to parents lead to problems when crossover/etc happens?

    def repair_backlink(self, parent: 'Node', root: 'Node'):
        """backlink was introduced on 23.04.2024,
        linking the root and parent nodes"""
        self.root_node = root
        self.parent_node = parent
        for cc in self.childs:
            cc.repair_backlink(self, root)
        # self.repair_depth()

    def replace_with(self, typus, childs):
        if typus is not None:
            self.set_typus(typus)
        if childs is not None:
            self.set_childs(childs)
        # self.repair()

        # self.repair_depth(self.depth)  # sfeh:discuss not required
        # no checks

    def list_terminal_nodes(self):
        base = self.list_mutable_nodes()
        base = [x for x in base if x.is_term()]
        return base

    def list_mutable_nodes(self, xtype=None, skip_first=False, allow_chain=False) -> ['Node']:
        """return all nodes that are mutable, aka suite for point- or branchmutation
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!"""

        # -> Check, if this node should be added
        if self.is_fix:
            node_list = []
        elif skip_first:  # ignore_first is automatically set to false during recursion
            node_list = []
        elif self.is_ExprCdPair():  # It is just a dummy holding a tuple
            node_list = []
        elif self.is_typus(Piecewise):
            node_list = []  # Piecewise has ambiguous xtype that can not be checked
        else:

            if xtype is None or xtype == self.get_xtype_self():
                if not allow_chain or (allow_chain and self.is_chain()):
                    node_list = [self]
                else:
                    node_list = []
            else:
                node_list = []

        # recursively add the other nodes
        if self.has_childs():  # sfeh:chain-operators discuss
            for cc in self.childs:
                a = cc.list_mutable_nodes(xtype=xtype, skip_first=False, allow_chain=allow_chain)
                node_list.extend(a)

        return node_list

    def evolve_mutate_filter_gauss(self):
        """Recursively filter the nodes in the branch of fintree
        sfeh:   random filter all terminal nodes /
                single node /
                nodes in a branch /
                random nodes in a branch /
                intelligent filtering
        """
        if self.has_childs():
            for cc in self.childs:
                cc.evolve_mutate_filter_gauss()

        else:
            if self.is_number():
                self.childs[0] = round(random.gauss(self.childs[0], 0.1),
                                       FLOAT_PRECISION)  # sfeh: -> no symbols -> userspecific

        return

    def tree_node_grouping(self):
        """
        If possible, this groups nodes to a simpler expression (if possible).
        E.g. a ** 2 -> square(a)
             a + b + c -> sum(a, b, c)  (chaining)

        # sfeh:idea Heavyside function. input a val, input b threshold
        """
        if self.is_term():  # good for runtime
            return

        if self.typus in (Pow, Powrounded):
            n_exp = self.childs[1].childs[0]  # must exist
            if n_exp == -1:
                self.replace_with(DivFraction, childs=[self.childs[0]])
            elif n_exp == 2:
                self.replace_with(Square, childs=[self.childs[0]])
            elif n_exp == 0.5:
                self.replace_with(Sqrt, childs=[self.childs[0]])
            elif n_exp == 0:
                self.replace_with(Number, childs=[1])
            elif self.childs[1].typus == Round:
                self.replace_with(Powrounded, childs=[self.childs[0], n_exp])
            # sfeh:discuss, powrounded here? not clear
            elif self.childs[1].typus == Number and n_exp % 1 == 0:
                self.set_typus(Powrounded)

            # sfehxxx sub, usub replace

        elif self.typus == Mul:
            if not self.is_chain():  # div only for
                if self.childs[0].typus == DivFraction:
                    self.replace_with(Div, childs=[self.childs[1], self.childs[0].childs[0]])
                elif self.childs[1].typus == DivFraction:
                    self.replace_with(Div, childs=[self.childs[0], self.childs[1].childs[0]])

                elif self.childs[0].typus == Number:
                    mul1 = self.childs[0].childs[0]
                    if mul1 == -1:  # aka sympy.S.NegativeOne -1, was -1 before
                        self.replace_with(Usub, childs=[self.childs[1]])  # sfeh Usub ONLY option, ignore usub tree len
                    elif 0 < mul1 < 1:
                        self.replace_with(Div, childs=[self.childs[1],
                                                       Node(Number, childs=[1 / mul1])])  # sfeh keep "div" as option

                elif self.childs[1].typus == Number:
                    # sfeh this code can be simplified
                    mul1 = self.childs[1].childs[0]
                    if mul1 == -1:  # aka sympy.S.NegativeOne -1, was -1 before
                        self.replace_with(Usub, childs=[self.childs[0]])  # sfeh Usub ONLY option, ignore usub tree len
                    elif 0 < mul1 < 1:
                        self.replace_with(Div, childs=[self.childs[0], 1 / mul1])
                        sfeh_div_by_test = 1 / mul1  # sfeh keep "div" as option
                        print(f'asd this needs testing when reached in future {sfeh_div_by_test}')

        for cc in self.childs:
            cc.tree_node_grouping()


def eval_parsimony(tree: Node, complexity_measure, origin_tree=None):
    if complexity_measure == 'tree_node_count_raw':  # number of nodes
        return tree.len_nodecount_raw()  # returns the number of nodes  # sfeh weights
    elif complexity_measure == 'tree_node_count':
        return tree.len_nodecount_fair()  # returns the number of nodes  # sfehxx weights
    elif complexity_measure == 'tree_edit_distance':  # tree_edit_distance, fintree-edit-distance
        apted1 = tree.get_apted_notation()
        apted2 = origin_tree.get_apted_notation()
        distance, mapping = apted_distance(apted1, apted2)  # sfeh the mapping could be useful somewhere
        return distance
    else:
        raise Exception(f'Complexity measurement not available: {complexity_measure}')


class RootNode_Dummy(Node):
    """Sfeh:discuss
    this node can be used as dummy and is used to mimic a root type"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def sympy_to_tree(s_expr: sympy.Basic, allow_chain=CHAINED_VERION) -> Node:
    """
    Important: start with the most specific rule
    # sfeh:discuss computational improvement when option to ignore args? do "raises" in args save time?
    # check is expr is an accepted operator, otherwise reconstruction probably fails"""
    if isinstance(s_expr, bool):
        return Node(Boolean, [s_expr])

    elif isinstance(s_expr, sympy.logic.boolalg.BooleanAtom):
        s_expr = True if isinstance(s_expr, (bool, sympy.logic.boolalg.BooleanTrue)) else False
        return Node(Boolean, [s_expr])

    # the following two lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    elif s_expr.is_Atom:
        if s_expr.is_Symbol:
            return Node(Symbol, [s_expr])  # str VERY important!! "Symbol type input" is not accepted
        else:
            # expr_eval = s_expr.evalf(FLOAT_PRECISION)  # sfeh
            # if abs(s_expr - expr_eval) > 0.001:  # sfeh use float recision here 0.1**FLOAT_PRECISION
            #     expr_eval = s_expr
            if s_expr.is_Boolean:
                # return Node(Boolean, [bool(expr_eval)])
                return Node(Boolean, [s_expr])
            elif s_expr.is_number:  # is_float does not match int
                # return Node(Float, [round(float(expr_eval), FLOAT_PRECISION)])
                return Node(Number, [s_expr])
                # "TypeError: Cannot convert complex to float" -> ignore the whole expression, let it fail
            else:
                raise NotImplementedError(f'What happened here? {s_expr}')

    else:  # **Operators**

        cc_nodes = []
        for arg in s_expr.args:
            cc_nodes.append(sympy_to_tree(arg, allow_chain=CHAINED_VERION))

        if CHAINED_VERION:
            op = d_sym2node_chain[type(s_expr)]
            return Node(op, childs=cc_nodes)

        if isinstance(s_expr, sympy.functions.elementary.piecewise.ExprCondPair):
            return Node(ExprCondPair, cc_nodes)

        elif isinstance(s_expr, sympy.Piecewise):
            # "Chained_VERSION" version is handled before
            reversed_pairs = list(s_expr.args[::-1])  # tuples to list, reversed: tuple must be nested the deepest
            reversed_pairs = [[sympy_to_tree(xx, allow_chain=CHAINED_VERION) for xx in list(i)] for i in
                              reversed_pairs]  # noqa
            otherwise = reversed_pairs[0][0]  # the last "True" condition
            for pairs in reversed_pairs[1:]:
                otherwise = Node(Ifte, [pairs[1], pairs[0], otherwise])
            return otherwise

        # sfehx include Usub, ignore usub in tree len()

        # elif isinstance(s_expr, RoundDummy):
        #     return Node(Round, [cc_nodes[0]])

        elif isinstance(s_expr, Mul):
            if s_expr.args[0].is_Rational:
                div_by = 1 / s_expr
                print(f'sfeh:open div by {div_by}')

        elif isinstance(s_expr, tuple(d_sym2node)):
            clss = d_sym2node[type(s_expr)]
            if len(s_expr.args) > len(clss.get_child_xts()):
                _cc = cc_nodes[0]
                for _c2 in cc_nodes[1:]:
                    _cc = Node(clss, [_cc, _c2])
                return _cc
                # raise TypeError(f"{clss} takes exactly {len(clss.get_child_xts())} args ({len(expr.args)} given)")
            else:
                return Node(clss, cc_nodes)

        elif isinstance(s_expr, sympy.re):
            # We assume, that these Functions occur due to assumptions.
            # (for now: sympy.re, the Real-part of a number)
            # hence, we can skip this function while rebuilding without loosing any information
            # as Symbols match their assumptions when they are rebuilt aswell
            # sfeh:debug: When does sympy.re occur? Are there other cases?
            # sfehxxx reintroduce piecewise? discuss: ignore trees which have real/complex numbers
            # return sympy_to_tree(expr.args[0], allow_chain=allow_chain)
            return cc_nodes  # sfeh:delete? never reached? 26.08.2024

    # sfeh:discuss
    # NotImplementedError: Expr missing: ITE(p > 13, tan(p - v) >= 2.578643, tan(p - v) >= 1)
    # this should not have occured, because it evaluates to bool, not to float
    raise NotImplementedError(f'Expr missing: {s_expr}')


if __name__ == '__main__':

    for x in d_sym2node.keys():
        lel = 4.5
        print(x, isinstance(lel, x))


class NodeRandomizer:

    def __init__(self, build_operator_dict, build_variables_list):
        """make all probabilities sum to 1 for each categoray (Add: 2, Mul: 1, Tan: 0.5) in"""

        self.pick_op, self.pick_op_match = operatorpool_to_picks(build_operator_dict)
        # sfeh: Acos: 0.33, Asin: 0.33, Atan: 0.33, Tanh: 0.5, Usub: 1, Xor: 1
        # Round: 0.5, Eq: 1,  # Ne: 0.5, #  # Log1p: 0.1, Gt: 0.1, Ge: 0.1,, Tan: 0.1, Sub: 1, Cos: 0.33
        # Powrounded: 0.5

        self.pick_symbol = {
            float: norm_choices([[sympy.Symbol(ii, real=True, imaginary=False), 1] for ii in build_variables_list]),
            bool: []}  # NotImplementedError

        # -> Choosing 50 random numeric values from the dataset for building trees ...just not zeros)
        # samples = [ii for ii in itertools.chain.from_iterable(df[build_variables_list].sample(n=50).values) if ii != 0]
        self.pick_constant = {float: norm_choices([
            [lambda: round(random.normalvariate(1, 1), FLOAT_PRECISION), 0.1],
            [lambda: round(random.randint(1, 20), FLOAT_PRECISION), 0.1],
            # [lambda: round(random.choice(samples), FLOAT_PRECISION), 0.5]
        ]),
            bool: norm_choices([[lambda: random.choice((True, False)), 1]])}

    def choose_operator(self, xt):
        # sfehxxxx allow_chain
        op = np.random.choice(self.pick_op[xt][0], p=self.pick_op[xt][1])  # no (), which would evaluate the op
        return op

    def choose_operator_match(self, xtype):
        # sfehxxx allow_chain
        op = np.random.choice(self.pick_op_match[xtype][0], p=self.pick_op_match[xtype][1])
        return op

    def choose_terminal_node(self, xt, p_observation=0.5):
        if np.random.random() > p_observation:
            try:
                _v = self.choose_symbol(xt)
                return Node(Symbol, [_v])
            except (TypeError, IndexError):
                pass  # return a constant (E.g. because there are no boolean observations)

        _v = self.choose_constant_node(xt)
        # sfeh expected str|int|long|float|Decimal|Number object but got 'Node'

        return _v

    def choose_constant_node(self, xt):
        _v = np.random.choice(self.pick_constant[xt][0], p=self.pick_constant[xt][1])()  # just dist. must be ()
        if xt == float:
            _v = sympy.Float(_v)  # sfeh:discuss allow "rational" inputs? 1/3, 3/4, ...
            # _v = sympy.Rational(_v)  # sfeh:discuss allow "rational" inputs? 1/3, 3/4, ...
            return Node(Number, [_v])  # round FLOAT_PRECISION was here
        else:
            # _v = sympy.logic.boolalg.BooleanAtom(_v)  # sfeh:discuss: vs. Boolean
            # -> sympy.sympify('And(True, BooleanAtom(False))')
            _v = _v  # BooleanAtom was here - why? Any purpose?
            return Node(Boolean, [_v])

    def choose_symbol_node(self, xt):
        """similar to choose_terminal_node()"""
        _v = self.choose_symbol(xt)
        return Node(Symbol, [_v])

    def choose_symbol(self, xt):
        _v = np.random.choice(self.pick_symbol[xt][0], p=self.pick_symbol[xt][1])
        return _v


def randomly_split_range(range_max: int, num_splits: int) -> list[int]:
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


def tree_simplification(tree, allow_chain=CHAINED_VERION) -> Node:
    """
    (Tries to) simplify/mathematically-reduce a tree. It is quite experimental
    # sfeh sympy-reconstruct patterns
    #   map symoy-sign to a sum
    #   map piecewise to if-then-else
    #   map power fractal - to sqrt?
    """
    tree_copy = copy.deepcopy(tree)
    expr_sym = tree.get_sympy_expr()
    tree = sympy_to_tree(expr_sym, allow_chain=allow_chain)
    if not CHAINED_VERION:
        tree.tree_node_grouping()
        if len(tree_copy) < len(tree):
            print(f'WHATTPPENDED SFEH'
                  f'\n\told: {tree_copy.str_as_list()}'
                  f'\n\tsym: {tree.str_as_list()}')
            if tree_copy.get_sympy_expr() != tree.get_sympy_expr():
                print_warning(f'\t{tree_copy.get_sympy_expr()}'
                              f'\n\t{tree.get_sympy_expr()}')
    return tree


def evolve_reduce_simplify(tree: Node, completely=True, force=False) -> Node:
    """Reducing a fintree to its most basic form with sympify.
    (completely = False: reduce just one branch. if you wanted to have more complexity)"""
    tree_copy = copy.deepcopy(tree)
    if completely:  # reduce the complete tree
        nodes_lv0 = tree.get_mutable_rootnodes(extend_lvls=0)  # only required for fixed-core trees
        for cc in nodes_lv0:
            cc.set_new_node(tree_simplification(cc))
    else:
        node_list = [n for n in tree.list_mutable_nodes() if
                     issubclass(n.typus, OperatorArity)]  # ignoring leaf nodes...
        if len(node_list) == 0:
            print_warning('wwww', f'Tree for simplification does not provide operators: {tree}')
            return tree
        node = np.random.choice(node_list)
        node.set_new_node(tree_simplification(node))  # sfeh chosen must be set again? or not? test it at least.
    if force:
        return tree
    else:
        if len(tree_copy) < len(tree):
            print_warning('w',
                          f'Tree grew larger during simplification:\n\t{tree_copy.str_as_list()}\n\t{tree.str_as_list()}')
            print_warning('ww', f'tree_copy: {len(tree_copy)} vs. {len(tree)}')
            # [Square, [Powrounded, [-0.91], [cartPos]]] < [Pow, [-0.91], [Mul, [2], [Round, [cartPos]]]]
            ######
            # sfeh:open simplify usub in front of terminal nodes
            # Warning (w): Tree grew larger during simplification:
            # 	[Add, [Sub, [1.06], [cartPos]], [Div, [Mul, [Add, [cartVel], [Mul, [9.00], [Add, [Max, [-0.341], [cartPos]], [Abs, [cartPos]]]]], [Sin, [cartVel]]], [0.0420]]]
            # 	[Add, [Add, [1.06], [Usub, [cartPos]]], [Mul, [Mul, [23.8], [Add, [Add, [cartVel], [Mul, [9.00], [Abs, [cartPos]]]], [Mul, [9.00], [Max, [-0.341], [cartPos]]]]], [Sin, [cartVel]]]]
            return tree_copy
        else:
            return tree


def node_deepcopy(tree: Node):
    _cpy = copy.deepcopy(tree)
    return _cpy


class Evolution:
    """
    was "TreeBuildRestrictions"
    functions to build trees, with the advantage of being able to use general build restrictions."""

    def __init__(self, origin_xtype, origin_tree, node_selector, build_restrictions, complexity_metric):
        """
        origin_tree: A tree, which
        """
        self.origin_xtype = origin_xtype
        self.origin_tree = origin_tree

        self.node_selector = node_selector

        self.complexity_metric = complexity_metric

        self.depth_max = build_restrictions.get('depth_max', 10)
        self.nodes_max = build_restrictions.get('nodes_max', 100)

    def evolve_prune_tree(self, tree: Node):
        """prune depth
        -> prune everything below a certain level... (should not happen in the first place)
        prune nodes
        -> get node difference, get nodelist, untill small enough: split the difference, prune nodes until

        sfeh:discussion there is a difference between parsimony and complexity...
        sfeh:discuss analyze the amount of trees that have to be pruned?
        sfeh:open add labelweight_max to"""
        nodelist = tree.list_mutable_nodes()
        for dnode in nodelist:
            if dnode.depth == self.nodes_max and dnode.get_arity() > 0:
                print_warning('wwww', f'Node in fintree is too deep: {dnode.depth}')
                new_node = self.node_selector.choose_terminal_node(dnode.get_xtype_self())
                new_node.depth = dnode.depth
                dnode.set_new_node(new_node)

        prune_amount = len(tree) - self.nodes_max
        while prune_amount > 0:
            print_warning('wwww', f'Tree too complex: {len(tree)} > {self.nodes_max}, pruning {prune_amount}.')
            nodelist = tree.list_mutable_nodes()
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

    def new_tree_nodes(self, nn, p_term=0):
        """insert a (random) number of branches at the first possible "layer" (not necessary depth)
        (If all nodes are modifiable, it is the root node. Otherwise, it is the first layer of modifiable nodes
        - get these nodes, randomly choose a subset of those
        - get the amount of nodes allowed to add. (max nodes without the core-fintree + the nodes about to delete)
        - split the amount of nodes up (randomly) and add these new branches to the fintree
        sfeh:idea mutate only the childs of a node! The label stays the same"""

        # if not isinstance(self.origin_tree, RootNode_Dummy):
        #
        # else:
        # layer0_nodes = self.origin_root
        # evotree = self.evolve_create_random(xtype, -1, num_rest=nodeamount, depth=0, p_term=p_term)

        evotree = node_deepcopy(self.origin_tree)
        layer0_nodes = evotree.get_mutable_rootnodes()
        layer0_splits = randomly_split_range(nn, len(layer0_nodes))

        for ii, node0 in enumerate(layer0_nodes):  # pareto_insert branches! get layer always, node ids might change
            lvl0_node = np.random.choice(node0.list_mutable_nodes())  # layer0_branch =
            # branch_size = layer0_nodes[ii]  # sfeh:idea + len(lvl0_node)
            new_subbranch = self.new_tree_nodes(lvl0_node.get_xtype_self(), p_term=p_term)
            lvl0_node.set_new_node(new_subbranch)

        return evotree

    def evolve_create_random(self, xt_out, depth_goal, num_rest=-1, depth=0, p_term=0.0):
        """num_rest: -1 ignores the node number restriction
        sfeh:open make depth_goal -> depth_rest"""

        if depth == self.depth_max or depth == depth_goal or num_rest == 0 or random.random() < p_term:
            node = self.node_selector.choose_terminal_node(xt_out)
            node.depth = depth

        else:
            # sfeh:allowchain?
            label = self.node_selector.choose_operator(xt_out)
            child_xts = label.get_child_xts()
            num = len(child_xts)
            childs = []

            if num_rest > 0:
                nums = randomly_split_range(num_rest - 1, num)
                for ii, xt in enumerate(child_xts):
                    cc = self.evolve_create_random(xt, depth_goal, num_rest=nums[ii], depth=depth + 1, p_term=p_term)
                    childs.append(cc)
            else:
                for xt in child_xts:
                    cc = self.evolve_create_random(xt, depth_goal, num_rest=-1, depth=depth + 1, p_term=p_term)
                    childs.append(cc)

            node = Node(label, childs, depth=depth)

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
                new_subbranch = self.evolve_new_endrecursive(lvl0_nodes.get_xtype_self(), depth_goal, num_rest=num_rest,
                                                             depth=lvl0_nodes.depth, p_term=p_term)
                lvl0_nodes.set_new_node(new_subbranch)

        else:
            evotree = self.evolve_new_endrecursive(self.origin_xtype, depth_goal, depth=depth, num_rest=num_rest,
                                                   p_term=p_term)
        return evotree

    def evolve_mutate_filter(self, tree):
        """Mutates a number of float terminal of a fintree
        - filter point/branch/all, branch can also affect a point only as well as all nodes
        - filter observations?
        - filter terminals
        - filter with which filter?"""

        _nd = np.random.choice(tree.list_mutable_nodes())
        # sfeh: does nothing if no float values are in this tree
        _nd.evolve_mutate_filter_gauss()

        return tree

    def evolve_mutate_point(self, tree: Node):
        """Mutate a single mutable point in any Tree.
        sfeh:debug is the fintree a fintree copy or the same fintree?"""
        evotree = copy.deepcopy(tree)

        node = rnd_choice(evotree.list_mutable_nodes(allow_chain=CHAINED_VERION))  # debug if ignores chains
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

    def evolve_mutate_branch_depth(self, tree: Node, depth_goal, p_term=0.0):
        """"""
        n_init = len(tree)
        node_list = tree.list_mutable_nodes()
        node = np.random.choice(node_list)
        xtype_out = node.get_xtype_self()
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
        nd = tree.list_mutable_nodes()
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

        a_nds = aa.list_mutable_nodes(skip_first=True)  # why actually ignore root node
        a_nds = [x for x in a_nds if len(x) > 1]

        if len(a_nds) == 0:
            raise ValueError(f'Crossover tree 1 has no mutable nodes')

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


if __name__ == '__main__':
    _test_open = '[Ifte, [Or, [b < -1], [And, [b < 0.1], [a < -0.05]]], 2, [Ifte, [And, [And, ' \
                 '[b > -0.45], [b < -0.05]], [a < -0.5]], 0, [Ifte, [a < 0], 0, 2]]]',
    _test_loadabls = ["['+',['-',['Ifte',['True'],['sign',['cartVel']],['/',[2.3],[4]]],['cartVel']],[-1.3]]",
                      '["Ifte:fix",["<",["cartVel"],[0]],["0:fix"],["2:fix"]]',
                      '["Ifte", ["Not", [False]], [0.0], [2.0]]']


class Candidate:
    """
    WAS: class FinalizedTree
    An actual individual (Tree + meta-infos/phenotypes)"""

    def __init__(self, tree: Node, fitness, parsimony, tag: str):
        self.tree = tree
        self.fitness = fitness
        self.parsimony = parsimony
        self.last_evolution = deque([tag], maxlen=10)  # sfeh:open

    def append_tag(self, tag):
        self.last_evolution.append(tag)

    def get_last_tag(self):
        return self.last_evolution[-1]

    def __str__(self):
        """Show the Parsimony and Fitness of a tree"""
        return f'[{self.get_parsimony():2.0f}: fit {self.get_fitness():4.2f}]'

    def full_string(self):
        return f'{self.__str__()}: {self.get_evotree().get_sympy_expr()}'

    def get_evotree(self):
        return self.tree

    def append_tag(self, tag):
        self.meta.append_tag(tag)

    def get_fitness(self):
        # return self.meta.fitness
        return self.fitness

    def get_parsimony(self):
        return self.parsimony

    def set_fitness(self, fitness):
        self.meta.fitness = fitness

    def set_parsimony(self, parsimony):
        self.meta.parsimony = parsimony

    def get_last_evolution(self):
        return self.meta.get_last_tag()  # sfeh same name?


class ExplainableGP:

    def __init__(self, name, pop_max, gen_max, rootdir, df_train, df_control, evolve: Evolution):
        self.time_start = time.perf_counter()
        self.name = name
        self.df_train = df_train
        self.df_control = df_control
        self.evolve = evolve
        self.pop_max = pop_max
        self.mp_cores = 1  # sfeh: open MP (multiprocessing)
        self.gen_max = gen_max
        self.rootdir = rootdir
        self.gen_id = 0

        printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET_COLOR}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        self.paretofront = []  # not a separate class; requires too much information
        self.pop_genepool = []  # sfeh:discuss maybe better names?
        self.pop_next = []

        self.lut_sym = {}
        self.lut_parsim = {}
        self.lut_fitness = {}  # Lookup-table for tree(-expressions) and its fitness/parsimony. Improving runtime a lot!

        # monitoring
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0
        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'parsim_avg', 'parsim_var', 'parsim_best',
                                                'gens_since_last_pareto'])

    # SFEH:xxx
    # WHATTPPENDED
    # [Div, [2.81], [cartVel]]
    # [Mul, [2.81], [DivFraction, [cartVel]]]
    # 2.81 / cartVel
    # 2.81 / cartVel
    # --> 3/a =sym> Mul(3, (1/a))
    # --> if Mul(DivFraction()) -> Divide

    def pop_print(self):
        """Print the expressions of all trees in a population"""
        n = [f'{k.full_string()}' for k in self.pop_next]
        n = [f'{BColors.BLUE}{x}' if ii % 2 == 0 else f'{BColors.YELLOW}{x}' for ii, x in enumerate(n)]
        n = [f'{k}\n' if ii % 10 == 9 else f'{k}\t' for ii, k in enumerate(n)]  # stop \n in line 0
        n = ''.join(n)
        n = re.sub(r'\n$', '', n)  # remove trailing \n (\t irrelevant)
        n = f'{n}{BColors.RESET_COLOR}'
        print(n)

    def run_update_paretofront(self, pop):
        """
        CAUTION: This Function was tried to be separated many times now. it never worked.
        This was tried =3= times now. Please increase the counter when you try.
        Reason: The paretocandidates should be simplified if possible and gens_since_last_pareto is reset.

        sfeh:discuss pareto-efficient, but different pareto entries?


        """
        pop_parcandidates = pareto_from_pop(pop)  # pareto-candidates in the pop, renamed to be clear

        for candidate_tree in pop_parcandidates:
            success = False
            fit = candidate_tree.get_fitness()
            par = candidate_tree.get_parsimony()

            if par < self.paretofront[0].get_parsimony():
                printyeah('a', f'Paretofront: New simplest entry. parsimony: {par} fitness: {fit:6.4f}, '
                               f'old simplest entry had {self.paretofront[0].get_parsimony()}')
                success = True

            # if all([self.fitness_compare(fit, p.get_fitness()) for p in self.paretofront]):  # sfeh-kernel
            elif fit < self.paretofront[-1].get_fitness():
                printyeah('a', f'Paretofront: New fittest entry. parsimony: {par} fitness: {fit:6.4f}')
                success = True
            else:
                for p in self.paretofront:
                    if par >= p.get_parsimony():
                        continue
                    else:
                        if fit < p.get_fitness():
                            success = True

            if success:
                self.gens_since_last_pareto = 0
                try:  # sfeh: trying to simplify the tree for improved pareto
                    symtree = evolve_reduce_simplify(candidate_tree.get_evotree(), force=True)
                    sym_candidate = self.tree_to_candidate(symtree, tag='sfeh:sym')
                    if sym_candidate.get_parsimony() < candidate_tree.get_parsimony():
                        printyeah('a', f'Paretofront: Even further simplified! '
                                       f'{sym_candidate.get_parsimony()} < {candidate_tree.get_parsimony()}')
                        self.pop_next_append(sym_candidate, force=True)

                    print(blue_string(f'Simplified symtree: {sym_candidate.get_parsimony()}: {symtree}'))

                except KeyError as ex:
                    print_caution(f'SFEH: this tree could whatever {ex}')  # -> piecewise function, mostly

                _obsoletes = [i for i in self.paretofront if
                              i.get_fitness() > candidate_tree.get_fitness() and i.get_parsimony() >= candidate_tree.get_parsimony()]
                if _obsoletes:
                    x = [f'{i.full_string()}' for i in _obsoletes]
                    printyeah('a', f'Paretofront: Removing obsolete entries {x}')
                self.paretofront = [ftree for ftree in self.paretofront if ftree not in _obsoletes]
                self.paretofront.append(candidate_tree)
                self.paretofront = pareto_sort(self.paretofront)

        return

    def end_generation(self):
        # sfeh:open end generation in every generation
        self.run_update_paretofront(self.pop_next)

        self.pop_genepool = self.pop_next[:]
        self.pop_print()
        self.pop_next = []
        self.analyze_generation()
        self.gen_id += 1

        self.time_genstart = time.perf_counter()

    # sfeh:idea sympy.NumberSy,bol

    def gen_create_initial(self):

        printpl('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')  # sfeh debug

        if self.evolve.origin_tree is not None:
            cand_origin = self.tree_to_candidate(self.evolve.origin_tree, raise_if_useless=False, tag='origin')
            self.pop_next_append(cand_origin)
        else:
            if CHAINED_VERION:
                @self.create_trees(rate=0.5)
                def init_rand1():
                    n = np.clip(int(random.normalvariate(4.0, 1.0)), 3, 6)
                    tree = self.evolve.evolve_new_tree_depth(n, float, p_term=0)
                    tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                    # sfeh trees can shrink to single-noded trees
                    if tree.get_max_depth() <= 1:
                        raise ValueError(f'Tree did not get complex enough')
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2():
                    n = np.clip(int(random.normalvariate(3.5, 1.0)), 3, 6)
                    tree = self.evolve.evolve_new_tree_depth(n, float, p_term=0)
                    tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                    return tree
            else:
                @self.create_trees(rate=0.5)
                def init_rand1a():
                    n = np.clip(int(random.normalvariate(3.0, 1.0)), 3, 5)
                    tree = self.evolve.evolve_new_tree_depth(n, float, p_term=0)  # sfeh: xtype not always float
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2a():
                    n = np.clip(int(random.normalvariate(2.5, 1.0)), 3, 5)
                    return self.evolve.evolve_new_tree_depth(n, float, p_term=0)

        self.paretofront = pareto_from_pop(self.pop_next)  # initialize
        self.pop_genepool = self.pop_next[:]
        self.pop_next = []
        self.analyze_generation()
        self.gen_id += 1
        return

    def pop_next_append(self, ct: Candidate, force=False):
        evotree = ct.get_evotree()
        # from visualization.pygraphviz import render_pygraphviz
        if force and ct.get_parsimony() < TREE_MIN_PARSIMONY:
            # sfeh raise ValueError(f'Tree not complex enough for population, sfeh')
            return
        printpl('gggg', f'|->{evotree.len_nodecount_fair():2.0f}: {evotree.str_as_expr()}')
        self.pop_next.append(ct)

    def create_trees(self, rate=0.0, crossover=False):
        """Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the final tree (candidate_tree) is refurbished."""

        def loop(create_tree_func):
            n = int(rate * self.pop_max)
            n_success = 0
            fails_list = []
            tag = create_tree_func.__name__
            printpl('ggg', f'->Evolving {n}x \'{tag}\'...')

            while n_success < n:
                try:
                    if crossover:
                        t1, t2 = create_tree_func()
                        ctree1 = self.tree_to_candidate(t1, tag=tag)
                        self.pop_next_append(ctree1)
                        n_success += 1
                        ctree2 = self.tree_to_candidate(t2, tag=tag)
                        self.pop_next_append(ctree2)
                        n_success += 1
                    else:
                        evotree = create_tree_func()
                        ctree = self.tree_to_candidate(evotree, tag=tag)
                        self.pop_next_append(ctree)
                        n_success += 1

                except (ValueError, ArithmeticError) as ex:
                    if 'ValueError: Tree did not get complex enough' in str(ex):
                        fails_list.append(ex)
                        print_warning('www', f'\'{tag}\' failed: {ex}')
                        if len(fails_list) > 2 * n_success + 5:  # allow more fails: fails_list > n
                            print_caution(f'Evolution fails too often: {tag}, {len(fails_list)}. ({n_success} successful).'
                                          f'\n{fails_list}')
                            return  # sfeh raise?
                    else:
                        raise
                except TypeError as ex:
                    # TypeError: Cannot convert complex to float
                    # ==> Okay!
                    # Value passed to parameter 'x' has DataType bool not in list of allowed values: bfloat16, float16..
                    # ==> sfeh probably this error: cond(): 'false_fn' argument required
                    # ==> Happens, when ITE is coming up. Ignoring for now.
                    # TypeError('Node.is_ExprCdPair() takes 1 positional argument but 2 were given')
                    # ==>
                    if str(ex) == "Cannot convert complex to float":
                        pass
                    else:
                        raise Exception  #  print(f'Typeerror, but why? {ex}')
                except AttributeError as ex:
                    print(f'("Okay", if sympy.im in expr) {ex}')
                #     # raise AttributeError(f'Probably sympy.im in expr {ex}')
                #     # AttributeError: Probably sympy.im in expr 'int' object
                #     raise AttributeError(f'AttributeError: {ex}')
                #     # f'\n\tint object has no attribute get_nodes_at_depth has no attribute get_nodes_at_depth
                #     # f'\n\tProbably sympy.im in expr has no attribute get_nodes_at_depth
                #     # print(f'Probably sympy.im in expr {ex}')
                #     #  AttributeError: 'Xor' object has no attribute '_eval_as_set'
                #     #    |--AttributeError: Probably sympy.im in expr 'Xor' object has no attribute '_eval_as_set'
                except KeyError as ex:
                    # KeyError(re) -> okay?, real part implies complex numbers, ignoring is okay
                    print(f'Keyerror, (probably sympy.lambdify expression not evaluable): {ex}')
                except RecursionError as ex:
                    print(f'RecursionError (probably Piecewise/relational combination?): {ex}')
                except NotImplementedError as nie:
                    print_caution(f'Notimplemented? {nie}')

        return loop

    def tree_to_candidate(self, evotree: Node, tag=None, raise_if_useless=True):
        """the "fixed" node information is not relevant"""

        evotree.force_input_node(self.evolve)
        evotree.repair_depth()

        tree_id = evotree.get_lut_id()

        if tree_id in self.lut_sym:
            sy_expr = self.lut_sym[tree_id]
        else:
            sy_expr = evotree.get_sympy_expr()
            sym_check(sy_expr)  # sfeh:discuss save bad trees in LUT aswell? Different LUT for bad trees?
            self.lut_sym[tree_id] = sy_expr

        if tree_id in self.lut_parsim:
            parsimony = self.lut_parsim[tree_id]
        else:
            parsimony = eval_parsimony(evotree, self.evolve.complexity_metric, origin_tree=self.evolve.origin_tree)
            self.lut_parsim[tree_id] = parsimony
            if raise_if_useless and parsimony > self.evolve.nodes_max:  # sfeh:open
                raise ValueError(f'Tree too complex: {parsimony} > {self.evolve.nodes_max}')

        if sy_expr in self.lut_fitness:
            fitness = self.lut_fitness[sy_expr]
        else:
            # sfeh:discuss sympy real=True might allow imaginary results
            # t0 = time.perf_counter()
            try:
                pairwise_results = eval_regression_sym_experimental(sy_expr, self.df_train)
            except Exception as TODO:
                pairwise_results = eval_regression_sym_experimental(sy_expr, self.df_train)

            fitness = regression_error(pairwise_results, self.df_train['action'])

            # t1 = time.perf_counter()
            # fitness2 = self.kernel.eval_tf(sy_expr)['mean_error']
            # t2 = time.perf_counter()
            # print(f'asd {t1-t0:4.4f} {t2-t1:4.4f} ({(t1-t0)-(t2-t1):4.2f}) {fitness:4.2f} {fitness2:4.2f}')
            # if DEBUG_DUMMY or fitness != self.kernel.eval_sym_experimental(sy_expr):
            #     print(f'FAILED: {fitness} vs. {self.kernel.eval_sym_experimental(sy_expr)}')

            self.lut_fitness[sy_expr] = fitness  # sfeh:discuss: lut update in finalize_tree_get_meta()?

        # return fitness, parsimony, sy_expr

        candidate = Candidate(evotree, fitness=fitness, parsimony=parsimony, tag=tag)
        return candidate

    def evoloop_monitoring_plots(self):
        """
        Create all run-related analysis plots in the root directory
        """
        try:
            plot_performance(self.monitor_df, self.name, self.rootdir / 'monitoring.png')
            plot_paretofront(self.paretofront, self.rootdir, self.name, self.evolve.nodes_max)
        except Exception as ex:
            printpl("e", f'Could not create plots: {ex}\n')

    def backup_save(self, opt_path_backup=None):
        """
        Load/safe backup of a run
        """

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
        run_backup_data = {}, self.gen_id, self.pop_genepool, self.paretofront, self.monitor_df
        path_backup = path_make_dir(path_backup)
        pickle_dump(path_backup, run_backup_data)
        # sfeh:debug

    def backup_load(self, opt_path_backup=None):
        """Load/safe backup of a run"""

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        if Path.is_file(path_backup):
            printpl('g', f'Loading data from backup-file {path_backup}')
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)
            except NotImplementedError as ex:
                raise Exception(f'NotImplementedError: {ex}')
            except EOFError as ex:
                raise Exception(f'EOFError: \n{ex}')

            help_dict, self.gen_id, self.pop_genepool, self.paretofront, self.monitor_df = run_data
            self.backup_save(opt_path_backup=self.rootdir / f'backup/backup-{self.gen_id}.pkl')  # sfeh:dis date?
            printpl('g', f'Successfully loaded backup file. Generation: {self.gen_id}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}')  # sfeh:beautify occurs 2x

    def analyze_generation(self):
        gen_time = time.perf_counter() - self.time_genstart
        tmp_dict = pop_analyze(self.pop_genepool, gen_time, self.gens_since_last_pareto)
        self.monitor_df.loc[self.gen_id] = tmp_dict
        printpl('gg',
                f"Created {len(self.pop_genepool)}/{self.pop_max} ({tmp_dict['pop_unique']} unique) in generation {self.gen_id}. "
                f"Trees in LUT: {len(self.lut_fitness)} Gen took {gen_time:4.2f}s")

        printpl('ggg', f'--- Gen {self.gen_id} took: {time.perf_counter() - self.time_genstart:4.2f}. ---')

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
        try:
            if self.gens_since_last_pareto > 100:  # .iloc[-1] > 100:  # sfeh discussion
                print('SFEH This condition made your program exit!')
                return True
            else:
                return False
        except Exception:
            return False

    # def plot_evolve_performance(self):
    #     """
    #     Plots for each tag in the evolution list
    #     (too much, I guess)
    #     sfeh: this should be saved within the trees. Everything else is a waste of memory!
    #     sfeh:open
    #     """
    #     try:
    #         with plt.rc_context(rc=pyplot_rc_tex):
    #             fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 9),
    #                                     sharex='all')  # , gridspec_kw={'height_ratios': [1,1,1]}
    #             fig.tight_layout()
    #             for tag in self.evolve_tags:
    #                 # ['fitness_train', 'parsimony', 'lentree', 'evolve_num', 'count']
    #                 axs[0].plot(self.monitor_df[tag]['fitness_train'], label=f'{tag}')
    #                 axs[1].plot(self.monitor_evol[tag]['parsimony'], label=f'{tag}')
    #                 axs[2].plot((self.monitor_evol[tag]['lentree'] / self.monitor_evol[tag]['evolve_num']),
    #                             label=f'{tag}')
    #
    #             plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
    #             path = self.rootdir / f'monitoring_evolutions.pdf'
    #             fig.savefig(path)
    #             self.printpl('f', f"monitoring_evolutions (pdf): {path}")  # .as_posix()
    #
    #     except Exception as ex:
    #         print_e(f'plot_evolution_analysis failed because of: {ex}')


def pop_analyze(popul, gen_time, gens_since_last_pareto):
    """Analysing the population (in each generation)
    - amount of trees
    - fittest tree
    - average fitness
    - average tree parsimony"""

    if len(popul) == 0:
        raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

    pop_fitness = [tree.get_fitness() for tree in popul]
    pop_parsim = [tree.get_parsimony() for tree in popul]
    pop_treelen = [len(candidate_tree.tree) for candidate_tree in popul]
    pop_fitness_best = np.min(pop_fitness)
    pop_unique = len(set([str(x.tree) for x in popul]))  # sfeh:analyze this?

    # sfeh:idea add the amount of actually new trees (compare with the LUT tree_ids)
    result = {'pop_len': len(popul),
              'pop_unique': pop_unique,
              'fit_avg': np.average(pop_fitness),
              'fit_var': np.std(pop_fitness),
              'fit_best': pop_fitness_best,
              'parsim_avg': np.average(pop_parsim),
              'parsim_var': np.std(pop_parsim),
              'parsim_best': np.min(pop_treelen),
              'time': gen_time,
              'gens_since_last_pareto': gens_since_last_pareto
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


if __name__ == '__main__':
    """
    Alpha tests
    """
    # t1 = tb.invent_core_depth(float, 3, p_term=0.5)
    # tree2 = tb.evolve_mutate_point(t1)
    # t1 = tb.invent_core_depth(float, 3, p_term=0.1)
    # t2 = tb.invent_core_depth(float, 3, p_term=0.1)
    # t1, t2 = tb.evolve_crossover(t1, t2)
    # for _ in range(5):
    #     print('x.D', t1, '===', t2)
    #     t1, t2 = tb.evolve_crossover(t1, t2)
    #     print('x~D', t1, '===', t2)

    # nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    nstr = '["+:fix",["-:fix",["Ifte",["True"],["sin",["2"]],["/",["2.043"],["4"]]],["cartVel"]],["-1.3"]]'
