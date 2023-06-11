import itertools
import random
from dataclasses import dataclass

import sympy.core.numbers

from plagih.plagih_tree import *

# For conversion from sympy into node
from plagih.util import FLOAT_PRECISION

d_sym2node = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul,
              sympy.Xor: Xor, sympy.Not: Not, sympy.And: And, sympy.Or: Or,
              sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.StrictGreaterThan: Gt,
              sympy.GreaterThan: Ge, sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos,
              sympy.asin: Asin, sympy.atan: Atan, sympy.tanh: tanh, sympy.sinh: Sinh, sympy.cosh: Cosh,
              sympy.Min: Min, sympy.Max: Max, sympy.ITE: ITE, sympy.exp: exp}
d_op2chain = {Add: AddChain, Mul: MulChain, Min: MinChain, Max: MaxChain, And: AndChain, Or: OrChain,
              Piecewise: Piecewise}  # todo Piecewise
d_sym2node_chain = {sympy.Add: AddChain, sympy.Mul: MulChain, sympy.Min: MinChain, sympy.Max: MaxChain,
                    sympy.And: AndChain, sympy.Or: OrChain, sympy.Piecewise: Piecewise}  # todo Piecewise
# , sympy.Equality: Eq
# sfeh:open = {sympy.Unequality: Ne, sympy.Equality: Eq}
sym_assumption_nodes = (sympy.re,)


@dataclass
class Node:
    """Recursively holds the nodes of a tree"""

    def __init__(self, label, childs: iter, depth=None, is_fix=False, is_chain=False):
        self.label = label
        self.childs = childs[:]

        self.is_fix = is_fix
        self.depth = depth
        # self.is_chain = is_chain  # sfeh:xxx check update required

    def is_chain(self):
        """is the node in chain mode?
        -> when there are more childs than input-xtypes
        ...if there are less. its weird, maybe node in construction?"""
        if len(self.label.xtype[0]) < len(self.childs):
            return False

    def str_as_list(self):
        try:
            label_str = self.label.__name__  # sfeh: can str(label) work? -> str with args recursively?
        except AttributeError as ex:
            # todo debug me
            label_str = self.label  # because Terminals are obj -> 'Symbol' obj has no attr __name__

        if self.childs:
            if issubclass(self.label, Operator):
                childstr = ', '.join([cc.str_as_list() for cc in self.childs])
                label_str = f'{label_str}, {childstr}'
            else:
                try:
                    if issubclass(self.label, Number):
                        label_str = f'{self.childs[0]:.3g}'  # '.5g'->5 decimals, trailing zeros, but rare (ugly) "E+04"
                    else:
                        label_str = f'{self.childs[0]}'
                except TypeError as ex:
                    label_str = str(self.childs[0].evalf())
                    # sfeh:open int, non-floats are handeled badly
                except Exception as ex:
                    print(f'SUCCESS sfeh:debug, delete? KEEP? {ex}')

        return f"[{label_str}]"

    def get_id(self):
        try:
            label_str = self.label.__name__  # sfeh: can str(label) work? -> str with args recursively?
        except AttributeError as ex:
            # todo debug me
            label_str = self.label  # because Terminals are obj -> 'Symbol' obj has no attr __name__

        if self.childs:
            if issubclass(self.label, Operator):
                childstr = ', '.join([cc.str_as_list() for cc in self.childs])
                label_str = f'{label_str}, {childstr}'
            else:
                try:
                    label_str = f'{self.childs[0]}'
                except TypeError as ex:
                    label_str = str(self.childs[0].evalf())
                    # sfeh:open int, non-floats are handeled badly
                except Exception as ex:
                    print(f'SUCCESS sfeh:debug, delete? KEEP? {ex}')

        return f"[{label_str}]"

    def str_as_expr(self):
        s = self.get_sympy_expr()
        return s

    def __str__(self):
        return str(self.str_as_expr())

    def get_sympy_expr(self):
        if issubclass(self.label, Operator):
            _sym = self.label.symfun
            _cs = [cc.get_sympy_expr() for cc in self.childs]
            return _sym(*_cs)
            # try:
            #     return _sym(*_cs)
            # except RecursionError as ex:
            #     print(f'sfeh:RecursionError, maybe Piecewise?: {self}, {ex}')
            #     raise RecursionError
            # except Exception as ex:
            #     print(f'sfeh:XXX this still occurs. {ex}')
            #     # The argument '-2.05444 + I*pi' is not comparable. <- that should be okay, just raise
            #     # <lambda>() missing 1 required positional argument: 'b'
            #     #   -> Probably in Sub-class lambda-function
            #     raise ex
        elif issubclass(self.label, Terminal):
            _sym = self.label.get_sym()  # _sym = self.label.symfun
            _cs = self.childs[0]
            return _sym(_cs)
        print('asddsa', len(self.childs), type(self.label), issubclass(self.label, Operator), issubclass(self.label, Terminal))
        raise NotImplementedError(f'get_sympy_expr no match for {self}, {type(self.label)}')

    def get_expr_raw(self):
        if issubclass(self.label, Operator):
            expr = [cc.get_expr_raw() for cc in self.childs]
            expr = ', '.join(expr)
            expr = f'{self.label.__name__}({expr})'
        elif issubclass(self.label, Terminal):
            expr = f'{self.childs[0]}'
        return expr

    def get_tf_expr(self):
        if self.childs and issubclass(self.label, Operator):
            _tf = self.label.tflow
            _cs = [cc.get_tf_expr() for cc in self.childs]
            return _tf(_cs)
        elif self.childs and issubclass(self.label, Terminal):
            _tf = self.label.tflow
            _cs = self.childs[0]
            return _tf(_cs)
        raise NotImplementedError(f'get_tf_expr no match for {self}, {type(self.label)}')

    def __repr__(self):
        """Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to str(), but adds the following information:
        - ":fix", when nodes are fixed"""
        label_str = self.label

        if self.is_fix:
            label_str += ':fix'

        if self.childs:
            childstr = ', '.join([repr(cc) for cc in self.childs])
            label_str = f"{label_str}, {childstr}"
        return f"[{label_str}]"

    def len_nodecount_raw(self):
        """counting the amount of nodes recursively"""
        if issubclass(self.label, Terminal):
            return 1  # childs can currently be floats
        else:
            return 1 + sum([cc.len_nodecount_raw() for cc in self.childs])

    def len_nodecount_fair(self):
        """counting the amount of nodes, but
            - ignoring "Usub!
        """
        if issubclass(self.label, Terminal):
            return 1
        elif issubclass(self.label, Usub):
            return sum([cc.len_nodecount_fair() for cc in self.childs])
        else:
            return 1 + sum([cc.len_nodecount_fair() for cc in self.childs])

    def __len__(self):
        return self.len_nodecount_fair()

    def get_label(self):
        return self.label

    def get_value(self):
        """sfeh: only usable, when terminal node?"""
        if self.is_term():
            return self.childs[0]

    def get_arity(self):
        return len(self.label.get_child_xts())

    def get_xtype(self):
        return self.label.xtype

    def get_xtype_childs(self):
        return self.label.xtype[0]

    def get_xtype_self(self):
        return self.label.xtype[1]

    def set_label(self, label: 'Label'):
        """all other values are automatically set by assigning the respected node"""
        self.label = label

    def set_childs(self, child_list):
        if isinstance(child_list, (list, tuple)):
            self.childs = child_list
        else:
            raise TypeError(f'childs must be set as list, not {type(child_list)}: {child_list}')

    def update_fixed_nodes(self, origin: 'Node'):
        """Updating the fixed nodes in a tree where they were lost for some reason.
        This should never be the case! But it happened during development of recreating a tree from expression.
        This might also be useful in tree checks"""
        if origin.is_fix:
            if str(self.label) != str(origin.label):
                raise
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.update_fixed_nodes(origin.childs[ii])

    def get_nodes_to_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """sum_layers=False, get_closest=True, return_all_layers=False"""
        child_results = []
        if self.depth < goal_depth:
            child_results = sum(
                [child.get_nodes_to_depth(goal_depth, only_mutable=only_mutable, force_depth=get_closest_depth) for
                 child in self.childs], [])

        if only_mutable and self.is_fix or get_closest_depth and self.depth != goal_depth:
            my_result = []
        else:
            my_result = [self]

        return my_result + child_results

    def get_all_nodes(self):
        if len(self.childs) == 0:
            return [self]
        else:
            return [self] + [cc.get_all_nodes() for cc in self.childs]

    def get_nodes_at_depth(self, goal_depth, allow_fixed=False, earliest_nonfix=False):
        """Returns a list with mutable ids which are *goal_depth* layers away from non-modifiable nodes
        last_leaves: if you want so save all leave nodes aswell
        sum_layers=False, get_closest=True, return_all_layers=False"""
        nodes = []
        if (not self.is_fix or allow_fixed) and (
                self.depth == goal_depth or (self.depth > goal_depth and earliest_nonfix)):
            return [self]
        elif self.depth <= goal_depth or earliest_nonfix:
            for cc in self.childs:
                res = cc.get_nodes_at_depth(goal_depth, allow_fixed=allow_fixed, earliest_nonfix=earliest_nonfix)
                nodes.extend(res)
            # nodes.extend(list(itertools.chain(
            #     *[cc.get_nodes_at_depth(goal_depth, allow_fixed=allow_fixed, expand_depth=expand_depth) for cc in
            #       self.childs])))
            return nodes
        else:
            return []

    def eval_apted_notation(self):
        """Calculating the TED requires this (weird) representation"""
        return f"{{{self.get_label()}{''.join([cc.eval_apted_notation() for cc in self.childs])}}}"

    def get_max_depth(self, depth=0):
        """Go through all nodes, save depth"""
        if len(self.childs) == 0:
            return depth
        else:
            return max(cc.get_max_depth(depth=depth + 1) for cc in self.childs)

    def is_regular(self):
        # Nodes that are notchained-nodes
        return issubclass(self.label, ArityNode)

    def is_chainop(self):
        return issubclass(self.label, ChainOp)

    def is_operator(self):
        # todo allow_chain check all
        return issubclass(self.label, Operator)

    def is_term(self):
        # sfeh:discuss is_atom, rename all to atom?
        return issubclass(self.label, Terminal)

    def repair_depth(self, depth=0):
        """
        aka set_depth recursively for all nodes in a branch
        mainly used in branch
        The depth is written inevery node (for whatever reason), and instead of having to propagate
        the depth through every crossover/branch mutation function, instead, we call it when replacing nodes
        """
        self.depth = depth
        if self.is_operator():
            for cc in self.childs:
                cc.repair_depth(depth=depth + 1)
        return

    def set_new_node(self, nd_new: 'Node'):
        """Replacing oneself with another node"""
        self.set_label(nd_new.label)  # sfeh remove childs, is_fix...
        self.childs = nd_new.childs  # sfeh maybe must be updated recursively
        self.repair_depth(self.depth)  # Especially required for crossover or branchesnd_new
        # sfeh check fixed or if type matches?

    def replace_with(self, label, childs):
        if label is not None:
            self.set_label(label)
        if childs is not None:
            self.set_childs(childs)

        # self.repair_depth(self.depth)  # todo, failed inside TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
        # no checks

    def eval_mutable_nodes(self, xt_match=None, ignore_first=False, allow_chain=False) -> ['Node']:
        """return all nodes that are mutable, aka suite for point- or branchmutation
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!"""

        # -> Check, if this node should be added
        if self.is_fix:
            node_list = []
        elif ignore_first:
            node_list = []  # ignore_first is automatically set to false during recursion
        elif allow_chain or self.is_chain():
            node_list = []
        else:
            if xt_match is None or xt_match == self.get_xtype_self():
                node_list = [self]
            else:
                node_list = []

        # -> recursively add the other nodes
        if self.is_regular() and self.is_operator():  # sfeh:chain-operators discuss
            for cc in self.childs:
                node_list.extend(cc.eval_mutable_nodes(xt_match=xt_match, ignore_first=False, allow_chain=allow_chain))

        return node_list

    def evolve_mutate_filter_gauss(self):
        """Recursively filter the nodes in the branch of fintree
        sfeh:   random filter all terminal nodes /
                single node /
                nodes in a branch /
                random nodes in a branch /
                intelligent filtering
        """
        if self.is_term():
            if issubclass(self.label, Number):
                self.childs[0] = round(random.gauss(self.childs[0], 0.1), FLOAT_PRECISION)  # sfeh: -> no symbols -> userspecific
        else:
            for cc in self.childs:
                cc.evolve_mutate_filter_gauss()
        return

    def mutate_self_filter_index(self):

        # new_index = int(max(min(round(random.gauss(self.time_index, 1)), self.index_minmax[1]), 0))
        # self.time_index = new_index
        # self.name = f'{self.fam}_{new_index}'
        pass

    def tree_node_grouping(self):
        """
        If possible, this groups nodes to a simpler expression (if possible).
        E.g. a ** 2 -> square(a)
             a + b + c -> sum(a, b, c)  (chaining)

        # sfeh:idea Heavyside function. input a val, input b threshold
        """
        if self.is_term():  # runtime
            return

        for cc in self.childs:
            # todo: check, if this actually alters the content
            cc.tree_node_grouping()

        if self.label in (Pow, Powrounded):
            n_exp = self.childs[1].childs[0]  # must exist
            if n_exp == -1:
                self.replace_with(DivFraction, childs=[self.childs[0]])
            elif n_exp == 2:
                self.replace_with(Square, childs=[self.childs[0]])
            elif n_exp == 0.5:
                self.replace_with(Sqrt, childs=[self.childs[0]])
            elif n_exp == 0:
                self.replace_with(Number, childs=[1])
            elif self.childs[1].label == Round:
                self.replace_with(Powrounded, childs=[self.childs[0], n_exp])
            # sfeh:discuss, powrounded here? not clear
            # elif self.childs[1].label == Number and n_exp % 1 == 0:
            #     self.set_label(Powrounded)

            # todo sub, usub replace
            # todo tree prints

        elif self.label == Mul:
            if not self.is_chain():  # div only for
                if self.childs[0].label == DivFraction:
                    self.replace_with(Div, childs=[self.childs[1], self.childs[0].childs[0]])
                elif self.childs[1].label == DivFraction:
                    self.replace_with(Div, childs=[self.childs[0], self.childs[1].childs[0]])

                elif self.childs[0].label == Number:
                    mul1 = self.childs[0].childs[0]
                    if mul1 == -1:  # aka sympy.S.NegativeOne -1, was -1 before
                        self.replace_with(Usub, childs=[self.childs[1]])  # todo Usub ONLY option, ignore usub tree len
                    elif 0 < mul1 < 1:
                        self.replace_with(Div, childs=[self.childs[1], Node(Number, childs=[1 / mul1])])  # todo keep "div" as option

                elif self.childs[1].label == Number:
                    mul1 = self.childs[1].childs[0]
                    if mul1 == -1:  # aka sympy.S.NegativeOne -1, was -1 before
                        self.replace_with(Usub, childs=[self.childs[0]])  # todo Usub ONLY option, ignore usub tree len
                    elif 0 < mul1 < 1:
                        self.replace_with(Div, childs=[self.childs[0], 1 / mul1])
                        todo_div_by_test = 1 / mul1  # todo keep "div" as option
                        print(f'asd todo {todo_div_by_test}')


def sympy_to_tree(s_expr: sympy.Basic, allow_chain=False) -> Node:
    """
    Important: start with the most specific rule
    # sfeh:discuss computational improvement when option to ignore args? do "raises" in args save time?
    # check is expr is an accepted operator, otherwise reconstruction probably fails"""
    if isinstance(s_expr, bool):
        return Node(Boolean, [s_expr])

    elif isinstance(s_expr, sympy.logic.boolalg.BooleanAtom):
        # s_expr = True if isinstance(s_expr, (bool, sympy.logic.boolalg.BooleanTrue)) else False
        s_expr = True if isinstance(s_expr, (bool, sympy.logic.boolalg.BooleanTrue)) else False  # todo todotodo
        return Node(Boolean, [s_expr])

    # the following two lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    elif s_expr.is_Atom:
        if s_expr.is_Symbol:
            # return Node(Symbol, [str(s_expr)])  # sfeh str VERY important!! "Symbol type input" is not accepted
            return Node(Symbol, [s_expr])  # todo todotodo str VERY important!! "Symbol type input" is not accepted
        else:
            expr_eval = s_expr  # todo check if this can be worked with .evalf(FLOAT_PRECISION)  # 15 digits
            if s_expr.is_Boolean:
                # return Node(Boolean, [bool(expr_eval)])
                return Node(Boolean, [expr_eval])  # todo todotodo
            elif s_expr.is_number:  # is_float does not match int
                # return Node(Float, [round(float(expr_eval), FLOAT_PRECISION)])
                return Node(Number, [expr_eval])  # todo todotodo check
                # "TypeError: Cannot convert complex to float" -> ignore the whole expression, let it fail
            else:
                raise NotImplementedError(f'What happened here? {s_expr}')

    else:  # **Operators**

        cc_nodes = []
        for arg in s_expr.args:
            cc_nodes.append(sympy_to_tree(arg, allow_chain=allow_chain))

        if allow_chain:
            op = d_sym2node[s_expr]  # if issubclass(clss, ChainableOp):
            return Node(op, childs=cc_nodes)

        if isinstance(s_expr, sympy.functions.elementary.piecewise.ExprCondPair):
            return Node(ExprCondPair, cc_nodes)

        elif isinstance(s_expr, sympy.Piecewise):
            # "Chained" version is handled before
            reversed_pairs = list(s_expr.args[::-1])  # tuples to list, reversed: tuple must be nested the deepest
            reversed_pairs = [[sympy_to_tree(xx, allow_chain=allow_chain) for xx in list(i)] for i in
                              reversed_pairs]  # noqa
            otherwise = reversed_pairs[0][0]  # the last "True" condition
            for pairs in reversed_pairs[1:]:
                otherwise = Node(Ifte, [pairs[1], pairs[0], otherwise])
            return otherwise

        # todo include Usub, ignore usub in tree len()

        elif isinstance(s_expr, RoundDummy):
            return Node(Round, [cc_nodes[0]])

        elif isinstance(s_expr, Mul):
            if s_expr.args[0].is_Rational:
                div_by = 1 / s_expr
                print(f'TODO TODO div by {div_by}')

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
            # todo reintroduce piecewise? discuss: ignore trees which have real/complex numbers
            # return sympy_to_tree(expr.args[0], allow_chain=allow_chain)
            return cc_nodes

    # sfeh:discuss
    # NotImplementedError: Expr missing: ITE(p > 13, tan(p - v) >= 2.578643, tan(p - v) >= 1)
    # this should not have occured, because it evaluates to bool, not to float
    raise NotImplementedError(f'Expr missing: {s_expr}')


if __name__ == '__main__':

    for x in d_sym2node.keys():
        lel = 4.5
        print(x, isinstance(lel, x))
