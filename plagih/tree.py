import itertools
import random
from dataclasses import dataclass

import sympy.core.numbers

from plagih.plagih_tree import *

# For conversion from sympy into node
from plagih.util import FLOAT_PRECISION

sym_nodes = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul,
             sympy.Xor: Xor, sympy.Not: Not, sympy.And: And, sympy.Or: Or,
             sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.StrictGreaterThan: Gt,
             sympy.GreaterThan: Ge, sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos,
             sympy.asin: Asin, sympy.atan: Atan, sympy.tanh: tanh, sympy.sinh: Sinh, sympy.cosh: Cosh,
             sympy.Min: Min, sympy.Max: Max, sympy.ITE: ITE, sympy.exp: exp}
sym_nodes_chain = {sympy.Add: AddChain, sympy.Mul: MulChain, sympy.Min: MinChain, sympy.Max: MaxChain,
                   sympy.And: AndChain, sympy.Or: OrChain, sympy.Piecewise: Piecewise}  # todo Piecewise
# , sympy.Equality: Eq
# sfeh:open = {sympy.Unequality: Ne, sympy.Equality: Eq}
sym_assumption_nodes = (sympy.re,)


@dataclass
class Node:
    """Recursively holds the nodes of a tree"""

    def __init__(self, label, childs, depth=None, is_fix=False, is_chain=False):
        self.label = label
        self.childs = childs

        self.is_fix = is_fix
        self.depth = depth
        self.is_chain = is_chain  # sfeh:xxx check update required

    def __str__(self):
        try:
            label_str = self.label.__name__  # sfeh: can str(label) work? -> str with args recursively?
        except AttributeError as ex:
            label_str = self.label  # because Terminals are obj -> 'Symbol' obj has no attr __name__

        if self.childs:
            if issubclass(self.label, Operator):
                childstr = ', '.join([str(cc) for cc in self.childs])
                label_str = f'{label_str}, {childstr}'
            else:
                try:
                    if issubclass(self.label, Float):
                        label_str = f'{self.childs[0]:.2g}'  # 2 decimals, no trailing zeros, but rare (ugly) "E+04"
                    else:
                        label_str = f'{self.childs[0]}'
                except Exception as ex:
                    print(f'sfeh:debug, delete if no occurs. "IndexError: invalid index to scalar variable"? {ex}')

        return f"[{label_str}]"

    def get_sympy_expr(self):
        if self.childs and issubclass(self.label, Operator):
            _sym = self.label.symfun
            _cs = [cc.get_sympy_expr() for cc in self.childs]
            # if self.label == Ifte:
            try:
                return _sym(*_cs)
            except RecursionError as ex:
                print(f'sfeh:RecursionError, maybe Piecewise?: {self}, {ex}')
                raise RecursionError
            # except ValueError as ex:
            #     raise ex
            # except Exception as ex:
            #     print(f'sfeh:XXX this still occurs. {ex}')
            #     # The argument '-2.05444 + I*pi' is not comparable.
            #     # <lambda>() missing 1 required positional argument: 'b'
            #     #   -> Probably in Sub-class lambda-function
            #     raise ex
        elif self.childs and issubclass(self.label, Terminal):
            _sym = self.label.get_sym()  # _sym = self.label.symfun
            _cs = self.childs[0]
            return _sym(_cs)
        print(len(self.childs), type(self.label), issubclass(self.label, Operator),
              issubclass(self.label, Terminal))
        raise NotImplementedError(f'get_sympy_expr no match for {self}, {type(self.label)}')

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

    def eval_str(self):
        return self.get_label()  # sfeh open

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

    def __len__(self):
        """counting the amount of nodes recursively"""
        if issubclass(self.label, Terminal):
            return 1  # childs can currently be floats
        else:
            return 1 + sum([len(cc) for cc in self.childs])

    def get_label(self):
        return self.label

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
        return issubclass(self.label, RegularNode)

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

    def set_new_node(self, new_node: 'Node'):
        self.set_label(new_node.label)  # sfeh remove childs, is_fix...
        self.childs = new_node.childs  # sfeh maybe must be updated recursively
        self.repair_depth(self.depth)  # Especially required for crossover or branches

    def eval_mutable_nodes(self, match_xt=None, ignore_first=False, ignore_chain=False) -> ['Node']:  # sfeh is this correct?
        """return all nodes that are mutable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!"""
        node_list = []
        if not any([self.is_fix, ignore_first, ignore_chain and self.is_chain]) and match_xt is None or match_xt == self.get_xtype_self():
            node_list.append(self)  # must be reference

        if self.is_regular():

            if self.is_operator():  # sfeh:chain-operators discuss
                for cc in self.childs:
                    node_list.extend(cc.eval_mutable_nodes(match_xt=match_xt, ignore_first=False, ignore_chain=ignore_chain))
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
            if issubclass(self.label, Float):
                self.childs[0] = round(random.gauss(self.childs[0], 0.1), FLOAT_PRECISION)   # sfeh: ->no symbols ->userspecific
        else:
            for cc in self.childs:
                cc.evolve_mutate_filter_gauss()
        return

    def mutate_self_filter_index(self):

        # new_index = int(max(min(round(random.gauss(self.time_index, 1)), self.index_minmax[1]), 0))
        # self.time_index = new_index
        # self.name = f'{self.fam}_{new_index}'
        pass


def s_tree(branch: Node, allow_chain=False) -> Node:
    """
    Check, if node can be simplified anyhow. 
    :param branch:
    :param allow_chain:
    :return:
    """
    if branch.is_term():
        return branch
    else:

        cc_nodes = [s_tree(cc, allow_chain=allow_chain) for cc in branch.childs]

        if allow_chain:
            if isinstance(nnnn, tuple(sym_nodes_chain)):
                clss = sym_nodes_chain[type(nnnn)]
                return Node(clss, cc_nodes, is_chain=True)

        if isinstance(nnnn, sympy.functions.elementary.piecewise.ExprCondPair):
            return Node(ExprCondPair, cc_nodes)

        elif isinstance(nnnn, sympy.Piecewise):
            # todo todotodo
            if allow_chain:
                return Node(Piecewise, cc_nodes)
            else:
                reversed_pairs = list(nnnn.args[::-1])  # tuples to list, reversed: tuple must be nested the deepest
                reversed_pairs = [[s_tree(xx, allow_chain=allow_chain) for xx in list(i)] for i in reversed_pairs]  # noqa
                otherwise = reversed_pairs[0][0]  # the last "True" condition
                for pairs in reversed_pairs[1:]:
                    otherwise = Node(Ifte, [pairs[1], pairs[0], otherwise])
                return otherwise

                # cc_nodes = cc_nodes[::-1]  # reversed: tuple, which is nested the deepest
                # otherwise = cc_nodes[0]  # the last "True" condition
                # # TypeError: cannot unpack non-iterable Node object
                # for ec in cc_nodes[1:]:
                #     otherwise = Node(Ifte, [ec[0], ec[1], otherwise])
                #
                # return otherwise

        # todo include Usub, ignore usub in tree len()
        # elif isinstance(expr, sympy.Mul) and expr.args[0] == -1 and expr.args[1].is_Atom:
        #     node = sympy_to_tree(expr.args[1], allow_chain=allow_chain)
        #
        #     return node

        elif isinstance(nnnn, sympy.Pow):
            # todo match nodes to simplified nodes in a different function
            if nnnn.args[1] == -1:  # sympy.S.NegativeOne= sfeh:check if assumptions are available
                _r = InverseFraction
                cc_nodes.pop(1)  # second arg was identified and must now be ignored
            elif nnnn.args[1] == 2:
                _r = Square
                cc_nodes.pop(1)
            elif nnnn.args[1] == sympy.S.Half:
                _r = Sqrt
                cc_nodes.pop(1)
            elif type(nnnn.args[1]) == RoundDummy:
                # todo check this BEFORE possibly the round operator is matched.
                _r = Powrounded
                cc_nodes[1] = cc_nodes[1].childs[0]
            else:
                _r = Pow
            return Node(_r, cc_nodes)

        elif isinstance(nnnn, RoundDummy):
            return Node(Round, [cc_nodes[0]])

        elif isinstance(nnnn, tuple(sym_nodes)):

            clss = sym_nodes[type(nnnn)]

            if isinstance(nnnn, sympy.Mul):
                for ii, cn in enumerate(cc_nodes):
                    if cn.label == InverseFraction:
                        divisor_node = cc_nodes.pop(ii)
                        divisor_node = divisor_node.childs[0]
                        dividend_node = cc_nodes[0]
                        return Node(Div, [dividend_node, divisor_node])

            if len(nnnn.args) > len(clss.get_child_xts()):
                if issubclass(clss, ChainableOp):
                    if allow_chain:
                        # todo actually do this at the very top
                        return Node(clss, cc_nodes, is_chain=True)
                    else:
                        _cc = cc_nodes[0]
                        for _c2 in cc_nodes[1:]:
                            _cc = Node(clss, [_cc, _c2])
                        return _cc
                else:
                    raise TypeError(f"{clss} takes exactly {len(clss.get_child_xts())} arguments ({len(nnnn.args)} given)")
            else:
                return Node(clss, cc_nodes)

        elif isinstance(nnnn, sympy.re):
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
    raise NotImplementedError(f'Expr missing: {nnnn}')


def sympy_to_tree(expr, allow_chain=False) -> Node:
    """Important: start with the most specific rule
    todo: check is expr is an accepted operator, otherwise reconstruction probably fails"""
    if isinstance(expr, bool):
        return Node(Boolean, [expr])

    elif isinstance(expr, sympy.logic.boolalg.BooleanAtom):
        expr = True if isinstance(expr, (bool, sympy.logic.boolalg.BooleanTrue)) else False
        return Node(Boolean, [expr])

    # the following two lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    elif expr.is_Atom:
        if expr.is_Symbol:
            # _r = Symbol  # sfeh str VERY important!! "Symbol type input" is not accepted
            # return Nested(Symbol(str(expr)), [])
            return Node(Symbol, [str(expr)])
        else:
            expr_eval = expr.evalf()  # standard 15 digits, sfeh prec=FLOAT_PRECISION?
            if expr.is_Boolean:
                # return Nested(Boolean(bool(expr_eval)), [])
                return Node(Boolean, [bool(expr_eval)])
            elif expr.is_number:  # is_float does not match int
                # return Nested(Float(float(expr_eval)), [])  # sfeh round
                return Node(Float, [round(float(expr_eval), FLOAT_PRECISION)])  # sfeh round
                # "TypeError: Cannot convert complex to float" -> ignore the whole expression, let it fail
            else:
                raise NotImplementedError(f'What happened here? {expr}')

    else:
        # try:
        #     if isinstance(expr, sympy.Mul) and   # sfeh check div here?:

        cc_nodes = [sympy_to_tree(ar, allow_chain=allow_chain) for ar in expr.args]
        # sfeh:discuss computational improvement when ability to ignore args? do "raises" in args save time?

        if allow_chain:
            if isinstance(expr, tuple(sym_nodes_chain)):
                clss = sym_nodes_chain[type(expr)]
                return Node(clss, cc_nodes, is_chain=True)

        if isinstance(expr, sympy.functions.elementary.piecewise.ExprCondPair):
            return Node(ExprCondPair, cc_nodes)

        elif isinstance(expr, sympy.Piecewise):
            # todo todotodo
            if allow_chain:
                return Node(Piecewise, cc_nodes)
            else:
                reversed_pairs = list(expr.args[::-1])  # tuples to list, reversed: tuple must be nested the deepest
                reversed_pairs = [[sympy_to_tree(xx, allow_chain=allow_chain) for xx in list(i)] for i in reversed_pairs]  # noqa
                otherwise = reversed_pairs[0][0]  # the last "True" condition
                for pairs in reversed_pairs[1:]:
                    otherwise = Node(Ifte, [pairs[1], pairs[0], otherwise])
                return otherwise

                # cc_nodes = cc_nodes[::-1]  # reversed: tuple, which is nested the deepest
                # otherwise = cc_nodes[0]  # the last "True" condition
                # # TypeError: cannot unpack non-iterable Node object
                # for ec in cc_nodes[1:]:
                #     otherwise = Node(Ifte, [ec[0], ec[1], otherwise])
                #
                # return otherwise

        # todo include Usub, ignore usub in tree len()
        # elif isinstance(expr, sympy.Mul) and expr.args[0] == -1 and expr.args[1].is_Atom:
        #     node = sympy_to_tree(expr.args[1], allow_chain=allow_chain)
        #
        #     return node

        elif isinstance(expr, sympy.Pow):
            # todo match nodes to simplified nodes in a different function
            if expr.args[1] == -1:  # sympy.S.NegativeOne= sfeh:check if assumptions are available
                _r = InverseFraction
                cc_nodes.pop(1)  # second arg was identified and must now be ignored
            elif expr.args[1] == 2:
                _r = Square
                cc_nodes.pop(1)
            elif expr.args[1] == sympy.S.Half:
                _r = Sqrt
                cc_nodes.pop(1)
            elif type(expr.args[1]) == RoundDummy:
                # todo check this BEFORE possibly the round operator is matched.
                _r = Powrounded
                cc_nodes[1] = cc_nodes[1].childs[0]
            else:
                _r = Pow
            return Node(_r, cc_nodes)

        elif isinstance(expr, RoundDummy):
            return Node(Round, [cc_nodes[0]])

        elif isinstance(expr, tuple(sym_nodes)):

            clss = sym_nodes[type(expr)]

            if isinstance(expr, sympy.Mul):
                for ii, cn in enumerate(cc_nodes):
                    if cn.label == InverseFraction:
                        divisor_node = cc_nodes.pop(ii)
                        divisor_node = divisor_node.childs[0]
                        dividend_node = cc_nodes[0]
                        return Node(Div, [dividend_node, divisor_node])

            if len(expr.args) > len(clss.get_child_xts()):
                if issubclass(clss, ChainableOp):
                    if allow_chain:
                        # todo actually do this at the very top
                        return Node(clss, cc_nodes, is_chain=True)
                    else:
                        _cc = cc_nodes[0]
                        for _c2 in cc_nodes[1:]:
                            _cc = Node(clss, [_cc, _c2])
                        return _cc
                else:
                    raise TypeError(f"{clss} takes exactly {len(clss.get_child_xts())} arguments ({len(expr.args)} given)")
            else:
                return Node(clss, cc_nodes)

        elif isinstance(expr, sympy.re):
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
    raise NotImplementedError(f'Expr missing: {expr}')


if __name__ == '__main__':

    for x in sym_nodes.keys():
        lel = 4.5
        print(x, isinstance(lel, x))
