import random
from dataclasses import dataclass

import sympy.core.numbers

from plagih.tree_labels import *
from plagih.tree_labels_chained import *

# For conversion from sympy into node
from plagih.tree_complexity.tree_edit_distance import apted_distance
from plagih.util import FLOAT_PRECISION, CHAINED_VERION

d_sym2node = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul,
              sympy.Xor: Xor, sympy.Not: Not, sympy.And: And, sympy.Or: Or,
              sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.StrictGreaterThan: Gt,
              sympy.GreaterThan: Ge, sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos,
              sympy.asin: Asin, sympy.atan: Atan, sympy.tanh: tanh, sympy.sinh: Sinh, sympy.cosh: Cosh,
              sympy.Min: Min, sympy.Max: Max, sympy.ITE: ITE, sympy.exp: exp}
d_op2chain = {Add: AddChain, Mul: MulChain, Min: MinChained, Max: MaxChained, And: AndChained, Or: OrChained,
              Piecewise: Piecewise}
# The chained version is the regular version updated with the following operators
d_sym2node_chain = d_sym2node | {sympy.Add: AddChain, sympy.Mul: MulChain, sympy.Min: MinChained, sympy.Max: MaxChained,
                                 sympy.And: AndChained, sympy.Or: OrChained, sympy.Piecewise: Piecewise,
                                 sympy.functions.elementary.piecewise.ExprCondPair: ExprCondPair}

# , sympy.Equality: Eq
# sfeh:open = {sympy.Unequality: Ne, sympy.Equality: Eq}
# sym_assumption_nodes = (sympy.re,)


@dataclass
class Node:
    """Recursively holds the nodes of a tree
    parent: pointer to parent node, 'None' implies root-node
        parent-parameter added for pseudo-backprop
    """

    def __init__(self, typus: Typus, childs: iter, depth=None, is_fix=False, is_chain=False):
        self.typus = typus
        self.childs = childs[:]  # ...usually a list, but can also be 'None'
        self.is_fix = is_fix
        self.depth = depth
        self.parent_node = None
        # self.is_chain = is_chain  # sfeh:xxx check update required

    def is_chain(self):
        """is the node in chain mode?
        -> when there are more childs than input-xtypes
        ...if there are less. its weird, maybe node in construction?"""
        if len(self.typus.xtype[0]) < len(self.childs):
            return False

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
                    if issubclass(self.typus, Number):
                        typus_str = f'{self.childs[0]:.3g}'  # '.5g'->5 decimals, trailing zeros, but rare (ugly) "E+04"
                    else:
                        typus_str = f'{self.childs[0]}'
                except TypeError as ex:
                    typus_str = str(self.childs[0].evalf())
                    # sympy.ONE -> 1.0000000...
                    # sfeh:open int, non-floats are handled badly
                except Exception as ex:
                    print(f'SUCCESS sfeh:debug, delete?2 KEEP? {ex}')

        return f"[{typus_str}]"

    def get_lut_id(self):
        """
        todo same node multiple times in tree, this is not "ID!"°!!
        Unique+simple representation of a tree (to check in a lut if it was calculated already)
        returns string Identificator
        sfeh:discuss is this just repr?
        ID=Identificator, which"""
        try:
            typus_str = self.typus.__name__  # sfeh: can str(typus) work? -> str with args recursively?
        except AttributeError as ex:
            print('XXX DEBUG delete me if you do not remember me :)')
            # sfeh debug me
            typus_str = self.typus  # because Terminals are obj -> 'Symbol' obj has no attr __name__

        if self.childs:
            if issubclass(self.typus, OperatorArity):
                childstr = ', '.join([cc.str_as_list() for cc in self.childs])
                typus_str = f'{typus_str}, {childstr}'
            elif issubclass(self.typus, OperatorChained):
                childstr = ', '.join([cc.str_as_list() for cc in self.childs])
                typus_str = f'{typus_str}, {childstr}'
            elif issubclass(self.typus, TerminalDummy):
                childstr = ', '.join([cc.str_as_list() for cc in self.childs])  # those are actual childs
                typus_str = f'({childstr})'
            else:
                try:
                    typus_str = f'{self.childs[0]}'
                except TypeError as ex:
                    typus_str = str(self.childs[0].evalf())
                    # sfeh:open int? rational?, non-floats are handled badly
                except Exception as ex:
                    print(f'sfeh:debug, delete? KEEP? {ex}')  # InvalidOperation([<class 'decimal.InvalidOperation'>])

        return f"[{typus_str}]"

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
            # sympy.Piecewise(sympy.functions.elementary.piecewise.ExprCondPair(1, True))
            # PW((a, b), (c, True))
            _cs = self.childs
            _cs = [(cc.childs[0], cc.childs[1]) for cc in _cs]
            _cs = [(cc[0].get_sympy_expr(), cc[1].get_sympy_expr()) for cc in _cs]
            _cs = [sympy.functions.elementary.piecewise.ExprCondPair(cc[0], cc[1]) for cc in _cs]
            _cs = _sym(*_cs)
            return _cs
        else:
            _cs = [cc.get_sympy_expr() for cc in self.childs]
            if issubclass(self.typus, OperatorArity):
                _sym = self.typus.get_sym()
            elif CHAINED_VERION:
                _sym = self.typus.get_sym()

            try:
                return _sym(*_cs)  # noqa (_sym is definitely assigned)
            except RecursionError as ex:
                print(f'sfeh:RecursionError, maybe Piecewise?: {self.typus}, {self.childs}, {ex}')
                raise RecursionError
            # except AttributeError as todo:
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
            except Exception as ex:
                raise ex

            raise NotImplementedError(f'get_sympy_expr no match for {self}, {type(self.typus)}')

    def get_expr_raw_fstring(self):
        """Add (1 + a)"""
        if issubclass(self.typus, Terminal):
            expr = f'{self.childs[0]}'
        else:
            expr = [cc.get_expr_raw_fstring() for cc in self.childs]
            expr = ', '.join(expr)
            if issubclass(self.typus, OperatorArity):
                expr = f'{self.typus.__name__}({expr})'
            elif CHAINED_VERION:
                try:
                    expr = f'{self.typus.expr_dmy}({expr})'
                except Exception as ex:
                    expr = f'({expr})'  # sfeh catching ExprCondPair here

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
        typus_str = self.typus

        if self.is_fix:
            typus_str += ':fix'

        if self.childs:
            childstr = ', '.join([self.represent(cc) for cc in self.childs])
            typus_str = f"{typus_str}, {childstr}"
        return f"[{typus_str}]"

    def __repr__(self):
        """sfeh:WRONG! Do NOT use __str__!"""
        return self.__str__()

    def len_nodecount_raw(self):
        """counting the amount of nodes recursively"""
        if issubclass(self.typus, Terminal):
            return 1  # childs can currently be floats
        else:
            return 1 + sum([cc.len_nodecount_raw() for cc in self.childs])

    def len_nodecount_fair(self):
        """counting the amount of nodes, but
            - ignoring "Usub!
        """
        if issubclass(self.typus, Terminal):
            return 1
        elif issubclass(self.typus, Usub):
            return sum([cc.len_nodecount_fair() for cc in self.childs])
        else:
            return 1 + sum([cc.len_nodecount_fair() for cc in self.childs])

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

    def set_typus(self, t: 'Typus'):
        """all other values are automatically set by assigning the respected node"""
        self.typus = t

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
            if str(self.typus) != str(origin.typus):
                raise
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.update_fixed_nodes(origin.childs[ii])

    def get_all_nodes_visualize(self, setid: str):
        """returns all nodes in a tree as list
        a+1 -> [+a1, a, 1]"""
        # if len(self.childs) == 0:
        try:
            showme = f'{self.childs[0]}' if self.is_term() else f'{self.get_typus().showme}'
        except Exception as ex:
            showme = f'ExCP'  # todo

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

    def is_regular(self):
        # Nodes that are notchained-nodes
        return issubclass(self.typus, (OperatorArity, Terminal))

    def is_chainop(self):
        return issubclass(self.typus, OperatorChained)

    def is_operator(self):
        return issubclass(self.typus, BaseOperator)

    def is_operator_chained(self):
        return issubclass(self.typus, OperatorChained)

    def is_operator_true(self):
        return issubclass(self.typus, OperatorArity)

    def is_term(self):
        # sfeh:discuss is_atom, rename all to atom?
        return issubclass(self.typus, Terminal)

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

    # todo:Do links to parents lead to problems when crossover/etc happens?

    def replace_with(self, typus, childs):
        if typus is not None:
            self.set_typus(typus)
        if childs is not None:
            self.set_childs(childs)

        # self.repair_depth(self.depth)  # sfeh:discuss not required
        # no checks

    def list_mutable_nodes(self, xt_match=None, ignore_first=False, allow_chain=False) -> ['Node']:
        """was eval_mutable_nodes,
        return all nodes that are mutable, aka suite for point- or branchmutation
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
                node_list.extend(cc.list_mutable_nodes(xt_match=xt_match, ignore_first=False, allow_chain=allow_chain))

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
            if issubclass(self.typus, Number):
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
                        self.replace_with(Div, childs=[self.childs[1], Node(Number, childs=[1 / mul1])])  # sfeh keep "div" as option

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


def sympy_to_tree(s_expr: sympy.Basic, allow_chain=False) -> Node:
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
            cc_nodes.append(sympy_to_tree(arg, allow_chain=allow_chain))

        if allow_chain:
            op = d_sym2node_chain[type(s_expr)]
            return Node(op, childs=cc_nodes)

        if isinstance(s_expr, sympy.functions.elementary.piecewise.ExprCondPair):
            # raise Exception(f'sfeh')
            return Node(ExprCondPair, cc_nodes)

        elif isinstance(s_expr, sympy.Piecewise):
            # "Chained_VERSION" version is handled before
            reversed_pairs = list(s_expr.args[::-1])  # tuples to list, reversed: tuple must be nested the deepest
            reversed_pairs = [[sympy_to_tree(xx, allow_chain=allow_chain) for xx in list(i)] for i in reversed_pairs]  # noqa
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
            return cc_nodes

    # sfeh:discuss
    # NotImplementedError: Expr missing: ITE(p > 13, tan(p - v) >= 2.578643, tan(p - v) >= 1)
    # this should not have occured, because it evaluates to bool, not to float
    raise NotImplementedError(f'Expr missing: {s_expr}')


# todo AttributeError: type object 'ExprCondPair' has no attribute 'symfun'
#   This also causes recusrion errors. rename maybe?


if __name__ == '__main__':

    for x in d_sym2node.keys():
        lel = 4.5
        print(x, isinstance(lel, x))
