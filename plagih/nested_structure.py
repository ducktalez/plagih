import itertools
from dataclasses import dataclass

from plagih.plagih_tree import *

# For sympy_to_Nested conversion
sptonode = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: Sign, sympy.log: Log, sympy.Mul: Mul,
            sympy.Xor: Xor, sympy.Not: Not, sympy.And: And, sympy.Or: Or,
            sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.StrictGreaterThan: Gt,
            sympy.GreaterThan: Ge, sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos,
            sympy.asin: Asin, sympy.atan: Atan, sympy.tanh: tanh, sympy.sinh: sinh, sympy.cosh: cosh,
            sympy.Min: Min, sympy.Max: Max}
# , sympy.Equality: Eq
# sfeh:open = {sympy.Unequality: Ne, sympy.Equality: Eq}
stn_keys = tuple(sptonode.keys())  # sfeh: debug this... relevant?


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
                    label_str = f'{self.childs[0]}'  # sfeh:hmmm
                except Exception as ex:
                    print(f'TODO IndexError: invalid index to scalar variable? {ex}')

        return f"[{label_str}]"

    def get_sympy_expr(self):
        if self.childs and issubclass(self.label, Operator):
            _sym = self.label.symfun
            _cs = [cc.get_sympy_expr() for cc in self.childs]
            # if self.label == Ifte:
            try:
                return _sym(*_cs)
            except RecursionError as ex:
                print(f'RecursionError, maybe Piecewise?: {self}, {ex}')
                raise RecursionError
        elif self.childs and issubclass(self.label, TerminalNode):
            _sym = self.label.symfun
            _cs = self.childs[0]
            return _sym(_cs)
        # else:
        #     if issubclass(type(self.label), TerminalNode):  # .--class--, as it is initiated?
        #         return self.label.get_sym()

        raise NotImplementedError(f'get_sympy_expr no match for {self}, {type(self.label)}')

    def eval_str(self):
        return self.get_label()  # sfeh open

    def __repr__(self):
        """
        Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to str(), but adds the following information:
        - ":fix", when nodes are fixed
        """
        label_str = self.label

        if self.is_fix:
            label_str += ':fix'

        if self.childs:
            childstr = ', '.join([repr(x) for x in self.childs])
            label_str = f"{label_str}, {childstr}"
        return f"[{label_str}]"

    def __len__(self):
        """counting the amount of nodes recursively"""
        if issubclass(self.label, TerminalNode):
            return 1  # childs can currently be floats
        else:
            return 1 + sum([len(cc) for cc in self.childs])

    def get_label(self):
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
        """does this work while building a fintree?"""
        return self.depth == 0

    def set_label(self, label: 'NodeBase'):
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
        """Calculating the TED requires this (weird) representation"""
        return f"{{{self.get_label()}{''.join([cc.eval_apted_notation() for cc in self.childs])}}}"

    def get_max_depth(self, depth=0):
        """Go through all nodes, save depth"""
        if len(self.childs) == 0:
            return depth
        else:
            return max(cc.get_max_depth(depth=depth + 1) for cc in self.childs)

    def is_operator(self):
        try:
            return issubclass(self.label, Operator)
        except Exception as ex:
            print(f'whats this? {self}; {self.label} {ex}')
            return False

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

    def set_new_nested(self, new_node: 'Node'):
        self.set_label(new_node.label)  # sfeh remove childs, is_fix...
        self.childs = new_node.childs  # sfeh maybe must be updated recursively
        self.repair_depth(self.depth)  # Especially required for crossover or branches

    def eval_mutable_nodes(self, xt_out=None, allow_root=True, allow_chain=False):
        """
        return all nodes that are mutable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!
        """
        node_list = []
        if not self.is_fix:  # requirement for mutability
            # crossover requires excluding types that are not matching, and excludes the root node
            # sfeh:open in anderer klasse
            if (xt_out is None or xt_out == self.get_xtype_out()) \
                    and (allow_root or not self.is_root()) \
                    and (allow_chain or not self.is_chain):
                node_list.append(self)
        if self.is_operator():
            for cc in self.childs:
                node_list.extend(cc.eval_mutable_nodes(xt_out=xt_out, allow_root=allow_root, allow_chain=allow_chain))
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
        if isinstance(self.get_label(), Operator):
            for cc in self.childs:
                cc.evolve_mutate_filter_branch(precision=precision)
        else:
            # self.label.mutate_self_filter(filter_type='gaussian_filter', precision=precision)
            # sfeh:xxx
            pass


def sympy_to_nsted(expr, allow_chain=False):
    if isinstance(expr, bool):
        # return Nested(Boolean(expr), [])
        return Node(Boolean, [expr])
    elif isinstance(expr, sympy.logic.boolalg.BooleanAtom):
        expr = True if isinstance(expr, (bool, sympy.logic.boolalg.BooleanTrue)) else False
        # return Nested(Boolean(expr), [])
        return Node(Boolean, [expr])

    # the following lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    # ==Terminal nodes==
    elif expr.is_Atom:
        if expr.is_Symbol:
            # _r = Symbol  # sfeh str VERY important!! Symbol type input is not accepted
            # return Nested(Symbol(str(expr)), [])
            return Node(Symbol, [str(expr)])
        else:
            expr_eval = expr.evalf()  # standard 15 digits, sfeh prec=FLOAT_PRECISION?
            if expr.is_Boolean:
                # return Nested(Boolean(bool(expr_eval)), [])
                return Node(Boolean, [bool(expr_eval)])
            elif expr.is_number:  # is_float does not match int
                # return Nested(Float(float(expr_eval)), [])  # sfeh round
                return Node(Float, [float(expr_eval)])  # sfeh round
                # "TypeError: Cannot convert complex to float" -> ignore the whole expression, let it fail
            else:
                print(f'XXX What happened here? {expr}')
                raise

    else:

        if isinstance(expr, sympy.Piecewise):
            if allow_chain:
                raise NotImplementedError
            else:
                _ccinv = list(expr.args[::-1])  # tuples to list, reverse: last tuple must be nested the deepest
                _ccinv = [[sympy_to_nsted(xx, allow_chain=allow_chain) for xx in list(i)] for i in _ccinv]
                otherwise = _ccinv[0][0]  # the last "True" condition
                for x in _ccinv[1:]:
                    otherwise = Node(Ifte, [x[1], x[0], otherwise])
                return otherwise
        elif isinstance(expr, sympy.Pow):
            if expr.args[1] in (-1, 2):
                if expr.args[1] == -1:  # sympy.S.NegativeOne= sfeh:check if assumptions are available
                    _r = InverseFraction
                elif expr.args[1] == 2:
                    _r = Square
                elif expr.args[1] == sympy.S.Half:
                    _r = Sqrt
                else:
                    raise
                return Node(_r, [sympy_to_nsted(expr.args[0], allow_chain=allow_chain)])  # can ignore args[1] now

            # sfeh:open
            # if isinstance(expr.args[1], sympy.Integer):
            #     _r = Powrounded
            # else:
            _r = Pow
            childnstd = [sympy_to_nsted(x, allow_chain=allow_chain) for x in expr.args]
            return Node(_r, childnstd)

        elif isinstance(expr, stn_keys):

            clss = sptonode[type(expr)]
            args = [sympy_to_nsted(x, allow_chain=allow_chain) for x in expr.args]

            if len(expr.args) > len(clss.xtype[0]):
                if issubclass(clss, ChainableOp):
                    if allow_chain:
                        return Node(clss, args, is_chain=True)
                    else:
                        # All have arity-2
                        childnstd = [sympy_to_nsted(x, allow_chain=allow_chain) for x in expr.args]
                        _cc = childnstd[0]
                        for _c2 in childnstd[1:]:
                            _cc = Node(clss, [_cc, _c2])
                        return _cc
                else:
                    raise TypeError(f"{clss} takes exactly {len(clss.xtype[0])} arguments ({len(expr.args)} given)")
            else:
                return Node(clss, args)  # same as in allow_chain

        else:
            # sfeh:discuss
            # NotImplementedError: Expr missing: ITE(p > 13, tan(p - v) >= 2.578643, tan(p - v) >= 1)
            # this should not have occured, because it evaluates to bool, not to float
            raise NotImplementedError(f'Expr missing: {expr}')


if __name__ == '__main__':
    # x = Nested(Add(), depth=0, childs=[Symbol('a'), Float(2.2)])
    # # x = Nested(Symbol(), childs=['a'])
    # print(x)
    for x in stn_keys:
        lel = 4.5
        print(x, isinstance(lel, x))
