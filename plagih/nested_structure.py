from dataclasses import dataclass

from plagih.plagih_tree import *


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

    def __init__(self, label, childs, depth=None, is_fix=False):
        self.label = label
        self.childs = childs

        self.is_fix = is_fix
        self.depth = depth

    def __str__(self):
        try:
            label_str = self.label.__name__
        except Exception as ex:
            label_str = self.label  # todo Float obj has no attr __name__

        if self.childs:
            childstr = ', '.join([str(x) for x in self.childs])
            label_str = f'{label_str}, {childstr}'

        return f"[{label_str}]"

    def get_sympy_expr(self):
        _sym = self.label.insym
        if self.childs:
            _cs = [cc.get_sympy_expr() for cc in self.childs]
            _sym = _sym(*_cs)
        else:
            if self.label.args:
                # _cs = self.label.args
                # try:
                #     _sym = _sym(_cs[0])
                # except Exception as ex:
                #     try:
                #         _sym = _sym(self.label)  # todo
                #         print('bbbbbbb')
                #     except Exception as ex:
                #         try:
                #             _sym = _sym()  # todo
                #         except Exception as ex:
                #             try:
                #                 _sym = _sym(_cs)  # todo
                #             except Exception as ex:
                #                 _sym = _sym(_cs)  # todo
                _sym = self.label.get_sym()
                _sym = _sym(*self.label.args)
                # try:
                #     _sym = _sym(*self.label.args)
                # except Exception as ex:
                #     _sym = _sym(*self.label.args)
            else:
                raise
        return _sym

    def repr_str(self):
        pass

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

    def update_fixed_nodes(self, origin: 'NestedStruc'):
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
        self.set_label(new_node.label)  # sfeh remove childs, is_fix...
        self.childs = new_node.childs  # sfeh maybe must be updated recursively
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


def sympy_to_nsted(expr):
    if isinstance(expr, bool):
        _r = Boolean(expr)
        return NestedStruc(_r, [])
    if isinstance(expr, sympy.logic.boolalg.BooleanAtom):
        expr = True if isinstance(expr, (bool, sympy.logic.boolalg.BooleanTrue)) else False
        _r = Boolean(expr)
        return NestedStruc(_r, [])

    # the following lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    # ==Terminal nodes==
    elif expr.is_Atom:
        if expr.is_Symbol:
            _r = Symbol(str(expr))  # sfeh str VERY important!!
            # _r = NestedStruc(Symbol(), [expr])
        else:
            expr_eval = expr.evalf()  # standard 15 digits
            if expr.is_Boolean:
                _r = Boolean(expr_eval)
                # _r = NestedStruc(Boolean(), [expr])
            elif expr.is_number:  # is_float does not match int
                _r = Float(float(expr_eval))  # sfeh round
                # _r = NestedStruc(Float(), [float(expr)])
            else:
                print(f'XXX What happened here? {expr}')
                raise
        return NestedStruc(_r, [])

    else:
        sptonode = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: sign, sympy.log: log, sympy.Mul: Mul,
                    sympy.Xor: Xor, sympy.Not: Not, sympy.Equality: Eq, sympy.And: And, sympy.Or: Or,
                    sympy.Unequality: Ne, sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.StrictGreaterThan: Gt,
                    sympy.GreaterThan: Ge, sympy.cos: cos, sympy.sin: sin, sympy.tan: tan, sympy.acos: acos,
                    sympy.asin: asin, sympy.atan: atan, sympy.tanh: tanh, sympy.sinh: sinh, sympy.cosh: cosh,
                    sympy.Min: Min, sympy.Max: Max}
        stn_keys = tuple(sptonode.keys())

        if isinstance(expr, sympy.Piecewise):
            _revlist = list(expr.args[::-1])  # tuples to list, reverse: last tuple must be nested the deepest
            _revlist = [[sympy_to_nsted(xx) for xx in list(x)] for x in _revlist]
            otherwise = _revlist[0][0]  # the last "True" condition
            for x in _revlist[1:]:
                otherwise = NestedStruc(Ifte, [x[1], x[0], otherwise])
            return otherwise

        elif isinstance(expr, stn_keys):

            clss = sptonode[type(expr)]
            args = [sympy_to_nsted(x) for x in expr.args]

            try:
                return NestedStruc(clss, args)
            except TypeError:
                result = args.pop()  # only commutative arity-2 functions here (Add, Mul, Max, Min)
                while args:
                    # sfeh:optimization
                    _r = clss(result, args.pop())
        else:
            raise NotImplementedError(f'Expr missing: {expr}')


if __name__ == '__main__':
    # x = NestedStruc(Add(), depth=0, childs=[Symbol('a'), Float(2.2)])
    # # x = NestedStruc(Symbol(), childs=['a'])
    # print(x)
    sptonode = {sympy.Add: Add, sympy.Pow: Pow, sympy.Abs: Abs, sympy.sign: sign, sympy.log: log, sympy.Mul: Mul,
                sympy.Xor: Xor, sympy.Not: Not, sympy.Equality: Eq, sympy.And: And, sympy.Or: Or,
                sympy.Unequality: Ne, sympy.StrictLessThan: Lt, sympy.LessThan: Le, sympy.StrictGreaterThan: Gt,
                sympy.GreaterThan: Ge, sympy.cos: cos, sympy.sin: sin, sympy.tan: tan, sympy.acos: acos,
                sympy.asin: asin, sympy.atan: atan, sympy.tanh: tanh, sympy.sinh: sinh, sympy.cosh: cosh,
                sympy.Min: Min, sympy.Max: Max}
    todo = {sympy.sqrt: Sqrt}
    stn_keys = tuple(sptonode.keys())
    for x in stn_keys:
        lel = 4.5
        print(x, isinstance(lel, x))
