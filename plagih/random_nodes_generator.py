from abc import ABC, abstractmethod

from plagih.util import print_warning


def check_operator_pool(ops: iter):
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


def norm_choices(val_p_tuples: (any, float)) -> [any, float]:
    """make a tuple-list callable for weighted numpy choice
    [['a', 1], ['b', 2]] -> [('a', 'b'), (0.333, 666)]"""
    xx = list(zip(*val_p_tuples))
    # normalizing the probabilities in every case to a sum of 1 (100%)
    psum = sum(xx[1])
    xx[1] = [i / psum for i in xx[1]]
    # lambda: np.random.choice(xx[0], p=xx[1])
    return xx


def operatorpool_to_picks(d_operator_pool):
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


# class NodeCreatorBase(ABC):
#
#     @abstractmethod
#     def choose_operator(self, xt):
#         pass
#
#     @abstractmethod
#     def choose_operator_match(self, xtype):
#         pass
#
#     @abstractmethod
#     def choose_terminal(self, xt):
#         pass
#
#     @abstractmethod
#     def choose_constant(self, xt):
#         pass
#
#     @abstractmethod
#     def choose_symbol(self, xt):
#         pass
