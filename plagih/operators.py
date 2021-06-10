"""
todo umbenennen in was neues
"""

import ast
from pathlib import Path
import sklearn.model_selection as skcv
from plagih.printing import *
import numpy as np
import pandas as pd
import re
from pydoc import locate
import random


import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # sfeh wasd wtf


# class Obs:
#
#     def mutate_filter(self):
#         if self.index_minmax is None:
#             return
#         else:
#             new_index = int(max(min(round(random.gauss(self.obs_index, 1)), self.index_minmax[1]), 0))
#             self.obs_index = new_index
#             self.name = f'{self.family}_{new_index}'
#
#     def __init__(self, name, ctype):
#         self.name = name
#         self.ctype = ctype
#         self.tf_type = tf.float32
#         self.iotype = float  # sfeh todo
#
#         obs_family, obs_index, prelabel = observation_get_family_and_time(name, none_return=None)
#         self.family = obs_family
#         self.obs_index = obs_index  # is None when no index but 0 when
#
#         self.index_minmax = None
#         self.fun_filter_index = lambda: None  # as default, return own index



# lol, lol. https://github.com/tensorflow/tensorflow/issues/27023 these messages are tingeling
# import tensorflow.python.util.deprecation as deprecation  # not possible on python 3.6
# deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # sfeh wasd wtf

"""
op: Dict to work as 'Database' for every expression-bit and its features
- KEY: expression-bit: can occur in various forms, which group into following uses:
    - ast.KEY: Found by pythons 'ast' when inline op like [+, -, *, ...] occurs
    - 'KEY': Found by pythons 'ast' when non-inline functions are found as string
    -> some operators are identical, but occur several times, e. g. ('And', '&', ast.BitAnd)
- VALUE: several features that have to be regarded. Some are irrelevant or have to be done.
    - name: (irrelevant) information what this function actually does. identical operators can be found like this.
    - arity: Amount of inputs an operator has to have. This must always be constant (reason why pythons min/max does not work)
    - tf: Tensorflow graph representation for the key. Was separated earlier. (!): [tf.bool, tf.float] are just variables of the used tf.cast function.
    - gpbp: Genetic Programming Backpropagation: An open idea from sfeh to introduce backpropagation for genetic operators.
    - latex: For visualizing trees with latex, these representations might look better

Features should always be defined, even though they might not occur at all. If not used, they CAN be filled with dummy values like äöü
Some, which are known of not being used yet are commented with '# not tested' or '# not used'
sfeh: write test that checks all operators for sympificytion (...+branch-combinations, and more?)
sfeh: use function-types (-> 'kommuttative'?)
"""


class Plabel:
    label = 'None'
    arity = 0
    tf = None
    sym_str = 'None'
    coolxtype = (tuple([None]), None)
    pycode = 'None'
    latex1 = 'None'
    latexF = 'None'

    def __str__(self):
        return str(self.label)

    def mutate_point(self):
        """
        evolve a point
        replace with same-arity, same type
        """
        pass

    def mutate_filter(self):
        pass

    def evolve_branch(self):
        """
        evolve a subbranch
        replace with same type, but any arity (child nodes are deleted)
        """
        pass


class BuildDummy(Plabel):
    """
    sfeh delete this? keep this?
    """
    def __init__(self, coolxtype_out, size=None):
        self.coolxtype = (tuple([None]), coolxtype_out)


class Ploperator(Plabel):
    pass


class Observation(Plabel):
    """

    """
    arity = 0

    def mutate_filter(self):
        """
        was filter_new_index
        """
        if self.index_minmax is None:
            return
        else:
            new_index = int(max(min(round(random.gauss(self.obs_index, 1)), self.index_minmax[1]), 0))
            self.obs_index = new_index
            self.name = f'{self.family}_{new_index}'

    def __init__(self, value, coolxtype_out=float, obs_indizes=None):
        obs_family, obs_index, prelabel = observation_get_family_and_time(value, none_return=None)

        self.label = value
        self.name = value if value[0] != '-' else value[1:]  # sfeh delete todo

        self.family = obs_family
        self.obs_index = obs_index  # is None when no index but 0 when

        self.iotype = coolxtype_out
        self.tf_type = tf.float32

        self.index_minmax = None
        self.fun_filter_index = lambda: None  # as default, return own index

        self.obs_indizes = obs_indizes
        if obs_index is None:
            self.latex1 = f'{prelabel}\\text{{{obs_family}}}'
            self.latexF = f'{prelabel}\\text{{{obs_family}}}'
        else:
            self.latex1 = f'{prelabel}\\text{{{obs_family}}}_{{{obs_index}}}'
            self.latexF = f'{prelabel}\\text{{{obs_family}}}_{{{obs_index}}}'
        self.sym_str = value  # sfeh delete?
        self.pycode = value  # sfeh delete?


class FloatConstant(Plabel):
    """

    """
    arity = 0
    otype = float
    coolxtype = (tuple([]), float)

    def __init__(self, value):
        self.label = value
        self.latex1 = f'{value:.3f}'
        self.latexF = f'{value:.3f}'
        self.sym_str = value
        self.pycode = value
        # self.name = value if value[0] != '-' else value[1:]  # sfeh delete todo

    def mutate_filter(self, filter_type='gaussian_filter', float_decimals=6):  # todo
        if filter_type == 'gaussian_filter':
            if random.choice(['v1', 'v2']) == 'v1' or self.label == 0:
                self.label += np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.label, 0.1)  # sfeh better adjustments?
                self.label = round(constant, float_decimals)  # sfeh be careful, might create zero sometimes


class BoolConstant(Plabel):
    """

    """
    arity = 0
    otype = bool
    coolxtype = (tuple([]), bool)

    def __init__(self, value):
        self.latex1 = f'{value}'
        self.latexF = f'{value}'
        self.sym_str = value
        self.pycode = value


class EvalAction(Plabel):
    """
    - minmax for histograms
    - minmax for regression-bounded
    """
    tf_type = tf.float32  # sfeh especiall when the type is integer
    iotype = float  # sfeh todo
    coolxtype = [None, float]

    def __init__(self, name, ctype, cmin, cmax, uniques):
        self.plabel = name
        self.name = name  # delete this
        self.ctype = ctype  # ! may be int!

        self.uniques = uniques
        self.minmax = (ctype(cmin), ctype(cmax))


class Add(Plabel):
    label = '+'
    arity = 2
    tf = tf.add
    latex1 = '+'
    latexF = '{}+{}'
    sym_str = '({} + {})'
    pycode = '({}+{})'
    coolxtype = (tuple([float, float]), float)


class Subtract(Plabel):
    """

    """
    label = '-'
    arity = 2
    tf = tf.subtract
    latex1 = '-'
    latexF = '{}-{}'
    sym_str = '({} - {})'
    pycode = '({}-{})'
    coolxtype = (tuple([float, float]), float)


class Usub(Plabel):
    label = 'Usub'
    arity = 1
    tf = tf.negative
    latex1 = '-'
    latexF = '-{}'
    sym_str = '(-{})'
    pycode = '(-{})'
    coolxtype = (tuple([float]), float)


class Multiply(Plabel):
    label = '*'
    arity = 2
    tf = tf.multiply
    latex1 = '\\cdot '
    latexF = '{}\\cdot {}'
    sym_str = '({} * {})'
    pycode = '({}*{})'
    coolxtype = (tuple([float, float]), float)


class Divide_no_nan(Plabel):
    label = '/'
    arity = 2
    tf = tf.math.divide_no_nan
    latex1 = '\\div '
    latexF = '\\frac{}{}'
    sym_str = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'
    coolxtype = (tuple([float, float]), float)


class Power(Plabel):
    label = '**'
    arity = 2
    tf = tf.pow
    latex1 = '{{x}}^{{y}}'
    latexF = '{}^{}'
    sym_str = '({}**Round({}))'
    pycode = '({}**round({}))'
    coolxtype = (tuple([float, float]), float)


class Abs(Plabel):
    label = 'abs'
    arity = 1
    tf = tf.abs
    latex1 = 'abs'
    latexF = '|{}|'
    sym_str = 'abs({})'
    pycode = 'abs({})'
    coolxtype = (tuple([float]), float)


class Sign(Plabel):
    label = 'sign'
    arity = 1
    tf = tf.sign
    latex1 = 'sign'
    latexF = 'sign({})'
    sym_str = 'sign({})'
    pycode = 'np.sign({})'
    coolxtype = (tuple([float]), float)


class Round(Plabel):
    label = 'Round'
    arity = 1
    tf = tf.round
    latex1 = 'round'
    latexF = 'round({})'
    sym_str = 'Round({})'
    pycode = 'round({})'
    coolxtype = (tuple([float]), float)


class Square(Plabel):
    label = 'Square'
    arity = 1
    tf = tf.square
    latex1 = 'x^2'
    latexF = '{}^2'
    sym_str = 'Square({})'
    pycode = '({})**2'
    coolxtype = (tuple([float]), float)


class Sqrt(Plabel):
    label = 'sqrt'
    arity = 1
    tf = tf.sqrt
    latex1 = '\\sqrt{x}'
    latexF = '\\sqrt{}'
    sym_str = 'sqrt({})'
    pycode = 'math.sqrt({})'
    coolxtype = (tuple([float]), float)


class Log(Plabel):
    label = 'log'
    arity = 1
    tf = tf.math.log
    latex1 = '\\log()'
    latexF = '\\log{}'
    sym_str = 'log({})'
    pycode = 'math.log({})'
    coolxtype = (tuple([float]), float)


class Log1p(Plabel):
    label = 'log1p'
    arity = 1
    tf = tf.math.log1p
    latex1 = '\\log(1+x)'
    latexF = '\\log(1+{})'
    sym_str = 'log1p({})'
    pycode = 'math.log1p({})'
    coolxtype = (tuple([float]), float)


class Cos(Plabel):
    label = 'cos'
    arity = 1
    tf = tf.cos
    latex1 = '\\cos '
    latexF = '\\cos({})'
    sym_str = 'cos({})'
    pycode = 'math.cos({})'
    coolxtype = (tuple([float]), float)


class Sin(Plabel):
    label = 'sin'
    arity = 1
    tf = tf.sin
    latex1 = '\\sin '
    latexF = '\\sin({})'
    sym_str = 'sin({})'
    pycode = 'math.sin({})'
    coolxtype = (tuple([float]), float)


class Tan(Plabel):
    label = 'tan'
    arity = 1
    tf = tf.tan
    latex1 = '\\tan '
    latexF = '\\tan({})'
    sym_str = 'tan({})'
    pycode = 'math.tan({})'
    coolxtype = (tuple([float]), float)


class Acos(Plabel):
    label = 'acos'
    arity = 1
    tf = tf.acos
    latex1 = '\\acos '
    latexF = '\\acos({})'
    sym_str = 'acos({})'
    pycode = 'math.acos({})'
    coolxtype = (tuple([float]), float)


class Asin(Plabel):
    label = 'asin'
    arity = 1
    tf = tf.asin
    latex1 = '\\asin '
    latexF = '\\asin({})'
    sym_str = 'asin({})'
    pycode = 'math.asin({})'
    coolxtype = (tuple([float]), float)


class Atan(Plabel):
    label = 'atan'
    arity = 1
    tf = tf.atan
    latex1 = '\\atan '
    latexF = '\\atan({})'
    sym_str = 'atan({})'
    pycode = 'math.atan({})'
    coolxtype = (tuple([float]), float)


class Tanh(Plabel):
    label = 'tanh'
    arity = 1
    tf = tf.tanh
    latex1 = '\\tanh '
    latexF = '\\tanh({})'
    sym_str = 'tanh({})'
    pycode = 'math.tanh({})'
    coolxtype = (tuple([float]), float)


class And(Plabel):
    label = 'Andb'
    arity = 2
    tf = tf.logical_and
    latex1 = 'and'
    latexF = '({}\\wedge{})'
    sym_str = 'Andb({}, {})'
    pycode = '({} and {})'
    coolxtype = (tuple([bool, bool]), bool)


class Or(Plabel):
    label = 'Orb'
    arity = 2
    tf = tf.logical_or
    latex1 = 'or'
    latexF = '({}\\vee{})'
    sym_str = 'Orb({}, {})'
    pycode = '({} or {})'
    coolxtype = (tuple([bool, bool]), bool)


class Xor(Plabel):
    label = 'Xor'
    arity = 2
    tf = tf.math.logical_xor
    latex1 = '\\oplus'
    latexF = '({}\\oplus{})'
    sym_str = 'Xor({}, {})'
    pycode = '({} ^ {})'
    coolxtype = (tuple([bool, bool]), bool)


class Not(Plabel):
    label = 'Notb'
    arity = 1
    tf = tf.logical_not
    latex1 = '\\neg'
    latexF = '\\neg{}'
    sym_str = 'Notb({})'
    pycode = 'not({})'
    coolxtype = (tuple([bool]), bool)


class Eq(Plabel):
    label = '=='
    arity = 2
    tf = tf.equal
    latex1 = '='
    latexF = '({}={})'
    sym_str = '({} == {})'
    pycode = '({}=={})'
    coolxtype = (tuple([bool, bool]), bool)


class Neq(Plabel):
    label = '!='
    arity = 2
    tf = tf.not_equal
    latex1 = '\\neq'
    latexF = '({}\\neq{})'
    sym_str = '({} != {})'
    pycode = '({}!={})'
    coolxtype = (tuple([bool, bool]), bool)


class Lt(Plabel):
    label = '<'
    arity = 2
    tf = tf.less
    latex1 = '<'
    latexF = '{}<{}'
    sym_str = '({} < {})'
    pycode = '({}<{})'
    coolxtype = (tuple([float, float]), bool)


class Le(Plabel):
    label = '<='
    arity = 2
    tf = tf.less_equal
    latex1 = '\\leq'
    latexF = '{}\\leq{}'
    sym_str = '({} <= {})'
    pycode = '({}<={})'
    coolxtype = (tuple([float, float]), bool)


class Gt(Plabel):
    label = '>'
    arity = 2
    tf = tf.greater
    latex1 = '>'
    latexF = '{}>{}'
    sym_str = '({} > {})'
    pycode = '({}>{})'
    coolxtype = (tuple([float, float]), bool)


class Ge(Plabel):
    label = '>='
    arity = 2
    tf = tf.greater_equal
    latex1 = '\\geq'
    latexF = '{}\\geq{}'
    sym_str = '({} >= {})'
    pycode = '({}>={})'
    coolxtype = (tuple([float, float]), bool)


class Ifte(Plabel):
    label = 'Ifte'
    arity = 3
    tf = tf.where
    latex1 = '\\text{if-then-else}'
    latexF = '\\text{{ if }} ({}) \\text{{ then }} ({}) \\text{{ else }} ({})'  # 'if({} then {} else {})'
    sym_str = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'
    coolxtype = (tuple([bool, float, float]), float)


class Min(Plabel):
    label = 'Mini'
    arity = 2
    tf = tf.minimum
    latex1 = '\\min'
    latexF = '\\min({}, {})'
    sym_str = 'Mini({}, {})'
    pycode = 'min({}, {})'
    coolxtype = (tuple([float, float]), float)


class Max(Plabel):
    label = 'Maxi'
    arity = 2
    tf = tf.maximum
    latex1 = '\\max'
    latexF = '\\max({}, {})'
    sym_str = 'Maxi({}, {})'
    pycode = 'max({}, {})'
    coolxtype = (tuple([float, float]), float)


def observation_select_index(obs_list, max_hist=10):
    """
    chooses variables but weighting how old they are.
    obs_list = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4']
    -> [0.28, 0.23, 0.19, 0.16, 0.13]
    sfeh: what about larger steps?
    0, 1, 2, 3 is good, but
    0, 5, 10, 15 is worse
    what if variables are not all of same diff?
    """
    obs_list = np.delete(obs_list, np.s_[max_hist:])
    x = len(obs_list)
    fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
    p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
    p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
    return p


class EnvVars:
    """
    observation
    - take ~100 samples as constants
    - get xtype for buildin tree from expr
    - eval - get tensors with correct tf-type
    - filter index
    eval_action
    """

    def __init__(self):
        # self.obs_krazy = {}  # lookup table with all observations - if an observation is not in here, it is a float # sfeh delete this
        self.obs_infos = {}
        self.eval_action: EvalAction = None
        self.choose_obs = {float: None,
                           bool: None}


def data_from_csv(path_data_csv, action_name, test_size=0.2, delimiter=','):
    """
    Loads .csv data files.
    Information that we need to extract for each column:
    - choose_xtype choosing a random observation for leaf nodes
    - filtering observation index
        is there an index (past values, e.g. velocity_0, velocity_1) -> performing filter-evolve on the variables index (velocity_2 -> velocity_3)
    - evalaction data_train, data_test, is the action for the regression? -> more than one action might be required (IB has three action dimensions)
        - action min max -> for kernel regression bounded. occuring min and max values might not be the theoretical min/max values


    deprecated:

    """

    def is_action(name, role):
        if role is None:
            is_act = any(x in role for x in ['out', 'action', 'act', 'result'])
        else:
            is_act = any(['a_' == name[:2],
                          ii == len(df.columns) - 1,
                          any(x in name for x in ['out', 'action', 'act', 'result'])])
        return is_act

    """
    - Reading the .csv-file (with pandas)
    - renaming column headers
    - saving header info for later use
    """
    with Path.open(path_data_csv) as file:
        df = pd.read_csv(file, delimiter=delimiter)
        # todo it is float64, float64, int64 with MTC.. does it work with Tensorflow?

    actionmain = None
    observations = []
    rename_columns = {}  # cartPos|type=int|... -> cartPos

    """
    1. split col name
    - check whether its an observation
    --> check whether there are indizes
    - check whether its an action
    --> check unique valuea, min, max
    - check if it should be ignored (deprecated action, irrelevant column)
    2. 
    """

    for ii, header in enumerate(df):

        header_split = header.split('|')  # split 1: cartVel|type=float|role=input --> {cartVel, type=float, role=input]
        column_name = header_split[0]  # The first entry is always the column name
        if column_name in op:
            raise Exception(f'Your samples hold a column that matches the potential tree operator {column_name}.\n'
                            f'That might end up in confusion, please rename the column.')

        # todo the column name is actually just the base name
        rename_columns[header] = column_name

        col_infos = {}  # column_name: {'type': 'float', 'role': None, 'colpos': ii}}
        try:
            for colparam in header_split[1:]:
                k, v = colparam.split('=')
                col_infos[k] = v
        except Exception as ex:
            print(f'Could not load samples csv correctly: {ex}')

        coolxtype_out = float

        if is_action(column_name, col_infos.get('role')):
            if column_name == action_name or action_name is None:
                # todo min/max SHOULD be either given manually, be set to [0,1] or [-1,1] or depend on the data
                cmin = col_infos.get('min', df[header].min())
                cmax = col_infos.get('max', df[header].max())
                uniques = df[header].unique()
                actionmain = EvalAction(column_name, coolxtype_out, cmin, cmax, uniques)
            else:
                # drop_actions.append(column_name)  # drop from data (only evaluate ONE action, drop other potential actions)
                df = df.drop(column_name, axis=1)  # no need to keep other actions
                printez('i', f'Ignoring action {column_name} for this run')  # , print_type=print_type print_type does not exist yet sfeh
        else:
            observations.append(Observation(column_name, coolxtype_out=coolxtype_out))

    df.rename(columns=rename_columns, inplace=True)
    df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P Following design pattern #YOLO

    """
    choosing random observations made easy
    """
    environment = EnvVars()

    obs_prop = []
    obs_info = {}

    obs_families = list(set(x.family for x in observations))
    for fam in obs_families:
        family_meeting = sorted([x for x in observations if x.family == fam], key=lambda o: o.obs_index)
        if len(family_meeting) > 1:
            observations.extend([x for x in family_meeting])
            obs_prop.extend(list(observation_select_index(family_meeting)))
            index_minmax = (family_meeting[0].obs_index, family_meeting[-1].obs_index)
            for obs in family_meeting:
                obs.index_minmax = index_minmax
                environment.obs_infos[obs.name] = obs
                obs_info[obs.name] = obs
        else:
            # LOL UMAD? only one family member (probably even more common)
            obs = family_meeting[0]
            obs_info[obs.name] = obs
            observations.append(obs)
            obs_prop.append(1)  # just one value

    environment.choose_obs = {float: lambda: np.random.choice(observations, p=obs_prop),
                           bool: None}  # sfeh None? no  lambda? yeah, not important but still...
    environment.obs_infos = obs_info

    if actionmain:
        environment.eval_action = actionmain
    else:
        raise

    data_train, data_test = skcv.train_test_split(df, test_size=test_size, random_state=0)

    return environment, data_train, data_test


def observation_get_family_and_time(name, re_pattern='_\\d+$', none_return=None):
    """
    When an observation is known, return the family, the time and the SIGN!!
    """

    core_label = re.split(re_pattern, name)[0]
    if core_label[0] == '-':
        core_label = core_label[1::]
        prelabel = '-'
    else:
        prelabel = ''
    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception:
        temp_diff = none_return
    return core_label, temp_diff, prelabel


op_what = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': Add,
    '-': Subtract,
    'Usub': Usub,
    '*': Multiply,
    # Division: SAFE division by zero! -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': Divide_no_nan,
    '**': Power,
    'Abs': Abs,
    'sign': Sign,
    'Round': Round,
    'Square': Square,
    'sqrt': Sqrt,
    'log': Log,  # sfeh log/ln?
    'log1p': Log1p,
    'cos': Cos,
    'sin': Sin,
    'tan': Tan,
    'acos': Acos,
    'asin': Asin,
    'atan': Atan,
    'tanh': Tanh,
    # 'Integer': {'fun_class': '', 'label': 'Integer', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 0.5, 'tf_name': '', 'tf': tf.cast({}, tf.int32), 'latex1': None, 'latexF': '{}',
    # 'sym_reduce': None, 'sym_str': 'N({}, )', 'pycode': 'math.tanh({})'},

    # 'b2b' Classical logical operators, evaluate from bool to bool
    # DON'T USE tf.bitwise.bitwise_and
    # sympify('Or')->'|', sympify('And')->'&', sympify('Not')->'~'
    'Andb': And,
    'Orb': Or,
    'Xor': Xor,
    'Notb': Not,

    # 'f2b' Classical comparative operators, evaluate from float to bool
    '==': Eq,
    '!=': Neq,
    '<': Lt,  # a < b
    '<=': Le,
    '>': Gt,  # a > b
    '>=': Ge,  # a >= 1

    # Functions which need separate handling in sympify
    'Ifte': Ifte,  # sfeh essential for evaluation
    # long version of Ifte-'pycode': 'if {0}:\n{1}\nelse:\n{2}'.format(a, textwrap.indent(str(b), '\t'), textwrap.indent(str(c), '\t'))
    'Mini': Min,  # with forced arity-2
    'Maxi': Max,  # with forced arity-2
}

## Currently not in use

op = {
    '+': op_what['+'],
    ast.Add: op_what['+'],
    '-': op_what['-'],
    ast.Sub: op_what['-'],
    '~': op_what['Usub'],
    'Usub': op_what['Usub'],
    'usub': op_what['Usub'],  # delete this sfeh
    ast.USub: op_what['Usub'],
    'Round': op_what['Round'],
    '*': op_what['*'],
    ast.Mult: op_what['*'],
    '/': op_what['/'],
    ast.Div: op_what['/'],
    '**': op_what['**'],
    ast.Pow: op_what['**'],
    'abs': op_what['Abs'],  # delete this
    'Abs': op_what['Abs'],
    'sign': op_what['sign'],
    'Square': op_what['Square'],
    'sqrt': op_what['sqrt'],
    'log': op_what['log'],
    'log1p': op_what['log1p'],
    'cos': op_what['cos'],
    'sin': op_what['sin'],
    'tan': op_what['tan'],
    'acos': op_what['acos'],
    'asin': op_what['asin'],
    'atan': op_what['atan'],
    'tanh': op_what['tanh'],
    'Andb': op_what['Andb'],
    'And': op_what['Andb'],
    ast.And: op_what['Andb'],
    '&': op_what['Orb'],
    ast.BitAnd: op_what['Andb'],
    'Orb': op_what['Orb'],
    'Or': op_what['Orb'],
    ast.Or: op_what['Orb'],
    'Xor': op_what['Xor'],
    'Notb': op_what['Notb'],
    ast.Not: op_what['Notb'],
    '==': op_what['=='],
    ast.Eq: op_what['=='],
    '!=': op_what['!='],
    ast.NotEq: op_what['!='],
    '<': op_what['<'],
    ast.Lt: op_what['<'],
    '<=': op_what['<='],
    ast.LtE: op_what['<='],
    '>': op_what['>'],
    ast.Gt: op_what['>'],
    '>=': op_what['>='],
    ast.GtE: op_what['>='],
    'Ifte': op_what['Ifte'],
    'Mini': op_what['Mini'],
    'Maxi': op_what['Maxi'],
}

latex_inline = ['+', '-', '*', '**', '==', '!=', '<', '<=', '>', '>=', 'Andb', 'Orb', 'Xor']

op_test = {
    '&': {'fun_class': '', 'label': '&', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 0.5, 'tf_name': '', 'tf': tf.logical_and, 'latex1': '\\land', 'latexF': '({}\\wedge{})',
          'sym_reduce': None, 'sym_str': '({} & {})', 'pycode': '({} and {})'},
    'Power3': {'fun_class': '', 'label': '', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 3, 'tf_name': '', 'tf': tf.pow, 'latex1': None, 'latexF': '{}',
               'sym_reduce': None, 'sym_str': '({}**2)', 'pycode': '({}**2)'},
    'Nand': {'fun_class': '', 'label': 'Nand', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'Nand({}, {})', 'pycode': 'Notb({} and {})'},
    'Xand': {'fun_class': '', 'label': 'Xand', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'Xand({}, {})', 'pycode': 'Notb({} ^ {})'},
    'Nor': {'fun_class': '', 'label': 'Nor', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_reduce': None, 'sym_str': 'Nor({}, {})', 'pycode': None},
    'Xnor': {'fun_class': '', 'label': 'Xnor', 'arity': 2, 'xtype': 'b2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'Xnor({}, {})', 'pycode': None},

    'log2': {'fun_class': '', 'label': 'log2', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 3, 'tf_name': '', 'tf': False, 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': 'log({})', 'pycode': 'math.log({})'},
    'log10': {'fun_class': '', 'label': 'log2', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 3, 'tf_name': '', 'tf': False, 'latex1': None, 'latexF': '{}',
              'sym_reduce': None, 'sym_str': 'log({})', 'pycode': 'math.log({})'},

    # I think this is only used when trying to get the xtype of type(variable), e.g. 0.3 -> float -> '2f'
    'float': {'fun_class': '', 'label': 'float', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': 'float({})'},  # not tested
    'int': {'fun_class': '', 'label': 'int', 'arity': 1, 'xtype': 'f2f', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
            'sym_reduce': None, 'sym_str': 'Integer({})', 'pycode': 'int({})'},  # not tested
    'bool': {'fun_class': '', 'label': 'bool', 'arity': 1, 'xtype': 'f2b', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
             'sym_reduce': None, 'sym_str': '', 'pycode': 'bool({})'},  # not tested

    # sfeh sqrt, is only 2nd root, also 3rd-root?

    # Loops. Never used yet, not working (sfeh). Loops that make GP very unsafe in terms of evaluation time.
    # Also very unclear how they should work.
    # - Let user specify them completely. Limit use to 1 (or 2) per tree.
    # - temporary variable(s) must be introduced that change within loop
    # - Condition must be change within loop
    'while': {'fun_class': '', 'label': 'while', 'arity': 2, 'xtype': 'b2?2?', 'coolxtype': ([], []), 'c-weight': 1, 'tf_name': '', 'tf': 'ä', 'latex1': None, 'latexF': '{}',
              'sym_str': None, 'pycode': None},  # sfeh: not working # Condition must change in loop

    # sfeh: not working # repeat n time, specify n (int) by user?
}

# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['label'], 1/v['coolxtype': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


if __name__ == "__main__":
    for k, v in op.items():
        print(f'Operators: {k}, {v}')
