from plagih.plagih_data import *


class Plabel:
    pass


class Add(Plabel):

    fun_label = '+'
    arity = 2
    xtype = 'f2f'
    c_weight = 1
    tf = tf.add
    latex1 = '+'
    latexF = '{}+{}'
    sym_str = '({} + {})'
    pycode = '({}+{})'


class Subtract(Plabel):

    fun_label = '-'
    arity = 2
    xtype = 'f2f'
    c_weight = 1
    tf = tf.subtract
    latex1 = '-'
    latexF = '{}-{}'
    sym_str = '({} - {})'
    pycode = '({}-{})'


class Usub(Plabel):

    fun_label = 'Usub'
    arity = 1
    xtype = 'f2f'
    c_weight = 0.5
    tf = tf.negative
    latex1 = '-'
    latexF = '-{}'
    sym_str = '(-{})'
    pycode = '(-{})'


class Multiply(Plabel):

    fun_label = '*'
    arity = 2
    xtype = 'f2f'
    c_weight = 1
    tf = tf.multiply
    latex1 = '\\cdot '
    latexF = '{}\\cdot {}'
    sym_str = '({} * {})'
    pycode = '({}*{})'


class Divide_no_nan(Plabel):

    fun_label = '/'
    arity = 2
    xtype = 'f2f'
    c_weight = 1
    tf = tf.math.divide_no_nan
    latex1 = '\\div '
    latexF = '\\frac{}{}'
    sym_str = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'


class Power(Plabel):

    fun_label = '**'
    arity = 2
    xtype = 'f2f'
    c_weight = 2
    tf = tf.pow
    latex1 = '{{x}}^{{y}}'
    latexF = '{}^{}'
    sym_str = '({}**Round({}))'
    pycode = '({}**round({}))'


class Abs(Plabel):

    fun_label = 'abs'
    arity = 1
    xtype = 'f2f'
    c_weight = 1
    tf = tf.abs
    latex1 = 'abs'
    latexF = '|{}|'
    sym_str = 'abs({})'
    pycode = 'abs({})'


class Sign(Plabel):

    fun_label = 'sign'
    arity = 1
    xtype = 'f2f'
    c_weight = 1
    tf = tf.sign
    latex1 = 'sign'
    latexF = 'sign({})'
    sym_str = 'sign({})'
    pycode = 'np.sign({})'


class Round(Plabel):

    fun_label = 'Round'
    arity = 1
    xtype = 'f2f'
    c_weight = 1
    tf = tf.round
    latex1 = 'round'
    latexF = 'round({})'
    sym_str = 'Round({})'
    pycode = 'round({})'


class Square(Plabel):

    fun_label = 'Square'
    arity = 1
    xtype = 'f2f'
    c_weight = 2
    tf = tf.square
    latex1 = 'x^2'
    latexF = '{}^2'
    sym_str = 'Square({})'
    pycode = '({})**2'


class Sqrt(Plabel):

    fun_label = 'sqrt'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.sqrt
    latex1 = '\\sqrt{x}'
    latexF = '\\sqrt{}'
    sym_str = 'sqrt({})'
    pycode = 'math.sqrt({})'


class Log(Plabel):

    fun_label = 'log'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.math.log
    latex1 = '\\log()'
    latexF = '\\log{}'
    sym_str = 'log({})'
    pycode = 'math.log({})'


class Log1p(Plabel):

    fun_label = 'log1p'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.math.log1p
    latex1 = '\\log(1+x)'
    latexF = '\\log(1+{})'
    sym_str = 'log1p({})'
    pycode = 'math.log1p({})'


class Cos(Plabel):

    fun_label = 'cos'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.cos
    latex1 = '\\cos '
    latexF = '\\cos({})'
    sym_str = 'cos({})'
    pycode = 'math.cos({})'


class Sin(Plabel):

    fun_label = 'sin'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.sin
    latex1 = '\\sin '
    latexF = '\\sin({})'
    sym_str = 'sin({})'
    pycode = 'math.sin({})'


class Tan(Plabel):

    fun_label = 'tan'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.tan
    latex1 = '\\tan '
    latexF = '\\tan({})'
    sym_str = 'tan({})'
    pycode = 'math.tan({})'


class Acos(Plabel):

    fun_label = 'acos'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.acos
    latex1 = '\\acos '
    latexF = '\\acos({})'
    sym_str = 'acos({})'
    pycode = 'math.acos({})'


class Asin(Plabel):

    fun_label = 'asin'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.asin
    latex1 = '\\asin '
    latexF = '\\asin({})'
    sym_str = 'asin({})'
    pycode = 'math.asin({})'


class Atan(Plabel):

    fun_label = 'atan'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.atan
    latex1 = '\\atan '
    latexF = '\\atan({})'
    sym_str = 'atan({})'
    pycode = 'math.atan({})'


class Tanh(Plabel):

    fun_label = 'tanh'
    arity = 1
    xtype = 'f2f'
    c_weight = 3
    tf = tf.tanh
    latex1 = '\\tanh '
    latexF = '\\tanh({})'
    sym_str = 'tanh({})'
    pycode = 'math.tanh({})'


class And(Plabel):

    fun_label = 'Andb'
    arity = 2
    xtype = 'b2b'
    c_weight = 0.5
    tf = tf.logical_and
    latex1 = 'and'
    latexF = '({}\\wedge{})'
    sym_str = 'Andb({}, {})'
    pycode = '({} and {})'


class Or(Plabel):

    fun_label = 'Orb'
    arity = 2
    xtype = 'b2b'
    c_weight = 0.5
    tf = tf.logical_or
    latex1 = 'or'
    latexF = '({}\\vee{})'
    sym_str = 'Orb({}, {})'
    pycode = '({} or {})'


class Xor(Plabel):

    fun_label = 'Xor'
    arity = 2
    xtype = 'b2b'
    c_weight = 0.5
    tf = tf.logical_xor
    latex1 = '\\oplus'
    latexF = '({}\\oplus{})'
    sym_str = 'Xor({}, {})'
    pycode = '({} ^ {})'


class Not(Plabel):

    fun_label = 'Notb'
    arity = 1
    xtype = 'b2b'
    c_weight = 0.5
    tf = tf.logical_not
    latex1 = '\\neg'
    latexF = '\\neg{}'
    sym_str = 'Notb({})'
    pycode = 'not({})'



class Eq(Plabel):

    fun_label = '=='
    arity = 2
    xtype = 'f2b'
    c_weight = 1
    tf = tf.equal
    latex1 = '='
    latexF = '({}={})'
    sym_str = '({} == {})'
    pycode = '({}=={})'


class Neq(Plabel):

    fun_label = '!='
    arity = 2
    xtype = 'f2b'
    c_weight = 1
    tf = tf.not_equal
    latex1 = '\\neq'
    latexF = '({}\\neq{})'
    sym_str = '({} != {})'
    pycode = '({}!={})'


class Lt(Plabel):

    fun_label = '<'
    arity = 2
    xtype = 'f2b'
    c_weight = 1
    tf = tf.less
    latex1 = '<'
    latexF = '{}<{}'
    sym_str = '({} < {})'
    pycode = '({}<{})'


class Le(Plabel):

    fun_label = '<='
    arity = 2
    xtype = 'f2b'
    c_weight = 1
    tf = tf.less_equal
    latex1 = '\\leq'
    latexF = '{}\\leq{}'
    sym_str = '({} <= {})'
    pycode = '({}<={})'


class Gt(Plabel):

    fun_label = '>'
    arity = 2
    xtype = 'f2b'
    c_weight = 1
    tf = tf.greater
    latex1 = '>'
    latexF = '{}>{}'
    sym_str = '({} > {})'
    pycode = '({}>{})'


class Ge(Plabel):

    fun_label = '>='
    arity = 2
    xtype = 'f2b'
    c_weight = 1
    tf = tf.greater_equal
    latex1 = '\\geq'
    latexF = '{}\\geq{}'
    sym_str = '({} >= {})'
    pycode = '({}>={})'


class Ifte(Plabel):

    fun_label = 'Ifte'
    arity = 3
    xtype = 'b2f2f'
    c_weight = 0.1
    tf = tf.where
    latex1 = '\\text{if-then-else}'
    latexF = 'if({} then {} else {})'
    sym_str = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'


class Min(Plabel):

    fun_label = 'Mini'
    arity = 2
    xtype = 'f2f'
    c_weight = 0.5
    tf = tf.minimum
    latex1 = '\\min'
    latexF = '\\min({}, {})'
    sym_str = 'Mini({}, {})'
    pycode = 'min({}, {})'


class Max(Plabel):

    fun_label = 'Maxi'
    arity = 2
    xtype = 'f2f'
    c_weight = 0.5
    tf = tf.maximum
    latex1 = '\\max'
    latexF = '\\max({}, {})'
    sym_str = 'Maxi({}, {})'
    pycode = 'max({}, {})'




class Observation(Plabel):

    arity = 0

    def __init__(self, obs_name, obs_type=float, obs_indizes=None, c_weight=0):
        self.fun_label = obs_name
        self.xtype = '2f' if obs_type == float else '2b'
        self.type =obs_type
        self.c_weight = c_weight
        family, temp = observation_get_family_and_time(obs_name, none_return=None)
        self.observation_family = family
        self.observation_index = temp
        self.obs_indizes = obs_indizes
        if temp is None:
            self.latex1 = f'{{\\text{{{family}}}}}'
            self.latexF = f'{{\\text{{{family}}}}}'
        else:
            self.latex1 = f'{{\\text{{{family}}}_{{{temp}}}}}'
            self.latexF = f'{{\\text{{{family}}}_{{{temp}}}}}'
        self.sym_str = obs_name
        self.pycode = obs_name


class FloatConstant(Plabel):
    arity = 0
    type = float
    xtype = '2f'

    def __init__(self, value, c_weight=0):
        self.c_weight = c_weight
        self.latex1 = f'{value:.3f}'
        self.latexF = f'{value:.3f}'
        self.sym_str = value
        self.pycode = value


class BoolConstant(Plabel):
    arity = 0
    type = bool
    xtype = '2b'

    def __init__(self, value, c_weight=0):
        self.c_weight = c_weight
        self.latex1 = f'{value}'
        self.latexF = f'{value}'
        self.sym_str = value
        self.pycode = value
