"""
plagih_tree contain a new implementation of trees that we use in genetic programming to display a program.
The old karoo "fintree" is replaced with, for now, "treer" in the code.
not all functions can use fintree for now and some fintree-functions require the old "fintree"
fintree splits the karoo fintree into the
- meta-info (fitness_train, parsimony, fintree-id, ...) and the
The core of the fintree, which "is" the fintree, is stored recursively
Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)

sfeh: write test that checks all operators for sympificytion (...+branch-combinations, and more?)
sfeh: use function-types (-> 'commutative'?)


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

    #sfeh:open combine the nodes with the sympy shizzle
    sfeh xxx input variables as locals? ?
    sfeh:open this is probably the reason for the capitalized class names in sympy: return eval(self, a)
    sfeh: I think we should get rid of sympy in the long term. A lot of problems are related to sympy.

    sfeh:sypyunification errors:
        - 'a and b', 'b and a'
        - 'And(a<2, a < 5)'
        - sympy.simplify('sign(-a)') -> -sign(a)

    sfeh:xxx sympy facttor (up/downfactor), so it adds stuff together
    sfeh:discus simplify/unify

Custom Operators /Functions/Nodes/Terminals/Nested:
    Any custom node must have a label subclassing NodeBase
    Also, make a case in sympy_to_nested to reconstruct trees from sympy expressions.
"""
import os

import sympy
import sympy.functions.elementary.piecewise  # sfeh: needs separate import?

from plagih.util import get_subclasses, PRECISION, DEBUG_DUMMY  # noqa

os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # https://github.com/tensorflow/tensorflow/issues/27023
import tensorflow as tf  # noqa (check if still required, tensorflow sends endless warnings)

# sfeh:check if this leads to tf warnings
tf.compat.v1.enable_eager_execution()  # sfeh possibly faster with disable


class NodeBase:
    symfun = None
    tflow = None
    xtype = None  # sfeh: will this b deprecated?

    def __new__(cls, *args, **kwargs):
        # if isinstance(cls, TerminalNode):
        #     cls.args = args
        obj = object.__new__(cls)
        # sfeh:open there are no instances of operator nodes
        # if args:
        #     obj.args = args
        # else:
        obj.args = args
        if issubclass(cls, TerminalNode):
            pass

        return obj

    def __str__(self):
        _str = self.__class__.__name__
        # if self.args:
        if issubclass(self.__class__, Operator):
            _childstr = ', '.join([str(a) for a in self.args])
            _str = f'{_str}({_childstr})'
        elif issubclass(type(self), TerminalNode):
            pass  # _str = f'{self.value}'
        else:
            raise
        return _str

    def __len__(self):
        """ONLY works, when args are there"""
        if issubclass(self.__class__, TerminalNode):
            return 1
        else:
            return 1 + sum([len(cc) for cc in self.args])

    def _sympy_(self, *args):  # -> sympy.Basic:
        _sym = self.symfun
        # _sym = _sym(*args)
        # childstr = [sympy.sympify(cc) for cc in args]
        # _sym = _sym(*childstr)

        # if isinstance(self, Operator):
        #     childstr = [sympy.sympify(cc) for cc in args]
        #     _sym = _sym(*childstr)
        #
        # elif isinstance(self, TerminalNode):
        #     return self.as_sym(self.args[0])
        #
        # else:
        #     raise NotImplementedError(f'sfeh:Specify exception. Class-type {type(self)}')

        return _sym

    @classmethod
    def get_sym(cls):
        _sym = cls.symfun
        return _sym

    def get_symstr(self):
        return self.symfun.__name__


class CustomOperator:
    # sfeh:xxx make an abstract class + mark all classes
    tflow = lambda *args: None
    symfun = lambda *args: None
    xtype = ((None, None), None)


class Operator(NodeBase):  # sfeh:xxx sympy.Function was here, also is_Function = True
    pass


class ChainableOp:
    """(Abstract) class for operators, that allow flexible arity (1-n args).
    Used e.g. while reconstructing trees from sympy expressions,
    to check whether it is possible to put more childs than planned into the node.

    The respected Operators are
    Add, Mul, Min, Max
    And, Or
    Piecewise/Ifte
    """
    chain_xtype = None


class MathOperator(Operator):
    # is_real = True
    # is_Boolean = False
    pass


class LogicOperator(Operator):
    # And, Or, Xor, Not
    # is_real/real = False
    # is_Boolean/bool = True
    pass


class RelationalOperator(Operator):
    pass


class AngleOperator(MathOperator):
    pass


class MinMaxBase(MathOperator):
    pass


class NoSymCapitalized:
    pass


class TerminalNode(NodeBase):  # sfeh sympy.Atom
    """
    Terminal nodes are leaf nodes which can not have children. e.g.:
    - constants (e.g. 2.3)
    - observations (e.g. b, aka data input)
    - user-functions (sfeh:open)
    """
    value = None

    # @abstractmethod
    # def __init__(self):
    #     pass
    #
    # def get_sym(self):
    #     return self.value


class Boolean(TerminalNode):
    # sfeh:discuss just for True/False?
    xtype = ((), bool)
    symfun = lambda *a: sympy.S.true if a[0] else ~sympy.S.true  # sympy.logic.boolalg.Boolean  # sfeh:discuss
    tflow = lambda arg: tf.constant(arg, dtype=tf.bool)

    # def __init__(self, value):
    #     self.value = sympy.S.true if value else ~sympy.S.true
    #
    # def get_sym(self):
    #     return self.value


class Float(TerminalNode):
    xtype = ((), float)
    symfun = lambda *a: sympy.Float(float(a[0]), PRECISION)
    tflow = lambda a: tf.constant(a, dtype=tf.float32)

    # def __init__(self, value):
    #     self.value = sympy.Float(value, PRECISION)


class Symbol(TerminalNode):
    """
    sfeh:discuss: should labels have a sign (-pos); can appear in observations
    This was used to deal with negative labels
        self.name = nlabl if nlabl[0] != '-' else nlabl[1:]
        sfeh:xxx option here for type float/bool
    sfeh:xxx how to set assumptions, Provide information such as real, integer, positive, range/interval
    """
    # as_sym = sympy.Symbol
    symfun = lambda *a: sympy.Symbol(a[0], real=True, imaginary=False)  # sfeh: real=/imaginary= faster.
    tflow = lambda a: tf.constant(a, dtype=tf.float32 if isinstance(a, float) else tf.bool)
    xtype = ((), float)

    # def __init__(self, value, **assumptions):
    #     self.value = sympy.Symbol(value, real=True, imaginary=False, **assumptions)
    #
    # def get_sym(self):
    #     return self.value


# class ExprCondPair(ChainableOp):
#     as_sym = sympy.functions.elementary.piecewise.ExprCondPair
#
#     # tflow = tf.where
#
#     def __init__(self, val, cond):
#         self.cond = cond
#         self.val = val


class Add(MathOperator, ChainableOp):
    symfun = sympy.Add
    tflow = tf.add
    xtype = ((float, float), float)
    chain_xtype = float


class InverseFraction(MathOperator):
    xtype = ((float, float), float)
    symfun = lambda a: sympy.Pow(a, sympy.S.NegativeOne)  # aka x**-1
    tflow = lambda a: tf.pow(a, -1)


class Pow(MathOperator):
    symfun = sympy.Pow
    tflow = tf.pow
    xtype = ((float, float), float)


class Abs(MathOperator):
    symfun = sympy.Abs
    tflow = tf.abs
    xtype = ((float,), float)


class Sign(MathOperator, NoSymCapitalized):
    # does not work in string, but irrelevant. sympy.simplify('sign(-a)') -> -sign(a)
    symfun = sympy.sign
    tflow = tf.sign
    xtype = ((float,), float)


class Log(MathOperator, NoSymCapitalized):
    symfun = sympy.log  # sfeh: Log isactually Ln (base e)
    tflow = tf.math.log
    xtype = ((float,), float)


class Cos(AngleOperator, NoSymCapitalized):
    symfun = sympy.cos
    tflow = tf.cos
    xtype = ((float,), float)


class Sin(AngleOperator, NoSymCapitalized):
    symfun = sympy.sin
    tflow = tf.sin
    xtype = ((float,), float)


class Tan(AngleOperator, NoSymCapitalized):
    # sfeh:discuss actually rename classes.
    # they do not have to match sympy expressions/classes
    symfun = sympy.tan
    tflow = tf.tan
    xtype = ((float,), float)


class Acos(AngleOperator, NoSymCapitalized):
    symfun = sympy.acos
    tflow = tf.acos
    xtype = ((float,), float)


class Asin(AngleOperator, NoSymCapitalized):
    symfun = sympy.asin
    tflow = tf.asin
    xtype = ((float,), float)


class Atan(AngleOperator, NoSymCapitalized):
    symfun = sympy.atan
    tflow = tf.atan
    xtype = ((float,), float)


class tanh(AngleOperator, NoSymCapitalized):
    symfun = sympy.tanh
    tflow = tf.tanh
    xtype = ((float,), float)


class Sinh(AngleOperator, NoSymCapitalized):
    symfun = sympy.sinh
    tflow = tf.sinh  # sfeh sinh, asinh
    xtype = ((float,), float)


class Cosh(AngleOperator, NoSymCapitalized):
    symfun = sympy.cosh
    tflow = tf.cosh  # sfeh acosh
    xtype = ((float,), float)


class Xor(LogicOperator, NoSymCapitalized):
    symfun = sympy.Xor
    tflow = tf.math.logical_xor
    xtype = ((bool, bool), bool)


class Not(LogicOperator):
    symfun = sympy.Not
    tflow = tf.logical_not
    xtype = ((bool,), bool)


class Eq(LogicOperator):
    # sfeh:debug Eq and Ne (), which also work for boolean inputs in sympy
    symfun = sympy.Eq
    tflow = tf.equal
    xtype = ((float, float), bool)


class Ne(LogicOperator):
    symfun = sympy.Ne
    tflow = tf.not_equal
    xtype = ((float, float), bool)


class Mul(MathOperator, ChainableOp):
    symfun = sympy.Mul
    tflow = tf.multiply
    xtype = ((float, float), float)
    chain_xtype = float


class And(LogicOperator, ChainableOp):
    symfun = sympy.And
    tflow = tf.logical_and
    xtype = ((bool, bool), bool)
    chain_xtype = bool


class Or(LogicOperator, ChainableOp):
    symfun = sympy.Or
    tflow = tf.logical_or
    xtype = ((bool, bool), bool)
    chain_xtype = bool


class ITE(LogicOperator):
    """sfeh:is this really required? currently not in use"""
    symfun = sympy.ITE
    tflow = lambda *args: tf.cond(args[0], true_fn=args[1], false_fn=args[2])
    xtype = ((bool, bool, bool), bool)


class Min(MinMaxBase, ChainableOp):
    symfun = sympy.Min
    tflow = tf.minimum
    xtype = ((float, float), float)
    chain_xtype = float


class Max(MinMaxBase, ChainableOp):
    symfun = sympy.Max
    tflow = tf.maximum
    xtype = ((float, float), float)
    chain_xtype = float


class Lt(RelationalOperator):
    symfun = sympy.Lt
    tflow = tf.less
    xtype = ((float, float), bool)


class Le(RelationalOperator):
    symfun = sympy.Le
    tflow = tf.less_equal
    xtype = ((float, float), bool)


class Gt(RelationalOperator):
    symfun = sympy.Gt
    tflow = tf.greater
    xtype = ((float, float), bool)


class Ge(RelationalOperator):
    symfun = sympy.Ge
    tflow = tf.greater_equal
    xtype = ((float, float), bool)


class Square(MathOperator):
    symfun = lambda a: sympy.Pow(a, 2)
    tflow = tf.square
    xtype = ((float,), float)


class Sub(MathOperator):
    tflow = tf.subtract
    xtype = ((float, float), float)
    symfun = lambda a, b: sympy.Add(a, -b)


class Ifte(Operator, ChainableOp):
    """Also class Piecewise"""
    tflow = tf.where
    xtype = ((bool, float, float), float)
    # symfun = lambda c, a, b: sympy.Piecewise((a, c), (b, True))
    symfun = lambda *args: sympy.Piecewise((args[1], args[0]), (args[2], True))
    chain_xtype = (float, bool)


Piecewise = Ifte


# def round_sympy(sexpr: sympy.Expr):
#     if sexpr.is_number:
#         return sympy.Float(sexpr, 1)
#     else:
#         return None  # XXX


class Round(MathOperator):
    """sfeh:XXX this does not work
    discuss:
    - sympy.Float(x, 1)  <-- sfeh:open
    - sympy.Integer(x)
    - sympy.N(x, 1)
        as_sym(Symbol('a'))
        as_sym(1.23)
        as_sym(sympy.Add(a, Symbol('a'))
    -> Write custom round function that evaluates only when is_number
    """
    # sfeh:xxx check conversion
    xtype = ((float,), float)
    symfun = lambda a: a.round(0) if a.is_number else sympy.Function('Rounddummy')(a)
    tflow = lambda a: tf.math.round(a, 1)


class Powrounded(Operator):
    tflow = lambda a, b: tf.pow(a, tf.round(b))
    symfun = lambda a, b: a**Round.symfun(b)  # todo sympy.Pow(a, Round(b))
    # sympy.lambdify  # sfeh:XXX
    xtype = ((float, float), float)


class Log1p(MathOperator):
    # https://docs.sympy.org/latest/modules/codegen.html#sympy.codegen.cfunctions.log1p
    xtype = ((float,), float)
    tflow = tf.math.log1p
    symfun = lambda a: sympy.log(a + 1)


class Div(MathOperator):
    tflow = tf.math.divide
    symfun = lambda a, b: sympy.Mul(a, 1 / b)
    xtype = ((float, float), float)


class Sqrt(MathOperator):
    """Capitalized class name, even though its a sympy function"""
    xtype = ((float,), float)
    symfun = sympy.sqrt  # same as: lambda a: sympy.Pow(a, sympy.S.Half)
    tflow = tf.sqrt


# class Divide_no_nan(Operator):
#     # class-name = 'Divide_no_nan'  # sfeh??
#     tflow = tf.math.divide_no_nan
#     as_sym = lambda a, b: sympy.Mul(a, )
#     xtype = ((float, float), float)


# class Usub(MathOperator, sympy.Function):
#     xtype = ((float,), float)
#     tflow = tf.negative
#     symfun = lambda a: sympy.Mul(a, -1)


class Clip(MinMaxBase, CustomOperator):
    # sfeh:open use this
    tflow = tf.clip_by_value
    symfun = lambda a, b, c: sympy.Min(sympy.Max(a, b), c)
    xtype = ((float, float, float), float)


class exp(MathOperator):
    tflow = tf.exp
    symfun = sympy.exp
    xtype = ((float,), float)


def expr_sympify(expr):
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

    sfeh: more sympy bugs
    sympify option evaluate=None does not work with custom functions
    """

    # loadable_ops_dict.update(eval_locals or {})  # sfeh:delete? irrelevant, cause every class defines the eval method?

    try:
        expr_sym = sympy.sympify(expr)
        # sfeh:xxx sympy expand, for evaluation probably
        # if DEBUG_DUMMY:
        #     try:
        #         # sfeh:XXX use or delete this!
        #         cartVel, cartPos = sympy.symbols('cartVel cartPos')
        #         expr_sym2 = expr_sym.factor()
        #         expr_sym22 = expr_sym.factor(cartVel + cartPos)
        #         expr_sym3 = expr_sym.expand()
        #         expr_sym32 = expr_sym.expand(cartVel + cartPos)
        #         if expr_sym2 != expr_sym22:
        #             pass
        #         if expr_sym != expr_sym2:
        #             print(f'COMPARE FACTOR:\n{expr_sym}\n{expr_sym2}')
        #         if expr_sym3 != expr_sym32:
        #             pass
        #         if expr_sym != expr_sym3:
        #             print(f'COMPARE EXTEND:\n{expr_sym}\n{expr_sym3}')
        #     except Exception as ex:
        #         print(f'sympify-factor ex: {expr_sym} {ex}')
        if expr_sym.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I, sympy.im):  # sfeh:discuss sympy.re
            raise ArithmeticError(f'Simplification failed: {expr_sym}')
        return expr_sym

    except ValueError as ex:
        # return 'nan'  # 'nan' always evaluates to nan. ALl nan bugs should be solved.
        raise ValueError(f'NaN in {ex}')
    except AttributeError as ex:
        # print(f'sfeh: This sympy bug happens, when sympifying "True": {ex}')
        raise
        # return sympy.true if expr else sympy.false

# sfeh:RANDOM IDEA when loading the data, check for every possible assupmtion. Better: load them manually.

# sympy_constants = {
#     sympy.numbers.Zero: 0,
#     sympy.numbers.Half: 0.5,
#     sympy.numbers.One: 1,
#     sympy.numbers.NegativeOne: -1,
#     sympy.numbers.Exp1: 2.71828182845904,  # sympy.numbers.Exp1().evalf(16)
#     sympy.numbers.Pi: 3.1415926535897932,  # sympy.numbers.Pi().evalf(17)
#     sympy.numbers.GoldenRatio: 1.61803398874989,  # sympy.numbers.GoldenRatio().evalf(16)
#     sympy.numbers.TribonacciConstant: 1.83928675521416,  # sympy.numbers.TribonacciConstant().evalf(16)
#     sympy.numbers.EulerGamma: 0.577215664901532,  # sympy.numbers.EulerGamma().evalf(16)
# }
# sfeh
#  sympy.numbers.Infinity: tensorflow.constant(np.Infinity),
#  sympy.numbers.NegativeInfinity: tensorflow.constant(-np.Infinity),
#  sympy.numbers.ComplexInfinity: tensorflow.complex(0, np.Infinity),sympy.numbers.ImaginaryUnit,
#  sympy.numbers.NumberSymbol
#  sympy.numbers.Catalan: tensorflow.constant(sympy.numbers.Catalan),
#  sympy.numbers.NaN: tensorflow.constant(sympy.numbers.NaN),

totf = {
    # sympy.Symbol: 'tf': lambda x: tf.cons
    sympy.Min: tf.minimum,
    sympy.Max: tf.maximum,
    sympy.Add: tf.add,
    sympy.Mul: tf.multiply,
    sympy.Pow: tf.pow,
    sympy.Abs: tf.abs,

    sympy.Not: tf.logical_not,
    sympy.And: tf.logical_and,
    sympy.Or: tf.logical_or,
    sympy.Xor: tf.math.logical_xor,

    sympy.Equality: tf.equal,
    sympy.Unequality: tf.not_equal,
    sympy.GreaterThan: tf.greater_equal,
    sympy.StrictGreaterThan: tf.greater,
    sympy.LessThan: tf.less_equal,
    sympy.StrictLessThan: tf.less,
    # sympy.N: tf.math.round,  # incorrect, sympy.N(x, 1) is right
    sympy.log: tf.math.log,
    sympy.cos: tf.cos,
    sympy.cosh: tf.cosh,
    sympy.sin: tf.sin,
    sympy.sinh: tf.sinh,
    sympy.tan: tf.tan,
    sympy.tanh: tf.tanh,
    sympy.acos: tf.acos,
    sympy.asin: tf.asin,
    sympy.atan: tf.atan,
    sympy.sign: tf.sign,
    # The real Part
    sympy.re: lambda a: tf.convert_to_tensor(a, dtype=tf.dtypes.float32),  # sfeh sympy-gotcha, comes up randomly
    # todo sfeh, shouldnt this automatically be acccepted as Round class?
    sympy.Function('Rounddummy'): tf.round,
    # Round: 'todo_OMGGOAWAYYY',
    sympy.exp: tf.exp,  # sfeh this occurs randomly...
    sympy.ITE: tf.cond
}


def sympy_to_tensorflow(expr, tensor_dict):
    """
    - check terminal-node
    -- check symbol
    --

    Bugs/gotchas:

    sympify('True')             -> True
    sympify('1')==True          ->
    sympify('~(True)')          -> -2
    sympify('~(False)')         -> -1
    sympify('a <= Min(a, b)')   ->

    sympy.logic.boolalg.ITE has only boolean inputs
    evaluate=None/false

    sympy.logic.boolalg.ITE is not If-then-else
    sympy.cosh can emerge out of sin-stuff
    sympy.sympify('True')->True is no sympy expression anymore
    # sfeh:bug gotcha sympy.re comes up randomly
    """
    # shape = tensors[list(tensors.keys())[0]].get_shape()  # sfeh:open:workaround:

    # ==Bug-handling==  sympy.sympify('True')->True is no sympy expression anymore
    if isinstance(expr, bool):  # e.g. '1'
        return tf.constant(expr, dtype=tf.dtypes.bool)
    if isinstance(expr, (bool, sympy.logic.boolalg.BooleanAtom)):
        expr = True if isinstance(expr, sympy.logic.boolalg.BooleanTrue) else False  # sfeh:collect sympy bug gotcha bug
        return tf.constant(expr, dtype=tf.dtypes.bool)

    # the following lines are not required, if sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    # ==Terminal nodes==
    elif expr.is_Atom:
        if expr.is_Symbol:
            # result = feed_dict[expr.name]  # sfeh:discuss placeholder

            # result = tf.compat.v1.placeholder(tf.bool, name=str(expr))
            # sfeh:runtime?
            result = tf.constant(tensor_dict[str(expr)], dtype=tf.bool if expr.assumptions0.get('bool') else tf.float32)
            return result

        else:
            expr_eval = expr.evalf()  # standard 15 digits
            if expr.is_Boolean:
                return tf.constant(expr_eval, dtype=tf.dtypes.bool)
            elif expr.is_number:  # is_float does not match int
                return tf.constant(float(expr_eval), dtype=tf.dtypes.float32)
            else:
                print(f'XXX What happened here? {expr}')

    else:  # Operator # len(expr.args) > 0:  # sfeh: line can be removed or replaced
        if isinstance(expr, sympy.Piecewise):
            args_reversed = list(expr.args[::-1])  # tuples to list, reverse: last tuple must be nested the deepest
            args_reversed = [[sympy_to_tensorflow(xx, tensor_dict) for xx in list(i)] for i in args_reversed]
            otherwise = args_reversed[0][0]  # the last "True" condition
            for cet in args_reversed[1:]:
                otherwise = tf.where(cet[1], cet[0], otherwise)
            return otherwise
        try:
            tf_fun = totf[type(expr)]
        except KeyError:
            # sfeh:debug can this work??
            #   - im(Rounddummy(cartVel))
            tf_fun = type(expr).tflow
            # sfeh:idea exception, try to map sympy to tf function with same name (sympy.cos -> tf.cos)

        tf_args = [sympy_to_tensorflow(a, tensor_dict) for a in expr.args]
        # SFEH:Missing and Problems:
        #   - Exception: eval-ex: type object 'cosh' has no attribute 'tflow'
        #   - AttributeError: type object 'Rounddummy' has no attribute 'tflow'
        try:
            result = tf_fun(*tf_args)  # fits, if the arguments match the expected arguments exactly Add(a, b)
        except TypeError:
            result = tf_args.pop()  # only commutative arity-2 functions here (Add, Mul, Max, Min)
            while tf_args:
                # sfeh:optimization
                result = tf_fun(result, tf_args.pop())
        return result
    raise NotImplementedError(f'Cannot convert {expr}')


# class ExperimentalBaseTree(NodeBase):
#     """
#     Is currently not in use and Experimental
#     states? sfeh:discuss
#     [None]: not set
#     [0]:    evolution/construction/build mode (potentially missing leaf nodes)
#     [1]:    structurally complete/finalized branch (node_depths correct, node_id set, ...)
#     [2]:    root-correct structure
#     [3]:    including meta-data (fitness_train, complexity)
#
#     # sfeh:discuss set tree depth at the end or so
#     # sfeh:xxx set NodeBase as metatype and make this class (ExpressionTree, or so...) a new thing
#     """
#
#     args = []
#     is_fix = False  # sfeh:xx here?
#
#     def __new__(cls, subcls, *args, **kwargs):
#         """
#         Why new instead of init?
#         -> https://docs.sympy.org/latest/tutorials/intro-tutorial/manipulation.html
#         " SymPy classes make heavy use of the __new__ class constructor, which, unlike __init__,
#         allows a different class to be returned from the constructor."
#         """
#         # if isinstance(cls, Operator):
#         #     pass
#         # elif isinstance(cls, TerminalNode):
#         #     if isinstance(cls, Symbol):
#         #         cls.fam, cls.time_index, _ = observation_get_family_and_time(args, none_return=None)
#         #         cls.index_minmax = None
#
#         obj = object.__new__(subcls)
#         obj.args = [arg for arg in args]
#         return obj
#
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls.args = kwargs.get('args', [])
#         cls.is_fix = kwargs.get('is_fix', [])
#
#     def __str__(self):
#         return self.get_str_recursive()
#
#     def get_str_recursive(self):
#         if isinstance(self, Operator):
#             childstr = ', '.join([str(x) for x in self.args])
#             _str = f"{self}({childstr})"
#         elif isinstance(self, TerminalNode):
#             _str = f'{self.value[0]}'
#         else:
#             raise NotImplementedError(f'sfeh:Specify exception. Class-type {type(self)}')
#
#         return _str
#
#     def __repr__(self):
#         return self.get_repr_recursive()
#
#     def get_repr_recursive(self):
#
#         _isfix = ', is_fix=True' if self.is_fix else ''
#         if isinstance(self, Operator):
#             childstr = ', '.join([repr(x) for x in self.args])
#         elif isinstance(self, TerminalNode):
#             childstr = self.args[0]
#         else:
#             raise NotImplementedError(f'sfeh:Specify exception. Class-type {type(self)}')
#
#         return f"{self.__class__.__name__}({childstr}{_isfix})"
#
#     def __len__(self):
#         return 1 + sum([len(cc) for cc in self.args])
#
#     def get_nclass(self):
#         return self.__class__.__name__


if __name__ == '__main__':

    ns = {
        'a': sympy.Symbol('a', real=True),
        'b': sympy.Symbol('b', real=True),
        'c': sympy.Symbol('c', bool=True),
        'd': sympy.Symbol('d', bool=True),
    }

    # tensors = {
    #     'a': tf.constant([1.0, 2, 3, 4, 5, 6], dtype=tf.dtypes.float32),
    #     'b': tf.constant([-1.0, -2, -3, -4, -5, -6], dtype=tf.float32),
    #     'c': tf.constant([True, False, True, False, True, False], dtype=tf.dtypes.bool),
    #     'd': tf.constant([True, True, True, True, True, True], dtype=tf.dtypes.bool)
    # }` # sfeh are tensors actually better??

    tensors = {
        'a': [1.0, 2, 3, 4, 5, 6],
        'b': [-1.0, -2, -3, -4, -5, -6],
        'c': [True, False, True, False, True, False],
        'd': [True, True, True, True, True, True]}

    tst = [
        '5', '1', '0', '0.5', '-1', 'True', 'False',
        'c & True', 'c | False', '~c',
        'a<1', 'a<b', 'a<=b', 'a>=b', 'a>b', 'a==b', 'a!=b', 'a',
        # 'oo', 'zoo', 'I',
        'a + 1', 'a + 2', 'a*2', 'a - 2', 'a/2', 'a < 2', 'a**2', '2/a',
        'a*b*2', 'a+b+a+2+4', 'Min(a, b, 3)', 'Max(a, b, 4, a**2, a+b)', 'a<3',
        'Piecewise((a, c), (b, d), (a+b, True))',
        'Eq(4, 4.0)',
        # 'Square((Min(-2.176629, b) - Abs(a)))', 'Round(-123.333334234) + Round(b)',
        # 'Ifte(c, 1, 2)', '1 < Max(2, Ifte(1 < a, 1, 1))', 'Max(a+1, 2**(5-b))'
    ]
    tst_custom = [
        'Ifte(a, b, c)',
        '(((0.326675 * b_2) - c_9) + (Ifte((-c_9 < b_5), c_7, Ifte((Square(Gain_6) < Max(a_2, Ifte((c_9 < '
        'c_4), -Gain_3, Gain_5))), c_9, c_4))))',
        'Max(a, 2)',
        'Min(b, b)',
        'Ifte(True, a, 3)',
        'sin(asin(b))',
        'And(False, True)',
        'Add(-1.490149, 14.0)',
        'Ifte(False, (3+a), 3)',
        'Eq(4, 4.0)',
        'Lt(a, a)',
        'Or(Ne(False, False), False)',
        'sqrt(5 * a)',
        'Multiply(log(acos(-0.212976)), asin(2))',
        'Not(False)',
        'acos(0.5)',
        # 'Round(1.2345)',
        'Min(Ifte(True, Multiply(a, 20.0), acos(-0.5)), 1)',
        'Div(Add(13.159398, 19.284178), 1)',
        'Pow(a, b)',
        'Add(-2, Min(Ifte(True, 1, b), 8))',
        '(Or(True, True) & c)',
        'Ifte(And(False, True), b, 0.046948)',
        'Ifte(Ne(True, Lt(Sub(a, 2), 1)), 1, 2)',
        'cos(tan(Square(Multiply(Add(Round(Ifte(Ne(Ge(b, 15), True), 7, Sub(a, 16.5))), 5), 4))))'
    ]
    xxx_problems = ['Ifte(Lt(Ifte(Eq(Min(b, 1), 3), Max(a, b), b), 0), 0, 2)']


    def test_basic_tfconversion():
        # sfeh:open tests

        for t in tst:
            x = sympy.sympify(t, locals=ns)
            x = sympy_to_tensorflow(x, tensors)

            print(f'{t} \t{x}')


    def test_sympify():
        print('Running sympify example')

        for x in tst + tst_custom:
            sx = expr_sympify(x)
            print(sx)


    def print_relevant_subclasses():

        st = {}
        for x in get_subclasses(Operator):
            try:
                st[f'sympy.{x.symfun.__name__}'] = x.__name__  # x.tflow.__name__
            except AttributeError as ex:
                print(f'Could not get {x}: {ex}')
                # st[x.__name__] = x.__name__
                pass
        st = ', '.join([f"{k}: {v}" for k, v in st.items()])
        print(f'sympy_to_node = {{{st}}}')


    # test_basic_tfconversion()  # sfeh all tests
    # test_sympify()

    x = Add(childs=[Symbol('a'), Mul(childs=[2, 3])])
    n1 = Float
    n2 = Symbol
    n3 = Boolean
    n4 = Add()
    # print(n1, n2, n3, n4)
    print(len(n1), len(n4))

    # for _subc in get_subclasses(BaseTree):
    #     if  in _subc.__bases__:
    #         # print(f'ignoring {_subc}')
    #         pass
    #     else:
    #         print(f'{_subc.__name__}')
    #
    # print(type(RelationalOperator))
    # print_relevant_subclasses()
