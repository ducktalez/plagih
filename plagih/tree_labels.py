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
from typing import Callable

import sympy
# import sympy.functions.elementary.piecewise  # sfeh: needs separate import?
from sympy.functions.elementary.piecewise import ExprCondPair

from plagih.util import get_subclasses, FLOAT_PRECISION, DEBUG_DUMMY  # noqa

os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # https://github.com/tensorflow/tensorflow/issues/27023
import tensorflow as tf  # noqa check if ignoring warnings still required (tensorflow sends endless warnings)

tf.compat.v1.disable_eager_execution()


# tf.compat.v1.enable_eager_execution()  # sfeh possibly faster with disable


class Label:
    symfun = None
    # tflow = None  # Not required anymore
    xtype = None

    def __new__(cls, *args, **kwargs):
        """Why use new as init method? -> Allows returning an instance of a different class"""

        obj = object.__new__(cls)
        obj.args = args
        if issubclass(cls, Terminal):
            pass

        return obj

    def __str__(self):
        _str = self.as_str()
        return _str

    def as_str(self):
        """"""
        _str = self.__class__.__name__
        _str = self.__class__
        if issubclass(self.__class__, (OperatorArity, OperatorChained)):
            _childstr = ', '.join([a.as_str() for a in self.args])
            _str = f'{_str}({_childstr})'
        elif issubclass(type(self), Terminal):
            pass  # _str = f'{self.value}'
        else:
            raise
        return _str

    def __len__(self):
        """ONLY works, when args are there"""
        if issubclass(self.__class__, Terminal):
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
        #     return self.symfun(self.args[0])
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

    @classmethod
    def get_child_xts(cls):
        return cls.xtype[0]


# class ArityNode(Label):
#     pass


class CustomOperator:
    # sfeh:xxx make an abstract class + mark all classes
    tflow = lambda *args: None
    symfun = lambda *args: None
    xtype = ((None, None), None)


class BaseOperator(Label):
    pass


class OperatorArity(BaseOperator):  # sfeh:xxx sympy.Function was here, also is_Function = True
    pass


class OperatorChained(BaseOperator):
    # no xtype, only input type
    # no tflow, separate handling in totf-function
    # Piecewise, AddChain, MulChain, MinChain, MaxChain, AndChain, OrChain
    # childs_min_max = [1, 5]
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


class MathOperator(OperatorArity):
    # is_real = True
    # is_Boolean = False
    pass


class LogicOperator(OperatorArity):
    # And, Or, Xor, Not
    # is_real/real = False
    # is_Boolean/bool = True
    pass


class RelationalOperator(OperatorArity):
    pass


class AngleOperator(MathOperator):
    pass


class MinMaxBase(MathOperator):
    pass


class NoSymCapitalized:
    """Does nothing, but maybe its good to know, which OG-sympy classes are lower case"""
    pass


class Terminal(Label):  # sfeh sympy.Atom
    """Terminal nodes are leaf nodes which can not have children. e.g.:
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


class Boolean(Terminal):
    # sfeh:discuss just for True/False?
    xtype = ((), bool)
    symfun = lambda *a: sympy.S.true if a[0] else ~sympy.S.true  # sympy.logic.boolalg.Boolean  # sfeh:discuss
    tflow = lambda arg: tf.constant(arg, dtype=tf.bool)

    # def __init__(self, value):
    #     self.value = sympy.S.true if value else ~sympy.S.true
    #
    # def get_sym(self):
    #     return self.value


class Number(Terminal):
    xtype = ((), float)
    symfun = lambda *a: sympy.Float(float(a[0]), FLOAT_PRECISION)
    # symfun = lambda *a: sympy.Rational(float(a[0]), FLOAT_PRECISION)
    tflow = lambda a: tf.constant(a, dtype=tf.float32)

    # def __init__(self, value):
    #     self.value = sympy.Float(value, FLOAT_PRECISION)


class Symbol(Terminal):
    """
    sfeh:discuss: should labels have a sign (-pos); can appear in observations
    This was used to deal with negative labels
        self.name = nlabl if nlabl[0] != '-' else nlabl[1:]
        sfeh:xxx option here for type float/bool
    sfeh:xxx how to set assumptions, Provide information such as real, integer, positive, range/interval
    """
    symfun = lambda *a: a[0]
    # symfun = lambda *a: sympy.Symbol(a[0], real=True, imaginary=False)  # sfeh: real=/imaginary= faster.

    tflow = lambda a: tf.constant(a, dtype=tf.float32 if isinstance(a, float) else tf.bool)
    xtype = ((), float)


class Add(MathOperator, ChainableOp):
    symfun = sympy.Add
    tflow = tf.add
    xtype = ((float, float), float)
    chain_xtype = float


# sfeh:idea expr.as_coeff_Mul() expr.as_numer_denom()
#     import sympy as sp
#     x, eps = sp.symbols('x E')
#     expr = eps * x**3 - x**2 + 2 + 3 * x * eps**(-2)
#     a = sp.Poly(expr, eps).coeffs()
#     b = sp.Poly(expr, eps).coeffs()
#     c = sp.Poly(expr, eps).coeffs()


class DivFraction(MathOperator):
    """x**-1
    aka InverseFraction"""
    xtype = ((float, float), float)
    symfun = lambda a: sympy.Pow(a, sympy.S.NegativeOne)
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
    #
    # sfeh
    # def __new__(cls, *args, **kwargs):
    #     return Node(cls, args)


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


class Ifte(OperatorArity, ChainableOp):
    """Also class Piecewise"""
    tflow = tf.where
    xtype = ((bool, float, float), float)
    symfun = lambda *args: sympy.Piecewise((args[1], args[0]), (args[2], True))
    chain_xtype = (float, bool)


# class RoundDummy(sympy.Function):
#     """Exists, as the Round-class must not be simplified
#     Only used, if a symbol (aka variable) is rounded (see class Round)
#     sfeh check if this is required
#     """
#     pass


class Round(MathOperator):
    """sfeh:XXX this does not work
    discuss:
    - sympy.Float(x, 1)  <-- sfeh:open
    - sympy.Integer(x)
    - sympy.N(x, 1)
        symfun(Symbol('a'))
        symfun(1.23)
        symfun(sympy.Add(a, Symbol('a'))
    -> Write custom round function that evaluates only when is_number
    """
    # sfeh:xxx check conversion
    xtype = ((float,), float)
    # symfun = lambda a: a.round(0) if a.is_number else RoundDummy(a)
    symfun: Callable[[sympy.Expr], sympy.Expr] = lambda a: a.round(0) if a.is_number else Round(a)  # sfeh (next line)
    # this is here to hint the type, as sympy will throw a warning otherwise, leading to this
    # https://docs.sympy.org/latest/explanation/active-deprecations.html#non-expr-args-deprecated
    tflow = lambda a: tf.math.round(a, 1)


class Reversable:
    """All the Operators which
    eg:
    Powrounded (If Pow + (child1 is round .../or child1 % 1)) -> """

    def revert_when(self):
        # sfeh:idea
        # something like
        # reverse_me = lambda labl, chld: Powrounded if (labl==Pow and chld[1]==Round) else Round
        pass


class Powrounded(OperatorArity):
    tflow = lambda a, b: tf.pow(a, tf.round(b))
    symfun = lambda a, b: sympy.Pow(a, Round.symfun(b))  # symfun = lambda a, b: a**Round.symfun(b)
    xtype = ((float, float), float)


class Log1p(MathOperator):
    # https://docs.sympy.org/latest/modules/codegen.html#sympy.codegen.cfunctions.log1p
    xtype = ((float,), float)
    tflow = tf.math.log1p  # sfeh: tflow is actually never used...
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
#     symfun = lambda a, b: sympy.Mul(a, )
#     xtype = ((float, float), float)


class Usub(MathOperator, sympy.Function):
    xtype = ((float,), float)
    tflow = tf.negative
    symfun = lambda a: sympy.Mul(a, -1)

    def __len__(self):
        """sfeh:currently not used"""
        return 0


class Clip(MinMaxBase, CustomOperator):
    # sfeh:open use this
    tflow = tf.clip_by_value
    symfun = lambda a, b, c: sympy.Min(sympy.Max(a, b), c)
    xtype = ((float, float, float), float)


class exp(MathOperator):
    tflow = tf.exp
    symfun = sympy.exp
    xtype = ((float,), float)


"""SFEH the following are operators that are to be handled completely different"""


class TerminalDummy(Label):
    @classmethod
    def get_child_xts(cls):
        return cls.xtype[0]


class ExprCondPair(TerminalDummy):
    """sfeh:discuss
    The only purpose is to wrap the results for a Node-structure, where every Node has childs with other nodes"""
    symfun = sympy.functions.elementary.piecewise.ExprCondPair
    xtype = ((float, bool), float)
    # tflow = tf.where
    expr_dmy = 'ExprCondPair'

# sfeh: remove?
# class Piecewise(ChainableOp):
#     """sfeh:discuss: the only Operator, which has tuples as input"""
#     symfun = sympy.Piecewise
#     # ogclass = Ifte
#     # xtype = ((float, bool), float)
#     xtype = ((ExprCondPair,), float)


# sfeh:discuss: there should probably be structural nodes and Operator nodes
# sfeh:discuss: Min/Max is just a ordeded list. ->taking element 1, -1, ...


def sym_check(expr_sym):
    if expr_sym.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I, sympy.im):  # sfeh:discuss sympy.re
        raise ArithmeticError(f'Simplification failed: {expr_sym}')
    return expr_sym


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
        sym_check(expr_sym)
        return expr_sym

    except ValueError as ex:
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
    # RoundDummy: tf.round,
    sympy.exp: tf.exp,  # sfeh this occurs randomly...
    sympy.ITE: tf.cond
}


# def sympy_to_tensorflow(expr_sy, d_tensors):
#     """
#     - check terminal-node
#     -- check symbol
#     --
#
#     Bugs/gotchas:
#
#     sympify('True')             -> True
#     sympify('1')==True          ->
#     sympify('~(True)')          -> -2
#     sympify('~(False)')         -> -1
#     sympify('a <= Min(a, b)')   ->
#
#     sympy.logic.boolalg.ITE has only boolean inputs
#     evaluate=None/false
#
#     sympy.logic.boolalg.ITE is not If-then-else
#     sympy.cosh can emerge out of sin-stuff
#     sympy.sympify('True')->True is no sympy expression anymore
#     # sfeh:bug gotcha sympy.re comes up randomly
#     """
#     # shape = tensors[list(tensors.keys())[0]].get_shape()  # sfeh:open:workaround:
#
#     # ==Bug-handling==  sympy.sympify('True')->True is no sympy expression anymore
#     if isinstance(expr_sy, bool):  # e.g. '1'
#         return tf.constant(expr_sy, dtype=tf.dtypes.bool)
#     if isinstance(expr_sy, (bool, sympy.logic.boolalg.BooleanAtom)):
#         expr_sy = True if isinstance(expr_sy,
#                                      sympy.logic.boolalg.BooleanTrue) else False  # sfeh:collect sympy bug gotcha bug
#         return tf.constant(expr_sy, dtype=tf.dtypes.bool)
#
#     # the following lines are not required, if sympy filters for bad expressions earlier
#     if expr_sy.is_imaginary or expr_sy.is_infinite:
#         raise ValueError(f'Cannot convert this to Tensorflow: {expr_sy}')
#
#     # ==Terminal nodes==
#     elif expr_sy.is_Atom:
#         if expr_sy.is_Symbol:
#             # result = feed_dict[expr.name]  # sfeh:discuss placeholder
#
#             # result = tf.compat.v1.placeholder(tf.bool, name=str(expr))
#             # sfeh:runtime?
#             result = tf.constant(d_tensors[str(expr_sy)],
#                                  dtype=tf.bool if expr_sy.assumptions0.get('bool') else tf.float32)
#             return result
#
#         else:
#             expr_eval = expr_sy.evalf()  # standard 15 digits
#             if expr_sy.is_Boolean:
#                 return tf.constant(bool(expr_eval), dtype=tf.dtypes.bool)
#             elif expr_sy.is_number:  # is_float does not match int
#                 return tf.constant(float(expr_eval), dtype=tf.dtypes.float32)
#             else:
#                 raise NotImplementedError(f'Atom, but no bool or number? {expr_sy}')
#
#     else:  # Operator # len(expr.args) > 0:  # sfeh: line can be removed or replaced
#         if isinstance(expr_sy, sympy.Piecewise):
#             args_reversed = list(expr_sy.args[::-1])  # tuples to list
#             # whY WOULD ONE    reverse the args? --> NOT REVERSER
#             # reverse: most specific (/last) to lowest,  tuple must be nested the deepest node:
#             try:
#                 args_reversed = [[sympy_to_tensorflow(xx, d_tensors) for xx in list(ii)] for ii in args_reversed]
#             except TypeError as ex:
#                 print(f'lolololol {ex}')
#                 # [(2.07, True), (8.0, ITE(cartVel <= 0.34, 0.755*cartPos > 2.0, cartPos/(cartVel**2*sign(cartVel) + 0.866) > 2.0))]
#                 # (8.0, ITE(cartVel <= 0.34, 0.755*cartPos > 2.0, cartPos/(cartVel**2*sign(cartVel) + 0.866) > 2.0))
#                 # args_reversed = [[sympy_to_tensorflow(xx, d_tensors) for xx in list(ii)] for ii in args_reversed]
#                 # args_reversed = [[sympy_to_tensorflow(xx, d_tensors) for xx in list(ii)] for ii in args_reversed]
#                 raise
#
#             otherwise = args_reversed[0][0]  # the last "True" condition
#             for cet in args_reversed[1:]:
#                 otherwise = tf.where(cet[1], cet[0], otherwise)
#             return otherwise
#
#         elif isinstance(expr_sy, sympy.ITE):
#             raise NotImplementedError(f'Sfeh: sympy.ITE not yet implemented, occurs as side-effect of boolean logic')
#
#         tf_fun = totf[type(expr_sy)]
#         # try:
#         #     tf_fun = totf[type(expr)]
#         # except KeyError:
#         #     # sfeh:debug can this work??
#         #     #   - im(Rounddummy(cartVel))
#         #     tf_fun = type(expr).tflow  # sfeh:debug-01.02 why does im come up here? (mut_br) im(Rounddummy(cartPos))
#         #     # sfeh:idea exception, try to map sympy to tf function with same name (sympy.cos -> tf.cos)
#
#         tf_args = [sympy_to_tensorflow(a, d_tensors) for a in expr_sy.args]
#         # SFEH:Missing and Problems:
#         #   - Exception: eval-ex: type object 'cosh' has no attribute 'tflow'
#         #   - AttributeError: type object 'Rounddummy' has no attribute 'tflow'
#         try:
#             result = tf_fun(*tf_args)  # fits, if the arguments match the expected arguments exactly Add(a, b)
#         except TypeError:
#             result = tf_args.pop()  # only commutative arity-2 functions here (Add, Mul, Max, Min)
#             while tf_args:
#                 # sfeh:optimization
#                 try:
#                     result = tf_fun(result, tf_args.pop())
#                 except Exception as ex:
#                     result = tf_fun(result, tf_args.pop())
#         return result
#     raise NotImplementedError(f'Cannot convert {expr_sy}')  # noqa: unreachable code, but was reached often while dev


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

    # def test_basic_tfconversion():
    #     # sfeh:open tests
    #
    #     for t in tst:
    #         x = sympy.sympify(t, locals=ns)
    #         x = sympy_to_tensorflow(x, tensors)
    #
    #         print(f'{t} \t{x}')


    def test_sympify():
        print('Running sympify example')

        for x in tst + tst_custom:
            sx = expr_sympify(x)
            print(sx)


    def print_relevant_subclasses():

        st = {}
        for x in get_subclasses(OperatorArity):
            try:
                st[f'sympy.{x.symfun.__name__}'] = x.__name__  # x.tflow.__name__
            except AttributeError as ex:
                print(f'Could not get {x}: {ex}')
                # st[x.__name__] = x.__name__
                pass
        st = ', '.join([f"{k}: {v}" for k, v in st.items()])
        print(f'sympy_to_node = {{{st}}}')


    def check_subclasses():

        for x in get_subclasses(OperatorArity):
            if len(x.__subclasses__()) > 0:
                # print('vdsfg', x)
                pass
            else:
                print(x.__name__)
                # pass


    check_subclasses()

    # test_basic_tfconversion()  # sfeh all tests
    # test_sympify()

    # x = Add(childs=[Symbol('a'), Mul(childs=[2, 3])])
    # n1 = Float
    # n2 = Symbol
    # n3 = Boolean
    # n4 = Add()
    # # print(n1, n2, n3, n4)
    # print(len(n1), len(n4))

    # for _subc in get_subclasses(BaseTree):
    #     if  in _subc.__bases__:
    #         # print(f'ignoring {_subc}')
    #         pass
    #     else:
    #         print(f'{_subc.__name__}')
    #
    # print(type(RelationalOperator))
    # print_relevant_subclasses()

#######################################
# sfeh:idea check these options
#     import sympy
#     a, b = sp.symbols('a b')
#     expr = b - a**2 * a**3 + 2 + 3 * a * b**(-2)
#     [expr,
#      expr.as_expr(),
#      expr.as_poly(),
#      expr.as_base_exp(),
#      expr.as_coeff_add(),
#      expr.as_coeff_Add(),
#      expr.as_coeff_mul(),
#      expr.as_coeff_Mul(),
#      # expr.as_coeff_exponent(),
#      # expr.leadterm(),  # not working
#      # expr.subs(),
#      expr.as_coefficients_dict(),
#      expr.as_content_primitive(),
#      expr.as_dummy(),
#      expr.as_expr(),
#      expr.as_leading_term(),
#      expr.as_numer_denom(),
#      expr.as_two_terms(),
#      expr.as_independent(),
#      expr.expand(),
#      expr.factor(),
#      expr.assumptions0,
#      expr.normal(),
#      expr.nsimplify(),
#      # expr.extract_multiplicatively(),
#      # USEFUL
#      expr.atoms(),  # for leaf nodes
#      ]
###########################
