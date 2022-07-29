"""
sfeh: I think we should get rid of sympy in the long term. A lot of problems are related to sympy.
sfeh:open this is probably the reason for the capitalized class names in sympy: return eval(self, a)

This class enrichens the python-core 'sympy'.
Sympy is used to reduce the functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.

- implementing missing functions in sympify, e. g. 'if a then b else c'.
- All number-related functions must have set
    is_real = True
    otherwise: '1 < Maxi(2, Ifte(1 < a, 1, 1))' will crash. (< operators only work on non-complex - aka real numbers)
    check for is_number if required.

- Classes must currently have the exact same name as their occurance (Ifte -> Ifte, not ifte or so)
    This is because when None is returned, the class name gets replaced at the function. could be solved, but why though :P

Useful information:
- These variables are set for every sympy object and thus can be tested, e.g. a.is_Boolean
    # To be overridden with True in the appropriate subclasses
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    is_MatAdd = False
    is_MatMul = False

    #sfeh discussion: sympify hat option rational=True, was aus Zahlen Brüche macht

    #sfeh:open combine the nodes with the sympy shizzle

    sfeh xxx inpput variables as locals? Provide information such as real, integer, positive, range/interval?
"""
import re

from sympy import Function, sympify, symbols, simplify  # , unify

# from sympy.core.numbers import ComplexInfinity
from plagih.util import DEBUG_DUMMY


class Ifte(Function):
    """
    my_sympify('Ifte(a, b, c)')
    """
    nargs = 3
    is_Function = True
    is_real = True

    @classmethod
    def eval(cls, a, b, c):
        if a.is_Boolean:
            return b if a else c  # search for 'gotcha' in https://docs.sympy.org/latest/_modules/sympy/core/relational.html
        else:
            return

    def _sympy_(self, *args):
        return eval(self, *args)


class Mini(Function):
    """
    Minimum function with arity-2.
    min() does not work (for now), as nested min() get accumulated, which leads to problems creating the tf-graph
    """
    nargs = 2
    is_Function = True
    is_real = True

    @classmethod
    def eval(cls, a, b):

        # if (a < b) == True or (a < b) == False: # first solution
        if a.is_number and b.is_number:  # must be real for a comparison
            return a if a <= b else b
        elif a == b:  # recent update for Mini(cartVel, cartVel)
            return a
        else:
            return

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Maxi(Function):
    """
    """
    nargs = 2
    is_Function = True
    is_real = True

    @classmethod
    def eval(cls, a, b):

        # if (a < b) == True or (a < b) == False: # first solution, was working.
        if a.is_number and b.is_number:
            return a if a > b else b
        elif a == b:  # recent update for Mini(cartVel, cartVel)
            return a
        else:
            return

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Andb(Function):
    """
    """
    nargs = 2
    is_Function = True

    @classmethod
    def eval(cls, a, b):

        # if (a == True or a == False) and (b == True or b == False):
        # sfeh xxx
        if a.is_Boolean and b.is_Boolean:
            return a and b
        elif a == b:
            return a
        else:
            return None

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Orb(Function):
    """
    """
    nargs = 2
    is_Function = True

    @classmethod
    def eval(cls, a, b):

        #
        # if ((a == True or a == False) and (b == True or b == False)) == (sympify(a).is_Boolean and sympify(b).is_Boolean):
        #     pass
        # else:
        #     raise

        # if (a == True or a == False) and (b == True or b == False):  # this works guaranteed
        if a.is_Boolean and b.is_Boolean:
            return a and b
        elif a == b:
            return a
        else:
            return None

    def _sympy_(self, a, b):
        return eval(self, a, b)

    # def _sympy_(self, a, b):
    #     """
    # sfeh:idea
    #     """
    #     return eval(a or b)


class Notb(Function):
    """
    Not (boolean)
    Problem was:
    - Not(a) evaluates to ~a
    - not(a<2) evaluates to nan
    """
    nargs = 1
    is_Function = True

    @classmethod
    def eval(cls, a):
        if sympify(a).is_Boolean:  # sfeh sympify.is_xxx here seems dumb
            return not a
        else:
            return None

    def _sympy_(self, a):
        return eval(self, a)


class Square(Function):
    """

    """
    nargs = 1
    is_Function = True
    is_real = True

    @classmethod
    def eval(cls, a):
        if sympify(a).is_real:
            return a ** 2  # see
        else:
            # print('sympy debug, Square(a). a is {} and of type {}'.forjmat(a, type(a)))
            return None

    def _sympy_(self, a):
        return eval(self, a)


class Usub(Function):
    """
    """
    nargs = 1
    is_Function = True
    is_real = True

    @classmethod
    def eval(cls, a):
        return -a  # see

    def _sympy_(self, a):
        return eval(self, a)


class Round(Function):
    """
    """
    nargs = 1
    is_Function = True
    is_real = True

    @classmethod
    def eval(cls, a):
        if a.is_number:  # sympify(a) evaluates first... but i guess it is evaluated already
            return round(a)  # see
        else:
            return  # f'Round({a})'

    def _sympy_(self, a):
        return eval(self, a)


class SignX(Function):
    """
    """
    nargs = 1
    is_Function = True
    is_real = True

    @classmethod
    def eval(cls, a):
        if a.is_number:  # sympify(a) evaluates first... but i guess it is evaluated already
            return round(a)  # see
        else:
            return

    def _sympy_(self, a):
        return eval(self, a)


def sympy_symbol_defaults(name_list):
    """
    sfeh workaround.
    sympy expressions like 'sign(((cartPos * cartVel) ** 151))' take forever.
    ignoring complex numbers with this trick (use this as locals)

    'sym_reduce': '({} ** {})'
    'sym_reduce': 'sign(re({}))'
    """
    symloc = {str(x): symbols(str(x), real=True, imaginary=False) for x in name_list}
    return symloc


# attention: exactly same capitals/letters! (gets replaced)
local_sympy_dict = {'Ifte': Ifte,
                    'Mini': Mini,
                    'Maxi': Maxi,
                    'Andb': Andb,
                    'Orb': Orb,
                    'Notb': Notb,
                    'Square': Square,
                    'Usub': Usub,
                    'Round': Round}


def plagih_sympify(expr, eval_locals=None):
    """
    Sympy bug #1:
    It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
    Or this issue: https://github.com/sympy/sympy/issues/17785

    Sympy bug #2:
    print(plagih_sympify('a<zoo'))
    throws an exception.
    -> Try-except block for this case
    """

    local_sympy_dict.update(eval_locals or {})

    try:
        # return sympify(sympify(function_string, locals=local_sympy_dict))
        expr = sympify(expr, locals=local_sympy_dict)  # rational=True?
        expr = sympify(expr, locals=local_sympy_dict)  # discussion
        return expr
    except Exception as ex:
        return 'nan'  # 'nan' always evaluates to nan. ALl nan bugs should be solved.


def expr_sympify(expr_raw):
    """
    Returns a simplified expression using sympify.
    - sympify the expression
    - If sympify evaluates to one of these errors: 'zoo', 'inf', '*I', 'nan', stop evaluation

    Sympify is a python core module which reduced mathematical expressions.
    Example: sympify('a+a+a+a') -> a*4
    Note that the sympify was extended in plagih_sympify_extras.py with extra functions

    Sympify fails: The results are, or contain, expressions that should/can not be evaluated
    'zoo': (Complex infinity) E.g. when a int-number is divided by zero
    'inf': (Regular infinity) E.g. when a float-number is divided by zero (...i know, why are there two infinities?)
    '*I': (Complex number) E.g. when putting a number to the power of negative fractals, 1**(-0.5)
    'nan': (Not a number) when Evaluation fails, E.g. types contradict, expression is empty, 'Mini(a, zoo' ...
    """

    try:
        expr_sym = plagih_sympify(expr_raw)
        expr_sym = str(expr_sym)  # discussion: debatable
    except Exception as ex:
        raise Exception(f'sympify_1: {expr_raw} reason: ({ex})')

    # I[^f] (searching "I", but ignoring "Ifte" though by not-having a "f" as second letter), old version: of \*I[^f]
    #  (pycharm also USUALLY does not check capitalized letters; i vs. I)
    # find: zoo, inf, *I, nan, (I)   , but ignore I in Ifte
    if re.search('(zoo|inf|nan|I[^f])', expr_sym):
        raise Exception(f'Simplification failed for expression: {expr_sym}')

    # if DEBUG_DUMMY:
    #     b = str(simplify(expr_sym))
    #
    #     if expr_sym != b:
    #         raise  # sfeh DISCUSSION debug

    return expr_sym


if __name__ == "__main__":
    print('Running sympify example')
    exprs = ['Square((Mini(-2.176629, Shift_2) - Abs(Fatigue_5)))',
             'Round(-123.333334234) + Round(Shift_2)',
             '1 < Maxi(2, Ifte(1 < a, 1, 1))']

    expr = '(((0.326675 * Consumption_2) - Shift_9) + (Ifte((-Shift_9 < Consumption_5), Shift_7, Ifte((Square(Gain_6) < Maxi(Fatigue_2, Ifte((Shift_9 < Shift_4), -Gain_3, Gain_5))), Shift_9, Shift_4))))'
    # expr = '-Consumption_0*sign(re(asdW**2)) - 0.004073'
    # expr = 'Mini(-1 - 1 + sqrt(1)'
    # expr = 'Maxi(2.202197, (Abs(cartVel) - sqrt(cartVel)))'
    # expr = '(vel + vel)'
    expr = 'cartPos - 0.4375'

    # obs = ['cartPos', 'cartVel']
    # symloc = {x: sympy.symbols(x, real=True, imaginary=False) for x in obs}
    # sympy_symbol_dict = {'a': sympy.symbols('a', real=True, imaginary=False),
    #                      'b': sympy.symbols('b', real=True, imaginary=False)}
    # sympify('sign(((cartPos * cartVel) ** 151))', symloc)

    # obs = {'cartVel': 0.5, 'cartPos': -0.8}
    # sympex = plagih_sympify(expr, eval_locals=obs)

    s2 = simplify(expr)
    print(s2)

    sympex = plagih_sympify(expr)
    print(sympex)
"""
sfeh
Lastly, it is recommended that you not use I, E, S, N, C, O, or Q 
"""
