"""
sfeh: I think we should get rid of sympy in the long term. A lot of problems are related to sympy.

This class enrichens the python-core 'sympy'.
Sympy is used to reduce the functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.

- implementing missing functions in sympify, e. g. 'if a then b else c'.
- All number-related functions must have set
    is_real = True
    otherwise: '1 < Maxi(2, Ifte(1 < a, 1, 1))' will crash. (< operators only only work on non-complex - aka real numbers)
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
"""

from sympy import Function, sympify
# from sympy.core.numbers import ComplexInfinity


class Ifte(Function):
    """
    plagih_sympify('Ifte(a, b, c)')
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
        return eval(self, *args)  # a, b, c don't know why c is unexpected. works though.


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
            return a if a < b else b
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
        if a.is_Boolean and b.is_Boolean:
            return a and b
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
        if a == True or b == True:
            return True  # sfeh this evaluation might end up in error in real experiment
        if a.is_Boolean and b.is_Boolean:
            return a and b
        else:
            return None

    def _sympy_(self, a, b):
        return eval(self, a, b)


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
            return a**2  # see
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


def plagih_sympify(function_string, eval_locals=None):
    """
    Sympy bug #1:
    It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
    Or this issue: https://github.com/sympy/sympy/issues/17785

    Sympy bug #2:
    print(plagih_sympify('a<zoo'))
    throws an exception.
    -> Try-except block for this case
    """

    # attention: exactly same capitals/letters! (gets replaced)
    local_sympy_dict = {'Ifte': Ifte,
                        'Mini': Mini,
                        'Maxi': Maxi,
                        'Andb': Andb,
                        'Orb': Orb,
                        'Notb': Notb,
                        'Square': Square,
                        'Usub': Usub,
                        'usub': Usub,  # sfeh delete this
                        'Round': Round}
    local_sympy_dict.update(eval_locals or {})

    try:
        # return sympify(sympify(function_string, locals=local_sympy_dict))
        return sympify(sympify(function_string, locals=local_sympy_dict))
    except Exception as ex:
        # raise
        # print(f'debugging further {ex}')
        return 'nan'  # 'nan' always evaluates to nan. ALl nan bugs should be solved.


if __name__ == "__main__":
    print('Running sympify example')
    exprs = ['Square((Mini(-2.176629, Shift_2) - Abs(Fatigue_5)))',
             'Round(-123.333334234) + Round(Shift_2)',
             '1 < Maxi(2, Ifte(1 < a, 1, 1))']

    expr = '(((0.326675 * Consumption_2) - Shift_9) + (Ifte((-Shift_9 < Consumption_5), Shift_7, Ifte((Square(Gain_6) < Maxi(Fatigue_2, Ifte((Shift_9 < Shift_4), -Gain_3, Gain_5))), Shift_9, Shift_4))))'
    expr = 'Maxi(2.202197, (Abs(cartVel) - sqrt(cartVel)))'
    # expr = 'sign(((a * b) ** 10))'  # takes too long
    # expr = '0.307785*Consumption_2 - 0.779543*Gain_3 + 0.779543*Gain_9 - Shift_9 + 0.779543*Ifte(-Shift_8 < Consumption_5, Shift_7, Shift_4)'
    obs = {'cartVel': 0.5, 'cartPos': -0.8}

    sympex = plagih_sympify(expr, eval_locals=obs)

    # print(plagih_sympify(expr))
    print(sympex)

"""
sfeh
Lastly, it is recommended that you not use I, E, S, N, C, O, or Q 
"""