"""
This class enrichens the python-core 'sympy'.
Sympy is used to reduce the functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.
Sympy does not have some functions, e. g. 'if a then b else c', which we want to use though.

This has currently only Ifte(a, b, c), which makes 'if then else' as if was a three-parametered function.

inputs a, b, c to a function can be actual actual values but also variables with out a value assigned yet.

Use:
1. Import function only
    from plagih.modules.plagih_sympy_extras import plagih_sympify
2. Use function

Also, please do not ask me about when to use Ifte() and ifte(), it somehow works.
"""

from sympy import Function, sympify
from sympy.core.numbers import ComplexInfinity


class Ifte(Function):
    """
    plagih_sympify('ifte(a, b, c)')
    """
    nargs = 3

    @classmethod
    def eval(cls, a, b, c):
        if a == True or a == False:  # you can not believe how long it took me to figure out why this is needed
            return b if a else c  # search for 'gotcha' in https://docs.sympy.org/latest/_modules/sympy/core/relational.html
        else:
            return

    def _sympy_(self, a, b, c):
        return eval(self, a, b, c)  # don't know why c is unexpected. works though.


class Mini(Function):
    """
    """
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        if (a < b) == True or (a < b) == False:
            return a if a < b else b
        else:
            return

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Maxi(Function):
    """
    """
    nargs = 2

    @classmethod
    def eval(cls, a, b):

        if (a < b) == True or (a < b) == False:
            return a if a > b else b
        else:
            return

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Andb(Function):
    """
    """
    nargs = 2

    @classmethod
    def eval(cls, a, b):

        if (a == True or a == False) and (b == True or b == False):
            return a and b
        else:
            return

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Orb(Function):
    """
    """
    nargs = 2

    @classmethod
    def eval(cls, a, b):

        if (a == True or a == False) and (b == True or b == False):
            return a and b
        else:
            return

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Square(Function):
    """
    """
    nargs = 2

    @classmethod
    def eval(cls, a):

        return a**2  # sfeh requires testing. a LOT of testing. check for num-type?

    def _sympy_(self, a, b):
        return eval(self, a, b)


class Ftob(Function):
    """
    Dummy function to convert Float to boolean
    """
    nargs = 1

    @classmethod
    def eval(cls, a):
        if (a > 0) == True or (a > 0) == False:
            return True if a > 0 else False
        else:
            return

    def _sympy_(self, a):
        return eval(self, a)


class Btof(Function):
    """
    Dummy function to convert Boolean to Float
    """
    nargs = 1

    @classmethod
    def eval(cls, a):
        if a == True or a == False:
            return 1 if a else 0
        else:
            return

    def _sympy_(self, a):
        return eval(self, a)


local_sympy_dict = {'Ifte': Ifte,
                    'Ftob': Ftob,
                    'Btof': Btof,
                    'Mini': Mini,
                    'Maxi': Maxi,
                    'Andb': Andb,
                    'Orb': Orb,
                    'Square': Square,}


def plagih_sympify(function_string):
    """
    Sympy bug #1:
    It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
    Or this issue: https://github.com/sympy/sympy/issues/17785

    Sympy bug #2:
    >>> print(plagih_sympify('a<zoo'))
    throws an exception.
    -> Try-except block for this case
    """
    try:
        return sympify(sympify(function_string, locals=local_sympy_dict))
    except:
        return 'nan'
