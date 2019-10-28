"""
This class enrichens the python-core "sympy".
Sympy is used to reduce the functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.
Sympy does not have some functions, e. g. 'if a then b else c', which we want to use though.

This has currently only Ifte(a, b, c), which makes 'if then else' as if was a three-parametered function.

Use:
1. Import this class: import karoo.modules.plagih_sympy_extras
2. Build up a dictionary: local_sympy_dict = {'Ifte': Ifte}
3. Use sympify():
    - Add your local dictionary to the sympify call in the 'local' parameter
    - when using sympify(), use it double.
    Like: sympify(sympify('Ifte(a, b, c)'))
Why?
It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
Or this issue: https://github.com/sympy/sympy/issues/17785
Also, please do not ask me about when to use Ifte() and ifte(), it somehow works.
"""

from sympy import Function, sympify


class Ifte(Function):
    nargs = 3

    @classmethod
    def eval(cls, a, b, c):
        return b if a else c

    def _sympy_(self, a, b, c): return eval(self, a, b, c)


local_sympy_dict = {'ifte': Ifte}


def plagih_sympify(function_string):
    return sympify(sympify(function_string, locals=local_sympy_dict))


# class BtoiSy(Function):
#     nargs = 1
#
#     @classmethod
#     def eval(cls, b):
#         if type(locate(b)) == type(True):
#             print("Was los")
#             return int(b)
#         else:
#             return 'int('+str(b)+')'
#
#     def _sympy_(self, b): return eval(self, b)
#
#
# class NtobSy(Function):
#     nargs = 1
#
#     @classmethod
#     def eval(cls, n):
#         if type(locate(n)) == type(1.0) or type(n) == type(1):
#             return bool(n)
#         else:
#             return 'bool('+str(n)+')'
#
#     def _sympy_(self, n): return eval(self, n)


class Lesy(Function):
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return a < b

    def _sympy_(self, a, b): return eval(self, a, b)


class Gesy(Function):
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return a > b

    def _sympy_(self, a, b): return eval(self, a, b)


class Eqsy(Function):
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return a == b

    def _sympy_(self, a, b): return eval(self, a, b)


class Sumsy(Function):
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return a+b

    def _sympy_(self, a, b): return eval(self, a, b)


class Difsy(Function):
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return a-b

    def _sympy_(self, a, b): return eval(self, a, b)


class Mulsy(Function):
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return a*b

    def _sympy_(self, a, b): return eval(self, a, b)


class Divsy(Function):
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        return a/b

    def _sympy_(self, a, b): return eval(self, a, b)


