"""
This class enrichens the python-core "sympy".
Sympy is used to reduce the functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.
Sympy does not have some functions, e. g. 'if a then b else c', which we want to use though.

This has currently only Ifte(a, b, c), which makes 'if then else' as if was a three-parametered function.

Use:
1. Import function only
    from karoo.modules.plagih_sympy_extras import plagih_sympify
2. Use function
    plagih_sympify('ifte(True, b, c)')
    plagih_sympify('Min(a, b)')

Also, please do not ask me about when to use Ifte() and ifte(), it somehow works.
"""

from sympy import Function, sympify


class Ifte(Function):
    """
    plagih_sympify('Ifte(a, b, c)')
    """
    nargs = 3

    @classmethod
    def eval(cls, a, b, c):
        return b if a else c

    def _sympy_(self, a, b, c): return eval(self, a, b, c)



local_sympy_dict = {'ifte': Ifte}

"""
It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
Or this issue: https://github.com/sympy/sympy/issues/17785
"""
def plagih_sympify(function_string):
    return sympify(sympify(function_string, locals=local_sympy_dict))
