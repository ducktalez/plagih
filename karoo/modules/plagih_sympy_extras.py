"""
This class enrichens the python-core "sympy".
Sympy is used to reduce the functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.
Sympy does not have some functions, e. g. 'if a then b else c', which we want to use though.

This has currently only Ifte(a, b, c), which makes 'if then else' as if was a three-parametered function.

Use:
1. Import this class: from <path-to-this-file>.plagih_sympy_extras import Ifte
2. Build up a dictionary: local_sympy_dict = {'Ifte': Ifte}
3. Use sympify():
    - Add your local dictionary to the sympify call in the 'local' parameter
    - when using sympify(), use it double.
    Like: sympify(sympify('Ifte(a, b, c)'))
Why?
It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
Or this issue: https://github.com/sympy/sympy/issues/17785
"""

from sympy import Function


class Ifte(Function):
    nargs = 3

    @classmethod
    def eval(cls, a, b, c):
        return b if a else c

    def _sympy_(self, a, b, c): return eval(self, a, b, c)
