"""
Rounding exponents is more complex than expected.
This file sums up all experiments and findings.
A sympy Issue was opened here https://github.com/sympy/sympy/issues/27326 on 28.11.2024

"Core operators no longer accept non-Expr args"
-> https://docs.sympy.org/latest/explanation/active-deprecations.html#non-expr-args-deprecated
"""
# import numpy as np
# import sympy
#
#
# # N( )
# a = sympy.sympify('Pow(a, N(1.2341, 1))')   # -> 1
# b = sympy.sympify('Pow(2, N(a, 1))')            # -> 2**a
# print('N( ):\n', a, b)
# # sympy.sympify('Pow(2, N(a, 1))', locals={'a': Symbol('a')})  # -> AttributeError: ->
# # # -> AttributeError: 'function' object has no attribute 'evalf'
#
# # Integer(1.234)
# a = sympy.sympify('Integer(1.234)')     # -> 1
# print('Integer()\n', a)
# # # -> TypeError: int() argument must be a string, a bytes-like object or a real number, not 'Symbol'
#
# # Modulo (%)
# a = sympy.sympify('1.234 - (1.234 % 1)')                                  # -> 1
# b = sympy.sympify('Pow(2, (a-(a % 1)))', locals={'a': 1.234})             # -> 2
# print(f'Modulo (%)\n', a, b)
# # sympy.sympify('Pow(2, (a-(a % 1)))', locals={'a': Symbol('a')})       # -> Typeerror ->
# # # -> TypeError: unsupported operand type(s) for %: 'Symbol' and 'One'
#
# # round()
# a = sympy.sympify('round(1.234)')             # -> 1
# b = sympy.sympify('Pow(2, round(1.234))')     # -> 2
# print(f'round(1.234)\n', a, b)
# # c = sympy.sympify('Pow(2, round(a))')         # TypeError: Cannot round symbolic expression
#
# # # custom Rnd_Dmy class
# # class Rnd_Dmy(sympy.Function):
# #
# #     @classmethod
# #     def eval(cls, a):
# #         if a.is_symbol:  # !! Must be checked before .is_number, crashes otherwise on symbols!
# #             return
# #         elif a.is_number:
# #             return round(a)
# #
# #     def _sympy_(self, a):
# #         return eval(self, a)
#
# class Rnd_Dmy(sympy.Function):
#     @classmethod
#     def eval(cls, a):
#         # Handle symbolic case
#         if a.is_symbol:
#             return None  # Keep symbolic
#         # Handle numerical case
#         elif a.is_number:
#             return round(a)  # sfeh sympy.Integer(round(a))  # Ensure it's a SymPy Integer
#
#     def __call__(self, a):
#         # Handle numerical evaluation (for lambdify or direct calls)
#         if isinstance(a, (int, float, np.ndarray)):
#             return np.round(a)
#         raise TypeError("Unsupported type for numerical evaluation in RoundDummy")
#
# a = sympy.sympify('Rnd_Dmy(1.234)', locals={'Rnd_Dmy': Rnd_Dmy})  # -> 1
# b = sympy.sympify('2**Rnd_Dmy(1.234)', locals={'Rnd_Dmy': Rnd_Dmy})  # -> 2
# c = sympy.sympify('Rnd_Dmy(a)', locals={'Rnd_Dmy': Rnd_Dmy})  # -> Rnd_Dmy(a)
# d = sympy.sympify('2**Rnd_Dmy(a)', locals={'Rnd_Dmy': Rnd_Dmy})  # -> 2**Rnd_Dmy(a)
# e = sympy.sympify('Rnd_Dmy(a)', locals={'Rnd_Dmy': Rnd_Dmy, 'a': Symbol('a', is_number=True)})
# x = sympy.sympify('2**Rnd_Dmy(a)', locals={'Rnd_Dmy': Rnd_Dmy, 'a': Symbol('a', is_number=True)})
# final = sympy.sympify('2**Rnd_Dmy(a)', locals={'Rnd_Dmy': Rnd_Dmy, 'a': Symbol('a', is_number=True)})
# final = final.evalf(subs={Symbol('a', is_number=True): 1.234})
# print('Rnd_Dmy(1.234)\n', a, b, c, d, e, x, final)
