# class BaseArity:
#     pass
#
#
# class ExactArity(BaseArity):
#     def __int__(self, num):
#         self.arity = num
#
#
# class NoArity(BaseArity):
#     def __int__(self):
#         self.arity = None

import sympy


class Operator:
    """
    operator nodes (+, +, *, /, sin(), sign(), ...)
    inner nodes of a fintree
    """
    is_Function = True

    def __str__(self):
        return self.get_nlabel()

    def get_nlabel(self):
        return __class__.__name__


class Testhis(Operator):
    nlabel = 'Testplus'  # ---------------------------------->
    xtype = (tuple([bool, bool]), bool)

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        return obj


class Testme(Operator):
    nlabel = 'Testplus'  # ---------------------------------->
    xtype = (tuple([bool, bool]), bool)

    def __init__(self, val):
        self.nlabel = val


xx = Testhis(1.23)
# yy = Testme(1.23)
# print(xx)
x = sympy.sin(sympy.Symbol('a'))
