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


class TestAdd(Operator, sympy.Add):
    nlabel = 'Testplus'  # ---------------------------------->
    xtype = (tuple([bool, bool]), bool)
    arity = 2


x = TestAdd()
print(x)
