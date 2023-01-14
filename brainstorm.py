import sympy


class Operator:
    """
    operator nodes (+, +, *, /, sin(), sign(), ...)
    inner nodes of a fintree
    """

    def __str__(self):
        return self.__class__.__name__


class Float:

    @classmethod
    def eval(cls, a):
        return sympy.Float(a)

    xtype = (tuple([float]), float)


class Add:

    @classmethod
    def eval(cls, a):
        return sympy.Float(a)

    xtype = (tuple([float]), float)


print(Add.insym.__name__)
