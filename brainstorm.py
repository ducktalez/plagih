import csv

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

import numpy as np
# import tensorflow as tf
#
# x = tf.constant(0, dtype=tf.float32)
# test = tf.dtypes.cast(x, tf.bool)
#
# print(test)

from sympy import Matrix, sympify

class MyList1(object):
    def __iter__(self):
        yield 1
        yield 2
        return
    def __getitem__(self, i): return list(self)[i]
    def _sympy_(self): return Matrix(self)


from sympy import Function

class Ifte(Function):
    nargs = 3

    @classmethod
    def eval(cls, a, b, c):
        return b if a else c

    def _sympy_(self, a, b, c): return eval(self, a, b, c)


local_dict= {"MyList1": MyList1, "ifte": Ifte}
print(sympify(Ifte('1', '2', '3'), locals=local_dict))
print(sympify(sympify('ifte(1==1, 2, 3)', locals=local_dict)))


# print(sympify('MyList1()'))  # MyList1()
# print(sympify(sympify('MyList1()', locals=local_dict)))  # <__main__.MyList1 object at 0x0000000006D0AA20>



# import sympy
# from sympy.parsing.sympy_parser import parse_expr
# t, k, y = sympy.symbols('t k y')
# parsed = parse_expr("log2(t + k) + k - y", local_dict={"log2": lambda x: sympy.log(x, 2)})
# print(sympy.solve(parsed, k))