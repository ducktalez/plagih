import csv

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from sympy import sympify, Function
import tensorflow  as tf

# pickle-version does make too much trouble for now    need to switch to .csv

# print(sympify('(1) + (2) + (b)'))
# print(sympify('1 + 2 + b'))


class ifte(Function):

        @classmethod
        def eval(cls, *_args):
            if (len(_args) == 3):
                if _args[0] > 0:
                    return 1
                else:
                    return 2

        def _sympy_(self, a, b, c):
            if a > 0:
                return b
            else:
                return c

from sympy.core.sympify import converter


# Function('ifte')
print(sympify('ifte(a,b,c'))
