from karoo.modules.plagih_sympy_extras import plagih_sympify
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

# print(plagih_sympify('b < Min(a*b + a, Max(a, b, b**3*Min(a, b))'))

# x = plagih_sympify('1+0/0')
# y = plagih_sympify('1/0')
#
# print(x, y)


class Hoho:

    def __init__(self):
        self.x = 5
        self.a=[1,2,3,4,5]
        self.b=[3,4,5,6,7]

    def whappens(self, **kwargs):
        print('Ehem', kwargs.items())
        return

    def testme(self):
        self.whappens(a)
        return


here = Hoho()
here.testme()
