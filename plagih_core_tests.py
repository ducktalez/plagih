from plagih.modules.plagih_sympy_extras import plagih_sympify
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def test_plagih_sympify():
    print(plagih_sympify('(Ifte(True, 0, 1))'))
    print(plagih_sympify('(Ifte(False, 0, 1))'))
    print(plagih_sympify('a == (a<b)'))
    print(plagih_sympify('a+a+b'))
    print(plagih_sympify('(Ifte((a<2), 0, 2))'))
    print(plagih_sympify('(Ifte((a<2), Mini(a, 2), 2))'))
    print(plagih_sympify('Mini(a, 2)'))
    print(plagih_sympify('Mini(1, 2)'))
    print(plagih_sympify('Maxi(a, 2)'))
    print(plagih_sympify('ftob(x)'))
    print(plagih_sympify('ftob(2)'))
    print(plagih_sympify('atan(1)'))

    return

test_plagih_sympify()
