from plagih.modules.plagih_sympy_extras import plagih_sympify
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

print(plagih_sympify('atan(1)'))


# print(plagih_sympify('(ifte(True, 0, 1))'))
# print(plagih_sympify('(ifte(False, 0, 1))'))
# print(plagih_sympify('a == (a<b)'))
# print(plagih_sympify('a+a+b'))
# print(plagih_sympify('(ifte((a<2), 0, 2))'))
# print(plagih_sympify('(ifte((a<2), mini(a, 2), 2))'))
# print(plagih_sympify('mini(a, 2)'))
# print(plagih_sympify('maxi(a, 2)'))
# print(plagih_sympify('ftob(x)'))
# print(plagih_sympify('ftob(2)'))