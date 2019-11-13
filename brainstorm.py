from karoo.modules.plagih_sympy_extras import plagih_sympify
import re

# print(plagih_sympify('b < Min(a*b + a, Max(a, b, b**3*Min(a, b))'))

x = plagih_sympify('(Ifte(((observation1/0)<(0)), (0), (2)))')

print(x)

if 'zoo' in str(x):
    print('booh')

x = re.sub('zoo', '10', str(x))

print(x)