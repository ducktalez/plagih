from plagih.modules.plagih_sympy_extras import plagih_sympify
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


print(plagih_sympify('(ifte(True, 0, 1))'))
print(plagih_sympify('(ifte(False, 0, 1))'))
print(plagih_sympify('a == (a<b)'))
print(plagih_sympify('a+a+b'))
print(plagih_sympify('(ifte((a<2), 0, 2))'))
print(plagih_sympify('(ifte((a<2), mini(a, 2), 2))'))
print(plagih_sympify('mini(a, 2)'))
print(plagih_sympify('maxi(a, 2)'))
print(plagih_sympify('ftob(x)'))
print(plagih_sympify('ftob2'))
print(plagih_sympify('a/0'))
print(plagih_sympify('a/(a/(a-a))'))

print(type(plagih_sympify('zoo')))
print(plagih_sympify('(Ifte((b<((Ifte(((((Ifte(((Mini((-0.9932952785512101), b))>a), (a*(0.06780742530309536)), b))-(((-0.7211696970358776)-b)*(Ifte(False, (Maxi((0.3096806245031116), a)), (b+(0.21136426839679912))))))*((-0.753417111601751)/(Maxi((Ifte(False, (0.014032479771852957), a)), ((Ifte((-0.3260846193322311), b, (0.6765217675917521)))-(-0.8467329233491687))))))>(a/(Ifte(False, (0.9713116326167917), (0.015012772791043627))))), b, (-0.4203299405142009)))*(Ifte((0>a), (Ifte(True, b, a)), (Mini(b, (Maxi((Ifte(True, (((a-(0.8622404989425894))-(-0.9540913171769931))/((Ifte(b, a, (0.3379741691735958)))+b)), (-0.2472563440018496))), ((-0.9065824434028913)/((a-a)/a)))))))))), 0, (Ifte(True, 2, 0))))'))
print(plagih_sympify('(Ifte((b<((Ifte(((((Ifte(((Mini((-0.9), b))>a), (a*(0.1)), b))-(((-0.7)-b)*b))*((-0.7)/(Maxi(a, ((Ifte((-0.3), b, (0.7)))-(-0.8))))))>(a/(Ifte(False, (0.9), (0.1))))), b, (-0.4)))*(Ifte((0>a), (Ifte(True, b, a)), (Mini(b, (Maxi((Ifte(True, (((a-(0.8))-(-0.9))/((Ifte(b, a, (0.3)))+b)), (-0.2))), ((-0.9)/((0)/a)))))))))), 0, 2))'))
print(plagih_sympify('((Ifte(((b*(7/a))>a), b, (-4)))*(Ifte(False, b, (Mini(b, (Maxi((Ifte(True, (a/((Ifte(b, a, 3))+b)), 0)), (1/(0/a)))))))))'))
print(plagih_sympify('Ifte(False, b, (Mini(b, (Maxi(((a)), (1/(0/a)))))))'))
print(plagih_sympify('1/(0/a)'))
print(plagih_sympify('Maxi(a, (1/(a)))'))
print(plagih_sympify('Maxi(1, (1/(a)))'))
print(plagih_sympify('Maxi(a, (1/(0/a)))'))
print(plagih_sympify('Mini(b, (Maxi(a, (1/(0/a)))))'))
print(plagih_sympify('Mini(b, (Maxi(a, (1/(0/a)))))'))
print(plagih_sympify('nan'))
print(plagih_sympify('a<zoo'))

