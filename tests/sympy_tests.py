from plagih.modules.plagih_sympy_extras import plagih_sympify
from plagih.modules.dicts import input_name
import re


def help_reduce_expr(expr):
    a = input_name + '0'
    b = input_name + '1'

    replacements = [(a, 'a'), (b, 'b'), ('(True)', 'True'), ('(False)', 'False'), ('(a)', 'a'), ('sina', 'sin(a)')]
    for repl in replacements:
        expr = expr.replace(repl[0], repl[1])

    return expr


def help_reduce_more(expr):
    # todo not working
    expr = help_reduce_expr(expr)

    expr_sym = plagih_sympify(expr)

    replacements = [('.', ''), ('\d', '1'), ('False', 'True'), ('True', 'a'), ('b', 'a'), ('(a)-(a)', '0'), ('sin(.)', '1')]

    for _ in range(10):
        for repl in replacements:
            tmp_expr = re.sub(repl[0], repl[1], expr)
            tmp_sym = plagih_sympify(expr)
            if tmp_sym != expr_sym and expr_sym == 'nan' or expr_sym == 'inf':
                return expr
            else:
                expr = tmp_expr

            print(expr, 'after', repl)

    return expr


def test_sympify_many():
    print(plagih_sympify('(ifte(True, 0, 1))'))
    print(plagih_sympify('(ifte(False, 0, 1))'))
    print(plagih_sympify('a == (a<b)'))
    print(plagih_sympify('a+a+b'))
    print(plagih_sympify('(ifte((a<2), 0, 2))'))
    print(plagih_sympify('(ifte((a<2), mini(a, 2), 2))'))
    print(plagih_sympify('mini(a, 2)'))
    print(plagih_sympify('maxi(a, 2)'))
    print(plagih_sympify('Ftob(x)'))
    print(plagih_sympify('Ftob2'))
    print(plagih_sympify('a/0'))
    print(plagih_sympify('a/(a/(a-a))'))

    print(type(plagih_sympify('zoo')))
    print(plagih_sympify(
        '(Ifte((b<((Ifte(((((Ifte(((Mini((-0.9932952785512101), b))>a), (a*(0.06780742530309536)), b))-(((-0.7211696970358776)-b)*(Ifte(False, (Maxi((0.3096806245031116), a)), (b+(0.21136426839679912))))))*((-0.753417111601751)/(Maxi((Ifte(False, (0.014032479771852957), a)), ((Ifte((-0.3260846193322311), b, (0.6765217675917521)))-(-0.8467329233491687))))))>(a/(Ifte(False, (0.9713116326167917), (0.015012772791043627))))), b, (-0.4203299405142009)))*(Ifte((0>a), (Ifte(True, b, a)), (Mini(b, (Maxi((Ifte(True, (((a-(0.8622404989425894))-(-0.9540913171769931))/((Ifte(b, a, (0.3379741691735958)))+b)), (-0.2472563440018496))), ((-0.9065824434028913)/(0/a)))))))))), 0, (Ifte(True, 2, 0))))'))
    print(plagih_sympify(
        '(Ifte((b<((Ifte(((((Ifte(((Mini((-0.9), b))>a), (a*(0.1)), b))-(((-0.7)-b)*b))*((-0.7)/(Maxi(a, ((Ifte((-0.3), b, (0.7)))-(-0.8))))))>(a/(Ifte(False, (0.9), (0.1))))), b, (-0.4)))*(Ifte((0>a), (Ifte(True, b, a)), (Mini(b, (Maxi((Ifte(True, (((a-(0.8))-(-0.9))/((Ifte(b, a, (0.3)))+b)), (-0.2))), ((-0.9)/((0)/a)))))))))), 0, 2))'))
    print(plagih_sympify('((Ifte(((b*(7/a))>a), b, (-4)))*(Ifte(False, b, (Mini(b, (Maxi((Ifte(True, (a/((Ifte(b, a, 3))+b)), 0)), (1/(0/a)))))))))'))
    print(plagih_sympify('Ifte(False, b, (Mini(b, (Maxi(((a)), (1/(0/a)))))))'))
    print(plagih_sympify('1/(0/a)'))
    print(plagih_sympify('Maxi(a, (1/(a)))'))
    print(plagih_sympify('Maxi(1, (1/(a)))'))
    print(plagih_sympify('Maxi(a, (1/(0/a)))'))
    print(plagih_sympify('Mini(b, (Maxi(a, (1/(0/a)))))'))
    print(plagih_sympify('Mini(b, (Maxi(a, (1/(0/a)))))'))  # nan
    print(plagih_sympify('nan'))  # nan
    print(plagih_sympify('a<zoo'))  # nan
    print(plagih_sympify('a/0'))  # zoo*a
    print(plagih_sympify('a/0.0'))  # inf*a
    print(plagih_sympify('observation0+(-0.09)**4'))  # observation0 + 6.561e-5
    print(plagih_sympify('And(a<2,b<3)'))
    print(plagih_sympify('(a<2) & (b<3)'))
    print(plagih_sympify('(-1)**(-0.5)'))
    print(plagih_sympify('(Ifte((And(((Mini((((((observation0)+(0.25))**(2))*(-0.09))+(0.03)), ((((observation0)+(-0.09))**(4))*(0.03))))<=(observation1)), ((observation1)<=(((-0.07)*(((observation0)+(0.38))**(2)))+(0.7))))), (0), (2)))'))
    print(plagih_sympify('a & True'))  # a
    print(plagih_sympify('(-0.09*(a**2))+0.03'))  # -0.09*a**2 + 0.03
    print(plagih_sympify('5 + (+6)'))
    print(plagih_sympify('5 + (-6)'))
    print(plagih_sympify('(-1)**(-0.5)'))
    print(plagih_sympify('(0**(-1.13))'))
    print(plagih_sympify('(a<b)'), '--a < b')
    print(plagih_sympify('(b>a)'), '--a < b')
    print(plagih_sympify('((-a)>(-b))'), '--a < b')
    print(plagih_sympify('((a<=b) & (a!=b))'), '--a < b')
    print(plagih_sympify('a*a*a*a*a'))
    print(plagih_sympify('N(2.345, 2)'))
    print(plagih_sympify('Or(True, False)'))

print('\nNew try:\n')

expr = '(Ifte((Or(((b)<(1)), (((b)<(0.1))&((a)<(-0.01))))), (2), (Ifte((((a)<(0.01))&(((b)>(-0.1))&((b)<(-0.01)))), (0), (Ifte(((a)<(0)), (0), (2)))))))'
# new_expr = help_reduce_expr(expr)
# print('Expression was strongly changed to: \n{}\n'.format(new_expr))
print(plagih_sympify(expr))
