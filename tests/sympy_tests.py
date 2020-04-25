from plagih.modules.plagih_sympy_extras import plagih_sympify
import re


def help_reduce_expr(expr):
    replacements = [('(True)', 'True'), ('(False)', 'False'), ('(a)', 'a'), ('sina', 'sin(a)')]
    for repl in replacements:
        expr = expr.replace(repl[0], repl[1])

    return expr


def help_reduce_more(expr):
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
    sympify_test_strings = [
        ('(Ifte(True, 0, 1))', '0'),
        ('(Ifte(False, 0, 1))', '1'),
        ('a+a+b', '2*a+b'),
        ('(Ifte((a<2), 0, 2))', 'ifte(a < 2, 0, 2)'),
        ('(Ifte((a<2), Mini(a, 2), 2))', None),
        ('Ftob(x)', 'Ftob(x)'),
        ('a/0', 'zoo'),
        ('a/(a/(a-a))', 'a*zoo'),
        ('(Ifte((b<((Ifte(((((Ifte(((Mini((-0.9932952785512101), b))>a), (a*(0.1)), b))-(((-0.7)-b)*b))*((-0.7)/(Maxi(a, ((Ifte((-0.3), b, (0.7)))-(-0.8))))))>(a/(Ifte(False, (0.9), (0.1))))), b, (-0.4)))*(Ifte((0>a), (Ifte(True, b, a)), (Mini(b, (Maxi((Ifte(True, (((a-(0.8))-(-0.9))/((Ifte(b, a, (0.3)))+b)), (-0.2))), ((-0.9)/((0)/a)))))))))), 0, 2))', None),
        ('((Ifte(((b*(7/a))>a), b, (-4)))*(Ifte(False, b, (Mini(b, (Maxi((Ifte(True, (a/((Ifte(b, a, 3))+b)), 0)), (1/(0/a)))))))))', None),
        ('Ifte(False, b, (Mini(b, (Maxi(((a)), (1/(0/a)))))))', None),
        ('1/(0/a)', None),
        ('Maxi(a, (1/(a)))', None),
        ('Maxi(1, (1/(a)))', None),
        ('Maxi(a, (1/(0/a)))', None),
        ('Mini(b, (Maxi(a, (1/(0/a)))))', None),  # nan
        ('a<zoo', None),  # nan
        ('a/0', None),  # zoo*a
        ('a/0.0', None),  # inf*a
        ('obs0+(-0.09)**4', None),  # obs0 + 6.561e-5
        ('And(a<2,b<3)', None),
        ('(a<2) & (b<3)', None),
        ('(-1)**(-0.5)', None),
        ('(Ifte((And(((Mini((((((obs0)+(0.25))**(2))*(-0.09))+(0.03)), ((((obs0)+(-0.09))**(4))*(0.03))))<=(obs1)), ((obs1)<=(((-0.07)*(((obs0)+(0.38))**(2)))+(0.7))))), (0), (2)))', None),
        ('a & True', None),  # a
        ('(-0.09*(a**2))+0.03', None),  # -0.09*a**2 + 0.03
        ('5 + (+6)', None),
        ('5 + (-6)', None),
        ('(-1)**(-0.5)', None),
        ('(0**(-1.13))', None),
        ('(a<b)', None),
        ('--a < b', None),
        ('(b>a)', None),
        ('--a < b', None),
        ('((-a)>(-b))', None),
        ('--a < b', None),
        ('((a<=b) & (a!=b))', None),
        ('--a < b', None),
        ('a*a*a*a*a', None),
        ('N(2.345, 2)', None),
        ('Or(True, False)', None),
        ('(Ifte((Or(((b)<(1)), (((b)<(0.1))&((a)<(-0.01))))), (2), (Ifte((((a)<(0.01))&(((b)>(-0.1))&((b)<(-0.01)))), (0), (Ifte(((a)<(0)), (0), (2)))))))', None),
        ('Andb(True, False)', None),
        ('Or(True, False)', None),
        ('Orb(True, False)', None),
        ('Or(True, False)', None),
        ('Orb(True, False)', None),
        ('Or(a, a)', None),
        ('Andb(True, a & b)', None),
        ('Ifte(Orb(pos < -1,  Andb(pos < 0.1, vel < -0.05)), 2, Ifte(Andb(Andb(pos > -0.45, pos < -0.05), vel < 0.02), 0,  Ifte(vel < 0, 0, 2)))', None),
        ('Andb(True, a & b)', None),
        ('not(Orb(True, Orb(False, (0.895)<(cartVel))))', None)]

    for expr in sympify_test_strings:
        print(plagih_sympify(expr[0]))


# test_sympify_many()

expr = 'not(Orb(True, Andb(False, 1 <(cartVel))))'
# expr = 'not(True || Orb(False, 1 <(cartVel)))'
# expr = '~(a)'
print(plagih_sympify(expr))
