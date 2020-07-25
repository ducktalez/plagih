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


expr = 'Square((Mini(-2.176629, Shift_2) - abs(Fatigue_5)))'
print(plagih_sympify(expr))
