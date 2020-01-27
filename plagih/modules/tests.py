from plagih.modules.plagih_eval import *
from plagih.modules.plagih_tree import *
from plagih.modules.Examples import *


def test_plagih_eval():
    label_list = MountainCarExamples.tree_v3

    tree = karoo_tree_from_labellist(label_list)
    expr_sym = tree_expr_sympify(tree=tree)
    expr_raw = tree_expr_raw(tree, root_id)

    print(expr_sym)
    graph = ast_convert_from_expr(expr_raw, build=True)
    # expr = labels_from_graphlist(graph, [])
    print(graph)


def test_sympify():
    pass


def test_plagih_tree():
    expr_raw = '(Ifte((b < (b / (Maxi((((((b - b) + (Mini(0.2, b))) / 2) ** (Maxi(1, b))) ** 0), a)))), 0, (Ifte((True), 2, 1))))'

    label_list = ast_convert_from_expr(expr_raw, build=True)
    olymp_winner = karoo_tree_from_labellist(label_list)
    print(olymp_winner)

    return


test_plagih_tree()
