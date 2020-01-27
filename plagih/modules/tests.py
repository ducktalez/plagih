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


def test_plagih_tree():

    tree3 = ['Ifte', '<', 0.0, 'Ifte', 'observation1', 0.0, '<', 2.0, 1.0, '/', '-', '**', 'observation1', 0.975, 'observation1', 0.17, 'Mini', 'observation0', '*', 0.185, 'Mini', '-', 'observation0', 0.145]
    # tree3 = ['+', 1, 2]
    tree3 = karoo_tree_from_labellist(tree3)
    print(tree3)
    return

# test_plagih_tree()