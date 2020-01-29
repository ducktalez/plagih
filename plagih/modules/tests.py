from plagih.modules.plagih_eval import *
from plagih.modules.plagih_tree import *
from plagih.modules.Examples import *


def test_plagih_eval():
    label_list = MountainCarExamples.tree_v3_list

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
    tree_list = MountainCarExamples.tree_plus_list
    tree_modify = MountainCarExamples.tree_plus_modify_v1
    tree = karoo_tree_from_labellist(tree_list, modify_list=tree_modify)
    # tree = karoo_tree_from_labellist(tree_list)
    print(tree)
    print(tree_get_mutatable_leaves(tree, 0))
    print(tree_get_mutatable_leaves(tree, 1))
    print(tree_get_mutatable_leaves(tree, 2))
    print(tree_get_mutatable_leaves(tree, 3))

    return


test_plagih_tree()
