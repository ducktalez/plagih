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
    tree_list = MountainCarExamples.tree_test_plus_list
    tree_modify = MountainCarExamples.tree_plus_modify_v1
    tree = karoo_tree_from_labellist(tree_list, modify_list=tree_modify)
    # tree = karoo_tree_from_labellist(tree_list)
    print(tree)
    print(tree_get_mutatable_leaves(tree, 0))
    print(tree_get_mutatable_leaves(tree, 1))
    print(tree_get_mutatable_leaves(tree, 2))
    print(tree_get_mutatable_leaves(tree, 3))

    return


def test_tmp():
    # tree = karoo_tree_from_labellist(MountainCarExamples.tree_v3_list, modify_list=MountainCarExamples.tree_v3_modify)
    tree = karoo_tree_from_labellist(MountainCarExamples.tree_test_minus_list)
    algo_raw = tree_expr_raw(tree, P_first_node)
    print('Algo raw:', algo_raw)
    algo_sym_1 = tree_expr_sympify(algo_raw=algo_raw)
    print('Also sym:', algo_sym_1)
    label_list_1 = ast_convert_from_expr(algo_sym_1, build=True)
    print('Label List 1:\n', label_list_1)
    tree2 = karoo_tree_from_labellist(label_list_1)
    if tree_check_expression(tree, karoo=True):
        print('True test')
    else:
        print('False test')
    algo_raw2 = tree_expr_raw(tree2, root_id)

    print('Algo raw2:', algo_raw2)
    algo_sym2 = tree_expr_sympify(tree=tree2)
    print('Algo_sym2:', algo_sym2)


test_tmp()
