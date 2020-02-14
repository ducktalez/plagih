from plagih.modules.plagih_eval import *
from plagih.modules.plagih_tree import *
from plagih.modules.Examples import *


def test_plagih_eval():
    label_list = MountainCarExamples.tree_v3_list

    tree = karoo_tree_from_labellist(label_list)
    expr_sym = tree_expr_sympify(tree=tree)
    expr_raw = tree_get_expr_raw(tree, root_id)

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
    print(tree_get_mutatable_layer(tree, 0))
    print(tree_get_mutatable_layer(tree, 1))
    print(tree_get_mutatable_layer(tree, 2))
    print(tree_get_mutatable_layer(tree, 3))

    return


def rebuild_same_tree():
    """
    just try to make the complete tree-transformation
    """
    # tree = karoo_tree_from_labellist(MountainCarExamples.tree_v3_list, modify_list=MountainCarExamples.tree_v3_modify)
    tree = karoo_tree_from_labellist(MountainCarExamples.tree_v3_list)
    algo_raw = tree_get_expr_raw(tree, P_first_node)
    algo_sym_1 = tree_expr_sympify(algo_raw=algo_raw)
    label_list_1 = ast_convert_from_expr(algo_sym_1, build=True)
    print('1 Label List:', label_list_1)
    tree2 = karoo_tree_from_labellist(label_list_1)
    algo_raw2 = tree_get_expr_raw(tree2, P_first_node)
    algo_sym_2 = tree_expr_sympify(algo_raw=algo_raw2)
    label_list_2 = ast_convert_from_expr(algo_sym_2, build=True)
    print('2 Label List:', label_list_2)

    if label_list_1 == label_list_2:
        print('SUCCESS')


def test_tree_build():
    label_list = ['Ifte', '<', 0.0, 2.0, 'observation1', 'Maxi', 'Mini', 'Mini', 'observation1', '-', 'Mini', '~', 'Mini', 0.855, 'observation0', '**', 0.455, 0.927014714644712, 'observation1', 'observation1', 'observation1']
    tree = karoo_tree_from_labellist(label_list)
    print(tree)


test_tree_build()
