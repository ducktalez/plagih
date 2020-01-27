from plagih.modules.plagih_eval import *
from plagih.modules.plagih_tree import *
from plagih.modules.Examples import *


def labels_from_algo(expr_array, expr):
    """
    Returns a list
    """
    for x in expr_array:
        if type(x) is not list:
            expr.append(x)

    only_lists = [x for x in expr_array if (type(x) == list)]
    if only_lists:
        from itertools import chain
        lists_removed = list(chain(*only_lists))
        expr = labels_from_algo(lists_removed, expr)
    return expr


def test_plagih_eval():
    label_list = MountainCarExamples.tree_v3

    tree = karoo_tree_from_labellist(label_list)
    expr_sym = tree_expr_sympify(tree=tree)
    expr_raw = tree_expr_raw(tree, root_id)

    print(expr_sym)
    graph = ast_convert_from_expr(expr_raw, build=True)
    expr = labels_from_algo(graph, [])
    print(expr)


def test_plagih_tree():
    core = test_trees(4)
    karoo_tree = tree_convert_plagih_to_karoo(core)
    another_list = ['Ifte', 'And', '0', '2', '<=', '<=', 'Mini', 'observation1', 'observation1', '+',
                    '+', '*', '*', '0.7', '*', '0.03', '**', '0.03', '-0.07', '**', '**', '-0.09', '+', '4', '+', '2',
                    '+', '2', 'observation0', '-0.09', 'observation0', '0.38', 'observation0', '0.25']
    label_list = ['&', 'a', 'True']
    arity_list = [3, 2, 0, 0, 0, 0]
    tree = karoo_tree_from_labellist(label_list)
    print(tree)
    tree_pretty_print(tree, karoo=True)

    # test()
    tree = karoo_tree_from_labellist(['**', '0.5', '1.12'])
    tree = tree_normalize_exponentiation(tree)
    print(tree)
    return

# test_plagih_eval()