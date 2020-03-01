from plagih.modules.plagih_eval import *
from plagih.modules.plagih_tree import *
from plagih.modules.Examples import *


class TestHelpers:
    # example func_array. Note that (for the random choice) functions can be included more often
    func_array = [[[], ['sin', 'cos', '~'], ['+', '+', '+', '-', '*', '/'], []],
                  [[], ['Ftob'], ['<', '>', '==', '!='], []],
                  [[], ['Not', 'Not'], ['&', 'Xor'], []],
                  [[], ['Btof'], [], []],
                  [[], [], [], ['Ifte']]]

    variables_dict = {'all': ['observation0', 'observation1'],
                      'types': ['float', 'float'],
                      'float': ['observation0', 'observation1'],
                      'bool': []}


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
    tree_modify = MountainCarExamples.tree_test_plus_modify_v1
    tree = karoo_tree_from_labellist(tree_list, modify_list=tree_modify)
    # tree = karoo_tree_from_labellist(tree_list)
    print(tree)
    print(tree_get_mutatable_layer(tree, 0))
    print(tree_get_mutatable_layer(tree, 1))
    print(tree_get_mutatable_layer(tree, 2))
    print(tree_get_mutatable_layer(tree, 3))

    return


def test_rebuild_loop_tree():
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
    label_list = ['Ifte', '<', 0.0, 2.0, 'observation1', 'Maxi', 'Mini', 'Mini', 'observation1', '-', 'Mini', '~', 'Mini', 0.855, 'observation0', '**', 0.455, 0.927014714644712, 'observation1',
                  'observation1', 'observation1']
    tree = karoo_tree_from_labellist(label_list)
    print(tree)


def test_choose_function():
    # example func_array. Note that (for the random choice) functions can be included more often
    func_array = TestHelpers.func_array
    arities = [0, 1, 2, 3]
    xtypes = ['f2f', 'f2b', 'b2b', 'b2f', 'b2f2f']
    do_not_forget_this_option = None

    # test_cases: (arity, xtype, result)
    test_cases = [(0, 'f2f', []),
                  (0, 'f2b', []),
                  (1, 'f2f', func_array[0][1]),
                  (1, 'f2b', func_array[1][1]),
                  (2, 'f2f', func_array[0][2]),
                  (2, 'b2b', func_array[2][2]),
                  (2, 'f2b', func_array[1][2]),

                  (2, None, func_array[0][2] + func_array[1][2] + func_array[2][2]),
                  (3, None, func_array[4][3]),

                  (None, 'f2f', func_array[f2f][1] + func_array[f2f][2] + func_array[b2f][1] + func_array[b2f2f][3]),

                  (None, None, sum(sum(func_array, []), [])),

                  (None, 'b2f2f', func_array[f2f][1] + func_array[f2f][2] + func_array[b2f][1] + func_array[b2f2f][3]),
                  (None, '2f', func_array[f2f][1] + func_array[f2f][2] + func_array[b2f][1] + func_array[b2f2f][3])
                  ]

    worked_fine = True
    for test_case in test_cases:
        arity = test_case[0]
        xtype = test_case[1]
        solution = test_case[2]
        result = xtype_get_func_list(func_array, xtype=xtype, arity=arity)
        if solution != result:
            print('Failed for {}. Result was: {}'.format(test_case, result))
            worked_fine = False

    # for arity in [0, 1, 2, 3, None]:
    #     for xtype in ['f2f', 'f2b', 'b2b', 'b2f2f', None]:
    #         func_list = xtype_get_func_list(func_array, xtype=xtype, arity=arity)
    #         print('xtype_get_func_list(func_array, arity={} xtype={})= {}'.format(arity, xtype, func_list))

    if worked_fine:
        print('test_choose_function() successful!')

    return worked_fine


def test_all():
    print('Starting several tests!')
    test_plagih_eval()
    test_sympify()
    test_plagih_tree()
    test_rebuild_loop_tree()
    test_tree_build()
    test_choose_function()
    test_build_tree_grow_nodecount()
    print('Testing procedure is custom_done!')


def test_build_tree_grow_nodecount(verbose=False):
    print_blue('test_build_tree_grow_nodecount()')
    worked_fine = True
    test_cases = [('f2f', 8),
                  ('f2f', 9),
                  ('f2f', 18),
                  ('f2f', 2),
                  ('b2f2f', 12),
                  ('b2b', 12),
                  ('2f', 12),
                  ('2b', 12),
                  ('f2b', 12)]
    variables_dict = TestHelpers.variables_dict
    func_array = TestHelpers.func_array
    for test_case in test_cases:
        for _ in range(10):
            old_xtype = test_case[0]
            max_nodes = test_case[1]
            label_list, arity_list = invent_label_list_nodes_grow(old_xtype, max_nodes, variables_dict, func_array)
            if verbose:
                print('Received the following list', len(label_list), label_list, arity_list)

            if sum(arity_list) + 1 != len(arity_list) or len(label_list) > max_nodes:
                print_warning('w', 'Something went wrong in {}. {} {}'.format(test_case, label_list, arity_list))
                print_warning('w', 'This is wrong: {}!={} {}>{}'.format(sum(arity_list) + 1, len(arity_list), len(label_list), max_nodes))
                worked_fine = False

            tree = karoo_tree_from_labellist(label_list)
            if not tree_check_child_xtype(tree, variables_dict=variables_dict):
                print('WHYY', tree[N_label])

    return worked_fine


def get_two_sample_trees():
    label_list = MountainCarExamples.tree_v2_list
    modify_list = MountainCarExamples.tree_v2_modify
    p_tree = Plagih_Tree(label_list, modify_list=modify_list)
    tree1 = p_tree.get_uninstanced_tree()

    p_tree = Plagih_Tree(label_list)
    tree2 = p_tree.get_uninstanced_tree()
    return tree1, tree2


def test_tree_layers():
    tree1, tree2 = get_two_sample_trees()
    print(tree1, '\n\n', tree2)

    for tree in [tree1, tree2]:
        print('\nnew tree')
        layer_ids = tree_get_layer_fix(tree)
        print('Last non-modifiable layer:', layer_ids)
        layer_ids = tree_get_layer_fix(tree, get_all_leaves=True)
        print('Last non-modifiable layer, all:', layer_ids)
        layer_ids = tree_get_mutatable_layer_lv0(tree)
        print('Fist modifiable layer:', layer_ids)
        layer_ids = tree_get_mutatable_layer(tree, 0, sum_layers=False)
        print('Layer dist = 0:', layer_ids)
        layer_ids = tree_get_mutatable_layer(tree, 1, sum_layers=False)
        print('Layer dist = 1:', layer_ids)
        layer_ids = tree_get_mutatable_layer(tree, 2, sum_layers=False)

        print('Layer dist = 2:', layer_ids)
        layer_ids = tree_get_mutatable_layer(tree, 0, sum_layers=True)
        print('Layer dist = 0, sum:', layer_ids)
        layer_ids = tree_get_mutatable_layer(tree, 1, sum_layers=True)
        print('Layer dist = 1, sum:', layer_ids)
        layer_ids = tree_get_mutatable_layer(tree, 2, sum_layers=True)
        print('Layer dist = 2, sum:', layer_ids)

        layer_ids = tree_get_mutatable_layer(tree, 1, sum_layers=True, get_closest=False, return_all_layers=True)
        print('All Layers till 1:', layer_ids)


def test_tree_evolve_branch_multiple():
    label_list = MountainCarExamples.tree_v2_list
    modify_list = MountainCarExamples.tree_v2_modify
    p_tree = Plagih_Tree(label_list, modify_list=modify_list)
    # p_tree = Plagih_Tree(label_list)
    tree = p_tree.get_uninstanced_tree()
    max_nodes = 15
    variables_dict = TestHelpers.variables_dict
    func_array = TestHelpers.func_array
    tree = tree_evolve_branch_multiple(tree, max_nodes, variables_dict, func_array)
    tree = tree_evolve_branch_multiple(tree, max_nodes, variables_dict, func_array)
    tree = tree_evolve_branch_multiple(tree, max_nodes, variables_dict, func_array)
    tree = tree_evolve_branch_multiple(tree, max_nodes, variables_dict, func_array)
    tree = tree_evolve_branch_multiple(tree, max_nodes, variables_dict, func_array)
    tree = tree_evolve_branch_multiple(tree, max_nodes, variables_dict, func_array)
    print(tree)


def test_list_split():
    fail_cnt = 0

    for sample_size in range(10, 100):
        x = randomly_split_range(sample_size, 3)
        print(x)
        if sum(x) < sample_size:
            fail_cnt += 1
        elif sum(x) > sample_size:
            fail_cnt += 1
    print('test_list_split failed {} times.'.format(fail_cnt))


def test_tree_get_ids_deepsearch():
    label_list = MountainCarExamples.tree_test_plus_list
    modify_list = MountainCarExamples.tree_test_plus_modify_v1
    p_tree = Plagih_Tree(label_list, modify_list=modify_list)
    tree = p_tree.get_uninstanced_tree()
    print('tree:', tree)

    nodes_deepsearch = tree_get_ids_depthfirst(tree)
    print('nodes_deepsearch', nodes_deepsearch)


def test_tree_viz_latex():
    label_list = MountainCarExamples.tree_test_plus_list
    modify_list = MountainCarExamples.tree_test_plus_modify_v1
    p_tree = Plagih_Tree(label_list, modify_list=modify_list)
    tree3 = p_tree.get_uninstanced_tree()

    result = tree_viz_get_latex(tree3)
    print(result)


def test_tree_set_modifyable_nodes():

    variables_dict = TestHelpers.variables_dict
    func_array = TestHelpers.func_array
    origin, tree2 = get_two_sample_trees()
    print('a origin', origin[N_modify])
    tree_new = tree_evolve_branch_multiple(origin.copy(), 25, variables_dict, func_array)
    print('b origin', origin[N_modify])
    tree_new = tree_set_modifyable_nodes(tree_new, origin)
    print('c origin', origin)
    print(tree_new)


test_tree_viz_latex()
