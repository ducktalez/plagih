from plagih.modules.plagih_eval import *
from plagih.modules.plagih_tree import *
from plagih.modules.Examples import *
from plagih.modules.operators import *
import time


# todo slowly make all of those random tests worth something


class TestHelpers:

    def __init__(self):
        self.func_arr_dummy = [[[], ['sin', 'cos', '~'], ['+', '+', '+', '-', '*', '/'], []],
                               [[], [], ['<', '>', '==', '!='], []],
                               [[], ['Not', 'Not'], ['Andb'], []],
                               [[], [], [], []],
                               [[], [], [], ['Ifte']]]

        self.env_bundle = {'obs_name': {'a': {'label': 'a', 'type': 'float', 'xtype': '2f'},
                                        'b': {'label': 'b', 'type': 'float', 'xtype': '2f'},
                                        'c': {'label': 'c', 'type': 'float', 'xtype': '2f'},
                                        'd': {'label': 'd', 'type': 'float', 'xtype': '2f'},
                                        'pos': {'label': 'cartPos', 'type': 'float', 'xtype': '2f'},
                                        'vel': {'label': 'cartVel', 'type': 'float', 'xtype': '2f'},
                                        'bool1': {'label': 'observation1', 'type': 'bool', 'xtype': '2b'},
                                        'bool2': {'label': 'observation1', 'type': 'bool', 'xtype': '2b'}
                                        },
                           '2f': ['a', 'b', 'c', 'd'],
                           '2b': ['bool1', 'bool2']}

        self.tree_MTC_simon_labels = ['Ifte',
                                      'Orb', '2', 'Ifte',
                                      '<', 'Andb', 'Andb', '0', 'Ifte',
                                      'cartVel', '1', '<', '<', '<', 'Andb', '<', '0', '2',
                                      'cartVel', '0.1', 'cartPos', '-0.05', 'cartPos', '0.02', '>', '<', 'cartPos', '0',
                                      'cartVel', '-0.45', 'cartVel', '-0.05']

        self.distributions_as_string = {'2f': [lambda: np.random.normal(1,2),
                                               lambda: np.random.normal(1,1),
                                               lambda: np.random.randint(0, 10)],
                                        '2b': [lambda: np.random.choice([True, False])]}

        self.tree_MTC_simon_expr = 'Ifte(Orb(pos < -1,  Andb(pos < 0.1, vel < -0.05)), 2, Ifte(Andb(Andb(pos > -0.45, pos < -0.05), vel < 0.02), 0,  Ifte(vel < 0, 0, 2)))'

    # example func_arr_dummy. Note that (for the random choice) functions can be included more often
    def karoo_tree_from_only_labellist(self, label_list, modify_list=None):
        xtype_list = xtypes_from_labels(label_list, self.env_bundle)
        p_tree = Plagih_Tree(label_list, xtype_list, modify_list=modify_list)
        tree = p_tree.get_uninstanced_tree()
        return tree

    def make_all_known_trees(self):
        tree = self.karoo_tree_from_only_labellist(self.tree_MTC_simon_labels)
        expr_raw = tree_get_expr_raw(tree, root_id)
        expr_sym = expr_sympify(expr_raw)
        print(expr_sym)
        # tree_sym = karoo_tree_from_expr(expr_sym, self.env_bundle)
        # print('sym\t', ','.join(tree_get_labellist(tree_sym)))
        sym_tree2 = tree_evolve_reduce(tree, self.env_bundle)
        print(tree_check_deep(sym_tree2, self.env_bundle))

        label_list = ast_convert_from_expr(self.tree_MTC_simon_expr, build=True)
        label_list = workaround_remove_tilde_operator(label_list)

    def auto_operator_tree_build(self):
        """
        test building all potential trees with all genetic operators
        """
        # sfeh float + bool
        result_xtype = '2f'
        goal_max_nodes = 14
        env_variables = self.env_bundle
        oparray = get_all_oparrays()
        choose_distributions = self.distributions_as_string
        build_type='grow'
        # origin_tree = karoo_tree_from_labellist(['+', '1', '2'], env_variables, modify_list=[0, 1, 1])

        label_list, arity_list, xtype_list = invent_label_list_nodes(result_xtype, goal_max_nodes, env_variables, oparray, choose_distributions, build_method=build_type)
        modify_list = [0] + ([1] * len(label_list))[1:]
        tree = karoo_tree_from_labellist(label_list, env_variables, modify_list=modify_list)
        for x in range(100):
            # print('ss', tree_pretty_print(tree))

            tree = self.tree_evolve_branch_multiple(tree, goal_max_nodes, env_variables, oparray, choose_distributions)
            # if not tree_check_deep(tree, env_variables):

        return



class MountainCarExamples:


    tree_v1_list = ['Ifte', '<', '0.0', '2.0', 'vel', '0.0']
    tree_v1_modify = [0, 1, 0, 0, 1, 1]
    tree_v1_expr = 'Ifte(vel < 0.0, 0.0, 2.0)'

    tree_v2_list = ['Ifte', '<', '0.0', 'Ifte', 'vel', '0.0', 'True', '2.0', '1.0']
    tree_v2_modify = [0, 1, 0, 0, 1, 1, 1, 0, 0]
    tree_v2_expr_sym = 'Ifte(vel < 0.0, 0.0, 2.0)'
    tree_v2_expr_raw = '(Ifte(((vel)<(0.0)), (0.0), (Ifte((True), (2.0), (1.0)))))'

    tree_v3_list = ['Ifte', 'Andb', '2', '0', '<=', '<=', 'Mini', 'vel', 'vel', '+', '+', '-', '*', '0.7',
                    '*', '0.03', '*', '0.008', '-0.07', '**',
                    '-0.09', '**', '0.3', '**', '+', '2', '+', '2', '+', '4', 'pos', '0.38', 'pos', '0.25',
                    'pos', '0.9']
    tree_v3_modify = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1]

    tree_v3_new = ['Ifte', 'Andb', 2, 0, '<=', '<=', 'Mini', 'vel', 'vel', '+', '+', '-', '*', 0.7, '*',
                   0.03, '*', 0.008, '-', '**', '-', '**', 0.3, '**', 0.07, '+', 2, 0.09, '+', 2, '+', 4,
                   'pos', 0.38, 'pos', 0.25, 'pos', 0.9]

    tree_test_minus_list = ['*', 'a', '-2']

    tree_test_plus_list = ['+', '-', '-', '*', '*', '*', '*', 1, 2, 3, 4, 5, 6, 7, 8]
    tree_test_plus_modify_v1 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    fake_tensors = {'pos': tf.constant(1.1, dtype=tf.float32),
                    'vel': tf.constant(2.2, dtype=tf.float32),
                    'bl': tf.constant(True, dtype=tf.bool)}

    expr_test1 = 'Ifte(1.019*(-0.09)**b*(0.98 - 0.13) + Mini(b, pos) > -0.97, 0.0, 2.0)'


class CartpoleExamples:

    files = {
        'samples_file': '../benchmarks/gym_cartpole/gp_files/samples.csv',
        'samples_pickle': '../benchmarks/gym_cartpole/gp_files/prepared_samples.p',
        'operators_file': '../benchmarks/gym_cartpole/gp_files/operators.csv'}

    label_list = ['Ifte', 'True', 0, 1]
    modify_list = [0, 1, 0, 0]


def check_op_names():
    # check if the fun-names are correct
    for key, value in op.items():
        if not value['fun'] in op:
            print('The op-dict has an entry: {} with fun: {} Which is not in op.'.format(key, value['fun']))
            return False

    # check if the pycode-functions are working with this arity
    return True


def runtime_exception_vs_if():
    value = '5'  # 5
    range_size = 2000000

    print('if     exVal  exExc  raw')

    for value in [5, '5', '']:

        v1_start = time.perf_counter()
        for _ in range(range_size):
            if value == '':
                x = value
            else:
                x = int(value)
        v1_end = time.perf_counter() - v1_start

        v2_start = time.perf_counter()
        for _ in range(range_size):
            try:
                x = int(value)
            except ValueError:
                x = value
        v2_end = time.perf_counter() - v2_start

        v3_start = time.perf_counter()
        for _ in range(range_size):
            try:
                x = int(value)
            except:
                x = value
        v3_end = time.perf_counter() - v3_start

        v4_start = time.perf_counter()
        for _ in range(range_size):
            x = value
        v4_end = time.perf_counter() - v4_start

        print('{:4.4f} {:4.4f} {:4.4f} {:4.4f}'.format(v1_end, v2_end, v3_end, v4_end))


def test_tree_visualize_reduced():
    obs_bundle = {'obs_name': {'a': {'label': 'a', 'type': 'float', 'xtype': '2f'},
                               'b': {'label': 'b', 'type': 'float', 'xtype': '2f'},
                               'c': {'label': 'c', 'type': 'float', 'xtype': '2f'},
                               'd': {'label': 'd', 'type': 'float', 'xtype': '2f'},
                               'bool1': {'label': 'observation1', 'type': 'bool', 'xtype': '2b'},
                               'bool2': {'label': 'observation1', 'type': 'bool', 'xtype': '2b'}
                               }}

    labellists = [['+', 'a', '2'],
                  ['&', '&', '<', 'True', 'bool1', 'Maxi', 4, '+', 1, '*', '/', 'a', 'b', 'c', 3],
                  ['Maxi', 1, '+', 'a', 'b'],
                  ['+', '-', 'Maxi', 1, 2, 3, 4]]

    forest_grouped = []
    for label_list in labellists:
        xtype_list = xtypes_from_labels(label_list, obs_bundle)
        tree = karoo_tree_from_labellist(label_list, xtype_list)
        vistree = visualize_tree_get_vistree(tree)
        forest_grouped.append(latex_tree_get_forest(tree))

    latex_file = latex_complete_tree_summary(forest_grouped)

    with Path.open(Path('textreetest.tex'), 'w') as file:
        file.write(latex_file)


live_test = TestHelpers()
live_test.auto_operator_tree_build()
