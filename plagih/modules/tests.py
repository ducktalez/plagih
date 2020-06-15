from plagih.modules.plagih_eval import *
from plagih.modules.plagih_tree import *
from plagih.modules.operators import *
import time
from pathlib import Path
from plagih.tree_distances.tree_edit_distance import *


class TestHelpers:

    def __init__(self):
        self.func_arr_dummy = [[[], ['sin', 'cos', '~'], ['+', '+', '+', '-', '*', '/'], []],
                               [[], [], ['<', '>', '==', '!='], []],
                               [[], ['Notb', 'Notb'], ['Andb'], []],
                               [[], [], [], []],
                               [[], [], [], ['Ifte']]]

        self.env_bundle = {'obs_name': {'a': {'label': 'a', 'type': 'float', 'xtype': '2f'},
                                        'b': {'label': 'b', 'type': 'float', 'xtype': '2f'},
                                        'c': {'label': 'c', 'type': 'float', 'xtype': '2f'},
                                        'd': {'label': 'd', 'type': 'float', 'xtype': '2f'},
                                        'pos': {'label': 'cartPos', 'type': 'float', 'xtype': '2f'},
                                        'vel': {'label': 'cartVel', 'type': 'float', 'xtype': '2f'},
                                        'bool1': {'label': 'cartVel', 'type': 'bool', 'xtype': '2b'},
                                        'bool2': {'label': 'cartVel', 'type': 'bool', 'xtype': '2b'}
                                        },
                           '2f': ['a', 'b', 'c', 'd'],
                           '2b': ['bool1', 'bool2']}

        self.tree_MTC_simon_labels = ['Ifte',
                                      'Orb', '2', 'Ifte',
                                      '<', 'Andb', 'Andb', '0', 'Ifte',
                                      'cartVel', '1', '<', '<', '<', 'Andb', '<', '0', '2',
                                      'cartVel', '0.1', 'cartPos', '-0.05', 'cartPos', '0.02', '>', '<', 'cartPos', '0',
                                      'cartVel', '-0.45', 'cartVel', '-0.05']

        self.tree_MTC_simon_expr = 'Ifte(Orb(pos < -1,  Andb(pos < 0.1, vel < -0.05)), 2, Ifte(Andb(Andb(pos > -0.45, pos < -0.05), vel < 0.02), 0,  Ifte(vel < 0, 0, 2)))'
        self.tree1 = TEST_karoo_tree_from_labellist(['+', '+', '*', '-', '1', '2', '3', '4', '5'], self.env_bundle)
        self.tree2 = TEST_karoo_tree_from_labellist(['+', '-', '*', '1', '2', '3', '4'], self.env_bundle)

    def test_ted_weighting(self):
        distance, mapping = tree_parsimony_ted(self.tree1, self.tree2)
        print('Mapping:\n' + '\n'.join([str(x) for x in mapping]))
        weighted_distance = weight_ted_mapping(mapping)
        print('\nDistance:', distance, 'Mapping_distance:', weighted_distance)

    def karoo_tree_from_only_labellist(self, label_list, modify_list=None):
        xtype_list = xtypes_from_labels(label_list, self.env_bundle)
        p_tree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list)
        tree = p_tree.get_uninstanced_tree()
        return tree

    def make_all_known_trees(self):
        tree = self.karoo_tree_from_only_labellist(self.tree_MTC_simon_labels)
        expr_raw = tree_get_expr_raw(tree, root_id)
        expr_sym = expr_sympify(expr_raw)
        print(expr_sym)
        # tree_sym = karoo_ptree_from_expr(expr_sym, self.env_bundle)
        # print('sym\t', ','.join(tree_get_labellist(tree_sym)))
        sym_tree2 = tree_evolve_reduce(tree, self.env_bundle)
        print(tree_check_deep(sym_tree2, self.env_bundle))

        label_list = ast_convert_from_expr(self.tree_MTC_simon_expr, build=True)
        label_list = workaround_remove_tilde_operator(label_list)


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


# live_test = TestHelpers()
# live_test.test_ted_weighting()

print(ast_convert_from_expr('Ifte((((Mini(((-0.09)*(((cartPos)+(0.25))**(2)))+(0.03), ((0.3)*(((cartPos)+(0.9))**(4)))-(0.008)))<=(cartVel))&((cartVel)<=(((-0.07)*(((cartPos)+(0.38))**(2)))+(0.7)))), (2), (0))', build=True))
