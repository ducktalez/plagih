import tensorflow as tf
from pathlib import Path

# todo: this file seems outdated, as runs are started within a folder now.


class MountainCarExamples:
    files = {
        'samples_file': '../benchmarks/gym_mountaincar/gp_files/data_samples/samples.csv',
        'samples_pickle': '../benchmarks/gym_mountaincar/gp_files/data_samples/plagih_data_prepared.p',
        'operators_file': '../benchmarks/gym_mountaincar/gp_files/operators/operators.csv'}

    tree_v1_list = ['Ifte', '<', '0.0', '2.0', 'vel', '0.0']
    tree_v1_modify = [0, 1, 0, 0, 1, 1]
    tree_v1_expr = 'Ifte(vel < 0.0, 0.0, 2.0)'

    tree_v2_list = ['Ifte', '<', '0.0', 'Ifte', 'vel', '0.0', 'True', '2.0', '1.0']
    tree_v2_modify = [0, 1, 0, 0, 1, 1, 1, 0, 0]
    tree_v2_expr_sym = 'Ifte(vel < 0.0, 0.0, 2.0)'
    tree_v2_expr_raw = '(Ifte(((vel)<(0.0)), (0.0), (Ifte((True), (2.0), (1.0)))))'

    tree_v3_list = ['Ifte', '&', '2', '0', '<=', '<=', 'Mini', 'vel', 'vel', '+', '+', '-', '*', '0.7',
                    '*', '0.03', '*', '0.008', '-0.07', '**',
                    '-0.09', '**', '0.3', '**', '+', '2', '+', '2', '+', '4', 'pos', '0.38', 'pos', '0.25',
                    'pos', '0.9']
    tree_v3_modify = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1]
    tree_v3_expr_raw = '(Ifte((((Mini((((-0.09)*(((pos)+(0.25))**(2)))+(0.03)), (((0.3)*(((pos)+(0.9))**(' \
                       '4)))-(0.008))))<=(vel))&((vel)<=(((-0.07)*(((pos)+(0.38))**(2)))+(' \
                       '0.7)))), (2), (0)))'
    tree_v3_expr_sym = 'Ifte((vel <= 0.7 - 0.07*(pos + 0.38)**2) & (Mini(0.03 - 0.09*(pos ' \
                       '+ 0.25)**2, 0.3*(pos + 0.9)**4 - 0.008) <= vel), 2, 0)'
    tree_v3_new = ['Ifte', '&', 2, 0, '<=', '<=', 'Mini', 'vel', 'vel', '+', '+', '-', '*', 0.7, '*',
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


