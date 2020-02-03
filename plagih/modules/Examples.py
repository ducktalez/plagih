import tensorflow as tf
from pathlib import Path


class MountainCarExamples:
    files = {
        'samples_file': '../mountaincar/karoo_files/data_samples/behaviour_samples.csv',
        'samples_pickle': '../mountaincar/karoo_files/data_samples/plagih_data_prepared.p',
        'operators_file': '../mountaincar/karoo_files/operators/operators.csv',
        'backup_pop': 'runs/Best_of/old_v1/population_new.csv'}

    tree_v1_list = ['Ifte', '<', '0.0', '2.0', 'observation1', '0.0']
    tree_v1_modify = [0, 1, 0, 0, 1, 1]
    tree_v1_expr = 'Ifte(observation1 < 0.0, 0.0, 2.0)'

    tree_v2_list = ['Ifte', '<', '0.0', 'Ifte', 'observation1', '0.0', 'True', '2.0', '1.0']
    tree_v2_modify = [0, 1, 0, 0, 1, 1, 1, 0, 0]
    tree_v2_expr_sym = 'Ifte(observation1 < 0.0, 0.0, 2.0)'
    tree_v2_expr_raw = '(Ifte(((observation1)<(0.0)), (0.0), (Ifte((True), (2.0), (1.0)))))'

    tree_v3_list = ['Ifte', '&', '2', '0', '<=', '<=', 'Mini', 'observation1', 'observation1', '+', '+', '-', '*', '0.7',
                    '*', '0.03', '*', '0.008', '-0.07', '**',
                    '-0.09', '**', '0.3', '**', '+', '2', '+', '2', '+', '4', 'observation0', '0.38', 'observation0', '0.25',
                    'observation0', '0.9']
    tree_v3_modify = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1]
    tree_v3_expr_raw = '(Ifte((((Mini((((-0.09)*(((observation0)+(0.25))**(2)))+(0.03)), (((0.3)*(((observation0)+(0.9))**(' \
                       '4)))-(0.008))))<=(observation1))&((observation1)<=(((-0.07)*(((observation0)+(0.38))**(2)))+(' \
                       '0.7)))), (2), (0)))'
    tree_v3_expr_sym = 'Ifte((observation1 <= 0.7 - 0.07*(observation0 + 0.38)**2) & (Mini(0.03 - 0.09*(observation0 ' \
                       '+ 0.25)**2, 0.3*(observation0 + 0.9)**4 - 0.008) <= observation1), 2, 0)'
    tree_v3_new = ['Ifte', '&', 2, 0, '<=', '<=', 'Mini', 'observation1', 'observation1', '+', '+', '-', '*', 0.7, '*',
                   0.03, '*', 0.008, '-', '**', '-', '**', 0.3, '**', 0.07, '+', 2, 0.09, '+', 2, '+', 4,
                   'observation0', 0.38, 'observation0', 0.25, 'observation0', 0.9]

    tree_plus_list = ['+', '-', '-', '*', '*', '*', '*', 1, 2, 3, 4, 5, 6, 7, 8]
    tree_plus_modify_v1 = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    fake_tensors = {'observation0': tf.constant(1.1, dtype=tf.float32),
                    'observation1': tf.constant(2.2, dtype=tf.float32),
                    'bl': tf.constant(True, dtype=tf.bool)}

    expr_test1 = 'Ifte(1.019*(-0.09)**b*(0.98 - 0.13) + Mini(b, observation0) > -0.97, 0.0, 2.0)'
