'''
This is a lightweight python-code to tree converter.
Note that every expression must be in a seperate line.
Most python operators are probably not defined.
'''

import numpy as np
import gym
import matplotlib.pyplot as plt
from mountaincar.agents.sarsa_best_approach import sarsa_start_training
import inspect
import re

from mountaincar.agents.move_directions import move_towards_direction, choose_action

op3 = ('if')
op2 = ('+','-','*','/','<','>','=')
op_del = ('\\', '#', 'def')

# def choose_action(state):
#     if \
#             state[1] < 0: #PLAGI
#         return 0
#     else:
#         return 2
#
# Karoo Tree components (Breite):
# --------> if, <, 0, 2, state[1], 0
#
# PLAGIH Tree Components
# --------> if, <, state[1], 0, 0, 2
# fixhere-> 0,  1,        1, 1, 0, 0
#
#                 if
#             /   |    \
#            <    0     2
#          /   \
#     state[1]   0
#
#   KAROO Representation
#
# TREE_ID,          1,,,,
# tree_type,        X,,,,
# tree_depth_base,  3,,,,
# NODE_ID,          1,      2,    3,      4,      5,      6
# node_depth,       0,      1,    1,      1,      2,      2
# node_type,        ifte,   bool, gyma,   gyma,   gymo, term
# node_label,       if,     <,    a=>0,   a=>2,   o[1],   0
# node_parent,      ,       1,    1,      1,      2,      2
# node_arity,       3,      2,    0,      0,      0,      0
# node_c1,          2,      4,    ,       ,        ,
# node_c2,          3,      5,    ,       ,        ,
# node_c3,          4,       ,    ,       ,        ,
# fitness,          1.0,  31,,,
# node_modify,      0,      1,    0,      0,      1,      0


def get_karoo_from_python(code):
    return


def split_line_to_exp(line):

    if line.startswith('def ') or line.startswith('#'):
        return
    elif 'if' in line:
        print('IF found. Handle if block')
        return 'if'
    return


code = inspect.getsource(choose_action)
line = ''
tmp = [a if a != '\n' else '@' for a in code]
code = ''.join(tmp)
print(code)

# for l in code:
#     line = line + l
#     if l == '\n':
#         # print('New Line found: ' + line)
#         #line_to_tree(line)
#         line = ''
print(re.findall(r'def', code))









# def choose_action(state):
#     if state[1] < 0: #PLAGI
#         return 0
#     else:
#         return 2

# def choose_action(state):
#     if \
#             state[1] < 0: #PLAGI
#         return 0
#     else:
#         return 2
# ['if', '<', '0', '2', 'o[1]', '0']
# [   0,   1,   0,   0,      1,   1]
