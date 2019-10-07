import numpy as np
import gym
import matplotlib.pyplot as plt
from mountaincar.agents.sarsa_best_approach import sarsa_start_training
import inspect
import re

from mountaincar.agents.move_directions import move_towards_direction, choose_action


# if \
#             state[1] < 0: #PLAGI
#         return 0
#     else:
#         return 2

def get_karooTree_from_pythonCode(code):
    try:
        print('Hi')
    except:
        print('Dont know what to do')

    return


print(inspect.getsource(choose_action))
# print(inspect.getsourcelines(choose_action))
