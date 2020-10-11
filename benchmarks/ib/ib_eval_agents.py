# coding=utf-8
from benchmarks.ib.IDS import IDS
import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ib.test_agents import *
from pathlib import Path
import csv
import math
from itertools import accumulate

'''
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Stefan Depeweg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation gp_files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


def envstate_normalize(env_state, to_daniel=True):
    """
    velocity, gain, shift (v, g, h)
    p, v, g, h, f, c
    'p', 'v', 'g', 'h', 'f', 'c'
    SetPoint_0 Velocity_0 Gain_0 Shift_0 Fatigue_0 Consumption_0
    """

    IB_norm_dict = {
        'p': [55.0, 28.72],
        'v': [48.75, 12.31],
        'g': [50.53, 29.91],
        'h': [49.45, 29.22],
        'f': [37.51, 31.17],
        'c': [166.33, 139.44]}

    if to_daniel:
        norm_values = {}
        for k, val in IB_norm_dict.items():
            norm_values[k] = (env_state[k] - val[0]) / val[1]
        return norm_values
    else:
        real_values = {}
        for k, val in IB_norm_dict.items():
            real_values[k] = (env_state[k] * val[1]) + val[0]
        return real_values


def eval_agents():

    # agents = [Agent_Daniel_Best()]
    agents = [
        Agent_random(),
        # Agent_nothing(),
        # Agent_daniel_21(),
        # Agent_daniel_27(),
        Agent_Daniel_29_Best(),
        # Agent_505050(),
        # Agent_Udluft(),
        # Agent_sim1(),
        # Agent_Test()
    ]

    # for k in range(n_trajectories):
    for k, agent in enumerate(agents):
        sum_all = eval_agent(agent, randomize=0)
        sum_safe = eval_agent(agent, safe_eval=True, randomize=0)
        sum_allr50 = eval_agent(agent, randomize=50, repeat_avg=10)
        sum_safer50 = eval_agent(agent, safe_eval=True, randomize=50, repeat_avg=10)
        print(f'Results : {agent.name} \t{sum_all:5.1f} \t (safe: {sum_safe:5.1f}) \t(r50: all: {sum_allr50:4.1f} \tsafe: {sum_safer50:4.1f})')


class AgentMerger(Ib_Agent):
    """

    """

    def __init__(self, name, a0, a1, a2):
        self.name = f'merged_{name}'
        super().__init__()
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    def decide(self, env_state):
        self.state_history.appendleft(env_state)

#        if len(self.state_history) > 10:
#            self.state_history.pop()

        at = np.array([0, 0, 0], dtype=np.float32)
        SetPoint = self.get_h('p', 0)

        exec('at[0] = min(max(-1, ' + self.a0 + '), 1)')
        exec('at[1] = min(max(-1, ' + self.a1 + '), 1)')
        exec('at[2] = min(max(-1, ' + self.a2 + '), 1)')

        return at


def eval_combined_agents(codes, complete=True, randomize=0, repeat_avg=1):

    if complete:
        a0, a1, a2 = codes
    else:
        a0, a1, a2 = codes
        a2 = '0'

    dummy_agent = AgentMerger('Herbert', a0, a1, a2)

    return eval_agent(dummy_agent, randomize=randomize, repeat_avg=repeat_avg)


def eval_agent(agent, safe_eval=False, randomize=0, repeat_avg=1):
    """
    Results: Daniel_21 -5263 (5651)
    Results: Daniel_27 -5275 (5628)
    Results: Daniel_29 -5270 (5611)
    """

    discount_factor = 0.97
    T = 100  # time_horizon
    discount_len = -100
    sum_discounted_p = []
    # setpoint_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # conuts_dict_copypaste = {10: 11, 20: 7, 30: 16, 40: 10, 50: 11, 60: 10, 70: 12, 80: 7, 90: 6, 100: 10}

    # for p in [30, 60, 10, 30, 70, 90, 30, 50, 20, 60]:
    # for p in [30, 60, 10, 30, 70, 90, 30, 50, 20, 60, 10, 50, 100, 90, 50, 20, 90, 50, 30, 50, 40, 100, 30, 30, 80, 80, 20, 70, 10, 70, 40, 10, 70, 60, 70, 40, 100, 30, 40, 50, 10, 40, 20, 30, 20, 80, 20, 70, 10, 30, 80, 10, 30, 30, 70, 40, 40, 30, 10, 40, 80, 100, 70, 60, 50, 10, 30, 40, 100, 90, 60, 60, 60, 30, 70, 30, 70, 30, 10, 60, 80, 20, 50, 100, 90, 80, 50, 70, 40, 90, 100, 50, 10, 100, 50, 60, 100, 70, 60, 100]:
    for rr in range(repeat_avg):
        for p in range(10, 101, 10):

            env = IDS(p=p)
            for num in range(randomize):  # get a more bad state
                at = 2 * np.random.rand(3) - 1  # "random"-function from og ib
                env.step(at)  # also returns markovStates, if needed

            sums_p = []
            for t in range(T):
                normal_state = envstate_normalize(env.state)
                at = agent.decide(normal_state)
                if safe_eval:
                    at[2] = 0
                env.step(at)  # also returns markovStates, if needed
                reward = env.visibleState()[-1]
                sums_p.append(reward)

            cumcum_discounted = list(accumulate(sums_p[discount_len:][::-1], lambda x_last, x: (x + (discount_factor * x_last))))
            sum_discounted_p.append(cumcum_discounted[-1])

    return float(np.average(sum_discounted_p))


def agent_create_samples_csv(T=10000):

    history_t = 5
    csv_data = np.zeros((T, 6 * history_t + 3))

    data = np.zeros(T)

    agent = Agent_Daniel_29_Best()
    env = IDS()  # p=100 in examples!

    for t in range(T):
        env_state = envstate_normalize(env.state)

        for ii in range(history_t):
            for enum_x, state_x in enumerate(env_state.values()):
                if t + ii < T:
                    csv_data[t+ii][enum_x * history_t + ii] = state_x

        # state_debug.append(list(env_state.values()))
        at = agent.decide(env_state)
        # at = at * np.array([1, 10, 5.75])
        # at = np.array([50-env.state['v'], 50-env.state['g'], 50-env.state['h']])
        # at = 2 * np.random.rand(3) - 1

        for ii, action in enumerate(at):
            csv_data[t][6 * history_t + ii] = action

        markovStates = env.step(at)
        data[t] = env.visibleState()[-1]
    # debug_array = np.array(state_debug).T
    # for x in range(6):
    #     print('Variable {}: {:.5f} {:.5f}'.format(x, np.mean(debug_array[x]), np.var(debug_array[x])))

    print(Path.cwd())
    print(Path('gp_files/ib_agent29_samples.csv'))
    print(Path('gp_files').is_dir())

    with Path('gp_files/ib_agent29_samples.csv').open('w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csv_data)

    print('DONE!')


if __name__ == "__main__":
    eval_agents()
