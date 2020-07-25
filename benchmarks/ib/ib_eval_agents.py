# coding=utf-8
from benchmarks.ib.IDS import IDS
import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ib.test_agents import *
from pathlib import Path
import csv

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


def envstate_normalize(env_state, to_normal=True):
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

    if to_normal:
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

    T = 100*1000

    # agents = [Agent_Daniel_Best()]
    agents = [
        # # Agent_random(),
        # Agent_nothing(),
        # Agent_daniel_21(),
        # Agent_daniel_27(),
        # Agent_Daniel_29_Best(),
        # Agent_Udluft(),
        Agent_sim1(),
        Agent_Test()
    ]

    data = np.zeros((len(agents), T))
    data_cost = np.zeros((len(agents), T))
    time_horizon = 100
    factor = 0.97

    # for k in range(n_trajectories):
    for k, agent in enumerate(agents):
        sum = 0
        for p in np.arange(10, 101, 10):
            env = IDS(p=p)

            # state_debug = []
            sum_t = 0
            for t in range(time_horizon):
                env_state = envstate_normalize(env.state)
                at = agent.decide(env_state)
                markovStates = env.step(at)

                entry = env.visibleState()[-1]
                data[k, t] = entry

                # entry = data[k][-1 - t]
                sum_t += factor ** (time_horizon-t) * entry
            sum += sum_t / 10

        print('Discounted reward sum (p=10,20,..,100; 1000 steps)', agent.name, sum)


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


agent_list = [
    ('06_50', '2', )
]
eval_agents()