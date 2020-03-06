# coding=utf-8
from benchmarks.industrial_benchmark_python.IDS import IDS
import numpy as np
import matplotlib.pyplot as plt
import collections

'''
The MIT License (MIT)

Copyright 2017 Siemens AG

Author: Stefan Depeweg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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


def get_max_history(index, history_list):
    """
    Returns
    """
    if index > len(history_list)-1:
        # print('Not enough history for index {}, len list: {}'.format(index, len(history_list)))
        return history_list[-1]
    else:
        return history_list[index]


class Agent_simple():

    def __init__(self):

        self.state_history = collections.deque()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0])
        at[0] = -get_max_history(5, self.state_history)['v'] - 0.91
        at[1] = 2*get_max_history(3, self.state_history)['f'] - state['p'] + 1.43
        at[2] = -3.48*get_max_history(3, self.state_history)['h'] - get_max_history(4, self.state_history)['h'] + 2*state['p'] + 0.81
        return at
        # at = np.array([50-env.state['v'], 50-env.state['g'], 50-env.state['h']])
        # at = 2 * np.random.rand(3) - 1


class Agent_blackbox():

    def __init__(self):
        self.state_history = collections.deque()

    def get_h(self, name, steps):
        return get_max_history(steps, self.state_history)[name]

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0])
        at[0] = -get_max_history(4, self.state_history)['v'] - 0.96
        at[1] = 0.41*get_max_history(1, self.state_history)['f'] + self.get_h('f', 4) - 0.59 * state['p'] + 0.77
        at[2] = -4.05*get_max_history(3, self.state_history)['h'] - get_max_history(4, self.state_history)['h'] + 2.26*state['p'] + 0.90
        return at


class Agent_nothing():
    """
    ...does nothing, kind of.
    """
    def __init__(self):
        self.state_history = collections.deque()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0])
        at[0] = 50
        at[1] = 50
        at[2] = 50
        return at


class Agent_random():
    """
    ...does nothing, kind of.
    """
    def __init__(self):
        self.state_history = collections.deque()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        # at = np.array([50-env.state['v'], 50-env.state['g'], 50-env.state['h']])
        at = 2 * np.random.rand(3) - 1
        return at


n_trajectories = 10
T = 10000
bad_agents = [Agent_nothing()]
agents = [Agent_random(), Agent_simple(), Agent_blackbox()]

data = np.zeros((n_trajectories*len(agents), T))
data_cost = np.zeros((n_trajectories, T))


# for k in range(n_trajectories):
for k, agent in enumerate(agents):
    env = IDS(p=100)
    for t in range(T):
        at = agent.decide(env.state)
        # at = np.array([50-env.state['v'], 50-env.state['g'], 50-env.state['h']])
        # at = 2 * np.random.rand(3) - 1
        markovStates = env.step(at)
        data[k, t] = env.visibleState()[-1]
    print('Average fitness: {}'.format(np.average(data[k])))

plt.plot(data.T)
plt.xlabel('T')
plt.ylabel('Reward')
plt.show()
