# coding=utf-8
from benchmarks.industrial_benchmark_python.IDS import IDS
import numpy as np
import matplotlib.pyplot as plt
import collections.deque

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

class Agent_simple():

    def __init__(self):
        self.remember = {'v': np.zeros(6),
                         'f': np.zeros(4),
                         'p': np.zeros(1),
                         'h': np.zeros(5)}

        self.f = np.zeros(4)
        self.p = np.zeros(4)
        self.h = np.zeros(4)}

    def decide(self, observation):
        observation0, observation1 = observation
        if -observation1 + min(observation1, observation0 + 1.025) > observation1:
            return 0
        else:
            return 2

n_trajectories = 10
T = 1000

data = np.zeros((n_trajectories, T))
data_cost = np.zeros((n_trajectories, T))

for k in range(n_trajectories):
    env = IDS(p=100)
    for t in range(T):
        at = np.array([0, 0, 0])
        # at = np.array([50-env.state['v'], 50-env.state['g'], 50-env.state['h']])
        # at = 2 * np.random.rand(3) - 1
        markovStates = env.step(at)
        data[k, t] = env.visibleState()[-1]

plt.plot(data.T)
plt.xlabel('T')
plt.ylabel('Reward')
plt.show()
