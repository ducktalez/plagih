# coding=utf-8
from benchmarks.industrial_benchmark_python.IDS import IDS
import numpy as np
import matplotlib.pyplot as plt
import collections
from pathlib import Path
import csv


def get_max_history(index, history_list):
    """
    Returns
    """
    if index > len(history_list) - 1:
        # print('Not enough history for index {}, len list: {}'.format(index, len(history_list)))
        return history_list[-1]
    else:
        return history_list[index]


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


class Agent_nothing():
    """
    ...does nothing, kind of.
    """

    def __init__(self):
        self.state_history = collections.deque()
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0])
        at[0] = 0
        at[1] = 0
        at[2] = 0
        return at


class Agent_random():
    """
    ...does nothing, kind of.
    """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = 'random'
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        # at = np.array([50-env.state['v'], 50-env.state['g'], 50-env.state['h']])
        at = 2 * np.random.rand(3) - 1
        return at


class Ib_Agent():

    def __init__(self):
        self.state_history = collections.deque()

    def get_h(self, name, steps):
        return get_max_history(steps, self.state_history)[name]


class Agent_daniel_21(Ib_Agent):
    """
    Daniel Hein's best agent for complexity 21
    """

    def __init__(self):
        self.name = 'Daniel_21'
        super().__init__()

    def decide(self, env_state):
        self.state_history.appendleft(env_state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = - self.get_h('v', 5) - 0.91
        at[1] = 2 * self.get_h('f', 3) - env_state['p'] + 1.43
        at[2] = -3.48 * self.get_h('h', 3) - self.get_h('h', 4) + 2 * self.get_h('p', 0) + 0.81
        return at


class Agent_daniel_27(Ib_Agent):

    def __init__(self):
        self.name = 'Daniel_27'
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)
        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0], dtype=np.float32)
        # # idk why, but these are the best??
        # at[0] = -self.get_h('v', 5) - 1.17
        # at[1] = 2 * self.get_h('f', 3) - self.get_h('p', 0) + 1.16
        # at[2] = -3.49 * self.get_h('h', 3) - self.get_h('h', 4) + 2 * self.get_h('p', 0) + 0.82
        at[0] = - self.get_h('v', 4) - 0.96
        at[1] = 0.4 * self.get_h('f', 1) + self.get_h('f', 4) - 0.6 * self.get_h('p', 0) + 0.76
        at[2] = - self.get_h('h', 3) - 1.57 * self.get_h('h', 4) + self.get_h('p', 0) + 0.52
        return at


class Agent_Daniel_29_Best(Ib_Agent):

    def __init__(self):
        self.name = 'Daniel_29'
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)
        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = -self.get_h('v', 4) - 0.96  # 27
        at[1] = 0.41 * self.get_h('f', 1) + self.get_h('f', 4) - 0.59 * self.get_h('p', 0) + 0.77  # 27
        at[2] = -4.05 * self.get_h('h', 3) - self.get_h('h', 4) + 2.26 * self.get_h('p', 0) + 0.90  # 27

        return at


class Agent_Test(Ib_Agent):
    """
    ...does nothing, kind of.
    """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = 'Testagent'
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = -self.get_h('g', 5) - 1.8  # 27
        at[1] = 0.41 * self.get_h('f', 1) + self.get_h('f', 4) - 0.59 * self.get_h('p', 0) + 0.77  # -13.8*Consumption_30
        at[2] = -4.05 * self.get_h('h', 3) - self.get_h('h', 4) + 2.26 * self.get_h('p', 0) + 0.90  # 27
        return at


class Agent_Udluft(Ib_Agent):
    """

    """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = 'Agent Udluft'
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = -self.get_h('f', 0) - self.get_h('v', 3)  # 27
        at[1] = self.get_h('f', 4)
        at[2] = -6 * self.get_h('h', 4)
        return at


class Agent_Test(Ib_Agent):
    """

    'p', 'v', 'g', 'h', 'f', 'c'
    SetPoint_0 Velocity_0 Gain_0 Shift_0 Fatigue_0 RewardTotal_0 Consumption_0
    """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = 'Agent Test'
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        if len(self.state_history) > 10:
            self.state_history.pop()

        at = np.array([0, 0, 0], dtype=np.float32)
        # at[0] = 12.31 * self.get_h('g', 8) - 11.079
        at[0] = ((((-89.824254)*self.get_h('v', 9))*(max(self.get_h('h', 1), (self.get_h('g', 2)+self.get_h('v', 3)))**2))-33.157891)
        at[1] = -2.6 / (self.get_h('c', 30) - self.get_h('f', 12) + self.get_h('f', 17))
        # at[1] = np.sign(max(self.get_h('f', 9), self.get_h('g', 3)))
        at[2] = 7.0442545165 * self.get_h('p', 30) - 9.3079051182 * self.get_h('h', 26) - 7.0442545165 * self.get_h('h', 27) + 3.0
        # at[2] = ((((((6.393297 * self.get_h('c', 5)) - (6.393297 * self.get_h('c', 9))) - (6.393297 * self.get_h('g', 4))) + (6.393297 * self.get_h('g', 9))) + (6.393297 * self.get_h('h', 4))) - (6.393297 * self.get_h('h', 9)))
        return at

# ('IB_udluft_0_0', IB_udluft_0_0(), 0, 7445.456), ('IB_udluft_0_1', IB_udluft_0_1(), 1.0, 6667.792), ('IB_udluft_0_2', IB_udluft_0_2(), 2.0, 6431.543), ('IB_udluft_0_3', IB_udluft_0_3(), 3.0, 6286.144), ('IB_udluft_0_4', IB_udluft_0_4(), 4.0, 6071.008), ('IB_udluft_0_6', IB_udluft_0_6(), 6.0, 5731.761), ('IB_udluft_0_7', IB_udluft_0_7(), 7.0, 5540.986), ('IB_udluft_0_10', IB_udluft_0_10(), 10.0, 5379.627), ('IB_udluft_0_13', IB_udluft_0_13(), 13.0, 5364.688), ('IB_udluft_0_14', IB_udluft_0_14(), 14.0, 5296.562)
# ('IB_udluft_1_0', IB_udluft_1_0(), 0, 10199.376), ('IB_udluft_1_1', IB_udluft_1_1(), 1.0, 4790.752), ('IB_udluft_1_2', IB_udluft_1_2(), 2.0, 4518.592), ('IB_udluft_1_3', IB_udluft_1_3(), 3.0, 4509.147), ('IB_udluft_1_4', IB_udluft_1_4(), 4.0, 4181.852), ('IB_udluft_1_5', IB_udluft_1_5(), 5.0, 4125.886), ('IB_udluft_1_6', IB_udluft_1_6(), 6.0, 3807.634), ('IB_udluft_1_8', IB_udluft_1_8(), 8.0, 3642.444), ('IB_udluft_1_10', IB_udluft_1_10(), 10.0, 3500.061), ('IB_udluft_1_12', IB_udluft_1_12(), 12.0, 3424.337)
# ('IB_udluft_2_0', IB_udluft_2_0(), 0, 9445.287), ('IB_udluft_2_1', IB_udluft_2_1(), 1.0, 5428.476), ('IB_udluft_2_2', IB_udluft_2_2(), 2.0, 5380.151), ('IB_udluft_2_4', IB_udluft_2_4(), 4.0, 4923.883), ('IB_udluft_2_5', IB_udluft_2_5(), 5.0, 4780.519), ('IB_udluft_2_6', IB_udluft_2_6(), 6.0, 4105.182), ('IB_udluft_2_7', IB_udluft_2_7(), 7.0, 3650.729), ('IB_udluft_2_10', IB_udluft_2_10(), 10.0, 3636.047), ('IB_udluft_2_11', IB_udluft_2_11(), 11.0, 3425.701), ('IB_udluft_2_12', IB_udluft_2_12(), 12.0, 3392.918), ('IB_udluft_2_15', IB_udluft_2_15(), 15.0, 3388.44)


def eval_agents():
    T = 100000

    # agents = [Agent_Daniel_Best()]
    agents = [
        # Agent_random(),
        # Agent_nothing(),
        # Agent_daniel_21(),
        # Agent_daniel_27(),

        # Agent_Daniel_29_Best(),
        # Agent_Udluft(),
        Agent_Test()
    ]
    print("Discounted reward sum after 100.000 steps random -9788.323176708134\n"
          "Discounted reward sum after 100.000 steps Daniel_29 -5014.727641016865\n"
          "Discounted reward sum after 100.000 steps Agent Udluft -5337.506606743116\n"
          "Discounted reward sum after 100.000 steps Agent Test -7601.404290299708\n"
          "Starting new ones")
    data = np.zeros((len(agents), T))
    data_cost = np.zeros((len(agents), T))

    # for k in range(n_trajectories):
    for k, agent in enumerate(agents):
        env = IDS()  # p=100 in examples!
        # state_debug = []
        for t in range(T):
            env_state = envstate_normalize(env.state)
            at = agent.decide(env_state)
            # v, g, h = at  # todo
            # v = max(0, min(100, ))
            markovStates = env.step(at)
            data[k, t] = env.visibleState()[-1]

        factor = 0.97
        sum = 0
        time_horizon = 100
        for x in range(time_horizon):
            entry = data[k][-1 - x]
            sum += factor ** x * entry

        print('Discounted reward sum after 100.000 steps', agent.name, sum)

    # plt.plot(data.T)
    # plt.xlabel('T')
    # plt.ylabel('Reward')
    # plt.show()


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
                    csv_data[t + ii][enum_x * history_t + ii] = state_x

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
    ('06_50', '2',)
]
eval_agents()
