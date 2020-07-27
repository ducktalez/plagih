# coding=utf-8
import numpy as np
import collections


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


class Ib_Agent:

    def __init__(self):
        self.state_history = collections.deque()

    def get_h(self, name, steps):

        def get_max_history(index, history_list):
            if index > len(history_list) - 1:
                # print('Not enough history for index {}, len list: {}'.format(index, len(history_list)))
                return history_list[-1]
            else:
                return history_list[index]
        return get_max_history(steps, self.state_history)[name]


# class Ib_Agent_Single:
#
#     def __init__(self):
#         self.state_history = collections.deque()
#
#     def get_h(self, name, steps):
#
#         def get_max_history(index, history_list):
#             if index > len(history_list) - 1:
#                 # print('Not enough history for index {}, len list: {}'.format(index, len(history_list)))
#                 try:
#                     return history_list[-1]
#                 except:
#                     return history_list[-1]
#             else:
#                 return history_list[index]
#         return get_max_history(steps, self.state_history)[name]


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
        at[2] = -3.48 * self.get_h('h', 3) - self.get_h('h', 4) + 2*self.get_h('p', 0) + 0.81
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


class Agent_sim1(Ib_Agent):
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

        SetPoint = self.get_h('p', 0)

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = np.sign(min(self.get_h('g', 7), ((self.get_h('f', 0)*max(SetPoint, self.get_h('h', 2)))+self.get_h('h', 3))))
        at[1] = np.sign(((((self.get_h('f', 9)+self.get_h('g', 4))-self.get_h('g', 9))-SetPoint)+1.387298))
        at[2] = -2 / self.get_h('h', 7)
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

        SetPoint = self.get_h('p', 0)

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = np.sign(min(min(self.get_h('g', 7), (-self.get_h('v', 9)-0.659044)), self.get_h('h', 3)))
        at[1] = np.sign((((self.get_h('g', 4)-self.get_h('g', 9))-SetPoint)+1.149521))
        at[2] = -2 / self.get_h('h', 7)
        return at