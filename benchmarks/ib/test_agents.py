# coding=utf-8
import collections

import numpy as np


class Ib_Agent:
    def __init__(self):
        self.state_history = collections.deque(maxlen=30)

    def only_save_history(self, env_state):
        """
        When random start, use this function to save the last states
        """
        self.state_history.appendleft(env_state)

    def get_h(self, name, steps):
        def get_max_history(index, history_list):
            if index >= len(history_list):
                return history_list[-1]  # return last value
            else:
                return history_list[index]

        return get_max_history(steps, self.state_history)[name]


class Agent_nothing(Ib_Agent):
    """
    ...does nothing, kind of.
    """

    def __init__(self):
        # self.state_history = collections.deque()
        self.name = "nothing"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        #        if len(self.state_history) > 10:
        #            self.state_history.pop_list()

        at = np.array([0, 0, 0])
        at[0] = 0
        at[1] = 0
        at[2] = 0
        return at


class Agent_random(Ib_Agent):
    """
    ...does nothing, kind of.
    """

    def __init__(self):
        # self.state_history = collections.deque()
        self.name = "randomly"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)
        at = 2 * np.random.rand(3) - 1
        return at


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
        self.name = "Hein_21"
        super().__init__()

    def decide(self, env_state):
        self.state_history.appendleft(env_state)

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = -self.get_h("v", 5) - 0.91
        at[1] = 2 * self.get_h("f", 3) - env_state["p"] + 1.43
        at[2] = -3.48 * self.get_h("h", 3) - self.get_h("h", 4) + 2 * self.get_h("p", 0) + 0.81
        return at


class Agent_daniel_27(Ib_Agent):
    def __init__(self):
        self.name = "Hein_27"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        at = np.array([0, 0, 0], dtype=np.float32)
        # # idk why, but these are the best??
        # at[0] = -self.get_h('v', 5) - 1.17
        # at[1] = 2 * self.get_h('f', 3) - self.get_h('p', 0) + 1.16
        # at[2] = -3.49 * self.get_h('h', 3) - self.get_h('h', 4) + 2 * self.get_h('p', 0) + 0.82
        at[0] = -self.get_h("v", 4) - 0.96
        at[1] = 0.4 * self.get_h("f", 1) + self.get_h("f", 4) - 0.6 * self.get_h("p", 0) + 0.76
        at[2] = -self.get_h("h", 3) - 1.57 * self.get_h("h", 4) + self.get_h("p", 0) + 0.52
        return at


class Agent_Daniel_29_Best(Ib_Agent):
    def __init__(self):
        self.name = "Hein_29"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = -self.get_h("v", 4) - 0.96  # 27
        at[1] = 0.41 * self.get_h("f", 1) + self.get_h("f", 4) - 0.59 * self.get_h("p", 0) + 0.77  # 27
        at[2] = -4.05 * self.get_h("h", 3) - self.get_h("h", 4) + 2.26 * self.get_h("p", 0) + 0.90  # 27

        return at


class Agent_Test(Ib_Agent):
    """
    ...does nothing, kind of.
    """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = "Testagent"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        #        if len(self.state_history) > 10:
        #            self.state_history.pop_list()

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = 0
        at[1] = 0
        at[2] = 0
        return at


class Agent_Udluft(Ib_Agent):
    """ """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = "Agent Udluft"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = -self.get_h("f", 0) - self.get_h("v", 3)  # 27
        at[1] = self.get_h("f", 4)
        at[2] = -6 * self.get_h("h", 4)
        return at


class Agent_sim1(Ib_Agent):
    """

    'p', 'v', 'g', 'h', 'f', 'c'
    SetPoint_0 Velocity_0 Gain_0 Shift_0 Fatigue_0 RewardTotal_0 Consumption_0
    """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = "Agent aim1"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        #        if len(self.state_history) > 10:
        #            self.state_history.pop_list()

        SetPoint = self.get_h("p", 0)

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = np.sign(
            min(self.get_h("g", 7), ((self.get_h("f", 0) * max(SetPoint, self.get_h("h", 2))) + self.get_h("h", 3)))
        )
        at[1] = np.sign(((((self.get_h("f", 9) + self.get_h("g", 4)) - self.get_h("g", 9)) - SetPoint) + 1.387298))
        at[2] = -2 / self.get_h("h", 7)
        return at


class Agent_505050(Ib_Agent):
    """
    'p', 'v', 'g', 'h', 'f', 'c'
    SetPoint_0 Velocity_0 Gain_0 Shift_0 Fatigue_0 RewardTotal_0 Consumption_0
    Results : randomly 	-6809.4 	 (safe: -6632.6)
    Results : nothing 	-6077.8 	 (safe: -6068.8)
    Results : Daniel_21 	-5278.1 	 (safe: -5573.3)
    Results : Daniel_27 	-5268.1 	 (safe: -5548.0)
    Results : Daniel_29 	-5232.4 	 (safe: -5542.0)
    Results : 505050 	-8273.8 	 (safe: -8321.5)
    Results : Agent Udluft 	-5905.2 	 (safe: -6193.2)
    """

    def __init__(self):
        self.state_history = collections.deque()
        self.name = "505050"
        super().__init__()

    def decide(self, state):
        self.state_history.appendleft(state)

        SetPoint = self.get_h("p", 0)

        at = np.array([0, 0, 0], dtype=np.float32)
        at[0] = 1.25 * (self.get_h("v", 0) * -12.31)
        at[1] = -0.53 * (self.get_h("g", 0) * -29.91)
        at[2] = 0.55 * (self.get_h("h", 0) * -29.22)
        return at
