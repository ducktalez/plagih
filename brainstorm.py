import torch
import torch.nn as nn
import torch.nn.functional as functional
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        super().__init__()

    def append_sample(self, *args):
        pass

    def policy(self, *args):
        pass

    def minimize_epsilon(self, *args):
        pass

    def train(self, *args):
        pass


class AgentDQN(Agent):
    def __init__(self):
        super(Agent).__init__()
        pass

    def append_sample(self, state, action, reward, next_state, policy):
        pass

    def policy(self, action):
        pass

    def minimize_epsilon(self):
        pass

    def train(self):
        pass