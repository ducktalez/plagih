import numpy as np


sarsa_file_75 = 'sarsa_agent_75.p'
sarsa_file_200 = 'sarsa_agent_200.p'
sarsa_file_1000 = 'sarsa_agent_1000.p'
sarsa_file_10000 = 'sarsa_agent_10000.p'


class PlagihAgent_A:

    def decide(self, observation):
        observation0, observation1 = observation
        if -observation1 + min(observation1, observation0 + 1.025) > observation1:
            return 0
        else:
            return 2


class PlagihAgent_A:

    def decide(self, observation):
        observation0, observation1 = observation
        if -observation1 + min(observation1, observation0 + 1.025) > observation1:
            return 0
        else:
            return 2


class SimpleAgent:

    def decide(self, observation):
        observation0, observation1 = observation
        if observation1 < 0:
            return 0
        else:
            return 2


class FixAgent:

    def decide(self, observation):
        pos, velocity = observation
        lb = min(-0.09 * (pos + 0.25) ** 2 + 0.03,
                 0.3 * (pos + 0.9) ** 4 - 0.008)
        ub = -0.07 * (pos + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2  # push right
        else:
            action = 0  # push left
        return action


class TestFixNoLowerbound:
    """
    I randomly found out, that the upper bound is not good for anything
    """
    def decide(self, observation):
        pos, velocity = observation
        lb = min(-0.09 * (pos + 0.25)**2 + 0.03,
                 0.3*(pos + 0.9)**4 - 0.008)

        if lb < velocity:
            return 2
        else:
            return 0


class TestCombined:
    """
    I found this candidate within 1 minute of gp
    """
    def decide(self, observation):
        pos, velocity = observation
        if (velocity <= 0.63) and (min(-0.09*(pos + 0.25)**2.0 + 0.03, 0.3*(pos + 0.9)**4.0 - 0.01) <= velocity):
            return 2
        else:
            return 0


class TestTmp:
    """
    random test
    """

    def decide(self, observation):
        pos, velocity = observation
        if (velocity <= 0.63) and (min(-0.09 * (pos + 0.25) ** 2.0 + 0.03, 0.3 * (pos + 0.9) ** 4.0 - 0.01) <= velocity):
            return 2
        else:
            return 0


class TileCoder:
    def __init__(self, layers, features):
        """
        Parameters
        - layers: int, the number of layers in tile coding
        - features: int, the number of features, also the shape of weights
        """
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:  # collide when codebook is full
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        """
        Parameters
        - floats: tuple of floats, each of which is within [0., 1.]
        - ints: tuple of ints
        Returns
        - features : list of ints
        """
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) / self.layers) \
                                        for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features


class SARSAAgent:
    def __init__(self, env, layers=8, features=2000, gamma=1.,
                 learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low
        self.encoder = TileCoder(layers, features)
        self.w = np.zeros(features)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def encode(self, observation, action):
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        features = self.encode(observation, action)
        return self.w[features].sum()

    def decide(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, observation_next, done, action_next=None):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(observation_next, action_next))
        delta = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * delta)


class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, layers=8, features=2000, gamma=1.,
                 learning_rate=0.03, epsilon=0.001, lambd=0.9):
        super().__init__(env=env, layers=layers, features=features,
                         gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.z = np.zeros(features)

    def learn(self, observation, action, reward, observation_next, done, action_next=None):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(observation_next, action_next))
            self.z *= (self.gamma * self.lambd)
            features = self.encode(observation, action)
            self.z[features] = 1.  # replacement trace
        delta = u - self.get_q(observation, action)
        self.w += (self.learning_rate * delta * self.z)
        if done:
            self.z = np.zeros_like(self.z)