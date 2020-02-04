import csv

import numpy as np

np.random.seed(0)
import gym
import matplotlib.pyplot as plt
import sys
from pathlib import Path, PurePath

import pickle


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


def play_sarsa(env, agent, train=False, render=False, save=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        observation_next, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            if train:
                agent.learn(observation, action, reward, observation_next, done)
            break
        action_next = agent.decide(observation_next)
        if train:
            agent.learn(observation, action, reward, observation_next, done, action_next)
        observation, action = observation_next, action_next

    if save == True:
        with open('./resources/agent.pkl', 'wb') as file:
            pickle.dump(agent, file)

    return episode_reward


def sarsa_start_training(env, episodes, rewardSample_interval, train=False, render=False):
    agent = SARSALambdaAgent(env)
    episode_rewards, episode_rewards_preaverage, episode_rewards_average = [], [], []
    for episode in range(episodes):
        episode_reward = play_sarsa(env, agent, train, render)
        episode_rewards_preaverage.append(episode_reward)
        if episode % rewardSample_interval == 0:
            episode_rewards.append(episode_reward)
            episode_rewards_average.append(np.mean(episode_rewards_preaverage[-rewardSample_interval:]))

            print("SARSA: " + str(episode_reward))
    return episode_rewards, episode_rewards_average


def sarsa_start_frommodel(env, episodes, rewardSample_interval, train=False, render=False):
    agent = SARSALambdaAgent(env)
    episode_rewards, episode_rewards_preaverage, episode_rewards_average = [], [], []
    for episode in range(episodes):
        episode_reward = play_sarsa(env, agent, train, render)
        episode_rewards_preaverage.append(episode_reward)  #
        if episode % rewardSample_interval == 0:
            episode_rewards.append(episode_reward)
            episode_rewards_average.append(np.mean(episode_rewards_preaverage[-rewardSample_interval:]))

            print("SARSA: " + str(episode_reward))
    return episode_rewards, episode_rewards_average


def plagih_get_behaviour_samples(env, agent, episodes, train=False, render=False):
    agent.epsilon = 0
    plagih_state_actions = []

    for episode in range(episodes):

        episode_reward = 0
        observation = env.reset()
        while True:
            action = agent.decide(observation)
            plagih_state_actions.append([observation, action])  # plagih
            observation_next, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                print('Adding samples which create this reward: ' + str(episode_reward))
                break
            action_next = agent.decide(observation_next)
            observation, action = observation_next, action_next

    return plagih_state_actions


def create_behaviour_samples_file(seed=0):
    env = gym.make('MountainCar-v0')
    env.seed(seed)
    agent = SARSALambdaAgent(env)

    # perform training
    episodes_training = 75
    episode_rewards = []
    for episode in range(episodes_training):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)

    # plt.plot(episode_rewards);
    # plt.show()

    print("Working on trained model now, no training, epsilon=0")

    plagih_behaviour_samples = plagih_get_behaviour_samples(env, agent, episodes=20, train=False)
    print(sys.getsizeof(plagih_behaviour_samples))
    print('Amount of samples: ' + str(len(plagih_behaviour_samples)))
    env.close()

    # samples_file = Path('../karoo_files/behaviour_samples.p')
    # with open(samples_file, "wb") as fp:  # Pickling
    #     pickle.dump(plagih_behaviour_samples, fp)

    # pickle-version does make too much trouble for now... need to switch to .csv
    samples_csv_ready = [['observation0:' + 'float', 'observation1:' + 'float', 'action0:' + 'int']]
    for sample in plagih_behaviour_samples:
        row = []
        row.append(sample[0][0])
        row.append(sample[0][1])
        row.append(sample[1])
        samples_csv_ready.append(row[:])
    print(samples_csv_ready)

    file_csv = Path('../karoo_files/behaviour_samples.csv')
    with open(file_csv, 'w+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(samples_csv_ready)
    csvFile.close()

    return


if __name__ == "__main__":
    # create_behaviour_samples_file()
    # dothis = input('Please press: create (s)amples, run coming later: ')
    # if dothis == 's':
    #     create_behaviour_samples_file()
    sarsa_start_frommodel()
