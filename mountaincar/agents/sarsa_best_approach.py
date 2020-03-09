import csv

import numpy as np

from mountaincar.agents.agents import *

np.random.seed(0)
import gym
import sys
from pathlib import Path, PurePath

import pickle


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

    if save is True:
        with open('sarsa_agent.p', 'wb') as file:
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


def train_sarsa_agent_pickle(seed=0):
    env = gym.make('MountainCar-v0')
    env.seed(seed)

    agent = SARSALambdaAgent(env)
    # steps_done = 0
    # for training_steps in [75, 200, 1000, 10000]:
    #     episode_rewards = []
    #     steps_left = training_steps - steps_done
    #     for episode in range(steps_left):
    #         episode_reward = play_sarsa(env, agent, train=True)
    #         episode_rewards.append(episode_reward)
    #
    #     with open(Path('pickle/sarsa_agent_{}.p'.format(str(training_steps))), 'wb') as file:
    #         pickle.dump(agent, file)

    episode_rewards = []
    for episode in range(75):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)
    with open(sarsa_file_75, 'wb') as file:
        pickle.dump(agent, file)
    print('...saved agent pickle')

    for episode in range(125):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)
    with open(sarsa_file_200, 'wb') as file:
        pickle.dump(agent, file)
    print('...saved agent pickle')

    for episode in range(800):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)
    with open(sarsa_file_1000, 'wb') as file:
        pickle.dump(agent, file)
    print('...saved agent pickle')

    for episode in range(9000):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)
    with open(sarsa_file_10000, 'wb') as file:
        pickle.dump(agent, file)
    print('...saved agent pickle')

    return


def create_samples_dataset(seed=0):
    env = gym.make('MountainCar-v0')
    env.seed(seed)
    with Path.open(sarsa_file_75, 'rb') as file:
        agent = pickle.load(file)

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
    train_sarsa_agent_pickle()
