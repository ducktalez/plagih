import csv

from benchmarks.mc.agents.agent_groups import *

np.random.seed(0)
import pickle
import sys
from pathlib import Path

import gym


def mountaincar_play_once(env, agent, train=False, render=False):
    """ """
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

    return episode_reward


def sarsa_start_frommodel(env, episodes, rewardSample_interval, train=False, render=False):
    agent = SARSALambdaAgent(env)
    episode_rewards, episode_rewards_preaverage, episode_rewards_average = [], [], []
    for episode in range(episodes):
        episode_reward = mountaincar_play_once(env, agent, train, render)
        episode_rewards_preaverage.append(episode_reward)  #
        if episode % rewardSample_interval == 0:
            episode_rewards.append(episode_reward)
            episode_rewards_average.append(np.mean(episode_rewards_preaverage[-rewardSample_interval:]))

            print("SARSA: " + str(episode_reward))
    return episode_rewards, episode_rewards_average


def plagih_get_behaviour_samples(env, agent, episodes):
    """
    Create obs-act-list: [[0.12, 1.55, 2], ...]
    - epsilon = 0 (no exploration)
    """
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
                print("Adding samples which create this reward: " + str(episode_reward))
                break
            action_next = agent.decide(observation_next)
            observation, action = observation_next, action_next

    return plagih_state_actions


def train_sarsa_agent_pickle(seed=0):
    """
    train and save agents with n steps for mountain car (75, 200, 1000, 10000)
    """
    env = gym.make("MountainCar-v0")
    env.seed(seed)
    agent = SARSALambdaAgent(env)

    for x in range(75):
        print(f"episode {x}")
        mountaincar_play_once(env, agent, train=True)
    with Path(f"sarsa_agent_75.p").open("wb") as file:
        pickle.dump(agent, file)
        print(f"done with sarsa_agent_75.p")

    for x in range(200 - 75):
        print(f"episode {x}")
        mountaincar_play_once(env, agent, train=True)
    with Path(f"sarsa_agent_200.p").open("wb") as file:
        pickle.dump(agent, file)
        print(f"done with sarsa_agent_200.p")

    for x in range(1000 - 200):
        print(f"episode {x}")
        mountaincar_play_once(env, agent, train=True)
    with Path(f"sarsa_agent_1000.p").open("wb") as file:
        pickle.dump(agent, file)
        print(f"done with sarsa_agent_1000.p")

    for x in range(10000 - 1000):
        print(f"episode {x}")
        mountaincar_play_once(env, agent, train=True)
    with Path(f"sarsa_agent_10000.p").open("wb") as file:
        pickle.dump(agent, file)
        print(f"done with sarsa_agent_10000.p")

    # episode_rewards = []
    # for episode in range(75):
    #     episode_reward = mountaincar_play_once(env, agent, train=True)
    #     episode_rewards.append(episode_reward)
    # print('...saved agent pickle')
    #
    # with open('sarsa_75.p', 'wb') as file:
    #     pickle.dump(agent, file)

    return


def create_plagih_samples_csv(agent, name, n=50):
    """
    Creates .csv data files for mountaincar runs
    """
    env = gym.make("MountainCar-v0")
    env.seed(0)

    plagih_behaviour_samples = plagih_get_behaviour_samples(env, agent, episodes=n)  # ((cartPos, cartVel), act)
    print(
        "Amount of samples: {}, {} bytes".format(len(plagih_behaviour_samples), sys.getsizeof(plagih_behaviour_samples))
    )
    env.close()

    # pickle-version does make too much trouble for now... need to switch to .csv

    samples_csv_ready = [
        ["cartPos|type=float|role=input", "cartVel|type=float|role=input", "action|type=int|role=output"]
    ]

    for sample_row in plagih_behaviour_samples:
        row = list(sample_row[0]) + [sample_row[1]]
        samples_csv_ready.append(row[:])

    with Path(f"../gp_files/samples_{name}.csv").open("w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(samples_csv_ready)

    return


if __name__ == "__main__":
    train_sarsa_agent_pickle()
    # # create_behaviour_samples_file()
    # # do_this = input('Please press: create (s)amples, run coming later: ')
    # # if do_this == 's':
    sarsa_agent_75, sarsa_agent_200, _, _ = load_sarsas(set_epsilon=0.001)
    create_plagih_samples_csv(sarsa_agent_75, "75")
    create_plagih_samples_csv(sarsa_agent_200, "200")
    # sarsa_agent_75, sarsa_agent_200, _, _ = load_sarsas(set_epsilon=0)
    # create_plagih_samples_csv(sarsa_agent_75, '75_0', n=100)
    # create_plagih_samples_csv(sarsa_agent_200, '200_0', n=100)
