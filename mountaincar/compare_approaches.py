import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
from mountaincar.agents.sarsa_best_approach import sarsa_start_training
from mountaincar.agents.move_directions import move_towards_direction
from mountaincar.agents.plagih_1 import improved_v1


def mtc_compare_approaches(approach1, approach2):
    env = gym.make('MountainCar-v0')
    env.seed(1)
    # env = gym.wrappers.Monitor(env, "./records", video_callable=lambda _: True)

    reward_sample_interval = 10
    episodes = 1000
    episodes_ez = episodes
    episodes_sarsa = episodes

    episode_rewards1, rewards1_avg = move_towards_direction(env, episodes_ez, reward_sample_interval)
    env.close()
    env = gym.make('MountainCar-v0')
    env.seed(1)
    # episode_rewards2, rewards2_avg = sarsa_start_training(env, episodes_sarsa, reward_sample_interval, train=True, render=False)
    episode_rewards2, rewards2_avg = improved_v1(env, episodes_sarsa, reward_sample_interval)
    env.close()

    # rewards = QLearning(env, 0.2, 0.9, 0.8, 0.001, episodes_q, 1000)

    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), episode_rewards1, label=str(approach1))
    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), episode_rewards2, label=str(approach2))
    plt.legend()
    # plt.yscale('linear')
    plt.xlabel('Episodes')
    plt.ylabel('Sampled Reward')
    plt.title('Sampled Reward vs Episodes')
    plt.ylim(-200, -50)
    plt.savefig('MTC-'+str(approach1)+'-vs-'+str(approach2)+'-'+str(episodes)+'-'+str(reward_sample_interval)+'.jpg')
    plt.show()

    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), rewards1_avg, label=str(approach1))
    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), rewards2_avg, label=str(approach2));
    plt.legend()
    # plt.yscale('linear')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.ylim(-200, -50)
    plt.savefig('MTC-'+str(approach1)+'-vs-'+str(approach2)+'-'+str(episodes)+'-'+str(reward_sample_interval)+'-average.jpg')
    plt.show()

    plt.close()


mtc_compare_approaches('simple', 'simple2')

# agent.epsilon = 0.
# episodes = 100
# episode_rewards = [play_sarsa(env, agent, train=False, render=False) for _ in range(episodes)]
# print('average award = {} / {} = {}'.format(
#         sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
















