import numpy as np
import gym
import matplotlib.pyplot as plt


def plot_approach(episodes, reward_sample_interval, episode_rewards1, rewards1_avg, approach):

    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), episode_rewards1, label=str(approach))
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Sampled Reward')
    plt.title('Sampled Reward vs Episodes')
    plt.ylim(-200, -50)
    plt.savefig('MTC-' + str(approach) + '-' + str(episodes) + '-' + str(reward_sample_interval) + '.jpg')
    plt.show()

    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), rewards1_avg, label=str(approach))
    plt.legend()
    # plt.yscale('linear')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.ylim(-200, -50)
    plt.savefig('MTC-' + str(approach) + '-' + str(episodes) + '-' + str(reward_sample_interval) + '-average.jpg')
    plt.show()

    plt.close()

    return


def mtc_approach_start(approach):
    env = gym.make('MountainCar-v0')
    env.seed(1)
    # env = gym.wrappers.Monitor(env, "./records", video_callable=lambda _: True)

    reward_sample_interval = 10
    episodes = 1000
    episodes_ez = episodes

    # episode_rewards2, rewards2_avg = sarsa_start_training(env, episodes_sarsa, rewardSample_interval, train=True, render=False)

    episode_rewards1, rewards1_avg = improved_v1(env, episodes_ez, reward_sample_interval)

    env.close()

    plot_approach(episodes, reward_sample_interval, episode_rewards1, rewards1_avg, approach)


mtc_approach_start('simple')
