import gym
import matplotlib.pyplot as plt
import numpy as np


def plot_approach(episodes, reward_sample_interval, episode_rewards1, rewards1_avg, approach):
    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), episode_rewards1, label=str(approach))
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Sampled Reward")
    plt.title("Sampled Reward vs Episodes")
    plt.ylim(-200, -50)
    plt.savefig(f"MTC-{approach}-{episodes}-{reward_sample_interval}.pdf")

    plt.plot(reward_sample_interval * (np.arange(len(episode_rewards1)) + 1), rewards1_avg, label=str(approach))
    plt.legend()
    # plt.yscale('linear')
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Episodes")
    plt.ylim(-200, -50)
    plt.savefig(f"MTC-{approach}-{episodes}-{reward_sample_interval}-average.pdf")

    plt.close("all")
    return


def mtc_approach_start(approach):
    env = gym.make("MountainCar-v0")
    env.seed(1)
    # env = gym.wrappers.Monitor(env, "./records", video_callable=lambda _: True)

    reward_sample_interval = 10
    episodes = 1000

    # episode_rewards2, rewards2_avg = sarsa_start_training(env, episodes_sarsa, rewardSample_interval, train=True, render=False)

    episode_rewards1, rewards1_avg = plot_approach(episodes, reward_sample_interval)

    env.close()

    plot_approach(episodes, reward_sample_interval, episode_rewards1, rewards1_avg, approach)


mtc_approach_start("simple")
