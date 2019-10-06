import numpy as np
import gym
import matplotlib.pyplot as plt
from mountaincar.agents.sarsa_best_approach import sarsa_best_start
from mountaincar.agents.move_directions import move_towards_direction

env = gym.make('MountainCar-v0')
env.seed(1)
# env = gym.wrappers.Monitor(env, "./records", video_callable=lambda _: True)


rewardSample_interval = 10
episodes = 1000
episodes_ez = episodes
episodes_sarsa = episodes

episode_rewards2, rewards2_avg = sarsa_best_start(env, episodes_sarsa,rewardSample_interval, train=True, render=False)
episode_rewards1, rewards1_avg = move_towards_direction(env, episodes_ez,rewardSample_interval)


# rewards = QLearning(env, 0.2, 0.9, 0.8, 0.001, episodes_q, 1000)
# # TODO plot nums auslagern

plt.plot(rewardSample_interval * (np.arange(len(episode_rewards1)) + 1), episode_rewards1, label="Heuristic")
plt.plot(rewardSample_interval * (np.arange(len(episode_rewards1)) + 1), episode_rewards2, label="Improved SARSA");
plt.legend()
# plt.yscale('linear')
plt.xlabel('Episodes')
plt.ylabel('Sampled Reward')
plt.title('Sampled Reward vs Episodes')
plt.ylim(-200, -50)
plt.savefig('MTC-heuristic_vs_sarsa-'+str(episodes)+'-'+str(rewardSample_interval)+'.jpg')
# plt.show()
# plt.close()
plt.show()

plt.plot(rewardSample_interval * (np.arange(len(episode_rewards1)) + 1), rewards1_avg, label="Heuristic")
plt.plot(rewardSample_interval * (np.arange(len(episode_rewards1)) + 1), rewards2_avg, label="Improved SARSA");
plt.legend()
# plt.yscale('linear')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.ylim(-200, -50)
plt.savefig('MTC-heuristic_vs_sarsa-'+str(episodes)+'-'+str(rewardSample_interval)+'-average.jpg')
plt.show()

plt.close()

# agent.epsilon = 0.
# episodes = 100
# episode_rewards = [play_sarsa(env, agent, train=False, render=False) for _ in range(episodes)]
# print('average award = {} / {} = {}'.format(
#         sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))











# Creates the plot which displays current position/speed and the corresponding action

# poses = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
# vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
# positions, velocities = np.meshgrid(poses, vels)
#
# @np.vectorize
# def decide(position, velocity):
#     return agent.decide((position, velocity))
#
# action_values = decide(positions, velocities)
#
# fig, ax = plt.subplots()
# c = ax.pcolormesh(positions, velocities, action_values)
# ax.set_xlabel('position')
# ax.set_ylabel('velocity')
# fig.colorbar(c, ax=ax, boundaries=[-.5, .5, 1.5, 2.5], ticks=[0, 1, 2])
# fig.show();


# with open('./resources/agent.pkl', 'wb') as file:
#     pickle.dump(agent, file)

env.close()



