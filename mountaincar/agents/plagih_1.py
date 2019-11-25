import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
# capsulated function "choose action" which gets a state and then chooses an action


def choose_action(state):
    observation1 = state[1]
    if observation1 < 1.249707159116757 * min(min(observation1, 0.5221824578829628), -0.0238622135678297) - 0.18682569636205436 + 1.249707159116757 * min(observation1 + min(observation1, 0.4814895922935718) + 0.49481152602372247, -0.7901782162226243) / observation1:
        return 0
    else:
        return 2


    Ifte(observation1 < b* b / observation1, 0, 2)

def improved_v1(env, episodes, reward_interval):

    reward_list, reward_list_avg = [], []

    for i in range(episodes):
        state = env.reset()
        done = False
        reward_tot = 0
        reward_list_preavg = []

        while(done is False):
            action = choose_action(state)
            state, reward, done, info = env.step(action)
            reward_tot += reward
        reward_list_preavg.append(reward_tot)

        if (i  % reward_interval == 0):

            reward_list.append(reward_tot)
            reward_list_avg.append(np.mean(reward_list[-reward_interval:]))
            print('EZ-Episode {} Average Reward: {}'.format(i + 1, reward_tot))

    return reward_list, reward_list_avg


