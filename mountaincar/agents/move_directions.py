import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
# capsulated function "choose action" which gets a state and then chooses an action
def choose_action(state):
    if \
            state[1] < 0: #PLAGI
        return 0
    else:
        return 2

def move_towards_direction(env, episodes, reward_interval):

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


