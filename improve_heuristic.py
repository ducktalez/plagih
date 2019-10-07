import numpy as np
import gym
import matplotlib.pyplot as plt
from mountaincar.agents.sarsa_best_approach import sarsa_start_training
from mountaincar.agents.move_directions import move_towards_direction

# plagih


# This function returns a key-value pair list to test the fitness on.
# - We use a pre-trained model from an agent
# - The agent has to provide a function called get_action(state), which gets a state and returns an action
# Several options may be helpful:
# sampleType="run_experiments": Runs random experiments from start to end, until sample_size
# sampleType="arbitrary_samples": Arbitrarily chooses samples
# Nice-to-have:
# - Non markovian / LSTM possibility?
# - sample size based on the number experiments? Like, "just make 10 experiments" and use as maximum then.

def get_fitness_samples(agent, sample_size=1000, sample_type="run_experiments"):
    return




env = gym.make('MountainCar-v0')
env.seed(1)
# env = gym.wrappers.Monitor(env, "./records", video_callable=lambda _: True)

env.close()



