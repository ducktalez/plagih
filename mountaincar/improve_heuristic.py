import numpy as np
import gym
import matplotlib.pyplot as plt
from mountaincar.agents.sarsa_best_approach import sarsa_start_training
from mountaincar.agents.move_directions import move_towards_direction
from pathlib import Path, PurePath
import pickle


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


# tree = Path('../samples/behaviour_samples.p')
# PLAGIH_display_one_tree_from_csv()

samples_file = Path('karoo_files/behaviour_samples.p')
with open(samples_file, "rb") as fp:
    plagih_behaviour_samples = pickle.load(fp)
test = plagih_behaviour_samples[:10]
# test = np.transpose(test)
# if test[0][0] == 'numpy.ndarray':
try:
    print(len(test[0][0]))
except:
    print('1')
try:
    print(len(test[0][1]))
except:
    print('1')

