import csv

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

# pickle-version does make too much trouble for now... need to switch to .csv


samples_file = Path('mountaincar/karoo_files/behaviour_samples.csv')
# TODO does this work with all types of data?
with open(samples_file) as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    num_observations = 0
    data_x, data_y = [], []
    for i, row in enumerate(reader):
        if i == 0:
            terminals = row
            for var_name in row:
                if var_name.startswith('o'):
                    num_observations += 1
                elif var_name.startswith('a'):
                    pass
                else:
                    raise print('Behaviour samples first line: Variables have to start with "o" or "a" to be recognized')
        else:
            data_x.append(row[:num_observations])
            data_y.append(row[num_observations:])

print(terminals)