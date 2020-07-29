# coding=utf-8
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import importlib.util
import itertools
from benchmarks.ib.test_agents import *
import yaml


def combined_lists(run_name):
    lsactions = ['_0/pycode_list.yaml', '_1/pycode_list.yaml', '_2/pycode_list.yaml']
    agents = []

    combipath = Path.cwd() / f'../slurm_runs/{run_name}'
    if not Path.is_dir(combipath):
        Path.mkdir(combipath)
    lut_file = combipath / 'lutfile.yaml'

    for lsaction in lsactions:
        lsfile = Path.cwd() / f'../slurm_runs/{run_name}{lsaction}'
        print(f'Looging at file: {lsfile}')
        with Path.open(lsfile, 'r') as file:
            yamload = yaml.load(file, Loader=yaml.FullLoader)
            agents.append(yamload)
        # todo normalize here aswell

    merged = list(itertools.product(agents[0], agents[1], agents[2]))
    combined_all = []

    for row in merged:
        parsim_sum = float(sum([x[0] for x in row]))
        fitness_sum = float(sum([x[1] for x in row]))
        parsims = [float(row[0][0]), float(row[1][0]), float(row[2][0])]
        codes = [x[3] for x in row]
        combined_all.append([parsim_sum, fitness_sum, parsims, codes])

    combined_best_dict = {}
    combined_best_msed_dict = {}
    for parsim_sum, fitness_sum, parsims, codes in combined_all:
        if parsim_sum in combined_best_dict:
            if fitness_sum < combined_best_dict[parsim_sum][0]:
                combined_best_dict[parsim_sum] = [fitness_sum, parsims, codes]
        else:
            combined_best_dict[parsim_sum] = [fitness_sum, parsims, codes]

        msed = math.sqrt(parsims[0] ** 2 + parsims[1] ** 2 + parsims[2] ** 2)
        if msed in combined_best_msed_dict:
            if fitness_sum < combined_best_msed_dict[msed][0]:
                combined_best_msed_dict[msed] = [fitness_sum, parsims, codes]
        else:
            combined_best_msed_dict[msed] = [fitness_sum, parsims, codes]

    combined_best = [[parsim_sum, fitness_sum, parsims, codes] for parsim_sum, (fitness_sum, parsims, codes) in combined_best_dict.items()]
    combined_mse_best = [[parsim_sum, fitness_sum, parsims, codes] for parsim_sum, (fitness_sum, parsims, codes) in combined_best_msed_dict.items()]

    analyse_results = []

    try:
        # with Path.open(lut_file, 'rb') as file:
        with Path.open(lut_file, 'r') as file:
            lut = yaml.load(file, Loader=yaml.FullLoader)
    except:
        lut = {}
    eval_count = 0
    not_evaled = 0

    print(f'Number of best combinations: {len(combined_best)}, total combinations: {len(combined_all)}')

    for parsim_sum, fitness_sum, parsims, codes in combined_all:  # todo todotodo combined_all or combined_best?

        lut_hash = f'{parsim_sum}_{parsims}_{fitness_sum}'

        if parsim_sum < 35:
            if lut_hash in lut:
                experiment, experiment_save = lut[lut_hash]
            else:
                experiment = float(eval_combined_agents(parsim_sum, parsims, codes))
                experiment_save = float(eval_combined_agents(parsim_sum, parsims, codes, complete=False))
                print(f'Combined parsimony {parsim_sum:3.0f} regression-error: {fitness_sum:.4f}. \t({parsims})\tcomplete: {experiment} \tsave: {experiment_save}')
                lut[lut_hash] = [experiment, experiment_save]
                eval_count += 1
                if eval_count % 20 == 0:
                    print(f'Saving lut after evaluating 20')
                    with Path.open(lut_file, 'w') as file:
                        _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)

            analyse_results.append([parsim_sum, experiment, experiment_save])
        else:
            not_evaled += 1

    print(f'Did not evaluate {not_evaled} combinations as they are too complex...')

    with Path.open(lut_file, 'w') as file:
        _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)

    analyse_results.sort(key=lambda x: x[0])
    nar = np.array(list(zip(*analyse_results)))

    fig, ax = plt.subplots()
    # ax.set_yscale(yscale)
    # ax.set_ylim(min(bottom, 0), new_top)
    # ax.set_xlim(min(left, 0), new_right)
    fig.tight_layout()
    ax.set_xlabel('Complexity')
    ax.set_ylabel('reward')
    ax.set_title('IB evaluation of GP-agents')
    # # only if one entry per parsimony
    # ax.plot(nar[0], nar[1], label='all actions', marker='.')
    # ax.plot(nar[0], nar[2], label='low risk', marker='.')
    # plt.show()
    # plt.cla()
    # only if one entry per parsimony
    # ax.scatter(nar[0], nar[1], color='r', label='all actions', marker='.')
    ax.scatter(nar[0], nar[2], color='b', label='low risk', marker='.')
    plt.show()
    plt.cla()


def main():

    parser = argparse.ArgumentParser(description='Plagih IB-Runs!)')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    # parser.add_argument("--data_dir", type=Path, default=Path(__file__).absolute().parent / "data", help="Path to the data directory",)

    parser.add_argument('-name', type=str, help='If the run has a name', default='IB_MSE_sim2')

    args = parser.parse_args()

    name = args.name

    combined_lists(name)


if __name__ == '__main__':
    main()
