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
    fit_normed = []
    fit_first = []

    root_dir_eval = Path.cwd() / f'../slurm_runs/{run_name}'
    if not Path.is_dir(root_dir_eval):
        Path.mkdir(root_dir_eval)
    lut_file = root_dir_eval / 'lutfile.yaml'

    for lsaction in lsactions:
        lsfile = Path.cwd() / f'../slurm_runs/{run_name}/{run_name}{lsaction}'
        print(f'Looging at file: {lsfile}')
        with Path.open(lsfile, 'r') as file:
            yamload = yaml.load(file, Loader=yaml.FullLoader)
        agents.append(yamload)

        fit_normed.append(np.mean([x[1] for x in yamload]))
        fit_first.append(yamload[0][1])

    merged = list(itertools.product(agents[0], agents[1], agents[2]))
    combined_all = []

    for row in merged:
        parsim_sum = float(sum([x[0] for x in row]))
        fitness_sum = float(sum([x[1] for x in row]))
        f_norm = float(sum([x[1]/fit_normed[ii] for ii, x in enumerate(row)]))
        f_1div = float(sum([x[1]/fit_first[ii] for ii, x in enumerate(row)]))
        f_1sub = float(sum([x[1]-fit_first[ii] for ii, x in enumerate(row)]))
        parsims = [float(row[0][0]), float(row[1][0]), float(row[2][0])]
        codes = [x[3] for x in row]
        combined_all.append({'parsim_sum': parsim_sum, 'parsims': parsims, 'codes': codes, 'fitness_sum': fitness_sum, 'f_norm': f_norm, 'f_1div': f_1div, 'f_1sub': f_1sub})

    """
    Creating the candidates to be analysed here
    """
    # combined_best = {}
    combined_best2 = {'fitness_sum': {}, 'f_norm': {}, 'f_1div': {}, 'f_1sub': {}}
    analyse_results = {'fitness_sum': [], 'f_norm': [], 'f_1div': [], 'f_1sub': []}

    for agd_row in combined_all:
        parsim_sum = agd_row['parsim_sum']

        # if parsim_sum in combined_best:
        #     if agd_row['fitness_sum'] < combined_best[parsim_sum]['fitness_sum']:
        #         combined_best[parsim_sum] = agd_row
        # else:
        #     combined_best[parsim_sum] = agd_row

        for measr in ['fitness_sum', 'f_norm', 'f_1div', 'f_1sub']:
            if parsim_sum in combined_best2[measr]:
                if agd_row[measr] < combined_best2[measr][parsim_sum][measr]:
                    combined_best2[measr][parsim_sum] = agd_row
            else:
                combined_best2[measr][parsim_sum] = agd_row

        # msed = math.sqrt(parsims[0] ** 2 + parsims[1] ** 2 + parsims[2] ** 2)
        # if msed in combined_best_msed_dict:
        #     if fitness_sum < combined_best_msed_dict[msed][0]:
        #         combined_best_msed_dict[msed] = [parsim_sum, fitness_sum, parsims, codes]
        # else:
        #     combined_best_msed_dict[msed] = [parsim_sum, fitness_sum, parsims, codes]

    # combined_best = [[parsim_sum, fitness_sum, parsims, codes] for parsim_sum, (fitness_sum, parsims, codes) in combined_best.items()]
    # combined_mse_best = [[parsim_sum, fitness_sum, parsims, codes] for msed, (parsim_sum, fitness_sum, parsims, codes) in combined_best2.items()]


    try:
        # with Path.open(lut_file, 'rb') as file:
        with Path.open(lut_file, 'r') as file:
            lut = yaml.load(file, Loader=yaml.FullLoader)
    except:
        lut = {}
    eval_count = 0
    not_evaled = 0

    print(f'Number of best combinations (fitness_sum): {len(combined_best2["fitness_sum"])}, total combinations: {len(combined_all)}')
    """
    Evaluating with real IB
    """
    for measr in ['fitness_sum', 'f_norm', 'f_1div', 'f_1sub']:  # todo todotodo combined_all or combined_best?
        for parsim_sum, arow in combined_best2[measr].items():
            if parsim_sum != arow['parsim_sum']:
                raise Exception('ASDASDDSADSAASDASD')
            parsims = arow['parsims']
            fitness_sum = arow['fitness_sum']
            lut_hash = f"{parsim_sum}_{parsims}_{fitness_sum}"

            if parsim_sum < 35:
                if lut_hash in lut:
                    experiment, experiment_save = lut[lut_hash]  # todo more evaluations? average?
                else:
                    codes = arow['codes']
                    experiment = float(eval_combined_agents(parsim_sum, parsims, codes))
                    experiment_save = float(eval_combined_agents(parsim_sum, parsims, codes, complete=False))
                    print(f'Combined parsimony {parsim_sum:3.0f} regression-error: {fitness_sum:.4f}. \t({parsims})\tcomplete: {experiment} \tsave: {experiment_save}')
                    lut[lut_hash] = [experiment, experiment_save]
                    eval_count += 1
                    if eval_count % 20 == 0:
                        print(f'Saving lut after evaluating 20')
                        with Path.open(lut_file, 'w') as file:
                            _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)

                analyse_results[measr].append([parsim_sum, experiment, experiment_save])
            else:
                not_evaled += 1

    print(f'Did not evaluate {not_evaled} combinations as they are too complex...')

    with Path.open(lut_file, 'w') as file:
        _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)
    for measr in ['fitness_sum', 'f_norm', 'f_1div', 'f_1sub']:
        plot_this = analyse_results[measr]
        plot_this.sort(key=lambda x: x[0])

        x = [x[0] for x in plot_this]
        y_all = [y[1] for y in plot_this]
        y_safe = [y[2] for y in plot_this]

        plt.tight_layout()
        fig, ax = plt.subplots()
        # ax.set_yscale(yscale)
        # ax.set_ylim(min(bottom, 0), new_top)
        # ax.set_xlim(min(left, 0), new_right)
        ax.set_xlabel('Complexity')
        ax.set_ylabel('reward')
        ax.set_title(f'IB eval {run_name} ({measr})')
        # # only if one entry per parsimony
        ax.plot(x, y_all, label='all actions', marker='.', color='r', )
        ax.plot(x, y_safe, label='low risk', marker='.', color='b')
        plt.savefig(root_dir_eval / f'{measr}-plot.png')
        plt.cla()
        # only if one entry per parsimony
        ax.scatter(x, y_all, color='r', label='all actions', marker='.')
        ax.scatter(x, y_safe, color='b', label='low risk', marker='.')
        plt.savefig(root_dir_eval / f'{measr}-scatter.png')
        plt.close()


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

# todo ein graph pro dimension
