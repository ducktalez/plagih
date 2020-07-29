# coding=utf-8
import argparse
import os

from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import itertools
import yaml

import numpy as np
import matplotlib.pyplot as plt

dir_slurm = Path(os.path.dirname(os.path.realpath(__file__))) / f'../slurm_runs/'
# dir_slurm = Path.cwd() / f'../slurm_runs/'


def get_combined_runs(agents):

    merged = list(itertools.product(agents[0], agents[1], agents[2]))
    combined_all = []

    fit_normed = [np.mean([a[1] for a in ag]) for ag in agents]
    fit_0st = [ag[0][1] for ag in agents]

    for row in merged:
        parsim_sum = float(sum([x[0] for x in row]))
        fitness_sum = float(sum([x[1] for x in row]))
        f_squared = float(sum([x[1] ** 2 for x in row]))
        f_norm = float(sum([x[1] / fit_normed[ii] for ii, x in enumerate(row)]))
        f_normsub = float(sum([x[1] - fit_normed[ii] for ii, x in enumerate(row)]))
        f_0div = float(sum([x[1] / fit_0st[ii] for ii, x in enumerate(row)]))
        f_0sub = float(sum([x[1] - fit_0st[ii] for ii, x in enumerate(row)]))
        parsims = [float(row[0][0]), float(row[1][0]), float(row[2][0])]
        codes = [x[3] for x in row]
        combined_all.append(
            {'parsim_sum': parsim_sum, 'experiment': None, 'experiment_save': None, 'parsims': parsims, 'codes': codes,
             'fitness_sum': fitness_sum, 'f_squared': f_squared, 'f_norm': f_norm, 'f_normsub': f_normsub, 'f_0div': f_0div, 'f_0sub': f_0sub})
    return combined_all


def eval_and_lut(root_dir_eval, combined_all, max_parsim):
    """
    Evaluating with real IB
    """

    lut_file = root_dir_eval / 'lutfile.yaml'
    try:
        # with Path.open(lut_file, 'rb') as file:
        with Path.open(lut_file, 'r') as file:
            lut = yaml.load(file, Loader=yaml.FullLoader)
    except:
        lut = {}
    eval_count = 0
    not_evaled = 0

    # print(f'Number of best combinations (fitness_sum): {len(combined_best2["fitness_sum"])}, total combinations: {len(combined_all)}')
    # for measr in ['fitness_sum', 'f_norm', 'f_norm', 'f_normsub', 'f_0div', 'f_0sub']:
    #     for parsim_sum, arow in combined_best2[measr].items():  # todo combined_all or combined_best?
    #         parsims = arow['parsims']
    #         fitness_sum = arow['fitness_sum']
    #         parsim_sum = arow['parsim_sum']
    #         lut_hash = f"{parsim_sum}_{parsims}_{fitness_sum}"
    # 
    #         if parsim_sum < max_parsim:
    #             if lut_hash in lut:
    #                 experiment, experiment_save = lut[lut_hash]  # todo more evaluations? average?
    #             else:
    #                 codes = arow['codes']
    #                 experiment = float(eval_combined_agents(parsim_sum, parsims, codes))
    #                 experiment_save = float(eval_combined_agents(parsim_sum, parsims, codes, complete=False))
    #                 print(f'Combined parsimony {parsim_sum:3.0f} regression-error: {fitness_sum:.4f}. \t({parsims})\tcomplete: {experiment} \tsave: {experiment_save}')
    #                 lut[lut_hash] = [experiment, experiment_save]
    #                 eval_count += 1
    #                 if eval_count % 20 == 0:
    #                     print(f'Saving lut after evaluating 20')
    #                     with Path.open(lut_file, 'w') as file:
    #                         _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)
    #             combined_best2[measr]['experiment'] = experiment
    #             combined_best2[measr]['experiment_save'] = experiment_save
    #             analyse_best[measr].append([parsim_sum, experiment, experiment_save])
    #         else:
    #             not_evaled += 1

    for ii, arow in enumerate(combined_all[:]):  # todo combined_all or combined_best?
        parsims = arow['parsims']
        fitness_sum = arow['fitness_sum']
        parsim_sum = arow['parsim_sum']
        lut_hash = f"{parsim_sum}_{parsims}_{fitness_sum}"

        if parsim_sum < max_parsim:
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

            combined_all[ii]['experiment'] = experiment
            combined_all[ii]['experiment_save'] = experiment_save
        else:
            not_evaled += 1

    with Path.open(lut_file, 'w') as file:
        _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)
    print(f'Did not evaluate {not_evaled} combinations as they are too complex...')
    
    return combined_all


def combined_lists(run_name, max_parsim):
    lsactions = ['_0/pycode_list.yaml', '_1/pycode_list.yaml', '_2/pycode_list.yaml']

    agents = []
    for act in lsactions:
        lfile = dir_slurm / run_name /f'{run_name}{act}'
        print(f'Looking at file: {lfile}')
        with Path.open(lfile, 'r') as file:
            yamload = yaml.load(file, Loader=yaml.FullLoader)
        agents.append(yamload)

    root_dir_eval = dir_slurm / f'{run_name}'
    if not Path.is_dir(root_dir_eval):
        Path.mkdir(root_dir_eval)

    combined_all = get_combined_runs(agents)
    combined_all.sort(key=lambda x: x['parsim_sum'])
    
    combined_all = eval_and_lut(root_dir_eval, combined_all, max_parsim)
    
    plotibeval(combined_all)
    
    """
    Creating the candidates to be analysed here
    """
    # # # combined_best = {}
    # combined_best2 = {'fitness_sum': {}, 'f_norm': {}, 'f_normsub': {}, 'f_0div': {}, 'f_0sub': {}}
    # # analyse_best = {'fitness_sum': [], 'f_norm': [], 'f_normsub': [], 'f_0div': [], 'f_0sub': []}
    # # analyse_all = {'fitness_sum': [], 'f_norm': [], 'f_normsub': [], 'f_0div': [], 'f_0sub': []}
    # 
    # for measr in ['fitness_sum', 'f_squared', 'f_norm', 'f_normsub', 'f_0div', 'f_0sub']:
    #     for row in combined_all:
    #         parsim_sum = row['parsim_sum']
    #         if parsim_sum in combined_best2[measr]:
    #             if row[measr] < combined_best2[measr][parsim_sum][measr]:
    #                 combined_best2[measr][parsim_sum] = row
    #         else:
    #             combined_best2[measr][parsim_sum] = row
    #     
    #     # todo simon msed?
    #     # msed = math.sqrt(parsims[0] ** 2 + parsims[1] ** 2 + parsims[2] ** 2)
    #     # if msed in combined_best_msed_dict:
    #     #     if fitness_sum < combined_best_msed_dict[msed][0]:
    #     #         combined_best_msed_dict[msed] = [parsim_sum, fitness_sum, parsims, codes]
    #     # else:
    #     #     combined_best_msed_dict[msed] = [parsim_sum, fitness_sum, parsims, codes]
    # 
    # # combined_best = [[parsim_sum, fitness_sum, parsims, codes] for parsim_sum, (fitness_sum, parsims, codes) in combined_best.items()]
    # # combined_mse_best = [[parsim_sum, fitness_sum, parsims, codes] for msed, (parsim_sum, fitness_sum, parsims, codes) in combined_best2.items()]
    
    
def plotibeval(combined_all):
    parsims = list(set([x['parsim_sum'] for x in combined_all])).sort()
    
    def get_best_lists(combined_all, measr):
        combined_all
        
    for measr in ['fitness_sum', 'f_norm', 'f_normsub', 'f_0div', 'f_0sub']:

        """
        plot best (plot)
        """
        plot_best = analyse_best[measr]
        plot_best.sort(key=lambda x: x[0])
        x = [x[0] for x in plot_best]
        y_all = [y[1] for y in plot_best]
        y_safe = [y[2] for y in plot_best]

        plt.tight_layout()
        fig, ax = plt.subplots()
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_title(f'IB eval {run_name} ({measr})')
        # # only if one entry per parsimony
        ax.plot(x, y_all, label='all actions', marker='.', color='r', )
        ax.plot(x, y_safe, label='low risk', marker='.', color='b')
        plt.savefig(root_dir_eval / f'{measr}-plot.png')
        plt.close()

        """
        plot all (scatter)
        """
        plot_all = analyse_all[measr]
        plot_all.sort(key=lambda x: x[0])
        x = [x[0] for x in plot_all]
        y_all = [y[1] for y in plot_all]
        y_safe = [y[2] for y in plot_all]

        plt.tight_layout()
        fig, ax = plt.subplots()
        ax.set_title(f'IB eval {run_name} ({measr})')
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        # only if one entry per parsimony
        ax.scatter(x, y_all, label='all actions', marker='.', color='r')
        ax.scatter(x, y_safe, label='low risk', marker='.', color='b')
        plt.savefig(root_dir_eval / f'{measr}-scatter.png')
        plt.close()

        for dim in range(3):
            plot_all = analyse_all[measr]
            plot_all.sort(key=lambda x: x[0])
            x = [x[0] for x in plot_all]
            y_all = [y[1] for y in plot_all]
            y_safe = [y[2] for y in plot_all]



def main():

    parser = argparse.ArgumentParser(description='Plagih IB-Run evaluation')
    parser.add_argument('-name', type=str, help='If the run has a name', default='IB_MSE_sim2')
    parser.add_argument('-auto', action='store_true')
    parser.add_argument('-max_parsim', type=int, default=30)
    args = parser.parse_args()

    name = args.name
    max_parsim = args.max_parsim

    if args.auto:
        rootdirstar = dir_slurm.glob('*')

        for x in rootdirstar:
            if x.is_dir():
                if x.name[:2] == 'IB':
                    print(f'\nEvaluating {x.name}')
                    combined_lists(x.name, max_parsim)
                else:
                    print(f'\nSkipping {x.name}')

    else:
        combined_lists(name, max_parsim)
    return


if __name__ == '__main__':
    main()

# todo ein graph pro dimension
