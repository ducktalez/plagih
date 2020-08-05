# coding=utf-8
import argparse
import os
import sys
sys.path.append('../../')
from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import itertools
import yaml
import multiprocessing as mp
import time

import numpy as np
import matplotlib.pyplot as plt

dir_slurm = Path(os.path.dirname(os.path.realpath(__file__))) / f'../slurm_runs/'
# lut_file = root_dir_eval / 'lutfile.yaml'
# dir_slurm = Path.cwd() / f'../slurm_runs/'

# def get_combined_runs(agents, parsim_max_sum, parsim_max_single):


def mp_evall(arow):
    parsims = arow['parsims']
    fitness_sum = arow['fitness_sum']
    parsim_sum = arow['parsim_sum']
    codes = arow['codes']
    lut_hash = f"{arow['codes']}"

    try:
        experiment = eval_combined_agents(codes)
        experiment_safe = eval_combined_agents(codes, complete=False)
        print(f'Combined parsimony {parsim_sum:3.0f} regression-error: {fitness_sum:.4f}. \t({parsims})\tcomplete: {experiment} \tsafe: {experiment_safe}')
        return [lut_hash, experiment, experiment_safe]
    except Exception as ex:
        print(f'WARNING: Something failed in the evaluation process: {ex}')
        print(f'WARNING: arow {arow}')
        return [lut_hash, None, None]


def eval_and_lut(combined_all, parsim_max_sum, parsim_max_single, lut_file):
    """
    Evaluating with real IB
    """

    try:
        # with Path.open(lut_file, 'rb') as file:
        with Path.open(lut_file, 'r') as file:
            lut = yaml.load(file, Loader=yaml.FullLoader)
        lut = lut or {}  # was none after loading empty file <.<
    except:
        lut = {}

    open_combinations = []
    for arow in combined_all:
        try:
            if f"{arow['codes']}" not in lut and arow['parsim_sum'] <= parsim_max_sum and all(x<=parsim_max_single for x in arow['parsims']):
                open_combinations.append(arow)
        except:
            pass

    mp.Process()
    print(f'Available cores for mp: {mp.cpu_count()-1}')  # sfeh -1 cause of my pc ;_;
    with mp.Pool(mp.cpu_count()) as p:
        results = p.map(mp_evall, open_combinations)

    result_dict = {a: [b, c] for a, b, c in results}  # [lut_hash, experiment, experiment_safe]

    lut.update(result_dict)
    print(f'Saving the updated lut file.')  # {not_evaled}')
    with Path.open(lut_file, 'w') as file:
        _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)

    combined_all_cpy = []
    for ii, arow in enumerate(combined_all[:]):
        if arow is not None:
            hashy = f"{arow['codes']}"
            try:
                e1, e2 = lut.get(hashy)
                arow['experiment'] = e1
                arow['experiment_safe'] = e2
                combined_all_cpy.append(arow)
            except:
                pass

        # combined_best2[measr]['experiment'] = experiment
        # combined_best2[measr]['experiment_safe'] = experiment_safe
        #         fitness_sum = arow['fitness_sum']
        #         parsim_sum = arow['parsim_sum']
        #         lut_hash = f"{parsim_sum}_{parsims}_{fitness_sum}"
        #
        #         if parsim_sum <= parsim_max_sum:
        #             if lut_hash in lut:
        #                 experiment, experiment_safe = lut[lut_hash]
    return combined_all


def plotibeval(combined_all, combined_all_p, combined_all_a, run_name, root_dir_eval):
    parsims = list(set([x['parsim_sum'] for x in combined_all if x['parsim_sum']]))
    parsims.sort()
        
    for measr in ['fitness_sum', 'f_norm', 'f_squared', 'f_normsub', 'f_0div', 'f_0sub']:
        """
        plot best (plot)
        todo kreuztabelle?
        todo are evaluations too different? take a look at start eval... YES! Make more evaluations... create eval_list. update list, once, first.
        """
        res = [min([row for row in combined_all_p[p]], key=lambda x: x[measr]) for p in parsims]
        # res = max([x for x in res1], key=lambda l: l[measr])
        res.sort(key=lambda x: x['parsim_sum'])
        xx = [x['parsim_sum'] for x in res]
        y_all = [y['experiment'] for y in res]
        y_safe = [y['experiment_safe'] for y in res]

        plt.tight_layout()
        fig, ax = plt.subplots()
        ax.set_title(f'plot ({run_name}, {measr})')  # todo compare measurements and remove one of them forever
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(-15000, -4000)
        # # only if one entry per parsimony
        ax.plot(xx, y_all, label='all actions', marker='.', color='r', )
        ax.plot(xx, y_all, label='all actions', marker='.', color='r', )
        ax.plot(xx, y_safe, label='low risk', marker='.', color='b')
        plt.savefig(root_dir_eval / f'plot-{measr}.png', dpi=300)
        plt.close()

    """
    Plotting the performance of each action
    # todo GP Idee: Robustheit durch Angriffe auf die Genetik?
    """
    for a, parsim_dict in enumerate(combined_all_a):
        xx = []
        exp = []
        exp_safe = []
        # res = [min(row, key=lambda x: x['f_squared']) for p, row in parsim_dict.items()]  # nah
        # for p, row in parsim_dict.items():
        #     res[p] =
        for p, rows in parsim_dict.items():
            xx.append(p)
            exp.append(np.mean([row['experiment'] for row in rows]))
            exp_safe.append(np.mean([row['experiment_safe'] for row in rows]))

        fig, ax = plt.subplots()
        ax.set_title(f'action {a}, ({run_name}, {measr})')
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(-15000, -4000)  # todo
        # # only if one entry per parsimony
        ax.plot(xx, exp, label='all actions', marker='.', color='r', )
        ax.plot(xx, exp_safe, label='low risk', marker='.', color='b')
        plt.savefig(root_dir_eval / f'act_{a}-plot-{measr}.png', dpi=150)
        # plt.savefig(root_dir_eval / f'act_{a}-plot-{measr}.svg')
        plt.close()

    """
    Plotting the best candidates per parsimony
    """
    parsim_dict = combined_all_p
    try:
        res = [max([row for row in parsim_dict[p]], key=lambda x: x['experiment']) for p in parsims]
    except:
        for p in parsims:
            print(f'at parsim {p}')
            for row in parsim_dict[p]:
                print(row)

    res.sort(key=lambda x: x['parsim_sum'])
    xx = [x['parsim_sum'] for x in res]
    y_all = [y['experiment'] for y in res]

    res2 = [max([row for row in parsim_dict[p]], key=lambda x: x['experiment_safe']) for p in parsims]
    res2.sort(key=lambda x: x['parsim_sum'])
    y_safe = [y['experiment_safe'] for y in res2]

    plt.tight_layout()
    fig, ax = plt.subplots()
    ax.set_title(f'Best{run_name} ({measr})')
    ax.set_xlabel('complexity')
    ax.set_ylabel('reward')
    ax.set_ylim(-15000, -4000)  # todo
    # # only if one entry per parsimony
    ax.plot(xx, y_all, label='all actions', marker='.', color='r', )
    ax.plot(xx, y_safe, label='low risk', marker='.', color='c')
    plt.savefig(root_dir_eval / f'best-real-expermiments.png', dpi=300)
    # plt.savefig(root_dir_eval / f'best-real-expermiments.svg')
    plt.close()

    # """
    # plot all (scatter)
    # """
    # plot_all = analyse_all[measr]
    # plot_all.sort(key=lambda x: x[0])
    # x = [x[0] for x in plot_all]
    # y_all = [y[1] for y in plot_all]
    # y_safe = [y[2] for y in plot_all]
    #
    # plt.tight_layout()
    # fig, ax = plt.subplots()
    # ax.set_title(f'IB eval {run_name} ({measr})')
    # ax.set_xlabel('complexity')
    # ax.set_ylabel('reward')
    # # only if one entry per parsimony
    # ax.scatter(x, y_all, label='all actions', marker='.', color='r')
    # ax.scatter(x, y_safe, label='low risk', marker='.', color='b')
    # plt.savefig(root_dir_eval / f'{measr}-scatter.png')
    # plt.close()
    #
    # for dim in range(3):
    #     plot_all = analyse_all[measr]
    #     plot_all.sort(key=lambda x: x[0])
    #     x = [x[0] for x in plot_all]
    #     y_all = [y[1] for y in plot_all]
    #     y_safe = [y[2] for y in plot_all]


def combined_lists(run_name, parsim_max_sum, parsim_max_single, local_yamls=False):
    lsactions = ['_0/pycode_list.yaml', '_1/pycode_list.yaml', '_2/pycode_list.yaml']
    if local_yamls:
        lut_file = dir_slurm / run_name / 'lutfile.yaml'
    else:
        lut_file = Path(os.path.dirname(os.path.realpath(__file__))) / 'lutfile.yaml'

    agents = []
    for act in lsactions:
        lfile = dir_slurm / run_name / f'{run_name}{act}'
        print(f'Looking at file: {lfile}')
        with Path.open(lfile, 'r') as file:
            yamload = yaml.load(file, Loader=yaml.FullLoader)
        agents.append(yamload)

    root_dir_eval = dir_slurm / f'{run_name}'
    if not Path.is_dir(root_dir_eval):
        Path.mkdir(root_dir_eval)

    merged = list(itertools.product(agents[0], agents[1], agents[2]))
    combined_all = []

    fit_normed = [np.mean([a[1] for a in ag]) for ag in agents]
    fit_0st = [ag[0][1] for ag in agents]

    for row in merged:
        parsim_sum = float(sum([x[0] for x in row]))
        if parsim_sum <= parsim_max_sum and all(x[0] <= parsim_max_single for x in row):
            fitness_sum = float(sum([x[1] for x in row]))
            f_squared = float(sum([x[1] ** 2 for x in row]))
            f_norm = float(sum([x[1] / fit_normed[ii] for ii, x in enumerate(row)]))
            f_normsub = float(sum([x[1] - fit_normed[ii] for ii, x in enumerate(row)]))
            f_0div = float(sum([x[1] / fit_0st[ii] for ii, x in enumerate(row)]))
            f_0sub = float(sum([x[1] - fit_0st[ii] for ii, x in enumerate(row)]))
            parsims = [float(row[0][0]), float(row[1][0]), float(row[2][0])]
            codes = [x[3] for x in row]
            combined_all.append(
                {'parsim_sum': parsim_sum, 'experiment': None, 'experiment_safe': None, 'parsims': parsims, 'codes': codes,
                 'fitness_sum': fitness_sum, 'f_squared': f_squared, 'f_norm': f_norm, 'f_normsub': f_normsub, 'f_0div': f_0div, 'f_0sub': f_0sub})
    # return combined_all
    # combined_all = get_combined_runs(agents, parsim_max_sum, parsim_max_single)

    combined_all.sort(key=lambda x: x['parsim_sum'])

    combined_all = eval_and_lut(combined_all, parsim_max_sum, parsim_max_single, lut_file)

    combined_all_p = {}
    combined_all_a = [{}, {}, {}]
    for row in combined_all:
        psum = row['parsim_sum']
        try:
            combined_all_p[psum].append(row)
        except:
            combined_all_p[psum] = [row]
        for a in range(3):
            px = row['parsims'][a]
            try:
                combined_all_a[a][px].append(row)
            except:
                combined_all_a[a][px] = [row]

    plotibeval(combined_all, combined_all_p, combined_all_a, run_name, root_dir_eval)


def main():

    parser = argparse.ArgumentParser(description='Plagih IB-Run evaluation')
    parser.add_argument('-name', type=str, help='If the run has a name', default='IB_MSE_sim2')
    parser.add_argument('-auto', action='store_true')
    parser.add_argument('-loyal_yaml', action='store_true')
    parser.add_argument('-parsim_max_sum', type=int, default=35)
    parser.add_argument('-parsim_max_single', type=int, default=14)
    args = parser.parse_args()

    name = args.name
    parsim_max_sum = args.parsim_max_sum
    parsim_max_single = args.parsim_max_single

    if args.auto:
        rootdirstar = dir_slurm.glob('*')
        # for parsim_tmp in range(5, parsim_max_sum, 5)
        for x in rootdirstar:
            if x.is_dir():
                if x.name[:2] == 'IB':
                    print(f'\nEvaluating {x.name}')
                    combined_lists(x.name, parsim_max_sum, parsim_max_single)
                else:
                    print(f'\nSkipping {x.name}')

    else:
        combined_lists(name, parsim_max_sum, parsim_max_single)
    return


if __name__ == '__main__':
    main()

# todo ein graph pro dimension
# todo print 20 best results in general?
