# coding=utf-8
import argparse
import os
import sys
# sys.path.append('modules/')
from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import itertools
import yaml

import numpy as np
import matplotlib.pyplot as plt

dir_slurm = Path(os.path.dirname(os.path.realpath(__file__))) / f'../slurm_runs/'
# lut_file = root_dir_eval / 'lutfile.yaml'
lut_file = Path(os.path.dirname(os.path.realpath(__file__))) / 'lutfile.yaml'
# dir_slurm = Path.cwd() / f'../slurm_runs/'


def get_combined_runs(agents, max_parsim_sum, max_parsim_single):

    merged = list(itertools.product(agents[0], agents[1], agents[2]))
    combined_all = []

    fit_normed = [np.mean([a[1] for a in ag]) for ag in agents]
    fit_0st = [ag[0][1] for ag in agents]

    for row in merged:
        parsim_sum = float(sum([x[0] for x in row]))
        if parsim_sum <= max_parsim_sum and all(x[0] <= max_parsim_single for x in row):
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
    return combined_all


def eval_and_lut(root_dir_eval, combined_all, max_parsim_sum, max_parsim_single):
    """
    Evaluating with real IB
    """

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
    #         if parsim_sum <= max_parsim_sum:
    #             if lut_hash in lut:
    #                 experiment, experiment_safe = lut[lut_hash]  # todo more evaluations? average?
    #             else:
    #                 codes = arow['codes']
    #                 experiment = float(eval_combined_agents(parsim_sum, parsims, codes))
    #                 experiment_safe = float(eval_combined_agents(parsim_sum, parsims, codes, complete=False))
    #                 print(f'Combined parsimony {parsim_sum:3.0f} regression-error: {fitness_sum:.4f}. \t({parsims})\tcomplete: {experiment} \tsave: {experiment_safe}')
    #                 lut[lut_hash] = [experiment, experiment_safe]
    #                 eval_count += 1
    #                 if eval_count % 20 == 0:
    #                     print(f'Saving lut after evaluating 20')
    #                     with Path.open(lut_file, 'w') as file:
    #                         _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)
    #             combined_best2[measr]['experiment'] = experiment
    #             combined_best2[measr]['experiment_safe'] = experiment_safe
    #             analyse_best[measr].append([parsim_sum, experiment, experiment_safe])
    #         else:
    #             not_evaled += 1

    for ii, arow in enumerate(combined_all[:]):  # todo combined_all or combined_best?
        parsims = arow['parsims']
        fitness_sum = arow['fitness_sum']
        parsim_sum = arow['parsim_sum']
        codes = arow['codes']
        lut_hash = f"{codes}"

        if parsim_sum <= max_parsim_sum:
            if lut_hash in lut:
                experiment, experiment_safe = lut[lut_hash]  # todo more evaluations? average?
            else:
                try:
                    experiment = float(eval_combined_agents(parsim_sum, parsims, codes))
                    experiment_safe = float(eval_combined_agents(parsim_sum, parsims, codes, complete=False))
                    print(f'Combined parsimony {parsim_sum:3.0f} regression-error: {fitness_sum:.4f}. \t({parsims})\tcomplete: {experiment} \tsave: {experiment_safe}')
                    lut[lut_hash] = [experiment, experiment_safe]
                    eval_count += 1
                    if eval_count % 50 == 0:
                        print(f'Saving lut after evaluating 50')
                        with Path.open(lut_file, 'w') as file:
                            _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)
                except Exception as ex:
                    print(f'WARNING: Something failed in the evaluation process: {ex}')
                    print(f'WARNING: arow {arow}')
                    # todo
                    # combined_all.remove(arow)
                    combined_all[ii] = None
                    continue

            combined_all[ii]['experiment'] = experiment
            combined_all[ii]['experiment_safe'] = experiment_safe
        else:
            not_evaled += 1

    combined_all = [x for x in combined_all if x is not None]

    with Path.open(lut_file, 'w') as file:
        _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)
    print(f'Did not evaluate {not_evaled} combinations as they are too complex...')
    
    return combined_all
    
    
def plotibeval(combined_all, combined_all_p, combined_all_a, run_name, root_dir_eval):
    parsims = list(set([x['parsim_sum'] for x in combined_all if x['parsim_sum']]))
    parsims.sort()
        
    for measr in ['fitness_sum', 'f_norm', 'f_squared', 'f_normsub', 'f_0div', 'f_0sub']:

        """
        plot best (plot)
        todo kreuztabelle?
        """
        res = [min([row for row in combined_all_p[p]], key=lambda x: x[measr]) for p in parsims]
        # res = max([x for x in res1], key=lambda l: l[measr])
        res.sort(key=lambda x: x['parsim_sum'])
        xx = [x['parsim_sum'] for x in res]
        y_all = [y['experiment'] for y in res]
        y_safe = [y['experiment_safe'] for y in res]

        plt.tight_layout()
        fig, ax = plt.subplots()
        ax.set_title(f'plot ({run_name}, {measr})')  # todo compare measurements
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(-15000, -4000)
        # # only if one entry per parsimony
        ax.plot(xx, y_all, label='all actions', marker='.', color='r', )
        ax.plot(xx, y_safe, label='low risk', marker='.', color='b')
        plt.savefig(root_dir_eval / f'plot-{measr}.png')
        plt.close()

    for a in range(3):
        tmp_comb = combined_all_a[a]
        res = [min(row, key=lambda x: x['f_squared']) for p, row in tmp_comb.items()]
        # res = max([x for x in res1], key=lambda l: l[measr])
        res.sort(key=lambda x: x['parsim_sum'])
        xx = [x['parsim_sum'] for x in res]
        y_all = [y['experiment'] for y in res]
        y_safe = [y['experiment_safe'] for y in res]

        plt.tight_layout()
        fig, ax = plt.subplots()
        ax.set_title(f'scatter-plot, action {a}, ({run_name}, {measr})')
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(-15000, -4000)  # todo
        # # only if one entry per parsimony
        ax.scatter(xx, y_all, label='all actions', marker='.', color='r', )
        ax.scatter(xx, y_safe, label='low risk', marker='.', color='b')
        plt.savefig(root_dir_eval / f'act_{a}-scatter-{measr}.png')
        plt.close()

    """
    Plotting the best candidates per parsimony
    """
    tmp_comb = combined_all_p
    try:
        res = [max([row for row in tmp_comb[p]], key=lambda x: x['experiment']) for p in parsims]
    except:
        for p in parsims:
            print(f'at parsim {p}')
            for row in tmp_comb[p]:
                print(row)

    res.sort(key=lambda x: x['parsim_sum'])
    xx = [x['parsim_sum'] for x in res]
    y_all = [y['experiment'] for y in res]

    res2 = [max([row for row in tmp_comb[p]], key=lambda x: x['experiment_safe']) for p in parsims]
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
    plt.savefig(root_dir_eval / f'best-real-expermiments.png')
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


def combined_lists(run_name, max_parsim_sum, max_parsim_single):
    lsactions = ['_0/pycode_list.yaml', '_1/pycode_list.yaml', '_2/pycode_list.yaml']

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

    combined_all = get_combined_runs(agents, max_parsim_sum, max_parsim_single)
    combined_all.sort(key=lambda x: x['parsim_sum'])

    combined_all = eval_and_lut(root_dir_eval, combined_all, max_parsim_sum, max_parsim_single)

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
    parser.add_argument('-max_parsim_sum', type=int, default=50)
    parser.add_argument('-max_parsim_single', type=int, default=20)
    args = parser.parse_args()

    name = args.name
    max_parsim_sum = args.max_parsim_sum
    max_parsim_single = args.max_parsim_single

    if args.auto:
        rootdirstar = dir_slurm.glob('*')
        # for parsim_tmp in range(5, max_parsim_sum, 5)
        for x in rootdirstar:
            if x.is_dir():
                if x.name[:2] == 'IB':
                    print(f'\nEvaluating {x.name}')
                    combined_lists(x.name, max_parsim_sum, max_parsim_single)
                else:
                    print(f'\nSkipping {x.name}')

    else:
        combined_lists(name, max_parsim_sum, max_parsim_single)
    return


if __name__ == '__main__':
    main()

# todo ein graph pro dimension
# todo print 20 best results in general?
