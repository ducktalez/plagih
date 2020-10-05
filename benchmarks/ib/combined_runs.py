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
from plagih.plagih_gp_base_class_xai import *

import numpy as np
import matplotlib.pyplot as plt

dir_slurm = Path(os.path.dirname(os.path.realpath(__file__))) / f'../slurm_runs/'
# lut_file = root_dir_eval / 'lutfile.yaml'
# dir_slurm = Path.cwd() / f'../slurm_runs/'

# def get_combined_runs(agents, parsim_max_sum, parsim_max_single):
funny_limits = (-15000, -4000)


def mp_evall(arow):
    """
    Use mp (multiprocessing) to evaluate parallelized
    """
    parsims = arow['parsims']
    regress_sum = arow['regress_sum']
    parsim_sum = arow['parsim_sum']
    codes = arow['codes']
    lut_hash = f"{arow['codes']}"

    try:
        ibx = eval_combined_agents(codes)
        ibx_safe = eval_combined_agents(codes, complete=False)
        ibx_r50 = eval_combined_agents(codes, randomize=50, repeat_avg=10)
        ibx_safe_r50 = eval_combined_agents(codes, complete=False, randomize=50, repeat_avg=10)
        print(f'Combined parsimony {parsim_sum:3.0f} regression-error: {regress_sum:.4f}. \t({parsims})\tcomplete: {ibx:0.1f} \tsafe: {ibx_safe:0.1f} \tall_r50: {ibx_r50:0.1f} \tsafe_r50: {ibx_safe_r50:0.1f}')
        return [lut_hash, ibx, ibx_safe, ibx_r50, ibx_safe_r50]
    except Exception as ex:
        print(f'WARNING: Something failed in the evaluation process: {ex}')
        print(f'WARNING: arow {arow}')
        return None  # [lut_hash, None, None]


def eval_and_lut(eval_list, parsim_MAX, parsim_1MAX, lut_file, mp_cpu_MAX):
    """
    Evaluating with real IB
    """

    try:
        with Path.open(lut_file, 'r') as file:
            lut = yaml.load(file, Loader=yaml.FullLoader)
        lut = lut or {}  # was none after loading empty file <.<
    except:
        lut = {}

    """
    Check if the value is in the lut file
    """
    open_combinations = []
    for arow in eval_list:
        try:
            if f"{arow['codes']}" not in lut and arow['parsim_sum'] <= parsim_MAX and all(x <= parsim_1MAX for x in arow['parsims']):
                open_combinations.append(arow)
        except:
            print('Fuk')

    mp.Process()
    mp_cores = min(mp.cpu_count(), mp_cpu_MAX)
    print(f'Using {mp_cores} for mp (available: {mp.cpu_count()})')
    with mp.Pool(mp_cores) as p:
        mp_result = p.map(mp_evall, open_combinations)

    mp_result_dict = {a: [b, c, d, e] for a, b, c, d, e in mp_result}  # [lut_hash, experiment, experiment_safe, experiment_r50, experiment_safe_r50]
    lut.update(mp_result_dict)

    if len(mp_result_dict) > 0:
        print(f'Saving the updated lut file.')
        with Path.open(lut_file, 'w') as file:
            _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)

    combined_all_cpy = []

    for arow in eval_list[:]:
        if arow is not None:
            hashy = f"{arow['codes']}"
            try:
                b, c, d, e = lut.get(hashy)
                arow['experiment'] = b
                arow['experiment_safe'] = c
                arow['experiment_r50'] = d
                arow['experiment_safe_r50'] = e
                combined_all_cpy.append(arow)
            except:
                pass

    # for arow in eval_list:

    return combined_all_cpy


def plot_best_prediction(root_dir_eval, run_name, parsims, combined_all_p, lut_file, parsim_MAX, parsim_1MAX, mp_cpu_MAX):
    """
    plot best guess
    """
    best_regrerr = [min([row for row in combined_all_p[p]], key=lambda x: x['regress_sum']) for p in parsims]
    best_regrerr_dict = eval_and_lut(best_regrerr, parsim_MAX, parsim_1MAX, lut_file, mp_cpu_MAX)

    """
    print the combined runs that belong together
    """
    # yaml_dump(root_dir_eval / 'best_regrerr.yaml', [' '.join(str(xx) for xx in x['parsims']) for x in best_regrerr_dict])  # todo delete this?

    """
    okay
    best_regrerr_dict
    """
    best_regrerr_dict.sort(key=lambda x: x['parsim_sum'])
    res_all = [best_regrerr_dict[0]]
    cnt = [0]
    best = best_regrerr_dict[0]

    """
    ignore if we assume worse examples by regress_sum
    """
    for p in parsims:
        prows = [row for row in combined_all_p[p]]
        best_row = min(prows, key=lambda x: x['regress_sum'])
        if best_row['regress_sum'] < best['regress_sum']:
            res_all.append(best_row)
            cnt.append(len(prows))
        else:
            print(f'Not a better entry at {p}')

    xx = [x['parsim_sum'] for x in res_all]
    y_all = [y['experiment'] for y in res_all]
    y_safe = [y['experiment_safe'] for y in res_all]
    y_all_r50 = [y['experiment_r50'] for y in res_all]
    y_safe_r50 = [y['experiment_safe_r50'] for y in res_all]

    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(funny_limits)
        ax.plot(xx, y_all, label='all actions', marker='.', color='r')
        ax.plot(xx, y_safe, label='low risk', marker='.', color='b')
        ax2 = ax.twinx()
        ax2.plot(xx, cnt, color='tab:gray', label='Possible combinations', linestyle='dashed', marker='.')  # linestyle='None'
        ax2.tick_params(axis='y', labelcolor='tab:gray')
        ax.legend(loc='lower right')
        ax2.legend(loc='lower left')
        fig.tight_layout()
        fig.savefig(root_dir_eval / f'{run_name}-regression_sum.png', dpi=300)
        plt.close()

    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(funny_limits)
        ax.plot(xx, y_all_r50, label='all actions (randomized)', marker='.', color='r')
        ax.plot(xx, y_safe_r50, label='low risk (randomized)', marker='.', color='b')
        ax2 = ax.twinx()
        ax2.plot(xx, cnt, color='tab:gray', label='Possible combinations', linestyle='dashed', marker='.')  # linestyle='None'
        ax2.tick_params(axis='y', labelcolor='tab:gray')
        ax.legend(loc='lower right')
        ax2.legend(loc='lower left')
        fig.tight_layout()
        fig.savefig(root_dir_eval / f'{run_name}-regression_sum_r50.png', dpi=300)
        plt.close()


def plot_per_action(root_dir_eval, run_name, parsims, combined_all_p):
    """
    Plotting the performance of each action
    """
    pass
    # for a, parsim_dict in enumerate(combined_all_a):
    #     xx = []
    #     exp = []
    #     exp_safe = []
    #     # res_all = [min(row, key=lambda x: x['f_squared']) for p, row in parsim_dict.items()]  # nah
    #     # for p, row in parsim_dict.items():
    #     #     res_all[p] =
    #     for p, rows in parsim_dict.items():
    #         xx.append(p)
    #         exp.append(np.mean([row['experiment'] for row in rows]))
    #         exp_safe.append(np.mean([row['experiment_safe'] for row in rows]))
    #
    #     fig, ax = plt.subplots()
    #     ax.set_title(f'action {a}, ({run_name})')
    #     ax.set_xlabel('complexity')
    #     ax.set_ylabel('reward')
    #     ax.set_ylim(funny_limits)
    #     # # only if one entry per parsimony
    #     ax.plot(xx, exp, label='all actions', marker='.', color='r', )
    #     ax.plot(xx, exp_safe, label='low risk', marker='.', color='b')
    #     ax.legend(loc='lower left')
    #     plt.savefig(root_dir_eval / f'act_{a}-plot.png', dpi=150)
    #     # plt.savefig(root_dir_eval / f'act_{a}-plot-{measr}.svg')
    #     plt.close()


def plot_actual_best(root_dir_eval, run_name, parsims, combined_all):
    """
    Plotting the best candidates per parsimony
    """
    parsim_dict = combined_all
    # try:
    res_all = [max([row for row in parsim_dict[p]], key=lambda x: x['experiment']) for p in parsims]
    # except:
    #     for p in parsims:
    #         print(f'at parsim {p}')
    #         for row in parsim_dict[p]:
    #             print(row)
    #     raise

    res_all.sort(key=lambda x: x['parsim_sum'])
    xx = [x['parsim_sum'] for x in res_all]
    y_all = [y['experiment'] for y in res_all]

    res_safe = [max([row for row in parsim_dict[p]], key=lambda x: x['experiment_safe']) for p in parsims]
    res_safe.sort(key=lambda x: x['parsim_sum'])
    y_safe = [y['experiment_safe'] for y in res_safe]

    res_r50 = [max([row for row in parsim_dict[p]], key=lambda x: x['experiment_r50']) for p in parsims]
    res_r50.sort(key=lambda x: x['parsim_sum'])
    y_all_r50 = [y['experiment_r50'] for y in res_r50]

    res_safe_r50 = [max([row for row in parsim_dict[p]], key=lambda x: x['experiment_safe_r50']) for p in parsims]
    res_safe_r50.sort(key=lambda x: x['parsim_sum'])
    y_safe_r50 = [y['experiment_safe_r50'] for y in res_safe_r50]

    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(funny_limits)
        # # only if one entry per parsimony
        ax.plot(xx, y_all, label='all actions', marker='.', color='r', )
        ax.plot(xx, y_safe, label='low risk', marker='.', color='c')
        ax.legend(loc='lower left')
        # ax.set_title(f'{run_name} (best)')
        fig.tight_layout()
        fig.savefig(root_dir_eval / f'{run_name} (best).png', dpi=300)
        # plt.savefig(root_dir_eval / f'best-real-expermiments.svg')

    with plt.rc_context(rc={}):
        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.set_xlabel('complexity')
        ax.set_ylabel('reward')
        ax.set_ylim(funny_limits)
        # # only if one entry per parsimony
        ax.plot(xx, y_all_r50, label='all actions, randomized(50)', marker='.', color='r', )
        ax.plot(xx, y_safe_r50, label='low risk, randomized(50)', marker='.', color='c')
        ax.legend(loc='lower left')
        fig.tight_layout()
        fig.savefig(root_dir_eval / f'{run_name} (best)_r50.png', dpi=300)
        # plt.savefig(root_dir_eval / f'best-real-expermiments.svg')


def plot_scatter_some():
    # """
    # plot all (scatter)
    # """
    pass
    # plot_all = combined_all
    # plot_all.sort(key=lambda x: x[0])
    # x = [x[0] for x in plot_all]
    # y_all = [y[1] for y in plot_all]
    # y_safe = [y[2] for y in plot_all]
    #
    # plt.tight_layout()
    #     fig.tight_layout()#
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


def combined_lists(run_name, parsim_MAX, parsim_1MAX, local_yamls=False, mp_cpu_MAX=16):
    """
    Make the combined evaluation of industrial benchmark runs.
    Three runs have to be combined from their raw code.
    (I now found a much better way by loading from the backup file, but I am lazy x~~~D)
    """
    merge_paretos(run_name)  # e.g. 'IB_MSE_s3m'
    root_dir_eval = dir_slurm / f'{run_name}'
    if not Path.is_dir(root_dir_eval):
        Path.mkdir(root_dir_eval)

    if local_yamls:
        lut_file = dir_slurm / run_name / 'lutfile.yaml'
    else:
        lut_file = Path(os.path.dirname(os.path.realpath(__file__))) / 'lutfile.yaml'

    agents = []
    for act in ['0', '1', '2']:
        lfile = dir_slurm / run_name / f'{run_name}_{act}/pycode_list.yaml'
        print(f'combining runs. Trying to load yaml file at: {lfile}')
        try:
            with Path.open(lfile, 'r') as file:
                yamload = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError as fnferr:
            raise FileNotFoundError(f'File does not exist (yet). Please check your runs first: {fnferr}')
        agents.append(yamload)

    iter_combinations = list(itertools.product(agents[0], agents[1], agents[2]))

    combined_all_less = []
    for row in iter_combinations:
        parsim_sum = float(sum([x[0] for x in row]))
        if parsim_sum <= parsim_MAX and all(x[0] <= parsim_1MAX for x in row):
            regress_sum = float(sum([x[1] for x in row]))
            parsims = [float(row[0][0]), float(row[1][0]), float(row[2][0])]
            codes = [x[3] for x in row]
            combined_all_less.append({'parsim_sum': parsim_sum,
                                      'experiment': None,
                                      'experiment_safe': None,
                                      'experiment_r50': None,
                                      'experiment_safe_r50': None,
                                      'parsims': parsims,
                                      'codes': codes,
                                      'regress_sum': regress_sum})

    combined_all_less.sort(key=lambda x: x['parsim_sum'])

    combined_all_p = {}
    combined_all_a = [{}, {}, {}]
    for row in combined_all_less:
        psum = int(row['parsim_sum'])
        try:
            combined_all_p[psum].append(row)
        except Exception as ex:
            combined_all_p[psum] = [row]
        for a in range(3):
            px = int(row['parsims'][a])
            try:
                combined_all_a[a][px].append(row)
            except Exception as ex:
                combined_all_a[a][px] = [row]

    """
    the main plotting procedures
    """
    parsims = sorted(list(set([x['parsim_sum'] for x in combined_all_less])))
    plot_best_prediction(root_dir_eval, run_name, parsims, combined_all_p, lut_file, parsim_MAX, parsim_1MAX, mp_cpu_MAX)


def merge_paretos(run_name):
    """
    load each IB run and add its pareto front to a merged plot
    """
    with plt.rc_context():
        fig, ax = plt.subplots(ncols=1)  # , figsize=(9, 9)  # todo opening warning (RuntimeWarning: More than 20 figures have been opened... through the pyplot interface) maybe it doesnt close
        plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
        for ii, color in enumerate(['blue', 'magenta', 'red']):
            lfile = dir_slurm / run_name / f'{run_name}_{ii}'  #
            with Path.open(lfile / 'backup/backup.p', 'rb') as file:
                gp_backup_data = pickle.load(file)
                gen_id, pareto, pop_base, monitor_pd, a_helping_dict = gp_backup_data
                # loaded_runs[ii] = pareto
            tuples = [[parsim, fitness] for (parsim, fitness, cooltree) in pareto]
            xx, yy = np.array(tuples).T
            ax.step(xx, yy, linestyle='dotted', marker='.', label=f'action {ii}', where='post')

        ax.set_xlabel('complexity')
        ax.set_ylabel('regression error')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(dir_slurm / run_name / f'{run_name}-pareto_combined', dpi=300)
        plt.close()

    print('combined runs: merged pareto entries into one plot!')


def main():

    parser = argparse.ArgumentParser(description='Plagih IB-Run evaluation')
    parser.add_argument('-name', type=str, help='If the run has a name', default='IB_MSE_sim2')
    parser.add_argument('-auto', action='store_true')
    parser.add_argument('-locallut', action='store_true')
    parser.add_argument('-mp_cpu_cores_max', type=int,  default=8)
    parser.add_argument('-parsim_max_sum', type=int, default=40)
    parser.add_argument('-parsim_max_single', type=int, default=40)
    args = parser.parse_args()

    name = args.name
    parsim_max_sum = args.parsim_max_sum
    parsim_max_single = args.parsim_max_single

    if args.auto:
        rootdirstar = dir_slurm.glob('*')
        for x in rootdirstar:
            if x.is_dir():
                if x.name[:2] == 'IB':
                    print(f'\nEvaluating {x.name}')
                    try:
                        combined_lists(x.name, parsim_max_sum, parsim_max_single, local_yamls=args.locallut, mp_cpu_MAX=args.mp_cpu_cores_max)
                    except Exception as ex:
                        print(f'Failed evaluation for {x.name}. ignoring. Reason: {ex}')
                        # sfeh except only the one fail that is required?
                else:
                    print(f'\nSkipping {x.name}')

    else:
        combined_lists(name, parsim_max_sum, parsim_max_single, local_yamls=args.locallut, mp_cpu_MAX=args.mp_cpu_cores_max)
    return


if __name__ == '__main__':
    main()
