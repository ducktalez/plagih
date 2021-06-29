# coding=utf-8
import sys
from pathlib import Path
import argparse
from benchmarks.ib.ib_eval_agents import eval_combined_agents
# from benchmarks.ib.ib_eval_agents import *
import itertools

sys.path.append('../../')
sys.path.insert(1, '../benchnmarks/ib/')
import os
from plagih.plagih_gp_base_class_xai import *
import yaml
import multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plagih.file_interaction import *


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
        return [lut_hash, -20000, -20000, -20000, -20000]  # sfeh


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
            tmp = lut[f"{arow['codes']}"]
            if tmp[0] and tmp[1] and tmp[2] and tmp[3]:
                pass
            else:
                raise Exception('Need to append this, not all 4 required values are contained')
        except:
            if arow['parsim_sum'] <= parsim_MAX and all(x <= parsim_1MAX for x in arow['parsims']):
                open_combinations.append(arow)

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


def plot_best_prediction(rootdir_eval, parsims, combined_all_p, lut_file, parsim_MAX, parsim_1MAX, mp_cpu_MAX):
    """
    plot best guess
    sfeh test RMSE?
    """
    best_regrerr = [min([row for row in combined_all_p[p]], key=lambda x: x['regress_sum']) for p in parsims]
    best_regrerr_dict = eval_and_lut(best_regrerr, parsim_MAX, parsim_1MAX, lut_file, mp_cpu_MAX)

    """
    print the combined runs that belong together
    """
    # relevant_agents = [list(set(xx)) for xx in zip(*[x['parsims'] for x in best_regrerr_dict])]  # delete if not required by sfeh
    bestregr_data = [[' '.join(f'{xx:0.0f}' for xx in x['parsims']), x['experiment'], x['experiment_safe'], x['experiment_r50'], x['experiment_safe_r50']]
                     for x in best_regrerr_dict]
    try:
        yaml_dump(rootdir_eval / 'best_regrerr.yaml', bestregr_data)  # sfeh delete this?
    except:
        # sfeh FFS this FUCKING includes

        with Path.open(rootdir_eval / 'best_regrerr.yaml', 'w') as file:
            _ = yaml.dump(bestregr_data, file, default_flow_style=False, sort_keys=False)
    # yaml_dump(rootdir_eval / 'best_regrerr.yaml', [' '.join(str(xx) for xx in x['parsims']) for x in best_regrerr_dict])  # sfeh delete this?

    """
    okay
    best_regrerr_dict
    """
    best_regrerr_dict.sort(key=lambda x: x['parsim_sum'])
    best_regrerr_dict[0]['cnt'] = 1  # sfeh
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
            cnt.append(len(prows))
            best_row['cnt'] = len(prows)
            res_all.append(best_row)
        else:
            print(f'Not a better entry at {p}')

    # xx = [x['parsim_sum'] for x in res_all]
    # y_all = [y['experiment'] for y in res_all]
    # y_safe = [y['experiment_safe'] for y in res_all]
    # y_all_r50 = [y['experiment_r50'] for y in res_all]
    # y_safe_r50 = [y['experiment_safe_r50'] for y in res_all]
    #
    # """
    # The two plots above in one plot
    # """
    # pyplot_size = (3.6, 2.7)
    # pyplot_rc_tex = {'figure.autolayout': True,
    #                  'text.usetex': True,
    #                  'backend': 'pgf',
    #                  'figure.figsize': pyplot_size,
    #                  }
    #
    # with plt.rc_context(rc=pyplot_rc_tex):
    #     fig, ax = plt.subplots()
    #     ax.set(xlabel='Pareto complexity sum', ylabel='reward', ylim=funny_limits)
    #
    #     plt.yticks(IB_XTICKS[0], IB_XTICKS[1])
    #
    #     ax.plot(xx, y_all, label='all actions', marker='.', color='r')
    #     ax.plot(xx, y_safe, label='low risk', marker='None', color='r', linestyle='dotted')
    #     ax.plot(xx, y_all_r50, label='all actions (randomized)', marker='.', color='b')
    #     ax.plot(xx, y_safe_r50, label='low risk (randomized)', marker='None', color='b', linestyle='dotted')
    #     ax2 = ax.twinx()
    #     ax2.plot(xx, cnt, color='tab:gray', label='Possible combinations', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
    #     ax2.tick_params(axis='y', labelcolor='tab:gray')
    #
    #     ax.legend(loc='lower right')
    #     ax2.legend(loc='lower left')
    #
    #     path_regrallplot = rootdir_eval / f'regression_all.pdf'
    #     fig.savefig(path_regrallplot)
    #     plt.close('all')

    return res_all


def combined_lists(path_main, parsim_MAX, parsim_1MAX, local_yamls=False, cpu_cores=16):
    """
    Make the combined evaluation of industrial benchmark runs.
    Three runs have to be combined from their raw code.
    (I now found a much better way by loading from the backup file, but I am lazy x~~~D)
    """
    main_name = path_main.name
    merge_paretos(path_main)

    if local_yamls:
        lut_file = path_main / 'lutfile.yaml'
    else:
        lut_file = Path(os.path.dirname(os.path.realpath(__file__))) / 'lutfile.yaml'

    agents = []
    for act in ['0', '1', '2']:
        lfile = path_main / f'{main_name}_{act}/pycode_list.yaml'
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
            regress_vals = [float(x[1]) for x in row]
            parsims = [float(row[0][0]), float(row[1][0]), float(row[2][0])]
            codes = [x[3] for x in row]
            combined_all_less.append({'parsim_sum': parsim_sum,
                                      'parsims': parsims,
                                      'experiment': None,
                                      'experiment_safe': None,
                                      'experiment_r50': None,
                                      'experiment_safe_r50': None,
                                      'codes': codes,
                                      'regress_sum': regress_sum,
                                      'regress_vals': regress_vals})
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

    parsims = sorted(list(set([x['parsim_sum'] for x in combined_all_less])))
    """
    the main plotting procedures
    """
    res_all = plot_best_prediction(path_main, parsims, combined_all_p, lut_file, parsim_MAX, parsim_1MAX, cpu_cores)

    return res_all


def merge_paretos(path_main):
    """
    load each IB run and add its paretos front to a merged plot
    """
    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
        for ii, color in enumerate(['blue', 'magenta', 'red']):
            path_combackup = path_main / f'{path_main.name}_{ii}/backup/backup.p'
            gp_backup_data = pickle_load(path_combackup)
            # except Exception:
            #     # sfeh this sucks, IB eval doesnt find it
            #     with Path.open(path_combackup, 'rb') as file:
            #         gp_backup_data = pickle.load(file)

            gen_id, pareto, pop_base, monitor_pd, a_helping_dict = gp_backup_data

            tuples = [[parsim, fitness] for (parsim, fitness, tree) in pareto]
            xx, yy = np.array(tuples).T
            ax.step(xx, yy, linestyle='dotted', marker='.', label=f'action {ii}', where='post')

        ax.set(xlabel='complexity', ylabel='regression error', xlim=(0, None), ylim=(0, None))  # 1.05  # top * 1.05 for better style
        ax.legend(loc='lower left')

        path_paretocombined = path_main / f'pareto_combined.pdf'
        fig.savefig(path_paretocombined)
        plt.close('all')

    print('IB combined runs: merged paretos paretos into one plot!')
    return


def main():
    """
    Custom method to evaluate separate industrial benchmark-runs combined
    """
    parser = argparse.ArgumentParser(description='Plagih IB-Run evaluation')
    parser.add_argument('-mainpath', type=str, help='lol does not work sf', default='IB_MSE_sim2')
    parser.add_argument('-auto', action='store_true')
    parser.add_argument('-locallut', action='store_true')
    parser.add_argument('-mp_cores', type=int,  default=8)
    parser.add_argument('-parsim_max_sum', type=int, default=40)
    parser.add_argument('-parsim_max_single', type=int, default=40)
    args = parser.parse_args()

    mainpath = args.mainpath
    parsim_max_sum = args.parsim_max_sum
    parsim_max_single = args.parsim_max_single

    if args.auto:
        slurm1 = list(Path('benchmarks/slurm_runs/').glob('*'))
        slurm2 = list(Path('benchmarks/slurm_runs2/').glob('*'))
        slurm3 = list(Path('benchmarks/slurm_runs3_easy/').glob('*'))
        rootdirstar = slurm1 + slurm2 + slurm3
        for runfolders in rootdirstar:
            if runfolders.is_dir():
                if runfolders.name[:2] == 'IB':
                    print(f'\nEvaluating {runfolders.name}')
                    try:
                        combined_lists(runfolders, parsim_max_sum, parsim_max_single, local_yamls=args.locallut, cpu_cores=args.mp_cores)
                    except Exception as ex:
                        print(f'Failed evaluation for {runfolders.name}. ignoring. Reason: {ex}')
                        # sfeh except only the one fail that is required?
                else:
                    print(f'\nSkipping {runfolders.name}')

    else:
        combined_lists(mainpath, parsim_max_sum, parsim_max_single, local_yamls=args.locallut, cpu_cores=args.mp_cores)
    return


if __name__ == '__main__':
    main()
