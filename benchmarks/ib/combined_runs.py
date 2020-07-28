# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import importlib.util
import itertools
from benchmarks.ib.test_agents import *
import yaml
import pickle
import yaml


# def combined_analyze(run_name):
#     actions = ['_0/agents/agents_a_velocity.py', '_1/agents/agents_a_gain.py', '_2/agents/agents_a_shift.py']
#     agents = []
#
#     for action in actions:
#         pyfile = Path.cwd() / f'../slurm_runs/{run_name}{action}'
#
#         imported = importlib.util.spec_from_file_location("module.name", pyfile)
#         imp_module = importlib.util.module_from_spec(imported)
#         imported.loader.exec_module(imp_module)
#         agents.append(imp_module.all_agents_more)
#
#     combined = list(itertools.product(agents[0], agents[1], agents[2]))
#     smaller = []
#
#     for row in combined:
#         fitness = sum([x[3] for x in row])
#         parsim = sum([x[2] for x in row])
#         agents = [x[1] for x in row]
#         parsims = [row[0][2], row[1][2], row[2][2]]
#         smaller.append([parsim, fitness, parsims, agents])
#
#     smaller.sort(key=lambda x: x[1], reverse=True)
#     smaller.sort(key=lambda x: x[0])
#     result = {sml[0]: [sml[1], sml[2], sml[3]] for sml in smaller}
#
#     for k, v in result.items():
#         print(f'Combined parsimony {k}({v[0]}) {v[1]}')
#         print(f'{eval_combined_agents(v[2])}')
#         print('\n')
#
#     # name, klasse, parsim, fitness


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

    merged = list(itertools.product(agents[0], agents[1], agents[2]))
    combined_all = []

    for row in merged:
        parsim_sum = float(sum([x[0] for x in row]))
        fitness_sum = float(sum([x[1] for x in row]))
        parsims = [float(row[0][0]), float(row[1][0]), float(row[2][0])]
        codes = [x[3] for x in row]
        combined_all.append([parsim_sum, fitness_sum, parsims, codes])

    combined_best_dict = {}
    for parsim_sum, fitness_sum, parsims, codes in combined_all:
        if parsim_sum in combined_best_dict:
            if fitness_sum < combined_best_dict[parsim_sum][0]:
                print(f'fitness_sum {fitness_sum} < {combined_best_dict[parsim_sum][0]} result[parsim_sum][0]')
                combined_best_dict[parsim_sum] = [fitness_sum, parsims, codes]
        else:
            combined_best_dict[parsim_sum] = [fitness_sum, parsims, codes]

    combined_best = [[parsim_sum, fitness_sum, parsims, codes] for parsim_sum, (fitness_sum, parsims, codes) in combined_best_dict.items()]

    # smaller.sort(key=lambda x: x[1])
    # smaller.sort(key=lambda x: x[0], reverse=True)
    #
    #
    # result = {s[0]: [s[1], s[2], s[3]] for s in smaller}

    analyse_results = []

    try:
        # with Path.open(lut_file, 'rb') as file:
        with Path.open(lut_file, 'r') as file:
            lut = yaml.load(file, Loader=yaml.FullLoader)
    except:
        lut = {}
    eval_count = 0; not_evaled = 0
    for parsim_sum, fitness_sum, parsims, codes in combined_all:  # todo todotodo combined_all or combined_best?

        lut_hash = f'{parsim_sum}_{parsims}_{fitness_sum}'

        if parsim_sum < 30:
            if lut_hash in lut:
                experiment, experiment_save = lut[lut_hash]
            else:
                experiment = float(eval_combined_agents(parsim_sum, parsims, codes))
                experiment_save = float(eval_combined_agents(parsim_sum, parsims, codes, complete=False))
                lut[lut_hash] = [experiment, experiment_save]
                eval_count += 1
                if eval_count % 20 == 0:
                    print(f'Saving lut after evaluating 20')
                    with Path.open(lut_file, 'w') as file:
                        _ = yaml.dump(lut, file, default_flow_style=False, sort_keys=False)

            print(f'Combined parsimony {parsim_sum:3.0f} regr. error: {fitness_sum:.4f} {parsims}\n'
                  f'complete: {experiment}\n'
                  f'save    : {experiment_save}\n')

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
    # ax.plot(nar[0], nar[1], label='all actions')
    # ax.plot(nar[0], nar[2], label='low risk')
    # plt.show()
    # plt.cla()
    # only if one entry per parsimony
    # ax.scatter(nar[0], nar[1], color='r', label='all actions')
    ax.scatter(nar[0], nar[2], color='b', label='low risk')
    plt.show()
    plt.cla()


combined_lists('IB_MSE_sim2')  # todo save the evaluation process as data file
