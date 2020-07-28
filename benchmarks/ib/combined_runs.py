# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import importlib.util
import itertools
from benchmarks.ib.test_agents import *
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

    for lsaction in lsactions:
        lsfile = Path.cwd() / f'../slurm_runs/{run_name}{lsaction}'
        print(f'Looging at file: {lsfile}')
        with Path.open(lsfile, 'r') as file:
            yamload = yaml.load(file, Loader=yaml.FullLoader)
            agents.append(yamload)

    merged = list(itertools.product(agents[0], agents[1], agents[2]))
    combined_all = []

    for row in merged:
        parsim_sum = sum([x[0] for x in row])
        fitness_sum = sum([x[1] for x in row])
        codes = [x[3] for x in row]
        codes2 = [x[3].replace('\\', '') for x in row]
        # for x in codes:
        #     print(x)
        #     x = x.replace('\\', '')
        #     print(x)
        parsims = [row[0][0], row[1][0], row[2][0]]
        combined_all.append([parsim_sum, fitness_sum, parsims, codes])

    # smaller = smaller[:50]  # todo todotodo

    combined_best_dict = {}
    for parsim_sum, fitness_sum, parsims, codes in combined_all:
        if parsim_sum in combined_best_dict:
            if fitness_sum < combined_best_dict[parsim_sum][0]:
                print(f'fitness_sum {fitness_sum} < {combined_best_dict[parsim_sum][0]} result[parsim_sum][0]')
                combined_best_dict[parsim_sum] = [fitness_sum, parsims, codes]
        else:
            combined_best_dict[parsim_sum] = [fitness_sum, parsims, codes]

    combined_best = [[parsim_sum, fitness_sum, parsims, codes] for parsim_sum, (fitness_sum, parsims, codes) in combined_best_dict.items()]

    print(combined_best)

    # smaller.sort(key=lambda x: x[1])
    # smaller.sort(key=lambda x: x[0], reverse=True)
    #
    #
    # result = {s[0]: [s[1], s[2], s[3]] for s in smaller}

    eval_analysis = []

    for parsim_sum, fitness_sum, parsims, codes in combined_best:

        experiment = eval_combined_agents(parsim_sum, parsims, codes)
        experiment_save = eval_combined_agents(parsim_sum, parsims, codes, complete=False)

        print(f'Combined parsimony {parsim_sum:3.0f} regr. error: {fitness_sum:.4f} {parsims}\n'
              f'complete: {experiment}\n'
              f'save    : {experiment_save}\n')

        eval_analysis.append([parsim_sum, experiment, experiment_save])

    nar = np.array(list(zip(*eval_analysis))).T
    plt.plot(nar[0], nar[1])
    plt.show()

combined_lists('IB_MSE_sim2')
