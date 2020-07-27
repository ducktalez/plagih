# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import importlib.util
import itertools
from benchmarks.ib.test_agents import *
import yaml


def combined_analyze(run_name):
    actions = ['_0/agents/agents_a_velocity.py', '_1/agents/agents_a_gain.py', '_2/agents/agents_a_shift.py']
    agents = []

    for action in actions:
        pyfile = Path.cwd() / f'../slurm_runs/{run_name}{action}'

        imported = importlib.util.spec_from_file_location("module.name", pyfile)
        imp_module = importlib.util.module_from_spec(imported)
        imported.loader.exec_module(imp_module)
        agents.append(imp_module.all_agents_more)

    combined = list(itertools.product(agents[0], agents[1], agents[2]))
    smaller = []

    for row in combined:
        fitness = sum([x[3] for x in row])
        parsim = sum([x[2] for x in row])
        agents = [x[1] for x in row]
        parsims = [row[0][2], row[1][2], row[2][2]]
        smaller.append([parsim, fitness, parsims, agents])

    smaller.sort(key=lambda x: x[1], reverse=True)
    smaller.sort(key=lambda x: x[0])
    result = {sml[0]: [sml[1], sml[2], sml[3]] for sml in smaller}

    for k, v in result.items():
        print(f'Combined parsimony {k}({v[0]}) {v[1]}')
        print(f'{eval_combined_agents(v[2])}')
        print('\n')

    # name, klasse, parsim, fitness


def combined_lists(run_name):
    lsactions = ['_0/pycode_list.yaml', '_1/pycode_list.yaml', '_2/pycode_list.yaml']
    agents = []

    for lsaction in lsactions:
        lsfile = Path.cwd() / f'../slurm_runs/{run_name}{lsaction}'
        with Path.open(lsfile, 'r') as file:
            yamload = yaml.load(file, Loader=yaml.FullLoader)
            agents.append(yamload)

    combined = list(itertools.product(agents[0], agents[1], agents[2]))
    smaller = []

    for row in combined:
        fitness = sum([x[3] for x in row])
        parsim = sum([x[2] for x in row])
        agents = [x[1] for x in row]
        parsims = [row[0][2], row[1][2], row[2][2]]
        smaller.append([parsim, fitness, parsims, agents])

    smaller.sort(key=lambda x: x[1], reverse=True)
    smaller.sort(key=lambda x: x[0])
    result = {sml[0]: [sml[1], sml[2], sml[3]] for sml in smaller}

    for k, v in result.items():
        print(f'Combined parsimony {k}({v[0]}) {v[1]}')
        print(f'{eval_combined_agents(v[2])}')
        print('\n')

    # name, klasse, parsim, fitness


combined_lists('IB_MSE_sim2')
