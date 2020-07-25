# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from benchmarks.ib.ib_eval_agents import *
from pathlib import Path
import importlib.util
import itertools


def combined_analyze(path):

    actions = ['_0/agents/agents_a_velocity.py', '_1/agents/agents_a_gain.py', '_2/agents/agents_a_shift.py']
    agents = []

    for action in actions:
        pyfile = Path(f'{path}{action}')

        imported = importlib.util.spec_from_file_location("module.name", pyfile)
        imp_module = importlib.util.module_from_spec(imported)
        imported.loader.exec_module(imp_module)
        agents.append(imp_module.all_agents_more)

    combined = list(itertools.product(agents[0], agents[1], agents[2]))
    smaller = []

    for row in combined:
        fitness = sum([x[2] for x in row])
        parsim = sum([x[3] for x in row])
        agents = [x[1] for x in row]
        smaller.append([fitness, parsim, agents])
    
    smaller.sort(key=lambda x: x[1], reverse=True)
    smaller.sort(key=lambda x: x[0])
    result = {sml[0]: [sml[1], sml[2]] for sml in smaller}

    for k, v in result.items():
        print(k, v)

    # name, klasse, parsim, fitness


combined_analyze('C:/Users/Rapid/PycharmProjects/plagih/benchmarks/benchmarks/slurm_runs/IB_scratch')
