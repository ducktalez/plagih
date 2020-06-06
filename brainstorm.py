"""
Simon's file for brainstorming new stuff
"""
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

def sdf():
    ipath = None
    task = 'run'
    force_new_run = False
    eval_action = 0  # Usually, the dataset only has one action

    try:
        opts, args = getopt.getopt(argv, 'i:o:h', ['ipath=', 'opath=', 'task=', 'force_new_run', 'action='])  # i: -> i requires arg, task= -> task requires arg
    except getopt.GetoptError:
        print('Failed, try: start.py -i <input FOLDER>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('start_run.py -i <input FOLDER>\n'
                  'options:\n'
                  '--task=<run, analyse, tree-latex>')
            sys.exit(2)
        elif opt in ('-i', '--ipath'):
            ipath = Path(arg)
        elif opt in ('-o', '--opath'):
            opath = arg
            print('Your input -o {} is not used'.format(opath))
        elif opt in '--task':
            task = arg
        elif opt in '--force_new_run':
            force_new_run = True
        elif opt in '--action':
            try:
                eval_action = int(arg)
            except Exception as ex:
                print('Could not convert action. Exception, just fyi, is: {}'.format(ex))

    if ipath is None:
        print('No run-folder provided. Starting NOTHING (TODO an example run).\n')
        # ipath = Path.cwd() / example_runs / 'cartpole_v1/'  # / 'plagih'
    # print('Starting plagih-run in {}'.format(Path(run_folder)))

    if task == 'run':
        plagih_gp.gp_run(ipath, force_new_run, eval_action=eval_action)
    elif task == 'analyse':
        plagih_gp.analyse(ipath)
    elif task == 'tree-latex':
        print('Creating a Latex-file from tree (complete tree) csv-file. Can not be used yet. ...')
        # plagih_gp.visualize_labellist(ipath)
    elif task == 'show-default_config':
        plagih_gp.show_default_config(ipath)
    elif task == 'show-default_operators':
        plagih_gp.show_default_operators(ipath)
    else:
        print('Task not known: {}'.format(task))




import argparse

parser = argparse.ArgumentParser(description='Plagih genetic programming (name changes!)')
# parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
# parser.add_argument("--data_dir", type=Path, default=Path(__file__).absolute().parent / "data", help="Path to the data directory",)
# parser.add_argument('-config_file', type=argparse.FileType('r'))


parser.add_argument('-config', type=Path, metavar='CONFIG_YAML', help='The config file in the run directory.')
parser.add_argument('-action', type=int, default=0, help='If there is more than one action, choose the right one.')
parser.add_argument('-data_prepared_p', type=Path)
parser.add_argument('-origin_tree', type=Path)
parser.add_argument('-data_csv', type=Path)
parser.add_argument('-force_new_run', action='store_true')
parser.add_argument('-analyse', '-analyze', '-files', '-results', action='store_true')
args = parser.parse_args()
print(args)





