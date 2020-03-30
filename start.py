"""
This starts the whole genetic programming.
This extra file was added to have a file in the root directory that can be started.
"""
from pathlib import Path
from plagih import plagih_gp
from plagih.modules.file_interaction import example_runs
import sys
import getopt

# sys.root_dir = sys.root_dir / 'plagih'
sys.path.append('plagih/')
sys.path.append('plagih/modules')
sys.path.append('mountaincar/')


def main(argv):
    """
   -h, -help
   -run_folder
   """
    ipath = None
    task = 'run'
    try:
        opts, args = getopt.getopt(argv, 'hi:o:f', ['ipath=', 'opath=', 'task='])
    except getopt.GetoptError:
        print('Failed, try: start.py -i <input FOLDER>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('start_run.py -i <input FOLDER>')  # todo write good + complete text
            sys.exit(2)
        elif opt in ('-i', '--ipath'):
            ipath = Path(arg)
        elif opt in ('-o', '--opath'):
            opath = arg
            print('Your input -o {} is not used'.format(opath))
        elif opt in ('--task'):
            task = arg

    if ipath is None:
        print('No run-folder provided. Starting an example run.\n')
        ipath = Path.cwd() / example_runs / 'cartpole_v1/'  # / 'plagih'
    # print('Starting plagih-run in {}'.format(Path(run_folder)))

    if task == 'run':
        plagih_gp.run(ipath)
    elif task == 'analyze':
        plagih_gp.analyze(ipath)
    elif task == 'tree-latex':
        print('Creating a Latex-file from tree (complete tree, todo) csv-file...')
        # plagih_gp.visualize_labellist(ipath)
    elif task == 'show-default_config':
        plagih_gp.show_default_config(ipath)
    elif task == 'show-default_operators':
        plagih_gp.show_default_operators(ipath)
    else:
        print('Task not known: {}'.format(task))
    # todo: options for other functions? run, visualise_tree, analyse_run, check_files, tests


if __name__ == "__main__":
    main(sys.argv[1:])
