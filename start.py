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
    run_folder = None
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('Failed, try: start.py -i <input FOLDER>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('start_run.py -i <input FOLDER>')
            sys.exit(2)
        elif opt in ("-i", "--ifile"):
            run_folder = Path(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            print('Your input -o {} is not used'.format(outputfile))

    if run_folder is None:
        print('No run-folder provided. Starting an example run.')
        run_folder = Path.cwd() / example_runs / 'cartpole_v1/'  # / 'plagih'
    # print('Starting plagih-run in {}'.format(Path(run_folder)))
    plagih_gp.run(run_folder)


if __name__ == "__main__":
    main(sys.argv[1:])
