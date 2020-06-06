"""
This starts the whole genetic programming.
This extra file was added to have a file in the root directory that can be started.
"""
from pathlib import Path
from plagih import plagih_gp
import sys
import getopt
import argparse

# sys.root_dir = sys.root_dir / 'plagih'
sys.path.append('plagih/')
sys.path.append('plagih/modules')
# sys.path.append('mountaincar/')


def main(argv):
    """
   -h, -help
   -run_folder
    # sfeh: options for other functions? run, visualise_tree, analyse_run, check_files, tests
   """

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
    # print(args)

    config_path = args.config
    force_new_run = args.force_new_run
    eval_action = args.action
    data_prepared_p = args.data_prepared_p
    origin_tree = args.origin_tree

    if args.analyse:
        plagih_gp.analyse(config_path)
    else:
        plagih_gp.gp_run(config_path, force_new_run, eval_action, data_prepared_p, origin_tree)


if __name__ == "__main__":
    main(sys.argv[1:])
