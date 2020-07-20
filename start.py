"""
This starts the whole genetic programming.
This extra file was added to have a file in the root directory that can be started.
"""

from pathlib import Path
from plagih import plagih_gp
import sys
import os
import argparse

sys.path.append('plagih/')
sys.path.append('plagih/modules')
# sys.path.append('mountaincar/')


def main(argv):
    """
   -h, -help
   -run_folder
    # sfeh: options for other functions? run, visualise_tree, analyse_run, check_files, tests
   """

    parser = argparse.ArgumentParser(description='Plagih genetic programming (name changes!)')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    # parser.add_argument("--data_dir", type=Path, default=Path(__file__).absolute().parent / "data", help="Path to the data directory",)

    parser.add_argument('-config', type=Path, metavar='CONFIG_YAML', help='The config file in the run directory.')
    parser.add_argument('-config_lookup', type=str, help='The config file is used by sfeh and can be found in a table')
    parser.add_argument('-load_backup', type=Path, help='Starting a run from a backup file (backup.p).')
    parser.add_argument('-out_dir', type=Path, help='A custom output folder (root_dir). Not stable yet.')  # sfeh
    parser.add_argument('-action', type=str, default=None, help='If there is more than one action, choose the right one. (action name)')  # sfeh type=int,
    parser.add_argument('-data_prepared', '-samples_ready', '-samples', type=Path)
    parser.add_argument('-origin_tree', type=Path)
    parser.add_argument('-data_csv', type=Path)
    parser.add_argument('-force_new_run', action='store_true')
    parser.add_argument('-analyse', '-analyze', '-files', '-results', action='store_true', default=None)

    args = parser.parse_args()
    # print(args)

    config_path = args.config
    config_lookup = args.config_lookup
    load_backup = args.load_backup
    out_dir = args.out_dir
    force_new_run = args.force_new_run
    eval_action = args.action
    data_prepared = args.data_prepared
    origin_tree = args.origin_tree
    analyze = args.analyse

    if config_lookup:
        print('using config that is known by sfeh')
        conf_paths = {'ib': 'benchmarks/run_sources/IB/config4ib.yaml',
                      'mtc': 'benchmarks/run_sources/MTC/config4mtc.yaml',
                      'mtc_rel': 'benchmarks/run_sources/MTC/config4mtc_relative.yaml'}

        config_path = Path(__file__).parent.absolute() / conf_paths[config_lookup]

    plagih_root = Path(os.path.dirname(os.path.realpath(__file__)))

    plagih_gp.gp_run(plagih_root, load_backup, config_path, out_dir, force_new_run, eval_action, data_prepared, origin_tree, analyze=analyze)


if __name__ == "__main__":
    main(sys.argv[1:])
