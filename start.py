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
    parser.add_argument('-load_backup', type=Path, help='Starting a run from a backup file (backup.p).')
    parser.add_argument('-out_dir', type=Path, help='A custom output folder (root_dir). Not stable yet.')  # sfeh
    parser.add_argument('-action', type=str, default=None, help='If there is more than one action, choose the right one. (action name)')  # sfeh type=int,
    parser.add_argument('-action_num', type=int, default=None, help='If there is more than one todo, choose the right one. (action number)')  # sfeh type=int,
    parser.add_argument('-data_prepared', '-samples_ready', '-samples', type=Path)
    parser.add_argument('-origin_tree', type=Path)
    parser.add_argument('-data_csv', type=Path)
    parser.add_argument('-force_new_run', action='store_true')
    parser.add_argument('-analyse', '-analyze', action='store_true', default=None)
    parser.add_argument('-kernel_name', default=None)

    parser.add_argument('-pop_max', '-pop_size', type=int, default=None)

    parser.add_argument('-tf_device_log', '-tf_log', action='store_true', default=False, help='Logs tensorflow evaluation feedback. (I recently used this to check if the GPU is actually used)')

    parser.add_argument('-prepared_run', '-config_lookup', '-run_prepared', type=str, help='Handy lookup for quick access to runs that (at least I) currently use a lot')

    args = parser.parse_args()
    # print(args)

    config_path = args.config
    load_backup = args.load_backup
    out_dir = args.out_dir
    force_new_run = args.force_new_run
    eval_action = args.action
    data_prepared = args.data_prepared
    origin_tree = args.origin_tree
    analyze = args.analyse
    tf_device_log = args.tf_device_log
    kernel_name = args.kernel_name
    pop_max = args.pop_max

    prepared_run = args.prepared_run

    if prepared_run:
        pathy = lambda x: Path(__file__).parent.absolute() / 'benchmarks/run_sources/' / x
        if 'IB' in prepared_run:
            data_prepared = pathy('IB/samples_prepared.csv')
            config_name = 'IB/config4ib'
            ori_trs = {'50_0': 'IB/ib_tree_50s_0.csv',
                       '50_1': 'IB/ib_tree_50s_1.csv',
                       '50_2': 'IB/ib_tree_50s_2.csv',
                       'udluft_0': 'IB/ib_tree_udluft_0.csv',
                       'udluft_1': 'IB/ib_tree_udluft_1.csv',
                       'udluft_2': 'IB/ib_tree_udluft_2.csv',
                       'mean_0': 'IB/ib_tree_mean_0.csv',
                       'mean_1': 'IB/ib_tree_mean_1.csv',
                       'mean_2': 'IB/ib_tree_mean_2.csv',
                       'sim1_0': 'IB/ib_sim1_0.csv',
                       'sim1_1': 'IB/ib_sim1_1.csv',
                       'sim1_2': 'IB/ib_sim1_2.csv'}
            for k, v in ori_trs.items():
                if k in prepared_run:
                    print(f'Using origin: {v}')
                    origin_tree = pathy(v)

            act_dct = {'_0': 'a_velocity',
                       '_1': 'a_gain',
                       '_2': 'a_shift'}
            for k, v in act_dct.items():
                if k in prepared_run:
                    print(f'Using action: {v}')
                    eval_action = v

        elif 'MTC' in prepared_run:
            config_name = 'MTC/config4mtc'
            if 'MTC200' in prepared_run:
                data_prepared = pathy('/MTC/samples200.csv')
            elif 'MTC75' in prepared_run:
                data_prepared = pathy('/MTC/samples75.csv')

            ori_trs = {'gpFfriendly': 'MTC/tree_gpFriendly_fix.csv',
                       'preset': 'MTC/tree_preset_fix.csv',
                       'simple': 'MTC/tree_simple.csv',
                       'simple_fix': 'MTC/tree_simple_fix.csv',
                       'simplePlus_fix': 'MTC/tree_simplePlus_fix.csv',
                       'simplePlus': 'MTC/tree_simplePlus.csv'}
            for k, v in ori_trs.items():
                if k in prepared_run:
                    print(f'Using origin: {v}')
                    origin_tree = pathy(v)
        else:
            raise

        config_name += '_rel' if 'rel' in prepared_run else ''
        config_name += '_tanh' if 'tanh' in prepared_run else ''
        config_path = pathy(f'{config_name}.yaml')
        out_dir = pathy(prepared_run)
        # else: scratch, aka run

    plagih_root = Path(os.path.dirname(os.path.realpath(__file__)))

    plagih_gp.gp_run(plagih_root, load_backup, config_path, out_dir, force_new_run, eval_action, data_prepared, origin_tree, kernel_name, analyze, tf_device_log, pop_max)


if __name__ == "__main__":
    main(sys.argv[1:])
    # todo rename start.py to cool name
