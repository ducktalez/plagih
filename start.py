"""
This starts the whole genetic programming.
This extra file was added to have a file in the root directory that can be started.
"""

from pathlib import Path
from plagih import plagih_gp
import sys
import os
import argparse
from plagih.modules.plagih_config import *


def main():  # argv sys.argv[1:]
    """
   -h, -help
   -run_folder
    # sfeh: options for other functions? run, visualise_tree, analyse_run, check_files, tests
   """

    parser = argparse.ArgumentParser(description='Plagih genetic programming (name changes!)')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    # parser.add_argument("--data_dir", type=Path, default=Path(__file__).absolute().parent / "data", help="Path to the data directory",)

    parser.add_argument('-load_config', '-config', type=Path, metavar='CONFIG_YAML', help='The config file in the run directory.', default=None)
    parser.add_argument('-name', type=str, help='If the run has a name')
    parser.add_argument('-load_backup', '-backup', type=Path, help='Starting a run from a backup file (backup.p).')
    parser.add_argument('-root_dir', '-out_dir', type=Path, help='A custom output folder (root_dir). Not stable yet.')  # sfeh
    parser.add_argument('-action_name', '-eval_action', '-action', type=str, help='If there is more than one action, choose the right one. (action name)')
    parser.add_argument('-data_csv', '-samples_csv', '-data_prepared', '-samples_ready', '-samples', type=Path)
    parser.add_argument('-origin_tree', type=Path)
    parser.add_argument('-analyse', '-analyze', action='store_true')
    parser.add_argument('-kernel_name', type=str, help='Kernel-name that will be analyzed to load the kernel. Currently only regression-versions.')
    parser.add_argument('-pop_max', '-pop_size', type=int)
    parser.add_argument('-gen_max', '-gen_size', type=int)
    parser.add_argument('-gen_additionally', '-gen_add', type=int)
    parser.add_argument('-tf_device_log', '-tf_log', action='store_true', help='Logs tensorflow evaluation feedback. (I recently used this to check if the GPU is actually used)')
    parser.add_argument('-force_new_run', action='store_true')
    parser.add_argument('-print_all', '-debug', action='store_true')
    parser.add_argument('-prepared_run', '-config_lookup', '-run_prepared', '-lookup', type=str, help='Handy lookup for quick access to runs that (at least I) currently use a lot')

    parser.add_argument('-developer_fix', action='store_true', help='(Developer only) Flag that can be activated is certain code should be executed. Now used to fix Linux/Windows paths-bug.')

    args = parser.parse_args()

    """
    Update the config
    """
    conf = GpConfig(args)

    prepared_run = args.prepared_run

    try:
        config_parent_folder = args.load_config.parent
    except:
        config_parent_folder = None
    root_dir = args.root_dir or config_parent_folder or Path.cwd()
    # todo
    # self.name = args.name or self.root_dir.resolve().name  # sfeh probably there are better names
    # self.load_backup_path_loaded_dummy = args.load_backup
    # self.data_csv_path_loaded_dummy = args.data_csv

    print('fuck debug')

    if prepared_run:

        def pathify(x):
            return Path(__file__).parent.absolute() / 'benchmarks/' / x

        path_origin_tree = None
        action_name = None
        root_dir = pathify(f'slurm_runs/{prepared_run[:-2]}/{prepared_run}')

        if 'IB' in prepared_run:
            path_data_csv = pathify('ib/gp_files/samples_prepared.csv')
            kernel_name = 'regression bounded'
            ori_trs = {'50_0': 'ib/gp_files//ib_tree_50s_0.csv',
                       '50_1': 'ib/gp_files/ib_tree_50s_1.csv',
                       '50_2': 'ib/gp_files/ib_tree_50s_2.csv',
                       'udluft_0': 'ib/gp_files/ib_tree_udluft_0.csv',
                       'udluft_1': 'ib/gp_files/ib_tree_udluft_1.csv',
                       'udluft_2': 'ib/gp_files/ib_tree_udluft_2.csv',
                       'mean_0': 'ib/gp_files/ib_tree_mean_0.csv',
                       'mean_1': 'ib/gp_files/ib_tree_mean_1.csv',
                       'mean_2': 'ib/gp_files/ib_tree_mean_2.csv',
                       'sim1_0': 'ib/gp_files/ib_sim1_0.csv',
                       'sim1_1': 'ib/gp_files/ib_sim1_1.csv',
                       'sim1_2': 'ib/gp_files/ib_sim1_2.csv',
                       'sim2_0': 'ib/gp_files/ib_sim2_0.csv',
                       'sim2_1': 'ib/gp_files/ib_sim2_1.csv',
                       'sim2_2': 'ib/gp_files/ib_sim2_2.csv'}
            for k, v in ori_trs.items():
                if k in prepared_run:
                    print(f'AUTOLOAD: Using origin: {v}')
                    path_origin_tree = pathify(v)

            act_dct = {'_0': 'a_velocity',
                       '_1': 'a_gain',
                       '_2': 'a_shift'}
            for k, v in act_dct.items():
                if k in prepared_run:
                    print(f'AUTOLOAD: Using action: {v}')
                    action_name = v

            # conf.gen_max = 2000
            conf.gen_max = 2500

        elif 'MTC' in prepared_run:
            kernel_name = 'regression bounded discrete'

            root_dir = pathify(f'slurm_runs/{prepared_run}')
            if 'MTC200' in prepared_run:
                path_data_csv = pathify('mc/gp_files/samples200.csv')
            elif 'MTC75' in prepared_run:
                path_data_csv = pathify('mc/gp_files/samples75.csv')
            else:
                raise

            ori_trs = {'gpfriendly': 'mc/gp_files/tree_gpFriendly_fix.csv',
                       'preset': 'mc/gp_files/tree_preset_fix.csv',
                       'simple': 'mc/gp_files/tree_simple.csv',
                       'simple_fix': 'mc/gp_files/tree_simple_fix.csv',
                       'simplePlus_fix': 'mc/gp_files/tree_simplePlus_fix.csv',
                       'simplePlus': 'mc/gp_files/tree_simplePlus.csv'}
            for k, v in ori_trs.items():
                if k in prepared_run:
                    print(f'AUTOLOAD: Using origin: {v}')
                    path_origin_tree = pathify(v)
        else:
            raise

        if 'RMSE' in prepared_run:
            kernel_name += ' RMSE'
        elif 'MSE' in prepared_run:
            kernel_name += ' MSE'
        else:
            kernel_name += ' MAE'
        kernel_name += ' tanhpenalize' if 'tanh' in prepared_run else ''
        kernel_name += ' explun' if 'explun' in prepared_run else ''
        kernel_name += ' explun01' if 'explun01' in prepared_run else ''  # explun: explore-punishment

        print(f'AUTOLOAD: path_origin_tree {path_origin_tree}')
        conf.kernel_name = kernel_name
        print(f'AUTOLOAD: kernel_name {kernel_name}')
        conf.action_name = action_name
        conf.name = prepared_run
        conf.gen_max = 1500
    else:
        path_data_csv = args.data_csv
        path_origin_tree = args.origin_tree

    # plagih_root = Path(os.path.dirname(os.path.realpath(__file__)))

    # loaded file-paths are not a good information
    path_load_backup = args.load_backup

    # Not run-specific
    analyse = args.analyse
    gen_additionally = args.gen_additionally
    force_new_run = args.force_new_run  # force_new_run should not be in the config... does not define the run

    # sfeh just for the conclusion...
    # self.origin_tree_path_loaded_dummy = args.origin_tree

    plagih_gp.gp_run(conf, root_dir, path_data_csv, path_origin_tree, path_load_backup, analyse, force_new_run, gen_additionally, developer_fix=args.developer_fix)


if __name__ == "__main__":
    main()
    # sfeh rename start.py to cool name
