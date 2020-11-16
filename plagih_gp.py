"""
This starts the whole genetic programming.
This extra file was added to have a file in the root directory that can be started.
"""

from plagih.file_interaction import *
from plagih.plagih_gp_base_class_xai import *
from plagih.plagih_data import *
from benchmarks.ib.combined_runs import *
from benchmarks.mc.agents.quick_eval import *
from pathlib import Path
import sys

import argparse

# import warnings
# warnings.filterwarnings('error')


def load_prepared_run(conf, prepared_run, slurm_runs_folder):

    def pathify(x):
        if x is None:
            return None
        else:
            return Path(__file__).parent.absolute() / 'benchmarks/' / x

    path_origin_tree = None
    action_name = None

    name_splits = prepared_run.split('_')

    """
    ['IB', 'RMSE', 'explun01', 'tanh', 's3m', '1']
    ['MTC75', 'MSE', 'simple']
    """

    if 'IB' == prepared_run[:2]:
        conf.gen_max = 6000   # todo
        root_dir = pathify(f'{slurm_runs_folder}/{prepared_run[:-2]}/{prepared_run}')
        path_data_csv = pathify('ib/gp_files/samples_prepared.csv')
        kernel_name = 'regression bounded'
        ori_trs = {'50_0': 'ib/gp_files//ib_tree_50s_0.csv',
                   '50_1': 'ib/gp_files/ib_tree_50s_1.csv',
                   '50_2': 'ib/gp_files/ib_tree_50s_2.csv',
                   'udluft_0': 'ib/gp_files/ib_tree_udluft_0.csv',
                   'udluft_1': 'ib/gp_files/ib_tree_udluft_1.csv',
                   'udluft_2': 'ib/gp_files/ib_tree_udluflont_2.csv',
                   'mean_0': 'ib/gp_files/ib_tree_mean_0.csv',
                   'mean_1': 'ib/gp_files/ib_tree_mean_1.csv',
                   'mean_2': 'ib/gp_files/ib_tree_mean_2.csv',
                   'sim1_0': 'ib/gp_files/ib_sim1_0.csv',
                   'sim1_1': 'ib/gp_files/ib_sim1_1.csv',
                   'sim1_2': 'ib/gp_files/ib_sim1_2.csv',
                   'sim2_0': 'ib/gp_files/ib_sim2_0.csv',
                   'sim2_1': 'ib/gp_files/ib_sim2_1.csv',
                   'sim2_2': 'ib/gp_files/ib_sim2_2.csv',
                   's3m_0': 'ib/gp_files/ib_s3m_0.csv',
                   's3m_1': 'ib/gp_files/ib_s3m_1.csv',
                   's3m_2': 'ib/gp_files/ib_s3m_2.csv',
                   'scratch': None}
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

    elif 'MTC' in prepared_run:
        conf.gen_max = 5000   # todo
        kernel_name = 'regression bounded discrete'

        root_dir = pathify(f'{slurm_runs_folder}/{prepared_run}')
        num_samples = '200' if 'MTC200' in prepared_run else '75'
        path_data_csv = pathify(f'mc/gp_files/samples{num_samples}.csv')

        ori_trs = {'gpFriendly': 'mc/gp_files/tree_gpFriendly.csv',
                   'gpFriendlyFix': 'mc/gp_files/tree_gpFriendly_fix.csv',
                   'preset': 'mc/gp_files/tree_preset.csv',
                   'presetFix': 'mc/gp_files/tree_preset_fix.csv',
                   'xiao': 'mc/gp_files/tree_xiao.csv',
                   'xiaoFix': 'mc/gp_files/tree_xiaoFix.csv',
                   'simple': 'mc/gp_files/tree_simple.csv',
                   'simpleFix': 'mc/gp_files/tree_simple_fix.csv',
                   'simplePlus': 'mc/gp_files/tree_simplePlus.csv',
                   'simplePlusFix': 'mc/gp_files/tree_simplePlus_fix.csv',
                   'simonBest': 'mc/gp_files/tree(simonBest).csv',
                   'simonBestFix': 'mc/gp_files/tree(simonBest)Fix.csv',
                   'simonBestFix2': 'mc/gp_files/tree(simonBest)Fix2.csv',
                   'simonOkay': 'mc/gp_files/tree_simonOkay.csv',
                   'simonOkayFix': 'mc/gp_files/tree_simonOkayFix.csv',
                   'scratch': None}

        path_origin_tree = pathify(ori_trs[name_splits[-1]])
    else:
        raise

    if 'RMSE' in prepared_run:
        kernel_name += ' RMSE'
    elif 'MSE' in prepared_run:
        kernel_name += ' MSE'
    elif 'MAE' in prepared_run:
        kernel_name += ' MAE'
    else:
        raise Exception(f'No kernel distance measurement found! (In old runs, MAE was automatically used)')
    kernel_name += ' tanhpenalize' if 'tanh' in prepared_run else ''
    kernel_name += ' explun' if 'explun' in prepared_run else ''
    kernel_name += ' explun01' if 'explun01' in prepared_run else ''  # explun: explore-punishment

    print(f'AUTOLOAD: path_origin_tree {path_origin_tree}')
    print(f'AUTOLOAD: kernel_name {kernel_name}')
    conf.kernel_name = kernel_name
    conf.action_name = action_name
    conf.name = prepared_run

    return conf, root_dir, path_data_csv, path_origin_tree


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
    parser.add_argument('-kernel_name', type=str, help='Kernel-name that will be analyzed to load the kernel. Currently only regression-versions.')
    parser.add_argument('-pop_max', '-pop_size', type=int, help='Set maximum pop for this run (updates the config)')
    parser.add_argument('-gen_max', '-gen_size', type=int)
    parser.add_argument('-gen_additionally', '-gen_add', type=int)
    parser.add_argument('-mp_cpu_cores_max', type=int, default=4, help='Maximum amount of cores for parallelisation. Sfeh: set default to max cores? 4 is for my old ass pc. Sorry^^')
    parser.add_argument('-prepared_run', '-config_lookup', '-run_prepared', '-lookup', type=str, help='Handy lookup for quick access to runs that (at least I) currently use a lot')
    # "action='store_true'"-arguments
    parser.add_argument('-analyse', '-analyze', '-analysis', action='store_true', default=None)
    parser.add_argument('-less_files', action='store_true', help='Creates less files by not analysing pareto candidates at the end. -analysis trumps this! (option to save disk space)')
    parser.add_argument('-no_files', action='store_true', help='Not used yet. Create no files. a sfeh wasd-dummy, that stops the program from writing any files whatsoever. Just to be sure.')
    parser.add_argument('-tf_device_log', '-tf_log', action='store_true', help='Logs (A LOT of) tensorflow evaluation feedback. (I recently used this to check if the GPU is actually used)')
    parser.add_argument('-force_new_run', action='store_true')
    parser.add_argument('-print_all', '-debug', '-verbose', action='store_true', help='Print all debug prints (very verbose and helps debugging)')
    parser.add_argument('-pop_kill', action='store_true', help="Force 'killing' the whole population, creating a new generation from scratch, but keeping the paretofront."
                                                                                              " Like a reboot, keeps local optima.")
    parser.add_argument('-testrun', action='store_true', help='SFEH (not used yet): Start a large test run. no origin (scratch) -> restart -> paretoentry as origin, new run -> restart -> analyse')
    parser.add_argument('-developer_fix', action='store_true', help='(Developer only) Flag that can be activated if certain code should be executed. Now used to fix Linux/Windows paths-bug.')
    parser.add_argument('-slurm_runs_folder', type=str, default='slurm_runs', help='sfeh for fore than one version of the same run')
    parser.add_argument('-sfeh_no_crazyops', action='store_true')

    args = parser.parse_args()

    """
    Update the config
    """
    conf = GpConfig(args)

    prepared_run = args.prepared_run
    # self.name = args.name or self.root_dir.resolve().name  # sfeh name? probably there are better names

    if prepared_run:
        conf, root_dir, path_data_csv, path_origin_tree = load_prepared_run(conf, prepared_run, args.slurm_runs_folder)
    else:
        path_data_csv = args.data_csv
        path_origin_tree = args.origin_tree
        try:
            root_dir = args.load_config.parent
        except:
            root_dir = None

    root_dir = args.root_dir or root_dir
    path_make_dir(root_dir)

    """
    Starting the actual run
    """
    gp = ExplainableGP(conf, root_dir, path_data_csv, path_origin_tree, args.mp_cpu_cores_max, developer_fix=args.developer_fix, sfeh_no_crazyops=args.sfeh_no_crazyops)

    if args.analyse:
        if args.force_new_run:
            raise Exception('Dude. Either analyse stuff or force a new run?')
        try:
            gp.backup_load(args.load_backup)
        except FileNotFoundError as no_file_ex:
            raise FileNotFoundError(f'You need to load a backup file to analyse! {no_file_ex}')
        # sfeh idea: track amount of created trees per parsimony? relevant for the pareto front

    else:
        if not args.force_new_run:
            try:
                gp.backup_load(args.load_backup)
            except FileNotFoundError as ex:
                gp.printpl('i', f'No backup file found at {ex}. Starting a new run.')

        if args.pop_kill:
            gp.pop_base = []
            gp.pop_tmp = []
            # sfeh test this!

        gp.plagih_gp_run(args.gen_additionally)

    gp.file_analysis_plots()

    if args.gen_max:
        conf.gen_max = args.gen_max  # workaroung for prepared run

    if args.analyse or not args.less_files:
        gp.analyse_pareto(cpu_cores=args.mp_cpu_cores_max)
    else:
        print_blue('You actively decided not to use analyse out run.\n'
                   'This option was created for distributed cluster evaluation on slurm. The files\n'
                   '1. May be deprecated if the GP process is restarted from here'
                   '2. take a lot of disc space (many images)\n'
                   '3. Need to be computed, after all\n'
                   '4. Computation (Already happened, although not lately ;~D)')

    print('***Program ending***\n********************\n\n')  # repeat 5 time for better view with "tail -n 10 slurm-*"
    sys.exit()


if __name__ == "__main__":
    main()
