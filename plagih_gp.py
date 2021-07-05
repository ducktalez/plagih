"""
This starts the whole genetic programming.
This extra file was added to have a file in the root directory that can be started.
"""
from plagih.plagih_config import Config
from plagih.plagih_gp_base_class_xai import *
import sys
from plagih.util import *

import argparse
from pathlib import Path
import yaml


def main():  # argv sys.argv[1:]
    """
   -h, -help
   -run_folder
    # sfeh: options for other functions? run, visualise_tree, analyze_run, check_files, tests
   """

    parser = argparse.ArgumentParser(description='Plagih genetic programming (name changes!)')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    # parser.add_argument("--data_dir", type=Path, default=Path(__file__).absolute().parent / "data", help="Path to the data directory",)
    # sfeh what is meta-var?
    parser.add_argument('-load_config', '-config', type=Path, metavar='CONFIG_YAML', default=None, help='The config file in the run directory.')
    parser.add_argument('-name', type=str, help='If the run has a name')
    parser.add_argument('-load_backup', '-backup', type=str,
                        help='Starting a run from a backup file (backup.p).')
    parser.add_argument('-action_name', '-action', type=str, help='Specify the .csv column holding the action (output) in the data. '
                                                                  '(if not clear or more than one action). If empty, the last column is taken.')
    # parser.add_argument('-origin_expr', type=str, default='', help='sfeh: open; string version, (expr or nested) directly loading the fintree')
    parser.add_argument('-kernel_name', type=str, help='Kernel-name that will be analyzed to load the kernel. Currently only regression-versions.')
    parser.add_argument('-dc', type=str, action='append', default=[], help='Drop columns from the loaded data-.csv file. Probably unused actions in the IB).')
    parser.add_argument('-pop_max', '-pop_size', type=int, help='Set maximum pop_list for this run (updates the config)')
    parser.add_argument('-gen_max', '-gen_size', type=int)
    parser.add_argument('-gen_additionally', '-gen_add', type=int)
    parser.add_argument('-mp_cores', type=int, default=4, help='Maximum amount of cores for parallelisation. Sfeh: set default to max cores? 4 is for my old ass pc. Sorry^^')
    parser.add_argument('-prepared_run', '-lookup', type=str, help='Handy lookup for quick access to runs that (at least I) currently use a lot')
    parser.add_argument('-analyze', '-analyse', '-analysis', action='store_true', default=None, help='Analyze a loaded run.')
    parser.add_argument('-force_new_run', action='store_true', help='Shortcut for forcing a new run (->developing)')
    parser.add_argument('-print_all', '-debug', '-verbose', action='store_true', help='Print all debug prints (very verbose and helps debugging)')
    parser.add_argument('-pop_kill', action='store_true', help="Kills/deletes the current population, but keeps the paretofront")
    parser.add_argument('-testrun', action='store_true', help='SFEH (not used yet): Start a large test run. no origin (scratch) -> restart -> paretoentry as origin, new run -> restart -> analyze')
    parser.add_argument('-slurm_runs_folder', type=str, default='slurm_runs', help='sfeh for fore than one version of the same run')
    parser.add_argument('-tf_gpu_allow_growth', type=bool, help='I dont know how a GPU can grow, but here you have the option.')
    parser.add_argument('-tf_device', default="/gpu:0", help='I hope your GPU nas Nvidia Cuda cores')
    parser.add_argument('-develop', '-dev', action='store_true', help='Extensive debugging and fintree testing during the developing process.')
    # parser.add_argument('-file_distrib', type=str, help='File with distributions for creating a fintree (maybe use the loaded config file aswell)')

    parser.add_argument('-rootdir', type=Path, help='A custom output folder (rootdir). Not stable yet.')  # sfeh
    parser.add_argument('-tf_device_log', type=Path, help='Logging (A LOT of) tensorflow evaluation feedback. '
                                                          '(I recently used this to check if the GPU is actually used)')
    parser.add_argument('-file_backup', type=str, help='rootdir->where the backup file is located')
    parser.add_argument('-path_data_csv', '-samples_csv', '-data_csv', type=str, help='rootdir->path of the data (.csv-file)')
    parser.add_argument('-path_origin', type=str)
    # paths: check if absolute path exists? separate absolute paths?
    # todo take last column as action instead
    # todo path/str contradiction... do not save real paths in config
    parser.add_argument('-less_files', action='store_true', help='Creating less files (e.g. no pareto analysis), "-analysis" trumps this command')
    parser.add_argument('-no_files', action='store_true', help='SFEH, unused. Create no files. dummy.')

    args = parser.parse_args()
    conf = Config(args)  # Update the config with the possibly loaded input args

    if args.prepared_run:
        rootdir, path_origin, path_data_csv = conf.load_prepared_run(args.prepared_run, args.slurm_runs_folder)
    else:
        try:
            rootdir = args.load_config.parent
        except:
            rootdir = None
    rootdir = path_make_dir(args.rootdir or rootdir)

    gp = ExplainableGP(conf, rootdir, path_origin, path_data_csv, args)
    time_start2 = time.perf_counter()
    gp.printpl('gg', f'Init. Time: {time.perf_counter() - time_start2:4.2f}s')

    if args.analyze:
        if args.force_new_run:
            raise Exception('Dude. Either analyze stuff or force a new run?')
        try:
            gp.backup_load(args.load_backup)
        except FileNotFoundError as no_file_ex:
            raise FileNotFoundError(f'You need to load a backup file to analyze! {no_file_ex}')
        # sfeh idea: track amount of created trees per parsimony? relevant for the paretofront front

    else:
        if not args.force_new_run:
            try:
                gp.backup_load(args.load_backup)
            except FileNotFoundError as ex:
                gp.printpl('i', f'No backup file found at {ex}. Starting a new run.')

        if args.pop_kill:
            gp.pop_base = []
            gp.pop_tmp = []
            # sfeh debug this!

        gp.plagih_gp_run(args.gen_additionally)

    gp.file_analysis_plots()

    if args.gen_max:
        conf.gen_max = args.gen_max  # workaroung for prepared run

    if args.analyze or not args.less_files:
        gp.paretofront.analyze_pareto(cpu_cores=args.mp_cores)
    else:
        print_blue('You actively decided not to use analyze out run.\n'
                   'This option was created for distributed cluster evaluation on slurm. The files\n'
                   '1. May be deprecated if the GP process is restarted from here'
                   '2. take a lot of disc space (many images)\n'
                   '3. Need to be computed, after all\n'
                   '4. Computation (Already happened, although not lately ;~D)')

    print('***Program ending***\n'
          '********************\n\n')
    sys.exit()


if __name__ == "__main__":
    """
    todo Ablauf:
    1. Datensatz laden. (gefundene Observationen präsentieren, Aktion präsentieren)
    2. Persönliche Anpassung des Entwicklers (z.B. andere Aktion, Verteilung, print verbosity, ...)
    3. Lauf starten
    """
    main()
