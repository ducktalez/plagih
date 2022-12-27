"""
This starts the whole genetic programming.
This (extra) file was added to have a file in the root directory that can be started.
"""
from sklearn.model_selection import train_test_split

from plagih.fitness_kernel import RegressionKernel
from plagih.plagih_gp_base_class_xai import *
from plagih.util import *


# def main():  # argv sys.argv[1:]
#     """
#     sfeh:command line only optional!
#     """
#
#     parser = argparse.ArgumentParser(
#         description='Plagih genetic programming (PLAusible Genetic Improvements to Heuristics, name changes!)')
#
#     # Loading files, starting a regular run
#     parser.add_argument('-name', type=str, help='If the run has a name')
#     parser.add_argument('-rootdir', type=Path, help='A custom output folder (rootdir). Not stable yet.')  # sfeh
#     parser.add_argument('-file_backup', type=str, help='rootdir->where the backup file is located')
#     parser.add_argument('-path_data_csv', '-data_csv', type=str, help='rootdir->path of the data (.csv-file)')
#     parser.add_argument('-path_origin', type=str)
#     parser.add_argument('-load_config', type=Path, metavar='CONFIG_YAML', default=None,
#                         help='The config file in the run directory.')
#     parser.add_argument('-load_backup', '-backup', type=str, help='Starting a run from a backup file (backup.p).')
#     parser.add_argument('-prepared_run', '-lookup', type=str,
#                         help='Handy lookup for quick access to runs that (at least I) currently use a lot')
#     parser.add_argument('-force_new_run', action='store_true', help='Shortcut for forcing a new run (->developing)')
#
#     # Parameters for a run which are not in files ;) (Parameters override loaded parameters AFAIK)
#     parser.add_argument('-kernel_name', type=str, help='Kernel-name loaded in run. Currently only regressions.')
#     parser.add_argument('-pop_max', '-pop_size', type=int,
#                         help='Set maximum pop_list for this run (updates the config)')
#     parser.add_argument('-gen_max', '-gen_size', type=int)
#     parser.add_argument('-action_name', '-action', type=str,
#                         help='Specify the .csv column holding the action (output) in the data. '
#                              '(if not clear or more than one action). If empty, the last column is taken.')
#
#     # Restartng a run
#     parser.add_argument('-pop_kill', action='store_true',
#                         help="Kills/deletes the current population, but keeps the paretofront")
#     parser.add_argument('-gen_additionally', '-gen_add', type=int)
#
#     parser.add_argument('-analyze', '-analysis', action='store_true', default=None, help='Analyze run in folder.')
#     parser.add_argument('-print_all', '-debug', '-verbose', action='store_true', help='Print all potential prints.')
#     # parser.add_argument('-print_type', type=str, help='Specifying the print type verbosity')  deprecated
#
#     # computation/runtime: Multicore processing, runtime performance, tensorflow feedback
#     parser.add_argument('-slurm_runs_folder', type=str, default='slurm_runs', help='sfeh:delete')
#     parser.add_argument('-tf_gpu_allow_growth', type=bool, help='sfeh:relevant?')
#     parser.add_argument('-tf_device', default="/gpu:0", help='I hope your GPU nas Nvidia Cuda cores')
#     parser.add_argument('-tf_device_log', type=Path,
#                         help='Logging tensorflow evaluation feedback. (recently: checked if GPU actually used)')
#     parser.add_argument('-mp_cores', type=int, default=4, help='Maximum amount of cores for parallelisation.')
#
#     # For developers only, halting at some code
#     parser.add_argument('-develop', '-dev', action='store_true',
#                         help='Extensive debugging and fintree testing during the developing process.')
#     parser.add_argument('-less_files', action='store_true',
#                         help='Less files (e.g. no pareto analysis), "-analysis" trumps this command')
#
#     parser.add_argument('-test', type=str, help='Continuous tests, start several test runs, reload them, etc...')
#
#     args = parser.parse_args()
#
#     conf = Config(args)  # hyperparameters are in config
#
#     path_origin, path_data_csv = None, None  # sfeh:open
#     if args.prepared_run:
#         rootdir, path_origin, path_data_csv = conf.load_prepared_run(args.prepared_run, args.slurm_runs_folder)
#     else:
#         try:
#             rootdir = args.load_config.parent
#         except:
#             rootdir = None
#     rootdir = path_make_dir(args.rootdir or rootdir)
#
#     kernel = RegressionKernel(path_data_csv, conf)
#     # gp = ExplainableGP(conf, rootdir, kernel, origin_tree)
#
#     if args.analyze:
#         if args.force_new_run:
#             raise Exception('Cannot -analyze and -force_new_run, remove one option.')
#         try:
#             gp.run_backup(path_load_custom_backup=args.load_backup, mode='load')
#         except FileNotFoundError as no_file_ex:
#             raise FileNotFoundError(f'You need to load a backup file to analyze! {no_file_ex}')
#
#     else:
#         if not args.force_new_run:
#             try:
#                 gp.run_backup(path_load_custom_backup=args.load_backup, mode='load')
#             except FileNotFoundError as ex:
#                 gp.printpl('i', f'No backup file found at {ex}. Starting a new run.')
#
#         if args.pop_kill:
#             gp.pop_kill()
#
#         gp.evoloop(args.gen_additionally)
#
#     gp.evoloop_monitoring_plots()
#
#     # if args.gen_max:
#     #     conf.gen_max = args.gen_max  # workaroung for prepared run. sfeh: why workaround?  # delete this
#
#     if args.analyze or not args.less_files:
#         gp.paretofront.analyze_pareto(cpu_cores=args.mp_cores)
#     else:
#         print_blue('You decided not to use analyze a run.\n'
#                    'This option was created for distributed cluster evaluation on slurm. The files\n'
#                    '1. May be deprecated if the GP process is restarted from here'
#                    '2. take a lot of disc space (many images)\n'
#                    '3. Need to be computed, after all\n'
#                    '4. Computation (Already happened, although not lately ;~D)')
#
#     print('***Program ending***\n'
#           '********************\n\n')
#     sys.exit()
#
#     # sfeh: load a file that does approximately the same as this hard coded stuff
#     # if 'IB' == prepared_run[:2]:
#     #     rootdir = Path.cwd() / {prepared_run}')
#     #     path_data_csv = pathify('ib/gp_files/samples_prepared.csv')
#     #     kernel_name = 'regression bounded'
#     #     ori_trees = {'s3m_0': 'ib/gp_files/ib_s3m_0.txt',
#     #                  's3m_1': 'ib/gp_files/ib_s3m_1.txt',
#     #                  's3m_2': 'ib/gp_files/ib_s3m_2.txt',
#     #                  'scratch': None}
#     #     for k, v in ori_trees.items():
#     #         if k in prepared_run:
#     #             print(f'AUTOLOAD: Using origin: {v}')
#     #             path_origin = pathify(v)
#     #
#     #     act_dct = {'_0': 'a_velocity',
#     #                '_1': 'a_gain',
#     #                '_2': 'a_shift'}
#     #     for k, v in act_dct.items():
#     #         if k in prepared_run:
#     #             print(f'AUTOLOAD: Using action: {v}')
#     #             self.action_name = v


def _test_random_pop():
    # ## Give your run a (folder) name
    name = 'MTC200_RMSE_scratch'
    rootdir = Path.cwd() / f'{name}'

    obs_list = ['cartVel', 'cartPos']
    action_name = 'action'

    # ## Load the training data into Kernel-class(...only offline training in this run).
    df = pd.read_csv(Path(__file__).parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv')
    df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P design pattern #YOLO
    data_train, data_control = train_test_split(df, test_size=0.2, random_state=0)
    use_RMSE_vs_MAE_dummy_todo = True  # RMSE
    root_type = float
    action_clip = [0, 2]
    action_round = 0
    kernel = RegressionKernel(use_RMSE_vs_MAE_dummy_todo, data_train, data_control, action_clip, action_round, action_name)

    ### Run/Computation restrictions
    pop_max = 100
    gen_max = 100
    nodeamount_max = 50
    depth_max = 10

    # GP Evolution
    period = {'gen_plots': 5, 'gen_save': 5}
    mp_cores = 1

    complexity_measure = 'tree_node_count'  # 'tree_edit_distance'  # tree_edit_distance / 'tree_node_count'  # sfeh idea todo

    # ### A simple tree and a simple tree with fixed nodes
    # '["Ifte",["<",["cartVel"],["0"]],["0"],["2"]]'
    # '["Ifte:fix",["<",["cartVel"],[0]],["0:fix"],["2:fix"]]'
    # sfeh:open give user feedback for tree
    origin_tree = tree_from_nested_labels('["Ifte",["<",["cartVel"],["0"]],["0"],["2"]]', kernel)
    # origin_tree = None

    # tree construction complexity
    obs_names = ['cartVel', 'cartPos']
    tb = TreeBuilder(obs_names, depth_max, nodeamount_max, root_xtype=root_type)

    period_plots = 10
    period_save = 10
    gp = ExplainableGP(name, pop_max, gen_max, rootdir, kernel, complexity_measure, origin_tree, tb, period_plots,
                       period_save)
    gp.evoloop()
    gp.evoloop_monitoring_plots()

    # try:
    #     gp.run_backup(path_load_custom_backup=load_backup, mode='load')
    # except FileNotFoundError as ex:
    #     gp.printpl('i', f'No backup file found at {ex}. Starting a new run.')
    #
    #     if args.pop_kill:
    #         gp.pop_kill()
    #
    #     gp.evoloop(args.gen_additionally)

    # gp.evoloop_monitoring_plots()
    #
    # # if args.gen_max:
    # #     conf.gen_max = args.gen_max  # workaroung for prepared run. sfeh: why workaround?  # delete this
    #
    # if args.analyze or not args.less_files:
    #     gp.paretofront.analyze_pareto(cpu_cores=args.mp_cores)
    # else:
    #     print_blue('You decided not to use analyze a run.\n'
    #                'This option was created for distributed cluster evaluation on slurm. The files\n'
    #                '1. May be deprecated if the GP process is restarted from here'
    #                '2. take a lot of disc space (many images)\n'
    #                '3. Need to be computed, after all\n'
    #                '4. Computation (Already happened, although not lately ;~D)')
    #
    # print('***Program ending***\n'
    #       '********************\n\n')
    # sys.exit()


# def autotests():
#     # sfeh:open


if __name__ == "__main__":
    """
    GP Workflow:
    1. Load (.csv) data. (gefundene Observationen präsentieren, Aktion präsentieren)
    2. Persönliche Anpassung des Entwicklers (z.B. andere Aktion, Verteilung, print verbosity, ...)
    3. Lauf starten
    """
    # main()
    _test_random_pop()
