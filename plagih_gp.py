"""
This starts the whole genetic programming.
This extra file was added to have a file in the root directory that can be started.
"""

from plagih.util import *
from plagih.plagih_gp_base_class_xai import *
from benchmarks.ib.combined_runs import *
from benchmarks.mc.agents.quick_eval import *
from pathlib import Path
import sys

import argparse
import yaml
from pathlib import Path


class Config:
    """
    (just to find this with quick search) self.conf self.config
    """

    def __init__(self, args):
        """
        SFEH ALERT: NEVER try to save the ui/paths. todo. switching between systems is worse than HitlerAIDS
        """
        self.pl_version = 1.1  # must only update if vital changes were made, version important when loading old run
        self.name = args.prepared_run or None  # sfeh

        try:
            with Path.open(args.load_config, 'r') as file:
                conf = yaml.load(file, Loader=yaml.FullLoader)
        except:
            conf = {}
        print_everything = 'wwwwaaaggggiiiff'
        self.print_type = print_everything if args.print_all else conf.get('print_type', print_everything)  # (a)lert, (w)arning, (g)en, (i)nfo, (f)ile written

        # can be updated from everywhere
        self.pop_max = args.pop_max or int(conf.get('pop_max', 1000))  #: 1000,  # amount is never tested
        self.gen_max = args.gen_max or int(conf.get('gen_max', 1000))  # : 1000,  # Maximum amount of generations
        self.action_name = args.action_name or conf.get('action_name', None)
        self.kernel_name = args.kernel_name or conf.get('kernel_name', 'regression')
        self.tree_depth_max = int(conf.get('tree_depth_max', 10))  #: 10,  # maximum Tree depth for entire run
        self.parsimony_max = conf.get('parsimony_max', 35)
        self.tourn_size = int(conf.get('tourn_size', 3))  #: 3,  # [7 per 100] number of trees selected for tournament
        self.dc = conf.get('dc', [])
        self.period = conf.get('period', {'gen_plots': 50, 'gen_save': 25})  # sfeh 10 or 5 for debugging, something higher for actual runs

        self.precision = args.pop_max or conf.get('precision', 6)  # makes the lut more practical - more hits are achieved. be careful with rounding to zero.  # sfeh check this

        # sfeh not here?
        self.evolve_list_random = conf.get('evolve_list_random', None)  # sfeh
        self.complexity_measure = conf.get('complexity_measure', 'tree_edit_distance')  # sfeh check used origin here? backup loaded origin?
        if self.complexity_measure in ['tree_edit_distance']:
            self.complexity_measure = 'tree_node_count'  # sfeh idea
            print_warning('w', "Complexity measurement 'tree_edit_distance' is not possible without origin!\n"
                               "Using 'tree_node_count' instead.", print_type=self.print_type)

        self.action_name = args.action_name or conf.get('action_name', None)  # sfeh type=float or maybe sometimes bool-.-.-
        self.mp_cores = args.mp_cores or conf.get('mp_cores', None)

        self.tf_device_log = args.tf_device_log or conf.get('tf_device_log', False)
        self.tf_gpu_allow_growth = args.tf_gpu_allow_growth or conf.get('tf_gpu_allow_growth', True)
        self.tf_device = args.tf_device or conf.get('tf_device', '/gpu:0')

        self.path_data_csv = args.path_data_csv or conf.get('path_data_csv')
        self.path_origin = args.path_origin or conf.get('path_origin', None)

        # self.parsimony_mean = conf.get('parsimony_mean', 15)  #: 20,  # If you wnt your population to be a certain size
        # self.tree_depth_min = conf.get('tree_depth_min', 1)  #: 2,
        # self.swim = 'p'  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        # self.gen_num_parsim_maxony = conf.get('gen_num_parsim_maxony', 50)  #: 50,  # Increase tmp_parsim to this generation
        # self.name = args.name or self.rootdir.resolve().name  # sfeh name? probably there are better names

        self.file_distrib = (args.file_distrib or 'run_files/distributions_file.yaml')
        self.file_backup = args.file_backup or conf.get('file_backup', 'backup/backup.p')

    def set_rootdir(self, rootdir: Path):
        self.rootdir = rootdir

    def absolute(self, path, make_dirs=False):
        p = self.rootdir / path
        if make_dirs and not p.parent.is_dir():
            p.parent.mkdir(parents=True)
        return p

    def check_update(self):
        """
        Update paretos front
        update population (fitness_train, parsimony, etc)
        also, raise if fitness_train problem
        """
        # oldsize = len(self.paretos)
        # pareto_list = self.paretos[:]
        # self.paretos = []
        # for (parsimony, fitness_train, tree) in pareto_list:
        #     fitness_train = round(fitness_train, self.conf.precision)
        #     tree.meta.fitness_train = round(fitness_train, self.conf.precision)
        #     entry = [parsimony, fitness_train, tree]
        #     self.insert(entry)
        # self.printpl('i', f'Updating paretos front from old run. length: {oldsize}, new paretos length: {len(self.paretos)}')

        # sfeh recompute parsimony and fitness_train for every tree (optional?)
        # also rebuild trees?

        pop_base_copy = self.pop_base[:]
        self.pop_base = []
        for tree in pop_base_copy:
            tree.meta.fitness_train = round(tree.meta.fitness_train, self.precision)
            self.pop_base.append(tree)
        printez('i', f'Updating population from old run. length: {len(self.pop_base)}, new population length: {len(self.pop_base)}', self.print_type)


def load_prepared_run(prepared_run, slurm_runs_folder):

    def pathify(x):
        if x is None:
            return None
        else:
            return Path(__file__).parent.absolute() / 'benchmarks/' / x
    #
    #     self.path_origin = None
    #     self.action_name = None
    #
    #     name_splits = prepared_run.split('_')
    #
    #     self.gen_max = 6000   # sfeh this is not used if >100 generations are without a new paretos entry
    #
    #     # sfeh: load a file that does approximately the same as this hard coded stuff
    #     if 'IB' == prepared_run[:2]:
    #         self.rootdir = pathify(f'{slurm_runs_folder}/{prepared_run[:-2]}/{prepared_run}')
    #         self.path_data_csv = pathify('ib/gp_files/samples_prepared.csv')
    #         kernel_name = 'regression bounded'
    #         ori_trs = {'s3m_0': 'ib/gp_files/ib_s3m_0.csv',
    #                    's3m_1': 'ib/gp_files/ib_s3m_1.csv',
    #                    's3m_2': 'ib/gp_files/ib_s3m_2.csv',
    #                    'scratch': None}
    #         for k, v in ori_trs.items():
    #             if k in prepared_run:
    #                 print(f'AUTOLOAD: Using origin: {v}')
    #                 path_origin = pathify(v)
    #
    #         act_dct = {'_0': 'a_velocity',
    #                    '_1': 'a_gain',
    #                    '_2': 'a_shift'}
    #         for k, v in act_dct.items():
    #             if k in prepared_run:
    #                 print(f'AUTOLOAD: Using action: {v}')
    #                 action_name = v
    #
    #     elif 'MTC' in prepared_run:
    #         kernel_name = 'regression bounded discrete'
    #
    #         self.rootdir = pathify(f'{slurm_runs_folder}/{prepared_run}')
    #         num_samples = '200' if 'MTC200' in prepared_run else '75'
    #         path_data_csv = pathify(f'mc/gp_files/samples{num_samples}.csv')
    #
    #         ori_trs = {'gpFriendly': 'mc/gp_files/tree_gpFriendly.csv',
    #                    'gpFriendlyFix': 'mc/gp_files/tree_gpFriendly_fix.csv',
    #                    'preset': 'mc/gp_files/tree_preset.csv',
    #                    'presetFix': 'mc/gp_files/tree_preset_fix.csv',
    #                    'xiao': 'mc/gp_files/tree_xiao.csv',
    #                    'xiaoFix': 'mc/gp_files/tree_xiaoFix.csv',
    #                    'simple': 'mc/gp_files/tree_simple.csv',
    #                    'simpleFix': 'mc/gp_files/tree_simple_fix.csv',
    #                    'simplePlus': 'mc/gp_files/tree_simplePlus.csv',
    #                    'simplePlusFix': 'mc/gp_files/tree_simplePlus_fix.csv',
    #                    'simonBest': 'mc/gp_files/tree(simonBest).csv',
    #                    'simonBestFix': 'mc/gp_files/tree(simonBest)Fix.csv',
    #                    'simonBestFix2': 'mc/gp_files/tree(simonBest)Fix2.csv',
    #                    'simonOkay': 'mc/gp_files/tree_simonOkay.csv',
    #                    'simonOkayFix': 'mc/gp_files/tree_simonOkayFix.csv',
    #                    'scratch': None}
    #
    #         path_origin = pathify(ori_trs[name_splits[-1]])
    #     else:
    #         raise
    #
    #     if 'RMSE' in prepared_run:
    #         kernel_name += ' RMSE'
    #     elif 'MSE' in prepared_run:
    #         kernel_name += ' MSE'
    #     elif 'MAE' in prepared_run:
    #         kernel_name += ' MAE'
    #     else:
    #         raise Exception(f'No kernel distance measurement found! (In old runs, MAE was automatically used)')
    #     kernel_name += ' tanhpenalize' if 'tanh' in prepared_run else ''
    #     kernel_name += ' explun' if 'explun' in prepared_run else ''
    #     kernel_name += ' explun01' if 'explun01' in prepared_run else ''  # explun: explore-punishment
    #
    #     print(f'AUTOLOAD: path_origin {path_origin}')
    #     print(f'AUTOLOAD: kernel_name {kernel_name}')
    #     self.kernel_name = kernel_name
    #     self.name = prepared_run
    #     # choose_distributions = ChooseConstants(path_distrib=path_distrib, csv_data_samples, n_samples=100)
    return


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

    parser.add_argument('-load_config', '-config', type=Path, metavar='CONFIG_YAML', default=None,
                        help='The config file in the run directory.')
    parser.add_argument('-name', type=str, help='If the run has a name')
    parser.add_argument('-load_backup', '-backup', type=str,
                        help='Starting a run from a backup file (backup.p).')
    parser.add_argument('-rootdir', '-out_dir', type=Path,
                        help='A custom output folder (rootdir). Not stable yet.')  # sfeh
    parser.add_argument('-path_data', type=str,
                        help='the data (.csv-file)')
    parser.add_argument('-action_name', '-action', type=str, default=None,
                        help='Specify the .csv column holding the action (output) in the data (if not clear or more than one action). If empty, the last column is taken.')  # todo tae last one
    parser.add_argument('-path_data_csv', '-samples_csv', '-data_prepared', '-samples_ready', '-samples', type=Path)
    parser.add_argument('-path_origin', type=str)
    # parser.add_argument('-origin_expr', type=str, default='', help='sfeh: open; string version, (expr or nested) directly loading the tree')
    parser.add_argument('-kernel_name', type=str,
                        help='Kernel-name that will be analyzed to load the kernel. Currently only regression-versions.')
    parser.add_argument('-dc', type=str, action='append', default=[],
                        help='Drop columns from the loaded data-.csv file. Probably unused actions in the IB).')
    parser.add_argument('-pop_max', '-pop_size', type=int, help='Set maximum pop_list for this run (updates the config)')
    parser.add_argument('-gen_max', '-gen_size', type=int)
    parser.add_argument('-gen_additionally', '-gen_add', type=int)
    parser.add_argument('-mp_cores', type=int, default=4,
                        help='Maximum amount of cores for parallelisation. Sfeh: set default to max cores? 4 is for my old ass pc. Sorry^^')
    parser.add_argument('-prepared_run', '-config_lookup', '-run_prepared', '-lookup', type=str,
                        help='Handy lookup for quick access to runs that (at least I) currently use a lot')

    parser.add_argument('-analyze', '-analyse', '-analysis', action='store_true', default=None)
    parser.add_argument('-less_files', action='store_true',
                        help='Creates less files by not analysing paretos candidates at the end. -analysis trumps this! (option to save disk space)')
    parser.add_argument('-no_files', action='store_true',
                        help='Not used yet. Create no files. a sfeh wasd-dummy, that stops the program from writing any files whatsoever. Just to be sure.')
    parser.add_argument('-tf_device_log', '-tf_log', action='store_true',
                        help='Logs (A LOT of) tensorflow evaluation feedback. (I recently used this to check if the GPU is actually used)')
    parser.add_argument('-force_new_run', action='store_true')
    parser.add_argument('-print_all', '-debug', '-verbose', action='store_true', help='Print all debug prints (very verbose and helps debugging)')
    parser.add_argument('-pop_kill', action='store_true',
                        help="Force 'killing' the whole population, creating a new generation from scratch, but keeping the paretofront."
                             " Like a reboot, keeps local optima.")
    parser.add_argument('-testrun', action='store_true',
                        help='SFEH (not used yet): Start a large test run. no origin (scratch) -> restart -> paretoentry as origin, new run -> restart -> analyze')
    parser.add_argument('-slurm_runs_folder', type=str, default='slurm_runs', help='sfeh for fore than one version of the same run')
    parser.add_argument('-tf_device_log', action='store_true', help='Logging (a lot of) tensorflow information.')
    parser.add_argument('-tf_gpu_allow_growth', type=bool, help='I dont know how a GPU can grow, but here you have the option.')
    parser.add_argument('-tf_device', default="/gpu:0", help='I hope your GPU nas Nvidia Cuda cores')

    args = parser.parse_args()
    conf = Config(args)  # Update the config with the possibly loaded input args

    if args.prepared_run:
        pass
        # load_prepared_run(args.prepared_run, args.slurm_runs_folder)
    else:
        try:
            rootdir = args.load_config.parent
        except:
            rootdir = None
    rootdir = path_make_dir(args.rootdir or rootdir)

    gp = ExplainableGP(conf)

    if args.analyze:
        if args.force_new_run:
            raise Exception('Dude. Either analyze stuff or force a new run?')
        try:
            gp.backup_load(args.load_backup)
        except FileNotFoundError as no_file_ex:
            raise FileNotFoundError(f'You need to load a backup file to analyze! {no_file_ex}')
        # sfeh idea: track amount of created trees per parsimony? relevant for the paretos front

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

    if args.analyze or not args.less_files:
        gp.pareto.analyze_pareto(cpu_cores=args.mp_cores)
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


def backup_save():
    """
    automatically saves everything important after a certain amount of time
    - save the paretos front (custom_done)
    - save the last generation (custom_done)
    - Save valuable meta-data_csv_path: current generation (custom_done)
    """

    # help_dict = {'self.monitor_evol': self.monitor_evol,
    #              'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh save complete config?    # sfeh i dont think we need the config
    # run_backup_data = self.gen_id, self.paretos, self.pop_base, self.monitor_df, help_dict  # sfeh use this later, help_dict
    # pickle_dump(self.rootdir / self.file_backup, run_backup_data)

    # run_backup_dict = {'self.gen_id': self.gen_id,
    #                    'self.paretos': self.paretos,
    #                    'self.pop_base': self.pop_base,
    #                    'self.monitor_df': self.monitor_df,
    #                    'help_dict': help_dict}
    #
    # path_backupyyy = path_make_dir(self.rootdir / 'backup/backup.yaml')
    # yaml_dump(path_backupyyy, run_backup_dict, self.ui.print_type=self.ui.print_type)

    return


def backup_load(path_load_backup=None):
    """
    If a backup-file is found...
    """
    # path_backup = path_load_backup or self.rootdir / self.file_backup  # sfeh file-load
    #
    # if Path.is_file(path_backup):
    #     self.print_ui('g', f'Loading data from backup-file {path_backup}')
    #     """
    #     Loading the state of the run from the pickle file
    #     """
    #     try:
    #         with Path.open(path_backup, 'rb') as file:
    #             run_data = pickle.load(file)
    #
    #     except NotImplementedError as nimp:
    #         raise Exception(f'NotImplementedError: {nimp}')
    #     except EOFError as eoferr:
    #         raise Exception(f'EOFError: \n{eoferr}')
    #
    #     try:
    #         self.gen_id, self.pareto, self.pop_base, monitor_pd, a_helping_dict = run_data  # sfeh use a helping dictt a_helping_dict is used for a useable sldifjsdfsdfg , a_helping_dict
    #         self.monitor_df = monitor_pd  # sfeh
    #         if 'gens_since_last_pareto' not in self.monitor_df.columns:
    #             self.monitor_df['gens_since_last_pareto'] = np.nan
    #         # self.monitor_evol = a_helping_dict.get('self.monitor_evol') or self.monitor_evol
    #         self.gens_since_last_pareto = a_helping_dict.get('gens_since_last_pareto') or 0  # sfeh
    #     except:
    #         self.gen_id, self.pareto, self.pop_base, m = run_data
    #
    #     self.check_update()
    #     self.print_ui('g', f'Successfully loaded backup file. Generation: {self.gen_id}')
    #
    #     # except Exception as ex:
    #     #     raise Exception(f'Even though a backup exists for this run, it could not be loaded, because of\n{ex}')
    # else:
    #     raise FileNotFoundError(f'No backup-file found at {path_backup}.')
    return


if __name__ == "__main__":
    """
    todo Ablauf:
    1. Datensatz laden. (gefundene Observationen präsentieren, Aktion präsentieren)
    2. Persönliche Anpassung des Entwicklers (z.B. andere Aktion, Verteilung, print verbosity, ...)
    3. Lauf starten
    """
    main()
