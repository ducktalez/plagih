from pathlib import Path
from plagih.util import *


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
        self.mp_cores = args.mp_cores or conf.get('mp_cores', 1)

        self.tf_device_log = args.tf_device_log or conf.get('tf_device_log', False)
        self.tf_gpu_allow_growth = args.tf_gpu_allow_growth or conf.get('tf_gpu_allow_growth', True)
        self.tf_device = args.tf_device or conf.get('tf_device', '/gpu:0')

        # self.parsimony_mean = conf.get('parsimony_mean', 15)  #: 20,  # If you wnt your population to be a certain size
        # self.tree_depth_min = conf.get('tree_depth_min', 1)  #: 2,
        # self.swim = 'p'  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
        # self.gen_num_parsim_maxony = conf.get('gen_num_parsim_maxony', 50)  #: 50,  # Increase tmp_parsim to this generation
        # self.name = args.name or self.rootdir.resolve().name  # sfeh name? probably there are better names

        # self.file_distrib = (args.file_distrib or 'run_files/distributions_file.yaml')
        # self.file_backup = args.file_backup or conf.get('file_backup', 'backup/backup.p')

        self.path_data_csv = args.path_data_csv or conf.get('path_data_csv', None)
        self.path_origin = args.path_origin or conf.get('path_origin', None)

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

    def load_prepared_run(self, prepared_run, slurm_runs_folder):

        def pathify(x):
            if x is None:
                return None
            else:
                return Path(__file__).parent.absolute() / 'benchmarks/' / x

        name_splits = prepared_run.split('_')
        path_origin = None  # not yet set

        self.gen_max = 6000  # sfeh this is not used if >100 generations are without a new paretos entry

        # sfeh: load a file that does approximately the same as this hard coded stuff
        if 'IB' == prepared_run[:2]:
            rootdir = pathify(f'{slurm_runs_folder}/{prepared_run[:-2]}/{prepared_run}')
            self.path_data_csv = pathify('ib/gp_files/samples_prepared.csv')
            kernel_name = 'regression bounded'
            ori_trs = {'s3m_0': 'ib/gp_files/ib_s3m_0.csv',
                       's3m_1': 'ib/gp_files/ib_s3m_1.csv',
                       's3m_2': 'ib/gp_files/ib_s3m_2.csv',
                       'scratch': None}
            for k, v in ori_trs.items():
                if k in prepared_run:
                    print(f'AUTOLOAD: Using origin: {v}')
                    path_origin = pathify(v)

            act_dct = {'_0': 'a_velocity',
                       '_1': 'a_gain',
                       '_2': 'a_shift'}
            for k, v in act_dct.items():
                if k in prepared_run:
                    print(f'AUTOLOAD: Using action: {v}')
                    self.action_name = v

        elif 'MTC' in prepared_run:
            kernel_name = 'regression bounded discrete'

            rootdir = pathify(f'{slurm_runs_folder}/{prepared_run}')
            num_samples = '200' if 'MTC200' in prepared_run else '75'
            self.path_data_csv = pathify(f'mc/gp_files/samples{num_samples}.csv')

            ori_trs = {
                # sfeh update other trees
                # 'gpFriendly': 'mc/gp_files/tree_gpFriendly.csv',
                # 'gpFriendlyFix': 'mc/gp_files/tree_gpFriendly_fix.csv',
                'preset': 'mc/gp_files/tree_preset.txt',  # todo
                'presetFix': 'mc/gp_files/tree_preset_fix.txt',  # todo
                # 'xiao': 'mc/gp_files/tree_xiao.csv',
                # 'xiaoFix': 'mc/gp_files/tree_xiaoFix.csv',
                'simple': 'mc/gp_files/tree_simple.txt',
                # 'simpleFix': 'mc/gp_files/tree_simple_fix.csv',
                # 'simplePlus': 'mc/gp_files/tree_simplePlus.csv',
                # 'simplePlusFix': 'mc/gp_files/tree_simplePlus_fix.csv',
                # 'simonBest': 'mc/gp_files/tree(simonBest).csv',
                # 'simonBestFix': 'mc/gp_files/tree(simonBest)Fix.csv',
                # 'simonBestFix2': 'mc/gp_files/tree(simonBest)Fix2.csv',
                # 'simonOkay': 'mc/gp_files/tree_simonOkay.csv',
                # 'simonOkayFix': 'mc/gp_files/tree_simonOkayFix.csv',
                'scratch': None
            }

            path_origin = pathify(ori_trs[name_splits[-1]])
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

        print(f'AUTOLOAD: path_origin {path_origin}')
        print(f'AUTOLOAD: kernel_name {kernel_name}')
        self.kernel_name = kernel_name
        self.name = prepared_run
        # choose_distributions = ChooseConstants(path_distrib=path_distrib, csv_data_samples, n_samples=100)

        return rootdir