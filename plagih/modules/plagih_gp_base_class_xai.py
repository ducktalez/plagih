"""
'f2f', 'b2b', f = float, b = bool. 'float to bool'

Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""
import time
from plagih.modules.file_interaction import *
from plagih.modules.viz_with_latex import *
import json
import collections.abc
import textwrap
from plagih.modules.plagih_data import *
from plagih.modules.Ptree2 import *
import random

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees
PLAGIH_VERSION = 0.963  # must only update if vital changes were made


class GpConfig():
    # def __init__(self, conf=None):
    #
    #     self.pl_version = PLAGIH_VERSION  # version important when loading old run
    #     self.force_new_run = False  # : False,  # especially for testing, ignores backup files when true
    #     self.time_max = None  # : None,  # int(60 * 60 * 12),  # 60 = 1 min
    #     self.gen_max = 1000  # : 1001,  # Maximum amount of generations
    #
    #     self.pop_max = 1000  # conf['pop_max']  #: 1000,  # amount is never tested
    #     self.tree_depth_max = 10  #: 10,  # maximum Tree depth for entire run
    #     self.tree_depth_min = 1  #: 2,
    #     self.tourn_size = 3  #: 3,  # [7 per 100] number of trees selected for tournament
    #     self.parsimony_mean = 20  #: 20,  # If you wnt your population to be a certain size
    #     self.parsimony_max = 50  #: 50,
    #     self.gen_num_max_parsimony = 50  #: 50,  # Increase tmp_parsim to this generation

    def __init__(self, conf):
        self.pl_version = PLAGIH_VERSION  # version important when loading old run
        self.print_type = conf.get('print_type', 'wwggaiiff')  # all: wwwwaaaggggsiiiivvvff  (a)lert, (w)arning, (g)en, (i)nfo
        self.force_new_run = conf.get('force_new_run', False)  # : False,  # especially for testing, ignores backup files when true
        self.time_max = conf.get('time_max', None)  # : None,  # int(60 * 60 * 12),  # 60 = 1 min
        self.gen_max = conf.get('gen_max', 1000)  # : 1001,  # Maximum amount of generations

        self.pop_max = conf.get('pop_max', 1000)  #: 1000,  # amount is never tested
        self.tree_depth_max = conf.get('tree_depth_max', 10)  #: 10,  # maximum Tree depth for entire run
        self.tree_depth_min = conf.get('tree_depth_min', 1)  #: 2,
        self.tourn_size = conf.get('tourn_size', 3)  #: 3,  # [7 per 100] number of trees selected for tournament
        self.parsimony_mean = conf.get('parsimony_mean', 15)  #: 20,  # If you wnt your population to be a certain size
        self.parsimony_max = conf.get('parsimony_max', 50)  #: 50,
        self.gen_num_max_parsimony = conf.get('gen_num_max_parsimony', 50)  #: 50,  # Increase tmp_parsim to this generation
        self.fitness_decimals = int(conf.get('fitness_decimals', 6))  # rounding the fitness
        self.float_decimals = int(conf.get('float_decimals', 6))  # None or 1-30 decimals

        self.kernel_name = conf.get('kernel_name', 'regression')

        self.plot_verbosity = conf['plot_verbosity']
        self.period = conf['period']

        # sfeh not here?
        self.evolve_list_random = conf.get('evolve_list_random')
        self.lambdadist_as_string = conf.get('lambdadist_as_string')
        self.complexity_measure = conf.get('complexity_measure')


class ExplainableGP(object):
    """

    """

    def __init__(self, plagih_root, root_dir, user_config, action_name, kernel_name=None, path_data=None, opth_operators=None, tf_device_log=False, pop_max=None, path_origin_csv=None):

        self.name = root_dir.resolve().name  # sfeh probably there are better names
        print(f'\n'
              f'\tInitializing Plagih. \n'
              f'\tName: {BColors.CYAN}{self.name}{BColors.RESET}. \n'
              f'\tLocated in: \n'
              f'\t{root_dir}\n')
        self.time_start = time.perf_counter()
        self.root_dir = root_dir
        self.sfeh_plagih_root = plagih_root

        self.config = {
            'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool

            'plot_verbosity': {'gen_fitness_average': 'y',
                               'sympify_errors': 'y',
                               'population_tmp_done-size': 'y',
                               'fitness_variance': 'n'},
            'period': {'time_plots': None,  # in sec
                       'gen_plots': 5,  # in gen counts
                       'time_save': None,  # in sec
                       'gen_save': 5,  # in gen counts
                       'time_analysis': None,  # in gen counts
                       'gen_analysis': 5},  # in gen counts

            'file_locs': {
                'example_runs': 'run_examples/',

                'folder_plots': 'plots/',
                'folder_steps': 'steps/',
                'folder_pop_analysis': 'pop_dist/',
                'folder_histograms': 'agents/',

                'file_backup_pickle': 'backup/backup.p',  # backup-version is set here
                'file_conclusion': 'conclusion.txt',

                # /agents/
                'trees_tex': 'agents/agents_trees.tex',
                'folder_pycode': 'agents/',

                # /info/
                'file_pareto': 'info/paretofront.yaml',
                'info_config_yaml': 'info/config.yaml',
                'file_info_config_json': 'info/config.json',
                'file_info_evolve_dict_yaml': 'info/evolve_list.yaml',
                'info_distributions_yaml': 'info/distributions_file.yaml',
                'env_vars_yaml': 'info/env_vars.yaml',

                # /run_files/
                'samples_ready_p': 'run_files/samples_ready.p',
                'samples_csv': 'run_files/samples.csv',
                'distributions_file': 'run_files/distributions_file.yaml',
            },
            'lambdadist_as_string':
                {'2f': ['lambda: random.normalvariate(0,1)',
                        'lambda: random.normalvariate(1,1)',
                        # 'lambda: random.randint(0, 10)',  # not required
                        ],
                 '2b': ['lambda: random.choice([True, False])'],
                 'observed_floats': 100}
        }

        def update_dict_nested(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dict_nested(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = update_dict_nested(self.config, user_config)  # overwrites the default config-values with user-loaded config

        self.file_locs = self.config['file_locs']

        self.conf = GpConfig(self.config)

        self.conf.pop_max = pop_max if pop_max is not None else self.conf.pop_max

        self.env_vars = EnvVars()
        self.data_train = None
        self.data_control = None

        self.activate_dataset(path_data=path_data, action_name=action_name)

        # Evaluating kernel (that uses tensorflow)
        self.tf_device = "/gpu:0"  # sfeh Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device. Is cpu otherwise
        self.tf_device_log = tf_device_log  # TF device usage logging (for debugging) (default false. I lately used it to check if the GPU is used)
        self.tf_config = tf.compat.v1.ConfigProto(log_device_placement=self.tf_device_log, allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        # sfeh for now, only regression kernel
        self.origin_results = None
        self.kernel = RegressionKernel(kernel_name if kernel_name else self.conf.kernel_name, self.data_train, self.tf_config, self.tf_device, self.env_vars.eval_action)
        self.print_type = self.conf.print_type

        if path_origin_csv:
            self.load_origin_tree(path_origin_csv)
        else:
            self.origin_cooltree: CoolTree = None
            self.pareto = None

        """
        load relevant stuff
        """
        self.activate_distributions(path_distrib=None)
        self.gp_load_oparray(opth_operators=opth_operators)

        """
        initialize some variables
        """
        # init values with dummies (just to have all self values here for overview)
        self.tree_lut = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        # self.parsimony_best_meta = None  # tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.pareto = []  # a dict with all pareto candidates. key is complexity, value is tree meta. [[1,344, meta], ...]
        self.population_tmp = []
        self.pop_base = []  # population that is taken to the next generation
        self.best_fitness = None  # keeps track of the current best fitness
        self.gen_id = 0
        self.custom_done = False  # sfeh
        self.restart_count = 0
        self.time_last_monitor = self.time_start
        self.time_last_backup = self.time_start

        # class MonitoringGenerations:
        #     # todo this might be better. maybe pandas?
        #     def __init__(self):
        #         self.population_tmp_done_size = {}
        #         self.pop_parsim = {}
        #         self.pop_last_evolves = {}
        #         self.fitness_average = {}
        #         self.fitness_variance = {}
        #         self.best_candidate = {}
        #         # 'complexity_list = {},  # not used, just uses memory
        #         self.complexity_average = {}
        #         self.complexity_mean = {}
        #         self.complexity_variance = {}  # variance can be deleted, only std-error is needed delete v1
        #         self.pop_trees_complexity_std_error = {}
        #         self.gen_time = {}

        self.monitoring_dict = {'population_tmp_done-size': {},
                                'pop_parsim': {},
                                'pop_last_evolves': {},
                                'fitness_average': {},
                                'fitness_variance': {},
                                'best_candidate': {},
                                # 'complexity_list': {},  # not used, just uses memory
                                'complexity_average': {},
                                'complexity_mean': {},
                                'complexity_variance': {},  # variance can be deleted, only std-error is needed delete v1
                                'pop:trees:complexity:std_error': {},
                                'gen_time': {}}

        self.print_g('ggg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return

    def make_evolve_rates(self):
        """

        """

        def evolve_safety_update(evolve_list):
            """
            Updates tournament size and evolve rates

            Example entry of the list could be:
            {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
            'custom_params': {'build_spec': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
            """

            for ii, evolve_spec in enumerate(evolve_list):
                tourn_size = evolve_spec.get('tourn_size', self.conf.tourn_size)
                evolve_list[ii]['tourn_size'] = tourn_size

                evolve_rate = evolve_list[ii].get('evolve_rate')
                evolve_list[ii]['evolve_num'] = int(evolve_rate * self.conf.pop_max)
                if evolve_list[ii].get('custom_params') is None:
                    evolve_list[ii]['custom_params'] = {}

            return evolve_list

        try:

            evolve_loop = self.conf.evolve_list
            self.printpl('i', 'Using evolve rates from config')
        except:

            evolve_loop = [
                # Reproduction (10%)
                {'tag': 'Repro', 'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.06, 'custom_params': {}},
                {'tag': 'Rsympy', 'evolve_name': 'reproduce', 'evolve_rate': 0.03, 'custom_params': {'sympify_tree': True}},
                {'tag': 'Pareto', 'evolve_name': 'revive pareto', 'evolve_rate': 0.01, 'custom_params': {}},

                # Mutation (25%)
                {'tag': 'Point', 'evolve_name': 'mutate point', 'evolve_rate': 0.05, 'custom_params': {}},

                # {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                #  'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8), 'full_or_grow': 'full'}}},
                # {'tag': 'BranchDG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                #  'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 5, 1), 'full_or_grow': 'grow'}}},
                {'tag': 'BranchNG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (8, 1, 12, 4), 'full_or_grow': 'full'}}},
                {'tag': 'BranchNG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (8, 1, 12, 4), 'full_or_grow': 'grow'}}},
                {'tag': 'BranchShrink', 'evolve_name': 'mutate branch', 'evolve_rate': 0.0,
                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0), 'full_or_grow': 'grow'}}},

                {'tag': 'FilterB', 'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                 'custom_params': {'mode': 'branch', 'filter_observations': True}},
                {'tag': 'FilterB', 'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                 'custom_params': {'mode': 'branch', 'filter_observations': False}},
                {'tag': 'FilterP', 'evolve_name': 'filter optimize', 'evolve_rate': 0.0, 'tourn_size': 5,
                 'custom_params': {'mode': 'point', 'filter_observations': True}},

                # Crossover (35%)
                {'tag': 'Xover', 'evolve_name': 'crossover branch', 'evolve_rate': 0.35, 'custom_params': {}},  # sum 0.70

                # Leftovers are automatically filled with random trees

                # Random (25%)
                # {'tag': 'Rand1', 'evolve_name': 'random trees', 'evolve_rate': 0.10,
                #  'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 3, 5, 1), 'full_or_grow': 'full'}}},
                # {'tag': 'Rand2', 'evolve_name': 'random trees', 'evolve_rate': 0.10,
                #  'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'full_or_grow': 'grow'}}},
                {'tag': 'Rand3', 'evolve_name': 'random trees', 'evolve_rate': 0.15,
                 'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (20, 5, None, 6), 'full_or_grow': 'grow'}}},
                {'tag': 'Rand4', 'evolve_name': 'random trees', 'evolve_rate': 0.15,
                 'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (20, 5, None, 6), 'full_or_grow': 'full'}}},
            ]

        self.evolve_loop = evolve_safety_update(evolve_loop)

        if self.origin_is_fix():
            try:
                evolve_random = self.conf.evolve_list_random['from_origin']
            except:
                evolve_random = [{'tag': 'RandO3', 'evolve_name': 'random trees', 'evolve_rate': 1.00,
                                  'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (14, 10, 45, 6), 'full_or_grow': 'full'}}}]
                self.conf.evolve_list_random = evolve_random
        else:
            try:
                evolve_random = self.conf.evolve_list_random['from_scratch']
            except:
                evolve_random = [{'tag': 'Rand1', 'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                  'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 4, 5, 1), 'full_or_grow': 'full'}}},
                                 {'tag': 'Rand2', 'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                  'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 6, 1), 'full_or_grow': 'grow'}}},
                                 {'tag': 'Rand3', 'evolve_name': 'random trees', 'evolve_rate': 0.40,
                                  'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (14, 10, None, 6), 'full_or_grow': 'full'}}}
                                 ]
                self.conf.evolve_list_random = evolve_random
        self.evolve_random = evolve_safety_update(evolve_random)

        return

    def file_loc(self, file_key):
        file_loc = self.file_locs.get(file_key)
        if file_loc is None:
            raise KeyError(f'Did not find for file_key {file_key}')
        else:
            return file_loc

    def file_make_dir_root(self, file_key):
        """
        Creates the folder only knowing the filekey
        """
        p = self.root_path(file_key)
        p = file_make_dir(p)
        return p

    def root_path(self, file_key):
        file_loc = self.file_locs.get(file_key)
        if file_loc is None:
            raise KeyError(f'Did not find for file_key {file_key}')
        else:
            return Path(self.root_dir / file_loc)

    def try_load_backup(self, path_backup=None):
        """
        If a backup-file is found...
        """
        path_backup = path_backup if path_backup else self.root_path('file_backup_pickle')  # sfeh file-load

        if Path.is_file(path_backup):
            self.print_g('g', 'Loading data from backup-file...')
            try:
                self.load_backup_pickle(path_backup)
            except Exception as excep:
                raise Exception(f'Even though a backup exists for this run, it could not be loaded because of\n{excep}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}.')

    def gp_analyze(self):
        try:
            self.try_load_backup(path_backup=None)
        except FileNotFoundError as no_file_ex:
            raise FileNotFoundError(f'You need to load a backup file to analyse! {no_file_ex}')
        self.terminate_run(make_a_backup=False)

    def load_backup_pickle(self, path_backup):
        """
        Loading the state of the run from the pickle file
        """

        with Path.open(path_backup, 'rb') as file_backup:
            run_data = pickle.load(file_backup)

        self.conf.pl_version, self.restart_count, self.gen_id, self.pareto, self.pop_base, self.monitoring_dict = run_data
        self.restart_count += 1

        printez('g', f'Starting at generation: {self.gen_id}', self.print_type)

        return

    def pareto_sort(self):
        """
        sorting the pareto entries in list for parsimony
        """
        self.pareto.sort(key=lambda x: x[0])

    def run_backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """

        # sfeh save complete config?
        run_backup_data = self.conf.pl_version, self.restart_count, self.gen_id, self.pareto, self.pop_base, self.monitoring_dict

        path_backup = self.file_make_dir_root('file_backup_pickle')
        with Path.open(path_backup, 'wb') as file:
            pickle.dump(run_backup_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.printpl('f', f'{path_backup.as_posix()}')
        return

    def plagih_gp_run(self):
        """
        regular plagih run
        """

        if not self.conf.force_new_run:
            try:
                self.try_load_backup()
            except FileNotFoundError as fnfex:
                print_warning('w', f'No backup file found. {fnfex}', print_type=self.print_type)
            except Exception:
                raise
            # sfeh: delete old files?

        if not self.origin_exists():
            if self.conf.complexity_measure in ['tree_edit_distance']:  # sfeh get all origin-based distances
                self.conf.complexity_measure = 'tree_node_count'
                print_warning('w', 'Complexity measurement \'tree_edit_distance\' is not possible without origin! Using \'tree_node_count\' instead.', print_type=self.print_type)

        self.write_config_yaml()  # just to see what config is running and what the user can set

        if self.gen_id == 0:
            self.print_g('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')
            self.gen_reset_parameters()
            self.gen_create_initial()

        while self.run_continues():  # max generation, max time, done...
            self.printpl('gggg', f'Evolving Generation {self.gen_id}')
            self.gen_create_loop()
            self.periodical_procedures()
            self.print_g('ggg', f'Generation {self.gen_id} took a total time of: {time.perf_counter() - self.time_genstart:4.2f}. ')
        else:
            printez('g', f'Done after Generation {self.gen_id}.', print_type=self.print_type)

        self.terminate_run()

        return

    def write_config_yaml(self):
        """
        write the parameters to a .csv file which can also be loaded
        """
        # filename = self.root_dir / info_config_yaml
        filename = self.file_make_dir_root('info_config_yaml')
        yaml_dump(filename, self.config, print_type=self.print_type)

        return

    def write_config_json(self):
        """
        write the parameters to a .csv file which can also be loaded
        """

        path = self.file_make_dir_root('file_info_config_json')

        with Path.open(path, 'w') as file:
            json.dump(self.config, file, indent=4)

        return

    def origin_is_fix(self):

        try:
            return self.origin_cooltree.core.is_fix
        except:
            return False

    def gen_create_random(self, amount):
        """
        Add an amount of random trees to the population
        """

    def gen_create_initial(self):
        """
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        if self.origin_exists():

            self.pop_append(self.origin_cooltree, last_evolution='initial')  # sfeh why not :P

        else:
            total_rate = sum([x['evolve_rate'] for x in self.evolve_random])

            for ii, evolve_specs in enumerate(self.evolve_random):
                evolve_num = int(self.conf.pop_max * (evolve_specs['evolve_rate'] / total_rate))
                call_params = evolve_specs.get('custom_params')
                tag = evolve_specs['tag']

                for nn in range(evolve_num):
                    if self.origin_is_fix():
                        new_tree = self.pop_random_from_origin_fix(call_params, self.origin_cooltree)
                    else:
                        new_tree = self.pop_random(call_params)
                    new_cooltree = cooltree_from_oldtree(new_tree)
                    self.pop_append(new_cooltree, last_evolution=tag)

        self.pop_analyse()
        self.pop_base = self.population_tmp[:]
        self.file_population_base_karoo('first')  # first gen only

    def gen_create_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        # All gp creators: name, function, num of trees from tournament selection

        self.gen_id += 1
        self.gen_reset_parameters()

        for ii, evolve_specs in enumerate(self.evolve_loop):  # all selected gp mutations

            time_evolve = time.perf_counter()
            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            call_params = evolve_specs.get('custom_params')
            tag = evolve_specs['tag']

            self.print_g('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')

            if evolve_name == 'reproduce':
                # sfeh one parameter
                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)

                    if call_params.get('sympify_tree'):
                        try:
                            cooltree.evolve_reduce(obs_krazy=self.env_vars.obs_krazy, completely=False)
                        except Exception as ex:
                            print_warning('w', f'not today, pal. {ex}')

                    self.pop_append(cooltree, last_evolution=tag)

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    cooltree.evolve_mutate_point(self.conf.float_decimals,
                                                 self.choose_oparray2,
                                                 self.env_vars.choose_obs,
                                                 self.choose_distributions)
                    self.pop_append(cooltree, last_evolution=tag)

            elif evolve_name == 'mutate branch':

                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    tree = cooltree.get_oldtree()
                    new_tree = self.pop_mutate_branch(call_params, tree)
                    new_cooltree = cooltree_from_oldtree(new_tree)
                    self.pop_append(new_cooltree, last_evolution=tag)

            elif evolve_name == 'crossover branch':
                for nn in range(int(evolve_num / 2)):  # two childs
                    parent_a = self.pop_selection_tournament(tourn_size)
                    parent_b = self.pop_selection_tournament(tourn_size)
                    parent_a = parent_a.get_oldtree()
                    parent_b = parent_b.get_oldtree()
                    child_a, child_b = self.pop_crossover_branch(parent_a, parent_b)
                    self.pop_append(child_a, last_evolution=tag)
                    self.pop_append(child_b, last_evolution=tag)

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    tree = cooltree.get_oldtree()
                    new_tree = self.pop_mutate_filter(call_params, tree)
                    new_cooltree = cooltree_from_oldtree(new_tree)
                    self.pop_append(new_cooltree, last_evolution=tag)

            elif evolve_name == 'revive pareto':

                for nn in range(evolve_num):
                    fitness_train, parsim, cooltree = random.choice(self.pareto)
                    self.pop_append(cooltree, last_evolution=tag)

            elif evolve_name == 'random trees':

                if self.origin_is_fix():
                    for nn in range(evolve_num):
                        new_tree = self.pop_random_from_origin_fix(call_params, self.origin_cooltree)
                        cooltree = cooltree_from_oldtree(new_tree)
                        self.pop_append(cooltree, last_evolution=tag)
                else:
                    for nn in range(evolve_num):
                        new_tree = self.pop_random(call_params)
                        cooltree = cooltree_from_oldtree(new_tree)
                        self.pop_append(cooltree, last_evolution=tag)
            else:
                print_e(f'the specified evolve call is not known: \'{evolve_name}\'')

            self.print_g('ggg', f'->Evolving \'{tag}\' {evolve_num} times took: {time.perf_counter() - time_evolve:4.2f}s.')

        # sfeh automatically fill with random trees
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        self.pop_analyse()
        self.pop_base = self.population_tmp[:]

    def run_continues(self):
        """
        checks if the run can continue
        """
        cond_1 = self.gen_id >= self.conf.gen_max
        cond_2 = False if self.conf.time_max is None else time.perf_counter() - self.time_start > self.conf.time_max
        cond_3 = self.custom_done
        if cond_1 or cond_2 or cond_3:
            return False
        else:
            return True

    def origin_exists(self):
        """
        A dummy method to check if the origin exists.
        This method is rather here to see how often this check is necessary...
        """
        if self.origin_cooltree is not None:
            return True
        else:
            return False

    def periodical_procedures(self, check_plots=None, check_backup=None, check_analysis=None):
        """
        Every few generations, update the created gp_files
        - default is in every generation, but saving every n-th gen or after time passed is possible aswell
        """

        time_now = time.perf_counter()

        def check_show_plots(plots_show):
            if self.conf.period['time_plots']:
                plots_show = self.conf.period['time_plots'] < (time_now - self.time_last_monitor)

            if self.conf.period['gen_plots']:
                plots_show = (self.gen_id % int(self.conf.period['gen_plots'])) == 0

            if plots_show:
                self.time_last_monitor = time_now
                # subfolder = Path(self.file_loc('folder_steps')) / f'Gen-{self.gen_id}'
                self.file_analysis_plots(self.root_dir)

        def check_save_backup(save_run):
            if self.conf.period['time_save']:
                if self.conf.period['time_save'] < (time_now - self.time_last_backup):
                    save_run = True

            if self.conf.period['gen_save']:
                if self.gen_id % int(self.conf.period['gen_save']) == 0:
                    save_run = True

            if save_run:
                self.time_last_backup = time_now
                self.run_backup_save()

        def check_write_analysis(write_analysis):
            if self.conf.period['time_analysis']:
                if self.conf.period['time_analysis'] < (time_now - self.time_last_backup):
                    write_analysis = True

            if self.conf.period['gen_analysis']:
                if self.gen_id % int(self.conf.period['gen_analysis']) == 0:
                    write_analysis = True

            if write_analysis:
                self.file_analysis_complex()
            return

        check_show_plots(check_plots)
        check_save_backup(check_backup)
        check_write_analysis(check_analysis)

        self.printpl('iii', 'Done with auto-procedures')

        return

    def file_pareto_txt(self):
        """
        Save all the pareto efficient candidates to file
        sfeh save as yaml?
        """

        path_pareto = self.file_make_dir_root('file_pareto')

        with Path.open(path_pareto, 'w') as file:
            for (parsim, fitness, cooltree) in self.pareto:
                file.write(f'\nParsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{cooltree.meta.expr_raw}')

        return

    def file_population_base_karoo(self, pop_name):
        """
        Save population_* to disk.

        """
        file_path = file_make_dir(self.root_dir / f'info/population_{pop_name}.csv')
        with Path.open(file_path, 'w', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
            target = csv.writer(csv_file, delimiter=',')
            if self.gen_id != 0:
                target.writerows([''])  # empty row before each generation
            target.writerows(['Plagih GP by Simon Fehrer, inspired by Kai Staats Karoo-gp', 'Generation:', str(self.gen_id)])

            for ii, cooltree in enumerate(self.pop_base):
                target.writerows([''])  # empty row before each Tree
                csv_formatted_tree = cooltree.pretty_format()
                target.writerows([csv_formatted_tree])
                stringtree = [str(cooltree).replace(',', ' ')]
                target.writerows(stringtree)

        return

    def file_analysis_complex(self):
        """
        writes all important gp_files

        """

        self.pareto_sort()

        self.file_pareto_txt()
        self.file_population_base_karoo('last')

        self.file_pareto_histograms()
        self.file_pareto_latex()
        self.file_pareto_pycode()

        return

    def get_path(self, file_key):
        """

        """
        real_path = self.root_path(file_key)
        return real_path

    def activate_dataset(self, path_data, action_name):
        """
        loading the data which the GP will be working on.
        The .csv-file is prepared (loading correct data-type, splitting data, ...)
        and saved as pickle-file for reloading runs.
        This is especially important, as the split in training and test-data must be the same.

        separate loading the prepared data into the main class.
        Why like this? I needed to find a bug in the data_from_csv file and
        did not want to start the whole stuff everytime

            # self.data_train_panda, self.data_control_panda
        """

        if path_data:
            if path_data.suffix == '.p':
                data_prepared = pickle_load(path_data)
            elif path_data.suffix == '.csv':
                data_prepared = data_from_csv(path_data, action_name=action_name)
            else:
                raise FileNotFoundError(f'File nust be a pickle (.p) or csv (.csv) file. Loaded file: {path_data}')

        elif Path.is_file(self.root_path('samples_ready_p')):  # maybe the data was already prepared earlier sfeh load file
            data_prepared = pickle_load(self.root_path('samples_ready_p'))
        elif Path.is_file(self.root_path('samples_csv')):  # Preprocess the raw data: training/test split, env-variables, ...  sfeh load file
            data_prepared = data_from_csv(self.root_path('samples_csv'), action_name=action_name)
            print(f'Prepared the raw {self.file_loc("samples_csv")} behaviour. Saving for next run.')
            pickle_dump(self.root_path('samples_ready_p'), data_prepared)
        else:
            raise FileNotFoundError('No data provided? Please provide data in your config-file(or in your command line call).')

        # dataspec_file = self.root_path / 'run_files/data_specification.yaml'  # sfeh load file is run_files?
        # if dataspec_file.is_file():
        #     pass  # sfeh: if you want to load information from extra file, check for this file here
        # else:
        #     # sfeh env_vars, _, _ = data_prepared. anyways, currently loading info via brackets in .csv-file
        #     pass
        #     # yaml_dump(self.root_path('env_vars_yaml'), data_prepared[0], print_type=self.print_type)

        self.env_vars, self.data_train, self.data_control = data_prepared  # data_control is data_test

        return

    def gp_load_oparray(self, opth_operators=None):
        """

        """

        try:
            operators = yaml_load(Path(opth_operators))

        except:
            print_warning('www', 'Opt-in not specified: Operators-file does not exist. Creating one with a default list of mathematical operators.', print_type=self.print_type)
            operators = [['+', 2],
                         ['-', 1], ['usub', 1],
                         ['*', 2], ['/', 1],
                         ['Square', 0.75], ['**', 0.25],
                         ['abs', 0.4], ['sign', 0.1], ['Round', 0.1],  # sfeh stop chain of arity-1 op in buid method?
                         ['sqrt', 0.1],
                         # ['log', 0.1], ['log1p', 0.1],  # sfeh
                         ['sin', 0.1],  # ['tan', 0.1], ['cos', 0.33], ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
                         ['tanh', 0.2],
                         ['Andb', 1], ['Orb', 1], ['Notb', 0.5],  # ['Xor', 1],
                         ['==', 1], ['!=', 0.5],
                         ['<', 0.5], ['<=', 0.5], ['>', 0.1], ['>=', 0.1],
                         ['Ifte', 2],
                         ['Mini', 1], ['Maxi', 1]]

        def check_allow_closure(operators):
            # sfeh dunno if that works... 2f not in x
            opxtypes = [op[oper]['xtype'] for oper, _ in operators]
            has_2f = any(['2f' in x for x in opxtypes])
            has_2b = any(['2b' in x for x in opxtypes])
            has_f2b = any(['f2b' in x for x in opxtypes])
            has_b2f = any(['b2f' in x for x in opxtypes])
            if not all([has_2f, has_2b, has_f2b, has_b2f]):
                print_warning('w', f'Operators are not complete', print_type=self.print_type)
            if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
                raise Exception(f'Operators do not allow closure')

        check_allow_closure(operators)

        self.choose_oparray2 = oparray_from_list(operators)

        return

    def activate_distributions(self, path_distrib=None):
        """

        """
        path_distrib = path_distrib if path_distrib is not None else self.root_path('distributions_file')

        if Path.is_file(path_distrib):
            lambdadist_as_string = yaml_load(path_distrib)
        else:
            print_warning('www', 'Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.', print_type=self.print_type)
            lambdadist_as_string = self.conf.lambdadist_as_string

        choose_distributions = {'2f': [], '2b': []}

        sample_amount = lambdadist_as_string.get('observed_floats')
        if sample_amount:
            obsnames = self.env_vars.obs_infos.keys()
            obs_samples = self.data_train[obsnames].to_numpy().flatten()

            obs_samples = np.random.choice(obs_samples, size=sample_amount)
            choose_distributions['2f'].extend([lambda: random.choice(obs_samples)]),  # take one

        choose_distributions['2f'].extend([eval(x) for x in lambdadist_as_string['2f']]),
        choose_distributions['2b'].extend([eval(x) for x in lambdadist_as_string['2b']])

        self.choose_distributions = choose_distributions

        return

    def file_pareto_histograms(self):
        """
        Make histograms for all pareto-efficient candidates
        sfeh: based on training data- maybe use test data...

        useful code?
        # histogram_data = np.digitize(histogram_data, bins)  # digitize: the bin index the entry belongs to
        # histogram_data = np.concatenate((histogram_data.reshape(-1, 1), pairwise_fitness.reshape(-1, 1)), axis=1)
        # histogram_data = np.multiply.reduce(histogram_data, axis=1)
        # hist, bins = np.histogram(histogram_data, bins=bins, weights=pairwise_fitness)
        """

        # for ii, (obs_name, obs_info) in enumerate(self.env_vars['obs_name'].items()):
        #     if 'RewardTotal' in obs_name:
        #         continue
        #     obs_min, obs_max = obs_info.get('minmax')
        #     obs_x_info[ii] = {'bins': np.linspace(obs_min, obs_max, 10 + 1)}  # sfeh 32 bins?
        # def histograms_actions(self):
        #
        #     # >>> Histograms for every dimension
        #
        #     data_dims = len(self.env_vars['obs_name'])
        #
        #     # for enum_aii, agent_ii in enumerate(agent_dimatrix):
        #     for enum_aii, (parsim, agent_info) in enumerate(agent_dimatrix.items()):
        #         fig, axs = plt.subplots(data_dims, 1)
        #
        #         pairwise_fitness = agent_info['pairwise_fitness']
        #         for obs_ii, agent_obs_info in agent_info['obs-specific'].items():  # histogram_data, parsim, tf_results['fitness']  -1? -> pairwise_fitness
        #             hist_data = agent_obs_info['histogram_data']
        #             axs[obs_ii].hist(hist_data, bins=obs_x_info[obs_ii]['bins'], weights=pairwise_fitness)
        #             axs[obs_ii].set_title(obs_x_info[obs_ii].get('obs_name'))
        #             axs[obs_ii].set_ylim(top=max_fails_per_bin)
        #
        #         plt.tight_layout()
        #         plt.savefig(path_hist / f'obs_hist_{parsim}.png')
        #         plt.clf()

        path_hist = folder_make_dir(self.root_dir / self.file_loc('folder_histograms'))
        #
        # agent_dimatrix = {}
        # obs_x_info = {}  # [None] * data_dims
        #
        # max_fails_per_bin = len(self.data_train)  # this value will define the y-axis height for all the histograms to look the same

        for (parsim, fitness, cooltree) in self.pareto:

            # pairwise_fitness = tf_results['pairwise_fitness']
            # agent_dimatrix[a_ii] = {}  # 'tf_fitness': None, 'pairwise_fitness': None, 'parsim': parsim
            # agent_dimatrix[a_ii]['tf_fitness'] = tf_results['fitness']
            # agent_dimatrix[a_ii]['pairwise_fitness'] = copy.deepcopy(pairwise_fitness)
            # agent_dimatrix[a_ii]['parsim'] = parsim
            # agent_dimatrix[a_ii]['result-solution'] = copy.deepcopy(deviation_per_action)  # this was: # action_hist_data[a_ii] = copy.deepcopy(kernel_result - solution_goal)

            # for ii, (obs_name, obs_info) in enumerate(self.env_vars['obs_name'].items()):
            #     col = obs_info.get('colpos')
            #     if delete_this_pandas:
            #         histogram_data = self.data_train[:, col]
            #     else:
            #         histogram_data = self.data_train[obs_name]
            #     # hist, _ = np.histogram(histogram_data, bins=16 + 1, weights=pairwise_fitness)
            #     # max_fails_per_bin = max(max(hist), max_fails_per_bin)
            #
            #     # agent_dimatrix[a_ii]['obs-specific'] = {}
            #     # agent_dimatrix[a_ii]['obs-specific'][ii] = histogram_data
            #     # obs_x_info[ii]['obs_name'] = obs_name  # sfeh meh

            # for agent_ii, (parsim, agent_info) in enumerate(agent_dimatrix.items()):  # Histograms for every action

            act_min, act_max = self.env_vars.eval_action.minmax
            act_range = act_max - act_min
            if self.kernel.discrete:  # [0, 1, 2] -> 2
                action_bins = np.linspace(-0.5 - act_range, 0.5 + act_range, 2 * act_range + 1 + 1)  # for +-0.5 and 0
            else:
                num_bins = 16 + 1  # +1 is extra bin for 0
                breite = 0.5 * (act_range * 2) / num_bins
                action_bins = np.linspace(-(breite + act_range), + (breite + act_range), num_bins + 1)  # sfeh 10 bins?

            expr_sym = cooltree.get_expr_sym()
            used_observations = cooltree.get_observation_list()
            pairwise_diff = self.kernel.eval_tf(expr_sym, used_observations)['pairwise_diff']

            fig, ax = plt.subplots()
            ax.hist(pairwise_diff, bins=action_bins, histtype="stepfilled", facecolor="none", edgecolor='k')
            ax.set_ylim(0, len(self.data_train))  # sfeh better size? max?
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Deviation')
            fig.tight_layout()
            plt.savefig(path_hist / f'acthist_{parsim}.png')
            plt.close()
        self.printpl('ff', f'{path_hist.as_posix()}')

    def file_pareto_latex(self):
        """
        Generates latex-file with the computational tree structure of all pareto agents
        - build tree from expression
        - fill tree meta-data, just in case we want to visualise anything of it
        - create latex-forest representation
        """
        latex_element = []

        for (parsim, fitness, cooltree) in self.pareto:

            try:
                # origin_oldtree = self.origin_cooltree.get_oldtree()
                # tree = tree_set_modifyable_nodes(tree, origin_tree=origin_oldtree)
                cooltree.set_fix_nodes(self.origin_cooltree)
            except Exception:
                pass

            cooltree.meta.last_evolution = 'texify'
            tree = cooltree.get_oldtree()
            latex_element.append(f'Pareto entry at parsimony {parsim} with mean Regression Error {fitness}:\n\n')

            forest_viz = latex_tree_get_forest(tree, tight_viz=False)
            latex_element.append(forest_viz)

            latex_element.append('Tight layout:\n')
            # try:
            forest_viz_tight = latex_tree_get_forest(tree)
            latex_element.append(forest_viz_tight)
            # except Exception as tvex:
            #     print_e(f'forest_viz_tight could not be created. {tree_get_labellist(tree)}\nReason: {tvex}')
            #     latex_element.append(f'forest_viz_tight could not be created. {tree_get_labellist(tree)}\nReason: {tvex}')

        latex_full_doc = latex_treeviz_full(latex_element)

        path_trees_tex = self.file_make_dir_root('trees_tex')
        with Path.open(path_trees_tex, 'w') as file:
            file.write(latex_full_doc)

        self.printpl('ff', f'{path_trees_tex.as_posix()}')

        return

    def file_population_base_latex(self):
        """
        """

        latex_element = []

        for ii, cooltree in enumerate(self.pop_base):

            latex_element.append(f'Pop base tree {ii} with Mean Regression Error {cooltree.meta.fitness_train} from last-mod {cooltree.meta.last_evolution}.\n')

            tree = cooltree.get_oldtree()

            forest_viz = latex_tree_get_forest(tree, tight_viz=False)
            latex_element.append(forest_viz)
            latex_element.append('Tight layout:\n')
            try:
                tight_forest_viz = latex_tree_get_forest(tree)
                latex_element.append(tight_forest_viz)
            except:
                print_e(f'tightviz at tree did not work. {ii} labels: {tree_get_labellist(tree)}')

        pop_viz = latex_treeviz_full(latex_element)
        path_tex = self.file_make_dir_root('info/test_pop_latex.tex')
        with Path.open(path_tex, 'w') as file:
            file.write(pop_viz)
        self.printpl('f', f'{path_tex}')

        return

    def file_pareto_pycode(self):
        """
        this auto-generation of real (executable) python files
        is strongly customized for my experiments with Mountaincar and industrial benchmark
        """

        py_return = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')

        assign_input = f"cartPos, cartVel = input\n"

        function_body = textwrap.indent(f"{assign_input}"
                                        "action = {}\n"
                                        f"{py_return}", '    ')  # aka tab (\t)

        complete_function = textwrap.indent(f"def decide(self, input):\n"
                                            f"{function_body}\n", '    ')  # aka tab (\t)

        all_agents = []
        all_agent_names = []
        all_more_info = []

        for (parsim, fitness, cooltree) in self.pareto:
            agent_name = f'{self.name}_{parsim:.0f}'

            agent_as_python = cooltree.get_pycode()
            all_agents.append(f"\nclass {agent_name}:\n{complete_function.format(agent_as_python)}\n")
            all_agent_names.append(agent_name)
            all_more_info.append(f"('{agent_name}', {agent_name}(), {parsim}, {fitness})")

        # all_agents = '\n\n'.join(all_agents)
        agent_tuples = ', '.join([f"('{x}', {x}())" for x in all_agent_names])
        all_more_info = ', '.join(all_more_info)
        pyc_complete = f"import math; import numpy as np\n" \
            f"{all_agents}\n" \
            f"all_agents_more = [{all_more_info}]\n" \
            f"agent_tuples = [{agent_tuples}]\n"

        pth = file_make_dir(self.root_path('folder_pycode') / f"agents_{self.env_vars.eval_action.name}.py")
        with Path.open(pth, 'w') as file:
            file.write(pyc_complete)
            self.printpl('ff', f'{pth.as_posix()}')

        return

    def origin_tree_get(self):
        """
        Safely return an origin_meta tree
        """
        if self.origin_exists():
            origin_cooltree = copy.deepcopy(self.origin_cooltree)
        else:
            origin_cooltree = None
        return origin_cooltree

    def treelut_tree_add(self, cooltree: CoolTree):
        """
        update selected values in self.tree_meta
        LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        """

        meta = {'fitness_train': cooltree.meta.fitness_train,
                'parsimony': cooltree.meta.parsimony,
                'expr_raw': cooltree.meta.expr_raw,
                'expr_sym': cooltree.meta.expr_sym}

        tree_ident = hash(cooltree)

        self.tree_lut[tree_ident] = meta
        return

    def gen_reset_parameters(self):
        """
        Sets the parameters for the generation
        - reset population_tmp_done
        - Linearly increase threshold for parsimony
        """

        self.time_genstart = time.perf_counter()
        self.population_tmp = []
        # self.parsimony_tmp = max(1 / min(self.gen_id, self.conf.gen_num_max_parsimony']) * self.parsimony_max, self.parsimony_max)

        return

    def pop_mutate_filter(self, call_params, tree):
        """
        Mutates a number of float terminal of a tree
        """
        mode = call_params['mode']  # point/branch/all
        filter_observations = call_params.get('filter_observations')  # point/branch/all
        mutate_filter = 'gaussian_filter'  # sfeh change?

        node_ids = tree_get_mutatable_nodes(tree)
        if mode == 'branch':
            node_id = random.choice(node_ids)  # sfeh should this be completely random?
            node_ids = tree_node_get_branch(tree, node_id)  # select the whole branch

        float_nodes = []
        obs_nodes = []
        for node_id in node_ids:
            if tree_node_get_xtype(tree, node_id) == '2f':
                try:
                    _ = float(tree_node_get_label(tree, node_id))
                    float_nodes.append(node_id)
                except ValueError:
                    obs_nodes.append(node_id)

        if mode == 'point':  # if poinemutation, return one nodeid as list
            if filter_observations:
                filter_id = [random.choice(float_nodes + obs_nodes)]
                if filter_id in float_nodes:
                    float_nodes = filter_id
                else:
                    obs_nodes = filter_id
            else:
                float_nodes = [random.choice(float_nodes)]

        if float_nodes:
            for node_id in float_nodes:
                val = float(tree_node_get_label(tree, node_id))
                val = label_constant_mutate(val, term_type=float, filter_type=mutate_filter, float_decimals=self.conf.float_decimals)
                tree = tree_node_set_label(tree, node_id, val)

        if obs_nodes and filter_observations:  # 'filtering' variables when they are from different times
            for nid in obs_nodes:
                obs_label = tree_node_get_label(tree, nid)

                is_negative = obs_label[0] == '-'  # workaround for negative labels
                if is_negative:
                    obs_label = obs_label[1:]

                hello_node = self.env_vars.obs_infos[obs_label]
                hello_node.filter_new_index()
                obs_label = hello_node.name

                new_obs = '-' + obs_label if is_negative else obs_label
                tree = tree_node_set_label(tree, nid, new_obs)
        else:
            print_warning('wwww', 'Tree does not seem to have any nodes for filtering.', print_type=self.print_type)  # usually happens with point-filtering
            pass

        return tree

    def invent_label_list(self, size_mode, first_xtype, build_size, full_or_grow, float_decimals):
        """
        Creates a random label list
        """
        if 'depth' in size_mode:
            # sfeh warning: Attention with this one. can get quite large with depth based
            label_list, arity_list, xtype_list = invent_label_list_depth(first_xtype, build_size, float_decimals,
                                                                         self.env_vars.choose_obs, self.env_vars.obs_krazy, self.choose_oparray2,
                                                                         self.choose_distributions, full_or_grow=full_or_grow)

        elif 'nodes' in size_mode:

            label_list, arity_list, xtype_list = invent_label_list_nodes(first_xtype, build_size, float_decimals,
                                                                         self.env_vars.choose_obs, self.env_vars.obs_krazy, self.choose_oparray2,
                                                                         self.choose_distributions, full_or_grow=full_or_grow)
        else:
            raise Exception('Known full_or_grow was not found for building random trees.')
        return label_list, arity_list, xtype_list

    def helper_evolve_params_branch(self, call_params):
        """

        """
        build_spec = call_params.get('build_spec')

        size_mode = build_spec['size_mode']

        mean_min_max_var = build_spec.get('mean_min_max_var')  # (base, min, max, normal_distrib)
        mean_min_max_var = list(mean_min_max_var)
        if 'depth' in size_mode:
            max_dummy = self.conf.tree_depth_max
        elif 'nodes' in size_mode:
            max_dummy = self.conf.parsimony_max
        else:
            raise
        if mean_min_max_var[2] is None:
            mean_min_max_var[2] = max_dummy
        else:
            mean_min_max_var[2] = min(mean_min_max_var[2], self.conf.parsimony_max)
        mean_min_max_var = tuple(mean_min_max_var)

        full_or_grow = build_spec['full_or_grow']

        return build_spec, size_mode, mean_min_max_var, full_or_grow

    def pop_random_from_origin_fix(self, call_params, origin_cooltree):
        """
        insert a (random) number of branches at the first possible "layer"
        (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
        - get these nodes, randomly choose a subset of those
        - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
        - split the amount of nodes up (randomly) and add these new branches to the tree
        """

        build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)

        # tree_base = tree.copy()
        # ('We are about to create new branches randomly at nodes {}.'.formjat(layer0_ids))
        origin_tree = origin_cooltree.get_oldtree()
        layer0_ids = tree_get_mutatable_layer(origin_tree, 0)

        build_split = []
        if 'depth' in size_mode:
            for ii in range(len(layer0_ids)):
                build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')
                build_split.append(build_size)

        elif 'nodes' in size_mode:
            build_nodes = choose_build_size(size_mode, mean_min_max_var, force='branch')
            build_split = randomly_split_range(build_nodes, len(layer0_ids))
        else:
            raise

        tree = origin_tree.copy()
        for i in range(len(layer0_ids)):  # insert branches! get layer every time (node ids might have changed)
            layer0_ids = tree_get_mutatable_layer_lv0(tree)
            node_id = layer0_ids[i]
            first_xtype = tree_node_get_xtype(tree, node_id)
            old_branch = tree_node_get_branch(tree, node_id, karoo=True)
            build_size = build_split[i]

            label_list, arity_list, xtype_list = self.invent_label_list(size_mode, first_xtype, build_size, full_or_grow, self.conf.float_decimals)

            c_core = Core_From_Labels(label_list, arity_list, xtype_list)
            core = c_core.get_uninstanced_core()
            tree = tree_insert_subtree(tree, core, old_branch, karoo=True)

        return tree

    def pop_random(self, call_params):
        """
        Creates completely random trees from scratch
        """

        build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)
        action_xtype = self.env_vars.eval_action.xtype
        build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')

        label_list, arity_list, xtype_list = self.invent_label_list(size_mode, action_xtype, build_size, full_or_grow, self.conf.float_decimals)

        p_tree = Ptree_karoo(label_list, xtype_list, arity_list=arity_list)
        tree = p_tree.get_uninstanced_tree()

        return tree

    def pop_mutate_branch(self, call_params, tree):

        """
        Mutate branch of a tree.

        """

        build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)

        full_or_grow = build_spec.get('full_or_grow')
        if full_or_grow is None:
            full_or_grow = random.choice(['full', 'grow'])
        node_ids = tree_get_mutatable_nodes(tree, no_root=True)
        old_node = random.choice(node_ids)
        old_xtype = tree_node_get_xtype(tree, old_node)
        build_size = choose_build_size(size_mode, mean_min_max_var, tree=tree, node_id=old_node)

        label_list, arity_list, xtype_list = self.invent_label_list(size_mode, old_xtype, build_size, full_or_grow, self.conf.float_decimals)

        if label_list:
            # core_insert = core_from_labels(label_list, arity_list, xtype_list)

            c_core = Core_From_Labels(label_list, arity_list, xtype_list)
            core_insert = c_core.get_uninstanced_core()

            branch_nodes_ids = tree_node_get_branch(tree, old_node, karoo=True)
            tree = tree_insert_subtree(tree, core_insert, branch_nodes_ids, karoo=True)
            tree = tree_prune_depth(tree, self.conf.tree_depth_max, self.env_vars.obs_krazy, self.env_vars.choose_obs, self.choose_distributions, self.conf.float_decimals)
        else:
            tree = None

        return tree

    def pop_crossover_branch(self, left_tree, right_tree):
        """
        swap branches of two trees
        - select parent a and b
        - select swappable branche for a_parent from b_parent
            - select a node in a (and crossover here, no matter what)
        - delete a_parent branch and insert b_parent branch (which tactic?)

        """

        # 1. two parents

        # 2. search nodes for left and right that can be exchanged. convert_needed
        left_id, right_id, success = tree_try_get_swapids(left_tree, right_tree)
        if not success:
            right_id, left_id, success = tree_try_get_swapids(right_tree, left_tree)

        left_ids, left_labels, left_aritys, left_xtypes = tree_get_branch_ilax(left_tree, left_id)
        right_ids, right_labels, right_aritys, right_xtypes = tree_get_branch_ilax(right_tree, right_id)

        if not success:
            print_warning('ww', f'Crossover conversion between trees not possible: \n{left_tree}\n{right_tree}', print_type=self.print_type)
            return None, None

        left_core = Core_From_Labels(left_labels, left_aritys, left_xtypes).get_uninstanced_core()
        right_core = Core_From_Labels(right_labels, right_aritys, right_xtypes).get_uninstanced_core()

        left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
        left_offspring = tree_prune_depth(left_offspring, self.conf.tree_depth_max, self.env_vars.obs_krazy, self.env_vars.choose_obs, self.choose_distributions, self.conf.float_decimals)

        right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
        right_offspring = tree_prune_depth(right_offspring, self.conf.tree_depth_max, self.env_vars.obs_krazy, self.env_vars.choose_obs, self.choose_distributions, self.conf.float_decimals)

        left_offspring = cooltree_from_oldtree(left_offspring)
        right_offspring = cooltree_from_oldtree(right_offspring)
        return left_offspring, right_offspring

    def update_pareto(self, cooltree: CoolTree):
        """
        inserts a tree into the pareto front
        """

        parsimony = cooltree.meta.parsimony
        expr_raw = cooltree.meta.expr_raw

        fitness_train = cooltree.meta.fitness_train
        tree_entry = [parsimony, fitness_train, cooltree]

        if len(self.pareto) == 0:  # no pareto entries found yet
            self.printpl('a', f"Paretofront inserted first candidate at {parsimony, fitness_train, expr_raw}")
            self.pareto.append(tree_entry)  # e.g. [3, 423, cooltree]
        else:
            try:
                p_simpler = [p for p in self.pareto if p[0] <= tree_entry[0]]  # all pareto entries that are less complex
                best = min(p_simpler, key=lambda p: p[1])  # the fittest of the less complex ones
            except:  # catches, when p_simpler is empty
                best = min(self.pareto, key=lambda p: p[1])  # fittest pareto entry - maybe the new entry is very complex, but the best

            if tree_entry[1] < best[1]:  # if true, at least one insertion
                self.printpl('a', f"Paretofront new entry was inserted: {parsimony, fitness_train, expr_raw}")
                self.pareto.append(tree_entry)  #
                self.pareto = [x for x in self.pareto[:] if x[0] < tree_entry[0] or x[1] < tree_entry[1] or x is tree_entry]
                self.pareto_sort()  # as far as I can tell, not really necessary without using iter()

                # simplify the tree ans save in pareto once again
                self.printpl('aaa', 'Trying to simplify for pareto entry.')
                cooltree_sym = copy.deepcopy(cooltree)
                try:
                    cooltree_sym.evolve_reduce(obs_krazy=self.env_vars.obs_krazy, completely=True)
                except Exception as ex:
                    print_warning('ww', f'Tree sympification did not work: {ex}', print_type=self.print_type)

                if len(cooltree_sym) < len(cooltree):
                    sym_fitness = self.tree_eval_fitness_train(cooltree_sym)
                    if sym_fitness != cooltree.meta.fitness_train and TEST_PHASE:
                        print_e(f'Fitness of a sympified tree is different! sym: {sym_fitness}, before: {cooltree.meta.fitness_train}\n'
                                f'tree: {cooltree.core.get_labellist()}\n'
                                f'symp: {cooltree_sym.core.get_labellist()}\n'
                                f'tree raw: {cooltree.get_expr_raw()}\n'
                                f'symp rawr: {cooltree_sym.get_expr_raw()}')
                        # raise Exception('FUCK')
                        return

                    self.printpl('aa', 'Successfully reduced pareto tree!')
                    cooltree_sym.meta.fitness_train = sym_fitness
                    self.update_pareto(cooltree_sym)
                else:
                    self.printpl('aaa', 'Pareto entry was already simplified')
            else:
                return
        self.pareto_sort()
        return

    def pop_append(self, cooltree: CoolTree, last_evolution=''):
        """
        Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the tree is refurbished.
        sfeh: if trees are 100% safely created, tree_check_deep() must not be used. Useful when trying out new gp-operators.
        - Enrich the raw tree for the next generation
        - check if the tree is actually valid
        ->
        """
        # sfeh this check might be important...

        try:
            cooltree.check_all()
        except Exception as ex:
            print_warning('w', f'tree failed the quick check. last-mod: {last_evolution}. Reason:\n{ex}', print_type=self.print_type)
            return

        # tree = self.tree_finish_nodes(tree, last_evolution=last_evolution)

        tree_ident = hash(cooltree)
        if tree_ident in self.tree_lut:
            tree_meta = self.tree_lut[tree_ident]
            parsimony = tree_meta['parsimony']
            fitness_train = tree_meta['fitness_train']
        else:
            parsimony = tree_eval_parsimony(cooltree, self.conf.complexity_measure, origin_cooltree=self.origin_cooltree)
            if parsimony > self.conf.parsimony_max:
                print_warning('wwww', f'Parsimony too high, last evolution: {last_evolution}', print_type=self.print_type)
                return

            try:
                fitness_train = self.tree_eval_fitness_train(cooltree)
            except Exception as evalex:
                print_warning('www', f'Exception while evaluating: {evalex}', print_type=self.print_type)
                # eval_fails.append(str(ex))
                # continue
                return

        expr_raw = cooltree.get_expr_raw()
        expr_sym = expr_sympify(expr_raw)
        try:
            cooltree.set_fix_nodes(self.origin_cooltree)
        except Exception as ex:
            print(f'Damn yo, failed tree finish: {ex}\n{cooltree}')

        cooltree.meta.fitness_train = fitness_train
        cooltree.meta.parsimony = parsimony
        cooltree.meta.last_evolution = last_evolution
        cooltree.meta.expr_raw = expr_raw
        cooltree.meta.expr_sym = expr_sym

        self.treelut_tree_add(cooltree)
        self.population_tmp.append(cooltree)

        self.update_pareto(cooltree)

    def pop_selection_tournament(self, tourn_size):

        """
        config-selection. takes a number of trees (we use 3) and returns the best one (winner)
        """
        tournament_list = [random.choice(self.pop_base) for _ in range(tourn_size)]
        tourn_winner = self.kernel.best_fitness_function(tournament_list, key=lambda cooltree: cooltree.meta.fitness_train)

        return copy.deepcopy(tourn_winner)

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def load_origin_tree(self, user_origin_csv, label_list=None, modify_list=None):
        """
        The origin tree (which was already loaded) gets activated for its use in the GP-process
        """
        # tree_expr_txt_path = root_dir / 'run_files/tree_expr.txt'
        # tree_numpy_csv_path = root_dir / 'run_files/tree_numpy.csv'
        # tree_labels_csv_path = self.root_dir / 'run_files/tree_labels.csv'

        """
        from the labellist-csv, loading the label list
        """

        with Path.open(user_origin_csv, newline='') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')
            for row in reader:
                if len(row) > 0:
                    if row[0] == 'label_list' or row[0] == 'node_label':
                        label_list = [x.replace(' ', '') for x in row[1:]]
                    elif row[0] == 'modify_list' or row[0] == 'node_modify':
                        modify_list = [int(x.replace(' ', '')) for x in row[1:]]
                    elif row[0] == '':
                        pass
                    else:
                        print_warning('ww', f'Unexpected row start: {row[0]}')
        # sfeh if file is a .txt file with an expression
        # elif Path.is_file(tree_expr_txt_path):  # karoo_tree_from_expr(expr)
        #     raise Exception('SFEH needs to create an option to make trees from expression')
        #     # with Path.open(tree_expr_txt_path) as txt_file:
        #     #     expr = txt_file.read()  # sfeh requires separate handling?
        #     #     print('Assuming all variables are floats, sfeh')
        #     #     tree = karoo_tree_from_expr(expr, 'sfeh')
        #     #     tree_pretty_print(tree)
        #     #     tree_save_csv(tree, tree_labels_csv_path)
        #     #     raise  # sfeh

        # if label_list is None:
        #     raise Exception('Labels could not be created from file.')
        # else:
        #     print_warning('ii', 'No origin-tree file was provided. Continuing.')
        # return label_list, modify_list

        try:
            origin_cooltree = cooltree_from_labellist(label_list, self.env_vars.obs_krazy, modify_list=modify_list)
            expr_sym = origin_cooltree.get_expr_sym()
        except Exception as sympex:
            raise Exception(f'Loaded origin_tree already failed because of: {sympex}')

        # sfeh, this does not work
        # if not tree_check_is_sympified(tree):
        #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
        #                          ''.format(expr_raw, expr_sym))

        used_observations = origin_cooltree.get_observation_list()
        tf_origin_results = self.kernel.eval_tf(expr_sym, used_observations)
        fitness_train = tf_origin_results['mean_error']  # fitness currently IS the mean error
        if self.kernel.exploration_risk:
            self.kernel.origin_results = tf_origin_results['results_kernel']  # now, these informations can be updated

        origin_cooltree.meta.fitness_train = fitness_train
        origin_cooltree.meta.parsimony = 0

        self.origin_cooltree = copy.deepcopy(origin_cooltree)

        self.pareto = [[0, fitness_train, origin_cooltree]]  # aka [3, 423, meta{}]
        self.print_g('gg', f'Loading origin tree, fitness {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return

    def tree_eval_fitness_train(self, cooltree: CoolTree):
        """
        Very fast eval-version that only computes fitness of the train data.
        tree_eval_complete gives more options
        Evaluating the fitness of a tree.
        - extract the expression the tree is holding
        - sympify the expression
        - (if sympify fails, evaluating does not make sense! Check sympify errors)
        - (sfeh: if sympify fails because of inf or zoo, tf could maybe still work due to save-tf-division)
        """

        try:
            expr_sym = cooltree.get_expr_sym()
        except Exception as evalex:
            raise Exception(f'eval:{evalex}')

        used_observations = cooltree.get_observation_list()
        fitness_train = self.kernel.eval_tf(expr_sym, used_observations, only_fitness=True)

        if not check_value_is_real(fitness_train):
            raise Exception(f'Error is {fitness_train}')  # happens, eg when values are soo wrong that it leaves the float-range
        fitness_train = round(fitness_train, self.conf.fitness_decimals)

        return fitness_train

    def tf_classify_labels_map(self, result):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the samples-csv.
        Outputs an array of tuples containing the predicted labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.uniques_num / 2) - 1 # '-1' keeps a binary classification splitting over the
            if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
            elif solution == self.uniques_num - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
            else: fitness = 0 # no class match
        sfeh remove

        """
        uniques_num = self.env_vars.eval_action.uniques
        skew = (uniques_num / 2) - 1
        label_rules = {uniques_num - 1: (
            tf.constant(uniques_num - 1), tf.constant(f' > {uniques_num - 2 - skew}'))}

        for class_label in range(uniques_num - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tf.cond(cond, lambda: (
                tf.constant(class_label), tf.constant(f' <= {class_label - skew}')), lambda: label_rules[class_label + 1])

        pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(f' <= {0 - skew}')), lambda: label_rules[1])

        return pred_label

    def file_analysis_plots(self, root_path, subfolder=''):
        """
        Make all plots
        """

        path_plots = folder_make_dir(root_path / self.file_loc('folder_plots') / subfolder)

        def plotendify_me(data_dict):
            """

            """
            # sfeh delete this version1 (working only with np-arrays, no dicts)
            npdata_dict = np.array([list(x) for x in sorted(data_dict.items())]).T
            return npdata_dict

        def get_pareto_plot_values():
            """

            """
            tuples = []
            for (parsim, fitness, cooltree) in self.pareto:
                tuples.append([parsim, fitness])
            npdata_dict = np.array(tuples).T
            return npdata_dict

        if self.conf.plot_verbosity['gen_fitness_average'] == 'y':
            data_tuples = plotendify_me(self.monitoring_dict['fitness_average'])
            plot_end(data_tuples, path_plots, title='average error', x_label='Generation', y_label='regression error', set_left=data_tuples[0][0], print_type=self.print_type)

        if self.conf.plot_verbosity['population_tmp_done-size'] == 'y':
            data_tuples = plotendify_me(self.monitoring_dict['population_tmp_done-size'])
            plot_end(data_tuples, path_plots, title='genepool size', x_label='Generation', y_label='amount', linestyle='None',
                     marker='.', set_left=data_tuples[0][0], print_type=self.print_type)

        data_tuples = plotendify_me(self.monitoring_dict['gen_time'])
        plot_end(data_tuples, path_plots, title='Generation time', x_label='Generation', y_label='time (sec)', linestyle='None',
                 marker='.',
                 set_left=data_tuples[0][0], print_type=self.print_type)

        data_tuples = get_pareto_plot_values()
        plot_end(data_tuples, root_path, title='pareto dominant candidates', x_label='parsimony',
                 y_label='regression error',
                 linestyle='dashed',
                 marker='.',
                 step_where='post',
                 set_right=self.conf.parsimony_max,
                 beyond_lines=True, print_type=self.print_type)

        if self.conf.plot_verbosity.get('fitness_variance') == 'y':
            data_tuples = plotendify_me(self.monitoring_dict['fitness_variance'])
            plot_end(data_tuples, path_plots, title='variance in error', x_label='Generation', y_label='variance',
                     marker='', print_type=self.print_type)

        data_tuples = plotendify_me(self.monitoring_dict['complexity_average'])
        data_tuples_variance = plotendify_me(self.monitoring_dict['complexity_variance'])  # sfeh update to standard error
        plot_end(data_tuples, path_plots, title='tree parsimony (avg and std. error)', x_label='Generation', y_label='variance',
                 marker='', fill_variance=data_tuples_variance, print_type=self.print_type)

        data_tuples = plotendify_me(self.monitoring_dict['best_candidate'])
        plot_end(data_tuples, path_plots, title='best candidate', x_label='Generation',
                 y_label='error', linestyle='dashed', step_where='post', print_type=self.print_type)

        return

    def pop_analyse(self):
        """
        Analysing this generation
        - amount of trees
        - fittest tree
        - average fitness
        - average tree parsimony
        """
        gen_id = self.gen_id
        popul = self.population_tmp

        if len(popul) == 0:
            raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

        pop_fitness = [cooltree.meta.fitness_train for cooltree in popul]
        pop_parsim = [cooltree.meta.parsimony for cooltree in popul]
        pop_treelen = [len(cooltree) for cooltree in popul]
        pop_last_evolve = [cooltree.meta.last_evolution for cooltree in popul]

        pop_fitness_best = self.kernel.np_best_fitness(pop_fitness)
        try:
            self.best_fitness = self.kernel.best_fitness_function(pop_fitness_best, self.best_fitness)
        except:
            self.best_fitness = pop_fitness_best

        self.monitoring_dict['population_tmp_done-size'][gen_id] = len(popul)
        self.monitoring_dict['pop_parsim'][gen_id] = pop_parsim
        self.monitoring_dict['pop_last_evolves'][gen_id] = pop_last_evolve
        self.monitoring_dict['fitness_average'][gen_id] = np.average(pop_fitness)
        self.monitoring_dict['fitness_variance'][gen_id] = np.var(pop_fitness)
        self.monitoring_dict['best_candidate'][gen_id] = self.best_fitness
        # self.monitoring_dict['complexity_list'][gen_id] = pop_treelen
        self.monitoring_dict['complexity_average'][gen_id] = np.average(pop_treelen)
        self.monitoring_dict['complexity_mean'][gen_id] = np.mean(pop_treelen)
        self.monitoring_dict['pop:trees:complexity:std_error'][gen_id] = np.std(pop_treelen)
        self.monitoring_dict['complexity_variance'][gen_id] = np.var(pop_treelen)  # sfeh version1 delete this shit

        gen_time = time.perf_counter() - self.time_genstart
        self.monitoring_dict['gen_time'][gen_id] = gen_time

        unique_tree_count = len([hash(x) for x in popul])  # sfeh analyze this?
        self.print_g('gg', f'Created {len(popul)}/{self.conf.pop_max} ({unique_tree_count} unique) in generation {gen_id}. Gen took {gen_time:4.2f}s')
        return

    def terminate_run(self, make_a_backup=True):
        """
        Program is done after writing all gp_files one last time.
        """
        if make_a_backup:
            self.run_backup_save()
        self.file_analysis_complex()

        self.file_analysis_plots(self.root_dir)
        self.print_g('gg', f'Terminating. \tTime since start: {time.perf_counter() - self.time_start:4.2f}s')

    def printpl(self, message_type, message_str):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        message_type options can be found in config
        """

        if message_type in self.print_type:
            printez(message_type, message_str, print_type=self.print_type, time_total=time.perf_counter() - self.time_start)

        return

    def print_g(self, message_type, text):
        """

        """

        if message_type in self.print_type:
            printez(message_type, text, time_total=time.perf_counter() - self.time_start)

        return


def choose_build_size(size_mode, mean_min_max_var, tree=None, node_id=None, force=None):
    """

    # branch_nodes, branch_depth, tree_depth, tree_nodes
    """
    mean, size_min, size_max, size_variance = mean_min_max_var
    if size_mode == 'branch_nodes' or size_mode == 'branch_depth' or force == 'branch':
        relative_size = 0
    else:
        if tree and node_id:
            pass
        else:
            raise Exception('No tree or node is given for computing the relative size')

        if size_mode == 'tree_depth':
            tree_size = tree_get_depth(tree)
            node_size = tree_node_get_depth(tree, node_id)
        elif size_mode == 'tree_nodes':
            tree_size = tree_get_size(tree)
            node_size = len(tree_node_get_branch(tree, node_id))
        else:
            raise Exception('Sizemode not known?')

        relative_size = tree_size - node_size

    build_size = int(random.normalvariate(mean, size_variance))
    if size_max is not None:
        build_size = min(size_max - relative_size, build_size)
    build_size = max(size_min, build_size)

    return int(build_size)


def check_value_is_real(fitness):
    """
    Returns bool value if we can use the calculated fitness
    Fitness values might evaluate to weird stuff
    e.g. 'nan' after dividing by zero or (inf) after 20**1234
    nan: fitness == fitness -> False
    inf: fitness is not float('inf') -> False
    """
    return fitness == fitness and fitness != float('inf')
