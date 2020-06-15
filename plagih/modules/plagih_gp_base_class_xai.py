"""
'f2f', 'b2b', f = float, b = bool. 'float to bool'

Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""
import time
from plagih.modules.file_interaction import *
# from plagih.modules.plagih_tree import *
from plagih.modules.viz_with_latex import *
import json
import collections.abc
from pathlib import Path
import textwrap
from plagih.modules.plagih_data import *

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees
PLAGIH_VERSION = 0.953  # must only update if vital changes were made


class ExplainableGP(object):
    """

    """

    def __init__(self, plagih_root, root_dir, user_config, user_file_paths=None, opt_evolve_list=None, data_prepared_path=None, opt_origin_tree_csv=None, out_dir=None):

        self.name = root_dir.name  # sfeh probably there are better names
        print('\n\tInitializing Plagih. Name: {}{}{}. Located in: \n\t{}\n'.format(BColors.CYAN, self.name, BColors.RESET, root_dir))
        self.time_start = time.perf_counter()
        self.restart_vers = 'v0.8'
        self.root_dir = root_dir
        self.sfeh_plagih_root = plagih_root

        self.config = {
            'remove superfluous config entries': False,  # guess you should do that

            'pl_version': PLAGIH_VERSION,  # version important when loading old run
            'description': 'No description set',
            'force_new_run': False,  # especially for testing. Otherwise, delete the folder. can be set via command line.
            # When to stop the run
            'time_max': None,  # int(60 * 60 * 12),  # 60 = 1 min
            'gen_max': 1001,  # Maximum amount of generations

            # (!) Relevant for result
            # 'pop': {
            #
            # },
            'pop_max': 1000,  # amount is never tested
            'tree_depth_max': 10,  # maximum Tree depth for entire run
            'tree_depth_min': 2,
            'tourn_size': 3,  # [7 per 100] number of trees selected for tournament
            'parsimony_mean': 20,  # If you wnt your population to be a certain size
            'parsimony_max': 50,
            'gen_num_max_parsimony': 50,  # Increase tmp_parsim to this generation

            # 'evalue': {
            #
            # },
            'kernel_name': 'regression discrete',  # [regression, regression bounded, classification, match]
            'complexity_measure': 'tree_edit_distance',
            'eval_action': 0,  # Only one action at a time! If the data has more than one action, runs have to be split. This can be specified here.
            'obs_past:max': 9,  # How long a variables history be used? (Only required if historic values do exist)
            'fitness_accuracy': 3,  # rounding the fitness

            'parsimony_tmp': 15,
            'float_accuracy': 10,  # None or 1-30 decimals
            'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool

            'user_feedback': {

            },
            'print_type': 'ggwsiivoaaf',  # To show absolutely all: wwwwggggsiiiivvvtoppptttff
            'overwrite periodic gp_files': True,  # If True, the file gets overwritten. If False, in every generation a new file is created.
            'plot_verbosity': {'gen_fitness_average': 'y',
                               'sympify_errors': 'y',
                               'population_tmp_done-size': 'y',
                               'fitness_variance': 'n'},
            'period': {'time_monitor': None,  # in sec
                       'time_save': None,  # in sec
                       'gen_monitor': 10,  # in gen counts
                       'gen_save': 50},  # in gen counts

            'evolve_list': [
                # Reproduction (10%)
                {'tag': 'Repro', 'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.06,
                 'custom_params': {}},
                {'tag': 'Rsympy', 'evolve_name': 'reproduce', 'evolve_rate': 0.03,
                 'custom_params': {'sympify_tree': True}},
                {'tag': 'Pareto', 'evolve_name': 'revive pareto', 'evolve_rate': 0.01,
                 'custom_params': {}},

                # Mutation (25%)
                {'tag': 'Point', 'evolve_name': 'mutate point', 'evolve_rate': 0.05,
                 'custom_params': {}},
                # todo point branch mutate?
                # {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                #  'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8), 'full_or_grow': 'full'}}},
                # {'tag': 'BranchDG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                #  'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 5, 1), 'full_or_grow': 'grow'}}},
                {'tag': 'BranchNG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (10, 1, 20, 5), 'full_or_grow': 'full'}}},
                {'tag': 'BranchNG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (10, 1, 20, 5), 'full_or_grow': 'grow'}}},
                {'tag': 'BranchShrink', 'evolve_name': 'mutate branch', 'evolve_rate': 0.0,
                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0), 'full_or_grow': 'grow'}}},
                {'tag': 'FilterB', 'evolve_name': 'filter optimize', 'evolve_rate': 0.10, 'tourn_size': 5,
                 'custom_params': {'mode': 'branch'}},
                {'tag': 'FilterP', 'evolve_name': 'filter optimize', 'evolve_rate': 0.0, 'tourn_size': 5,
                 'custom_params': {'mode': 'point'}},

                # Crossover (35%)
                {'tag': 'Xover', 'evolve_name': 'crossover branch', 'evolve_rate': 0.35,  # sum 0.70
                 'custom_params': {}},

                # Leftovers are automatically filled with random trees

                # Random (25%)
                # {'tag': 'Rand1', 'evolve_name': 'random trees', 'evolve_rate': 0.10,
                #  'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 3, 5, 1), 'full_or_grow': 'full'}}},
                # {'tag': 'Rand2', 'evolve_name': 'random trees', 'evolve_rate': 0.10,
                #  'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'full_or_grow': 'grow'}}},
                {'tag': 'Rand3', 'evolve_name': 'random trees', 'evolve_rate': 0.15,
                 'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (20, 12, None, 6), 'full_or_grow': 'grow'}}},
                {'tag': 'Rand4', 'evolve_name': 'random trees', 'evolve_rate': 0.15,
                 'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (20, 12, None, 6), 'full_or_grow': 'full'}}},
                ],

            # todo
            'evolve_list_random': {
                'from_origin': [
                    {'tag': 'RandO3', 'evolve_name': 'random trees', 'evolve_rate': 1.00,
                     'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (20, 10, 45, 6), 'full_or_grow': 'full'}}}
                ],
                'from_scratch': [
                    {'tag': 'Rand1', 'evolve_name': 'random trees', 'evolve_rate': 0.30,
                     'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'full_or_grow': 'full'}}},
                    {'tag': 'Rand2', 'evolve_name': 'random trees', 'evolve_rate': 0.30,
                     'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 6, 1), 'full_or_grow': 'grow'}}},
                    {'tag': 'Rand3', 'evolve_name': 'random trees', 'evolve_rate': 0.40,
                     'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (20, 10, 45, 6), 'full_or_grow': 'full'}}}
                ],

            },

            # todo these are irrelevant, when the actual paths are used. file-loc updates might even be wrong.
            #  make a difference between write/read?
            'file_locs': {
                'pycode_load': 'benchmarks/gym_mountaincar/agents/quick_eval.py',  # todo make pretty solution
                'example_runs': 'run_examples/',

                'folder_plots': 'plots/',
                'folder_steps': 'steps/',
                'folder_pop_analysis': 'pop_dist/',
                'folder_histograms': 'agents/',

                'file_backup_pickle': 'backup/backup.p',  # backup-version is set here
                'file_conclusion': 'conclusion.txt',

                # /agents/
                'trees_tex': 'agents/agents_trees.tex',
                'file_pycode': 'agents/agents.py',
                'file_pycode_eval': 'agents/eval_agents.py',

                # /info/
                'file_pareto': 'info/pareto.yaml',
                'info_config_yaml': 'info/config.yaml',
                'file_info_config_json': 'info/config.json',
                'file_info_evolve_dict_yaml': 'info/evolve_list.yaml',
                'info_distributions_yaml': 'info/distributions_file.yaml',
                'env_variables_yaml': 'info/env_variables.yaml',

                # /run_files/
                'file_config_yaml': 'run_files/config.yaml',
                'file_config_json': 'run_files/config.json',
                'samples_ready_p': 'run_files/samples_ready.p',
                'file_evolve_functions': 'run_files/evolve_functions.yaml',
                'samples_csv': 'run_files/samples.csv',
                'operators_yaml': 'run_files/operators.yaml',
                'distributions_file': 'run_files/distributions_file.yaml',
                'tree_expr_txt': 'run_files/tree_expr.txt',
                'tree_labels_csv': 'run_files/tree_labels.csv',
                'tree_numpy_csv': 'run_files/tree_numpy.csv',
            },
            'distributions_as_string':
                {'2f': ['lambda: np.random.normal(1,2)',
                        'lambda: np.random.normal(1,1)',
                        'lambda: np.random.randint(0, 10)'],
                 '2b': ['lambda: np.random.choice([True, False])'],
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

        if True:  # some config_dict values have to be used quite often...
            self.kernel = FitnessKernel(self.config['kernel_name'])
            self.print_type = self.config['print_type']
            self.precision = self.config['fitness_accuracy']  # the number of floating points for the round function
            self.parsimony_max = self.config['parsimony_max']
            self.monitoring_verbosity = self.config['plot_verbosity']
            self.tourn_size = self.config['tourn_size']
            self.evolve_list = self.config['evolve_list']
            self.file_locs = self.config['file_locs']

        # Making useable files from the raw string format
        self.root_paths = {}
        for file_key, file_loc in self.config['file_locs'].items():
            self.root_paths[file_key] = Path(root_dir / file_loc)
        # self.root_paths.update(user_file_paths)

        self.activate_dataset(data_prepared_p_path=data_prepared_path)

        # distributions_as_string = self.config['distributions_as_string']
        # sfeh
        # if opt_distributions_as_string:
        #     distributions_as_string.update(opt_distributions_as_string)
        self.load_tree_builders_distributions(path_user_distributions=None)

        if opt_evolve_list:
            self.evolve_list.update(opt_evolve_list)

        self.load_tree_builders_choose_oparray(opt_path_opyaml=None)  # todo

        # init values with dummies (just to have all self values here for overview)
        self.tree_lut = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.parsimony_best_meta = {}  # tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.pareto = {}  # a dict with all pareto candidates. key is complexity, value is tree meta
        self.population_tmp_done = []
        self.population_tmp_eval = []
        self.population_base = []  # population that is taken to the next generation
        self.best_fitness = None  # keeps track of the current best fitness
        self.origin_meta = None
        self.origin_tree = None
        self.gen_id = 0
        self.custom_done = False
        self.restart_count = 0
        self.time_last_monitor = self.time_start
        self.time_last_files = self.time_start
        self.pop_next = None

        # special variables
        self.tf_device = "/gpu:0"  # sfeh Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device. Is cpu otherwise
        self.tf_device_log = False  # TF device usage logging (for debugging)

        self.tf_config = tf.compat.v1.ConfigProto(log_device_placement=self.tf_device_log, allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True

        # self.node_choose_dict = {
        #     '2f': {
        #         0: {'observ': [],
        #             'distribution': []},
        #         1: {'f2f': [],
        #             'b2f': []},
        #         2: {'f2f': [],
        #             'b2f': []},
        #         3: {'b2f2f': []}},
        #     '2b': {
        #         0: {'observ': [],
        #             'distribution': []},
        #         1: {'f2b': [],
        #             'b2b': []},
        #         2: {'f2b': [],
        #             'b2b': []},
        #         3: {None: []}}}

        # some useful stuff
        self.monitoring_dict = {'population_tmp_done-size': {},
                                'fitness_average': {},
                                'fitness_variance': {},
                                'best_candidate': {},
                                'total_found_trees': {},
                                'gen_time': {},
                                'complexity_average': {},
                                'complexity_variance': {},  # variance can be deleted, only std-error is needed
                                'pop:trees:complexity:std_error': {}}

        self.print_g('ggg', 'Init. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        return

    def try_load_backup(self):
        """
        If a backup-file is found...
        """
        path_backup = self.root_paths['file_backup_pickle']
        if Path.is_file(path_backup):
            self.print_g('g', 'Backup-file was found. Loading data...')
            try:
                self.load_backup_pickle(path_backup)
                return True
            except Exception as ex:
                print_warning('w',
                              'Even though a backup exists for this run, it could not be loaded because of: {}.'.format(ex))
                raise
        else:
            return False

    def update_old_runs(self):
        """
        If features were added between versions, you can try to update this from here
        """
        if self.env_variables.get('env_observation_family') is None:
            self.env_variables['env_observation_family'] = {}
            # sfeh delete this sometimes
            for x in self.env_variables['obs_name'].values():
                col_label = x['label']
                temp_diff = envvariable_get_tempdiff(col_label)
                core_label = envvariable_get_corelabel(col_label)
                try:
                    self.env_variables['env_observation_family'][core_label].extend([col_label])
                except (IndexError, KeyError):
                    self.env_variables['env_observation_family'][core_label] = [col_label]

                self.env_variables['obs_name'][col_label]['temp_diff'] = temp_diff
                self.env_variables['obs_name'][col_label]['core_label'] = core_label
            self.printpl('w', 'Attention, we updated self.env_variables for an old run')
            # todo should we save the update?

        return

    def load_backup_pickle(self, path_backup):
        """
        Loading the state of the run from the pickle file
        """

        with Path.open(path_backup, 'rb') as file_backup:
            run_data = pickle.load(file_backup)

        try:
            self.config['pl_version'], self.restart_count, self.gen_id, self.parsimony_best_meta, self.pareto, self.population_base, self.monitoring_dict = run_data
            # todo test
        except:
            # todo remove this in version1
            self.restart_count, self.gen_id, self.parsimony_best_meta, self.pareto, self.population_base, self.monitoring_dict = run_data
            self.version = 0.9

        self.restart_count += 1

        self.update_old_runs()

        # update monitoring dict (average vs variance)
        if self.config['pl_version'] <= 0.9:
            pass
            # self.monitoring_dict['pop:trees:complexity:std_error'] = np.sqrt(self.monitoring_dict['complexity_variance'])

        printez('g', 'Loading Generation: {}'.format(self.gen_id), self.print_type)

        return

    def run_backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """

        run_backup_data = self.config['pl_version'], self.restart_count, self.gen_id, self.parsimony_best_meta, self.pareto, self.population_base, self.monitoring_dict

        path_backup = file_make_dir(self.root_paths['file_backup_pickle'])
        with Path.open(path_backup, 'wb') as file:
            pickle.dump(run_backup_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.printpl('ff', 'Saved: {}'.format(path_backup))
        return

    def plagih_gp_run(self):
        """
        regular plagih run
        """

        if not self.config['force_new_run']:
            self.try_load_backup()
            # sfeh: delete old files?

        if not self.origin_exists():
            if self.config['complexity_measure'] in ['tree_edit_distance']:  # sfeh get all origin-based distances
                self.config['complexity_measure'] = 'tree_node_count'
                print_warning('w', 'Complexity measurement \'tree_edit_distance\' is not possible without origin! Using \'tree_node_count\' instead.', print_type=self.print_type)

        self.write_config_yaml()  # just to see what config is running and what the user can set

        if self.gen_id == 0:
            self.print_g('gg', 'Preparing to create first Generation. Gen {}.'.format(self.gen_id))
            self.gen_id = 0
            self.gen_reset_parameters()
            self.gen_create_initial()

        while self.run_continues():  # max generation, max time, done...
            self.printpl('gggg', 'Evolving Generation {}'.format(self.gen_id))
            self.gen_create_loop()
            self.periodical_procedures()
            self.print_g('ggg', 'Generation {} took a total time of: {:4.2f}. '.format(self.gen_id, time.perf_counter() - self.time_genstart))
        else:
            printez('g', 'Done after Generation {}.'.format(self.gen_id), print_type=self.print_type)

        self.terminate_run(self.root_dir)
        return

    def plagih_update_analysis(self):
        """
        Without starting a new run, get the most important gp_files
        """

        loading_done = self.try_load_backup()
        if not loading_done:
            print_e('You need to load a backup file to analyse!')
        else:
            self.terminate_run(self.root_dir)

    def write_config_yaml(self):
        """
        write the parameters to a .csv file which can also be loaded
        """

        # filename = self.root_dir / info_config_yaml
        filename = file_make_dir(self.root_paths['info_config_yaml'])
        yaml_dump(filename, self.config, print_type=self.print_type)

        return

    def write_config_json(self):
        """
        write the parameters to a .csv file which can also be loaded
        """

        path = file_make_dir(self.root_paths['file_info_config_json'])

        with Path.open(path, 'w') as file:
            json.dump(self.config, file, indent=4)

        return

    def origin_is_fix(self):

        if self.origin_exists():
            if not tree_node_is_modifiable(self.origin_tree_get(), root_id):
                return True
        return False

    def gen_create_random(self, amount):
        """
        Add an amount of random trees to the population
        """
        if self.origin_is_fix():
            evolve_list_random = self.config['evolve_list_random']['from_origin']
        else:
            evolve_list_random = self.config['evolve_list_random']['from_scratch']

        total_rate = sum([x['evolve_rate'] for x in evolve_list_random.values()])

        for ii, evolve_specs in enumerate(evolve_list_random.items()):
            evolve_num = int(amount * (evolve_specs['evolve_rate'] / total_rate))
            call_params = evolve_specs.get('custom_params')
            tag = evolve_specs['tag']

            for nn in range(evolve_num):
                if self.origin_is_fix():
                    new_tree = self.pop_random_from_origin_fix(call_params, self.origin_tree)
                else:
                    new_tree = self.pop_random(call_params)
                self.pop_append(new_tree, last_evolution=tag)

    def gen_create_initial(self):
        """
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        if self.origin_exists():
            # self.config['evolve_list_random']['from_origin']
            # sfeh why not :P
            self.pop_append(self.origin_tree, last_evolution='initial')
            # if self.origin_is_fix():
            #     origin_tree = self.origin_tree_get()
            #     for nn in range(evolve_num):
            #         new_tree = self.pop_random_from_origin_fix(call_params, origin_tree)
            #         self.pop_append(new_tree, last_evolution=tag)
        else:
            self.gen_create_random(self.config['pop_max'])

        self.gen_finalize()
        write_file_population_karoo(self.population_base, 'first', self.root_dir, self.gen_id, print_type=self.print_type)  # first gen only

    def gen_create_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        # All gp creators: name, function, num of trees from tournament selection
        # sfehsfeh idee: print every written file

        self.gen_id += 1
        self.gen_reset_parameters()

        # ########

        #     {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
        #     'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean': 12, 'min': 1, 'max': 30, 'gauss_var': 1, 'method': 'grow'}}},
        for ii, evolve_specs in enumerate(self.evolve_list):  # all selected gp mutations

            time_evolve = time.perf_counter()
            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            call_params = evolve_specs.get('custom_params')
            tag = evolve_specs['tag']

            # believe me, debugging with this is much more fun

            self.print_g('gggg', '->Evolving \'{}\' {}x starting...'.format(tag, evolve_num))

            if evolve_name == 'reproduce':
                # sfeh one parameter
                for nn in range(evolve_num):
                    tree = self.pop_selection_tournament(tourn_size)
                    new_tree = self.pop_reproduce(call_params, tree)
                    self.pop_append(new_tree, last_evolution=tag)

            elif evolve_name == 'mutate point':

                for nn in range(evolve_num):
                    tree = self.pop_selection_tournament(tourn_size)
                    new_tree = self.pop_mutate_point(tree)
                    self.pop_append(new_tree, last_evolution=tag)

            elif evolve_name == 'mutate branch':

                for nn in range(evolve_num):
                    tree = self.pop_selection_tournament(tourn_size)
                    new_tree = self.pop_mutate_branch(call_params, tree)
                    self.pop_append(new_tree, last_evolution=tag)

            elif evolve_name == 'crossover branch':

                for nn in range(int(evolve_num / 2)):  # two childs
                    parent_a = self.pop_selection_tournament(tourn_size)
                    parent_b = self.pop_selection_tournament(tourn_size)
                    child_a, child_b = self.pop_crossover_branch(parent_a, parent_b)
                    self.pop_append(child_a, last_evolution=tag)
                    self.pop_append(child_b, last_evolution=tag)

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    tree = self.pop_selection_tournament(tourn_size)
                    new_tree = self.pop_mutate_filter(call_params, tree)
                    self.pop_append(new_tree, last_evolution=tag)

            elif evolve_name == 'revive pareto':

                for nn in range(evolve_num):
                    new_tree = self.pop_reproduce_olymp()
                    self.pop_append(new_tree, last_evolution=tag)

            elif evolve_name == 'random trees':

                if self.origin_is_fix():
                    origin_tree = self.origin_tree_get()
                    for nn in range(evolve_num):
                        new_tree = self.pop_random_from_origin_fix(call_params, origin_tree)
                        self.pop_append(new_tree, last_evolution=tag)
                else:
                    for nn in range(evolve_num):
                        new_tree = self.pop_random(call_params)
                        self.pop_append(new_tree, last_evolution=tag)
            else:
                print_e('the specified evolve call is not known: \'{}\''.format(evolve_name))

            self.print_g('ggg', '->Evolving \'{}\' {} times took: {:4.2f}s.'.format(tag, evolve_num, time.perf_counter() - time_evolve))

        # sfeh automatically fill with random trees
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.config['pop_max'] * (1 - total_rate)))

        self.gen_finalize()

    def run_continues(self):
        """
        checks if the run can continue
        """
        cond_1 = self.gen_id >= self.config['gen_max']
        cond_2 = False if self.config['time_max'] is None else time.perf_counter() - self.time_start > self.config[
            'time_max']
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
        if self.origin_tree is not None:
            return True
        else:
            return False

    def periodical_procedures(self, plots_show=None, save_run=None):
        """
        Every few generations, update the created gp_files
        - default is in every generation, but saving every n-th gen or after time passed is possible aswell
        """

        time_now = time.perf_counter()

        if self.config['period']['time_monitor']:
            if self.config['period']['time_monitor'] < (time_now - self.time_last_monitor):
                self.printpl('iii', 'auto-plots (time)')
                plots_show = True
                self.time_last_monitor = time_now

        if self.config['period']['time_save']:
            if self.config['period']['time_save'] < (time_now - self.time_last_files):
                self.printpl('iii', 'auto-save (time)')
                save_run = True
                self.time_last_files = time_now

        if self.config['period']['gen_monitor']:
            if self.gen_id % int(self.config['period']['gen_monitor']) == 0:
                self.printpl('iii', 'auto-plots (gen)')
                plots_show = True

        if self.config['period']['gen_save']:
            if self.gen_id % int(self.config['period']['gen_save']) == 0:
                self.printpl('iii', 'auto-save (gen)')
                save_run = True

        if plots_show:
            if self.config['overwrite periodic gp_files']:
                subfolder = ''
            else:
                subfolder = Path(self.file_locs['folder_steps']) / 'Gen-{}'.format(self.gen_id)
            self.file_all_plots(self.root_dir, subfolder=subfolder)

        if save_run:
            self.run_backup_save()
            # self.file_save_files(tmp_path)  # todo

        self.printpl('iii', 'Done with auto-procedures')

        return 0

    def file_save_files(self, root_path):
        """
        writes all important gp_files

        """
        self.file_pareto_histograms(root_path)
        self.file_conclusion(root_path)
        write_file_pareto_txt(self.pareto, root_path, self.file_locs['file_pareto'])  # todo "get path" instead?
        self.file_pareto_latex(self.pareto, root_path)
        self.file_generate_pycode(self.pareto, root_path)
        write_file_population_karoo(self.population_base, 'last', root_path, self.gen_id, print_type=self.print_type)

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Load and Archive Data                     +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def remove_old_variables(self):
        """

        """
        max_past = self.config['obs_past:max']

        for obs_name, obs_info in self.env_variables['obs_name'].items():
            if obs_info['temp_diff'] > max_past:
                xtype = obs_info['xtype']
                self.env_variables[xtype].remove(obs_name)
        return

    def get_path(self, file_key):
        real_path = self.root_paths[file_key]
        return real_path

    def activate_dataset(self, data_prepared_p_path=None, delimiter=','):
        """
        loading the data which the GP will be working on.
        The .csv-file is prepared (loading correct data-type, splitting data, ...)
        and saved as pickle-file for reloading runs.
        This is especially important, as the split in training and test-data must be the same.

        separate loading the prepared data into the main class.
        Why like this? I needed to find a bug in the data_from_csv file and
        did not want to start the whole stuff everytime

            # self.data_train_panda, self.data_control_panda  # todo version1 data_test_panda
        """
        if data_prepared_p_path:
            data_prepared = pickle_load(data_prepared_p_path)
        elif Path.is_file(self.root_paths['samples_ready_p']):  # maybe the data was already prepared earlier
            data_prepared = pickle_load(self.root_paths['samples_ready_p'])
        elif Path.is_file(self.root_paths['samples_csv']):  # Preprocess the raw data: training/test split, env-variables, ...
            self.env_variables, _, _, self.data_train, self.data_control = data_from_csv(self.root_paths['samples_csv'], delimiter=delimiter)

            print('Prepared the raw {} behaviour. Saving for next run.'.format(self.file_locs['samples_csv']))
            data_prepared = (self.env_variables, self.data_train, self.data_control)  # sfeh version1 remove numpy version
            pickle_dump(self.root_paths['samples_ready_p'], data_prepared)
        else:
            raise FileNotFoundError('No data provided? Please provide data in your config-file(or in your command line call).')  # samples_ready_p or samples_csv required

        dataspec_file = self.root_dir / 'run_files/data_specification.yaml'  # todo
        if dataspec_file.is_file():
            pass  # sfeh: if you want to load information from extra file, check for this file here
        else:
            # sfeh env_variables, _, _ = data_prepared. anyways, currently loading info via brackets in .csv-file
            yaml_dump(self.root_paths['env_variables_yaml'], data_prepared[0], print_type=self.print_type)

        self.env_variables, self.data_train, self.data_control = data_prepared  # data_control is data_test
        self.update_old_runs()
        self.remove_old_variables()

        return

    def load_evolve_functions(self, file_evolverates=None):
        """

        """
        if not file_evolverates:
            file_evolverates = self.root_paths['evolve_file']

        file_evolverates = Path(file_evolverates)
        if Path.is_file(file_evolverates):
            evolve_list = yaml_load(file_evolverates)
            # sfeh this was never (?) tested?
            self.config['evolve_list'] = evolve_list  # sfeh build a check for this loaded file?
        else:
            print_warning('ww', 'Opt-in not specified: Evolve-file for GP evolve functions defined! Trying to choose them for you.')

        # yaml_dump(self.root_paths['file_info_evolve_dict_yaml'], evolve_list)  # todo save the config
        # sfeh: if you want to load information from extra file, check for this file here

        return

    def load_tree_builders_choose_oparray(self, opt_path_opyaml=None):
        """

        """
        # double check load
        if not opt_path_opyaml:
            opt_path_opyaml = self.file_locs['operators_yaml']
        opt_path_opyaml = Path(self.root_dir / opt_path_opyaml)

        if Path.is_file(opt_path_opyaml):
            operators = yaml_load(opt_path_opyaml)
        else:
            # raise FileNotFoundError('File does not exist: {}.'.format(operators_csv))
            print_warning('ww', 'Opt-in not specified: Operators-file does not exist. Creating one with a default list of mathematical operators.')
            operators = np.array([['+', 3],
                                  ['-', 1], ['usub', 2],
                                  ['*', 2], ['/', 1],
                                  ['Square', 0.75], ['**', 0.25],
                                  ['abs', 0.4], ['sign', 0.1],
                                  ['sqrt', 0.2],
                                  ['log', 0.1], ['log1p', 0.1],
                                  ['sin', 0.1],  # ['tan', 0.1], ['cos', 0.33], ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
                                  ['tanh', 0.2],
                                  ['Andb', 1], ['Orb', 1], ['Notb', 0.5],  # ['Xor', 1],
                                  ['==', 1], ['!=', 0.5],
                                  ['<', 0.5], ['<=', 0.5], ['>', 0.1], ['>=', 0.1],
                                  ['Ifte', 2],
                                  ['Mini', 1], ['Maxi', 1]])

            # np.savetxt(operators_csv, functions, delimiter=',', fmt='%s') # sfeh save in config?

        self.choose_oparray2 = oparray_from_list(operators)

        return

    def load_tree_builders_distributions(self, path_user_distributions=None):
        """

        """
        # double check load
        if not path_user_distributions:
            path_user_distributions = self.root_paths.get('distributions_file')

        if Path.is_file(path_user_distributions):
            distributions_as_string = yaml_load(path_user_distributions)
        else:
            print_warning('www', 'Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.')
            distributions_as_string = self.config['distributions_as_string']
            # distributions_as_string is already given...
            # sfeh samples from csv?
            # info_file = file_make_dir(self.root_paths['info_distributions_yaml'])
            # yaml_dump(info_file, distributions_as_string)

        choose_distributions = {'2f': [], '2b': []}

        take_data_samples = distributions_as_string.get('observed_floats')
        if take_data_samples:
            action_columns = list(range(len(self.env_variables['obs_name']), len(self.data_train[0])))  # remove these
            # non_float_columns = ... # sfeh: data types must be float for this to work, remove non-float values. probably, these do not really exist.
            observ_values = np.delete(self.data_train, action_columns, 1)
            variables_set = np.random.choice(observ_values.flatten(), take_data_samples)  # 2nd param is probably '100'
            choose_distributions['2f'].extend([lambda: np.random.choice(variables_set)]),

        choose_distributions['2f'].extend([eval(x) for x in distributions_as_string['2f']]),
        choose_distributions['2b'].extend([eval(x) for x in distributions_as_string['2b']])

        self.choose_distributions = choose_distributions

        return

    def prepare_evolve_functions(self):
        """
        Updates tournament size and evolve rates

        Example entry of the list could be:
        {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
        'custom_params': {'build_spec': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
        """
        evolve_list = self.config['evolve_list']
        for ii, evolve_spec in enumerate(evolve_list):

            # -> tournament size (can be custom set. e.g. a larger tournament for a high-end tree optimisation)
            tourn_size = evolve_spec.get('tourn_size')
            if not tourn_size:
                tourn_size = self.tourn_size
                evolve_list[ii]['tourn_size'] = tourn_size

            # -> Evolve rate
            evolve_rate = evolve_list[ii].get('evolve_rate')
            evolve_list[ii]['evolve_num'] = int(evolve_rate * self.config['pop_max'])
            if evolve_list[ii].get('custom_params') is None:
                evolve_list[ii]['custom_params'] = {}

        # self.config['evolve_list'] = evolve_list  # not needed, is reference ?
        self.evolve_list = evolve_list
        return

    def file_pareto_histograms(self, root_path):
        """
        Make histograms for all pareto-efficient candidates
        sfeh: based on training data- maybe use test data...

        useful code?
        # histogram_data = np.digitize(histogram_data, bins)  # digitize: the bin index the entry belongs to
        # histogram_data = np.concatenate((histogram_data.reshape(-1, 1), pairwise_fitness.reshape(-1, 1)), axis=1)
        # histogram_data = np.multiply.reduce(histogram_data, axis=1)
        # hist, bins = np.histogram(histogram_data, bins=bins, weights=pairwise_fitness)
            # todo only histograms for the variables that are in a tree?
        # todo random histogramme werte am Rand?
        """

        # todo allow very simple IB-solutions...

        path_hist = folder_make_dir(root_path / self.file_locs['folder_histograms'])

        np_data = self.data_train  # sfeh, also non-train data?
        agent_dimatrix = {}
        obs_x_info = {}  # [None] * data_dims

        max_fails_per_bin = 0  # this value will define the y-axis height for all the histograms to look the same

        for ii, (obs_name, obs_info) in enumerate(self.env_variables['obs_name'].items()):
            obs_x_info[ii] = {}  # 'bins': None, 'obs_name': None
            histogram_data = np_data[:, obs_info.get('pos')]  # pos is the same as col
            obs_minmax = obs_info.get('minmax')  # todo this should be set, no matter what.
            if obs_minmax is None:
                # print('SHOULD NOT HAPPEN, SFEH. Must be set earlier!')  # todo
                obs_minmax = (np.min(histogram_data), np.max(histogram_data))

            # todo delete this
            if obs_name == 'cartVel':  # todo
                obs_minmax = (-0.07, 0.07)
            elif obs_name == 'cartPos':
                obs_minmax = (-1.2, 0.6)

            obs_x_info[ii]['bins'] = np.linspace(obs_minmax[0], obs_minmax[1], 32 + 1)  # todo 32 bins?

        for a_ii, (parsim, meta) in enumerate(sorted(list(self.pareto.items()))):
            expr_raw = meta['expr_raw']  # sfeh: tree should already be sympified as much as possible
            ptree = karoo_ptree_from_expr(expr_raw, self.env_variables)
            tree = ptree.get_uninstanced_tree()
            expr_sym = tree_get_expr_sym(tree)

            tf_results = eval_tf(expr_sym,
                                 np_data,
                                 self.kernel,
                                 self.env_variables,
                                 self.tf_config, self.tf_device, self.tf_classify_labels_map, complete=True, specific_action=self.config['eval_action'])  # ['fitness'] only needed if return is dict

            pairwise_fitness = tf_results['pairwise_fitness']
            tf_fitness = tf_results['fitness']
            agent_dimatrix[a_ii] = {}  # 'tf_fitness': None, 'pairwise_fitness': None, 'parsim': parsim
            agent_dimatrix[a_ii]['tf_fitness'] = tf_fitness
            agent_dimatrix[a_ii]['pairwise_fitness'] = copy.deepcopy(pairwise_fitness)
            agent_dimatrix[a_ii]['parsim'] = parsim
            deviation_per_action = (tf_results['kernel_result'] - tf_results['solution_goal'])
            agent_dimatrix[a_ii]['result-solution'] = copy.deepcopy(deviation_per_action)  # this was: # action_hist_data[a_ii] = copy.deepcopy(kernel_result - solution_goal)

            for ii, (obs_name, obs_info) in enumerate(self.env_variables['obs_name'].items()):
                col = obs_info.get('pos')
                histogram_data = np_data[:, col]
                hist, _ = np.histogram(histogram_data, bins=obs_x_info[ii]['bins'], weights=pairwise_fitness)
                max_fails_per_bin = max(max(hist), max_fails_per_bin)
                agent_dimatrix[a_ii]['obs-specific'] = {}
                agent_dimatrix[a_ii]['obs-specific'][ii] = histogram_data
                obs_x_info[ii]['obs_name'] = obs_name  # sfeh meh

        # todo when there are more than 3 (?) dimensions, these plots make no sense.
        #  check for all vars in the tree, mark all observations of the same type with the same color?
        # # >>> Histograms for every dimension
        #
        # data_dims = len(self.env_variables['obs_name'])
        #
        # # for enum_aii, agent_ii in enumerate(agent_dimatrix):
        # for enum_aii, (parsim, agent_info) in enumerate(agent_dimatrix.items()):
        #     fig, axs = plt.subplots(data_dims, 1)
        #
        #     # pairwise_fitness = agent_ii[-1]
        #     pairwise_fitness = agent_info['pairwise_fitness']
        #     # parsim = agent_ii['parsim']
        #     for obs_ii, agent_obs_info in agent_info['obs-specific'].items():  # histogram_data, parsim, tf_results['fitness']  -1? -> pairwise_fitness
        #         # histogram_data, parsim, obs_name, fitness = dim_ii
        #         hist_data = agent_obs_info['histogram_data']
        #         axs[obs_ii].hist(hist_data, bins=obs_x_info[obs_ii]['bins'], weights=pairwise_fitness)  # todo
        #         # ax[enum_ii].hist(histogram_data, bins='auto', weights=pairwise_fitness)  # todo
        #         axs[obs_ii].set_title(obs_x_info[obs_ii].get('obs_name'))
        #         axs[obs_ii].set_ylim(top=max_fails_per_bin)
        #
        #     plt.tight_layout()
        #     plt.savefig(path_hist / 'obs_hist_{}.png'.format(parsim))
        #     plt.clf()

        # >>> Histograms for every action
        # for enum_aii, agent_ii in enumerate(agent_dimatrix):
        for agent_ii, (parsim, agent_info) in enumerate(agent_dimatrix.items()):

            # Histograms action-based
            act_min, act_max = self.env_variables['action_at'][self.config['eval_action']]['minmax']
            if 'discrete' in self.kernel.kernel:
                # todo test
                unique_actions = self.env_variables['action_at'][self.config['eval_action']]['unique_outputs_num']
                action_bins = np.linspace(-0.5 + act_min, 0.5 + act_max, 2 * unique_actions + 1)
            else:
                action_bins = np.linspace(act_min, act_max, 10)  # check out histogram_bin_edges, maybe it is better todo also 10 bins?

            fig, ax = plt.subplots()
            # ax.hist(action_hist_data[enum_aii], bins=action_bins)  # , weights=np.abs(np.sign(pairwise_fitness))  # bins='auto
            # todo options: density = True, cumulative = True, np.clip(), np.diff(np.unique(data)).min() for
            ax.hist(agent_info['result-solution'], bins=action_bins, histtype="stepfilled", facecolor="none", edgecolor='k')  # , weights=np.abs(np.sign(pairwise_fitness))  # bins='auto
            ax.set_ylim(0, len(self.data_train))  # sfeh better size? max? # self.env_variables['action_at'][self.config['eval_action']]
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Deviation')
            fig.tight_layout()
            plt.savefig(path_hist / 'acthist_{}.png'.format(parsim))
            plt.close()  # todo

    def file_conclusion(self, path):
        """
        write the performance of the config to disc
        """
        # path_conclusion = path / folder_info
        # if not Path.is_dir(path_conclusion):
        #     Path.mkdir(path_conclusion)
        #
        # open(path_conclusion / file_conclusion, 'w')
        # file.write('Plagih GP\n launched: {}'.format(str(date_time)))
        #
        # if self.origin_exists():
        #     origin_fitness = eval_tf(self.origin_meta['expr_sym'], self.data_control, self.tf_parameters, get_predicted_labels=True)['fitness']
        #     # fitness_control_best = origin_result['fitness']
        #
        #     fittest_algo = self.origin_meta['expr_sym']
        #     fittest_parsimony = 0
        #
        #     file.write('\n\t Origin fitness score: {}'.format(origin_fitness))
        #
        # elif self.pareto:
        #     file.write('\n No origin_meta was provided')
        #     meta = next(iter(self.pareto.items()))[1]
        #     fittest_parsimony = int(meta['parsimony'])
        #     fittest_algo = meta['expr_sym']
        #     return  # sfeh fittest_parsimony must be set, do not return
        # else:
        #     file.write('\n There are no candidates to be mentioned at all. Maybe change your config?')
        #     return
        #
        # for parsimony, fitness in self.pareto.items():
        #     algo_sym = self.parsimony_best_meta[parsimony]['expr_sym']
        #     result = eval_tf(algo_sym, self.data_control, self.tf_parameters, get_predicted_labels=True)
        #     fit_control = result['fitness']
        #
        #     if self.kernel.fitness_compare(fit_control, fitness_control_best, mode='better_or_equal'):  # find the Tree with a perfect match for all data_csv_path rows
        #         fitness_control_best = fit_control
        #         fittest_algo = algo_sym
        #         fittest_parsimony = parsimony
        #
        #     no_fault = True
        #     for enum, entry in enumerate(result['agent_result']):
        #         if not self.check_value_is_real(entry):
        #             no_fault = False
        #             # sfeh this is a bad workaround
        #             result['agent_result'][enum] = 1
        #
        #     if no_fault:
        #         kernel_result = self.kernel.conclusion_get_text(result, fitness_control_best)
        #         file.write(kernel_result)
        #     else:
        #         file.write('\n\n Error in this tree')
        #
        # else:
        #     # Info about the best Tree
        #     file.write('\n\n The best candidate has parsimony: {}'.format(str(fittest_parsimony)))
        #     file.write('\n With fitness: {}'.format(fitness_control_best))
        #     file.write('\n\n With the following sympify-algorithm:\n {}'.format(fittest_algo))
        #     file.write('\n\n')

        return

    def file_pareto_latex(self, pareto, root_path):
        """
        Generates latex-file with the computational tree structure of all pareto agents
        - build tree from expression
        - fill tree meta-data, just in case we want to visualise anything of it
        - create latex-forest representation
        """

        forest_grouped = []

        for parsim, meta in sorted(list(pareto.items())):
            expr_raw = meta['expr_raw']  # sfeh: tree should already be sympified as much as possible
            ptree = karoo_ptree_from_expr(expr_raw, self.env_variables)
            tree = ptree.get_uninstanced_tree()
            tree = self.tree_finish(tree, last_evolution='texify')
            ###
            vistree = latex_tree_get_vistree(tree)
            ###

            tikz_code = latex_tree_get_forest(vistree)  # generate the small forest inputs

            # save a ready-to-use tex file with all pareto trees
            forest_grouped.append(latex_get_forest_title(parsim, meta['fitness_train'], tikz_code, tree_sep))

        latex_full_doc = latex_complete_tree_summary(forest_grouped)
        # self.file_locs = {'agents_trees.tex': 'agents/agents_trees.tex'}
        path_trees_tex = file_make_dir(root_path / self.file_locs['trees_tex'])
        with Path.open(path_trees_tex, 'w') as file:
            file.write(latex_full_doc)

        self.printpl('f', '{}'.format(path_trees_tex))

        return

    def file_generate_pycode(self, pareto, root_path):
        """

        """

        if self.env_variables['action_at'][0]['type'] == 'int':
            action_min, action_max = self.env_variables['action_at'][0]['minmax']
            action_min, action_max = int(action_min), int(action_max)
            py_return_action = 'return max({}, min({}, int(round(action))))\n'.format(action_min, action_max)
        else:
            py_return_action = 'return action\n'

        py_operations_assign = '{} = input\n'.format(', '.join(self.env_variables['obs_name']))
        py_decide_body = '{}{{}}\n{}'.format(py_operations_assign, py_return_action)
        py_decide = 'def decide(self, input):\n{}\n'.format(textwrap.indent(py_decide_body, '\t'))
        py_class_code = 'class {{}}:\n\n{}'.format(textwrap.indent(py_decide, '\t'))

        all_agents = []
        all_agent_names = []
        for parsim, meta in sorted(list(pareto.items())):
            expr_raw = meta['expr_raw']
            expr_sym = expr_sympify(expr_raw)
            # label_list_sym = ast_convert_from_expr(expr_sym, build=True)
            # tree = TEST_karoo_tree_from_labellist(label_list_sym, self.env_variables)
            ptree = karoo_ptree_from_expr(expr_sym, self.env_variables)
            tree = ptree.get_uninstanced_tree()
            py_action = 'action = {}'.format(tree_get_pycode(tree))

            py_agent_name = '{}{:.0f}'.format(self.name, parsim)
            all_agent_names.append(py_agent_name)
            all_agents.append(py_class_code.format(py_agent_name, py_action))

        pycode_names = 'all_agents = [{}]\n'.format(', '.join(all_agent_names))
        py_agent_tuples = 'agent_tuples = [{}]\n'.format(
            ', '.join(['(\'{}\', {}())'.format(x, x) for x in all_agent_names]))

        pycode_agents = '{}'.format('\n'.join(all_agents))

        pycode_complete_agents = 'import math\n\n' \
                                 '{}\n\n' \
                                 '{}\n' \
                                 '{}\n'.format(pycode_agents, pycode_names, py_agent_tuples)

        pth = file_make_dir(root_path / self.config['file_locs']['file_pycode'])
        with Path.open(pth, 'w') as file:
            file.write(pycode_complete_agents)
            self.printpl('ff', '{}'.format(pth))

        self.call_custom_mountaincar_file(root_path, pycode_complete_agents)  # sfeh root path is instance variabel

        return

    def call_custom_mountaincar_file(self, root_dir, pycode_complete_agents):

        # sfeh remove this (?)

        if Path.is_file(Path.cwd() / self.file_locs.get('pycode_load')):
            #  if direct execution is wished...# exec(Path.open("custom_eval_agents.py").read())

            # auto_import_eval = 'import sys\n' \
            #                    'from pathlib import Path\n' \
            #                    'sys.path.append(Path({}))\n' \
            #                    'import {} as custom_eval_agents\n' \
            #                    'custom_eval_agents.eval_agent_list(agent_tuples, folder=Path(\'img\'))'.format(pycode_load, Path(pycode_load).stem)

            auto_import_eval = ''

            # 'from benchmarks.gym_mountaincar.agents.mtc_agent_sarsa import *\n\n' + \

            executable_python_evaluation = 'import sys\n' + \
                                           'from pathlib import Path\n' \
                                           'sys.path.append(str(Path(\'' + str(self.sfeh_plagih_root.absolute().as_posix()) + '\')))\n' + \
                                           'from benchmarks.gym_mountaincar.agents.quick_eval import *\n' + \
                                           pycode_complete_agents + '\n' + \
                                           'from pathlib import Path\n' + \
                                           'folder = Path.cwd() / \'custom_files\'\n\n' + \
                                           auto_import_eval + '\n' + \
                                           'from benchmarks.gym_mountaincar.agents.mtc_agent_sarsa import * \n' + \
                                           'with Path.open(Path(\'' + str(Path(self.sfeh_plagih_root / 'benchmarks/gym_mountaincar/agents/sarsa_agent_200.p').absolute().as_posix()) + '\'), \'rb\') as file:\n' + \
                                           '\tsarsa_agent = pickle.load(file)\n\n' + \
                                           'if __name__ == \'__main__\':\n' + \
                                           '\tprint(\'executing!\')\n' + \
                                           '\teval_agent_list(agent_tuples, folder=folder, goal_agent=sarsa_agent)\n'

            with Path.open(root_dir / self.file_locs['file_pycode_eval'], 'w') as file:
                file.write(executable_python_evaluation)
                self.printpl('f', '{}'.format(self.file_locs['file_pycode_eval']))

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Population specific                       +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_eval_remaining(self):

        """
        Evaluate all trees in population_tmp_eval.
        This is the part of the population that could not be found in the dict.
        - evaluate tree fitness
        - store fitness in tree, update the dictionary with known trees
        - append tree to the population

        ...if anything in the try-block fails, the tree will not be appended to the population
        """

        eval_fails = []

        for tree in self.population_tmp_eval:

            try:
                fitness_train = self.tree_eval_fitness_train(tree)
            except Exception as ex:
                print_warning('wwww', 'Exception while evaluating: {}'.format(ex), print_type=self.print_type)
                eval_fails.append(str(ex))
                continue
            else:
                tree = tree_set_fitness(tree, fitness_train)
                self.treelut_tree_add(tree, fitness_train=fitness_train)
                last_evolution = tree_get_last_evolution(tree)
                self.pop_append(tree, last_evolution=last_evolution)

        if len(eval_fails) > 0:
            print_warning('www', 'Evaluating {} new trees in gen {} caused {} exceptions:\n{}'.format(
                len(self.population_tmp_eval), self.gen_id, len(eval_fails), ', '.join(eval_fails)), print_type=self.print_type)

        return

    def origin_tree_get(self):
        """
        Safely return an origin_meta tree
        """
        if self.origin_exists():
            tree_origin = self.origin_tree.copy()
        else:
            tree_origin = None
        return tree_origin

    def treelut_tree_add(self, tree, fitness_train=None, parsimony=None, expr_raw=None, expr_sym=None):
        """
        update selected values in self.tree_meta
        LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        """
        if not fitness_train:
            fitness_train = tree_get_fitness(tree)
        if not parsimony:
            parsimony = tree_get_parsimony(tree)
        if not expr_raw:
            expr_raw = tree_get_expr_raw(tree, node_id=root_id)
        if not expr_sym:
            expr_sym = expr_sympify(expr_raw)

        meta = {'fitness_train': fitness_train,
                'parsimony': parsimony,
                'expr_raw': expr_raw,
                'expr_sym': expr_sym}

        tree_ident = tree_hash(tree)

        self.tree_lut[tree_ident] = meta
        return

    def tree_eval_parsimony_easywrapper(self, tree):
        """
        loading the two extra parameters every time is just boring
        """
        parsimony = tree_eval_parsimony(tree, self.config['complexity_measure'], origin_tree=self.origin_tree_get())
        return parsimony

    def pop_add_tree_midrun(self, tree):
        """
        Trying to add another tree to the current population
        - evaluate the tree
        - append to population
        - update gene_pool
        -
        """
        self.printpl('i', 'Trying to add tree mid-run...')

        tree = self.tree_finish(tree, last_evolution='par-s')

        if not tree_check_deep(tree, self.env_variables):
            self.printpl('w', 'todo: remove w, tree could not be added midrun.')
            return

        parsimony = self.tree_eval_parsimony_easywrapper(tree)
        tree = tree_set_parsimony(tree, parsimony)
        self.treelut_tree_add(tree, parsimony=parsimony)
        self.population_tmp_done.append(tree)

        self.pareto_update_insert()
        return

    def pareto_update_insert(self):
        """
        update new entries in the pareto dict (part of the whole update process)
        Requires: self.parsimony_best_meta entries
        """

        self.parsimony_best_update()

        sorted_parsimony_best = sorted(self.parsimony_best_meta.items(), key=lambda x: x[0])
        try:
            best_fit = next(iter(sorted_parsimony_best))[1]['fitness_train']  # [1] accesses the meta, ['fitness_train'] the fitness
        except StopIteration:
            best_fit = None

        for key, meta in sorted_parsimony_best:  # tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
            fitness = meta['fitness_train']
            parsim = meta['parsimony']
            pareto_improved = None

            if self.kernel.fitness_compare(fitness, best_fit):
                if self.pareto.get(parsim):
                    pareto_fit = self.pareto.get(parsim)['fitness_train']
                    if self.kernel.fitness_compare(fitness, pareto_fit):
                        self.pareto[parsim] = meta
                        self.printpl('a', 'Pareto update at {}, with new {}-error: {}. Old was: {}'.format(
                            parsim, self.config['kernel_name'], fitness, best_fit))
                        pareto_improved = True
                else:
                    self.pareto[parsim] = meta
                    self.printpl('a', 'New pareto entry at {:.0f} with {}-error: {:4.2f}'.format(parsim, self.config[
                        'kernel_name'], fitness))
                    pareto_improved = True
                best_fit = fitness
            if pareto_improved:
                expr_raw = meta['expr_raw']  # expy_sym will can cause exceptions while setting fix nodes
                ptree = karoo_ptree_from_expr(expr_raw, self.env_variables)
                tree = ptree.get_uninstanced_tree()
                tree = tree_set_modifyable_nodes(tree, origin_tree=self.origin_tree_get())
                sym_tree = tree_evolve_reduce(tree, self.env_variables, completely=True)
                if list(tree_get_labellist(sym_tree)) != list(tree_get_labellist(tree)):
                    if len(list(tree_get_labellist(sym_tree))) > len(list(tree_get_labellist(tree))):
                        print_warning('w', 'Sympified tree is larger than raw version?')
                    self.printpl('aa', 'Pareto entry could be further sympified!')
                    sym_tree = tree_set_fitness(sym_tree, fitness)
                    self.pop_add_tree_midrun(sym_tree)
                else:
                    print_warning('iii', 'The found pareto entry was already in its sympified version')
        return

    def pareto_update_clean(self):
        """
        remove superfluous pareto entries
        """
        sorted_pareto = sorted(self.pareto.items(), key=lambda x: x[0])
        try:
            last_fitness = copy.deepcopy(next(iter(sorted_pareto))[1]['fitness_train'])
        except StopIteration:
            last_fitness = None  # todo is None actually correct here?
        for parsim, meta in sorted_pareto[1:]:
            fitness = meta['fitness_train']
            if self.kernel.fitness_compare(fitness, last_fitness):
                last_fitness = fitness
            else:
                self.pareto.pop(parsim)
                self.printpl('aa', 'Pareto entry at {} became obsolete. Its fitness: {} was surpassed by simpler entry with fitness: {}.'.format(
                    parsim, last_fitness, fitness))
        return

    def todo_update_pareto(self, pareto, candidates):

        """
        kernel_fun = min() todo
        todo
        """
        pareto = []
        candidates = [(20, 2000), (5, 900), (5, 800), (10, 700), (7, 800)]

        if not pareto:  # no pareto entries found yet
            pareto.append(candidates[0])  # print('First pareto entry')

        for entry in candidates:
            try:
                p_simpler = [p for p in pareto if p[0] <= entry[0]]
                best = min(p_simpler, key=lambda p: p[1])
            except:
                best = min(pareto, key=lambda p: p[1])  # fittest pareto entry

            if entry[1] < best[1]:
                pareto.append(entry)  #
                pareto = [x for x in pareto[:] if x[0] < entry[0] or x[1] < entry[1] or x is entry]
                pareto.sort(key=lambda x: x[0])  # as far as I can tell, not really necessary without using iter()
            else:
                continue  # Not a pareto entry

        # print('Pareto', '\t\t', pareto)
        return

    def pareto_update(self):
        """
        Builds up the pareto front
        1. adds new entries from parsimony_best_meta
        2. deletes entries that are now obsolete


        """

        self.pareto_update_insert()
        self.pareto_update_clean()

        return

    def pareto_update_try(self):
        """
        sfeh tbd #
        """
        for i, tree in enumerate(self.population_tmp_done):
            fitness = tree_get_fitness(tree, precision=self.precision)
            parsimony = tree_get_parsimony(tree)
            meta = tree_get_meta(tree)
            for p_fit, p_meta in self.pareto.items():
                p_parsim = p_meta['parsimony']
                if self.kernel.fitness_compare(fitness, p_fit):
                    if parsimony < p_parsim:
                        # Found a new entry on pareto
                        # 1. insert new entry
                        self.pareto[parsimony] = meta
                        # 2. clean pareto
                        self.pareto_update_clean()

                    else:
                        # pareto is already sufficient
                        break

    def parsimony_best_update(self):
        """
        Updates a list with the best candidates for each parsimony.
        Do not confuse with pareto-entries
        """

        for ii, tree in enumerate(self.population_tmp_done):

            parsim = tree_get_parsimony(tree)
            fitness_train = tree_get_fitness(tree)

            # 3. is the tree better than the current best at this parsimony dim_y?
            if parsim in self.parsimony_best_meta:
                comp_fit = self.parsimony_best_meta[parsim]['fitness_train']
                if self.kernel.fitness_compare(fitness_train, comp_fit):
                    self.printpl('iii', 'Found a better candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                    self.parsimony_best_meta[parsim] = tree_get_meta(tree)
            else:
                self.printpl('iii', 'Found a new candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                self.parsimony_best_meta[parsim] = tree_get_meta(tree)

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   What happens in a Generation              +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gen_reset_parameters(self):
        """
        Sets the parameters for the generation
        - reset population_tmp_done
        - Linearly increase threshold for parsimony
        """

        self.time_genstart = time.perf_counter()
        self.population_tmp_done = []
        self.population_tmp_eval = []
        # self.parsimony_tmp = max(1 / min(self.gen_id, self.config['gen_num_max_parsimony']) * self.parsimony_max,
        #                          self.parsimony_max)

        return

    def pop_reproduce(self, call_params, tree):

        """

        copy a tree from the last population without changing its outcome
        """
        if call_params.get('sympify_tree'):
            tree = tree_evolve_reduce(tree, self.env_variables, completely=False)

        return tree

    def pop_reproduce_olymp(self):

        """
        Copy an entry from the pareto candidates into the population
        """

        if self.parsimony_best_meta:
            meta = np.random.choice(list(self.parsimony_best_meta.values()))
            expr_raw = meta['expr_raw']
            label_list = ast_convert_from_expr(expr_raw, build=True)
            xtype_list = xtypes_from_labels(label_list, self.env_variables)
            p_tree = Ptree_karoo(label_list, xtype_list)
            tree = p_tree.get_uninstanced_tree()
        else:
            tree = None

        return tree

    def pop_mutate_point(self, tree):

        """
        Point mutation, One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """
        tree = tree_evolve_mutate_point(tree, self.config['float_accuracy'], self.choose_oparray2, self.env_variables,
                                        self.choose_distributions)

        return tree

    def pop_mutate_filter(self, call_params, tree):
        """
        Mutates a number of float terminal of a tree
        """
        mode = call_params['mode']  # point/branch/all
        mutate_filter = 'gaussian_filter'  # sfeh change?

        node_ids = tree_get_mutatable_nodes(tree)
        if mode == 'branch':
            node_id = np.random.choice(node_ids)  # todo should this be completely random?
            node_ids = tree_node_get_branch(tree, node_id)  # select the whole branch

        float_nodes = []
        variables_node = []
        for node_id in node_ids:
            if tree_node_get_xtype(tree, node_id) == '2f':
                try:
                    _ = float(tree_node_get_label(tree, node_id))
                    float_nodes.append(node_id)
                except ValueError:
                    variables_node.append(node_id)

        if float_nodes:
            if mode == 'point':
                float_nodes = [np.random.choice(float_nodes)]
            for node_id in float_nodes:
                val = float(tree_node_get_label(tree, node_id))
                val = gp_mutate_constants(val, term_type='float', filter_type=mutate_filter, float_accuracy=self.config['float_accuracy'])
                tree = tree_node_set_label(tree, node_id, val)

        # todo this should probably be done somewhere else...
        if variables_node:  # 'filtering' variables when they are from different times
            for nodeobs_id in variables_node:
                obs_name = tree_node_get_label(tree, nodeobs_id)
                core_label = envvariable_get_corelabel(obs_name)
                var_list = self.env_variables['env_observation_family'].get(core_label)
                if var_list is not None:
                    new_obs = random_choose_tempobs(var_list)
                    tree = tree_node_set_label(tree, nodeobs_id, new_obs)

        else:
            # print_warning('iii', 'Tree does not seem to have any float nodes for filtering.')  # usually happens with point-filtering
            pass

        return tree

    def invent_label_list(self, size_mode, first_xtype, build_size, full_or_grow, float_accuracy):
        """
        Creates a random label list
        """
        if 'depth' in size_mode:
            # sfeh warning: Attention with this one. can get quite large with depth based
            label_list, arity_list, xtype_list = invent_label_list_depth(first_xtype, build_size, float_accuracy,
                                                                         self.env_variables, self.choose_oparray2,
                                                                         self.choose_distributions,
                                                                         full_or_grow=full_or_grow)

        elif 'nodes' in size_mode:

            label_list, arity_list, xtype_list = invent_label_list_nodes(first_xtype, build_size, float_accuracy,
                                                                         self.env_variables, self.choose_oparray2,
                                                                         self.choose_distributions,
                                                                         full_or_grow=full_or_grow)
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
            max_dummy = self.config['depth_max']
        elif 'nodes' in size_mode:
            max_dummy = self.config['parsimony_max']
        else:
            raise
        if mean_min_max_var[2] is None:
            mean_min_max_var[2] = max_dummy
        else:
            mean_min_max_var[2] = min(mean_min_max_var[2], self.config['parsimony_max'])
        mean_min_max_var = tuple(mean_min_max_var)

        full_or_grow = build_spec['full_or_grow']

        return build_spec, size_mode, mean_min_max_var, full_or_grow

    def pop_random_from_origin_fix(self, call_params, origin_tree):
        """
        insert a (random) number of branches at the first possible "layer"
        (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
        - get these nodes, randomly choose a subset of those
        - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
        - split the amount of nodes up (randomly) and add these new branches to the tree
        """

        # tree_origin = self.origin_tree_get()origin_tree
        build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)

        # tree_base = tree.copy()
        # ('We are about to create new branches randomly at nodes {}.'.format(layer0_ids))
        layer0_ids = tree_get_mutatable_layer(origin_tree, 0)

        build_split = []
        if 'depth' in size_mode:
            # print_warning('ww', 'notagoodidea.origin adding depths without a plan? sfeh sfeh')
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

            label_list, arity_list, xtype_list = self.invent_label_list(size_mode, first_xtype, build_size, full_or_grow, self.config['float_accuracy'])

            c_core = Core_From_Labels(label_list, arity_list, xtype_list)
            core = c_core.get_uninstanced_core()
            tree = tree_insert_subtree(tree, core, old_branch, karoo=True)

        return tree

    def pop_random(self, call_params):
        """
        Creates completely random trees from scratch
        """

        build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)
        action_xtype = self.env_variables['action_at'][0]['xtype']
        build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')  # sfeh anderer name für branch

        label_list, arity_list, xtype_list = self.invent_label_list(size_mode, action_xtype, build_size, full_or_grow,
                                                                    self.config['float_accuracy'])

        p_tree = Ptree_karoo(label_list, xtype_list, arity_list=arity_list)
        tree = p_tree.get_uninstanced_tree()

        return tree

    def pop_mutate_branch(self, call_params, tree):

        """
        Mutate branch of a tree.

        todo read this, care about same branch size
        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each old_node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """

        build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)

        full_or_grow = build_spec.get('full_or_grow')
        if full_or_grow is None:
            full_or_grow = np.random.choice(['full', 'grow'])

        node_ids = tree_get_mutatable_nodes(tree, no_root=True)
        old_node = np.random.choice(node_ids)
        old_xtype = tree_node_get_xtype(tree, old_node)
        build_size = choose_build_size(size_mode, mean_min_max_var, tree=tree, node_id=old_node)

        label_list, arity_list, xtype_list = self.invent_label_list(size_mode, old_xtype, build_size, full_or_grow,
                                                                    self.config['float_accuracy'])

        if label_list:
            # core_insert = core_from_labels(label_list, arity_list, xtype_list)

            c_core = Core_From_Labels(label_list, arity_list, xtype_list)
            core_insert = c_core.get_uninstanced_core()

            branch_nodes_ids = tree_node_get_branch(tree, old_node, karoo=True)
            tree = tree_insert_subtree(tree, core_insert, branch_nodes_ids, karoo=True)
            tree = tree_prune_depth(tree, self.config['tree_depth_max'], self.env_variables, self.choose_distributions,
                                    self.config['float_accuracy'])
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
            print_warning('ww', 'Crossover conversion between trees not possible: \n{}\n{}'.format(left_tree, right_tree))
            return None, None

        left_core = core_from_labels(left_labels, left_aritys, left_xtypes)
        right_core = core_from_labels(right_labels, right_aritys, right_xtypes)

        left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
        left_offspring = tree_prune_depth(left_offspring, self.config['tree_depth_max'], self.env_variables,
                                          self.choose_distributions, self.config['float_accuracy'])

        right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
        right_offspring = tree_prune_depth(right_offspring, self.config['tree_depth_max'], self.env_variables,
                                           self.choose_distributions, self.config['float_accuracy'])

        return left_offspring, right_offspring

    def tree_finish(self, tree, last_evolution=''):
        """
        The np-tree needs more information than only the expression.
        -> set modifyable nodes (mandatory)

        -> round all distributions_file
        -> try to envstate_normalize exponents ('**'). sfeh, not really working.
        -> set last evolution (for analysing gp operators. e.g. if no good trees originate from crossover, something might be wrong)

        ! The tree must exist, it is not checked whether the tree is None
        """

        # if tree is None:
        #     print_warning('ww', 'Tree from last_evolution: \'{}\' failed. Continuing.'.format(last_evolution))  # reasons: sympify, last tree too large
        # else:

        tree = tree_set_modifyable_nodes(tree, origin_tree=self.origin_tree_get())  # sfeh: somewhere else
        tree = tree_normalize_exponentiation(tree)  # sfeh: somewhere else
        tree = tree_set_last_evolution(tree, last_evolution)  # sfeh: should this be done during the evolve process?

        return tree

    def pop_base_transfer(self):
        """

        """
        self.population_base = []

        for i, tree in enumerate(self.population_tmp_done):
            tree_copy = np.copy(tree)
            # tree_copy = tree_set_id(tree_copy, i)  # sfeh delete this?
            self.population_base.append(tree_copy)

        return

    def pop_append(self, tree, last_evolution=''):
        """
        Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the tree is refurbished.
        sfeh: if trees are 100% safely created, tree_check_deep() must not be used. Useful when trying out new gp-operators.
        - Enrich the raw tree for the next generation
        - check if the tree is actually valid
        ->
        """

        if not tree_check_quick(tree):
            return

        tree = self.tree_finish(tree, last_evolution=last_evolution)

        tree_ident = tree_hash(tree)
        if tree_ident in self.tree_lut:
            tree_meta = self.tree_lut[tree_ident]  # tree_hash: fitness_train, parsimony, expr_sym, expr_raw
            parsimony = tree_meta['parsimony']
            fitness_train = tree_meta['fitness_train']
            tree = tree_set_parsimony(tree, parsimony)
            tree = tree_set_fitness(tree, fitness_train)

            self.population_tmp_done.append(tree)
        else:
            # sfeh idea: evaluate already when the trees are constructed? evaluating later seems kind of dull
            parsimony = self.tree_eval_parsimony_easywrapper(tree)
            if parsimony <= self.parsimony_max:
                tree = tree_set_parsimony(tree, parsimony)
                tree = tree_set_fitness(tree, '')
                self.population_tmp_eval.append(tree)
            else:
                print_warning('wwww', 'Parsimony too high, last evolution: {}'.format(last_evolution), print_type=self.print_type)

        return

    def gen_finalize(self):

        """
        From raw population_tmp_done
        - Evaluate leftover

        """

        self.pop_eval_remaining()
        self.pareto_update()
        self.pop_base_transfer()
        self.pop_analyse()
        # write_file_population_karoo(self.population_tmp_done, 'last', self.root_dir, self.gen_id, print_type=self.print_type)  # better in periodic file write

        self.monitoring_dict['total_found_trees'][self.gen_id] = len(self.tree_lut)
        gen_time = time.perf_counter() - self.time_genstart
        if delete_this and self.monitoring_dict.get('gen_time') is None:
            self.monitoring_dict['gen_time'] = {}
        self.monitoring_dict['gen_time'][self.gen_id] = gen_time
        self.print_g('gg', 'Created {}/{} unique trees in generation {}. Gen took {:4.2f}s'.format(
            len(self.population_tmp_done), self.config['pop_max'], self.gen_id,
            time.perf_counter() - self.time_genstart))
        # todo why is gen_time going up with generations? why does ib take so much longer than mc?
        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Perform the 3 genetic prog. operations    +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_selection_tournament(self, tourn_size):

        """
        config-selection. takes a number of trees (we use 3) and returns the best one (winner)

        """

        best_id = None
        best_fitness = None

        for n in range(tourn_size):

            tree_id = pop_tree_choose(self.population_base)
            tree = self.population_base[tree_id]

            fitness = tree_get_fitness(tree, precision=self.config['fitness_accuracy'])

            if self.kernel.fitness_compare(fitness, best_fitness, mode='better'):
                best_id = tree_id
                best_fitness = fitness

        tourn_winner = copy.deepcopy(self.population_base[best_id])

        return tourn_winner

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def activate_origin_tree(self, ptree: Ptree_karoo):
        """
        The origin tree (which was already loaded) gets activated for its use in the GP-process
        """
        tree = ptree.get_uninstanced_tree()
        if not tree_check_deep(tree, self.env_variables):
            raise

        expr_raw = tree_get_expr_raw(tree, node_id=root_id)

        try:
            expr_sym = expr_sympify(expr_raw=expr_raw)
        except Exception as ex:
            raise Exception('Your tree\'s algorithm could not be sympified. excep: {}'.format(ex))

        # sfeh, this does not work
        # if not tree_check_is_sympified(tree):
        #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
        #                          ''.format(expr_raw, expr_sym))

        self.origin_tree = copy.deepcopy(tree)

        fitness_train = self.tree_eval_fitness_train(tree)

        self.origin_meta = {'expr_raw': expr_raw,
                            'expr_sym': expr_sym,
                            'parsimony': 0,
                            'fitness_train': fitness_train}

        self.parsimony_best_meta[0] = self.origin_meta
        self.pareto[0] = copy.deepcopy(self.origin_meta)

        self.print_g('gg', 'Loading origin_meta, fitness {}. Time: {:4.2f}s'.format(fitness_train,
                                                                                    time.perf_counter() - self.time_start))

        return

    def tree_eval_fitness_train(self, tree):
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
            expr_sym = tree_get_expr_sym(tree)
        except Exception as ex:
            raise Exception('eval:{}'.format(ex))

        fitness_train = eval_tf(expr_sym,
                                self.data_train,
                                self.kernel,
                                self.env_variables,
                                self.tf_config, self.tf_device, self.tf_classify_labels_map, specific_action=self.config['eval_action'])

        if not check_value_is_real(fitness_train):
            raise Exception('Fitness is inf or nan: {}'.format(fitness_train))  # happens, eg when values are soo wrong that it leaves the float-range

        fitness_train = round(fitness_train, self.config['fitness_accuracy'])

        return fitness_train

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to use evaluate (tensorflow)      +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def tf_classify_labels_map(self, result):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the samples-csv.
        Outputs an array of tuples containing the predicted labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.unique_outputs_num / 2) - 1 # '-1' keeps a binary classification splitting over the origin_meta
            if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
            elif solution == self.unique_outputs_num - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
            else: fitness = 0 # no class match

        """
        unique_outputs_num = self.env_variables['action_at'][0]['unique_outputs_num']
        skew = (unique_outputs_num / 2) - 1
        label_rules = {unique_outputs_num - 1: (
            tf.constant(unique_outputs_num - 1), tf.constant(' > {}'.format(unique_outputs_num - 2 - skew)))}

        for class_label in range(unique_outputs_num - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tf.cond(cond, lambda: (
                tf.constant(class_label), tf.constant(' <= {}'.format(class_label - skew))),
                                               lambda: label_rules[class_label + 1])

        pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))), lambda: label_rules[1])

        return pred_label

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def file_all_plots(self, root_path, subfolder=''):
        """
        Make all plots
        'fitness_average'
        'population_tmp_done-size'
        'complexity_average'
        'total_found_trees'
        'pareto dominant candidates'
        'tmp_pop_fitness_distribution'
        """

        path_plots = folder_make_dir(root_path / self.file_locs['folder_plots'] / subfolder)

        def plotendify_me(data_dict):
            # sfeh delete this version1 (working only with np-arrays, no dicts)
            npdata_dict = np.array([list(x) for x in sorted(data_dict.items())]).T
            return npdata_dict

        def get_pareto_plot_values():
            # sfeh i think there is a more beautiful solution?
            tuples = []
            for key in sorted(self.pareto):
                tuples.append([key, self.pareto[key]['fitness_train']])
            npdata_dict = np.array(tuples).T
            return npdata_dict

        if self.monitoring_verbosity['gen_fitness_average'] == 'y':
            data_tuples = plotendify_me(self.monitoring_dict['fitness_average'])
            plot_end(data_tuples, path_plots, plt_title='average error', plt_x_label='Generation', plt_y_label='fitness',
                     set_left=data_tuples[0][0])

        if self.monitoring_verbosity['population_tmp_done-size'] == 'y':
            data_tuples = plotendify_me(self.monitoring_dict['population_tmp_done-size'])
            plot_end(data_tuples, path_plots, plt_title='genepool size', plt_x_label='Generation', plt_y_label='amount', linestyle='None',
                     marker='.',
                     set_left=data_tuples[0][0])

        data_tuples = plotendify_me(self.monitoring_dict['gen_time'])
        plot_end(data_tuples, path_plots, plt_title='Generation time', plt_x_label='Generation', plt_y_label='time (sec)', linestyle='None',
                 marker='.',
                 set_left=data_tuples[0][0])

        data_tuples = get_pareto_plot_values()
        plot_end(data_tuples, root_path, plt_title='pareto dominant candidates', plt_x_label='parsimony',
                 plt_y_label='fitness',
                 linestyle='dashed',
                 marker='.',
                 step_where='post',
                 set_right=self.parsimony_max,
                 beyond_lines=True,
                 save_tikz=True)

        if self.monitoring_verbosity.get('fitness_variance') == 'y':
            data_tuples = plotendify_me(self.monitoring_dict['fitness_variance'])
            plot_end(data_tuples, path_plots, plt_title='variance in error', plt_x_label='Generation', plt_y_label='variance',
                     marker='')

        data_tuples = plotendify_me(self.monitoring_dict['complexity_average'])
        data_tuples_variance = plotendify_me(self.monitoring_dict['complexity_variance'])  # sfeh update to standard error
        plot_end(data_tuples, path_plots, plt_title='tree complexity (avg and std. error)', plt_x_label='Generation', plt_y_label='variance',
                 marker='', fill_variance=data_tuples_variance)

        data_tuples = plotendify_me(self.monitoring_dict['best_candidate'])
        plot_end(data_tuples, path_plots, plt_title='best candidate', plt_x_label='Generation',
                 plt_y_label='error',
                 linestyle='dashed',
                 step_where='post')

        # data_tuples = plotendify_me(self.monitoring_dict['total_found_trees'])
        # plot_end(data_tuples, path_plots, plt_title='number of created trees', plt_x_label='Generation', plt_y_label='amount', linestyle='None',
        #          marker='.',
        #          set_left=data_tuples[0][0])

        # data_tuples = plotendify_me(self.monitoring_dict['complexity_average'])
        # plot_end(data_tuples, path_plots, plt_title='average tree complexity', plt_y_label='#nodes',
        #          set_left=data_tuples[0][0])
        #
        # data_tuples = plotendify_me(self.monitoring_dict['complexity_variance'])
        # plot_end(data_tuples, path_plots, plt_title='variance in complexity', plt_y_label='variance',
        #          marker='')

        # data_tuples = np.array([list(x) for x in self.monitoring_dict['tmp_pop_fitness_distribution']]).T  # sfeh delete list version1 [(1,234),(2,544), ..]
        # plot_end(data_tuples, path_plots, plt_title='population distribution Gen {}'.format(self.gen_id),
        #          plt_x_label='tree ids',
        #          plt_y_label='fitness',
        #          marker='',
        #          set_right=self.config['pop_max'],
        #          right_padding=1,
        #          subfolder=folder_pop_analysis)

        # sfeh https://github.com/linkedin/naarad/issues/114 UserWarning: Attempting to set identical bottom==top results

        return

    def pop_analyse(self):
        """
        Analysing this generation
        - amount of trees
        - fittest tree
        - average fitness
        - average tree complexity
        """

        # How many survived in the selection?
        self.monitoring_dict['population_tmp_done-size'][int(self.gen_id)] = len(self.population_tmp_done)
        if len(self.population_tmp_done) <= 0:
            self.terminate_run(self.root_dir)

        # Find the fittest + average fitness
        pop_best_fitness = tree_get_fitness(self.population_tmp_done[FIRST_TREE])
        fitness_train_sum = 0
        tree_cnt = 0
        pop_tree_analysis = []
        for ii, tree in enumerate(self.population_tmp_done):
            fitness = tree_get_fitness(tree)
            complexity = tree_get_parsimony(tree)
            last_evolve = tree_get_last_evolution(tree)
            pop_tree_analysis.append({'fitness': fitness, 'complexity': complexity, 'last_evolve': last_evolve})

            fitness_train_sum += fitness  # for fitness average
            tree_cnt += 1
            if self.kernel.fitness_compare(fitness, pop_best_fitness):
                pop_best_fitness = fitness
        average_fitness = fitness_train_sum / max(tree_cnt, 1)
        self.monitoring_dict['fitness_average'][self.gen_id] = average_fitness
        if self.best_fitness is None:
            self.best_fitness = pop_best_fitness
        else:
            if self.kernel.fitness_compare(pop_best_fitness, self.best_fitness):
                self.best_fitness = pop_best_fitness
        self.monitoring_dict['best_candidate'][self.gen_id] = self.best_fitness

        # Tree fitness distribution
        # dist_fit = [[i, x] for i, x in enumerate([x['fitness'] for x in pop_tree_analysis])]
        # self.monitoring_dict['tmp_pop_fitness_distribution'] = dist_fit

        # Tree complexity
        complexity_sum = 0
        for ii, tree in enumerate(self.population_tmp_done):
            complexity_sum += len(tree_nodes_get_ids(tree, karoo=True))
        avg_complexity = complexity_sum / tree_cnt
        self.monitoring_dict['complexity_average'][self.gen_id] = avg_complexity

        # fitness variance
        todo = [x['fitness'] for x in pop_tree_analysis]
        fitness_variance = np.var(todo)
        self.monitoring_dict['fitness_variance'][self.gen_id] = fitness_variance

        # complexity standard-error
        if delete_this and self.monitoring_dict.get('pop:trees:complexity:std_error') is None:
            self.monitoring_dict['pop:trees:complexity:std_error'] = {}  # sfeh delete this version1
        self.monitoring_dict['pop:trees:complexity:std_error'][self.gen_id] = np.std([x['complexity'] for x in pop_tree_analysis])
        self.monitoring_dict['complexity_variance'][self.gen_id] = np.var([x['complexity'] for x in pop_tree_analysis])  # sfeh version1 delete this shit

        return

    def get_env_variables(self):
        """
        xtypes_list is required to build trees
        this helps creating it
        """
        return self.env_variables

    def terminate_run(self, root_dir):
        """
        Program is done after writing all gp_files one last time.
        """
        self.file_save_files(root_dir)

        # if Path.is_file(root_dir / file_pycode_eval):
        #     # exec(Path.open(root_dir / file_pycode_eval).read())
        #     os.system('python ' + str(root_dir / file_pycode_eval))  # sfeh nothing to be proud of

        self.file_all_plots(root_dir)
        self.print_g('gg', ' Terminating. \tTime since start: {:4.2f}s'.format(time.perf_counter() - self.time_start))

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to print_type output information  +
    # +++++++++++++++++++++++++++++++++++++++++++++

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

    build_size = int(np.random.normal(mean, size_variance))
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
