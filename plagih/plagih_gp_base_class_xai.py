"""
'f2f', 'b2b', f = float, b = bool. 'float to bool'

Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""
import time

import math
import matplotlib.pyplot as plt
from plagih.file_interaction import *
from plagih.viz_with_latex import *
from plagih.plagih_config import *
from plagih.plagih_data import *
from plagih.Ptree2 import *
import random

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


class FileLocations:
    folder_plots = 'plots/'
    folder_histograms = 'agents/'

    file_backup_pickle = 'backup/backup.p'
    file_backup_yaml = 'backup/backup.yaml'

    trees_tex = 'agents_trees.tex'
    folder_pycode = 'agents/'

    file_pareto = 'info/paretofront.yaml'
    info_config_yaml = 'info/config.yaml'

    samples_ready_p = 'run_files/samples_ready.p'
    samples_csv = 'run_files/samples.csv'
    distributions_file = 'run_files/distributions_file.yaml'


class ExplainableGP(object):
    """

    """

    def __init__(self, conf: GpConfig, root_dir, path_data_csv, path_origin_tree, developer_fix=None):
        self.conf = conf
        self.root_dir = Path(root_dir)

        self.developer_fix = developer_fix  # sfeh

        print(f'\n'
              f'\tInitializing Plagih. \n'
              f'\tName: {BColors.CYAN}{self.root_dir.name}{BColors.RESET}. \n'
              f'\tLocated in: \n'
              f'\t{self.root_dir}\n')
        self.time_start = time.perf_counter()

        self.file_locs = FileLocations()

        self.env_vars = EnvVars()
        self.data_train = None
        self.data_control = None

        self.env_vars, self.data_train, self.data_control = self.activate_dataset(path_data_csv, self.conf.action_name)

        # Evaluating kernel (that uses tensorflow)
        self.tf_config = tf.compat.v1.ConfigProto(log_device_placement=self.conf.tf_device_log,
                                                  allow_soft_placement=True)  # TF device usage logging (for debugging) (default false. I lately used it to check if the GPU is used)
        self.tf_config.gpu_options.allow_growth = True
        # sfeh for now, only regression kernel
        self.origin_results = None
        self.kernel = RegressionKernel(self.conf.kernel_name, self.data_train, self.tf_config, "/gpu:0",
                                       self.env_vars.eval_action)  # sfeh Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device. Is cpu otherwise
        self.print_type = self.conf.print_type

        self.pareto = []  # a dict with all pareto candidates. key is complexity, value is tree meta. [[1,344, meta], ...]
        if path_origin_tree:
            self.origin_cooltree = self.load_origin_tree(path_origin_tree)
            self.origin_is_fix = self.origin_cooltree.core.is_fix
        else:
            self.origin_cooltree = None
            self.origin_is_fix = False

            if self.conf.complexity_measure in ['tree_edit_distance']:  # all origin-based distances  # todo tree_edit_distancev2
                self.conf.complexity_measure = 'tree_node_count'
                print_warning('w', "Complexity measurement 'tree_edit_distance' is not possible without origin! Using 'tree_node_count' instead.", print_type=self.print_type)

        """
        load relevant stuff
        """
        self.choose_distributions = self.activate_distributions(path_distrib=None)  # sfeh path_distrib not None
        self.choose_oparray2 = self.gp_load_oparray()  # path_operators sfeh this file from config version1

        """
        initialize some variables
        """
        # init values with dummies (just to have all self values here for overview)
        self.tree_lut = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.population_tmp = []
        self.pop_base = []  # population that is taken to the next generation
        self.best_fitness = None  # keeps track of the current best fitness
        self.gen_id = 0

        class MonitoringGenerations:
            # sfeh this might be better. maybe pandas?
            def __init__(self):
                self.population_tmp_done_size = {}
                self.fitness_average = {}
                self.fitness_variance = {}
                self.best_candidate = {}
                # 'complexity_list = {},  # not used, just uses memory
                self.complexity_average = {}
                self.complexity_variance = {}  # variance can be deleted, only std-error is needed delete v1
                self.pop_trees_complexity_std_error = {}
                self.gen_time = {}

            def load_old_monitoring_dict(self, monitoring_dict):
                self.population_tmp_done_size = monitoring_dict.get('population_tmp_done-size', {})

                self.fitness_average = monitoring_dict.get('fitness_average', {})
                self.fitness_variance = monitoring_dict.get('fitness_variance', {})
                self.best_candidate = monitoring_dict.get('best_candidate', {})
                # 'complexity_list = monitoring_dict.get('complexity_list', {}),  # not used, just uses memory

                self.complexity_average = monitoring_dict.get('complexity_average', {})
                self.complexity_variance = monitoring_dict.get('complexity_variance', {})  # variance can be deleted, only std-error is needed delete v1
                self.gen_time = monitoring_dict.get('gen_time', {})
                self.evol_performance = monitoring_dict.get('evol_performance', {})

        self.monitor_pd = pd.DataFrame(columns=['pop_len', 'pop_unique',
                                                'fit_avg', 'fit_std', 'fit_best',
                                                'parsim_avg', 'parsim_std',
                                                'complexity_avg',
                                                'time'])

        self.evolve_loop, self.evolve_random = self.make_evolve_rates()

        self.print_g('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return

    def make_evolve_rates(self):
        """

        """

        def evolve_safety_update(evolve_dict):
            """
            Updates tournament size and evolve rates

            Example entry of the list could be:
            {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
            'custom_params': {'build_spec': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
            """

            for tag, evolve_spec in evolve_dict.items():
                evolve_dict[tag]['tourn_size'] = evolve_spec.get('tourn_size', self.conf.tourn_size)

                evolve_rate = evolve_spec.get('evolve_rate')
                evolve_dict[tag]['evolve_num'] = int(evolve_rate * self.conf.pop_max)
                if evolve_spec.get('custom_params') is None:
                    evolve_spec['custom_params'] = {}

            return evolve_dict

        try:
            evolve_loop = self.evolve_list
            self.printpl('i', 'Using evolve rates from config')
        except:

            evolve_loop = {
                # Reproduction (10%)
                'Repro': {'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.06, 'custom_params': {}},
                'Rsympy': {'evolve_name': 'reproduce', 'evolve_rate': 0.03, 'custom_params': {'sympify_tree': True}},
                'Pareto': {'evolve_name': 'revive pareto', 'evolve_rate': 0.01, 'custom_params': {}},

                # Mutation (25%)
                'Point': {'evolve_name': 'mutate point', 'evolve_rate': 0.05, 'custom_params': {}},

                'BranchDF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                             'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8), 'full_or_grow': 'full'}}},
                'BranchDG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                             'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 5, 1), 'full_or_grow': 'grow'}}},
                'BranchNF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                             'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'full_or_grow': 'full'}}},
                'BranchNG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                             'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'full_or_grow': 'grow'}}},
                'BranchShrink': {'evolve_name': 'mutate branch', 'evolve_rate': 0.0,
                                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0), 'full_or_grow': 'grow'}}},

                'FilterBO': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                             'custom_params': {'mode': 'branch', 'filter_observations': True}},
                'FilterB': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                            'custom_params': {'mode': 'branch', 'filter_observations': False}},
                'FilterP': {'evolve_name': 'filter optimize', 'evolve_rate': 0.0, 'tourn_size': 5,
                            'custom_params': {'mode': 'point', 'filter_observations': True}},

                # Crossover (35%)
                'Xover': {'evolve_name': 'crossover branch', 'evolve_rate': 0.30, 'custom_params': {}},  # sum 0.70

                # Leftovers are automatically filled with random trees

                # Random (25%)
                'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.05,
                          'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 3, 5, 1), 'full_or_grow': 'full'}}},
                'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.00,
                          'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'full_or_grow': 'grow'}}},
                'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                          'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (12, 1, None, 5), 'full_or_grow': 'grow'}}},
                'Rand4': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                          'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (12, 1, None, 5), 'full_or_grow': 'full'}}},  # param 'max' can be None
            }

        evolve_loop = evolve_safety_update(evolve_loop)

        if self.origin_is_fix:
            try:
                evolve_random = self.evolve_list_random['from_origin']
            except:
                evolve_random = {'RandO3': {'evolve_name': 'random trees', 'evolve_rate': 1.00,
                                            'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (10, 3, None, 4), 'full_or_grow': 'full'}}}}
        else:
            try:
                evolve_random = self.evolve_list_random['from_scratch']
            except:
                evolve_random = {'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                           'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (3.5, 2, 5, 1), 'full_or_grow': 'full'}}},
                                 'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                           'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 2, 6, 1), 'full_or_grow': 'grow'}}},
                                 'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.40,
                                           'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 3, None, 4), 'full_or_grow': 'full'}}}
                                 }
        evolve_random = evolve_safety_update(evolve_random)

        return evolve_loop, evolve_random

    def file_make_dir_root(self, file):
        """
        Creates the folder only knowing the filekey
        """
        p = self.root_dir / file
        p = file_make_dir(p)
        return p

    def run_backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """

        a_helping_dict = {'old_config': None}  # sfeh save complete config?    # sfeh i dont think we need the config
        run_backup_data = self.gen_id, self.pareto, self.pop_base, self.monitor_pd, a_helping_dict  # sfeh use this later, a_helping_dict

        path_backup = self.file_make_dir_root(self.file_locs.file_backup_pickle)
        with Path.open(path_backup, 'wb') as file:
            pickle.dump(run_backup_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        with Path.open(path_backup, 'wb') as file:
            pickle.dump(run_backup_data, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.printpl('f', f'Backup: {path_backup.as_posix()}')
        return

    def backup_load(self, path_load_backup=None):
        """
        backup_load
        If a backup-file is found...
        """
        path_backup = path_load_backup or self.root_dir / self.file_locs.file_backup_pickle  # sfeh file-load

        if Path.is_file(path_backup):
            self.print_g('g', f'Loading data from backup-file {path_backup}')
            # try:
            """
            Loading the state of the run from the pickle file
            """
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)
            except NotImplementedError as nimp:
                raise Exception(f'NotImplementedError: {nimp}')
            except EOFError as eoferr:
                raise Exception(f'EOFError: \n{eoferr}')

            try:
                self.gen_id, self.pareto, self.pop_base, monitor_pd, a_helping_dict = run_data  # sfeh use a helping dictt a_helping_dict is used for a useable sldifjsdfsdfg , a_helping_dict
                self.monitor_pd = monitor_pd  # sfeh
            except:
                self.gen_id, self.pareto, self.pop_base, m = run_data
                # self.conf.restart_count += 1

                # lelz_dict = {}
                # for skey, sval in m.items():
                #     if skey == 'population_tmp_done-size':
                #         lelz_dict['pop_len'] = [val for row, val in sval.items()]
                #         self.monitor_pd.loc[row]['pop_len'] = [val]
                #     else:
                #         raise
                # m = {m[k]: v[1] for k, v in xxx}
                for time in m['population_tmp_done-size'].keys():
                    entr = {'pop_len': m['population_tmp_done-size'][time],
                            'pop_unique': None,
                            'fit_avg': m['fitness_average'][time],
                            'fit_std': math.sqrt(max(m['fitness_variance'][time], 0)),  # variance to std
                            'fit_best': m['best_candidate'][time],
                            'parsim_avg': m['complexity_average'][time],
                            'parsim_std': math.sqrt(max(m['complexity_variance'][time], 0)),
                            'complexity_avg': None,  # np.var(pop_treelen),
                            'time': m['gen_time'][time]}
                    self.monitor_pd.loc[time] = entr  # sfeh version1 delete this sh

            self.check_update()

            printez('g', f'Successfully loaded backup file. Generation: {self.gen_id}', self.print_type)

            # except Exception as ex:
            #     raise Exception(f'Even though a backup exists for this run, it could not be loaded, because of\n{ex}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}.')
        return

    def check_update(self):
        """
        Update paretofront
        update population (fitness, parsimony, etc)
        also, raise if fitness problem
        """
        oldsize = len(self.pareto)
        pareto_list = self.pareto[:]
        self.pareto = []
        for (parsimony, fitness_train, cooltree) in pareto_list:
            fitness_train = round(fitness_train, self.conf.float_decimals)
            cooltree.meta.fitness_train = round(fitness_train, self.conf.float_decimals)
            entry = [parsimony, fitness_train, cooltree]
            self.pareto_append(entry)
        self.printpl('i', f'Updating paretofront from old run. length: {oldsize}, new pareto length: {len(self.pareto)}')

        # sfeh recompute parsimony and fitness for every tree (optional?)
        # also rebuild trees?

        poplen = len(self.pop_base)
        pop_base_copy = self.pop_base[:]
        self.pop_base = []
        for cooltree in pop_base_copy:
            cooltree.meta.fitness_train = round(cooltree.meta.fitness_train, self.conf.float_decimals)
            self.pop_base.append(cooltree)
        self.printpl('i', f'Updating population from old run. length: {poplen}, new population length: {len(self.pop_base)}')

    def pareto_sort(self):
        """
        sorting the pareto entries in list for parsimony
        """
        self.pareto.sort(key=lambda x: x[0])

    def plagih_gp_run(self, path_load_backup, force_new_run=False, gen_additionally=None):
        """
        regular plagih run
        """

        if not force_new_run:
            try:
                self.backup_load(path_load_backup)
            except FileNotFoundError as fnfex:
                print_warning('i', f'No backup file found at {fnfex}. Starting a new run.', print_type=self.print_type)
            except Exception:
                raise

        if gen_additionally is not None:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.gen_id + gen_additionally)
            self.printpl('i', f'Adding new generations, gen_max was {printdummy}, current gen {self.gen_id}. gen_additionally: {gen_additionally}.\nNew max gen: {self.conf.gen_max}')

        self.write_config_yaml()  # just to see what config is running and what the user can set

        if self.gen_id == 0:
            self.print_g('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')

            self.time_genstart = time.perf_counter()
            self.population_tmp = []
            self.gen_create_initial()

            self.pop_analyse()
            self.pop_base = self.population_tmp[:]
            self.file_population_base_karoo('first')  # first gen only

        while self.run_continues():  # max generation, max time, done...
            # self.printpl('gggg', f'Evolving Generation {self.gen_id}')
            self.gen_id += 1
            self.time_genstart = time.perf_counter()
            self.population_tmp = []
            self.gen_next_population()
            self.periodical_procedures()

            self.pop_analyse()
            self.pop_base = self.population_tmp[:]
            self.print_g('ggg', f'Generation {self.gen_id} took a total time of: {time.perf_counter() - self.time_genstart:4.2f}. ')
        else:
            printez('g', f'Done after Generation {self.gen_id}.', print_type=self.print_type)

        self.run_backup_save()
        # self.terminate_run()

        return

    def write_config_yaml(self):
        """
        write the parameters to a .csv file which can also be loaded
        """
        # filename = self.root_dir / info_config_yaml
        filename = self.file_make_dir_root(self.file_locs.info_config_yaml)
        yaml_dump(filename, self.conf, print_type=self.print_type)

        return

    # def write_config_json(self):
    #     """
    #     write the parameters to a .csv file which can also be loaded
    #     """
    #
    #     path = self.file_make_dir_root(self.file_locs.file_info_config_json)
    #
    #     with Path.open(path, 'w') as file:
    #         json.dump(self.conf, file, indent=4)
    #
    #     return

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

        if self.origin_cooltree is not None:

            self.pop_append(self.origin_cooltree, last_evolution='origin')  # sfeh why not :P

        else:
            total_rate = sum([x['evolve_rate'] for x in self.evolve_random.values()])

            for tag, evolve_specs in self.evolve_random.items():
                evolve_num = int(self.conf.pop_max * (evolve_specs['evolve_rate'] / total_rate))
                call_params = evolve_specs.get('custom_params')

                for nn in range(evolve_num):
                    if self.origin_is_fix:
                        new_tree = self.pop_random(call_params, from_origin=True)
                    else:
                        new_tree = self.pop_random(call_params)
                    new_cooltree = cooltree_from_oldtree(new_tree)
                    self.pop_append(new_cooltree, last_evolution=tag)
        return

    def gen_next_population(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        # All gp creators: name, function, num of trees from tournament selection

        for tag, evolve_specs in self.evolve_loop.items():  # all selected gp mutations

            time_evolve = time.perf_counter()
            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            call_params = evolve_specs.get('custom_params')

            self.print_g('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')
            len_pop_before = len(self.population_tmp)
            if evolve_name == 'reproduce':
                """
                
                """
                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    if call_params.get('sympify_tree'):
                        try:
                            cooltree.evolve_reduce(obs_krazy=self.env_vars.obs_krazy, completely=False)
                        except Exception as ex:
                            print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.print_type)

                    self.pop_append(cooltree, last_evolution=tag)

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    cooltree.evolve_mutate_point(self.choose_oparray2,
                                                 self.env_vars.choose_obs,
                                                 self.choose_distributions, self.conf.float_decimals)
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

                if self.origin_is_fix:
                    for nn in range(evolve_num):
                        new_tree = self.pop_random(call_params, from_origin=True)
                        cooltree = cooltree_from_oldtree(new_tree)
                        self.pop_append(cooltree, last_evolution=tag)
                else:
                    for nn in range(evolve_num):
                        new_tree = self.pop_random(call_params)
                        cooltree = cooltree_from_oldtree(new_tree)
                        self.pop_append(cooltree, last_evolution=tag)
            else:
                print_e(f'the specified evolve call is not known: \'{evolve_name}\'')
            created_trees = (len(self.population_tmp) - len_pop_before)
            self.print_g('ggg', f'->Evolving \'{tag}\' (success {created_trees}/{evolve_num}) took: {time.perf_counter() - time_evolve:4.2f}s pop.size is now {len(self.population_tmp)}.')

        # sfeh automatically fill with random trees
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        return

    def run_continues(self):
        """
        checks if the run can continue
        """
        cond_1 = self.gen_id <= self.conf.gen_max

        return cond_1

    def periodical_procedures(self):
        """
        Every few generations, update the created gp_files
        - default is in every generation, but saving every n-th gen or after time passed is possible aswell
        """

        if self.gen_id % int(self.conf.period.get('gen_plots', 1)) == 0:
            self.file_analysis_plots()

        if self.gen_id % int(self.conf.period.get('gen_save', 1)) == 0:
            self.run_backup_save()
        return

    def file_pareto_txt(self):
        """
        Save all the pareto efficient candidates to file
        sfeh save as yaml?
        """

        path_pareto = self.file_make_dir_root(self.file_locs.file_pareto)

        with Path.open(path_pareto, 'w') as file:
            for (parsim, fitness, cooltree) in self.pareto:
                file.write(f'\nParsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{cooltree.meta.expr_raw}')

        return

    def file_population_base_karoo(self, pop_name):
        """
        Save population_* to disk.
        # sfeh also save pareto/pop as labellists for easy loading?
        """
        file_path = file_make_dir(self.root_dir / f'info/population_{pop_name}.txt')
        with Path.open(file_path, 'w', newline='') as txt_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
            # target = csv.writer(csv_file, delimiter=',')
            # if self.gen_id != 0:
            #     target.writerows([''])  # empty row before each generation
            txt_file.write(f'Plagih GP by Simon Fehrer, inspired by Kai Staats Karoo-gp. Generation: {self.gen_id}\n')

            for ii, cooltree in enumerate(self.pop_base):
                txt_file.write(f'\nTree meta: {cooltree.meta}')
                txt_file.write(f'\nas string: {cooltree}')
                csv_formatted_tree = cooltree.pretty_format()
                txt_file.write(f'\nFormatted in layers:\n{csv_formatted_tree}\n')  # writerows only for csv

        return

    def analyse_pareto(self):
        """
        Giving all the results
        # sfeh discussion: This is only relevant at the end. (aka not in persidical analysis)
        # in between, this might create wrong files
        # e.g. pareto entries, that do not exist at the end leave files behind
        """
        self.printpl('i', f'Analysing the pareto candidates of your run!')
        # if self.conf.period['gen_analysis']:
        #     if self.gen_id % int(self.conf.period['gen_analysis']) == 0:
        #         self.analyse_pareto()

        # self.pareto_sort()  # is pareto not sorted?  sfeh working? check if sorted.

        self.file_pareto_txt()
        self.file_population_base_karoo('last')
        self.file_pareto_histograms()
        self.file_pareto_latex()
        if 'MTC' in self.conf.name:
            self.file_pareto_pycode()
        elif 'IB' in self.conf.name:
            self.file_pareto_listcode()
        else:
            print_warning('w', f'This should actually never happen right now. name: {self.conf.name}')

        return

    def activate_dataset(self, path_data_csv, action_name):
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

        if path_data_csv:
            if path_data_csv.suffix == '.p':
                data_prepared = pickle_load(path_data_csv)
            elif path_data_csv.suffix == '.csv':
                data_prepared = data_from_csv(path_data_csv, action_name=action_name)
            else:
                raise FileNotFoundError(f'File nust be a pickle (.p) or csv (.csv) file. Loaded file: {path_data_csv}')
        elif Path.is_file(self.root_dir / self.file_locs.samples_ready_p):  # maybe the data was already prepared earlier sfeh load file
            data_prepared = pickle_load(self.root_dir / self.file_locs.samples_ready_p)
        elif Path.is_file(self.root_dir / self.file_locs.samples_csv):  # Preprocess the raw data: training/test split, env-variables, ...  sfeh load file
            data_prepared = data_from_csv(self.file_locs.samples_csv, action_name=action_name)
            print(f'Prepared the raw {self.file_locs.samples_csv} behaviour. Saving for next run.')
            pickle_dump(self.root_dir / self.file_locs.samples_ready_p, data_prepared)
        else:
            raise FileNotFoundError(f'No data provided? Please provide data in your config-file(or in your command line call). {path_data_csv}')

        env_vars, data_train, data_control = data_prepared  # data_control is data_test

        return env_vars, data_train, data_control

    def gp_load_oparray(self, path_operators=None):
        """

        """

        try:
            operator_tuples = yaml_load(Path(path_operators))
            # sfeh lel, 100% excepts as never loaded this
        except:
            self.printpl('i', 'Opt-in not specified: Operators-file does not exist. Creating one with a default list of mathematical operator_tuples.')
            operator_tuples = [['+', 2],
                               ['-', 1], ['Usub', 1],
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

        """
        Check, if the user-specified operators allow closure
        """
        # sfeh dunno if that works... 2f not in x
        opxtypes = [op[oper]['xtype'] for oper, _ in operator_tuples]
        has_2f = any(['2f' in x for x in opxtypes])
        has_2b = any(['2b' in x for x in opxtypes])
        has_f2b = any(['f2b' in x for x in opxtypes])
        has_b2f = any(['b2f' in x for x in opxtypes])
        if not all([has_2f, has_2b, has_f2b, has_b2f]):
            print_warning('w', f'Operators are not complete', print_type=self.print_type)
        if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
            raise Exception(f'Operators do not allow closure')

        """
        Load all operator_tuples ready-to-use from a file
        """
        choose_oparray2 = {
            # # all operator_tuples (not needed)
            # None: {0: [], 1: [], 2: [], 3: [], None: []},

            # all operator_tuples with a certain xtype-result
            '2f': {0: [], 1: [], 2: [], 3: [], None: []},
            '2b': {0: [], 1: [], 2: [], None: []},

            # all operator_tuples for point mutation
            'f2f': {1: [], 2: [], None: []},
            'f2b': {1: [], 2: [], None: []},
            'b2b': {1: [], 2: [], None: []},
            'b2f': {1: []},
            'b2f2f': {3: [], None: []}
        }

        for ops_tupel in operator_tuples:
            label = ops_tupel[0]
            ops_tupel = (ops_tupel[0], float(ops_tupel[1]))

            op_info = op[label]
            xtype = op_info['xtype']
            arity = op_info['arity']

            # choose_oparray[xtype_index(xtype)][arity].append(ops_tupel)

            choose_oparray2[xtype][None].append(ops_tupel)
            choose_oparray2[xtype][arity].append(ops_tupel)
            choose_oparray2[xtype[-2:]][None].append(ops_tupel)
            choose_oparray2[xtype[-2:]][arity].append(ops_tupel)

        for xtype, xrow in choose_oparray2.items():
            for arity in xrow.keys():
                # from a list of tuples [[op, prob], [+, 1], ...] to two lists [operator_tuples] [probability]
                operators_probabilities = list(zip(*choose_oparray2[xtype][arity]))
                choose_oparray2[xtype][arity] = operators_probabilities

        return choose_oparray2

    def activate_distributions(self, path_distrib=None):
        """

        """
        path_distrib = path_distrib if path_distrib is not None else self.root_dir / self.file_locs.distributions_file

        if Path.is_file(path_distrib):
            lambdadist_as_string = yaml_load(path_distrib)
        else:
            self.printpl('i', 'Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.')
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

        return choose_distributions

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

        path_hist = folder_make_dir(self.root_dir / self.file_locs.folder_histograms)  # todo make this at the start and only use the path

        for (parsim, fitness, cooltree) in self.pareto:

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
            ax.set_ylim(0, len(self.data_train)), ax.set_ylabel('Frequency'), ax.set_xlabel('Deviation')
            fig.tight_layout()
            fig.savefig(path_hist / f'acthist_{parsim}.svg')
            fig.savefig(path_hist / f'acthist_{parsim}.png', dpi=300)
            plt.close()
        self.printpl('ff', f'Histograms: {path_hist.as_posix()}')

    def file_pareto_latex(self):
        """
        Generates latex-file with the computational tree structure of all pareto agents
        - build tree from expression
        - fill tree meta-data, just in case we want to visualise anything of it
        - create latex-forest representation
        """
        latex_elements = []

        for (parsim, fitness, cooltree) in self.pareto:

            try:
                # origin_oldtree = self.origin_cooltree.get_oldtree()
                # tree = tree_set_modifyable_nodes(tree, origin_tree=origin_oldtree)
                cooltree.set_fix_nodes(self.origin_cooltree)
            except Exception:
                pass

            cooltree.meta.last_evolution = 'texify'
            tree = cooltree.get_oldtree()
            latex_elements.append(f'Pareto entry at parsimony {parsim} with mean Regression Error {fitness}:\n\n')

            forest_viz = latex_tree_get_forest(tree, tight_viz=False)
            latex_elements.append(forest_viz)

            latex_elements.append('Tight layout:')
            # try:
            forest_viz_tight = latex_tree_get_forest(tree)
            latex_elements.append(forest_viz_tight)
            # except Exception as tvex:
            #     print_e(f'forest_viz_tight could not be created. {tree_get_labellist(tree)}\nReason: {tvex}')
            #     latex_elements.append(f'forest_viz_tight could not be created. {tree_get_labellist(tree)}\nReason: {tvex}')

        latex_full_doc = latex_treeviz_full(latex_elements)

        path_trees_tex = self.file_make_dir_root(self.file_locs.trees_tex)
        with Path.open(path_trees_tex, 'w') as file:
            file.write(latex_full_doc)

        self.printpl('ff', f'Latex-trees: {path_trees_tex.as_posix()}')

        return

    def file_pareto_pycode(self):
        """
        this auto-generation of real (executable) python files
        is strongly customized for my experiments with Mountaincar and industrial benchmark

        very useful: textwrap.indent
        example: complete_function = textwrap.indent(f"def decide(self, input):\n"
                                            f"{function_body}\n", '    ')  # aka tab (\t)
        """

        py_return = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')

        complete_function = f"    def decide(self, input):\n" \
            f"        cartPos, cartVel = input\n" \
            f"        action = {{}}\n" \
            f"        return {py_return}\n"

        all_agents = []
        all_agent_names = []
        all_more_info = []

        for (parsim, fitness, cooltree) in self.pareto:
            agent_name = f'{self.conf.name}_{parsim:.0f}'

            agent_as_python = cooltree.get_pycode()
            all_agents.append(f"class {agent_name}:\n{complete_function.format(agent_as_python)}")
            all_agent_names.append(agent_name)
            all_more_info.append(f"('{agent_name}', {agent_name}(), {parsim}, {fitness})")

        all_agents = '\n\n'.join(all_agents)
        agent_tuples = ', '.join([f"('{x}', {x}())" for x in all_agent_names])
        all_more_info = ', '.join(all_more_info)
        pyc_complete = f"import math; import numpy as np\n\n" \
            f"{all_agents}\n" \
            f"all_agents_more = [{all_more_info}]\n" \
            f"agent_tuples = [{agent_tuples}]\n"

        pth = file_make_dir(self.root_dir / self.file_locs.folder_pycode / f"agents_{self.env_vars.eval_action.name}.py")
        with Path.open(pth, 'w') as file:
            file.write(pyc_complete)
            self.printpl('ff', f'Pycode: {pth.as_posix()}')

        return

    def file_pareto_listcode(self):
        """

        """

        # pycode_agent = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')

        pygents_list = []

        for (parsim, fitness, cooltree) in self.pareto:
            # agent_name = f'{self.conf.name}_{parsim:.0f}'
            agent_name = f'{self.conf.name}_{self.env_vars.eval_action.name}_{parsim:.0f}'
            agent_as_python = cooltree.get_pycode()
            pygents_list.append([parsim, float(fitness), agent_name, agent_as_python])

        path = file_make_dir(self.root_dir / 'pycode_list.yaml')
        with Path.open(path, 'w') as file:
            _ = yaml.dump(pygents_list, file)  # , default_flow_style=False, sort_keys=False)
            printez('ff', f'IB pycode-list: {path.as_posix()}')  # sfeh always the same print structure... just pass the path?

        return

    def origin_tree_get(self):
        """
        Safely return an origin_meta tree
        """
        if self.origin_cooltree is not None:
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

    def pop_mutate_filter(self, call_params, tree):
        """
        Mutates a number of float terminal of a tree
        """
        mode = call_params['mode']  # point/branch/all
        yes_observations = call_params.get('yes_observations')  # point/branch/all
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

        if mode == 'point':  # if pointmutation, return one nodeid as list
            if yes_observations:
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
                val = label_constant_mutate(val, term_type=float, float_decimals=self.conf.float_decimals, filter_type=mutate_filter)
                tree = tree_node_set_label(tree, node_id, val)

        if obs_nodes and yes_observations:  # 'filtering' variables when they are from different times
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

        # print_warning('www', 'Tree does not seem to have any nodes for filtering.', print_type=self.print_type)  # usually happens with point-filtering
        # pass

        return tree

    def invent_label_list(self, size_mode, first_xtype, build_size, full_or_grow):
        """
        Creates a random label list
        """
        if 'depth' in size_mode:
            # sfeh warning: Attention with this one. can get quite large with depth based
            label_list, arity_list, xtype_list = invent_label_list_depth(first_xtype, build_size,
                                                                         self.env_vars.choose_obs, self.env_vars.obs_krazy, self.choose_oparray2,
                                                                         self.choose_distributions, float_decimals=self.conf.float_decimals, full_or_grow=full_or_grow)

        elif 'nodes' in size_mode:

            label_list, arity_list, xtype_list = invent_label_list_nodes(first_xtype, build_size,
                                                                         self.env_vars.choose_obs, self.env_vars.obs_krazy, self.choose_oparray2,
                                                                         self.choose_distributions, float_decimals=self.conf.float_decimals, full_or_grow=full_or_grow)
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

    def pop_random(self, call_params, from_origin=False):
        """
        Creates random trees
        """

        build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)

        if from_origin:
            """
            insert a (random) number of branches at the first possible "layer"
            (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
            - get these nodes, randomly choose a subset of those
            - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
            - split the amount of nodes up (randomly) and add these new branches to the tree
            """

            from_origin = self.origin_cooltree.get_oldtree()
            layer0_ids = tree_get_mutatable_layer(from_origin, 0)

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

            tree = from_origin.copy()
            for i in range(len(layer0_ids)):  # insert branches! get layer every time (node ids might have changed)
                layer0_ids = tree_get_mutatable_layer_lv0(tree)
                node_id = layer0_ids[i]
                first_xtype = tree_node_get_xtype(tree, node_id)
                old_branch = tree_node_get_branch(tree, node_id, karoo=True)
                build_size = build_split[i]

                label_list, arity_list, xtype_list = self.invent_label_list(size_mode, first_xtype, build_size, full_or_grow)

                c_core = Core_From_Labels(label_list, arity_list, xtype_list)
                core = c_core.get_uninstanced_core()
                tree = tree_insert_subtree(tree, core, old_branch, karoo=True)
        else:
            action_xtype = self.env_vars.eval_action.xtype
            build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')

            label_list, arity_list, xtype_list = self.invent_label_list(size_mode, action_xtype, build_size, full_or_grow)

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

        label_list, arity_list, xtype_list = self.invent_label_list(size_mode, old_xtype, build_size, full_or_grow)

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

    def pareto_append(self, tree_entry, msg=None):
        if msg:
            self.printpl('a', f"Paretofront new entry ({msg}): {tree_entry[0]}, {tree_entry[1]}: {tree_entry[2].meta.expr_raw}")
        self.pareto.append(tree_entry)

        self.pareto = [x for x in self.pareto[:] if x[0] < tree_entry[0] or x[1] < tree_entry[1] or (x[0] == tree_entry[0] and x[1] == tree_entry[1])]
        self.pareto_sort()  # as far as I can tell, not really necessary without using iter()

        cooltree = tree_entry[2]
        cooltree_sym = copy.deepcopy(cooltree)
        try:
            self.printpl('aaa', 'Trying to simplify for pareto entry.')  # simplify the tree and save in pareto once again
            cooltree_sym.evolve_reduce(obs_krazy=self.env_vars.obs_krazy, completely=True)
            parsimony = cooltree_sym.eval_parsimony(self.conf.complexity_measure, origin_cooltree=self.origin_cooltree)
            if parsimony < cooltree.meta.parsimony:
                self.printpl('aa', 'Successfully reduced pareto tree!')
                sym_fitness = self.tree_eval_fitness_train(cooltree_sym)  # sfeh actually not required, delete this
                cooltree_sym.meta.fitness_train = sym_fitness
                cooltree_sym.meta.parsimony = parsimony
                self.update_pareto(cooltree_sym)
        except Exception as ex:
            print_warning('www', f'Tree sympification did not work: {ex}', print_type=self.print_type)

        else:
            self.printpl('aaa', 'Pareto entry was already simplified')

    def update_pareto(self, cooltree: CoolTree):
        """
        inserts a tree into the pareto front
        """

        parsimony = cooltree.meta.parsimony
        fitness_train = cooltree.meta.fitness_train
        tree_entry = [parsimony, fitness_train, cooltree]

        p_simpler = [p for p in self.pareto if p[0] <= tree_entry[0]]  # all pareto entries that are less complex

        if len(p_simpler) == 0:  # all other pareto entries are more complex
            self.pareto_append(tree_entry, msg=f'new simplest entry')
        else:
            best = min(p_simpler, key=lambda p: p[1])  # the fittest of the less complex ones
            if tree_entry[1] < best[1]:
                self.kernel.better_fitness_relation(tree_entry[1], best[1])  # if true, at least one insertion
                self.pareto_append(tree_entry, msg=f'old fitness: {best[1]}')

        self.pareto_sort()  # sfeh check if required
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
            fitness_train = round(tree_meta['fitness_train'], self.conf.float_decimals)  # sfeh just for now
        else:
            parsimony = cooltree.eval_parsimony(self.conf.complexity_measure, origin_cooltree=self.origin_cooltree)
            if parsimony > self.conf.parsimony_max:
                print_warning('wwww', f'Parsimony too high, last evolution: {last_evolution}', print_type=self.print_type)  # sfeh care about wwww. should not
                return
            try:
                fitness_train = self.tree_eval_fitness_train(cooltree)
            except Exception as evalex:
                print_warning('wwww', f'Exception while evaluating: {evalex}', print_type=self.print_type)
                # eval_fails.append(str(ex))  # sfeh
                # continue
                return

        expr_raw = cooltree.get_expr_raw()
        expr_sym = expr_sympify(expr_raw)
        try:
            cooltree.set_fix_nodes(self.origin_cooltree)
        except Exception as ex:
            print(f'NOOOOOOPE, failed tree finish: {ex}\n{cooltree}')

        cooltree.meta.fitness_train = fitness_train
        cooltree.meta.parsimony = parsimony
        cooltree.meta.last_evolution = last_evolution
        cooltree.meta.expr_raw = expr_raw
        cooltree.meta.expr_sym = expr_sym
        # cooltree.set_meta(fitness_train, parsimony, last_evolution, expr_raw, expr_sym)

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

    def load_origin_tree(self, path_origin_tree, label_list=None, modify_list=None):
        """
        The origin tree (which was already loaded) gets activated for its use in the GP-process
        """
        # tree_expr_txt_path = root_dir / 'run_files/tree_expr.txt'
        # tree_numpy_csv_path = root_dir / 'run_files/tree_numpy.csv'
        # tree_labels_csv_path = self.root_dir / 'run_files/tree_labels.csv'

        """
        from the labellist-csv, loading the label list
        """

        with Path.open(path_origin_tree, newline='') as csvFile:
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
                        print_e(f'Unexpected row start: {row[0]}')
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
        fitness_train = round(float(tf_origin_results['mean_error']), self.conf.float_decimals)  # fitness currently IS the mean error
        if self.kernel.exploration_risk:
            self.kernel.origin_results = tf_origin_results['results_kernel']  # after getting the origin-results, these informations can be updated

        origin_cooltree.meta.fitness_train = fitness_train
        origin_cooltree.meta.parsimony = 0

        self.pareto.append([0, fitness_train, origin_cooltree])  # the origin tree is the only candidate for now -> it is in the paretofront
        self.print_g('gg', f'Loading origin tree, fitness {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return origin_cooltree  # self.origin_cooltree = copy.deepcopy(origin_cooltree)

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
        fitness_train = round(float(self.kernel.eval_tf(expr_sym, used_observations, only_fitness=True)), self.conf.float_decimals)

        if not check_value_is_real(fitness_train):
            raise Exception(f'Error is {fitness_train}')  # happens, eg when values are soo wrong that it leaves the float-range
        # fitness_train = round(fitness_train, self.conf.fitness_decimals)

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

    def plot_end(self, xx, yy,
                 title='', ax_label='', x_label='Generation', y_label='', yscale='linear',
                 step_post='',  # plt_xparam='',
                 linestyle='-',
                 marker='',
                 set_left=None, set_right=None, set_top=None,
                 padd_r=1.05, padd_top=1.05,
                 beyond_lines=False,
                 fill_variance=None):
        """
        Make all plots in the same style - and also saving space.
        - Makes pyplots

        :param set_top:
        :param fill_variance:
        :param padd_r:
        :param marker:
        :param title:
        :param ax_label: irrelevant for a single curve
        :param x_label: label the x-axis
        :param y_label: label the y-axis
        :param yscale: only 'linear'.
        :param step_post: makes 'step' plots- can be 'post', 'pre' or [pls google]

        :param linestyle: E. g. 'None', 'dashed', '-', ''
        :param set_left: Smallest left value
        :param set_right: E. g. if parsim_maxony is 100 -> show complete width, even if entries only go to 40
        :param padd_top: How much padding to the top border
        :param beyond_lines: in step plots, draw the line further to the left and right

        :return:

        Options that are not used
        # plt.legend()
        sfeh: max_height=None,  # when creating a plot in every generation, fix the maximum height and width?
        """

        if len(xx) == 0:
            print_e(f'Plotting empty array is not possible! Data={xx, yy}')
            return

        x, y = xx, yy

        top, bottom, left, right, new_right, new_top = plot_sexyfy(x, y, set_left=set_left, set_right=set_right, set_top=set_top, right_padding=padd_r, top_padding=padd_top)

        if beyond_lines:  # adding a point to the edges to imply that there are no more values (pareto-plot)
            x = np.concatenate([[x[0]], x, [new_right + 1]])
            y = np.concatenate([[new_top + 1], y, [y[-1]]])

        fig, ax = plt.subplots()
        ax.set_yscale(yscale)
        ax.set_xlim(min(left, 0), new_right)
        ax.set_ylim(min(bottom, 0), new_top)
        fig.tight_layout()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if step_post:
            ax.step(x, y, linestyle=linestyle, marker=marker, label=ax_label, where=step_post)  # , plt_xparam
        else:
            ax.plot(x, y, linestyle=linestyle, marker=marker, label=ax_label)  # , plt_xparam
            if fill_variance is not None:
                x_std, y_var = fill_variance
                y_std = np.sqrt(y_var)
                lower_bound_stderr = y - y_std
                upper_bound_stderr = y + y_std
                ax.fill_between(x_std, lower_bound_stderr, upper_bound_stderr, alpha=0.2)
                ax.set_ylim(min(bottom, 0), new_top + np.max(upper_bound_stderr))

        path_plot = folder_make_dir(self.root_dir / self.file_locs.folder_plots)

        plt.tight_layout()
        try:
            plt.savefig(path_plot / f'{title}.png', dpi=300)
            plt.savefig(path_plot / f'{title}.svg')
        except PermissionError as permerr:
            print_e(f'Could not save plot: {permerr}')

        plt.close()  # Stackoverflow said that this is too much, # plt.clf() should be better, but does not seem to work
        return

    def file_analysis_plots(self):
        """
        Make all plots
        """
        try:
            # self.monitoring_pd['complexity_avg'].plot()  # sfeh rename
            fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]}, sharex='all')  # , figsize=(9, 9)
            fig.tight_layout()
            # plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
            xx = list(self.monitor_pd.index)

            axs0 = axs[0]
            axs0.plot(self.monitor_pd['fit_avg'], label='Fitness (average)')
            try:
                avg = self.monitor_pd['fit_avg']
                std = self.monitor_pd['fit_std']
                axs0.fill_between(xx, avg - std, avg + std, alpha=0.2)  # axs0.set_title('Regression Error (average)')
            except:
                pass
            axs0.step(x=xx, y=self.monitor_pd['fit_best'], linestyle='dashed', marker='', where='post', color='g', label='Best candidate')  # , label=ax_label
            axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

            axs1 = axs[1]
            axs1.plot(self.monitor_pd['parsim_avg'], label='Parsimony (average)')
            try:
                p_avg = self.monitor_pd['parsim_avg']
                p_std = self.monitor_pd['parsim_std']
                axs1.fill_between(xx, p_avg - p_std, p_avg + p_std, alpha=0.2)  # axs1.set_title('Parsimony (average)')
            except:
                pass
            axs1.set_ylim(ymin=0), axs1.legend(loc='lower left')

            axs2 = axs[2]
            # self.monitor_pd.plot(y=['pop_len', 'pop_unique'], ax=axs2), # axs2.set_title('Created trees (average)')
            axs2.plot(self.monitor_pd['pop_len'], label='pop size')
            axs2.plot(self.monitor_pd['pop_unique'], label='unique')
            axs2.margins(y=0.25), axs2.set_ylim(ymin=0), axs2.legend(loc='lower left')

            axs3 = axs[3]
            between_outliers = self.monitor_pd['time'].between(0, 2*self.monitor_pd['time'].mean())
            axs3.plot(self.monitor_pd['time'][between_outliers], label='time (s)')  # sfeh could be a better rule...
            axs3.set_ylim(ymin=0), axs3.legend(loc='lower left')

            """
            Top level style
            """
            axs3.set_xlim(xmin=0), axs3.set_xlabel('Generations')
            axs0.set_title('Population Monitoring')  # sfeh
            fig.tight_layout()
            fig.savefig(self.root_dir / f'monitoring.png', dpi=300)
            # fig.savefig(self.root_dir / f'monitoring.pdf')  # asd sfeh?
            # fig.savefig(self.root_dir / f'monitoring.svg')

            plt.close()
        except:
            print_e(f'Yeah... should have used old monitoring dict')

        tuples = [[parsim, fitness] for (parsim, fitness, cooltree) in self.pareto]
        xx, yy = np.array(tuples).T
        self.plot_end(xx, yy,
                      step_post='post', title='pareto dominant candidates', x_label='parsimony',
                      y_label='regression error',
                      linestyle='dashed',
                      marker='.',
                      set_right=self.conf.parsimony_max,
                      beyond_lines=True)


        # def plot_evolution_analysis():
        #     try:
        #         # data_tuples = plotendify_me(self.monitoring_dict['best_candidate'])
        #         # plot_end(data_tuples, path_plots, title='best candidate',
        #         #          y_label='error', linestyle='dashed', step_where='post')
        #
        #         """
        #         Plots for each tag (too much, i guess)
        #         """
        #         evolution_tag = [k for k in self.evolve_loop.keys() if k not in ['reproduce', 'revive pareto']]
        #
        #         fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)  # , figsize=(9, 9)
        #         # axs[0].set_ylabel(y_label)  # sfeh
        #         fig.suptitle('GP Evolution analysis')
        #         evol_performance = self.monitoring_dict['evol_performance']
        #         for tag in evolution_tag:
        #
        #             evolve_function = self.evolve_loop[tag]['evolve_name']
        #             # [last_evol]['fitness_avg'][gen_id]
        #             try:
        #                 xx = [[]] * 4
        #                 yy = [[]] * 4
        #                 xx[0], yy[0] = zip(*evol_performance['fitness_avg'][tag].items())
        #                 xx[1], yy[1] = zip(*evol_performance['parsimony_avg'][tag].items())
        #                 # xx[2], yy[2] = zip(*evol_performance['evolve_num'][tag].items())
        #                 # xx2, yy2 = zip(*evol_performance['evolve_num'][tag].items())
        #                 # size_x, size_y = zip(*evol_performance['lentree_avg'][tag].items())
        #
        #                 axs[0].plot(xx[0], yy[0], label=tag)
        #                 axs[1].plot(xx[1], yy[1], label=tag)
        #                 # axs[2].plot(xx[2], yy[2], label=tag)
        #                 # axs[2].plot(size_x, size_y, label=tag)
        #             except:
        #                 print_warning('wwww', f'Could not analyse last evolution with  tag: {tag}', print_type=self.print_type)
        #
        #         fig.tight_layout()
        #         plt.savefig(path_plots / f'evolve_analysis.png')
        #         plt.close()
        #
        #         """
        #         Plots for each tag (too much, i guess)
        #         """
        #         evol_funs = dict.fromkeys([v['evolve_name'] for v in self.evolve_loop.values()], [])
        #         for k, v in self.evolve_loop.items():
        #             evol_funs['evolve_name'].append(k)
        #
        #         fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)  # , figsize=(9, 9)
        #         # axs[0].set_ylabel(y_label)  # sfeh
        #         fig.suptitle('GP Evolution analysis')
        #         evol_performance = self.monitoring_dict['evol_performance']
        #         for evoname, tags in evol_funs.items():
        #             xx = [[]] * 4
        #             yy = [[]] * 4
        #             for tag in tags:
        #                 xx[0], yy[0] = zip(*evol_performance['fitness_avg'][tag].items())
        #                 xx[1], yy[1] = zip(*evol_performance['parsimony_avg'][tag].items())
        #                 # xx[2], yy[2] = zip(*evol_performance['evolve_num'][tag].items())
        #                 # xx[3], yy[3] = zip(*evol_performance['lentree_avg'][tag].items())
        #                 xxc, yyc = zip(*evol_performance['count'][tag].items())
        #
        #                 axs[0].plot(xx0, yy0, label=tag)
        #                 axs[1].plot(xx1, yy1, label=tag)
        #                 # axs[2].plot(xx2, yy2, label=tag)
        #                 # axs[2].plot(size_x, size_y, label=tag)
        #
        #         fig.tight_layout()
        #         plt.savefig(path_plots / f'evolve_analysis.png')
        #         plt.close()
        #
        #     except Exception as ex:
        #         print_e(f'plot_evolution_analysis failed because of: {ex}')
        #
        # plot_evolution_analysis()

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
        # tmp_evol_performance = {cooltree.meta.last_evolution: {'fitness': cooltree.meta.fitness_train,
        #                                                        'parsimony': cooltree.meta.parsimony,
        #                                                        'lentree': len(cooltree)} for cooltree in popul}
        # for last_evol in tmp_evol_performance:
        #     if last_evol in self.evolve_loop:
        #         for evolinfo in ['fitness_avg', 'parsimony_avg', 'lentree_avg', 'evolve_num', 'count']:
        #
        #             if self.monitoring_dict['evol_performance'][evolinfo].get(last_evol) is None:
        #                 self.monitoring_dict['evol_performance'][evolinfo][last_evol] = {}
        #         try:
        #             self.monitoring_dict['evol_performance']['fitness_avg'][last_evol][gen_id] = np.sum(tmp_evol_performance[last_evol]['fitness'])
        #             self.monitoring_dict['evol_performance']['parsimony_avg'][last_evol][gen_id] = np.sum(tmp_evol_performance[last_evol]['parsimony'])
        #             self.monitoring_dict['evol_performance']['lentree_avg'][last_evol][gen_id] = np.sum(tmp_evol_performance[last_evol]['lentree'])
        #             self.monitoring_dict['evol_performance']['evolve_num'][last_evol][gen_id] = self.evolve_loop[last_evol]['evolve_num']
        #             self.monitoring_dict['evol_performance']['count'][last_evol][gen_id] = len(tmp_evol_performance[last_evol])  # currently not used
        #             # sfeh fitness - last fitness?
        #         except Exception as ex:
        #             print_e(f'Could not save evol_performance analysis. {ex}')
        #     else:
        #         if last_evol != 'origin':
        #             print_warning('w', f'delete_this, sfeh, okay when the following is origin: {last_evol}')

        pop_parsim = [cooltree.meta.parsimony for cooltree in popul]
        pop_treelen = [len(cooltree) for cooltree in popul]

        pop_fitness_best = self.kernel.np_best_fitness(pop_fitness)
        try:
            self.best_fitness = self.kernel.best_fitness_function(pop_fitness_best, self.best_fitness)
        except:
            self.best_fitness = pop_fitness_best

        # self.monitoring_dict['population_tmp_done-size'][gen_id] = len(popul)
        # self.monitoring_dict['fitness_average'][gen_id] = np.average(pop_fitness)
        # self.monitoring_dict['fitness_variance'][gen_id] = np.var(pop_fitness)
        # self.monitoring_dict['best_candidate'][gen_id] = self.best_fitness
        # self.monitoring_dict['complexity_average'][gen_id] = np.average(pop_treelen)
        # self.monitoring_dict['complexity_variance'][gen_id] = np.var(pop_treelen)  # sfeh version1 delete this shit
        # self.monitoring_dict['gen_time'][gen_id] = gen_time

        unique_tree_count = len(set([hash(x) for x in popul]))  # sfeh analyze this?

        gen_time = time.perf_counter() - self.time_genstart

        self.monitor_pd.loc[gen_id] = {'pop_len': len(popul),
                                       'pop_unique': unique_tree_count,
                                       'fit_avg': np.average(pop_fitness),
                                       'fit_std': np.std(pop_fitness),
                                       'fit_best': self.best_fitness,
                                       'parsim_avg': np.average(pop_parsim),
                                       'parsim_std': np.std(pop_parsim),
                                       'complexity_avg': np.var(pop_treelen),
                                       'time': gen_time}  # sfeh version1 delete this shit

        self.print_g('gg', f'Created {len(popul)}/{self.conf.pop_max} ({unique_tree_count} unique) in generation {gen_id}. Gen took {gen_time:4.2f}s')
        # sfeh check if there are really unique... doubt it.
        return

    def terminate_run(self):
        """
        Program is done after writing all gp_files one last time.
        For new usage: Check if an update should be made before
        """

        self.file_analysis_plots()
        self.print_g('gg', f'Terminating. \tTime since start: {time.perf_counter() - self.time_start:4.2f}s')
        return

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
