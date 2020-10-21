"""
'f2f', 'b2b', f = float, b = bool. 'float to bool'

Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""
import time

import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import matplotlib.ticker as ticker

from benchmarks.ib.combined_runs import *
from benchmarks.mc.agents.quick_eval import auto_evaluate_run_end
from plagih.file_interaction import *
from plagih.viz_with_latex import *
from plagih.plagih_config import *
from plagih.plagih_data import *
from plagih.Ptree2 import *
import random

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


class FileLocations:

    backup_p = 'backup/backup.p'
    trees_sub_tex = 'visualisation/'
    folder_pycode = ''

    # Files that can can (in theory) be used to prepare a runable folder. maybe deprecated now, sfeh.
    use_distributions_file = 'run_files/distributions_file.yaml'

    def absolute(self, path, make_dirs=False):
        p = self.rootpath / path
        if make_dirs and not p.parent.is_dir():
            p.parent.mkdir(parents=True)
        return p

    def __init__(self, rootpath):
        self.rootpath = Path(rootpath)
        # sfeh check stuff?


class ParetoFront:
    """
    sfeh
    """
    def __init__(self):
        pass
        # self.entries = []


class ExplainableGP(object):
    """
    sfeh
    """

    def __init__(self, conf: GpConfig, root_dir: Path, path_data, path_origin_tree, mp_cpu_cores_max=1, developer_fix=None):
        self.conf = conf
        self.root_dir = Path(root_dir)
        self.mp_cpu_cores_max = mp_cpu_cores_max  # sfeh
        self.developer_fix = developer_fix  # sfeh

        print(f'\n'
              f'\tInitializing Plagih. \n'
              f'\tName: {BColors.CYAN}{self.root_dir.name}{BColors.RESET}. \n'
              f'\tLocated in: \n'
              f'\t{self.root_dir}\n')
        self.time_start = time.perf_counter()

        self.paths = FileLocations(self.root_dir)

        self.env_vars = EnvVars()
        self.data_train = None
        self.data_control = None

        # if path_data.suffix == '.p': data_prepared = pickle_load(path_data)
        self.env_vars, self.data_train, self.data_control = data_from_csv(path_data, action_name=self.conf.action_name)
        # FileNotFoundError(f'No data provided? File must be a pickle (.p) or csv (.csv) file. Loaded file: {path_data}')

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
            self.origin_cooltree.meta.last_evolution = 'origin'
            self.origin_is_fix = self.origin_cooltree.core.is_fix
        else:
            self.origin_cooltree = None
            self.origin_is_fix = False

            if self.conf.complexity_measure in ['tree_edit_distance']:
                self.conf.complexity_measure = 'tree_node_count'  # sfeh idea
                print_warning('w', "Complexity measurement 'tree_edit_distance' is not possible without origin! Using 'tree_node_count' instead.", print_type=self.print_type)

        """
        load relevant stuff
        """
        self.choose_distributions = self.activate_distributions(path_distrib=None)  # asd sfeh path_distrib not None
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

            # def load_old_monitoring_dict(self, monitoring_dict):
            #     self.population_tmp_done_size = monitoring_dict.get('population_tmp_done-size', {})
            #
            #     self.fitness_average = monitoring_dict.get('fitness_average', {})
            #     self.fitness_variance = monitoring_dict.get('fitness_variance', {})
            #     self.best_candidate = monitoring_dict.get('best_candidate', {})
            #     # 'complexity_list = monitoring_dict.get('complexity_list', {}),  # not used, just uses memory
            #
            #     self.complexity_average = monitoring_dict.get('complexity_average', {})
            #     self.complexity_variance = monitoring_dict.get('complexity_variance', {})  # variance can be deleted, only std-error is needed delete v1
            #     self.gen_time = monitoring_dict.get('gen_time', {})
            #     self.evol_performance = monitoring_dict.get('evol_performance', {})

        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique',
                                                'fit_avg', 'fit_std', 'fit_best',
                                                'parsim_avg', 'parsim_std',
                                                'complexity_avg',
                                                'time', 'gens_since_last_pareto'])

        self.evolve_loop, self.evolve_random = self.make_evolve_rates()
        self.evolve_tags = list(self.evolve_loop.keys()) + list(self.evolve_random.keys())
        # self.monitor_evoldf = pd.DataFrame(columns=self.evolve_tags)
        self.monitor_evol = dict.fromkeys(self.evolve_tags, pd.DataFrame(columns=['fitness', 'parsimony', 'lentree', 'evolve_num', 'count']))

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
                          'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'full_or_grow': 'grow'}}},
                'Rand4': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                          'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'full_or_grow': 'full'}}},  # param 'max' can be None
            }

        evolve_loop = evolve_safety_update(evolve_loop)

        if self.origin_is_fix:
            try:
                evolve_random = self.evolve_list_random['from_origin']
            except:
                evolve_random = {'Rand3o': {'evolve_name': 'random trees', 'evolve_rate': 1.00,
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

    def run_backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """

        a_helping_dict = {'self.monitor_evol': self.monitor_evol,
                          'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh save complete config?    # sfeh i dont think we need the config
        run_backup_data = self.gen_id, self.pareto, self.pop_base, self.monitor_df, a_helping_dict  # sfeh use this later, a_helping_dict
        pickle_dump(self.root_dir / self.paths.backup_p, run_backup_data)

        # run_backup_dict = {'self.gen_id': self.gen_id,
        #                    'self.pareto': self.pareto,
        #                    'self.pop_base': self.pop_base,
        #                    'self.monitor_df': self.monitor_df,
        #                    'a_helping_dict': a_helping_dict}
        #
        # path_backupyyy = path_make_dir(self.root_dir / 'backup/backup.yaml')
        # yaml_dump(path_backupyyy, run_backup_dict, print_type=self.print_type)

        return

    def backup_load(self, path_load_backup=None):
        """
        If a backup-file is found...
        """
        path_backup = path_load_backup or self.root_dir / self.paths.backup_p  # sfeh file-load

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
                self.monitor_df = monitor_pd  # sfeh
                if 'gens_since_last_pareto' not in self.monitor_df.columns:
                    self.monitor_df['gens_since_last_pareto'] = np.nan
                self.monitor_evol = a_helping_dict.get('self.monitor_evol') or self.monitor_evol
                self.gens_since_last_pareto = a_helping_dict.get('gens_since_last_pareto') or 0  # sfeh
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
                            'time': m['gen_time'][time],
                            'gens_since_last_pareto': 0}
                    self.monitor_df.loc[time] = entr  # sfeh version1 delete this sh

            self.check_update()

            self.print_g('g', f'Successfully loaded backup file. Generation: {self.gen_id}')

            # except Exception as ex:
            #     raise Exception(f'Even though a backup exists for this run, it could not be loaded, because of\n{ex}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}.')
        return

    def check_update(self):
        """
        Update pareto front
        update population (fitness, parsimony, etc)
        also, raise if fitness problem
        """
        # oldsize = len(self.pareto)
        # pareto_list = self.pareto[:]
        # self.pareto = []
        # for (parsimony, fitness_train, cooltree) in pareto_list:
        #     fitness_train = round(fitness_train, self.conf.float_decimals)
        #     cooltree.meta.fitness_train = round(fitness_train, self.conf.float_decimals)
        #     entry = [parsimony, fitness_train, cooltree]
        #     self.pareto_append(entry)
        # self.printpl('i', f'Updating pareto front from old run. length: {oldsize}, new pareto length: {len(self.pareto)}')

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

    def plagih_gp_run(self, gen_additionally):
        """
        regular plagih run
        """

        if gen_additionally:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.gen_id + gen_additionally)
            self.printpl('i', f'Adding new generations, gen_max was {printdummy}, current gen {self.gen_id}. gen_additionally: {gen_additionally}. New max gen: {self.conf.gen_max}')

        yaml_dump(self.root_dir / 'used_config.yaml', self.conf, print_type=self.print_type)

        self.gens_since_last_pareto = 0

        while self.gen_id <= self.conf.gen_max:  # max generation, max time, done...
            self.time_genstart = time.perf_counter()

            if self.gen_id == 0:
                self.print_g('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')
                self.gen_create_initial()
            else:

                # You can avoid this situation by calling multiprocessing.Process before you load your huge data.
                # Then the additional memory allocations will not be reflected in the child process when you load the data in the parent.
                # sfeh: In python 3.8, this might be availably: multiprocessing.shared_memory https://docs.python.org/3/library/multiprocessing.shared_memory.html
                # sfeh: check memory usage! should not scale with the number of processes, only one pop_base is required, it does not change.

                self.mp_cpu_cores_max = 1  # sfeh wasd
                if self.mp_cpu_cores_max >= 2:
                    pass
                    # sfeh asd
                    mp.Process()  # sfeh maybe good for memory? https://stackoverflow.com/questions/14749897/python-multiprocessing-memory-usage

                    print(f'Trying to make parallel new population: {mp.cpu_count()}')
                    with mp.Pool(min(mp.cpu_count(), self.mp_cpu_cores_max)) as p:

                        evolve_list = [[tag, evolve_specs] for tag, evolve_specs in self.evolve_loop.items()]
                        results = p.map(self.the_fun, evolve_list)
                    time_evolve = time.perf_counter()
                else:
                    self.gen_next_population()
                    pass

            self.update_pareto_from_pop_tmp()

            self.pop_analyse()

            self.pop_base = self.population_tmp[:]
            self.population_tmp = []
            self.print_g('ggg', f'Generation {self.gen_id} took a total time of: {time.perf_counter() - self.time_genstart:4.2f}.')
            self.gen_id += 1
        else:
            self.print_g('g', f'Done after Generation {self.gen_id}.\nTime since start: {time.perf_counter() - self.time_start:4.2f}s')

        self.run_backup_save()

        return

    def gen_create_initial(self):
        """
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        if self.origin_cooltree is not None:

            self.pop_append(self.origin_cooltree)  # sfeh why not :P

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
                    new_cooltree.meta.last_evolution = tag
                    self.pop_append(new_cooltree)
        return

    def gen_next_population(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        # All gp creators: name, function, num of trees from tournament selection

        for tag, evolve_specs in self.evolve_loop.items():  # all selected gp mutations

            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            call_params = evolve_specs.get('custom_params')

            self.print_g('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')
            if evolve_name == 'reproduce':
                """
                
                """
                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    if call_params.get('sympify_tree'):
                        try:
                            cooltree.evolve_reduce(obs_krazy=self.env_vars.obs_krazy, completely=False)
                            cooltree.meta.last_evolution = tag
                        except Exception as ex:
                            print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.print_type)

                    self.pop_append(cooltree)  # append anyways... it was worth a try :P

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    cooltree.evolve_mutate_point(self.choose_oparray2,
                                                 self.env_vars.choose_obs,
                                                 self.choose_distributions, self.conf.float_decimals)
                    cooltree.meta.last_evolution = tag
                    self.pop_append(cooltree)

            elif evolve_name == 'mutate branch':

                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)

                    new_tree = cooltree.get_oldtree()
                    build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)
                    full_or_grow = build_spec.get('full_or_grow') or random.choice(['full', 'grow'])
                    node_ids = tree_get_mutatable_nodes(new_tree, no_root=True)
                    old_node = random.choice(node_ids)
                    old_xtype = tree_node_get_xtype(new_tree, old_node)
                    build_size = choose_build_size(size_mode, mean_min_max_var, tree=new_tree, node_id=old_node)

                    label_list, arity_list, xtype_list = self.invent_label_list(size_mode, old_xtype, build_size, full_or_grow)

                    c_core = Core_From_Labels(label_list, arity_list, xtype_list)
                    core_insert = c_core.get_uninstanced_core()
                    branch_nodes_ids = tree_node_get_branch(new_tree, old_node, karoo=True)
                    new_tree = tree_insert_subtree(new_tree, core_insert, branch_nodes_ids, karoo=True)
                    new_tree = tree_prune_depth(new_tree, self.conf.tree_depth_max, self.env_vars.obs_krazy, self.env_vars.choose_obs, self.choose_distributions, self.conf.float_decimals)
                    # else:
                    #     new_tree = None

                    new_cooltree = cooltree_from_oldtree(new_tree)

                    new_cooltree.meta.last_evolution = tag
                    self.pop_append(new_cooltree)

            elif evolve_name == 'crossover branch':
                for nn in range(int(evolve_num / 2)):  # two childs
                    parent_a = self.pop_selection_tournament(tourn_size)
                    parent_b = self.pop_selection_tournament(tourn_size)
                    parent_a = parent_a.get_oldtree()
                    parent_b = parent_b.get_oldtree()
                    child_a, child_b = self.pop_crossover_branch(parent_a, parent_b)
                    child_a.meta.last_evolution = tag
                    child_b.meta.last_evolution = tag
                    self.pop_append(child_a)
                    self.pop_append(child_b)

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    cooltree = self.pop_selection_tournament(tourn_size)
                    tree = cooltree.get_oldtree()
                    new_tree = self.pop_mutate_filter(call_params, tree)
                    new_cooltree = cooltree_from_oldtree(new_tree)
                    new_cooltree.meta.last_evolution = tag
                    self.pop_append(new_cooltree)

            elif evolve_name == 'revive pareto':

                for nn in range(evolve_num):
                    fitness_train, parsim, cooltree = random.choice(self.pareto)
                    cooltree.meta.last_evolution = tag
                    self.pop_append(cooltree)

            elif evolve_name == 'random trees':

                if self.origin_is_fix:
                    for nn in range(evolve_num):
                        new_tree = self.pop_random(call_params, from_origin=True)
                        cooltree = cooltree_from_oldtree(new_tree)
                        cooltree.meta.last_evolution = tag
                        self.pop_append(cooltree)
                else:
                    for nn in range(evolve_num):
                        new_tree = self.pop_random(call_params)
                        cooltree = cooltree_from_oldtree(new_tree)
                        cooltree.meta.last_evolution = tag
                        self.pop_append(cooltree)
            else:
                print_e(f'the specified evolve call is not known: \'{evolve_name}\'')

            # self.print_g('ggg', f'->Evolving \'{tag}\' (success {len(self.population_tmp)}/{evolve_num}) took: {time.perf_counter() - time_evolve:4.2f}s pop.size is now {len(self.population_tmp)}.')

        # sfeh automatically fill with random trees
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        return

    def file_pareto_txt(self):
        """
        Save all the pareto efficient candidates to file
        sfeh save as yaml?
        """
        pareto_yaml = [f'Parsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{cooltree.meta.expr_raw}' for (parsim, fitness, cooltree) in self.pareto]
        yaml_dump(self.root_dir / 'paretofront.yaml', pareto_yaml, print_type=self.print_type)

        return

    def file_population(self, pop_name):
        """
        Save population_* to disk.
        """
        txt_file = f'Plagih GP by Simon Fehrer, inspired by Kai Staats Karoo-gp. Generation: {self.gen_id}\n'
        for ii, cooltree in enumerate(self.pop_base):
            txt_file += f'\nTree meta: {cooltree.meta}\nas string: {cooltree}\nFormatted in layers:\n{cooltree.pretty_format()}\n'

        file_dump(self.root_dir / f'info/population_{pop_name}.txt', txt_file)

        return

    def analyse_pareto(self, cpu_cores=16):
        """
        Giving all the results
        # sfeh discussion: This is only relevant at the end. (aka not in persidical analysis)
        # in between, this might create wrong files
        # e.g. pareto entries, that do not exist at the end leave files behind
        """
        self.printpl('i', f'Analysing the pareto candidates of your run!')

        # sfeh
        # class ParetoEntry(object):
        #     def __init__(self):
        #         self.histogram = None
        #         self.latex = None
        #         self.yamltext = None
        #         self.realcode = None
        #         self.texforest_bracket = {0: None, 1: None, 2: None}  # 0=fulltree, 1=tighttree, 2=oneliner

        # if self.conf.period['gen_analysis']:
        #     if self.gen_id % int(self.conf.period['gen_analysis']) == 0:
        #         self.analyse_pareto()

        # self.pareto_sort()  # is pareto not sorted?  sfeh working? check if sorted.
        # self.file_population('last')

        dir_benchmarks = Path(__file__).parent.parent.absolute() / 'benchmarks/'
        path_hist = path_make_dir(self.root_dir / 'histograms/')
        pareto_agents = {}

        for (parsim, fitness, cooltree) in self.pareto:
            # self.file_pareto_txt()  # sfeh

            # todo todotodo histpath
            histpath = None
            # histpath = self.plot_agent_histogram(parsim, fitness, cooltree, path_hist)

            texvis = self.file_pareto_latex(parsim, fitness, cooltree)
            forestree0, forestree1, texpression_2, forestree2 = texvis

            pareto_agents[parsim] = {'parsim': parsim,
                                     'fitness': fitness,
                                     'cooltree': cooltree,
                                     'forestree0': forestree0, 'forestree1': forestree1, 'texpression_2': texpression_2, 'forestree2': forestree2,  # 'texvis': texvis,
                                     'histogram': histpath,
                                     }
            # finaline = [f'{parsim} & {fitness} & {cooltree} % {}{}{}{}', 'regression error', 'real fitness', 'forestree0', 'forestree1', 'texpression_2', 'forestree2']

        tex_include_pdf = lambda x: f"\\includegraphics{{{str(x).replace('.pdf', '')}}}"
        tex_tabuline = lambda x: f"{' & '.join(x)}\\tabularnewline\n"

        if 'MTC' in self.conf.name:
            """
            complexity, regr. error, real evaluation, decision plot, spiral plot, diff-plot
            """
            self.file_pareto_pycode()
            sarsa_agent_steps = 200 if 'MTC200' in self.conf.name else 75 if 'MTC75' in self.conf.name else 'NO_MC_AGENT'
            sarsa_agent = pickle_load(dir_benchmarks / f'mc/agents/sarsa_agent_{sarsa_agent_steps}.p')

            agent_performance, _ = auto_evaluate_run_end(self.root_dir, sarsa_agent, n=100)
            df_mtc = pd.DataFrame(columns=['complexity', 'regr. error', 'avg. reward', 'fails', 'expression'])
            tex_lines = []

            for pp, x in agent_performance.items():
                _, pp, avg_reward, fails, path_mcmeshplot, path_mcmeshplot_diff = x
                # pareto_agents[pp].update({})
                tex_line = [f"{int(pp)}",
                            f"{pareto_agents[pp]['fitness']}",
                            f"{avg_reward:0.1f}",
                            f'{fails}',
                            f"{pareto_agents[pp]['forestree2']}"]
                tex_line_large = [tex_include_pdf(f'sfehs_eval/{pp}'),  # path_mcmeshplot
                                  tex_include_pdf(f'sfehs_eval/diff-{pp}'),  # path_mcmeshplot_diff),
                                  'sfeh decision plot',
                                  tex_include_pdf(f'histograms/acthist_{pp}')]

                df_mtc.loc[pp] = tex_line

                tex_lines += [tex_line + tex_line_large]

            todo = df_mtc.to_latex(escape=False)  # escape, why? -> no \textbackslash
            file_dump(self.root_dir / f'df_mtc.tex', todo, print_type=self.print_type)  # agents_trees.tex

            paste = ''.join([tex_tabuline(x[:5]) for x in tex_lines])
            tex_analysis = "\\begin{tabular}{llllllllll}\n \\hline\n" \
                           "complexity & regr. error & avg. reward & fails & expression & decision plot & spiral plot & spiral difference-plot & histogram \\tabularnewline \\hline\n" \
                           f"{paste}" \
                           "\\hline\n\\end{tabular}\n\n"
            tex_analysis = latex_treeviz_full_document([tex_analysis])  # sfeh
            file_dump(self.root_dir / f'analysis_overview.tex', tex_analysis, print_type=self.print_type)  # agents_trees.tex

            paste2 = ''.join([tex_tabuline(x[:]) for x in tex_lines])
            tex_analysis = "\\begin{tabular}{llllllllll}\n \\hline\n" \
                           "complexity & regr. error & avg. reward & fails & expression & decision plot & spiral plot & spiral difference-plot & histogram \\tabularnewline \\hline\n" \
                           f"{paste2}" \
                           "\\hline\n\\end{tabular}\n\n"
            tex_analysis = latex_treeviz_full_document([tex_analysis])  # sfeh
            file_dump(self.root_dir / f'analysis_overview_plus.tex', tex_analysis, print_type=self.print_type)  # agents_trees.tex

        elif 'IB' in self.conf.name:
            self.file_pareto_listcode()

            if self.conf.name[-2:] == '_0':
                """
                - complexity_sum, [complexities], regression_sum, [regression_errors], real_sum, [formulas]
                - plot with pareto candidates
                """
                tex_combined = "\\begin{tabular}{llllllllll}\n" \
                               "\\hline \n" \
                               "\\multicolumn{2}{c}{complexity} & \\multicolumn{2}{c}{regression error} & \\multicolumn{4}{c}{Real evaluation} & expression \\tabularnewline\n" \
                               "sum &  & sum &  & & (safe) & (random) & (random, safe)  \\tabularnewline \\hline\n"

                """
                res_all: ordered list
                    {'parsim_sum': parsim_sum,
                      'experiment': None,
                      'experiment_safe': None,
                      'experiment_r50': None,
                      'experiment_safe_r50': None,
                      'parsims': parsims,
                      'codes': codes,
                      'regress_sum': regress_sum,
                      'regress_vals': regress_vals}
                """
                res_all = combined_lists(self.root_dir.parent, 40, 40, local_yamls=True, cpu_cores=cpu_cores)  # sfeh use self.conf.mp_cpu_cores_max

                xx = [x['parsim_sum'] for x in res_all]
                y_all = [y['experiment'] for y in res_all]
                y_safe = [y['experiment_safe'] for y in res_all]
                y_all_r50 = [y['experiment_r50'] for y in res_all]
                y_safe_r50 = [y['experiment_safe_r50'] for y in res_all]

                cnt = [y['cnt'] for y in res_all]

                # tex_combined += ' & '.join([xx, cnt, y_all, y_safe, y_all_r50, y_safe_r50])
                for y in res_all:
                    parsims = y['parsims']

                    try:
                        # \input{C:/Users/Rapid/PycharmProjects/plagih/benchmarks/slurm_runs/IB_MSE_scratch/IB_MSE_scratch_2/visualisation/01_input.tex}\tabularnewline
                        # {pars_ii:02d}
                        input_agentex = lambda run_ii: f"\\input{{{self.root_dir.parent.name}_{run_ii}/visualisation/{int(parsims[run_ii]):02d}_input_forest.tex}}"
                        oneplot_input = f"\\shortstack[l]{{{input_agentex(0)}\\\\{input_agentex(1)}\\\\{input_agentex(2)}}}"

                    except:
                        raise

                    tex_line = [f"{int(y['parsim_sum'])}",
                                ' '.join([f'{int(x)}' for x in y['parsims']]),
                                f"{y['regress_sum']:0.3f}",
                                ' '.join([f'{x:0.2f}' for x in y['regress_vals']]),
                                f"{y['experiment']:0.0f}",
                                f"{y['experiment_safe']:0.0f}",
                                f"{y['experiment_r50']:0.0f}",
                                f"{y['experiment_safe_r50']:0.0f}",
                                f"{y['cnt']}",
                                oneplot_input]
                    tex_combined += f"{' & '.join(tex_line)}\\tabularnewline \\hline \n"

                tex_combined += f"\\hline\n\\end{{tabular}}\n\n"
                tex_combined = latex_treeviz_full_document([tex_combined])
                file_dump(self.root_dir.parent / 'combined_overview.tex', tex_combined)

                with plt.rc_context(rc=pyplot_rc_tex):
                    fig, ax = plt.subplots()
                    ax.set(xlabel='Pareto complexity sum', ylabel='reward', ylim=funny_limits)
                    ax.plot(xx, y_all, label='all actions', marker='.', color='r')
                    ax.plot(xx, y_safe, label='low risk', marker='None', color='r', linestyle='dotted')
                    ax.plot(xx, y_all_r50, label='all actions (randomized)', marker='.', color='b')
                    ax.plot(xx, y_safe_r50, label='low risk (randomized)', marker='None', color='b', linestyle='dotted')
                    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

                    axx = ax.twinx()
                    axx.plot(xx, cnt, color='tab:gray', label='regression error', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
                    axx.tick_params(axis='y', labelcolor='tab:gray')

                    ax2 = ax.twinx()
                    ax2.plot(xx, cnt, color='tab:gray', label='Possible combinations', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
                    ax2.tick_params(axis='y', labelcolor='tab:gray')

                    ax.legend(loc='lower right')
                    ax2.legend(loc='lower left')

                    fig.savefig(self.root_dir.parent / f'regression_all-TEST.pdf')
                    plt.close('all')

        else:
            raise Exception(f'This should actually never happen right now. name: {self.conf.name}')

        return

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

        if path_data.suffix == '.p':
            data_prepared = pickle_load(path_data)
        elif path_data.suffix == '.csv':
            data_prepared = data_from_csv(path_data, action_name=action_name)
        else:
            raise FileNotFoundError(f'No data provided? File must be a pickle (.p) or csv (.csv) file. Loaded file: {path_data}')

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
                               ['Abs', 0.4], ['sign', 0.1], ['Round', 0.1],  # sfeh stop chain of arity-1 op in buid method?
                               ['sqrt', 0.25],
                               # ['log', 0.1], ['log1p', 0.1],  # sfeh
                               ['sin', 0.5],  # ['tan', 0.1], ['cos', 0.33], ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
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
        path_distrib = path_distrib or self.root_dir / self.paths.use_distributions_file

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

    def plot_agent_histogram(self, parsim, fitness, cooltree, path_hist):
        """
        Make histograms for all pareto-efficient candidates
        sfeh: based on training data- maybe use test data...

        useful code?
        # histogram_data = np.digitize(histogram_data, bins)  # digitize: the bin index the entry belongs to
        # histogram_data = np.concatenate((histogram_data.reshape(-1, 1), pairwise_fitness.reshape(-1, 1)), axis=1)
        # histogram_data = np.multiply.reduce(histogram_data, axis=1)
        # hist, bins = np.histogram(histogram_data, bins=bins, weights=pairwise_fitness)
        """

        # def plot_agent_histogram()

        action_bins = self.kernel.histogram_bins(self.env_vars.eval_action.minmax)
        expr_sym = cooltree.get_expr_sym()
        used_observations = cooltree.get_observation_list()
        pairwise_diff = self.kernel.eval_tf(expr_sym, used_observations)['pairwise_diff']

        with plt.rc_context(rc=pyplot_rc_tex):
            fig, ax = plt.subplots()
            ax.hist(pairwise_diff, bins=action_bins, histtype="stepfilled", facecolor="none", edgecolor='k')
            ax.set(ylim=(0, len(self.data_train)), ylabel='Frequency', xlabel='Deviation')
            histpath = path_hist / f'acthist_{parsim}.pdf'
            fig.savefig(histpath)
            plt.close('all')
        # self.printpl('ff', f'Histogram: {histpath.as_posix()}')

        return histpath

    def file_pareto_latex(self, parsim, fitness, cooltree):
        """
        Generates latex-file with the computational tree structure of all pareto agents
        - build tree from expression
        - fill tree meta-data, just in case we want to visualise anything of it
        - create latex-forest representation
        """

        """
        whole procedure from tree to forest core
        tight_viz:
            0: display every node
            1: clever tight-visualisation where possible
            2: one single mathematical expression
        """

        cooltree.set_fix_nodes(self.origin_cooltree)
        # cooltree.meta.last_evolution = 'texify'
        tree = cooltree.get_oldtree()

        plforest = lambda x: f'\\plforest{{{x}}}\n'

        forestree0 = plforest(latex_brackettree(tree))
        forestree1 = plforest(latex_brackettree_tight(latex_tree_semitight(tree)))
        texpression_2 = f'${latex_tight_node(tree)}$'
        forestree2 = plforest(f'[{texpression_2}]')

        """
        save every tree-visualisation in subfolder
        """
        path_subfolder_tex = path_make_dir(self.root_dir / self.paths.trees_sub_tex)  # sfeh running this in every tree seems unneccesary
        # create full document including just one tree (file must have a nice name)

        """
        The following lines delete this
        """
        # todo todotodo
        # full_tex0 = latex_treeviz_full_document([forestree0], doc_border='')
        # full_tex1 = latex_treeviz_full_document([forestree1], doc_border='')
        #
        # file_dump(path_subfolder_tex / f'full_{parsim:02d}.tex', full_tex0, print_type=self.print_type)
        # file_dump(path_subfolder_tex / f'full_{parsim:02d}_tight.tex', full_tex1, print_type=self.print_type)
        #
        # file_dump(path_subfolder_tex / f'{parsim:02d}_input.tex', texpression_2, print_type=self.print_type)
        # file_dump(path_subfolder_tex / f'{parsim:02d}_input_forest.tex', forestree2, verbose='ff', print_type=self.print_type)
        # f'{parsim} with mean Regression Error {fitness}:\n {forestree0} tight: {forestree1} tight2: {texpression_2}\n\n\n'

        return forestree0, forestree1, texpression_2, forestree2

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

        """
        dude sfeh
        sfeh for fully executable code
        bad code
        """
        if 'MTC75' in self.conf.name:
            srsaagnt = 75
        elif 'MTC200' in self.conf.name:
            srsaagnt = 200
        else:
            raise

        pyc_complete = f"import math; import numpy as np\n" \
            "import sys\n" \
            "from pathlib import Path\n" \
            "sys.path.append(str(Path(Path.cwd() / '../../../'))\n" \
            "from benchmarks.MC.agents.quick_eval import *\n" \
            "from pathlib import Path\n" \
            "folder = Path.cwd() / 'custom_files'\n" \
            "from benchmarks.MC.agents.mtc_agent_sarsa import * \n" \
            f"with Path.open(Path(Path.cwd() / '../../../benchmarks/MC/agents/sarsa_agent_{srsaagnt}.p'), 'rb') as file:\n" \
            "\tsarsa_agent = pickle.load(file)\n" \
            "\n" \
            f"{all_agents}\n" \
            f"all_agents_more = [{all_more_info}]\n" \
            f"agent_tuples = [{agent_tuples}]\n" \
            f"\n\n" \
            "if __name__ == '__main__':\n" \
            "\tprint('executing!')\n" \
            "\teval_agent_list(agent_tuples, folder=folder, goal_agent=sarsa_agent)\n"

        pth = path_make_dir(self.root_dir / self.paths.folder_pycode / f"agents.py")
        with Path.open(pth, 'w') as file:
            file.write(pyc_complete)
            self.printpl('ff', f'Pycode: {pth.as_posix()}')

        return

    def file_pareto_listcode(self):
        """
        save python code for Industrial Benchmark runs
        delete sometime
        """

        # pycode_agent = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')

        pygents_list = []

        for (parsim, fitness, cooltree) in self.pareto:
            # agent_name = f'{self.conf.name}_{parsim:.0f}'
            agent_name = f'{self.conf.name}_{self.env_vars.eval_action.name}_{parsim:.0f}'
            agent_as_python = cooltree.get_pycode()
            pygents_list.append([parsim, float(fitness), agent_name, agent_as_python])

        yaml_dump(self.root_dir / 'pycode_list.yaml', pygents_list, print_type=self.print_type)
        path = path_make_dir(self.root_dir / 'pycode_list.yaml')
        with Path.open(path, 'w') as file:
            _ = yaml.dump(pygents_list, file)  # , default_flow_style=False, sort_keys=False)
            printez('ff', f'IB pycode-list: {path.as_posix()}', print_type=self.print_type)  # sfeh always the same print structure... just pass the path?

        return

    def treelut_tree_add(self, cooltree: CoolTree):
        """
        update selected values in self.tree_meta
        LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        """

        meta = {'fitness_train': cooltree.meta.fitness_train,
                'parsimony': cooltree.meta.parsimony,
                'expr_raw': cooltree.meta.expr_raw,
                'expr_sym': cooltree.meta.expr_sym}

        tree_ident = hash(cooltree)  # attention: hashes change between python runs. do not save anything on their hash values <.<

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
        The call parameters in the evolution file need to be adjusted
        delete if possible
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
        Creates random trees for the population
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
            print_warning('ww', f'Crossover conversion for these trees not possible: \n{left_tree}\n{right_tree}', print_type=self.print_type)
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
            self.printpl('a', f"New entry found! ({msg}): {BColors.RESET}{tree_entry[0]}, {tree_entry[1]}:{BColors.RESET} {tree_entry[2].meta.expr_raw}")
        self.pareto.append(tree_entry)
        self.gens_since_last_pareto = 0

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
                self.update_pareto_with_tree(cooltree_sym)
        except Exception as ex:
            print_warning('www', f'Tree sympification did not work: {ex}', print_type=self.print_type)

        else:
            self.printpl('aaa', 'Pareto entry was already simplified')

    def update_pareto_from_pop_tmp(self):
        """
        inserts a tree into the pareto front
        """

        # first, make a local pareto front
        self.gens_since_last_pareto += 1
        pop_parsimonies_dict = dict.fromkeys(sorted(set([cooltree.meta.parsimony for cooltree in self.population_tmp])))
        for ct in self.population_tmp:
            p = ct.meta.parsimony
            try:
                best_yet = pop_parsimonies_dict[p]
                if ct.meta.fitness_train < best_yet.meta.fitness_train:  # sfeh asd kernel required for fitness comparison
                    pop_parsimonies_dict[p] = ct
            except:
                pop_parsimonies_dict[p] = ct

        for p, cooltree in pop_parsimonies_dict.items():
            self.update_pareto_with_tree(cooltree)

        self.pareto_sort()  # sfeh check if required
        return

    def update_pareto_with_tree(self, cooltree: CoolTree):
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

    def pop_append(self, cooltree: CoolTree):
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
            print_warning('w', f'tree failed the quick check. last-mod: {cooltree.meta.last_evolution}. Reason:\n{ex}', print_type=self.print_type)
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
                print_warning('wwww', f'Parsimony too high, last evolution: {cooltree.meta.last_evolution}', print_type=self.print_type)  # sfeh care about wwww. should not
                return
            try:
                fitness_train = self.tree_eval_fitness_train(cooltree)
            except Exception as evalex:
                print_warning('wwww', f'Exception while evaluating: {evalex}', print_type=self.print_type)
                return

        expr_raw = cooltree.get_expr_raw()
        expr_sym = expr_sympify(expr_raw)
        try:
            cooltree.set_fix_nodes(self.origin_cooltree)
        except Exception as ex:
            print(f'NOOOOOOPE, failed tree finish: {ex}\n{cooltree}')

        cooltree.meta.fitness_train = fitness_train
        cooltree.meta.parsimony = parsimony
        cooltree.meta.expr_raw = expr_raw
        cooltree.meta.expr_sym = expr_sym
        # cooltree.set_meta(fitness_train, parsimony, last_evolution, expr_raw, expr_sym)

        self.treelut_tree_add(cooltree)
        self.population_tmp.append(cooltree)
        return

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
        fitness_train = round(tf_origin_results['mean_error'], self.conf.float_decimals)  # fitness currently IS the mean error
        if self.kernel.exploration_risk:
            self.kernel.origin_results = tf_origin_results['results_kernel']  # after getting the origin-results, these informations can be updated

        origin_cooltree.meta.fitness_train = fitness_train
        origin_cooltree.meta.parsimony = 0

        self.pareto.append([0, fitness_train, origin_cooltree])  # the origin tree is the only candidate for now -> it is in the pareto front
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
        fitness_train = round(self.kernel.eval_tf(expr_sym, used_observations, only_fitness=True), self.conf.float_decimals)

        """
        Returns bool value if we can use the calculated fitness
        Fitness values might evaluate to weird stuff
        e.g. 'nan' after dividing by zero or (inf) after 20**1234
        nan: fitness == fitness -> False
        inf: fitness is not float('inf') -> False
        """
        if fitness_train != fitness_train or fitness_train == float('inf'):
            raise Exception(f"fitness is: '{fitness_train}'")  # happens, eg when values are soo wrong that it leaves the float-range
        # fitness_train = round(fitness_train, self.conf.fitness_decimals)

        return fitness_train

    # def plot_rc_default(self):
    #     rc('font', weight='bold')    # bold fonts are easier to see
    #     rc('tick', labelsize=15)     # tick labels bigger
    #     rc('lines', lw=1, color='k') # thicker black lines
    #     rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    #     rc('savefig', dpi=300)       # higher res outputs

    def plot_gen_performance(self):
        """
        All monitoring infos
        sfeh den shit in Funktionen aufteilen
        """
        with plt.rc_context(rc={'axes.grid': True}):
            fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]}, sharex='all')  # , figsize=(9, 9)
            plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
            xx = list(self.monitor_df.index)

            axs0 = axs[0]
            axs0.plot(self.monitor_df['fit_avg'], marker='', label='Regression error (average)')
            try:
                avg = self.monitor_df['fit_avg']
                std = self.monitor_df['fit_std']
                axs0.fill_between(xx, avg - std, avg + std, alpha=0.2)  # axs0.set_title('Regression Error (average)')  # sfeh not stderr... upper/lower bound?
            except Exception as ex:
                raise Exception(f'Delete this. were there any problems? {ex}')
            axs0.step(x=xx, y=self.monitor_df['fit_best'], linestyle='dashed', marker='', where='post', color='g', label='Best candidate')  # , label=ax_label
            axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

            axs0_twin = axs0.twinx()
            axs0_twin.plot(xx, self.monitor_df['gens_since_last_pareto'], color='tab:gray', label='Generations since last pareto entry', linestyle='dashed', marker='')  # linestyle='None'
            axs0_twin.tick_params(axis='y', labelcolor='tab:gray')
            try:
                axs0_twin.set_ylim(ymin=0, ymax=max(self.monitor_df['gens_since_last_pareto'].max() or 1, 50))
            except Exception as ex:
                try:
                    print_e(f'damn setting ylim not worksening :s {ex}')
                    axs0_twin.set_ylim(ymin=0, ymax=max(self.monitor_df['gens_since_last_pareto'].notnull().max() or 1, 50))
                    # print(self.monitor_df['gens_since_last_pareto'].notnull().max())
                except Exception as ex2:
                    print_e(f'damn setting ylim not working, version 2! {ex2}')
                    axs0_twin.set_ylim(ymin=0, ymax=50)

            axs0_twin.legend(loc='lower right')

            axs1 = axs[1]
            axs1.plot(self.monitor_df['parsim_avg'], label='Complexity (average)')
            # self.conf.complexity_measure

            try:
                p_avg = self.monitor_df['parsim_avg']
                p_std = self.monitor_df['parsim_std']
                axs1.fill_between(xx, p_avg - p_std, p_avg + p_std, alpha=0.2)  # axs1.set_title('TED (average)')
            except Exception as ex:
                raise Exception(f'Delete this if no raise since some time. {ex}')
            axs1.set_ylim(ymin=0), axs1.legend(loc='lower left')

            axs2 = axs[2]
            axs2.plot(self.monitor_df['pop_len'], label='pop size')
            axs2.plot(self.monitor_df['pop_unique'], label='unique')
            axs2.margins(y=0.25), axs2.set_ylim(ymin=0), axs2.legend(loc='lower left')

            axs3 = axs[3]
            between_outliers = self.monitor_df['time'].between(0, 2 * self.monitor_df['time'].mean())
            axs3.plot(self.monitor_df['time'][between_outliers], label='time (s)')  # sfeh could be a better rule...
            axs3.set_ylim(ymin=0), axs3.legend(loc='lower left')

            # Top level style
            axs3.set_xlim(xmin=0, xmax=max(xx)), axs3.set_xlabel('generations')
            axs0.set_title(f'monitoring gp generations {self.conf.name}')  # sfeh
            fig.tight_layout()
            path = self.root_dir / f'monitoring-{self.conf.name}.pdf'
            fig.savefig(path, dpi=300)
            self.printpl('f', f"monitoring: {path.as_posix()}")
            plt.close('all')

    def plot_paretofront(self):
        """
        Plot pareto candidates
        """
        tuples = [[parsim, fitness] for (parsim, fitness, cooltree) in self.pareto]
        xx, yy = np.array(tuples).T

        if len(xx) == 0:
            print_e(f'Plotting empty array is not possible! Data={xx, yy}')
            return

        run_name = self.conf.name
        run_name = str(run_name).replace('_', '-')  # todo workaround for latex version

        with plt.rc_context(rc=pyplot_rc_tex):
            # fig, ax = plt.subplots(figsize=pyplot_size)
            fig, ax = plt.subplots(figsize=pyplot_size)  # , figsize=(9, 9)
            right = max(max(xx), self.conf.parsimony_max) * 1.05  # sfeh check this out 1.05  # if set_right:

            # beyond_lines:  # adding a point to the edges to imply that there are no more values (pareto-plot)
            xx = np.concatenate([[xx[0]], xx, [right + 1]])
            yy = np.concatenate([[max(yy) + 1], yy, [yy[-1]]])

            ax.step(xx, yy, linestyle='dashed', marker='.', label=f'{run_name}', where='post')

            # ax.legend(loc='lower left')

            ax.set(xlabel='complexity', ylabel='regression error',
                   xlim=(min(min(xx), 0), right),
                   ylim=(min(min(yy), 0), (max(yy) - min(min(yy), 0)) * 1.05))

            try:
                path_plot = path_make_dir(self.root_dir / 'plots/')
                fig.savefig(path_plot / f'paretofront.pdf', backend='pgf')  # run_name?
                # fig.savefig(path_plot / f'paretofront.png', dpi=300)
                self.printpl('f', f"paretofront (pdf): {path_plot.as_posix()}")
            except PermissionError as permerr:
                print_e(f'Could not save plot: {permerr}')  # sfeh for everything?

        return

    def real_evaluation(self):
        pass

    def plot_evolve_performance(self):
        """
        Plots for each tag in the evolution list
        (too much, i guess)
        sfeh: this should be saved within the trees. Everything else is a waste of memory!
        """
        try:
            with plt.rc_context(rc=pyplot_rc_tex):
                fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 9), sharex='all')  # , gridspec_kw={'height_ratios': [1,1,1]}
                fig.tight_layout()
                for tag in self.evolve_tags:
                    # ['fitness', 'parsimony', 'lentree', 'evolve_num', 'count']
                    axs[0].plot(self.monitor_evol[tag]['fitness'], label=f'{tag}')
                    axs[1].plot(self.monitor_evol[tag]['parsimony'], label=f'{tag}')
                    axs[2].plot((self.monitor_evol[tag]['lentree'] / self.monitor_evol[tag]['evolve_num']), label=f'{tag}')
                    # axs[0].set_ylim(ymin=0), axs[0].legend(loc='lower left')
                    # axs[1].set_ylim(ymin=0), axs[1].legend(loc='lower left')
                    # axs[2].set_ylim(ymin=0), axs[2].legend(loc='lower left')

                plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
                path = self.root_dir / f'monitoring_evolutions.pdf'
                fig.savefig(path, dpi=300)
                self.printpl('f', f"monitoring_evolutions (pdf): {path.as_posix()}")

        except Exception as ex:
            print_e(f'plot_evolution_analysis failed because of: {ex}')

    def file_analysis_plots(self):
        """
        Make all plots
        """

        self.plot_gen_performance()  # largest plot analysing the
        self.plot_paretofront()
        # self.plot_evolve_performance()  # sfeh
        return

    def pop_analyse(self):
        """
        Analysing this generation
        - amount of trees
        - fittest tree
        - average fitness
        - average tree parsimony
        """
        # t = time.perf_counter()  # should this be used?
        gen_id = self.gen_id
        popul = self.population_tmp

        if len(popul) == 0:
            raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

        pop_fitness = [cooltree.meta.fitness_train for cooltree in popul]

        # tmp_evol_performance = dict.fromkeys(self.monitor_evol.keys(), pd.DataFrame(columns=['fitness', 'parsimony', 'lentree']))
        # for cooltree in popul:
        #     last_evol = cooltree.meta.last_evolution
        #     if last_evol in self.evolve_tags:
        #         row = {'fitness': cooltree.meta.fitness_train,
        #                'parsimony': cooltree.meta.parsimony,
        #                'lentree': len(cooltree)}
        #         tmp_evol_performance[last_evol].loc[self.gen_id] = row
        #
        # for last_evol, evodata in tmp_evol_performance.items():
        #     if last_evol in self.evolve_loop:
        #         try:
        #             row = {'fitness_avg': evodata['fitness'].mean(),
        #                    'parsimony_avg': evodata['parsimony'].mean(),
        #                    'lentree_avg': evodata['lentree'].mean(),
        #                    'evolve_num': self.evolve_loop[last_evol]['evolve_num'],
        #                    'count': len(tmp_evol_performance[last_evol])}
        #             self.monitor_evol[last_evol].append(row, ignore_index=True)  #
        #             # sfeh fitness - last fitness?
        #         except Exception as ex:
        #             print_e(f'Could not save evol_performance analysis. {ex}')
        #     else:
        #         if last_evol != 'origin' and last_evol != 'Rand3o':
        #             print_warning('w', f'delete_this, sfeh, okay when the following is origin: {last_evol}')

        pop_parsim = [cooltree.meta.parsimony for cooltree in popul]
        pop_treelen = [len(cooltree) for cooltree in popul]

        pop_fitness_best = self.kernel.np_best_fitness(pop_fitness)
        try:
            self.best_fitness = self.kernel.best_fitness_function(pop_fitness_best, self.best_fitness)
        except:
            self.best_fitness = pop_fitness_best

        unique_tree_count = len(set([hash(x) for x in popul]))  # sfeh analyze this?

        gen_time = time.perf_counter() - self.time_genstart

        self.monitor_df.loc[gen_id] = {'pop_len': len(popul),
                                       'pop_unique': unique_tree_count,
                                       'fit_avg': np.average(pop_fitness),
                                       'fit_std': np.std(pop_fitness),
                                       'fit_best': self.best_fitness,
                                       'parsim_avg': np.average(pop_parsim),
                                       'parsim_std': np.std(pop_parsim),
                                       'complexity_avg': np.var(pop_treelen),
                                       'time': gen_time,
                                       'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh version1 delete this shit

        self.print_g('gg', f'Created {len(popul)}/{self.conf.pop_max} ({unique_tree_count} unique) in generation {gen_id}. Gen took {gen_time:4.2f}s')
        # sfeh check if there are really unique... doubt it.

        """
        show plots if necessary
        """
        if self.gen_id % int(self.conf.period.get('gen_plots', 1)) == 0:
            self.file_analysis_plots()

        if self.gen_id % int(self.conf.period.get('gen_save', 1)) == 0:
            self.run_backup_save()
        return

    def printpl(self, message_type, message_str):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        message_type options can be found in config
        """
        if message_type in self.print_type:
            printez(message_type, message_str, print_type=self.print_type)
        return

    def print_g(self, message_type, text):
        """
        print informations about generations (g) progress.
        Very common print, showing the progress of the generations.
        always printing the time since start. used to be colored blue.
        """
        if message_type in self.print_type:
            # time_passed = time.perf_counter() - self.time_start
            time_now = time.strftime("%H:%M", time.localtime())
            print(f'{time_now}. {text}')
        return
