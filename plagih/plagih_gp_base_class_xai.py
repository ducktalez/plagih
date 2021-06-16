"""

Functions, that might be addable in the future:
"""
import copy
import math
import random
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from benchmarks.ib.combined_runs import *
from benchmarks.mc.agents.quick_eval import auto_evaluate_run_end
from plagih.file_interaction import *
from plagih.plagih_config import *
from plagih.tree_factory import ChooseConstants
from plagih.viz_with_latex import *

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


# def mp_dummy(arg, **kwarg):
#     return Nu.e2(arg, **kwarg)


class FileLocations:
    backup_p = 'backup/backup.p'
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

        """
        load relevant stuff
        """
        with Path.open(path_data) as file:
            # self.env_vars, choose_observations = data_from_csv(df, action_name=self.conf.action_name)
            df = pd.read_csv(file, delimiter=',')
            # todo it is float64, float64, int64 with MTC.. does it work with Tensorflow?
            df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P Following design pattern #YOLO

            for ii, header in enumerate(df):
                if header in op:
                    raise Exception(f'Your samples hold a column that matches the potential tree operator {header}.\n'
                                    f'That might end up in confusion, please rename the column.')

            if self.conf.action_name is None:
                self.conf.action_name = df[len(df.columns) - 1]

            df = df.drop(self.conf.dc, axis=1)  # no need to keep other actions
            printez('i', f'Ignoring columns: {self.conf.dc}')  # , print_type=print_type print_type does not exist yet sfeh
            csv_observations = list(df.columns)
            csv_observations.remove(self.conf.action_name)
            choose_observations = ChooseObservation(csv_observations)

            self.data_train, self.data_control = train_test_split(df, test_size=0.2, random_state=0)  # discussion: random state 0 okay? test_size 0.2?
            self.kernel = RegressionKernel(self.conf.kernel_name, self.data_train, self.tf_config, "/gpu:0",
                                           self.conf.action_name)  # sfeh Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device. Is cpu otherwise

        choose_distributions = self.activate_distributions(path_distrib=None)  # asd sfeh path_distrib not None
        choose_ops = self.gp_load_oparray(path_operators=None)  # path_operators sfeh this file from config version1

        self.tb = TreeBuilder(choose_ops, choose_observations, choose_distributions, self.conf.float_decimals)

        # Evaluating kernel (that uses tensorflow)
        self.tf_config = tf.compat.v1.ConfigProto(log_device_placement=self.conf.tf_device_log,
                                                  allow_soft_placement=True)  # TF device usage logging (for debugging) (default false. I lately used it to check if the GPU is used)
        self.tf_config.gpu_options.allow_growth = True
        self.print_type = self.conf.print_type  # sfeh hmmm remove

        self.pareto = []  # a list with all pareto candidates. key is complexity, value is tree meta. [[1,344, meta], ...]

        if path_origin_tree:
            # todo make origin class
            self.origin = self.load_origin_tree(path_origin_tree)
            self.origin.meta.last_evolution = 'origin'
            self.origin_is_fix = self.origin.core.is_fix
        else:
            self.origin = None
            self.origin_is_fix = False

            if self.conf.complexity_measure in ['tree_edit_distance']:
                self.conf.complexity_measure = 'tree_node_count'  # sfeh idea
                print_warning('w', "Complexity measurement 'tree_edit_distance' is not possible without origin!\n"
                                   "Using 'tree_node_count' instead.", print_type=self.print_type)

        # init values with dummies (just to have all self values here for overview)
        self.tree_lut = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.population = []
        self.pop_base = []  # population that is taken to the next generation
        self.best_fitness = None  # keeps track of the current best fitness
        self.gen_id = 0

        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'complexity_avg', 'complexity_var', 'complexity_stderr',
                                                'gens_since_last_pareto'])

        self.evolve_loop, self.evolve_random = self.make_evolve_rates()
        self.evolve_tags = list(self.evolve_loop.keys()) + list(self.evolve_random.keys())

        # self.monitor_evol = dict.fromkeys(self.evolve_tags, pd.DataFrame(columns=['fitness', 'parsimony', 'lentree', 'evolve_num', 'count']))

        self.print_g('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return

    def make_evolve_rates(self):
        """
        The evolve_dict is converted into a list of the population length.
        The evolve_loop is for the regular evolution, the evolve_random is especially required for the first generation.
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
        # for (parsimony, fitness_train, tree) in pareto_list:
        #     fitness_train = round(fitness_train, self.conf.float_decimals)
        #     tree.meta.fitness_train = round(fitness_train, self.conf.float_decimals)
        #     entry = [parsimony, fitness_train, tree]
        #     self.pareto_append(entry)
        # self.printpl('i', f'Updating pareto front from old run. length: {oldsize}, new pareto length: {len(self.pareto)}')

        # sfeh recompute parsimony and fitness for every tree (optional?)
        # also rebuild trees?

        poplen = len(self.pop_base)
        pop_base_copy = self.pop_base[:]
        self.pop_base = []
        for tree in pop_base_copy:
            tree.meta.fitness_train = round(tree.meta.fitness_train, self.conf.float_decimals)
            self.pop_base.append(tree)
        self.printpl('i', f'Updating population from old run. length: {poplen}, new population length: {len(self.pop_base)}')

    def pareto_sort(self):
        """
        sorting the pareto entries in list for parsimony
        """
        self.pareto.sort(key=lambda x: x[0])

    def custom_exit_condition(self):
        """
        Special condition to exit the evolve-loop
        1. when >100 generations, no new pareto entries were found
        """
        try:
            if self.monitor_df['gens_since_last_pareto'].iloc[-1] > 100:
                print('SFEH This condition made your program exit!')
                return True
            else:
                return False
        except Exception:
            return False

    def plagih_gp_run(self, gen_additionally):
        """
        regular plagih run
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.gen_id + gen_additionally)
            self.printpl('i', f'Adding new generations, gen_max was {printdummy}, current gen {self.gen_id}. gen_additionally: {gen_additionally}. New max gen: {self.conf.gen_max}')

        # sfeh
        # yaml_dump(self.root_dir / 'used_config.yaml', self.conf, print_type=self.print_type)

        self.gens_since_last_pareto = 0

        while self.gen_id <= self.conf.gen_max and not self.custom_exit_condition():  # max generation, max time, done...
            self.time_genstart = time.perf_counter()

            if self.gen_id == 0:
                self.print_g('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')
                self.gen_create_initial()  # sfeh stattdessen einfach checken, ob die letzte population leer ist und info/warnung: neue generation?
            else:
                # This might be a solution for multiprocessing:
                # You can avoid this situation by calling multiprocessing.Process before you load your huge data.
                # Then the additional memory allocations will not be reflected in the child process when you load the data in the parent.
                # sfeh: In python 3.8, this might be availably: multiprocessing.shared_memory https://docs.python.org/3/library/multiprocessing.shared_memory.html
                # sfeh: check memory usage! should not scale with the number of processes, only one pop_base is required, it does not change.

                self.mp_cpu_cores_max = 1  # sfeh wasd
                if self.mp_cpu_cores_max >= 2:
                    pass
                    # # sfeh asd
                    # mp.Process()  # sfeh maybe good for memory? https://stackoverflow.com/questions/14749897/python-multiprocessing-memory-usage
                    # print(f'Trying to make parallel new population: {mp.cpu_count()}')
                    # with mp.Pool(min(mp.cpu_count(), self.mp_cpu_cores_max)) as p:
                    #     evolve_list = [[tag, evolve_specs] for tag, evolve_specs in self.evolve_loop.items()]
                    #     results = p.map(fun, evolve_list)
                    # time_evolve = time.perf_counter()
                else:
                    self.gen_next_population()
                    pass

            self.update_pareto_from_pop_tmp()

            self.pop_analyse()

            self.pop_base = self.population[:]
            self.population = []
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

        if self.origin is not None:
            self.pop_append(self.origin)  # sfeh why not :P
        else:
            total_rate = sum([x['evolve_rate'] for x in self.evolve_random.values()])

            for tag, evolve_specs in self.evolve_random.items():
                evolve_num = int(self.conf.pop_max * (evolve_specs['evolve_rate'] / total_rate))
                call_params = evolve_specs.get('custom_params')

                for nn in range(evolve_num):
                    if self.origin_is_fix:
                        tree = self.tb.pop_random(call_params, from_origin=True)
                    else:
                        tree = self.tb.pop_random(call_params)
                    # tree.meta.last_evolution = tag  # todo
                    self.pop_append(tree)
        return

    def gen_next_population(self):
        """
        Creates all new Generations by applying the evolutions in the evolve-loop.

        brainstorm:
        - the tree can be reproduced (selection), random/new, olymp-reproduction
        - (1 tree) mutations can affect a point, branch, terminal nodes
        - (2 trees) can make a crossover
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
                    tree = self.pop_selection_tournament(tourn_size)
                    if call_params.get('sympify_tree'):
                        try:
                            tree.evolve_reduce(obs_infos=self.env_vars.obs_infos, completely=False)
                            tree.meta.last_evolution = tag
                        except Exception as ex:
                            print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.print_type)

                    self.pop_append(tree)  # append anyways... it was worth a try :P

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    tree = self.pop_selection_tournament(tourn_size)
                    tree.evolve_mutate_point_random(self.tb)
                    # tree.meta.last_evolution = tag
                    self.pop_append(tree)

            elif evolve_name == 'mutate branch':
                # todo tree version
                # todo question: tree.copy() or .deepcopy necessary??
                for nn in range(evolve_num):
                    build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)
                    full_or_grow = build_spec.get('full_or_grow') or random.choice(['full', 'grow'])
                    tree = self.pop_selection_tournament(tourn_size)
                    cool_build_size = choose_build_size(size_mode, mean_min_max_var, tree=tree)
                    tree.evolve_mutate_branch(cool_build_size, self.tb, size_mode=size_mode, full_or_grow=full_or_grow)
                    # sfeh delete this?
                    # new_tree = tree_prune_depth(new_tree, self.conf.tree_depth_max, self.env_vars.choose_obs, self.choose_distributions, self.conf.float_decimals)
                    # new_tree.meta.last_evolution = tag
                    self.pop_append(tree)

            elif evolve_name == 'crossover branch':
                for nn in range(int(evolve_num / 2)):  # two childs
                    left_parent = self.pop_selection_tournament(tourn_size)
                    right_parent = self.pop_selection_tournament(tourn_size)

                    """
                    swap branches of two trees
                    - select parent a and b
                    - select swappable branche for a_parent from b_parent
                        - select a node in a (and crossover here, no matter what)
                    - delete a_parent branch and insert b_parent branch (which tactic?)
                    todo into main tree?
                    """
                    # 1. two parents
                    # 2. search nodes for left and right that can be exchanged. convert_needed
                    try:
                        left_parent_nodes = left_parent.get_mutatable_nodes(allow_root=False)
                        left_rnd = random.choice(left_parent_nodes)
                        swap_coolxtype_out = left_rnd.label.coolxtype[1]
                        right_parent_nodes = right_parent.get_mutatable_nodes(coolxtype_out=swap_coolxtype_out)
                        if right_parent_nodes:
                            right_rnd = random.choice(right_parent_nodes)
                        else:
                            swap_coolxtype_out = float if swap_coolxtype_out == bool else bool  # the other swap type now
                            right_parent_nodes = right_parent.get_mutatable_nodes(allow_root=False, coolxtype_out=swap_coolxtype_out)
                            right_rnd = random.choice(right_parent_nodes)
                            left_rnd = left_parent.get_mutatable_nodes(coolxtype_out=swap_coolxtype_out)
                    except:
                        raise Exception

                    # todo deepcopy required??
                    left_parent.new_core(right_rnd)
                    right_parent.new_core(left_rnd)

                    left_parent.meta.last_evolution = tag
                    right_parent.meta.last_evolution = tag
                    self.pop_append(left_parent)
                    self.pop_append(right_parent)

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    tree = self.pop_selection_tournament(tourn_size)
                    tree.evolve_mutate_filter_random(call_params, self.tb)
                    tree.meta.last_evolution = tag
                    self.pop_append(tree)

            elif evolve_name == 'revive pareto':

                for nn in range(evolve_num):
                    fitness_train, parsim, tree = random.choice(self.pareto)
                    tree.meta.last_evolution = tag
                    self.pop_append(tree)

            elif evolve_name == 'random trees':

                if self.origin_is_fix:
                    for nn in range(evolve_num):
                        tree = self.tb.pop_random(call_params, from_origin=True)
                        tree.meta.last_evolution = tag
                        self.pop_append(tree)
                else:
                    for nn in range(evolve_num):
                        tree = self.tb.pop_random(call_params)
                        tree.meta.last_evolution = tag
                        self.pop_append(tree)
            else:
                print_e(f"Evolution not known: '{evolve_name}'")

        missing_trees = self.conf.pop_max - len(self.population)
        if missing_trees > 0:
            if missing_trees > 0.05 * self.conf.pop_max:
                self.printpl('ii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            else:
                self.printpl('iii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            while len(self.population) < self.conf.pop_max:
                return  # sfeh aka create trees here if desired

        # sfeh automatically fill with random trees (check this at the initiation)
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        return

    def file_pareto_txt(self):
        """
        Save all the pareto candidates to a file.
        (Quick feedback that requires little overhead)
        """
        pareto_yaml = [f'Parsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{tree.meta.expr_raw}' for (parsim, fitness, tree) in self.pareto]
        yaml_dump(self.root_dir / 'paretofront.yaml', pareto_yaml, print_type=self.print_type)

        return

    def analyse_pareto(self, cpu_cores=16):
        """
        Writing all analysis files after evaluating the paretofront.
        (Currently strongly customized by sfeh for the mountaincar and industrial benchmark)
        """
        self.printpl('i', f'Analysing the pareto candidates of the run.')

        dir_benchmarks = Path(__file__).parent.parent.absolute() / 'benchmarks/'
        path_hist = path_make_dir(self.root_dir / 'histograms/')
        pareto_agents = {}

        for (parsim, fitness, tree) in self.pareto:
            histograms_path = self.plot_agent_histogram(parsim, tree, path_hist)  # sfeh todo

            forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest = self.file_pareto_latex(parsim, tree)

            pareto_agents[parsim] = {'parsim': parsim,
                                     'fitness': fitness,
                                     'tree': tree,
                                     'forest_tree_full': forest_tree_full, 'forest_tree_tight': forest_tree_tight, 'tex_expr_raw': tex_expr_raw, 'tex_expr_forest': tex_expr_forest,
                                     'histogram': histograms_path,
                                     }

        tex_include_pdf = lambda x: f"\\includegraphics{{{str(x).replace('.pdf', '')}}}"
        tex_tabuline = lambda x: f"{' & '.join(x)}\\tabularnewline\n"
        tex_stacklist = lambda x: "\\shortstack[l]{{{}}}".format('\\\\'.join([str(xx) for xx in x]))

        if 'MTC' in self.conf.name:
            """
            complexity, regr. error, real evaluation, decision plot, spiral plot, diff-plot
            """
            # sfeh i guess not necessary anymore (?)
            # self.file_pareto_pycode()
            sarsa_agent_steps = 200 if 'MTC200' in self.conf.name else 75 if 'MTC75' in self.conf.name else 'NO_MC_AGENT'
            sarsa_agent = pickle_load(dir_benchmarks / f'mc/agents/sarsa_agent_{sarsa_agent_steps}.p')

            agent_performance = auto_evaluate_run_end(self.root_dir, sarsa_agent, n=100)
            # df_mtc = pd.DataFrame(columns=['complexity', 'regr. error', 'avg. reward', 'fails', 'expression'])
            tex_lines = []

            for pp, x in agent_performance.items():
                pp, fitness, avg_reward, fails, path_mcmeshplot, path_mcmeshplot_diff = x

                tex_lines.append([f"{int(pp)}",
                                  f"{pareto_agents[pp]['fitness']:.2f}",
                                  f"{avg_reward:0.1f}",
                                  f"{pareto_agents[pp]['tex_expr_raw']}",
                                  f"{tex_include_pdf(f'sfehs_eval/{pp}.pdf')}",  # path_mcmeshplot
                                  # tex_include_pdf(f'sfehs_eval/diff-{pp}.pdf'),  # path_mcmeshplot_diff),
                                  tex_include_pdf(f'sfehs_eval/space-{pp}.pdf'),
                                  # tex_include_pdf(f'histograms/acthist_{pp}'),
                                  f"{pareto_agents[pp]['forest_tree_full']}",  # forest_tree_full, forest_tree_tight
                                  f"{pareto_agents[pp]['forest_tree_tight']}",
                                  f'{fails}'])

            paste = ''.join([tex_tabuline(x[:4]) for x in tex_lines])
            paste = "\\begin{longtable}[c]{>{\\LTleft}p{5mm}>{\\LTleft}p{6mm}>{\\LTleft}p{10mm}>{\\LTleft}p{102mm}}\n\\hline\n" \
                    "dist & error & reward  & expression \\tabularnewline \\hline\n" \
                    f"{paste}" \
                    "\\hline\n\\end{longtable}\n"
            file_dump(self.root_dir / f'analysis_input.tex', paste, print_type=self.print_type)
            file_dump(self.root_dir / f'analysis_overview.tex', latex_treeviz_full_document(paste), print_type=self.print_type)

            paste_full = ''.join([tex_tabuline(x[:]) for x in tex_lines])
            paste_full = f"{str(self.conf.name).replace('_', '-')}  {tex_include_pdf('monitoring.png')}  {tex_include_pdf('sfehs_eval/evaled_overview.pdf')}\n\n" \
                         "\\begin{tabular}{lllllllllllll}\n \\hline\n" \
                         "dist & error & reward & parsimony & expression \\tabularnewline \\hline\n" \
                         f"{paste_full}" \
                         "\\hline\n\\end{tabular}\n\n"
            file_dump(self.root_dir / f'analysis_overview_plus.tex', latex_treeviz_full_document(paste_full), print_type=self.print_type)

        elif 'IB' in self.conf.name:

            self.file_pareto_listcode()

            if self.conf.name[-2:] == '_0':
                """
                - complexity_sum, [complexities], regression_sum, [regression_errors], real_sum, [formulas]
                - plot with pareto candidates
                """
                # "\\multicolumn{2}{c}{complexity} & \\multicolumn{2}{c}{regression error} & \\multicolumn{4}{c}{IB reward} & combinations & expression \\tabularnewline\n" \
                tex_line_input, tex_line_overview = '', ''

                res_all = combined_lists(self.root_dir.parent, 40, 40, local_yamls=True, cpu_cores=cpu_cores)  # sfeh use self.conf.mp_cpu_cores_max

                xx = [x['parsim_sum'] for x in res_all]
                y_all = [y['experiment'] for y in res_all]
                y_safe = [y['experiment_safe'] for y in res_all]
                y_all_r50 = [y['experiment_r50'] for y in res_all]
                y_safe_r50 = [y['experiment_safe_r50'] for y in res_all]
                cnt = [y['cnt'] for y in res_all]

                for y in res_all:
                    parsims = y['parsims']

                    input_agentex = lambda run_ii, prepth: f"\\input{{{prepth}{self.root_dir.parent.name}_{run_ii}/visualisation/{int(parsims[run_ii]):02d}_input.tex}}"  # _forest

                    tex_line_overview += tex_tabuline([f"{int(y['parsim_sum'])}",
                                                       f"{y['regress_sum']:0.3f}",
                                                       f"{y['experiment']:0.0f}",
                                                       tex_stacklist([f'{int(x)}' for x in y['parsims']]),
                                                       tex_stacklist([input_agentex(x, '') for x in [0, 1, 2]])])

                    tex_line_input += tex_tabuline([f"{int(y['parsim_sum'])}",
                                                    f"{y['regress_sum']:0.3f}",
                                                    f"{y['experiment']:0.0f}",
                                                    tex_stacklist([f'{int(x)}' for x in y['parsims']]),
                                                    tex_stacklist([input_agentex(x, f'../benchmarks/{self.root_dir.parent.parent.name}/{self.root_dir.parent.name}/') for x in [0, 1, 2]])])

                    tex_line_input += tex_tabuline([f"{int(y['parsim_sum'])}",
                                                    f"{y['regress_sum']:0.3f}",
                                                    f"{y['experiment']:0.0f}",
                                                    tex_stacklist([f'{int(x)}' for x in y['parsims']]),
                                                    tex_stacklist([input_agentex(x, f'../benchmarks/{self.root_dir.parent.parent.name}/{self.root_dir.parent.name}/') for x in [0, 1, 2]])])

                combined_overview = "\\begin{tabular}{llllllllll}\n\\hline \n" \
                                    f"{tex_tabuline(['dist', 'error', 'reward', 'dist', 'Agent code'])} \\hline\n" \
                                    f"{tex_line_overview}" \
                                    f"\\hline\n\\end{{tabular}}\n\n"

                combined_input = "\\begin{longtable}[c]{>{\\centering}p{10mm}>{\\centering}p{10mm}>{\\centering}p{12mm}>{\\centering}p{12mm}>{\\centering}p{90mm}} \\hline\n" \
                                 f"{tex_tabuline(['dist', 'error', 'reward', 'parsimony', 'expressions'])}" \
                                 f"{tex_line_input}" \
                                 "\\hline\n\\end{longtable}\n"

                combined_fulltrees = "\\begin{longtable}[c]{>{\\centering}p{10mm}>{\\centering}p{10mm}>{\\centering}p{12mm}>{\\centering}p{12mm}>{\\centering}p{90mm}} \\hline\n" \
                                     f"{tex_tabuline(['dist', 'error', 'reward', 'parsimony', 'expressions'])}" \
                                     f"{tex_line_input}" \
                                     "\\hline\n\\end{longtable}\n"

                combined_overview = latex_treeviz_full_document(combined_overview)
                file_dump(self.root_dir.parent / 'combined_overview.tex', combined_overview)
                file_dump(self.root_dir.parent / 'combined_input.tex', combined_input)

                with plt.rc_context(rc=pyplot_rc_tex):
                    fig, ax = plt.subplots()
                    ax.set(xlabel='Pareto complexity sum', ylabel='reward [x1000]', ylim=funny_limits)
                    ax.plot(xx, y_all, label='average', marker='.', color='r')
                    ax.plot(xx, y_safe, marker='None', color='r', linestyle='dotted')  # , label='low risk'
                    ax.plot(xx, y_all_r50, label='randomized start', marker='.', color='b')
                    ax.plot(xx, y_safe_r50, marker='None', color='b', linestyle='dotted')  # , label='low risk (randomized)'
                    ax.legend(loc='lower right')
                    plt.yticks(IB_YICKS[0], IB_YICKS[1])

                    # drawing the regression error, but plots seem to be too overloaded
                    # axx = ax.twinx()
                    # axx.plot(xx, cnt, color='tab:gray', label='regression error', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
                    # axx.tick_params(axis='y', labelcolor='tab:gray')
                    # # axx.plot(xx, y['regression_sum'])
                    #
                    # ax2 = ax.twinx()
                    # ax2.plot(xx, cnt, color='tab:gray', label='combos', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
                    # # ax2.set(ylabel='possible combinations', color='tab:gray')
                    # ax2.tick_params(axis='y', labelcolor='tab:gray')
                    # ax2.legend(loc='lower left')

                    fig.savefig(self.root_dir.parent / f'regression_all.pdf')
                    plt.close('all')

        else:
            raise Exception(f'This should actually never happen right now. name: {self.conf.name}')

        return

    def gp_load_oparray(self, path_operators=None):
        """
        Offers the possibility for the user to load a .yaml-file of operators for the gp-process.
        operator_pool is used otherwise.
        The Operator must match its version in the 'op'-Array (alternatively search for op_what)
        The second value in the tuple denotes the probability of choosing an operator

        @:param sfeh_no_crazyops: sfeh's workaround for being too lazy to load a file with the actual operators
        """

        try:
            operator_pool = yaml_load(Path(path_operators))
        except:
            self.printpl('i', 'Opt-in not specified: Operators-file does not exist.\n'
                              'Creating one with a default list of mathematical operator_pool.')
            operator_pool = None
        choose_operators = ChooseOperators(operator_pool=operator_pool)

        yaml_dump(self.root_dir / 'backup/operators_used.yaml', operator_pool, default_flow_style=True)  # delete this??

        """
        Load all operator_pool ready-to-use from a file
        """

        return choose_operators

    def activate_distributions(self, path_distrib=None):
        """
        Optional custom distributions specified by the user.
        """
        path_distrib = path_distrib or self.root_dir / self.paths.use_distributions_file
        choose_distributions = ChooseConstants(path_distrib, data_train=self.data_train, n_samples=100)
        return choose_distributions

    def plot_agent_histogram(self, parsim, tree, path_hist):
        """
        Make histograms for all pareto-efficient candidates
        sfeh: based on training data- maybe use test data...

        useful code?
        # histogram_data = np.digitize(histogram_data, bins)  # digitize: the bin index the entry belongs to
        # histogram_data = np.concatenate((histogram_data.reshape(-1, 1), pairwise_fitness.reshape(-1, 1)), axis=1)
        # histogram_data = np.multiply.reduce(histogram_data, axis=1)
        # hist, bins = np.histogram(histogram_data, bins=bins, weights=pairwise_fitness)
        """

        action_bins = self.kernel.histogram_bins(self.env_vars.eval_action.minmax)
        expr_sym = tree.get_expr_sym()
        used_observations = tree.get_observation_list()
        pairwise_diff = self.kernel.eval_tf(expr_sym, used_observations)['pairwise_diff']

        with plt.rc_context(rc=pyplot_rc_tex):
            fig, ax = plt.subplots()
            ax.hist(pairwise_diff, bins=action_bins, histtype="stepfilled", facecolor="none", edgecolor='k')
            ax.set(ylim=(0, len(self.data_train)), ylabel='frequency', xlabel='deviation')
            histpath = path_hist / f'acthist_{parsim}.pdf'
            fig.savefig(histpath)
            plt.close('all')

        return histpath

    def file_pareto_latex(self, parsim, tree):
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

        tree.set_fix_nodes(self.origin)
        tree = tree.get_oldtree()

        pl_forest = lambda x: f'\\plforest{{{x}}}\n'

        forest_tree_full = None  # todo pl_forest(latex_brackettree(tree))
        forest_tree_tight = None  # todo pl_forest(latex_brackettree_tight(latex_tree_semitight(tree)))
        # sfeh workaround delete this
        tex_expr_raw = f'${tree.export_visualization_latex()}$'  # sfeh dollars
        tex_expr_forest = pl_forest(f'[{tex_expr_raw}]')

        path_subfolder_tex = path_make_dir(self.root_dir / 'visualisation')  # sfeh running this in every tree seems unneccesary

        """
        The following lines delete this
        """
        # sfeh
        file_dump(path_subfolder_tex / f'full_{parsim:02d}.tex', forest_tree_full, verbose='ff', print_type=self.print_type)
        file_dump(path_subfolder_tex / f'full_{parsim:02d}_tight.tex', forest_tree_tight, verbose='ff', print_type=self.print_type)
        file_dump(path_subfolder_tex / f'{parsim:02d}_input.tex', tex_expr_raw, print_type=self.print_type)
        file_dump(path_subfolder_tex / f'{parsim:02d}_input_forest.tex', tex_expr_forest, verbose='ff', print_type=self.print_type)
        file_dump(path_subfolder_tex / f'{parsim:02d}_doc.tex', latex_treeviz_full_document(forest_tree_full), verbose='ff', print_type=self.print_type)  # delete this

        return forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest

    # def file_pareto_pycode(self):
    #     """
    #     this auto-generation of real (executable) python files
    #     is strongly customized for my experiments with Mountaincar and industrial benchmark
    #
    #     very useful: textwrap.indent
    #     example: complete_function = textwrap.indent(f"def decide(self, input):\n"
    #                                         f"{function_body}\n", '    ')  # aka tab (\t)
    #     """
    #     pass
    #     # py_return = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')
    #
    #     # complete_function = f"    def decide(self, input):\n" \
    #     #     f"        cartPos, cartVel = input\n" \
    #     #     f"        action = {{}}\n" \
    #     #     f"        return {py_return}\n"
    #     #
    #     # all_agents = []
    #     # all_agent_names = []
    #     # all_more_info = []
    #     #
    #     # for (parsim, fitness, tree) in self.pareto:
    #     #     agent_name = f'{self.conf.name}_{parsim:.0f}'
    #     #
    #     #     agent_as_python = tree.get_pycode()
    #     #     all_agents.append(f"class {agent_name}:\n{complete_function.format(agent_as_python)}")
    #     #     all_agent_names.append(agent_name)
    #     #     all_more_info.append(f"('{agent_name}', {agent_name}(), {parsim}, {fitness})")
    #     #
    #     # all_agents = '\n\n'.join(all_agents)
    #     # agent_tuples = ', '.join([f"('{x}', {x}())" for x in all_agent_names])
    #     # all_more_info = ', '.join(all_more_info)
    #     #
    #     # """
    #     # dude sfeh
    #     # sfeh for fully executable code
    #     # bad code
    #     # """
    #     # if 'MTC75' in self.conf.name:
    #     #     sarsa_agent = 75
    #     # elif 'MTC200' in self.conf.name:
    #     #     sarsa_agent = 200
    #     # else:
    #     #     raise
    #
    #     # pyc_complete = f"import math; import numpy as np\n" \
    #     #     "import sys\n" \
    #     #     "from pathlib import Path\n" \
    #     #     "sys.path.append(str(Path(Path.cwd() / '../../../'))\n" \
    #     #     "from benchmarks.MC.agents.quick_eval import *\n" \
    #     #     "from pathlib import Path\n" \
    #     #     "folder = Path.cwd() / 'custom_files'\n" \
    #     #     "from benchmarks.MC.agents.mtc_agent_sarsa import * \n" \
    #     #     f"with Path.open(Path(Path.cwd() / '../../../benchmarks/MC/agents/sarsa_agent_{sarsa_agent}.p'), 'rb') as file:\n" \
    #     #     "\tsarsa_agent = pickle.load(file)\n" \
    #     #     "\n" \
    #     #     f"{all_agents}\n" \
    #     #     f"all_agents_more = [{all_more_info}]\n" \
    #     #     f"agent_tuples = [{agent_tuples}]\n" \
    #     #     f"\n\n" \
    #     #     "if __name__ == '__main__':\n" \
    #     #     "\tprint('executing!')\n" \
    #     #     "\teval_agent_list(agent_tuples, folder=folder, goal_agent=sarsa_agent)\n"
    #
    #     # pth = path_make_dir(self.root_dir / self.paths.folder_pycode / f"agents.py")
    #     # with Path.open(pth, 'w') as file:
    #     #     file.write(pyc_complete)
    #     #     self.printpl('ff', f'Pycode: {pth.as_posix()}')
    #
    #     return

    def file_pareto_listcode(self):
        """
        save python code for Industrial Benchmark runs
        delete sometime
        """

        # pycode_agent = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')

        pygents_list = []

        for (parsim, fitness, tree) in self.pareto:
            # agent_name = f'{self.conf.name}_{parsim:.0f}'
            agent_name = f'{self.conf.name}_{self.env_vars.eval_action.name}_{parsim:.0f}'
            agent_as_python = tree.get_pycode()
            pygents_list.append([parsim, float(fitness), agent_name, agent_as_python])

        yaml_dump(self.root_dir / 'pycode_list.yaml', pygents_list, print_type=self.print_type)
        path = path_make_dir(self.root_dir / 'pycode_list.yaml')
        with Path.open(path, 'w') as file:
            _ = yaml.dump(pygents_list, file)  # , default_flow_style=False, sort_keys=False)
            printez('ff', f'IB pycode-list: {path.as_posix()}', print_type=self.print_type)  # sfeh always the same print structure... just pass the path?

        return

    def treelut_tree_add(self, tree: Node):
        """
        update selected values in self.tree_meta
        LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        """
        # todo
        meta = {'fitness_train': tree.meta.fitness_train,
                'parsimony': tree.meta.parsimony,
                'expr_raw': tree.meta.expr_raw,
                'expr_sym': tree.meta.expr_sym}

        tree_ident = hash(tree)  # attention: hashes change between python runs. do not save anything on their hash values <.<

        self.tree_lut[tree_ident] = meta
        return

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

    def pareto_append(self, tree_entry, msg=None):
        """
        Appending a candidate to the paretofront.
        - append the entry to the paretofront
        - reset gens_since_last_pareto
        - try to add the tree in its sympified version
        """
        if msg:
            self.printpl('a', f"New entry found! ({msg}): {BColors.RESET}{tree_entry[0]}, {tree_entry[1]}:{BColors.RESET} {tree_entry[2].meta.expr_raw}")
        self.pareto.append(tree_entry)
        self.gens_since_last_pareto = 0

        self.pareto = [x for x in self.pareto[:] if x[0] < tree_entry[0] or x[1] < tree_entry[1] or (x[0] == tree_entry[0] and x[1] == tree_entry[1])]
        self.pareto_sort()  # as far as I can tell, not really necessary without using iter()

        tree = tree_entry[2]
        tree_sym = copy.deepcopy(tree)
        try:
            self.printpl('aaa', 'Trying to simplify for pareto entry.')  # simplify the tree and save in pareto once again
            tree_sym.evolve_reduce(obs_infos=self.env_vars.obs_infos, completely=True)
            parsimony = tree_sym.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
            if parsimony < tree.meta.parsimony:
                self.printpl('aa', 'Successfully reduced pareto tree!')
                sym_fitness = self.tree_eval_fitness_offline_train(tree_sym)  # sfeh actually not required, delete this
                tree_sym.meta.fitness_train = sym_fitness
                tree_sym.meta.parsimony = parsimony
                self.update_pareto_with_tree(tree_sym)
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
        pop_parsimonies_dict = dict.fromkeys(sorted(set([tree.meta.parsimony for tree in self.population])))
        for ct in self.population:
            p = ct.meta.parsimony
            try:
                best_yet = pop_parsimonies_dict[p]
                if ct.meta.fitness_train < best_yet.meta.fitness_train:  # sfeh asd kernel required for fitness comparison
                    pop_parsimonies_dict[p] = ct
            except:
                pop_parsimonies_dict[p] = ct

        for p, tree in pop_parsimonies_dict.items():
            self.update_pareto_with_tree(tree)

        self.pareto_sort()  # sfeh check if required
        return

    def update_pareto_with_tree(self, tree: FinalizedNode):
        """
        inserts a tree into the pareto front
        """
        parsimony = tree.meta.parsimony
        fitness_train = tree.meta.fitness_train
        tree_entry = [parsimony, fitness_train, tree]

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

    def pop_append(self, tree: Node):
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
            tree.check_all()
        except Exception as ex:
            print_warning('w', f'tree failed the quick check. last-mod: {tree.meta.last_evolution}. Reason:\n{ex}', print_type=self.print_type)

        # tree = self.tree_finish_nodes(tree, last_evolution=last_evolution)

        tree_ident = hash(tree)
        if tree_ident in self.tree_lut:
            tree_meta = self.tree_lut[tree_ident]
            parsimony = tree_meta['parsimony']
            fitness_train = round(tree_meta['fitness_train'], self.conf.float_decimals)  # sfeh just for now
        else:
            parsimony = tree.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
            if parsimony > self.conf.parsimony_max:
                print_warning('wwww', f'Parsimony too high, last evolution: {tree.meta.last_evolution}', print_type=self.print_type)  # sfeh care about wwww. should not
                return
            try:
                fitness_train = self.tree_eval_fitness_offline_train(tree)
            except Exception as evalex:
                print_warning('wwww', f'Exception while evaluating: {evalex}, tree: {tree}.', print_type=self.print_type)
                return

        expr_raw = tree.get_expr()
        expr_sym = expr_sympify(expr_raw)
        try:
            tree.set_fix_nodes(self.origin)
        except Exception as ex:
            print(f'Nope, failed tree finish: {ex}\n{tree}')

        tree.meta.fitness_train = fitness_train
        tree.meta.parsimony = parsimony
        tree.meta.expr_raw = expr_raw
        tree.meta.expr_sym = expr_sym
        # tree.set_meta(fitness_train, parsimony, last_evolution, expr_raw, expr_sym)

        self.treelut_tree_add(tree)
        self.population.append(tree)
        return

    def pop_selection_tournament(self, tourn_size):
        """
        config-selection. takes a number of trees (we use 3) and returns the best one (winner)
        """
        tournament_list = [random.choice(self.pop_base) for _ in range(tourn_size)]
        tourn_winner = self.kernel.best_fitness_function(tournament_list, key=lambda tree: tree.meta.fitness_train)

        return copy.deepcopy(tourn_winner)

    def load_origin_tree(self, path_origin_tree):
        """
        The origin tree (which was already loaded) gets activated for its use in the GP-process
        """
        # tree_expr_txt_path = root_dir / 'run_files/tree_expr.txt'
        # tree_numpy_csv_path = root_dir / 'run_files/tree_numpy.csv'
        # tree_labels_csv_path = self.root_dir / 'run_files/tree_labels.csv'

        """
        from the labellist-csv, loading the label list
        """

        with Path.open(path_origin_tree, newline='') as file:
            file.read()
        try:
            origin_tree = tree_from_labellist(label_list, modify_list=modify_list)
            expr_sym = origin_tree.get_expr_sym()
        except Exception as sympex:
            raise Exception(f'Loaded origin_tree already failed because of: {sympex}')

        # sfeh, this does not work
        # if not tree_check_is_sympified(tree):
        #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
        #                          ''.format(expr_raw, expr_sym))

        used_observations = origin_tree.get_observation_list()
        tf_origin_results = self.kernel.eval_tf(expr_sym, used_observations)
        fitness_train = round(tf_origin_results['mean_error'], self.conf.float_decimals)  # fitness currently IS the mean error
        if self.kernel.exploration_risk:
            self.kernel.origin_results = tf_origin_results['results_kernel']  # after getting the origin-results, these informations can be updated

        origin_tree.meta.fitness_train = fitness_train
        origin_tree.meta.parsimony = 0

        self.pareto.append([0, fitness_train, origin_tree])  # the origin tree is the only candidate for now -> it is in the pareto front
        self.print_g('gg', f'Loading origin tree, regr. error {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return origin_tree  # self.origin_tree = copy.deepcopy(origin_tree)

    def tree_eval_fitness_offline_train(self, tree: Node):
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
            expr_raw = tree.get_expr()
            expr_sym = expr_sympify(expr_raw)
        except Exception as evalex:
            raise Exception(f'eval:{evalex}')

        used_observations = tree.get_observation_list()
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

        return fitness_train

    def tree_eval_fitness_online(self, tree: FinalizedNode, episodes=1, seed=0, env_parameters=None):
        """
        an online evaluation of a tree
        todo
        """
        pass

        return

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
                    print_e(f'damn setting ylim not working sfeh :s {ex}')
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
            path = self.root_dir / f'monitoring.png'  # -{self.conf.name}
            fig.savefig(path)
            self.printpl('f', f"monitoring: {path.as_posix()}")
            plt.close('all')

    def plot_paretofront(self):
        """
        Write pyplot with pareto candidates
        """
        tuples = [[parsim, fitness] for (parsim, fitness, tree) in self.pareto]
        xx, yy = np.array(tuples).T

        if len(xx) == 0:
            print_e(f'Plotting empty array is not possible! Data={xx, yy}')
            return

        run_name = self.conf.name
        run_name = str(run_name).replace('_', '-')  # sfeh asd workaround for latex version

        with plt.rc_context(rc=pyplot_rc_tex):
            fig, ax = plt.subplots()
            right = max(max(xx), self.conf.parsimony_max) * 1.05  # sfeh check this out 1.05  # if set_right:

            # beyond_lines:  # adding a point to the edges to imply that there are no more values (pareto-plot)
            xx = np.concatenate([[xx[0]], xx, [right + 1]])
            yy = np.concatenate([[max(yy) + 1], yy, [yy[-1]]])

            ax.step(xx, yy, linestyle='dashed', marker='.', label=f'{run_name}', where='post')
            ax.set(xlabel='complexity', ylabel='regression error',
                   xlim=(0, right),
                   ylim=(0, (max(yy) - min(min(yy), 0)) * 1.05))

            try:
                fig.savefig(self.root_dir / f'paretofront.pdf')
                self.printpl('f', f"paretofront (pdf): {self.root_dir / f'paretofront.pdf'}")
            except PermissionError as perm_error:
                print_e(f'Could not save plot: {perm_error}')  # sfeh for everything?

        return

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

                plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
                path = self.root_dir / f'monitoring_evolutions.pdf'
                fig.savefig(path)
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
        gen_id = self.gen_id
        popul = self.population

        if len(popul) == 0:
            raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

        pop_fitness = [tree.meta.fitness_train for tree in popul]

        # tmp_evol_performance = dict.fromkeys(self.monitor_evol.keys(), pd.DataFrame(columns=['fitness', 'parsimony', 'lentree']))
        # for tree in popul:
        #     last_evol = tree.meta.last_evolution
        #     if last_evol in self.evolve_tags:
        #         row = {'fitness': tree.meta.fitness_train,
        #                'parsimony': tree.meta.parsimony,
        #                'lentree': len(tree)}
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

        pop_parsim = [tree.meta.parsimony for tree in popul]
        pop_treelen = [len(tree) for tree in popul]

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
        save_gen = int(self.conf.period.get('gen_save', 1))
        plot_gen = int(self.conf.period.get('gen_plots', 1))
        if self.gen_id >= plot_gen and self.gen_id % plot_gen == 0:
            self.file_analysis_plots()

        if self.gen_id >= save_gen and self.gen_id % save_gen == 0 or self.gen_id == 10:  # sfeh extra save at 10 for early feedback while testing
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
            time_now = time.strftime("%d.%m %H:%M", time.localtime())
            print(f'[{time_now}] {text}')
        return

# def activate_dataset(path_data, action_name):
#     """
#     loading the data which the GP will be working on.
#     The .csv-file is prepared (loading correct data-type, splitting data, ...)
#     and saved as pickle-file for reloading runs.
#     This is especially important, as the split in training and test-data must be the same.
#
#     separate loading the prepared data into the main class.
#     Why like this? I needed to find a bug in the data_from_csv file and
#     did not want to start the whole stuff everytime
#
#         # self.data_train_panda, self.data_control_panda
#     """
#
#     if path_data.suffix == '.p':
#         data_prepared = pickle_load(path_data)
#     elif path_data.suffix == '.csv':
#         data_prepared = data_from_csv(path_data, action_name=action_name)
#     else:
#         raise FileNotFoundError(f'No data provided? File must be a pickle (.p) or csv (.csv) file. Loaded file: {path_data}')
#
#     env_vars, data_train, data_control = data_prepared  # data_control is data_test
#
#     return env_vars, data_train, data_control


def data_from_csv(df, action):
    """
    todo
    Loads .csv data files.
    - Reading the .csv-file (with pandas)
    - renaming column headers
    - saving header info for later use

    Information that we need to extract for each column:
    - choose_xtype choosing a random observation for leaf nodes
    - filtering observation index
        is there an index (past values, e.g. velocity_0, velocity_1) -> performing filter-evolve on the variables index (velocity_2 -> velocity_3)
    - evalaction data_train, data_test, is the action for the regression? -> more than one action might be required (IB has three action dimensions)
        - action min max -> for kernel regression bounded. occuring min and max values might not be the theoretical min/max values


    deprecated:

    """

    """
    1. split col name
    - check whether its an observation
    --> check whether there are indizes
    - check whether its an action
    --> check unique valuea, min, max
    - check if it should be ignored (deprecated action, irrelevant column)
    2. 
    """
    # todo remove dat shit

    return