"""
'f2f', 'b2b', f = float, b = bool. 'float to bool'

Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""
import matplotlib.pyplot as plt
import time
from plagih.modules.file_interaction import *
import subprocess
import json

try:
    import tikzplotlib
except Exception as ex:
    print_e('Need to install tikzplotlib? matplotlib2tikz is outdated. Exception:\n{}'.format(ex))
### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


class ExplainableGP(object):
    """

    """

    def __init__(self, root_dir, config=None):

        self.name = root_dir.name
        print('\n\tInitializing Plagih. Name: {}{}{}. Located in: \n\t{}\n'.format(BColors.CYAN, self.name, BColors.RESET, root_dir))
        self.time_start = time.perf_counter()
        self.restart_vers = 'v0.8'

        self.root_dir = root_dir
        print(self.root_dir)

        self.config = {
            # 'root_dir': root_dir,  # TODO
            'mode': 'run',  # ['run', 'analyze']
            'description': 'No description set',
            'choose': {
                'choose rules': ['first', 'choose', 'random'],
                'operators': ['random', 'from distribution'],
                'distributions_file': ['gauss', 'random', 'data samples'],
                'default': 'random'
            },
            # (!) Relevant for result
            'pop_max': 1000,  # Maximum amount of trees in a population. Only used evolve rates, condition is never tested.
            'parsimony_max': 80,  # right value is the maximum parsimony. left value not used, but was meant to set parsimony for the first generations. [3 to 2^(bas +1) - 1]
            'kernel_name': 'regression discrete',  # [regression, regression bounded, classification, match]
            'complexity_measure': 'tree edit distance',

            # rather irrelevant
            'parsimony_tmp': 15,
            'precision': 3,  # rounding the fitness
            'float_accuracy': 200,
            'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
            'print_type': 'gggwwsivoaa',  # To show absolutely all: wggggsiiiivvvtopppttt
            'overwrite periodic gp_files': True,  # If True, the file gets overwritten. If False, in every generation a new file is created.
            'force_new_run': False,  # especially for testing. Instead of deleting the old folder each time, you can set this to False to init a new run again #
            'delete_old_file': False,  # sfeh, delete old gp_files. be very careful
            'monitor': {'gen_fitness_average': 'y',
                        'sympify_errors': 'y',
                        'population_tmp_done-size': 'y',
                        'fitness_variance': 'n'},
            'period': {'time_monitor': None,  # in sec
                       'time_save': None,  # in sec
                       'gen_monitor': 1,  # in gen counts
                       'gen_save': 1},  # in gen counts

            # GP-evolve specific parameters
            'evolve_rates': {'repro one': 0.05,
                             'repro pareto': 0.01,
                             'repro reduced one': 0.03,
                             'filter floats': 0.05,
                             'point mutate function': 0.1,
                             'branch mutate insert': 0.10,
                             'crossover branches': 0.36,
                             'random from origin_tree': 0.15,
                             'random from scratch': 0.15,
                             },
            'evolve': {'r': {'evolve_fun': 'reproduce lucky', 'rate': 0.05, 'params': ()},  # tbd sfeh tbd
                       'rP': {'evolve_fun': 'reproduce pareto', 'rate': 0.01, 'params': ()},
                       'rS': {'evolve_fun': 'reproduce sympify', 'rate': 0.03, 'params': ()},
                       'p': {'evolve_fun': 'parameters filter floats', 'rate': 0.05, 'params': ('gaussian')},  # sfeh bool aswell
                       'mN': {'evolve_fun': 'mutate node', 'rate': 0.1, 'params': ()},  # sfeh mutate node or mutate function?
                       'mB': {'evolve_fun': 'mutate branch', 'rate': 0.1, 'params': ()},
                       'c': {'evolve_fun': 'crossover branche', 'rate': 0.36, 'params': ()},
                       'nO': {'evolve_fun': 'new origin-based', 'rate': 0.15, 'params': ()},
                       'n': {'evolve_fun': 'new random', 'rate': 0.15, 'params': ()}
                       },
            'crossover_type_safety_mode': 'replace_same_types',
            'gen_num_max_parsimony': 50,  # Increase tmp_parsim to this generation
            'tree_growth': 'node-based',  # node-based, depth-based
            'tree_depth_base': 7,  # [3..10]
            'tree_depth_max': 15,  # maximum Tree depth for entire run
            'tree_depth_min': 5,
            'tree from scratch min_nodes': 8,
            'random from scratch max nodes': 50,
            'tree branch base nodes': 20,
            'tourn_size': 3,  # [7 per 100] number of trees selected for tournament

            # When to stop the run
            'time_max': None,  # int(60 * 60 * 12),  # 60 = 1 min
            'gen_max': 800,  # Maximum amount of generations

            'env': {
                'name': None,
                'observations': {

                }
            }
        }

        self.config.update(config)

        # init values with dummies (just to have all self values here for overview)
        self.tree_meta = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw', 'gen+nr'}
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

        # some config_dict values have to be used quite often...
        self.kernel = FitnessKernel(self.config['kernel_name'])
        self.print_type = self.config['print_type']
        self.precision = self.config['precision']  # the number of floating points for the round function
        self.parsimony_tmp = self.config['parsimony_tmp']
        self.parsimony_max = self.config['parsimony_max']
        self.monitor_dict = self.config['monitor']
        self.evolve_rates = self.config['evolve_rates']
        self.tourn_size = self.config['tourn_size']

        # special variables
        self.tf_device = "/gpu:0"  # Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device_log = False  # TF device usage logging (for debugging)

        self.node_choose_dict = {
            '2f': {
                0: {'observ': [],
                    'distribution': []},
                1: {'f2f': [],
                    'b2f': []},
                2: {'f2f': [],
                    'b2f': []},
                3: {'b2f2f': []}},
            '2b': {

                0: {'observ': [],
                    'distribution': []},
                1: {'f2b': [],
                    'b2b': []},
                2: {'f2b': [],
                    'b2b': []},
                3: {None: []}}}

        # some useful stuff
        self.monitoring_dict = {'population_tmp_done-size': {},
                                'fitness_average': {},
                                'fitness_variance': {},
                                'best_candidate': {},
                                'total_found_trees': {},
                                'complexity_average': {},
                                'complexity_variance': {},
                                'tmp_pop_fitness_distribution': {}}
        self.pop_analysis_dict = {}  # similar to monitoring_dict

        # self.file_directories_create(self.config['root_dir'])  # deprecated, folder MUST already exist
        self.print_g('ggg', 'Init. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        return

    def try_load_backup(self):
        """
        If a backup-file is found
        """
        path_backup = self.root_dir / file_backup_pickle
        if Path.is_file(path_backup):
            self.print_g('g', 'Backup-file was found. Loading data...')
            try:
                self.run_backup_load(path_backup)
                return True
            except Exception as ex:
                print_warning('w', 'Even though a backup exists for this run, it could not be loaded because of: {}.'.format(ex))
                raise
        else:
            return False

    def run_backup_load(self, path_backup):
        """

        """

        with Path.open(path_backup, 'rb') as file:
            run_data = pickle.load(file)

        self.restart_count, self.gen_id, self.parsimony_best_meta, self.pareto, self.population_base, self.monitoring_dict = run_data

        self.restart_count += 1
        printez('g', 'Loading Generation: {}'.format(self.gen_id), self.print_type)

        return

    def run_backup_load_old(self, path_backup):
        """

        """

        with Path.open(path_backup, 'rb') as file:
            run_data = pickle.load(file)

        self.restart_count = run_data['self.restart_count']  # counting how often this was restarted
        self.gen_id = run_data['self.gen_id']
        self.parsimony_best_meta = run_data['self.parsimony_best_meta']
        self.pareto = run_data['self.pareto']
        self.population_base = run_data['self.population_base']
        self.monitoring_dict = run_data['self.monitoring_dict']

        # check if entries exist
        for key, value in self.monitoring_dict:
            self.monitoring_dict[key] = value
        # if self.monitoring_dict.get('complexity_variance') is None:
        #     self.monitoring_dict['complexity_variance'] = {}
        # if self.monitoring_dict.get('fitness_variance') is None:
        #     self.monitoring_dict['fitness_variance'] = {}

        # updating the population header line (removing it)
        if isinstance(self.population_base[0], str):
            self.population_base.pop(0)

        # force fix of all trees if they are incorrect in last versions
        self.version_pop_fix_trees(self.population_base)

        # update monitoring dict name (genepool_size -> population_tmp_done-size)
        try:
            self.monitoring_dict['population_tmp_done-size'] = run_data['self.monitoring_dict']['genepool_size']
            print_warning('w', 'Delete this. Restarting from an old run, where gene_pool existed.')
        except:
            pass

        # update pareto entries (pareto now contains meta data)
        first_pareto = next(iter(self.pareto.items()))
        if isinstance(first_pareto[1], float):
            self.pareto = {}
            self.pareto_update()

        self.restart_count += 1
        printez('g', 'Loading Generation: {}'.format(self.gen_id), self.print_type)

        return

    def run_backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """
        path_backup = self.root_dir / file_backup_pickle

        run_backup_data = self.restart_count, self.gen_id, self.parsimony_best_meta, self.pareto, self.population_base, self.monitoring_dict

        with Path.open(path_backup, 'wb') as file:
            pickle.dump(run_backup_data, file)
        self.printpl('iii', 'Saved run in pickle file.')
        return

    def plagih_gp_run(self):
        """
        regular plagih run
        """

        if self.origin_exists():  # Just handeling a special case here
            if not tree_node_is_modifiable(self.origin_tree, root_id):  # Modify-nodes is not "activated"
                self.config['evolve_rates']['random from origin_tree'] += float(self.config['evolve_rates']['random from scratch'])
                self.config['evolve_rates']['random from scratch'] = 0
                print_warning('ww', 'Generating new trees \'randomly\' is not possible when the origin tree has fix nodes!'
                                    '\nFixed by adding the evolve_rate to \'random from origin_tree\'.')
        else:

            if self.config['evolve_rates']['random from origin_tree'] > 0:
                self.config['evolve_rates']['random from scratch'] += float(self.config['evolve_rates']['random from origin_tree'])
                self.config['evolve_rates']['random from origin_tree'] = 0
                print_warning('ww', 'Generating mew trees \'random from fix_tree\' is not possible without a fix_tree!'
                                    '\nFixed by adding the evolve_rate to \'random\'.')

            if self.config['complexity_measure'] in ['tree edit distance']:  # sfeh get all origin-based distances
                raise Exception('Can not use relative distance without providing a reference/origin tree!')

        self.write_config_yaml()  # sfeh or json?

        if self.gen_id == 0:
            self.gen_create_first()

        self.gen_create_loop()
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
        filename = file_make_dir(self.root_dir / info_config_yaml)
        yaml_dump(filename, self.config)

        return

    def write_config_json(self):
        """
        write the parameters to a .csv file which can also be loaded
        """

        path_config = make_dir(self.root_dir / run_files)
        filename = self.root_dir / file_config_json

        with Path.open(filename, 'w') as file:
            json.dump(self.config, file, indent=4)

        return

    def gen_create_first(self):
        """
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """
        self.print_g('gg', 'Preparing to create first Generation. Gen {}.'.format(self.gen_id))
        self.gen_reset_parameters()

        rate_o = self.evolve_rates['random from origin_tree']
        rate_s = self.evolve_rates['random from scratch']
        rate_sum = sum([rate_o, rate_s])

        first_rate_o = int((self.config['pop_max'] * (rate_o / rate_sum))/2)
        first_rate_s = int((self.config['pop_max'] * (rate_s / rate_sum))/2)

        for ii in range(first_rate_o):
            call_params = {'fix_tree': self.origin_tree_get(),
                           'depth_tuple': (6, 1, 10, 2),
                           'build_method': 'full'}
            tree = self.pop_random_from_origin(call_params)
            self.pop_append(tree, last_evolution='f_o1')

        for ii in range(first_rate_o):
            call_params = {'fix_tree': self.origin_tree_get(),
                           'nodes_tuple': (20, 1, 40, 7),
                           'build_method': 'grow'}
            tree = self.pop_random_from_origin(call_params)
            self.pop_append(tree, last_evolution='f_o2')

        for ii in range(first_rate_s):
            call_params = {'depth_tuple': (6, 1, 10, 2),
                           'build_method': 'full'}
            tree = self.pop_random(call_params)
            self.pop_append(tree, last_evolution='f_r1')

        for ii in range(first_rate_s):
            call_params = {'depth_tuple': (6, 1, 10, 2),
                           'build_method': 'grow'}
            tree = self.pop_random(call_params)
            self.pop_append(tree, last_evolution='f_r2')

        self.gen_finalize()
        file_population_karoo(self.population_base, '1_first', self.root_dir, self.gen_id)  # first gen only

    def gen_create_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        # todo da lässt sich doch was machen...
        # All gp creators: name, function, num of trees from tournament selection

        if self.origin_exists():
            origin_tree = self.origin_tree
        else:
            origin_tree = None

        gp_dict_user = {
            'Repro': {'gp_func': 'reproduce', 'params': {}, 'repro_rate': 0.05},
            'Rsympy': {'gp_func': 'reproduce', 'tourn_size': None, 'repro_rate': 0.03},
            'Pareto': {'gp_func': 'revive pareto', 'tourn_size': None, 'repro_rate': 0.01},
            'Point': {'gp_func': 'mutate point', 'tourn_size': None, 'repro_rate': 0.05},
            'Branch1': {'gp_func': 'mutate branch', 'tourn_size': None, 'repro_rate': 0.05, 'custom_params': {'branch_nodes': 20, 'build_method': 'grow'}},
            'Branch2': {'gp_func': 'mutate branch', 'tourn_size': None, 'repro_rate': 0.05, 'custom_params': {'branch_depth': 6, 'build_method': 'full'}},
            'Branch3': {'gp_func': 'mutate branch', 'tourn_size': None, 'repro_rate': 0.05, 'custom_params': {'branch_depth': 7, 'build_method': 'grow'}},
            'Xover': {'gp_func': 'crossover branch', 'tourn_size': None, 'repro_rate': 0.36},
            'Filter1': {'gp_func': 'filter', 'tourn_size': None, 'repro_rate': 0.10},
            'Filter2': {'gp_func': 'filter', 'tourn_size': None, 'repro_rate': 0.10},
            'RandFix1': {'gp_func': 'random from fix_tree', 'tourn_size': 0, 'repro_rate': 0.15},
            'RandFix2': {'gp_func': 'random', 'tourn_size': 0, 'repro_rate': 0.15, 'custom_params': {'branch_depth': 7, 'build_method': 'grow'}}
        }

        evolve_dict = {
            'reproduce': (self.pop_reproduce, 1, None),
            'revive pareto': (self.pop_reproduce_olymp, 0, None),
            'mutate point': (self.pop_mutate_point, 1, None),
            'mutate branch': (self.pop_mutate_branch, 1, None),
            'crossover branch': (self.pop_crossover_branch, 2, None),
            'filter': (self.pop_mutate_filter, 1, None),
            'random from fix_tree': (self.pop_random_from_origin, 0, origin_tree),
            'random': (self.pop_random, 0, None)}

        # gp_list = [  # name of the function,    implementation in plagih,       number of tournament selections needed
        #     ('repro one', self.pop_reproduce, 1),
        #     ('repro pareto', self.pop_reproduce_olymp, 0),
        #     ('repro reduced one', self.pop_reproduce_reduce, 1),
        #     ('point mutate function', self.pop_mutate_point, 1),
        #     ('filter floats', self.pop_mutate_filter, 1),
        #     ('branch mutate insert', self.pop_mutate_branch, 1),
        #     ('crossover branch', self.pop_crossover_branch, 2),
        #     ('random from origin_tree', self.pop_random_from_origin, 0),
        #     ('random from scratch', self.pop_random, 0)]
        # for k, v in gp_dict_user:
        #
        #
        # # if gpfunc_call2[evolve_call][1]:
        # #     call_params['fix_tree'] = self.origin_tree_get()

        # sfehsfeh: pring every written file

        gp_dict2 = {
            # 'Repro': {'gp_func': 'reproduce', 'evolve_call': self.pop_reproduce, 'tourn_size': None,
            #           'custom_params': {}, 'evolve_num': 50},
            # 'Rsympy': {'gp_func': 'reproduce', 'evolve_call': self.pop_reproduce, 'tourn_size': None,
            #            'custom_params': {'sympify_tree': True},
            #            'evolve_num': 30},
            #
            # 'Pareto': {'gp_func': 'revive pareto', 'evolve_call': self.pop_reproduce_olymp, 'tourn_size': 0,
            #            'custom_params': {}, 'evolve_num': 10},

            # 'Point': {'gp_func': 'mutate point', 'evolve_call': self.pop_mutate_point, 'tourn_size': None,
            #           'custom_params': {}, 'evolve_num': 50},
            # 'Branch1': {'gp_func': 'mutate branch', 'evolve_call': self.pop_mutate_branch, 'tourn_size': None,
            #             'custom_params': {'nodes_tuple': (20, 1, 40, 5), 'build_method': 'grow'},
            #             'evolve_num': 50},
            # 'Branch2': {'gp_func': 'mutate branch', 'evolve_call': self.pop_mutate_branch, 'tourn_size': None,
            #             'custom_params': {'depth_tuple': (6, 1, 9, 1), 'build_method': 'full'},
            #             'evolve_num': 50},
            # 'Branch3': {'gp_func': 'mutate branch', 'evolve_call': self.pop_mutate_branch, 'tourn_size': None,
            #             'custom_params': {'depth_tuple': (7, 1, 9, 1), 'build_method': 'grow'},
            #             'evolve_num': 50},
            'Xover': {'gp_func': 'crossover branch', 'evolve_call': self.pop_crossover_branch, 'tourn_size': None,
                      'custom_params': {}, 'evolve_num': 50},

            'Filter1': {'gp_func': 'filter', 'evolve_call': self.pop_mutate_filter, 'tourn_size': None,
                        'custom_params': {'mode': 'branch'},
                        'evolve_num': 50},  # todo
            'Filter2': {'gp_func': 'filter', 'evolve_call': self.pop_mutate_filter, 'tourn_size': None,
                        'custom_params': {'mode': 'point'}, 'evolve_num': 50},
            #
            # 'Rand1': {'gp_func': 'random', 'evolve_call': self.pop_random, 'tourn_size': 0,
            #           'custom_params': {'depth_tuple': (6, 1, 9, 2), 'build_method': 'full'},  'evolve_num': 50},
            # 'Rand2': {'gp_func': 'random', 'evolve_call': self.pop_random, 'tourn_size': 0,
            #           'custom_params': {'depth_tuple': (7, 1, 9, 2), 'build_method': 'grow'}, 'evolve_num': 50},

            # 'RandFix1': {'gp_func': 'random from fix_tree', 'evolve_call': self.pop_random_from_origin, 'tourn_size': 0,
            #              'custom_params': {'depth_tuple': (6, 1, 9, 2), 'build_method': 'full'},
            #              'evolve_num': 50},
            # 'RandFix2': {'gp_func': 'random from fix_tree', 'evolve_call': self.pop_random_from_origin, 'tourn_size': 0,
            #              'custom_params': {'nodes_tuple': (20, 1, 30, 6), 'build_method': 'grow'},
            #              'evolve_num': 50}
            }

        for k in gp_dict2.keys():  # all selected gp mutations  # sfeh seems like dumb code
            if gp_dict2[k]['tourn_size'] is None:
                gp_dict2[k]['tourn_size'] = self.tourn_size

        while self.run_continues():  # max generation, max time, done...

            self.gen_reset_parameters()

            # ########
            # call_params{'old_tree', 'partner_tree', 'fix_tree', 'branch_nodes', 'branch_depth': 6, 'build_method': <'full', 'grow', 'half'>}
            for tag, fun_infos in gp_dict2.items():  # all selected gp mutations
                time_evolve = time.perf_counter()
                evolve_call = fun_infos['evolve_call']
                gp_func = fun_infos['gp_func']
                tourn_size = fun_infos['tourn_size']

                call_params = fun_infos['custom_params']

                if evolve_dict[gp_func][2] is not None:
                    call_params['fix_tree'] = self.origin_tree_get()

                evolve_num = gp_dict2[tag]['evolve_num']  # todo

                tourn_num = evolve_dict[gp_func][1]
                if tourn_num == 1:  # one tournament winner. mainly mutations
                    for ii in range(evolve_num):
                        call_params['old_tree'] = self.pop_selection_tournament(tourn_size)
                        new_tree = evolve_call(call_params)
                        self.pop_append(new_tree, last_evolution=tag)

                elif tourn_num == 0:  # no tournament winner. mainly new random ones
                    for ii in range(evolve_num):
                        new_tree = evolve_call(call_params)
                        self.pop_append(new_tree, last_evolution=tag)

                elif tourn_num == 2:  # only crossover
                    for ii in range(int(evolve_num/2)):
                        call_params['old_tree'] = self.pop_selection_tournament(tourn_size)
                        call_params['partner_tree'] = self.pop_selection_tournament(tourn_size)
                        new_tree, new_tree2 = list(evolve_call(call_params))
                        self.pop_append(new_tree, last_evolution=tag)
                        self.pop_append(new_tree2, last_evolution=tag)
                else:
                    raise Exception('Tournament selection is only meant to be performed [0, 1, 2] times. This was: {}'.format(tourn_num))

                self.print_g('ggg', '->Evolving \'{}\' {}x. Took: {:4.2f}s.'.format(tag, evolve_num, time.perf_counter() - time_evolve))

            # ########

            # # Creating a new population ############
            # for name, gp_function, tourn_rep in gp_list:
            #     time_evolve = time.perf_counter()
            #
            #     repro_rate = int(self.evolve_rates[name] * self.config['pop_max'])
            #     gp_function(repro_rate)
            #     self.print_g('ggg', '-->Evolve ({}) {}x. Took: {:4.2f}s.'.format(name, repro_rate, time.perf_counter() - time_evolve))
            # # ######################################

            self.gen_finalize()

            self.periodical_procedures()

            self.print_g('ggg', 'Generation took a total time of: {:4.2f}'.format(time.perf_counter() - self.time_genstart))
        else:
            printez('g', 'Done after Generation {}.'.format(self.gen_id), print_type=self.print_type)
            return

    def run_continues(self):
        """
        checks if the run can continue
        """
        cond_1 = self.gen_id >= self.config['gen_max']
        cond_2 = False if self.config['time_max'] is None else time.perf_counter() - self.time_start > self.config['time_max']
        cond_3 = self.custom_done
        if cond_1 or cond_2 or cond_3:
            return False
        else:
            return True

    def origin_exists(self):
        if self.origin_tree is not None:
            return True
        else:
            return False

    def periodical_procedures(self):
        """
        Every few generations, update the created gp_files
        - default is in every generation, but saving every n-th gen or after time passed is possible aswell
        """
        if self.config['overwrite periodic gp_files']:
            tmp_path = make_dir(self.root_dir)
        else:
            tmp_path = make_dir(self.root_dir / folder_steps / 'Gen-{}'.format(self.gen_id))

        time_now = time.perf_counter()
        plots_show, save_run = False, False

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
            self.file_all_plots(tmp_path)

        if save_run:
            self.run_backup_save()
            self.file_save_files(tmp_path)

        self.printpl('ii', 'Done with auto-procedures')

        return 0

    def file_save_files(self, root_path, pop_name=''):
        """
        writes all important gp_files

        """
        self.file_conclusion(root_path)
        self.file_pareto_text(self.pareto, root_path)
        self.file_pareto_latex(self.pareto, root_path)
        self.file_generate_pycode(self.pareto, root_path)

        file_population_karoo(self.population_base, pop_name, root_path, self.gen_id)  # save the final generation of Trees to disk

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Load and Archive Data                     +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def activate_dataset(self, data_prepared):
        """
        separate loading the prepared data into the main class.
        Why like this? I needed to find a bug in the data_from_csv file and
        did not want to start the whole stuff everytime
        """
        env_variables, data_train, data_control = data_prepared  # sfeh what data is used?
        self.env_variables = env_variables
        self.data_train, self.data_control = data_train, data_control  # what is that good for: self.data_train_rows, = data_train_rows,

        return

    def activate_operators(self, choose_oparray, choose_distributions):
        """
        operators were loaded already and need to be set in the gp run
        """
        self.choose_oparray = choose_oparray
        self.choose_distributions = choose_distributions  # sfeh samples from csv?

        if not self.config['force_new_run']:
            self.try_load_backup()
            # sfeh: delete old files?

        return

    def version_pop_fix_trees(self, population):
        """
        Some old runs are now inconsistent as the trees now hold some important information
        """

        # Fix the tree node's xtypes (old node's 'type', e.g. 'Term', ...) with 'f2f', ...
        cnt = 0
        for ii, tree in enumerate(population):
            if tree_node_get_xtype(tree, root_id) == '':
                cnt += 1
                population[ii] = tree_set_xtypes(tree, self.env_variables)
        if cnt > 0:
            print_warning('ww', 'Amount of trees with node_xtype inconsistency: {}.'.format(cnt))

        # Fix the trees missing parsimony
        cnt = 0
        for ii, tree in enumerate(population):
            if str(tree_get_parsimony(tree)) == '':
                cnt += 1
                parsimony = self.tree_eval_parsimony_easywrapper(tree)
                population[ii] = tree_set_parsimony(tree, parsimony)
        if cnt > 0:
            print_warning('ww', 'Amount of trees without parsimony: {}.'.format(cnt))

        return

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
        #     origin_fitness = eval_tf(self.origin_meta['expr_sym'], self.data_control, self.tf_parameters, get_pred_labels=True)['fitness']
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
        #     result = eval_tf(algo_sym, self.data_control, self.tf_parameters, get_pred_labels=True)
        #     fit_control = result['fitness']
        #
        #     if self.kernel.fitness_compare(fit_control, fitness_control_best, mode='better_or_equal'):  # find the Tree with a perfect match for all data_csv_path rows
        #         fitness_control_best = fit_control
        #         fittest_algo = algo_sym
        #         fittest_parsimony = parsimony
        #
        #     no_fault = True
        #     for enum, entry in enumerate(result['tf_result']):
        #         if not self.check_value_is_real(entry):
        #             no_fault = False
        #             # sfeh this is a bad workaround
        #             result['tf_result'][enum] = 1
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

    def file_pareto_text(self, pareto, root_path):
        """
        Save all the pareto efficient candidates to file
        """

        pth = file_make_dir(root_path / file_pareto)

        with Path.open(pth, 'w') as file:
            for parsim, meta in sorted(list(pareto.items())):
                fitness = meta['fitness_train']
                algo_sym = meta['expr_sym']  # save raw version, not the sympified one
                file.write('\nParsimony: \t{0} Fitness: \t{1} Expr: \t{2}'.format(parsim, fitness, algo_sym))

    def file_pareto_latex(self, pareto, root_path):
        """
        Save all pareto entries as latex gp_files
        - build tree from expression
        - fill tree meta-data, just in case we want to visualise anything of it
        - create latex-forest representation
        """

        forest_grouped = []

        for parsim, meta in sorted(list(pareto.items())):
            expr_raw = meta['expr_raw']  # sfeh: tree should already be sympified as much as possible
            tree = karoo_tree_from_expr(expr_raw, self.env_variables)
            tree = self.tree_beautify(tree, last_evolution='texify')
            ###
            vistree = visualize_tree_get_vistree(tree)
            ###

            tikz_code = latex_tree_get_forest(vistree)  # generate the small forest inputs

            # save a ready-to-use tex file with all pareto trees
            forest_grouped.append(latex_get_forest_title(parsim, meta['fitness_train'], tikz_code, tree_sep))

        latex_full_doc = latex_complete_tree_summary(forest_grouped)

        pth = file_make_dir(root_path / trees_tex)
        with Path.open(pth, 'w') as file:
            file.write(latex_full_doc)

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
            # todo idea round constants when generating

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
            # tree = karoo_tree_from_labellist(label_list_sym, self.env_variables)
            tree = karoo_tree_from_expr(expr_sym, self.env_variables)
            py_action = 'action = {}'.format(tree_get_pycode(tree))

            py_agent_name = '{}{:.0f}'.format(self.name, parsim)
            all_agent_names.append(py_agent_name)
            all_agents.append(py_class_code.format(py_agent_name, py_action))

        pycode_names = 'all_agents = [{}]\n'.format(', '.join(all_agent_names))
        py_agent_tuples = 'agent_tuples = [{}]\n'.format(', '.join(['(\'{}\', {}())'.format(x, x) for x in all_agent_names]))

        pycode_agents = '{}'.format('\n'.join(all_agents))

        pycode_complete_agents = 'import math\n\n' \
                                 '{}\n\n' \
                                 '{}\n\n' \
                                 '{}'.format(pycode_agents, pycode_names, py_agent_tuples)

        pth = file_make_dir(root_path / file_pycode)
        with Path.open(pth, 'w') as file:
            file.write(pycode_complete_agents)

        self.call_custom_file(root_path, pycode_complete_agents)  # sfeh root path is instance variabel

        return

    # todo idee: gp vs. nn entscheidungen clustern.

    def call_custom_file(self, root_dir, pycode_complete_agents):

        if Path.is_file(root_dir / callable_user_python_script):
            #  if direct execution is wished...# exec(Path.open("custom_eval_agents.py").read())

            # auto_import_eval = 'import sys\n' \
            #                    'from pathlib import Path\n' \
            #                    'sys.path.append(Path({}))\n' \
            #                    'import {} as custom_eval_agents\n' \
            #                    'custom_eval_agents.eval_agent_list(agent_tuples, folder=Path(\'img\'))'.format(callable_user_python_script, Path(callable_user_python_script).stem)

            with Path.open(root_dir / callable_user_python_script, 'r') as file:
                auto_import_eval = file.read()

            executable_python_evaluation = pycode_complete_agents + \
                                           '\nfrom pathlib import Path\n' + \
                                           'folder = Path.cwd() / \'custom_files\'\n\n' + \
                                           auto_import_eval

            with Path.open(root_dir / file_pycode_eval, 'w') as file:
                file.write(executable_python_evaluation)

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

        count_fails = 0

        for tree in self.population_tmp_eval:

            try:
                fitness_train = self.tree_eval_fitness_train(tree)
                tree = tree_set_fitness(tree, fitness_train)
                self.tree_meta_update(tree, fitness_train=fitness_train)
                self.pop_append(tree)
            except Exception as ex:
                print_warning('www', 'Exception while evaluating: {}'.format(ex), print_type=self.print_type)
                count_fails += 1
                continue

        if count_fails > 0:
            print_warning('ww', 'Evaluating {} trees in gen {} caused {} exceptions.'.format(
                len(self.population_tmp_eval), self.gen_id, count_fails), print_type=self.print_type)  # todo why more trees?

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

    def tree_meta_update(self, tree, fitness_train=None, parsimony=None, expr_raw=None, expr_sym=None):
        """
        update self.tree_meta
        # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw', 'gen+nr'}
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

        self.tree_meta[tree_ident] = meta

    def tree_eval_parsimony_easywrapper(self, tree):
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

        if tree_check_deep(tree, self.env_variables):
            tree = self.tree_beautify(tree, last_evolution='par-s')
            parsimony = self.tree_eval_parsimony_easywrapper(tree)
            tree = tree_set_parsimony(tree, parsimony)
            self.tree_meta_update(tree, parsimony=parsimony)
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
        best_fit = next(iter(sorted_parsimony_best))[1]['fitness_train']  # [1] accesses the meta, ['fitness_train'] the fitness

        for key, meta in sorted_parsimony_best:  # tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
            fitness = meta['fitness_train']
            parsim = meta['parsimony']
            pareto_improved = None

            if self.kernel.fitness_compare(fitness, best_fit):
                if self.pareto.get(parsim):
                    pareto_fit = self.pareto.get(parsim)['fitness_train']
                    if self.kernel.fitness_compare(fitness, pareto_fit):
                        self.pareto[parsim] = meta
                        self.printpl('a', 'Pareto update at {}, with new {}-error: {}. Old was: {}!'.format(parsim, self.config['kernel_name'], fitness, best_fit))
                        pareto_improved = True
                else:
                    self.pareto[parsim] = meta
                    self.printpl('a', 'New pareto entry at {:.0f} with {}-error: {:4.2f}!'.format(parsim, self.config['kernel_name'], fitness))
                    pareto_improved = True
                best_fit = fitness
            # todo idea plot #pareto entries / gen
            if pareto_improved:
                expr_raw = meta['expr_raw']  # expy_sym will can cause exceptions while setting fix nodes
                tree = karoo_tree_from_expr(expr_raw, self.env_variables)
                tree = tree_set_modifyable_nodes(tree, origin_tree=self.origin_tree_get())
                sym_tree = tree_evolve_reduce(tree, self.env_variables, completely=True)
                if list(tree_get_labellist(sym_tree)) != list(tree_get_labellist(tree)):
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
        last_fitness = copy.deepcopy(next(iter(sorted_pareto))[1]['fitness_train'])
        for parsim, meta in sorted_pareto[1:]:
            fitness = meta['fitness_train']
            if self.kernel.fitness_compare(fitness, last_fitness):
                last_fitness = fitness
            else:
                self.pareto.pop(parsim)
                self.printpl('aa', 'Pareto entry at {} became obsolete. Its fitness: {} was surpassed by simpler entry with fitness: {}.'.format(parsim, last_fitness, fitness))
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

        self.gen_id += 1
        self.time_genstart = time.perf_counter()
        self.population_tmp_done = []
        self.population_tmp_eval = []
        self.parsimony_tmp = max(1 / min(self.gen_id, self.config['gen_num_max_parsimony']) * self.parsimony_max, self.parsimony_max)

        return

    def tree_evolve_branch_multiple(self, tree, goal_nodes, env_variables, oparray, choose_distributions):

        # todo
        return tree

    def pop_reproduce(self, call_params):

        """

        copy a tree from the last population without changing its outcome
        """
        tree = call_params['old_tree']
        if call_params.get('sympify_tree'):
            tree = tree_evolve_reduce(tree, self.env_variables, completely=False)

        return tree

    def pop_reproduce_olymp(self, call_params):

        """
        Copy an entry from the pareto candidates into the population
        """

        if self.parsimony_best_meta:
            meta = np.random.choice(list(self.parsimony_best_meta.values()))
            expr_raw = meta['expr_raw']
            label_list = ast_convert_from_expr(expr_raw, build=True)
            xtype_list = xtypes_from_labels(label_list, self.env_variables)
            p_tree = Plagih_Tree(label_list, xtype_list)
            tree = p_tree.get_uninstanced_tree()
        else:
            tree = None

        return tree

    def pop_mutate_point(self, call_params):

        """
        Point mutation, One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """
        tree = call_params['old_tree']
        tree = tree_evolve_mutate_point(tree, self.choose_oparray, self.env_variables, self.choose_distributions)

        return tree

    def pop_mutate_filter(self, call_params):
        """

        """
        tree = call_params['old_tree']
        mode = call_params['mode']  # point/branch/all
        filter = 'gaussian_filter'  # sfeh change?

        # new_tree = tree_evolve_mutate_filter_one(tree)
        """
        Mutates a number of float terminal of a tree
        """
        # 1. choose a node
        node_ids = tree_get_mutatable_nodes(tree)
        if mode == 'branch':
            node_id = np.random.choice(node_ids)
            node_ids = tree_node_get_branch(tree, node_id)

        float_nodes = []
        for node_id in node_ids:
            if tree_node_get_xtype(tree, node_id) == '2f':
                try:
                    _ = float(tree_node_get_label(tree, node_id))
                    float_nodes.append(node_id)
                except ValueError:
                    pass

        if float_nodes:
            if mode == 'point':
                float_nodes = [np.random.choice(float_nodes)]
            for node_id in float_nodes:
                val = float(tree_node_get_label(tree, node_id))
                val = gp_mutate_constants(val, term_type='float', filter_type=filter)
                tree = tree_node_set_label(tree, node_id, val)
        else:
            print_warning('ww', 'Tree does not seem to have any float nodes for filtering.')

        return tree

    def choose_build_goal_depth(self, depth_tuple, old_node_depth):
        """

        """
        build_depth = int(np.random.normal(depth_tuple[0], depth_tuple[3]))  #
        build_depth = max(depth_tuple[1], build_depth)
        build_depth = min(depth_tuple[2] - old_node_depth, build_depth)
        return build_depth

    def choose_build_goal_nodes(self, nodes_tuple, existing_nodes):
        """

        """

        build_nodes = int(np.random.normal(nodes_tuple[0], nodes_tuple[3]))
        build_nodes = max(nodes_tuple[1], build_nodes)
        build_nodes = min(nodes_tuple[2] - existing_nodes, build_nodes)
        return build_nodes

    def pop_random_from_origin(self, call_params):
        """
        insert a (random) number of branches at the first possible "layer"
        (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
        - get these nodes, randomly choose a subset of those
        - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
        - split the amount of nodes up (randomly) and add these new branches to the tree
        todo fix min and max border
        """

        # tree_origin = self.origin_tree_get()
        fix_tree = call_params.get('fix_tree')
        build_mode = call_params.get('build_method')
        depth_tuple = call_params.get('depth_tuple')  # (base, min, max, normal_distrib)
        nodes_tuple = call_params.get('nodes_tuple')  # (base, min, max, normal_distrib)

        # tree_base = tree.copy()
        layer0_ids = tree_get_mutatable_layer(fix_tree, 0)  # ('We are about to create new branches randomly at nodes {}.'.format(layer0_ids))

        build_split = []
        if depth_tuple:
            for ii in range(len(layer0_ids)):
                build_depth = self.choose_build_goal_depth(depth_tuple, 0)
                build_split.append(build_depth)

        elif nodes_tuple:
            build_nodes = self.choose_build_goal_nodes(nodes_tuple, 0)  # sfeh actually, this does not care about tree depth
            build_split = randomly_split_range(build_nodes, len(layer0_ids))

        tree = fix_tree.copy()
        for i in range(len(layer0_ids)):  # finally, insert branches. need to get layer every time as node ids might have changed.
            layer0_ids = tree_get_mutatable_layer_lv0(tree)
            node_id = layer0_ids[i]
            node_xtype = tree_node_get_xtype(tree, node_id)
            old_branch = tree_node_get_branch(tree, node_id, karoo=True)
            branch_build = build_split[i]

            # label_list, arity_list, xtype_list = invent_label_list_nodes(node_xtype, branch_goal_nodes,
            #                                                              env_variables, oparray, choose_distributions,
            #                                                              build_mode='grow')

            if depth_tuple:
                # sfeh warning: Attention with this one. can get quite large with depth based
                label_list, arity_list, xtype_list = invent_label_list_depth(node_xtype, branch_build,
                                                                             self.env_variables, self.choose_oparray, self.choose_distributions,
                                                                             build_mode=build_mode)

            elif nodes_tuple:

                label_list, arity_list, xtype_list = invent_label_list_nodes(node_xtype, branch_build,
                                                                             self.env_variables, self.choose_oparray, self.choose_distributions,
                                                                             build_mode=build_mode)
            else:
                raise Exception('Known build_mode was not found for building random trees.')

            core = core_from_labels(label_list, arity_list, xtype_list)
            tree = tree_insert_subtree(tree, core, old_branch, karoo=True)

        return tree

    def pop_random(self, call_params):
        """
        Creates completely random trees from scratch
        """

        build_mode = call_params['build_method']
        if self.origin_exists():  # todo
            if tree_node_get_modify(self.origin_tree, root_id) != node_is_modifiable:
                print_warning('w', 'You can not create new trees from scratch when origin has fix nodes! This should be handeled earlier')
                return
        action_xtype = self.env_variables['action_at'][0]['xtype']

        if call_params.get('depth_tuple'):

            depth_tuple = call_params.get('depth_tuple')  # (base, min, max, normal_distrib)
            build_depth = self.choose_build_goal_depth(depth_tuple, 0)
            label_list, arity_list, xtype_list = invent_label_list_depth(action_xtype, build_depth,
                                                                         self.env_variables, self.choose_oparray, self.choose_distributions,
                                                                         build_mode=build_mode)

        elif call_params.get('nodes_tuple'):

            nodes_tuple = call_params.get('nodes_tuple')  # (base, min, max, normal_distrib)
            build_nodes = self.choose_build_goal_nodes(nodes_tuple, 0)
            label_list, arity_list, xtype_list = invent_label_list_nodes(action_xtype, build_nodes,
                                                                         self.env_variables, self.choose_oparray, self.choose_distributions,
                                                                         build_mode=build_mode)
        else:
            raise Exception('Known build_mode was not found for building random trees.')

        p_tree = Plagih_Tree(label_list, xtype_list, arity_list=arity_list)
        tree = p_tree.get_uninstanced_tree()

        return tree

    def pop_mutate_branch(self, call_params):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each old_node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """

        tree = call_params['old_tree']
        build_mode = call_params['build_method']
        # 'nodes_tuple', 'depth_tuple': 6, 'build_method': <'full', 'grow', 'half'>}

        node_ids = tree_get_mutatable_nodes(tree, no_root=True)
        old_node = np.random.choice(node_ids)
        old_xtype = tree_node_get_xtype(tree, old_node)

        # build_mode = np.random.choice(['full', 'grow'])

        if call_params.get('depth_tuple'):

            depth_tuple = call_params.get('depth_tuple')  # (base, min, max, normal_distrib)
            old_node_depth = tree_node_get_depth(tree, old_node)
            build_depth = self.choose_build_goal_depth(depth_tuple, old_node_depth)
            label_list, arity_list, xtype_list = invent_label_list_depth(old_xtype, build_depth,
                                                                         self.env_variables, self.choose_oparray, self.choose_distributions,
                                                                         build_mode=build_mode)



        elif call_params.get('nodes_tuple'):

            nodes_tuple = call_params.get('nodes_tuple')  # (base, min, max, normal_distrib)
            tree_size = tree_get_last_nodeid(tree)
            build_nodes = self.choose_build_goal_nodes(nodes_tuple, tree_size)

            label_list, arity_list, xtype_list = invent_label_list_nodes(old_xtype, build_nodes,
                                                                         self.env_variables, self.choose_oparray, self.choose_distributions,
                                                                         build_mode=build_mode)

        else:
            raise Exception('Known build_mode was not found for branch mutation.')

        if label_list:
            core_insert = core_from_labels(label_list, arity_list, xtype_list)
            branch_nodes_ids = tree_node_get_branch(tree, old_node, karoo=True)
            tree = tree_insert_subtree(tree, core_insert, branch_nodes_ids, karoo=True)
        else:
            tree = None

        return tree

    def pop_crossover_branch(self, call_params):
        """
        swap branches of two trees
        - select parent a and b
        - select swappable branche for a_parent from b_parent
            - select a node in a (and crossover here, no matter what)
        - delete a_parent branch and insert b_parent branch (which tactic?)

        """

        # 1. two parents
        left_tree = call_params['old_tree']
        right_tree = call_params['partner_tree']

        # 2. search nodes for left and right that can be exchanged. convert_needed
        left_id, right_id, success = self.tree_try_get_swapids(left_tree, right_tree)
        if not success:
            right_id, left_id, success = self.tree_try_get_swapids(right_tree, left_tree)

        left_ids, left_labels, left_aritys, left_xtypes = tree_get_branch_ilax(left_tree, left_id)
        right_ids, right_labels, right_aritys, right_xtypes = tree_get_branch_ilax(right_tree, right_id)

        if not success:
            print_warning('ww', 'Crossover conversion between trees not possible: \n{}\n{}'.format(left_tree, right_tree))
            # left_xtype = xtype_get_from_label(tree_node_get_label(left_tree, left_id), self.env_var_dummy)
            # conv_to_left, conv_to_right = xtype_get_converters(left_xtype)
            # right_labels.insert(0, conv_to_left)
            # right_aritys.insert(0, 1)
            # left_labels.insert(0, conv_to_right)
            # left_aritys.insert(0, 1)
            return

        left_core = core_from_labels(left_labels, left_aritys, left_xtypes)  # todo this is not necessary, switch branches
        right_core = core_from_labels(right_labels, right_aritys, right_xtypes)

        left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
        left_offspring = tree_prune_depth(left_offspring, self.config['tree_depth_max'], self.env_variables, self.choose_distributions)

        right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
        right_offspring = tree_prune_depth(right_offspring, self.config['tree_depth_max'], self.env_variables, self.choose_distributions)

        return left_offspring, right_offspring

    def tree_beautify(self, tree, last_evolution=''):
        """
        The np-tree needs more information than only the expression.
        -> set modifyable nodes (mandatory)

        -> round all distributions_file
        -> try to normalize exponents ('**'). sfeh, not really working.
        -> set last evolution (for analysing gp operators. e.g. if no good trees originate from crossover, something might be wrong)
        -> set xtype for all nodes.
        """

        if tree is None:
            print_warning('ww', 'Tree from last_evolution: \'{}\' failed. probably sympify. Continuing.'.format(last_evolution))
        else:
            tree = tree_set_modifyable_nodes(tree, origin_tree=self.origin_tree_get())
            tree = tree_round_constants(tree, self.config['float_accuracy'], karoo=True)  # sfeh: test 08.04.2020
            tree = tree_normalize_exponentiation(tree)
            tree = tree_set_last_evolution(tree, last_evolution)
            # tree = tree_set_xtypes(tree, self.env_variables)

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

        tree = self.tree_beautify(tree, last_evolution=last_evolution)  # sfeh beautify somewhere else

        if tree_check_quick(tree):
            tree_ident = tree_hash(tree)

            if tree_ident in self.tree_meta:

                tree_meta = self.tree_meta[tree_ident]
                tree = tree_set_evalutaion(tree, tree_meta)
                self.population_tmp_done.append(tree)
            else:
                parsimony = self.tree_eval_parsimony_easywrapper(tree)
                if parsimony <= self.parsimony_max:
                    tree = tree_set_parsimony(tree, parsimony)
                    tree = tree_set_fitness(tree, '')
                    self.population_tmp_eval.append(tree)
                else:
                    print_warning('www', 'Tree was too complex! Last Evolution: {}'.format(last_evolution), print_type=self.print_type)

        return

    def gen_finalize(self):

        """
        From raw population_tmp_done
        - Evaluate leftover

        """

        self.pop_eval_remaining()
        self.pareto_update()
        self.pop_base_transfer()
        self.pop_analyze()
        file_population_karoo(self.population_tmp_done, 'tmp', self.root_dir, self.gen_id)

        self.monitoring_dict['total_found_trees'][self.gen_id] = len(self.tree_meta)
        self.print_g('gg', 'Created {}/{} unique trees in generation {}. Gen took {:4.2f}s'.format(
            len(self.population_tmp_done), self.config['pop_max'], self.gen_id, time.perf_counter() - self.time_genstart))

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

            fitness = tree_get_fitness(tree, precision=self.config['precision'])

            if self.kernel.fitness_compare(fitness, best_fitness, mode='better'):
                best_id = tree_id
                best_fitness = fitness

        tourn_winner = copy.deepcopy(self.population_base[best_id])
        tourn_winner = tree_set_meta_wipe(tourn_winner)

        return tourn_winner

    def tree_try_get_swapids(self, a_tree, b_tree, version='default'):
        """
        Try to return two branches (aka ids) [for crossover] that can be crossed

        """
        if version == 'default':
            # choose a node from parent a
            a_ids = tree_get_mutatable_nodes(a_tree, no_root=True)
            # a_ids = tree_get_mutatable_layer_lv0(a_tree)  # todo
            a_id = np.random.choice(a_ids)
            a_label = tree_node_get_label(a_tree, a_id)
            a_label, _, a_xtype = tree_node_get_lax_v3(a_tree, a_id)

            # create a list from parent b with same xtype
            b_node_ids = tree_get_mutatable_nodes(b_tree, no_root=True)
            b_sametype_ids = b_node_ids[:]
            for b_id in b_node_ids:
                b_label, _, b_xtype = tree_node_get_lax_v3(b_tree, b_id)
                if not xtype_equi_outcome(b_xtype, a_xtype):
                    b_sametype_ids.remove(b_id)  # remove one-by-one false partner nodes.

            if b_sametype_ids:  # if entries were found, choose one. we are custom_done
                b_id = np.random.choice(b_sametype_ids)
                success = True
            else:
                b_id = np.random.choice(b_node_ids)
                success = False

            return a_id, b_id, success
        else:
            raise

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def activate_origin_tree(self, tree):
        """

        """
        if not tree_check_deep(tree, self.env_variables):
            raise

        expr_raw = tree_get_expr_raw(tree, node_id=root_id)
        expr_sym = expr_sympify(expr_raw=expr_raw)
        tree_check_expr(tree)

        # sfeh, this does not work
        # if not tree_check_is_sympified(tree):
        #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
        #                          ''.format(expr_raw, expr_sym))

        if not tree_check_deep(tree, self.env_variables):
            raise

        self.origin_tree = copy.deepcopy(tree)
        self.origin_meta = {'expr_raw': expr_raw, 'expr_sym': expr_sym, 'parsimony': 0}
        try:
            fitness_train = self.tree_eval_fitness_train(tree)
        except Exception:
            raise Exception('Your origin_meta algorithm already caused an exception!')
        self.origin_meta['fitness_train'] = fitness_train

        self.parsimony_best_meta[0] = self.origin_meta
        self.pareto[0] = copy.deepcopy(self.origin_meta)

        self.print_g('gg', 'Loading origin_meta, fitness {}. Time: {:4.2f}s'.format(fitness_train, time.perf_counter() - self.time_start))

        return

    def tree_eval_fitness_train(self, tree):
        """
        Evaluating the fitness of a tree.
        - extract the expression the tree is holding
        - sympify the expression
        - (if sympify fails, evaluating does not make sense! Check sympify errors)
        - (sfeh: if sympify fails because of inf or zoo, tf could maybe still work due to save-tf-division)

        This evaluation should only be called inside a try-block.
        sympification is allowed to fail and also tf-eval showed some exceptions in the past
        (now, tf-problems are all gone, but still, the program should never crash because of one tricky tree)
        """

        try:
            expr_sym = tree_get_expr_sym(tree)
        except Exception as ex:
            raise Exception('Expr could not be sympified: {}'.format(ex))

        fitness_train = eval_tf(expr_sym, self.data_train, self.kernel, self.env_variables, self.tf_device_log, self.tf_device, self.tf_classify_labels_map)['fitness']

        # print(str(fitness_train), 'str fitness train')
        # if str(fitness_train) == 'inf':
        #     print('fitness_train', fitness_train, type(fitness_train), fitness_train != float('inf'))
        if not check_value_is_real(fitness_train):
            raise Exception('Fitness_train is not a real number: {}'.format(fitness_train))
        # else:
        #     print('debug aa1', fitness_train, 'is of type', type(fitness_train), 'the check is', check_value_is_real(fitness_train))

        return fitness_train

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to use evaluate (tensorflow)      +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def tf_classify_labels_map(self, result):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the data_csv_path .csv. Outputs an array of tuples containing the predicted
        labels based upon the result and corresponding boolean condition triggered.

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

        pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))),
                             lambda: label_rules[1])

        return pred_label

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                +
    # +++++++++++++++++++++++++++++++++++++++++++++

    def get_pareto_plot_values(self):
        """
        sfeh i think there is a more beautiful solution?
        """
        tuples = []
        for key in sorted(self.pareto):
            tuples.append((key, self.pareto[key]['fitness_train']))
        return tuples

    def file_all_plots(self, path):
        """
        Make all plots
        """

        path_plots = make_dir(path / folder_plots)

        if self.monitor_dict['gen_fitness_average'] == 'y':
            data_tuples = sorted(list(self.monitoring_dict['fitness_average'].items()))
            self.plot_end(data_tuples, path_plots, plt_title='average error', plt_y_label='fitness',
                          linestyle='-',
                          set_left=data_tuples[0][0])

        if self.monitor_dict['population_tmp_done-size'] == 'y':
            data_tuples = sorted(list(self.monitoring_dict['population_tmp_done-size'].items()))
            self.plot_end(data_tuples, path_plots, plt_title='genepool size', plt_y_label='amount', linestyle='',
                          set_left=data_tuples[0][0])

        data_tuples = sorted(list(self.monitoring_dict['complexity_average'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='average tree complexity', plt_y_label='#nodes',
                      linestyle='-',
                      set_left=data_tuples[0][0])

        data_tuples = sorted(list(self.monitoring_dict['total_found_trees'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='number of created trees', plt_y_label='amount', linestyle='',
                      set_left=data_tuples[0][0])

        data_tuples = self.get_pareto_plot_values()
        self.plot_end(data_tuples, path_plots, plt_title='pareto dominant candidates', plt_x_label='parsimony', plt_y_label='fitness',
                      linestyle='dashed',
                      step_where='post',
                      set_right=self.parsimony_max,
                      beyond_lines=True,
                      save_tikz=True)

        dist_fit = self.monitoring_dict['tmp_pop_fitness_distribution']
        self.plot_end(dist_fit, path_plots, plt_title='population distribution Gen {}'.format(self.gen_id), plt_y_label='fitness',
                      linestyle='-',
                      marker='',
                      set_right=self.config['pop_max'],
                      right_padding=1,
                      subfolder=folder_pop_analysis)

        if self.monitor_dict.get('fitness_variance') == 'y':
            data_tuples = sorted(list(self.monitoring_dict['fitness_variance'].items()))
            self.plot_end(data_tuples, path_plots, plt_title='variance in error', plt_y_label='variance',
                          linestyle='-',
                          marker='')

        data_tuples = sorted(list(self.monitoring_dict['complexity_variance'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='variance in complexity', plt_y_label='variance',
                      linestyle='-',
                      marker='')

        data_tuples = sorted(list(self.monitoring_dict['best_candidate'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='best candidate', plt_x_label='generation', plt_y_label='error',
                      linestyle='dashed',
                      step_where='post')

        # sfeh https://github.com/linkedin/naarad/issues/114 UserWarning: Attempting to set identical bottom==top results

        return

    def pop_analyze(self):
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
            parsimony = tree_get_parsimony(tree)
            last_modi = tree_get_last_evolution(tree)
            pop_tree_analysis.append({'fitness': fitness, 'complexity': parsimony, 'last_evolve': last_modi})

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
        dist_fit = [(i, x) for i, x in enumerate(sorted([x['fitness'] for x in pop_tree_analysis]))]  # sorting based on fitness
        self.monitoring_dict['tmp_pop_fitness_distribution'] = dist_fit

        # Tree complexity
        complexity_sum = 0
        for ii, tree in enumerate(self.population_tmp_done):
            complexity_sum += len(tree_nodes_get_ids(tree, karoo=True))
        avg_complexity = complexity_sum / tree_cnt
        self.monitoring_dict['complexity_average'][self.gen_id] = avg_complexity

        # fitness variance
        fitness_variance = np.var([x['fitness'] for x in pop_tree_analysis])
        self.monitoring_dict['fitness_variance'][self.gen_id] = fitness_variance

        # complexity variance
        parsimony_variance = np.var([x['complexity'] for x in pop_tree_analysis])
        self.monitoring_dict['complexity_variance'][self.gen_id] = parsimony_variance

        return

    def get_observation_bundle(self):
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

    def plot_end(self, data_2d, path,
                 plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='', yscale='linear', step_where='', plt_xparam='',
                 linestyle='None',
                 marker='.',
                 set_left=None, set_right=None, set_top=None,
                 right_padding=1.05, top_padding=1.05,
                 beyond_lines=False,
                 save_tikz=False,
                 subfolder=None):
        """
        Make all plots in the same style - and also saving space.
        - Makes pyplots


        :param data_2d: array with data, e.g. [[1, 5],[2, 4], [3, 4]]
        :param path: where to save the result
        :param plt_title:
        :param plt_curve_label: irrelevant for a single curve
        :param plt_x_label: label the x-axis
        :param plt_y_label: label the y-axis
        :param yscale: only 'linear'.
        :param step_where: makes 'step' plots- can be 'post', 'pre' or [pls google]
        :param plt_xparam: not in use, the same adjustment can be done with optional parameters
        :param linestyle: E. g. 'None', 'dashed', '-', ''
        :param set_left: Smallest left value
        :param set_right: E. g. if max_parsimony is 100 -> show complete width, even if entries only go to 40
        :param beyond_lines: in step plots, draw the line further to the left and right
        :param save_tikz: Also save the plot as tikzpicture (for Latex)(requires tikzplotlib)
        :param subfolder: save plot in plots/*subfolder*, e.g. if this plot is created in every generation
        :return:

        sfeh: max_height=None,  # when creating a plot in every generation, fix the maximum height and width?
        """

        if len(data_2d) == 0:
            print_e('Plotting empty array is not possible! Data={}'.format(data_2d))
            return

        x, y = [], []
        for a, b in data_2d:
            x.append(a)
            y.append(b)

        # x, y = data_2d.reshape(-1, 2).T  # sfeh this could be a more pythonic way, but tuples can not be reshaped.

        # bottom, top = plt.ylim()
        # left, right = plt.xlim()

        top, bottom, left, right = max(y), min(y), min(x), max(x)
        if set_left:
            left = set_left
        if set_top:
            new_top = set_top
        else:
            new_top = (top - min(bottom, 0)) * top_padding  # top * 1.05 for better style

        if set_right:
            right = max(right, set_right)
        new_right = right * right_padding

        if beyond_lines:
            x = [x[0]] + x + [new_right + 1]
            y = [new_top + 1] + y + [y[-1]]

        if step_where:
            plt.step(x, y, plt_xparam, linestyle=linestyle, marker=marker, label=plt_curve_label, where=step_where)
        else:
            plt.plot(x, y, plt_xparam, linestyle=linestyle, marker=marker, label=plt_curve_label)

        # let it start at (0,0) but +5% margin to the top and right
        plt.yscale(yscale)
        plt.ylim(min(bottom, 0), new_top)
        plt.xlim(min(left, 0), new_right)
        plt.margins(x=0, y=0)

        plt.xlabel(plt_x_label)
        plt.ylabel(plt_y_label)
        plt.title(plt_title)

        # plt.legend()
        if subfolder:  #
            path = make_dir(path / subfolder)

        plt.savefig(path / '{}.png'.format(plt_title))
        if save_tikz:
            try:
                tikzplotlib.save(path / '{}.tex'.format(plt_title))
            except Exception as ex:
                pass

        plt.close()
        return

    def printpl(self, message_type, text):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        :param message_type: 'gggiiiivvv...' options can be found in config
        """

        if message_type in self.print_type:
            printez(message_type, text, print_type=self.print_type, time_total=time.perf_counter() - self.time_start)

        return

    def print_g(self, message_type, text):
        """

        """

        if message_type in self.print_type:
            printez(message_type, text, time_total=time.perf_counter() - self.time_start)

        return


def check_value_is_real(fitness):
    """
    Returns bool value if we can use the calculated fitness
    Fitness values might evaluate to weird stuff
    e.g. 'nan' after dividing by zero or (inf) after 20**1234
    nan: fitness == fitness -> False
    inf: fitness is not float('inf') -> False
    """
    return fitness == fitness and fitness != float('inf')
