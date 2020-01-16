"""
Explaination:
> 'f2f', 'b2b', etc.: my personal silly naming. f = float, b = bool.
   f2b means float to boolean, e. g. '<' takes 'float' and returns a 'bool'
> node_modify (0 or 1), specifies, whether this node is supposed to be




Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""

import sys
import csv
import sklearn.metrics as skm
import sklearn.model_selection as skcv
from datetime import datetime
from plagih.modules.plagih_tree import *
import pickle
import re
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
import numpy as np

import matplotlib.pyplot as plt
from plagih.modules.tree_distances.tree_edit_distance import apted_distance
import time
from pathlib import Path

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# TODO hash dict based on the sympy version?
# Load backupfrom a .csv File (TODO), check, if it is compartible with tree_origin (TODO)
# todo write why Min and Max is crap (sympy multielement, tf problem with ast)
# TODO all functions have to be within one of these lists. check it.
# random TODO replace and with &, see https://docs.sympy.org/latest/_modules/sympy/core/relational.html
# TODO count new tree_ident in each evolve step
# TODO point mutation mehrfach anwenden? branch mutation mehrfach anwenden?
# TODO ...somit wäre garantiert, dass jedes mal die richtige Lösung harauskommen könnte. Nur mit
# TODO ...einer Mutation pro Kandidat wäre das nicht möglich
# TODO: Wenn average fitness convergiert oder alte Bäume neu auftreten, dann sollten zufällige Kandidaten erzeugt werden!
# todo. "zufällige" Kandidaten bedeuten auch: Alle branches ausprobieren- mehrfach branch mutation?
# TODO: TED with values- just assume values are elements? 0.12 == 0.1 distance wise? ...
# Genepool_create: TODO stop equal candidates from being in the gene pool multiple times?
# ( and { in karoo and TED. sfeh/todo: this can be optimized to create a nicer brackets-styled algorithm
# Todos in evolve_subtree_depth_choose
# # TODO consider tree size of last tree,
# # TODO consider random tree size,
# # TODO consider always maximum tree size,
# # TODO is this already considered by 50:50 func-term?
# TODO point mutation should also reduce arities if needed?
# TODO tree_choose_node_id only works for same arity functions
#  todo random samples out of dataset values as new constants?
# TODO zoo and inf and nan in plagih_sympify... other solution?
# TODO what is "swim" in karoo, what is it good for?
# TODo check memory usage?
# random TODO grow depth anpassen!
# TODO anzahl bereits bekannter bäume
# TODO Field Guide programming lesen

# TODO save sympifyed versions of trees

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees


class ExplainableGP(object):
    """
    The main class performing all the important stuff
    """

    def __init__(self, config_dict):

        self.time_start = time.perf_counter()
        self.tree_hash_meta = {}
        self.parsimony_best_dict = {}
        self.pareto = {}
        self.population_base = []  # population that is taken to the next generation

        # 1. set global variables to those local values passed from the user script
        self.config = config_dict
        self.kernel = config_dict['kernel']  # fitness function
        self.display = config_dict['display']
        self.precision = config_dict['precision']  # the number of floating points for the round function
        self.parsimony_min_max = config_dict['tree_parsimony_min_max']

        self.tnode = {}

        self.tf_device = "/gpu:0"  # Set TF computation backend device (CPU or GPU); gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device_log = False  # TF device usage logging (for debugging)

        # What trees should be built
        # TODO
        self.asdf = {'tree_depth_max': 10,
                     'strongly_typed': True,
                     'mutate_branch_depth_max': 6,
                     'mutate_branch_build_method': 'grow',
                     'mutate_branch_grow_term_probability': 0.5,
                     'crossover_strategy': ['same_type', 'same_type_switched', 'convert', 'plagih_switcharoo'],
                     'tree_min_nodes': 3}

        self.monitor_dict = config_dict['monitor']
        self.evolve_rates = config_dict['evolve_rates']
        evolve_missing = 1-sum(self.evolve_rates.values())
        self.evolve_rates['Create Random'] += evolve_missing

        self.fitness_type = fitt_dict[self.kernel]  # load fitness type
        if self.fitness_type == 'max':
            self.fitness_bad_dummy = 0
        else:
            self.fitness_bad_dummy = float("inf")
        self.gene_pool = {}
        self.xype_func_dict = {'f2f': [], 'f2b': [], 'b2b': [], 'b2f': [], 'b2f2f': [],
                               '2b': [], '2f': [],
                               'b2': [], 'f2': []}  # todo delete this

        # some useful stuff
        self.debug_warnings = {}
        self.monitoring_dict = {'genepool_size': {},
                                'fitness_average': {},
                                'total_found_trees': {}}

        self.file_directories_create()
        self.done = False
        self.printpl('gg', 'Init. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        return

    def plagih_gp_run(self):
        """
        regular plagih-config run from scratch
        """

        self.gen_id = 0  # set initial generation ID    # first gen only
        self.file_config()
        self.main_generation_first_origin()
        self.main_generation_loop()  # (main loop)
        self.printpl('gg', 'GP-run. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        self.file_autowrite(self.path, 'f')  # archive populations and return to plagih_gp.py for a clean exit
        self.file_autoplots(self.path)
        self.printpl('gg', 'Completely. Exit. \tTime: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        sys.exit()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Top level      functions                  |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def file_population_write(self, population, key, path):

        """
        Save population_* to disk.

        """
        file_path = path / 'population_{}.csv'.format(str(key))

        with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
            target = csv.writer(csv_file, delimiter=',')
            if self.gen_id != 0:
                target.writerows([''])  # empty row before each generation
            target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(self.gen_id)]])

            for tree in range(1, len(population)):
                target.writerows([''])  # empty row before each Tree
                for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                    target.writerows([population[tree][row]])

        return

    def file_directories_create(self):
        """
        Create all files that will be saved after all
        """

        self.datetime = datetime.now().strftime('%Y%m%d-%H%M%S')
        cwd = Path.cwd()
        self.path = cwd / 'runs' / '{}{}'.format(self.datetime, self.config['name'])
        if not Path.is_dir(self.path):
            Path.mkdir(self.path)

        return

    def main_generation_first_origin(self):
        """
        Everything that needs to be done for the first generation
        - Extracts "origin Tree" from file
        - Creates all other trees: origin tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """
        self.printpl('gg', 'Preparing to evolve Generation {}'.format(self.gen_id))
        self.gen_prepare_parameters()
        self.pop_first_create()
        self.pareto[0] = self.origin['fitness_train']
        self.gen_finalize()
        self.file_population_write(self.population_base, '1_first', self.path)  # first gen only

    def main_generation_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """

        gp_list = [('Reproduce gen', self.gen_reproduce),
                   ('Point Mutation', self.gen_mutate_point),
                   ('Point Filter', self.gen_mutate_filter),
                   ('Branch nodebased', self.gen_mutate_branch),
                   ('Crossover one Branch', self.gen_crossover_branch),
                   ('Create Random', self.gen_create_random)]
        tourn_size = self.config['gp_tourn_size']

        # todo add stop after we achieved our goal
        while self.gen_id < self.config['gen_max'] and \
                time.perf_counter() - self.time_start < self.config['time_max'] and \
                not self.done:

            self.gen_id += 1
            self.gen_prepare_parameters()

            for name, gp_function in gp_list:
                self.printpl('gggg', '{}...'.format(name))
                evolve_num = int(self.evolve_rates[name] * self.config['pop_max'])
                time_evolve = time.perf_counter()

                # n = 0
                # while n < evolve_rate:
                #     n += 1
                #     pass
                gp_function(evolve_num, tourn_size)
                self.printpl('ggg', '{} took: {:4.2f}.'.format(name, time.perf_counter() - time_evolve))

            self.autosave_stuff()

            self.gen_finalize()
            self.printpl('ggg', 'Generation took a total time of: {:4.2f}'.format(time.perf_counter() - self.time_genstart))
        else:
            self.printpl('p', '{} Enter {}?{} to review your options or {}q{}uit{}'.format(BColors.GREEN, BColors.BOLD, BColors.GREEN, BColors.BOLD, BColors.GREEN, BColors.RESET))
            # menu_continue = 0

    def autosave_stuff(self, overwrite=True):
        """

        """
        if overwrite:
            path_auto = self.path / 'autosave'
            auto_enumname = 'tmp'
        else:
            path_auto = self.path / 'Gen-{}'.format(self.gen_id)
            auto_enumname = str(self.gen_id)
        if not Path.is_dir(path_auto):
            Path.mkdir(path_auto)

        time_now = time.perf_counter()

        if self.config['period']['time_monitor']:
            if self.config['period']['time_monitor'] < (time_now - self.time_last_monitor):
                self.printpl('ii', 'auto-plots (time)')
                self.file_autoplots(self.path)
                self.time_last_monitor = time_now

        if self.config['period']['time_save']:
            if self.config['period']['time_save'] < (time_now - self.time_last_files):
                self.printpl('ii', 'auto-plots (time)')
                self.file_autowrite(path_auto, auto_enumname)
                self.time_last_files = time_now

        if self.config['period']['gen_monitor']:
            if self.gen_id % int(self.config['period']['gen_monitor']) == 0:
                self.printpl('ii', 'auto-plots (time)')
                self.file_autoplots(self.path)

        if self.config['period']['gen_save']:
            if self.gen_id % int(self.config['period']['gen_save']) == 0:
                self.printpl('ii', 'auto-plots (time)')
                self.file_autowrite(path_auto, auto_enumname)

        return 0

    def file_autowrite(self, path, gen):
        """
        writes all important files

        """
        self.file_conclusion(path)
        self.file_pareto(self.pareto, path)
        self.file_population_write(self.population_new, str(gen), path)  # save the final generation of Trees to disk

    def gen_olympus_update(self):
        """
        The olymp is where the godlike contestants reside.
        In each generation, the olymp searches for new god contestants
        """
        self.printpl('e', 'TODO Olympus for candidates')
        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Load and Archive Data                     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def data_from_csv(self, samples_file, save_pickle_path=None):

        """
        loads the goal-data_csv_path from .csv file. first observations then actions.
        Both can have any shape specified in the gym.env "spaces" (dimensions: 1-n, type: int-floatstring?)

        Mountaincar .csv first lines (11.12.2019):
        --------------------------------------------------------
        observation0:float,     observation1:float, action0:float
        -0.5031261704876531,    0.0,                2
        --------------------------------------------------------
        """

        num_observations, num_actions = 0, 0
        var_types = []
        # TODO Terminal types as dictionary? would be much prettier.
        input_dict = {'all': {},
                      'float': {},
                      'bool': {}}
        variables_dict = {'all': [],
                          'types': [],
                          'float': [],
                          'bool': []}

        action_dict = {}
        actions, action_types = [], []

        # 1. Read file
        with Path.open(samples_file) as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            for i, row in enumerate(reader):
                if i == 0:  # variable identifiers
                    # all_variables = [x.rsplit(':', 1)[0] for x in row]  # ['observation0:float'] -> ['observation0']
                    for var_name in row:
                        var_types.append(var_name.split(':', 1)[1])
                        if var_name.startswith('o'):  # found an observation
                            num_observations += 1
                            term = var_name.rsplit(':', 1)[0]
                            term_type = var_name.split(':', 1)[1]
                            input_dict[term] = term_type  # todo austauschen
                            variables_dict['all'].append(term)
                            variables_dict['types'].append(term_type)
                            if term_type == 'float':
                                variables_dict['float'].append(term)
                            elif term_type == 'bool':
                                variables_dict['bool'].append(term)
                            else:
                                raise
                        elif var_name.startswith('a'):  # found an action
                            num_actions += 1
                            action = var_name.split(':', 1)[0]
                            action_type = var_name.split(':', 1)[1]
                            action_dict[action] = action_type  # Do not use this:# '2b' if 'bool' in action_type else '2f'
                        else:
                            self.printpl('e', 'Behaviour samples first line: Variables have to start with "o" or "a" to be recognized. Is actually: {}'.format(var_name))
                            raise

                    data_x, data_y = [], []

                else:  # convert every 'string' element to its data_csv_path type
                    # TODO var_types ist genau dasselbe wie self.terminal , oder? eines ersetzen?
                    row_as_data = [locate(var_types[i])(x) for i, x in enumerate(row)]  # ['observation0:float'] + ['0.123'] --> float(['0.123']) --> 0.123
                    data_x.append(row_as_data[:num_observations])
                    data_y.append(row_as_data[num_observations:])
            csvFile.close()

        self.input_dict = input_dict
        self.variables_dict = variables_dict
        self.action_dict = action_dict
        self.class_labels = len(np.unique(data_y))  # load the user defined true labels for classification or solutions for regression
        self.data_train_rows, self.data_train, self.data_control = data_load_data_split(data_x, data_y, test_size=0.2)

        if save_pickle_path:
            pickle_data = {'self.input_dict': self.input_dict,
                           'self.variables_dict': self.variables_dict,
                           'self.action_dict': self.action_dict,
                           'self.class_labels': self.class_labels,
                           'self.data_train_rows': self.data_train_rows,
                           'self.data_train': self.data_train,
                           'self.data_control': self.data_control}
            with Path.open(save_pickle_path, 'wb') as file:
                pickle.dump(pickle_data, file)

        self.printpl('g', 'Loading samples. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        return

    def data_load_pickle(self, prepared_data_pickle_path):
        """
        loads a data_csv_path file that was already split with the csv reader
        """
        with Path.open(prepared_data_pickle_path, 'rb') as file:
            pickle_data = pickle.load(file)
        # pickle_data = pickle.load(Path.open(prepared_data_pickle_path, 'rb'))

        self.input_dict = pickle_data['self.input_dict'],
        self.variables_dict = pickle_data['self.variables_dict'],
        self.action_dict = pickle_data['self.action_dict'],
        self.class_labels = pickle_data['self.class_labels'],
        self.data_train_rows = pickle_data['self.data_train_rows'],
        self.data_train = pickle_data['self.data_train'],
        self.data_control = pickle_data['self.data_control']

        self.printpl('g', 'Pickle-loading samples. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        return

    def data_load_operators(self, operators_file_path):
        """
        Load all operators ready-to-use from a file
        """

        functions = np.loadtxt(operators_file_path, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
        # Part 3.5: Split the functions in 5 types

        # rows are the function types (f2f)
        # columns are the arity
        self.op_type_arity_array = [[[], [], [], []],
                                    [[], [], [], []],
                                    [[], [], [], []],
                                    [[], [], [], []],
                                    [[], [], [], []]]
        for fun in functions:
            label = fun[0]
            arity = op[label]['arity']  # arity = int(fun[1])
            xtype = op[label]['xtype']

            if xtype == 'f2f':
                self.xype_func_dict['f2f'].append(label)
                self.xype_func_dict['2f'].append(label)
                self.xype_func_dict['f2'].append(label)
                self.op_type_arity_array[f2f][arity].append(label)
            elif xtype == 'f2b':
                self.xype_func_dict['f2b'].append(label)
                self.xype_func_dict['2b'].append(label)
                self.xype_func_dict['f2'].append(label)
                self.op_type_arity_array[f2b][arity].append(label)
            elif xtype == 'b2b':
                self.xype_func_dict['b2b'].append(label)
                self.xype_func_dict['2b'].append(label)
                self.xype_func_dict['b2'].append(label)
                self.op_type_arity_array[b2b][arity].append(label)
            elif xtype == 'b2f':
                self.xype_func_dict['b2f'].append(label)
                self.xype_func_dict['2f'].append(label)
                self.xype_func_dict['b2'].append(label)
                self.op_type_arity_array[b2f][arity].append(label)
            elif xtype == 'b2f2f':
                self.xype_func_dict['b2f2f'].append(label)
                self.xype_func_dict['2f'].append(label)
                self.xype_func_dict['f2'].append(label)
                self.xype_func_dict['b2'].append(label)
                self.op_type_arity_array[b2f2f][arity].append(label)

        if not self.xype_func_dict['2f'] and not self.xype_func_dict['2b']:
            self.printpl('w', 'Neither Boolean nor Float Values can be created!')
        elif not self.xype_func_dict['2f']:
            self.printpl('w', 'No Float Values can be created!')
        elif not self.xype_func_dict['2b']:
            self.printpl('w', 'Neither Boolean Values can be created!')

        self.printpl('g', 'Loading operators. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        return

    def data_pickle_save(self):
        """
        save all data_csv_path every few rounds to restore them
        - save the pareto front (done)
        - save the last generation (done)
        - Save valuable meta-data_csv_path: current generation (done)
        TODO not complete
        """
        run_data = {'gen_id': self.gen_id,
                    'parsimony_front_fitness': '',
                    'pareto': self.pareto,
                    'hash_trees_meta': self.tree_hash_meta,
                    'population_new': self.population_new
                    }
        pickle.dump(run_data, Path.open(self.path / 'Gen-{}-backup.p'.format(str(self.gen_id)), 'wb'))

    def autosave(self):
        """
        automatically saves everything important after a certain amount of time
        """
        # saving data_csv_path (including the split)
        # saving pareto front
        # saving hash dict
        pass

    def file_conclusion(self, path):

        """
        write the performance of the config to disc
        """

        file = Path.open(path / 'conclusion.txt', 'w')
        file.write('Plagih GP\n launched: ' + str(self.datetime))

        result = self.eval_tf(self.origin['algo_sym'], self.data_control, get_pred_labels=True)
        self.origin['fit_control'] = result['fitness']
        fit_best = result['fitness']

        fittest_algo = self.origin['algo_sym']
        fittest_parsimony = 0

        for parsimony, tree_hash in self.pareto.items():

            algo_sym = self.tree_hash_meta[self.parsimony_best_dict[parsimony]]['algo_sym']
            result = self.eval_tf(algo_sym, self.data_control, get_pred_labels=True)
            fit_control = result['fitness']

            if self.fitness_compare(fit_control, fit_best, mode='better_or_equal') or\
                    self.kernel == 'regression' and fit_control <= fit_best or\
                    self.kernel == 'match' and fit_control == self.data_train_rows:  # find the Tree with a perfect match for all data_csv_path rows

                fit_best = fit_control
                fittest_algo = algo_sym
                fittest_parsimony = parsimony

            if self.kernel == 'classification':
                file.write('\n\n Classification fitness score: {}'.format(fit_best))
                file.write('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution'], result['pred_labels'][0])))
                file.write('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution'], result['pred_labels'][0])))

            elif self.kernel == 'regression':
                mse, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
                file.write('\n\n Regression fitness score: {}'.format(fitness))
                file.write('\n Mean Squared Error: {}'.format(mse))

            elif self.kernel == 'match':
                file.write('\n\n Matching fitness score: {}'.format(result['fitness']))

        else:  # pareto dict is empty
            file.write('\n\n No solution was better than the origin... your species has gone extinct!')

        # Info about the origin tree
        file.write('\n\t Origin fitness score: {}'.format(self.origin['fit_control']))

        # Info about the best Tree
        file.write('\n\n The best candidate has parsimony: {}'.format(str(fittest_parsimony)))
        file.write('\n With fitness: {}'.format(fit_best))
        file.write('\n\n With the following sympify-algorithm:\n {}'.format(fittest_algo))
        file.write('\n\n')
        file.close()

        return

    def file_config(self):
        """
        write the parameters to a file
        Todo update
        """

        file = Path.open(self.path / 'config.txt', 'w')
        file.write('Plagih GP. This config is not complete, TODO!')
        file.write('\n launched: {}'.format(self.datetime))
        file.write('\n kernel: {}'.format(self.kernel))
        file.write('\n precision: {}\n'.format(self.config['precision']))
        file.write('\n tree depth max: ' + str(self.config['tree_depth_max']))
        file.write('\n')
        file.write('\n tournament size: ' + str(self.config['gp_tourn_size']))
        file.write('\n population: ' + str(self.config['pop_max']))
        file.write('\n number of generations: ' + str(self.gen_id))
        file.write('\n\n')
        file.close()

    def file_pareto(self, pareto, path):
        """
        Save all the pareto efficient candidates to file
        """
        file = Path.open(path / 'pareto.txt', 'w')
        file.write('\nParsimony: \t<num> Fitness: \t<fitness> Expr: \t<expression>')

        for parsim, fit in sorted(list(pareto.items())):
            tree_meta = self.tree_hash_meta[self.parsimony_best_dict[parsim]]
            fitness = tree_meta['fitness_train']
            algo_sym = tree_meta['algo_sym']
            file.write('\nParsimony: \t{0} Fitness: \t{1} Expr: \t{2}'.format(str(parsim), str(fitness), str(algo_sym)))

        file.close()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Population specific                       |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_data_load_backup_population(self, population_backup_file):
        """
        Loads a saved population from an earlier run
        """
        with Path.open(population_backup_file, 'rb') as csv_file:
            target = csv.reader(csv_file, delimiter=',')
            n = 0  # track row count

            for row in target:

                n = n + 1
                if n == 1:
                    pass  # skip first empty row

                elif n == 2:
                    self.population_base = [row]  # write header to population_genepool

                else:
                    if not row:
                        self.tree = np.array([[]])  # initialise Tree array

                    else:
                        if self.tree.shape[1] == 0:
                            self.tree = np.append(self.tree, [row], axis=1)  # append first row to Tree

                        else:
                            self.tree = np.append(self.tree, [row], axis=0)  # append subsequent rows to Tree

                    if self.tree.shape[0] == T_num_lines:  # (current tree rows + row 0
                        self.population_base.append(self.tree)  # append complete Tree to population list

        self.printpl('i', 'We loaded the following population_genepool: {}'.format(self.population_base))
        return

    def pop_genepool_create(self, population):

        """
        Create the gene pool
        - Add a candidate if its parsimony is within the threshold


        """
        self.printpl('gggg', 'Gene Pool for Generation: {}...'.format(self.gen_id))
        dominator_count = 0
        gene_pool_hash_dict = {}

        for tree_id in range(1, len(population)):
            tree = population[tree_id]
            tree_ident, tree_meta = self.tree_store_meta_get_hash(tree)

            if tree_meta['parsimony'] < self.parsimony_min_max[1]:  # Tree -> gene_pool?

                gene_pool_hash_dict[tree_id] = tree_ident
                if self.fitness_compare(tree_meta['fitness_train'], self.origin['fitness_train']):
                    self.printpl('vvv', 'A candidate is fitter than the origin (might have occurred already)')
                    dominator_count += 1

        self.printpl('g', '{}: {} Candidates were better than the origin.'.format(self.gen_id, dominator_count))

        return gene_pool_hash_dict

    def pop_pareto_update(self):
        """
        Builds up the pareto front

        """

        # 1. Find lowest complexity
        best_fitness = self.tree_hash_meta[self.parsimony_best_dict[0]]['fitness_train']

        for parsim, ident in sorted(list(self.parsimony_best_dict.items())):
            fitness = self.tree_hash_meta[ident]['fitness_train']

            # kick
            if self.fitness_compare(best_fitness, fitness):
                self.pareto.pop(parsim, None)

            elif self.fitness_compare(fitness, best_fitness):
                if self.pareto.get(parsim):
                    if self.fitness_compare(fitness, self.pareto.get(parsim)):
                        self.pareto[parsim] = fitness
                        self.printpl('a', 'Updated pareto at {}. New fitness is: {}, old was: {}'.format(parsim, fitness, best_fitness))
                else:
                    self.pareto[parsim] = fitness
                    self.printpl('a', 'New pareto entry at {} with fitness: {}'.format(parsim, fitness))
                best_fitness = fitness

        return

    def fitness_compare(self, fitness1, fitness2, mode='better'):
        """
        Compares the fitness of two candidates according to the kernel

        Example:
            >
            fitness_compare
        """
        if (self.fitness_type == 'max' and fitness1 > fitness2) or \
                (self.fitness_type == 'min' and fitness1 < fitness2):
            return True
        elif fitness1 == fitness2:
            if mode == 'better_or_equal':
                return True
            elif mode == 'better':
                return False
            else:
                self.printpl('e', 'Mode not known')
        else:
            return False

    def pop_first_create(self):
        """
        Constructs the first generation
        - loads the origin-tree from file
        - constructs the first generation from this tree with branc. mutation
        """

        # TODO branch mutation in ALL subtrees? if more options are available
        # TODO safely create a complete generation?
        self.printpl('gg', 'First population...')
        self.time_last_monitor = self.time_start
        self.time_last_files = self.time_start
        tree_origin = self.origin['tree'].copy()
        origin_ids = tree_get_mutatable_list(tree_origin, no_root=True)

        for tree_id in range(1, self.config['pop_max'] + 1):
            # vary this tree with mutation
            branch_top = np.random.choice(origin_ids)
            branch_nodes_ids = tree_get_ids_karoo(tree_origin, branch_top)  # [6, 9, 10] select point of mutation and all nodes beneath
            tree = self.tree_insert_branch_random(tree_origin, branch_nodes_ids)  # tree with new branch

            # Fill the correct meta-data_csv_path into the tree (and wipe the old fitness)
            # tree = self.tree_store_meta_lastgen(tree, modification='i')  # wipe fitness data_csv_path
            tree = self.tree_modifyable_nodes_set(tree)
            tree[TR_ID][1] = tree_id

            self.popnew_append(tree, last_modification='first')

        self.printpl('ggg', 'We have constructed the first population of {} trees, saved to disk'.format(self.config['pop_max']))

    def pop_parsimony_best_update(self, gene_pool_hash_dict):
        """

        """
        # 1. Check every potential candidate
        for tree_id, tree_hash in gene_pool_hash_dict.items():
            tree_meta = self.tree_hash_meta[gene_pool_hash_dict[tree_id]]
            parsim = tree_meta['parsimony']
            fitness_train = tree_meta['fitness_train']

            # 3. is the tree better than the current best in this parsimony level?
            if parsim in self.parsimony_best_dict:
                cmp_fitness = self.tree_hash_meta[self.parsimony_best_dict[parsim]]['fitness_train']
                if self.fitness_compare(fitness_train, cmp_fitness, mode='better'):
                    self.parsimony_best_dict[parsim] = gene_pool_hash_dict[tree_id]
                    self.printpl('aa', 'Found a better candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                else:
                    return  # The "regular" case
            else:
                self.parsimony_best_dict[parsim] = gene_pool_hash_dict[tree_id]

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   What happens in a Generation              |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gen_prepare_parameters(self):
        """
        Sets the parameters for the generation
        - reset population_new
        - Linearly increase threshold for parsimony
        """

        self.time_genstart = time.perf_counter()
        self.debug_warnings = {}

        self.population_new = ['Plagih GP - Evolving Generation']  # initialise population_new to host the next generation
        full_parsimony_factor = 2  # working with the maximum parsimony for at least some generations
        gen_relation = min((full_parsimony_factor * self.gen_id) / self.config['gen_max'], 1)
        self.parsimony_min_max[0] = int(gen_relation * self.parsimony_min_max[1])

        return

    def gen_reproduce(self, repro_rate, tourn_size):

        """
        A single Tree from the prior generation is copied without mutation
        """

        for n in range(repro_rate):  # quantity of Trees to be copied without mutation
            tourn_winner = self.gp_selection_tournament(tourn_size)
            self.popnew_append(tourn_winner, last_modification='repro')  # i know, tests are not necessary...

        return

    def gen_mutate_point(self, repro_rate, tourn_size):

        """
        One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """

        for i in range(repro_rate):  # quantity of Trees to be generated through mutation
            tree = self.gp_selection_tournament(tourn_size)
            tree, node = self.treegp_mutate_point_evolve(tree, same_arity=True)

            self.popnew_append(tree, last_modification='point')

        return

    def gen_mutate_filter(self, repro_rate, tourn_size):
        """

        """

        for i in range(repro_rate):
            tree = self.gp_selection_tournament(tourn_size)
            try:
                self.debug_warnings['824 tree'] = tree
                new_tree = self.treegp_mutate_filter_one(tree)
                if len(new_tree) > 1:
                    self.popnew_append(new_tree, last_modification='filter')
            except Exception as ex:
                self.printpl('www', 'Tree in mutate filter could not be changed, {}'.format(ex))

        return

    def gen_mutate_branch(self, repro_rate, tourn_size):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """

        for i in range(repro_rate):  # quantity of Trees to be generated through mutation

            tourn_winner = self.gp_selection_tournament(tourn_size)  # perform tournament selection for each mutation
            node_ids = tree_get_mutatable_list(tourn_winner, no_root=True)
            node = np.random.choice(node_ids)
            branch_nodes_ids = tree_get_ids_karoo(tourn_winner, node)  # select point of mutation and all nodes beneath [6, 9, 10]
            tourn_winner = self.tree_insert_branch_random(tourn_winner, branch_nodes_ids)

            self.popnew_append(tourn_winner, last_modification='branch')

        return

    def gen_create_random(self, repro_rate, tourn_size):
        """
        TODO, this is currently only branch mutation

        """

        for i in range(repro_rate):  # quantity of Trees to be generated through mutation

            tourn_winner = self.gp_selection_tournament(tourn_size)  # perform tournament selection for each mutation
            node_ids = tree_get_mutatable_list(tourn_winner, no_root=True)
            node = np.random.choice(node_ids)
            branch_nodes_ids = tree_get_ids_karoo(tourn_winner, node)  # select point of mutation and all nodes beneath [6, 9, 10]
            tourn_winner = self.tree_insert_branch_random(tourn_winner, branch_nodes_ids)

            self.popnew_append(tourn_winner, last_modification='miss(br)')

        return

    def gen_crossover_point(self, repro_rate, tourn_size):
        """
        swap points of two trees
        """

    def gen_crossover_branch(self, repro_rate, tourn_size):
        """
        swap branches of two trees
        - select parent a and b
        - select swappable branche for a_parent from b_parent
            - select a node in a (and crossover here, no matter what)
        - delete a_parent branch and insert b_parent branch (which tactic?)

        """

        half_rate = int(repro_rate / 2)
        for n in range(half_rate):

            # 1. two parents
            left_tree = self.gp_selection_tournament(tourn_size)  # perform tournament selection for 'a_parent'
            right_tree = self.gp_selection_tournament(tourn_size)  # perform tournament selection for 'b_parent'

            force_convert = False

            # 2. search nodes for left and right that can be exchanged. convert_needed
            left_id, right_id, success = self.tree_try_get_swapids(left_tree, right_tree)
            if not success:
                right_id, left_id, success = self.tree_try_get_swapids(right_tree, left_tree)
            if not success:
                force_convert = True

            left_ids, left_labels, left_aritys = tree_get_branchinfo(left_tree, left_id)
            right_ids, right_labels, right_aritys = tree_get_branchinfo(right_tree, right_id)

            if force_convert:
                self.printpl('w', 'Crossover conversion between trees forced. \n{}\n{}'.format(left_tree, right_tree))
                left_xtype = self.xtype_get(left_tree[N_label][left_id])
                conv_to_left, conv_to_right = xtype_get_converters(left_xtype)
                right_labels.insert(0, conv_to_left)
                right_aritys.insert(0, 1)
                left_labels.insert(0, conv_to_right)
                left_aritys.insert(0, 1)

            left_core = core_from_labels(left_labels, left_aritys)
            right_core = core_from_labels(right_labels, right_aritys)

            left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
            left_offspring = self.treegp_crossover_tree_prune(left_offspring, self.config['tree_depth_max'])
            self.popnew_append(left_offspring, last_modification='cross')

            right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
            right_offspring = self.treegp_crossover_tree_prune(right_offspring, self.config['tree_depth_max'])
            self.popnew_append(right_offspring, last_modification='cross')
            # right_offspring = self.treegp_crossover_insert(right_tree, right_ids, left_tree, left_ids, convert_needed=convert_needed)
        return

    def popnew_append(self, tree, last_modification=''):
        """
        some stuff to definitely do before actually appending to pop
        """
        tree = tree_round_constants(tree, self.config['float_accuracy'], karoo=True)
        tree = self.tree_modifyable_nodes_set(tree)
        tree[TR_type][1] = last_modification
        if not tree_test_check_children(tree):
            self.printpl('e', 'Tree is not consistent:\n{}'.format(tree))
        self.population_new.append(tree)

    def gen_finalize(self):

        """
        From raw population_new to new population_genepool
        - Gene_pool with tree's parsimony (and store info in the tree)
        -

        """

        self.population_new = pop_enum_trees(self.population_new)  # pop +tree_id
        gene_pool_hash_dict = self.pop_genepool_create(self.population_new)
        self.monitor_genepool(gene_pool_hash_dict)

        self.pop_parsimony_best_update(gene_pool_hash_dict)
        self.pop_pareto_update()

        self.population_base = pop_copy_genepool(self.population_new, gene_pool_hash_dict, self.gen_id)
        self.file_population_write(self.population_new, 'new', self.path)

        self.monitoring_dict['total_found_trees'][self.gen_id] = len(self.tree_hash_meta)
        self.printpl('gg', 'Monitoring: Created {}/{} unique trees in generation {}. Gen-time: {:4.2f}'.format(
            len(set(gene_pool_hash_dict.values())),
            self.config['pop_max'],
            self.gen_id,
            time.perf_counter()-self.time_genstart))

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Perform the 3 genetic prog. operations    |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gp_selection_tournament(self, tourn_size):

        """
        config-selection. takes a number of trees (we use 3) and returns the best one (winner)
        Uses:
            self.population a
            self.genepool a
        """

        # Start with dummies
        best_id = -1
        best_fitness = self.fitness_bad_dummy

        # Get several values
        for n in range(tourn_size):

            tree_id = pop_util_random(self.population_base)

            fitness = float(self.population_base[tree_id][T_fitness][1])  # extract the fitness from the array
            fitness = round(fitness, self.config['precision'])  # force 'result' and 'solution' to the same number of floating points

            if self.fitness_compare(fitness, best_fitness, mode='better'):
                best_id = tree_id
                best_fitness = fitness

        tourn_winner = util_tree_copy(self.population_base, best_id)

        return tourn_winner

    def gp_mutate_constantfilter(self, constant, term_type=None, filter_type='gaussian_filter'):
        """
        When this happens, constants get a a small variance
        """

        if term_type == 'float':
            if filter_type == 'gaussian_filter':
                constant = np.random.normal(constant, 0.1)
            else:
                self.printpl('w', 'Warning: Filter  not specified. Please specify a filter_type.')
                constant = np.random.normal(constant, 0.1)

        if term_type == 'int':
            constant = int(np.random.normal(constant, 2))

        if term_type == 'bool':
            constant = not constant
            # random by 50:50?

        return constant

    def treegp_mutate_point_evolve(self, tree, same_arity=True):

        """
        Mutate a single mutatable point in any Tree.
        """

        # 1. choose a node
        node_id = np.random.choice(tree_get_mutatable_list(tree))
        arity = int(tree[N_arity][node_id])
        xtype = self.xtype_get(tree[N_label][node_id])  # '>' -> 'f2b'

        if same_arity:
            # 2. perform point mutation on that specific node
            if arity > 0:
                tree[N_label][node_id] = self.xtype_choose_func_pointmutation(xtype=xtype, arity=arity)  # Function is same type, same arity
            elif arity == 0:  # aka a terminal
                tree[N_label][node_id] = self.xtype_choose_term(xtype)  # 3 -> '2f' -> 5
            else:
                self.printpl('e', 'Arity not as expected: {}'.format(tree[N_arity][node_id]))
                raise
        elif arity == 'plagih_switcharoo':
            self.printpl('e', 'SFEH this is TODO')
        else:
            self.printpl('e', 'treegp_mutate_point_evolve dies not know this method to handle the arity: {}'.format(arity))

        return tree, node_id  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping

    def treegp_mutate_filter_one(self, tree):
        """
        Mutates one float terminal of a tree
        """
        # 1. choose a node
        node_ids = tree_get_mutatable_list(tree)
        float_nodes = []
        for node_id in node_ids:
            label = tree_get_label(tree, node_id)
            if xtype_get_constant(label) == '2f':
                float_nodes.append(node_id)
        if float_nodes:
            float_id = np.random.choice(float_nodes)
            val = float(tree_get_label(tree, float_id))
            new_value = self.gp_mutate_constantfilter(val, term_type='float', filter_type='gaussian_filter')
            tree[N_label][float_id] = str(new_value)
            return tree
        else:
            raise Exception('No mutatable node found!')
            # return None

    def treegp_crossover_tree_prune(self, tree, depth):
        """
        reduces the depth of a Tree (in case it is too deep).
        Arguments required: tree, depth
        """

        nodes = []

        for n in range(1, len(tree[3])):

            if int(tree[N_depth][n]) == depth and int(tree[N_arity][n]) > 0:
                tree[N_arity][n] = 0
                node_xtype = self.xtype_get(tree[N_label][n])
                tree[N_label][n] = self.xtype_choose_term(node_xtype)  # replace label

            elif int(tree[N_depth][n]) > depth:  # record nodes deeper than the maximum allowed Tree depth
                nodes.append(n)

        tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
        tree = evolve_node_arity_fix(tree)  # fix all node arities

        return tree

    def treegp_crossover_get_partner_node_id(self, function_label, partner_tree, partner_branch_id, mode='same_type'):
        """
        -> Crossover: Returns a node_id in the partner tree, that can be swapped
        """

        node_xtype = self.xtype_get(function_label)
        node_options = []

        if mode == 'same_type':  # only return a node with the same function type
            for i, label in enumerate(partner_tree[N_label][1:]):
                if self.xtype_get(label) == node_xtype:
                    node_options.append(i + 1)  # +1, we skipped the first element

            if node_options:
                np.random.shuffle(node_options)  # otherwise, the first closest element is always taken (-> smallest)
                return min(node_options, key=lambda x: abs(x - partner_branch_id))  # return closest node
            else:
                return 0  # No matching node found :(
        elif mode == 'random':
            self.printpl('e', 'mode: Do the same as in the upper function, but choose randomly?')
        else:
            self.printpl('e', 'Mode not found {}'.format(mode))
            raise

    def tree_try_get_swapids(self, a_tree, b_tree):
        """
        Returns two branches (node ids) that can be replaced and a converter (if needed)

        # same position & node, same node, same type, reversed type | convert_needed type
        """

        # choose a node from parent a
        a_ids = tree_get_mutatable_list(a_tree, no_root=True)
        a_id = np.random.choice(a_ids)
        a_xtype = self.xtype_get(a_tree[N_label][a_id])

        # create a list from parent b with same xtype
        b_ids = tree_get_mutatable_list(b_tree, no_root=True)
        b_sametype_ids = b_ids[:]
        for i in b_ids:
            b_xtype = self.xtype_get(b_tree[N_label][i])
            if not xtype_equi_outcome(b_xtype, a_xtype):
                b_sametype_ids.remove(i)

        if b_sametype_ids:  # if it has entries, choose one. we are done
            b_id = np.random.choice(b_sametype_ids)
            success = True
        else:
            b_id = np.random.choice(b_ids)
            success = False
        aaa = self.xtype_get(a_tree[N_label][a_id])
        bbb = self.xtype_get(b_tree[N_label][b_id])
        if not xtype_equi_outcome(aaa, bbb):
            self.printpl('vv', 'sfeh dummy. This might happen sometimes. {}, {}\n{}\n{}'.format(aaa, bbb, a_tree[N_label], b_tree[N_label]))
            # raise

        return a_id, b_id, success

    # def treegp_crossover_insert(self, left_parent, left_ids, right_parent, right_ids):
    #
    #     """
    #     Perform a crossover between nodes that are crossoverable in terms of function types
    #     get: parent x, y and their branches
    #     return: puts right_ids into parent_x
    #     """
    #
    #     right_top_id = int(right_ids[0])
    #     left_top_id = int(left_ids[0])
    #
    #     r_labels, r_aritys = tree_branch_get_label_list(right_parent, right_ids, karoo=True)
    #
    #     if len(right_ids) == 1:
    #         # if branch of new parent contains only one node (terminal)
    #         # Remember: if a conversion was needed, a terminal would now have a function in front of it!
    #
    #         new_label = right_parent[N_label][right_top_id]
    #         left_parent[N_label][left_top_id] = new_label  # replace label with that of a particular node in 'right_ids'
    #         left_parent[N_arity][left_top_id] = 0  # set terminal arity
    #
    #         left_parent = np.delete(left_parent, left_ids[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
    #         left_parent = tree_fix_link_child_karoo(left_parent)  # fix all child links
    #         left_parent = evolve_node_renum_karoo(left_parent)  # renumber all 'node_id's
    #     else:
    #
    #         right_core = core_from_labels(r_labels, r_aritys)
    #         left_parent = tree_insert_subtree(left_parent, right_core, left_ids, karoo=True)
    #         left_parent = self.treegp_crossover_tree_prune(left_parent, self.config['tree_depth_max'])  # sfeh: not sure if this is necessary?
    #
    #     return left_parent

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Utility  functions to evolve a tree       |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def invent_label_list(self, xtype, depth_goal):
        """
        build a random, but within itself consistent label list
        Also, return the arities aswell (they are searched anyways)
        """
        todo_xtypes = [xtype]
        result_label_list = []
        result_arity_list = []

        # Build a list with labels in row, and a list with their arities
        for depth in range(0, depth_goal):
            next_xtype_list = []

            if depth == depth_goal - 1:  # now, we are on the lowest level.

                for t in todo_xtypes:  # Build terminals now.
                    label = self.xtype_choose_term(t)
                    arity = 0

                    # Add the label to the result list
                    result_label_list.append(label)
                    result_arity_list.append(arity)

            else:
                for t in todo_xtypes:

                    # Randomly choose a new label

                    if insert_function_or_term(depth, depth_goal) == 'terminal':
                        label = self.xtype_choose_term(t)
                        arity = 0
                    else:
                        label, arity = self.xtype_choose_func_branchmutate(xtype=t, arity=False)

                    # xtype-'To-do' list for the next depth to give values to these functions
                    if label == 'Ifte':
                        next_xtype_list.extend(['2b', '2f', '2f'])
                    else:
                        tmp_xtype = self.xtype_get(label)
                        child_type = tmp_xtype[:2][::-1]  # the input of our function "reverted" is the xtype
                        for _ in range(0, arity):  # when arity==2, add 2 times
                            next_xtype_list.append(child_type)

                    # Add the label to the result list
                    result_label_list.append(label)
                    result_arity_list.append(arity)

            # Finally, update the list for the next round
            todo_xtypes = next_xtype_list[:]

        return result_label_list, result_arity_list

    def tree_insert_branch_random(self, tree, branch_ids):

        """
        replaces the branch_ids in a tree with a new branch
        Given: Tree and a list of node ids
        - checks how far to build down
        - checks the old nodes xtype, etc.
        - checks if we are not too far down the tree
        -

        returns: new tree
        """

        grow_method = self.config['tree_growth']

        if grow_method == 'depth_base_uniform':
            """
            We allow base depth (which is a little lower than max)
            but every node has 0.5 chance to become a terminal
            - iterate over depths
            - fill with as many funcs as possible

            """
            # calculate depth restriction
            depth_upper_bound = self.config['tree_depth_max'] - int(tree[N_depth][branch_ids[0]])
            depth_goal = min(self.config['tree_depth_base'], depth_upper_bound)

            # Get information about the top-node we have to replace
            old_label = tree[N_label][branch_ids[0]]
            old_xtype = self.xtype_get(old_label)

            # Build a new tree
            label_list, arity_list = self.invent_label_list(old_xtype, depth_goal)  # Build a complete tree

            if not label_list:
                self.printpl('ww', 'Wanted to branch-mutate a node that is on the lowest level')
                return tree

            core_insert = core_from_labels(label_list, arity_list)
            result_tree = tree_insert_subtree(tree, core_insert, branch_ids, karoo=True)

            return result_tree
        elif grow_method == 'old plagih code':
            self.printpl('e', 'Not yet')
        elif grow_method == 'nodes_max_uniform':
            """
            We allow a certain amount of new nodes instead tree depth.
            This could be calculated respectively to the parsimony level
            which the tree might have up his sleeve
            """
            num_new_nodes = np.random.randint(10, 30)
            # max_

        else:
            self.printpl('e', 'That did not work')

        return tree

    def evolve_subtree_insert_child(self, tree, node, c_buffer):

        """
        Insert child node into the copy of a parent Tree.

        """

        if int(tree[N_arity][node]) == 0:  # if arity = 0
            self.printpl('e', 'In evolve_child_insert: node {} has arity 0'.format(node))
            # self.plagih_pause()  # consider special instructions for this

        elif int(tree[N_arity][node]) == 1:  # if arity = 1
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[N_depth][c_buffer] = int(tree[N_depth][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

        elif int(tree[N_arity][node]) == 2:  # if arity = 2
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[N_depth][c_buffer] = int(tree[N_depth][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

            tree = np.insert(tree, c_buffer + 1, '', axis=1)  # insert node for 'node_c2'
            tree[3][c_buffer + 1] = c_buffer + 1  # node ID
            tree[N_depth][c_buffer + 1] = int(tree[N_depth][node]) + 1  # node_depth
            tree[7][c_buffer + 1] = int(tree[3][node])  # parent ID

        elif int(tree[N_arity][node]) == 3:  # if arity = 3
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[N_depth][c_buffer] = int(tree[N_depth][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

            tree = np.insert(tree, c_buffer + 1, '', axis=1)  # insert node for 'node_c2'
            tree[3][c_buffer + 1] = c_buffer + 1  # node ID
            tree[N_depth][c_buffer + 1] = int(tree[N_depth][node]) + 1  # node_depth
            tree[7][c_buffer + 1] = int(tree[3][node])  # parent ID

            tree = np.insert(tree, c_buffer + 2, '', axis=1)  # insert node for 'node_c3'
            tree[3][c_buffer + 2] = c_buffer + 2  # node ID
            tree[N_depth][c_buffer + 2] = int(tree[N_depth][node]) + 1  # node_depth
            tree[7][c_buffer + 2] = int(tree[3][node])  # parent ID

        else:
            self.printpl('e', 'In evolve_child_insert: node {} arity > 3'.format(node))
            # self.plagih_pause()  # consider special instructions for this (pause)

        return tree

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def load_origin_tree(self, origin_tree_file_path=None, label_list=None, permanent_list=None):
        """
        This loads the 'origin' and evaluates it
        Two loading options:
            - path to csv with tree (outdated)
            - an array with labels ['+','1','observation0']. optional, the permanent nodes as separate array

        returns: tree
        """

        # Check if the user provided an origin
        if origin_tree_file_path:

            # Load origin from file
            with Path.open(origin_tree_file_path, 'r') as csv_file:
                target = csv.reader(csv_file, delimiter=',')
                tree = np.array([[]])
                for row in target:
                    if tree.shape[1] == 0:  # looks if tree is empty
                        tree = np.append(tree, [row], axis=1)  # append first row to Tree ('tree_id')
                    else:
                        tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree
                if tree.shape[0] == T_num_lines:  # (+ row 0)
                    pass  # print('Origin Tree is: \n' + str(tree))
                else:
                    self.printpl('e', "Tree could not be imported correctly from .csv file.")
                    raise
        elif label_list:
            tree = karoo_tree_from_user(label_list, permanent_list)
        else:
            self.printpl('w', 'No origin provided. Todo. starting from scratch with random generation?')
            raise

        origin_algo_raw = self.tree_expr_raw(tree, P_first_node)
        self.origin = {'tree': tree,
                       'algo_raw': origin_algo_raw,
                       'algo_sym': self.tree_expr_sympify(algo_raw_str=origin_algo_raw),
                       'parsimony': 0}

        origin_hash, origin_meta = self.tree_store_meta_get_hash(tree)
        self.origin['fitness_train'] = origin_meta['fitness_train']

        self.parsimony_best_dict[0] = origin_hash

        self.hashtable_fitness_train = {}

        self.printpl('g', 'Loading origin. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        return

    def tree_store_meta_get_hash(self, tree, store_in_tree=True):
        """
        gets all the main tree information
        1. algo_raw
        2. tree_identifier (algo_raw)
        2. algo_sym
        3. parsimony
        4. fitness_train
        """
        # 1. get algo_raw - what is needed to compute the tree identifier
        algo_raw_str = self.tree_expr_raw(tree, 1)
        tree_ident = hash(algo_raw_str)  # sfeh: potential for improvement- use algo_sym in separate dict as identifier.

        # 2.1 Did we have this tree already? -> Nice, we have everything
        if tree_ident in self.tree_hash_meta:
            tree_meta = self.tree_hash_meta[tree_ident]

        # 2.2 New tree, but still Skip fitness eval for complex trees
        else:
            parsimony = self.tree_parsimony(tree)

            # 3. compute fitness
            if parsimony < self.parsimony_min_max[1]:
                # 3.1 With tensorflow
                algo_sym = self.tree_expr_sympify(algo_raw_str=str(algo_raw_str))
                fitness_train = self.eval_tf(algo_sym, self.data_train)['fitness']

            else:
                # 3.2 just fill with bad values
                algo_sym = sympy_dummy
                fitness_train = self.fitness_bad_dummy

            # 4. All the tree-specific meta into dict
            tree_meta = {'algo_raw': str(algo_raw_str), 'tree_ident': tree_ident, 'algo_sym': str(algo_sym), 'parsimony': float(parsimony), 'fitness_train': float(fitness_train)}
            self.tree_hash_meta[tree_ident] = tree_meta

        # 5. store fitness in 'old' Karoo tree structure
        if store_in_tree:
            self.tree_store_parsimony(tree, tree_meta['parsimony'])
            self.tree_store_fitness(tree, tree_meta['fitness_train'])
            # self.tree_store_meta_lastgen(tree)

            return tree_ident, tree_meta

    def tree_modifyable_nodes_set(self, chosen_tree):
        """
        Sets all the origin core nodes back to non-modifyable
        """
        # Set all nodes to be modifiable (=1)
        for i, tmp in enumerate(chosen_tree[N_modify][1:]):
            chosen_tree[N_modify][i + 1] = '1'

        # Find no-modifyables in Origin
        non_modifiable_nodes = []
        if self.origin['tree'][N_modify][1] == '0':  # check is modifiable nodes are specified
            non_modifiable_nodes.extend(self.tree_permanent_nodes_get(1, chosen_tree, 1))

        for non_modifiable in non_modifiable_nodes:
            chosen_tree[N_modify][non_modifiable] = '0'

        return chosen_tree

    def tree_permanent_nodes_get(self, origin_node, chosen_tree, chosen_node):
        """
        Returns a list of nodes that are not supposed to be modified
        """

        if self.origin['tree'][N_modify][origin_node] == '0':
            permanent_nodes = [int(chosen_tree[N_id][chosen_node])]
            for child in [N_c1, N_c2, N_c3]:
                if self.origin['tree'][child][origin_node] != '':
                    next_origin_node = int(self.origin['tree'][child][origin_node])
                    next_chosen_node = int(chosen_tree[child][chosen_node])
                    tmp = self.tree_permanent_nodes_get(next_origin_node, chosen_tree, next_chosen_node)
                    if tmp is not None:
                        permanent_nodes.extend(tmp)
            return permanent_nodes
        else:
            return

    def tree_build_type_constant_get(self, term_type='', mode='float-1to1', uniform_range=''):
        """

        Returns a constant that fits into the position
        -- term_type = 'float'
        """
        if uniform_range:
            return np.random.uniform(uniform_range[0], uniform_range[1])

        if term_type == 'bool':
            const = np.random.choice([True, False])
        elif term_type == 'float':
            if mode == 'float-1to1':
                const = np.random.uniform(-1, 1)
            elif mode == 'intTotal_10':
                const = np.random.random_integers(-10, 10)
            elif mode == 'random_optimised':
                const = np.random.choice([-10, -5, -2, -1, -1, -0.8, -0.6, -0.5, -0.4, -0.2, 0, 10,
                                          5, 2, 1, 1, 0.8, 0.6, 0.5, 0.4, 0.2, 0])
            else:
                # sfeh: gibt viele Verteilungen: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html
                self.printpl('e', 'You did not take care of the kind of numbers you want to have')
                raise
        elif term_type == 'int':
            # TODO give more opportunities, similar to random floats
            const = np.random.random_integers(-10, 10)
        else:
            self.printpl('w', 'Please specify your desired datatype if possible. Trying to return c1 similar to terminals.')
            self.printpl('e', 'This term type should not occur, I guess {}'.format(term_type))
            # term_type = np.random.choice(self.variables_dict['types'])
            const = self.tree_build_type_constant_get(term_type=term_type)
        return str(const)

    def tree_expr_sympify(self, algo_raw_str=None, tree=None):

        """
        returns the sympifyed expression
        """
        if tree:  # If we got a tree, we generate the expression
            algo_raw_str = str(self.tree_expr_raw(tree, 1))

        try:
            x = plagih_sympify(algo_raw_str)
            strx = str(x)

            if 'zoo' in strx:
                self.printpl('ww', 'zoo in expression? sfeh: not anymore... {}'.format(algo_raw_str))
                x = re.sub('zoo', '10', strx)

            if 'inf' in strx:
                self.printpl('ww', 'Inf in expression? {}'.format(algo_raw_str))
                x = re.sub('inf', '10', strx)

            if 'nan' in strx:  # Happens when 0/0 occurs. This tree is worth nothing anyways
                self.printpl('ww', 'We had a "nan" in {}'.format(algo_raw_str))
                self.remove_this_tree()
                return str(sympy_dummy)
            else:
                return str(x)
        except Exception:
            self.printpl('w', 'In sympify. Caused by this raw algorithm: ' + str(algo_raw_str))
            # todo.
            self.remove_this_tree()
            return str(sympy_dummy)

    def remove_this_tree(self):
        self.printpl('ww', 'This still is a todo')
        """
        If a tree makes problems, delete it somehow.
        - set parsimony very high?
        todo
        """

    def tree_expr_raw(self, tree, node_id):

        """
        Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').

        """
        node_id = int(node_id)

        if tree[N_arity, node_id] == '0':  # arity of 0 for the pattern '[term]'
            return '(' + tree[N_label, node_id] + ')'  # 'node_label' (function or terminal)

        elif tree[N_arity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
            return '(' + self.tree_expr_raw(tree, tree[9, node_id]) + tree[N_label, node_id] + ')'

        elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
            # This if case is for 2-ary ops that is prefix. like Min(a, b)
            if tree[N_label, node_id] not in functions_infix_dict:
                return '(' + tree[N_label, node_id] + '(' + self.tree_expr_raw(tree, tree[9, node_id]) + ', ' + self.tree_expr_raw(tree, tree[10, node_id]) + '))'
            else:
                return '(' + self.tree_expr_raw(tree, tree[9, node_id]) + tree[N_label, node_id] + self.tree_expr_raw(tree, tree[10, node_id]) + ')'  # Klammern, da sympify sonst abkacnen könnte

        elif tree[N_arity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
            return '(Ifte(' + self.tree_expr_raw(tree, tree[9, node_id]) + ', ' + self.tree_expr_raw(tree, tree[10, node_id]) + ', ' + self.tree_expr_raw(tree, tree[11, node_id]) + '))'

    def tree_raw_depth_prefix(self, tree, node_id):

        """
        Does the same as tree_expr_raw, but evaluates infix functions in prefix notation (functional form)

        """

        node_id = int(node_id)

        if tree[N_arity, node_id] == '0':  # arity of 0 for the pattern '[term]'
            return '{' + tree[N_label, node_id] + '}'  # 'node_label' (function or terminal)

        elif tree[N_arity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
            return '{' + tree[N_label, node_id] + self.tree_raw_depth_prefix(tree, tree[9, node_id]) + '}'

        elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
            return '{' + tree[N_label, node_id] + '' + self.tree_raw_depth_prefix(tree, tree[9, node_id]) + self.tree_raw_depth_prefix(tree, tree[10, node_id]) + '' + '}'

        elif tree[N_arity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
            return '{Ifte' + self.tree_raw_depth_prefix(tree, tree[9, node_id]) + self.tree_raw_depth_prefix(tree, tree[10, node_id]) + self.tree_raw_depth_prefix(tree, tree[11, node_id]) + '' + '}'

    def tree_parsimony(self, tree, parsimony_distance='ted'):
        """
        parsimony_distance: compute the chosen distance by the user.

        """
        if parsimony_distance == 'ted':
            return self.tree_parsimony_ted(self.origin['tree'], tree)
        elif parsimony_distance == 'total_count_nodes':
            return int(tree[3][-1:])  # returns the tree size
        elif parsimony_distance == 'total_tree_depth':
            return tree[N_depth][1]  # returns the tree size
        elif parsimony_distance == 'total_karoo_original':  # do not use with long variable names
            algo_raw_str = str(self.tree_expr_raw(tree, 1))
            return len(str(algo_raw_str))
        # elif parsimony_distance == 'total_simplified':
        #     algo_sym = self.tree_expr_sympify(tree=tree)
        #     return count_ops(algo_sym)
        elif parsimony_distance == 'rel_ari_1':  # Does this work?
            return self.tree_parsimony_relari(tree)
        else:
            self.printpl('i', 'Parsimony distance not specified! Use default.')
            self.tree_parsimony(tree)

    def tree_parsimony_relari(self, tree):
        """
        This distance penalizes non-original functions with its arity
        - ignore node[0] [description]
        - look within the subtree if the original function is on origin spot
        """

        # If the new tree is actually less complex than the original one, just return 1
        if len(tree[N_label]) < len(self.origin['tree'][N_label]):
            return 1

        distance = 0

        # iterate over every node in the new tree
        for i, arity in enumerate(tree[N_arity]):
            if i == 0:  # skip node 0. the description
                continue
            elif i < len(self.origin['tree'][N_label]):  # Make sure we stay within the tree index. <= does not work
                if self.origin['tree'][N_label][i] != tree[N_label][i]:  # is it different from the origin?
                    distance = distance + int(arity)  # add the nodes arity. double-punishes large trees
            else:
                distance = distance + int(arity)

        return max(distance, 1)  # make sure, it does not return 0

    def tree_parsimony_ted(self, tree1, tree2):
        """
        The Tree Edit distance (TED) ('coolest' distance)
        - the amount of changes that have to be applied to the origin to equality are counted
        """
        # TODO TED soll geänderte Werte ignorieren
        apted_tree1 = self.tree_raw_depth_prefix(tree1, 1)
        apted_tree2 = self.tree_raw_depth_prefix(tree2, 1)
        distance, mapping = apted_distance(apted_tree1, apted_tree2)
        # sfeh the mapping could be handy somewhere
        return distance

    def tree_store_fitness(self, tree, fitness):

        """
        Store the fitness within the tree np-array

        """

        fitness = float(fitness)
        fitness = round(fitness, self.config['precision'])

        tree[T_fitness][1] = fitness  # store the fitness with each tree

        return

    def tree_store_parsimony(self, tree, parsimony):
        """
        Store the parsimony within the tree np-array
        """
        if parsimony < 0:
            self.printpl('w', 'Parsimony is: {}'.format(parsimony))
        tree[T_parsimony][1] = parsimony

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to use evaluate (tensorflow)      |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def eval_tf(self, expr, data, get_pred_labels=False):

        """
        computes config-tree results and fitness scores.
        - Computes tree expression using TensorFlow (TF)
        - parsing input string 'expression' and converting it into a TF operation graph
        - processing tf graph in an isolated TF session (results and corresponding fitness)

            'self.tf_device' - controls which device will be used for computations (CPU or GPU).
            'self.tf_device_log' - controls device placement logging (debug only).

        Args:
            'expr' - a string expression to be computed on the data_csv_path. Variable -> 'self.terminals'
            'data_csv_path' - an 'n by m' matrix of the data_csv_path points containing n observations like 'self.terminals'.
            'get_pred_labels' - (Classify Kernel) a boolean flag which controls whether the predicted labels should be
            extracted from the evolved results.

        Returns:
            A dict mapping keys to the following outputs:
                'result'            - array of the results of applying given expression to the data_csv_path
                'pred_labels'       - (Classify) an array of the predicted labels extracted from the results
                'solution'          - array of the solution values extracted from the data_csv_path (variable 's' in the dataset)
                'pairwise_fitness'  - array of the element-wise results of applying the fitness kernel function
                'fitness'           - aggregated scalar fitness score

        """

        # Initialize TensorFlow session
        tf.compat.v1.reset_default_graph()  # tf.reset_default_graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=self.tf_device_log, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            with sess.graph.device(self.tf_device):

                # 1. data_csv_path (observations, actions) to tensors
                tensors = {}

                num_terminals = len(self.variables_dict['all'])

                for i in range(num_terminals):
                    var = self.variables_dict['all'][i]
                    if '2f' in self.xtype_get(var, node_arity=0):
                        tensors[var] = tf.constant(data[:, i], dtype=tf.float32)  # converts data_csv_path into vectors
                    else:  # '2b'
                        tensors[var] = tf.constant(data[:, i], dtype=tf.bool)

                # for i in range(num_actions):
                #     self.action_dict
                #     action_xtype = self.xtype_get(var, node_arity=0)
                #     if '2f' in action_xtype:
                #         tensors[var] = tf.constant(data[:, num_terminals + i], dtype=tf.float32)  # converts data_csv_path into vectors

                for i, action in enumerate(self.action_dict):
                    py_type = self.action_dict[action]
                    if 'float' in py_type:
                        tensors['action' + str(i)] = tf.constant(data[:, num_terminals + i], dtype=tf.float32)  # converts data_csv_path into vectors
                    # elif '2b' in action_xtype:  # '2b'
                    #     self.printpl('e', 'Currently no kernel available for boolean fitness')
                    #     tensors[var] = tf.constant(data[:, i], dtype=tf.bool)
                    else:
                        self.printpl('e', 'action_dict type for {} is: {}.'.format(action, py_type))

                # 2- Transform string expression into TF operation graph
                tf_result = self.eval_tf_ast_expr(expr, tensors)
                pred_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

                # TODO currently does only support one label
                solution = tensors['action0']  # solution c1 is assumed to be stored in this terminal
                # 3- Add fitness computation into TF graph
                if self.kernel == 'classification':  # CLASSIFY kernel

                    """
                    Creates element-wise fitness computation TensorFlow (TF) sub-graph for CLASSIFY kernel.
                    - tree-label vs. true label
                    This multiclass classifer compares each row of a given Tree to the known solution, comparing predicted labels 
                    generated by plagih GP against the true class labels. This method is able to work with any number of class 
                    labels, from 2 to n. The left-most bin includes -inf. The right-most bin includes +inf. Those inbetween are 
                    by default confined to the spacing of 1.0 each, as defined by:

                        (solution - 1) < result <= solution

                    The skew adjusts the boundaries of the bins such that they fall on both the negative and positive sides of the 
                    origin. At the time of this writing, an odd number of class labels will generate an extra bin on the positive 
                    side of origin as it has not yet been determined the effect of enabling the middle bin to include both a 
                    negative and positive result.
                    """

                    if len(self.action_dict) > 1:
                        self.printpl('e', 'TODO multidimensional input. To be done, there is no solution yet.')

                    if get_pred_labels:
                        pred_labels = tf.map_fn(self.eval_tf_classify_labels_map, tf_result, dtype=(tf.int32, tf.string), swap_memory=True)

                    skew = (self.class_labels / 2) - 1

                    rule11 = tf.equal(solution, 0)
                    rule12 = tf.less_equal(tf_result, 0 - skew)
                    rule13 = tf.logical_and(rule11, rule12)

                    rule21 = tf.equal(solution, self.class_labels - 1)
                    rule22 = tf.greater(tf_result, solution - 1 - skew)
                    rule23 = tf.logical_and(rule21, rule22)

                    rule31 = tf.less(solution - 1 - skew, tf_result)
                    rule32 = tf.less_equal(tf_result, solution - skew)
                    rule33 = tf.logical_and(rule31, rule32)

                    pairwise_fitness = tf.dtypes.cast(tf.logical_or(tf.logical_or(rule13, rule23), rule33), tf.int32)

                elif self.kernel == 'regression':  # REGRESSION kernel

                    """
                    A very, very basic REGRESSION kernel which is not designed to perform well in the real world. It requires
                    that you raise the minimum node count to keep it from converging on the c1 of '1'. Consider writing or 
                    integrating a more sophisticated kernel.
                    """

                    pairwise_fitness = tf.abs(solution - tf_result)

                elif self.kernel == 'match':  # MATCH kernel

                    """
                    This is used for demonstration purposes only.
                    """

                    # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
                    RTOL, ATOL = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
                    pairwise_fitness = tf.dtypes.cast(tf.less_equal(tf.abs(solution - tf_result), ATOL + RTOL * tf.abs(tf_result)), tf.int32)

                # elif self.kernel == '[other]': # use others as a template

                else:
                    raise Exception('Kernel type is wrong or missing. You entered {}'.format(self.kernel))

                fitness = tf.reduce_sum(pairwise_fitness)

                # Process TF graph and collect the results
                tf_result, pred_labels, solution, fitness, pairwise_fitness = sess.run([tf_result, pred_labels, solution, fitness, pairwise_fitness])

        return {'result': tf_result, 'pred_labels': pred_labels, 'solution': solution, 'fitness': float(fitness),  # this was changed
                'pairwise_fitness': pairwise_fitness, 'old_fitness': float(fitness)}

    def eval_tf_ast_expr(self, expr, tensors):

        """
        Extract expression tree from the string algo_sym and transform into TensorFlow (TF) graph.

        """
        # print('Current expr:', expr)  # importantprint for debugging failed expressions
        tree = ast.parse(expr, mode='eval').body

        # TODO diesen try-except block entfernen
        self.debug_warnings['expr'] = str(expr)
        try:
            return self.eval_tf_expr_graph(tree, tensors)
        except:
            return self.fitness_dummy_get()

    def eval_tf_expr_graph(self, node, tensors):

        """
        Recursively transforms parsed expression tree into TensorFlow (TF) graph.

        """

        if isinstance(node, ast.Name):  # <tensor_name>
            return tensors[node.id]

        elif isinstance(node, ast.Num):  # <number>
            shape = tensors[list(tensors.keys())[0]].get_shape()
            return tf.constant(node.n, shape=shape, dtype=tf.float32)

        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>, e.g., x + y
            return ast_tensor_dict[type(node.op)](
                self.eval_tf_expr_graph(node.left, tensors),
                self.eval_tf_expr_graph(node.right, tensors))

        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            return ast_tensor_dict[type(node.op)](
                self.eval_tf_expr_graph(node.operand, tensors))

        elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)
            # special case: If-then-else
            if node.func.id == 'Ifte':
                return ast_tensor_dict[node.func.id](tf.dtypes.cast(
                    self.eval_tf_expr_graph(node.args[0], tensors), tf.bool),
                    self.eval_tf_expr_graph(node.args[1], tensors),
                    self.eval_tf_expr_graph(node.args[2], tensors))

            if node.func.id == 'Ftob' or node.func.id == 'Btof':
                return tf.dtypes.cast(*[self.eval_tf_expr_graph(arg, tensors) for arg in node.args], dtype=ast_tensor_dict[node.func.id])

            if len(node.args) > 2:
                self.printpl('e', 'This has more than 2 args and is not Ifte? {}'.format(str(node.func.id)))
            else:
                try:
                    return ast_tensor_dict[node.func.id](*[self.eval_tf_expr_graph(arg, tensors) for arg in node.args])
                except Exception as ex:
                    self.printpl('w', 'debug warning expr: {}'.format(self.debug_warnings['expr']))
                    self.printpl('e', 'node.func.id caused an exception:{}\n'
                                      'node.func.id: {}\n'
                                      'node.args: {}\n'
                                      'and expression: {}'.format(ex, node.func.id, str(node.args), self.debug_warnings['expr']))

        elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
            return self.eval_tf_chain_bool(node.values, ast_tensor_dict[type(node.op)], tensors)

        elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
            return self.eval_tf_chain_compare([node.left] + node.comparators, node.ops, tensors)

        elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
            return tf.constant(node.value)

        else:
            raise TypeError(node)

    def eval_tf_chain_bool(self, values, operation, tensors):

        """
        Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.

        """

        x = tf.dtypes.cast(self.eval_tf_expr_graph(values[0], tensors), tf.bool)
        if len(values) > 1:
            return operation(x, self.eval_tf_chain_bool(values[1:], operation, tensors))
        else:
            return x

    def eval_tf_chain_compare(self, comparators, ops, tensors):

        """
        Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

        Called by: fitness_node_parse

        Arguments required: comparators, ops, tensors
        """

        x = self.eval_tf_expr_graph(comparators[0], tensors)
        y = self.eval_tf_expr_graph(comparators[1], tensors)

        if len(comparators) > 2:
            return tf.logical_and(ast_tensor_dict[type(ops[0])](x, y), self.eval_tf_chain_compare(comparators[1:], ops[1:], tensors))
        else:
            return ast_tensor_dict[type(ops[0])](x, y)

    def eval_tf_classify_labels_map(self, result):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the data_csv_path .csv. Outputs an array of tuples containing the predicted
        labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.class_labels / 2) - 1 # '-1' keeps a binary classification splitting over the origin
            if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
            elif solution == self.class_labels - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
            else: fitness = 0 # no class match

        Called by: fitness_eval

        Arguments required: result
        """

        skew = (self.class_labels / 2) - 1
        label_rules = {self.class_labels - 1: (
            tf.constant(self.class_labels - 1), tf.constant(' > {}'.format(self.class_labels - 2 - skew)))}

        for class_label in range(self.class_labels - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tf.cond(cond, lambda: (
                tf.constant(class_label), tf.constant(' <= {}'.format(class_label - skew))),
                                               lambda: label_rules[class_label + 1])

        pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))),
                             lambda: label_rules[1])

        return pred_label

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to receive correct xtypes (f2b,.) |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def xtype_choose_func_pointmutation(self, xtype=None, arity=None):
        """
        returns a function for a function in point mutation
        This only accepts functions as inputs. (point mutation)
        No need to handle terminals
        """

        if arity:
            if xtype == 'f2f':
                return np.random.choice(self.op_type_arity_array[f2f][arity])
            elif xtype == 'f2b':
                return np.random.choice(self.op_type_arity_array[f2b][arity])
            elif xtype == 'b2b':
                return np.random.choice(self.op_type_arity_array[b2b][arity])
            elif xtype == 'b2f':
                return np.random.choice(self.op_type_arity_array[b2f][arity])
            elif xtype == 'b2f2f':
                return np.random.choice(self.op_type_arity_array[b2f2f][arity])  # sfeh okay that does not make sense tbh
            else:
                self.printpl('e', 'Function was not found in function_types_dict {}'.format(xtype))
                raise
        else:
            raise

    def xtype_choose_func_branchmutate(self, xtype=None, arity=False):
        """
        This fills in a function that fits the type of the function/terminal before.
        terminal  '2f' -> '_2f', arity
        function 'f2f' -> '_2f', arity
        function 'b2f2f' -> '_2f', arity
        > ->
        """
        if xtype:
            if '2f' in xtype:
                choose_func = sum(self.op_type_arity_array[f2f] + self.op_type_arity_array[b2f] + self.op_type_arity_array[b2f2f], [])
            elif '2b' in xtype:
                choose_func = sum(self.op_type_arity_array[f2b] + self.op_type_arity_array[b2b], [])
            else:
                raise
        else:
            choose_func = sum(self.op_type_arity_array[f2f] +
                              self.op_type_arity_array[b2f] +
                              self.op_type_arity_array[b2f2f] +
                              self.op_type_arity_array[f2b] +
                              self.op_type_arity_array[b2b], [])

        # Attention! do not choose out of an dictionary.
        # every function is inside there only once, so no higher chance for functions that are more often in the list
        # label = np.random.choice(self.xype_func_dict['2f'])

        # choose out of a list, or add another way. maybe automatic?

        label = np.random.choice(choose_func)

        return label, op[str(label)]['arity']

    def xtype_get(self, label, node_arity=None):
        """
        returns xtype for a label
        """
        if not node_arity:
            node_arity = op_label_get_arity(label)

        if node_arity == 0:  # arity=0 -> terminal
            if 'True' in label or 'False' in label:
                node_xtype = '2b'
            elif 'observation' in label:
                term_position = self.variables_dict['all'].index(label)
                node_xtype = op[self.variables_dict['types'][term_position]]['xtype']
            elif 'action' in label:
                self.printpl('w', 'Does this happen? Test it!')
                node_xtype = self.action_dict[label]

            else:  # only 'float' left
                node_xtype = '2f'
        elif node_arity > 0:
            node_xtype = op[label]['xtype']
        else:
            self.printpl('e', 'This arity is not known: {}'.format(node_arity))
            raise

        return node_xtype

    def xtype_choose_term(self, node_xtype):
        """
        Returns a terminal of xtype.

        function: f2b -> 2b needed
        terminal:  2f -> 2f needed
        --> check if it is function, aka _2f
        --> check if it is terminal, aka f2

        Modes:
        var_and_const: return randomly (50:50) a variable or a constant
        terminal_only: return                  a variable
        Todo Introduce constants-mode, where the user can give constant types (similar to functions)?

        input options: f2f, f2b, b2f, b2b, f2b2b, 2f, 2b
        """

        # node_xtype == '2f' or 'f2' in node_xtype:
        if '2f' in node_xtype:
            terminals_type = self.variables_dict['float']
            the_type = 'float'
        elif '2b' in node_xtype:
            terminals_type = self.variables_dict['bool']
            the_type = 'bool'
        else:
            self.printpl('e', 'Probably, you have to check if your "function" is actually a terminal. xtype {}'.format(node_xtype))
            raise

        if np.random.choice(['var', 'const']) == 'var':  # our choice is variable
            if terminals_type:  # Is there an entry in the list?
                return np.random.choice(terminals_type)  # ...so we return one
        return self.tree_build_type_constant_get(term_type=the_type)  # otherwise: constant (There are always constants :P)

    def pnode_function_select(self, pnode):

        """
        Returns a function with the same outcome

        """

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def file_autoplots(self, path, post_path=None):
        """
        monitors everything

        Helper:
        # plot_end(self, y, mode='', plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='')
        """
        if self.monitor_dict['gen_fitness_average'] == 'y':
            data_tupels = sorted(list(self.monitoring_dict['fitness_average'].items()))
            self.plot_end(data_tupels, path, plt_title='Average Fitness', plt_y_label='Fitness')

        if self.monitor_dict['genepool_size'] == 'y':
            data_tupels = sorted(list(self.monitoring_dict['genepool_size'].items()))
            self.plot_end(data_tupels, path, plt_title='Genepool size', plt_y_label='Amount')

        data_tupels = sorted(list(self.monitoring_dict['total_found_trees'].items()))
        self.plot_end(data_tupels, path, plt_title='Number of created Trees', plt_y_label='Amount')

        data_tupels = sorted(list(self.pareto.items()))
        self.plot_end(data_tupels, path, plt_title='Pareto Dominant Candidates', plt_x_label='Parsimony', plt_y_label='Fitness')
        return

    def monitor_genepool(self, gene_pool_hash_dict):
        """
        Give the user some feedback
        """

        # How many survived in the selection?
        self.monitoring_dict['genepool_size'][int(self.gen_id)] = len(gene_pool_hash_dict)
        if len(gene_pool_hash_dict) > 0:
            self.printpl('ggg', 'The generation`s population is: {}'.format(len(gene_pool_hash_dict)))
        else:  # the evolutionary constraints were too tight, killing off the entire population
            self.printpl('e', 'There are no Trees in the gene pool. You should archive your population and (q)uit.')
            self.file_autowrite(self.path, self.gen_id)
            self.file_autoplots(self.path)
            sys.exit()

        # average fitness our genepool?
        fitness_train_sum = 0
        for key, value in gene_pool_hash_dict.items():
            fitness_train_sum += float(self.tree_hash_meta[value]['fitness_train'])
        average_fitness = fitness_train_sum / len(gene_pool_hash_dict)
        self.monitoring_dict['fitness_average'][int(self.gen_id)] = average_fitness
        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to display output information     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def fitness_dummy_get(self):
        if self.fitness_type == 'min':
            return float('inf')
        else:
            return float(0)

    def plot_end(self, data_2d, path, plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='', yscale='linear', variance=None):

        x, y = [], []
        for a, b in data_2d:
            x.append(a)
            y.append(b)

        if variance:
            printez('e', 'variance not available')
            means = np.mean(y, axis=0)
            stds = np.std(y, axis=0)
            n = means.size

        plt.plot(x, y, label=plt_curve_label)

        if plt_x_label and plt_y_label:
            plt.xlabel(plt_x_label)
            plt.ylabel(plt_y_label)

        if plt_title:
            plt.title(plt_title)

        # plt.legend()
        plt.yscale(yscale)
        plt.ylim(0)
        plt.xlim(0)
        path_plot = path / 'plots'
        if not Path.is_dir(path_plot):
            Path.mkdir(path_plot)
        plt.savefig(path_plot / '{}-plot.jpg'.format(plt_title))
        plt.close()
        return

    # def plagih_pause(self):
    #
    #     """
    #     Pause the program execution and engage the user, providing a number of options.
    #
    #     Arguments required: [0,1,2] where (0) refers to an end-of-run; (1) refers to any use of the (pause) menu from
    #     within the run, and anticipates ENTER as an escape from the menu to continue the run; and (2) refers to an
    #     'ERROR!' for which the user may want to archive data_csv_path before terminating. At this point in time, (2) is
    #     associated with each error but does not provide any special options).
    #     """
    #
    #     ### PART 1 - reset and pack values to send to menu.pause ###
    #     menu_dict = {'input_a': '',
    #                  'input_b': 0,
    #                  'display': self.display,
    #                  'tree_depth_max': self.config['tree_depth_max'],
    #                  'pop_max': self.pop_max,
    #                  'gen_id': self.gen_id,
    #                  'gen_max': self.config['gen_max'],
    #                  'tourn_size': self.tourn_size,
    #                  'evolve_repro': self.evolve_rates['reproduce'],
    #                  'evolve_point': self.evolve_rates['mutate_point'],
    #                  'evolve_branch': self.evolve_rates['mutate_branch'],
    #                  'evolve_cross': self.evolve_rates['Crossover one Branch'],
    #                  # 'fittest_dict': self.origin_dominators,
    #                  'pop_last_len': len(self.population_base),
    #                  'pop_new_len': len(self.population_new),
    #                  'path': self.path}
    #
    #     menu_dict = menu.pause(menu_dict)  # call the external function menu.pause
    #
    #     ### PART 2 - unpack values returned from menu.pause ###
    #     input_a = menu_dict['input_a']
    #     input_b = menu_dict['input_b']
    #     self.display = menu_dict['display']
    #     self.config['gen_max'] = menu_dict['gen_max']
    #
    #     self.evolve_rates['reproduce'] = menu_dict['evolve_repro']
    #     self.evolve_rates['mutate_point'] = menu_dict['evolve_point']
    #     self.evolve_rates['mutate_branch'] = menu_dict['evolve_branch']
    #     self.evolve_rates['Crossover one Branch'] = menu_dict['evolve_cross']
    #
    #     ### PART 3 - execute the user queries returned from menu.pause ###
    #     if input_a == 'esc':
    #         return 2  # breaks out of the plagih_gp() or plagih_pause_refer() loop
    #
    #     elif input_a == 'eval':  # evaluate a Tree against the TEST data_csv_path
    #         algo_sym = self.tree_expr_sympify(tree=self.population_new[input_b])  # generate the raw and sympified expression for the given Tree using SymPy
    #         self.printpl('i', '\n\t\033[36mTree', input_b, 'yields (sym):\033[1m', algo_sym, BColors.RESET)  # print the sympified expression
    #         result = self.eval_tf(str(algo_sym), self.data_control, get_pred_labels=True)  # might change to algo_raw_str evaluation
    #         self.pause_fitness_test(result)  # TF tested 2017 02/02
    #
    #     elif input_a == 'print_last':  # print a Tree from population_genepool
    #         self.display_tree(self.population_base[input_b])
    #
    #     elif input_a == 'print_new':  # print a Tree from population_new
    #         self.display_tree(self.population_new[input_b])
    #
    #     elif input_a == 'pop_last':  # list all Trees in population_genepool
    #         self.printpl('i', '')
    #         for tree_id in range(1, len(self.population_base)):
    #             algo_sym = self.tree_expr_sympify(tree=self.population_base[tree_id])
    #             self.printpl('i', '\t\033[36m Tree', self.population_base[tree_id][TR_ID][1], 'yields (sym):\033[1m', algo_sym, BColors.RESET)
    #
    #     elif input_a == 'pop_new':  # list all Trees in population_new
    #         self.printpl('ii', '')
    #         for tree_id in range(1, len(self.population_new)):
    #             algo_sym = self.tree_expr_sympify(tree=self.population_new[tree_id])  # extract the expression
    #             self.printpl('ii', '\t\033[36m Tree', self.population_new[tree_id][TR_ID][1], 'yields (sym):\033[1m', algo_sym, BColors.RESET)
    #
    #     elif input_a == 'load':  # load population_s to replace population_genepool
    #         pass
    #         # self.data_pickle_recover(self.filename['s'])  #
    #
    #     elif input_a == 'write':  # write the evolving population_new to disk
    #         self.file_population_write(self.population_new, 'new')
    #
    #     elif input_a == 'add':  # check for added generations, then exit plagih_pause and continue the run
    #         self.config['gen_max'] = self.config['gen_max'] + input_b  # if input_b > 0: self.gen_max = self.gen_max + input_b - REMOVED 2019 06/05
    #
    #     elif input_a == 'quit':
    #         self.main_terminate()  # archive populations and exit
    #
    #     return _pause(self):
    #
    #     """
    #     Pause the program execution and engage the user, providing a number of options.
    #
    #     Arguments required: [0,1,2] where (0) refers to an end-of-run; (1) refers to any use of the (pause) menu from
    #     within the run, and anticipates ENTER as an escape from the menu to continue the run; and (2) refers to an
    #     'ERROR!' for which the user may want to archive data_csv_path before terminating. At this point in time, (2) is
    #     associated with each error but does not provide any special options).
    #     """
    #
    #     ### PART 1 - reset and pack values to send to menu.pause ###
    #     menu_dict = {'input_a': '',
    #                  'input_b': 0,
    #                  'display': self.display,
    #                  'tree_depth_max': self.config['tree_depth_max'],
    #                  'pop_max': self.pop_max,
    #                  'gen_id': self.gen_id,
    #                  'gen_max': self.config['gen_max'],
    #                  'tourn_size': self.tourn_size,
    #                  'evolve_repro': self.evolve_rates['reproduce'],
    #                  'evolve_point': self.evolve_rates['mutate_point'],
    #                  'evolve_branch': self.evolve_rates['mutate_branch'],
    #                  'evolve_cross': self.evolve_rates['Crossover one Branch'],
    #                  # 'fittest_dict': self.origin_dominators,
    #                  'pop_last_len': len(self.population_base),
    #                  'pop_new_len': len(self.population_new),
    #                  'path': self.path}
    #
    #     menu_dict = menu.pause(menu_dict)  # call the external function menu.pause
    #
    #     ### PART 2 - unpack values returned from menu.pause ###
    #     input_a = menu_dict['input_a']
    #     input_b = menu_dict['input_b']
    #     self.display = menu_dict['display']
    #     self.config['gen_max'] = menu_dict['gen_max']
    #
    #     self.evolve_rates['reproduce'] = menu_dict['evolve_repro']
    #     self.evolve_rates['mutate_point'] = menu_dict['evolve_point']
    #     self.evolve_rates['mutate_branch'] = menu_dict['evolve_branch']
    #     self.evolve_rates['Crossover one Branch'] = menu_dict['evolve_cross']
    #
    #     ### PART 3 - execute the user queries returned from menu.pause ###
    #     if input_a == 'esc':
    #         return 2  # breaks out of the plagih_gp() or plagih_pause_refer() loop
    #
    #     elif input_a == 'eval':  # evaluate a Tree against the TEST data_csv_path
    #         algo_sym = self.tree_expr_sympify(tree=self.population_new[input_b])  # generate the raw and sympified expression for the given Tree using SymPy
    #         self.printpl('i', '\n\t\033[36mTree', input_b, 'yields (sym):\033[1m', algo_sym, BColors.RESET)  # print the sympified expression
    #         result = self.eval_tf(str(algo_sym), self.data_control, get_pred_labels=True)  # might change to algo_raw_str evaluation
    #         self.pause_fitness_test(result)  # TF tested 2017 02/02
    #
    #     elif input_a == 'print_last':  # print a Tree from population_genepool
    #         self.display_tree(self.population_base[input_b])
    #
    #     elif input_a == 'print_new':  # print a Tree from population_new
    #         self.display_tree(self.population_new[input_b])
    #
    #     elif input_a == 'pop_last':  # list all Trees in population_genepool
    #         self.printpl('i', '')
    #         for tree_id in range(1, len(self.population_base)):
    #             algo_sym = self.tree_expr_sympify(tree=self.population_base[tree_id])
    #             self.printpl('i', '\t\033[36m Tree', self.population_base[tree_id][TR_ID][1], 'yields (sym):\033[1m', algo_sym, BColors.RESET)
    #
    #     elif input_a == 'pop_new':  # list all Trees in population_new
    #         self.printpl('ii', '')
    #         for tree_id in range(1, len(self.population_new)):
    #             algo_sym = self.tree_expr_sympify(tree=self.population_new[tree_id])  # extract the expression
    #             self.printpl('ii', '\t\033[36m Tree', self.population_new[tree_id][TR_ID][1], 'yields (sym):\033[1m', algo_sym, BColors.RESET)
    #
    #     elif input_a == 'load':  # load population_s to replace population_genepool
    #         pass
    #         # self.data_pickle_recover(self.filename['s'])  #
    #
    #     elif input_a == 'write':  # write the evolving population_new to disk
    #         self.file_population_write(self.population_new, 'new')
    #
    #     elif input_a == 'add':  # check for added generations, then exit plagih_pause and continue the run
    #         self.config['gen_max'] = self.config['gen_max'] + input_b  # if input_b > 0: self.gen_max = self.gen_max + input_b - REMOVED 2019 06/05
    #
    #     elif input_a == 'quit':
    #         self.main_terminate()  # archive populations and exit
    #
    #     return 1

    def pause_fitness_test(self, result):

        if self.kernel == 'classification':
            """
            Print the Precision-Recall and Confusion Matrix for a CLASSIFICATION run against the test data_csv_path.

            From scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
                Precision (P) = true_pos / true_pos + false_pos
                Recall (R) = true_pos / true_pos + false_neg
                harmonic mean of Precision and Recall (F1) = 2(P x R) / (P + R)

            From scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
                y_pred = result, the predicted labels generated by Plagih GP
                y_true = solution, the true labels associated with the data_csv_path

            """
            for i in range(len(result['result'])):
                self.printpl('iii', '\t Data row {} predicts class:\033[1m {} ({} True)\033[0;0m\033[36m as {:.2f}{}\033[0;0m'.format(
                    i, int(result['pred_labels'][0][i]), int(result['solution'][i]), result['result'][i],
                    result['pred_labels'][1][i]))

            self.printpl('iii', '\n Fitness score: {}\n'
                                'Precision-Recall report:\n{}\n'
                                'Confusion matrix:\n{}'.format(result['fitness'],
                                                               skm.classification_report(result['solution'], result['pred_labels'][0]),
                                                               skm.confusion_matrix(result['solution'], result['pred_labels'][0])))

        elif self.kernel == 'regression':
            """
            Print the Fitness score and Mean Squared Error for a REGRESSION run against the test data_csv_path.

            """

            for i in range(len(result['result'])):
                self.printpl('iii', '\tData row {} predicts c1:\033[1m {:.2f} ({:.2f} True)'.
                             format(i, result['result'][i], result['solution'][i]))

            mse, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']

            self.printpl('iii', '\n\t Origin fitness score: {}'.format(self.origin['fitness_train']))
            self.printpl('iii', '\n\t Regression fitness score: {}'.format(fitness))
            self.printpl('iii', '\t Mean Squared Error: {}'.format(mse))

            return

        elif self.kernel == 'match':

            """
            Print the accuracy for a MATCH kernel run against the test data_csv_path.

            """

            for i in range(len(result['result'])):
                self.printpl('vv', '\tData row {} predicts match: {} {:.2f} ({:.2f} True)'.format(i, BColors.BOLD, result['result'][i], result['solution'][i]))

            self.printpl('arity', 'Matching fitness score: {}'.format(result['fitness']))

            return
        else:
            self.printpl('e', 'This fitness test is not available:')

        return

    def display_tree(self, tree):

        """
        dummy for displaying a tree
        """

        self.printpl('i', '\n{} Tree ID {}{}'.format(BColors.CYAN, int(tree[TR_ID][1]), BColors.RESET))

        for depth in range(0, self.config['tree_depth_max'] + 1):  # increment through all possible Tree depths - tested 2016 07/09

            self.printpl('i', 'Tree: {}\n'.format(tree))

        algo_raw_str = str(self.tree_expr_raw(tree, 1))
        self.printpl('i', '\tTree {} yields (raw): {}'.format(tree[TR_ID][1], algo_raw_str))
        algo_sym = self.tree_expr_sympify(algo_raw_str=algo_raw_str)
        self.printpl('i', '\tTree {} yields (sym): {}'.format(tree[TR_ID][1], algo_sym))

        return

    def printpl(self, message_type, text):

        """
        Gets a verbosity (e.g. 'i')

        PLAGIH-Display-modes:
            Errors (e): always on, does print Errors
            Info (i): informations. (need further specification)
            NextGen (n): Prints infos about the current mutation process

        Display modes: (Just a reminder from original plagih)
            Generational (g): pauses after each generation is complete
            Interactive (i): pauses with the completion of each section (e.g. tournament, gene pool, genetic operators)
            DeBug (db): displays the internal workings of the genetic operators
            minimal (m): displays only the multivariate expression of each tree
            Silent (s): displays only the summary of each generations

        """

        if message_type in self.display:
            printez(message_type, text, time_total=time.perf_counter() - self.time_start)

        return

    # def evolve_subtree_depth_choose(self, ptree, top_id, bottom_id, amount_replaced_nodes=None, mode='base_depth'):  # sfeh other default
    #     """
    #     Return the size of the tree to be inserted.
    #     Should not be set to maximum to reduce complexity!
    #     """
    #
    #     depth_old = int(ptree[N_depth][bottom_id]) - int(ptree[N_depth][top_id])  # subtract depth of 'branch_top' from the last in 'branch'
    #     depth_upper_bound = self.config['tree_depth_max'] - int(ptree[N_depth][top_id])  # = 10 - (node_depth)
    #     if mode == 'maximum':
    #         branch_depth = depth_upper_bound
    #     elif mode == 'same_length':
    #         branch_depth = depth_old
    #     elif mode == 'base_depth':
    #         branch_depth = min(self.config['tree_depth_base'], depth_upper_bound)
    #     elif mode == 'random':
    #         branch_depth = min(depth_upper_bound, np.random.randint(0, 1 + max(depth_upper_bound, 3)))  # SFEH random depth, I hope this is enough to guarantee tree size
    #     else:
    #         self.printpl('e', 'evolve_subtree_depth_choose does not accept this mode: {}'.format(mode))
    #         raise
    #     return branch_depth
    # def tree_store_meta_lastgen(self, tree, modification=''):
    #
    #     """
    #     Remove all fitness data_csv_path from a given tree.
    #
    #     This is required after a new generation is evolved as the fitness of the same Tree prior to its mutation will
    #     no longer apply.
    #
    #     """
    #
    #     # save information about how good last changes were
    #     # for i in range(min(self.tree_depth_min, 5), 2, -1):  # 5,4,3,2
    #     #     tree[TR_type][i] = tree[TR_type][i-1]    # The last modifications
    #     #     tree[T_fitness][i] = tree[T_fitness][i-1]  # The last fitness
    #     #     tree[T_parsimony][i] = tree[T_parsimony][i-1]  # The last parsimony (TODO) # tree_id,1,a,b,c -> tree_id,1,a,a,b
    #
    #     # What needs to be assigned later
    #     # tree[TR_type][2] = modification  # wipe last modification data_csv_path
    #     # tree[TR_ID][1] = ''  # -> tree_id,,
    #     # tree[T_fitness][1] = ''  # wipe fitness data_csv_path
    #     # tree[T_parsimony][1] = ''  # wipe parsimony data_csv_path
    #
    #     return tree

    # Stuff, that is not needed
    #
    # def treegp_mutate_branch_terminal_build(self):
    #
    #     """
    #     Build the Terminal nodes for the tree.
    #
    #     """
    #
    #     self.tnode[N_depth] = self.tnode[TR_depth]  # set the final node_depth (same as 'config.pop[N_depth]' + 1)
    #
    #     for j in range(1, len(self.tree[N_id])):  # go through all nodes
    #         if int(self.tree[N_depth][j]) == self.tnode[N_depth] - 1:  # this node is a parent
    #             for k in range(1, (int(self.tree[N_arity][j]) + 1)):  # increment through each degree of arity for each parent node
    #                 self.tnode[N_parent] = int(self.tree[3][j])  # set the parent 'node_id'  ...
    #                 self.treegp_branch_terminal(op[self.tree[N_label][j]]['xtype'])  # ... generate a Terminal node
    #
    #     return
    # def plot_live(self):
    #     many_values = np.array([[6, 5, 4, 5, 6, 5, 4, 5, 5, 6, 5, 8, 5, 5, 5, 5, 5, 5, 4, 5, 6, 5],
    #                             [2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 8, 9, 8, 7],
    #                             [2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 8, 9, 8, 7],
    #                             [2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 7, 8, 9, 0],
    #                             [2, 3, 4, 5, 4, 3, 2, 3, 8, 9, 8, 7, 6, 5, 6, 7, 8, 9, 0, 9, 8, 7],
    #                             [2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 7, 1],
    #                             [8, 7, 6, 5, 6, 7, 8, 9, 0, 9, 8, 7, 6, 3, 4, 5, 6, 7, 8, 7, 6, 1]])
    #
    #     means = np.mean(many_values, axis=0)
    #     stds = np.std(many_values, axis=0)
    #     n = means.size
    #
    #     # helpers
    #     epsilon = 0.1
    #     num_plays = np.shape(many_values)[1]
    #
    #     ### start episode loop
    #     # compute upper/lower confidence bounds
    #     ci = 0.95
    #     e = 6  # if e%200 == 0:, the current episode we are in. aka the last one for us
    #     test_stat = st.t.ppf((ci + 1) / 2, e)
    #     lower_bound = means - test_stat * stds / np.sqrt(e)
    #     upper_bound = means + test_stat * stds / np.sqrt(e)
    #
    #     # clear plot frame
    #     plt.clf()
    #
    #     # plot average reward
    #     plt.plot(means, color='blue', label="epsilon=%.2f" % epsilon)
    #
    #     # plot upper/lower confidence bound
    #     x = np.arange(0, num_plays, 1)
    #     plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)
    #
    #     plt.grid()
    #     # plt.ylim(0, 2)  # limit y axis
    #     plt.title('Avg. Reward per step in experiment {}: {}'.format(e, sum(means) / num_plays))
    #     plt.ylabel("Reward per step")
    #     plt.xlabel("Play")
    #     plt.legend()
    #     plt.show()
    #     plt.pause(0.1)
    #     ### end episode loop
    #
    #     ## disable interactive plotting => otherwise window terminates
    #     plt.ioff()
    #     plt.show()
    #     return

    # # 03.01. not needed?
    # def tree_replace_branch_nodelist(self, karoo_tree, tree_ids, insert_tree):
    #
    #     """
    #     This method enables the insertion of ptree_branch_node_ids in place of a branch
    #     ptree_branch_node_ids = [5,6,8,9] node that are changed
    #
    #     The end result is a Tree with a mutated branch.
    #     """
    #
    #     branch_top = int(tree_ids[0])
    #     karoo_tree[N_label][branch_top] = insert_tree[N_label][1]  # copy node_label from new karoo_tree
    #     karoo_tree[N_arity][branch_top] = insert_tree[N_arity][1]  # copy node_arity from new karoo_tree
    #     karoo_tree = np.delete(karoo_tree, tree_ids[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
    #
    #     c_buffer = evolve_c_buffer_karoo(karoo_tree, branch_top)  # generate c_buffer for point of mutation ('branch_top')
    #     karoo_tree = tree_insert_node_child_dummies(karoo_tree, branch_top, c_buffer, wrapper=True)  # insert a single new node ('branch_top')
    #     karoo_tree = evolve_node_renum_karoo(karoo_tree)  # renumber all 'node_id's
    #
    #     ### PART 2 - insert b_branchody from 'config.karoo_tree' into 'karoo_tree' ###
    #     node_count = 2  # set node count for 'config.karoo_tree' to 2 as the new root has already replaced 'branch_top' (above)
    #
    #     while node_count < len(insert_tree[3]):  # increment through all nodes in the new Tree ('config.karoo_tree'), starting with node 2
    #
    #         for j in range(1, len(karoo_tree[3])):  # increment through all nodes in tourn_winner ('karoo_tree')
    #
    #             if karoo_tree[N_type][j] == '':
    #                 karoo_tree[N_type][j] = insert_tree[N_type][node_count]  # copy 'node_type' from branch to karoo_tree
    #                 karoo_tree[N_label][j] = insert_tree[N_label][node_count]  # copy 'node_label' from branch to karoo_tree
    #                 karoo_tree[N_arity][j] = insert_tree[N_arity][node_count]  # copy 'node_arity' from branch to karoo_tree
    #
    #                 if karoo_tree[N_arity][j] == '0':  # terminal
    #                     karoo_tree = tree_fix_link_child_karoo(karoo_tree)  # fix all child links
    #                     karoo_tree = evolve_node_renum_karoo(karoo_tree)  # renumber all 'node_id's
    #
    #                 if int(karoo_tree[N_arity][j]) > 0:
    #                     c_buffer = evolve_c_buffer_karoo(karoo_tree, j)  # generate 'c_buffer' for point of mutation ('branch_top')
    #                     karoo_tree = tree_insert_node_child_dummies(karoo_tree, j, c_buffer, wrapper=True)  # insert new nodes
    #                     karoo_tree = tree_fix_link_child_karoo(karoo_tree)  # fix all child links
    #                     karoo_tree = evolve_node_renum_karoo(karoo_tree)  # renumber all 'node_id's
    #
    #                 node_count = node_count + 1  # exit loop when 'node_count' reaches the number of columns in the array 'config.karoo_tree'
    #
    #     return karoo_tree

    # def tree_branch(self, root_xype, max_depth=''):
    #
    #     """
    #     Builds a new 'branch'-tree
    #     """
    #
    #     tree_id, last_mod = 'x', 'b',
    #     tree = self.ptree_init()
    #     if max_depth:
    #
    #         label, arity = self.xtype_choose_func(root_xype)
    #
    #         if np.random.choice(['function', 'terminal']) == 'terminal':
    #             tree[N_type][top_id] = 'term'
    #             tree[N_label][top_id] = self.xtype_choose_term(xype)  # replace with a correct label
    #             tree = np.delete(tree, branch_ids[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
    #             tree = evolve_node_arity_fix(tree)  # fix all node arities (term)
    #             tree = self.tree_fix_link_child(tree)  # fix all child links (func)
    #             tree = evolve_node_renum(tree)  # renumber all 'node_id's
    #
    #             # 5.2 We insert a function here
    #             tree = self.ptree_replace_branch_nodelist(tree, branch_ids, self.tree)  # insert new 'branch' at point of mutation 'branch_top' in tourn_winner 'tree'
    #
    #         ptree = self.ptree_init()
    #         ptree = self.ptree_node_add_fromdnode(ptree, pnode)
    #
    #         self.treegp_mutate_branch_terminal_build()  # build the Terminal nodes
    #         return


def util_tree_copy(population, tree_id):
    """
    copy a tree from a population
    """
    return np.copy(population[tree_id])


def pop_util_random(population):
    """
    Returns a random tree_id from a population
    """
    return np.random.randint(1, len(population))


def pop_util_copy(population_x, title):
    """
    Copy one population to another.
    """
    population_y = [title]  # an empty list stores a copy of the prior generation

    for tree_id in range(1, len(population_x)):  # increment through each Tree in the current population
        tree_copy = util_tree_copy(population_x, tree_id)  # copy each array in the current population
        population_y.append(tree_copy)  # add each copied Tree to the new population list

    return population_y


def insert_function_or_term(depth, depth_goal):
    """
    with a certain probability, insert terminals or functions
    SFEH TODO this need to be changed
    """
    if np.random.choice(['50', '50', '50', '50', 'larger']) == 'larger':
        probability = np.random.uniform(0, depth_goal)
        if probability > min(depth, depth_goal / 2):
            decision = 'function'
        else:
            decision = 'terminal'
        return decision
    else:
        decision = np.random.choice(['terminal', 'function'])

    return decision


def pop_copy_genepool(population_new, gene_pool_hash_dict, gen_id):
    """
    Copy the genepool of a gen
    """
    pop_y = ['Population Selection in Generation {}.'.format(str(gen_id))]  # empty list

    for i, (tree_id, tree_ident) in enumerate(gene_pool_hash_dict.items()):
        tree_copy = util_tree_copy(population_new, tree_id)
        tree_copy[TR_ID][1] = i + 1
        pop_y.append(tree_copy)

    return pop_y


def pop_enum_trees(population):
    """
    outsourced enumeration of trees in a population
    """
    for tree_id in range(1, len(population)):  #
        population[tree_id][TR_ID][1] = tree_id
    return population


def data_load_data_split(data_x, data_y, test_size):
    # TODO die func kann sicher nicht mit 2d labels umgehen. Funktion macht das echt super uneffizient.
    x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size=test_size)  # 80/20 TRAIN/TEST split
    data_train = np.c_[x_train, y_train]  # recombine each row of data_csv_path with its associated class label (right column)
    data_control = np.c_[x_test, y_test]  # recombine each row of data_csv_path with its associated class label (right column)

    data_train_rows = len(data_train[:, 0])

    return data_train_rows, data_train, data_control


def printez(message_type, text, time_total=0.0):
    """
    giving prints colours, accessable from everywhere
    """
    message_pretxt = BColors.RESET  # default color
    message_posttxt = BColors.RESET
    if 'i' in message_type:
        message_style = BColors.CYAN
        message_pretxt = 'Info: '
    elif 'e' in message_type:
        message_style = BColors.RED
        message_pretxt = 'ERROR: '
        raise_error = True
    elif 'w' in message_type:  # warning
        message_style = BColors.WARNING
        message_pretxt = 'Warning: '  # Warning-yellow
    elif 'g' in message_type:
        message_style = BColors.BLUE
        message_pretxt = '{:4.2f} Gen: '.format(time_total)  # green
    elif 'v' in message_type:  # verbose
        message_style = BColors.WHITE  # white
        message_pretxt = 'Verbose: '
    elif 'p' in message_type:  # pause
        message_style = BColors.YELLOW
        message_pretxt = 'Pause(TODO): '  # Yellow
        pause = True
    elif 'f' in message_type:  # function
        message_style = BColors.MAGENTA  # Magenta
        message_pretxt = 'Func: '
    elif 'a' in message_type:  # Timer
        message_style = BColors.GREEN
        message_pretxt = 'Alert: '
    else:
        message_style = ''
        printez('w', 'Display-mode {} not known.'.format(message_type))

    print('{}{}{}{}'.format(message_style, message_pretxt, str(text), message_posttxt))
    return
