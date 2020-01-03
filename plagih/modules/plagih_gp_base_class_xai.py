"""
Explaination:
> 'f2f', 'b2b', etc.: my personal silly naming. f = float, b = bool.
   f2b means float to boolean, e. g. '<' takes 'float' and returns a 'bool'
> node_modify (0 or 1), specifies, whether this node is supposed to be




Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""

import sys
import os
import csv
import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skcv
from sympy import sympify, count_ops
from datetime import datetime
import plagih.modules.plagih_gp_pause as menu
# sfeh import the pause later, maybe
# import plagih.modules.plagih_gp_pause as menu
import tensorflow as tf
import ast
import pickle

# PLAGI imports
import re
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
from plagih.modules.plagih_sympy_extras import plagih_sympify
from plagih.modules.dicts import *
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats as st
from plagih.modules.tree_distances.tree_edit_distance import apted_distance
import time
from plagih.modules.plagih_tree import *

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

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

function_infix_to_prefix = {  # currently obsolete
    '+': 'add',
    '-': 'sub',
    '*': 'mult',
    '/': 'div',
    '**': 'power',
    '==': 'eq',
    '!=': 'neq',
    '<': 'lt',
    '<=': 'leq',
    '>': 'gt',
    '>=': 'geq',
}

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees


class BColors:  # sfeh can be deleted
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[39m'

    BLACK2 = '\033[40m'
    RED2 = '\033[41m'


class ExplainableGP(object):
    """

    """

    def __init__(self, config_dict, file_dict, evolve_ratio_dict, monitor_dict):

        self.tree_hash_meta = {}  # All tree informations ->
        self.parsimony_best_dict = {}  # the best tree for each distance c1 -> {distance: tree_meta_tuple}
        self.pareto = {}
        self.population_base = []  # population that is taken to the next generation

        # 1. set global variables to those local values passed from the user script
        self.gp = config_dict
        self.kernel = config_dict['kernel']  # fitness function
        self.tree_pop_max = config_dict['pop_max']  # maximum number of Trees per generation
        self.tourn_size = config_dict['gp_tourn_size']  # number of Trees selected for each tournament
        self.display = config_dict['display']
        self.precision = config_dict['precision']  # the number of floating points for the round function
        self.swim = config_dict['swim']  # pass along the gene_pool restriction methodology
        self.parsimony_min_max = config_dict['parsimony_min_max']

        self.tnode = {}

        self.tf_device = "/gpu:0"  # Set TF computation backend device (CPU or GPU); gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device_log = False  # TF device usage logging (for debugging)

        # What trees should be built
        self.asdf = {'tree_depth_max': 10,
                     'strongly_typed': True,
                     'mutate_branch_depth_max': 6,
                     'mutate_branch_build_method': 'grow',
                     'mutate_branch_grow_term_probability': 0.5,
                     'crossover_strategy': ['same_type', 'same_type_switched', 'convert', 'plagih_switcharoo'],
                     'tree_min_nodes': 3}

        self.monitor_dict = monitor_dict
        self.evolve_rates = evolve_ratio_dict

        self.evolve_missing = evolve_ratio_dict['missing']  # fill up the generation with candidates... todo?
        if self.evolve_missing > 0:
            exit()

        fitt_dict = {'c': 'max', 'r': 'min', 'm': 'max'}
        self.fitness_type = fitt_dict[self.kernel]  # load fitness type
        if self.fitness_type == 'max':
            self.fitness_bad_dummy = 0
        else:
            self.fitness_bad_dummy = float("inf")
        self.gene_pool = {}
        self.xype_func_dict = {'f2f': [], 'f2b': [], 'b2b': [], 'b2f': [], 'b2f2f': [],
                               '2b': [], '2f': [],
                               'b2': [], 'f2': []}
        self.data_load(file_dict['operators_file'], file_dict['samples_file'], file_dict['origin_tree_file'])

        # some useful stuff
        self.debug_warnings = ''
        self.time_start = time.perf_counter()  # process_time() would be even more accurate, but not necessary
        self.monitoring_dict = {'genepool_size': {},
                                'fitness_average': {}}

        self.file_directories_create()

        return

    def plagih_gp_run(self):
        """
        regular plagih-gp run from scratch
        """

        self.gen_id = 1  # set initial generation ID    # first gen only
        self.file_config()
        self.main_generation_first_origin()
        self.main_generation_loop()  # (main loop)
        self.main_terminate()  # archive populations and return to plagih_gp.py for a clean exit

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Top level      functions                  |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def file_population_write(self, population, key):

        """
        Save population_* to disk.

        """
        file_path = self.path + 'population_' + str(key) + '.csv'
        with open(file_path, 'a', newline='') as csv_file:
            target = csv.writer(csv_file, delimiter=',')
            if self.gen_id != 1:
                target.writerows([''])  # empty row before each generation
            target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(self.gen_id)]])

            for tree in range(1, len(population)):
                target.writerows([''])  # empty row before each Tree
                for row in range(0, TRn_um_lines):  # increment through each row in the array Tree (+ row 0)
                    target.writerows([population[tree][row]])

        return

    def file_directories_create(self):
        """
        Create all files that will be saved after all
        """

        self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        cwd = os.getcwd()
        self.path = os.path.join(cwd, 'runs/' + self.datetime + self.gp['name'] + '/')  # generate a unique directory name
        if not os.path.isdir(self.path):
            os.makedirs(self.path)  # make a unique director
        return

    def main_generation_first_origin(self, population_backup_file='', pareto_backup_file=''):
        """
        Everything that needs to be done for the first generation
        - Extracts "origin Tree" from file
        - Creates all other trees: origin tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        self.gen_prepare_parameters()
        self.pop_first_create()

        # Special case: load Data
        try:
            if pareto_backup_file:
                self.pop_data_load_backup_population(population_backup_file)
        except BaseException:
            self.printpl('e', 'Could not load data')
            raise
        self.gen_finalize()

        self.file_population_write(self.population_base, '1_first')  # first gen only

    def main_generation_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        menu_continue = 1
        while menu_continue != 0:  # this allows the user to add generations mid-run and not get buried in nested iterations
            for self.gen_id in range(self.gen_id + 1, self.gp['gen_max'] + 1):  # generation 2 to *max generation*

                # 1. set parameters for the generation
                self.gen_prepare_parameters()

                # 2. Create new generation (from last genepool)

                tourn_size = self.gp['gp_tourn_size']
                self.gen_reproduce(self.evolve_rates['reproduce'], tourn_size)  # method 1 - Reproduction
                self.gen_mutate_point()  # method 2 - Point Mutation
                self.gen_mutate_branch()  # method 3 - Branch Mutation
                self.gen_crossover()  # method 4 - Crossover

                self.gen_finalize()

            else:
                self.printpl('p', '{} Enter {}?{} to review your options or {}q{}uit{}'.format('\033[32m', '\033[1m', '\033[32m', '\033[1m', '\033[32m', '\033[0;0m'))
                menu_continue = 0

    def main_terminate(self):
        """
        Terminates the evolutionary run (if yet in progress), saves parameters and data to disk, and cleanly returns
        the user to plagih_gp.py and the command line.

        """
        self.file_conclusion()
        self.file_pareto()
        # target = open(self.filename['f'], 'w')
        # target.close()  # initialize the .csv file for the final population
        self.file_population_write(self.population_new, 'f')  # save the final generation of Trees to disk

        self.printpl('i', '\n\t\033[32m Your Trees and runtime parameters are archived in plagih_gp/runs/[date-time]/\033[0;0m'
                          '\n\033[3m "It is not the strongest of the species that survive, nor the most intelligent,'
                          '\nbut the one most responsive to change."\033[0;0m --Charles Darwin\n'
                          '\033[3m Congrats!\033[0;0m Your Plagih GP run is complete.')

        self.monitor_show()

        sys.exit()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Generation loop                           |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gen_olymp_update(self):
        """
        The olymp is where the godlike contestants reside.
        In each generation, the olymp searches for new god contestants
        """
        self.printpl('t', 'TODO Olymp for candidates')
        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Load and Archive Data                     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def data_load(self, operators_file, samples_file, origin_tree_file_path):

        """
        Loads all user input and prepares

        """

        self.data_load_dataset(samples_file)
        self.data_load_operators(operators_file)  # Ia a little complex now, outsourced into this function
        self.tree_data_load_origin_tree(origin_tree_file_path)  # construct the first population of Trees

        return

    def data_load_dataset(self, samples_file):

        """
        loads the goal-data from .csv file. first observations then actions.
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
        self.variables_dict = {'all': [],
                               'types': [],
                               'float': [],
                               'bool': []}

        self.action_dict = {}
        self.actions, self.action_types = [], []

        # 1. Read file
        with open(samples_file) as csvFile:
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
                            self.variables_dict['all'].append(term)
                            self.variables_dict['types'].append(term_type)
                        elif var_name.startswith('a'):  # found an action
                            num_actions += 1
                            action = var_name.split(':', 1)[0]
                            action_type = var_name.split(':', 1)[1]
                            self.action_dict[action] = action_type
                            self.actions.append(action)  # action0
                            self.action_types.append(action_type)  # float
                        else:
                            self.printpl('e', 'Behaviour samples first line: Variables have to start with "o" or "a" to be recognized. Is actually: {}'.format(var_name))
                            raise

                    data_x, data_y = [], []

                else:  # convert every 'string' element to its data type
                    # TODO var_types ist genau dasselbe wie self.terminal , oder? eines ersetzen?
                    row_as_data = [locate(var_types[i])(x) for i, x in enumerate(row)]  # ['observation0:float'] + ['0.123'] --> float(['0.123']) --> 0.123
                    data_x.append(row_as_data[:num_observations])
                    data_y.append(row_as_data[num_observations:])
            csvFile.close()
        self.data = samples_file

        # sfeh this is a workaround to have your terminals in typed lists. rework all of this please
        for i, term_type in enumerate(self.variables_dict['types']):  # TODO get all of the actions out, not only one
            term = self.variables_dict['all'][i]
            if term_type == 'float':  # sfeh check if this is enough, int
                self.variables_dict['float'].append(term)
            elif term_type == 'bool':
                self.variables_dict['bool'].append(term)
            elif term_type == 'int':
                self.printpl('w', 'term_type is neither float or bool. Trying to make float out of:', term_type)
                self.variables_dict['float'].append(term)

        # sfeh das funktioniert nur bei diskreten Actions
        self.class_labels = len(np.unique(data_y))  # load the user defined true labels for classification or solutions for regression

        self.data_load_dataset_split(data_x, data_y, test_size=0.2)

        return

    def data_load_dataset_split(self, data_x, data_y, test_size):

        # TODO die func kann sicher nicht mit 2d labels umgehen. Funktion macht das echt super uneffizient.
        x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size=test_size)  # 80/20 TRAIN/TEST split
        data_train = np.c_[x_train, y_train]  # recombine each row of data with its associated class label (right column)
        data_control = np.c_[x_test, y_test]  # recombine each row of data with its associated class label (right column)

        data_x, data_y, x_train, y_train, x_test, y_test = [], [], [], [], [], []  # clear from memory
        # self.data_train_cols = len(data_train[0, :])
        self.data_train_rows = len(data_train[:, 0])
        # self.data_control_cols = len(data_control[0, :])
        # self.data_control_rows = len(data_control[:, 0])

        ### PART 5 - load TRAINING and TEST data for TensorFlow processing
        self.data_train = data_train  # Store train data for processing in TF
        self.data_control = data_control  # Store test data for processing in TF

        return

    def data_load_operators(self, operators_file_path):
        """
        Load all operators ready-to-use from a file
        """
        self.functions = np.loadtxt(operators_file_path, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
        # Part 3.5: Split the functions in 5 types

        # rows are the function types (f2f)
        # columns are the arity
        self.op_type_arity_array = [[[], [], [], []],
                                    [[], [], [], []],
                                    [[], [], [], []],
                                    [[], [], [], []],
                                    [[], [], [], []]]
        for fun in self.functions:
            label = fun[0]
            arity = op[label]['arity']  # arity = int(fun[1])
            xtype = op_xtype_dict[label]

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

        # self.printpl('i', 'self.functions_array: ')
        # pprint(self.functions_array)
        return

    def data_pickle_save(self):
        """
        save all data every few rounds to restore them
        - save the pareto front (done)
        - save the last generation (done)
        - Save valuable meta-data: current generation (done)
        TODO not complete
        """
        run_data = {'gen_id': self.gen_id,
                    'parsimony_front_fitness': '',
                    'pareto': self.pareto,
                    'hash_trees_meta': self.tree_hash_meta,
                    'population_new': self.population_new
                    }
        pickle.dump(run_data, open(self.path + 'Gen-' + str(self.gen_id) + '-backup.p', 'wb'))

    def data_pickle_recover(self, samples_file, operators_file, origin_tree_file_path, pareto_file):

        """
        Restarts a run 'midway' by mainly loading the already found pareto-front
        - Warn user about pickling (todo)
        """

        self.data_load_dataset(samples_file)
        self.data_load_operators(operators_file)  # Ia a little complex now, outsourced into this function
        self.tree_data_load_origin_tree(origin_tree_file_path)  # construct the first population of Trees

        with open(pareto_file, 'rb') as csv_file:
            target = csv.reader(csv_file, delimiter=',')
            n = 0  # track row count

            for row in target:
                print('row', row)

                n = n + 1
                if n == 1:
                    pass  # skip first empty row

                elif n == 2:
                    self.population_base = [row]  # write header to population_genepool

                else:
                    if row == []:
                        self.tree = np.array([[]])  # initialise Tree array

                    else:
                        if self.tree.shape[1] == 0:
                            self.tree = np.append(self.tree, [row], axis=1)  # append first row to Tree

                        else:
                            self.tree = np.append(self.tree, [row], axis=0)  # append subsequent rows to Tree

                    if self.tree.shape[0] == TRn_um_lines:  # (current tree rows + row 0
                        self.population_base.append(self.tree)  # append complete Tree to population list

        self.printpl('i', 'Recovered gene_pool: {} with size {}'.format(self.population_base, len(self.population_base)))

        return

    def file_conclusion(self):

        """
        write the performance of the gp to disc
        """

        file = open(self.path + 'conclusion.txt', 'w')
        file.write('Plagih GP\n launched: ' + str(self.datetime))
        file.write('\n data set: {} \n'.format(str(self.data)))

        result = self.eval_tf(self.origin['algo_sym'], self.data_control, get_pred_labels=True)
        self.origin['fitness_control'] = result['fitness']
        fittest_fitness = result['fitness']

        fittest_algo = self.origin['algo_sym']
        fittest_parsimony = 0

        for parsimony, tree_hash in self.pareto.items():

            # Get relevant data of solution candidate
            tree_meta = self.tree_hash_meta[tree_hash]
            algo_sym = tree_meta['algo_sym']
            parsimony = tree_meta['parsimony']
            result = self.eval_tf(algo_sym, self.data_control, get_pred_labels=True)
            fitness_control = result['fitness']

            update_best = False
            if self.kernel == 'c' and fitness_control >= fittest_fitness:  # find the Tree with maximum fitness score
                update_best = True

            elif self.kernel == 'r' and fitness_control <= fittest_fitness:  # find the Tree with minimum fitness score
                update_best = True

            elif self.kernel == 'm' and fitness_control == self.data_train_rows:  # find the Tree with a perfect match for all data rows
                # TODO what was in the line above that worked?
                update_best = True

            if update_best:
                fittest_fitness = fitness_control
                fittest_algo = algo_sym
                fittest_parsimony = parsimony

            if self.kernel == 'c':
                file.write('\n\n Classification fitness score: {}'.format(fittest_fitness))
                file.write('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution'], result['pred_labels'][0])))
                file.write('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution'], result['pred_labels'][0])))

            elif self.kernel == 'r':
                mse, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
                file.write('\n\n Regression fitness score: {}'.format(fitness))
                file.write('\n Mean Squared Error: {}'.format(mse))

            elif self.kernel == 'm':
                file.write('\n\n Matching fitness score: {}'.format(result['fitness']))

        else:  # pareto dict is empty
            file.write('\n\n No solution was better than the origin... your species has gone extinct!')

        # Info about the origin tree
        file.write('\n\t Origin fitness score: {}'.format(self.origin['fitness_control']))

        # Info about the best Tree
        file.write('\n\n The best candidate has parsimony:' + str(fittest_parsimony))
        file.write('\n With fitness:' + str(fittest_fitness))
        file.write('\n\n With the following sympify-algorithm:\n' + fittest_algo)
        file.write('\n\n')
        file.close()

        return

    def file_config(self):
        """
        write the parameters to a file
        """

        file = open(self.path + 'config.txt', 'w')
        file.write('Plagih GP')
        file.write('\n launched: {}'.format(self.datetime))
        file.write('\n dataset: {}\n'.format(self.data))
        file.write('\n kernel: {}'.format(self.kernel))
        file.write('\n precision: {}\n'.format(self.precision))
        file.write('\n tree depth max: ' + str(self.gp['tree_depth_max']))
        file.write('\n genetic operator Reproduction: ' + str(self.evolve_rates['reproduce']))
        file.write('\n genetic operator Point Mutation: ' + str(self.evolve_rates['mutate_point']))
        file.write('\n genetic operator Branch Mutation: ' + str(self.evolve_rates['mutate_branch']))
        file.write('\n genetic operator Crossover: ' + str(self.evolve_rates['crossover']))
        file.write('\n')
        file.write('\n tournament size: ' + str(self.gp['gp_tourn_size']))
        file.write('\n population: ' + str(self.gp['pop_max']))
        file.write('\n number of generations: ' + str(self.gen_id))
        file.write('\n\n')
        file.close()

    def file_pareto(self):
        """
        Save all the pareto efficient candidates to file
        """
        file = open(self.path + 'pareto.txt', 'w')

        for parsim_key, tree_hash in sorted(list(self.pareto.items())):
            tree_meta = self.tree_hash_meta[tree_hash]
            fitness = tree_meta['fitness_train']
            algo_sym = tree_meta['algo_sym']
            file.write('\nParsimony: \t' + str(parsim_key) + ' Fitness: \t' + str(fitness) + ' Expr: \t' + str(algo_sym))

        file.close()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Population specific     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_data_load_backup_population(self, population_backup_file):
        """
        Loads a saved population from an earlier run
        - Load from a .csv File (TODO)
        - check, if it is compartible with tree_origin (TODO)
        """
        with open(population_backup_file, 'rb') as csv_file:
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

                    if self.tree.shape[0] == TRn_um_lines:  # (current tree rows + row 0
                        self.population_base.append(self.tree)  # append complete Tree to population list

        self.printpl('i', 'We loaded the following population_genepool: {}'.format(self.population_base))
        return

    def pop_genepool_create(self, population):

        """
        Create the gene pool
        - Add a candidate if its parsimony is within the threshold

        # Add the BEST ones to the olymp?

        """
        self.printpl('gg', 'Gene Pool for Generation: {}...'.format(self.gen_id))
        dominator_count = 0

        # Empty old gene_pool first
        gene_pool_hash_dict = {}

        for tree_id in range(1, len(population)):  # Every tree
            tree = population[tree_id]
            tree_ident, tree_meta = self.tree_store_meta_get_hash(tree)

            if tree_meta['parsimony'] < self.parsimony_min_max[1]:  # Tree -> gene_pool?

                gene_pool_hash_dict[tree_id] = tree_ident
                if self.fitness_compare(tree_meta['fitness_train'], self.origin['fitness_train']):
                    self.printpl('vvv', 'A candidate is fitter than the origin (might have occurred already)')
                    dominator_count += 1

        self.printpl('gg', dominator_count, ' Candidates were better than the origin.')

        self.monitor_performance_generation(gene_pool_hash_dict)

        return gene_pool_hash_dict

    # random TODO grow depth anpassen!
    # TODO anzahl bereits bekannter bäume
    # TODO Field Guide programming lesen

    def pop_pareto_update(self):
        """
        Builds up the pareto front
        - iterate over all parsimonys
            - always check for the best of each parsimony
        """

        # 1. the current best fitness is the origin
        fitness_best = self.tree_hash_meta[self.parsimony_best_dict[0]]['fitness_train']

        # 2. look at all the best ones in each parsimony
        for parsim, parsim_ident in sorted(list(self.parsimony_best_dict.items())):

            fitness_parsimony = self.tree_hash_meta[parsim_ident]['fitness_train']

            # 3. is it the same fitness as in pareto?
            if self.fitness_compare(fitness_parsimony, fitness_best, mode='better_or_equal'):
                # actually 'pareto efficient' means not worse. optionally, change mode to 'better'
                if parsim not in self.pareto:
                    self.pareto[parsim] = parsim_ident

                # Get
                pareto_ident = self.pareto[parsim]
                fitness_pareto = self.tree_hash_meta[pareto_ident]['fitness_train']

                # 4. was a lower parsimony already better?
                if self.fitness_compare(fitness_parsimony, fitness_pareto, mode='better'):
                    self.printpl('arity', 'Pareto updated at parsimony {}. New fitness {}. Old fitness: {}'.format(parsim, fitness_parsimony, fitness_best))

                    fitness_best = fitness_parsimony
                    self.pareto[parsim] = parsim_ident

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
        # TODO safely create a complete generation
        self.printpl('g', 'Initial population...')

        tree = self.origin['tree'].copy()
        # tree = self.tree_store_meta_lastgen(tree, modification='i')  # wipe fitness data
        tree[TR_ID][1] = 1
        self.population_new.append(tree)

        for tree_id in range(2, self.gp['pop_max'] + 1):  # range(2,10) goes from 2 to 9. No need for the extra -1?

            # Copy reference tree
            tree = self.origin['tree'].copy()

            # vary this tree with mutation
            branch_nodes_ids = self.ptree_choose_branch_ids(tree)  # [6, 9, 10] select point of mutation and all nodes beneath
            tree = self.evolve_subtree_build(tree, branch_nodes_ids)  # tree with new branch

            # Fill the correct meta-data into the tree (and wipe the old fitness)
            # tree = self.tree_store_meta_lastgen(tree, modification='i')  # wipe fitness data
            tree = self.tree_modifyable_nodes_set(tree)
            tree[TR_ID][1] = tree_id

            self.population_new.append(tree)

        self.printpl('gg', 'We have constructed a single, stochastic population of {} Trees, and saved to disk'.format(self.tree_pop_max))

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

                else:
                    return  # The "regular" case
            else:
                self.parsimony_best_dict[parsim] = gene_pool_hash_dict[tree_id]

        return

    def pop_enum_trees(self, population):
        """
        outsourced enumeration of trees in a population
        """
        for tree_id in range(1, len(self.population_new)):  #
            population[tree_id][TR_ID][1] = tree_id

    def pop_util_copy(self, population_x, title):

        """
        Copy one population to another.
        """
        popolation_y = [title]  # an empty list stores a copy of the prior generation

        for tree_id in range(1, len(population_x)):  # increment through each Tree in the current population
            tree_copy = self.pop_util_tree_copy(population_x, tree_id)  # copy each array in the current population
            popolation_y.append(tree_copy)  # add each copied Tree to the new population list

        return popolation_y

    def pop_copy_genepool(self, population_new, gene_pool_hash_dict):

        """
        Copy the genepool of a gen
        """
        pop_y = ['Population Selection in Generation ' + str(self.gen_id)]  # empty list

        for i, (tree_id, tree_ident) in enumerate(gene_pool_hash_dict.items()):
            tree_copy = np.copy(population_new[tree_id])
            tree_copy[TR_ID] = i + 1
            pop_y.append(tree_copy)

        return pop_y

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   What happens in a Generation              |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gen_prepare_parameters(self):
        """
        Sets the parameters for this generation
        - reset population_new
        - Lineary increase threshold for parsimony
        """
        self.time_gen = time.perf_counter()
        self.debug_warnings = ''

        self.printpl('g', 'Preparing to evolve Generation', self.gen_id, '...')
        self.population_new = ['Plagih GP - Evolving Generation']  # initialise population_new to host the next generation

        full_parsimony_factor = 2  # working with the maximum parsimony for at least some generations
        gen_relation = min((full_parsimony_factor * self.gen_id) / self.gp['gen_max'], 1)
        self.parsimony_min_max[0] = int(gen_relation * self.parsimony_min_max[1])

        return

    def gen_reproduce(self, repro_rate, tourn_size):

        """
        A single Tree from the prior generation is copied without mutation
        """
        self.printpl('gg', 'Reproduce...')
        time_start = time.perf_counter()

        for n in range(repro_rate):  # quantity of Trees to be copied without mutation
            tourn_winner = self.gp_selection_tournament(self.gp['gp_tourn_size'])
            tourn_winner[TR_type][1] = 'r'
            self.population_new.append(tourn_winner)  # append array to next generation population of Trees

        self.printpl('gg', 'gen_reproduce took: {:4.2f}'.format(time.perf_counter() - time_start))

        return

    def gen_mutate_point(self):

        """
        One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """
        self.printpl('gg', 'Point Mutation...')
        time_start = time.perf_counter()

        for i in range(self.evolve_rates['mutate_point']):  # quantity of Trees to be generated through mutation
            tree = self.gp_selection_tournament(self.gp['gp_tourn_size'])
            tree, node = self.treegp_mutate_point_evolve(tree)
            self.population_new.append(tree)

        self.printpl('gg', 'gen_mutate_point took: {:4.2f}'.format(time.perf_counter() - time_start))

        return

    def gen_mutate_branch(self):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """
        self.printpl('gg', 'Branch Mutation...')
        time_start = time.perf_counter()
        time_tmp = time.perf_counter()

        mutate_step_times = [0, 0, 0, 0, 0]

        for i in range(self.evolve_rates['mutate_branch']):  # quantity of Trees to be generated through mutation

            mutate_step_times[0] += time.perf_counter() - time_tmp
            tourn_winner = self.gp_selection_tournament(self.gp['gp_tourn_size'])  # perform tournament selection for each mutation
            branch_nodes_ids = self.ptree_choose_branch_ids(tourn_winner)  # select point of mutation and all nodes beneath [6, 9, 10]
            tourn_winner = self.evolve_subtree_build(tourn_winner, branch_nodes_ids)
            tourn_winner = self.tree_modifyable_nodes_set(tourn_winner)

            self.population_new.append(tourn_winner)  # append array to next generation population of Trees

        self.printpl('gg', 'gen_mutate_point took: {:4.2f}'.format(time.perf_counter() - time_start))

        return

    def gen_crossover(self):

        """
        TODO currently, partners do not exchange their branches, just parent a takes a branch of b_parent
        - select parent a and b
        - select swappable branche for a_parent from b_parent
            - select a node in a (and crossover here, no matter what)
        - delete a_parent branch and insert b_parent branch (which tactic?)

        """
        self.printpl('gg', 'Crossover...')
        time_start = time.perf_counter()

        for n in range(self.evolve_rates['crossover']):

            # 1. Select two parents and their branches
            left_p = self.gp_selection_tournament(self.gp['gp_tourn_size'])  # perform tournament selection for 'a_parent'
            right_p = self.gp_selection_tournament(self.gp['gp_tourn_size'])  # perform tournament selection for 'b_parent'

            # 2. Get the branches for left and right that can be exchanged
            left_ids, right_ids, x_convert_bool = self.treegp_crossover_get_swap_branches(left_p, right_p)

            # 3. is a forced conversion needed between the parents?
            if x_convert_bool:
                self.printpl('w', 'Forced conversion is needed between two trees.')
                offspring = self.treegp_crossover_insert(left_p, left_ids, right_p, right_ids, left_cast=x_convert_bool)
            else:
                offspring = self.treegp_crossover_insert(left_p, left_ids, right_p, right_ids)

            offspring = self.tree_modifyable_nodes_set(offspring)
            self.population_new.append(offspring)  # append the 1st child to next generation of Trees

        self.printpl('gg', 'gen_mutate_point took: {:4.2f}'.format(time.perf_counter() - time_start))

        return

    def gen_finalize(self):

        """
        From raw population_new to new population_genepool
        - Gene_pool with tree's parsimony (and store info in the tree)
        -

        """
        self.printpl('gg', 'Finalizing...', time.perf_counter() - self.time_gen)
        self.pop_enum_trees(self.population_new)  # pop +tree_id
        gene_pool_hash_dict = self.pop_genepool_create(self.population_new)
        # self.pop_tree_store_fitness_parsimony_analysis(self.population_new, gene_pool_hash_dict)     # gene +fitness

        self.pop_parsimony_best_update(gene_pool_hash_dict)
        self.pop_pareto_update()

        self.population_base = self.pop_copy_genepool(self.population_new, gene_pool_hash_dict)
        self.file_population_write(self.population_new, 'new')

        self.printpl('p', 'Time needed for this Generation:', time.perf_counter() - self.time_gen)

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Perform the 3 genetic prog. operations    |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gp_selection_tournament(self, tourn_size):

        """
        gp-selection. takes a number of trees (we use 3) and returns the best one (winner)
        Uses:
            self.population a
            self.genepool a
        """

        # Start with dummies
        best_id = -1
        best_fitness = self.fitness_bad_dummy

        # Get several values
        for n in range(tourn_size):

            tree_id = self.pop_util_random(self.population_base)

            fitness = float(self.population_base[tree_id][TR_fitness][1])  # extract the fitness from the array
            fitness = round(fitness, self.precision)  # force 'result' and 'solution' to the same number of floating points

            if self.fitness_compare(fitness, best_fitness, mode='better'):
                best_id = tree_id
                best_fitness = fitness

        tourn_winner = self.pop_util_tree_copy(self.population_base, best_id)

        return tourn_winner

    def pop_util_tree_copy(self, population, tree_id):
        """
        copy a tree from a population
        """
        return np.copy(population[tree_id])

    def pop_util_random(self, population):
        """
        Returns a random tree_id from a population
        """
        return np.random.randint(1, len(population))

    def gp_mutate_termfilter(self, constant, term_type='', filter='gaussian_filter'):
        """
        When this happens, constants get a a small variance
        """

        if term_type == 'float':
            if filter == 'gaussian_filter':
                constant = np.random.normal(constant, 0.1)
            else:
                self.printpl('w', 'Warning: Filter  not specified. Please specify a filter.')
                constant = np.random.normal(constant, 0.1)

        if term_type == 'int':
            constant = int(np.random.normal(constant, 2))

        if term_type == 'bool':
            constant = not constant
            # random by 50:50?

        return constant

    def treegp_mutate_point_evolve(self, tree, arity='same'):

        """
        Mutate a single mutatable point in any Tree.
        """

        # 1. choose a node
        node_id = self.evolve_choose_mutatable_node_id(tree, mode='mutate_point')
        label = tree[N_label][node_id]
        node_xtype = self.xtype_label_get_xtype(tree[N_label][node_id])  # '>' -> 'f2b'

        if arity == 'same':
            # 2. perform point mutation on that specific node
            if int(tree[N_arity][node_id]) > 0:
                tree[N_label][node_id] = self.evolve_func_get_func(label)  # Function is same type, same arity
            elif int(tree[N_arity][node_id]) == 0:  # aka a terminal
                tree[N_label][node_id] = self.xtype_choose_term(node_xtype)  # 3 -> '2f' -> 5
            else:
                self.printpl('e', 'Operator type is not specified for PLAGIH ("term", "func",...)', tree[N_type][node_id])
                raise
        elif arity == 'plagih_switcharoo':
            self.printpl('e', 'SFEH this is TODO')
        else:
            self.printpl('e', 'treegp_mutate_point_evolve dies not know this method to handle the arity:', arity)

        return tree, node_id  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping

    # def tree_branch(self, root_xype, max_depth=''):
    #
    #     """
    #     Builds a new 'branch'-tree
    #     TODO max_depth or max_nodes
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
    #         # TODO set tree_depth_base in tree.
    #         return

    def treegp_mutate_branch_terminal_build(self):

        """
        Build the Terminal nodes for the tree.

        """

        self.tnode[N_depth] = self.tnode[TR_depth]  # set the final node_depth (same as 'gp.pop[N_depth]' + 1)

        for j in range(1, len(self.tree[N_id])):  # go through all nodes
            if int(self.tree[N_depth][j]) == self.tnode[N_depth] - 1:  # this node is a parent
                for k in range(1, (int(self.tree[N_arity][j]) + 1)):  # increment through each degree of arity for each parent node
                    self.tnode[N_parent] = int(self.tree[3][j])  # set the parent 'node_id'  ...
                    self.treegp_branch_terminal(op_xtype_dict[self.tree[N_label][j]])  # ... generate a Terminal node

        return

    def treegp_branch_terminal(self, terminal_xtype):

        """
        Generate a single Terminal node.

        """

        self.xtype_xtype_get_terminal(terminal_xtype)
        self.tnode[N_c1] = ''
        self.tnode[N_c2] = ''
        self.tnode[N_c3] = ''

        self.tree = self.ptree_node_add_frominstance(self.tree)  # commit new node to array

        return

    # def treegp_branch_node(self, parent_arity_sum, prior_sibling_arity, prior_siblings, xtype):
    #
    #     """
    #     Generate a single node (func or term) for
    #
    #     """
    #
    #     if np.random.choice(['func', 'term']) == 'func':  # randomly selected as Function
    #         label, arity = self.xtype_choose_func(xtype)
    #         self.xtype_choose_func()
    #         self.pnode_function_select(pnode, xtype)  # retrieve a function, input-reverse the parent-function (f2b -> we need 2f input)
    #         self.tnode = self.treegp_node_link_child(self.tree, self.tnode, parent_arity_sum, prior_sibling_arity, prior_siblings)  # establish links to children
    #     else:
    #         self.xtype_xtype_get_terminal(xtype)  # was here
    #         self.tnode[N_c1] = ''
    #         self.tnode[N_c2] = ''
    #         self.tnode[N_c3] = ''
    #
    #     self.tree = self.ptree_node_add_frominstance(self.tree)  # commit new node to array
    #     prior_sibling_arity = prior_sibling_arity + self.tnode[N_arity]  # sum the arity of prior siblings
    #
    #     return prior_sibling_arity

    def treegp_node_link_child(self, tree, tr_node, parent_arity_sum, prior_sibling_arity, prior_siblings):

        """
        Fill in the ptree-nodes metadata

        """
        if len(tree[3]) < 2:
            print('WHAAT \n{}'.format(tree))
        for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'
            if int(tree[N_depth][n]) == tr_node[N_depth] - 1:  # find all nodes that reside at the prior (parent) 'node_depth'
                c_buffer = tr_node[N_id] + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

                if tr_node[N_arity] == 0:  # terminal in a Grow Tree
                    tr_node[N_c1] = ''
                    tr_node[N_c2] = ''
                    tr_node[N_c3] = ''

                elif tr_node[N_arity] == 1:  # 1 child
                    tr_node[N_c1] = c_buffer
                    tr_node[N_c2] = ''
                    tr_node[N_c3] = ''

                elif tr_node[N_arity] == 2:  # 2 children
                    tr_node[N_c1] = c_buffer
                    tr_node[N_c2] = c_buffer + 1
                    tr_node[N_c3] = ''

                elif tr_node[N_arity] == 3:  # 3 children
                    tr_node[N_c1] = c_buffer
                    tr_node[N_c2] = c_buffer + 1
                    tr_node[N_c3] = c_buffer + 2

                else:
                    self.printpl('e', 'tree_build_child_link: pop[N_arity] = {}'.format(self.tnode[N_arity]))

        return tr_node

    def ptree_init(self):
        ptree = np.array(
            [['tree_id'],
             ['tree_type'],
             ['tree_depth_base'],
             ['node_id'],
             ['node_depth'],
             ['node_type'],
             ['node_label'],
             ['node_parent'],
             ['node_arity'],
             ['node_c1'],
             ['node_c2'],
             ['node_c3'],
             ['fitness'],
             ['node_modify'],
             ['parsimony']])
        return ptree

    def tree_branch_labels(self, tree, branch):

        """
        This method prepares a stand-alone Tree as a copy of the given branch.

        """
        label_list = []
        arity_list = []
        type_list = []
        for node_id in branch:

            label_list.append(tree[N_label][node_id])
            arity_list.append(tree[N_arity][node_id])
            type_list.append(tree[N_type][node_id])

        return label_list, arity_list, type_list

    def treegp_crossover_tree_prune(self, tree, depth):
        """
        reduces the depth of a Tree (in case it is too deep).
        Arguments required: tree, depth
        """

        nodes = []

        for n in range(1, len(tree[3])):

            if int(tree[N_depth][n]) == depth and int(tree[N_arity][n]) > 0:
                tree[N_type][n] = 'term'  # mutate type 'func' to 'term'
                node_xtype = self.xtype_label_get_xtype(tree[N_label][n])
                tree[N_label][n] = self.xtype_choose_term(node_xtype)  # replace label

            elif int(tree[N_depth][n]) > depth:  # record nodes deeper than the maximum allowed Tree depth
                nodes.append(n)

            else:
                pass  # as int(tree[N_depth][n]) < depth and will remain untouched

        tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
        tree = evolve_node_arity_fix(tree)  # fix all node arities

        return tree

    def treegp_crossover_get_partner_node_id(self, function_label, partner_tree, partner_branch_id, mode='same_type'):
        """
        -> Crossover: Returns a node_id in the partner tree, that can be swapped
        """

        node_xtype = self.xtype_label_get_xtype(function_label)
        node_options = []
        # TODO check if the tree is large enough?
        if mode == 'same_type':  # only return a node with the same function type
            for i, label in enumerate(partner_tree[N_label][1:]):
                if self.xtype_label_get_xtype(label) == node_xtype:
                    node_options.append(i + 1)  # +1, we skipped the first element

            if node_options:  # Found at least one!
                np.random.shuffle(node_options)  # otherwise, the first closest element is always taken (-> smallest)
                return min(node_options, key=lambda x: abs(x - partner_branch_id))  # return closest node
            else:
                return 0  # No matching node found :(
        elif mode == 'random':
            self.printpl('t', 'mode: Do the same as in the upper function, but choose randomly?')
        else:
            self.printpl('e', 'Mode not found', mode)
            raise

    def treegp_crossover_get_swap_branches(self, a_parent, b_parent):
        """
        Returns two branches (node ids) that can be replaced and a converter (if needed)

        - Option1: Try swapping based on xtype
            - Choose a random node in tree a
            - Has b an equal xtype-node?
            -> yes: swap them, RETURN
        - Option2: Try swapping with the other xtype
            - Choose a random node in tree a of different xtype
            - Has b an equal xtype-node?
            -> yes: swap them, RETURN
        - Option3: No matching xtypes. Force conversion.
            - Choose a random node in tree a
            - Choose a random node in tree b
            - Convert b_parent's node to a_parent's node
        -> only returns x_convert_bool if we need to force a conversion
        """

        a_node = self.evolve_choose_mutatable_node_id(a_parent, mode='crossover_no_root')
        a_xtype = self.xtype_label_get_xtype(a_parent[N_label][a_node])
        x_convert_bool = False
        try:  # swapping with random xtype
            b_node = self.evolve_choose_mutatable_node_id(b_parent, mode='crossover', same_xtype=a_xtype)
            mode = 'done'
        except BaseException:  # no matching xtypes. maybe the other one?
            if '2f' in a_xtype:
                a_xtype = '2b'
            elif '2b' in a_xtype:
                a_xtype = '2f'

            try:  # swapping with other xtype
                a_node = self.evolve_choose_mutatable_node_id(a_parent, mode='crossover_no_root', same_xtype=a_xtype)  # now
                b_node = self.evolve_choose_mutatable_node_id(b_parent, mode='crossover', same_xtype=a_xtype)  # also a_xtype
            except BaseException:  # no matching xtypes. maybe the other one?

                a_node = self.evolve_choose_mutatable_node_id(a_parent, mode='crossover_no_root')
                b_node = self.evolve_choose_mutatable_node_id(b_parent, mode='crossover')
                a_xtype = self.xtype_label_get_xtype(a_parent[N_label][a_node])
                b_xtype = self.xtype_label_get_xtype(b_parent[N_label][b_node])
                # check if the two labels are compartibel
                if self.xtype_outcome_equi_test(a_xtype, b_xtype):
                    self.printpl('w', 'Crossover: Forcing conversion between', a_parent[N_label][a_node], b_parent[N_label][b_node])
                    x_convert_bool = True

        # TODO add try-except-case with point mutation (same arity, swapping xtype)
        a_branch = self.ptree_choose_branch_ids(a_parent, node=a_node)
        b_branch = self.ptree_choose_branch_ids(b_parent, node=b_node)

        return a_branch, b_branch, x_convert_bool

    def treegp_crossover_insert(self, left_parent, left_branch, right_parent, right_branch, left_cast=False):

        """
        Perform a crossover between nodes that are crossoverable in terms of function types
        get: parent x, y and their branches
        return: puts right_branch into parent_x
        """

        right_top_id = int(right_branch[0])
        left_top_id = int(left_branch[0])

        if len(right_branch) == 1:  # if branch of new parent contains only one node (terminal)
            if left_cast:
                new_label = right_parent[N_label][right_top_id]
            else:
                new_label = right_parent[N_label][right_top_id]

            left_parent[N_label][left_top_id] = new_label  # replace label with that of a particular node in 'right_branch'
            left_parent[N_type][left_top_id] = 'term'  # replace type
            left_parent[N_arity][left_top_id] = 0  # set terminal arity

            left_parent = np.delete(left_parent, left_branch[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
            left_parent = self.tree_fix_link_child(left_parent)  # fix all child links
            left_parent = evolve_node_renum_karoo(left_parent)  # renumber all 'node_id's

        else:  # we are working with a branch from 'parent' >= depth 1 (min 2 nodes)
            if left_cast:
                self.printpl('e', 'This is a large TODO')

                # TODO
                # left_xtype = self.xtype_label_get_xtype(main_parent[N_label][main_root_node])
                # b_xtype = self.xtype_label_get_xtype(right_parent[N_label][b_root_node])
                # func_convert = self.xtype_get_converter(main_xtype, b_xtype)
                # flist = [func_convert].extend()
            else:
                pass
            label_list, arity_list, type_list = self.tree_branch_labels(right_parent, right_branch)

            right_core = tree_from_labels(label_list, arity_list, type_list)
            left_parent = tree_insert_subtree(left_parent, right_core, left_branch, wrapper=True)
            # left_parent = self.tree_replace_branch_nodelist(left_parent, left_branch, right_core)  # insert new nodes at point of mutation 'branch_top' in tourn_winner 'offspring'
            left_parent = self.treegp_crossover_tree_prune(left_parent, self.gp['tree_depth_max'])  # prune to the max Tree depth + adjustment

        return left_parent

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Utility  functions to evolve a tree       |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def tree_create_core_from_labels(self, xtype, depth_goal):
        """
        build a random tree based on a base depth and 50% chance for every node to become a terminal
        """
        todo_xtypes = [xtype]
        result_label_list = []
        result_arity_list = []
        result_type_list = []

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
                    result_type_list.append('term')

            else:
                for t in todo_xtypes:

                    # Randomly choose a new label
                    if np.random.choice(['fun', 'trm']) == 'trm':
                        label = self.xtype_choose_term(t)
                        arity = 0
                    else:
                        label, arity = self.xtype_choose_func(t)

                    # xtype-'To-do' list for the next depth to give values to these functions
                    if label == 'Ifte':
                        next_xtype_list.extend(['2b', '2f', '2f'])
                    else:
                        tmp_xtype = self.xtype_label_get_xtype(label)
                        child_type = tmp_xtype[:2][::-1]  # the input of our function "reverted" is the xtype
                        for _ in range(0, arity):  # when arity==2, add 2 times
                            next_xtype_list.append(child_type)

                    # Add the label to the result list
                    result_label_list.append(label)
                    result_arity_list.append(arity)
                    result_type_list.append(self.op_label_get_basictype(label))

            # Finally, update the list for the next round
            todo_xtypes = next_xtype_list[:]

        # print('Result_label_list', result_label_list, 'result_arity_list', result_arity_list)
        return result_label_list, result_arity_list, result_type_list

    def evolve_subtree_build_dbu(self, tree, branch_ids):
        """
        builds subtree with depth_base_uniform method
        """
        top_depth = tree[N_depth][branch_ids[0]]
        depth_upper_bound = self.gp['tree_depth_max'] - int(top_depth)
        depth_goal = min(self.gp['tree_depth_base'], depth_upper_bound)

        label = tree[N_label][branch_ids[0]]
        xtype = self.xtype_label_get_xtype(label)

        label_list, arity_list, type_list = self.tree_create_core_from_labels(xtype, depth_goal)
        core_insert = tree_from_labels(label_list, arity_list, type_list)
        result_tree = tree_insert_subtree(tree, core_insert, branch_ids, wrapper=True)

        if not tree_test_plausibility(result_tree):
            self.printpl('e', 'Tree values are not plausible!\n{} \n and core\n{}'.format(result_tree, core_insert))

        return tree

    def evolve_subtree_build(self, tree, branch_ids):

        """
        Given: Tree and a node list
        - checks how far to build down
        - checks the old nodes xtype, etc.
        - checks if we are not too far down the tree
        -

        returns: new tree
        """

        grow_method = self.gp['tree_growth']

        if grow_method == 'depth_base_uniform':
            """
            We allow base depth (which is a little lower than max)
            but every node has 0.5 chance to become a terminal
            - iterate over depths
            - fill with as many funcs as possible
            """
            tree = self.evolve_subtree_build_dbu(tree, branch_ids)
            return tree
        elif grow_method == 'old plagih code':
            self.printpl('e', 'Not yet')
            #
            # width_goal = 1
            # nodes_cnt = 0  # reset for 'c_buffer' in 'children_link'
            # prior_s = 0  # reset for 'c_buffer' in 'children_link'
            #
            # # go as far wide as needed
            # for count in range(1, width_goal + 1):
            #
            #     # Check, how many of the lower nodes
            #     if label == 'Ifte':
            #         # build up "parent" list
            #         nodes_cnt = self.treegp_branch_node(width_goal, nodes_cnt, prior_s, '2b')
            #         prior_s += 1
            #         nodes_cnt = self.treegp_branch_node(width_goal, nodes_cnt, prior_s, '2f')
            #         prior_s += 1
            #         nodes_cnt = self.treegp_branch_node(width_goal, nodes_cnt, prior_s, '2f')
            #         prior_s += 1
            #     else:
            #         pnode[N_parent] = todo_xypes[count]  # set the nodes parent
            #         parent_func_xtype = op_xtype_dict[tree[N_label][pnode[N_parent]]]  # find parents node
            #         xype = parent_func_xtype[:2][::-1]
            #         nodes_cnt = self.treegp_branch_node(width_goal, nodes_cnt, prior_s, xype)
            #         prior_s += 1
            #
            #
            #     pnode[N_label] = label
            #     # needen: label, c1, c2, c3, depth, parent, arity, depth, type, modify
            #
            # for count in range(1, len(tree[N_depth])):  # increment through all nodes in array 'tree'
            #     if int(tree[N_depth][count]) == pnode[N_depth] - 1:  # find parent nodes which reside at the prior depth
            #         width_goal = width_goal + int(tree[N_arity][count])  # sum arities of all parent nodes at the prior depth
            #
            # # how many nodes
            # pnode = self.pnode_update()
            #
            # self.tnode = self.treegp_branch_functions(tree, pnode)  # build all the Function nodes
            #
            # pnode = self.pnode_asdf(pnode, label, arity)
            #
            # if branch_depth == 0:  # the point of mutation ('branch_top') chosen resides at the maximum allowable depth, so mutate term to term
            #     tree[N_label][top_node_id] = self.xtype_choose_term(xype)
            #
            # else:
            #     self.tree_branch(xype, max_depth=branch_depth)  # build new Tree ('gp.tree') with a maximum depth which matches 'branch'
        elif grow_method == 'nodes_max_uniform':
            """
            We allow a certain amount of new nodes instead tree depth.
            This could be calculated respectively to the parsimony level
            which the tree might have up his sleeve
            """
            raise
        else:
            self.printpl('e', 'That did not work')

        return tree

    def evolve_subtree_depth_choose(self, ptree, top_id, bottom_id, amount_replaced_nodes, mode='base_depth'):  # sfeh other default
        """
        Return the size of the tree to be inserted.
        Should not be set to maximum to reduce complexity!
        """

        # TODO consider tree size of last tree,
        # TODO consider random tree size,
        # TODO consider always maximum tree size,
        # TODO is this already considered by 50:50 func-term?

        depth_old = int(ptree[N_depth][bottom_id]) - int(ptree[N_depth][top_id])  # subtract depth of 'branch_top' from the last in 'branch'
        depth_upper_bound = self.gp['tree_depth_max'] - int(ptree[N_depth][top_id])  # = 10 - (node_depth)
        if mode == 'maximum':
            branch_depth = depth_upper_bound
        elif mode == 'same_length':
            branch_depth = depth_old
        elif mode == 'base_depth':
            branch_depth = min(self.gp['tree_depth_base'], depth_upper_bound)
        elif mode == 'random':
            branch_depth = min(depth_upper_bound, np.random.randint(0, 1 + max(depth_upper_bound, 3)))  # SFEH random depth, I hope this is enough to guarantee tree size
        else:
            self.printpl('e', 'evolve_subtree_depth_choose does not accept this mode: {}'.format(mode))
            raise
        return branch_depth

    def tree_replace_branch_nodelist(self, karoo_tree, tree_ids, insert_tree):

        """
        This method enables the insertion of ptree_branch_node_ids in place of a branch
        ptree_branch_node_ids = [5,6,8,9] node that are changed

        The end result is a Tree with a mutated branch.
        """

        branch_top = int(tree_ids[0])
        karoo_tree[N_label][branch_top] = insert_tree[N_label][1]  # copy node_label from new karoo_tree
        karoo_tree[N_arity][branch_top] = insert_tree[N_arity][1]  # copy node_arity from new karoo_tree
        karoo_tree = np.delete(karoo_tree, tree_ids[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')

        c_buffer = self.evolve_c_buffer(karoo_tree, branch_top)  # generate c_buffer for point of mutation ('branch_top')
        karoo_tree = tree_insert_node_child_dummies(karoo_tree, branch_top, c_buffer, wrapper=True)  # insert a single new node ('branch_top')
        karoo_tree = evolve_node_renum_karoo(karoo_tree)  # renumber all 'node_id's

        ### PART 2 - insert b_branchody from 'gp.karoo_tree' into 'karoo_tree' ###
        node_count = 2  # set node count for 'gp.karoo_tree' to 2 as the new root has already replaced 'branch_top' (above)

        while node_count < len(insert_tree[3]):  # increment through all nodes in the new Tree ('gp.karoo_tree'), starting with node 2

            for j in range(1, len(karoo_tree[3])):  # increment through all nodes in tourn_winner ('karoo_tree')

                if karoo_tree[N_type][j] == '':
                    karoo_tree[N_type][j] = insert_tree[N_type][node_count]  # copy 'node_type' from branch to karoo_tree
                    karoo_tree[N_label][j] = insert_tree[N_label][node_count]  # copy 'node_label' from branch to karoo_tree
                    karoo_tree[N_arity][j] = insert_tree[N_arity][node_count]  # copy 'node_arity' from branch to karoo_tree

                    if karoo_tree[N_arity][j] == '0':  # terminal
                        karoo_tree = self.tree_fix_link_child(karoo_tree)  # fix all child links
                        karoo_tree = evolve_node_renum_karoo(karoo_tree)  # renumber all 'node_id's

                    if int(karoo_tree[N_arity][j]) > 0:
                        c_buffer = self.evolve_c_buffer(karoo_tree, j)  # generate 'c_buffer' for point of mutation ('branch_top')
                        karoo_tree = tree_insert_node_child_dummies(karoo_tree, j, c_buffer, wrapper=True)  # insert new nodes
                        karoo_tree = self.tree_fix_link_child(karoo_tree)  # fix all child links
                        karoo_tree = evolve_node_renum_karoo(karoo_tree)  # renumber all 'node_id's

                    node_count = node_count + 1  # exit loop when 'node_count' reaches the number of columns in the array 'gp.karoo_tree'

        return karoo_tree

    def evolve_subtree_insert_child(self, tree, node, c_buffer):

        """
        Insert child node into the copy of a parent Tree.

        """

        if int(tree[N_arity][node]) == 0:  # if arity = 0
            self.printpl('e', 'In evolve_child_insert: node', node, 'has arity 0')
            self.plagih_pause()  # consider special instructions for this

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
            self.printpl('e', 'In evolve_child_insert: node', node, 'arity > 3')
            self.plagih_pause()  # consider special instructions for this (pause)

        return tree

    def ptree_choose_branch_ids(self, tree, node=None):

        """
        chooses a mutatable branch to mutate
        - specify a starting node
        - return all child-nodes as list
        """

        branch = np.array([])  # the array is necessary in order to len(branch) when 'branch' has only one element

        if node:  # Crossover: Option to specify own starting node
            branch_top = node
        else:
            branch_top = self.evolve_choose_mutatable_node_id(tree, mode='mutate_branch_no_root')  # "2" returns mutable node (except root node)

        # 2. Also return all child nodes
        branch_eval = self.tree_node_get_childlist(tree, branch_top)  # generate tuple of 'branch_top' and subsequent nodes
        branch_symp = plagih_sympify(branch_eval)  # convert string into something useful # sfeh: simple sympy might be faster
        branch = np.append(branch, branch_symp)  # append list to array
        branch = np.sort(branch)  # sort nodes in branch for Crossover.

        return branch

    def evolve_c_buffer(self, tree, node):

        """
        Generates the c_buffer for a node of a ptree

        """

        parent_arity_sum = 0
        prior_sibling_arity = 0
        prior_siblings = 0

        for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'

            if int(tree[N_depth][n]) == int(tree[N_depth][node]) - 1:  # find parent nodes at the prior depth
                if tree[N_arity][n] != '':
                    parent_arity_sum = parent_arity_sum + int(tree[N_arity][n])  # sum arities of all parent nodes at the prior depth

            if int(tree[N_depth][n]) == int(tree[N_depth][node]) and int(tree[3][n]) < int(tree[3][node]):  # find prior siblings at the current depth
                if tree[N_arity][n] != '':
                    prior_sibling_arity = prior_sibling_arity + int(tree[N_arity][n])  # sum prior sibling arity
                prior_siblings = prior_siblings + 1  # sum quantity of prior siblings

        c_buffer = node + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

        return c_buffer

    def tree_fix_link_child(self, tree):

        """
        In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

        This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.

        """

        for node_id in range(1, len(tree[3])):
            c_buffer = evolve_c_buffer(tree, node_id, wrapper=True)  # generate c_buffer for each node
            tree = evolve_fix_link_child_doit(tree, node_id, c_buffer, wrapper=True)  # update child links for each node

        return tree

    def evolve_fix_link_child_doit(self, tree, node, c_buffer):

        """
        Link each parent node to its children.

        """

        if int(tree[3][node]) == 1:
            # SFEH Root can only be ignored, if root was not changed
            c_buffer = c_buffer + 1  # if root (node 1) is passed through this method

        if tree[N_arity][node] != '':

            if int(tree[N_arity][node]) == 0:  # if arity = 0
                tree[9][node] = ''
                tree[10][node] = ''
                tree[11][node] = ''

            elif int(tree[N_arity][node]) == 1:  # if arity = 1
                tree[9][node] = c_buffer
                tree[10][node] = ''
                tree[11][node] = ''

            elif int(tree[N_arity][node]) == 2:  # if arity = 2
                tree[9][node] = c_buffer
                tree[10][node] = c_buffer + 1
                tree[11][node] = ''

            elif int(tree[N_arity][node]) == 3:  # if arity = 3
                tree[9][node] = c_buffer
                tree[10][node] = c_buffer + 1
                tree[11][node] = c_buffer + 2

            else:
                self.printpl('e', 'evolve_child_link: node', node, 'has arity', tree[N_arity][node])
                raise  # self.plagih_pause()  # consider special instructions for this (pause)

        return tree

    def evolve_fix_link_parent(self, tree):

        """
        In a given Tree, fix 'parent_id' for all nodes.

        This is automatically handled in all mutations except with Crossover due to the need to copy branches 'a' and
        'b' to their own trees before inserting them into copies of	the parents.

        Technically speaking, the 'node_parent' c1 is not used by any methods. The parent ID can be completely out
        of whack and the expression will work perfectly. This is maintained for the sole purpose of granting the user
        a friendly, makes-sense interface which can be read in both directions.

        Called by: evolve_branch_copy

        Arguments required: tree
        """

        ### THIS METHOD MAY NOT BE REQUIRED AS SORTING 'branch' SEEMS TO HAVE FIXED 'parent_id' ###

        for node in range(1, len(tree[3])):

            if tree[9][node] != '':
                child = int(tree[9][node])
                tree[7][child] = node

            if tree[10][node] != '':
                child = int(tree[10][node])
                tree[7][child] = node

            if tree[11][node] != '':
                child = int(tree[11][node])
                tree[7][child] = node

        return tree

    def evolve_node_renum(self, tree):

        """
        Renumber all 'node_id' in a given tree.

        This is required after a new generation is evolved as the node_id numbers are carried forward from the previous
        generation but are no longer in order.

        """

        for n in range(1, len(tree[3])):
            tree[3][n] = n  # renumber all nodes

        return tree

    def evolve_choose_mutatable_node_id(self, tree, mode='', same_xtype=''):
        """
        Returns a mutatable node for point-mutation
        -> no_root handles
        """
        # TODO only works for 2-array functions

        node_ids = []

        # 1. Build up a list with nodes
        if same_xtype:
            for i, label in enumerate(tree[N_label]):
                if tree[TRn_modify][i] == '1':  # also skips node 0
                    # TODO make this faster
                    node_xtype = self.xtype_label_get_xtype(tree[N_label][i])
                    if self.xtype_outcome_equi_test(node_xtype, same_xtype):
                        node_ids.append(int(tree[3][i]))
        else:
            for i, x in enumerate(tree[N_type]):
                if tree[TRn_modify][i] == '1':
                    node_ids.append(int(tree[3][i]))

        # 2. Kick out root if it is there?
        if 'no_root' in mode:  # delete root node
            node_ids = [x for x in node_ids if x != 1]

        # 3: return the node. Not safe, could be try-except block.
        # eg: all nodes are not modifiable
        # eg. all nodes are not of correct type
        node_id = np.random.choice(node_ids)
        return node_id

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def ptree_node_add_frominstance(self, tree):

        """
        Commit the values of a new node (root, function, or terminal) to the array 'tree'.
        TODO
        """

        tree = np.append(tree, [[self.tnode[TR_ID]],
                                [self.tnode[TR_type]],
                                [self.tnode[TR_depth]],
                                [self.tnode[N_id]],
                                [self.tnode[N_depth]],
                                [self.tnode[N_type]],
                                [self.tnode[N_label]],
                                [self.tnode[N_parent]],
                                [self.tnode[N_arity]],
                                [self.tnode[N_c1]],
                                [self.tnode[N_c2]],
                                [self.tnode[N_c3]],
                                '',  # [self.tnode[TR_fitness]],
                                ['1'],
                                '',  # [self.tnode[TR_parsimony]]
                                ], 1)

        self.tnode[N_id] = self.tnode[N_id] + 1

        return tree

    def ptree_node_add_fromvalues(self, tree, node_id, node_depth,
                                  node_type, node_label, node_parent, node_arity, node_c1,
                                  node_c2, node_c3):
        np.append(tree,
                  ['', '', '', [node_id], [node_depth], [node_type],
                   [node_label], [node_parent], [node_arity], [node_c1], [node_c2], [node_c3],
                   '', '', ''], 1)
        return tree

    def ptree_node_add_fromdnode(self, tree, dnode):
        np.append(tree, ['',
                         '',
                         '',
                         dnode[N_id],
                         dnode[N_depth],
                         dnode[N_type],
                         dnode[N_label],
                         dnode[N_parent],
                         dnode[N_arity],
                         dnode[N_c1],
                         dnode[N_c2],
                         dnode[N_c3],
                         '',
                         '',
                         ''], 1)
        return tree

    def tree_data_load_origin_tree(self, origin_tree_file_path):
        """
        This loads the 'origin' and evaluates it

        Arguments required: path to csv
        returns: tree
        """

        # Check if the user provided an origin
        if origin_tree_file_path == '':
            # Probably the best idea is to specify the outcome only. e.g. float
            self.printpl('t', 'No origin provided. Need to rework everything for this case.')
            raise

        # Load origin from file
        with open(origin_tree_file_path, 'r') as csv_file:
            target = csv.reader(csv_file, delimiter=',')
            tree = np.array([[]])
            for row in target:
                if tree.shape[1] == 0:  # looks if tree is empty
                    tree = np.append(tree, [row], axis=1)  # append first row to Tree ('tree_id')
                else:
                    tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree
            if tree.shape[0] == TRn_um_lines:  # (+ row 0)
                pass  # print('Origin Tree is: \n' + str(tree))
            else:
                self.printpl('e', "Tree could not be imported correctly from .csv file.")
                raise
        tree[TR_parsimony][1] = 0  # the distance to itself is 0 by definition
        origin_algo_raw = self.tree_expr_raw(tree, P_first_node)
        self.origin = {'tree': tree,
                       'algo_raw': origin_algo_raw,
                       'algo_sym': self.tree_expr_sympify(algo_raw_str=origin_algo_raw),
                       'parsimony': 0}

        origin_hash, origin_meta = self.tree_store_meta_get_hash(tree)
        self.origin['fitness_train'] = origin_meta['fitness_train']

        self.hashtable_fitness_train = {}
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

            # 4. All the tree-specific meta data into the
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
        for i, tmp in enumerate(chosen_tree[TRn_modify][1:]):
            chosen_tree[TRn_modify][i + 1] = '1'

        # Find no-modifyables in Origin
        non_modifiable_nodes = []
        if self.origin['tree'][TRn_modify][1] == '0':  # check is modifiable nodes are specified
            non_modifiable_nodes.extend(self.tree_nomodifyable_nodes_get(1, chosen_tree, 1))

        for non_modifiable in non_modifiable_nodes:
            chosen_tree[TRn_modify][non_modifiable] = '0'

        return chosen_tree

    def tree_nomodifyable_nodes_get(self, origin_node, chosen_tree, chosen_node):
        """
        Returns a list of nodes that are not supposed to be modified
        """

        if self.origin['tree'][TRn_modify][origin_node] == '0':
            non_modifiables = []
            non_modifiables.append(int(chosen_tree[N_id][chosen_node]))
            for child in [9, 10, 11]:
                if self.origin['tree'][child][origin_node] != '':
                    next_origin_node = int(self.origin['tree'][child][origin_node])
                    next_chosen_node = int(chosen_tree[child][chosen_node])
                    tmp = self.tree_nomodifyable_nodes_get(next_origin_node, chosen_tree, next_chosen_node)
                    if tmp is not None:
                        non_modifiables.extend(tmp)
            return non_modifiables
        else:
            return

    def tree_build_type_constant_get(self, term_type='', mode='float-1to1', uniform_range=''):
        """
        todo random samples
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
            self.printpl('e', 'This term type should not occur, I guess', term_type)
            term_type = np.random.choice(self.variables_dict['types'])
            const = self.tree_build_type_constant_get(term_type=term_type)
        return str(const)

    def tree_expr_sympify(self, algo_raw_str='', tree=''):

        """
        returns the sympifyed expression
        """
        if len(tree) > 0:  # If we got a tree, we generate the expression
            algo_raw_str = str(self.tree_expr_raw(tree, 1))

        try:
            x = plagih_sympify(algo_raw_str)
            strx = str(x)

            if 'zoo' in strx:
                x = re.sub('zoo', '10', strx)  # TODO how to handle zoo?

            if 'nan' in strx:  # Happens when 0/0 occurs. This tree is worth nothing anyways
                self.printpl('w', 'We had a "nan"')
                self.remove_this_tree()
                return str(sympy_dummy)
            else:
                return str(x)
        except:
            self.printpl('w', 'In sympify. Caused by this raw algorithm: ' + str(algo_raw_str))
            # todo.
            self.remove_this_tree()
            return str(sympy_dummy)

    def remove_this_tree(self):
        """
        If a tree makes problems, delete it somehow.
        - set parsimony very high?
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

    def tree_node_get_childlist(self, tree, node_id):

        """
        return a list of s nodes childs.
        + Evaluate all or part of a Tree and

        This method generates a list of all 'node_id's from the given Node and below. It is used primarily to generate
        'branch' for the multi-generational mutation of Trees.
        TODO what does this exactly?
        """

        node_id = int(node_id)

        if tree[N_arity, node_id] == '0':  # arity of 0 for the pattern '[node_id]'
            return tree[3, node_id]  # 'node_id'

        else:
            if tree[N_arity, node_id] == '1':  # arity of 1 for the pattern '[node_id], [node_id]'
                return '{}, {}'.format(tree[3, node_id], self.tree_node_get_childlist(tree, tree[9, node_id]))

            elif tree[N_arity, node_id] == '2':  # arity of 2 for the pattern '[node_id], [node_id], [node_id]'
                return '{}, {}, {}'.format(
                    tree[3, node_id],
                    self.tree_node_get_childlist(tree, tree[9, node_id]),
                    self.tree_node_get_childlist(tree, tree[10, node_id]))

            elif tree[N_arity, node_id] == '3':  # arity of 3 for the pattern '[node_id], [node_id], [node_id], [node_id]'
                return '{}, {}, {}, {}'.format(
                    tree[3, node_id],
                    self.tree_node_get_childlist(tree, tree[9, node_id]),
                    self.tree_node_get_childlist(tree, tree[10, node_id]),
                    self.tree_node_get_childlist(tree, tree[11, node_id]))

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
        fitness = round(fitness, self.precision)

        tree[TR_fitness][1] = fitness  # store the fitness with each tree

        return

    def tree_store_parsimony(self, tree, parsimony):
        """
        Store the parsimony within the tree np-array
        """
        if parsimony < 0:
            self.printpl('w', 'Parsimony is:', parsimony)
        tree[TR_parsimony][1] = parsimony

    def tree_store_meta_lastgen(self, tree, modification=''):

        """
        Remove all fitness data from a given tree.

        This is required after a new generation is evolved as the fitness of the same Tree prior to its mutation will
        no longer apply.

        """

        # save information about how good last changes were
        # for i in range(min(self.tree_depth_min, 5), 2, -1):  # 5,4,3,2
        #     tree[TR_type][i] = tree[TR_type][i-1]    # The last modifications
        #     tree[TR_fitness][i] = tree[TR_fitness][i-1]  # The last fitness
        #     tree[TR_parsimony][i] = tree[TR_parsimony][i-1]  # The last parsimony (TODO) # tree_id,1,a,b,c -> tree_id,1,a,a,b

        # What needs to be assigned later
        # tree[TR_type][2] = modification  # wipe last modification data
        # tree[TR_ID][1] = ''  # -> tree_id,,
        # tree[TR_fitness][1] = ''  # wipe fitness data
        # tree[TR_parsimony][1] = ''  # wipe parsimony data

        return tree

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to use evaluate (tensorflow)      |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def eval_tf(self, expr, data, get_pred_labels=False):

        """
        computes gp-tree results and fitness scores.
        - Computes tree expression using TensorFlow (TF)
        - parsing input string 'expression' and converting it into a TF operation graph
        - processing tf graph in an isolated TF session (results and corresponding fitness)

            'self.tf_device' - controls which device will be used for computations (CPU or GPU).
            'self.tf_device_log' - controls device placement logging (debug only).

        Args:
            'expr' - a string expression to be computed on the data. Variable -> 'self.terminals'
            'data' - an 'n by m' matrix of the data points containing n observations like 'self.terminals'.
            'get_pred_labels' - (Classify Kernel) a boolean flag which controls whether the predicted labels should be
            extracted from the evolved results.

        Returns:
            A dict mapping keys to the following outputs:
                'result'            - array of the results of applying given expression to the data
                'pred_labels'       - (Classify) an array of the predicted labels extracted from the results
                'solution'          - array of the solution values extracted from the data (variable 's' in the dataset)
                'pairwise_fitness'  - array of the element-wise results of applying the fitness kernel function
                'fitness'           - aggregated scalar fitness score

        """

        # Initialize TensorFlow session
        tf.compat.v1.reset_default_graph()  # tf.reset_default_graph()
        config = tf.compat.v1.ConfigProto(log_device_placement=self.tf_device_log, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            with sess.graph.device(self.tf_device):

                # 1. data (observations, actions) to tensors
                tensors = {}

                num_terminals = len(self.variables_dict['all'])
                num_actions = len(self.actions)

                for i in range(num_terminals):
                    var = self.variables_dict['all'][i]
                    if '2f' in self.xtype_node_get_xtype(var, 'term'):
                        tensors[var] = tf.constant(data[:, i], dtype=tf.float32)  # converts data into vectors
                    else:  # '2b'
                        tensors[var] = tf.constant(data[:, i], dtype=tf.bool)

                for i in range(num_actions):
                    var = self.actions[i]
                    action_xtype = self.xtype_node_get_xtype(var, 'term')
                    if '2f' in action_xtype:
                        tensors[var] = tf.constant(data[:, num_terminals + i], dtype=tf.float32)  # converts data into vectors
                    elif '2b' in action_xtype:  # '2b'
                        self.printpl('t', 'Currently no kernel available for boolean fitness')
                        tensors[var] = tf.constant(data[:, i], dtype=tf.bool)
                    else:
                        self.printpl('e', 'Kernel not known for: {} which is {}.'.format(var, action_xtype))

                # 2- Transform string expression into TF operation graph
                tf_result = self.eval_tf_ast_expr(expr, tensors)
                pred_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

                # TODO currently does only support one label
                solution = tensors['action0']  # solution c1 is assumed to be stored in this terminal
                # 3- Add fitness computation into TF graph
                if self.kernel == 'c':  # CLASSIFY kernel

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

                    if len(self.actions) > 1:
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

                elif self.kernel == 'r':  # REGRESSION kernel

                    """
                    A very, very basic REGRESSION kernel which is not designed to perform well in the real world. It requires
                    that you raise the minimum node count to keep it from converging on the c1 of '1'. Consider writing or 
                    integrating a more sophisticated kernel.
                    """

                    pairwise_fitness = tf.abs(solution - tf_result)

                elif self.kernel == 'm':  # MATCH kernel

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

        # todo delete this
        # self.printpl('c', ('arity', self.fitness_compare_better(fitness, self.origin['fitness_train'])), 'Fitness was better than original fitness')
        # if self.fitness_compare_better(fitness, self.origin['fitness_train']):
        #     print('Fitness was better than original fitness:', fitness, ' better than:', self.origin['fitness_train'])

        return {'result': tf_result, 'pred_labels': pred_labels, 'solution': solution, 'fitness': float(fitness),  # this was changed
                'pairwise_fitness': pairwise_fitness, 'old_fitness': float(fitness)}

    def eval_tf_ast_expr(self, expr, tensors):

        """
        Extract expression tree from the string algo_sym and transform into TensorFlow (TF) graph.

        """
        # print('Current expr:', expr)  # importantprint for debugging failed expressions
        tree = ast.parse(expr, mode='eval').body

        # TODO diesen try-except block entfernen
        self.debug_warnings = str(expr)
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
            # # This was here for Min and Max. complicated stuff, did not work.
            # if node.func.id in functions_multiparam_dict:
            #     return operator_dict[node.func.id]([self.eval_tf_expr_graph(arg, tensors) for arg in node.args])

            if node.func.id == 'Ftob':
                self.printpl('i', 'float was converted to bool in tensorflow')
                return tf.dtypes.cast(
                    *[self.eval_tf_expr_graph(arg, tensors) for arg in node.args], dtype=tf.bool)
            elif node.func.id == 'Btof':
                return tf.dtypes.cast(
                    *[self.eval_tf_expr_graph(arg, tensors) for arg in node.args], dtype=tf.float32)

            if len(node.args) > 2:
                self.printpl('e', 'This has more than 2 args?', str(node.func.id))
            else:
                try:
                    return ast_tensor_dict[node.func.id](*[self.eval_tf_expr_graph(arg, tensors) for arg in node.args])
                except Exception as ex:
                    self.printpl('w', 'debug warning:', self.debug_warnings)
                    self.printpl('e', 'node.func.id caused an exception, type:\n', ex,
                                 '\nnode.func.id:\n', node.func.id,
                                 '\nnode.args:', str(node.args),
                                 '\nand expression:\n', self.debug_warnings)

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
        # sfeh idea: note: we have to convert all values to the action space if not discrete

    def eval_tf_classify_labels_map(self, result):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the data .csv. Outputs an array of tuples containing the predicted
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

    def op_label_get_basictype(self, node_label):
        """
        return terminal or function according to the label
        """

        if node_label in op_xtype_dict:
            return 'func'
        else:
            return 'term'

    def xtype_xtype_get_terminal(self, node_xtype):

        """
        Define a single Terminal (variable extracted from the top row of the associated TRAINING data)

        """

        self.tnode[N_type] = 'term'
        self.tnode[N_label] = self.xtype_choose_term(node_xtype)  # get a terminal
        self.tnode[N_arity] = 0

        return

    def evolve_func_get_func(self, label, mode='same_arity_same_type'):
        """
        returns a function for a function in point mutation
        This only accepts functions as inputs. (point mutation)
        No need to handle terminals
        """

        arity = op[label]['arity']
        xtype = op[label]['xtype']

        if mode == 'same_arity_same_type':

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
                self.printpl('e', 'Function was not found in function_types_dict', xtype)
                raise

        else:
            self.printpl('e', 'Mode not known: ', mode)

    def xtype_choose_func(self, xtype):
        """
        This fills in a function that fits the type of the function/terminal before.
        terminal  '2f' -> '_2f', arity
        function 'f2f' -> '_2f', arity
        function 'b2f2f' -> '_2f', arity
        > ->
        """
        if '2f' in xtype:
            label = np.random.choice(self.xype_func_dict['2f'])
        elif '2b' in xtype:
            label = np.random.choice(self.xype_func_dict['2b'])
        else:
            self.printpl('e', 'Warning: Function was not found in function_types_dict', xtype)
            raise

        return label, op[str(label)]['arity']

    def xtype_node_get_xtype(self, node_label, node_type):
        """
        input: (+, 'func')
        'term' or 'func'
        """

        if node_type == 'term':
            if 'True' in node_label or 'False' in node_label:
                return '2b'
            elif 'observation' in node_label:
                term_position = self.variables_dict['all'].index(node_label)
                return op_xtype_dict[self.variables_dict['types'][term_position]]
            elif 'action' in node_label:
                term_position = self.actions.index(node_label)
                return op_xtype_dict[self.action_types[term_position]]
            else:  # only 'float' left
                return '2f'
        elif node_type == 'func':
            return op_xtype_dict[node_label]
        else:
            self.printpl('e', 'This node_type is not known', node_type)
            raise

    def xtype_label_get_xtype(self, label, node_type=''):
        """
        returns xtype for a label
        todo runtime compared to xtype_node_get_xtype?
        """
        if not node_type:
            node_type = self.op_label_get_basictype(label)

        node_xtype = self.xtype_node_get_xtype(label, node_type)

        if node_type == 'term':
            if 'True' in label or 'False' in label:
                return '2b'
            elif 'observation' in label:
                term_position = self.variables_dict['all'].index(label)
                return op_xtype_dict[self.variables_dict['types'][term_position]]
            elif 'action' in label:
                term_position = self.actions.index(label)
                return op_xtype_dict[self.action_types[term_position]]
            else:  # only 'float' left
                return '2f'
        elif node_type == 'func':
            return op_xtype_dict[label]
        else:
            self.printpl('e', 'This node_type is not known', node_type)
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
            self.printpl('e', 'Probably, you have to check if your "function" is actually a terminal. xtype', node_xtype)
            raise

        if np.random.choice(['var', 'const']) == 'var':  # our choice is variable
            if terminals_type:  # Is there an entry in the list?
                return np.random.choice(terminals_type)  # ...so we return one
        return self.tree_build_type_constant_get(term_type=the_type)  # otherwise: constant (There are always constants :P)

    def xtype_outcome_equi_test(self, a_xtype, b_xtype):
        """
        Dummy. Returns, whether two xtypes are equal
        """
        return a_xtype in b_xtype or b_xtype in a_xtype

    def xtype_get_converter(self, a_xtype, b_xtype):
        """
        convert b-to-a dummy
        """
        if '2b' in a_xtype and '2f' in b_xtype:
            return 'Ftob'
        if '2f' in a_xtype and '2b' in b_xtype:
            return 'Btof'
        else:
            self.printpl('e', 'One of those two cases should happen', a_xtype, b_xtype)
            raise

    def pnode_function_select(self, pnode):

        """
        Returns a function with the same outcome

        """

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def monitor_show(self, mode=''):
        """
        monitors everything

        Helper:
        # plot_end(self, y, mode='', plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='')
        """

        if self.monitor_dict['gen_fitness_average'] == 'y':
            self.plot_end('fitness_average', plt_title='Average Fitness', plt_y_label='Fitness')

        if self.monitor_dict['genepool_size'] == 'y':
            self.plot_end('genepool_size', plt_title='Genepool size', plt_y_label='Amount')

        return

    def monitor_performance_generation(self, gene_pool_hash_dict):
        """
        Give the user some feedback
        """

        # How many survived in the selection?
        self.monitoring_dict['genepool_size'][int(self.gen_id)] = len(gene_pool_hash_dict)
        if len(gene_pool_hash_dict) > 0:
            self.printpl('gg', 'The generation has:', len(gene_pool_hash_dict), '')
        else:  # the evolutionary constraints were too tight, killing off the entire population
            self.printpl('e', 'There are no Trees in the gene pool. You should archive your population and (q)uit.')

        # What is the average fitness in our genepool?
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

    def plot_end(self, name, mode='', plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label=''):
        # insert artificial data

        # # self.plot_end('s', plt_title='Sympify zoo and nan', plt_y_label='Amount')

        # # if variance
        # if mode == 'variance':
        #     self.printpl('e', 'variance not available', y_array)
        #     means = np.mean(y, axis=0)
        #     stds = np.std(y, axis=0)
        #     n = means.size
        x, y = [], []
        for a, b in sorted(list(self.monitoring_dict[name].items())):
            x.append(a)
            y.append(b)

        x = np.arange(self.gen_id)  # sfeh -gen start id?
        plt.plot(x, y, label=plt_curve_label)

        if plt_x_label and plt_y_label:
            plt.xlabel(plt_x_label)
            plt.ylabel(plt_y_label)

        if plt_title:
            plt.title(plt_title)

        # plt.legend()
        plt.yscale('linear')
        plt.ylim(0)
        plt.xlim(0)
        # plt.savefig('save.jpg')
        # plt.show()
        # plt.close()
        plt.savefig(self.path + plt_title + '-plot.jpg')
        plt.close()
        return

    def plot_live(self):
        many_values = np.array([[6, 5, 4, 5, 6, 5, 4, 5, 5, 6, 5, 8, 5, 5, 5, 5, 5, 5, 4, 5, 6, 5],
                                [2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 8, 9, 8, 7],
                                [2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 8, 9, 8, 7],
                                [2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 6, 7, 6, 5, 4, 5, 6, 7, 7, 8, 9, 0],
                                [2, 3, 4, 5, 4, 3, 2, 3, 8, 9, 8, 7, 6, 5, 6, 7, 8, 9, 0, 9, 8, 7],
                                [2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 7, 6, 5, 4, 3, 4, 5, 6, 7, 8, 7, 1],
                                [8, 7, 6, 5, 6, 7, 8, 9, 0, 9, 8, 7, 6, 3, 4, 5, 6, 7, 8, 7, 6, 1]])

        means = np.mean(many_values, axis=0)
        stds = np.std(many_values, axis=0)
        n = means.size

        # helpers
        epsilon = 0.1
        num_plays = np.shape(many_values)[1]

        ### start episode loop
        # compute upper/lower confidence bounds
        ci = 0.95
        e = 6  # if e%200 == 0:, the current episode we are in. aka the last one for us
        test_stat = st.t.ppf((ci + 1) / 2, e)
        lower_bound = means - test_stat * stds / np.sqrt(e)
        upper_bound = means + test_stat * stds / np.sqrt(e)

        # clear plot frame
        plt.clf()

        # plot average reward
        plt.plot(means, color='blue', label="epsilon=%.2f" % epsilon)

        # plot upper/lower confidence bound
        x = np.arange(0, num_plays, 1)
        plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)

        plt.grid()
        # plt.ylim(0, 2)  # limit y axis
        plt.title('Avg. Reward per step in experiment {}: {}'.format(e, sum(means) / num_plays))
        plt.ylabel("Reward per step")
        plt.xlabel("Play")
        plt.legend()
        plt.show()
        plt.pause(0.1)
        ### end episode loop

        ## disable interactive plotting => otherwise window terminates
        plt.ioff()
        plt.show()
        return

    def plagih_pause_refer(self):

        """

        """

        menu = 1
        while menu == 1:
            menu = self.plagih_pause()

        return

    def plagih_pause(self):

        """
        Pause the program execution and engage the user, providing a number of options.

        Arguments required: [0,1,2] where (0) refers to an end-of-run; (1) refers to any use of the (pause) menu from
        within the run, and anticipates ENTER as an escape from the menu to continue the run; and (2) refers to an
        'ERROR!' for which the user may want to archive data before terminating. At this point in time, (2) is
        associated with each error but does not provide any special options).
        """

        ### PART 1 - reset and pack values to send to menu.pause ###
        menu_dict = {'input_a': '',
                     'input_b': 0,
                     'display': self.display,
                     'tree_depth_max': self.gp['tree_depth_max'],
                     'tree_pop_max': self.tree_pop_max,
                     'gen_id': self.gen_id,
                     'gen_max': self.gp['gen_max'],
                     'tourn_size': self.tourn_size,
                     'evolve_repro': self.evolve_rates['reproduce'],
                     'evolve_point': self.evolve_rates['mutate_point'],
                     'evolve_branch': self.evolve_rates['mutate_branch'],
                     'evolve_cross': self.evolve_rates['crossover'],
                     # 'fittest_dict': self.origin_dominators,
                     'pop_last_len': len(self.population_base),
                     'pop_new_len': len(self.population_new),
                     'path': self.path}

        menu_dict = menu.pause(menu_dict)  # call the external function menu.pause

        ### PART 2 - unpack values returned from menu.pause ###
        input_a = menu_dict['input_a']
        input_b = menu_dict['input_b']
        self.display = menu_dict['display']
        self.gp['gen_max'] = menu_dict['gen_max']

        self.evolve_rates['reproduce'] = menu_dict['evolve_repro']
        self.evolve_rates['mutate_point'] = menu_dict['evolve_point']
        self.evolve_rates['mutate_branch'] = menu_dict['evolve_branch']
        self.evolve_rates['crossover'] = menu_dict['evolve_cross']

        ### PART 3 - execute the user queries returned from menu.pause ###
        if input_a == 'esc':
            return 2  # breaks out of the plagih_gp() or plagih_pause_refer() loop

        elif input_a == 'eval':  # evaluate a Tree against the TEST data
            algo_sym = self.tree_expr_sympify(tree=self.population_new[input_b])  # generate the raw and sympified expression for the given Tree using SymPy
            self.printpl('o', '\n\t\033[36mTree', input_b, 'yields (sym):\033[1m', algo_sym, '\033[0;0m')  # print the sympified expression
            result = self.eval_tf(str(algo_sym), self.data_control, get_pred_labels=True)  # might change to algo_raw_str evaluation
            self.pause_fitness_test(result)  # TF tested 2017 02/02

        elif input_a == 'print_last':  # print a Tree from population_genepool
            self.display_tree(self.population_base[input_b])

        elif input_a == 'print_new':  # print a Tree from population_new
            self.display_tree(self.population_new[input_b])

        elif input_a == 'pop_last':  # list all Trees in population_genepool
            self.printpl('o', '')
            for tree_id in range(1, len(self.population_base)):
                algo_sym = self.tree_expr_sympify(tree=self.population_base[tree_id])
                self.printpl('i', '\t\033[36m Tree', self.population_base[tree_id][TR_ID][1], 'yields (sym):\033[1m', algo_sym, '\033[0;0m')

        elif input_a == 'pop_new':  # list all Trees in population_new
            self.printpl('ii', '')
            for tree_id in range(1, len(self.population_new)):
                algo_sym = self.tree_expr_sympify(tree=self.population_new[tree_id])  # extract the expression
                self.printpl('ii', '\t\033[36m Tree', self.population_new[tree_id][TR_ID][1], 'yields (sym):\033[1m', algo_sym, '\033[0;0m')

        elif input_a == 'load':  # load population_s to replace population_genepool
            pass
            # self.data_pickle_recover(self.filename['s'])  #

        elif input_a == 'write':  # write the evolving population_new to disk
            self.file_population_write(self.population_new, 'new')

        elif input_a == 'add':  # check for added generations, then exit plagih_pause and continue the run
            self.gp['gen_max'] = self.gp['gen_max'] + input_b  # if input_b > 0: self.gen_max = self.gen_max + input_b - REMOVED 2019 06/05

        elif input_a == 'quit':
            self.main_terminate()  # archive populations and exit

        return 1

    def pause_fitness_test(self, result):

        if self.kernel == 'c':
            """
            Print the Precision-Recall and Confusion Matrix for a CLASSIFICATION run against the test data.

            From scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
                Precision (P) = true_pos / true_pos + false_pos
                Recall (R) = true_pos / true_pos + false_neg
                harmonic mean of Precision and Recall (F1) = 2(P x R) / (P + R)

            From scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
                y_pred = result, the predicted labels generated by Plagih GP
                y_true = solution, the true labels associated with the data

            """
            for i in range(len(result['result'])):
                self.printpl('o', '\t\033[36m Data row {} predicts class:\033[1m {} ({} True)\033[0;0m\033[36m as {:.2f}{}\033[0;0m'.format(
                    i, int(result['pred_labels'][0][i]), int(result['solution'][i]), result['result'][i],
                    result['pred_labels'][1][i]))

            self.printpl('o', '\n Fitness score: {}'.format(result['fitness']))
            self.printpl('o', '\n Precision-Recall report:\n', skm.classification_report(result['solution'], result['pred_labels'][0]))
            self.printpl('o', ' Confusion matrix:\n', skm.confusion_matrix(result['solution'], result['pred_labels'][0]))

        elif self.kernel == 'r':
            """
            Print the Fitness score and Mean Squared Error for a REGRESSION run against the test data.

            """

            for i in range(len(result['result'])):
                self.printpl('o', '\t\033[36m Data row {} predicts c1:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.
                             format(i, result['result'][i], result['solution'][i]))

            MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']

            self.printpl('o', '\n\t Origin fitness score: {}'.format(self.origin['fitness_train']))
            self.printpl('o', '\n\t Regression fitness score: {}'.format(fitness))
            self.printpl('o', '\t Mean Squared Error: {}'.format(MSE))

            return

        elif self.kernel == 'm':

            """
            Print the accuracy for a MATCH kernel run against the test data.

            """

            for i in range(len(result['result'])):
                self.printpl('vv', '\t\033[36m Data row {} predicts match:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.format(i, result['result'][i], result['solution'][i]))

            self.printpl('arity', 'Matching fitness score: {}'.format(result['fitness']))

            return
        else:
            self.printpl('e', 'This fitness test is not available:')

        return

    def display_tree(self, tree):

        """
        Display all or part of a Tree on-screen.

        This method displays all sequential node_ids from 'start' node through bottom, within the given tree.

        """

        ind = ''
        self.printpl('i', '\n\033[1m\033[36m Tree ID', int(tree[TR_ID][1]), '\033[0;0m')

        for depth in range(0, self.gp['tree_depth_max'] + 1):  # increment through all possible Tree depths - tested 2016 07/09
            self.printpl('o', '\n', ind, '\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m')

            for node in range(1, len(tree[3])):  # increment through all nodes (redundant, I know)
                if int(tree[N_depth][node]) == depth:
                    self.printpl('i', '')
                    self.printpl('i', ind, '\033[1m\033[36m NODE:', tree[3][node], '\033[0;0m')
                    self.printpl('i', ind, '  type:', tree[N_type][node])
                    self.printpl('i', ind, '  label:', tree[6][node], '\tparent node:', tree[7][node])
                    self.printpl('i', ind, '  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node])

            ind = ind + '\t'

        self.printpl('i', 'TODO')
        # self.eval_tf(tree)  # generate the raw and sympified expression for the entire Tree
        algo_raw_str = str(self.tree_expr_raw(tree, 1))
        self.printpl('i', '\t\033[36mTree', tree[TR_ID][1], 'yields (raw):', algo_raw_str, '\033[0;0m')
        self.printpl('i', '\t\033[36mTree', tree[TR_ID][1], 'yields (sym):\033[1m', '\033[0;0m')

        return

    def manual_expr_fitness(self, expr):
        fitness = self.eval_tf(expr, self.data_train)['fitness']
        self.printpl('i', 'Your algos fitness:', fitness)
        return

    def printpl(self, message_type, *args):

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

        message_posttxt = '\033[39m'
        raise_error, pause = False, False

        if message_type in self.display:
            message_pretxt = '\033[39m'  # default color
            if 'i' in message_type:
                message_style = '\033[36m'
                message_pretxt = 'Info: '  # cyan
            elif 'e' in message_type:
                message_style = '\033[31m'
                message_pretxt = 'ERROR: '  # red
                raise_error = True
            elif 'w' in message_type:  # warning
                message_style = '\033[93m'
                message_pretxt = 'Warning: '  # Warning-yellow
            elif 'g' in message_type:
                message_style = '\033[32m'
                message_pretxt = 'Generation: '  # green
            elif 'arity' in message_type:  # verbose
                message_style = '\033[37m'  # white
                message_pretxt = 'Verbose: '
            elif 'p' in message_type:  # pause
                message_style = '\033[33m'
                message_pretxt = 'Pause(TODO): '  # Yellow
                pause = True
            elif 'f' in message_type:  # function
                message_style = '\033[35m'  # Magenta
                message_pretxt = 'Func: '
            elif 't' in message_type:  # Timer
                message_style = '\033[32m'
                message_pretxt = 'Timer: '
            elif 'o' in message_type:  # original
                # Just show it
                message_style = ''
                message_pretxt = ''
            elif 'c' in message_type:
                if args[0][1]:
                    self.printpl(args[0][0], '', args[1:])
                return
            else:
                message_style = ''
                self.printpl('w', 'Display-mode', message_type, 'not known.')

            print(message_style + message_pretxt + ' '.join(map(str, args)) + message_posttxt)
            #
            # if raise_error:
            #     raise

            if pause:
                self.plagih_pause()  # correct pause?
        return
