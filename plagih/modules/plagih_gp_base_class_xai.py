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
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats as st

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

TR_ID = 0
TR_type = 1
TR_depth = 2
TR_nID = 3
TR_ndepth = 4
TR_ntype = 5
TR_nlabel = 6
TR_nparent = 7
TR_narity = 8
TR_nc1 = 9
TR_nc2 = 10
TR_nc3 = 11
TR_fitness = 12
TR_nmodify = 13
TR_parsimony = 14
TR_num_lines = 15

operator_dict = {ast.Add: tf.add,  # e.g., a + b
                 ast.Sub: tf.subtract,  # e.g., a - b
                 ast.Mult: tf.multiply,  # e.g., a * b
                 ast.Div: tf.divide,  # e.g., a / b
                 ast.Pow: tf.pow,  # e.g., a ** 2
                 ast.USub: tf.negative,  # e.g., -a
                 ast.And: tf.logical_and,  # e.g., a and b
                 ast.Or: tf.logical_or,  # e.g., a or b
                 ast.Not: tf.logical_not,  # e.g., not a
                 ast.Eq: tf.equal,  # e.g., a == b
                 ast.NotEq: tf.not_equal,  # e.g., a != b
                 ast.Lt: tf.less,  # e.g., a < b
                 ast.LtE: tf.less_equal,  # e.g., a <= b
                 ast.Gt: tf.greater,  # e.g., a > b
                 ast.GtE: tf.greater_equal,  # e.g., a >= 1
                 'abs': tf.abs,  # e.g., abs(a)
                 'sign': tf.sign,  # e.g., sign(a)
                 'square': tf.square,  # e.g., square(a)
                 'sqrt': tf.sqrt,  # e.g., sqrt(a)
                 'pow': tf.pow,  # e.g., pow(a, b)
                 'log': tf.math.log,  # e.g., log(a)
                 'log1p': tf.math.log1p,  # e.g., log1p(a)
                 'cos': tf.cos,  # e.g., cos(a)
                 'sin': tf.sin,  # e.g., sin(a)
                 'tan': tf.tan,  # e.g., tan(a)
                 'acos': tf.acos,  # e.g., acos(a)
                 'asin': tf.asin,  # e.g., asin(a)
                 'atan': tf.atan,  # e.g., atan(a)
                 'Ifte': tf.compat.v2.where,  # e.g., Ifte(a, b, c)
                 'Min': tf.math.reduce_min,  # do not use tf.math.minimum,  # min is apparently a string
                 'Max': tf.math.reduce_max,  # e.g. min(a, b)
                 'Mini': tf.math.minimum,  # if reduce_min does not work...
                 'Maxi': tf.math.maximum,
                 # Note: These need separate handling for their conversion type
                 # 'Ftob': tf.dtypes.cast,
                 # 'Ftob': tf.dtypes.cast,
                 }
function_dtypes_dict = {  # Needs A LOT OF further testing
    'float': '2f',  # these three are dummies
    'int': '2f',    # neede to use the dict for function types aswell
    'bool': '2f',   # so we can "workarounded" use them

    '+': 'f2f',
    '-': 'f2f',
    '*': 'f2f',
    '/': 'f2f',
    '**': 'f2f',
    'abs': 'f2f',
    'sign': 'f2f',
    'square': 'f2f',
    'sqrt': 'f2f',
    'log': 'f2f',
    'log1p': 'f2f',
    'cos': 'f2f',
    'sin': 'f2f',
    'tan': 'f2f',
    'acos': 'f2f',
    'asin': 'f2f',
    'atan': 'f2f',

    'And': 'b2b',
    'Or': 'b2b',
    'Xor': 'b2b',
    'Nand': 'b2b',
    'Xand': 'b2b',
    'Nor': 'b2b',
    'Xnor': 'b2b',
    'Not': 'b2b',
    'ITE': 'b2b',

    '==': 'f2b',
    '!=': 'f2b',
    '<': 'f2b',
    '<=': 'f2b',
    '>': 'f2b',
    '>=': 'f2b',

    'Ftob': 'f2b',
    'Btof': 'b2f',  # False->0, True->1, dummy-function
    'Btof_extreme': 'b2f',  # False->-1, True->1. Does that make sense?

    'Ifte': 'b2f2f',  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Min': 'f2f',
    'Max': 'f2f',
    'Mini': 'f2f',
    'Maxi': 'f2f',
}
functions_multiparam_dict = []  # = ['Min', 'Max']  # Currently not in use
functions_wrap_dict = ['Mini', 'Maxi', 'abs', 'sign', 'square', 'sqrt', 'log', 'log1p', 'cos', 'sin', 'tan', 'acos', 'asin', 'atan']
functions_inline_dict = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=']  # 'Min', 'Max',
# TODO all functions have to be within one of these lists. check it.
function_arity_dict = {  # Needs A LOT OF further testing
    'float': 0,  # these three are dummies
    'int': 0,    # neede to use the dict for function types aswell
    'bool': 0,   # so we can "workarounded" use them

    '+': 2,
    '-': 2,
    '*': 2,
    '/': 2,
    '**': 2,
    'abs': 1,
    'sign': 1,
    'square': 1,
    'sqrt': 1,
    'log': 1,
    'log1p': 1,
    'cos': 1,
    'sin': 1,
    'tan': 1,
    'acos': 1,
    'asin': 1,
    'atan': 1,

    'And': 2,
    'Or': 2,
    'Xor': 2,
    'Nand': 2,
    'Xand': 2,
    'Nor': 2,
    'Xnor': 2,
    'Not': 1,
    'ITE': 3,

    '==': 2,
    '!=': 2,
    '<': 2,
    '<=': 2,
    '>': 2,
    '>=': 2,
    'Ftob': 1,
    'Btof': 1,  # False->0, True->1, dummy-function
    'Btof_extreme': 1,  # False->-1, True->1. Does that make sense?

    'Ifte': 3,  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    # 'Min': 2,
    # 'Max': 2,
    'Mini': 2,
    'Maxi': 2,
}
sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees


class ExplainableGP(object):
    """

    """

    def __init__(self, config_dict, file_dict, gp_ops_distribution_list, monitor_dict):

        # self.algo_raw = []      # the raw expression generated by Sympy per Tree -- CONSIDER MAKING THIS VARIABLE LOCAL
        # self.algo_sym = []      # the expression generated by Sympy per Tree -- CONSIDER MAKING THIS VARIABLE LOCAL
        self.origin_dominators = {}  # all Trees which share the best fitness score
        self.pareto_parsimony_to_fitness = {}   # parsimony: best_fitness
        self.pareto_parsimony_get_hash = {}     # parsimony: tree-hash (algo_raw)
        self.pareto_best_trees = {}             # tree-hash: algo_raw
        self.gene_pool = []                 # store all Tree IDs for use by Tournament
        self.class_labels = 0               # the number of true class labels (data_y) todo warum 0?
        self.population_genepool = []

        # 1. set global variables to those local values passed from the user script
        self.crossover_type_safety_mode = config_dict['crossover_type_safety_mode']
        self.gene_pool_threshold = config_dict['gene_pool_threshold']
        self.kernel = config_dict['kernel']                            # fitness function
        self.tree_depth_max = config_dict['tree_depth_max']            # maximum Tree depth for the entire run; limits bloat
        self.tree_depth_min = config_dict['tree_depth_min']            # minimum number of nodes
        self.tree_pop_max = config_dict['tree_pop_max']                # maximum number of Trees per generation
        self.gen_max = config_dict['gen_max']                          # maximum number of generations
        self.tourn_size = config_dict['tourn_size']                    # number of Trees selected for each tournament
        self.display = config_dict['display']
        self.precision = config_dict['precision']  # the number of floating points for the round function
        self.swim = config_dict['swim']  # pass along the gene_pool restriction methodology
        self.parsimony_min_max = config_dict['parsimony_min_max']

        self.monitor_dict = monitor_dict

        self.evolve_repro = gp_ops_distribution_list[0]      # quantity of a population generated through Reproduction
        self.evolve_point = gp_ops_distribution_list[1]      # quantity of a population generated through Point Mutation
        self.evolve_branch = gp_ops_distribution_list[2]     # quantity of a population generated through Branch Mutation
        self.evolve_cross = gp_ops_distribution_list[3]      # quantity of a population generated through Crossover
        self.evolve_missing = gp_ops_distribution_list[4]    # fill up the generation with candidates... todo?
        if self.evolve_missing > 0:
            exit()

        fitt_dict = {'c': 'max', 'r': 'min', 'm': 'max'}
        self.fitness_type = fitt_dict[self.kernel]  # load fitness type
        if self.fitness_type == 'max':
            self.fitness_bad_dummy = 0
        else:
            self.fitness_bad_dummy = float("inf")

        self.data_load(file_dict['operators_file'], file_dict['samples_file'], file_dict['origin_tree_file'])

        return

    def plagih_gp_run(self):
        """
        We will now
        """
        self.main_directories_create()
        self.main_generation_first()
        self.main_generation_loop()   # (main loop)
        self.main_terminate()  # archive populations and return to plagih_gp.py for a clean exit

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Top level      functions                  |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def main_directories_create(self):
        """
        Create all files that will be saved after all
        """

        self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        cwd = os.getcwd()
        self.path = os.path.join(cwd, 'runs/' + self.datetime + '/')  # generate a unique directory name
        if not os.path.isdir(self.path):
            os.makedirs(self.path)  # make a unique directory
        self.filename = {}  # a dictionary to hold .csv filenames

        self.filename.update({'1_first': self.path + 'population_1_first.csv'})
        target = open(self.filename['1_first'], 'w')
        target.close()  # initialise a .csv file for population 'a' (foundation)

        self.filename.update({'new': self.path + 'population_new.csv'})
        target = open(self.filename['new'], 'w')
        target.close()  # initialise a .csv file for population 'new' (evolving)

        self.filename.update({'f': self.path + 'population_f.csv'})
        target = open(self.filename['f'], 'w')
        target.close()  # initialise a .csv file for the final population (test)

        self.filename.update({'s': self.path + 'population_s.csv'})
        target = open(self.filename['s'], 'w')
        target.close()  # initialise a .csv file to manually load (seed)

        self.monitor_performance(mode='init')

    def main_generation_first(self, population_backup_file='', paretofront_backup_file=''):
        """
        Everything that needs to be done for the first generation
        - Extracts "origin Tree" from file
        - Creates all other trees: origin tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        self.gen_id = 1  # set initial generation ID    # first gen only
        self.monitor_performance(mode='init')           # first gen only

        self.gen_prepare_parameters()
        self.gp_first_create()     # creates pop_num of trees

        # Special case: load Data
        try:
            if paretofront_backup_file:
                # load
                self.data_load_backup_population(population_backup_file)
        except:
            self.printpl('e', 'Could not load data')
        self.gen_finalize()

        self.data_save_population(self.population_genepool, '1_first')        # first gen only

    def main_generation_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        menu = 1
        while menu != 0:  # this allows the user to add generations mid-run and not get buried in nested iterations
            for self.gen_id in range(self.gen_id + 1, self.gen_max + 1):  # generation 2 to *max generation*

                # 1. set parameters for the generation
                self.gen_prepare_parameters()

                # 2. Create new generation (from last genepool)
                self.gp_reproduce()  # method 1 - Reproduction
                self.gp_mutate_point()  # method 2 - Point Mutation
                self.gp_mutate_branch()  # method 3 - Branch Mutation
                self.gp_crossover()  # method 4 - Crossover

                self.gen_finalize()

                self.monitor_performance(mode='update')

            else:
                self.printpl('p', '\t\033[32m Enter \033[1m?\033[0;0m\033[32m to review your options or \033[1mq\033[0;0m\033[32muit\033[0;0m')
                menu = 0

    def main_terminate(self):
        """
        Terminates the evolutionary run (if yet in progress), saves parameters and data to disk, and cleanly returns
        the user to plagih_gp.py and the command line.

        """
        self.data_save_config()
        self.data_save_conclusion()
        self.data_save_paretofront()
        target = open(self.filename['f'], 'w')
        target.close()  # initialize the .csv file for the final population
        self.data_save_population(self.population_new, 'f')  # save the final generation of Trees to disk

        self.printpl('o', '\n\t\033[32m Your Trees and runtime parameters are archived in plagih_gp/runs/[date-time]/\033[0;0m')
        self.printpl('o', '\n\033[3m "It is not the strongest of the species that survive, nor the most intelligent,\033[0;0m')
        self.printpl('o', '\033[3m  but the one most responsive to change."\033[0;0m --Charles Darwin\n')
        self.printpl('o', '\033[3m Congrats!\033[0;0m Your Plagih GP run is complete.')

        self.monitor_performance(mode='show')

        sys.exit()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def monitor_pop_fitness(self):
        pass

    def monitor_performance(self, mode=''):
        """
        monitors everything

        Helper:
        # plot_end(self, y, mode='', plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='')
        """

        if self.monitor_dict['gen_fitness_avg'] == 'y':
            if mode == 'update':  # save value to final list and reset counter
                average_fitness = self.monitor_gen_fitness_eval()
                self.mon_fitness_avg.append(average_fitness)
            elif mode == 'reset':
                pass
            elif mode == 'show':  # show the plot
                self.plot_end(self.mon_fitness_avg, plt_title='Average Fitness', plt_y_label='Fitness')
            elif mode == 'init':  # initialize the variables (saves memory)
                self.mon_fitness_avg = []
            else:
                self.printpl('e', 'Display-mode not known or empty:', mode)

        if self.monitor_dict['genepool_size'] == 'y':
            if mode == 'update':
                self.mon_genepool_size.append(self.gene_pool_size)
            elif mode == 'reset':
                pass
            elif mode == 'show':
                self.plot_end(self.mon_genepool_size, plt_title='Genepool size', plt_y_label='Amount')
            elif mode == 'init':
                self.mon_genepool_size = []
            else:
                self.printpl('e', 'Display-mode not known or empty:', mode)

        # TODO anzahl doppelte bäume pro generation

        # # Sympify errors
        # if self.monitor['sympify_errors'] == 'y':
        #     if mode == 'update':
        #         if self.monitor_failed_sympys_amount:
        #             self.plot_failed_sympys_amount.append(self.monitor_failed_sympys_amount)
        #     elif mode == 'reset':
        #         self.monitor_failed_sympys_amount = 0
        #     elif mode == 'init':
        #         self.plot_failed_sympys_amount = []
        #     elif mode == 'show':
        #         if self.plot_failed_sympys_amount:
        #             try:
        #                 self.plot_end('s', plt_title='Sympify zoo and nan', plt_y_label='Amount')
        #             except:
        #                 self.printpl('e', 'plotting did not work for sympify')
        #     else:
        #         self.printpl('e', 'Display-mode not known or empty:', mode)
        #
        # # Average Tree size, parsimony

        return

    def monitor_gen_fitness_eval(self):
        """

        :return:
        """
        tmp_sum = 0
        for tree_id in self.gene_pool:
            tmp_sum += float(self.population_new[tree_id][12][1])
        return tmp_sum / len(self.gene_pool)

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Load and Archive Data                     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def data_load(self, operators_file, samples_file, origin_tree_file_path):

        """
        Loads all user input and prepares

        """

        self.data_load_dataset(samples_file)
        self.data_load_operators(operators_file)  # Ia a little complex now, outsourced into this function
        self.data_load_origin_tree(origin_tree_file_path)  # construct the first population of Trees

        return

    def data_load_origin_tree(self, origin_tree_file_path):
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
                    tree = np.append(tree, [row], axis=1)  # append first row to Tree ('TREE_ID')
                else:
                    tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree
            if tree.shape[0] == TR_num_lines:  # (+ row 0)
                pass  # print('Origin Tree is: \n' + str(tree))
            else:
                self.printpl('e', "Tree could not be imported correctly from .csv file.")
                raise

        # As we need it quite often, safe the origin tree's data
        self.origin_tree = tree
        # algo_sym = self.tree_expr_sympify(self.origin_tree)
        algo_sym_origin = self.tree_expr_sympify(self.origin_tree)

        self.hashtable_fitness_train = {}
        self.origin_fitness_train = self.eval_fitness_traindata(algo_sym_origin)

        return

    def data_load_dataset(self, samples_file):

        """
        loads the goal-data from .csv file
        """
        # Load the 'good' samples file. first observations then actions.
        # Both can have any shape specified in the gym.env "spaces" (dimensions: 1-n, type: int-floatstring?)
        #
        # Mountaincar .csv first lines (11.12.2019):
        ######################################
        # observation0:float, observation1_float, action0:float
        # -0.5031261704876531, 0.0, 2
        # ####################################

        # TODO hier unterscheiden, welche distanz genommen werden soll?
        # TODO Funktioniert das mit allen Datentypen?

        num_observations, num_actions = 0, 0
        var_types = []
        # TODO Terminal types as dictionary? would be much prettier.
        self.terminals, self.terminal_types = [], []
        self.actions, self.action_types = [], []

        # 1. File einlesen
        with open(samples_file) as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            for i, row in enumerate(reader):
                if i == 0:  # variable identifiers
                    # all_variables = [x.rsplit(':', 1)[0] for x in row]  # ['observation0:float'] -> ['observation0']
                    for var_name in row:
                        var_types.append(var_name.split(':', 1)[1])
                        if var_name.startswith('o'):  # found an observation
                            num_observations += 1
                            self.terminals.append(var_name.rsplit(':', 1)[0])
                            self.terminal_types.append(var_name.split(':', 1)[1])
                        elif var_name.startswith('a'):  # found an action
                            num_actions += 1
                            self.actions.append(var_name.split(':', 1)[0])  # action0
                            self.action_types.append(var_name.split(':', 1)[1])  # float
                        else:
                            self.printpl('e', 'Behaviour samples first line: Variables have to start with "o" or "a" to be recognized. Is actually:', var_name)
                            raise

                    data_x, data_y = [], []

                else:  # convert every 'string' element to its datatype
                    # TODO var_types ist genau dasselbe wie self.terminal , oder? eines ersetzen?
                    row_as_data = [locate(var_types[i])(x) for i, x in enumerate(row)]  # ['observation0:float'] + ['0.123'] --> float(['0.123']) --> 0.123
                    data_x.append(row_as_data[:num_observations])
                    data_y.append(row_as_data[num_observations:])
            csvFile.close()
        self.dataset = samples_file

        # Part 1.5: load terminals into specific terminal types
        self.terminals_bool, self.terminals_float = [], []

        # sfeh this is a workaround to have your terminals in typed lists. rework all of this please
        for i, term_type in enumerate(self.terminal_types):  # TODO get all of the actions out, not only one
            if term_type == 'float' or term_type == 'int':  # sfeh check if this is enough, int
                self.terminals_float.append(self.terminals[i])
            elif term_type == 'bool':
                self.terminals_bool.append(self.terminals[i])
            else:
                self.printpl('e', 'term_type is neither float or bool:', term_type)
                raise

        # sfeh das funktioniert nur bei diskreten Actions
        self.class_labels = len(np.unique(data_y))  # load the user defined true labels for classification or solutions for regression

        self.data_load_dataset_split(data_x, data_y, test_size=0.2)

        return

    def data_load_dataset_split(self, data_x, data_y, test_size):

        # Part 4 - from the dataset, extract TRAINING and TEST data ###
        # TODO die func kann sicher nicht mit 2d labels umgehen. Funktion macht das echt super uneffizient.
        x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size=test_size)  # 80/20 TRAIN/TEST split
        data_train = np.c_[x_train, y_train]  # recombine each row of data with its associated class label (right column)
        data_test = np.c_[x_test, y_test]  # recombine each row of data with its associated class label (right column)
        data_x, data_y, x_train, y_train, x_test, y_test = [], [], [], [], [], []  # clear from memory
        self.data_train_cols = len(data_train[0, :])  # qty count
        self.data_train_rows = len(data_train[:, 0])  # qty count
        self.data_test_cols = len(data_test[0, :])  # qty count
        self.data_test_rows = len(data_test[:, 0])  # qty count

        ### PART 5 - load TRAINING and TEST data for TensorFlow processing
        self.data_train = data_train  # Store train data for processing in TF
        self.data_test = data_test  # Store test data for processing in TF
        self.tf_device = "/gpu:0"  # Set TF computation backend device (CPU or GPU); gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device_log = False  # TF device usage logging (for debugging)

        # TODO what to do with test data?
        if True:
            self.data_test = self.data_train

    def data_load_operators(self, operators_file_path):
        """
        Load all operators ready-to-use from a file
        """
        self.functions = np.loadtxt(operators_file_path, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
        # Part 3.5: Split the functions in 5 types
        self.functions_f2f, self.functions_f2b, self.functions_b2b, self.functions_b2f, self.functions_b2f2f = [], [], [], [], []
        self.functions_array = [[[], [], [], []],  # rows are the function types (f2f)
                                [[], [], [], []],  # columns are the arity
                                [[], [], [], []],
                                [[], [], [], []],
                                [[], [], [], []]]
        for fun in self.functions:
            if function_dtypes_dict[fun[0]] == 'f2f':
                self.functions_f2f.append(fun[0])
                self.functions_array[0][int(fun[1])].append(fun[0])
            elif function_dtypes_dict[fun[0]] == 'f2b':
                self.functions_f2b.append(fun[0])
                self.functions_array[1][int(fun[1])].append(fun[0])
            elif function_dtypes_dict[fun[0]] == 'b2b':
                self.functions_b2b.append(fun[0])
                self.functions_array[2][int(fun[1])].append(fun[0])
            elif function_dtypes_dict[fun[0]] == 'b2f':
                self.functions_b2f.append(fun[0])
                self.functions_array[3][int(fun[1])].append(fun[0])
            elif function_dtypes_dict[fun[0]] == 'b2f2f':
                self.functions_b2f2f.append(fun[0])
                self.functions_array[4][int(fun[1])].append(fun[0])

        # self.printpl('i', 'self.functions_array: ')
        # pprint(self.functions_array)

        func_2f, func_2b = [], []

        # The Functions that create float (aka number) values
        if self.functions_f2f:
            func_2f.extend(self.functions_f2f)
        if self.functions_b2f:
            func_2f.extend(self.functions_b2f)
        if self.functions_b2f2f:
            func_2f.extend(self.functions_b2f2f)
        if func_2f:
            self.functions_2f = func_2f[:]
        else:
            self.printpl('w', 'No Functions that create numbers were found')
            self.functions_2f = []

        # The Functions that create boolean values
        if self.functions_f2b:
            func_2b.extend(self.functions_f2b)
        if self.functions_b2b:
            func_2b.extend(self.functions_b2b)
        if func_2b:
            self.functions_2b = func_2b[:]
        else:
            self.printpl('w', 'No Functions that create bool were found')
            self.functions_2b = []

    def data_load_backup_population(self, population_backup_file):
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
                    self.population_genepool = [row]  # write header to population_genepool

                else:
                    if row == []:
                        self.tree = np.array([[]])  # initialise Tree array

                    else:
                        if self.tree.shape[1] == 0:
                            self.tree = np.append(self.tree, [row], axis=1)  # append first row to Tree

                        else:
                            self.tree = np.append(self.tree, [row], axis=0)  # append subsequent rows to Tree

                    if self.tree.shape[0] == TR_num_lines:  # (current tree rows + row 0
                        self.population_genepool.append(self.tree)  # append complete Tree to population list

        self.printpl('i', 'We loaded the following population_genepool:', self.population_genepool)
        return

    def data_pickle_save(self):
        """
        save all data every few rounds to restore them
        - save the pareto front (done)
        - save the last generation (done)
        - Save important meta-data: current generation (done)
        """
        run_data = {'gen_id': self.gen_id,
                    'parsimony_front_fitness': '',
                    'pareto_front': self.pareto_parsimony_get_hash,
                    'pareto_front_hash': self.pareto_parsimony_to_fitness,
                    'best_trees': self.pareto_best_trees,
                    'population_new': self.population_new
                    }
        pickle.dump(run_data, open(self.path + 'Gen-' + str(self.gen_id) + '-backup.p', 'wb'))

    def data_pickle_recover(self, samples_file, operators_file, origin_tree_file_path, paretofront_file):

        """
        Restarts a run 'midway' by mainly loading the already found pareto-front
        - Warn user about pickling (todo)
        """

        self.data_load_dataset(samples_file)
        self.data_load_operators(operators_file)  # Ia a little complex now, outsourced into this function
        self.data_load_origin_tree(origin_tree_file_path)  # construct the first population of Trees

        with open(paretofront_file, 'rb') as csv_file:
            target = csv.reader(csv_file, delimiter=',')
            n = 0  # track row count

            for row in target:
                print('row', row)

                n = n + 1
                if n == 1:
                    pass  # skip first empty row

                elif n == 2:
                    self.population_genepool = [row]  # write header to population_genepool

                else:
                    if row == []:
                        self.tree = np.array([[]])  # initialise Tree array

                    else:
                        if self.tree.shape[1] == 0:
                            self.tree = np.append(self.tree, [row], axis=1)  # append first row to Tree

                        else:
                            self.tree = np.append(self.tree, [row], axis=0)  # append subsequent rows to Tree

                    if self.tree.shape[0] == TR_num_lines:  # (current tree rows + row 0
                        self.population_genepool.append(self.tree)  # append complete Tree to population list

        self.printpl('i', 'Recovered gene_pool:', self.population_genepool, 'with size', len(self.population_genepool))

        return

    def data_save_population(self, population, key):

        """
        Save population_* to disk.

        """

        with open(self.filename[key], 'a', newline='') as csv_file:
            target = csv.writer(csv_file, delimiter=',')
            if self.gen_id != 1:
                target.writerows([''])  # empty row before each generation
            target.writerows([['Plagih GP by Kai Staats, improved by Simon Fehrer', 'Generation:', str(self.gen_id)]])

            for tree in range(1, len(population)):
                target.writerows([''])  # empty row before each Tree
                for row in range(0, TR_num_lines):  # increment through each row in the array Tree (+ row 0)
                    target.writerows([population[tree][row]])

        return

    def data_save_conclusion(self):

        """
        write the performance of the gp to disc
        """

        file = open(self.path + 'results.txt', 'w')
        file.write('Plagih GP')
        file.write('\n launched: ' + str(self.datetime))
        file.write('\n dataset: ' + str(self.dataset))
        file.write('\n')

        if len(self.origin_dominators) > 0:

            fitness_best = 0
            fittest_tree = 0

            # revised method, re-evaluating all Trees from stored fitness score
            # for tree_id in range(1, len(self.population_new)):
            for tree_id in self.gene_pool:

                fitness = float(self.population_new[tree_id][12][1])

                if self.kernel == 'c':  # display best fit Trees for the CLASSIFY kernel
                    if fitness >= fitness_best:  # find the Tree with maximum fitness score
                        fitness_best = fitness
                        fittest_tree = tree_id  # set best fitness Tree

                elif self.kernel == 'r':  # display best fit Trees for the REGRESSION kernel
                    if fitness_best == 0:
                        fitness_best = fitness  # set the baseline first time through
                    if fitness <= fitness_best:  # find the Tree with minimum fitness score
                        fitness_best = fitness
                        fittest_tree = tree_id  # set best fitness Tree

                elif self.kernel == 'm':  # display best fit Trees for the MATCH kernel
                    if fitness == self.data_train_rows:  # find the Tree with a perfect match for all data rows
                        fitness_best = fitness
                        fittest_tree = tree_id  # set best fitness Tree

            # test the most fit Tree and write to the .txt log
            algo_sym = self.tree_expr_sympify(self.population_new[int(fittest_tree)])  # generate the raw and sympified expression for the given Tree using SymPy
            result = self.eval_tf(str(algo_sym), self.data_test, get_pred_labels=True)

            algo_sym = self.tree_expr_sympify(self.origin_tree)
            origin_fitness_test = self.eval_tf(str(algo_sym), self.data_test)['fitness']

            file.write('\n\t Origin fitness score: {}'.format(origin_fitness_test))
            file.write('\n\n Tree ' + str(fittest_tree) + ' is the most fit, with expression:')
            file.write('\n\n ' + str(algo_sym))

            if self.kernel == 'c':
                file.write('\n\n Classification fitness score: {}'.format(result['fitness']))
                file.write('\n\n Precision-Recall report:\n {}'.format(
                    skm.classification_report(result['solution'], result['pred_labels'][0])))
                file.write('\n Confusion matrix:\n {}'.format(
                    skm.confusion_matrix(result['solution'], result['pred_labels'][0])))

            elif self.kernel == 'r':
                MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
                file.write('\n\n Regression fitness score: {}'.format(fitness))
                file.write('\n Mean Squared Error: {}'.format(MSE))

            elif self.kernel == 'm':
                file.write('\n\n Matching fitness score: {}'.format(result['fitness']))

        else:
            file.write('\n\n No solution was better than the origin... your species has gone extinct!')

        file.write('\n\n')
        file.close()

        return

    def data_save_config(self):
        """
        write the parameters to a file
        """

        file = open(self.path + 'config.txt', 'w')
        file.write('Plagih GP')
        file.write('\n launched: ' + str(self.datetime))
        file.write('\n dataset: ' + str(self.dataset))
        file.write('\n')
        file.write('\n kernel: ' + str(self.kernel))
        file.write('\n precision: ' + str(self.precision))
        file.write('\n')
        file.write('\n tree depth max: ' + str(self.tree_depth_max))
        file.write('\n min node count: ' + str(self.tree_depth_min))
        file.write('\n')
        file.write('\n genetic operator Reproduction: ' + str(self.evolve_repro))
        file.write('\n genetic operator Point Mutation: ' + str(self.evolve_point))
        file.write('\n genetic operator Branch Mutation: ' + str(self.evolve_branch))
        file.write('\n genetic operator Crossover: ' + str(self.evolve_cross))
        file.write('\n')
        file.write('\n tournament size: ' + str(self.tourn_size))
        file.write('\n population: ' + str(self.tree_pop_max))
        file.write('\n number of generations: ' + str(self.gen_id))
        file.write('\n\n')
        file.close()

    def data_save_paretofront(self):
        """
        Save all the pareto efficient candidates to file
        """
        file = open(self.path + 'paretofront.txt', 'w')

        for key in self.pareto_parsimony_to_fitness:
            value = self.pareto_parsimony_to_fitness[key]
            tmp_hash = self.pareto_parsimony_get_hash[key]
            algo = self.pareto_best_trees[tmp_hash]
            file.write('\nParsimony: ' + str(key) + ' ' + str(value) + ' ' + str(algo))

        file.close()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Perform the 3 genetic prog. operations    |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gp_first_create(self):
        """
        Constructs the first generation
        - loads the origin-tree from file
        - constructs the first generation from this tree with branch mutation
        """

        #TODO branch mutation in ALL subtrees? if more options are available
        #TODO safely create a complete generation
        self.printpl('g', 'Initial population...')

        self.origin_tree[0][1] = 1
        self.population_new.append(self.origin_tree)

        for TREE_ID in range(2, self.tree_pop_max + 1 - 1):

            # Copy reference tree
            tree = self.origin_tree.copy()

            # vary this tree with branch mutation
            branch_nodes_list = self.evolve_subtree_get(tree)  # [6, 9, 10] select point of mutation and all nodes beneath
            tree = self.evolve_subtree_build(tree, branch_nodes_list)  # tree with new branch

            # Fill the correct meta-data into the tree (and wipe the old fitness)
            tree = self.tree_store_meta_lastgen(tree, modification='i')  # wipe fitness data
            tree = self.tree_modifyable_nodes_set(tree)
            tree[0][1] = TREE_ID

            self.population_new.append(tree)

        self.printpl('gg', 'We have constructed a single, stochastic population of', self.tree_pop_max, 'Trees, and saved to disk')

    def gp_selection_tournament(self, tourn_size):

        """
        gp-selection. takes a number of trees (usually 3) and returns the best one (winner)
        Uses:
            self.population a
            self.genepool a
        """

        tourn_test = 0
        # short_test = 0 # an incomplete parsimony test (seeking shortest solution)

        for n in range(tourn_size):

            # 1. choose a random gene_pool tree from population_genepool
            # old code, now population_genepool has a complete genepool
            # rnd = np.random.randint(len(self.gene_pool))  # select one Tree at random from the gene pool
            # tree_id = int(self.gene_pool[rnd])

            # select one Tree at random from the gene pool
            tree_id = 1 + np.random.randint(self.gene_pool_size)

            fitness = float(self.population_genepool[tree_id][12][1])  # extract the fitness from the array
            fitness = round(fitness, self.precision)  # force 'result' and 'solution' to the same number of floating points

            if self.fitness_type == 'max':  # if the fitness function is maximising

                # first time through, 'tourn_test' will be initialised below

                if fitness > tourn_test:  # if the current Tree's 'fitness' is greater than the priors'
                    self.printpl('ggg', '\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # set 'TREE_ID' for the new leader
                    tourn_test = fitness  # set 'fitness' of the new leader
                # short_test = int(self.population_genepool[tree_id][TR_parsimony][1]) # set len(algo_raw) of new leader

                elif fitness == tourn_test:  # if the current Tree's 'fitness' is equal to the priors'
                    self.printpl('ggg', '\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # in case there is no variance in this tournament
                # tourn_test unchanged

                elif fitness < tourn_test:  # if the current Tree's 'fitness' is less than the priors'
                    self.printpl('ggg', '\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test,'and is ignored\033[0;0m')
                    # tourn_lead unchanged
                    # tourn_test unchanged

                else:
                    self.printpl('e', '\t\033[31m pop_selection_tournament: fitness =', fitness, 'and tourn_test =', tourn_test, '\033[0;0m')
                    raise  # no reason to pause

            elif self.fitness_type == 'min':  # if the fitness function is minimising

                if tourn_test == 0:  # first time through, 'tourn_test' is given a baseline value
                    tourn_test = fitness

                if fitness < tourn_test:  # if the current Tree's 'fitness' is less than the priors'
                    self.printpl('ggg', '\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # set 'TREE_ID' for the new leader
                    tourn_test = fitness  # set 'fitness' of the new leader

                elif fitness == tourn_test:  # if the current Tree's 'fitness' is equal to the priors'
                    self.printpl('ggg', '\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # in case there is no variance in this tournament
                # tourn_test unchanged

                elif fitness > tourn_test:  # if the current Tree's 'fitness' is greater than the priors'
                    self.printpl('ggg', '\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and is ignored\033[0;0m')
                    # tourn_lead unchanged
                    # tourn_test unchanged
                else:
                    self.printpl('i', 'fitness', self.population_genepool[tree_id])
                    self.printpl('e', '\033[31m pop_selection_tournament: fitness =', fitness, 'and tourn_test =', tourn_test, '\033[0;0m')
                    raise
                if tourn_lead:
                    tourn_winner = np.copy(self.population_genepool[tourn_lead])  # copy full Tree so as to not inadvertantly modify the original tree
                    self.printpl('vv', '\t\033[36mThe winner of the tournament is Tree:\033[1m', tourn_winner[0][1], '\033[0;0m')
                else:
                    self.printpl('e', 'This line has to be here to stop pycharm from finding a warning in the next line')
                    raise

        return tourn_winner

    def gp_genepool_parsimony(self):

        """
        Create the gene pool
        -> self.gene_pool
        - Add a candidate if its parsimony is within the threshold

        # TODO find rules that automatically stop plagih from finding solutions that are too complex? adjust the nextgen functions (mutate, ...)?
        Ideas:
        # Create a rule when a candidate can come into the gene pool
        #  - When its better than original threshold
        #  - epsilon-threshold, which is auto initialised in the first generation?
        #  - when there was an improvement from the last change
        # Add good ones to the gene pool
        # Add the BEST ones to the olymp
        # TODO stop equal candidates from being in the gene pool multiple times?

        """
        """
        select a genepool from population_new
        """
        self.printpl('gg', 'Gene Pool for Generation:', self.gen_id, '...')

        # Empty old gene_pool first
        self.gene_pool = []
        self.gene_pool_size = 0
        for tree_id in range(1, len(self.population_new)):  # Every tree

            # Get genepool requirements. Only criteria: parsominy
            tree = self.population_new[tree_id]
            parsimony = self.tree_parsimony(self.population_new[tree_id])

            # Requirement
            if parsimony < self.parsimony_min_max[0]:  # Can this tree go in the gene_pool?
                self.gene_pool_size += 1
                self.tree_store_parsimony(tree, parsimony)
                self.gene_pool.append(int(tree[0][1]))

        if len(self.gene_pool) > 0:
            self.printpl('i', 'The total population of the gene pool is', len(self.gene_pool))
        else:  # the evolutionary constraints were too tight, killing off the entire population
            self.printpl('e', 'There are no Trees in the gene pool. You should archive your population and (q)uit.')

        return

    def gp_reproduce(self):

        """
        Through tournament selection, a single Tree from the prior generation is copied without mutation to the next
        generation. This is analogous to a member of the prior generation directly entering the gene pool of the
        subsequent (younger) generation.

        """
        self.printpl('gg', 'Reproduce One...')

        for n in range(self.evolve_repro):  # quantity of Trees to be copied without mutation
            tourn_winner = self.gp_selection_tournament(self.tourn_size)  # perform tournament selection for each reproduction
            tourn_winner = self.tree_store_meta_lastgen(tourn_winner, modification='r')  # wipe fitness data
            tourn_winner[1][1] = 'r'
            self.population_new.append(tourn_winner)  # append array to next generation population of Trees

        return

    def gp_mutate_point(self):

        """
        One point (terminal or function) gets mutated.
        Currently only mutating with functions/terminals of the exactly same type.
        """
        self.printpl('gg', 'Point Mutation...')

        for n in range(self.evolve_point):  # quantity of Trees to be generated through mutation
            tourn_winner = self.gp_selection_tournament(self.tourn_size)  # get a tournament winner
            tourn_winner, node = self.gp_mutate_point_evolve(tourn_winner)  # point mutation; return single point for record keeping
            tourn_winner = self.tree_store_meta_lastgen(tourn_winner, modification='p')  # wipe fitness data
            self.population_new.append(tourn_winner)  # append array to next generation population of Trees
        return

    def gp_mutate_point_evolve(self, tree, mode='random'):

        """
        Mutate a single mutatable point in any Tree.
        """

        # 1. choose a node
        node = self.evolve_get_mutatable_node_id(tree, mode='mutate_point')  # randomly select a point in the Tree (including root)
        node_dtype = self.dtype_label_get_dtype(tree[TR_nlabel][node])  # '>' -> 'f2b'

        # 2. perform point mutation on that specific node
        if tree[5][node] == 'func':
            func_arity = int(tree[TR_narity][node])
            tree[TR_nlabel][node] = self.dtype_func_get_func(node_dtype, arity=func_arity)  # Function is same type, same arity
            # Take care of the modify specs
        elif tree[5][node] == 'term':
            tree[TR_nlabel][node] = self.dtype_dtype_get_term(node_dtype)  # 3 -> '2f' -> 5
        else:
            self.printpl('e', 'Operator type is not specified for PLAGIH ("term", "func",...)', tree[5][node])
            raise

        return tree, node  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping

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

    def gp_mutate_branch(self):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """
        self.printpl('gg', 'Branch Mutation...')

        for n in range(self.evolve_branch):  # quantity of Trees to be generated through mutation
            tourn_winner = self.gp_selection_tournament(self.tourn_size)  # perform tournament selection for each mutation
            branch_nodes_list = self.evolve_subtree_get(tourn_winner)  # select point of mutation and all nodes beneath [6, 9, 10]
            tourn_winner = self.evolve_subtree_build(tourn_winner, branch_nodes_list)
            tourn_winner = self.tree_store_meta_lastgen(tourn_winner, modification='b')  # wipe fitness data
            tourn_winner = self.tree_modifyable_nodes_set(tourn_winner)
            self.population_new.append(tourn_winner)  # append array to next generation population of Trees

        return

    def gp_mutate_branch_new_tree_build(self, TREE_ID, last_modification, old_node_type, tree_depth):

        """
        This method combines 4 sub-methods into a single method for ease of deployment. It is designed to executed
        within a loop such that an entire population is built. However, it may also be run from the command line,
        passing a single TREE_ID to the method.

        if terminal: 2f
        if func:    f2f
        """

        self.gp_mutate_branch_tree_initialise(TREE_ID, tree_depth, last_modification)  # Create empty tree np-array
        self.gp_mutate_branch_root_build(old_node_type)  # insert the first node with either '_2b' or '_2f'
        self.gp_mutate_branch_function_build()  # build all the Function nodes
        self.gp_mutate_branch_terminal_build()  # build the Terminal nodes
        # TODO set tree_depth_base in tree.
        return  # each Tree is written to 'gp.tree'

    def gp_mutate_branch_tree_initialise(self, TREE_ID, tree_depth, last_modification):

        """
        Assign 15 (14+1) global variables to the array 'tree'.

        Build the array 'tree' with 15 rows and initally, just 1 column of labels. This array will grow horizontally as
        each new node is appended. The values of this array are stored as string characters, numbers forced to integers at
        the point of execution.

        Use of the debug (db) interface mode enables the user to watch the genetic operations as they work on the Trees.

        """

        self.pop_TREE_ID = TREE_ID  # pos 0: a unique identifier for each tree
        self.pop_tree_type = last_modification  # pos 1: a global constant based upon the initial user setting
        self.pop_tree_depth_base = tree_depth  # pos 2: a global variable which conveys 'tree_depth_base' as unique to each new Tree
        self.pop_NODE_ID = 1  # pos 3: unique identifier for each node; this is the INDEX KEY to this array
        self.pop_node_depth = 0  # pos 4: depth of each node when committed to the array
        self.pop_node_type = ''  # pos 5: root, function, or terminal
        self.pop_node_label = ''  # pos TR_nlabel: operator [+, -, *, ...] or terminal [a, b, c, ...]
        self.pop_node_parent = ''  # pos 7: parent node
        self.pop_node_arity = ''  # pos 8: number of nodes attached to each non-terminal node
        self.pop_node_c1 = ''  # pos 9: child node 1
        self.pop_node_c2 = ''  # pos 10: child node 2
        self.pop_node_c3 = ''  # pos 11: child node 3 (assumed max of 3 with boolean operator 'if')
        self.pop_fitness = ''  # pos 12: fitness score following Tree evaluation
        self.pop_node_modify = ''  # pos 13: dummy value whether this node is allowed to be modified
        self.pop_parsimony = ''  # pos 14: separate line for the latest distance to origin

        self.tree = np.array(
            [['TREE_ID'], ['tree_type'], ['tree_depth_base'], ['NODE_ID'], ['node_depth'], ['node_type'], ['node_label'], ['node_parent'], ['node_arity'], ['node_c1'], ['node_c2'], ['node_c3'],
             ['fitness'], ['node_modify'], ['parsimony']])

        return

    def gp_mutate_branch_root_build(self, old_node_type):

        """
        Build a root node for a branch insert.
        """
        # eg b2f2f
        self.dtype_function_select(old_node_type)  # select the operator for root

        if self.pop_node_arity == 1:  # 1 child
            self.pop_node_c1 = 2
            self.pop_node_c2 = ''
            self.pop_node_c3 = ''

        elif self.pop_node_arity == 2:  # 2 children
            self.pop_node_c1 = 2
            self.pop_node_c2 = 3
            self.pop_node_c3 = ''

        elif self.pop_node_arity == 3:  # 3 children
            self.pop_node_c1 = 2
            self.pop_node_c2 = 3
            self.pop_node_c3 = 4

        else:
            self.printpl('p', 'branch_root_build: pop_node_arity =', self.pop_node_arity, '\033[0;0m')

        self.pop_node_type = 'func'  # used to be r00t, but what is it good for?

        self.evolve_node_commit()

        return

        ### Function Nodes ###

    def gp_mutate_branch_function_build(self):

        """
        Build the branch full depth
        Builds
        """

        for i in range(1, self.pop_tree_depth_base):  # the tree depth (-1, where the last functions are) sfeh: actually NO -1?

            self.pop_node_depth = i  # increment 'node_depth'
            parent_arity_sum = 0
            prior_sibling_arity = 0  # reset for 'c_buffer' in 'children_link'
            prior_siblings = 0  # reset for 'c_buffer' in 'children_link'

            # parent_arity_sum = amount of nodes (that have to be on this level)
            for j in range(1, len(self.tree[3])):  # increment through all nodes in array 'tree'
                if int(self.tree[4][j]) == self.pop_node_depth - 1:  # find parent nodes which reside at the prior depth
                    parent_arity_sum = parent_arity_sum + int(self.tree[TR_narity][j])  # sum arities of all parent nodes at the prior depth

            # Set for every "free space" a function node (func)
            for j in range(1, len(self.tree[3])):  # increment through all nodes
                if int(self.tree[4][j]) == self.pop_node_depth - 1:  # ... find all parent nodes, one level above...
                    if self.tree[TR_nlabel][j] == 'Ifte':
                        prior_sibling_arity = self.gp_mutate_branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2b')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                        prior_sibling_arity = self.gp_mutate_branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                        prior_sibling_arity = self.gp_mutate_branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                    else:
                        for k in range(1, int(self.tree[TR_narity][j]) + 1):  # k = 1,2
                            self.pop_node_parent = int(self.tree[3][j])  # set the nodes parent
                            parent_func_dtype = function_dtypes_dict[self.tree[TR_nlabel][self.pop_node_parent]]  # find parents node
                            func_dtype = parent_func_dtype[:2][::-1]  # parent 'f2b' -> '2f' child needed. Aka, the first two characters reversed
                            prior_sibling_arity = self.gp_mutate_branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, func_dtype)  # ... generate a Function node
                            prior_siblings = prior_siblings + 1  # sum sibling nodes (current depth) who will spawn their own children (cousins? :)

        return

    def gp_mutate_branch_terminal_build(self):

        """
        Build the Terminal nodes for the tree.

        """

        self.pop_node_depth = self.pop_tree_depth_base  # set the final node_depth (same as 'gp.pop_node_depth' + 1)

        for j in range(1, len(self.tree[3])):  # go through all nodes
            if int(self.tree[4][j]) == self.pop_node_depth - 1:  # this node is a parent
                for k in range(1, (int(self.tree[TR_narity][j]) + 1)):  # increment through each degree of arity for each parent node
                    self.pop_node_parent = int(self.tree[3][j])  # set the parent 'NODE_ID'  ...
                    self.gp_mutate_branch_terminal_gen(function_dtypes_dict[self.tree[TR_nlabel][j]])  # ... generate a Terminal node

        return

    def gp_mutate_branch_terminal_gen(self, terminal_dtype):

        """
        Generate a single Terminal node.

        """

        self.evolve_dtype_get_terminal(terminal_dtype)
        self.pop_node_c1 = ''
        self.pop_node_c2 = ''
        self.pop_node_c3 = ''

        self.evolve_node_commit()  # commit new node to array

        return

    def gp_mutate_branch_node_gen(self, parent_arity_sum, prior_sibling_arity, prior_siblings, node_dtype):

        """
        Generate a single label (func or term) for
        -- parent_arity_sum
        -- prior_sibling_arity
        -- prior_siblings
        -- '2b')

        """

        if np.random.choice(['func', 'term']) == 'func':  # randomly selected as Function
            self.dtype_function_select(node_dtype)  # retrieve a function, input-reverse the parent-function (f2b -> we need 2f input)
            self.gp_mutate_branch_link_child(parent_arity_sum, prior_sibling_arity, prior_siblings)  # establish links to children
        else:
            self.evolve_dtype_get_terminal(node_dtype)  # was here
            self.pop_node_c1 = ''
            self.pop_node_c2 = ''
            self.pop_node_c3 = ''

        self.evolve_node_commit()  # commit new node to array
        prior_sibling_arity = prior_sibling_arity + self.pop_node_arity  # sum the arity of prior siblings

        return prior_sibling_arity

    def gp_mutate_branch_link_child(self, parent_arity_sum, prior_sibling_arity, prior_siblings):

        """


        """

        for n in range(1, len(self.tree[3])):  # increment through all nodes (exclude 0) in array 'tree'
            if int(self.tree[4][n]) == self.pop_node_depth - 1:  # find all nodes that reside at the prior (parent) 'node_depth'
                c_buffer = self.pop_NODE_ID + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

                if self.pop_node_arity == 0:  # terminal in a Grow Tree
                    self.pop_node_c1 = ''
                    self.pop_node_c2 = ''
                    self.pop_node_c3 = ''

                elif self.pop_node_arity == 1:  # 1 child
                    self.pop_node_c1 = c_buffer
                    self.pop_node_c2 = ''
                    self.pop_node_c3 = ''

                elif self.pop_node_arity == 2:  # 2 children
                    self.pop_node_c1 = c_buffer
                    self.pop_node_c2 = c_buffer + 1
                    self.pop_node_c3 = ''

                elif self.pop_node_arity == 3:  # 3 children
                    self.pop_node_c1 = c_buffer
                    self.pop_node_c2 = c_buffer + 1
                    self.pop_node_c3 = c_buffer + 2

                else:
                    self.printpl('e', '\n\t\033[31m ERROR! In tree_build_child_link: pop_node_arity =', self.pop_node_arity, '\033[0;0m')
                    self.plagih_pause()  # consider special instructions for this

        return

    def gp_crossover(self, mode='replace_same_types'):

        """
        TODO now, partners do not exchange their branches, just parent a takes a branch of parent_b
        - select parent a and b
        - select swappable branche for parent_a from parent_b
            - select a node in a (and crossover here, no matter what)
        - delete parent_a branch and insert parent_b branch (which tactic?)

        """
        self.printpl('gg', 'Crossover...')

        for n in range(self.evolve_cross):  # quantity of Trees to be generated through Crossover, (now not accounting for 2 children each, changed)

            # 1. Select two parents and their branches
            parent_a = self.gp_selection_tournament(self.tourn_size)  # perform tournament selection for 'parent_a'
            parent_b = self.gp_selection_tournament(self.tourn_size)  # perform tournament selection for 'parent_b'

            # 2. Get the branches for parent a that can be exchanged
            branch_a, branch_b, convert_a = self.gp_crossover_get_swap_branches(parent_a, parent_b)

            if convert_a:
                self.printpl('i', 'Forced conversion is needed between two trees.')
                offspring = self.gp_crossover_insert(parent_a, branch_a, parent_b, branch_b, converter=convert_a)  # perform Crossover
            else:
                offspring = self.gp_crossover_insert(parent_a, branch_a, parent_b, branch_b)  # perform Crossover

            offspring = self.tree_store_meta_lastgen(offspring, modification='c')
            offspring = self.tree_modifyable_nodes_set(offspring)
            self.population_new.append(offspring)  # append the 1st child to next generation of Trees

        return

    def gp_crossover_tree_branch_copy(self, tree, branch):

        """
        This method prepares a stand-alone Tree as a copy of the given branch.

        """

        new_tree = np.array(
            [['TREE_ID'], ['tree_type'], ['tree_depth_base'], ['NODE_ID'], ['node_depth'], ['node_type'],
             ['node_label'], ['node_parent'], ['node_arity'], ['node_c1'], ['node_c2'], ['node_c3'], ['fitness'], ['node_modify'], ['parsimony']])

        for n in range(len(branch)):
            node = branch[n]
            branch_top = int(branch[0])
            TREE_ID = 'copy'
            tree_type = tree[1][1]
            tree_depth_base = int(tree[4][branch[-1]]) - int(tree[4][branch_top])  # subtract depth of 'branch_top' from the last in 'branch'
            NODE_ID = tree[3][node]
            node_depth = int(tree[4][node]) - int(tree[4][branch_top])  # subtract the depth of 'branch_top' from the current node depth
            node_type = tree[5][node]
            node_label = tree[TR_nlabel][node]
            node_parent = ''  # updated by evolve_parent_link_fix(), below
            node_arity = tree[TR_narity][node]
            node_c1 = ''  # updated by evolve_child_link_fix(), below
            node_c2 = ''
            node_c3 = ''
            fitness = ''
            node_modify = '1'  # sfeh Test this
            parsimony = ''

            new_tree = np.append(new_tree,
                                 [[TREE_ID], [tree_type], [tree_depth_base], [NODE_ID], [node_depth], [node_type],
                                  [node_label], [node_parent], [node_arity], [node_c1], [node_c2], [node_c3],
                                  [fitness], [node_modify], [parsimony]], 1)

        new_tree = self.evolve_node_renum(new_tree)
        new_tree = self.evolve_fix_link_child(new_tree)
        new_tree = self.evolve_fix_link_parent(new_tree)

        # new_tree = self.data_tree_clean(new_tree)

        return new_tree

    def gp_crossover_tree_prune(self, tree, depth):

        '''
        This method reduces the depth of a Tree. Used with Crossover, the input value 'branch' can be a partial Tree
        (branch) or a full tree, and it will operate correctly. The input value 'depth' becomes the new maximum depth,
        where depth is defined as the local maximum + the user defined adjustment.

        Arguments required: tree, depth
        '''

        nodes = []

        for n in range(1, len(tree[3])):

            if int(tree[4][n]) == depth and tree[5][n] == 'func':
                tree[5][n] = 'term'  # mutate type 'func' to 'term'
                node_dtype = self.dtype_label_get_dtype(tree[TR_nlabel][n])
                tree[TR_nlabel][n] = self.dtype_dtype_get_term(node_dtype)  # replace label

            elif int(tree[4][n]) > depth:  # record nodes deeper than the maximum allowed Tree depth
                nodes.append(n)

            else:
                pass  # as int(tree[4][n]) < depth and will remain untouched

        tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
        tree = self.evolve_node_arity_fix(tree)  # fix all node arities

        return tree

    def gp_crossover_get_partner_node_id(self, function_label, partner_tree, partner_branch_id, mode='same_type'):
        """
        -> Crossover: Returns a node_id in the partner tree, that can be swapped
        """

        node_dtype = self.dtype_node_get_dtype(function_label, self.evolve_label_get_terminal(function_label))
        node_options = []
        # TODO check if the tree is large enough?
        if mode == 'same_type':  # only return a node with the same function type
            for i, label in enumerate(partner_tree[TR_nlabel][1:]):
                if self.dtype_label_get_dtype(label) == node_dtype:
                    node_options.append(i+1)  # +1, we skipped the first element

            if node_options:  # Found at least one!
                np.random.shuffle(node_options)  # otherwise, the first closest element is always taken (-> smallest)
                return min(node_options, key=lambda x: abs(x-partner_branch_id))  # return closest node
            else:
                return 0  # No matching node found :(
        elif mode == 'random':
            self.printpl('t', 'mode: Do the same as in the upper function, but choose randomly?')
        else:
            self.printpl('e', 'Mode not found', mode)
            raise

    def gp_crossover_get_swap_branches(self, parent_a, parent_b):
        """
        Returns two branches (node ids) that can be replaced and a converter (if needed)

        - Option1: Try swapping based on dtype
            - Choose a random node in tree a
            - Has b an equal dtype-node?
            -> yes: swap them, RETURN
        - Option2: Try swapping with the other dtype
            - Choose a random node in tree a of different dtype
            - Has b an equal dtype-node?
            -> yes: swap them, RETURN
        - Option3: No matching dtypes. Force conversion.
            - Choose a random node in tree a
            - Choose a random node in tree b
            - Convert parent_b's node to parent_a's node
        -> only returns convert_a if we need to force a conversion
        """
        a_convert, b_convert = '', ''
        mode = self.crossover_type_safety_mode

        # 1. swappable nodes exist?
        if mode == 'replace_same_types':
            a_node = self.evolve_get_mutatable_node_id(parent_a, mode='crossover_no_root')
            a_dtype = self.dtype_label_get_dtype(parent_a[TR_nlabel][a_node])
            try:  # swapping with random dtype
                b_node = self.evolve_get_mutatable_node_id(parent_b, mode='crossover', same_dtype=a_dtype)
            except:  # no matching dtypes. maybe the other one?
                mode = 'try_dtype'  # better luck next time
                # TODO directly force_conversion?

        if mode == 'try_dtype':
            if '2f' in a_dtype:
                a_dtype = '2b'
            elif '2b' in a_dtype:
                a_dtype = '2f'
            else:
                self.printpl('e', 'dtype should be either 2f or 2b, it is', a_dtype)
                raise
            try:  # swapping with other dtype
                a_node = self.evolve_get_mutatable_node_id(parent_a, mode='crossover_no_root', same_dtype=a_dtype)  # now
                b_node = self.evolve_get_mutatable_node_id(parent_b, mode='crossover', same_dtype=a_dtype)  # also a_dtype
            except:  # no matching dtypes. maybe the other one?
                mode = 'force_conversion'

        if mode == 'force_conversion':
            a_node = self.evolve_get_mutatable_node_id(parent_a, mode='crossover_no_root')
            b_node = self.evolve_get_mutatable_node_id(parent_b, mode='crossover')
            a_dtype = self.dtype_label_get_dtype(parent_a[TR_nlabel][a_node])
            b_dtype = self.dtype_label_get_dtype(parent_b[TR_nlabel][b_node])
            # check if the two labels are compartibel
            if self.dtype_outcome_equi_test(a_dtype, b_dtype):
                self.printpl('w', 'Crossover: Forcing conversion between', parent_a[TR_nlabel][a_node], parent_b[TR_nlabel][b_node])
                a_convert = self.dtype_convert_dtypes_get_dummylabel(a_dtype, b_dtype)
                # b_convert = self.sfeh_convert_dtypes_get_dummylabel(b_dtype, a_dtype)

        # TODO add try-except-case with point mutation (same arity, swapping dtype)

        a_branch = self.evolve_subtree_get(parent_a, node=a_node)
        b_branch = self.evolve_subtree_get(parent_b, node=b_node)

        return a_branch, b_branch, a_convert

    def gp_crossover_insert(self, parent_x, branch_x, parent_y, branch_y, converter=0):

        """
        Perform a crossover between nodes that are crossoverable in terms of function txpes
        get: parent a, b and their branches
        return: puts branch_y into parent_x
        """

        y_root = int(branch_y[0])
        x_root = int(branch_x[0])

        if len(branch_y) == 1:  # if the branch from the incoming parent contains only one node (terminal)

            parent_x[TR_nlabel][x_root] = parent_y[TR_nlabel][y_root]  # replace label with that of a particular node in 'branch_y'
            parent_x[5][x_root] = 'term'  # replace type
            parent_x[TR_narity][x_root] = 0  # set terminal arity

            parent_x = np.delete(parent_x, branch_x[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
            parent_x = self.evolve_fix_link_child(parent_x)  # fix all child links
            parent_x = self.evolve_node_renum(parent_x)  # renumber all 'NODE_ID's

        else:  # we are working with a branch from 'parent' >= depth 1 (min 3 nodes)

            self.tree = self.gp_crossover_tree_branch_copy(parent_y, branch_y)  # generate stand-alone 'gp.tree' with properties of 'branch_y'
            parent_x = self.evolve_subtree_build_insert(parent_x, branch_x)  # insert new 'branch_x' at point of mutation 'branch_top' in tourn_winner 'offspring'
            parent_x = self.gp_crossover_tree_prune(parent_x, self.tree_depth_max)  # prune to the max Tree depth + adjustment

        return parent_x

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Utility  functions to evolve a tree       |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def evolve_subtree_build(self, chosen_tree, branch_nodes_list):

        """
        Given: Tree and a node list
        - checks how far to build down (todo?)
        - checks the old nodes dtype, etc.
        - checks if we are not too far down the tree
        -

        returns: new tree
        """

        # 1. How far can we build down?
        branch_top = int(branch_nodes_list[0])
        branch_depth = self.evolve_subtree_depth_get(chosen_tree, branch_top, len(branch_nodes_list))  # sfeh solution to keep tree kind of small, dont forget the mode

        # 2. Get the old-node's information
        old_node_label = chosen_tree[TR_nlabel][branch_nodes_list[0]]  # <,        +,-,*,8,action0 ...
        old_node_type = chosen_tree[5][branch_nodes_list[0]]   # func,     term, ...
        old_node_dtype = self.dtype_node_get_dtype(old_node_label, old_node_type)  # '2f', 'f2b', ...

        # 3. check if we are on a too-low level to branch mutate...
        if branch_depth < 0:  # this has never occured ... yet
            self.printpl('e', 'In evolve_grow_mutate: branch_depth', branch_depth, '< 0')
            self.plagih_pause()

        elif branch_depth == 0:  # the point of mutation ('branch_top') chosen resides at the maximum allowable depth, so mutate term to term
            # 50:50 decision in function below if constant or variable
            self.printpl('vv', 'Ended in the lowest depth with old label', old_node_label)
            chosen_tree[TR_nlabel][branch_top] = self.dtype_dtype_get_term(old_node_dtype)

        # 4. We can now mutate the branch!
        else:
            # 5. 50:50 terminal or function
            if np.random.choice(['func', 'term']) == 'term':  # mutate 'branch_top' to a terminal and delete all nodes beneath (no subsequent nodes are added to this branch)
                # 5.1 We insert a terminal here
                chosen_tree[5][branch_top] = 'term'  # replace type ('func' to 'term' or 'term' to 'term')
                term_type = self.dtype_node_get_dtype(old_node_label, old_node_type)
                chosen_tree[TR_nlabel][branch_top] = self.dtype_dtype_get_term(term_type)  # replace with a correct label
                chosen_tree = np.delete(chosen_tree, branch_nodes_list[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
                chosen_tree = self.evolve_node_arity_fix(chosen_tree)  # fix all node arities (term)
                chosen_tree = self.evolve_fix_link_child(chosen_tree)  # fix all child links (func)
                chosen_tree = self.evolve_node_renum(chosen_tree)  # renumber all 'NODE_ID's
            else:
                # 5.2 We insert a function here
                # self.branch_new_tree_build('mutant', 'b', old_node_dtype, branch_depth)  # build new Tree ('gp.tree') with a maximum depth which matches 'branch'
                self.gp_mutate_branch_new_tree_build('mutant', 'b', old_node_dtype, branch_depth)  # build new Tree ('gp.tree') with a maximum depth which matches 'branch'
                chosen_tree = self.evolve_subtree_build_insert(chosen_tree, branch_nodes_list)  # insert new 'branch' at point of mutation 'branch_top' in tourn_winner 'tree'
            # because we already know the maximum depth to which this branch can grow, there is no need to prune after insertion

        return chosen_tree

    def evolve_subtree_depth_get(self, chosen_tree, branch_top, amount_replaced_nodes, mode='random'):  # sfeh other default
        """
        Return the size of the tree to be inserted.
        Should not be set to maximum to reduce complexity!
        """

        # TODO consider tree size of last tree, # TODO consider random tree size, # TODO consider always maximum tree size, TODO is this already considered by 50:50 func-term?

        branch_depth_upper_bound = self.tree_depth_max - int(chosen_tree[4][branch_top])  # 'tree_depth_max' - depth at 'branch_top' to set max size of new branch
        if mode == 'maximum':
            branch_depth = branch_depth_upper_bound
        elif mode == 'base_depth':
            branch_depth = min(branch_depth_upper_bound, self.pop_tree_depth_base)
        elif mode == 'random':
            branch_depth = max(branch_depth_upper_bound, np.random.randint(0, 1+max(branch_depth_upper_bound, 3)))  # SFEH random depth, I hope this is enough to guarantee tree size
        else:
            self.printpl('e', 'sfeh_get_new_tree_size does not accept this mode: ' + str(mode))
            raise
        return branch_depth

    def evolve_subtree_build_insert(self, winner_tree, branch_nodes):

        """
        This method enables the insertion of insert_tree in place of a branch (which is a node_id). It works with 3 inputs: local 'tree' is being
        modified; local 'branch' is a section of 'tree' which will be removed; and the global 'gp.tree' (recycling this
        variable from initial population generation) is the new Tree to be inserted into 'tree', replacing 'branch'.

        branch_nodes = [5,6,8,9] node that are changed

        The end result is a Tree with a mutated branch. Pretty cool, huh?
        """

        ### PART 1 - insert branch_top from 'gp.tree' into 'tree' ###
        branch_top = int(branch_nodes[0])
        winner_tree[5][branch_top] = 'func'  # update type ('func' to 'term' or 'term' to 'term'); this modifies gp.tree[5][1] from 'root' to 'func'
        winner_tree[TR_nlabel][branch_top] = self.tree[TR_nlabel][1]  # copy node_label from new tree
        winner_tree[TR_narity][branch_top] = self.tree[TR_narity][1]  # copy node_arity from new tree
        winner_tree = np.delete(winner_tree, branch_nodes[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')

        c_buffer = self.evolve_c_buffer(winner_tree, branch_top)  # generate c_buffer for point of mutation ('branch_top')
        winner_tree = self.evolve_subtree_insert_child(winner_tree, branch_top, c_buffer)  # insert a single new node ('branch_top')
        winner_tree = self.evolve_node_renum(winner_tree)  # renumber all 'NODE_ID's

        ### PART 2 - insert branch_body from 'gp.tree' into 'tree' ###
        node_count = 2  # set node count for 'gp.tree' to 2 as the new root has already replaced 'branch_top' (above)

        while node_count < len(self.tree[3]):  # increment through all nodes in the new Tree ('gp.tree'), starting with node 2

            for j in range(1, len(winner_tree[3])):  # increment through all nodes in tourn_winner ('tree')

                if winner_tree[5][j] == '':
                    winner_tree[5][j] = self.tree[5][node_count]  # copy 'node_type' from branch to tree
                    winner_tree[TR_nlabel][j] = self.tree[TR_nlabel][node_count]  # copy 'node_label' from branch to tree
                    winner_tree[TR_narity][j] = self.tree[TR_narity][node_count]  # copy 'node_arity' from branch to tree

                    if winner_tree[5][j] == 'term':
                        winner_tree = self.evolve_fix_link_child(winner_tree)  # fix all child links
                        winner_tree = self.evolve_node_renum(winner_tree)  # renumber all 'NODE_ID's

                    if winner_tree[5][j] == 'func':
                        c_buffer = self.evolve_c_buffer(winner_tree, j)  # generate 'c_buffer' for point of mutation ('branch_top')
                        winner_tree = self.evolve_subtree_insert_child(winner_tree, j, c_buffer)  # insert new nodes
                        winner_tree = self.evolve_fix_link_child(winner_tree)  # fix all child links
                        winner_tree = self.evolve_node_renum(winner_tree)  # renumber all 'NODE_ID's

                    node_count = node_count + 1  # exit loop when 'node_count' reaches the number of columns in the array 'gp.tree'

        return winner_tree

    def evolve_subtree_insert_child(self, tree, node, c_buffer):

        """
        Insert child node into the copy of a parent Tree.

        """

        if int(tree[TR_narity][node]) == 0:  # if arity = 0
            self.printpl('e', 'In evolve_child_insert: node', node, 'has arity 0')
            self.plagih_pause()  # consider special instructions for this

        elif int(tree[TR_narity][node]) == 1:  # if arity = 1
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[4][c_buffer] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

        elif int(tree[TR_narity][node]) == 2:  # if arity = 2
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[4][c_buffer] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

            tree = np.insert(tree, c_buffer + 1, '', axis=1)  # insert node for 'node_c2'
            tree[3][c_buffer + 1] = c_buffer + 1  # node ID
            tree[4][c_buffer + 1] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer + 1] = int(tree[3][node])  # parent ID

        elif int(tree[TR_narity][node]) == 3:  # if arity = 3
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[4][c_buffer] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

            tree = np.insert(tree, c_buffer + 1, '', axis=1)  # insert node for 'node_c2'
            tree[3][c_buffer + 1] = c_buffer + 1  # node ID
            tree[4][c_buffer + 1] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer + 1] = int(tree[3][node])  # parent ID

            tree = np.insert(tree, c_buffer + 2, '', axis=1)  # insert node for 'node_c3'
            tree[3][c_buffer + 2] = c_buffer + 2  # node ID
            tree[4][c_buffer + 2] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer + 2] = int(tree[3][node])  # parent ID

        else:
            self.printpl('e', 'In evolve_child_insert: node', node, 'arity > 3')
            self.plagih_pause()  # consider special instructions for this (pause)

        return tree

    def evolve_subtree_get(self, tree, node=0):

        """
        chooses a mutatable branch to mutate
        - specify a starting node
        - return all childnodes as list
        """
        # 1. branch_top = a valid node
        branch = np.array([])  # the array is necessary in order to len(branch) when 'branch' has only one element
        if node > 0:  # Crossover: Option to specify own starting node
            branch_top = node
        else:
            branch_top = self.evolve_get_mutatable_node_id(tree, mode='mutate_branch_no_root')  # "2" returns mutable node (except root node)

        # 2. Also return all child nodes
        branch_eval = self.tree_node_get_childlist(tree, branch_top)  # generate tuple of 'branch_top' and subsequent nodes
        branch_symp = sympify(branch_eval)  # convert string into something useful
        branch = np.append(branch, branch_symp)  # append list to array
        branch = np.sort(branch)  # sort nodes in branch for Crossover.

        return branch

    def evolve_label_get_terminal(self, node_label):
        """
        return terminal or function according to the label
        """

        if node_label in function_dtypes_dict:
            return 'func'
        else:
            return 'term'

    def evolve_node_commit(self):

        """
        Commit the values of a new node (root, function, or terminal) to the array 'tree'.
        TODO
        """

        self.tree = np.append(self.tree, [[self.pop_TREE_ID], [self.pop_tree_type], [self.pop_tree_depth_base], [self.pop_NODE_ID], [self.pop_node_depth], [self.pop_node_type], [self.pop_node_label],
                                          [self.pop_node_parent], [self.pop_node_arity], [self.pop_node_c1], [self.pop_node_c2], [self.pop_node_c3], [self.pop_fitness], [1], [self.pop_parsimony]], 1)

        self.pop_NODE_ID = self.pop_NODE_ID + 1

        return

    def evolve_c_buffer(self, tree, node):

        """
        This method serves the very important function of determining the links from parent to child for any given
        node. The single, simple formula [parent_arity_sum + prior_sibling_arity - prior_siblings] perfectly determines
        the correct position of the child node, already in place or to be inserted, no matter the depth nor complexity
        of the tree.

        This method is currently called from the evolution methods, but will soon (I hope) be called from the first
        generation Tree generation methods (above) such that the same method may be used repeatedly.

        Called by: evolve_child_link_fix, evolve_banch_top_copy, evolve_branch_body_copy

        Arguments required: tree, node
        """

        parent_arity_sum = 0
        prior_sibling_arity = 0
        prior_siblings = 0

        for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'

            if int(tree[4][n]) == int(tree[4][node]) - 1:  # find parent nodes at the prior depth
                if tree[TR_narity][n] != '': parent_arity_sum = parent_arity_sum + int(
                    tree[TR_narity][n])  # sum arities of all parent nodes at the prior depth

            if int(tree[4][n]) == int(tree[4][node]) and int(tree[3][n]) < int(
                    tree[3][node]):  # find prior siblings at the current depth
                if tree[TR_narity][n] != '':
                    prior_sibling_arity = prior_sibling_arity + int(tree[TR_narity][n])  # sum prior sibling arity
                prior_siblings = prior_siblings + 1  # sum quantity of prior siblings

        c_buffer = node + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

        return c_buffer

    def evolve_fix_link_child(self, tree):

        """
        In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

        This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.

        """

        for node in range(1, len(tree[3])):
            c_buffer = self.evolve_c_buffer(tree, node)  # generate c_buffer for each node
            tree = self.evolve_fix_link_child_doit(tree, node, c_buffer)  # update child links for each node

        return tree

    def evolve_fix_link_child_doit(self, tree, node, c_buffer):

        """
        Link each parent node to its children.

        """

        if int(tree[3][node]) == 1:
            # SFEH Root can only be ignored, if root was not changed
            c_buffer = c_buffer + 1  # if root (node 1) is passed through this method

        if tree[TR_narity][node] != '':

            if int(tree[TR_narity][node]) == 0:  # if arity = 0
                tree[9][node] = ''
                tree[10][node] = ''
                tree[11][node] = ''

            elif int(tree[TR_narity][node]) == 1:  # if arity = 1
                tree[9][node] = c_buffer
                tree[10][node] = ''
                tree[11][node] = ''

            elif int(tree[TR_narity][node]) == 2:  # if arity = 2
                tree[9][node] = c_buffer
                tree[10][node] = c_buffer + 1
                tree[11][node] = ''

            elif int(tree[TR_narity][node]) == 3:  # if arity = 3
                tree[9][node] = c_buffer
                tree[10][node] = c_buffer + 1
                tree[11][node] = c_buffer + 2

            else:
                self.printpl('e', '\n\t\033[31m In evolve_child_link: node', node, 'has arity', tree[TR_narity][node])
                raise  # self.plagih_pause()  # consider special instructions for this (pause)

        return tree

    def evolve_fix_link_parent(self, tree):

        """
        In a given Tree, fix 'parent_id' for all nodes.

        This is automatically handled in all mutations except with Crossover due to the need to copy branches 'a' and
        'b' to their own trees before inserting them into copies of	the parents.

        Technically speaking, the 'node_parent' value is not used by any methods. The parent ID can be completely out
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

    def evolve_node_arity_fix(self, tree):

        """
        In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.

        This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and
        Crossover.

        """

        for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'

            if tree[5][n] == 'term':  # check for discrepency
                tree[TR_narity][n] = '0'  # set arity to 0
                tree[9][n] = ''  # wipe 'node_c1'
                tree[10][n] = ''  # wipe 'node_c2'
                tree[11][n] = ''  # wipe 'node_c3'
                tree[TR_nmodify][n] = '1'

        return tree

    def evolve_node_renum(self, tree):

        """
        Renumber all 'NODE_ID' in a given tree.

        This is required after a new generation is evolved as the NODE_ID numbers are carried forward from the previous
        generation but are no longer in order.

        """

        for n in range(1, len(tree[3])):
            tree[3][n] = n  # renumber all Trees in given population

        return tree

    def evolve_get_mutatable_node_id(self, tree, mode='', same_dtype=''):
        """
        Returns a mutatable node for point-mutation
        -> no_root handles
        """
        # TODO only works for 2-array functions

        node_ids = []

        # 1. Build up a list with nodes
        if same_dtype:
            for i, label in enumerate(tree[TR_nlabel]):
                if tree[TR_nmodify][i] == '1':  # also skips node 0
                    # TODO make this faster
                    node_dtype = self.dtype_label_get_dtype(tree[TR_nlabel][i])
                    if self.dtype_outcome_equi_test(node_dtype, same_dtype):
                        node_ids.append(int(tree[3][i]))
        else:
            for i, x in enumerate(tree[5]):
                if tree[TR_nmodify][i] == '1':
                    node_ids.append(int(tree[3][i]))

        # 2. Kick out root if it is there?
        if 'no_root' in mode:  # delete root node
            node_ids = [x for x in node_ids if x != 1]

        # 3: return the node. Not safe, could be try-except block.
        # eg: all nodes are not modifiable
        # eg. all nodes are not of correct type
        node_id = np.random.choice(node_ids)
        return node_id

    def evolve_dtype_get_terminal(self, node_dtype):

        """
        Define a single Terminal (variable extracted from the top row of the associated TRAINING data)

        """

        self.pop_node_type = 'term'
        self.pop_node_label = self.dtype_dtype_get_term(node_dtype)  # get a terminal
        self.pop_node_arity = 0

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def tree_modifyable_nodes_set(self, chosen_tree):
        """
        Sets all the origin core nodes back to non-modifyable
        """
        # Set all nodes to be modifiable (=1)
        for i, tmp in enumerate(chosen_tree[TR_nmodify][1:]):
            chosen_tree[TR_nmodify][i + 1] = 1

        # Find no-modifyables in Origin
        non_modifiable_nodes = []
        if self.origin_tree[TR_nmodify][1] == '0':  # check is modifiable nodes are specified
            non_modifiable_nodes.extend(self.tree_nomodifyable_nodes_get(1, chosen_tree, 1))

        for non_modifiable in non_modifiable_nodes:
            chosen_tree[TR_nmodify][non_modifiable] = 0

        return chosen_tree

    def tree_nomodifyable_nodes_get(self, origin_node, chosen_tree, chosen_node):
        """
        Returns a list of nodes that are not supposed to be modified
        """

        if self.origin_tree[TR_nmodify][origin_node] == '0':
            non_modifiables = []
            non_modifiables.append(int(chosen_tree[3][chosen_node]))
            for child in [9, 10, 11]:
                if self.origin_tree[child][origin_node] != '':
                    next_origin_node = int(self.origin_tree[child][origin_node])
                    next_chosen_node = int(chosen_tree[child][chosen_node])
                    tmp = self.tree_nomodifyable_nodes_get(next_origin_node, chosen_tree, next_chosen_node)
                    if tmp is not None:
                        non_modifiables.extend(tmp)
            return non_modifiables
        else:
            return

    def tree_build_type_constant_get(self, term_type='', mode='float-1to1', range=[]):
        """
        todo random samples
        Returns a constant that fits into the position
        -- term_type = 'float'
        """
        if range:
            return np.random.uniform(range[0], range[1])

        if term_type == 'bool':
            return np.random.choice([True, False])
        elif term_type == 'float':
            if mode == 'float-1to1':
                return np.random.uniform(-1, 1)
            elif mode == 'intTotal_10':
                return np.random.random_integers(-10, 10)
            elif mode == 'random_optimised':
                return np.random.choice([-10, -5, -2, -1, -1, -0.8, -0.6, -0.5, -0.4, -0.2, 0, 10,
                                         5, 2, 1, 1, 0.8, 0.6, 0.5, 0.4, 0.2, 0])
            else:
                # sfeh: gibt viele Verteilungen: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html
                self.printpl('e', 'You did not take care of the kind of numbers you want to have')
                raise
        elif term_type == 'int':
            # TODO give more opportunities, similar to random floats
            return np.random.random_integers(-10, 10)
        else:
            self.printpl('w', 'Please specify your desired datatype if possible. Trying to return value similar to terminals.')
            self.printpl('e', 'This term type should not occur, I guess', term_type)
            term_type = np.random.choice(self.terminal_types)
            return self.tree_build_type_constant_get(term_type=term_type)

    def tree_expr_sympify(self, tree):

        """
        returns the sympifyed expression
        -> self.monitor_failed_sympys_amount
        """

        algo_raw_str = str(self.tree_expr_raw(tree, 1))  # pass the root 'node_id', then flatten the Tree to a string
        try:  # plagih: try block needed. simpify can not handle if then else.
            x = plagih_sympify(algo_raw_str)
            strx = str(x)

            # if 'zoo' in strx or 'nan' in strx:
            #     self.monitor_failed_sympys_amount += 1

            if 'zoo' in strx:
                x = re.sub('zoo', '10', strx)  # TODO how to handle zoo?

            if 'nan' in strx:  # Happens when 0/0 occurs. This tree is worth nothing anyways
                self.printpl('w', 'We had a "nan"')
                return sympy_dummy
            else:
                return x
        except:
            self.printpl('w', 'In sympify. Caused by this raw algorithm: ' + str(algo_raw_str))
            # todo.
            return sympy_dummy

    def tree_expr_raw(self, tree, node_id):

        """
        Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').

        This method is called once per Tree, but may be called at any time to prepare an expression for any full or
        partial (branch) Tree contained in 'population'. Pass the starting node for recursion via the local variable
        'node_id' where the local variable 'tree' is a copy of the Tree you desire to evaluate.

        """

        node_id = int(node_id)

        if tree[TR_narity, node_id] == '0':  # arity of 0 for the pattern '[term]'
            return '(' + tree[TR_nlabel, node_id] + ')'  # 'node_label' (function or terminal)
        else:
            if tree[TR_narity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
                return '(' + self.tree_expr_raw(tree, tree[9, node_id]) + tree[TR_nlabel, node_id] + ')'

            elif tree[TR_narity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
                # This if case is for 2-ary ops that can not be inline. like Min(a, b)
                if tree[TR_nlabel, node_id] not in functions_inline_dict:  # in non_inline_functions:
                    return '(' + tree[TR_nlabel, node_id] + '(' + self.tree_expr_raw(tree, tree[9, node_id]) + ', ' + self.tree_expr_raw(tree, tree[10, node_id]) + '))'
                else:
                    return '(' + self.tree_expr_raw(tree, tree[9, node_id]) + tree[TR_nlabel, node_id] + self.tree_expr_raw(tree, tree[10, node_id]) + ')'  # Klammern, da sympify sonst abkacnen könnte

            # if then else
            elif tree[TR_narity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
                return '(Ifte(' + self.tree_expr_raw(tree, tree[9, node_id]) + ', ' + self.tree_expr_raw(tree, tree[10, node_id]) + ', ' + self.tree_expr_raw(tree, tree[11, node_id]) + '))'

    def tree_node_get_childlist(self, tree, node_id):

        """
        Evaluate all or part of a Tree and return a list of all 'NODE_ID's.

        This method generates a list of all 'NODE_ID's from the given Node and below. It is used primarily to generate
        'branch' for the multi-generational mutation of Trees.

        Pass the starting node for recursion via the local variable 'node_id' where the local variable 'tree' is a copy
        of the Tree you desire to evaluate.

        """

        node_id = int(node_id)

        if tree[TR_narity, node_id] == '0':  # arity of 0 for the pattern '[NODE_ID]'
            return tree[3, node_id]  # 'NODE_ID'

        else:
            if tree[TR_narity, node_id] == '1':  # arity of 1 for the pattern '[NODE_ID], [NODE_ID]'
                return tree[3, node_id] + ', ' \
                       + self.tree_node_get_childlist(tree, tree[9, node_id])

            elif tree[TR_narity, node_id] == '2':  # arity of 2 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID]'
                return tree[3, node_id] + ', ' \
                       + self.tree_node_get_childlist(tree, tree[9, node_id]) + ', ' \
                       + self.tree_node_get_childlist(tree, tree[10, node_id])

            elif tree[TR_narity, node_id] == '3':  # arity of 3 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID], [NODE_ID]'
                return tree[3, node_id] + ', ' \
                       + self.tree_node_get_childlist(tree, tree[9, node_id]) + ', ' \
                       + self.tree_node_get_childlist(tree, tree[10, node_id]) + ', ' \
                       + self.tree_node_get_childlist(tree, tree[11, node_id])

    def tree_parsimony(self, tree, parsimony_distance='rel_ari_1'):
        """
        parsimony_distance: compute the chosen distance by the user. Default is best historic distance

        """
        if parsimony_distance == 'total_count_nodes':
            return int(tree[3][-1:])  # returns the tree size
        elif parsimony_distance == 'total_tree_depth':
            return tree[4][1]     # returns the tree size
        elif parsimony_distance == 'total_karoo_original':  # do not use with long variable names
            algo_raw_str = str(self.tree_expr_raw(tree, 1))
            return len(str(algo_raw_str))
        # elif parsimony_distance == 'total_simplified':
        #     algo_sym = self.tree_expr_sympify(tree)
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
        if len(tree[TR_nlabel]) < len(self.origin_tree[TR_nlabel]):
            return 1

        distance = 0

        # iterate over every node in the new tree
        for i, arity in enumerate(tree[TR_narity]):
            if i == 0:  # skip node 0. the description
                continue
            elif i < len(self.origin_tree[TR_nlabel]):  # Make sure we stay within the tree index. <= does not work
                if self.origin_tree[TR_nlabel][i] != tree[TR_nlabel][i]:  # is it different from the origin?
                    distance = distance + int(arity)  # add the nodes arity. double-punishes large trees
            else:
                distance = distance + int(arity)

        return max(distance, 1)  # make sure, it does not return 0

    def tree_parsimony_ted(self, tree):
        """
        The Tree Edit distance (TED) ('coolest' distance)
        - the amount of changes that have to be applied to the origin to equality are counted
        """
        # TODO distanzfunktion für Anzahl der Änderungen schreiben
        og_expr = self.tree_expr_sympify(self.origin_tree)
        changed_expr = self.tree_expr_sympify(tree)
        return

    def tree_store_fitness(self, tree, fitness):

        """
        Store the fitness within the tree np-array

        """

        fitness = float(fitness)
        fitness = round(fitness, self.precision)

        tree[12][1] = fitness  # store the fitness with each tree

        return

    def tree_store_parsimony(self, tree, parsimony):
        """
        Store the parsimony within the tree np-array
        """
        tree[TR_parsimony][1] = parsimony

    def tree_store_meta_lastgen(self, tree, modification=''):

        """
        Remove all fitness data from a given tree.

        This is required after a new generation is evolved as the fitness of the same Tree prior to its mutation will
        no longer apply.

        """

        # save information about how good last changes were
        for i in range(min(self.tree_depth_min, 5), 2, -1):  # 5,4,3,2
            tree[1][i] = tree[1][i-1]    # The last modifications
            tree[12][i] = tree[12][i-1]  # The last fitness
            tree[TR_parsimony][i] = tree[TR_parsimony][i-1]  # The last parsimony (TODO) # TREE_ID,1,a,b,c -> TREE_ID,1,a,a,b

        # What needs to be assigned later
        tree[1][1] = modification  # wipe last modification data
        tree[0][1] = ''  # -> TREE_ID,,
        tree[12][1] = ''  # wipe fitness data
        tree[TR_parsimony][1] = ''  # wipe parsimony data

        return tree

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Generation loop        |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gen_prepare_parameters(self):
        """
        Sets the parameters for this generation
        - reset population_new
        - Lineary increase threshold for parsimony
        """
        self.printpl('g', 'Preparing to evolve Generation', self.gen_id, '...')
        self.population_new = ['Plagih GP - Evolving Generation']  # initialise population_new to host the next generation

        full_parsimony_factor = 2  # working with the maximum parsimony for at least some generations
        gen_relation = min((full_parsimony_factor * self.gen_id) / self.gen_max, 1)
        self.parsimony_min_max[0] = int(gen_relation * self.parsimony_min_max[1])

        self.monitor_performance(mode='reset')

        return

    def gen_finalize(self):

        """
        From raw population_new to new population_genepool
        - Gene_pool with tree's parsimony (and store info in the tree)
        -

        """

        self.pop_enum_trees()           # pop +tree_id
        self.gp_genepool_parsimony()   # gene +parsimony, self.gene_pool: [1,2,6,8,14,20]
        self.pop_genepool_fitness()     # gene +fitness
        self.pop_pareto_front()

        # self.population_genepool = self.pop_copy(self.population_new, ['Plagih GP Generation ' + str(self.gen_id)])
        self.population_genepool = self.pop_copy_genepool(self.population_new)
        self.data_save_population(self.population_new, 'new')
        self.monitor_performance(mode='update')

        return

    def gen_olymp_update(self):
        """
        The olymp is where the godlike contestants reside.
        In each generation, the olymp searches for new god contestants
        """
        self.printpl('t', 'TODO Olymp for candidates')
        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Population specific     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_pareto_front(self):
        """
        Builds up the pareto front
        - iterate over all parsimonys
            - always check for the best of each parsimony
        """
        # 1. Check every potential candidate
        for candidate in self.gene_pool:

            # 2.
            tree = self.population_new[candidate]
            tmp_parsim = tree[TR_parsimony][1]
            tmp_fitness = tree[TR_fitness][1]
            algo_raw_str = str(self.tree_expr_raw(tree, 1))  #str(self.tree_expr_sympify(tree))
            expr_raw_hash = hash(algo_raw_str)

            # 3. is the tree better than the current best in this parsimony level?
            if tmp_parsim in self.pareto_parsimony_to_fitness:
                cmp_fitness = self.pareto_parsimony_to_fitness[tmp_parsim]
                if (self.fitness_type == 'max' and tmp_fitness > cmp_fitness) or \
                        (self.fitness_type == 'min' and tmp_fitness < cmp_fitness):

                    self.printpl('v', 'Found new tree with parsimony', tmp_parsim, 'and fitness', tmp_fitness, 'which dominated old fitness', )

                    # delete the old tree from the paretofront
                    old_tree_hash = self.pareto_parsimony_get_hash[tmp_parsim]
                    del self.pareto_best_trees[old_tree_hash]

                    # update everything to the new tree
                    self.pareto_best_trees.update({expr_raw_hash: algo_raw_str})
                    self.pareto_parsimony_to_fitness[tmp_parsim] = tmp_fitness
                    self.pareto_parsimony_get_hash[tmp_parsim] = expr_raw_hash

                    # sfeh add the best trees to the olymp?
                else:
                    return  # not a special tree
            else:
                self.pareto_best_trees.update({expr_raw_hash: algo_raw_str})
                self.pareto_parsimony_to_fitness.update({tmp_parsim: tmp_fitness})
                self.pareto_parsimony_get_hash.update({tmp_parsim: expr_raw_hash})

        return

    def pop_enum_trees(self):
        """
        outsourced enumeration of trees in a population
        """
        for tree_id in range(1, len(self.population_new)):  #
            self.population_new[tree_id][0][1] = tree_id

    def pop_copy(self, population_x, title):

        """
        Copy one population to another.
        """
        popolation_y = [title]  # an empty list stores a copy of the prior generation

        for tree in range(1, len(population_x)):  # increment through each Tree in the current population
            tree_copy = np.copy(population_x[tree])  # copy each array in the current population
            popolation_y.append(tree_copy)  # add each copied Tree to the new population list

        return popolation_y

    def pop_copy_genepool(self, pop_x):

        """
        Copy the genepool.
        """
        pop_y = ['Genepool-Population in Generation ' + str(self.gen_id)]  # empty list

        for tree in self.gene_pool:  # increment through each Tree in the current population
            tree_copy = np.copy(pop_x[tree])  # copy each array in the current population
            pop_y.append(tree_copy)  # add each copied Tree to the new population list

        return pop_y

    def pop_genepool_fitness(self):
        """
        Compute the fitness for every tree
        """

        for tree_id in self.gene_pool:
            algo_sym = self.tree_expr_sympify(self.population_new[tree_id])
            fitness_train = self.eval_fitness_traindata(algo_sym)
            self.tree_store_fitness(self.population_new[tree_id], fitness_train)  # store Fitness and parsimony with each Tree

            if self.fitness_type == 'max':
                if fitness_train > self.origin_fitness_train:
                    self.origin_dominators.update({tree_id: algo_sym})
            else:
                if fitness_train < self.origin_fitness_train:
                    self.origin_dominators.update({tree_id: algo_sym})

        self.printpl('p', '\n\033[36m ', len(list(self.origin_dominators.keys())), 'trees\033[1m', np.sort(list(self.origin_dominators.keys())), '\033[0;0m\033[36moffer the highest fitness scores.\033[0;0m')

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to use evaluate (tensorflow)      |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def eval_fitness_traindata(self, algo_sym):
        """
        returns the fitness of a tree
        - kick out dummy
        - check, if fitness is in hash table
        - otherwise, compute with tensorflow
        """

        # If algo_sym is '1' - aka an error occured in sympy
        if algo_sym == sympy_dummy:
            return self.fitness_bad_dummy

        # 1. Try to get the fitness from hashtable
        expr_sym_str = str(algo_sym)
        expr_hash = hash(expr_sym_str)
        if expr_hash in self.hashtable_fitness_train:
            return self.hashtable_fitness_train[expr_hash]

        # 2. calculate the fitness and store it in the hash table
        fitness = self.eval_tf(expr_sym_str, self.data_train)['fitness']
        self.hashtable_fitness_train[expr_hash] = fitness
        return fitness

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

                num_terminals = len(self.terminals)
                num_actions = len(self.actions)

                for i in range(num_terminals):
                    var = self.terminals[i]
                    if '2f' in self.dtype_node_get_dtype(var, 'term'):
                        tensors[var] = tf.constant(data[:, i], dtype=tf.float32)  # converts data into vectors
                    else:  # '2b'
                        tensors[var] = tf.constant(data[:, i], dtype=tf.bool)

                for i in range(num_actions):
                    var = self.actions[i]
                    action_dtype = self.dtype_node_get_dtype(var, 'term')
                    if '2f' in action_dtype:
                        tensors[var] = tf.constant(data[:, num_terminals + i], dtype=tf.float32)  # converts data into vectors
                    elif '2b' in action_dtype:  # '2b'
                        self.printpl('t', 'Currently no kernel available for boolean fitness')
                        tensors[var] = tf.constant(data[:, i], dtype=tf.bool)
                    else:
                        self.printpl('e', 'Kernel not known for:', var, 'which is', action_dtype)

                # 2- Transform string expression into TF operation graph
                tf_result = self.eval_tf_ast_expr(expr, tensors)
                pred_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

                # TODO currently does only support one label
                solution = tensors['action0']  # solution value is assumed to be stored in this terminal
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
                    that you raise the minimum node count to keep it from converging on the value of '1'. Consider writing or 
                    integrating a more sophisticated kernel.
                    """

                    pairwise_fitness = tf.abs(solution - tf_result)

                elif self.kernel == 'm':  # MATCH kernel

                    """
                    This is used for demonstration purposes only.
                    """

                    # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
                    RTOL, ATOL = 1e-05, 1e-08  # fixes above issue by checking if a float value lies within a range of values
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
        # print('Current expr:', expr)  # importantprint
        tree = ast.parse(expr, mode='eval').body

        return self.eval_tf_expr_graph(tree, tensors)

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
            return operator_dict[type(node.op)](self.eval_tf_expr_graph(node.left, tensors), self.eval_tf_expr_graph(node.right, tensors))

        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            return operator_dict[type(node.op)](self.eval_tf_expr_graph(node.operand, tensors))

        elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)
            # special case: If-then-else
            if node.func.id == 'Ifte':
                return operator_dict[node.func.id](
                    tf.dtypes.cast(self.eval_tf_expr_graph(node.args[0], tensors), tf.bool),
                    self.eval_tf_expr_graph(node.args[1], tensors),
                    self.eval_tf_expr_graph(node.args[2], tensors))
            # special case: Min and Max accept 2 or more arguments. Many errors. Therefore not in use anymore.
            # if node.func.id in non_inline_multielem_functions:  # Min, Max are now Mini and Maxi. These only accept two inputs.
            #     # self.printpl('e', [self.tf_expr_graph(arg, tensors) for arg in node.args])
            #     return operators[node.func.id]([self.tf_expr_graph(arg, tensors) for arg in node.args])  # the star '*' makes the difference

            if node.func.id == 'Ftob':
                self.printpl('i', 'float was converted to bool in tensorflow')
                return tf.dtypes.cast(*[self.eval_tf_expr_graph(arg, tensors) for arg in node.args], dtype=tf.bool)
            elif node.func.id == 'Btof':
                return tf.dtypes.cast(*[self.eval_tf_expr_graph(arg, tensors) for arg in node.args], dtype=tf.float32)

            # The actual handling. The '*' inserts all the arguments (in this case 2) into the function.
            return operator_dict[node.func.id](*[self.eval_tf_expr_graph(arg, tensors) for arg in node.args])

        elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
            return self.eval_tf_chain_bool(node.values, operator_dict[type(node.op)], tensors)

        elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
            return self.eval_tf_chain_compare([node.left] + node.comparators, node.ops, tensors)

        elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
            if type(node.value) is not type(True):
                self.printpl('e', 'This True/False name constant is something else', node.value)
                raise
            try:
                return tf.constant(node.value)
            except:
                self.printpl('e', 'Oh no this was not True or False')
                raise
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
            return tf.logical_and(operator_dict[type(ops[0])](x, y), self.eval_tf_chain_compare(comparators[1:], ops[1:], tensors))
        else:
            return operator_dict[type(ops[0])](x, y)
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

    # random TODO replace and with &, see https://docs.sympy.org/latest/_modules/sympy/core/relational.html

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to receive correct dtypes (f2b,.) |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def dtype_func_get_func(self, function_type, arity=0):
        """
        This only accepts functions as inputs. (point mutation)
        No need to handle terminals
        """
        if arity > 0:  # Point mutation only works within same arity
            # Do this, if a
            if function_type == 'f2f':
                return np.random.choice(self.functions_array[0][arity])
            elif function_type == 'f2b':
                return np.random.choice(self.functions_array[1][arity])
            elif function_type == 'b2b':
                return np.random.choice(self.functions_array[2][arity])
            elif function_type == 'b2f':
                return np.random.choice(self.functions_array[3][arity])
            elif function_type == 'b2f2f':
                return np.random.choice(self.functions_array[4][arity])  # sfeh okay that does not make sense tbh
            else:
                self.printpl('e', 'Function was not found in function_types_dict', function_type)
                raise

        if function_type == 'f2f':
            return np.random.choice(self.functions_f2f)
        elif function_type == 'f2b':
            return np.random.choice(self.functions_f2b)
        elif function_type == 'b2b':
            return np.random.choice(self.functions_b2b)
        elif function_type == 'b2f':
            return np.random.choice(self.functions_b2f)
        elif function_type == 'b2f2f':
            return np.random.choice(self.functions_b2f2f)  # sfeh okay that does not make sense tbh
        else:
            self.printpl('e', 'Function was not found in function_types_dict', function_type)
            raise

    def dtype_dtype_get_func(self, function_dtype):
        """
        This fills in a function that fits the type of the function/terminal before.
        terminal  '2f' -> '_2f', arity
        function 'f2f' -> '_2f', arity
        function 'b2f2f' -> '_2f', arity
        > ->
        """
        if '2f' in function_dtype:
            new_label = np.random.choice(self.functions_2f)
            return new_label, function_arity_dict[str(new_label)]
        elif '2b' in function_dtype:
            new_label = np.random.choice(self.functions_2b)
            return new_label, function_arity_dict[str(new_label)]
        else:
            self.printpl('e', 'Warning: Function was not found in function_types_dict', function_dtype)
            raise

    def dtype_node_get_dtype(self, node_label, node_type):
        """
        'term' or 'func'
        """
        self.printpl('f', 'dtype_get_dtype4node')
        if node_type == 'term':
            if 'True' in node_label or 'False' in node_label:
                return '2b'
            elif 'observation' in node_label:
                term_position = self.terminals.index(node_label)
                return function_dtypes_dict[self.terminal_types[term_position]]
            elif 'action' in node_label:
                term_position = self.actions.index(node_label)
                return function_dtypes_dict[self.action_types[term_position]]
            else:  # only 'float' left
                return '2f'
        elif node_type == 'func':
            return function_dtypes_dict[node_label]
        else:
            self.printpl('e', 'This node_type is not known', node_type)
            raise

    def dtype_label_get_dtype(self, node_label):
        """
        returns dtype for a label
        """

        node_type = self.evolve_label_get_terminal(node_label)
        node_dtype = self.dtype_node_get_dtype(node_label, node_type)
        return node_dtype

    def dtype_dtype_get_term(self, node_dtype):
        """
        Returns a terminal that fits to the given type.

        function: f2b -> 2b needed
        terminal:  2f -> 2f needed
        --> check if it is function, aka _2f
        --> check if it is terminal, aka f2

        Modes:
        var_and_const: return randomly (50:50) a variable or a constant
        terminal_only: return                  a variable
        Todo Introduce constants-mode, where the user can give constant types (similar to functions)?
        """

        # node_dtype == '2f' or 'f2' in node_dtype:
        if '2f' in node_dtype:
            terminals_correct = self.terminals_float
            the_type = 'float'
        elif '2b' in node_dtype:
            terminals_correct = self.terminals_bool
            the_type = 'bool'
        else:
            self.printpl('e', 'Probably, you have to check if your "function" is actually a terminal. dtype', node_dtype)
            raise

        try:
            if np.random.choice(['var', 'const']) == 'var':  # our choice is variable
                if terminals_correct:  # Is there an entry in the list?
                    return np.random.choice(terminals_correct)  # ...so we return one
            return self.tree_build_type_constant_get(term_type=the_type)  # otherwise: constant (There are always constants :P)

        except ValueError:
            self.printpl('w', 'Should not happen. Did not find a terminal. Made up a ' + the_type + ' constant.')
            return self.tree_build_type_constant_get(term_type=the_type)

    def dtype_outcome_equi_test(self, a_dtype, b_dtype):
        """
        Dummy. Returns, whether two dtypes are equal
        """
        return a_dtype in b_dtype or b_dtype in a_dtype

    def dtype_convert_dtypes_get_dummylabel(self, a_dtype, b_dtype):
        """
        convert a-to-b dummy
        """
        if '2b' in a_dtype and '2f' in b_dtype:
            return 'Btof'
        if '2f' in a_dtype and '2b' in b_dtype:
            return 'Ftob'
        else:
            self.printpl('e', 'One of those two cases should happen', a_dtype, b_dtype)
            raise

    def dtype_function_select(self, func_dtype):

        """
        Returns a function with the same outcome

        """

        self.pop_node_type = 'func'
        new_function = self.dtype_dtype_get_func(func_dtype)
        self.pop_node_label = new_function[0]
        self.pop_node_arity = int(new_function[1])
        self.pop_node_modify = 1

        return

        ### Terminal Nodes ###

        # Idea: values are recognized and adjusted (+0.3, e.g.)?
        # Idea: insert numbers as 0-arity functions? -> naah

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to display output information     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def printpl(self, message_type, *args):  # plagih naming

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

        Colors: (examples)
            BLACK   = '\033[30m'
            RED     = '\033[31m'
            GREEN   = '\033[32m'
            YELLOW  = '\033[33m'
            BLUE    = '\033[34m'
            MAGENTA = '\033[35m'
            CYAN    = '\033[36m'
            WHITE   = '\033[37m'
            RESET   = '\033[39m'

        Background: (examples)
            BLACK   = '\033[40m'
            RED     = '\033[41m'

        ...maybe these look better?
        class BColors:  # sfeh can be deleted
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'
        """

        message_posttxt = '\033[39m'
        raise_error, pause = False, False

        if message_type in self.display:
            message_pretxt = '\033[39m'  # default color
            if message_type == 'i':
                message_style = '\033[36m'
                message_pretxt = 'Info: '  # cyan
            elif message_type == 'e':
                message_style = '\033[31m'
                message_pretxt = 'ERROR: '  # red
                raise_error = True
            elif message_type == 'w':  # warning
                message_style = '\033[93m'
                message_pretxt = 'Warning: '  # Warning-yellow
            elif message_type == 'g':
                message_style = '\033[32m'
                message_pretxt = 'Generation: '  # green
            elif message_type == 'v':  # verbose
                message_style = '\033[37m'
                message_pretxt = 'Verbose: '  # white
            elif message_type == 'p':  # pause
                message_style = '\033[33m'
                message_pretxt = 'Pause(TODO): '  # Yellow
                pause = True
            elif message_type == 'f':  # function
                message_style = '\033[35m'
                message_pretxt = 'Func: '  # Magenta
            elif message_type == 'o':  # original
                # Just show it
                message_style = ''
                message_pretxt = ''
            else:
                message_style = ''
                self.printpl('e', 'Display-mode', message_type, 'not known.')

            print(message_style + message_pretxt + ' '.join(map(str, args)) + message_posttxt)

            if raise_error:
                raise

            if pause:
                self.plagih_pause()  # correct pause?
        return

    def plot_end(self, y, mode='', plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label=''):
        # insert artificial data

        # # self.plot_end('s', plt_title='Sympify zoo and nan', plt_y_label='Amount')
        # if y_name == 's':  # there must be a better sulotion
        #     y = self.plot_failed_sympys_amount
        # else:
        #     self.printpl('e', 'Monitoring, this is not available', y_name)

        # # if variance
        # if mode == 'variance':
        #     self.printpl('e', 'variance not available', y_array)
        #     means = np.mean(y, axis=0)
        #     stds = np.std(y, axis=0)
        #     n = means.size

        x = np.arange(len(y))
        plt.plot(x, y, label=plt_curve_label)

        if plt_x_label and plt_y_label:
            plt.xlabel(plt_x_label)
            plt.ylabel(plt_y_label)

        if plt_title:
            plt.title(plt_title)

        # plt.legend()
        # plt.yscale('linear')
        # plt.ylim(-200, -50)
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
        Enables (g)eneration, (i)nteractive, and (d)e(b)ug display modes to offer the (pause) menu at each prompt.

        See plagih_pause() for an explanation of the value being passed.

        Called by: the functions called by PART 4 of plagih_gp()

        Arguments required: none
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
                     'tree_depth_max': self.tree_depth_max,
                     'tree_depth_min': self.tree_depth_min,
                     'tree_pop_max': self.tree_pop_max,
                     'gen_id': self.gen_id,
                     'gen_max': self.gen_max,
                     'tourn_size': self.tourn_size,
                     'evolve_repro': self.evolve_repro,
                     'evolve_point': self.evolve_point,
                     'evolve_branch': self.evolve_branch,
                     'evolve_cross': self.evolve_cross,
                     'fittest_dict': self.origin_dominators,
                     'pop_last_len': len(self.population_genepool),
                     'pop_new_len': len(self.population_new),
                     'path': self.path}

        menu_dict = menu.pause(menu_dict)  # call the external function menu.pause

        ### PART 2 - unpack values returned from menu.pause ###
        input_a = menu_dict['input_a']
        input_b = menu_dict['input_b']
        self.display = menu_dict['display']
        self.tree_depth_min = menu_dict['tree_depth_min']
        self.gen_max = menu_dict['gen_max']
        self.tourn_size = menu_dict['tourn_size']
        self.evolve_repro = menu_dict['evolve_repro']
        self.evolve_point = menu_dict['evolve_point']
        self.evolve_branch = menu_dict['evolve_branch']
        self.evolve_cross = menu_dict['evolve_cross']

        ### PART 3 - execute the user queries returned from menu.pause ###
        if input_a == 'esc':
            return 2  # breaks out of the plagih_gp() or plagih_pause_refer() loop

        elif input_a == 'eval':  # evaluate a Tree against the TEST data
            algo_sym = self.tree_expr_sympify(self.population_new[input_b])  # generate the raw and sympified expression for the given Tree using SymPy
            self.printpl('o', '\n\t\033[36mTree', input_b, 'yields (sym):\033[1m', algo_sym, '\033[0;0m')  # print the sympified expression
            result = self.eval_tf(str(algo_sym), self.data_test, get_pred_labels=True)  # might change to algo_raw_str evaluation
            self.pause_fitness_test(result)  # TF tested 2017 02/02

        elif input_a == 'print_last':  # print a Tree from population_genepool
            self.display_tree(self.population_genepool[input_b])

        elif input_a == 'print_new':  # print a Tree from population_new
            self.display_tree(self.population_new[input_b])

        elif input_a == 'pop_last':  # list all Trees in population_genepool
            self.printpl('o', '')
            for tree_id in range(1, len(self.population_genepool)):
                algo_sym = self.tree_expr_sympify(self.population_genepool[tree_id])
                self.printpl('i', '\t\033[36m Tree', self.population_genepool[tree_id][0][1], 'yields (sym):\033[1m', algo_sym, '\033[0;0m')

        elif input_a == 'pop_new':  # list all Trees in population_new
            self.printpl('o', '')
            for tree_id in range(1, len(self.population_new)):
                algo_sym = self.tree_expr_sympify(self.population_new[tree_id])  # extract the expression
                self.printpl('o', '\t\033[36m Tree', self.population_new[tree_id][0][1], 'yields (sym):\033[1m', algo_sym, '\033[0;0m')

        elif input_a == 'load':  # load population_s to replace population_genepool
            self.data_pickle_recover(self.filename['s'])  # NEED TO replace 's' with a user defined filename

        elif input_a == 'write':  # write the evolving population_new to disk
            self.data_save_population(self.population_new, 'new')
            self.printpl('o', '\n\t All current members of the evolving population_new saved to plagih_gp/runs/[date-time]/population_new.csv')

        elif input_a == 'add':  # check for added generations, then exit plagih_pause and continue the run
            self.gen_max = self.gen_max + input_b  # if input_b > 0: self.gen_max = self.gen_max + input_b - REMOVED 2019 06/05

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
                self.printpl('o', '\t\033[36m Data row {} predicts value:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.
                      format(i, result['result'][i], result['solution'][i]))

            MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']

            self.printpl('o', '\n\t Origin fitness score: {}'.format(self.origin_fitness_train))
            self.printpl('o', '\n\t Regression fitness score: {}'.format(fitness))
            self.printpl('o', '\t Mean Squared Error: {}'.format(MSE))

            return

        elif self.kernel == 'm':

            """
            Print the accuracy for a MATCH kernel run against the test data.

            """

            for i in range(len(result['result'])):
                self.printpl('vv', '\t\033[36m Data row {} predicts match:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.format(i, result['result'][i], result['solution'][i]))

            self.printpl('v', 'Matching fitness score: {}'.format(result['fitness']))

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
        self.printpl('o','\n\033[1m\033[36m Tree ID', int(tree[0][1]), '\033[0;0m')

        for depth in range(0, self.tree_depth_max + 1):  # increment through all possible Tree depths - tested 2016 07/09
            self.printpl('o','\n', ind, '\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m')

            for node in range(1, len(tree[3])):  # increment through all nodes (redundant, I know)
                if int(tree[4][node]) == depth:
                    self.printpl('o','')
                    self.printpl('o',ind, '\033[1m\033[36m NODE:', tree[3][node], '\033[0;0m')
                    self.printpl('o',ind, '  type:', tree[5][node])
                    self.printpl('o',ind, '  label:', tree[6][node], '\tparent node:', tree[7][node])
                    self.printpl('o',ind, '  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node])

            ind = ind + '\t'

        self.printpl('o', 'TODO')
        self.eval_tf(tree)  # generate the raw and sympified expression for the entire Tree
        algo_raw_str = str(self.tree_expr_raw(tree, 1))
        self.printpl('o', '\t\033[36mTree', tree[0][1], 'yields (raw):', algo_raw_str, '\033[0;0m')
        self.printpl('o', '\t\033[36mTree', tree[0][1], 'yields (sym):\033[1m', '\033[0;0m')

        return

    def manual_expr_fitness(self, expr):
        fitness = self.eval_tf(expr, self.data_train)['fitness']
        self.printpl('i', 'Your algos fitness:', fitness)
        return

    # def obsolete_compare_genepool_fitness(self, tree_id, fitness, fitness_best):
    #     """
    #     obsolete?
    #     """
    #     algo_sym = self.tree_expr_sympify(self.population_new[tree_id])
    #
    #     ### PART 3 - COMPARE FITNESS OF ALL TREES IN CURRENT GENERATION ###
    #     if self.kernel == 'c':  # display best fit Trees for the CLASSIFY kernel
    #         if fitness >= fitness_best:  # find the Tree with maximum fitness score
    #             fitness_best = fitness  # set best fitness score
    #             self.origin_dominators.update({tree_id: algo_sym})  # add to dictionary if fitness >= prior
    #
    #     elif self.kernel == 'r':  # display best fit Trees for the REGRESSION kernel
    #         if fitness_best == 0:
    #             fitness_best = fitness  # set the baseline first time through
    #         if fitness <= fitness_best:  # find the Tree with minimum fitness score
    #             fitness_best = fitness  # set best fitness score
    #             self.origin_dominators.update({tree_id: algo_sym})  # add to dictionary if fitness <= prior
    #
    #     elif self.kernel == 'm':  # display best fit Trees for the MATCH kernel
    #         if fitness == self.data_train_rows:  # find the Tree with a perfect match for all data rows
    #             fitness_best = fitness  # set best fitness score
    #             self.origin_dominators.update({tree_id: algo_sym})  # add to dictionary if all rows match
    #
    #     else:
    #         raise self.printpl('e', 'No proper kernel was selected', self.kernel)
    #     return fitness_best

    # def obsolete_mutate_branch_function_build_grow(self):
    #
    #     """
    #     Build the branch full depth
    #     Builds
    #     """
    #
    #     for i in range(1, self.pop_tree_depth_base):  # the tree depth (-1, where the last functions are) sfeh: actually NO -1?
    #
    #         self.pop_node_depth = i  # increment 'node_depth'
    #         parent_arity_sum = 0
    #         prior_sibling_arity = 0  # reset for 'c_buffer' in 'children_link'
    #         prior_siblings = 0  # reset for 'c_buffer' in 'children_link'
    #
    #         # parent_arity_sum = amount of nodes (that have to be on this level)
    #         for j in range(1, len(self.tree[3])):  # increment through all nodes in array 'tree'
    #             if int(self.tree[4][j]) == self.pop_node_depth - 1:  # find parent nodes which reside at the prior depth
    #                 parent_arity_sum = parent_arity_sum + int(self.tree[TR_narity][j])  # sum arities of all parent nodes at the prior depth
    #
    #         # Set for every "free space" a function node (func)
    #         for j in range(1, len(self.tree[3])):  # increment through all nodes
    #             if int(self.tree[4][j]) == self.pop_node_depth - 1:  # ... find all parent nodes, one level above...
    #                 if self.tree[TR_nlabel][j] == 'Ifte':
    #                     prior_sibling_arity = self.evolve_subtree_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2b')  # ... generate a Function node
    #                     prior_siblings = prior_siblings + 1
    #                     prior_sibling_arity = self.evolve_subtree_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
    #                     prior_siblings = prior_siblings + 1
    #                     prior_sibling_arity = self.evolve_subtree_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
    #                     prior_siblings = prior_siblings + 1
    #                 else:
    #                     for k in range(1, int(self.tree[TR_narity][j]) + 1):  # k = 1,2
    #                         self.pop_node_parent = int(self.tree[3][j])  # set the nodes parent
    #                         parent_func_dtype = function_dtypes_dict[self.tree[TR_nlabel][self.pop_node_parent]]  # find parents node
    #                         func_dtype = parent_func_dtype[:2][::-1]  # parent 'f2b' -> '2f' child needed. Aka, the first two characters reversed
    #                         prior_sibling_arity = self.evolve_subtree_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, func_dtype)  # ... generate a Function node
    #                         prior_siblings = prior_siblings + 1  # sum sibling nodes (current depth) who will spawn their own children (cousins? :)
    #
    #     return
