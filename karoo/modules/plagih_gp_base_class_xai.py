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
import karoo.modules.plagih_gp_pause as menu
# sfeh import the pause later, maybe
# import karoo.modules.karoo_gp_pause as menu
import tensorflow as tf
import ast

# PLAGI imports
import re
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
from karoo.modules.plagih_sympy_extras import plagih_sympify
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats as st

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

operators = {ast.Add: tf.add,  # e.g., a + b
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
             # 'ftob': tf.dtypes.cast,
             # 'ftob': tf.dtypes.cast,
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

    'ftob': 'f2b',
    'btof': 'b2f',  # False->0, True->1, dummy-function
    'btof_extreme': 'b2f',  # False->-1, True->1. Does that make sense?

    'Ifte': 'b2f2f',  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Min': 'f2f',
    'Max': 'f2f',
    'Mini': 'f2f',
    'Maxi': 'f2f',
}
    # Storytime:
    # plagih_sympify can reduce an expression to 'Min(a, b, c)', but tensorflow and our framework does not like this
non_inline_multielem_functions = ['Min', 'Max']  # There is probably an actual way to call 'non_inline_functions'

non_inline_functions = ['Min', 'Max', 'Mini', 'Maxi', 'abs', 'sign', 'square', 'sqrt', 'log', 'log1p', 'cos', 'sin', 'tan', 'acos', 'asin', 'atan']
inline_functions = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=']

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
    'ftob': '1',
    'btof': 1,  # False->0, True->1, dummy-function
    'btof_extreme': 1,  # False->-1, True->1. Does that make sense?

    'Ifte': 3,  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Min': 2,
    'Max': 2,
    'Mini': 2,
    'Maxi': 2,
}

sympy_dummy = plagih_sympify(1)

np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees



class Base_GP(object):
    """
    This Base_BP class contains all methods for Karoo GP. Method names are differentiated from global variable names
    (defined below) by the prefix 'fx_' followed by an object and action, as in fx_display_tree(), with a few
    expections, such as fx_fitness_gene_pool().

    The method categories (denoted by +++ banners +++) are as follows:
        fx_karoo_					Methods to Run Karoo GP
        fx_data_					Methods to Load and Archive Data
        fx_init_					Methods to Construct the 1st Generation
        fx_eval_					Methods to Evaluate a Tree
        fx_fitness_					Methods to Train and Test a Tree for Fitness
        fx_nextgen_					Methods to Construct the next Generation
        fx_evolve_					Methods to Evolve a Population
        fx_display_					Methods to Visualize a Tree

    Error checks are quickly located by searching for 'ERROR!'
    """

    def __init__(self):

        """
        ### Global variables used for data management ###
        self.data_train				store train data for processing in TF
        self.data_test				store test data for processing in TF
        self.tf_device				set TF computation backend device (CPU or GPU)
        self.tf_device_log			employed for TensorFlow debugging

        self.data_train_cols		number of cols in the TRAINING data - see fx_data_load()
        self.data_train_rows		number of rows in the TRAINING data - see fx_data_load()
        self.data_test_cols			number of cols in the TEST data - see fx_data_load()
        self.data_test_rows			number of rows in the TEST data - see fx_data_load()

        self.functions				user defined functions (operators) from the associated files/[functions].csv
        self.terminals				user defined variables (operands) from the top row of the associated [data].csv
        self.coeff					user defined coefficients (NOT YET IN USE)
        self.fitness_type			fitness type
        self.datetime				date-time stamp of when the unique directory is created
        self.path					full path to the unique directory created with each run
        self.dataset				local path and dataset filename

        ### Global variables used for evolutionary management ###
        self.population_a			the root generation from which Trees are chosen for mutation and reproduction
        self.population_b			the generation constructed from gp.population_a (recyled)
        self.gene_pool				once-per-generation assessment of trees that meet min and max boundary conditions
        self.gen_id					simple n + 1 increment
        self.fitness_type			set in fx_data_load() as either a minimising or maximising function
        self.tree					axis-1, 13 element Numpy array that defines each Tree, stored in 'gp.population'
        self.pop_*					13 variables that define each Tree - see fx_init_tree_initialise()
        """

        self.algo_raw = []  # the raw expression generated by Sympy per Tree -- CONSIDER MAKING THIS VARIABLE LOCAL
        self.algo_sym = []  # the expression generated by Sympy per Tree -- CONSIDER MAKING THIS VARIABLE LOCAL
        self.fittest_dict = {}  # all Trees which share the best fitness score
        self.gene_pool = []  # store all Tree IDs for use by Tournament
        self.class_labels = 0  # the number of true class labels (data_y)

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Run Karoo GP                  |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def plagih_gp(self, kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, gen_max,
                  tourn_size, operators_file, samples_file, origin_tree_file, evolve_distribution, display, precision,
                  swim, mode, gene_pool_threshold, parsimony_min_max, monitor):

        # 1. set global variables to those local values passed from the user script
        self.kernel = kernel  # fitness function
        self.tree_depth_max = tree_depth_max  # maximum Tree depth for the entire run; limits bloat
        self.tree_depth_min = tree_depth_min  # minimum number of nodes
        self.tree_pop_max = tree_pop_max  # maximum number of Trees per generation
        self.gen_max = gen_max  # maximum number of generations
        self.tourn_size = tourn_size  # number of Trees selected for each tournament
        self.evolve_repro = evolve_distribution[0]  # quantity of a population generated through Reproduction
        self.evolve_point = evolve_distribution[1]  # quantity of a population generated through Point Mutation
        self.evolve_branch = evolve_distribution[2]  # quantity of a population generated through Branch Mutation
        self.evolve_cross = evolve_distribution[3]  # quantity of a population generated through Crossover
        self.evolve_missing = evolve_distribution[4]  # fill up the generation with candidates
        if self.evolve_missing > 0:
            exit()
        self.display = display
        self.precision = precision  # the number of floating points for the round function
        self.swim = swim  # pass along the gene_pool restriction methodology
        self.gene_pool_threshold = gene_pool_threshold
        self.parsimony_min_max = parsimony_min_max
        self.monitor = monitor
        fitt_dict = {'c': 'max', 'r': 'min', 'm': 'max'}
        self.fitness_type = fitt_dict[self.kernel]  # load fitness type

        # 2. Perform genetic programming
        self.main_data_load(operators_file, samples_file, origin_tree_file)
        self.main_directories_create()
        self.main_gen_first()
        self.main_generation_new_repeat()   # (main loop)
        self.main_terminate()  # archive populations and return to karoo_gp.py for a clean exit

        return

    def main_gen_first(self):
        """
        Everything that needs to be done for the first generation
        - Extracts "origin Tree" from file
        - Creates all other trees: origin tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        # 1.  set parameters for the generation
        self.gen_id = 1  # set initial generation ID
        self.gen_prepare_parameters()
        self.population_a = ['PLAGIH Karoo GP Extension by Simon Fehrer, Generation ' + str(self.gen_id)]  # initialise population_a to host the first generation
        self.population_b = ['placeholder']  # initialise population_b to satisfy fx_karoo_pause()

        self.pop_first_create()

        # if self.gen_max == 1:  # terminate here if constructing just one generation
        #     self.data_tree_write(self.population_a, 'a')  # save this single population to disk
        #     sys.exit()

        ### PART 3 - evaluate first generation of Trees ###
        self.printpl('g', 'Evaluate the first generation of Trees ...')
        self.fitness_gym(self.population_a)  # generate expression, evaluate fitness, compare fitness
        self.data_tree_write(self.population_a, 'a')  # save the first generation of Trees to disk

        self.monitor_performance(mode='collect')

    def main_generation_new_repeat(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """
        menu = 1
        while menu != 0:  # this allows the user to add generations mid-run and not get buried in nested iterations
            for self.gen_id in range(self.gen_id + 1, self.gen_max + 1):  # generation 2 to *max generation*
                self.gen_prepare_parameters()
                self.printpl('g', 'Evolve a population of Trees for Generation', self.gen_id, '...')
                self.population_b = ['Karoo GP - Evolving Generation']  # initialise population_b to host the next generation
                
                self.fitness_gene_pool()  # generate the viable gene pool
                self.fitness_olymp()

                #    Genetic programming on the old population
                self.pop_reproduce()  # method 1 - Reproduction
                self.pop_mutate_point()  # method 2 - Point Mutation
                self.pop_mutate_branch()  # method 3 - Branch Mutation
                self.pop_crossover()  # method 4 - Crossover

                self.eval_generation()  # evaluate all Trees in a single generation
                self.population_a = self.fx_evolve_pop_copy(self.population_b, ['Karoo GP Generation ' + str(self.gen_id)])
                self.monitor_performance(mode='collect')
            # SFEH das war mal da, hat nach dem Durchlaufen ohne Menu abgekackt
            # if mode == 's':
            #     menu = 0  # (s)erver mode - termination with completiont of prescribed run
            else:  # (d)esktop mode - user is given an option to quit, review, and/or modify parameters; 'add' generations continues the run
                print('\n\t\033[32m Enter \033[1m?\033[0;0m\033[32m to review your options or \033[1mq\033[0;0m\033[32muit\033[0;0m')
                # menu = self.fx_karoo_pause()
                # SFEH statt oben steht da jetzt da unten :D
                menu = 0

    def monitor_performance(self, mode=''):
        """
        monitors everything

        Helper:
        # plot_end(self, y, mode='', plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='')
        """

        # Init is here to safe memory
        if self.monitor['gen_fitness_avg'] == 'y':
            if mode == 'g':  # update every generation
                pass
                # self.plagih_candidate_fitness_eval()

        # Sympify errors
        if self.monitor['sympify_errors'] == 'y':
            if mode == 'collect':
                if self.monitor_failed_sympys_amount:
                    self.plot_failed_sympys_amount.append(self.monitor_failed_sympys_amount)
            elif mode == 'init':
                self.plot_failed_sympys_amount = []
            elif mode == 'show':
                if self.plot_failed_sympys_amount:
                    try:
                        self.plot_end('s', plt_title='Sympify zoo and nan', plt_y_label='Amount')
                    except:
                        self.printpl('e', 'plotting did not work for sympify')
            else:
                self.printpl('e', 'Display-mode not known or empty:', mode)

        # Average Tree size, parsimony

        return

    def main_terminate(self):
        """
        Terminates the evolutionary run (if yet in progress), saves parameters and data to disk, and cleanly returns
        the user to karoo_gp.py and the command line.

        Called by: fx_karoo_gp() and fx_karoo_pause_refer()

        Arguments required: none
        """
        self.monitor_performance(mode='show')
        self.data_params_write()
        target = open(self.filename['f'], 'w')
        target.close()  # initialize the .csv file for the final population
        self.data_tree_write(self.population_b, 'f')  # save the final generation of Trees to disk

        print('\n\t\033[32m Your Trees and runtime parameters are archived in karoo_gp/runs/[date-time]/\033[0;0m')
        print('\n\033[3m "It is not the strongest of the species that survive, nor the most intelligent,\033[0;0m')
        print('\033[3m  but the one most responsive to change."\033[0;0m --Charles Darwin\n')
        print('\033[3m Congrats!\033[0;0m Your Karoo GP run is complete.\n')
        sys.exit()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Load and Archive Data         |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def main_data_load(self, operators_file, samples_file, origin_tree_file_path):

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
            if tree.shape[0] == 14+1:  # (+ row 0)
                pass  # print('Origin Tree is: \n' + str(tree))
            else:
                raise print("Tree could not be imported correctly from .csv file.")

        # As we need it quite often, safe the origin tree's data
        self.origin_tree = tree
        self.tree_algo_expr_sympify(self.origin_tree)

        self.hashtable_fitness = {}
        self.origin_fitness = self.tree_fitness(self.origin_tree)

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
                            print('actions:', self.actions)
                        else:
                            raise print('Behaviour samples first line: Variables have to start with "o" or "a" to be recognized')

                    data_x, data_y = [], []

                else:  # convert every 'string' element to its datatype
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

        # Part 4 - from the dataset, extract TRAINING and TEST data ###
        # TODO die func kann sicher nicht mit 2d labels umgehen. Funktion macht das echt super uneffizient.
        x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size=0.2)  # 80/20 TRAIN/TEST split
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

        return

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

        # TODO do not always print this
        # print('self.functions_array: ')
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
            print('No Functions that create numbers were found')
            self.functions_2f = []

        # The Functions that create boolean values
        if self.functions_f2b:
            func_2b.extend(self.functions_f2b)
        if self.functions_b2b:
            func_2b.extend(self.functions_b2b)
        if func_2b:
            self.functions_2b = func_2b[:]
        else:
            print('No Functions that create bool were found')
            self.functions_2b = []

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
        self.filename.update({'a': self.path + 'population_a.csv'})
        target = open(self.filename['a'], 'w')
        target.close()  # initialise a .csv file for population 'a' (foundation)
        self.filename.update({'b': self.path + 'population_b.csv'})
        target = open(self.filename['b'], 'w')
        target.close()  # initialise a .csv file for population 'b' (evolving)
        self.filename.update({'f': self.path + 'population_f.csv'})
        target = open(self.filename['f'], 'w')
        target.close()  # initialise a .csv file for the final population (test)
        self.filename.update({'s': self.path + 'population_s.csv'})
        target = open(self.filename['s'], 'w')
        target.close()  # initialise a .csv file to manually load (seed)

        self.monitor_performance(mode='init')


    def data_tree_write(self, population, key):

        """
        Save population_* to disk.

        Called by: fx_karoo_gp, fx_eval_generation

        Arguments required: population, key
        """

        with open(self.filename[key], 'a', newline='') as csv_file:
            target = csv.writer(csv_file, delimiter=',')
            if self.gen_id != 1:
                target.writerows([''])  # empty row before each generation
            target.writerows([['Karoo GP by Kai Staats, improved by Simon Fehrer', 'Generation:', str(self.gen_id)]])

            for tree in range(1, len(population)):
                target.writerows([''])  # empty row before each Tree
                for row in range(0, 14+1):  # increment through each row in the array Tree (+ row 0)
                    target.writerows([population[tree][row]])

        return

    def data_params_write(self):

        """
        Save run-time configuration parameters to disk.

        """

        file = open(self.path + 'log_config.txt', 'w')
        file.write('Karoo GP')
        file.write('\n launched: ' + str(self.datetime))
        file.write('\n dataset: ' + str(self.dataset))
        file.write('\n')
        file.write('\n kernel: ' + str(self.kernel))
        file.write('\n precision: ' + str(self.precision))
        file.write('\n')
        # file.write('tree type: ' + tree_type)
        # file.write('tree depth base: ' + str(tree_depth_base))
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

        file = open(self.path + 'log_test.txt', 'w')
        file.write('Karoo GP')
        file.write('\n launched: ' + str(self.datetime))
        file.write('\n dataset: ' + str(self.dataset))
        file.write('\n')

        if len(self.fittest_dict) > 0:

            fitness_best = 0
            fittest_tree = 0

            # revised method, re-evaluating all Trees from stored fitness score
            for tree_id in range(1, len(self.population_b)):

                fitness = float(self.population_b[tree_id][12][1])

                if self.kernel == 'c':  # display best fit Trees for the CLASSIFY kernel
                    if fitness >= fitness_best:  # find the Tree with Maximum fitness score
                        fitness_best = fitness
                        fittest_tree = tree_id  # set best fitness Tree

                elif self.kernel == 'r':  # display best fit Trees for the REGRESSION kernel
                    if fitness_best == 0: fitness_best = fitness  # set the baseline first time through
                    if fitness <= fitness_best:  # find the Tree with Minimum fitness score
                        fitness_best = fitness
                        fittest_tree = tree_id  # set best fitness Tree

                elif self.kernel == 'm':  # display best fit Trees for the MATCH kernel
                    if fitness == self.data_train_rows:  # find the Tree with a perfect match for all data rows
                        fitness_best = fitness
                        fittest_tree = tree_id  # set best fitness Tree

            # elif self.kernel == '[other]': # use others as a template

            # print ('fitness_best:', fitness_best, 'fittest_tree:', fittest_tree)

            # test the most fit Tree and write to the .txt log
            self.tree_algo_expr_sympify(self.population_b[int(fittest_tree)])  # generate the raw and sympified expression for the given Tree using SymPy
            result = self.expr_fitness_eval(str(self.algo_sym), self.data_test, get_pred_labels=True)

            file.write('\n\t Origin fitness score: {}'.format(self.origin_fitness))

            file.write('\n\n Tree ' + str(fittest_tree) + ' is the most fit, with expression:')
            file.write('\n\n ' + str(self.algo_sym))

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

        # elif self.kernel == '[other]': # use others as a template

        else:
            file.write('\n\n There were no evolved solutions generated in this run... your species has gone extinct!')

        file.write('\n\n')
        file.close()

        return

    def data_tree_clean(self, tree):

        """
        This method aesthetically cleans the Tree array, removing redundant data.
        TODO this is unnecessary
        """

        # tree[0][2:] = ''  # A little clean-up to make things look pretty :)
        # tree[1][2:] = ''  # Ignore the man behind the curtain!
        # tree[2][2:] = ''  # Yes, I am a bit OCD ... but you *know* you appreciate clean arrays.

        return tree

    def fx_data_recover(self, population):

        """
        This method is used to load a saved population of Trees, as invoked through the (pause) menu where population_r
        replaces population_a in the karoo_gp/runs/[date-time]/ directory.

        Called by: fx_karoo_pause

        Arguments required: population (filename['s'])
        """

        with open(population, 'rb') as csv_file:
            target = csv.reader(csv_file, delimiter=',')
            n = 0  # track row count

            for row in target:
                print('row', row)

                n = n + 1
                if n == 1:
                    pass  # skip first empty row

                elif n == 2:
                    self.population_a = [row]  # write header to population_a

                else:
                    if row == []:
                        self.tree = np.array([[]])  # initialise Tree array

                    else:
                        if self.tree.shape[1] == 0:
                            self.tree = np.append(self.tree, [row], axis=1)  # append first row to Tree

                        else:
                            self.tree = np.append(self.tree, [row], axis=0)  # append subsequent rows to Tree

                    if self.tree.shape[0] == 14+1:  # (current tree rows + row 0
                        self.population_a.append(self.tree)  # append complete Tree to population list

        print('\n', self.population_a)

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Construct the 1st Generation  |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_first_create(self):
        """
        Constructs the first generation
        - loads the origin-tree from file
        - constructs the first generation from this tree with branch mutation
        """

        #TODO branch mutation in ALL subtrees? if more options are available
        #TODO safely create a complete generation
        self.printpl('g', 'Initial population...')

        self.origin_tree[0][1] = 1
        self.pop_first_tree_append(self.origin_tree)

        for TREE_ID in range(2, self.tree_pop_max + 1 - 1):

            tree = self.origin_tree.copy()  # is this necessary? probably.
            branch_nodes_list = self.tree_branch_get(tree)  # [6, 9, 10] select point of mutation and all nodes beneath
            tree = self.pop_mutate_branch_evolve(tree, branch_nodes_list)  # tree with new branch
            # 6 Fill the correct meta-data into the tree (and wipe the old fitness)
            tree = self.tree_store_meta_lastgen(tree, modification='i')  # wipe fitness data
            tree = self.tree_store_set_modify_nodes(tree)

            tree[0][1] = TREE_ID
            self.pop_first_tree_append(tree)

        self.printpl('g', '\n We have constructed a single, stochastic population of', self.tree_pop_max, 'Trees, and saved to disk')

    def pop_first_tree_append(self, tree):

        """
        Append Tree array to the foundation Population.

        """

        self.data_tree_clean(tree)  # clean 'tree' prior to storing
        self.population_a.append(tree)  # append 'tree' to population list

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Construct the 1st Generation  |
    # +++++++++++++++++++++++++++++++++++++++++++++
    def pop_mutate_branch_evolve(self, chosen_tree, branch_nodes_list):

        """
        Given: Tree and a node list

        Mutate a branch of one Tree and return it.
        The 'grow' method is used, 'full' does not exist in my world.

        returns: new tree
        """

        # 1. How far can we build down?
        branch_top = int(branch_nodes_list[0])
        branch_depth = self.pop_branch_depth_get(chosen_tree, branch_top)  # sfeh solution to keep tree kind of small, dont forget the mode

        # 2. Get the old-node's information
        old_node_label = chosen_tree[6][branch_nodes_list[0]]  # <,        +,-,*,8,action0 ...
        old_node_type = chosen_tree[5][branch_nodes_list[0]]   # func,     term, ...
        old_node_dtype = self.dtype_get_dtype4node(old_node_label, old_node_type)  # '2f', 'f2b', ...

        # 3. check if we are on a too-low level to branch mutate...
        if branch_depth < 0:  # this has never occured ... yet
            self.printpl('e', 'In fx_evolve_grow_mutate: branch_depth', branch_depth, '< 0')
            self.fx_karoo_pause()  # consider special instructions for this (pause) - 2019 06/08

        elif branch_depth == 0:  # the point of mutation ('branch_top') chosen resides at the maximum allowable depth, so mutate term to term
            # 50:50 decision in function below if constant or variable
            self.printpl('v', 'Ended in the lowest depth with old label', old_node_label)
            chosen_tree[6][branch_top] = self.dtype_get_term4dtype(old_node_dtype)

        # 4. We can now mutate the branch!
        else:
            # 5. 50:50 terminal or function
            if np.random.choice(['func', 'term']) == 'term':  # mutate 'branch_top' to a terminal and delete all nodes beneath (no subsequent nodes are added to this branch)
                # 5.1 We insert a terminal here
                chosen_tree[5][branch_top] = 'term'  # replace type ('func' to 'term' or 'term' to 'term')
                term_type = self.dtype_get_dtype4node(old_node_label, old_node_type)
                chosen_tree[6][branch_top] = self.dtype_get_term4dtype(term_type)  # replace with a correct label
                chosen_tree = np.delete(chosen_tree, branch_nodes_list[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
                chosen_tree = self.tree_node_arity_fix(chosen_tree)  # fix all node arities (term)
                chosen_tree = self.fx_evolve_child_link_fix(chosen_tree)  # fix all child links (func)
                chosen_tree = self.tree_node_renum(chosen_tree)  # renumber all 'NODE_ID's
            else:
                # 5.2 We insert a function here
                # self.branch_new_tree_build('mutant', 'b', old_node_dtype, branch_depth)  # build new Tree ('gp.tree') with a maximum depth which matches 'branch'
                self.evolve_branch_new_tree_build('mutant', 'b', old_node_dtype, branch_depth)  # build new Tree ('gp.tree') with a maximum depth which matches 'branch'
                chosen_tree = self.evolve_branch_insert(chosen_tree, branch_nodes_list)  # insert new 'branch' at point of mutation 'branch_top' in tourn_winner 'tree'
            # because we already know the maximum depth to which this branch can grow, there is no need to prune after insertion


        return chosen_tree

    def gen_prepare_parameters(self):
        """
        Sets the parameters for this generation
        - Lineary increase threshold for parsimony
        """

        self.monitor_failed_sympys_amount = 0  # monitoring

        # adjust parsimony threshold lineary
        gen_done_percentage = self.gen_id / self.gen_max
        self.parsimony_min_max[0] = int(gen_done_percentage * self.parsimony_min_max[1])

        return

    def pop_crossover_get_swap_branches(self, parent_a, parent_b, mode='replace_same_types'):
        """
        Returns two branches (node ids) that can be replaced and a converter (if needed)

        - Try swapping based on dtype
            - Choose a random node in tree a
            - Has b an equal dtype-node?
            -> yes: swap them, RETURN
        - Try swapping with the other dtype
            - Choose a random node in tree a of different dtype
            - Has b an equal dtype-node?
            -> yes: swap them, RETURN
        - No matching dtypes. Force conversion.
            - Choose a random node in tree a
            - Choose a random node in tree b
            - Convert parent_b's node to parent_a's node
        -> only returns convert_a if we need to force a conversion
        """
        a_convert, b_convert = '', ''

        # 1. swappable nodes exist?
        if mode == 'replace_same_types':
            a_node = self.tree_get_mutatable_node_id(parent_a, mode='crossover_no_root')
            a_dtype = self.dtype_get_dtype4label(parent_a[6][a_node])
            try:  # swapping with random dtype
                b_node = self.tree_get_mutatable_node_id(parent_b, mode='crossover', same_dtype=a_dtype)
            except:  # no matching dtypes. maybe the other one?
                mode = 'try_dtype'  # better luck next time
                # TODO directly force_conversion?

        # obsolete?
        if mode == 'try_dtype':
            if '2f' in a_dtype:
                a_dtype = '2b'
            elif '2b' in a_dtype:
                a_dtype = '2f'
            else:
                raise self.printpl('e', 'dtype should be either 2f or 2b, it is', a_dtype)
            try:  # swapping with other dtype
                a_node = self.tree_get_mutatable_node_id(parent_a, mode='crossover_no_root', same_dtype=a_dtype)  # now
                b_node = self.tree_get_mutatable_node_id(parent_b, mode='crossover', same_dtype=a_dtype)  # also a_dtype
            except:  # no matching dtypes. maybe the other one?
                mode = 'force_conversion'

        if mode == 'force_conversion':
            a_node = self.tree_get_mutatable_node_id(parent_a, mode='crossover_no_root')
            b_node = self.tree_get_mutatable_node_id(parent_b, mode='crossover')
            a_dtype = self.dtype_get_dtype4label(parent_a[6][a_node])
            b_dtype = self.dtype_get_dtype4label(parent_b[6][b_node])
            # check if the two labels are compartibel
            if self.dtype_outcome_equi_test(a_dtype, b_dtype):
                self.printpl('w', 'Crossover: Forcing conversion between', parent_a[6][a_node], parent_b[6][b_node])
                a_convert = self.dtype_convert_dtypes_get_dummylabel(a_dtype, b_dtype)
                # b_convert = self.sfeh_convert_dtypes_get_dummylabel(b_dtype, a_dtype)

        # TODO check if tree is too large
        # TODO add try-except-case with point mutation (same arity, swapping dtype)

        a_branch = self.tree_branch_get(parent_a, node=a_node)
        b_branch = self.tree_branch_get(parent_b, node=b_node)

        return a_branch, b_branch, a_convert

    def pop_crossover_insert(self, parent_x, branch_x, parent_y, branch_y, converter=0):

        """
        Perform a crossover between nodes that are crossoverable in terms of function txpes
        get: parent a, b and their branches
        return: puts branch_y into parent_x
        """
        # TODO converter is not used yet

        y_root = int(branch_y[0])
        x_root = int(branch_x[0])

        if len(branch_y) == 1:  # if the branch from the incoming parent contains only one node (terminal)

            parent_x[6][x_root] = parent_y[6][y_root]  # replace label with that of a particular node in 'branch_y'
            parent_x[5][x_root] = 'term'  # replace type
            parent_x[8][x_root] = 0  # set terminal arity

            parent_x = np.delete(parent_x, branch_x[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
            parent_x = self.fx_evolve_child_link_fix(parent_x)  # fix all child links
            parent_x = self.tree_node_renum(parent_x)  # renumber all 'NODE_ID's

        else:  # we are working with a branch from 'parent' >= depth 1 (min 3 nodes)

            self.tree = self.tree_branch_copy(parent_y, branch_y)  # generate stand-alone 'gp.tree' with properties of 'branch_y'
            parent_x = self.evolve_branch_insert(parent_x, branch_x)  # insert new 'branch_x' at point of mutation 'branch_top' in tourn_winner 'offspring'
            parent_x = self.evolve_tree_prune(parent_x, self.tree_depth_max)  # prune to the max Tree depth + adjustment - tested 2016 07/10

        return parent_x

    def evolve_branch_new_tree_build(self, TREE_ID, last_modification, old_node_type, tree_depth):

        """
        This method combines 4 sub-methods into a single method for ease of deployment. It is designed to executed
        within a loop such that an entire population is built. However, it may also be run from the command line,
        passing a single TREE_ID to the method.

        if terminal: 2f
        if func:    f2f
        """

        self.branch_tree_initialise(TREE_ID, tree_depth, last_modification)  # Create empty tree np-array
        self.branch_root_build(old_node_type)  # insert the first node with either '_2b' or '_2f'
        self.branch_function_build_full()  # build all the Function nodes
        self.init_terminal_build()  # build the Terminal nodes
        # TODO set tree_depth_base in tree.
        return  # each Tree is written to 'gp.tree'

    def branch_tree_initialise(self, TREE_ID, tree_depth, last_modification):

        """
        Assign 15 (14+1) global variables to the array 'tree'.

        Build the array 'tree' with 15 rows and initally, just 1 column of labels. This array will grow horizontally as
        each new node is appended. The values of this array are stored as string characters, numbers forced to integers at
        the point of execution.

        Use of the debug (db) interface mode enables the user to watch the genetic operations as they work on the Trees.

        Called by: fx_init_tree_build

        Arguments required: TREE_ID, tree_type, tree_depth_base
        """

        self.pop_TREE_ID = TREE_ID  # pos 0: a unique identifier for each tree
        self.pop_tree_type = last_modification  # pos 1: a global constant based upon the initial user setting
        self.pop_tree_depth_base = tree_depth  # pos 2: a global variable which conveys 'tree_depth_base' as unique to each new Tree
        self.pop_NODE_ID = 1  # pos 3: unique identifier for each node; this is the INDEX KEY to this array
        self.pop_node_depth = 0  # pos 4: depth of each node when committed to the array
        self.pop_node_type = ''  # pos 5: root, function, or terminal
        self.pop_node_label = ''  # pos 6: operator [+, -, *, ...] or terminal [a, b, c, ...]
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

    def branch_root_build(self, old_node_type):

        """
        Build a root node for a branch insert.
        """
        # eg b2f2f
        self.branch_function_select(old_node_type)  # select the operator for root

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
            print('\n\t\033[31m ERROR! In branch_root_build: pop_node_arity =', self.pop_node_arity, '\033[0;0m'); self.fx_karoo_pause()  # consider special instructions for this (pause) - 2019 06/08

        self.pop_node_type = 'func'  # used to be r00t, but what is it good for?

        self.branch_node_commit()

        return

        ### Function Nodes ###

    def tree_store_set_modify_nodes(self, chosen_tree):
        """
        Sets all the origin core nodes back to non-modifyable
        """
        # Set all nodes to be modifiable (=1)
        for i, tmp in enumerate(chosen_tree[13][1:]):
            chosen_tree[13][i + 1] = 1

        # Find no-modifyables in Origin
        non_modifiable_nodes = []
        if self.origin_tree[13][1] == '0':  # check is modifiable nodes are specified
            non_modifiable_nodes.extend(self.tree_nomodify_nodes_get(1, chosen_tree, 1))

        for non_modifiable in non_modifiable_nodes:
            chosen_tree[13][non_modifiable] = 0

        return chosen_tree

    def tree_nomodify_nodes_get(self, origin_node, chosen_tree, chosen_node):
        """
        Returns a list of nodes that are not supposed to be modified
        """

        if self.origin_tree[13][origin_node] == '0':
            non_modifiables = []
            non_modifiables.append(int(chosen_tree[3][chosen_node]))
            for child in [9, 10, 11]:
                if self.origin_tree[child][origin_node] != '':
                    next_origin_node = int(self.origin_tree[child][origin_node])
                    next_chosen_node = int(chosen_tree[child][chosen_node])
                    tmp = self.tree_nomodify_nodes_get(next_origin_node, chosen_tree, next_chosen_node)
                    if tmp is not None:
                        non_modifiables.extend(tmp)
            return non_modifiables
        else:
            return

    def branch_function_build_full(self):

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
                    parent_arity_sum = parent_arity_sum + int(self.tree[8][j])  # sum arities of all parent nodes at the prior depth

            # Set for every "free space" a function node (func)
            for j in range(1, len(self.tree[3])):  # increment through all nodes
                if int(self.tree[4][j]) == self.pop_node_depth - 1:  # ... find all parent nodes, one level above...
                    if self.tree[6][j] == 'Ifte':
                        prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2b')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                        prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                        prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                    else:
                        for k in range(1, int(self.tree[8][j]) + 1):  # k = 1,2
                            self.pop_node_parent = int(self.tree[3][j])  # set the nodes parent
                            parent_func_dtype = function_dtypes_dict[self.tree[6][self.pop_node_parent]]  # find parents node
                            func_dtype = parent_func_dtype[:2][::-1]  # parent 'f2b' -> '2f' child needed. Aka, the first two characters reversed
                            prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, func_dtype)  # ... generate a Function node
                            prior_siblings = prior_siblings + 1  # sum sibling nodes (current depth) who will spawn their own children (cousins? :)

        return

    def branch_function_build_grow(self):

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
                    parent_arity_sum = parent_arity_sum + int(self.tree[8][j])  # sum arities of all parent nodes at the prior depth

            # Set for every "free space" a function node (func)
            for j in range(1, len(self.tree[3])):  # increment through all nodes
                if int(self.tree[4][j]) == self.pop_node_depth - 1:  # ... find all parent nodes, one level above...
                    if self.tree[6][j] == 'Ifte':
                        prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2b')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                        prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                        prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, '2f')  # ... generate a Function node
                        prior_siblings = prior_siblings + 1
                    else:
                        for k in range(1, int(self.tree[8][j]) + 1):  # k = 1,2
                            self.pop_node_parent = int(self.tree[3][j])  # set the nodes parent
                            parent_func_dtype = function_dtypes_dict[self.tree[6][self.pop_node_parent]]  # find parents node
                            func_dtype = parent_func_dtype[:2][::-1]  # parent 'f2b' -> '2f' child needed. Aka, the first two characters reversed
                            prior_sibling_arity = self.branch_node_gen(parent_arity_sum, prior_sibling_arity, prior_siblings, func_dtype)  # ... generate a Function node
                            prior_siblings = prior_siblings + 1  # sum sibling nodes (current depth) who will spawn their own children (cousins? :)

        return

    def branch_node_gen(self, parent_arity_sum, prior_sibling_arity, prior_siblings, node_dtype):

        """
        (parent_arity_sum, prior_sibling_arity, prior_siblings, '2b')
        Generate a single Function node for the initial population.

        """

        if np.random.choice(['func', 'term']) == 'func':  # randomly selected as Function
            self.branch_function_select(node_dtype)  # retrieve a function, input-reverse the parent-function (f2b -> we need 2f input)
            self.fx_init_child_link(parent_arity_sum, prior_sibling_arity, prior_siblings)  # establish links to children
        else:
            self.branch_terminal_select(node_dtype) #  was here
            self.pop_node_c1 = ''
            self.pop_node_c2 = ''
            self.pop_node_c3 = ''

        self.branch_node_commit()  # commit new node to array
        prior_sibling_arity = prior_sibling_arity + self.pop_node_arity  # sum the arity of prior siblings

        return prior_sibling_arity

    def branch_function_select(self, func_dtype):

        """
        Returns a function with the same outcome

        """

        self.pop_node_type = 'func'
        new_function = self.dtype_get_func4any(func_dtype)
        self.pop_node_label = new_function[0]
        self.pop_node_arity = int(new_function[1])
        self.pop_node_modify = 1
        # print('SFEHs:', func_dtype, self.pop_node_label)

        return

        ### Terminal Nodes ###

        # Idea: values are recognized and adjusted (+0.3, e.g.)?
        # Idea: insert numbers as 0-arity functions? -> naah

    def tree_get_constant4type(self, term_type='', mode='float-1to1'):
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
            return self.tree_get_constant4type(term_type=term_type)

    def init_terminal_build(self):

        """
        Build the Terminal nodes for the tree.

        Arguments required: none
        """

        self.pop_node_depth = self.pop_tree_depth_base  # set the final node_depth (same as 'gp.pop_node_depth' + 1)

        for j in range(1, len(self.tree[3])):  # go through all nodes
            if int(self.tree[4][j]) == self.pop_node_depth - 1:  # this node is a parent
                for k in range(1, (int(self.tree[8][j]) + 1)):  # increment through each degree of arity for each parent node
                    self.pop_node_parent = int(self.tree[3][j])  # set the parent 'NODE_ID'  ...
                    self.init_terminal_gen(function_dtypes_dict[self.tree[6][j]])  # ... generate a Terminal node

        return

    def init_terminal_gen(self, terminal_dtype):

        """
        Generate a single Terminal node for the initial population.

        Called by: fx_init_terminal_build

        Arguments required: none
        """
        self.branch_terminal_select(terminal_dtype)
        # self.fx_init_terminal_select()  # retrieve a terminal
        self.pop_node_c1 = ''
        self.pop_node_c2 = ''
        self.pop_node_c3 = ''

        self.branch_node_commit()  # commit new node to array

        return

    def branch_terminal_select(self, node_dtype):

        """
        Define a single Terminal (variable extracted from the top row of the associated TRAINING data)

        Called by: fx_init_terminal_gen, fx_init_function_gen

        Arguments required: none
        """

        self.pop_node_type = 'term'
        self.pop_node_label = self.dtype_get_term4dtype(node_dtype)  # get a terminal
        self.pop_node_arity = 0

        return

    ### The Lovely Children ###

    def fx_init_child_link(self, parent_arity_sum, prior_sibling_arity, prior_siblings):

        """
        Link each parent node to its children in the intial population.

        Called by: fx_init_function_gen

        Arguments required: parent_arity_sum, prior_sibling_arity, prior_siblings
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
                    self.printpl('o', '\n\t\033[31m ERROR! In fx_init_child_link: pop_node_arity =', self.pop_node_arity, '\033[0;0m')
                    self.fx_karoo_pause()  # consider special instructions for this (pause) - 2019 06/08

        return

    def branch_node_commit(self):

        '''
        Commit the values of a new node (root, function, or terminal) to the array 'tree'.

        Called by: fx_init_root_build, fx_init_function_gen, fx_init_terminal_gen

        Arguments required: none
        '''

        self.tree = np.append(self.tree, [[self.pop_TREE_ID], [self.pop_tree_type], [self.pop_tree_depth_base], [self.pop_NODE_ID], [self.pop_node_depth], [self.pop_node_type], [self.pop_node_label],
                                          [self.pop_node_parent], [self.pop_node_arity], [self.pop_node_c1], [self.pop_node_c2], [self.pop_node_c3], [self.pop_fitness], [1], [self.pop_parsimony]], 1)

        self.pop_NODE_ID = self.pop_NODE_ID + 1

        return


    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Evaluate a Tree                |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def tree_algo_expr_sympify(self, tree):

        """
        Evaluate a Tree and generate its multivariate expression (both raw and Sympified).

        We need to extract the variables from the expression. However, these variables are no longer correlated
        to the original variables listed across the top of each column of data.csv. Therefore, we must re-assign
        the respective values for each subsequent row in the data .csv, for each Tree's unique expression.

        Called by: fx_karoo_pause, data_params_write, fx_eval_label, fitness_gym, fx_fitness_gene_pool, fx_display_tree

        Arguments required: tree
        """

        self.algo_raw = self.tree_expr_raw(tree, 1)  # pass the root 'node_id', then flatten the Tree to a string
        try:  # plagih: try block needed. simpify can not handle if then else.
            x = plagih_sympify(self.algo_raw)
            strx = str(x)

            if 'zoo' in strx or 'nan' in strx:
                self.monitor_failed_sympys_amount = self.monitor_failed_sympys_amount + 1

            if 'zoo' in strx:
                x = re.sub('zoo', '10', strx)  # TODO how to handle zoo?

            if 'nan' in strx:  # Happens when 0/0 occurs. This tree is worth nothing anyways
                self.printpl('w', 'We had a "nan"')
                self.algo_sym == sympy_dummy  # "nan" workaround

            self.algo_sym = x  # convert string to a functional expression (the coolest line in Karoo! :)
        except:
            self.printpl('e', 'In sympify. Caused by this raw algorithm: ' + str(self.algo_raw))
            self.algo_sym = 1
            self.printpl('w', 'We had a "nan" which lead to an Exception')
            # todo.
        return

    def tree_expr_raw(self, tree, node_id):

        """
        Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').

        This method is called once per Tree, but may be called at any time to prepare an expression for any full or
        partial (branch) Tree contained in 'population'. Pass the starting node for recursion via the local variable
        'node_id' where the local variable 'tree' is a copy of the Tree you desire to evaluate.

        Called by: eval_poly, fx_eval_label (recursively)

        Arguments required: tree, node_id
        """

        # if tree[6, node_id] == 'not': tree[6, node_id] = ', not' # temp until this can be fixed at data_load

        node_id = int(node_id)

        if tree[8, node_id] == '0':  # arity of 0 for the pattern '[term]'
            return '(' + tree[6, node_id] + ')'  # 'node_label' (function or terminal)
        else:
            if tree[8, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
                return '(' + self.tree_expr_raw(tree, tree[9, node_id]) + tree[6, node_id] + ')'

            elif tree[8, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
                # This if case is for 2-ary ops that can not be inline. like Min(a, b)
                if tree[6, node_id] not in inline_functions:  # in non_inline_functions:
                    return '(' + tree[6, node_id] + '(' + self.tree_expr_raw(tree, tree[9, node_id]) + ', ' + self.tree_expr_raw(tree, tree[10, node_id]) + '))'
                else:
                    return '(' + self.tree_expr_raw(tree, tree[9, node_id]) + tree[6, node_id] + self.tree_expr_raw(tree, tree[10, node_id]) + ')'  # Klammern, da sympify sonst abkacnen könnte

            # if then else
            elif tree[8, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
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

        if tree[8, node_id] == '0':  # arity of 0 for the pattern '[NODE_ID]'
            return tree[3, node_id]  # 'NODE_ID'

        else:
            if tree[8, node_id] == '1':  # arity of 1 for the pattern '[NODE_ID], [NODE_ID]'
                return tree[3, node_id] + ', '\
                       + self.tree_node_get_childlist(tree, tree[9, node_id])

            elif tree[8, node_id] == '2':  # arity of 2 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID]'
                return tree[3, node_id] + ', ' \
                       + self.tree_node_get_childlist(tree, tree[9, node_id]) + ', ' \
                       + self.tree_node_get_childlist(tree, tree[10, node_id])

            elif tree[8, node_id] == '3':  # arity of 3 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID], [NODE_ID]'
                return tree[3, node_id] + ', ' \
                       + self.tree_node_get_childlist(tree, tree[9, node_id]) + ', ' \
                       + self.tree_node_get_childlist(tree, tree[10, node_id]) + ', ' \
                       + self.tree_node_get_childlist(tree, tree[11, node_id])

    def eval_generation(self):

        """
        This method invokes the evaluation of an entire generation of Trees. It automatically evaluates population_b
        before invoking the copy of _b to _a.

        Called by: fx_karoo_gp

        Arguments required: none
        """

        self.printpl('g', 'Evaluate all Trees in Generation', self.gen_id)
        self.printpl('p', 'Want to adjust anything?')

        for tree_id in range(1, len(self.population_b)):  # renumber all Trees in given population
            self.printpl('vv', 'Evaluating Tree', tree_id)
            self.population_b[tree_id][0][1] = tree_id

        self.fitness_gym(self.population_b)
        self.data_tree_write(self.population_b, 'a')  # archive current population as foundation for next generation

        self.printpl('v', 'Copy gp.population_b to gp.population_a')

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Train and Test a Tree         |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def fitness_gym(self, population):

        """
        Part 1 evaluates each expression against the data, line for line. This is the most time consuming and
        computationally expensive part of genetic programming. When GPUs are available, the performance can increase
        by many orders of magnitude for datasets measured in millions of data.

        Part 2 evaluates every Tree in each generation to determine which have the best, overall fitness score. This
        could be the highest or lowest depending upon if the fitness function is maximising (higher is better) or
        minimising (lower is better). The total fitness score is then saved with each Tree in the external .csv file.

        Part 3 compares the fitness of each Tree to the prior best fit in order to track those that improve with each
        comparison. For matching functions, all the Trees will have the same fitness score, but they may present more
        than one solution. For minimisation and maximisation functions, the final Tree should present the best overall
        fitness for that generation. It is important to note that Part 3 does *not* in any way influence the Tournament
        Selection which is a stand-alone process.

        Called by: fx_karoo_gp, fx_eval_generations

        Arguments required: population
        """

        fitness_best = 0
        self.fittest_dict = {}

        for tree_id in range(1, len(population)):

            ### PART 1 - GENERATE MULTIVARIATE EXPRESSION FOR EACH TREE ###
            self.tree_algo_expr_sympify(population[tree_id])  # extract the expression

            ### PART 2 - EVALUATE FITNESS FOR EACH TREE AGAINST TRAINING DATA ###
            fitness = 0
            expr = str(self.algo_sym)  # get sympified expression and process it with TF
            result = self.expr_fitness_eval(expr, self.data_train)
            parsimony = self.tree_parsimony_distance(population[tree_id])
            fitness = result['fitness']  # extract fitness score

            self.fitness_store(population[tree_id], fitness, parsimony)  # store Fitness and parsimony with each Tree

            ### PART 3 - COMPARE FITNESS OF ALL TREES IN CURRENT GENERATION ###
            if self.kernel == 'c':  # display best fit Trees for the CLASSIFY kernel
                if fitness >= fitness_best:  # find the Tree with Maximum fitness score
                    fitness_best = fitness  # set best fitness score
                    self.fittest_dict.update({tree_id: self.algo_sym})  # add to dictionary if fitness >= prior

            elif self.kernel == 'r':  # display best fit Trees for the REGRESSION kernel
                if fitness_best == 0: fitness_best = fitness  # set the baseline first time through
                if fitness <= fitness_best:  # find the Tree with Minimum fitness score
                    fitness_best = fitness  # set best fitness score
                    self.fittest_dict.update({tree_id: self.algo_sym})  # add to dictionary if fitness <= prior

            elif self.kernel == 'm':  # display best fit Trees for the MATCH kernel
                if fitness == self.data_train_rows:  # find the Tree with a perfect match for all data rows
                    fitness_best = fitness  # set best fitness score
                    self.fittest_dict.update({tree_id: self.algo_sym})  # add to dictionary if all rows match

            else:
                raise self.printpl('e', 'No peoper kernel was selected', self.kernel)

        self.printpl('o', '\n\033[36m ', len(list(self.fittest_dict.keys())), 'trees\033[1m', np.sort(list(self.fittest_dict.keys())), '\033[0;0m\033[36moffer the highest fitness scores.\033[0;0m')
        if self.display == 'g': self.fx_karoo_pause_refer()

        return

    def sfeh_treedist_rel_ari(self, tree):
        """
        This distance penalizes non-original functions with its arity
        - ignore node[0] [description]
        - look within the subtree if the original function is on origin spot
        """

        # If the new tree is actually less complex than the original one, just return 1
        if len(tree[6]) < len(self.origin_tree[6]):
            return 0

        distance = 0

        # iterate over every node in the new tree
        for i, arity in enumerate(tree[8]):
            if i == 0:  # skip node 0. the description
                continue
            elif i < len(self.origin_tree[6]):  # Make sure we stay within the tree index. <= does not work
                if self.origin_tree[6][i] != tree[6][i]:  # is it different from the origin?
                    distance = distance + int(arity)  # add the nodes arity. double-punishes large trees
            else:
                distance = distance + int(arity)

        return distance

    def tree_parsimony_distance(self, tree, parsimony_distance='rel_ari_1'):
        """

        :param tree: The tree
        :param parsimony_distance: compute the chosen distance by the user. Default is best historic distance
        :return: The chosen distance
        """
        if parsimony_distance == 'total_count_nodes':
            return int(tree[3][-1:])  # returns the tree size
        elif parsimony_distance == 'total_tree_depth':
            return tree[4][1]     # returns the tree size
        elif parsimony_distance == 'total_karoo_original':  # do not use with long variable names
            return len(str(self.algo_raw))
        elif parsimony_distance == 'total_simplified':
            return count_ops(self.algo_sym)
        elif parsimony_distance == 'rel_ari_1':  # Does this work?
            return self.sfeh_treedist_rel_ari(tree)
        elif parsimony_distance == 'print':   # Please inser all of the measurements with example
            self.printpl('i', 'No distance chosen. Available parsimony measurements:')
            self.printpl('i', 'pcount_nodes' + '    : count_nodes. Amount of literals in the program.       ' + str(tree[3][-1:]))
            self.printpl('i', 'tree_depth' + '     : Only use the depth of the tree as measurement.        ' + str(tree[4][1]))
            self.printpl('i', 'karoo_original' + ' : Karoo`s OG parsimony. Do not use it with PLAGIH.      ' + str(len(str(self.algo_raw))))
            self.printpl('i', 'Choose wisely. More to come soon.')
            return
        else:
            self.printpl('i', 'Parsimony distance not specified! Use default.')
            self.tree_parsimony_distance(tree)

    def pop_tree_exists(self):
        return

    def tree_fitness(self, tree):
        """
        returns the fitness of a tree
        - first check in the hashtable
        - otherwise, compute with tensorflow
        """

        # If algo_sym is '1' - aka an error occured in sympy
        if self.algo_sym == sympy_dummy:
            return -1

        # 1. Try to get the fitness from hashtable
        expr_sym = str(self.algo_sym)
        expr_hash = hash(expr_sym)
        if expr_hash in self.hashtable_fitness:
            return self.hashtable_fitness[expr_hash]

        # 2. calculate the fitness and store it in the hash table
        result = self.expr_fitness_eval(expr_sym, self.data_train)
        fitness = result['fitness']  # extract fitness score
        self.hashtable_fitness[expr_hash] = fitness
        return fitness

    def expr_fitness_eval(self, expr, data, get_pred_labels=False):

        """
        Computes tree expression using TensorFlow (TF) returning results and fitness scores.

        This method orchestrates most of the TF routines by parsing input string 'expression' and converting it into a TF
        operation graph which is then processed in an isolated TF session to compute the results and corresponding fitness
        values.

            'self.tf_device' - controls which device will be used for computations (CPU or GPU).
            'self.tf_device_log' - controls device placement logging (debug only).

        Args:
            'expr' - a string expression to be computed on the data. Variable names should match 'self.terminals' names.
            'data' - an 'n by m' matrix of the data points containing n observations like 'self.terminals'.
            'get_pred_labels' - a boolean flag which controls whether the predicted labels should be extracted from the
            evolved results. This applies only to the CLASSIFY kernel and defaults to 'False'.

        Returns:
            A dict mapping keys to the following outputs:
                'result' - an array of the results of applying given expression to the data
                'pred_labels' - an array of the predicted labels extracted from the results; defined only for CLASSIFY kernel, else None
                'solution' - an array of the solution values extracted from the data (variable 's' in the dataset)
                'pairwise_fitness' - an array of the element-wise results of applying corresponding fitness kernel function
                'fitness' - aggregated scalar fitness score

        """

        # Initialize TensorFlow session
        tf.compat.v1.reset_default_graph()  # Reset TF internal state and cache (after previous processing). sfeh: updated as recommended by tensorflow
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
                    if '2f' in self.dtype_get_dtype4node(var, 'term'):
                        tensors[var] = tf.constant(data[:, i], dtype=tf.float32)  # converts data into vectors
                    else:  # '2b'
                        tensors[var] = tf.constant(data[:, i], dtype=tf.bool)

                for i in range(num_actions):
                    var = self.actions[i]
                    if '2f' in self.dtype_get_dtype4node(var, 'term'):
                        tensors[var] = tf.constant(data[:, num_terminals + i], dtype=tf.float32)  # converts data into vectors
                    else:  # '2b'
                        self.printpl('t', 'Currently no kernel available for boolean fitness')
                        tensors[var] = tf.constant(data[:, i], dtype=tf.bool)

                # 2- Transform string expression into TF operation graph
                tf_result = self.fitness_expr_parse(expr, tensors)
                pred_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

                # TODO currently does only support one label
                solution = tensors['action0']  # solution value is assumed to be stored in this terminal
                # 3- Add fitness computation into TF graph
                if self.kernel == 'c':  # CLASSIFY kernel

                    """
                    Creates element-wise fitness computation TensorFlow (TF) sub-graph for CLASSIFY kernel.
                    - tree-label vs. true label
                    This multiclass classifer compares each row of a given Tree to the known solution, comparing predicted labels 
                    generated by Karoo GP against the true class labels. This method is able to work with any number of class 
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
                        pred_labels = tf.map_fn(self.fitness_labels_map, tf_result, dtype=(tf.int32, tf.string), swap_memory=True)

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
                print('Result:', tf_result, 'Fitness:', fitness)

        return {'result': tf_result, 'pred_labels': pred_labels, 'solution': solution, 'fitness': float(fitness),
                'pairwise_fitness': pairwise_fitness}

    def fitness_expr_parse(self, expr, tensors):

        """
        Extract expression tree from the string algo_sym and transform into TensorFlow (TF) graph.

        """
        # print('Current expr:', expr)  # importantprint
        tree = ast.parse(expr, mode='eval').body

        return self.fitness_node_parse(tree, tensors)

    def fx_fitness_chain_bool(self, values, operation, tensors):

        """
        Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.

        Called by: fitness_node_parse

        Arguments required: values, operation, tensors
        """

        x = tf.dtypes.cast(self.fitness_node_parse(values[0], tensors), tf.bool)
        if len(values) > 1:
            return operation(x, self.fx_fitness_chain_bool(values[1:], operation, tensors))
        else:
            return x

    def fx_fitness_chain_compare(self, comparators, ops, tensors):

        """
        Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

        Called by: fitness_node_parse

        Arguments required: comparators, ops, tensors
        """

        x = self.fitness_node_parse(comparators[0], tensors)
        y = self.fitness_node_parse(comparators[1], tensors)

        if len(comparators) > 2:
            return tf.logical_and(operators[type(ops[0])](x, y), self.fx_fitness_chain_compare(comparators[1:], ops[1:], tensors))
        else:
            return operators[type(ops[0])](x, y)
        # sfeh idea: note: we have to convert all values to the action space if not discrete

    def fitness_node_parse(self, node, tensors):

        """
        Recursively transforms parsed expression tree into TensorFlow (TF) graph.

        Called by: fx_fitness_expr_parse, fx_fitness_chain_bool, fx_fitness_chain_compare

        Arguments required: node, tensors
        """

        if isinstance(node, ast.Name):  # <tensor_name>
            return tensors[node.id]

        elif isinstance(node, ast.Num):  # <number>
            shape = tensors[list(tensors.keys())[0]].get_shape()
            return tf.constant(node.n, shape=shape, dtype=tf.float32)

        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>, e.g., x + y
            return operators[type(node.op)](self.fitness_node_parse(node.left, tensors), self.fitness_node_parse(node.right, tensors))

        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            return operators[type(node.op)](self.fitness_node_parse(node.operand, tensors))

        elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or ftob(a)
            if node.func.id == 'Ifte':
                return operators[node.func.id](
                        tf.dtypes.cast(self.fitness_node_parse(node.args[0], tensors), tf.bool),
                        self.fitness_node_parse(node.args[1], tensors),
                        self.fitness_node_parse(node.args[2], tensors))

            if node.func.id in non_inline_multielem_functions:  # Min, Max Goddamn. yeah, min and max need the same type, apparently. TODO? Does this work now?
                self.printpl('e', [self.fitness_node_parse(arg, tensors) for arg in node.args])
                return operators[node.func.id]([self.fitness_node_parse(arg, tensors) for arg in node.args])  # the star '*' makes the difference

            if node.func.id == 'ftob':
                self.printpl('i', 'float was converted to bool in tensorflow')
                return tf.dtypes.cast(*[self.fitness_node_parse(arg, tensors) for arg in node.args], dtype=tf.bool)
            elif node.func.id == 'btof':
                return tf.dtypes.cast(*[self.fitness_node_parse(arg, tensors) for arg in node.args], dtype=tf.float32)

            return operators[node.func.id](*[self.fitness_node_parse(arg, tensors) for arg in node.args])

        elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
            return self.fx_fitness_chain_bool(node.values, operators[type(node.op)], tensors)

        elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
            return self.fx_fitness_chain_compare([node.left] + node.comparators, node.ops, tensors)

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

    def fitness_labels_map(self, result):

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

        Called by: fx_fitness_eval

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

    def fitness_store(self, tree, fitness, parsimony):

        """
        Records the fitness and length of the raw algorithm (multivariate expression) to the Numpy array. Parsimony can
        be used to apply pressure to the evolutionary process to select from a set of trees with the same fitness function
        the one(s) with the simplest (shortest) multivariate expression.

        Called by: fx_fitness_gym

        Arguments required: tree, fitness
        """

        fitness = float(fitness)
        fitness = round(fitness, self.precision)

        tree[12][1] = fitness  # store the fitness with each tree
        tree[14][1] = parsimony  # store the length of the raw algo for parsimony
        # if len(tree[3]) > 4: # if the Tree array is wide enough -- SEE SCRATCHPAD

        return

    def fitness_tournament(self, tourn_size):

        """
        Multiple contenders ('tourn_size') are randomly selected and then compared for their respective fitness, as
        determined in fx_fitness_gym(). The tournament is engaged to select a single Tree for each invocation of the
        genetic operators: reproduction, mutation (point, branch), and crossover (sexual reproduction).

        The original Tournament Selection drew directly from the foundation generation (gp.generation_a). However,
        with the introduction of a minimum number of nodes as defined by the user ('gp.tree_depth_min'),
        'gp.gene_pool' limits the Trees to those which meet all criteria.

        Stronger boundary parameters (a reduced gap between the min and max number of nodes) may invoke more compact
        solutions, but also runs the risk of elitism, even total population die-off where a healthy population once existed.
        """

        tourn_test = 0
        # short_test = 0 # an incomplete parsimony test (seeking shortest solution)

        for n in range(tourn_size):
            # tree_id = np.random.randint(1, self.tree_pop_max + 1) # former method of selection from the unfiltered population
            rnd = np.random.randint(len(self.gene_pool))  # select one Tree at random from the gene pool
            tree_id = int(self.gene_pool[rnd])

            fitness = float(self.population_a[tree_id][12][1])  # extract the fitness from the array
            fitness = round(fitness, self.precision)  # force 'result' and 'solution' to the same number of floating points

            if self.fitness_type == 'max':  # if the fitness function is Maximising

                # first time through, 'tourn_test' will be initialised below

                if fitness > tourn_test:  # if the current Tree's 'fitness' is greater than the priors'
                    if self.display == 'i':
                        print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # set 'TREE_ID' for the new leader
                    tourn_test = fitness  # set 'fitness' of the new leader
                # short_test = int(self.population_a[tree_id][14][1]) # set len(algo_raw) of new leader

                elif fitness == tourn_test:  # if the current Tree's 'fitness' is equal to the priors'
                    if self.display == 'i':
                        print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # in case there is no variance in this tournament
                # tourn_test remains unchanged

                # TODO NEED TO add option for parsimony
                # if int(self.population_a[tree_id][14][1]) < short_test:
                # short_test = int(self.population_a[tree_id][14][1]) # set len(algo_raw) of new leader
                # print ('\t\033[36m with improved parsimony score of:\033[1m', short_test, '\033[0;0m')

                elif fitness < tourn_test:  # if the current Tree's 'fitness' is less than the priors'
                    if self.display == 'i':
                        print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test,'and is ignored\033[0;0m')
                # tourn_lead remains unchanged
                # tourn_test remains unchanged

                else:
                    print('\n\t\033[31m ERROR! In fx_fitness_tournament: fitness =', fitness, 'and tourn_test =',tourn_test,'\033[0;0m')
                    self.fx_karoo_pause()  # consider special instructions for this (pause) - 2019 06/08

            elif self.fitness_type == 'min':  # if the fitness function is Minimising

                if tourn_test == 0:  # first time through, 'tourn_test' is given a baseline value
                    tourn_test = fitness

                if fitness < tourn_test:  # if the current Tree's 'fitness' is less than the priors'
                    if self.display == 'i':
                        print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # set 'TREE_ID' for the new leader
                    tourn_test = fitness  # set 'fitness' of the new leader

                elif fitness == tourn_test:  # if the current Tree's 'fitness' is equal to the priors'
                    if self.display == 'i':
                        print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id  # in case there is no variance in this tournament
                # tourn_test remains unchanged

                elif fitness > tourn_test:  # if the current Tree's 'fitness' is greater than the priors'
                    if self.display == 'i':
                        print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and is ignored\033[0;0m')
                # tourn_lead remains unchanged
                # tourn_test remains unchanged

                else:
                    self.printpl('e', 'fitness', self.population_a[tree_id])
                    raise print('\n\033[31m ERROR! In fx_fitness_tournament: fitness =', fitness, 'and tourn_test =', tourn_test, '\033[0;0m')

        tourn_winner = np.copy(self.population_a[tourn_lead])  # copy full Tree so as to not inadvertantly modify the original tree

        if self.display == 'i': print('\n\t\033[36mThe winner of the tournament is Tree:\033[1m', tourn_winner[0][1], '\033[0;0m')

        return tourn_winner

    def fitness_gene_pool(self):

        """
        Create the gene pool
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

        self.printpl('g', 'Gene Pool for Generation:', self.gen_id, '...')
        self.gene_pool = []

        for tree_id in range(1, len(self.population_a)):  # Every tree
            self.tree_algo_expr_sympify(self.population_a[tree_id])  # extract the expression
            if self.algo_sym != 1 and self.parsimony_min_max[0] > int(self.population_a[tree_id][14][1]):  # dummy exit and parsimony condition
                self.gene_pool.append(self.population_a[tree_id][0][1])

        if len(self.gene_pool) > 0:
            self.printpl('p', 'The total population of the gene pool is', len(self.gene_pool))
        else:  # the evolutionary constraints were too tight, killing off the entire population
            self.printpl('p', 'There are no Trees in the gene pool. You should archive your population and (q)uit.')
            self.fx_karoo_pause_refer()  # 2019 06/07

        return

    def fitness_olymp(self):
        """
        The olymp is where the godlike contestants reside.
        In each generation, the olymp searches for new god contestants
        """
        self.printpl('t', 'TODO Olymp for candidates')
        return

    def fx_fitness_test(self, result):

        if self.kernel == 'c':
            """
            Print the Precision-Recall and Confusion Matrix for a CLASSIFICATION run against the test data.

            From scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
                Precision (P) = true_pos / true_pos + false_pos
                Recall (R) = true_pos / true_pos + false_neg
                harmonic mean of Precision and Recall (F1) = 2(P x R) / (P + R)

            From scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
                y_pred = result, the predicted labels generated by Karoo GP
                y_true = solution, the true labels associated with the data

            Called by: fx_karoo_pause

            Arguments required: result
            """
            for i in range(len(result['result'])):
                print(
                    '\t\033[36m Data row {} predicts class:\033[1m {} ({} True)\033[0;0m\033[36m as {:.2f}{}\033[0;0m'.format(
                        i, int(result['pred_labels'][0][i]), int(result['solution'][i]), result['result'][i],
                        result['pred_labels'][1][i]))

            print('\n Fitness score: {}'.format(result['fitness']))
            print('\n Precision-Recall report:\n', skm.classification_report(result['solution'], result['pred_labels'][0]))
            print(' Confusion matrix:\n', skm.confusion_matrix(result['solution'], result['pred_labels'][0]))

        elif self.kernel == 'r':
            """
            Print the Fitness score and Mean Squared Error for a REGRESSION run against the test data.

            Called by: fx_karoo_pause

            Arguments required: result

            """

            for i in range(len(result['result'])):
                print('\t\033[36m Data row {} predicts value:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.
                      format(i, result['result'][i], result['solution'][i]))

            MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
            print('\n\t Origin fitness score: {}'.format(self.origin_fitness))
            print('\n\t Regression fitness score: {}'.format(fitness))
            print('\t Mean Squared Error: {}'.format(MSE))

            return

        elif self.kernel == 'm':

            """
            Print the accuracy for a MATCH kernel run against the test data.

            Arguments required: result
            """

            for i in range(len(result['result'])):
                self.printpl('o', '\t\033[36m Data row {} predicts match:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.format(i, result['result'][i], result['solution'][i]))

            self.printpl('o', 'Matching fitness score: {}'.format(result['fitness']))

            return
        else:
            self.printpl('e', 'This fitness test is not available:')

        return
    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Genetic Operators appliable               |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_reproduce(self):

        """
        Through tournament selection, a single Tree from the prior generation is copied without mutation to the next
        generation. This is analogous to a member of the prior generation directly entering the gene pool of the
        subsequent (younger) generation.

        """
        self.printpl('g', 'Reproduce One...')

        for n in range(self.evolve_repro):  # quantity of Trees to be copied without mutation
            tourn_winner = self.fitness_tournament(self.tourn_size)  # perform tournament selection for each reproduction
            tourn_winner = self.tree_store_meta_lastgen(tourn_winner, modification='r')  # wipe fitness data
            tourn_winner[1][1] = 'r'
            self.population_b.append(tourn_winner)  # append array to next generation population of Trees

        return

    def pop_mutate_point(self):

        """
        One point (terminal or function) gets mutated.
        Currently only mutating with functions/terminals of the exactly same type.
        """
        self.printpl('g', 'Point Mutation...')

        for n in range(self.evolve_point):  # quantity of Trees to be generated through mutation
            tourn_winner = self.fitness_tournament(self.tourn_size)  # get a tournament winner
            tourn_winner, node = self.pop_mutate_point_evolve(tourn_winner)  # point mutation; return single point for record keeping
            tourn_winner = self.tree_store_meta_lastgen(tourn_winner, modification='p')  # wipe fitness data
            self.population_b.append(tourn_winner)  # append array to next generation population of Trees
        return

    def pop_mutate_branch(self):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """
        self.printpl('g', 'Branch Mutation...')

        for n in range(self.evolve_branch):  # quantity of Trees to be generated through mutation
            tourn_winner = self.fitness_tournament(self.tourn_size)  # perform tournament selection for each mutation
            branch_nodes_list = self.tree_branch_get(tourn_winner)  # select point of mutation and all nodes beneath [6, 9, 10]
            tourn_winner = self.pop_mutate_branch_evolve(tourn_winner, branch_nodes_list)
            tourn_winner = self.tree_store_meta_lastgen(tourn_winner, modification='b')  # wipe fitness data
            tourn_winner = self.tree_store_set_modify_nodes(tourn_winner)
            self.population_b.append(tourn_winner)  # append array to next generation population of Trees

        return

    def pop_crossover(self, mode='replace_same_types'):

        """
        TODO now, partners do not exchange their branches, just parent a takes a branch of parent_b
        - select parent a and b
        - select swappable branche for parent_a from parent_b
            - select a node in a (and crossover here, no matter what)
        - delete parent_a branch and insert parent_b branch (which tactic?)

        """
        self.printpl('g', 'Crossover...')

        for n in range(self.evolve_cross):  # quantity of Trees to be generated through Crossover, (now not accounting for 2 children each, changed)

            # 1. Select two parents and their branches
            parent_a = self.fitness_tournament(self.tourn_size)  # perform tournament selection for 'parent_a'
            parent_b = self.fitness_tournament(self.tourn_size)  # perform tournament selection for 'parent_b'

            # 2. Get the branches for parent a that can be exchanged
            branch_a, branch_b, convert_a = self.pop_crossover_get_swap_branches(parent_a, parent_b)

            if convert_a:
                self.printpl('i', 'Forced conversion is needed between two trees.')
                offspring = self.pop_crossover_insert(parent_a, branch_a, parent_b, branch_b, converter=convert_a)  # perform Crossover
            else:
                offspring = self.pop_crossover_insert(parent_a, branch_a, parent_b, branch_b)  # perform Crossover

            offspring = self.tree_store_meta_lastgen(offspring, modification='c')  # wipe fitness data
            offspring = self.tree_store_set_modify_nodes(offspring)
            self.population_b.append(offspring)  # append the 1st child to next generation of Trees

        return

    def pop_mutate_termfilter(self, constant, term_type='', filter='gaussian_filter'):
        """
        When this happens, constants get a a small variance
        """

        if term_type == 'float':
            if filter == 'gaussian_filter':
                constant = np.random.normal(constant, 0.1)
            else:
                print('Warning: Filter  not specified. Please specify a filter.')
                constant = np.random.normal(constant, 0.1)

        if term_type == 'int':
            constant = int(np.random.normal(constant, 2))

        if term_type == 'bool':
            constant = not constant
            # random by 50:50?

        return constant

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Evolve a Population            |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_crossover_get_partner_node_id(self, function_label, partner_tree, partner_branch_id, mode='same_type'):
        """
        -> Crossover: Returns a node_id in the partner tree, that can be swapped
        """

        node_dtype = self.dtype_get_dtype4node(function_label, self.evolve_get_nodetype4label(function_label))
        node_options = []
        # TODO check if the tree is large enough?
        if mode == 'same_type':  # only return a node with the same function type
            for i, label in enumerate(partner_tree[6][1:]):
                if self.dtype_get_dtype4label(label) == node_dtype:
                    node_options.append(i+1)  # +1, we skipped the first element

            if node_options:  # Found at least one!
                np.random.shuffle(node_options)  # otherwise, the first closest element is always taken (-> smallest)
                return min(node_options, key=lambda x: abs(x-partner_branch_id))  # return closest node
            else:
                return 0  # No matching node found :(
        elif mode == 'random':
            self.printpl('TODO', 'mode: Do the same as in the upper function, but choose randomly?')
        else:
            raise self.printpl('e', 'Mode not found', mode)

    def pop_mutate_point_evolve(self, tree, mode='random'):

        """
        Mutate a single mutatable point in any Tree.
        """

        # 1. choose a node
        node = self.tree_get_mutatable_node_id(tree, mode='mutate_point')  # randomly select a point in the Tree (including root)
        node_dtype = self.dtype_get_dtype4label(tree[6][node])  # '>' -> 'f2b'

        # 2. perform point mutation on that specific node
        if tree[5][node] == 'func':
            func_arity = int(tree[8][node])
            tree[6][node] = self.dtype_get_func4func(node_dtype, arity=func_arity)  # Function is same type, same arity
            # Take care of the modify specs
        elif tree[5][node] == 'term':
            tree[6][node] = self.dtype_get_term4dtype(node_dtype)  # 3 -> '2f' -> 5
        else:
            raise self.printpl('e', 'Operator type is not specified for PLAGIH ("term", "func",...)', tree[5][node])

        return tree, node  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping

    def pop_branch_depth_get(self, chosen_tree, branch_top, mode='random'):  # sfeh other default
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
            raise self.printpl('e', 'sfeh_get_new_tree_size does not accept this mode: ' + str(mode))
        return branch_depth

    def evolve_get_nodetype4label(self, node_label):
        """
        return terminal or function according to the label
        """

        if node_label in function_dtypes_dict:
            return 'func'
        else:
            return 'term'

    def tree_get_mutatable_node_id(self, tree, mode='', same_dtype=''):
        """
        Returns a mutatable node for point-mutation
        -> no_root handles
        """
        # TODO only works for 2-array functions

        node_ids = []

        # 1. Build up a list with nodes
        if same_dtype:
            for i, label in enumerate(tree[6]):
                if tree[13][i] == '1':  # also skips node 0
                    # TODO make this faster
                    node_dtype = self.dtype_get_dtype4label(tree[6][i])
                    if self.dtype_outcome_equi_test(node_dtype, same_dtype):
                        node_ids.append(int(tree[3][i]))
        else:
            for i, x in enumerate(tree[5]):
                if tree[13][i] == '1':
                    node_ids.append(int(tree[3][i]))

        # 2. Kick out root if it is there?
        if 'no_root' in mode:  # delete root node
            node_ids = [x for x in node_ids if x != 1]

        # 3: return the node. Not safe, could be try-except block.
        # eg: all nodes are not modifiable
        # eg. all nodes are not of correct type
        node_id = np.random.choice(node_ids)
        return node_id

    def tree_branch_get(self, tree, node=0):

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
            branch_top = self.tree_get_mutatable_node_id(tree, mode='mutate_branch_no_root')  # "2" returns mutable node (except root node)

        # 2. Also return all child nodes
        branch_eval = self.tree_node_get_childlist(tree, branch_top)  # generate tuple of 'branch_top' and subsequent nodes
        branch_symp = sympify(branch_eval)  # convert string into something useful
        branch = np.append(branch, branch_symp)  # append list to array
        branch = np.sort(branch)  # sort nodes in branch for Crossover.

        return branch

    def evolve_branch_insert(self, winner_tree, branch_nodes):

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
        winner_tree[6][branch_top] = self.tree[6][1]  # copy node_label from new tree
        winner_tree[8][branch_top] = self.tree[8][1]  # copy node_arity from new tree
        winner_tree = np.delete(winner_tree, branch_nodes[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')

        c_buffer = self.fx_evolve_c_buffer(winner_tree, branch_top)  # generate c_buffer for point of mutation ('branch_top')
        winner_tree = self.fx_evolve_child_insert(winner_tree, branch_top, c_buffer)  # insert a single new node ('branch_top')
        winner_tree = self.tree_node_renum(winner_tree)  # renumber all 'NODE_ID's

        ### PART 2 - insert branch_body from 'gp.tree' into 'tree' ###
        node_count = 2  # set node count for 'gp.tree' to 2 as the new root has already replaced 'branch_top' (above)

        while node_count < len(self.tree[3]):  # increment through all nodes in the new Tree ('gp.tree'), starting with node 2

            for j in range(1, len(winner_tree[3])):  # increment through all nodes in tourn_winner ('tree')

                if winner_tree[5][j] == '':
                    winner_tree[5][j] = self.tree[5][node_count]  # copy 'node_type' from branch to tree
                    winner_tree[6][j] = self.tree[6][node_count]  # copy 'node_label' from branch to tree
                    winner_tree[8][j] = self.tree[8][node_count]  # copy 'node_arity' from branch to tree

                    if winner_tree[5][j] == 'term':
                        winner_tree = self.fx_evolve_child_link_fix(winner_tree)  # fix all child links
                        winner_tree = self.tree_node_renum(winner_tree)  # renumber all 'NODE_ID's

                    if winner_tree[5][j] == 'func':
                        c_buffer = self.fx_evolve_c_buffer(winner_tree, j)  # generate 'c_buffer' for point of mutation ('branch_top')
                        winner_tree = self.fx_evolve_child_insert(winner_tree, j, c_buffer)  # insert new nodes
                        winner_tree = self.fx_evolve_child_link_fix(winner_tree)  # fix all child links
                        winner_tree = self.tree_node_renum(winner_tree)  # renumber all 'NODE_ID's

                    node_count = node_count + 1  # exit loop when 'node_count' reaches the number of columns in the array 'gp.tree'

        return winner_tree

    def tree_branch_copy(self, tree, branch):

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
            node_label = tree[6][node]
            node_parent = ''  # updated by fx_evolve_parent_link_fix(), below
            node_arity = tree[8][node]
            node_c1 = ''  # updated by fx_evolve_child_link_fix(), below
            node_c2 = ''
            node_c3 = ''
            fitness = ''
            node_modify = '1'  # sfeh Test this
            parsimony = ''

            new_tree = np.append(new_tree,
                                 [[TREE_ID], [tree_type], [tree_depth_base], [NODE_ID], [node_depth], [node_type],
                                  [node_label], [node_parent], [node_arity], [node_c1], [node_c2], [node_c3],
                                  [fitness], [node_modify], [parsimony]], 1)

        new_tree = self.tree_node_renum(new_tree)
        new_tree = self.fx_evolve_child_link_fix(new_tree)
        new_tree = self.fx_evolve_parent_link_fix(new_tree)
        new_tree = self.data_tree_clean(new_tree)

        return new_tree

    def fx_evolve_c_buffer(self, tree, node):

        """
        This method serves the very important function of determining the links from parent to child for any given
        node. The single, simple formula [parent_arity_sum + prior_sibling_arity - prior_siblings] perfectly determines
        the correct position of the child node, already in place or to be inserted, no matter the depth nor complexity
        of the tree.

        This method is currently called from the evolution methods, but will soon (I hope) be called from the first
        generation Tree generation methods (above) such that the same method may be used repeatedly.

        Called by: fx_evolve_child_link_fix, fx_evolve_banch_top_copy, fx_evolve_branch_body_copy

        Arguments required: tree, node
        """

        parent_arity_sum = 0
        prior_sibling_arity = 0
        prior_siblings = 0

        for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'

            if int(tree[4][n]) == int(tree[4][node]) - 1:  # find parent nodes at the prior depth
                if tree[8][n] != '': parent_arity_sum = parent_arity_sum + int(
                    tree[8][n])  # sum arities of all parent nodes at the prior depth

            if int(tree[4][n]) == int(tree[4][node]) and int(tree[3][n]) < int(
                    tree[3][node]):  # find prior siblings at the current depth
                if tree[8][n] != '':
                    prior_sibling_arity = prior_sibling_arity + int(tree[8][n])  # sum prior sibling arity
                prior_siblings = prior_siblings + 1  # sum quantity of prior siblings

        c_buffer = node + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

        return c_buffer

    def fx_evolve_child_link(self, tree, node, c_buffer):

        """
        Link each parent node to its children.

        Called by: fx_evolve_child_link_fix

        Arguments required: tree, node, c_buffer
        """

        if int(tree[3][node]) == 1:
            # SFEH Root can only be ignored, if root was not changed
            c_buffer = c_buffer + 1  # if root (node 1) is passed through this method

        if tree[8][node] != '':

            if int(tree[8][node]) == 0:  # if arity = 0
                tree[9][node] = ''
                tree[10][node] = ''
                tree[11][node] = ''

            elif int(tree[8][node]) == 1:  # if arity = 1
                tree[9][node] = c_buffer
                tree[10][node] = ''
                tree[11][node] = ''

            elif int(tree[8][node]) == 2:  # if arity = 2
                tree[9][node] = c_buffer
                tree[10][node] = c_buffer + 1
                tree[11][node] = ''

            elif int(tree[8][node]) == 3:  # if arity = 3
                tree[9][node] = c_buffer
                tree[10][node] = c_buffer + 1
                tree[11][node] = c_buffer + 2

            else:
                self.printpl('o', '\n\t\033[31m ERROR! In fx_evolve_child_link: node', node, 'has arity', tree[8][node])
                self.fx_karoo_pause()  # consider special instructions for this (pause) - 2019 06/08

        return tree

    def fx_evolve_child_link_fix(self, tree):

        """
        In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

        This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.

        """

        for node in range(1, len(tree[3])):
            c_buffer = self.fx_evolve_c_buffer(tree, node)  # generate c_buffer for each node
            tree = self.fx_evolve_child_link(tree, node, c_buffer)  # update child links for each node

        return tree

    def fx_evolve_child_insert(self, tree, node, c_buffer):

        """
        Insert child node into the copy of a parent Tree.

        Called by: fx_evolve_branch_insert

        Arguments required: tree, node, c_buffer
        """

        if int(tree[8][node]) == 0:  # if arity = 0
            self.printpl('e', 'In fx_evolve_child_insert: node', node, 'has arity 0')
            self.fx_karoo_pause()  # consider special instructions for this (pause) - 2019 06/08

        elif int(tree[8][node]) == 1:  # if arity = 1
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[4][c_buffer] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

        elif int(tree[8][node]) == 2:  # if arity = 2
            tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
            tree[3][c_buffer] = c_buffer  # node ID
            tree[4][c_buffer] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer] = int(tree[3][node])  # parent ID

            tree = np.insert(tree, c_buffer + 1, '', axis=1)  # insert node for 'node_c2'
            tree[3][c_buffer + 1] = c_buffer + 1  # node ID
            tree[4][c_buffer + 1] = int(tree[4][node]) + 1  # node_depth
            tree[7][c_buffer + 1] = int(tree[3][node])  # parent ID

        elif int(tree[8][node]) == 3:  # if arity = 3
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
            self.printpl('e', 'In fx_evolve_child_insert: node', node, 'arity > 3')
            self.fx_karoo_pause()  # consider special instructions for this (pause) - 2019 06/08

        return tree

    def fx_evolve_parent_link_fix(self, tree):

        """
        In a given Tree, fix 'parent_id' for all nodes.

        This is automatically handled in all mutations except with Crossover due to the need to copy branches 'a' and
        'b' to their own trees before inserting them into copies of	the parents.

        Technically speaking, the 'node_parent' value is not used by any methods. The parent ID can be completely out
        of whack and the expression will work perfectly. This is maintained for the sole purpose of granting the user
        a friendly, makes-sense interface which can be read in both directions.

        Called by: fx_evolve_branch_copy

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

    def tree_node_arity_fix(self, tree):

        """
        In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.

        This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and
        Crossover.

        """

        for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'

            if tree[5][n] == 'term':  # check for discrepency
                tree[8][n] = '0'  # set arity to 0
                tree[9][n] = ''  # wipe 'node_c1'
                tree[10][n] = ''  # wipe 'node_c2'
                tree[11][n] = ''  # wipe 'node_c3'
                tree[13][n] = '1'

        return tree

    def tree_node_renum(self, tree):

        """
        Renumber all 'NODE_ID' in a given tree.

        This is required after a new generation is evolved as the NODE_ID numbers are carried forward from the previous
        generation but are no longer in order.

        """

        for n in range(1, len(tree[3])):
            tree[3][n] = n  # renumber all Trees in given population

        return tree

    def tree_store_parsimony(self, tree, parsimony):
        """
        Store the parsimony within the tree np-array
        """
        tree[14][1] = parsimony

    def tree_store_fitness(self, tree, fitness):

        """
        Store the parsimony within the tree np-array
        """

        fitness = float(fitness)
        fitness = round(fitness, self.precision)

        tree[12][1] = fitness  # store the fitness with each tree

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
            tree[14][i] = tree[14][i-1]  # The last parsimony (TODO) # TREE_ID,1,a,b,c -> TREE_ID,1,a,a,b

        # What needs to be assigned later
        tree[0][1] = ''  # -> TREE_ID,,
        tree[1][1] = modification  # wipe last modification data
        tree[12][1] = ''  # wipe fitness data
        tree[14][1] = ''  # wipe parsimony data

        return tree

    def evolve_tree_prune(self, tree, depth):

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
                node_dtype = self.dtype_get_dtype4label(tree[6][n])
                tree[6][n] = self.dtype_get_term4dtype(node_dtype)  # replace label

            elif int(tree[4][n]) > depth:  # record nodes deeper than the maximum allowed Tree depth
                nodes.append(n)

            else:
                pass  # as int(tree[4][n]) < depth and will remain untouched

        tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
        tree = self.tree_node_arity_fix(tree)  # fix all node arities

        return tree

    def fx_evolve_pop_copy(self, pop_a, title):

        """
        Copy one population to another.

        Simply copying a list of arrays generates a pointer to the original list. Therefore we must append each array
        to a new, empty array and then build a list of those new arrays.

        Called by: fx_karoo_gp

        Arguments required: pop_a, title
        """

        pop_b = [title]  # an empty list stores a copy of the prior generation

        for tree in range(1, len(pop_a)):  # increment through each Tree in the current population

            tree_copy = np.copy(pop_a[tree])  # copy each array in the current population
            pop_b.append(tree_copy)  # add each copied Tree to the new population list

        return pop_b

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to receive correct dtypes (f2b,.) |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def dtype_get_func4func(self, function_type, arity=0):
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
                raise self.printpl('e', 'Function was not found in function_types_dict', function_type)

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
            raise self.printpl('e', 'Function was not found in function_types_dict', function_type)

    def dtype_get_func4any(self, function_dtype):
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
            raise self.printpl('e', 'Warning: Function was not found in function_types_dict', function_dtype)

    def dtype_get_dtype4node(self, node_label, node_type):
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
            raise self.printpl('e', 'This node_type is not known', node_type)

    def dtype_get_dtype4label(self, node_label):
        """
        returns dtype for a label
        """

        node_type = self.evolve_get_nodetype4label(node_label)
        node_dtype = self.dtype_get_dtype4node(node_label, node_type)
        return node_dtype

    def dtype_outcome_equi_test(self, a_dtype, b_dtype):
        return a_dtype in b_dtype or b_dtype in a_dtype

    def dtype_get_term4dtype(self, node_dtype):
        """
        Returns a terminal that fits to the given type.

        function: f2b -> 2b needed
        terminal:  2f -> 2f needed
        --> check if it is function, aka _2f
        --> check if it is terminal, aka f2

        Modes:
        var_and_const: return randomly (50:50) a variable or a constant
        terminal_only: return                  a variable
        Todo: Introduce constants-mode, where the user can give constant types (similar to functions)?
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
            return self.tree_get_constant4type(term_type=the_type)  # otherwise: constant (There are always constants :P)

        except ValueError:
            self.printpl('w', 'Should not happen. Did not find a terminal. Made up a ' + the_type + ' constant.')
            return self.tree_get_constant4type(term_type=the_type)

    def dtype_convert_dtypes_get_dummylabel(self, a_dtype, b_dtype):
        """
        convert a-to-b dummy
        """
        if '2b' in a_dtype and '2f' in b_dtype:
            return 'btof'
        if '2f' in a_dtype and '2b' in b_dtype:
            return 'ftob'
        else:
            self.printpl('e', 'One of those two cases should happen', a_dtype, b_dtype)
            raise
        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to display output information     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def printpl(self, verbosity, *args):  # plagih naming
        """
        Gets a verbosity (e.g. 'i')

        PLAGIH-Display-modes:
            Errors (e): always on, does print Errors
            Info (i): informations. (need further specification)
            NextGen (n): Prints infos about the current mutation process

        Display modes: (Just a reminder from original Karoo)
            Generational (g): pauses after each generation is complete
            Interactive (i): pauses with the completion of each section (e.g. tournament, gene pool, genetic operators)
            DeBug (db): displays the internal workings of the genetic operators
            Minimal (m): displays only the multivariate expression of each tree
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

        if verbosity in self.display:
            message_style = '\033[39m'  # default color
            if verbosity == 'i':
                message_style = '\033[36mInfo: '  # cyan
            elif verbosity == 'e':
                message_style = '\033[31mERROR: '  # red
            elif verbosity == 'g':
                message_style = '\033[32mGeneration: '  # green
            elif verbosity == 'v':  # verbose
                message_style = '\033[37mVerbose: '  # white
            elif verbosity == 'p':  # pause
                message_style = '\033[33mPause(TODO): '  # Yellow
            elif verbosity == 'f':  # function
                message_style = '\033[35mFunc: '  # Magenta
            elif verbosity == 'w':  # warning
                message_style = '\033[93mWarning: '  # Warning-yellow
            elif verbosity == 'o':  # original Karoo message. Has its own format.
                message_style = ''  # Do nothing
            else:
                # Just show it
                self.printpl('e', 'Display-mode', verbosity, 'not known.')

            print(message_style + ' '.join(map(str, args))+'\033[39m')
        return

    def plotpl(self):  # plagih naming
        plt.plot([1, 2, 3, 4, 7, 8, 8, 8])
        plt.ylabel('some numbers')
        plt.show()

    def plot_end(self, y_name, mode='', plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label=''):
        # insert artificial data

        # self.plot_end('s', plt_title='Sympify zoo and nan', plt_y_label='Amount')
        if y_name == 's':  # there must be a better sulotion
            y = self.plot_failed_sympys_amount
        else:
            self.printpl('e', 'Monitoring this is not available', y_name)

        # if variance
        if mode == 'variance':
            self.printpl('e', 'variance not available', y_name)
            means = np.mean(y, axis=0)
            stds = np.std(y, axis=0)
            n = means.size

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
        plt.show()
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

    def fx_karoo_pause_refer(self):

        """
        Enables (g)eneration, (i)nteractive, and (d)e(b)ug display modes to offer the (pause) menu at each prompt.

        See fx_karoo_pause() for an explanation of the value being passed.

        Called by: the functions called by PART 4 of fx_karoo_gp()

        Arguments required: none
        """

        menu = 1
        while menu == 1:
            menu = self.fx_karoo_pause()

        return

    def fx_karoo_pause(self):

        """
        Pause the program execution and engage the user, providing a number of options.

        Called by: fx_karoo_pause_refer

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
                     'fittest_dict': self.fittest_dict,
                     'pop_a_len': len(self.population_a),
                     'pop_b_len': len(self.population_b),
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
            return 2  # breaks out of the fx_karoo_gp() or fx_karoo_pause_refer() loop

        elif input_a == 'eval':  # evaluate a Tree against the TEST data
            self.tree_algo_expr_sympify(self.population_b[input_b])  # generate the raw and sympified expression for the given Tree using SymPy
            print('\n\t\033[36mTree', input_b, 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')  # print the sympified expression

            result = self.expr_fitness_eval(str(self.algo_sym), self.data_test, get_pred_labels=True)  # might change to algo_raw evaluation
            self.fx_fitness_test(result)  # TF tested 2017 02/02

        # elif self.kernel == '[other]': # use others as a template

        elif input_a == 'print_a':  # print a Tree from population_a
            self.fx_display_tree(self.population_a[input_b])

        elif input_a == 'print_b':  # print a Tree from population_b
            self.fx_display_tree(self.population_b[input_b])

        elif input_a == 'pop_a':  # list all Trees in population_a
            print('')
            for tree_id in range(1, len(self.population_a)):
                self.tree_algo_expr_sympify(self.population_a[tree_id])  # extract the expression
                print('\t\033[36m Tree', self.population_a[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')

        elif input_a == 'pop_b':  # list all Trees in population_b
            print('')
            for tree_id in range(1, len(self.population_b)):
                self.tree_algo_expr_sympify(self.population_b[tree_id])  # extract the expression
                print('\t\033[36m Tree', self.population_b[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym,'\033[0;0m')

        elif input_a == 'load':  # load population_s to replace population_a
            self.fx_data_recover(self.filename['s'])  # NEED TO replace 's' with a user defined filename

        elif input_a == 'write':  # write the evolving population_b to disk
            self.data_tree_write(self.population_b, 'b')
            print('\n\t All current members of the evolving population_b saved to karoo_gp/runs/[date-time]/population_b.csv')

        elif input_a == 'add':  # check for added generations, then exit fx_karoo_pause and continue the run
            self.gen_max = self.gen_max + input_b  # if input_b > 0: self.gen_max = self.gen_max + input_b - REMOVED 2019 06/05

        elif input_a == 'quit':
            self.main_terminate()  # archive populations and exit

        return 1


    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Visualize a Tree              |
    # +++++++++++++++++++++++++++++++++++++++++++++

    # def fx_display_tree(self, tree):

    # def fx_display_branch(self, tree, start):
