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
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
import numpy as np

import matplotlib.pyplot as plt
import time
from pathlib import Path
from plagih.modules.plagih_eval import *
from plagih.modules.printing import *
from plagih.modules.plagih_pop import *

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


class ExplainableGP(object):
    """
    The main class performing all the important stuff
    """

    def __init__(self, config_dict):

        self.time_start = time.perf_counter()

        self.tree_meta = {}
        self.parsimony_best_meta = {}  #tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.pareto = {}
        self.population_base = []  # population that is taken to the next generation
        self.best_fitness = None

        # 1. set global variables to those local values passed from the user script
        self.config = config_dict
        self.kernel = config_dict['kernel']  # fitness function
        self.display = config_dict['display']
        self.precision = config_dict['precision']  # the number of floating points for the round function
        self.parsimony_min_max = config_dict['tree_parsimony_min_max']

        self.tnode = {}

        self.tf_device = "/gpu:0"  # Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device_log = False  # TF device usage logging (for debugging)

        self.monitor_dict = config_dict['monitor']
        self.evolve_rates = config_dict['evolve_rates']
        evolve_missing = 1 - sum(self.evolve_rates.values())
        self.evolve_rates['Create Random'] += evolve_missing

        if self.kernel == 'regression':
            self.fitness_bad_dummy = float("inf")
        else:  # 'classification' or 'match'
            self.fitness_bad_dummy = 0
        self.gene_pool = {}
        self.xype_func_dict = {'f2f': [], 'f2b': [], 'b2b': [], 'b2f': [], 'b2f2f': [],
                               '2b': [], '2f': [],
                               'b2': [], 'f2': []}  # todo, is this necessary? could be deleted

        # some useful stuff
        self.debug_warnings = {}
        self.monitoring_dict = {'genepool_size': {},
                                'fitness_average': {},
                                'best_candidate': {},
                                'total_found_trees': {}}
        runs_path = self.config['path']
        self.gen_id = 0
        self.done = False

        self.file_directories_create(runs_path)
        self.print_g('ggg', 'Init. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        return

    def plagih_gp_run(self):
        """
        regular plagih-config run from scratch
        """

        write_config_file(self.path, self.config, self.gen_id, self.kernel, self.datetime)
        self.main_generation_first_origin()
        self.main_generation_loop()  # (main loop)
        self.print_g('gg', 'GP-run. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        self.file_autowrite(self.path, 'f')
        self.auto_plots(self.path)
        self.print_g('gg', 'Completely. Exit. \tTime: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        sys.exit()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Top dim_y      functions                  |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def file_directories_create(self, path_cwd):
        """
        Create all files that will be saved after all
        """

        # self.datetime = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.datetime = datetime.now().strftime('%H%M%S')
        cwd = path_cwd
        self.path = cwd / 'runs' / '{}'.format(self.config['name'])
        path_backup = self.path / 'backup.p'
        # self.path = cwd / 'runs' / '{}{}'.format(self.config['name'], self.datetime)
        if not Path.is_dir(self.path):
            Path.mkdir(self.path)

        elif Path.is_file(path_backup):
            self.printpl('g', 'Restarting old run now...')
            self.load_backup(path_backup)
        else:
            print_e('Old run started here, but no pickle file was found')
            raise

        return

    def main_generation_first_origin(self):
        """
        Everything that needs to be done for the first generation
        - Extracts "origin Tree" from file
        - Creates all other trees: origin tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """
        self.print_g('gg', 'Preparing to evolve Generation {}'.format(self.gen_id))
        self.gen_prepare_parameters()
        self.pop_first_create()
        self.pareto[0] = self.origin['fitness_train']
        self.gen_finalize()
        file_population_write_karoo(self.population_base, '1_first', self.path, self.gen_id)  # first gen only

    def main_generation_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """

        gp_list = [('Reproduce gen', self.gen_reproduce),
                   ('Reproduce Olymp', self.gen_reproduce_olymp),
                   ('Reproduce reduce', self.gen_reproduce_reduce),
                   ('Point Mutation', self.gen_mutate_point),
                   ('Point Filter', self.gen_mutate_filter),
                   ('Branch nodebased', self.gen_mutate_branch),
                   ('Crossover one Branch', self.gen_crossover_branch),
                   ('Create Random', self.gen_create_random)]
        tourn_size = self.config['gp_tourn_size']

        while self.gen_id < self.config['gen_max'] and \
                time.perf_counter() - self.time_start < self.config['time_max'] and \
                not self.done:

            self.gen_id += 1
            self.gen_prepare_parameters()

            for name, gp_function in gp_list:
                evolve_num = int(self.evolve_rates[name] * self.config['pop_max'])
                time_evolve = time.perf_counter()
                gp_function(evolve_num, tourn_size)
                self.print_g('ggg', '{} took: {:4.2f}.'.format(name, time.perf_counter() - time_evolve))

            self.auto_procedures()

            self.gen_finalize()
            self.print_g('ggg', 'Generation took a total time of: {:4.2f}'.format(time.perf_counter() - self.time_genstart))
        else:
            self.printpl('p', '{} Enter {}?{} to review your options or {}q{}uit{}'.format(BColors.GREEN, BColors.BOLD, BColors.GREEN, BColors.BOLD, BColors.GREEN, BColors.RESET))
            # menu_continue = 0

    def auto_procedures(self, overwrite=True):
        """
        Every few generations, save the current status
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
                self.auto_plots(self.path)
                self.time_last_monitor = time_now

        if self.config['period']['time_save']:
            if self.config['period']['time_save'] < (time_now - self.time_last_files):
                self.printpl('ii', 'auto-save (time)')
                self.file_autowrite(path_auto, auto_enumname)
                self.auto_save_pickle()
                self.time_last_files = time_now

        if self.config['period']['gen_monitor']:
            if self.gen_id % int(self.config['period']['gen_monitor']) == 0:
                self.printpl('ii', 'auto-plots (gen)')
                self.auto_plots(self.path)

        if self.config['period']['gen_save']:
            if self.gen_id % int(self.config['period']['gen_save']) == 0:
                self.printpl('ii', 'auto-save (gen)')
                self.auto_save_pickle()
                self.file_autowrite(path_auto, auto_enumname)

        self.printpl('ii', 'Done with auto-procedures')

        return 0

    def file_autowrite(self, path, gen):
        """
        writes all important files

        """
        self.file_conclusion(path, datetime=self.datetime)
        self.file_pareto(self.pareto, path)
        file_population_write_karoo(self.population_new, str(gen), path, self.gen_id)  # save the final generation of Trees to disk
        return

    def gen_olympus_update(self):
        """
        thgis is actually only the pareto front?
        """
        self.printpl('e', 'TODO Olympus for candidates')
        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Load and Archive Data                     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def activate_data(self, data_prepared):
        """
        separate loading the prepared data into the main class.
        Why like this? I needed to find a bug in the data_from_csv file and
        did not want to start the whole stuff everytime
        """
        input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control = data_prepared
        self.input_dict = input_dict
        self.variables_dict = variables_dict
        self.action_dict = action_dict
        self.unique_outputs_num = unique_outputs_num
        self.data_train_rows, self.data_train, self.data_control = data_train_rows, data_train, data_control

    def activate_operators(self, op_array):
        self.op_array = op_array
        return

    def activate_pop(self, pop):
        self.population_new = pop
        return

    def load_backup(self, path_backup):
        """

        """
        with Path.open(path_backup, 'rb') as file:
            run_data = pickle.load(file)
        self.gen_id = run_data['self.gen_id']
        self.parsimony_best_meta = run_data['self.parsimony_best_meta']
        self.pareto = run_data['self.pareto']
        self.population_new = run_data['self.population_new']
        self.monitoring_dict = run_data['self.monitoring_dict']
        print('Generation:', self.gen_id)

        return

    def auto_save_pickle(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (done)
        - save the last generation (done)
        - Save valuable meta-data_csv_path: current generation (done)
        """
        path_backup = self.path / 'backup.p'

        run_data = {'self.gen_id': self.gen_id,
                    'self.parsimony_best_meta': self.parsimony_best_meta,
                    'self.pareto': self.pareto,  # todo as raw trees
                    'self.population_new': self.population_new,
                    'self.monitoring_dict': self.monitoring_dict
                    }
        pickle.dump(run_data, Path.open(path_backup, 'wb'))
        # pickle.dump(run_data, Path.open(backup_path / 'Gen-{}-backup.p'.format(str(self.gen_id)), 'wb'))
        return

    def file_conclusion(self, path, datetime=None):

        """
        write the performance of the config to disc
        """

        file = Path.open(path / 'conclusion.txt', 'w')
        file.write('Plagih GP\n launched: ' + str(datetime))

        result = eval_tf(self.origin['expr_sym'], self.data_control, self.eval_parameters, get_pred_labels=True)
        self.origin['fit_control'] = result['fitness']
        fit_control_best = result['fitness']

        fittest_algo = self.origin['expr_sym']
        fittest_parsimony = 0

        for parsimony, tree_hash in self.pareto.items():
            try:
                algo_sym = self.parsimony_best_meta[parsimony]['expr_sym']
            except:
                raise
            result = eval_tf(algo_sym, self.data_control, self.eval_parameters, get_pred_labels=True)
            fit_control = result['fitness']

            if self.fitness_compare(fit_control, fit_control_best, mode='better_or_equal'):  # find the Tree with a perfect match for all data_csv_path rows

                fit_control_best = fit_control
                fittest_algo = algo_sym
                fittest_parsimony = parsimony

            if self.kernel == 'classification':
                file.write('\n\n Classification fitness score: {}'.format(fit_control_best))
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
        file.write('\n With fitness: {}'.format(fit_control_best))
        file.write('\n\n With the following sympify-algorithm:\n {}'.format(fittest_algo))
        file.write('\n\n')
        file.close()

        return

    def file_pareto(self, pareto, path):
        """
        Save all the pareto efficient candidates to file
        """
        file = Path.open(path / 'pareto.txt', 'w')
        file.write('\nParsimony: \t<num> Fitness: \t<fitness> Expr: \t<expression>')

        for parsim, fit in sorted(list(pareto.items())):
            tree_meta = self.parsimony_best_meta[parsim]
            fitness = tree_meta['fitness_train']
            algo_sym = tree_meta['expr_sym']
            file.write('\nParsimony: \t{0} Fitness: \t{1} Expr: \t{2}'.format(str(parsim), str(fitness), str(algo_sym)))

        file.close()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Population specific                       |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def load_backup_population(self, population_backup_file):
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
        self.print_g('gggg', 'Gene Pool for Generation: {}...'.format(self.gen_id))
        dominator_count = 0
        gene_pool = {}

        for tree_num in range(1, len(population)):
            tree = population[tree_num]
            try:
                tree_ident, tree_meta = self.tree_get_meta(tree)
            except:
                continue

            population[tree_num] = tree_store_fitness(tree, tree_meta['fitness_train'], precision=self.config['precision'])
            gene_pool[tree_num] = tree_meta

            if self.fitness_compare(tree_meta['fitness_train'], self.origin['fitness_train']):
                dominator_count += 1

        self.print_g('gg', 'Generation {}, {} Candidates were better than the origin.'.format(self.gen_id, dominator_count))

        return gene_pool, population

    def pop_pareto_update(self):
        """
        Builds up the pareto front

        """

        # 1. Find lowest complexity
        best_fit = self.parsimony_best_meta[0]['fitness_train']

        sorted_parsimony_best = sorted(self.parsimony_best_meta.items(), key=lambda x: x[0])
        for parsim, meta in sorted_parsimony_best:
            fit = meta['fitness_train']

            if self.fitness_compare(best_fit, fit, mode='better_or_equal'):
                self.pareto.pop(parsim, None)

            elif self.fitness_compare(fit, best_fit):
                if self.pareto.get(parsim):
                    pareto_fit = self.pareto.get(parsim)
                    if self.fitness_compare(fit, pareto_fit):
                        self.pareto[parsim] = fit
                        self.printpl('a', 'Updated pareto at {}. New fitness is: {}, old was: {}'.format(parsim, fit, best_fit))
                else:
                    self.pareto[parsim] = fit
                    self.printpl('a', 'New pareto entry at {} with fitness: {}'.format(parsim, fit))
                best_fit = fit
        return

    def fitness_compare(self, fitness1, fitness2, mode='better'):
        """
        Compares the fitness of two candidates according to the kernel

        Example:
            >
            fitness_compare
        """
        if self.kernel == 'regression' and fitness1 < fitness2:
            return True
        elif self.kernel == 'classification' and fitness1 > fitness2:
            return True
        elif self.kernel == 'match' and fitness1 > fitness2:
            return True
        elif fitness1 == fitness2 and mode == 'better_or_equal':
            return True
        else:
            return False

    def pop_first_create(self):
        """
        Constructs the first generation
        - loads the origin-tree from file
        - constructs the first generation from this tree with branc. mutation
        """

        self.print_g('ggg', 'First population...')
        self.time_last_monitor = self.time_start
        self.time_last_files = self.time_start
        tree_origin = self.origin['tree'].copy()
        origin_ids = tree_get_mutatable_nodes(tree_origin, no_root=True)

        for tree_id in range(1, self.config['pop_max'] + 1):
            # vary this tree with mutation
            branch_top = np.random.choice(origin_ids)
            branch_nodes_ids = tree_get_branch(tree_origin, branch_top)  # [6, 9, 10] select point of mutation and all nodes beneath
            tree = self.tree_insert_branch_random(tree_origin, branch_nodes_ids)  # tree with new branch

            # Fill the correct meta-data_csv_path into the tree (and wipe the old fitness)
            tree = tree_modifyable_nodes_set(tree, self.origin['tree'])
            tree = tree_set_id(tree, 1)

            self.popnew_append(tree, last_modification='first')

        self.print_g('ggg', 'We have constructed the first population of {} trees, saved to disk'.format(self.config['pop_max']))

    def pop_parsimony_best_update(self, gene_pool):
        """

        """

        # 1. Check every potential candidate
        for key, meta in gene_pool.items():

            parsim = meta['parsimony']
            fitness_train = meta['fitness_train']

            # 3. is the tree better than the current best at this parsimony dim_y?
            if parsim in self.parsimony_best_meta:
                comp_fit = self.parsimony_best_meta[parsim]['fitness_train']
                if self.fitness_compare(fitness_train, comp_fit):
                    self.printpl('aa', 'Found a better candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                    self.parsimony_best_meta[parsim] = meta
            else:
                self.printpl('aa', 'Found a new candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                self.parsimony_best_meta[parsim] = meta

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
            tourn_winner = self.pop_selection_tournament(tourn_size)
            self.popnew_append(tourn_winner, last_modification='repro')  # i know, tests are not necessary...

        return

    def gen_reproduce_olymp(self, repro_rate, tourn_size):

        """
        Copy an entry from the pareto candidates into the population
        """
        # print('before olymp:', len(self.population_new))
        for n in range(repro_rate):  # quantity of Trees to be copied without mutation
            if self.parsimony_best_meta:
                meta = np.random.choice(list(self.parsimony_best_meta.values()))
                # expr_sym = meta['expr_sym']; print('sym', expr_sym)
                expr_raw = meta['expr_raw']
                # label_list = ast_convert_from_expr(expr_sym, build=True); print('label_list', label_list)
                label_list = ast_convert_from_expr(expr_raw, build=True)
                olymp_winner = karoo_tree_from_labellist(label_list)
                olymp_winner = tree_modifyable_nodes_set(olymp_winner, self.origin['tree'])
                self.popnew_append(olymp_winner, last_modification='repro')

                # print('Olymp winner:\n', olymp_winner)

        # print('after olymp:', len(self.population_new))
        return

    def gen_reproduce_reduce(self, repro_rate, tourn_size):

        """
        copy a tree from the last population and sympify parts of it
        """

        for i in range(repro_rate):
            tree = self.pop_selection_tournament(tourn_size)
            tree = self.treegp_reduce_parts(tree, completely=False)
            self.popnew_append(tree, last_modification='point')

        return

    def gen_mutate_point(self, repro_rate, tourn_size):

        """
        One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """

        for i in range(repro_rate):  # quantity of Trees to be generated through mutation
            tree = self.pop_selection_tournament(tourn_size)
            tree, node = self.treegp_mutate_point_evolve(tree, same_arity=True)

            self.popnew_append(tree, last_modification='point')

        return

    def gen_mutate_filter(self, repro_rate, tourn_size):
        """

        """

        for i in range(repro_rate):
            tree = self.pop_selection_tournament(tourn_size)
            try:
                self.debug_warnings['824 tree'] = tree
                new_tree = self.treegp_mutate_filter_one(tree)
                if len(new_tree) > 1:
                    self.popnew_append(new_tree, last_modification='filter')
            except Exception as ex:
                self.printpl('www', 'Tree in mutate filter could not be changed, {}'.format(ex))

        return

    def tree_random_build(self, tree):
        tree_origin = tree.copy()

        node_ids = tree_get_mutatable_layer(tree, 0)
        num_new_branches = len(node_ids)
        if num_new_branches > 3:
            print_warning('w', 'That is a lot of new trees')
        for i in range(0, num_new_branches):
            node_ids = tree_get_mutatable_layer(tree, 0)
            node_id = node_ids[i]
            old_branch = tree_get_branch(tree, node_id, karoo=True)
            tree = self.tree_insert_branch_random(tree_origin, old_branch)  # tree with new branch

        tree = tree_modifyable_nodes_set(tree, self.origin['tree'])
        tree = tree_set_id(tree, 1)
        return tree

    def gen_create_random(self, repro_rate, tourn_size):
        """

        """
        tree_origin = self.origin['tree'].copy()
        for i in range(repro_rate):

            tree = self.tree_random_build(tree_origin)
            tree = tree_modifyable_nodes_set(tree, self.origin['tree'])
            tree = tree_set_id(tree, 1)

            self.popnew_append(tree, last_modification='miss(br)')

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

            tourn_winner = self.pop_selection_tournament(tourn_size)  # perform tournament selection for each mutation
            node_ids = tree_get_mutatable_nodes(tourn_winner, no_root=True)
            node = np.random.choice(node_ids)
            branch_nodes_ids = tree_get_branch(tourn_winner, node)  # select point of mutation and all nodes beneath [6, 9, 10]
            tourn_winner = self.tree_insert_branch_random(tourn_winner, branch_nodes_ids)

            self.popnew_append(tourn_winner, last_modification='branch')

        return

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
            left_tree = self.pop_selection_tournament(tourn_size)  # perform tournament selection for 'a_parent'
            right_tree = self.pop_selection_tournament(tourn_size)  # perform tournament selection for 'b_parent'

            force_convert = False

            # 2. search nodes for left and right that can be exchanged. convert_needed
            left_id, right_id, success = self.tree_try_get_swapids(left_tree, right_tree)
            if not success:
                right_id, left_id, success = self.tree_try_get_swapids(right_tree, left_tree)
            if not success:
                force_convert = True

            left_ids, left_labels, left_aritys = tree_get_branch_plus(left_tree, left_id)
            right_ids, right_labels, right_aritys = tree_get_branch_plus(right_tree, right_id)

            if force_convert:
                self.printpl('w', 'Crossover conversion between trees forced. \n{}\n{}'.format(left_tree, right_tree))
                left_xtype = self.xtype_get(tree_get_label(left_tree, left_id))  # left_tree[N_label][left_id]
                conv_to_left, conv_to_right = xtype_get_converters(left_xtype)
                right_labels.insert(0, conv_to_left)
                right_aritys.insert(0, 1)
                left_labels.insert(0, conv_to_right)
                left_aritys.insert(0, 1)

            left_core = core_from_labels(left_labels, left_aritys)
            right_core = core_from_labels(right_labels, right_aritys)

            left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
            left_offspring = self.treegp_tree_prune(left_offspring, self.config['tree_depth_max'])
            self.popnew_append(left_offspring, last_modification='cross')

            right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
            right_offspring = self.treegp_tree_prune(right_offspring, self.config['tree_depth_max'])
            self.popnew_append(right_offspring, last_modification='cross')
            # right_offspring = self.treegp_crossover_insert(right_tree, right_ids, left_tree, left_ids, convert_needed=convert_needed)
        return

    def popnew_append(self, tree, last_modification=''):
        """
        some stuff to definitely do before actually appending to pop
        """
        tree = tree_round_constants(tree, self.config['float_accuracy'], karoo=True)
        tree = tree_modifyable_nodes_set(tree, self.origin['tree'])
        tree = tree_normalize_exponentiation(tree)
        tree = tree_set_history(tree, last_modification)
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
        gene_pool, self.population_new = self.pop_genepool_create(self.population_new)

        self.monitor_genepool(gene_pool, self.path, self.gen_id)

        self.pop_parsimony_best_update(gene_pool)

        self.pop_pareto_update()

        self.population_base = pop_copy_genepool(self.population_new, gene_pool, self.gen_id)
        file_population_write_karoo(self.population_new, 'new', self.path, self.gen_id)

        self.monitoring_dict['total_found_trees'][self.gen_id] = len(self.tree_meta)
        self.print_g('gg', 'Monitoring: Created {}/{} unique trees in generation {}. Gen-time: {:4.2f}'.format(
            len(set(gene_pool.keys())),
            self.config['pop_max'],
            self.gen_id,
            time.perf_counter() - self.time_genstart))

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Perform the 3 genetic prog. operations    |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_selection_tournament(self, tourn_size):

        """
        config-selection. takes a number of trees (we use 3) and returns the best one (winner)

        """

        # Start with dummies
        best_id = -1
        best_fitness = self.fitness_bad_dummy

        # Get several values
        for n in range(tourn_size):

            tree_id = pop_random(self.population_base)
            tree = self.population_base[tree_id]
            fitness = tree_get_fitness(tree, precision=self.config['precision'])  # extract the fitness from the array

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
                printez('w', 'Warning: Filter  not specified. Please specify a filter_type.', display=self.display)
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
        node_id = np.random.choice(tree_get_mutatable_nodes(tree))
        label = tree_get_label(tree, node_id)
        arity = tree_label_get_arity(label)  # int(tree[N_arity][node_id])
        xtype = self.xtype_get(label)  # '>' -> 'f2b'

        if same_arity:
            # 2. perform point mutation on that specific node
            if arity > 0:
                tree[N_label][node_id] = xtype_choose_func_pointmutation(self.op_array, xtype=xtype, arity=arity)  # Function is same type, same arity
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

    def treegp_reduce_branch(self, tree, node_id):
        delete_ids = tree_get_branch(tree, node_id)
        expr_raw = tree_expr_raw(tree, node_id)
        try:
            expr_sym = tree_expr_sympify(algo_raw=expr_raw)
            label_list = ast_convert_from_expr(expr_sym, build=True)
            arity_list = [tree_label_get_arity(label) for label in label_list]  # todo zeile auslagern?
            core = core_from_labels(label_list, arity_list)
            tree_sympified = tree_insert_subtree(tree, core, delete_ids, karoo=True)

            return tree_sympified
        except:
            print_warning('w', 'reducing expr raw: {}'.format(expr_raw))
            print_warning('w', 'Delete this tree! nan tree or other error.')
            return tree

    def treegp_reduce_parts(self, tree, completely=False):

        """
        Mutate a single mutatable point in any Tree.
        """

        if completely:
            # todo test this
            nodes_lv0 = tree_get_mutatable_layer_lv0(tree, karoo=True)
            for node_id in nodes_lv0:
                tree = self.treegp_reduce_branch(tree, node_id)
        else:
            node_ids = tree_get_mutatable_nodes(tree)
            # todo only functions? maybe always all?
            node_id = np.random.choice(node_ids)  # choose
            tree = self.treegp_reduce_branch(tree, node_id)
        return tree

    def treegp_mutate_filter_one(self, tree):
        """
        Mutates one float terminal of a tree
        """
        # 1. choose a node
        node_ids = tree_get_mutatable_nodes(tree)
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

    def treegp_tree_prune(self, tree, depth):
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
        a_ids = tree_get_mutatable_nodes(a_tree, no_root=True)
        a_id = np.random.choice(a_ids)
        a_xtype = self.xtype_get(a_tree[N_label][a_id])

        # create a list from parent b with same xtype
        b_ids = tree_get_mutatable_nodes(b_tree, no_root=True)
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
    #     r_labels, r_aritys = tree_branch_get_label_list(right_parent, right_ids, tests=True)
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
    #         left_parent = tree_insert_subtree(left_parent, right_core, left_ids, tests=True)
    #         left_parent = self.treegp_crossover_tree_prune(left_parent, self.config['tree_depth_max'])  # sfeh: not sure if this is necessary?
    #
    #     return left_parent

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Utility  functions to evolve a tree       |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def tree_insert_branch_random(self, tree, branch_ids, grow_method='depth_base_random'):

        """
        # TODO would be nicer is this just returned a new branch and insert is separately
        replaces the branch_ids in a tree with a new branch
        Given: Tree and a list of node ids
        - checks how far to build down
        - checks the old nodes xtype, etc.
        - checks if we are not too far down the tree
        -

        returns: new tree
        """

        if grow_method == 'depth_base_random':
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
            label_list, arity_list = invent_label_list_depth_random(old_xtype, depth_goal, self.variables_dict, self.action_dict, self.op_array)

            if not label_list:
                result_tree = tree
            else:
                core_insert = core_from_labels(label_list, arity_list)
                result_tree = tree_insert_subtree(tree, core_insert, branch_ids, karoo=True)

        elif grow_method == 'num_nodes':
            return None
        elif grow_method == 'nodes_max_uniform':
            """
            We allow a certain amount of new nodes instead tree depth.
            This could be calculated respectively to the parsimony dim_y
            which the tree might have up his sleeve
            """
            return None
            # num_new_nodes = np.random.randint(10, 30)
            # max_

        else:
            print_e('That did not work')
            return None

        return result_tree

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def load_origin_tree(self, origin_tree_file_path=None, label_list=None, modify_list=None):
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
                    print_e('Tree could not be imported correctly from .csv file.')
                    raise
        elif label_list:
            tree = karoo_tree_from_labellist(label_list, modify_list=modify_list)
        else:
            print_warning('w', 'No origin provided. Todo. starting from scratch with random generation?')
            raise

        self.eval_parameters = {
            'kernel': self.kernel,
            'action_dict': self.action_dict,
            'variables_dict': self.variables_dict,
            'tf_device_log': self.tf_device_log,
            'tf_device': self.tf_device,
            'unique_outputs_num': self.unique_outputs_num,
            'tf_classify_labels_map': self.tf_classify_labels_map}

        origin_algo_raw = tree_expr_raw(tree, P_first_node)
        try:
            expr_sym = tree_expr_sympify(algo_raw=origin_algo_raw)
        except:
            raise Exception('Your origin algorithm can already not be sympified. Aborting.')
        self.origin = {'tree': tree,
                       'expr_raw': origin_algo_raw,
                       'expr_sym': expr_sym,
                       'parsimony': 0}
        try:
            origin_hash, origin_meta = self.tree_get_meta(tree)
        except Exception as ex:
            whats_wrong = 'Your origin algorithm already caused an exception. {}'.format(ex)
            raise Exception(whats_wrong)
        self.origin['fitness_train'] = origin_meta['fitness_train']

        self.parsimony_best_meta[0] = origin_meta

        self.hashtable_fitness_train = {}

        self.print_g('gg', 'Loading origin, fitness {}. Time: {:4.2f}s'.format(origin_meta['fitness_train'], time.perf_counter() - self.time_start))
        return

    def tree_get_meta(self, tree):
        """
        gets all the main tree information
        1. tree_identifier (expr_raw)
        2. expr_sym
        3. parsimony
        4. fitness_train
        """
        # 1. get expr_raw - what is needed to compute the tree identifier
        expr_raw = tree_expr_raw(tree, root_id)
        tree_ident = hash(expr_raw)  # sfeh: potential for improvement- use expr_sym in separate dict as identifier.

        # 2 Did we have this tree already? -> Nice, we have everything
        if tree_ident in self.tree_meta:
            tree_meta = self.tree_meta[tree_ident]

        else:  # 2.2 New tree, but still skip fitness eval for complex trees
            parsimony = self.tree_parsimony(tree, origin_tree=self.origin['tree'])

            if parsimony < self.parsimony_min_max[1]:  # 3. compute fitness
                # print('Algo raw:', str(expr_raw))  # importantprint 2 for expr_raw
                try:  # 3. With tensorflow
                    expr_sym = tree_expr_sympify(algo_raw=str(expr_raw))
                except:
                    raise Exception('Expr could not be sympified: {}.'.format(expr_raw))

                fitness_train = eval_tf(expr_sym, self.data_train, self.eval_parameters)['fitness']
                tree_meta = {'parsimony': float(parsimony), 'fitness_train': float(fitness_train), 'expr_sym': str(expr_sym), 'expr_raw': str(expr_raw)}
                self.tree_meta[tree_ident] = tree_meta

            else:
                # 3.2 just fill with bad values
                # expr_sym = sympy_dummy
                # fitness_train = self.fitness_bad_dummy
                raise Exception('Tree too complex, parsimony is too high.')

        return tree_ident, tree_meta

    def remove_this_tree(self):
        self.printpl('ww', 'This still is a todo')
        """
        If a tree makes problems, delete it somehow.
        - set parsimony very high?
        todo
        """

    def tree_parsimony(self, tree, origin_tree=None, parsimony_distance='ted'):
        """
        parsimony_distance: compute the chosen distance by the user.

        """
        if parsimony_distance == 'ted':
            return tree_parsimony_ted(tree, origin_tree)
        elif parsimony_distance == 'total_count_nodes':
            return int(tree[3][-1:])  # returns the tree size
        elif parsimony_distance == 'total_tree_depth':
            return tree[N_depth][1]  # returns the tree size
        elif parsimony_distance == 'total_karoo_original':  # do not use with long variable names
            algo_raw_str = str(tree_expr_raw(tree, root_id))
            return len(str(algo_raw_str))
        # elif parsimony_distance == 'total_simplified':
        #     algo_sym = self.tree_expr_sympify(tree=tree)
        #     return count_ops(algo_sym)
        elif parsimony_distance == 'rel_ari_1':  # Does this work?
            return tree_parsimony_relari(tree, origin_tree)
        else:
            raise Exception('Parsimony distance not specified!')

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to use evaluate (tensorflow)      |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def tf_classify_labels_map(self, result):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the data_csv_path .csv. Outputs an array of tuples containing the predicted
        labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.unique_outputs_num / 2) - 1 # '-1' keeps a binary classification splitting over the origin
            if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
            elif solution == self.unique_outputs_num - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
            else: fitness = 0 # no class match

        """

        skew = (self.unique_outputs_num / 2) - 1
        label_rules = {self.unique_outputs_num - 1: (
            tf.constant(self.unique_outputs_num - 1), tf.constant(' > {}'.format(self.unique_outputs_num - 2 - skew)))}

        for class_label in range(self.unique_outputs_num - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tf.cond(cond, lambda: (
                tf.constant(class_label), tf.constant(' <= {}'.format(class_label - skew))),
                                               lambda: label_rules[class_label + 1])

        pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))),
                             lambda: label_rules[1])

        return pred_label

    def xtype_choose_func(self, xtype=None):
        """
        This fills in a function that fits the type of the function/terminal before.
        terminal  '2f' -> '_2f', arity
        function 'f2f' -> '_2f', arity
        function 'b2f2f' -> '_2f', arity
        > ->
        """
        if xtype:
            if '2f' in xtype:
                choose_func = sum(self.op_array[f2f] + self.op_array[b2f] + self.op_array[b2f2f], [])
            elif '2b' in xtype:
                choose_func = sum(self.op_array[f2b] + self.op_array[b2b], [])
            else:
                raise
        else:
            choose_func = sum(self.op_array[f2f] +
                              self.op_array[b2f] +
                              self.op_array[b2f2f] +
                              self.op_array[f2b] +
                              self.op_array[b2b], [])

        # Attention! do not choose out of an dictionary.
        # every function is inside there only once, so no higher chance for functions that are more often in the list
        # label = np.random.choice(self.xype_func_dict['2f'])

        # choose out of a list, or add another way. maybe autmatic?

        label = np.random.choice(choose_func)

        return label, op[str(label)]['arity']

    def xtype_get(self, label, node_arity=None):
        """
        returns xtype for a label
        """
        if not node_arity:
            node_arity = tree_label_get_arity(label)

        if node_arity == 0:  # arity=0 -> terminal
            if 'True' in label or 'False' in label:
                node_xtype = '2b'
            elif 'observation' in label:
                term_position = self.variables_dict['all'].index(label)
                node_xtype = op[self.variables_dict['types'][term_position]]['xtype']
            elif 'action' in label:
                print_warning('w', 'Does this happen? Test it!')
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
        return tree_build_type_constant_get(term_type=the_type)  # otherwise: constant (There are always constants :P)

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def auto_plots(self, path, post_path=None):
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
        self.plot_end(data_tupels, path, plt_title='Pareto Dominant Candidates', only_dots=True, plt_x_label='Parsimony', plt_y_label='Fitness')

        data_tupels = sorted(list(self.monitoring_dict['best_candidate'].items()))
        self.plot_end(data_tupels, path, plt_title='Best candidate', only_dots=True, plt_x_label='Generation', plt_y_label='Fitness')

        return

    def monitor_genepool(self, gene_pool, path, gen_id):
        """
        Give the user some feedback
        """

        # How many survived in the selection?
        self.monitoring_dict['genepool_size'][int(gen_id)] = len(gene_pool)
        if len(gene_pool) > 0:
            self.print_g('ggg', 'The generation`s population is: {}'.format(len(gene_pool)))
        else:
            self.printpl('e', 'There are no Trees in the gene pool. You should archive your population and (q)uit.')
            self.file_autowrite(path, gen_id)
            self.auto_plots(path)
            sys.exit()

        if not self.best_fitness:
            self.best_fitness = self.origin['fitness_train']

        fitness_train_sum = 0
        for _, meta in gene_pool.items():
            fitness = float(meta['fitness_train'])
            fitness_train_sum += fitness  # for fitness average
            if self.fitness_compare(fitness, self.best_fitness):
                self.best_fitness = fitness

        average_fitness = fitness_train_sum / len(gene_pool)
        self.monitoring_dict['fitness_average'][gen_id] = average_fitness
        self.monitoring_dict['best_candidate'][gen_id] = self.best_fitness

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to display output information     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def plot_end(self, data_2d, path, plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='', yscale='linear', only_dots=None, variance=None):

        x, y = [], []
        for a, b in data_2d:
            x.append(a)
            y.append(b)

        if variance:
            print_e('variance not available')
            means = np.mean(y, axis=0)
            stds = np.std(y, axis=0)
            n = means.size

        if only_dots:
            plt.plot(x, y, linestyle='', marker='o', label=plt_curve_label)
        else:
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

    def printpl(self, message_type, text):

        """

        """

        if message_type in self.display:
            printez(message_type, text, display=self.display, time_total=time.perf_counter() - self.time_start)

        return

    def print_g(self, message_type, text):

        """

        """

        if message_type in self.display:
            printez(message_type, text, time_total=time.perf_counter() - self.time_start)

        return


def pop_util_copy(population_x, title):
    """
    Copy one population to another.
    """
    population_y = [title]  # an empty list stores a copy of the prior generation

    for tree_id in range(1, len(population_x)):  # increment through each Tree in the current population
        tree_copy = util_tree_copy(population_x, tree_id)  # copy each array in the current population
        population_y.append(tree_copy)  # add each copied Tree to the new population list

    return population_y


def pop_enum_trees(population):
    """
    outsourced enumeration of trees in a population
    """
    for tree_id in range(FIRST_TREE, len(population)):  #
        population[tree_id][TR_ID][1] = tree_id
    return population


def data_load_data_split(data_x, data_y, test_size):

    x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size=test_size)  # 80/20 TRAIN/TEST split
    data_train = np.c_[x_train, y_train]  # recombine each row of data_csv_path with its associated class label (right column)
    data_control = np.c_[x_test, y_test]  # recombine each row of data_csv_path with its associated class label (right column)

    data_train_rows = len(data_train[:, 0])

    return data_train_rows, data_train, data_control


def data_from_csv(samples_file):
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
    input_dict = {'all': {},
                  'float': {},
                  'bool': {}}
    variables_dict = {'all': [],
                      'types': [],
                      'float': [],
                      'bool': []}

    action_dict = {}

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
                        input_dict[term] = term_type
                        variables_dict['all'].append(term)
                        variables_dict['types'].append(term_type)
                        if term_type == 'float':
                            variables_dict['float'].append(term)
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
                        print_e('Behaviour samples first line: Variables have to start with "o" or "a" to be recognized. Is actually: {}'.format(var_name))
                        raise

                data_x, data_y = [], []

            else:  # convert every 'string' element to its data_csv_path type
                row_as_data = [locate(var_types[i])(x) for i, x in enumerate(row)]  # ['observation0:float'] + ['0.123'] --> float(['0.123']) --> 0.123
                data_x.append(row_as_data[:num_observations])
                data_y.append(row_as_data[num_observations:])
        csvFile.close()
    unique_outputs_num = len(np.unique(data_y))  # load the user defined true labels for classification or solutions for regression

    data_train_rows, data_train, data_control = data_load_data_split(data_x, data_y, test_size=0.2)

    # self.printplg('g', 'Loading samples. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
    return input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control


def data_load_pickle(prepared_data_pickle_path):
    """
    loads a data_csv_path file that was already split with the csv reader
    """
    with Path.open(prepared_data_pickle_path, 'rb') as file:
        pickle_data = pickle.load(file)

    # self.printplg('g', 'Pickle-loading samples. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
    return pickle_data  # input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control


def save_data_pickle(prepared_data, data_pickle_path):
    """
    saves prepared plagih data to pickle file
    """

    with Path.open(data_pickle_path, 'wb') as file:
        pickle.dump(prepared_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    return


def file_population_write_plagih(population, pop_name, path, gen_id):

    file_path = path / 'population_plagih_{}.csv'.format(str(pop_name))

    with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
        target = csv.writer(csv_file, delimiter=',')
        if gen_id != 0:
            target.writerows([''])  # empty row before each generation
        target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(gen_id)]])

        for tree in range(1, len(population)):
            target.writerows([''])  # empty row before each Tree
            for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                target.writerows([population[tree][row]])

    return


def file_population_write_karoo(population, pop_name, path, gen_id):
    """
    Save population_* to disk.

    """
    file_path = path / 'population_{}.csv'.format(str(pop_name))

    with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
        target = csv.writer(csv_file, delimiter=',')
        if gen_id != 0:
            target.writerows([''])  # empty row before each generation
        target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(gen_id)]])

        for tree in range(1, len(population)):
            target.writerows([''])  # empty row before each Tree
            for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                target.writerows([population[tree][row]])

    return


def write_config_file(path, config, gen_id, kernel, datetime):
    """
    write the parameters to a file
    """

    file = Path.open(path / 'config.txt', 'w')
    file.write('Plagih GP. This config is not complete, sfeh!')
    file.write('\n launched: {}'.format(datetime))
    file.write('\n kernel: {}'.format(kernel))
    file.write('\n precision: {}\n'.format(config['precision']))
    file.write('\n tree depth max: ' + str(config['tree_depth_max']))
    file.write('\n')
    file.write('\n tournament size: ' + str(config['gp_tourn_size']))
    file.write('\n population: ' + str(config['pop_max']))
    file.write('\n number of generations: ' + str(gen_id))
    file.write('\n\n')
    file.close()
    return


def load_operators_from_csv(op_csv_path):
    """
    Load all operators ready-to-use from a file
    """

    functions = np.loadtxt(op_csv_path, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
    # Part 3.5: Split the functions in 5 types

    # rows are the function types (f2f)
    # columns are the arity
    op_array = [[[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []]]
    for fun in functions:
        label = fun[0]
        arity = op[label]['arity']  # arity = int(fun[1])
        xtype = op[label]['xtype']

        if xtype == 'f2f':
            op_array[f2f][arity].append(label)
        elif xtype == 'f2b':
            op_array[f2b][arity].append(label)
        elif xtype == 'b2b':
            op_array[b2b][arity].append(label)
        elif xtype == 'b2f':
            op_array[b2f][arity].append(label)
        elif xtype == 'b2f2f':
            op_array[b2f2f][arity].append(label)

    return op_array


def load_pop_from_csv(pop_csv):

    """
    This method is used to load a saved population of Trees, as invoked through the (pause) menu where population_r
    replaces population_a in the karoo_gp/runs/[date-time]/ directory.
    """

    with Path.open(pop_csv, 'r') as csv_file:
        target = csv.reader(csv_file, delimiter=',')
        n = 0  # track row count

        for row in target:

            n = n + 1
            if n == 1:
                pass  # skip first empty row

            elif n == 2:
                population_a = [row]  # write header to population_a

            else:
                if not row:
                    tree = np.array([[]])  # initialise Tree array

                else:
                    if tree.shape[1] == 0:
                        tree = np.append(tree, [row], axis=1)  # append first row to Tree

                    else:
                        tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree

                if tree.shape[0] == T_num_lines:
                    population_a.append(tree)  # append complete Tree to population list

    return population_a
