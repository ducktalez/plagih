"""
Explaination:
> 'f2f', 'b2b', etc.: my personal silly naming. f = float, b = bool.
   f2b means float to boolean, e. g. '<' takes 'float' and returns a 'bool'
> node_modify (0 or 1), specifies, whether this node is supposed to be




Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""
import copy
import random
import sys
import sklearn.metrics as skm
from datetime import datetime
from plagih.modules.plagih_tree import *
import time
from plagih.modules.plagih_eval import *
from plagih.modules.file_interaction import *

import tikzplotlib

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


class ExplainableGP(object):
    """
    The main class performing all the important stuff
    """

    def __init__(self, config_dict):

        print('\n\tInitializing Plagih. Name: {}{}{}.\n'.format(BColors.CYAN, config_dict['name'], BColors.RESET))
        self.time_start = time.perf_counter()
        self.restart_vers = 'v0.75'

        # init values with dummies (just to have all self values here for overview)
        self.tree_meta = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw', 'gen+nr'}
        self.parsimony_best_meta = {}  # tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.pareto = {}  # a dict with all pareto candidates. key is complexity, value is tree meta
        self.population_base = []  # population that is taken to the next generation
        self.best_fitness = None  # keeps track of the current best fitness
        self.action_min_max = [None, None]  # list with [0] = min and [1] = max, For kernel "regression bounded" (or so)
        self.origin = None
        self.gene_pool = {}
        self.debug_warnings = {}
        self.gen_id = 0
        self.custom_done = False
        self.restart_count = 0
        self.time_last_monitor = self.time_start
        self.time_last_files = self.time_start
        self.pop_next = None
        self.output_xtype = None

        # some config_dict values have to be used quite often...
        self.config = config_dict
        # self.kernel = config_dict['kernel_name']  # fitness function
        self.kernel = FitnessKernel(config_dict['kernel_name'])
        self.print_type = config_dict['print_type']
        self.precision = config_dict['precision']  # the number of floating points for the round function
        self.parsimony_tmp = config_dict['parsimony_tmp']
        self.parsimony_max = config_dict['parsimony_max']
        self.monitor_dict = config_dict['monitor']
        self.evolve_rates = config_dict['evolve_rates']
        self.tourn_size = config_dict['tourn_size']

        # special variables
        self.tf_device = "/gpu:0"  # Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device_log = False  # TF device usage logging (for debugging)

        # some useful stuff
        self.monitoring_dict = {'population_tmp_done-size': {},
                                'fitness_average': {},
                                'best_candidate': {},
                                'total_found_trees': {},
                                'complexity_average': {}}

        self.file_directories_create(self.config['root_dir'])
        self.print_g('ggg', 'Init. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        return

    def plagih_update_files(self):
        """
        Without starting a new run, get the most important files
        """

        path_backup = self.root_dir / file_backup_pickle
        if Path.is_file(path_backup):
            self.print_g('g', 'Backup file for updating analysis files exists...')
            try:
                self.plagih_load_backup(path_backup)
            except Exception as ex:
                print_warning('w', 'Even though a backup exists for this run, it could not be loaded because of: {}\nStarting a new run.'.format(ex))
                raise
            self.terminate_run(self.root_dir)
        else:
            raise IOError('Backup file does not exist!')

    def plagih_gp_run(self):
        """
        regular plagih-config run
        """

        path_backup = self.root_dir / file_backup_pickle
        if Path.is_file(path_backup) and not self.config['force_new_run']:
            self.print_g('g', 'Restarting old run now...')
            try:
                self.plagih_load_backup(path_backup)
            except Exception as ex:
                print_warning('w', 'Even though a backup exists for this run, it could not be loaded because of: {}\nStarting a new run.'.format(ex))
                # todo delete old plots, append to old config,

        if self.gen_id == 0:
            self.gen_create_first()

        write_config_file(self.root_dir, self.config, self.gen_id, self.kernel, self.datetime)

        self.gen_create_loop()
        self.terminate_run(self.root_dir)

    def file_directories_create(self, path_cwd):
        """
        Create all files that will be saved after all
        """

        # self.datetime = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.datetime = datetime.now().strftime('%H%M%S')

        self.root_dir = make_dir(path_cwd / folder_runs / '{}'.format(self.config['name']))

        return

    def gen_create_first(self):
        """
        Everything that needs to be custom_done for the first generation
        - Extracts "origin Tree" from file
        - Creates all other trees: origin tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """
        self.print_g('gg', 'Preparing to create first Generation. Gen {}.'.format(self.gen_id))
        self.gen_reset_parameters()

        rate_o = self.evolve_rates['random from origin']
        rate_s = self.evolve_rates['random from scratch']
        pop_max = self.config['pop_max']

        if self.origin_exists() and rate_o > 0:
            rate_sum = rate_s + rate_o
            rate_s = int((rate_s / rate_sum) * pop_max)
            rate_o = pop_max - rate_s
            self.pop_random_from_origin(rate_o)
        else:
            rate_s = pop_max

        self.pop_random_from_scratch(rate_s)

        # if delete_this and self.gen_id != 0:
        #     print('why is this gen id:', self.gen_id)

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

        gp_list = [  # name of the function,    implementation in plagih,       number of tournament selections needed
            ('repro one', self.pop_reproduce, 1),
            ('repro pareto', self.pop_reproduce_olymp, 0),
            ('repro reduced one', self.pop_reproduce_reduce, 1),
            ('point mutate function', self.pop_mutate_point, 1),
            ('filter floats', self.pop_mutate_filter, 1),
            ('branch mutate insert', self.pop_mutate_branch, 1),
            ('crossover branches', self.pop_crossover_branch, 2),
            ('random from origin', self.pop_random_from_origin, 0),
            ('random from scratch', self.pop_random_from_scratch, 0)]

        if self.origin_exists():
            origin_tree = self.origin['tree']
        else:
            origin_tree = None

        gp_dict = {  # name of the function,    implementation in plagih,       number of tournament selections needed
            'repro one': (self.pop_reproduce, 1, None),
            'repro pareto': (self.pop_reproduce_olymp, 0, None),
            'repro reduced one': (self.pop_reproduce_reduce, 1, None),
            'point mutate function': (self.pop_mutate_point, 1, None),
            'filter floats': (self.pop_mutate_filter, 1, None),
            'branch mutate insert': (self.pop_mutate_branch, 1, None),
            'crossover branches': (self.pop_crossover_branch, 2, None),
            'random from origin': (self.pop_random_from_origin, 0, origin_tree),
            'random from scratch': (self.pop_random_from_scratch, 0, None)}

        while self.run_continues():  # max generation, max time, done...

            self.gen_reset_parameters()

            # Creating a new population ############
            for name, gp_function, tourn_rep in gp_list:
                time_evolve = time.perf_counter()
                count_tries = 0
                while count_tries < self.evolve_rates[name] * self.config['pop_max']:
                    break
                    # smallest inserted rate
                repro_rate = int(self.evolve_rates[name] * self.config['pop_max'])
                gp_function(repro_rate)
                self.print_g('ggg', '-->Evolve: ({}) took: {:4.2f}sec.'.format(name, time.perf_counter() - time_evolve))
            # ######################################

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
        cond_2 = time.perf_counter() - self.time_start > self.config['time_max']
        cond_3 = self.custom_done
        if cond_1 or cond_2 or cond_3:
            return False
        else:
            return True

    def origin_exists(self):
        if self.origin is not None:
            return True
        else:
            return False

    def periodical_procedures(self):
        """
        Every few generations, update the created files
        - default is in every generation, but saving every n-th gen or after time passed is possible aswell
        """
        if self.config['overwrite periodic files']:
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
            self.auto_plots(tmp_path)

        if save_run:
            self.run_save_pickle()
            self.file_save_files(tmp_path)

        self.printpl('ii', 'Done with auto-procedures')

        return 0

    def file_save_files(self, root_path, pop_name=''):
        """
        writes all important files

        """
        self.file_conclusion(root_path, date_time=self.datetime)
        self.file_pareto(self.pareto, root_path)
        self.file_pareto_latex(self.pareto, root_path)
        file_population_karoo(self.population_base, pop_name, root_path, self.gen_id)  # save the final generation of Trees to disk

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Load and Archive Data                     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def activate_dataset(self, data_prepared):
        """
        separate loading the prepared data into the main class.
        Why like this? I needed to find a bug in the data_from_csv file and
        did not want to start the whole stuff everytime
        """
        input_dict, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control, action_min_max = data_prepared
        self.input_dict = input_dict
        self.variables_dict = variables_dict
        self.action_dict = action_dict
        self.unique_outputs_num = unique_outputs_num
        self.data_train_rows, self.data_train, self.data_control = data_train_rows, data_train, data_control
        self.action_min_max = action_min_max

        self.eval_parameters = {
            'kernel_name': self.kernel,
            'action_dict': self.action_dict,
            'variables_dict': self.variables_dict,
            'tf_device_log': self.tf_device_log,
            'tf_device': self.tf_device,
            'unique_outputs_num': self.unique_outputs_num,
            'tf_classify_labels_map': self.tf_classify_labels_map,
            'action_min_max': self.action_min_max}

        self.output_xtype = self.output_get_xtype()

        return

    def activate_operators(self, func_array):
        """
        operators were loaded already and need to be set in the gp run
        """
        self.func_array = func_array
        return

    def plagih_load_backup(self, path_backup):
        """

        """

        # restart_vers = '0.7'  # change this version is backups change. Or dont. Fiilures will come anyways.
        # run_info = [('self.restart_count', self.restart_count),
        #                       ('self.restart_vers', self.restart_vers),
        #                       ('self.gen_id', self.gen_id),
        #                       ('self.parsimony_best_meta', self.parsimony_best_meta),
        #                       ('self.pareto', self.pareto),
        #                       ('self.population_base', self.population_base),
        #                       ('self.monitoring_dict', self.monitoring_dict)]

        with Path.open(path_backup, 'rb') as file:
            run_data = pickle.load(file)

        try:
            self.monitoring_dict['population_tmp_done-size'] = run_data['genepool_size']
            print_warning('w', 'Delete this. Restarting from an old run, where gene_pool existed.')
        except:
            print_warning('w', 'Success, delete this. This is a newer version where gene_pool was kicked out')

        self.restart_count = run_data['self.restart_count']  # counting how often this was restarted
        self.gen_id = run_data['self.gen_id']
        self.parsimony_best_meta = run_data['self.parsimony_best_meta']
        self.pareto = run_data['self.pareto']
        self.population_base = run_data['self.population_base']
        self.monitoring_dict = run_data['self.monitoring_dict']

        if type(next(iter(run_data['self.pareto']))) == type(0.6):
            self.pareto = None
            self.pareto_update()
            raise Exception('TODO pareto is outdated')

        self.restart_count += 1
        printez('g', 'Loading Generation: {}'.format(self.gen_id), self.print_type)

        return

    def run_save_pickle(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """
        path_backup = self.root_dir / file_backup_pickle

        run_data = {'self.restart_count': self.restart_count,
                    'self.gen_id': self.gen_id,
                    'self.parsimony_best_meta': self.parsimony_best_meta,
                    'self.pareto': self.pareto,  # todo as raw trees
                    'self.population_base': self.population_base,
                    'self.monitoring_dict': self.monitoring_dict
                    }
        pickle.dump(run_data, Path.open(path_backup, 'wb'))
        return

    def file_conclusion(self, path, date_time=None):

        """
        write the performance of the config to disc
        """
        path_conclusion = path / folder_info
        if not Path.is_dir(path_conclusion):
            Path.mkdir(path_conclusion)

        file = Path.open(path_conclusion / file_conclusion, 'w')
        file.write('Plagih GP\n launched: {}'.format(str(date_time)))

        if self.origin_exists():
            origin_result = eval_tf(self.origin['expr_sym'], self.data_control, self.eval_parameters, get_pred_labels=True)
            fitness_control_best = origin_result['fitness']

            fittest_algo = self.origin['expr_sym']
            fittest_parsimony = 0

            file.write('\n\t Origin fitness score: {}'.format(origin_result['fitness']))

        elif self.pareto:
            file.write('\n No origin was provided')
            fitness_control_best = next(iter(self.pareto.keys()))
            tree_meta = self.parsimony_best_meta[fitness_control_best]
            fittest_parsimony = int(tree_meta['parsimony'])
            fittest_algo = tree_meta['expr_sym']
        else:
            file.write('\n There are no candidates to be mentioned at all. Maybe change your config?')
            return

        for parsimony, fitness in self.pareto.items():
            algo_sym = self.parsimony_best_meta[parsimony]['expr_sym']
            result = eval_tf(algo_sym, self.data_control, self.eval_parameters, get_pred_labels=True)
            fit_control = result['fitness']

            if self.kernel.fitness_compare(fit_control, fitness_control_best, mode='better_or_equal'):  # find the Tree with a perfect match for all data_csv_path rows
                fitness_control_best = fit_control
                fittest_algo = algo_sym
                fittest_parsimony = parsimony

            no_fault = True
            for enum, entry in enumerate(result['tf_result']):
                if not self.check_value_is_real(entry):
                    no_fault = False
                    # todo this is a bad workaround
                    result['tf_result'][enum] = 1

            if no_fault:
                kernel_result = self.kernel.conclusion_get_text(result, fitness_control_best)
                file.write(kernel_result)
            else:
                file.write('\n\n Error in this tree')

        else:
            # Info about the best Tree
            file.write('\n\n The best candidate has parsimony: {}'.format(str(fittest_parsimony)))
            file.write('\n With fitness: {}'.format(fitness_control_best))
            file.write('\n\n With the following sympify-algorithm:\n {}'.format(fittest_algo))
            file.write('\n\n')
            file.close()

        return

    # todo wenn ein gp lauf immer wieder dieselben Lösungen findet, verbiete einige Grundstrukturen.
    # todo ...oder andere einschränkungen. sin verbieten. wenn nichts besseres gefunden wird, weiter

    def file_pareto(self, pareto, root_path):
        """
        Save all the pareto efficient candidates to file
        """

        path_pareto = root_path / folder_info
        if not Path.is_dir(path_pareto):
            Path.mkdir(path_pareto)

        file = Path.open(path_pareto / file_pareto, 'w')

        for parsim, fit in sorted(list(pareto.items())):
            tree_meta = self.parsimony_best_meta[parsim]
            fitness = tree_meta['fitness_train']
            algo_sym = tree_meta['expr_raw']  # save raw version, not the sympified one
            #  todo automatically sympify pareto candidates?
            file.write('\nParsimony: \t{0} Fitness: \t{1} Expr: \t{2}'.format(str(parsim), str(fitness), str(algo_sym)))

        file.close()

    def file_pareto_latex(self, pareto, root_path):
        """
        Save all pareto entries as latex files
        - create trees from pareto expressions
        """

        # todo delete old entries? -> Maybe start visualisation separately?

        forest_grouped = []
        path_trees = make_dir(root_path / folder_trees)

        for parsim, meta in sorted(list(pareto.items())):
            expr_raw = meta['expr_raw']
            label_list = ast_convert_from_expr(expr_raw, build=True)
            tree = karoo_tree_from_labellist(label_list)
            tree = self.tree_enrich(tree, last_evolution='')

            # save the small forest inputs
            tikz_code = tree_viz_get_tex_forest(tree)

            file = Path.open(path_trees / 'tree-{}.tex'.format(str(meta['parsimony'])), 'w')
            file.write(tikz_code)
            file.close()

            # save a ready-to-use tex file with all pareto trees
            forest_grouped.append(
                'Pareto entry at parsimony {} with fitness {}.\n{}\n\\newpage'.format(parsim, meta['fitness_train'], tree_viz_get_tex_forest(tree)))  # todo this kind of is latex aswell

        latex_full_doc = latex_standalone_doc_forest(forest_grouped)
        file = Path.open(path_trees / '#all_trees.tex', 'w')
        file.write(latex_full_doc)
        file.close()

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Population specific                       |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def pop_eval_remaining(self):

        """
        Evaluate all trees in self.population_tmp_eval.
        This is the part of the population that could not be found in the dict.
        """

        count_fails = 0

        for tree_id in pop_iterate_trees(self.population_tmp_eval):

            tree = self.population_tmp_eval[tree_id]

            try:
                fitness_train = self.tree_eval_fitness_train(tree)
                tree = tree_set_fitness(tree, fitness_train)
                self.tree_meta_update(tree, fitness_train=fitness_train)
                self.pop_append(tree)  # todo the appendix appendaroo
            except Exception as ex:
                print_warning('ww', 'Error while getting meta Info: {}'.format(ex), print_type=self.print_type)
                count_fails += 1
                continue



        print_warning('ww', 'Evaluating {} trees in gen {} caused {} exceptions.'.format(len(self.population_tmp_eval), self.gen_id, count_fails), print_type=self.print_type)

        return

    def origin_tree_get(self):
        """
        Safely return an origin tree
        """
        if self.origin_exists():
            tree_origin = self.origin['tree']
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

        tree_ident = tree_get_ident(tree)

        self.tree_meta[tree_ident] = meta

    def pop_add_tree_midrun(self, tree):
        """
        Trying to add another tree to the current population
        - evaluate the tree
        - append to population
        - update gene_pool
        -
        """
        self.printpl('i', 'Trying to add tree mid-run...')

        # tree = self.tree_enrich(tree, last_evolution='p-sym')  # todo test added trees
        if self.tree_check_core_all(tree):

            tree = self.tree_enrich(tree, last_evolution='ps')
            parsimony = tree_eval_parsimony(tree, self.config['complexity_measure'], origin_tree=self.origin_tree_get())
            tree = tree_set_parsimony(tree, parsimony)
            self.tree_meta_update(tree, parsimony=parsimony)
            self.population_tmp_done.append(tree)  # todo, if we late insert a tree, will the pop-loop find it?

            self.parsimony_best_update()
            self.pareto_update_insert()

        return

    def pareto_update_insert(self):
        """
        update new entries in the pareto dict (part of the whole update process)
        Requires: self.parsimony_best_meta entries
        # todo this whole parsimony_best thing seems bad
        """
        sorted_parsimony_best = sorted(self.parsimony_best_meta.items(), key=lambda x: x[0])
        best_fit = next(iter(sorted_parsimony_best))[1]['fitness_train']  # [1] accesses the meta, ['fitness_train'] the fitness

        for key, meta in sorted_parsimony_best:  # tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
            # todo idea delete self.parsimony_best??
            # tree = self.population_tmp_done[tree_id]
            fitness = meta['fitness_train']
            parsim = meta['parsimony']
            pareto_improved = None

            if self.kernel.fitness_compare(fitness, best_fit):
                if self.pareto.get(parsim):
                    pareto_fit = self.pareto.get(parsim)
                    if self.kernel.fitness_compare(fitness, pareto_fit):
                        self.pareto[parsim] = meta
                        self.printpl('a', 'Pareto update at {}, with new fitness: {}. Old was: {}.'.format(parsim, fitness, best_fit))
                        pareto_improved = True
                else:
                    self.pareto[parsim] = meta
                    self.printpl('a', 'Pareto new entry at {} with fitness: {:4.2f}.'.format(parsim, fitness))
                    pareto_improved = True
                best_fit = fitness

            if pareto_improved:
                expr_raw = meta['expr_raw']
                tree = karoo_tree_from_expr(expr_raw)
                sym_tree = tree_evolve_reduce(tree, completely=True)
                if tree_get_expr_raw(sym_tree, node_id=root_id) != tree_get_expr_raw(tree, node_id=root_id):
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
        last_pareto_fit = next(iter(sorted_pareto))[1]
        for parsim, fitness in sorted_pareto[1:]:
            if self.kernel.fitness_compare(fitness, last_pareto_fit):
                last_pareto_fit = fitness
            else:
                self.pareto.pop(parsim)
                self.printpl('aa', 'Pareto entry at {} got obsolete. Its fitness: {} was surpassed by simpler entry with fitness: {}.'.format(parsim, last_pareto_fit, fitness))
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

    def parsimony_best_update(self):
        """
        Updates a list with the best candidates for each parsimony.
        Do not confuse with pareto-entries
        """

        for tree_id in pop_iterate_trees(self.population_tmp_done):  # todo, these are not ordered in parsimony nor fitness
            tree = self.population_tmp_done[tree_id]

            parsim = tree_get_parsimony(tree)  # todo random is eval parsimony used correctly everywhere?
            fitness_train = tree_get_fitness(tree)

            # 3. is the tree better than the current best at this parsimony dim_y?
            if parsim in self.parsimony_best_meta:
                comp_fit = self.parsimony_best_meta[parsim]['fitness_train']
                if self.kernel.fitness_compare(fitness_train, comp_fit):
                    self.printpl('aa', 'Found a better candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                    self.parsimony_best_meta[parsim] = tree_get_meta(tree)
            else:
                self.printpl('aa', 'Found a new candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                self.parsimony_best_meta[parsim] = tree_get_meta(tree)

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   What happens in a Generation              |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gen_reset_parameters(self):
        """
        Sets the parameters for the generation
        - reset population_tmp_done
        - Linearly increase threshold for parsimony
        """

        self.gen_id += 1
        self.time_genstart = time.perf_counter()
        self.debug_warnings = {}
        self.population_tmp_done = ['DELETE THIS']  # todo pop-delete initialise population_tmp_done to host the next generation
        self.parsimony_tmp = max(1 / min(self.gen_id, self.config['gen_num_max_parsimony']) * self.parsimony_max, self.parsimony_max)
        self.population_tmp_eval = ['DELETE THIS']  # todo pop-delete

        return

    # random todo, save last sympify expr in debug info...
    def pop_reproduce(self, repro_rate):

        """
        A single Tree from the prior generation is copied without mutation
        """

        for _ in range(repro_rate):  # quantity of Trees to be copied without mutation
            tourn_winner = self.pop_selection_tournament(self.tourn_size)
            self.pop_append(tourn_winner, last_evolution='r1')  # i know, tests are not necessary...

        return

    def pop_reproduce_olymp(self, repro_rate):

        """
        Copy an entry from the pareto candidates into the population
        """

        for _ in range(repro_rate):  # quantity of Trees to be copied without mutation
            if self.parsimony_best_meta:
                meta = np.random.choice(list(self.parsimony_best_meta.values()))
                # expr_sym = meta['expr_sym']; print('sym', expr_sym)
                expr_raw = meta['expr_raw']
                # label_list = ast_convert_from_expr(expr_sym, build=True); print('label_list', label_list)
                label_list = ast_convert_from_expr(expr_raw, build=True)
                p_tree = Plagih_Tree(label_list)
                olymp_winner = p_tree.get_uninstanced_tree()
                self.pop_append(olymp_winner, last_evolution='r(oly)')

        return

    def pop_reproduce_reduce(self, repro_rate):

        """
        copy a tree from the last population and sympify parts of it
        """

        for _ in range(repro_rate):
            tree = self.pop_selection_tournament(self.tourn_size)
            tree = tree_evolve_reduce(tree, completely=False)
            self.pop_append(tree, last_evolution='point')

        return

    def pop_mutate_point(self, repro_rate):

        """
        One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """

        for _ in range(repro_rate):  # quantity of Trees to be generated through mutation
            tree = self.pop_selection_tournament(self.tourn_size)
            tree = tree_evolve_mutate_point(tree, self.func_array, self.variables_dict)
            self.pop_append(tree, last_evolution='point')

        return

    def pop_mutate_filter(self, repro_rate):
        """

        """

        for _ in range(repro_rate):
            tree = self.pop_selection_tournament(self.tourn_size)
            try:
                new_tree = tree_evolve_mutate_filter_one(tree)
                if len(new_tree) > 1:
                    self.pop_append(new_tree, last_evolution='filter')
            except Exception as ex:
                self.printpl('www', 'Tree in mutate filter could not be changed, {}'.format(ex))

        return

    def pop_random_from_origin(self, repro_rate):
        """

        """

        if self.origin_exists():
            tree_origin = self.origin['tree'].copy()
            for _ in range(repro_rate):
                tree = tree_evolve_branch_multiple(tree_origin, self.parsimony_max, self.variables_dict, self.func_array)
                self.pop_append(tree, last_evolution='new(o)')

        return

    def pop_random_from_scratch(self, repro_rate):
        """
        todo make available half ramped
        """

        for i in range(repro_rate):
            max_nodes = np.random.randint(self.config['tree_branch_nodes_base'], self.parsimony_max)  # todo 3 auslagern und testen ob 3 entstehen kann
            label_list, arity_list = invent_label_list_nodes_grow(self.output_xtype, max_nodes, self.variables_dict, self.func_array)
            p_tree = Plagih_Tree(label_list)
            tree = p_tree.get_uninstanced_tree()
            tree = tree_set_id(tree, i)

            self.pop_append(tree, last_evolution='0s')

        return

    def pop_mutate_branch(self, repro_rate, last_evolution='mb'):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """

        for _ in range(repro_rate):  # quantity of Trees to be generated through mutation

            tree = self.pop_selection_tournament(self.tourn_size)  # perform tournament selection for each mutation
            node_ids = tree_get_mutatable_nodes(tree, no_root=True)
            node = np.random.choice(node_ids)
            branch_nodes_ids = tree_get_branch(tree, node, karoo=True)  # select point of mutation and all nodes beneath [6, 9, 10]
            if self.config['tree_growth'] == 'v1':
                tree = tree_evolve_insert_branch_v1(tree, branch_nodes_ids, self.variables_dict, self.func_array,
                                                    depth_max=self.config['tree_depth_max'],
                                                    depth_min=self.config['tree_depth_min'],
                                                    depth_goal=self.config['tree_depth_base'])
            elif self.config['tree_growth'] == 'v2':
                tree = tree_evolve_insert_branch_v2(tree, branch_nodes_ids, self.variables_dict, self.func_array,
                                                    self.config['tree_branch_nodes_base'])
            else:
                raise Exception('Tree growth version not known')

            self.pop_append(tree, last_evolution=last_evolution)

        return

    def pop_crossover_branch(self, repro_rate):
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
            left_tree = self.pop_selection_tournament(self.tourn_size)  # perform tournament selection for 'a_parent'
            right_tree = self.pop_selection_tournament(self.tourn_size)  # perform tournament selection for 'b_parent'

            force_convert = False

            # 2. search nodes for left and right that can be exchanged. convert_needed
            left_id, right_id, success = self.tree_try_get_swapids(left_tree, right_tree)
            if not success:
                right_id, left_id, success = self.tree_try_get_swapids(right_tree, left_tree)
            if not success:
                force_convert = True

            left_ids, left_labels, left_aritys = tree_get_branch_lax(left_tree, left_id)
            right_ids, right_labels, right_aritys = tree_get_branch_lax(right_tree, right_id)

            if force_convert:
                self.printpl('w', 'Crossover conversion between trees forced. \n{}\n{}'.format(left_tree, right_tree))
                left_xtype = xtype_get(tree_node_get_label(left_tree, left_id), self.variables_dict)
                conv_to_left, conv_to_right = xtype_get_converters(left_xtype)
                right_labels.insert(0, conv_to_left)
                right_aritys.insert(0, 1)
                left_labels.insert(0, conv_to_right)
                left_aritys.insert(0, 1)

            left_core = core_from_labels(left_labels, left_aritys)
            right_core = core_from_labels(right_labels, right_aritys)

            left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
            left_offspring = tree_prune_depth(left_offspring, self.config['tree_depth_max'], self.variables_dict)
            self.pop_append(left_offspring, last_evolution='cross')

            right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
            right_offspring = tree_prune_depth(right_offspring, self.config['tree_depth_max'], self.variables_dict)
            self.pop_append(right_offspring, last_evolution='cross')

        return

    def output_get_xtype(self):
        """
        Return the xtype of the action we want to create.
        """
        action_type_one = next(iter(self.action_dict.values()))
        if action_type_one == 'float':
            xtype = '2f'
        elif action_type_one == 'bool':
            xtype = '2b'
        else:
            raise
        return xtype

    # todo sarsa policy performance
    # todo decision plot (wie bei sarsa) für alle bei MTC

    def tree_enrich(self, tree, last_evolution=''):
        """
        The np-tree needs more information than only the expression.
        -> set modifyable nodes (mandatory)
        -> round all constants
        -> try to normalize exponents ('**'). sfeh, not really working.
        -> set last evolution (for analysing gp operators. e.g. if no good trees originate from crossover, something might be wrong)
        -> set xtype for all nodes. todo make this when the node is added
        """

        if tree is None:
            print_e('Tree from last_evolution: {} failed.'.format(last_evolution))
            raise Exception('Tree is None')
        else:
            if self.origin_exists():
                tree = tree_set_modifyable_nodes(tree, self.origin['tree'])
            else:
                tree = tree_set_modifyable_nodes_true(tree)

            tree = tree_round_constants(tree, self.config['float_accuracy'], karoo=True)
            tree = tree_normalize_exponentiation(tree)
            tree = tree_set_history(tree, last_evolution)
            tree = tree_set_xtypes(tree, self.variables_dict)  # delete if this is made separately

        return tree

    def tree_check_core_all(self, tree):
        """
        Performs all checks that we currently have
        # todo do not use this if trees are sefely generated
        # todo check meta values in separate method? update those aswell?
        """

        if not tree_check_children(tree):
            print_e('Tree is not consistent:\n{}'.format(tree))
            tree_works = False
        elif not tree_check_typed(tree, self.variables_dict):
            print_e('Tree children xtypes are not correct:\n{}'.format(tree_labels(tree)))
            tree_works = False
        elif tree_node_get_arity(tree, root_id) == 0:
            print_warning('w', 'Tree is only a root node')
            tree_works = False
        # elif tree_get_meta(tree):
        #     print_warning('w', 'Could not get meta from tree.')
        #     tree_works = False
        else:
            tree_works = True

        return tree_works

    def pop_append_late(self, tree):
        """
        Everything is done, as we filled all the other information in pop_append()
        - enumerate
        """
        if tree_get_id(tree) != '':
            print('todo? delete this', tree, tree_get_id(tree))
        else:
            tree = tree_set_id(tree, len(self.population_tmp_done))
        self.population_tmp_done.append(tree)
        return

    def pop_base_transfer(self):
        """
        Copy the genepool of a gen
        """
        self.population_base = ['Population Selection in Generation {}.'.format(str(self.gen_id))]  # todo pop_delete

        for tree_id in pop_iterate_trees(self.population_tmp_done):
            # tree = self.population_tmp_done[tree_id]
            tree_copy = pop_tree_copy(self.population_tmp_done, tree_id)  # what about entry #1?
            tree_copy = tree_set_id(tree_copy, tree_id)  # todo +1`??
            self.population_base.append(tree_copy)
            # todo could enum trees here again

        return

    def pop_append(self, tree, last_evolution=''):
        """
        Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the tree is refurbished.
        todo: if trees are 100% safely created, tree_check_all() must not be used. Useful when trying out new gp-operators.
        - Enrich the raw tree for the next generation
        - check if the tree is actually valid
        ->
        """

        tree = self.tree_enrich(tree, last_evolution=last_evolution)

        if self.tree_check_core_all(tree):
            tree_ident = tree_get_ident(tree)

            if tree_ident in self.tree_meta:

                tree_meta = self.tree_meta[tree_ident]
                tree = tree_set_meta(tree, tree_meta)
                tree = tree_set_id(tree, len(self.population_tmp_done))  # todo test and find better solution
                self.population_tmp_done.append(tree)
            else:
                parsimony = tree_eval_parsimony(tree, self.config['complexity_measure'], origin_tree=self.origin_tree_get())
                if parsimony <= self.parsimony_max:
                    tree = tree_set_parsimony(tree, parsimony)
                    tree = tree_set_fitness(tree, '')
                    tree = tree_set_id(tree, '')  # todo test and find better solution
                    self.population_tmp_eval.append(tree)
                else:
                    print_warning('w', 'Tree was too complex!')

        return

    def gen_finalize(self):

        """
        From raw population_tmp_done
        - Evaluate leftover

        """

        # self.population_tmp_done = pop_enum_trees(self.population_tmp_done)  # todo not neccessary?

        # gene_pool = self.pop_genepool_create()
        self.pop_eval_remaining()

        self.parsimony_best_update()
        self.pareto_update()  # todo todo save sympified tree

        self.pop_base_transfer()
        self.pop_analyze()
        file_population_karoo(self.population_tmp_done, 'tmp', self.root_dir, self.gen_id)

        self.monitoring_dict['total_found_trees'][self.gen_id] = len(self.tree_meta)
        self.print_g('gg', 'Monitoring: Created {}/{} unique trees in generation {}. Gen-time: {:4.2f}'.format(
            len(self.population_tmp_done),
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
        # todo idea layerwise-mutations not only for layer0?

        best_id = None
        best_fitness = None

        for n in range(tourn_size):

            tree_id = pop_tree_choose(self.population_base)
            tree = self.population_base[tree_id]
            print('last--tree\n', tree)
            fitness = tree_get_fitness(tree, precision=self.config['precision'])

            if self.kernel.fitness_compare(fitness, best_fitness, mode='better'):
                best_id = tree_id
                best_fitness = fitness

        tourn_winner = copy.deepcopy(self.population_base[best_id])
        tourn_winner = tree_set_meta_wipe(tourn_winner)

        return tourn_winner

    def tree_try_get_swapids(self, a_tree, b_tree):
        """
        Returns two branches (node ids) that can be replaced and a converter (if needed)

        # same position & node, same node, same type, reversed type | convert_needed type
        """

        # choose a node from parent a
        a_ids = tree_get_mutatable_nodes(a_tree, no_root=True)
        a_id = np.random.choice(a_ids)
        a_node_label = tree_node_get_label(a_tree, a_id)
        a_xtype = xtype_get(a_node_label, self.variables_dict)

        # create a list from parent b with same xtype
        b_node_ids = tree_get_mutatable_nodes(b_tree, no_root=True)
        b_sametype_ids = b_node_ids[:]
        for b_id in b_node_ids:
            b_label = tree_node_get_label(b_tree, b_id)
            b_xtype = xtype_get(b_label, self.variables_dict)
            if not xtype_equi_outcome(b_xtype, a_xtype):
                b_sametype_ids.remove(b_id)  # remove one-by-one false partner nodes.

        if b_sametype_ids:  # if entries were found, choose one. we are custom_done
            b_id = np.random.choice(b_sametype_ids)
            success = True
        else:
            b_id = np.random.choice(b_node_ids)
            success = False

        return a_id, b_id, success

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Work with trees                           |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def load_origin_tree(self, origin_tree_file_path=None, label_list=None, modify_list=None):
        """
        This loads the 'origin' and evaluates it
        Two loading options:
            - root_dir to csv with tree (outdated)
            - an array with labels ['+','1','observation0']. optional, the permanent nodes as separate array

        returns: tree
        """

        # Check if the user provided an origin
        if origin_tree_file_path:
            tree = tree_single_from_csv(origin_tree_file_path)
        elif label_list:
            p_tree = Plagih_Tree(label_list, modify_list=modify_list)
            tree = p_tree.get_uninstanced_tree()
        else:
            print_warning('w', 'No origin provided. starting from scratch with random generation?')
            tree = None

        expr_raw = tree_get_expr_raw(tree, node_id=root_id)

        try:
            expr_sym = expr_sympify(expr_raw=expr_raw)
        except:
            raise Exception('Your origin algorithm could not be sympified. Aborting.')

        if expr_sym != expr_raw:  # todo check if both expressions have the same structure, not if they are equal. try eval=False?
            print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}'.format(expr_raw, expr_sym))

        self.origin = {'tree': tree, 'expr_raw': expr_raw, 'expr_sym': expr_sym, 'parsimony': 0}
        try:
            fitness_train = self.tree_eval_fitness_train(tree)
        except Exception:
            raise Exception('Your origin algorithm already caused an exception!')
        self.origin['fitness_train'] = fitness_train

        self.parsimony_best_meta[0] = self.origin
        self.pareto[0] = copy.deepcopy(self.origin)

        self.print_g('gg', 'Loading origin, fitness {}. Time: {:4.2f}s'.format(fitness_train, time.perf_counter() - self.time_start))

        return

    def tree_eval_fitness_train(self, tree):
        """
        Evaluating the fitness of a tree.
        - extract the expression the tree is holding
        - sympify the expression
        - (if sympify fails, evaluating does not make sense! Check sympify errors)
        """

        expr_raw = tree_get_expr_raw(tree, node_id=root_id)

        try:
            expr_sym = expr_sympify(expr_raw=expr_raw)
        except Exception as ex:
            raise Exception('Expr could not be sympified: {}. Ex: {}'.format(expr_raw, ex))

        fitness_train = eval_tf(expr_sym, self.data_train, self.eval_parameters)['fitness']

        if not self.check_value_is_real(fitness_train):
            raise Exception('fitness_train is not a real number: {}'.format(fitness_train))

        return fitness_train

    def remove_this_tree(self):
        self.printpl('ww', 'This still is a sfeh')
        """
        If a tree makes problems, delete it somehow.
        - set parsimony very high?
        todo
        """

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

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def get_pareto_plot_values(self):
        """
        todo i think there is a more beautiful solution?
        """
        tuples = []
        for key in sorted(self.pareto):
            tuples.append((key, self.pareto[key]['fitness_train']))
        return tuples

    def auto_plots(self, path):
        """
        Make all plots
        """
        import warnings
        warnings.filterwarnings('error')

        path_plots = make_dir(path / folder_plots)

        if self.monitor_dict['gen_fitness_average'] == 'y':
            data_tuples = sorted(list(self.monitoring_dict['fitness_average'].items()))
            self.plot_end(data_tuples, path_plots, plt_title='average fitness', plt_y_label='fitness', linestyle='-', min_left=data_tuples[0][0])

        if self.monitor_dict['population_tmp_done-size'] == 'y':
            data_tuples = sorted(list(self.monitoring_dict['population_tmp_done-size'].items()))
            self.plot_end(data_tuples, path_plots, plt_title='genepool size', plt_y_label='amount', linestyle='', min_left=data_tuples[0][0])

        data_tuples = sorted(list(self.monitoring_dict['complexity_average'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='average tree complexity', plt_y_label='#nodes', linestyle='-', min_left=data_tuples[0][0])

        data_tuples = sorted(list(self.monitoring_dict['total_found_trees'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='number of created trees', plt_y_label='amount', linestyle='', min_left=data_tuples[0][0])

        data_tuples = self.get_pareto_plot_values()
        self.plot_end(data_tuples, path_plots, plt_title='pareto dominant candidates', plt_x_label='parsimony', plt_y_label='fitness', linestyle='dashed',
                      step_where='post', max_right=self.parsimony_max, beyond_lines=True, save_tikz=True)  # todo beyond_lines

        data_tuples = sorted(list(self.monitoring_dict['best_candidate'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='best candidate', plt_x_label='generation', plt_y_label='fitness', linestyle='dashed',
                      step_where='post')

        # todo https://github.com/linkedin/naarad/issues/114 UserWarning: Attempting to set identical bottom==top results

        return

    def check_value_is_real(self, fitness):
        """
        Returns bool value if we can use the calculated fitness
        Fitness values might evaluate to weird stuff
        e.g. 'nan' after dividing by zero or (inf) after 20**1234
        nan: fitness == fitness -> False
        inf: fitness is not float('inf') -> False
        """
        return fitness == fitness and fitness is not float('inf')

    def pop_analyze(self):
        """
        Analysing this generation
        - amount of trees
        - fittest tree
        - average fitness
        - average tree complexity
        # todo len(self.population_tmp_done) is used too often
        """

        # How many survived in the selection?
        self.monitoring_dict['population_tmp_done-size'][int(self.gen_id)] = len(self.population_tmp_done)

        if len(self.population_tmp_done) <= 0:
            self.terminate_run(self.root_dir)

        # Find the fittest tree, also average fitness

        pop_best_fitness = tree_get_fitness(self.population_tmp_done[FIRST_TREE])

        fitness_train_sum = 0

        # dominator_count = 0

        for tree_id in pop_iterate_trees(self.population_tmp_done):
            tree = self.population_tmp_done[tree_id]
            fitness = tree_get_fitness(tree)
            if self.check_value_is_real(fitness):  # todo take care of this earlier
                fitness_train_sum += fitness  # for fitness average
                if self.kernel.fitness_compare(fitness, pop_best_fitness):
                    pop_best_fitness = fitness

        if self.best_fitness is None:
            self.best_fitness = pop_best_fitness
        else:
            if self.best_fitness < pop_best_fitness:
                self.best_fitness = pop_best_fitness

        #     # Count dominators # todo why
        #     if self.origin_exists():
        #         if self.kernel.fitness_compare(tree_meta['fitness_train'], self.origin['fitness_train']):
        #             dominator_count += 1
        #     else:
        #         pass
        # self.print_g('gg', 'Generation {}, {} Candidates were better than the origin.'.format(self.self.gen_id, dominator_count))

        average_fitness = fitness_train_sum / max(len(self.population_tmp_done), 1)
        self.monitoring_dict['fitness_average'][self.gen_id] = average_fitness
        self.monitoring_dict['best_candidate'][self.gen_id] = self.best_fitness

        # Tree complexity
        complexity_sum = 0
        for tree_id in pop_iterate_trees(self.population_tmp_done):
            tree = self.population_tmp_done[tree_id]
            complexity_sum += len(tree_nodes_get_ids(tree, karoo=True))
        avg_complexity = complexity_sum / len(self.population_tmp_done)
        self.monitoring_dict['complexity_average'][self.gen_id] = avg_complexity

        return

    def terminate_run(self, path):
        """
        Program is done after writing all files one last time.
        :param path:
        :return:
        """
        self.file_save_files(path)
        self.auto_plots(path)
        self.print_g('gg', ' Terminating. \tTotal time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        # sys.exit()  # todo sys.exit prevents further stuff

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to print_type output information     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def plot_end(self, data_2d, path,
                 plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='', yscale='linear', step_where='', plt_xparam='',
                 linestyle='None', min_left=None, max_right=None, beyond_lines=False, save_tikz=False):
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
        :param min_left: Smallest left value
        :param max_right: E. g. if max_parsimony is 100 -> show complete width, even if entries only go to 40
        :param beyond_lines: in step plots, draw the line further to the left and right
        :param save_tikz: Also save the plot as tikzpicture (for Latex)
        :return:
        """

        if len(data_2d) == 0:
            print_e('Plotting empty array is not possible! Data={}'.format(data_2d))
            return

        # todo sklearn can split this nicer split data in x and y values
        x, y = [], []
        for a, b in data_2d:
            x.append(a)
            y.append(b)

        # bottom, top = plt.ylim()
        # left, right = plt.xlim()

        top, bottom, left, right = max(y), min(y), min(x), max(x)
        if min_left:
            left = min_left
        new_top = (top - min(bottom, 0)) * 1.05  # todo beautify plots...
        if max_right:
            right = max(right, max_right)
        new_right = right * 1.05

        if beyond_lines:
            x = [x[0]] + x + [new_right + 1]
            y = [new_top + 1] + y + [y[-1]]

        if step_where:
            plt.step(x, y, plt_xparam, linestyle=linestyle, marker='.', label=plt_curve_label, where=step_where)
        else:
            plt.plot(x, y, plt_xparam, linestyle=linestyle, marker='.', label=plt_curve_label)

        # let it start at (0,0) but +5% margin to the top and right
        plt.yscale(yscale)
        plt.ylim(min(bottom, 0), new_top)
        plt.xlim(min(left, 0), new_right)
        plt.margins(x=0, y=0)

        plt.xlabel(plt_x_label)
        plt.ylabel(plt_y_label)
        plt.title(plt_title)

        # plt.legend()

        plt.savefig(path / '{}.jpg'.format(plt_title))
        if save_tikz:
            try:
                tikzplotlib.save(path / '{}.tex'.format(plt_title))
            except Exception as ex:
                print_e('Need to install tikzplotlib? matplotlib2tikz is outdated. Exception:\n{}'.format(ex))

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


def pop_util_copy(population_x, title):
    """
    Copy one population to another.
    """
    population_y = [title]  # an empty list stores a copy of the prior generation

    for tree_id in range(1, len(population_x)):  # increment through each Tree in the current population
        tree_copy = pop_tree_copy(population_x, tree_id)  # copy each array in the current population
        population_y.append(tree_copy)  # add each copied Tree to the new population list

    return population_y


def pop_enum_trees(population):
    """
    outsourced enumeration of trees in a population
    """
    for tree_id in range(FIRST_TREE, len(population)):  #
        tree = population[tree_id]
        population[tree_id] = tree_set_id(tree, tree_id)
    return population


def pop_iterate_trees(population):
    """
    Ich schwöre welcher Hurensohn hat diese Kopfzeile gemacht?
    """
    tree_ids = []
    for tree_id in range(FIRST_TREE, len(population)):  #
        tree_ids.append(tree_id)
    return tree_ids


def file_population_karoo(population, pop_name, path, gen_id):
    """
    Save population_* to disk.

    """

    pop_path = make_dir(path / folder_info)

    file_path = pop_path / 'population_{}.csv'.format(str(pop_name))

    # todo function to tree_ and append each tree
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


def write_config_file(path, config, gen_id, kernel, date_time):
    """
    write the parameters to a file
    """

    path_config = make_dir(path / folder_info)

    file = Path.open(path_config / file_config, 'a')
    file.write('This config is not complete, sfeh!')
    file.write('\n launched: {}'.format(date_time))
    file.write('\n kernel: {}'.format(kernel))
    file.write('\n precision: {}\n'.format(config['precision']))
    file.write('\n tree depth max: ' + str(config['tree_depth_max']))
    file.write('\n')
    file.write('\n tournament size: ' + str(config['tourn_size']))
    file.write('\n population: ' + str(config['pop_max']))
    file.write('\n number of generations: ' + str(gen_id))
    file.write('\n\n')
    file.close()
    return


def load_funcarray_from_csv(op_csv_path):
    """
    Load all operators ready-to-use from a file
    """

    functions = np.loadtxt(op_csv_path, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
    # Part 3.5: Split the functions in 5 types

    # rows are the function types (f2f)
    # columns are the arity
    func_array = [[[], [], [], []],
                  [[], [], [], []],
                  [[], [], [], []],
                  [[], [], [], []],
                  [[], [], [], []]]

    for fun in functions:
        label = fun[0]
        arity = op[label]['arity']  # arity = int(fun[1])
        xtype = op[label]['xtype']

        if xtype == 'f2f':
            func_array[f2f][arity].append(label)
        elif xtype == 'f2b':
            func_array[f2b][arity].append(label)
        elif xtype == 'b2b':
            func_array[b2b][arity].append(label)
        elif xtype == 'b2f':
            func_array[b2f][arity].append(label)
        elif xtype == 'b2f2f':
            func_array[b2f2f][arity].append(label)

    return func_array


def tree_get_ident(tree):
    """
    What is used as identificator for a tree...
    - hash(expr_raw)
    """
    expr_raw = tree_get_expr_raw(tree, node_id=root_id)
    tree_ident = hash(expr_raw)
    return tree_ident


def tree_get_parsimony(tree):
    parsimony = tree[T_parsimony][1]
    if parsimony != '':
        parsimony = float(parsimony)
    return parsimony


def tree_eval_parsimony(tree, parsimony_distance, origin_tree=None):
    """
    parsimony_distance: compute the chosen distance by the user.

    """

    if parsimony_distance == 'total_count_nodes':  # number of nodes
        return tree_get_last_nodeid(tree)  # returns the number of nodes
    elif parsimony_distance == 'total_tree_depth':
        return 0

    if parsimony_distance == 'ted':
        return tree_parsimony_ted(tree, origin_tree)
    elif parsimony_distance == 'rel_ari_1':  # Does this work?
        return tree_parsimony_relari(tree, origin_tree)
    else:
        print_e('Complexity measurement not available: {}'.format(parsimony_distance))
        raise
