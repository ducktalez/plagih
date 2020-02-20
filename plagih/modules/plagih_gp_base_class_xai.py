"""
Explaination:
> 'f2f', 'b2b', etc.: my personal silly naming. f = float, b = bool.
   f2b means float to boolean, e. g. '<' takes 'float' and returns a 'bool'
> node_modify (0 or 1), specifies, whether this node is supposed to be




Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""

import sys
import sklearn.metrics as skm
from datetime import datetime
from plagih.modules.plagih_tree import *
import time
from plagih.modules.plagih_eval import *
from plagih.modules.file_interaction import *

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
        self.restart_vers = 'v0.7'

        # init values with dummies (just to have all self values here for overview)
        self.tree_meta = {}  # LUT to save all expressions with their corresponding meta data (e.g. fitness). needs memory.
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
        self.xype_func_dict = {'f2f': [], 'f2b': [], 'b2b': [], 'b2f': [], 'b2f2f': [],
                               '2b': [], '2f': [],
                               'b2': [], 'f2': []}  # todo, is this necessary? could be deleted

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
        self.monitoring_dict = {'genepool_size': {},
                                'fitness_average': {},
                                'best_candidate': {},
                                'total_found_trees': {},
                                'complexity_average': {}}

        self.file_directories_create(self.config['root_dir'])
        self.print_g('ggg', 'Init. Time: {:4.2f}s'.format(time.perf_counter() - self.time_start))

        return

    def plagih_gp_run(self):
        """
        regular plagih-config run
        """

        path_backup = self.root_dir / file_backup_pickle
        if Path.is_file(path_backup) and not self.config['force_new_run']:
            self.print_g('g', 'Restarting old run now...')
            try:
                self.run_load_backup(path_backup)
            except Exception as ex:
                print_warning('w', 'Even though a backup exists for this run, it could not be loaded because of: {}\nStarting a new run.'.format(ex))
                # todo delete old plots, append to old config,

        if self.gen_id == 0:
            self.main_generation_first()

        write_config_file(self.root_dir, self.config, self.gen_id, self.kernel, self.datetime)

        self.main_generation_loop()
        self.terminate_run(self.root_dir)

    def file_directories_create(self, path_cwd):
        """
        Create all files that will be saved after all
        """

        # self.datetime = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.datetime = datetime.now().strftime('%H%M%S')

        self.root_dir = path_cwd / folder_runs / '{}'.format(self.config['name'])
        # self.root_dir = cwd / 'runs' / '{}{}'.format(self.config['name'], self.datetime)

        if not Path.is_dir(self.root_dir):
            Path.mkdir(self.root_dir)

        return

    def main_generation_first(self):
        """
        Everything that needs to be custom_done for the first generation
        - Extracts "origin Tree" from file
        - Creates all other trees: origin tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """
        self.print_g('gg', 'Preparing to evolve first Generation. Gen {}.'.format(self.gen_id))
        self.gen_reset_parameters()

        if self.origin_exists():
            self.pop_first_create_from_origin(self.config['pop_max'], self.origin['tree'])
        else:
            self.gen_random_from_scratch(self.config['pop_max'])

        self.gen_finalize()
        file_population_karoo(self.population_base, '1_first', self.root_dir, self.gen_id)  # first gen only

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

    def main_generation_loop(self):
        """
        Creates all new Generations.
        - adjust parameters for this generation (parsimony threshold)
        - Create a gene pool (kick out too complex candidates)
        """

        gp_list = [('repro one', self.gen_reproduce),
                   ('repro pareto', self.gen_reproduce_olymp),
                   ('repro reduced one', self.gen_reproduce_reduce),
                   ('point mutate function', self.gen_mutate_point),
                   ('filter floats', self.gen_mutate_filter),
                   ('branch mutate insert', self.gen_mutate_branch),
                   ('crossover branches', self.gen_crossover_branch),
                   ('random from origin', self.gen_random_from_origin),
                   ('random from scratch', self.gen_random_from_scratch)]

        while self.run_continues():

            self.gen_reset_parameters()

            for name, gp_function in gp_list:
                time_evolve = time.perf_counter()
                self.print_g('ggg', '-->Evolve: ({}) took: {:4.2f}sec.'.format(name, time.perf_counter() - time_evolve))

                evolve_num = int(self.evolve_rates[name] * self.config['pop_max'])
                gp_function(evolve_num)

            self.gen_finalize()

            self.periodical_procedures()

            self.print_g('ggg', 'Generation took a total time of: {:4.2f}'.format(time.perf_counter() - self.time_genstart))
        else:
            printez('g', 'Done after Generation {}.'.format(self.gen_id), print_type=self.print_type)
            return

    def origin_exists(self):
        if self.origin is not None:
            return True
        else:
            return False

    def periodical_procedures(self):
        """
        Every few generations, save the current status
        """
        if self.config['overwrite periodic files']:
            tmp_path = self.root_dir
        else:
            tmp_path = self.root_dir / folder_steps / 'Gen-{}'.format(self.gen_id)

        if not Path.is_dir(tmp_path):
            Path.mkdir(tmp_path)

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
        self.file_conclusion(root_path, datetime=self.datetime)
        self.file_pareto(self.pareto, root_path)
        file_population_karoo(self.population_base, pop_name, root_path, self.gen_id)  # save the final generation of Trees to disk

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

        return

    def activate_operators(self, func_array):
        """
        operators were loaded already and need to be set in the gp run
        """
        self.func_array = func_array
        return

    def run_load_backup(self, path_backup):
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

        self.restart_count = run_data['self.restart_count']  # counting how often this was restarted
        self.gen_id = run_data['self.gen_id']
        self.parsimony_best_meta = run_data['self.parsimony_best_meta']
        self.pareto = run_data['self.pareto']
        self.population_base = run_data['self.population_base']
        self.monitoring_dict = run_data['self.monitoring_dict']

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

    def file_conclusion(self, path, datetime=None):

        """
        write the performance of the config to disc
        """
        path_conclusion = path / folder_info
        if not Path.is_dir(path_conclusion):
            Path.mkdir(path_conclusion)

        file = Path.open(path_conclusion / file_conclusion, 'w')
        file.write('Plagih GP\n launched: {}'.format(str(datetime)))

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
            for enum, todo in enumerate(result['result']):
                if todo != todo or todo == float('inf'):
                    no_fault = True
                    # todo we handle this now...
                    result['result'][enum] = 1

            if no_fault:
                kernel_result = self.kernel.conclusion_info(result, fitness_control_best)
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

    def file_pareto(self, pareto, root_path):
        """
        Save all the pareto efficient candidates to file
        """

        path_pareto = root_path / folder_info
        if not Path.is_dir(path_pareto):
            Path.mkdir(path_pareto)

        file = Path.open(path_pareto / file_pareto, 'w')

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

    def pop_genepool_create(self, population):

        """
        Create the gene pool
        - Add a candidate if its parsimony is within the threshold

        """
        self.print_g('gggg', 'Gene Pool for Generation: {}...'.format(self.gen_id))
        dominator_count = 0
        gene_pool = {}
        fail_count = [0, 0]

        for tree_num in range(0, len(population)):  # todo was 1 not from 0
            tree = population[tree_num]
            try:
                if self.origin_exists():
                    tree_ident, tree_meta = self.tree_get_meta(tree, tree_origin=self.origin['tree'])
                else:
                    tree_ident, tree_meta = self.tree_get_meta(tree)
            except Exception as ex:
                printez('www', 'Error while getting meta Info: {}'.format(ex), self.print_type)
                fail_count[0] += 1
                continue

            if tree_meta['fitness_train'] != tree_meta['fitness_train'] or tree_meta['fitness_train'] == float('inf'):
                fail_count[1] += 1
                continue

            population[tree_num] = tree_set_fitness(tree, tree_meta['fitness_train'], precision=self.config['precision'])
            gene_pool[tree_num] = tree_meta

            if self.origin_exists():
                if self.kernel.fitness_compare(tree_meta['fitness_train'], self.origin['fitness_train']):
                    dominator_count += 1
            else:
                pass

        self.print_g('gg', 'Generation {}, {} Candidates were better than the origin.'.format(self.gen_id, dominator_count))

        return gene_pool, population

    def pop_pareto_update(self):
        """
        Builds up the pareto front
        1. adds new entries from parsimony_best_meta
        2. deletes entries that are now obsolete
        """

        sorted_parsimony_best = sorted(self.parsimony_best_meta.items(), key=lambda x: x[0])
        best_fit = next(iter(sorted_parsimony_best))[1]['fitness_train']  # [1] accesses the meta, ['fitness_train'] the fitness

        for parsim, meta in sorted_parsimony_best:
            fitness = meta['fitness_train']

            if self.kernel.fitness_compare(fitness, best_fit):
                if self.pareto.get(parsim):
                    pareto_fit = self.pareto.get(parsim)
                    if self.kernel.fitness_compare(fitness, pareto_fit):
                        self.pareto[parsim] = fitness
                        self.printpl('a', 'Pareto update at {}, with new fitness: {}. Old was: {}.'.format(parsim, fitness, best_fit))

                else:
                    self.pareto[parsim] = fitness
                    self.printpl('a', 'Pareto new entry at {} with fitness: {:4.2f}.'.format(parsim, fitness))
                best_fit = fitness

        sorted_pareto = sorted(self.pareto.items(), key=lambda x: x[0])
        last_pareto_fit = next(iter(sorted_pareto))[1]
        for parsim, fitness in sorted_pareto:
            if self.kernel.fitness_compare(last_pareto_fit, fitness):
                self.pareto.pop(parsim)
                self.printpl('aa', 'Pareto entry at {} got obsolete. Its fitness: {} was surpassed by simpler entry with fitness: {}.'.format(parsim, last_pareto_fit, fitness))
                last_pareto_fit = fitness

        # todo tile parsimony

        return

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
                if self.kernel.fitness_compare(fitness_train, comp_fit):
                    self.printpl('aa', 'Found a better candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                    self.parsimony_best_meta[parsim] = meta
            else:
                self.printpl('aa', 'Found a new candidate. Fit: {} Parsim: {}'.format(fitness_train, parsim))
                self.parsimony_best_meta[parsim] = meta

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   What happens in a Generation              |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def gen_reset_parameters(self):
        """
        Sets the parameters for the generation
        - reset population_tmp
        - Linearly increase threshold for parsimony
        """

        self.gen_id += 1
        self.time_genstart = time.perf_counter()
        self.debug_warnings = {}
        self.population_tmp = ['Plagih GP - Evolving Generation']  # initialise population_tmp to host the next generation
        self.parsimony_tmp = max(1 / min(self.gen_id, self.config['gen_id_max_parsimony']) * self.parsimony_max, self.parsimony_max)

        return

    def gen_reproduce(self, repro_rate):

        """
        A single Tree from the prior generation is copied without mutation
        """

        for n in range(repro_rate):  # quantity of Trees to be copied without mutation
            tourn_winner = self.pop_selection_tournament(self.tourn_size)
            self.pop_append(tourn_winner, last_modification='repro')  # i know, tests are not necessary...

        return

    def gen_reproduce_olymp(self, repro_rate):

        """
        Copy an entry from the pareto candidates into the population
        """
        # print('before olymp:', len(self.population_tmp))
        for n in range(repro_rate):  # quantity of Trees to be copied without mutation
            if self.parsimony_best_meta:
                meta = np.random.choice(list(self.parsimony_best_meta.values()))
                # expr_sym = meta['expr_sym']; print('sym', expr_sym)
                expr_raw = meta['expr_raw']
                # label_list = ast_convert_from_expr(expr_sym, build=True); print('label_list', label_list)
                label_list = ast_convert_from_expr(expr_raw, build=True)
                olymp_winner = karoo_tree_from_labellist(label_list)
                self.pop_append(olymp_winner, last_modification='repro')

        return

    def gen_reproduce_reduce(self, repro_rate):

        """
        copy a tree from the last population and sympify parts of it
        """

        for i in range(repro_rate):
            tree = self.pop_selection_tournament(self.tourn_size)
            tree = tree_evolve_reduce_parts(tree, completely=False)
            self.pop_append(tree, last_modification='point')

        return

    def gen_mutate_point(self, repro_rate):

        """
        One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """

        for i in range(repro_rate):  # quantity of Trees to be generated through mutation
            tree = self.pop_selection_tournament(self.tourn_size)
            tree = self.tree_evolve_mutate_point(tree)

            self.pop_append(tree, last_modification='point')

        return

    def gen_mutate_filter(self, repro_rate):
        """

        """

        for i in range(repro_rate):
            tree = self.pop_selection_tournament(self.tourn_size)
            try:
                self.debug_warnings['824 tree'] = tree
                new_tree = self.treegp_mutate_filter_one(tree)
                if len(new_tree) > 1:
                    self.pop_append(new_tree, last_modification='filter')
            except Exception as ex:
                self.printpl('www', 'Tree in mutate filter could not be changed, {}'.format(ex))

        return

    def pop_first_create_from_origin(self, pop_size, tree_origin):
        """
        Constructs the first generation
        - loads the origin-tree from file
        - constructs the first generation from this tree with branc. mutation
        """

        tree_origin = tree_origin.copy()
        origin_ids = tree_get_mutatable_nodes(tree_origin, no_root=True)

        for tree_id in range(pop_size):
            tree = self.tree_evolve_branch_multiple(tree_origin)

            self.pop_append(tree, last_modification='first')

        self.print_g('ggg', 'We have constructed the first population of {} trees, saved to disk'.format(self.config['pop_max']))

    def tree_evolve_branch_multiple(self, tree):
        """
        """
        tree_origin = tree.copy()
        node_ids = tree_get_mutatable_layer(tree, 0)
        num_new_branches = len(node_ids)
        if num_new_branches > 3:
            print_warning('w', 'That is a lot of new branches: {}'.format(num_new_branches))

        each_nodes = int(self.parsimony_max / num_new_branches)

        for i in range(num_new_branches):
            # todo could gat last non modify layer instead
            node_ids = tree_get_mutatable_layer(tree, 0)
            node_id = node_ids[i]
            old_branch = tree_get_branch(tree, node_id, karoo=True)

            tree = self.tree_insert_branch_random(tree_origin, old_branch, max_nodes=each_nodes)  # tree with new branch

        return tree

    def gen_random_from_origin(self, repro_rate):
        """

        """
        if self.origin_exists():
            tree_origin = self.origin['tree'].copy()
            for i in range(repro_rate):
                tree = self.tree_evolve_branch_multiple(tree_origin)

                self.pop_append(tree, last_modification='miss(br)')

        return

    def gen_random_from_scratch(self, pop_size):
        """
        todo make available half ramped
        """

        action_type_one = next(iter(self.action_dict.values()))
        if action_type_one == 'float':
            xtype = '2f'
        elif action_type_one == 'bool':
            xtype = '2b'
        else:
            raise
        for i in range(pop_size):
            max_nodes = np.random.randint(10, self.config['parsimony_max'][1])  # todo 3 auslagern und testen ob 3 entstehen kann
            label_list, arity_list = invent_label_list_nodes_grow(xtype, max_nodes, self.variables_dict, self.func_array)
            tree = karoo_tree_from_labellist(label_list)
            tree = tree_set_id(tree, i)

            self.pop_append(tree, last_modification='random')

        return

    def gen_mutate_branch(self, repro_rate):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """

        for i in range(repro_rate):  # quantity of Trees to be generated through mutation

            tourn_winner = self.pop_selection_tournament(self.tourn_size)  # perform tournament selection for each mutation
            node_ids = tree_get_mutatable_nodes(tourn_winner, no_root=True)
            node = np.random.choice(node_ids)
            branch_nodes_ids = tree_get_branch(tourn_winner, node, karoo=True)  # select point of mutation and all nodes beneath [6, 9, 10]
            tourn_winner = self.tree_insert_branch_random(tourn_winner, branch_nodes_ids)

            self.pop_append(tourn_winner, last_modification='branch')

        return

    def gen_crossover_branch(self, repro_rate):
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
                left_xtype = xtype_get(tree_get_label(left_tree, left_id), variables_dict=self.variables_dict)
                conv_to_left, conv_to_right = xtype_get_converters(left_xtype)
                right_labels.insert(0, conv_to_left)
                right_aritys.insert(0, 1)
                left_labels.insert(0, conv_to_right)
                left_aritys.insert(0, 1)

            left_core = core_from_labels(left_labels, left_aritys)
            right_core = core_from_labels(right_labels, right_aritys)

            left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
            left_offspring = self.treegp_tree_prune(left_offspring, self.config['tree_depth_max'])
            self.pop_append(left_offspring, last_modification='cross')

            right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
            right_offspring = self.treegp_tree_prune(right_offspring, self.config['tree_depth_max'])
            self.pop_append(right_offspring, last_modification='cross')

        return

    def pop_append(self, tree, last_modification=''):
        """
        some stuff to definitely do before actually appending to pop
        """
        if tree is None:
            return
        else:
            if self.origin_exists():
                tree = tree_set_modifyable_nodes(tree, self.origin['tree'])
            else:
                tree = tree_set_modifyable_nodes_true(tree)

            tree = tree_round_constants(tree, self.config['float_accuracy'], karoo=True)
            tree = tree_normalize_exponentiation(tree)
            tree = tree_set_history(tree, last_modification)

            if not tree_test_check_children(tree):
                print_e('Tree is not consistent:\n{}'.format(tree))
            elif not tree_check_child_xtype(tree, variables_dict=self.variables_dict):
                print_e('Tree childrens xtypes are not correct:\n{}'.format(tree[N_label]))
            elif tree_node_get_arity(tree, root_id) == 0:
                print_warning('w', 'Tree is only a root node')
            else:
                self.population_tmp.append(tree)

        return

    def gen_finalize(self):

        """
        From raw population_tmp to new population_genepool
        - Gene_pool with tree's parsimony (and store info in the tree)
        -

        """

        self.population_tmp = pop_enum_trees(self.population_tmp)  # pop +tree_id
        gene_pool, self.population_tmp = self.pop_genepool_create(self.population_tmp)
        self.analyze_genepool(gene_pool, self.root_dir, self.gen_id)
        self.pop_parsimony_best_update(gene_pool)
        self.pop_pareto_update()
        self.population_base = pop_copy_genepool(self.population_tmp, gene_pool, self.gen_id)
        file_population_karoo(self.population_tmp, 'tmp', self.root_dir, self.gen_id)

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
        best_id = None
        best_fitness = None

        # Get several values
        for n in range(tourn_size):

            tree_id = pop_random(self.population_base)
            tree = self.population_base[tree_id]
            fitness = tree_get_fitness(tree, precision=self.config['precision'])  # extract the fitness from the array

            if self.kernel.fitness_compare(fitness, best_fitness, mode='better'):
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
                printez('w', 'Warning: Filter  not specified. Please specify a filter_type.', print_type=self.print_type)
                constant = np.random.normal(constant, 0.1)

        if term_type == 'int':
            constant = int(np.random.normal(constant, 2))

        if term_type == 'bool':
            constant = not constant
            # random by 50:50?

        return constant

    def tree_evolve_complexify(self, tree, same_arity=True):
        """
        todo
        a function that inserts certain functions that hopefully give good opportunities for next generations
        eg: in old_node '+', inserting Ifte(True, '+', 1.23) or so...
        """
        pass

    def tree_evolve_mutate_point(self, tree):

        """
        Mutate a single mutatable point in any Tree.
        """

        # 1. choose a node
        node_ids = tree_get_mutatable_nodes(tree)
        node_id = np.random.choice(node_ids)
        label, arity, xtype = tree_node_get_lax(tree, node_id, variables_dict=self.variables_dict)

        if arity > 0:
            new_label, new_arity = xtype_choose_func(self.func_array, xtype=xtype, arity=arity)  # Function is same type, same arity
            tree = tree_node_set_label(tree, node_id, new_label)
        else:  # arity == 0:  # aka a terminal
            new_label = self.xtype_choose_term(xtype)  # 3 -> '2f' -> 5
            tree = tree_node_set_label(tree, node_id, new_label)

        return tree  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping

    def treegp_mutate_point_insert_evolve(self, tree, same_arity=True):
        """

        """

        node_ids = tree_get_mutatable_nodes(tree)
        insert_id = None
        for node_id in node_ids:
            label = tree_get_label(tree, node_id)
            xtype = xtype_get(label, variables_dict=self.variables_dict)  # '>' -> 'f2b'
            if label == '**' and tree_node_get_child(tree, node_id, 1) != 'Power':  # todo
                insert_id = node_id
                break

        if insert_id:
            old_ids, old_labels, old_aritys = tree_get_branch_lax(tree, insert_id)
            new_labels = ['Power'] + old_labels
            new_aritys = [1] + old_aritys
            insert_core = core_from_labels(new_labels, new_aritys)
            tree_insert_subtree(tree, insert_core, old_ids, karoo=False)

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
            # todo modify multiple float nodes at once?
            float_id = np.random.choice(float_nodes)
            val = float(tree_get_label(tree, float_id))
            new_value = self.gp_mutate_constantfilter(val, term_type='float', filter_type='gaussian_filter')
            tree = tree_node_set_label(tree, float_id, new_value)
            return tree
        else:
            raise Exception('No mutatable node found!')
            # return None

    def treegp_tree_prune(self, tree, max_depth):
        """
        reduces the depth of a Tree (in case it is too deep).
        Arguments required: tree, depth
        """

        nodes = []

        for node_id in range(root_id, len(tree[3])):

            node_depth = tree_get_label(tree, node_id)
            node_arity = tree_node_get_arity(tree, node_id)
            if node_depth == max_depth and node_arity > 0:  # replace this node with terminal
                label = tree_get_label(tree, node_id)
                node_xtype = xtype_get(label, variables_dict=self.variables_dict)
                tree = tree_node_set_arity(tree, node_id, 0)
                new_term = self.xtype_choose_term(node_xtype)  # replace label
                tree = tree_node_set_label(tree, node_id, new_term)

            elif tree_node_get_depth(tree, node_id) > max_depth:  # record nodes deeper than the maximum allowed Tree depth
                nodes.append(node_id)

        tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
        tree = evolve_node_arity_fix(tree)  # fix all node arities

        return tree

    def treegp_crossover_get_partner_node_id(self, function_label, partner_tree, partner_branch_id, mode='same_type'):
        """
        -> Crossover: Returns a node_id in the partner tree, that can be swapped
        """

        node_xtype = xtype_get(function_label, variables_dict=self.variables_dict)
        node_options = []

        if mode == 'same_type':  # only return a node with the same function type
            # for i, label in enumerate(partner_tree[N_label][1:]):
            for i, label in enumerate(tree_get_ids(partner_tree, karoo=True, skip_nodes=1)):
                partner_node_xtype = xtype_get(label, variables_dict=self.variables_dict)
                if node_xtype == partner_node_xtype:
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
        a_node_label = tree_get_label(a_tree, a_id)
        a_xtype = xtype_get(a_node_label, variables_dict=self.variables_dict)

        # create a list from parent b with same xtype
        b_node_ids = tree_get_mutatable_nodes(b_tree, no_root=True)
        b_sametype_ids = b_node_ids[:]
        for b_id in b_node_ids:
            b_label = tree_get_label(b_tree, b_id)
            b_xtype = xtype_get(b_label, variables_dict=self.variables_dict)
            if not xtype_equi_outcome(b_xtype, a_xtype):
                b_sametype_ids.remove(b_id)  # remove one-by-one false partner nodes.

        if b_sametype_ids:  # if entries were found, choose one. we are custom_done
            b_id = np.random.choice(b_sametype_ids)
            success = True
        else:
            b_id = np.random.choice(b_node_ids)
            success = False

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

    def tree_insert_branch_random(self, tree, branch_ids, grow_method='nodes_max_uniform', max_nodes=None):

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

        # Get information about the top-node we have to replace
        old_label = tree_get_label(tree, branch_ids[0])
        old_xtype = xtype_get(old_label, variables_dict=self.variables_dict)

        if grow_method == 'depth_base_random':
            """
            We allow base depth (which is a little lower than max)
            but every node has 0.5 chance to become a terminal
            - iterate over depths
            - fill with as many funcs as possible

            """
            # calculate depth restriction
            depth_upper_bound = self.config['tree_depth_max'] - tree_node_get_depth(tree, branch_ids[0])
            depth_goal = min(self.config['tree_depth_base'], depth_upper_bound)

            # Build a new tree
            label_list, arity_list = invent_label_list_depth_random(old_xtype, depth_goal, self.variables_dict, self.action_dict, self.func_array, min_depth=self.config['tree_depth_min'])

        elif grow_method == 'nodes_max_uniform':
            """
            We allow a certain amount of new nodes instead tree depth.
            This could be calculated respectively to the parsimony dim_y
            which the tree might have up his sleeve
            """
            if not max_nodes:
                max_nodes = self.config['parsimony_max']  # todo too high
            label_list, arity_list = invent_label_list_nodes_grow(old_xtype, max_nodes, self.variables_dict, self.func_array)

        else:
            print_e('That did not work')
            return None

        if not label_list:
            result_tree = tree
        else:
            core_insert = core_from_labels(label_list, arity_list)
            result_tree = tree_insert_subtree(tree, core_insert, branch_ids, karoo=True)

        return result_tree

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
            tree = karoo_tree_from_labellist(label_list, modify_list=modify_list)
        else:
            print_warning('w', 'No origin provided. starting from scratch with random generation?')
            tree = None

        origin_algo_raw = tree_get_expr_raw(tree, P_first_node)
        try:
            expr_sym = tree_expr_sympify(algo_raw=origin_algo_raw)
        except:
            raise Exception('Your origin algorithm can already not be sympified. Aborting.')
        self.origin = {'tree': tree,
                       'expr_raw': origin_algo_raw,
                       'expr_sym': expr_sym,
                       'parsimony': 0}
        try:
            origin_hash, origin_meta = self.tree_get_meta(tree, tree_origin=self.origin['tree'])
        except Exception:
            whats_wrong = 'Your origin algorithm already caused an exception!'
            raise Exception(whats_wrong)
        self.origin['fitness_train'] = origin_meta['fitness_train']
        self.parsimony_best_meta[0] = origin_meta
        self.pareto[0] = self.origin['fitness_train']

        self.print_g('gg', 'Loading origin, fitness {}. Time: {:4.2f}s'.format(origin_meta['fitness_train'], time.perf_counter() - self.time_start))

        return

    def tree_get_meta(self, tree, tree_origin=None):
        """
        gets all the main tree information
        1. tree_identifier (expr_raw)
        2. expr_sym
        3. parsimony
        4. fitness_train
        """
        # 1. get expr_raw - what is needed to compute the tree identifier
        expr_raw = tree_get_expr_raw(tree, root_id)
        tree_ident = hash(expr_raw)  # sfeh: potential for improvement- use expr_sym in separate dict as identifier.

        # 2 Did we have this tree already? -> Nice, we have everything
        if tree_ident in self.tree_meta:
            tree_meta = self.tree_meta[tree_ident]

        else:  # 2.2 New tree, but still skip fitness eval for complex trees
            parsimony = tree_get_parsimony(tree, self.config['complexity_measure'], origin_tree=tree_origin)

            if parsimony < self.parsimony_max:  # 3. compute fitness
                # print('Algo raw:', str(expr_raw))  # importantprint 2 for expr_raw
                try:  # 3. With tensorflow
                    expr_sym = tree_expr_sympify(algo_raw=str(expr_raw))
                except Exception as ex:
                    raise Exception('Expr could not be sympified: {}. Ex: {}'.format(expr_raw, ex))

                fitness_train = eval_tf(expr_sym, self.data_train, self.eval_parameters)['fitness']
                # print('Nftr', fitness_train)
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
        return choose_constant(term_type=the_type)  # otherwise: constant (There are always constants :P)

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Monitoring                                |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def auto_plots(self, path):
        """
        Make all plots
        """

        path_plots = path / folder_plots
        if not Path.is_dir(path_plots):
            Path.mkdir(path_plots)

        if self.monitor_dict['gen_fitness_average'] == 'y':
            data_tuples = sorted(list(self.monitoring_dict['fitness_average'].items()))
            self.plot_end(data_tuples, path_plots, plt_title='average fitness', plt_y_label='fitness', linestyle='-')

        if self.monitor_dict['genepool_size'] == 'y':
            data_tuples = sorted(list(self.monitoring_dict['genepool_size'].items()))
            self.plot_end(data_tuples, path_plots, plt_title='genepool size', plt_y_label='amount', linestyle='')

        data_tuples = sorted(list(self.monitoring_dict['complexity_average'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='average tree complexity', plt_y_label='#nodes', linestyle='-')

        data_tuples = sorted(list(self.monitoring_dict['total_found_trees'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='number of created trees', plt_y_label='amount', linestyle='')

        data_tuples = sorted(list(self.pareto.items()))
        self.plot_end(data_tuples, path_plots, plt_title='pareto dominant candidates', plt_x_label='parsimony', plt_y_label='fitness', linestyle='dashed',
                      step_where='pre', max_right=30)  # todo 30 is related to parsimony_max

        data_tuples = sorted(list(self.monitoring_dict['best_candidate'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='best candidate', plt_x_label='generation', plt_y_label='fitness', linestyle='dashed',
                      step_where='post')

        return

    def analyze_genepool(self, gene_pool, root_path, gen_id):
        """
        Give the user some feedback
        """

        # How many survived in the selection?
        self.monitoring_dict['genepool_size'][int(gen_id)] = len(gene_pool)
        if len(gene_pool) > 0:
            self.print_g('ggg', 'The generation`s population is: {}'.format(len(gene_pool)))
        else:
            self.printpl('e', 'There are no Trees in the gene pool.')
            self.terminate_run(root_path)

        if not self.best_fitness:
            if self.origin_exists():
                self.best_fitness = self.origin['fitness_train']
            else:
                self.best_fitness = next(iter(gene_pool.values()))['fitness_train']

        fitness_train_sum, count_success = 0, 0
        for _, meta in gene_pool.items():
            fitness = float(meta['fitness_train'])
            if fitness == fitness and fitness is not float('inf'):  # weird comparison is NaN Test
                count_success += 1
                fitness_train_sum += fitness  # for fitness average
                if self.kernel.fitness_compare(fitness, self.best_fitness):
                    self.best_fitness = fitness

        average_fitness = fitness_train_sum / max(count_success, 1)
        self.monitoring_dict['fitness_average'][gen_id] = average_fitness

        self.monitoring_dict['best_candidate'][gen_id] = self.best_fitness

        complexity_sum = 0
        for tree in self.population_tmp:
            complexity_sum += len(tree_get_ids(tree, karoo=True))
        avg_complexity = complexity_sum / len(self.population_tmp)
        self.monitoring_dict['complexity_average'][gen_id] = avg_complexity

        return

    def terminate_run(self, path):

        self.file_save_files(path)
        self.auto_plots(path)
        self.print_g('gg', ' Terminating this run. \tTotal time: {:4.2f}s'.format(time.perf_counter() - self.time_start))
        sys.exit()

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to print_type output information     |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def plot_end(self, data_2d, path,
                 plt_title='', plt_curve_label='', plt_x_label='Generation', plt_y_label='', yscale='linear', step_where='', plt_xparam='',
                 linestyle='None', color='', max_right=None):

        x, y = [], []
        for a, b in data_2d:
            x.append(a)
            y.append(b)

        if step_where:
            plt.step(x, y, plt_xparam, linestyle=linestyle, marker='.', label=plt_curve_label, where=step_where)
        else:
            plt.plot(x, y, plt_xparam, linestyle=linestyle, marker='.', label=plt_curve_label)

        plt.xlabel(plt_x_label)
        plt.ylabel(plt_y_label)
        plt.title(plt_title)

        # plt.legend()

        # let it start at (0,0) but +5% margin to the top and right
        plt.yscale(yscale)
        bottom, top = plt.ylim()
        new_top = (top - min(bottom, 0)) * 1.05  # todo beautify plots...
        plt.ylim(min(bottom, 0), new_top)
        left, right = plt.xlim()
        if max_right:
            right = max(right, max_right)
        new_right = right * 1.05
        plt.xlim(min(left, 0), new_right)
        plt.margins(x=0, y=0)

        plt.savefig(path / '{}.jpg'.format(plt_title))
        plt.close()
        return

    def printpl(self, message_type, text):

        """

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


def file_population_karoo(population, pop_name, path, gen_id):
    """
    Save population_* to disk.

    """

    pop_path = path / folder_info
    if not Path.is_dir(pop_path):
        Path.mkdir(pop_path)

    file_path = pop_path / 'population_{}.csv'.format(str(pop_name))

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

    path_config = path / folder_info

    if not Path.is_dir(path_config):
        Path.mkdir(path_config)

    file = Path.open(path_config / file_config, 'a')
    file.write('This config is not complete, sfeh!')
    file.write('\n launched: {}'.format(datetime))
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


def tree_get_parsimony(tree, parsimony_distance, origin_tree=None):
    """
    parsimony_distance: compute the chosen distance by the user.

    """

    if parsimony_distance == 'total_count_nodes':  # number of nodes
        return tree_get_last_nodeid(tree)  # returns the number of nodes
    elif parsimony_distance == 'total_tree_depth':
        return tree[N_depth][1]  # returns the tree depth

    if parsimony_distance == 'ted':
        return tree_parsimony_ted(tree, origin_tree)
    elif parsimony_distance == 'rel_ari_1':  # Does this work?
        return tree_parsimony_relari(tree, origin_tree)
    else:
        print_e('Complexity measurement not available: {}'.format(parsimony_distance))
        raise
