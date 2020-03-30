"""
Explaination:
> 'f2f', 'b2b', etc.: my personal silly naming. f = float, b = bool.
   f2b means float to boolean, e. g. '<' takes 'float' and returns a 'bool'
> node_modify (0 or 1), specifies, whether this node is supposed to be

# todo todo creating random trees is mandatory!! to find base genes... or is it?
# todo idea: split up population after convergence to a certain structure?

Functions, that might be addable in the future:
'Integer': 'f2f', # converts a number to an integer.
"""
import matplotlib.pyplot as plt
import time
from plagih.modules.file_interaction import *
import yaml
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
    The main class performing all the important stuff
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
                'constants': ['gauss', 'random', 'observation samples'],
                'default': 'random'
            },
            # (!) Relevant for result
            'pop_max': 1000,  # Maximum amount of trees in a population. Only used evolve rates, condition is never tested.
            'parsimony_max': 100,  # right value is the maximum parsimony. left value not used, but was meant to set parsimony for the first generations. [3 to 2^(bas +1) - 1]
            'kernel_name': 'regression bounded',  # [regression, regression bounded, classification, match]
            'complexity_measure': 'ted',

            # rather irrelevant
            'parsimony_tmp': 15,
            'precision': 3,  # rounding the fitness
            'float_accuracy': 200,
            'swim': 'p',  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
            'print_type': 'gggwwsivoaa',  # To print_type absolutely all: wggggsiiiivvvtopppttt
            'overwrite periodic gp_files': True,  # If True, the file gets overwritten. If False, in every generation a new file is created.
            'force_new_run': False,  # especially for testing. Instead of deleting the old folder each time, you can set this to False to init a new run again #
            'delete_old_file': False,  # sfeh, delete old gp_files. be very careful
            'monitor': {'gen_fitness_average': 'y',
                        'sympify_errors': 'y',
                        'population_tmp_done-size': 'y'
                        },
            'period': {'time_monitor': None,  # in sec
                       'time_save': None,  # in sec
                       'gen_monitor': 1,  # in gen counts
                       'gen_save': 1},  # in gen counts

            # GP-evolve specific parameters
            'evolve_rates': {'repro one': 0.03,
                             'repro pareto': 0.04,
                             'repro reduced one': 0.03,
                             'filter floats': 0.05,
                             'point mutate function': 0.1,
                             'branch mutate insert': 0.10,
                             'crossover branches': 0.40,
                             'random from origin_tree': 0.15,
                             'random from scratch': 0.15,
                             },
            'crossover_type_safety_mode': 'replace_same_types',
            'gen_num_max_parsimony': 50,  # Increase tmp_parsim to this generation
            'tree_growth': 'node-based',  # node-based, depth-based
            'tree_depth_base': 7,  # [3..10]
            'tree_depth_max': 25,  # maximum Tree depth for entire run
            'tree_depth_min': 5,
            'tree from scratch min_nodes': 8,
            'random from scratch max nodes': 50,
            'tree branch base nodes': 20,
            'tourn_size': 4,  # [7 per 100] number of trees selected for tournament

            # When to stop the run
            'time_max': None,  # int(60 * 60 * 12),  # 60 = 1 min
            'gen_max': 800,  # Maximum amount of generations

            'env': {
                'name': None,
                'observations': {

                }
            }
        }

        self.config.update(config)  # todo check if config is correct

        # init values with dummies (just to have all self values here for overview)
        self.tree_meta = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw', 'gen+nr'}
        self.parsimony_best_meta = {}  # tree_meta = {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.pareto = {}  # a dict with all pareto candidates. key is complexity, value is tree meta
        self.population_tmp_done = []
        self.population_tmp_eval = []
        self.population_base = []  # population that is taken to the next generation
        self.best_fitness = None  # keeps track of the current best fitness
        self.action_min_max = [None, None]  # list with [0] = min and [1] = max, For kernel "regression bounded" (or so)
        self.origin_meta = None
        self.origin_tree = None
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

    def load_backup(self):
        """
        If a backup-file is found
        """
        path_backup = self.root_dir / file_backup_pickle
        if Path.is_file(path_backup):
            self.print_g('g', 'Backup-file was found. Loading data...')
            try:
                self.plagih_load_backup(path_backup)
                return True
            except Exception as ex:
                print_warning('w', 'Even though a backup exists for this run, it could not be loaded because of: {}.'.format(ex))
                raise
        else:
            return False

    def plagih_gp_run(self):
        """
        regular plagih-config run
        """
        if not self.config['force_new_run']:
            self.load_backup()

        # check for 'random from scratch' + 'origin has fix nodes' fail?
        if self.origin_exists():
            if tree_node_is_modifiable(self.origin_tree, root_id):  # Modify-nodes is not "activated"
                self.config['evolve_rates']['random from origin_tree'] += float(self.config['evolve_rates']['random from scratch'])
                self.config['evolve_rates']['random from scratch'] = 0
        else:
            if self.config['complexity_measure'] in ['ted', 'rel_ari_1']:  # sfeh make 'ted', 'rel_ari_1' variables
                raise Exception('Can not use relative distance without providing a reference/origin tree!')
        if self.gen_id == 0:
            self.gen_create_first()

        self.write_config_yaml()  # sfeh or json?

        self.gen_create_loop()
        self.terminate_run(self.root_dir)
        return

    def plagih_update_analysis(self):
        """
        Without starting a new run, get the most important gp_files
        """

        loading_done = self.load_backup()
        if not loading_done:
            print_e('You need to load a backup file to analyse!')
        else:
            self.terminate_run(self.root_dir)

    def write_config_yaml(self):
        """
        write the parameters to a .csv file which can also be loaded
        """

        path_config = make_dir(self.root_dir / folder_info)
        filename = path_config / file_config_yaml

        with Path.open(filename, 'w') as file:
            _ = yaml.dump(self.config, file, indent=4)

        return

    def write_config_json(self):
        """
        write the parameters to a .csv file which can also be loaded
        """

        path_config = make_dir(self.root_dir / folder_info)
        filename = path_config / file_config_json

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

        rate_o = int(self.config['pop_max'] * self.evolve_rates['random from origin_tree'])
        rate_s = int(self.config['pop_max'] * self.evolve_rates['random from scratch'])

        self.pop_random_from_origin(rate_o)

        self.pop_random_from_scratch(rate_s)

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
            ('random from origin_tree', self.pop_random_from_origin, 0),
            ('random from scratch', self.pop_random_from_scratch, 0)]

        if self.origin_exists():
            origin_tree = self.origin_tree
        else:
            origin_tree = None

        gp_dict = {  # name of the function:    implementation in plagih,       number of tournament selections needed
            'repro one': (self.pop_reproduce, 1, None),
            'repro pareto': (self.pop_reproduce_olymp, 0, None),
            'repro reduced one': (self.pop_reproduce_reduce, 1, None),
            'point mutate function': (self.pop_mutate_point, 1, None),
            'filter floats': (self.pop_mutate_filter, 1, None),
            'branch mutate insert': (self.pop_mutate_branch, 1, None),
            'crossover branches': (self.pop_crossover_branch, 2, None),
            'random from origin_tree': (self.pop_random_from_origin, 0, origin_tree),
            'random from scratch': (self.pop_random_from_scratch, 0, None)}

        gp_dict2 = {
            'repro one': {'fun': self.pop_reproduce, 'tourn_size': 1, 'origin': None},
            'repro pareto': {'fun': self.pop_reproduce_olymp, 'tourn_size': 0, 'origin': None},
            'repro reduced one': {'fun': self.pop_reproduce_reduce, 'tourn_size': 1, 'origin': None},
            'point mutate function': {'fun': self.pop_mutate_point, 'tourn_size': 1, 'origin': None},
            'filter floats': {'fun': self.pop_mutate_filter, 'tourn_size': 1, 'origin': None},
            'branch mutate insert': {'fun': self.pop_mutate_branch, 'tourn_size': 1, 'origin': None},
            'crossover branches': {'fun': self.pop_crossover_branch, 'tourn_size': 2, 'origin': None},
            'random from origin_tree': {'fun': self.pop_random_from_origin, 'tourn_size': 0, 'origin': origin_tree},
            'random from scratch': {'fun': self.pop_random_from_scratch, 'tourn_size': 0, 'origin': None}}

        while self.run_continues():  # max generation, max time, done...

            self.gen_reset_parameters()

            # Creating a new population ############
            for name, gp_function, tourn_rep in gp_list:
                time_evolve = time.perf_counter()
                count_tries = 0
                # while count_tries < self.evolve_rates[name] * self.config['pop_max']:
                #     break

                repro_rate = int(self.evolve_rates[name] * self.config['pop_max'])
                gp_function(repro_rate)
                self.print_g('ggg', '-->Evolve ({}) {}x. Took: {:4.2f}s.'.format(name, repro_rate, time.perf_counter() - time_evolve))
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
            self.run_save_pickle()
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
        self.file_pareto_pycode(self.pareto, root_path)

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
        # _, variables_dict, action_dict, unique_outputs_num, data_train_rows, data_train, data_control, action_min_max = data_prepared
        observations_bundle, actions, param_at, unique_outputs_num, data_train_rows, data_train, data_control, action_min_max = data_prepared

        self.variables_dict = observations_bundle  # todo remane
        self.action_dict = actions
        self.param_at = param_at
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

        self.output_xtype = self.action_get_xtype()

        return

    def activate_operators(self, func_array):
        """
        operators were loaded already and need to be set in the gp run
        """
        self.func_array = func_array
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
                population[ii] = tree_set_xtypes(tree, self.variables_dict)
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

    def plagih_load_backup(self, path_backup):
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
        if self.monitoring_dict.get('complexity_variance') is None:
            self.monitoring_dict['complexity_variance'] = {}
        if self.monitoring_dict.get('fitness_variance') is None:
            self.monitoring_dict['fitness_variance'] = {}

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
                    'self.pareto': self.pareto,
                    'self.population_base': self.population_base,
                    'self.monitoring_dict': self.monitoring_dict
                    }
        pickle.dump(run_data, Path.open(path_backup, 'wb'))
        self.printpl('iii', 'Saved run in pickle file.')
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
        #     origin_fitness = eval_tf(self.origin_meta['expr_sym'], self.data_control, self.eval_parameters, get_pred_labels=True)['fitness']
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
        #     result = eval_tf(algo_sym, self.data_control, self.eval_parameters, get_pred_labels=True)
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

        path_pareto = root_path / folder_info
        if not Path.is_dir(path_pareto):
            Path.mkdir(path_pareto)

        with Path.open(path_pareto / file_pareto, 'w') as file:

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
        path_trees = make_dir(root_path / folder_trees)

        for parsim, meta in sorted(list(pareto.items())):
            expr_raw = meta['expr_raw']  # sfeh: use raw or sym?
            label_list = ast_convert_from_expr(expr_raw, build=True)
            xtype_list = xtypes_from_labels(label_list, self.variables_dict)
            tree = karoo_tree_from_labellist(label_list, xtype_list)
            tree = self.tree_beautify(tree, last_evolution='texify')
            ###
            vistree = visualize_tree_get_vistree(tree)
            ###

            tikz_code = latex_tree_get_forest(vistree)  # generate the small forest inputs

            # save a ready-to-use tex file with all pareto trees
            forest_grouped.append(latex_get_forest_title(parsim, meta['fitness_train'], tikz_code, tree_sep))

        latex_full_doc = latex_complete_tree_summary(forest_grouped)

        with Path.open(path_trees / '#all_trees.tex', 'w') as file:
            file.write(latex_full_doc)

        return

    def file_pareto_pycode(self, pareto, root_path):
        """

        """
        try:
            path_trees = make_dir(root_path / folder_trees)

            all_agents = []
            all_agent_names = []

            for parsim, meta in sorted(list(pareto.items())):
                agent_name = 'agent{:.0f}'.format(parsim)  # todo set better name

                expr_raw = meta['expr_raw']
                expr_sym = expr_sympify(expr_raw)
                label_list_sym = ast_convert_from_expr(expr_sym, build=True)
                xtype_list_sym = xtypes_from_labels(label_list_sym, self.variables_dict)  # todo sym or raw?
                tree = karoo_tree_from_labellist(label_list_sym, xtype_list_sym)
                pycode = tree_get_pycode(tree)

                all_agent_names.append(agent_name)
                code_tmp = 'class {}:\n\n' \
                           '\tdef decide(self, {}):\n' \
                           '\t\tcartPos, cartVel = observation[0], observation[1]\n' \
                           '\t\treturn max(0, min(2, int(round({}))))\n\n'.format(agent_name, name_observation, pycode)  # sfeh: maybe cast
                # todo
                all_agents.append(code_tmp)

            pycode_names = 'all_agents = [,(\'{}\', {}())'.format(all_agent_names[0], all_agent_names[0])
            pycode_agents = '{}'.format(all_agents[0])
            for ii, agent_name in enumerate(all_agent_names):
                if ii == 0:
                    continue
                pycode_names += '(\'{}\', {}())'.format(agent_name, agent_name)
                pycode_agents += all_agents[ii]
            else:
                pycode_names += ']\n'
            complete_file = 'import math;\n\n' \
                            '{}\n\n' \
                            '{}'.format(pycode_agents, pycode_names)

            with Path.open(path_trees / file_pycode, 'w') as file:
                file.write(complete_file)
        except Exception as ex:
            print_e('Wooops! no py-files created. sfeh-todo? ex: {}'.format(ex))

        return

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Population specific                       |
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
            print_warning('ww', 'Evaluating {} trees in gen {} caused {} exceptions.'.format(len(self.population_tmp_eval), self.gen_id, count_fails), print_type=self.print_type)

        return

    def origin_tree_get(self):
        """
        Safely return an origin_meta tree
        """
        if self.origin_exists():
            tree_origin = self.origin_tree
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

        if self.tree_check_core_all(tree):
            tree = self.tree_beautify(tree, last_evolution='par-s')
            parsimony = self.tree_eval_parsimony_easywrapper(tree)
            tree = tree_set_parsimony(tree, parsimony)
            self.tree_meta_update(tree, parsimony=parsimony)
            self.population_tmp_done.append(tree)

            self.pareto_update_insert()

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
                        self.printpl('a', 'Pareto update at {}, with new fitness: {}. Old was: {}!'.format(parsim, fitness, best_fit))
                        pareto_improved = True
                else:
                    self.pareto[parsim] = meta
                    self.printpl('a', 'New pareto entry at {} with fitness: {:4.2f}!'.format(parsim, fitness))
                    pareto_improved = True
                best_fit = fitness

            if pareto_improved:
                expr_raw = meta['expr_raw']  # expy_sym will can cause exceptiops while setting fix nodes
                tree = karoo_tree_from_expr(expr_raw, self.variables_dict)
                tree = tree_set_modifyable_nodes(tree, origin_tree=self.origin_tree_get())
                sym_tree = tree_evolve_reduce(tree, self.variables_dict, completely=True)
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
        self.population_tmp_done = []
        self.population_tmp_eval = []
        self.parsimony_tmp = max(1 / min(self.gen_id, self.config['gen_num_max_parsimony']) * self.parsimony_max, self.parsimony_max)

        return

    def pop_random_from_origin(self, repro_rate):
        """

        """

        if self.origin_exists():
            tree_origin = self.origin_tree.copy()
            for _ in range(repro_rate):
                goal_nodes = np.random.randint(self.config['tree from scratch min_nodes'], 1 + self.config['random from scratch max nodes'])
                tree = tree_evolve_branch_multiple(tree_origin, goal_nodes, self.variables_dict, self.func_array)
                self.pop_append(tree, last_evolution='new-o')

        return

    def pop_random_from_scratch(self, repro_rate):
        """
        sfeh
        """
        if self.origin_exists():
            if repro_rate > 0 and tree_node_get_modify(self.origin_tree, root_id) != node_is_modifiable:
                print_warning('w', 'You can not create new trees from scratch when origin has fix nodes! {} This should be handeled earlier'.format(repro_rate))
                # print('TEST: removing a return statement here')
                return

        for i in range(repro_rate):
            goal_nodes = np.random.randint(self.config['tree from scratch min_nodes'], 1 + self.config['random from scratch max nodes'])
            label_list, arity_list, xtype_list = invent_label_list_nodes_grow(self.output_xtype, goal_nodes, self.variables_dict, self.func_array)
            p_tree = Plagih_Tree(label_list, xtype_list)
            tree = p_tree.get_uninstanced_tree()
            # tree = tree_set_id(tree, i)

            self.pop_append(tree, last_evolution='new-s')

        return

    def pop_reproduce(self, repro_rate):

        """
        A single Tree from the prior generation is copied as is
        """

        for _ in range(repro_rate):
            tourn_winner = self.pop_selection_tournament(self.tourn_size)
            self.pop_append(tourn_winner, last_evolution='r-one')  # i know, tests are not necessary...

        return

    def pop_reproduce_olymp(self, repro_rate):

        """
        Copy an entry from the pareto candidates into the population
        """

        for _ in range(repro_rate):
            if self.parsimony_best_meta:
                meta = np.random.choice(list(self.parsimony_best_meta.values()))
                expr_raw = meta['expr_raw']
                label_list = ast_convert_from_expr(expr_raw, build=True)
                xtype_list = xtypes_from_labels(label_list, self.variables_dict)
                p_tree = Plagih_Tree(label_list, xtype_list)
                olymp_winner = p_tree.get_uninstanced_tree()
                self.pop_append(olymp_winner, last_evolution='r-par')

        return

    def pop_reproduce_reduce(self, repro_rate):

        """
        copy a tree from the last population and sympify parts of it
        """

        for _ in range(repro_rate):
            tree = self.pop_selection_tournament(self.tourn_size)
            tree = tree_evolve_reduce(tree, self.variables_dict, completely=False)
            self.pop_append(tree, last_evolution='r-sym')

        return

    def pop_mutate_point(self, repro_rate):

        """
        Point mutation, One point (terminal or function) gets mutated.
        SFEH: Currently only mutating with functions/terminals of the exactly same type.
        """

        for _ in range(repro_rate):  # quantity of Trees to be generated through mutation
            tree = self.pop_selection_tournament(self.tourn_size)
            tree = tree_evolve_mutate_point(tree, self.func_array, self.variables_dict)
            self.pop_append(tree, last_evolution='m-poi')

        return

    def pop_mutate_filter(self, repro_rate):
        """

        """

        for _ in range(repro_rate):
            tree = self.pop_selection_tournament(self.tourn_size)
            try:
                new_tree = tree_evolve_mutate_filter_one(tree)
                if len(new_tree) > 1:
                    self.pop_append(new_tree, last_evolution='m-fil')
            except Exception as ex:
                self.printpl('www', 'Tree in mutate filter could not be changed, {}'.format(ex))

        return

    def pop_mutate_branch(self, repro_rate):

        """
        Mutates a whole tree branch.

        If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        """

        for _ in range(repro_rate):  # quantity of Trees to be generated through mutation
            # time_start = time.perf_counter()
            tree = self.pop_selection_tournament(self.tourn_size)  # perform tournament selection for each mutation
            node_ids = tree_get_mutatable_nodes(tree, no_root=True)
            node = np.random.choice(node_ids)
            branch_nodes_ids = tree_node_get_branch(tree, node, karoo=True)  # select point of mutation and all nodes beneath [6, 9, 10]
            if self.config['tree_growth'] == 'depth-based':
                tree = tree_evolve_insert_branch_v1(tree, branch_nodes_ids, self.variables_dict, self.func_array,
                                                    depth_max=self.config['tree_depth_max'],
                                                    depth_min=self.config['tree_depth_min'],
                                                    depth_goal=self.config['tree_depth_base'])
            elif self.config['tree_growth'] == 'node-based':
                goal_nodes = np.random.randint(1, 1 + max(min(self.config['tree branch base nodes'], self.parsimony_max - tree_get_parsimony(tree)), 1))  # max just for safety reasons
                tree = tree_insert_branch_v2(tree, branch_nodes_ids, self.variables_dict, self.func_array, goal_nodes)
            else:
                raise Exception('Tree growth version not known')

            self.pop_append(tree, last_evolution='m-bra')
            # print('Mutated a trees branch. Took {:4.2f} sec'.format(time.perf_counter()-time_start))
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

            # 2. search nodes for left and right that can be exchanged. convert_needed
            left_id, right_id, success = self.tree_try_get_swapids(left_tree, right_tree)
            if not success:
                right_id, left_id, success = self.tree_try_get_swapids(right_tree, left_tree)

            left_ids, left_labels, left_aritys, left_xtypes = tree_get_branch_ilax(left_tree, left_id)
            right_ids, right_labels, right_aritys, right_xtypes = tree_get_branch_ilax(right_tree, right_id)

            if not success:
                print_warning('ww', 'Crossover conversion between trees not possible: \n{}\n{}'.format(left_tree, right_tree))
                # left_xtype = xtype_get_from_label(tree_node_get_label(left_tree, left_id), self.variables_dict)
                # conv_to_left, conv_to_right = xtype_get_converters(left_xtype)
                # right_labels.insert(0, conv_to_left)
                # right_aritys.insert(0, 1)
                # left_labels.insert(0, conv_to_right)
                # left_aritys.insert(0, 1)
                return

            left_core = core_from_labels(left_labels, left_aritys, left_xtypes)  # todo this is not necessary, switch branches
            right_core = core_from_labels(right_labels, right_aritys, right_xtypes)

            left_offspring = tree_insert_subtree(left_tree, right_core, left_ids, karoo=True)
            left_offspring = tree_prune_depth(left_offspring, self.config['tree_depth_max'], self.variables_dict)
            self.pop_append(left_offspring, last_evolution='cross')

            right_offspring = tree_insert_subtree(right_tree, left_core, right_ids, karoo=True)
            right_offspring = tree_prune_depth(right_offspring, self.config['tree_depth_max'], self.variables_dict)
            self.pop_append(right_offspring, last_evolution='cross')

        return

    def action_get_xtype(self):
        """
        Return the xtype of the action we want to create.
        sfeh: currently only float kernels supported
        """
        # action_type_one = next(iter(self.action_dict.values()))['type']
        # if action_type_one == 'float':
        #     xtype = '2f'
        # elif action_type_one == 'bool':
        #     xtype = '2b'
        # else:
        #     raise
        return '2f'

    def tree_beautify(self, tree, last_evolution=''):
        """
        The np-tree needs more information than only the expression.
        -> set modifyable nodes (mandatory)

        -> round all constants
        -> try to normalize exponents ('**'). sfeh, not really working.
        -> set last evolution (for analysing gp operators. e.g. if no good trees originate from crossover, something might be wrong)
        -> set xtype for all nodes.
        """

        if tree is None:
            print_warning('ww', 'Tree from last_evolution: {} failed. probably sympify. Continuing.'.format(last_evolution))
        else:
            tree = tree_set_modifyable_nodes(tree, origin_tree=self.origin_tree_get())
            tree = tree_round_constants(tree, self.config['float_accuracy'], karoo=True)
            tree = tree_normalize_exponentiation(tree)
            tree = tree_set_last_evolution(tree, last_evolution)
            tree_check_xtypes(tree)
            tree = tree_set_xtypes(tree, self.variables_dict)  # todo

        return tree

    def tree_check_core_all(self, tree):
        """
        Performs all checks that we currently have
        # sfeh do not use this if trees are safely generated
        # sfeh check meta values in separate method? update those aswell?
        """

        if tree is None:
            return False

        if not tree_check_children(tree):
            print_e('Tree is not consistent:\n{}'.format(tree))
            tree_works = False
        elif not tree_check_types(tree, self.variables_dict):
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
        sfeh: if trees are 100% safely created, tree_check_all() must not be used. Useful when trying out new gp-operators.
        - Enrich the raw tree for the next generation
        - check if the tree is actually valid
        ->
        """

        tree = self.tree_beautify(tree, last_evolution=last_evolution)

        if self.tree_check_core_all(tree):
            tree_ident = tree_get_ident(tree)

            if tree_ident in self.tree_meta:

                tree_meta = self.tree_meta[tree_ident]
                tree = tree_set_meta(tree, tree_meta)
                # tree = tree_set_id(tree, len(self.population_tmp_done))
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
    #   Perform the 3 genetic prog. operations    |
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
    #   Work with trees                           |
    # +++++++++++++++++++++++++++++++++++++++++++++

    def activate_origin_tree(self, tree):
        """

        """
        tree_check_all(tree)

        expr_raw = tree_get_expr_raw(tree, node_id=root_id)
        expr_sym = expr_sympify(expr_raw=expr_raw)
        tree_check_expr(tree)

        # sfeh, this does not work
        # if not tree_check_is_sympified(tree):
        #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
        #                          ''.format(expr_raw, expr_sym))

        if not self.tree_check_core_all(tree):
            print('TODO', tree_labels(tree))
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

        expr_raw = tree_get_expr_raw(tree, node_id=root_id)

        try:
            expr_sym = expr_sympify(expr_raw=expr_raw)
        except Exception as ex:
            raise Exception('Expr could not be sympified: {}'.format(ex))

        fitness_train = eval_tf(expr_sym, self.data_train, self.eval_parameters)['fitness']

        if not check_value_is_real(fitness_train):
            raise Exception('fitness_train is not a real number: {}'.format(fitness_train))

        return fitness_train

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to use evaluate (tensorflow)      |
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
            self.plot_end(data_tuples, path_plots, plt_title='average fitness', plt_y_label='fitness',
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

        data_tuples = sorted(list(self.monitoring_dict['fitness_variance'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='variance in fitness', plt_y_label='variance',
                      linestyle='-',
                      marker='')

        data_tuples = sorted(list(self.monitoring_dict['complexity_variance'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='variance in parsimony', plt_y_label='variance',
                      linestyle='-',
                      marker='')

        data_tuples = sorted(list(self.monitoring_dict['best_candidate'].items()))
        self.plot_end(data_tuples, path_plots, plt_title='best candidate', plt_x_label='generation', plt_y_label='fitness',
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
        return self.variables_dict

    def terminate_run(self, path):
        """
        Program is done after writing all gp_files one last time.
        :param path:
        :return:
        """
        self.file_save_files(path)
        self.file_all_plots(path)
        self.print_g('gg', ' Terminating. \tTime since start: {:4.2f}s'.format(time.perf_counter() - self.time_start))

    # +++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to print_type output information     |
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

        todo max_height=None,  # when creating a plot in every generation, fix the maximum height and width?
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

        plt.savefig(path / '{}.jpg'.format(plt_title))
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


def funcarray_from_list(functions):
    """
    Load all operators ready-to-use from a file
    """

    # rows are the function types (f2f)
    # columns are the arity
    op_array = [[[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []],
                [[], [], [], []]]

    # sfeh make this a np.array

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


def check_value_is_real(fitness):
    """
    Returns bool value if we can use the calculated fitness
    Fitness values might evaluate to weird stuff
    e.g. 'nan' after dividing by zero or (inf) after 20**1234
    nan: fitness == fitness -> False
    inf: fitness is not float('inf') -> False
    """
    return fitness == fitness and fitness is not float('inf')
