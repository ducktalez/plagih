"""
Functions, that might be addable in the future:
"""
import pandas as pd

from plagih.fitness_kernel import *
from plagih.monitoring import plot_gen_performance
from plagih.paretofront import *
from plagih.tree_factory import *

import copy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


def pop_selection_tournament(pop_base, tourn_size=3):
    """

    """
    tree_list = [np.random.choice(pop_base) for _ in range(tourn_size)]
    fintree: 'FinalizedTree' = min(tree_list, key=lambda tree: tree.get_fitness())
    evotree = fintree.get_evotree()
    return copy.deepcopy(evotree)  # sfeh deepcopy not required if it is copied later


class ExplainableGP:
    """

    """

    def __init__(self, conf, rootdir, path_origin, path_data_csv, args):
        self.conf = conf
        # self.develop = args.develop  # more testing and stuff during development phase
        self.time_start = time.perf_counter()
        kernel = RegressionKernel(path_data_csv, conf)  # sfeh relative to rootdir? -> nah -> absolute path... discuss
        self.kernel = kernel
        self.tb = TreeBuilder(kernel.obs_names, self.conf, root_xtype=float)
        self.origin = OriginTree(kernel, path_origin=path_origin)
        # self.evolve_loop = EvolutionLoop()
        self.rootdir = rootdir
        self.gen_id = 0

        self.printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        self.paretofront = []  # not a separate class as requires too much information
        self.pop_next = []
        self.pop_base = []  # sfeh maybe better names

        # Lookup-table for tree(-expressions) and tits fitness/parsimony. Improving runtime a lot!
        self.lut = {}  # using str(), hash() is only temporary, repr() currently not required for LUT info

        # monitoring
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0
        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'parsim_avg', 'parsim_var', 'parsim_best',
                                                'gens_since_last_pareto'])
        # sfeh:open confidence interval NOT symmetric!
        # self.monitor_evolutions = {}
        # for tag in self.evolve_tags:
        #     self.monitor_evolutions[tag] = pd.DataFrame(columns=['fit_avg', 'fit_var', 'time_per_tree'])

        """
        was Evolution class
        The evolve_dict is converted into a list of the population length.
        The evolve_loop is for the regular evolution, the evolve_random is especially required for the first generation.
        """
        evolve_loop = {
            # Reproduction (10%)
            'Repro': {'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.09,
                      'custom_params': {}},
            'Rsympy': {'evolve_name': 'reproduce', 'evolve_rate': 0.05,
                       'custom_params': {'simplify': True}},
            # sfeh 0.03
            'Pareto': {'evolve_name': 'revive paretofront', 'evolve_rate': 0.01,
                       'custom_params': {}},

            # Mutation (25%)
            'Point': {'evolve_name': 'mutate point', 'evolve_rate': 0.05,
                      'custom_params': {}},

            'BranchDF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {
                             'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8),
                                            'p_full': 1.0}}},
            'BranchDG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {
                             'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 5, 1),
                                            'p_full': 0.5}}},
            'BranchNF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {
                             'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3),
                                            'p_full': 1.0}}},
            'BranchNG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {
                             'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 15, 3),
                                            'p_full': 0.5}}},
            'BranchShrink': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                             'custom_params': {
                                 'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0),
                                                'p_full': 0.5}}},

            'FilterBO': {'evolve_name': 'filter optimize', 'evolve_rate': 0.03, 'tourn_size': 5,
                         'custom_params': {'filter_mode': 'branch', 'filter_observations': True}},
            'FilterB': {'evolve_name': 'filter optimize', 'evolve_rate': 0.02, 'tourn_size': 5,
                        'custom_params': {'filter_mode': 'branch', 'filter_observations': False}},
            'FilterP': {'evolve_name': 'filter optimize', 'evolve_rate': 0.03, 'tourn_size': 5,
                        'custom_params': {'filter_mode': 'point', 'filter_observations': True}},

            # Crossover (35%)
            'Xover': {'evolve_name': 'crossover branch', 'evolve_rate': 0.20, 'custom_params': {}},

            # Leftovers are automatically filled with random trees

            # Random (25%)
            'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.05,
                      'custom_params': {
                          'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 3, 4, 1), 'p_full': 1.0}}},
            'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.02,
                      'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1),
                                                       'p_full': 0.5}}},
            'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.10,
                      'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (12, 3, None, 5),
                                                       'p_full': 0.5}}},
            'Rand4': {'evolve_name': 'random trees', 'evolve_rate': 0.10,
                      'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (12, 3, None, 5),
                                                       'p_full': 1.0}}},  # param 'max' can be None
        }
        sum_rates = sum(x['evolve_rate'] for x in evolve_loop.values())
        if sum_rates != 1:
            print_warning('w', f'Evolution rates do not add up to 1: {sum_rates}', print_type=self.conf.print_type)

        self.evolve_loop = self.verify_evolution_params(evolve_loop)

        if self.origin.origin_is_fix:
            evolve_random = {'Rand3o': {'evolve_name': 'random trees', 'evolve_rate': 1.00,
                                        'custom_params': {'build_spec': {'size_mode': 'branch_nodes',
                                                                         'mean_min_max_var': (10, 3, None, 4),
                                                                         'p_full': 1.0}}}}
        else:
            # try:  # sfeh still not sure about this
            #     evolve_random = self.evolve_list_random['from_scratch']
            # except:
            evolve_random = {'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                       'custom_params': {
                                           'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (3.5, 2, 5, 1),
                                                          'p_full': 1.0}}},
                             'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                       'custom_params': {
                                           'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 2, 6, 1),
                                                          'p_full': 0.5}}},
                             'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.40,
                                       'custom_params': {'build_spec': {'size_mode': 'tree_nodes',
                                                                        'mean_min_max_var': (12, 3, None, 5),
                                                                        'p_full': 1.0}}}
                             }
        self.evolve_random = self.verify_evolution_params(evolve_random)

        self.evolve_tags = list(self.evolve_loop.keys()) + list(self.evolve_random.keys())
        return

    def verify_evolution_params(self, evolve_dict):
        """
        Updates tournament size and evolve rates
    
        Example entry of the list could be:
        {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
        'custom_params': {'build_spec': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
        """

        for k, v in evolve_dict.items():
            evolve_dict[k]['tourn_size'] = v.get('tourn_size', self.conf.tourn_size)
            evolve_dict[k]['evolve_num'] = int(v.get('evolve_rate') * self.conf.pop_max)
            v['custom_params'] = v.get('custom_params', {})
        return evolve_dict

    # self.printpl('i', 'Using evolve rates from config')

    def evoloop(self, gen_additionally):
        """
        Start a regular GP run.
        The only other option is to analyze a loaded run.
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.gen_id + gen_additionally)
            self.printpl('i',
                         f'Adding {gen_additionally} more generations in gen {self.gen_id}, '
                         f'increasing gen_max from {printdummy} to {self.conf.gen_max}.')

        # sfeh check if any roots
        # yaml_dump(self.rootdir / 'used_config.yaml', self.conf, print_type=self.print_type)

        # self.pop.gens_since_last_pareto = 0

        while self.gen_id <= self.conf.gen_max and not self.run_custom_exit_condition():
            self.time_genstart = time.perf_counter()  # sfeh here?

            if self.gen_id == 0:
                self.printpl('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')  # sfeh debug
                self.gen_create_initial()  # sfeh stattdessen einfach checken, ob die letzte population leer ist und info/warnung: neue generation?
            else:
                # This might be a solution for multiprocessing: You can avoid this situation by calling
                # multiprocessing.Process before you load your huge data. Then the additional memory allocations will
                # not be reflected in the child process when you load the data in the parent. sfeh: In python 3.8,
                # this might be availably: multiprocessing.shared_memory
                # https://docs.python.org/3/library/multiprocessing.shared_memory.html sfeh: check memory usage!
                # should not scale with the number of processes, only one pop_base is required, it does not change.

                self.conf.mp_cores = 1  # sfeh wasd
                if self.conf.mp_cores >= 2:
                    pass
                    # # sfeh asd
                    # mp.Process()  # sfeh maybe good for memory? https://stackoverflow.com/questions/14749897/python-multiprocessing-memory-usage
                    # print(f'Trying to make parallel new population: {mp.cpu_count()}')
                    # with mp.Pool(min(mp.cpu_count(), self.mp_cores)) as p:
                    #     evolve_list = [[tag, evolve_specs] for tag, evolve_specs in self.evolve_loop.items()]
                    #     results = p.map(fun, evolve_list)
                    # time_evolve = time.perf_counter()
                else:
                    self.gen_create_next()

            if len(self.paretofront) == 0:
                self.paretofront = [self.pop_next[0]]  # initialize
                printez('a',
                        f'Starting a new paretofront with parsimony: {self.paretofront[0].get_parsimony()} fitness: {self.paretofront[0].get_fitness():6.4f}',
                        self.conf.print_type)

            tmp_pareto = pareto_from_pop(self.pop_next)  # sfeh:optional paretofront in each generation?
            self.paretofront = self.run_update_paretofront(self.paretofront)

            gen_time = time.perf_counter() - self.time_genstart
            pop_analysis_dict = pop_analyze(self.pop_next)
            pop_analysis_dict['gen_time'] = gen_time
            pop_analysis_dict['gens_since_last_pareto'] = self.gens_since_last_pareto
            self.monitor_df.loc[self.gen_id] = pop_analysis_dict

            pop_tree_analysis(self.pop_next)

            self.printpl('gg',
                         f"Created {len(self.pop_next)}/{self.conf.pop_max} ({pop_analysis_dict['pop_unique']} unique) "
                         f"in generation {self.gen_id}. Gen took {gen_time:4.2f}s")

            self.pop_base = self.pop_next[:]  # or: deepcopy
            self.pop_next = []

            self.printpl('ggg', f'Gen {self.gen_id} took: {time.perf_counter() - self.time_genstart:4.2f}.')
            self.evoloop_monitoring_scheduled_io(self.conf, self.gen_id)
            self.gen_id += 1
        else:
            self.printpl('g',
                         f'Done after Generation {self.gen_id}.\n'
                         f'Time since start: {time.perf_counter() - self.time_start:4.2f}s')

        self.run_backup(mode='save')

        return

    def run_update_paretofront(self, paretofront):
        """
        CAUTION: This Function was tried to be separated many times now. it never worked.
        This was tried =3= times now. Please increase the counter when you try.
        sfeh:debug, might be faulty
        sfeh:discuss pareto-efficient, but different pareto entries?
        """
        pop_parcandidates = pareto_from_pop(self.pop_next)  # pareto-candidates in the pop, renamed to be clear

        for fintree in pop_parcandidates:
            success = False
            fit = fintree.get_fitness()
            par = fintree.get_parsimony()

            if par < paretofront[0].get_parsimony():
                printyeah('a', f'Paretofront: New simplest entry. parsimony: {par} fitness: {fit:6.4f}, '
                               f'old simplest entry had {paretofront[0].get_parsimony()}')
                success = True

            # if all([self.fitness_compare(fit, p.get_fitness()) for p in paretofront]):  # sfeh-kernel
            elif fit < paretofront[-1].get_fitness():
                printyeah('a', f'Paretofront: New fittest entry. parsimony: {par} fitness: {fit:6.4f}')
                success = True
            else:
                for p in paretofront:
                    if par >= p.get_parsimony():
                        continue
                    else:
                        if fit < p.get_fitness():
                            success = True

            if success:
                self.gens_since_last_pareto = 0
                printyeah('a', f'Paretofront: New entry. parsimony: {par} fitness: {fit:6.4f}')

                # sfeh: trying to simplify the tree for a even improved pareto # todo
                try:
                    symtree = evolve_reduce_simplify(fintree.get_evotree(), force=True)
                    symmeta = self.finalize_tree_get_meta(symtree)
                    symtree_fin = FinalizedTree(symtree, symmeta)
                    if symtree_fin.get_parsimony() < fintree.get_parsimony():
                        printyeah('a', f'Paretofront: Even further simplified! '
                                       f'{symtree_fin.get_parsimony()} < {fintree.get_parsimony()}')

                    self.genloop_performance_append_tree(symtree, tag='sfeh-sympifyed_pareto')
                except Exception as ex:
                    raise  # todo debug
                obsolete_entries = [x for x in paretofront if
                                    x.get_fitness() > fintree.get_fitness() and x.get_parsimony() >= fintree.get_parsimony()]
                if obsolete_entries:
                    printyeah('a', f'Paretofront: Removing obsolete entries {[str(x) for x in obsolete_entries]}')
                paretofront = [x for x in paretofront if x not in obsolete_entries]
                paretofront.append(fintree)
                paretofront = pareto_sort(paretofront)

        return paretofront

    def gen_create_initial(self):
        """
        was: gen_create_initial
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta fintree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        if self.origin.existing:
            self.pop_next.append(self.origin.fintree)
            # self.pop_append_evotree(self.origin.fintree)  # sfeh why not :P
            # self.pareto.pareto_insert(self.origin)  # the origin fintree is the only candidate (automatically added)
        else:
            total_rate = sum([x['evolve_rate'] for x in self.evolve_random.values()])

            for tag, evolve_specs in self.evolve_random.items():
                evolve_num = int(self.conf.pop_max * (evolve_specs['evolve_rate'] / total_rate))
                custom_params = evolve_specs.get('custom_params')

                for nn in range(evolve_num):
                    evotree = self.tb.pop_random(custom_params, origin=self.origin)
                    self.genloop_performance_append_tree(evotree, tag)
        return

    def gen_create_next(self):
        """
        Creates all new Generations by applying the evolutions in the evolve-loop.

        brainstorm:
        - the fintree can be reproduced (selection), random/new, olymp-reproduction
        - (1 fintree) mutations can affect a point, branch, terminal nodes
        - (2 trees) can make a crossover
        """
        # All gp creators: name, function, num of trees from tournament selection

        global custom_params
        for tag, evolve_specs in self.evolve_loop.items():  # all selected gp mutations

            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            custom_params = evolve_specs.get('custom_params')

            self.printpl('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')
            if evolve_name == 'reproduce':
                """
                Reproduction
                """
                for nn in range(evolve_num):
                    evotree = pop_selection_tournament(self.pop_base, tourn_size=tourn_size)
                    if custom_params.get('simplify'):  # sfeh==>debug
                        try:
                            evotree = evolve_reduce_simplify(evotree, completely=False)
                            # fintree.meta.last_evolution = tag
                        except Exception as ex:
                            print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.conf.print_type)
                            # raise ex  # sfeh debug

                    self.genloop_performance_append_tree(evotree, tag)  # pop_append_evotree anyways... it was worth a try :P

            elif evolve_name == 'mutate point':
                """
                Point mutation, one point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    evotree = pop_selection_tournament(self.pop_base, tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_point(evotree)
                    # fintree.meta.last_evolution = tag
                    self.genloop_performance_append_tree(evotree, tag)

            elif evolve_name == 'mutate branch':
                """
                One node is replaced with a random branch
                """
                for nn in range(evolve_num):
                    _, size_mode, mean_min_max_var, p_full = helper_evolve_params_branch(custom_params,
                                                                                         tree_depth_max=self.conf.tree_depth_max,
                                                                                         parsimony_max=self.conf.parsimony_max)
                    evotree = pop_selection_tournament(self.pop_base, tourn_size=tourn_size)
                    build_size = choose_build_size(size_mode, mean_min_max_var, tree=evotree, force='branch')
                    # sfeh:test options, depth, in this case

                    if size_mode == 'branch_depth':  # building a branch to a depth
                        evotree = self.tb.evolve_mutate_branch_depth(evotree, build_size, p_full=p_full)

                    elif size_mode == 'branch_nodes':  # building a branch with an amount of nodes
                        evotree = self.tb.evolve_mutate_branch_nodes(evotree, build_size, p_full=p_full)
                    else:
                        raise
                    evotree = self.tb.evolve_prune(evotree)  # sfeh:performance runtime-wise, do this somewhere else
                    self.genloop_performance_append_tree(evotree, tag)

            elif evolve_name == 'crossover branch':
                for nn in range(int(evolve_num / 2)):  # two childs
                    """
                    swap branches of two trees
                    - select parent a and b
                    - select swappable branche for a_parent from b_parent
                        - select a node in a (and crossover here, no matter what)
                    - delete a_parent branch and pareto_insert b_parent branch (which tactic?)
                    sfeh:idea into main fintree?
                    """
                    atree = pop_selection_tournament(self.pop_base, tourn_size=tourn_size)
                    btree = pop_selection_tournament(self.pop_base, tourn_size=tourn_size)

                    # 1. two parents
                    # 2. search nodes for left and right that can be exchanged. convert_needed
                    try:
                        atree, btree = self.tb.evolve_crossover(atree, btree)
                        # fintree = self.tb.finalize(etree)  # ==>state
                        self.genloop_performance_append_tree(atree, tag=tag)
                        self.genloop_performance_append_tree(btree, tag=tag)
                    except Exception as ex:
                        print_warning("www", f'Crossover failed: {ex}')
                        # sfeh: mostly 1-noded trees

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    evotree = pop_selection_tournament(self.pop_base, tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_filter_random(evotree, custom_params)
                    self.genloop_performance_append_tree(evotree, tag)

            elif evolve_name == 'revive paretofront':

                for nn in range(evolve_num):
                    fintree = np.random.choice(self.paretofront)
                    self.pop_next.append(fintree)

            elif evolve_name == 'random trees':

                for nn in range(evolve_num):
                    evotree = self.tb.pop_random(custom_params, origin=self.origin)
                    self.genloop_performance_append_tree(evotree, tag=tag)

            else:
                print_e(f"Evolution not known: '{evolve_name}'")

        missing_trees = self.conf.pop_max - len(self.pop_next)
        if missing_trees > 0:
            if missing_trees > 0.05 * self.conf.pop_max:
                self.printpl('ii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            else:
                self.printpl('iii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')

            for _ in range(self.conf.pop_max - len(self.pop_next)):
                # sfeh aka create trees here if desired. Is this desired? ->correct tag, ->correct randoms
                # while len(self.pop_next) < self.conf.pop_max:
                evotree = self.tb.pop_random(custom_params, origin=self.origin)
                self.genloop_performance_append_tree(evotree, tag='taggyRnd')  # sfeh

        return

    def evoloop_monitoring_plots(self):
        """
        Create all run-related analysis plots in the root directory
        """
        try:
            path_monitoring = self.rootdir / 'monitoring.png'
            plot_gen_performance(self.monitor_df, self.conf.name, path_monitoring)  # largest plot analysing the
            self.printpl('f', f"monitoring: {path_monitoring}")  # .as_posix()
            pareto_plot(self.paretofront, self.rootdir, self.conf)
        except Exception as ex:
            self.printpl("e", f'Could not create plots: {ex}\n')

    def evoloop_monitoring_scheduled_io(self, conf, gen_id):
        """
        every x generations, save a backup and/or save plots
        """
        plot_gen = int(conf.period.get('gen_plots', 10))
        if gen_id >= plot_gen and gen_id % plot_gen == 0:
            self.evoloop_monitoring_plots()

        save_gen = int(conf.period.get('gen_save', 10))
        if gen_id >= save_gen and gen_id % save_gen == 0 or gen_id == 10:
            self.run_backup(mode='save')

    def run_backup(self, path_load_custom_backup=None, mode='load'):
        """
        load/safe backup of a run. Both done in this method
        """

        path_backup = self.rootdir / 'backup/backup.pkl'

        if mode == 'save':
            """
            automatically saves everything important after a certain amount of time
            - save the paretofront front (custom_done)
            - save the last generation (custom_done)
            - Save valuable meta-data_csv_path: current generation (custom_done)
            """
            # sfeh:discuss: saving the yaml is not required
            path_backup_yaml = path_make_dir(path_backup)
            yaml_dump(path_backup_yaml, self.conf, print_type=self.conf.print_type)

            # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
            run_backup_data = {}, self.gen_id, self.pop_base, self.paretofront, self.monitor_df  # sfeh use this later, help_dict
            path_backup = path_make_dir(path_backup)
            pickle_dump(path_backup, run_backup_data)
            # sfeh:debug

        elif mode == 'load':
            # run_backup_load(self, argpath_backup):
            """
            If a backup-file is found...
            Loading the state of the run from the pickle file
            """
            path_backup = path_load_custom_backup or self.rootdir / 'backup/backup.pkl'

            if Path.is_file(path_backup):
                self.printpl('g', f'Loading data from backup-file {path_backup}')
                try:
                    with Path.open(path_backup, 'rb') as file:
                        run_data = pickle.load(file)
                except NotImplementedError as ex:
                    raise Exception(f'NotImplementedError: {ex}')
                except EOFError as ex:
                    raise Exception(f'EOFError: \n{ex}')

                help_dict, self.gen_id, self.pop_base, self.paretofront, self.monitor_df = run_data
                self.printpl('g', f'Successfully loaded backup file. Generation: {self.gen_id}')

            else:
                raise FileNotFoundError(f'No backup-file found at {path_backup}.')

        else:
            raise

    def run_custom_exit_condition(self):
        """
        Special condition to exit the evolve-loop
        1. when >100 generations, no new paretofront paretofront were found
        """
        try:
            if self.gens_since_last_pareto > 100:  # .iloc[-1] > 100:  # sfeh discussion
                print('SFEH This condition made your program exit!')
                return True
            else:
                return False
        except Exception:
            return False

    def plot_evolve_performance(self):
        """
        Plots for each tag in the evolution list
        (too much, i guess)
        sfeh: this should be saved within the trees. Everything else is a waste of memory!
        todo
        """
        try:
            with plt.rc_context(rc=pyplot_rc_tex):
                fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 9), sharex='all')  # , gridspec_kw={'height_ratios': [1,1,1]}
                fig.tight_layout()
                for tag in self.evolve_tags:
                    # ['fitness_train', 'parsimony', 'lentree', 'evolve_num', 'count']
                    axs[0].plot(self.monitor_df[tag]['fitness_train'], label=f'{tag}')
                    axs[1].plot(self.monitor_evol[tag]['parsimony'], label=f'{tag}')
                    axs[2].plot((self.monitor_evol[tag]['lentree'] / self.monitor_evol[tag]['evolve_num']), label=f'{tag}')

                plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
                path = self.rootdir / f'monitoring_evolutions.pdf'
                fig.savefig(path)
                self.printpl('f', f"monitoring_evolutions (pdf): {path}")  # .as_posix()

        except Exception as ex:
            print_e(f'plot_evolution_analysis failed because of: {ex}')

    def genloop_performance_append_tree(self, evotree: Node, tag):
        """
        Safely append a fintree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the fintree is refurbished.
        - Enrich the raw fintree for the next generation
        - check if the fintree is actually valid
        ->
        """
        if DEBUG_DUMMY:
            # if trees are 100% safely created, tree checks are not required. Useful when trying out new gp-operators.
            self.tb.check_all(evotree, fatal=True)

        try:
            meta = self.lut[str(evotree)]
        except KeyError as ex:
            try:
                meta = self.finalize_tree_get_meta(evotree)
            except ValueError as ex:
                print_warning('wwww', f'Could not append fintree to population because: {ex}', print_type=self.conf.print_type)
                return  # sfeh:print
            except Exception as ex:
                print_warning('www', f'Could not append tree to population because: {ex}\n'
                                     f'=>tree: {evotree}', print_type=self.conf.print_type)
                return

        fintree = FinalizedTree(evotree, meta)
        fintree.append_tag(tag)
        self.pop_next.append(fintree)

    def finalize_tree_get_meta(self, evotree):
        """
        Very fast eval-version that only computes fitness_train of the train data.
        tree_eval_complete gives more options
        Evaluating the fitness_train of a fintree.
        - extract the expression the fintree is holding
        - sympify the expression
        - (if sympify fails, evaluating does not make sense! Check sympify errors)
        - (sfeh: if sympify fails because of inf or zoo, tf could maybe still work due to save-tf-division)

        fintree->fintree!
        ===
        # try:
        #     self.check_all()
        # attention: regular hashes may change between python runs. do not save anything on their hash values <.<

        ===

        Returns bool value if we can use the calculated fitness_train
        Fitness values might evaluate to weird stuff
        e.g. 'nan' after dividing by zero or (inf) after 20**1234
        nan: fitness_train == fitness_train -> False
        inf: fitness_train is not float('inf') -> False
        """
        parsimony = evotree.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin.fintree)
        if parsimony > self.conf.parsimony_max:
            # sfeh last_evolution leads to error atm
            # print_warning('wwww', f'Parsimony too high, last evolution: {fintree.meta.last_evolution}', print_type=self.print_type)  # sfeh care about wwww. should not
            raise ValueError(f'Tree too complex: {parsimony} > {self.conf.parsimony_max}')

        expr_raw = evotree.eval_expr()

        try:
            expr_sym = expr_sympify(expr_raw)
        except ValueError as ex:
            print_warning('wwww', f'fintree: {evotree} had ex: {ex}', print_type=self.conf.print_type)
            raise ValueError(ex)

        treeobs = evotree.get_observation_list()
        try:
            fitness = self.kernel.eval_tf(expr_sym, treeobs, only_fitness=True)
            # if fitness != fitness or fitness == float('inf'):
            #     raise Exception(f"fitness_train is: '{fitness}'")  # eg very wrong values that exceed the float-range
        except Exception as ex:
            raise Exception(f'eval-ex: {ex}')

        meta = TreeMeta(fitness, parsimony, expr_raw, expr_sym)
        self.lut[str(evotree)] = meta
        return meta

    def printpl(self, message_type, message_str):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        message_type options can be found in config
        """
        printez(message_type, message_str, print_type=self.conf.print_type)
        return


def pop_analyze(popul):
    """
    Analysing the population (done in each generation)
    - amount of trees
    - fittest fintree
    - average fitness_train
    - average fintree parsimony
    """
    # popul = self.pop_next

    if len(popul) == 0:
        raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

    pop_fitness = [tree.get_fitness() for tree in popul]
    pop_parsim = [tree.get_parsimony() for tree in popul]
    pop_treelen = [len(fintree.tree) for fintree in popul]
    pop_fitness_best = np.min(pop_fitness)
    pop_unique = len(set([hash(x) for x in popul]))  # sfeh analyze this?
    result = {'pop_len': len(popul),
              'pop_unique': pop_unique,
              'fit_avg': np.average(pop_fitness),
              'fit_var': np.std(pop_fitness),
              'fit_best': pop_fitness_best,
              'parsim_avg': np.average(pop_parsim),
              'parsim_var': np.std(pop_parsim),
              'parsim_best': np.min(pop_treelen),
              # 'time': gen_time,  # todo
              # 'gens_since_last_pareto': self.gens_since_last_pareto  # was here!
              }  # sfeh delete?
    return result


def pop_tree_analysis(popul):
    """

    """
    evo_perfo = {}
    for tree in popul:
        tag = tree.get_last_evolution()
        fit = tree.get_fitness()
        parsim = tree.get_parsimony()

        if evo_perfo.get(tag) is None:
            evo_perfo[tag] = {'fitness': [], 'parsim': []}

        evo_perfo[tag]['fitness'].append(fit)
        evo_perfo[tag]['parsim'].append(parsim)

    for tag, v in evo_perfo.items():
        evo_perfo[tag]['fitness'] = np.average(evo_perfo[tag]['fitness'])
        evo_perfo[tag]['parsim'] = np.average(evo_perfo[tag]['parsim'])
    # todo
    return


def printpl(message_type, message_str, print_type):
    """
    Lightweight print function.
    Instead of checking if you should print every time, this is done here.
    message_type options can be found in config
    """
    printez(message_type, message_str, print_type=print_type)
    return


if __name__ == '__main__':
    """
    Alpha tests
    """
    tb = TreeBuilder(obs_names=['cartPos', 'cartVel'], conf=None)
    # t1 = tb.invent_core_depth(float, 3, p_full=0.5)
    # tree2 = tb.evolve_mutate_point(t1)
    # t1 = tb.invent_core_depth(float, 3, p_full=0.9)
    # t2 = tb.invent_core_depth(float, 3, p_full=0.9)
    # t1, t2 = tb.evolve_crossover(t1, t2)
    # for _ in range(5):
    #     print('x.D', t1, '===', t2)
    #     t1, t2 = tb.evolve_crossover(t1, t2)
    #     print('x~D', t1, '===', t2)

    # nstr = "['Ifte', ['<', ['*', [2.85], ['cartVel']], ['Square', ['cartVel']]], ['*', ['cartPos'], ['*', ['cartPos'], [0.014]]], ['/', [2.0], ['cartPos']]]"
    # nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    nstr = '["+:fix",["-:fix",["Ifte",["True"],["sin",["2"]],["/",["2.043"],["4"]]],["cartVel"]],["-1.3"]]'
    tr = tree_from_nested_string(nstr)
    check_expression_reconstruction(tr)
