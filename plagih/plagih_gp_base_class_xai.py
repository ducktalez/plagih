"""

Functions, that might be addable in the future:
"""

from plagih.evolution import EvolutionLoop
from plagih.paretofront import ParetoFront
from plagih.tree_factory import *
from plagih.viz_with_latex import *

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees

import copy

import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow
import numpy as np

tensorflow.compat.v1.disable_eager_execution()  # sfeh damn what was this line good for?


class Population:

    def __init__(self, kernel, conf, origin=None):
        self.kernel = kernel
        self.conf.gen_id = 0
        self.size = 1000
        self.pop_next = []
        self.pop_base = []  # sfeh maybe better names

        self.conf = conf  # todo

        self.lut = {}
        self.origin = origin
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0
        self.print_type_g = conf.print_type

        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'complexity_avg', 'complexity_var', 'complexity_stderr',
                                                'gens_since_last_pareto'])

    def print_g(self, message_type, text):
        """
        todo
        print informations about generations (g) progress.
        Very common print, showing the progress of the generations.
        always printing the time since start. used to be colored blue.
        """
        printez(message_type, text, print_type=self.print_type_g)
        return

    def __len__(self):
        return len(self.pop_next)  # todo hmmm

    # def plot_evolve_performance(self):
    #     """
    #     Plots for each tag in the evolution list
    #     (too much, i guess)
    #     sfeh: this should be saved within the trees. Everything else is a waste of memory!
    #     """
    #     try:
    #         with plt.rc_context(rc=pyplot_rc_tex):
    #             fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 9), sharex='all')  # , gridspec_kw={'height_ratios': [1,1,1]}
    #             fig.tight_layout()
    #             for tag in self.evolve_tags:
    #                 # ['fitness_train', 'parsimony', 'lentree', 'evolve_num', 'count']
    #                 axs[0].plot(self.monitor_evol[tag]['fitness_train'], label=f'{tag}')
    #                 axs[1].plot(self.monitor_evol[tag]['parsimony'], label=f'{tag}')
    #                 axs[2].plot((self.monitor_evol[tag]['lentree'] / self.monitor_evol[tag]['evolve_num']), label=f'{tag}')
    #
    #             plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
    #             path = self.ui.rootdir / f'monitoring_evolutions.pdf'
    #             fig.savefig(path)
    #             self.printpl('f', f"monitoring_evolutions (pdf): {path.as_posix()}")
    #
    #     except Exception as ex:
    #         print_e(f'plot_evolution_analysis failed because of: {ex}')

    def pass_population(self):
        """

        """
        self.pop_base = self.pop_next[:]  # otherwise: deepcopy
        self.pop_next = []

    def pop_analyze(self):
        """
        Analysing this generation
        - amount of trees
        - fittest tree
        - average fitness_train
        - average tree parsimony
        """
        popul = self.pop_next

        if len(popul) == 0:
            raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

        pop_fitness = [tree.get_fitness() for tree in popul]

        # tmp_evol_performance = dict.fromkeys(self.monitor_evol.keys(), pd.DataFrame(columns=['fitness_train', 'parsimony', 'lentree']))
        # for tree in popul:
        #     last_evol = tree.meta.last_evolution
        #     if last_evol in self.evolve_tags:
        #         row = {'fitness_train': tree.meta.fitness_train,
        #                'parsimony': tree.meta.parsimony,
        #                'lentree': len(tree)}
        #         tmp_evol_performance[last_evol].loc[self.conf.gen_id] = row
        #
        # for last_evol, evodata in tmp_evol_performance.items():
        #     if last_evol in self.evolve_loop:
        #         try:
        #             row = {'fitness_avg': evodata['fitness_train'].mean(),
        #                    'parsimony_avg': evodata['parsimony'].mean(),
        #                    'lentree_avg': evodata['lentree'].mean(),
        #                    'evolve_num': self.evolve_loop[last_evol]['evolve_num'],
        #                    'count': len(tmp_evol_performance[last_evol])}
        #             self.monitor_evol[last_evol].append_tree(row, ignore_index=True)  #
        #             # sfeh fitness_train - last fitness_train?
        #         except Exception as ex:
        #             print_e(f'Could not save evol_performance analysis. {ex}')
        #     else:
        #         if last_evol != 'origin' and last_evol != 'Rand3o':
        #             print_warning('w', f'delete_this, sfeh, okay when the following is origin: {last_evol}')

        pop_parsim = [tree.get_parsimony() for tree in popul]
        pop_treelen = [len(tree.tree) for tree in popul]

        pop_fitness_best = self.kernel.np_best_fitness(pop_fitness)
        try:
            self.best_fitness = self.kernel.get_fitness_extreme_function(pop_fitness_best, self.best_fitness)
        except:
            self.best_fitness = pop_fitness_best

        unique_tree_count = len(set([hash(x) for x in popul]))  # sfeh analyze this?

        gen_time = time.perf_counter() - self.time_genstart

        self.monitor_df.loc[self.conf.gen_id] = {'pop_len': len(popul),
                                                 'pop_unique': unique_tree_count,
                                                 'fit_avg': np.average(pop_fitness),
                                                 'fit_std': np.std(pop_fitness),
                                                 'fit_best': self.best_fitness,
                                                 'parsim_avg': np.average(pop_parsim),
                                                 'parsim_std': np.std(pop_parsim),
                                                 'complexity_avg': np.var(pop_treelen),
                                                 'time': gen_time,
                                                 'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh version1 delete this shit

        self.print_g('gg', f'Created {len(popul)}/{self.conf.pop_max} ({unique_tree_count} unique) in generation {self.conf.gen_id}. Gen took {gen_time:4.2f}s')
        return

    def selection_tournament(self, tourn_size=3):
        """

        """
        tree_list = [np.random.choice(self.pop_base) for _ in range(tourn_size)]
        fintree: 'TreeFinalized' = self.kernel.get_fitness_extreme_function(tree_list, key=lambda tree: tree.get_fitness())
        evotree = fintree.get_evotree()
        return copy.deepcopy(evotree)  # sfeh deepcopy not required if it is copied later

    def finalize_tree(self, tree):
        """
        tree->fintree!
        ===
        # try:
        #     self.check_all()
        # attention: regular hashes may change between python runs. do not save anything on their hash values <.<

        ===
        Very fast eval-version that only computes fitness_train of the train data.
        tree_eval_complete gives more options
        Evaluating the fitness_train of a tree.
        - extract the expression the tree is holding
        - sympify the expression
        - (if sympify fails, evaluating does not make sense! Check sympify errors)
        - (sfeh: if sympify fails because of inf or zoo, tf could maybe still work due to save-tf-division)

        Returns bool value if we can use the calculated fitness_train
        Fitness values might evaluate to weird stuff
        e.g. 'nan' after dividing by zero or (inf) after 20**1234
        nan: fitness_train == fitness_train -> False
        inf: fitness_train is not float('inf') -> False
        """

        try:
            meta = self.lut[hash(tree)]
        except KeyError:
            parsimony = tree.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
            if parsimony > self.conf.parsimony_max:
                # sfeh last_evolution leads to error atm
                # print_warning('wwww', f'Parsimony too high, last evolution: {tree.meta.last_evolution}', print_type=self.print_type_g)  # sfeh care about wwww. should not
                raise Exception('Tree too complex')
            try:
                expr_raw = tree.eval_expr_sym()
                expr_sym = expr_sympify(expr_raw)
                treeobs = tree.get_observation_list()
                fitness = round(self.kernel.eval_tf(expr_sym, treeobs, only_fitness=True), self.conf.precision)
                if fitness != fitness or fitness == float('inf'):
                    raise Exception(f"fitness_train is: '{fitness}'")  # happens, eg when values are soo wrong that it leaves the float-range
                meta = TreeMeta(fitness, parsimony, expr_raw, expr_sym, STATE_EVALUATED)
                self.lut[hash(tree)] = meta
            except Exception as ex:
                print_warning('wwww', f'Exception while evaluating: {ex}, tree: {tree}.', print_type=self.print_type_g)
                raise Exception(f'eval: {ex}')

        fintree = TreeFinalized(tree, meta)
        return copy.deepcopy(fintree)  # sfeh guess no deepcopy is required

    def append_tree(self, tree: Node, tag=None):
        """
        was: def pop_append(self, tree: Node):
            # sfeh this check might be important...
        Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the tree is refurbished.
        sfeh: if trees are 100% safely created, tree_check_deep() must not be used. Useful when trying out new gp-operators.
        - Enrich the raw tree for the next generation
        - check if the tree is actually valid
        ->
        """
        try:
            fintree = self.finalize_tree(tree)
            fintree.append_tag(tag)
            self.pop_next.append(fintree)
        except Exception as ex:
            logging.warning(f'Could not append tree to population because: {ex} =>tree: {tree}')
            # print_warning('w', f'tree failed the quick check. last-mod: {self.meta.last_evolution}. Reason:\n{ex}', print_type=self.ui.print_type)
            return

    def plot_gen_performance(self):
        """
        All monitoring infos
        sfeh den shit in Funktionen aufteilen
        """
        with plt.rc_context(rc={'axes.grid': True}):
            fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]}, sharex='all')  # , figsize=(9, 9)
            plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
            xx = list(self.monitor_df.index)

            axs0 = axs[0]
            axs0.plot(self.monitor_df['fit_avg'], marker='', label='Regression error (average)')
            # todo: not just the stderr on both sides...
            # try:
            #     avg = self.monitor_df['fit_avg']
            #     std = self.monitor_df['fit_std']
            #     axs0.fill_between(xx, avg - std, avg + std, alpha=0.2)  # axs0.set_title('Regression Error (average)')  # sfeh not stderr... upper/lower bound?
            # except Exception as ex:
            #     raise Exception(f'Delete this. were there any problems? {ex}')
            axs0.step(x=xx, y=self.monitor_df['fit_best'], linestyle='dashed', marker='', where='post', color='g', label='Best candidate')  # , label=ax_label
            axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

            axs0_twin = axs0.twinx()
            axs0_twin.plot(xx, self.monitor_df['gens_since_last_pareto'], color='tab:gray', label='Generations since last paretos entry', linestyle='dashed', marker='')  # linestyle='None'
            axs0_twin.tick_params(axis='y', labelcolor='tab:gray')
            try:
                axs0_twin.set_ylim(ymin=0, ymax=max(self.monitor_df['gens_since_last_pareto'].max() or 1, 50))
            except Exception as ex:
                try:
                    print_e(f'damn setting ylim not working sfeh :s {ex}')
                    axs0_twin.set_ylim(ymin=0, ymax=max(self.monitor_df['gens_since_last_pareto'].notnull().max() or 1, 50))
                    # print(self.monitor_df['gens_since_last_pareto'].notnull().max())
                except Exception as ex2:
                    print_e(f'damn setting ylim not working, version 2! {ex2}')
                    axs0_twin.set_ylim(ymin=0, ymax=50)

            axs0_twin.legend(loc='lower right')

            axs1 = axs[1]
            axs1.plot(self.monitor_df['parsim_avg'], label='Complexity (average)')
            # self.conf.complexity_measure

            try:
                p_avg = self.monitor_df['parsim_avg']
                p_std = self.monitor_df['parsim_std']
                axs1.fill_between(xx, p_avg - p_std, p_avg + p_std, alpha=0.2)  # axs1.set_title('TED (average)')
            except Exception as ex:
                raise Exception(f'Delete this if no raise since some time. {ex}')
            axs1.set_ylim(ymin=0), axs1.legend(loc='lower left')

            axs2 = axs[2]
            axs2.plot(self.monitor_df['pop_len'], label='pop_list size')
            axs2.plot(self.monitor_df['pop_unique'], label='unique')
            axs2.margins(y=0.25), axs2.set_ylim(ymin=0), axs2.legend(loc='lower left')

            axs3 = axs[3]
            between_outliers = self.monitor_df['time'].between(0, 2 * self.monitor_df['time'].mean())
            axs3.plot(self.monitor_df['time'][between_outliers], label='time (s)')  # sfeh could be a better rule...
            axs3.set_ylim(ymin=0), axs3.legend(loc='lower left')

            # Top level style
            axs3.set_xlim(xmin=0, xmax=max(xx)), axs3.set_xlabel('generations')
            axs0.set_title(f'monitoring gp generations {self.conf.name}')  # sfeh
            fig.tight_layout()
            path = self.rootdir / f'monitoring.png'  # -{self.conf.name}
            fig.savefig(path)
            printez('f', f"monitoring: {path.as_posix()}", print_type=self.print_type_g)
            # plt.close('all')


class OriginTree:
    def __init__(self, kernel, tree=None, path_origin=None):
        if path_origin:
            """
            The origin tree (which was already loaded) gets activated for its use in the GP-process
            """

            with Path.open(path_origin, newline='') as file:
                file.read()
                origin_tree = Node(label=Observation('Todo'))  # todo
                origin_meta = TreeMeta()
                origin_tree = TreeFinalized(origin_tree, origin_meta)
            try:
                expr_sym = origin_tree.meta.expr_sym  # todo
            except Exception as sympex:
                raise Exception(f'Loaded origin_tree already failed because of: {sympex}')

            # sfeh, this does not work
            # if not tree_check_is_sympified(tree):
            #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
            #                          ''.format(expr_raw, expr_sym))

            used_observations = origin_tree.tree.get_observation_list()
            tf_origin_results = kernel.eval_tf(expr_sym, used_observations)
            fitness_train = round(tf_origin_results['mean_error'], kernel.precision)  # fitness_train currently IS the mean error
            if kernel.exploration_risk:
                kernel.origin_results = tf_origin_results['results_kernel']  # after getting the origin-results, these informations can be updated

            origin_tree.set_fitness(fitness_train)
            origin_tree.set_parsimony(0)

            # self.printpl('gg', f'Loading origin tree, regr. error {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')
        try:
            self.tree = tree
            self.meta = TreeMeta()

            self.tree.meta.last_evolution = 'origin'
            self.origin_is_fix = self.tree.is_fix  # sfeh the root node!?!
        except Exception as ex:
            return
        self.origin_is_fix = False

    def exists(self):
        return


class ExplainableGP(object):
    """

    """

    def __init__(self, rootdir, conf):
        self.conf = conf
        self.time_start = time.perf_counter()
        self.kernel = RegressionKernel(conf)
        self.tb = TreeBuilder(self.kernel.obs_names, root_xtype=float)
        self.origin = OriginTree(self.kernel, self.conf.path_origin)
        self.evolve_loop = EvolutionLoop(self.origin.origin_is_fix, self.conf)
        self.pop = Population(self.kernel, conf, origin=self.origin)
        self.pareto = ParetoFront(self.kernel.fitness_compare, self.conf, origin=self.origin)
        self.printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        self.rootdir = rootdir
        self.conf.set_rootdir()  # sfeh better solution?

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')
        return

    def gen_create_initial(self):
        """
        was: gen_create_initial
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        if self.origin.exists():
            self.pop.append_tree(self.origin.tree)  # sfeh why not :P
            # self.pareto.insert(self.origin)  # the origin tree is the only candidate (automatically added, i hope)
        else:
            total_rate = sum([x['evolve_rate'] for x in self.evolve_loop.evolve_random.values()])

            for tag, evolve_specs in self.evolve_loop.evolve_random.items():
                evolve_num = int(self.conf.pop_max * (evolve_specs['evolve_rate'] / total_rate))
                call_params = evolve_specs.get('custom_params')

                for nn in range(evolve_num):
                    if self.origin.origin_is_fix:
                        evotree = self.tb.pop_random(call_params, from_origin=True)
                    else:
                        evotree = self.tb.pop_random(call_params)
                    # tree.meta.last_evolution = tag  # todo
                    self.pop.append_tree(evotree, tag=tag)
        return

    def gen_next_population(self):
        """
        Creates all new Generations by applying the evolutions in the evolve-loop.

        brainstorm:
        - the tree can be reproduced (selection), random/new, olymp-reproduction
        - (1 tree) mutations can affect a point, branch, terminal nodes
        - (2 trees) can make a crossover
        """
        # All gp creators: name, function, num of trees from tournament selection

        for tag, evolve_specs in self.evolve_loop.evolve_loop.items():  # all selected gp mutations

            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            call_params = evolve_specs.get('custom_params')

            self.printpl('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')
            if evolve_name == 'reproduce':
                """
                
                """
                for nn in range(evolve_num):
                    evotree = self.pop.selection_tournament(tourn_size=tourn_size)
                    if call_params.get('simplify'):  # sfeh==>debug
                        try:
                            evotree.evolve_reduce(completely=False)
                            # tree.meta.last_evolution = tag
                        except Exception as ex:
                            print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.conf.print_type)
                            # raise ex  # sfeh debug

                    self.pop.append_tree(evotree, tag=tag)  # append_tree anyways... it was worth a try :P

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    evotree = self.pop.selection_tournament(tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_point(evotree)
                    # tree.meta.last_evolution = tag
                    self.pop.append_tree(evotree, tag=tag)

            elif evolve_name == 'mutate branch':
                for nn in range(evolve_num):
                    build_spec, size_mode, mean_min_max_var, p_op = helper_evolve_params_branch(call_params)
                    evotree = self.pop.selection_tournament(tourn_size=tourn_size)
                    cool_build_size = choose_build_size(size_mode, mean_min_max_var, tree=evotree)
                    evotree = self.tb.evolve_mutate_branch_depth(evotree, cool_build_size)  # sfeh , full_or_grow=full_or_grow), size_mode=size_mode
                    # new_tree.meta.last_evolution = tag
                    self.pop.append_tree(evotree, tag=tag)

            elif evolve_name == 'crossover branch':
                for nn in range(int(evolve_num / 2)):  # two childs
                    """
                    swap branches of two trees
                    - select parent a and b
                    - select swappable branche for a_parent from b_parent
                        - select a node in a (and crossover here, no matter what)
                    - delete a_parent branch and insert b_parent branch (which tactic?)
                    todo into main tree?
                    """
                    atree = self.pop.selection_tournament(tourn_size=tourn_size)
                    btree = self.pop.selection_tournament(tourn_size=tourn_size)

                    # 1. two parents
                    # 2. search nodes for left and right that can be exchanged. convert_needed
                    atree, btree = self.tb.evolve_crossover(atree, btree)
                    # tree = self.tb.finalize(etree)  # ==>state
                    self.pop.append_tree(atree, tag=tag)
                    self.pop.append_tree(btree, tag=tag)

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    evotree = self.pop.selection_tournament(tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_filter_random(evotree)  # todo , call_params
                    # tree.remember_evolution(tag)
                    self.pop.append_tree(evotree, tag=tag)

            elif evolve_name == 'revive paretos':

                for nn in range(evolve_num):
                    evotree = self.pareto.random_choice()
                    # evotree.remember_evolution(tag)
                    self.pop.append_tree(evotree, tag=tag)

            elif evolve_name == 'random trees':

                if self.origin.origin_is_fix:
                    for nn in range(evolve_num):
                        evotree = self.tb.pop_random(call_params, from_origin=True)
                        # tree.meta.last_evolution = tag
                        self.pop.append_tree(evotree, tag=tag)
                else:
                    for nn in range(evolve_num):
                        evotree = self.tb.pop_random(call_params)
                        # tree.meta.last_evolution = tag
                        self.pop.append_tree(evotree, tag=tag)
            else:
                print_e(f"Evolution not known: '{evolve_name}'")

        missing_trees = self.conf.pop_max - len(self.pop.pop_next)
        if missing_trees > 0:
            if missing_trees > 0.05 * self.conf.pop_max:
                self.printpl('ii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            else:
                self.printpl('iii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')

            while len(self.pop) < self.conf.pop_max:
                return  # sfeh aka create trees here if desired

        # sfeh automatically fill with random trees (check this at the initiation)
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        return

    def custom_exit_condition(self):
        """
        Special condition to exit the evolve-loop
        1. when >100 generations, no new paretos paretos were found
        """
        try:
            if self.pop.gens_since_last_pareto > 100:  # .iloc[-1] > 100:  # sfeh discussion
                print('SFEH This condition made your program exit!')
                return True
            else:
                return False
        except Exception:
            return False

    def plagih_gp_run(self, gen_additionally):
        """
        regular plagih run
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.conf.gen_id + gen_additionally)
            self.printpl('i', f'Adding new generations, gen_max was {printdummy}, current gen {self.conf.gen_id}. gen_additionally: {gen_additionally}. New max gen: {self.conf.gen_max}')

        # sfeh
        # yaml_dump(self.rootdir / 'used_config.yaml', self.conf, print_type=self.print_type)

        # self.pop.gens_since_last_pareto = 0

        while self.conf.gen_id <= self.conf.gen_max and not self.custom_exit_condition():  # max generation, max time, done...
            self.pop.time_genstart = time.perf_counter()  # sfeh here?

            if self.conf.gen_id == 0:
                # self.print_g('gg', f'Preparing to create first Generation. Gen {self.conf.gen_id}.')  # todo
                self.gen_create_initial()  # sfeh stattdessen einfach checken, ob die letzte population leer ist und info/warnung: neue generation?
            else:
                # This might be a solution for multiprocessing:
                # You can avoid this situation by calling multiprocessing.Process before you load your huge data.
                # Then the additional memory allocations will not be reflected in the child process when you load the data in the parent.
                # sfeh: In python 3.8, this might be availably: multiprocessing.shared_memory https://docs.python.org/3/library/multiprocessing.shared_memory.html
                # sfeh: check memory usage! should not scale with the number of processes, only one pop_base is required, it does not change.

                self.conf.mp_cores = 1  # sfeh wasd
                if self.mp_cores >= 2:
                    pass
                    # # sfeh asd
                    # mp.Process()  # sfeh maybe good for memory? https://stackoverflow.com/questions/14749897/python-multiprocessing-memory-usage
                    # print(f'Trying to make parallel new population: {mp.cpu_count()}')
                    # with mp.Pool(min(mp.cpu_count(), self.mp_cores)) as p:
                    #     evolve_list = [[tag, evolve_specs] for tag, evolve_specs in self.evolve_loop.items()]
                    #     results = p.map(fun, evolve_list)
                    # time_evolve = time.perf_counter()
                else:
                    self.gen_next_population()

            self.pareto.pareto_from_population(self.pop.pop_next)

            self.pop.pop_analyze()
            # sfeh check if there are really unique... doubt it.

            """
            show plots if necessary
            """
            save_gen = int(self.conf.period.get('gen_save', 1))
            plot_gen = int(self.conf.period.get('gen_plots', 1))
            if self.conf.gen_id >= plot_gen and self.conf.gen_id % plot_gen == 0:
                self.file_analysis_plots()

            if self.conf.gen_id >= save_gen and self.conf.gen_id % save_gen == 0 or self.conf.gen_id == 10:  # sfeh extra save at 10 for early feedback while testing
                self.conf.backup_save()

            self.pop.pass_population()

            self.printpl('ggg', f'Generation {self.conf.gen_id} took a total time of: {time.perf_counter() - self.pop.time_genstart:4.2f}.')
            self.conf.gen_id += 1
        else:
            self.printpl('g', f'Done after Generation {self.conf.gen_id}.\nTime since start: {time.perf_counter() - self.time_start:4.2f}s')

        self.conf.backup_save()

        return

    def file_analysis_plots(self):
        """
        Make all plots
        """

        self.pop.plot_gen_performance()  # largest plot analysing the
        self.pareto.plot_paretofront()
        # self.plot_evolve_performance()  # sfeh
        return

    def printpl(self, message_type, message_str):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        message_type options can be found in config
        """
        printez(message_type, message_str, print_type=self.conf.print_type)
        return


# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['label'], 1/v['xtype': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


if __name__ == '__main__':
    """
    Alpha tests
    """
    tb = TreeBuilder(obs_names=['aaa', 'bbb'])
    # t1 = tb.invent_core_depth(float, 3, p_op=0.5)
    # tree2 = tb.evolve_mutate_point(t1)
    # t1 = tb.invent_core_depth(float, 3, p_op=0.9)
    # t2 = tb.invent_core_depth(float, 3, p_op=0.9)
    # t1, t2 = tb.evolve_crossover(t1, t2)
    # for _ in range(5):
    #     print('x.D', t1, '===', t2)
    #     t1, t2 = tb.evolve_crossover(t1, t2)
    #     print('x~D', t1, '===', t2)
    from plagih.sympy_extras import *

    trexpt = '[+,1,2]'
    # tree = tb.tree_from_nested_trick(trexpt)
    nstr = '["+",["-",["Ifte",["True"],[2],[2.043]],["cartVel"]],[-1.3]]'
    nstr = '["+",["-",["Ifte","True",["sin",2],2.043],"cartVel"],-1.3]'
    x = eval(x, {'sin': Sin, 'Ifte': Ifte})
