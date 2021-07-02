"""

Functions, that might be addable in the future:
"""
from plagih.node_labels import *
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

    def __init__(self, kernel, conf, tb, origin):
        self.kernel = kernel
        self.conf = conf  # todo
        # self.conf.gen_id = 0  # todo?
        self.size = conf.pop_max
        self.tb = tb

        self.pop_next = []
        self.pop_base = []  # sfeh maybe better names

        self.lut = {}
        self.origin = origin
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0
        self.print_type_g = conf.print_type

        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'complexity_avg', 'complexity_var', 'complexity_stderr',
                                                'gens_since_last_pareto'])
        """
        was Evolution class
        The evolve_dict is converted into a list of the population length.
        The evolve_loop is for the regular evolution, the evolve_random is especially required for the first generation.
        """
        self.conf = conf
        evolve_loop = {
            # Reproduction (10%)
            'Repro': {'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.06, 'custom_params': {}},
            'Rsympy': {'evolve_name': 'reproduce', 'evolve_rate': 0.03, 'custom_params': {'simplify': True}},
            'Pareto': {'evolve_name': 'revive paretos', 'evolve_rate': 0.01, 'custom_params': {}},
    
            # Mutation (25%)
            'Point': {'evolve_name': 'mutate point', 'evolve_rate': 0.05, 'custom_params': {}},
    
            'BranchDF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                         'custom_params': {'build_max': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8), 'p_op': 1.0}}},
            'BranchDG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                         'custom_params': {'build_max': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 5, 1), 'p_op': 0.5}}},
            'BranchNF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'p_op': 1.0}}},
            'BranchNG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'p_op': 0.5}}},
            'BranchShrink': {'evolve_name': 'mutate branch', 'evolve_rate': 0.0,
                             'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0), 'p_op': 0.5}}},
    
            'FilterBO': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                         'custom_params': {'mode': 'branch', 'filter_observations': True}},
            'FilterB': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                        'custom_params': {'mode': 'branch', 'filter_observations': False}},
            'FilterP': {'evolve_name': 'filter optimize', 'evolve_rate': 0.0, 'tourn_size': 5,
                        'custom_params': {'mode': 'point', 'filter_observations': True}},
    
            # Crossover (35%)
            'Xover': {'evolve_name': 'crossover branch', 'evolve_rate': 0.30, 'custom_params': {}},  # sum 0.70
    
            # Leftovers are automatically filled with random trees
    
            # Random (25%)
            'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.05,
                      'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 3, 5, 1), 'p_op': 1.0}}},
            'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.00,
                      'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'p_op': 0.5}}},
            'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'p_op': 0.5}}},
            'Rand4': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'p_op': 1.0}}},  # param 'max' can be None
        }
        self.evolve_loop = self.evolve_safety_update(evolve_loop)
    
        if origin.origin_is_fix:
            evolve_random = {'Rand3o': {'evolve_name': 'random trees', 'evolve_rate': 1.00,
                                        'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (10, 3, None, 4), 'p_op': 1.0}}}}
        else:
            # try:  # sfeh still not sure about this
            #     evolve_random = self.evolve_list_random['from_scratch']
            # except:
            evolve_random = {'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                       'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (3.5, 2, 5, 1), 'p_op': 1.0}}},
                             'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                       'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 2, 6, 1), 'p_op': 0.5}}},
                             'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.40,
                                       'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 3, None, 4), 'p_op': 1.0}}}
                             }
        self.evolve_random = self.evolve_safety_update(evolve_random)
    
        self.evolve_tags = list(self.evolve_loop.keys()) + list(self.evolve_random.keys())
        return
    
    def evolve_safety_update(self, evolve_dict):
        """
        Updates tournament size and evolve rates
    
        Example entry of the list could be:
        {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
        'custom_params': {'build_max': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
        """
    
        for tag, evolve_spec in evolve_dict.items():
            evolve_dict[tag]['tourn_size'] = evolve_spec.get('tourn_size', self.conf.tourn_size)
            evolve_dict[tag]['evolve_num'] = int(evolve_spec.get('evolve_rate') * self.conf.pop_max)
            evolve_spec['custom_params'] = evolve_spec.get('custom_params', {})
        return evolve_dict
    
    # evolve_loop = self.evolve_list  # todo
    # self.printpl('i', 'Using evolve rates from config')
    
    # class Evolution:
    #     # todo rate not in here
    #     # def __init__(self, id=None, evolution=None, rate=None, params=None, custom_params=None):
    #     #     self.id = id
    #     #     self.evolution = evolution
    #     #     self.params = params or {}
    #     #     self.rate = rate
    #     #     self.custom_params = custom_params
    
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
            self.pop_next.append(self.origin.tree)
            # self.append_tree(self.origin.tree)  # sfeh why not :P
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
                    self.append_tree(evotree, tag=tag)
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

            self.print_g('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')
            if evolve_name == 'reproduce':
                """
                
                """
                for nn in range(evolve_num):
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    if call_params.get('simplify'):  # sfeh==>debug
                        try:
                            evotree.evolve_reduce(completely=False)
                            # tree.meta.last_evolution = tag
                        except Exception as ex:
                            print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.conf.print_type)
                            # raise ex  # sfeh debug

                    self.append_tree(evotree, tag=tag)  # append_tree anyways... it was worth a try :P

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_point(evotree)
                    # tree.meta.last_evolution = tag
                    self.append_tree(evotree, tag=tag)

            elif evolve_name == 'mutate branch':
                for nn in range(evolve_num):
                    build_spec, size_mode, mean_min_max_var, p_op = helper_evolve_params_branch(call_params)
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    cool_build_size = choose_build_size(size_mode, mean_min_max_var, tree=evotree)
                    evotree = self.tb.evolve_mutate_branch_depth(evotree, cool_build_size)  # sfeh , full_or_grow=full_or_grow), size_mode=size_mode
                    # new_tree.meta.last_evolution = tag
                    self.append_tree(evotree, tag=tag)

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
                    atree = self.selection_tournament(tourn_size=tourn_size)
                    btree = self.selection_tournament(tourn_size=tourn_size)

                    # 1. two parents
                    # 2. search nodes for left and right that can be exchanged. convert_needed
                    atree, btree = self.tb.evolve_crossover(atree, btree)
                    # tree = self.tb.finalize(etree)  # ==>state
                    self.append_tree(atree, tag=tag)
                    self.append_tree(btree, tag=tag)

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_filter_random(evotree)  # todo , call_params
                    self.append_tree(evotree, tag=tag)

            elif evolve_name == 'revive paretos':

                for nn in range(evolve_num):
                    evotree = self.pareto.random_choice()
                    self.append_tree(evotree, tag=tag)

            elif evolve_name == 'random trees':

                if self.origin.origin_is_fix:
                    for nn in range(evolve_num):
                        evotree = self.tb.pop_random(call_params, from_origin=True)
                        self.append_tree(evotree, tag=tag)
                else:
                    for nn in range(evolve_num):
                        evotree = self.tb.pop_random(call_params)
                        self.append_tree(evotree, tag=tag)
            else:
                print_e(f"Evolution not known: '{evolve_name}'")

        missing_trees = self.conf.pop_max - len(self.pop_next)
        if missing_trees > 0:
            if missing_trees > 0.05 * self.conf.pop_max:
                self.print_g('ii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            else:
                self.print_g('iii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')

            while len(self.pop_next) < self.conf.pop_max:
                return  # sfeh aka create trees here if desired

        # sfeh automatically fill with random trees (check this at the initiation)
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        return

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
        fintree: 'FinalizedTree' = self.kernel.get_fitness_extreme_function(tree_list, key=lambda tree: tree.get_fitness())
        evotree = fintree.get_evotree()
        return copy.deepcopy(evotree)  # sfeh deepcopy not required if it is copied later

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

    def plot_gen_performance(self, path_monitoring: Path):
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
            fig.savefig(path_monitoring)
            printez('f', f"monitoring: {path_monitoring.as_posix()}", print_type=self.print_type_g)
            # plt.close('all')


class OriginTree:
    """
    The origin tree (which was already loaded) gets activated for its use in the GP-process
    """

    def __init__(self, kernel, path_origin=None):
        if path_origin:

            with Path.open(path_origin, newline='') as file:
                nested_expr = file.read()

            tree = tree_from_nested_string(nested_expr)
            expr_raw = tree.eval_expr()
            try:
                expr_sym = expr_sympify(expr_raw)
            except Exception as sympex:
                raise Exception(f'Loaded origin_tree expression could not be mathematically simplified: {sympex}')

            # sfeh, this does not work
            # if not tree_check_is_sympified(tree):
            #     print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
            #                          ''.format(expr_raw, expr_sym))

            used_observations = tree.get_observation_list()
            tf_origin_results = kernel.eval_tf(expr_sym, used_observations)
            fitness_train = round(tf_origin_results['mean_error'], kernel.precision)  # todo do in line above
            if kernel.exploration_risk:
                kernel.origin_results = tf_origin_results['results_kernel']  # after getting the origin-results, these informations can be updated

            meta = TreeMeta(fitness=fitness_train, parsimony=0, expr_raw=expr_raw, expr_sym=expr_sym)
            meta.append_tag('origin')
            self.tree = FinalizedTree(tree, meta)
            self.origin_is_fix = False  # TODO self.tree.is_fix  # sfeh the root node!?!
            self.existing = True

            # origin_tree.set_fitness(fitness_train)
            # origin_tree.set_parsimony(0)
            # self.printpl('gg', f'Loading origin tree, regr. error {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')
        else:
            self.existing = False
            self.tree = None  # sfeh probably the 'existing' above is deprecated
            self.origin_is_fix = False  # ...if non-existent, it is also not fix

    def exists(self):
        return self.exists


class ExplainableGP:
    """

    """

    def __init__(self, conf, rootdir, args=None):
        self.conf = conf
        self.develop = args.develop  # more testing and stuff during development phase
        self.time_start = time.perf_counter()
        kernel = RegressionKernel(conf)
        self.kernel = kernel
        tb = TreeBuilder(kernel.obs_names, root_xtype=float)
        origin = OriginTree(kernel, path_origin=self.conf.path_origin)
        self.origin = origin
        # self.evolve_loop = EvolutionLoop()
        self.pareto = ParetoFront(kernel.fitness_compare, origin=origin)
        self.pop = Population(kernel, conf, tb, origin)
        self.rootdir = rootdir

        self.printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        # self.conf.set_rootdir()  # sfeh better solution?

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        self.lut = {}
        return

    def backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the paretos front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """

        # help_dict = {'self.monitor_evol': self.monitor_evol,
        #              'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh save complete config?    # sfeh i dont think we need the config
        # run_backup_data = self.gen_id, self.paretos, self.pop_base, self.monitor_df, help_dict  # sfeh use this later, help_dict
        # pickle_dump(self.rootdir / self.file_backup, run_backup_data)

        # run_backup_dict = {'self.gen_id': self.gen_id,
        #                    'self.paretos': self.paretos,
        #                    'self.pop_base': self.pop_base,
        #                    'self.monitor_df': self.monitor_df,
        #                    'help_dict': help_dict}
        #
        # path_backupyyy = path_make_dir(self.rootdir / 'backup/backup.yaml')
        # yaml_dump(path_backupyyy, run_backup_dict, self.ui.print_type=self.ui.print_type)

        return

    def backup_load(self, path_load_backup=None):
        """
        If a backup-file is found...
        """
        # path_backup = path_load_backup or self.rootdir / self.file_backup  # sfeh file-load
        #
        # if Path.is_file(path_backup):
        #     self.print_ui('g', f'Loading data from backup-file {path_backup}')
        #     """
        #     Loading the state of the run from the pickle file
        #     """
        #     try:
        #         with Path.open(path_backup, 'rb') as file:
        #             run_data = pickle.load(file)
        #
        #     except NotImplementedError as nimp:
        #         raise Exception(f'NotImplementedError: {nimp}')
        #     except EOFError as eoferr:
        #         raise Exception(f'EOFError: \n{eoferr}')
        #
        #     try:
        #         self.gen_id, self.pareto, self.pop_base, monitor_pd, a_helping_dict = run_data  # sfeh use a helping dictt a_helping_dict is used for a useable sldifjsdfsdfg , a_helping_dict
        #         self.monitor_df = monitor_pd  # sfeh
        #         if 'gens_since_last_pareto' not in self.monitor_df.columns:
        #             self.monitor_df['gens_since_last_pareto'] = np.nan
        #         # self.monitor_evol = a_helping_dict.get('self.monitor_evol') or self.monitor_evol
        #         self.gens_since_last_pareto = a_helping_dict.get('gens_since_last_pareto') or 0  # sfeh
        #     except:
        #         self.gen_id, self.pareto, self.pop_base, m = run_data
        #
        #     self.check_update()
        #     self.print_ui('g', f'Successfully loaded backup file. Generation: {self.gen_id}')
        #
        #     # except Exception as ex:
        #     #     raise Exception(f'Even though a backup exists for this run, it could not be loaded, because of\n{ex}')
        # else:
        #     raise FileNotFoundError(f'No backup-file found at {path_backup}.')
        return

    def custom_exit_condition(self):
        """
        Special condition to exit the evolve-loop
        1. when >100 generations, no new paretos paretos were found
        """
        try:
            if self.gens_since_last_pareto > 100:  # .iloc[-1] > 100:  # sfeh discussion
                print('SFEH This condition made your program exit!')
                return True
            else:
                return False
        except Exception:
            return False

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
            parsimony = tree.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin.tree)
            if parsimony > self.conf.parsimony_max:
                # sfeh last_evolution leads to error atm
                # print_warning('wwww', f'Parsimony too high, last evolution: {tree.meta.last_evolution}', print_type=self.print_type_g)  # sfeh care about wwww. should not
                raise Exception('Tree too complex')
            try:
                expr_raw = tree.eval_expr()
                expr_sym = expr_sympify(expr_raw)
                treeobs = tree.get_observation_list()
                fitness = round(self.kernel.eval_tf(expr_sym, treeobs, only_fitness=True), self.conf.precision)
                if fitness != fitness or fitness == float('inf'):
                    raise Exception(f"fitness_train is: '{fitness}'")  # happens, eg when values are soo wrong that it leaves the float-range
                meta = TreeMeta(fitness, parsimony, expr_raw, expr_sym)
                self.lut[hash(tree)] = meta
            except Exception as ex:
                # print_warning('wwww', f'Exception while evaluating: {ex}, tree: {tree}.', print_type=self.print_type_g)
                raise Exception(f'eval-ex: {ex}')

        fintree = FinalizedTree(tree, meta)
        return copy.deepcopy(fintree)  # sfeh guess no deepcopy is required

    def plagih_gp_run(self, gen_additionally):
        """
        regular plagih run
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.conf.gen_id + gen_additionally)
            self.printpl('i', f'Adding {gen_additionally} more generations in gen {self.conf.gen_id}, increasing gen_max from {printdummy} to {self.conf.gen_max}.')

        # sfeh check if any roots
        # yaml_dump(self.rootdir / 'used_config.yaml', self.conf, print_type=self.print_type)

        # self.pop.gens_since_last_pareto = 0

        while self.conf.gen_id <= self.conf.gen_max and not self.custom_exit_condition():  # max generation, max time, done...
            self.pop.time_genstart = time.perf_counter()  # sfeh here?

            if self.conf.gen_id == 0:
                self.printpl('gg', f'Preparing to create first Generation. Gen {self.conf.gen_id}.')  # sfeh debug
                self.pop.gen_create_initial()  # sfeh stattdessen einfach checken, ob die letzte population leer ist und info/warnung: neue generation?
            else:
                # This might be a solution for multiprocessing:
                # You can avoid this situation by calling multiprocessing.Process before you load your huge data.
                # Then the additional memory allocations will not be reflected in the child process when you load the data in the parent.
                # sfeh: In python 3.8, this might be availably: multiprocessing.shared_memory https://docs.python.org/3/library/multiprocessing.shared_memory.html
                # sfeh: check memory usage! should not scale with the number of processes, only one pop_base is required, it does not change.

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
                    self.pop.gen_next_population()

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

        self.pop.plot_gen_performance(self.rootdir / 'monitoring.png')  # largest plot analysing the
        self.pareto.plot_paretofront(self.rootdir / f'paretofront.pdf', self.conf.name, self.conf.parsimony_max)
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


def reci(lst, depth=0, obs_list=None):
    """
    nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    """
    strlabel = lst[0]
    if strlabel in ['True', 'False']:
        print(f'Label {strlabel} is boolean.')
        label = BoolConstant(strlabel)
    else:
        try:
            float(strlabel)
            print(f'Label {strlabel} is float.')
            label = FloatConstant(strlabel)
        except:
            pass

            if strlabel in ops:
                print(f'Label {strlabel} is an operator.')
                label = ops[strlabel]
            else:
                if obs_list:
                    if strlabel in obs_list:
                        print(f'Label {strlabel} is an observation.')
                        label = Observation(strlabel)
                    else:
                        raise Exception(f'Label "{strlabel}" can not be assigned to a node-label!')
                else:
                    print(f'Label {strlabel} is assumed to be an observation.')
                    label = Observation(strlabel)

    node = Node(label=label, depth=depth)

    if len(lst[1:]) == node.get_arity():
        childs = [reci(x, depth=depth + 1) for x in lst[1:]]
        node.set_childs(childs)

    else:
        raise Exception(f'Tree-building list length {len(lst[1:])} does not match the nodes arity {node.get_arity()}.')

    return node


def tree_from_nested_string(nested_str):
    """
    all_input_options = ['1', '0', '-1.132', 'True', 'False', 'vel', 'Ifte', 'max', 'Maxi', '-vel']
    nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    """

    evaled_expr = eval(nested_str, ops)

    tree = reci(evaled_expr, 0)
    print('debugsfeh', tree)
    return tree


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

    nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'

    tree = tree_from_nested_string(nstr)
