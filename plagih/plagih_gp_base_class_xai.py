"""
The main class of a gp run. It holds the following functionalities
- run informations (config, evolution-specifications for the loop, success monitoring])
- population (pop_base, pop_next)
-
"""
from plagih.monitoring import plot_gen_performance
from plagih.paretofront import *
from plagih.tree_factory import *

import copy
from pathlib import Path
import numpy as np
import pandas as pd

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


def printpl(message_type, message_str):
    """Lightweight print function.
    Instead of checking if you should print every time, this is done here.
    message_type options can be found in config
    """
    printez(message_type, message_str)
    return


class ExplainableGP:

    def __init__(self, name, pop_max, gen_max, rootdir, kernel, origintree, tb: TreeBuildRestrictions, selection_default):
        self.time_start = time.perf_counter()
        self.kernel = kernel
        self.origintree = origintree
        self.name = name
        self.tb = tb
        self.pop_max = pop_max
        self.mp_cores = 1  # sfeh: open MP (multiprocessing)
        self.gen_max = gen_max
        self.rootdir = rootdir
        self.gen_id = 0
        self.selection_default = selection_default

        printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        self.paretofront = []  # not a separate class; requires too much information
        self.pop_next = []
        self.population = []  # sfeh:discuss maybe better names?

        self.lut = {}  # Lookup-table for tree(-expressions) and tits fitness/parsimony. Improving runtime a lot!

        # monitoring
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0
        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'parsim_avg', 'parsim_var', 'parsim_best',
                                                'gens_since_last_pareto'])

    def pop_kill(self):
        """Delets the current population"""
        self.population = []
        self.pop_next = []

    # self.printpl('i', 'Using evolve rates from config')

    def evoloop(self, period_plots=10, period_save=10, gen_additionally=0):
        """
        Start a regular GP run.
        The only other option is to analyze a loaded run.
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.gen_max)
            self.gen_max = max(self.gen_max, self.gen_id + gen_additionally)
            printpl('i', f'Adding {gen_additionally} more generations in gen {self.gen_id}, '
                         f'increasing gen_max from {printdummy} to {self.gen_max}.')

        # sfeh check if any roots
        # yaml_dump(self.rootdir / 'used_config.yaml', self.conf)

        while self.gen_id <= self.gen_max and not self.run_custom_exit_condition():
            self.time_genstart = time.perf_counter()  # sfeh here?

            if self.gen_id == 0:
                printpl('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')  # sfeh debug
                self.gen_create_initial()  # sfeh check if last pop is empty? +info/warnung: neue generation?
                self.gen_id += 1
            else:
                # This might be a solution for multiprocessing: You can avoid this situation by calling
                # multiprocessing.Process before you load your huge data. Then the additional memory allocations will
                # not be reflected in the child process when you load the data in the parent. sfeh: In python 3.8,
                # this might be availably: multiprocessing.shared_memory
                # https://docs.python.org/3/library/multiprocessing.shared_memory.html sfeh: check memory usage!
                # should not scale with the number of processes, only one pop_base is required, it does not change.

                self.gen_create_next()

            if len(self.paretofront) == 0:
                self.paretofront = [self.pop_next[0]]  # initialize
                printez('a', f'Starting a new paretofront with parsimony: {self.pop_next[0].get_parsimony()} '
                             f'fitness: {self.paretofront[0].get_fitness():6.4f}')

            # tmp_pareto = pareto_from_pop(self.pop_next)  # sfeh:idea paretofront in each generation?
            self.paretofront = self.run_update_paretofront(self.paretofront)

            gen_time = time.perf_counter() - self.time_genstart
            pop_analysis_dict = pop_analyze(self.pop_next, gen_time, self.gens_since_last_pareto)
            self.monitor_df.loc[self.gen_id] = pop_analysis_dict
            printpl('gg', f"Created {len(self.pop_next)}/{self.pop_max} ({pop_analysis_dict['pop_unique']}"
                          f" unique) in generation {self.gen_id}. Gen took {gen_time:4.2f}s")

            self.population = self.pop_next[:]
            self.pop_next = []

            printpl('ggg', f'--- Gen {self.gen_id} took: {time.perf_counter() - self.time_genstart:4.2f}. ---')
            self.monitoring_scheduled_io(self.gen_id, period_plots, period_save)
            self.gen_id += 1

        printpl('g', f'Done after Generation {self.gen_id}.\n'
                     f'Time since start: {time.perf_counter() - self.time_start:4.2f}s')

        self.backup_save()

        return

    def run_update_paretofront(self, paretofront):
        """
        CAUTION: This Function was tried to be separated many times now. it never worked.
        This was tried =3= times now. Please increase the counter when you try.
        Reason: The paretocandidates should be simplified if possible and gens_since_last_pareto is reset.
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
                try:
                    # sfeh: trying to simplify the tree for a even improved pareto # sfeh:open
                    symtree = evolve_reduce_simplify(fintree.get_evotree(), force=True)
                    symmeta = self.tree_eval_meta(symtree, tag='sfeh:sym')
                    _symtree_fin = FinalizedTree(symtree, symmeta)
                    if _symtree_fin.get_parsimony() < fintree.get_parsimony():
                        printyeah('a', f'Paretofront: Even further simplified! '
                                       f'{_symtree_fin.get_parsimony()} < {fintree.get_parsimony()}')

                    self.pop_append(symtree)  # sfeh:open , tag='sfeh-sympifyed_pareto'

                except KeyError as ex:
                    print_e(f'SFEH: this tree could whatever {ex}')  # -> piecewise function, mostly

                _obsoletes = [i for i in paretofront if
                              i.get_fitness() > fintree.get_fitness() and i.get_parsimony() >= fintree.get_parsimony()]
                if _obsoletes:
                    printyeah('a', f'Paretofront: Removing obsolete entries {[str(i) for i in _obsoletes]}')
                paretofront = [ftree for ftree in paretofront if ftree not in _obsoletes]
                paretofront.append(fintree)
                paretofront = pareto_sort(paretofront)

        return paretofront

    # sfeh:idea sympy.NumberSy,bol

    def gen_create_initial(self):

        if self.origintree is not None:
            self.pop_next.append(self.origintree)  # the origin fintree is the only candidate (automatically added)
        else:
            @self.create_trees(rate=0.5)
            def init_rand1():
                return self.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.0, 1.0)), 2, 3), p_term=0)

            @self.create_trees(rate=0.5)
            def init_rand2():
                return self.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(2.5, 1.0)), 2, 3), p_term=0)
        return

    def create_trees(self, rate=0.0, select_n=0, crossover=False):
        """Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the final tree (fintree) is refurbished."""

        def loop(create_tree_func):
            n = int(rate * self.pop_max)
            n_success = 0
            n_fails = 0
            tag = create_tree_func.__name__
            printpl('gggg', f'->Evolving {n}x \'{tag}\'...')

            while n_success < n:
                trees = [self.selection_default(self.population) for _ in range(select_n)]
                try:
                    if not crossover:
                        evotree = create_tree_func(*trees)
                        self.pop_append(evotree, tag=tag)
                        n_success += 1
                    else:
                        t1, t2 = create_tree_func(*trees)
                        self.pop_append(t1, tag=tag)
                        n_success += 1
                        self.pop_append(t2, tag=tag)
                        n_success += 1

                except (ValueError, ArithmeticError) as ex:
                    n_fails += 1  # sfeh:use this for something?
                    print_warning('www', f'\'{tag}\' failed: {ex}')
                    if n_fails > n_success + 5:  # allow more fails: n_fails > n
                        print_e(f'Evolution "{tag}" fails too often: {n_fails}x. {n_success}.')
                        return  # sfeh raise?
                except TypeError as ex:
                    print(f'Typeerror, but why? {ex}')
                except AttributeError as ex:
                    raise AttributeError(f'Probably sympy.im in expr {ex}')
                    # print(f'Probably sympy.im in expr {ex}')
        return loop

    def pop_append(self, evotree, tag=None):
        try:
            meta = self.lut[str(evotree)]  # fixed nodes not relevant
        except KeyError:
            meta = self.tree_eval_meta(evotree, tag=tag)

        fintree = FinalizedTree(evotree, meta)
        self.pop_next.append(fintree)

    def gen_create_next(self):
        """Creates all new Generations by applying the evolutions in the evolve-loop.

        brainstorm:
        - the fintree can be reproduced (selection), random/new, olymp-reproduction
        - (1 fintree) mutations can affect a point, branch, terminal nodes
        - (2 trees) can make a crossover"""

        @self.create_trees(rate=0.1)
        def repro1():
            return selection_tournament(self.population, tournsize=4)

        @self.create_trees(rate=0.1, select_n=1)
        def mut_br(tree):
            return self.tb.evolve_mutate_branch_nodes(tree, 4, p_term=0)

        @self.create_trees(rate=0.2, select_n=2, crossover=True)
        def xover(*trees):
            evo1, evo2 = self.tb.evolve_crossover(trees[0], trees[1])
            return evo1, evo2

        @self.create_trees(rate=0.1, select_n=1)
        def re_sym_all(tree):
            return evolve_reduce_simplify(tree, completely=True)

        @self.create_trees(rate=0.1, select_n=1)
        def re_sym(tree):
            return evolve_reduce_simplify(tree, completely=False)

        @self.create_trees(rate=0.1, select_n=1)
        def mut_pt(tree):
            return self.tb.evolve_mutate_point(tree)

        # @self.create_trees(rate=0.1)
        # def mxPointXXX():
        #     evotree = selection_tournament(self.pop_base, tournsize=3)
        #     return self.tb.evolve_mutate_pointxxx(evotree)

        @self.create_trees(rate=0.1, select_n=1)
        def mx_ranch_d(tree):
            return self.tb.evolve_mutate_branch_depth(tree, 4, p_term=0.5)

        @self.create_trees(rate=0.1, select_n=1)
        def mx_branch_n(tree):
            nodeamount_goal = np.clip(int(random.normalvariate(12, 4)), 0, 20)
            return self.tb.evolve_mutate_branch_nodes(tree, nodeamount_goal)

        @self.create_trees(rate=0.1, select_n=1)
        def filter_optimize(tree):
            return self.tb.evolve_mutate_filter(tree)

        @self.create_trees(rate=0.1)
        def pareto_revive():
            fintree = np.random.choice(self.paretofront)
            return fintree.tree

        @self.create_trees(rate=0.1)
        def rand1():
            return self.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3, 1)), 2, 4), p_term=0)

        @self.create_trees(rate=0.1)
        def rand2():
            # sfeh float? nope
            return self.tb.evolve_new_tree_depth(np.clip(int(random.normalvariate(3.5, 1)), 2, 4), p_term=0)

        return

    def evoloop_monitoring_plots(self):
        """
        Create all run-related analysis plots in the root directory
        """
        try:
            path_monitoring = self.rootdir / 'monitoring.png'
            plot_gen_performance(self.monitor_df, self.name, path_monitoring)  # largest plot analysing the
            printpl('f', f"monitoring: {path_monitoring}")  # .as_posix()
            pareto_plot(self.paretofront, self.rootdir, self.name, self.tb.nodes_max)
        except Exception as ex:
            printpl("e", f'Could not create plots: {ex}\n')

    def monitoring_scheduled_io(self, gen_id, period_plots, period_save):
        """
        Every x generations, save a backup and/or save plots
        """
        plot_gen = period_plots
        if gen_id >= plot_gen and gen_id % plot_gen == 0:
            self.evoloop_monitoring_plots()

        save_gen = period_save
        if gen_id >= save_gen and gen_id % save_gen == 0 or gen_id == 10:
            self.backup_save()

    def backup_save(self):
        """
        Load/safe backup of a run
        """

        path_backup = self.rootdir / 'backup/backup.pkl'

        # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
        run_backup_data = {}, self.gen_id, self.population, self.paretofront, self.monitor_df
        path_backup = path_make_dir(path_backup)
        pickle_dump(path_backup, run_backup_data)
        # sfeh:debug

    def backup_load(self, path_load_custom_backup=None):
        """Load/safe backup of a run
        """

        path_backup = path_load_custom_backup or self.rootdir / 'backup/backup.pkl'

        if Path.is_file(path_backup):
            printpl('g', f'Loading data from backup-file {path_backup}')
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)
            except NotImplementedError as ex:
                raise Exception(f'NotImplementedError: {ex}')
            except EOFError as ex:
                raise Exception(f'EOFError: \n{ex}')

            help_dict, self.gen_id, self.population, self.paretofront, self.monitor_df = run_data
            printpl('g', f'Successfully loaded backup file. Generation: {self.gen_id}')

        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}')  # sfeh:beautify occurs 2x

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

    # def plot_evolve_performance(self):
    #     """
    #     Plots for each tag in the evolution list
    #     (too much, I guess)
    #     sfeh: this should be saved within the trees. Everything else is a waste of memory!
    #     sfeh:open
    #     """
    #     try:
    #         with plt.rc_context(rc=pyplot_rc_tex):
    #             fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 9),
    #                                     sharex='all')  # , gridspec_kw={'height_ratios': [1,1,1]}
    #             fig.tight_layout()
    #             for tag in self.evolve_tags:
    #                 # ['fitness_train', 'parsimony', 'lentree', 'evolve_num', 'count']
    #                 axs[0].plot(self.monitor_df[tag]['fitness_train'], label=f'{tag}')
    #                 axs[1].plot(self.monitor_evol[tag]['parsimony'], label=f'{tag}')
    #                 axs[2].plot((self.monitor_evol[tag]['lentree'] / self.monitor_evol[tag]['evolve_num']),
    #                             label=f'{tag}')
    #
    #             plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
    #             path = self.rootdir / f'monitoring_evolutions.pdf'
    #             fig.savefig(path)
    #             self.printpl('f', f"monitoring_evolutions (pdf): {path}")  # .as_posix()
    #
    #     except Exception as ex:
    #         print_e(f'plot_evolution_analysis failed because of: {ex}')

    def tree_eval_meta(self, tree: Node, tag):
        """Evaluating the fitness of a tree.
        - extract the expression
        - try to simplify the expression (sympy)
        - create tensorflow-graph from sympy-expr and evaluate it

        Fitness values might evaluate to
            - 'nan' when dividing by zero or
            - 'inf' when 20**1234
        """
        parsimony = eval_parsimony(tree, self.tb.complexity_metric, origin_tree=self.origintree)
        if parsimony > self.tb.nodes_max:
            raise ValueError(f'Tree too complex: {parsimony} > {self.tb.nodes_max}')

        sy_expr = tree.get_sympy_expr()
        sym_control(sy_expr)  # raise if "weird" stuff in it
        # expr_raw = tree.get_expr_raw()
        # with sympy.evaluate(False):
        #     expr_raw = tree.get_sympy_expr()  # todo
        # v2_sym = expr_sympify(expr_raw)  # todo delete
        # sfeh:discuss sympy real=True might allow imaginary results

        fitness = self.kernel.eval_tf(sy_expr)['mean_error']
        # if DEBUG_DUMMY or fitness != self.kernel.eval_sym_experimental(sy_expr):
        #     print(f'FAILED: {fitness} vs. {self.kernel.eval_sym_experimental(sy_expr)}')

        meta = TreeMeta(fitness=fitness, parsimony=parsimony, expr_sym=sy_expr, tag=tag)
        self.lut[str(tree)] = meta  # sfeh:discuss: lut update in finalize_tree_get_meta()?
        return meta


def pop_analyze(popul, gen_time, gens_since_last_pareto):
    """Analysing the population (in each generation)
    - amount of trees
    - fittest tree
    - average fitness
    - average tree parsimony"""

    if len(popul) == 0:
        raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

    pop_fitness = [tree.get_fitness() for tree in popul]
    pop_parsim = [tree.get_parsimony() for tree in popul]
    pop_treelen = [len(fintree.tree) for fintree in popul]
    pop_fitness_best = np.min(pop_fitness)
    pop_unique = len(set([hash(x) for x in popul]))  # sfeh:analyze this?
    result = {'pop_len': len(popul),
              'pop_unique': pop_unique,
              'fit_avg': np.average(pop_fitness),
              'fit_var': np.std(pop_fitness),
              'fit_best': pop_fitness_best,
              'parsim_avg': np.average(pop_parsim),
              'parsim_var': np.std(pop_parsim),
              'parsim_best': np.min(pop_treelen),
              'time': gen_time,
              'gens_since_last_pareto': gens_since_last_pareto
              }
    return result


if __name__ == '__main__':
    """
    Alpha tests
    """
    # t1 = tb.invent_core_depth(float, 3, p_term=0.5)
    # tree2 = tb.evolve_mutate_point(t1)
    # t1 = tb.invent_core_depth(float, 3, p_term=0.1)
    # t2 = tb.invent_core_depth(float, 3, p_term=0.1)
    # t1, t2 = tb.evolve_crossover(t1, t2)
    # for _ in range(5):
    #     print('x.D', t1, '===', t2)
    #     t1, t2 = tb.evolve_crossover(t1, t2)
    #     print('x~D', t1, '===', t2)

    # nstr = '["+",["-",["Ifte",["True"],["sin",[2]],["/",[2.043],[4]]],["cartVel"]],[-1.3]]'
    nstr = '["+:fix",["-:fix",["Ifte",["True"],["sin",["2"]],["/",["2.043"],["4"]]],["cartVel"]],["-1.3"]]'
