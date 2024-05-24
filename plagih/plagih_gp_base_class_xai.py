"""
The main class of a gp run. It holds the following functionalities
- run information (config, evolution-specifications for the loop, success monitoring])
- population (pop_base, pop_next)
-
"""
import re
from collections import deque

from plagih.monitoring import plot_performance
from plagih.paretofront import *
from plagih.tree_factory import *

from pathlib import Path
import numpy as np
import pandas as pd

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


def printpl(msg_type, message_str):
    """Lightweight print function.
    Instead of checking if you should print every time, this is done here.
    message_type options can be found in config
    """
    printez(msg_type, message_str)
    return


class Candidate:
    """
    WAS: class FinalizedTree
    An actual individual (Tree + meta-infos/phenotypes)"""

    def __init__(self, tree: Node, fitness, parsimony, tag: str):
        self.tree = tree
        self.fitness = fitness
        self.parsimony = parsimony
        self.last_evolution = deque([tag], maxlen=10)  # sfeh:open

    def append_tag(self, tag):
        self.last_evolution.append(tag)

    def get_last_tag(self):
        return self.last_evolution[-1]

    def __str__(self):
        """Show the Parsimony and Fitness of a tree"""
        return f'[{self.get_parsimony():2.0f}: fit {self.get_fitness():4.2f}]'

    def full_string(self):
        return f'{self.__str__()}: {self.get_evotree().get_sympy_expr()}'

    def get_evotree(self):
        return self.tree

    def append_tag(self, tag):
        self.meta.append_tag(tag)

    def get_fitness(self):
        # return self.meta.fitness
        return self.fitness

    def get_parsimony(self):
        return self.parsimony

    def set_fitness(self, fitness):
        self.meta.fitness = fitness

    def set_parsimony(self, parsimony):
        self.meta.parsimony = parsimony

    def get_last_evolution(self):
        return self.meta.get_last_tag()  # sfeh same name?


class ExplainableGP:

    def __init__(self, name, pop_max, gen_max, rootdir, kernel, tb: Evolution):
        self.time_start = time.perf_counter()
        self.kernel = kernel
        self.name = name
        self.tb = tb
        self.pop_max = pop_max
        self.mp_cores = 1  # sfeh: open MP (multiprocessing)
        self.gen_max = gen_max
        self.rootdir = rootdir
        self.gen_id = 0

        printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET_COLOR}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        self.paretofront = []  # not a separate class; requires too much information
        self.pop_genepool = []  # sfeh:discuss maybe better names?
        self.pop_next = []

        self.lut_sym = {}
        self.lut_parsim = {}
        self.lut_fitness = {}  # Lookup-table for tree(-expressions) and its fitness/parsimony. Improving runtime a lot!

        # monitoring
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0
        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'parsim_avg', 'parsim_var', 'parsim_best',
                                                'gens_since_last_pareto'])

    # todo
    # WHATTPPENDED
    # SFEH
    # [Div, [2.81], [cartVel]]
    # [Mul, [2.81], [DivFraction, [cartVel]]]
    # 2.81 / cartVel
    # 2.81 / cartVel

    def pop_print(self):
        """Print the expressions of all trees in a population"""
        n = [f'{k.full_string()}' for k in self.pop_next]
        n = [f'{BColors.BLUE}{x}' if ii % 2 == 0 else f'{BColors.YELLOW}{x}' for ii, x in enumerate(n)]
        n = [f'{k}\n' if ii % 10 == 9 else f'{k}\t' for ii, k in enumerate(n)]  # stop \n in line 0
        n = ''.join(n)
        n = re.sub(r'\n$', '', n)  # remove trailing \n (\t irrelevant)
        n = f'{n}{BColors.RESET_COLOR}'
        print(n)

    def run_update_paretofront(self, pop):
        """
        CAUTION: This Function was tried to be separated many times now. it never worked.
        This was tried =3= times now. Please increase the counter when you try.
        Reason: The paretocandidates should be simplified if possible and gens_since_last_pareto is reset.

        sfeh:discuss pareto-efficient, but different pareto entries?


        """
        pop_parcandidates = pareto_from_pop(pop)  # pareto-candidates in the pop, renamed to be clear

        for candidate_tree in pop_parcandidates:
            success = False
            fit = candidate_tree.get_fitness()
            par = candidate_tree.get_parsimony()

            if par < self.paretofront[0].get_parsimony():
                printyeah('a', f'Paretofront: New simplest entry. parsimony: {par} fitness: {fit:6.4f}, '
                               f'old simplest entry had {self.paretofront[0].get_parsimony()}')
                success = True

            # if all([self.fitness_compare(fit, p.get_fitness()) for p in self.paretofront]):  # sfeh-kernel
            elif fit < self.paretofront[-1].get_fitness():
                printyeah('a', f'Paretofront: New fittest entry. parsimony: {par} fitness: {fit:6.4f}')
                success = True
            else:
                for p in self.paretofront:
                    if par >= p.get_parsimony():
                        continue
                    else:
                        if fit < p.get_fitness():
                            success = True

            if success:
                self.gens_since_last_pareto = 0
                try:  # sfeh: trying to simplify the tree for improved pareto
                    symtree = evolve_reduce_simplify(candidate_tree.get_evotree(), force=True)
                    sym_candidate = self.tree_to_candidate(symtree, tag='sfeh:sym')
                    if sym_candidate.get_parsimony() < candidate_tree.get_parsimony():
                        printyeah('a', f'Paretofront: Even further simplified! '
                                       f'{sym_candidate.get_parsimony()} < {candidate_tree.get_parsimony()}')
                        self.pop_next_append(sym_candidate, force=True)

                    print(blue_string(f'Simplified symtree: {sym_candidate.get_parsimony()}: {symtree}'))

                except KeyError as ex:
                    print_caution(f'SFEH: this tree could whatever {ex}')  # -> piecewise function, mostly

                _obsoletes = [i for i in self.paretofront if
                              i.get_fitness() > candidate_tree.get_fitness() and i.get_parsimony() >= candidate_tree.get_parsimony()]
                if _obsoletes:
                    x = [f'{i.full_string()}' for i in _obsoletes]
                    printyeah('a', f'Paretofront: Removing obsolete entries {x}')
                self.paretofront = [ftree for ftree in self.paretofront if ftree not in _obsoletes]
                self.paretofront.append(candidate_tree)
                self.paretofront = pareto_sort(self.paretofront)

        return

    def end_generation(self):
        # sfeh:open end generation in every generation
        self.run_update_paretofront(self.pop_next)

        self.pop_genepool = self.pop_next[:]
        self.pop_print()
        self.pop_next = []
        self.analyze_generation()
        self.gen_id += 1

        self.time_genstart = time.perf_counter()

    # sfeh:idea sympy.NumberSy,bol

    def gen_create_initial(self):

        printpl('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')  # sfeh debug

        if self.tb.origin_tree is not None:
            cand_origin = self.tree_to_candidate(self.tb.origin_tree, raise_if_useless=False, tag='origin')
            self.pop_next_append(cand_origin)
        else:
            if CHAINED_VERION:
                @self.create_trees(rate=0.5)
                def init_rand1():
                    n = np.clip(int(random.normalvariate(4.0, 1.0)), 3, 6)
                    tree = self.tb.evolve_new_tree_depth(n, float, p_term=0)
                    tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                    # sfeh trees can shrink to single-noded trees
                    if tree.get_max_depth() <= 1:
                        raise ValueError(f'Tree did not get complex enough')
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2():
                    n = np.clip(int(random.normalvariate(3.5, 1.0)), 3, 6)
                    tree = self.tb.evolve_new_tree_depth(n, float, p_term=0)
                    tree = tree_simplification(tree, allow_chain=CHAINED_VERION)
                    return tree
            else:
                @self.create_trees(rate=0.5)
                def init_rand1a():
                    n = np.clip(int(random.normalvariate(3.0, 1.0)), 3, 5)
                    tree = self.tb.evolve_new_tree_depth(n, float, p_term=0)  # sfeh: xtype not always float
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2a():
                    n = np.clip(int(random.normalvariate(2.5, 1.0)), 3, 5)
                    return self.tb.evolve_new_tree_depth(n, float, p_term=0)

        self.paretofront = pareto_from_pop(self.pop_next)  # initialize
        self.pop_genepool = self.pop_next[:]
        self.pop_next = []
        self.analyze_generation()
        self.gen_id += 1
        return

    def pop_next_append(self, ct: Candidate, force=False):
        evotree = ct.get_evotree()
        # from visualization.pygraphviz import render_pygraphviz
        if force and ct.get_parsimony() < TREE_MIN_PARSIMONY:
            # sfeh raise ValueError(f'Tree not complex enough for population, sfeh')
            return
        printpl('gggg', f'|->{evotree.len_nodecount_fair():2.0f}: {evotree.str_as_expr()}')
        self.pop_next.append(ct)

    def create_trees(self, rate=0.0, crossover=False):
        """Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the final tree (candidate_tree) is refurbished."""

        def loop(create_tree_func):
            n = int(rate * self.pop_max)
            n_success = 0
            n_fails = 0
            tag = create_tree_func.__name__
            printpl('ggg', f'->Evolving {n}x \'{tag}\'...')

            while n_success < n:
                try:
                    if crossover:
                        t1, t2 = create_tree_func()
                        ctree1 = self.tree_to_candidate(t1, tag=tag)
                        self.pop_next_append(ctree1)
                        n_success += 1
                        ctree2 = self.tree_to_candidate(t2, tag=tag)
                        self.pop_next_append(ctree2)
                        n_success += 1
                    else:
                        evotree = create_tree_func()
                        ctree = self.tree_to_candidate(evotree, tag=tag)
                        self.pop_next_append(ctree)
                        n_success += 1
                        # todo force at least one input variable

                except (ValueError, ArithmeticError) as ex:
                    n_fails += 1
                    print_warning('www', f'\'{tag}\' failed: {ex}')
                    if n_fails > 2*n_success + 5:  # allow more fails: n_fails > n
                        print_caution(f'Evolution fails too often: {tag}, {n_fails}x. ({n_success}x successful).')
                        return  # sfeh raise?
                except TypeError as ex:
                    if str(ex) == "Cannot convert complex to float":
                        pass
                    else:
                        print(f'Typeerror, but why? {ex}')  # ok: TypeError: Cannot convert complex to float
                    # todo: Typeerror, but why? 'NoneType' object is not subscriptable -> ?
                    # Problem: TypeError: Typeerror, but why? expecting bool or Boolean, not `(a - 0.53 <= -0.361, a - 0.801 < v/a**1.0)`
                    # Value passed to parameter 'x' has DataType bool not in list of allowed values: bfloat16, float16..
                    # sfeh probably this error: cond(): 'false_fn' argument required
                    # Happens, when ITE is coming up. Ignoring for now.
                except AttributeError as ex:
                    print(f'("Okay", if sympy.im in expr) {ex}')
                #     # raise AttributeError(f'Probably sympy.im in expr {ex}')
                #     # AttributeError: Probably sympy.im in expr 'int' object
                #     raise AttributeError(f'AttributeError: {ex}')
                #     # f'\n\tint object has no attribute get_nodes_at_depth has no attribute get_nodes_at_depth
                #     # f'\n\tProbably sympy.im in expr has no attribute get_nodes_at_depth
                #     # print(f'Probably sympy.im in expr {ex}')
                #     #  AttributeError: 'Xor' object has no attribute '_eval_as_set'
                #     #    |--AttributeError: Probably sympy.im in expr 'Xor' object has no attribute '_eval_as_set'
                except KeyError as ex:
                    # KeyError(re) -> okay?, real part implies complex numbers, ignoring is okay
                    print(f'Keyerror, (probably sympy.lambdify expression not evaluable): {ex}')
                except RecursionError as ex:
                    print(f'RecursionError (probably Piecewise/relational combination?): {ex}')
                except NotImplementedError as nie:
                    print_caution(f'Notimplemented? {nie}')

        return loop

    def tree_to_candidate(self, evotree: Node, tag=None, raise_if_useless=True):
        """the "fixed" node information is not relevant"""

        evotree.force_input_node(self.tb)
        evotree.repair_depth()

        tree_id = evotree.get_lut_id()

        if tree_id in self.lut_sym:
            sy_expr = self.lut_sym[tree_id]
        else:
            sy_expr = evotree.get_sympy_expr()
            # sy_expr = sym_check(sy_expr)  # raise if "weird" stuff in it
            self.lut_sym[tree_id] = sy_expr
            sym_check(sy_expr)

        if tree_id in self.lut_parsim:
            parsimony = self.lut_parsim[tree_id]
        else:
            parsimony = eval_parsimony(evotree, self.tb.complexity_metric, origin_tree=self.tb.origin_tree)
            self.lut_parsim[tree_id] = parsimony
            if raise_if_useless and parsimony > self.tb.nodes_max:  # sfeh:open
                raise ValueError(f'Tree too complex: {parsimony} > {self.tb.nodes_max}')

        if sy_expr in self.lut_fitness:
            fitness = self.lut_fitness[sy_expr]
        else:
            # sfeh:discuss sympy real=True might allow imaginary results
            # t0 = time.perf_counter()
            fitness = self.kernel.eval_sym_experimental(sy_expr)
            # t1 = time.perf_counter()
            # fitness2 = self.kernel.eval_tf(sy_expr)['mean_error']
            # t2 = time.perf_counter()
            # print(f'asd {t1-t0:4.4f} {t2-t1:4.4f} ({(t1-t0)-(t2-t1):4.2f}) {fitness:4.2f} {fitness2:4.2f}')
            # if DEBUG_DUMMY or fitness != self.kernel.eval_sym_experimental(sy_expr):
            #     print(f'FAILED: {fitness} vs. {self.kernel.eval_sym_experimental(sy_expr)}')

            self.lut_fitness[sy_expr] = fitness  # sfeh:discuss: lut update in finalize_tree_get_meta()?
        
        # return fitness, parsimony, sy_expr
        
        candidate = Candidate(evotree, fitness=fitness, parsimony=parsimony, tag=tag)
        return candidate

    def evoloop_monitoring_plots(self):
        """
        Create all run-related analysis plots in the root directory
        """
        try:
            plot_performance(self.monitor_df, self.name, self.rootdir / 'monitoring.png')
            plot_paretofront(self.paretofront, self.rootdir, self.name, self.tb.nodes_max)
        except Exception as ex:
            printpl("e", f'Could not create plots: {ex}\n')

    def backup_save(self, opt_path_backup=None):
        """
        Load/safe backup of a run
        """

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
        run_backup_data = {}, self.gen_id, self.pop_genepool, self.paretofront, self.monitor_df
        path_backup = path_make_dir(path_backup)
        pickle_dump(path_backup, run_backup_data)
        # sfeh:debug

    def backup_load(self, opt_path_backup=None):
        """Load/safe backup of a run"""

        path_backup = opt_path_backup or self.rootdir / 'backup/backup.pkl'

        if Path.is_file(path_backup):
            printpl('g', f'Loading data from backup-file {path_backup}')
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)
            except NotImplementedError as ex:
                raise Exception(f'NotImplementedError: {ex}')
            except EOFError as ex:
                raise Exception(f'EOFError: \n{ex}')

            help_dict, self.gen_id, self.pop_genepool, self.paretofront, self.monitor_df = run_data
            self.backup_save(opt_path_backup=self.rootdir / f'backup/backup-{self.gen_id}.pkl')  # sfeh:dis date?
            printpl('g', f'Successfully loaded backup file. Generation: {self.gen_id}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}')  # sfeh:beautify occurs 2x

    def analyze_generation(self):
        gen_time = time.perf_counter() - self.time_genstart
        tmp_dict = pop_analyze(self.pop_genepool, gen_time, self.gens_since_last_pareto)
        self.monitor_df.loc[self.gen_id] = tmp_dict
        printpl('gg', f"Created {len(self.pop_genepool)}/{self.pop_max} ({tmp_dict['pop_unique']} unique) in generation {self.gen_id}. "
                      f"Trees in LUT: {len(self.lut_fitness)} Gen took {gen_time:4.2f}s")

        printpl('ggg', f'--- Gen {self.gen_id} took: {time.perf_counter() - self.time_genstart:4.2f}. ---')

        # def monitoring_scheduled_io(self, gen_id, plots_interval=10, backup_interval=10):
        """
        Every x generations, save a backup and/or save plots
        """
        if self.gen_id >= PLOTS_INTERVAL and self.gen_id % PLOTS_INTERVAL == 0:
            self.evoloop_monitoring_plots()

        if self.gen_id >= BACKUP_INTERVAL and self.gen_id % BACKUP_INTERVAL == 0 or self.gen_id == 10:
            self.backup_save()

    def run_custom_exit_condition(self):
        """
        Special condition to exit the evolve-loop
        1. when >100 generations, no new paretofront were found
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
    pop_treelen = [len(candidate_tree.tree) for candidate_tree in popul]
    pop_fitness_best = np.min(pop_fitness)
    pop_unique = len(set([str(x.tree) for x in popul]))  # sfeh:analyze this?
    # sfeh:idea add the amount of actually new trees (compare with the LUT tree_ids)
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
