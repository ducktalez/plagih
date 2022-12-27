"""
The main class of a gp run. It holds the following functionalities
- run informations (config, evolution-specifications for the loop, success monitoring])
- population (pop_base, pop_next)
-
"""
from plagih.monitoring import plot_gen_performance
from plagih.paretofront import *
from plagih.tree_complexity.tree_edit_distance import apted_distance
from plagih.tree_factory import *

import copy
from pathlib import Path
import numpy as np
import pandas as pd

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees


def eval_parsimony(tree: TreeNode, complexity_measure, origin_tree=None):
    """
    complexity_measure: compute the chosen distance by the user.
    #     'tree_node_count': tree_get_size,
    #     'tree_depth': tree_get_depth,
    #     'tree_edit_distance': tree_parsimony_ted,

    sfeh open: weights
    """
    if complexity_measure == 'tree_node_count':  # number of nodes
        return len(tree)  # returns the number of nodes  # sfeh weights
    elif complexity_measure == 'tree_edit_distance':  # tree_edit_distance, fintree-edit-distance
        apted1 = tree.eval_apted_notation()
        apted2 = origin_tree.eval_apted_notation()
        distance, mapping = apted_distance(apted1, apted2)  # sfeh the mapping could be useful somewhere
        return distance
    else:
        raise Exception(f'Complexity measurement not available: {complexity_measure}')


class ExplainableGP:

    def __init__(self, name, pop_max, gen_max, rootdir, kernel, complexity_measure, origintree, tb: TreeBuilder,
                 period_plots, period_save):
        self.time_start = time.perf_counter()
        self.kernel = kernel
        self.origintree = origintree
        self.name = name
        self.tb = tb
        self.pop_max = pop_max
        self.mp_cores = 1  # sfeh: open MP (multiprocessing)
        self.complexity_measure = complexity_measure
        self.gen_max = gen_max
        self.rootdir = rootdir
        self.gen_id = 0

        self.period_plots = period_plots
        self.period_save = period_save

        self.printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        self.paretofront = []  # not a separate class; requires too much information
        self.pop_next = []
        self.pop_base = []  # sfeh:discuss maybe better names?

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
        self.pop_base = []
        self.pop_next = []

    # self.printpl('i', 'Using evolve rates from config')

    def evoloop(self, gen_additionally=0):
        """
        Start a regular GP run.
        The only other option is to analyze a loaded run.
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.gen_max)
            self.gen_max = max(self.gen_max, self.gen_id + gen_additionally)
            self.printpl('i', f'Adding {gen_additionally} more generations in gen {self.gen_id}, '
                              f'increasing gen_max from {printdummy} to {self.gen_max}.')

        # sfeh check if any roots
        # yaml_dump(self.rootdir / 'used_config.yaml', self.conf)

        while self.gen_id <= self.gen_max and not self.run_custom_exit_condition():
            self.time_genstart = time.perf_counter()  # sfeh here?

            if self.gen_id == 0:
                self.printpl('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')  # sfeh debug
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
                printez('a', f'Starting a new paretofront with parsimony: {self.paretofront[0].get_parsimony()} '
                             f'fitness: {self.paretofront[0].get_fitness():6.4f}')

            # tmp_pareto = pareto_from_pop(self.pop_next)  # sfeh:optional paretofront in each generation?
            self.paretofront = self.run_update_paretofront(self.paretofront)

            gen_time = time.perf_counter() - self.time_genstart
            pop_analysis_dict = pop_analyze(self.pop_next)
            pop_analysis_dict['gen_time'] = gen_time
            pop_analysis_dict['gens_since_last_pareto'] = self.gens_since_last_pareto
            self.monitor_df.loc[self.gen_id] = pop_analysis_dict

            self.printpl('gg', f"Created {len(self.pop_next)}/{self.pop_max} ({pop_analysis_dict['pop_unique']}"
                               f" unique) in generation {self.gen_id}. Gen took {gen_time:4.2f}s")

            self.pop_base = self.pop_next[:]  # deepcopy
            self.pop_next = []

            self.printpl('ggg', f'Gen {self.gen_id} took: {time.perf_counter() - self.time_genstart:4.2f}.')
            self.evoloop_monitoring_scheduled_io(self.gen_id)
            self.gen_id += 1

        self.printpl('g', f'Done after Generation {self.gen_id}.\n'
                          f'Time since start: {time.perf_counter() - self.time_start:4.2f}s')

        self.run_backup(mode='save')

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
                printyeah('a', f'Paretofront: New entry. parsimony: {par} fitness: {fit:6.4f}')
                try:
                    # sfeh: trying to simplify the tree for a even improved pareto # sfeh:open
                    symtree = evolve_reduce_simplify(fintree.get_evotree(), force=True)
                    symmeta = self.finalize_tree_get_meta(symtree)
                    symtree_fin = FinalizedTree(symtree, symmeta)
                    if symtree_fin.get_parsimony() < fintree.get_parsimony():
                        printyeah('a', f'Paretofront: Even further simplified! '
                                       f'{symtree_fin.get_parsimony()} < {fintree.get_parsimony()}')

                    self.evaluate_and_append(symtree)  # sfeh:open , tag='sfeh-sympifyed_pareto'

                except Exception as ex:
                    print(f'SFEH: this tree could todo whatever {ex}')
                    # -> piecewise function, mostly
                obsolete_entries = [x for x in paretofront if
                                    x.get_fitness() > fintree.get_fitness() and
                                    x.get_parsimony() >= fintree.get_parsimony()]
                if obsolete_entries:
                    printyeah('a', f'Paretofront: Removing obsolete entries {[str(x) for x in obsolete_entries]}')
                paretofront = [ftree for ftree in paretofront if ftree not in obsolete_entries]
                paretofront.append(fintree)
                paretofront = pareto_sort(paretofront)

        return paretofront

    def gen_create_initial(self):

        if self.origintree is not None:
            self.pop_next.append(self.origintree)
            # self.pop_append_evotree(self.origin.fintree)  # sfeh why not :P
            # self.pareto.pareto_insert(self.origin)  # the origin fintree is the only candidate (automatically added)
        else:
            # total_rate = sum([x['evolve_rate'] for x in self.evolve_random.values()])
            #
            # for tag, evolve_specs in self.evolve_random.items():
            #     evolve_num = int(self.pop_max * (evolve_specs['evolve_rate'] / total_rate))
            #
            #     for nn in range(evolve_num):
            #         evotree = tb.pop_random(goaldepth_randomizer=None, goalnodes_randomizer, p_full, origin=self.origin)
            #         self.genloop_performance_append_tree(evotree, tag)
            @self.create_trees(1)
            def rand1():
                return self.tb.pop_random_depth(np.clip(int(random.normalvariate(4, 1)), 1, 5), p_full=1, xtype=float)

            # def rand2():
            #     # todo float? nope
            #     return [self.tb.pop_random_depth(np.clip(int(random.normalvariate(3.5, 1)), 2, 5), p_full=1, xtype=float)]
            # self.create_trees(0.5, rand2)
        return

    def lut_to_meta(self, evotree):
        if DEBUG_DUMMY:
            # if trees are 100% safely created, tree checks are not required. Useful when trying out new gp-operators.
            self.tb.check_all(evotree, fatal=False)  # sfeh fatal=True? (raise)

        try:
            meta = self.lut[str(evotree)]  # fixed nodes not relevant
        except KeyError:
            try:
                meta = self.finalize_tree_get_meta(evotree)
            except ValueError as ex:
                print_warning('www', f'ValueError: {ex}')
                raise
            except ArithmeticError as ex:
                print_warning('www', f'ArithmeticError: {ex}')
                raise
            except TypeError as ex:
                print_warning('ww', f'TypeError: {ex}')
                raise
            except Exception as ex:
                print_warning('ww', f'Could not append tree to population because: {ex}\n'
                                    f'=>tree: {evotree}')
                raise
        return meta

    def create_trees(self, rate):
        """Safely append a fintree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the fintree is refurbished.
        - Enrich the raw fintree for the next generation
        - check if the fintree is actually valid"""
        # self.printpl('gggg', f'->Evolving \'{createTreeFunc.__name__}\' {n}x starting...')

        def loop(createTreeFunc):
            n = rate * self.pop_max
            n_success = 0
            n_fails = 0
            while n_success < n:
                try:
                    # evotree = selection_tournament(self.pop_base, tournsize=3)
                    evotree = createTreeFunc()  # self.tb.evolve_mutate_branch_nodes(evotree, np.clip(int(random.normalvariate(12, 4)), 0, 30))
                    self.evaluate_and_append(evotree)
                    # self.printpl('ggggg', f'Success with tree: {evotree}')
                    n_success += 1
                except (ValueError, ArithmeticError, TypeError):
                    n_fails += 1

        return loop

    def create_trees_crossover(self, rate):
        """sfeh:workaround, there are better solutions"""

        def loop(createTreeFunc):
            n = rate * self.pop_max
            n_success = 0
            while n_success < n:
                try:
                    t1, t2 = createTreeFunc()
                    self.evaluate_and_append(t1)
                    self.evaluate_and_append(t2)
                    # self.printpl('ggggg', f'Success with 2xtrees: {t1}, {t2}')
                    n_success += 2
                except Exception as ex:
                    print(f'{ex}')

        return loop

    def evaluate_and_append(self, evotree):
        meta = self.lut_to_meta(evotree)
        fintree = FinalizedTree(evotree, meta)
        # sfeh: tag
        self.pop_next.append(fintree)

    def gen_create_next(self):
        """
        Creates all new Generations by applying the evolutions in the evolve-loop.

        brainstorm:
        - the fintree can be reproduced (selection), random/new, olymp-reproduction
        - (1 fintree) mutations can affect a point, branch, terminal nodes
        - (2 trees) can make a crossover
        """

        # All gp creators: name, function, num of trees from tournament selection
        @self.create_trees(0.1)
        def repro1():
            return selection_tournament(self.pop_base, tournsize=3)

        # self.create_trees(0.10, repro1)

        # sfeh:xxx repro-sympify
        @self.create_trees(0.1)
        def mutateB():
            evotree = selection_tournament(self.pop_base, tournsize=3)
            return self.tb.evolve_mutate_branch_nodes(evotree, 4)  # sfeh:todo also

        @self.create_trees_crossover(0.2)
        def xover():
            # try:
            t1 = selection_tournament(self.pop_base, tournsize=3)
            t2 = selection_tournament(self.pop_base, tournsize=3)
            # except ValueError as ex:
            #     print_warning("www", f'Crossover failed: {ex}')
            #     # sfeh: mostly 1-noded trees
            evo1, evo2 = self.tb.evolve_crossover(t1, t2)
            return evo1, evo2

        @self.create_trees(0.1)
        def rand1():
            return self.tb.pop_random_depth(np.clip(int(random.normalvariate(4, 1)), 1, 5), p_full=1, xtype=float)

        @self.create_trees(0.1)
        def rand2():
            # todo float? nope
            return self.tb.pop_random_depth(np.clip(int(random.normalvariate(3.5, 1)), 2, 5), p_full=1, xtype=float)

        # @self.create_trees(0.1)
        # def reproSym():
        #     evotree = selection_tournament(self.pop_base, tournsize=3)
        #     return evolve_reduce_simplify(evotree, completely=False)

        @self.create_trees(0.1)
        def mutatePoint():
            evotree = selection_tournament(self.pop_base, tournsize=3)
            return self.tb.evolve_mutate_point(evotree)

        # @self.create_trees(0.1)
        # def mxPointXXX():
        #     evotree = selection_tournament(self.pop_base, tournsize=3)
        #     return self.tb.evolve_mutate_pointxxx(evotree)

        @self.create_trees(0.1)
        def mxBranchD():
            evotree = selection_tournament(self.pop_base, tournsize=3)
            return self.tb.evolve_mutate_branch_depth(evotree, self.tb.depth_max, p_full=0.5)

        @self.create_trees(0.1)
        def mxBranchN():
            evotree = selection_tournament(self.pop_base, tournsize=3)
            nodeamount_goal = np.clip(int(random.normalvariate(12, 4)), 0, 30)
            return self.tb.evolve_mutate_branch_nodes(evotree, nodeamount_goal)

        @self.create_trees(0.1)
        def filterOptimize():
            evotree = selection_tournament(self.pop_base, tournsize=3)
            return self.tb.evolve_mutate_filter_random(evotree)

        @self.create_trees(0.1)
        def pareto_revive():
            fintree = np.random.choice(self.paretofront)
            return fintree.tree

        return

    def evoloop_monitoring_plots(self):
        """
        Create all run-related analysis plots in the root directory
        """
        try:
            path_monitoring = self.rootdir / 'monitoring.png'
            plot_gen_performance(self.monitor_df, self.name, path_monitoring)  # largest plot analysing the
            self.printpl('f', f"monitoring: {path_monitoring}")  # .as_posix()
            pareto_plot(self.paretofront, self.rootdir, self.name, self.tb.nodeamount_max)
        except Exception as ex:
            self.printpl("e", f'Could not create plots: {ex}\n')

    def evoloop_monitoring_scheduled_io(self, gen_id):
        """
        Every x generations, save a backup and/or save plots
        """
        plot_gen = self.period_plots
        if gen_id >= plot_gen and gen_id % plot_gen == 0:
            self.evoloop_monitoring_plots()

        save_gen = self.period_save
        if gen_id >= save_gen and gen_id % save_gen == 0 or gen_id == 10:
            self.run_backup(mode='save')

    def run_backup(self, path_load_custom_backup=None, mode='load'):
        """
        Load/safe backup of a run
        """

        path_backup = self.rootdir / 'backup/backup.pkl'

        if mode == 'save':
            # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
            run_backup_data = {}, self.gen_id, self.pop_base, self.paretofront, self.monitor_df  # sfeh use this later, help_dict
            path_backup = path_make_dir(path_backup)
            pickle_dump(path_backup, run_backup_data)
            # sfeh:debug

        elif mode == 'load':
            # run_backup_load(self, argpath_backup):
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

    # def plot_evolve_performance(self):
    #     """
    #     Plots for each tag in the evolution list
    #     (too much, i guess)
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
        parsimony = eval_parsimony(evotree, self.complexity_measure, origin_tree=self.origintree)
        if parsimony > self.tb.nodeamount_max:
            # sfep:discuss: information about last evolution? currently not saved in tree. Is this not autopruned?
            raise ValueError(f'Tree too complex: {parsimony} > {self.tb.nodeamount_max}')

        expr_raw = evotree.eval_expr_str()
        expr_sym = expr_sympify(expr_raw)

        try:
            fitness = self.kernel.eval_tf(expr_sym)['mean_error']
            # if fitness != fitness or fitness == float('inf'):
            #     raise Exception(f"fitness_train is: '{fitness}'")  # eg very wrong values that exceed the float-range
        except ValueError as ex:
            raise ValueError(f'eval-ex nan: {ex}')
        except TypeError as ex:
            # Value passed to parameter 'x' has DataType bool not in list of allowed values: bfloat16, float16, float32, float64, int8, int16, int32, int64, complex64, complex128
            # probably because datatypes in trees do not match (xtypes, build). Happens in random tree generation
            raise TypeError(f'eval-ex-te: {ex}')
        except Exception as ex:
            raise Exception(f'eval-ex: {ex}')

        meta = TreeMeta(fitness=fitness, parsimony=parsimony, expr_raw=expr_raw, expr_sym=expr_sym)

        self.lut[str(evotree)] = meta  # sfeh:discuss: lut update in finalize_tree_get_meta()?

        return meta

    def printpl(self, message_type, message_str):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        message_type options can be found in config
        """
        printez(message_type, message_str)
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


def printpl(message_type, message_str):
    """
    Lightweight print function.
    Instead of checking if you should print every time, this is done here.
    message_type options can be found in config
    """
    printez(message_type, message_str)
    return


if __name__ == '__main__':
    """
    Alpha tests
    """
    tb = TreeBuilder(obs_names=['cartPos', 'cartVel'])
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
    tr = node_from_nested_labels(nstr)
    check_tree_loadable_reconstruction(tr)
