from plagih.evaluation import eval_predict_df
from plagih.evolve import Evolution
from plagih.monitoring import plot_performance
from plagih.paretofront import *

from plagih.trees import *
from plagih.util import *


class Candidate:
    """
    WAS: class FinalizedTree
    An actual individual (Tree + meta-infos/phenotypes)"""

    def __init__(self, tree: Node, fitness, parsimony, tag: str):
        self.tree = tree
        self.fitness = fitness
        self.parsimony = parsimony
        # self.last_evolution = deque([tag], maxlen=10)  # sfeh:open

    # def append_tag(self, tag):
    #     self.last_evolution.append(tag)
    #
    # def get_last_tag(self):
    #     return self.last_evolution[-1]

    def __str__(self):
        """Show the Parsimony and Fitness of a tree"""
        return f'[{self.get_parsim():2.0f}: fit {self.get_fitness():4.2f} ({self.tree.__str__()})]'

    def full_string(self):
        return f'{self.__str__()}: {BColors.BOLD}{self.get_evotree().get_sympy_expr()}{BColors.RESET}'

    def get_evotree(self):
        return self.tree

    def get_fitness(self):
        # return self.meta.fitness
        return self.fitness

    def get_parsim(self):
        return self.parsimony


class ExplainableGP:
    """
    sfeh:open
        - class Population -> pop_max should be part of a pop_creation_list?
        - class data-specific/eval -> df_train, normalize_numpy
        - class
    """
    def __init__(self, evolve: Evolution, df_train, rootdir=None, allow_chain=False, normalize_numpy=None, pop_max=100, gen_max=15):
        self.time_start = time.perf_counter()
        if rootdir is None:
            self.rootdir = None
        elif isinstance(rootdir, Path):
            self.rootdir = rootdir
        else:
            raise NotImplementedError
        self.df_train = df_train
        # self.df_control = df_control
        self.evolve = evolve
        self.pop_max = pop_max
        self.gen_max = gen_max
        self.gen_id = 0
        self.normalize_numpy = normalize_numpy
        self.allow_chain = allow_chain
        # self.mp_cores = 1  # sfeh: open MP (multiprocessing)

        # printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        print(f'\n'
              f'\tInitializing Plagih.\n'
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
                                                'fit_avg', 'fit_var',
                                                'fit_quantile_25', 'fit_quantile_50', 'fit_quantile_75', 'fit_best',
                                                'parsim_avg', 'parsim_var', 'parsim_quantile_25', 'parsim_quantile_50',
                                                'parsim_quantile_75', 'parsim_best',
                                                'gens_since_last_pareto'])

    def get_name(self):
        if isinstance(self.rootdir, Path):
            s = self.rootdir.name
        else:
            s = None
        return s

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
            par = candidate_tree.get_parsim()

            if par < self.paretofront[0].get_parsim():
                printez('a', f'Paretofront: New simplest entry. parsimony: {par} fitness: {fit:6.4f}, '
                               f'old simplest entry had {self.paretofront[0].get_parsim()}')
                success = True

            # if all([self.fitness_compare(fit, p.get_fitness()) for p in self.paretofront]):  # sfeh-kernel
            elif fit < self.paretofront[-1].get_fitness():
                printez('a', f'Paretofront: New fittest entry. parsimony: {par} fitness: {fit:6.4f}')
                success = True
            else:
                for p in self.paretofront:
                    if par >= p.get_parsim():
                        continue
                    else:
                        if fit < p.get_fitness():
                            success = True

            if success:
                self.gens_since_last_pareto = 0
                try:  # sfeh: trying to simplify the tree for improved pareto
                    symtree = evolve_reduce_simplify(candidate_tree.get_evotree(), self.allow_chain, force=True)
                    sym_candidate = self.tree_to_candidate(symtree, tag='sfeh:sym')
                    if sym_candidate.get_parsim() < candidate_tree.get_parsim():
                        printez('a', f'Paretofront: Further simplified! {sym_candidate.get_parsim()} < {candidate_tree.get_parsim()}')
                        self.pop_next_append(sym_candidate, force=True)

                    print(blue_string(f'Simplified symtree: {sym_candidate.get_parsim()}: {symtree}'))

                except KeyError as ex:
                    print_caution(f'SFEH: this tree could whatever {ex}')  # -> piecewise function, mostly

                _obsoletes = [i for i in self.paretofront if
                              i.get_fitness() > candidate_tree.get_fitness() and i.get_parsim() >= candidate_tree.get_parsim()]
                if _obsoletes:
                    x = [f'{i.full_string()}' for i in _obsoletes]
                    printez('a', f'Paretofront: Removing obsolete entries {x}')
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

    def gen_create_initial(self, origin_tree=None):
        """
        sfeh:prio4 this is a very specific function...
        """
        printpl('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')  # sfeh debug

        if origin_tree is not None:
            cand_origin = self.tree_to_candidate(origin_tree, raise_if_useless=False, tag='origin')
            self.pop_next_append(cand_origin)
        else:
            if self.allow_chain:
                @self.create_trees(rate=0.5)
                def init_rand1():
                    n = np.clip(int(random.normalvariate(4.0, 1.0)), 3, 6)
                    tree = self.evolve.evolve_new_tree_depth(n, float, p_term=0)
                    tree = tree_simplification(tree, allow_chain=self.allow_chain)
                    # sfeh trees can shrink to single-noded trees
                    if tree.get_max_depth() < 1:
                        raise TreeSizeError(f'Tree did not get complex enough')
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2():
                    n = np.clip(int(random.normalvariate(3.5, 1.0)), 3, 6)
                    tree = self.evolve.evolve_new_tree_depth(n, float, p_term=0)
                    tree = tree_simplification(tree, allow_chain=self.allow_chain)
                    return tree
            else:
                @self.create_trees(rate=0.5)
                def init_rand1a():
                    n = np.clip(int(random.normalvariate(3.0, 1.0)), 3, 5)
                    tree = self.evolve.evolve_new_tree_depth(n, float, p_term=0)  # sfeh: xtype not always float
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2a():
                    n = np.clip(int(random.normalvariate(2.5, 1.0)), 3, 5)
                    return self.evolve.evolve_new_tree_depth(n, float, p_term=0)

        self.paretofront = pareto_from_pop(self.pop_next)
        self.pop_genepool = self.pop_next[:]
        self.pop_next = []
        self.analyze_generation()
        self.gen_id += 1
        return self.pop_genepool

    def pop_next_append(self, ct: Candidate, force=False):
        evotree = ct.get_evotree()
        # from visualization.pygraphviz import render_pygraphviz
        if force and ct.get_parsim() < TREE_MIN_PARSIMONY:
            # sfeh raise ValueError(f'Tree not complex enough for population, sfeh')
            return
        printpl('gggg', f'|->{evotree.len_nodecount_fair():2.0f}: {evotree.str_as_expr()}')
        self.pop_next.append(ct)

    def create_trees(self, rate=0.0, crossover=False, simplify=False):
        """Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the final tree (candidate_tree) is refurbished."""

        def loop(create_tree_func):
            n = int(rate * self.pop_max)
            n_success = 0
            fails_list = []
            tag = create_tree_func.__name__
            printpl('ggg', f'->Evolving {n}x \'{tag}\'...')

            while n_success < n:
                try:
                    if crossover:
                        t1, t2 = create_tree_func()
                        # if simplify:
                        #     t1 = tree_simplification(t1, allow_chain=self.allow_chain)
                        #     t2 = tree_simplification(t2, allow_chain=self.allow_chain)
                        ctree1 = self.tree_to_candidate(t1, tag=tag)
                        self.pop_next_append(ctree1)
                        n_success += 1
                        ctree2 = self.tree_to_candidate(t2, tag=tag)
                        self.pop_next_append(ctree2)
                        n_success += 1
                    else:
                        evotree = create_tree_func()
                        # if simplify:
                        #     evotree = tree_simplification(evotree, allow_chain=self.allow_chain)
                        ctree = self.tree_to_candidate(evotree, tag=tag)
                        self.pop_next_append(ctree)
                        n_success += 1

                except (TreeSizeError, SympySimplificationError) as ex:

                    fails_list.append(ex)
                    print_warning('www', f'\'{tag}\' failed: {ex}')
                    if len(fails_list) > 2 * n_success + 5:  # allow more fails: fails_list > n
                        print_caution(f'Evolution fails too often: {tag}, {len(fails_list)}. ({n_success} successful).'
                                      f'\n{fails_list}')
                        return  # sfeh raise?

                # except (ValueError, ArithmeticError) as ex:
                #     # if 'Crossover tree 1 has no mutable nodes!' in str(ex):
                #     # if "'a' cannot be empty unless no samples are taken" in str(ex):
                #     print_warning('ww', f'OnlyPrintException: Value/Arithmetic: {ex}')
                except TypeError as ex:
                    # Value passed to parameter 'x' has DataType bool not in list of allowed values: bfloat16, float16..
                    # ==> sfeh probably this error: cond(): 'false_fn' argument required
                    # ==> Happens, when ITE is coming up. Ignoring for now.
                    if str(ex) == "Cannot convert complex to float":
                        pass
                # except AttributeError as ex:
                # todo comented
                #     print(f'OnlyPrintException: (Okay, if sympy.im in expr) {ex}')
                except KeyError as ex:
                    # KeyError(re) -> okay?, real part implies complex numbers, ignoring is okay
                    # (probably sympy.lambdify expression not evaluable)
                    print(f'OnlyPrintException: Keyerror?: {ex}')
                except RecursionError as ex:
                    print(f'OnlyPrintException: RecursionError (probably Piecewise/relational combination?): {ex}')
                # except NotImplementedError as nie:
                #     print_caution(f'Notimplemented? {nie}')
                # except Exception as ex:
                #     print(f'OnlyPrintException: Why are we not here??? {ex}')
        return loop

    def eval_fitness(self, sy_expr):

        df_results = eval_predict_df(sy_expr, self.df_train, normalize_numpy=self.normalize_numpy)
        # pairwise_results = pairwise_results['result']
        fitness = np.sqrt(np.mean((df_results - self.df_train['action']) ** 2))  # discuss: np.square vs. **2: should be mainly irrelevant
        fitness = round(fitness, FLOAT_PRECISION)
        return fitness

    def tree_to_candidate(self, evotree: Node, origin_tree=None, tag=None, raise_if_useless=True):
        """the "fixed" node information is not relevant

        Tree MUST NOT be altered from here!
        raise_if_useless is here in order to show, where the maximum nodes is exceeded!
        """

        evotree.force_input_node(self.evolve)
        evotree.repair_depth()

        tree_id = evotree.get_lut_id()

        if tree_id in self.lut_sym:
            sy_expr = self.lut_sym[tree_id]
        else:
            sy_expr = evotree.get_sympy_expr()
            sym_check(sy_expr)  # sfeh:discuss save bad trees in LUT aswell? Different LUT for bad trees?
            self.lut_sym[tree_id] = sy_expr

        if tree_id in self.lut_parsim:
            parsimony = self.lut_parsim[tree_id]
        else:
            parsimony = eval_parsimony(evotree, self.evolve.complexity_metric, origin_tree=origin_tree)
            self.lut_parsim[tree_id] = parsimony
            if raise_if_useless and parsimony > self.evolve.nodes_max:  # sfeh:open
                raise TreeSizeError(f'Tree too complex: {parsimony} > {self.evolve.nodes_max}')

        if sy_expr in self.lut_fitness:
            fitness = self.lut_fitness[sy_expr]
        else:
            fitness = self.eval_fitness(sy_expr)

            self.lut_fitness[sy_expr] = fitness  # sfeh:discuss: lut update in finalize_tree_get_meta()?

        candidate = Candidate(evotree, fitness=fitness, parsimony=parsimony, tag=tag)
        return candidate

    def evoloop_monitoring_plots(self):
        """
        Create all run-related analysis plots in the root directory
        """
        plot_performance(self.monitor_df, self.rootdir / 'monitoring.png')
        plot_paretofront(self.paretofront, self.rootdir, self.evolve.nodes_max)

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
        printpl('gg',
                f"Created {len(self.pop_genepool)}/{self.pop_max} ({tmp_dict['pop_unique']} unique) in generation {self.gen_id}. "
                f"Trees in LUT: {len(self.lut_fitness)} Generation took {gen_time:4.2f}s")

        printpl('ggg', f'--- Generation {self.gen_id} took: {time.perf_counter() - self.time_genstart:4.2f}. ---')

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
        if self.gens_since_last_pareto > 100:  # .iloc[-1] > 100:  # sfeh discussion
            print('SFEH This condition made your program exit!')
            return True
        else:
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
    pop_parsim = [tree.get_parsim() for tree in popul]
    pop_treelen = [len(candidate_tree.tree) for candidate_tree in popul]
    pop_fitness_best = np.min(pop_fitness)
    pop_unique = len(set([str(x.tree) for x in popul]))  # sfeh:analyze this?

    # sfeh:idea add the amount of actually new trees (compare with the LUT tree_ids)
    result = {'pop_len': len(popul),
              'pop_unique': pop_unique,
              'time': gen_time,
              'gens_since_last_pareto': gens_since_last_pareto,
              'fit_avg': np.average(pop_fitness),
              'fit_var': np.std(pop_fitness),
              'fit_best': pop_fitness_best,
              'fit_quantile_50': np.quantile(pop_fitness, 0.5),
              'fit_quantile_25': np.quantile(pop_fitness, 0.25),
              'fit_quantile_75': np.quantile(pop_fitness, 0.75),
              'parsim_avg': np.average(pop_parsim),
              'parsim_var': np.std(pop_parsim),
              'parsim_best': np.min(pop_treelen),
              'parsim_quantile_50': np.quantile(pop_parsim, 0.5),
              'parsim_quantile_25': np.quantile(pop_parsim, 0.25),
              'parsim_quantile_75': np.quantile(pop_parsim, 0.75)
              }
    return result


def selection_tournament(pop, n=3):
    """
    Survival of the fittest
    Returns the fittest from n random trees of the last population
    """
    tree_list = [np.random.choice(pop) for _ in range(n)]
    fintree: 'Candidate' = min(tree_list, key=lambda tree: tree.get_fitness())
    evotree = fintree.get_evotree()
    evotree = copy.deepcopy(evotree)
    return evotree


# if __name__ == "__main__":
#     df = pd.read_csv(Path(__file__).parent.parent.absolute() / f'benchmarks/mc/gp_files/samples200.csv').astype('float32')
#     evo = Evolution()
#     gp = ExplainableGP(evo, df)
#     pop = gp.gen_create_initial()
#     for _ in range(10):
#         gp.