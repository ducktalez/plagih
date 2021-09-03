"""
Functions, that might be addable in the future:
"""
from plagih.util import *
from plagih.fitness_kernel import *
from plagih.sympy_extras import expr_sympify

from plagih.tree_factory import *
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees

tensorflow.compat.v1.disable_eager_execution()  # sfeh damn what was this line good for?


class ExplainableGP:
    """

    """

    # def __init__(self, conf, rootdir, args=None):

    def __init__(self, conf, rootdir, path_origin, path_data_csv, args):
        """
        sfeh conf? meh...
        """
        self.conf = conf
        self.develop = args.develop  # more testing and stuff during development phase
        self.time_start = time.perf_counter()
        kernel = RegressionKernel(path_data_csv,
                                  conf)  # sfeh relative to rootdir? -> nah -> some absolute path... discussion
        self.kernel = kernel
        tb = TreeBuilder(kernel.obs_names, self.conf, root_xtype=float)
        self.origin = OriginTree(kernel, path_origin=path_origin)
        # self.evolve_loop = EvolutionLoop()
        self.rootdir = rootdir

        self.printpl('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        # self.conf.set_rootdir()  # sfeh better solution?

        print(f'\n'
              f'\tInitializing Plagih UI.\n'
              f'\tName: {BColors.CYAN}{rootdir.name}{BColors.RESET}.\n'
              f'\tLocated in: \n'
              f'\t{rootdir}\n')

        # ==================================

        self.paretofront = []
        self.tb = tb

        self.pop_next = []
        self.pop_base = []  # sfeh maybe better names

        self.lut = {}
        self.time_genstart = time.perf_counter()
        self.gens_since_last_pareto = 0

        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'parsim_avg', 'parsim_var', 'parsim_best',
                                                'gens_since_last_pareto'])
        """
        was Evolution class
        The evolve_dict is converted into a list of the population length.
        The evolve_loop is for the regular evolution, the evolve_random is especially required for the first generation.
        """
        evolve_loop = {
            # Reproduction (10%)
            'Repro': {'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.09, 'custom_params': {}},
            'Rsympy': {'evolve_name': 'reproduce', 'evolve_rate': 0.00, 'custom_params': {'simplify': True}},
            # sfeh 0.03
            'Pareto': {'evolve_name': 'revive paretofront', 'evolve_rate': 0.01, 'custom_params': {}},

            # Mutation (25%)
            'Point': {'evolve_name': 'mutate point', 'evolve_rate': 0.05, 'custom_params': {}},

            'BranchDF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                         'custom_params': {
                             'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8),
                                            'p_full': 1.0}}},
            'BranchDG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
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
            'BranchShrink': {'evolve_name': 'mutate branch', 'evolve_rate': 0.0,
                             'custom_params': {
                                 'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0),
                                                'p_full': 0.5}}},

            'FilterBO': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                         'custom_params': {'filter_mode': 'branch', 'filter_observations': True}},
            'FilterB': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                        'custom_params': {'filter_mode': 'branch', 'filter_observations': False}},
            'FilterP': {'evolve_name': 'filter optimize', 'evolve_rate': 0.0, 'tourn_size': 5,
                        'custom_params': {'filter_mode': 'point', 'filter_observations': True}},

            # Crossover (35%)
            'Xover': {'evolve_name': 'crossover branch', 'evolve_rate': 0.30, 'custom_params': {}},  # sum 0.70

            # Leftovers are automatically filled with random trees

            # Random (25%)
            'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.05,
                      'custom_params': {
                          'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 3, 4, 1), 'p_full': 1.0}}},
            'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.00,
                      'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1),
                                                       'p_full': 0.5}}},
            'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (12, 3, None, 5),
                                                       'p_full': 0.5}}},
            'Rand4': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (12, 3, None, 5),
                                                       'p_full': 1.0}}},  # param 'max' can be None
        }
        self.evolve_loop = self.evolve_safety_update(evolve_loop)

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
        self.evolve_random = self.evolve_safety_update(evolve_random)

        self.evolve_tags = list(self.evolve_loop.keys()) + list(self.evolve_random.keys())
        return

    # class Population:

    def file_pareto(self, path):
        # sfeh where to use? path-> self.conf.rootdir / 'paretofront.yaml',
        yaml_dump(path, self.paretofront)  # sfeh this printing is probably shit

    #     self.pareto_insert_again_simplified(fintree)

    # def pareto_insert_again_simplified(self, fintree):
    #     """
    #     # sfeh:open
    #     """
    #     # tree_sym = copy.deepcopy(evotree)
    #     #
    #     # try:
    #     #     # printez('aaa', 'Trying to simplify for paretofront entry.')  # simplify the fintree and save in paretofront once again
    #     #     tree_sym.evolve_reduce(obs_infos=obs_infos, completely=True)
    #     #     parsimony = tree_sym.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
    #     #     if parsimony < evotree.meta.parsimony:
    #     #         # self.printpl('aa', 'Successfully reduced paretofront fintree!')
    #     #         sym_fitness = self.eval_tf_fitness(tree_sym)  # sfeh actually not required, delete this
    #     #         tree_sym.meta.fitness_train = sym_fitness
    #     #         tree_sym.meta.parsimony = parsimony
    #     #         self.update_pareto_with_tree(tree_sym)
    #     # except Exception as ex:
    #     #     print_warning('www', f'Tree sympification did not work: {ex}', print_type=self.conf.print_type)
    #     #
    #     # else:
    #     #     self.printpl('aaa', 'Pareto entry was already simplified')
    #     pass

    def pareto_plot(self, path, run_name, parsimony_max):
        """
        Write pyplot with paretofront candidates
        """

        run_name = str(run_name).replace('_', '-')  # sfeh asd workaround for latex version

        tuples = [[tree.get_parsimony(), tree.get_fitness()] for tree in self.paretofront]
        xx, yy = np.array(tuples).T

        if len(xx) == 0:
            print_e(f'Plotting empty array is not possible! Data={xx, yy}')
            return

        with plt.rc_context(rc=pyplot_rc_tex):
            fig, ax = plt.subplots()
            right = max(max(xx), parsimony_max) * 1.05  # sfeh check this out 1.05  # if set_right:

            # beyond_lines:  # adding a point to the edges to imply that there are no more values (paretofront-plot)
            xx = np.concatenate([[xx[0]], xx, [right + 1]])
            yy = np.concatenate([[max(yy) + 1], yy, [yy[-1]]])

            ax.step(xx, yy, linestyle='dashed', marker='.', label=f'{run_name}', where='post')
            ax.set(xlabel='complexity', ylabel='regression error',
                   xlim=(0, right),
                   ylim=(0, (max(yy) - min(min(yy), 0)) * 1.05))

            try:
                fig.savefig(path)
                self.printpl('f', f"paretofront (pdf): {self.rootdir / f'paretofront.pdf'}")
            except PermissionError as perm_error:
                print_e(f'Could not save plot: {perm_error}')  # sfeh for everything?

        return

    def pareto_append_clean(self, tree: FinalizedTree):
        """

        """
        self.gens_since_last_pareto = 0
        self.paretofront.append(tree)
        self.paretofront = self.poplist_paretosort(self.paretofront)

    def pareto_random_choice(self):
        try:
            return np.random.choice(self.paretofront)
        except:
            raise  # sfeh

    def pareto_txt(self):
        """
        Save all the paretofront candidates to a file.
        (Quick feedback that requires little overhead)
        """
        return [f'Parsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{tree.meta.expr_raw}' for (parsim, fitness, tree)
                in self.paretofront]

    # def update_pareto_with_tree(self, fintree: FinalizedTree):
    #     """
    #     inserts a fintree into the paretofront front
    #     """
    #     parsim = fintree.get_parsimony()
    #     fit = fintree.get_fitness()
    #
    #     p_simpler = [p for p in self.paretofront if p.get_parsimony() <= parsim]  # all paretofront paretofront that are less complex
    #
    #     if len(p_simpler) == 0:  # all other paretofront paretofront are more complex
    #         self.pareto_insert(fintree, msg=f'new simplest entry')
    #     else:
    #         best = min(p_simpler, key=lambda p: p[1])  # the fittest of the less complex ones
    #         if fit < best.get_fitness():
    #             self.kernel.fitness_compare(fit, best.get_fitness())  # if true, at least one insertion
    #             self.pareto_insert(fintree, msg=f'old fitness: {best[1]}')
    #
    #     # self.pareto_clean_sort()  # sfeh check if required
    #     return

    def poplist_paretosort(self, pop_list):
        """
        sfeh check this!! op_next mostly has 2 pareto entries??
        """
        # sfeh:open:kernel sorting with x.get_fitness OnLY when fitness has "<" relation. (otherwise: -x.get_fitness)
        fitness_sign = self.kernel.fitness_sign
        pop_list = sorted(pop_list, key=lambda x: (x.get_parsimony(), fitness_sign * x.get_fitness()))

        try:
            best = pop_list[0]
        except Exception as ex:
            raise Exception(f'The list is empty, i guess: {pop_list}. {ex}')

        best_par = best.get_parsimony()
        best_fit = best.get_fitness()
        pop_pareto = [best]

        for tree in pop_list:
            parsim = tree.get_parsimony()
            if parsim == best_par:
                continue  # sfeh:discuss does sorted keep the order?
            else:
                fitness = tree.get_fitness()
                if self.kernel.fitness_compare(fitness, best_fit):
                    pop_pareto.append(tree)
                    best_par = parsim
                    best_fit = fitness

        # print(f'The paretofront-efficient candidates of the population are: {pop_pareto}')

        return pop_pareto

    def pareto_from_population(self):
        """
        sfeh:debug, might be faulty
        sfeh:discuss pareto-efficient, but different pareto entries?
        """
        pop_parcandidates = self.poplist_paretosort(self.pop_next)  # pareto-candidates in the pop, renamed to be clear

        if len(self.paretofront) == 0:
            firstpareto = pop_parcandidates[0]
            self.printpl('a',
                         f'Starting a new paretofront with parsimony: {firstpareto.get_parsimony()} fitness: {firstpareto.get_fitness():6.4f}')
            self.paretofront.append(firstpareto)

        for fintree in pop_parcandidates:
            success = False
            fit = fintree.get_fitness()
            par = fintree.get_parsimony()

            if all([par < p.get_parsimony() for p in self.paretofront]):
                self.printpl('a', f'Paretofront: New simplest entry. parsimony: {par} fitness: {fit:6.4f}')
                success = True
                # optional: print now deprecated entries in the paretofront

            if all([self.kernel.fitness_compare(fit, p.get_fitness()) for p in self.paretofront]):
                self.printpl('a', f'Paretofront: New best-fitness entry. parsimony: {par} fitness: {fit:6.4f}')
                success = True
                # optional: print now deprecated entries in the paretofront

            for p in self.paretofront:
                # better fitness at a comparatively good
                if self.kernel.fitness_compare(fit, p.get_fitness()) and par <= p.get_parsimony():
                    # self.paretofront.remove(p)  # sfeh: remove in the other cases aswell or not at all
                    success = True

            if success:
                self.printpl('a', f'Paretofront: New entry. parsimony: {par} fitness: {fit:6.4f}')
                self.pareto_append_clean(fintree)
                # self.printpl('a', f"New entry found! {BColors.RESET}{fintree.get_parsimony()}, {fintree.get_fitness()}:{BColors.RESET} {fintree.meta.expr_raw}")

        self.paretofront = self.poplist_paretosort(self.paretofront)

        return

    # # V0 (never used)
    # pop_iter = iter(pop_sorted)
    # pareto_iter = iter(self.paretofront)
    #
    # p = next(pareto_iter)
    # p_par = p.get_parsimony()
    # t = next(pop_iter)
    #
    # while True:
    #
    #     # lets try the next (or first) fintree out!
    #     while t.get_parsimony() < p_par:
    #         print('Found a NEW, simpler entry!')
    #         self.paretofront.pop_append_evotree(t)  # sfeh (every entry in this list is already paretoefficient)
    #         t = next(pop_iter)
    #
    #     t_fit = t.get_fitness()
    #     while not cmp(t_fit, p.get_fitness()):
    #         print(f'removing the deprecated entries: Fitness: {p.get_fitness()}')
    #         self.paretofront.remove(p)
    #         p = next(pareto_iter)
    #
    #     while cmp(t.get_fitness(), p.get_fitness()):
    #         t = next(pop_iter)
    #
    #     p_fit = p.get_fitness()
    #     t_fit = t.get_fitness()
    #
    #     while cmp(t_fit, p_fit) and t_par < p_par:
    #         # find a potential entry with at least the parsimony
    #         print(f'SHEESH! We kicked out another paretofront entry! old: {p_fit}, new fitness: {t_fit}.')
    #         self.paretofront.remove(p)
    #         self.paretofront.pop_append_evotree(t)
    #         p = next(pareto_iter)
    #         p_par = p.get_parsimony()
    #         p_fit = p.get_fitness()
    #
    #     while t_par < p_par:
    #         t = next(pop_iter)
    #         t_par = t.get_parsimony()
    #
    #     while p_par > t_par:
    #         p = next(pareto_iter)
    #         p_par = p.get_parsimony()
    #         try:
    #             t = next(pop_iter)
    #             t_par = t.get_parsimony()
    #         except StopIteration:
    #             print('ASD Pareto feddich :3!!')  # sfeh debug paretofront wurde erstellt
    #             return
    #
    # # sfeh sort everything
    #
    # # pop_list = sorted(pop_list, key=lambda x: (x.get_parsimony(), x.get_fitness()))
    # # pop_iter = iter(pop_list)
    # # fintree = next(pop_iter)
    # # pop_sorted = [fintree]
    # # best_par = fintree.get_parsimony()
    # # best_fit = fintree.get_fitness()
    # #
    # # while True:
    # #     try:
    # #         fintree = next(pop_iter)
    # #         parsim = fintree.get_parsimony()
    # #         if parsim == best_par:
    # #             continue
    # #         else:
    # #             fitness = fintree.get_fitness()
    # #             if self.kernel.fitness_compare(fitness, best_fit):
    # #                 pop_sorted.pop_append_evotree(fintree)
    # #                 best_par = parsim
    # #                 best_fit = fitness
    # #     except StopIteration:
    # #         break

    # def file_pareto_listcode(self):
    #     """
    #     save python code for Industrial Benchmark runs
    #     delete sometime
    #     """
    #
    #     # pycode_agent = self.kernel.pycode_wrap_result(self.env_vars.action.minmax).format('action')
    #
    #     pygents_list = []
    #
    #     for (parsim, fitness_train, fintree) in self.paretofront:
    #         # agent_name = f'{self.conf.name}_{parsim:.0f}'
    #         agent_name = f'{self.conf.name}_{self.env_vars.action.name}_{parsim:.0f}'
    #         agent_as_python = fintree.eval_pycode()
    #         pygents_list.pop_append_evotree([parsim, float(fitness_train), agent_name, agent_as_python])
    #
    #     yaml_dump(self.rootdir / 'pycode_list.yaml', pygents_list, print_type=self.print_type)
    #     path = path_make_dir(self.rootdir / 'pycode_list.yaml')
    #     with Path.open(path, 'w') as file:
    #         _ = yaml.dump(pygents_list, file)  # , default_flow_style=False, sort_keys=False)
    #         printez('ff', f'IB pycode-list: {path.as_posix()}', print_type=self.print_type)  # sfeh always the same print structure... just pass the path?
    #
    #     return
    #

    def analyze_pareto(self, cpu_cores=16):  # sfeh 16 cores? nope
        """
        sfeh:open
        Writing all analysis files after evaluating the paretofront.
        (Currently strongly customized by sfeh for the mountaincar and industrial benchmark)
        """
        # self.printpl('i', f'Analysing the paretofront candidates of the run.')
        #
        # dir_benchmarks = Path(__file__).parent.parent.absolute() / 'benchmarks/'
        # path_hist = path_make_dir(self.rootdir / 'histograms/')
        # pareto_agents = {}
        #
        # for (parsim, fitness_train, fintree) in self.paretofront:
        #     histograms_path = self.plot_agent_histogram(parsim, fintree, path_hist)  # sfeh
        #
        #     forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest = self.file_pareto_latex(parsim, fintree)
        #
        #     pareto_agents[parsim] = {'parsim': parsim,
        #                              'fitness_train': fitness_train,
        #                              'fintree': fintree,
        #                              'forest_tree_full': forest_tree_full, 'forest_tree_tight': forest_tree_tight, 'tex_expr_raw': tex_expr_raw, 'tex_expr_forest': tex_expr_forest,
        #                              'histogram': histograms_path,
        #                              }
        #
        # tex_include_pdf = lambda x: f"\\includegraphics{{{str(x).replace('.pdf', '')}}}"
        # tex_tabuline = lambda x: f"{' & '.join(x)}\\tabularnewline\n"
        # tex_stacklist = lambda x: "\\shortstack[l]{{{}}}".format('\\\\'.join([str(xx) for xx in x]))
        #
        # if 'MTC' in self.conf.name:
        #     """
        #     complexity, regr. error, real evaluation, decision plot, spiral plot, diff-plot
        #     """
        #     # sfeh i guess not necessary anymore (?)
        #     # self.file_pareto_pycode()
        #     sarsa_agent_steps = 200 if 'MTC200' in self.conf.name else 75 if 'MTC75' in self.conf.name else 'NO_MC_AGENT'
        #     sarsa_agent = pickle_load(dir_benchmarks / f'mc/agents/sarsa_agent_{sarsa_agent_steps}.p')
        #
        #     agent_performance = auto_evaluate_run_end(self.rootdir, sarsa_agent, n=100)
        #     # df_mtc = pd.DataFrame(columns=['complexity', 'regr. error', 'avg. reward', 'fails', 'expression'])
        #     tex_lines = []
        #
        #     for pp, x in agent_performance.items():
        #         pp, fitness_train, avg_reward, fails, path_mcmeshplot, path_mcmeshplot_diff = x
        #
        #         tex_lines.pop_append_evotree([f"{int(pp)}",
        #                           f"{pareto_agents[pp]['fitness_train']:.2f}",
        #                           f"{avg_reward:0.1f}",
        #                           f"{pareto_agents[pp]['tex_expr_raw']}",
        #                           f"{tex_include_pdf(f'sfehs_eval/{pp}.pdf')}",  # path_mcmeshplot
        #                           # tex_include_pdf(f'sfehs_eval/diff-{pp}.pdf'),  # path_mcmeshplot_diff),
        #                           tex_include_pdf(f'sfehs_eval/space-{pp}.pdf'),
        #                           # tex_include_pdf(f'histograms/acthist_{pp}'),
        #                           f"{pareto_agents[pp]['forest_tree_full']}",  # forest_tree_full, forest_tree_tight
        #                           f"{pareto_agents[pp]['forest_tree_tight']}",
        #                           f'{fails}'])
        #
        #     paste = ''.join([tex_tabuline(x[:4]) for x in tex_lines])
        #     paste = "\\begin{longtable}[c]{>{\\LTleft}p{5mm}>{\\LTleft}p{6mm}>{\\LTleft}p{10mm}>{\\LTleft}p{102mm}}\n\\hline\n" \
        #             "dist & error & reward  & expression \\tabularnewline \\hline\n" \
        #             f"{paste}" \
        #             "\\hline\n\\end{longtable}\n"
        #     file_dump(self.rootdir / f'analysis_input.tex', paste, print_type=self.print_type)
        #     file_dump(self.rootdir / f'analysis_overview.tex', latex_treeviz_full_document(paste), print_type=self.print_type)
        #
        #     paste_full = ''.join([tex_tabuline(x[:]) for x in tex_lines])
        #     paste_full = f"{str(self.conf.name).replace('_', '-')}  {tex_include_pdf('monitoring.png')}  {tex_include_pdf('sfehs_eval/evaled_overview.pdf')}\n\n" \
        #                  "\\begin{tabular}{lllllllllllll}\n \\hline\n" \
        #                  "dist & error & reward & parsimony & expression \\tabularnewline \\hline\n" \
        #                  f"{paste_full}" \
        #                  "\\hline\n\\end{tabular}\n\n"
        #     file_dump(self.rootdir / f'analysis_overview_plus.tex', latex_treeviz_full_document(paste_full), print_type=self.print_type)
        #
        # elif 'IB' in self.conf.name:
        #
        #     self.file_pareto_listcode()
        #
        #     if self.conf.name[-2:] == '_0':
        #         """
        #         - complexity_sum, [complexities], regression_sum, [regression_errors], real_sum, [formulas]
        #         - plot with paretofront candidates
        #         """
        #         # "\\multicolumn{2}{c}{complexity} & \\multicolumn{2}{c}{regression error} & \\multicolumn{4}{c}{IB reward} & combinations & expression \\tabularnewline\n" \
        #         tex_line_input, tex_line_overview = '', ''
        #
        #         res_all = combined_lists(self.rootdir.parent, 40, 40, local_yamls=True, cpu_cores=cpu_cores)  # sfeh use self.conf.mp_cores
        #
        #         xx = [x['parsim_sum'] for x in res_all]
        #         y_all = [y['experiment'] for y in res_all]
        #         y_safe = [y['experiment_safe'] for y in res_all]
        #         y_all_r50 = [y['experiment_r50'] for y in res_all]
        #         y_safe_r50 = [y['experiment_safe_r50'] for y in res_all]
        #         cnt = [y['cnt'] for y in res_all]
        #
        #         for y in res_all:
        #             parsims = y['parsims']
        #
        #             input_agentex = lambda run_ii, prepth: f"\\input{{{prepth}{self.rootdir.parent.name}_{run_ii}/visualisation/{int(parsims[run_ii]):02d}_input.tex}}"  # _forest
        #
        #             tex_line_overview += tex_tabuline([f"{int(y['parsim_sum'])}",
        #                                                f"{y['regress_sum']:0.3f}",
        #                                                f"{y['experiment']:0.0f}",
        #                                                tex_stacklist([f'{int(x)}' for x in y['parsims']]),
        #                                                tex_stacklist([input_agentex(x, '') for x in [0, 1, 2]])])
        #
        #             tex_line_input += tex_tabuline([f"{int(y['parsim_sum'])}",
        #                                             f"{y['regress_sum']:0.3f}",
        #                                             f"{y['experiment']:0.0f}",
        #                                             tex_stacklist([f'{int(x)}' for x in y['parsims']]),
        #                                             tex_stacklist([input_agentex(x, f'../benchmarks/{self.rootdir.parent.parent.name}/{self.rootdir.parent.name}/') for x in [0, 1, 2]])])
        #
        #             tex_line_input += tex_tabuline([f"{int(y['parsim_sum'])}",
        #                                             f"{y['regress_sum']:0.3f}",
        #                                             f"{y['experiment']:0.0f}",
        #                                             tex_stacklist([f'{int(x)}' for x in y['parsims']]),
        #                                             tex_stacklist([input_agentex(x, f'../benchmarks/{self.rootdir.parent.parent.name}/{self.rootdir.parent.name}/') for x in [0, 1, 2]])])
        #
        #         combined_overview = "\\begin{tabular}{llllllllll}\n\\hline \n" \
        #                             f"{tex_tabuline(['dist', 'error', 'reward', 'dist', 'Agent code'])} \\hline\n" \
        #                             f"{tex_line_overview}" \
        #                             f"\\hline\n\\end{{tabular}}\n\n"
        #
        #         combined_input = "\\begin{longtable}[c]{>{\\centering}p{10mm}>{\\centering}p{10mm}>{\\centering}p{12mm}>{\\centering}p{12mm}>{\\centering}p{90mm}} \\hline\n" \
        #                          f"{tex_tabuline(['dist', 'error', 'reward', 'parsimony', 'expressions'])}" \
        #                          f"{tex_line_input}" \
        #                          "\\hline\n\\end{longtable}\n"
        #
        #         combined_fulltrees = "\\begin{longtable}[c]{>{\\centering}p{10mm}>{\\centering}p{10mm}>{\\centering}p{12mm}>{\\centering}p{12mm}>{\\centering}p{90mm}} \\hline\n" \
        #                              f"{tex_tabuline(['dist', 'error', 'reward', 'parsimony', 'expressions'])}" \
        #                              f"{tex_line_input}" \
        #                              "\\hline\n\\end{longtable}\n"
        #
        #         combined_overview = latex_treeviz_full_document(combined_overview)
        #         file_dump(self.rootdir.parent / 'combined_overview.tex', combined_overview)
        #         file_dump(self.rootdir.parent / 'combined_input.tex', combined_input)
        #
        #         with plt.rc_context(rc=pyplot_rc_tex):
        #             fig, ax = plt.subplots()
        #             ax.set(xlabel='Pareto complexity sum', ylabel='reward [x1000]', ylim=funny_limits)
        #             ax.plot(xx, y_all, label='average', marker='.', color='r')
        #             ax.plot(xx, y_safe, marker='None', color='r', linestyle='dotted')  # , label='low risk'
        #             ax.plot(xx, y_all_r50, label='randomized start', marker='.', color='b')
        #             ax.plot(xx, y_safe_r50, marker='None', color='b', linestyle='dotted')  # , label='low risk (randomized)'
        #             ax.legend(loc='lower right')
        #             plt.yticks(IB_YICKS[0], IB_YICKS[1])
        #
        #             # drawing the regression error, but plots seem to be too overloaded
        #             # axx = ax.twinx()
        #             # axx.plot(xx, cnt, color='tab:gray', label='regression error', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
        #             # axx.tick_params(axis='y', labelcolor='tab:gray')
        #             # # axx.plot(xx, y['regression_sum'])
        #             #
        #             # ax2 = ax.twinx()
        #             # ax2.plot(xx, cnt, color='tab:gray', label='combos', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
        #             # # ax2.set(ylabel='possible combinations', color='tab:gray')
        #             # ax2.tick_params(axis='y', labelcolor='tab:gray')
        #             # ax2.legend(loc='lower left')
        #
        #             fig.savefig(self.rootdir.parent / f'regression_all.pdf')
        #             plt.close('all')
        #
        # else:
        #     raise Exception(f'This should actually never happen right now. name: {self.conf.name}')

        return

    #
    # def file_pareto_latex(self, parsim, fintree):
    #     """
    #     Generates latex-file with the computational fintree structure of all paretofront agents
    #     - build fintree from expression
    #     - fill fintree meta-data, just in case we want to visualise anything of it
    #     - create latex-forest representation
    #     """
    #
    #     """
    #     whole procedure from fintree to forest core
    #     tight_viz:
    #         0: display every node
    #         1: clever tight-visualisation where possible
    #         2: one single mathematical expression
    #     """
    #
    #     fintree.set_fix_nodes(self.origin)
    #     fintree = fintree.get_oldtree()
    #
    #     pl_forest = lambda x: f'\\plforest{{{x}}}\n'
    #
    #     forest_tree_full = None  # pl_forest(latex_brackettree(fintree))
    #     forest_tree_tight = None  # pl_forest(latex_brackettree_tight(latex_tree_semitight(fintree)))
    #     # sfeh workaround delete this
    #     tex_expr_raw = f'${fintree.export_visualization_latex()}$'  # sfeh dollars
    #     tex_expr_forest = pl_forest(f'[{tex_expr_raw}]')
    #
    #     path_subfolder_tex = path_make_dir(self.rootdir / 'visualisation')  # sfeh running this in every fintree seems unneccesary
    #
    #     """
    #     The following lines delete this
    #     """
    #     # sfeh
    #     file_dump(path_subfolder_tex / f'full_{parsim:02d}.tex', forest_tree_full, verbose='ff', print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'full_{parsim:02d}_tight.tex', forest_tree_tight, verbose='ff', print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_input.tex', tex_expr_raw, print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_input_forest.tex', tex_expr_forest, verbose='ff', print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_doc.tex', latex_treeviz_full_document(forest_tree_full), verbose='ff', print_type=self.print_type)  # delete this
    #
    #     return forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest

    # def file_pareto_pycode(self):
    #     """
    #     this auto-generation of real (executable) python files
    #     is strongly customized for my experiments with Mountaincar and industrial benchmark
    #
    #     very useful: textwrap.indent
    #     example: complete_function = textwrap.indent(f"def decide(self, input):\n"
    #                                         f"{function_body}\n", '    ')  # aka tab (\t)
    #     """
    #     pass
    #     # py_return = self.kernel.pycode_wrap_result(self.env_vars.action.minmax).format('action')
    #
    #     # complete_function = f"    def decide(self, input):\n" \
    #     #     f"        cartPos, cartVel = input\n" \
    #     #     f"        action = {{}}\n" \
    #     #     f"        return {py_return}\n"
    #     #
    #     # all_agents = []
    #     # all_agent_names = []
    #     # all_more_info = []
    #     #
    #     # for (parsim, fitness_train, fintree) in self.paretofront:
    #     #     agent_name = f'{self.conf.name}_{parsim:.0f}'
    #     #
    #     #     agent_as_python = fintree.eval_pycode()
    #     #     all_agents.pop_append_evotree(f"class {agent_name}:\n{complete_function.format(agent_as_python)}")
    #     #     all_agent_names.pop_append_evotree(agent_name)
    #     #     all_more_info.pop_append_evotree(f"('{agent_name}', {agent_name}(), {parsim}, {fitness_train})")
    #     #
    #     # all_agents = '\n\n'.join(all_agents)
    #     # agent_tuples = ', '.join([f"('{x}', {x}())" for x in all_agent_names])
    #     # all_more_info = ', '.join(all_more_info)
    #     #
    #     # """
    #     # dude sfeh
    #     # sfeh for fully executable code
    #     # bad code
    #     # """
    #     # if 'MTC75' in self.conf.name:
    #     #     sarsa_agent = 75
    #     # elif 'MTC200' in self.conf.name:
    #     #     sarsa_agent = 200
    #     # else:
    #     #     raise
    #
    #     # pyc_complete = f"import math; import numpy as np\n" \
    #     #     "import sys\n" \
    #     #     "from pathlib import Path\n" \
    #     #     "sys.path.pop_append_evotree(str(Path(Path.cwd() / '../../../'))\n" \
    #     #     "from benchmarks.MC.agents.quick_eval import *\n" \
    #     #     "from pathlib import Path\n" \
    #     #     "folder = Path.cwd() / 'custom_files'\n" \
    #     #     "from benchmarks.MC.agents.mtc_agent_sarsa import * \n" \
    #     #     f"with Path.open(Path(Path.cwd() / '../../../benchmarks/MC/agents/sarsa_agent_{sarsa_agent}.p'), 'rb') as file:\n" \
    #     #     "\tsarsa_agent = pickle.load(file)\n" \
    #     #     "\n" \
    #     #     f"{all_agents}\n" \
    #     #     f"all_agents_more = [{all_more_info}]\n" \
    #     #     f"agent_tuples = [{agent_tuples}]\n" \
    #     #     f"\n\n" \
    #     #     "if __name__ == '__main__':\n" \
    #     #     "\tprint('executing!')\n" \
    #     #     "\teval_agent_list(agent_tuples, folder=folder, goal_agent=sarsa_agent)\n"
    #
    #     # pth = path_make_dir(self.rootdir / self.dir_pycode / f"agents.py")
    #     # with Path.open(pth, 'w') as file:
    #     #     file.write(pyc_complete)
    #     #     self.printpl('ff', f'Pycode: {pth.as_posix()}')
    #
    #     return

    def evolve_safety_update(self, evolve_dict):
        """
        Updates tournament size and evolve rates
    
        Example entry of the list could be:
        {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
        'custom_params': {'build_spec': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
        """

        for tag, evolve_spec in evolve_dict.items():
            evolve_dict[tag]['tourn_size'] = evolve_spec.get('tourn_size', self.conf.tourn_size)
            evolve_dict[tag]['evolve_num'] = int(evolve_spec.get('evolve_rate') * self.conf.pop_max)
            evolve_spec['custom_params'] = evolve_spec.get('custom_params', {})
        return evolve_dict

    # evolve_loop = self.evolve_list
    # self.printpl('i', 'Using evolve rates from config')

    # class Evolution:
    #     # sfeh rate not in here
    #     # def __init__(self, id=None, evolution=None, rate=None, params=None, custom_params=None):
    #     #     self.id = id
    #     #     self.evolution = evolution
    #     #     self.params = params or {}
    #     #     self.rate = rate
    #     #     self.custom_params = custom_params

    def printpl(self, message_type, message_str):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        message_type options can be found in config
        """
        printez(message_type, message_str, print_type=self.conf.print_type)
        return

    def gen_create_initial(self):
        """
        was: gen_create_initial
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta fintree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        if self.origin.exists():
            self.pop_next.append(self.origin.fintree)
            # self.pop_append_evotree(self.origin.fintree)  # sfeh why not :P
            # self.pareto.pareto_insert(self.origin)  # the origin fintree is the only candidate (automatically added, i hope)
        else:
            total_rate = sum([x['evolve_rate'] for x in self.evolve_random.values()])

            for tag, evolve_specs in self.evolve_random.items():
                evolve_num = int(self.conf.pop_max * (evolve_specs['evolve_rate'] / total_rate))
                custom_params = evolve_specs.get('custom_params')

                for nn in range(evolve_num):
                    if self.origin.origin_is_fix:
                        evotree = self.tb.pop_random(custom_params, origin=self.origin)
                    else:
                        evotree = self.tb.pop_random(custom_params)
                    self.pop_append_evotree(evotree, tag)
        return

    def gen_next_population(self):
        """
        Creates all new Generations by applying the evolutions in the evolve-loop.

        brainstorm:
        - the fintree can be reproduced (selection), random/new, olymp-reproduction
        - (1 fintree) mutations can affect a point, branch, terminal nodes
        - (2 trees) can make a crossover
        """
        # All gp creators: name, function, num of trees from tournament selection

        for tag, evolve_specs in self.evolve_loop.items():  # all selected gp mutations

            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            custom_params = evolve_specs.get('custom_params')

            self.printpl('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')
            if evolve_name == 'reproduce':
                """
                
                """
                for nn in range(evolve_num):
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    if custom_params.get('simplify'):  # sfeh==>debug
                        try:
                            evotree.evolve_reduce(completely=False)
                            # fintree.meta.last_evolution = tag
                        except Exception as ex:
                            print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.conf.print_type)
                            # raise ex  # sfeh debug

                    self.pop_append_evotree(evotree, tag)  # pop_append_evotree anyways... it was worth a try :P

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_point(evotree)
                    # fintree.meta.last_evolution = tag
                    self.pop_append_evotree(evotree, tag)

            elif evolve_name == 'mutate branch':
                """
                branch_nodes
                """
                for nn in range(evolve_num):
                    _, size_mode, mean_min_max_var, p_full = helper_evolve_params_branch(custom_params,
                                                                                         tree_depth_max=self.conf.tree_depth_max,
                                                                                         parsimony_max=self.conf.parsimony_max)
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    build_size = choose_build_size(size_mode, mean_min_max_var, tree=evotree, force='branch')  # sfeh:test options, depth, in this case

                    if size_mode == 'branch_depth':
                        evotree = self.tb.evolve_mutate_branch_depth(evotree, build_size, p_full=p_full)

                    elif size_mode == 'branch_nodes':
                        evotree = self.tb.evolve_mutate_branch_nodes(evotree, build_size, p_full=p_full)
                    else:
                        raise

                    self.pop_append_evotree(evotree, tag)

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
                    atree = self.selection_tournament(tourn_size=tourn_size)
                    btree = self.selection_tournament(tourn_size=tourn_size)

                    # 1. two parents
                    # 2. search nodes for left and right that can be exchanged. convert_needed
                    try:
                        atree, btree = self.tb.evolve_crossover(atree, btree)
                        # fintree = self.tb.finalize(etree)  # ==>state
                        self.pop_append_evotree(atree, tag=tag)
                        self.pop_append_evotree(btree, tag=tag)
                    except Exception as ex:
                        pass

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    evotree = self.selection_tournament(tourn_size=tourn_size)
                    evotree = self.tb.evolve_mutate_filter_random(evotree, custom_params)
                    self.pop_append_evotree(evotree, tag)

            elif evolve_name == 'revive paretofront':

                for nn in range(evolve_num):
                    fintree = self.pareto_random_choice()
                    self.pop_next.append(fintree)

            elif evolve_name == 'random trees':

                if self.origin.origin_is_fix:
                    for nn in range(evolve_num):
                        evotree = self.tb.pop_random(custom_params, origin=self.origin)
                        self.pop_append_evotree(evotree, tag=tag)
                else:
                    for nn in range(evolve_num):
                        evotree = self.tb.pop_random(custom_params)
                        self.pop_append_evotree(evotree, tag=tag)
            else:
                print_e(f"Evolution not known: '{evolve_name}'")

        missing_trees = self.conf.pop_max - len(self.pop_next)
        if missing_trees > 0:
            if missing_trees > 0.05 * self.conf.pop_max:
                self.printpl('ii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            else:
                self.printpl('iii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')

            while len(self.pop_next) < self.conf.pop_max:
                return  # sfeh aka create trees here if desired

        # sfeh automatically fill with random trees (check this at the initiation)
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        return

    # # sfeh open
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
    #             path = self.rootdir / f'monitoring_evolutions.pdf'
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
        - fittest fintree
        - average fitness_train
        - average fintree parsimony
        """
        popul = self.pop_next

        if len(popul) == 0:
            raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

        pop_fitness = [tree.get_fitness() for tree in popul]

        # tmp_evol_performance = dict.fromkeys(self.monitor_evol.keys(), pd.DataFrame(columns=['fitness_train', 'parsimony', 'lentree']))
        # for fintree in popul:
        #     last_evol = fintree.meta.last_evolution
        #     if last_evol in self.evolve_tags:
        #         row = {'fitness_train': fintree.meta.fitness_train,
        #                'parsimony': fintree.meta.parsimony,
        #                'lentree': len(fintree)}
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
        #             self.monitor_evol[last_evol].pop_append_evotree(row, ignore_index=True)  #
        #             # sfeh fitness_train - last fitness_train?
        #         except Exception as ex:
        #             print_e(f'Could not save evol_performance analysis. {ex}')
        #     else:
        #         if last_evol != 'origin' and last_evol != 'Rand3o':
        #             print_warning('w', f'delete_this, sfeh, okay when the following is origin: {last_evol}')

        pop_parsim = [tree.get_parsimony() for tree in popul]
        pop_treelen = [len(fintree.tree) for fintree in popul]

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
                                                 'fit_var': np.std(pop_fitness),
                                                 'fit_best': pop_fitness_best,
                                                 'parsim_avg': np.average(pop_parsim),
                                                 'parsim_var': np.std(pop_parsim),
                                                 'parsim_best': np.min(pop_treelen),
                                                 'time': gen_time,
                                                 'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh version1 delete this shit

        self.printpl('gg',
                     f'Created {len(popul)}/{self.conf.pop_max} ({unique_tree_count} unique) in generation {self.conf.gen_id}. Gen took {gen_time:4.2f}s')
        return

    def selection_tournament(self, tourn_size=3):
        """

        """
        tree_list = [np.random.choice(self.pop_base) for _ in range(tourn_size)]
        fintree: 'FinalizedTree' = self.kernel.get_fitness_extreme_function(tree_list,
                                                                            key=lambda tree: tree.get_fitness())
        evotree = fintree.get_evotree()
        return copy.deepcopy(evotree)  # sfeh deepcopy not required if it is copied later

    def pop_append_evotree(self, evotree: Node, tag):
        """
        was: def pop_append(self, fintree: Node):
            # sfeh this check might be important...
        Safely append a fintree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the fintree is refurbished.
        sfeh: if trees are 100% safely created, tree_check_deep() must not be used. Useful when trying out new gp-operators.
        - Enrich the raw fintree for the next generation
        - check if the fintree is actually valid
        ->
        """
        evotree = self.tb.evolve_prune(evotree)  # sfeh:performance runtime-wise, do this somewhere else
        try:
            meta = self.lut[hash(evotree)]
        except KeyError as ex:
            try:
                meta = self.finalize_tree_get_meta(evotree)
            except ValueError as ex:
                print_warning('wwww', f'Could not append fintree to population because: {ex}',
                              print_type=self.conf.print_type)
                return  # sfeh:print
            except Exception as ex:
                print_warning('www', f'Could not append fintree to population because: {ex}\n'
                                     f'=>fintree: {evotree}', print_type=self.conf.print_type)
                # print_warning('w', f'fintree failed the quick check. last-mod: {self.meta.last_evolution}. Reason:\n{ex}', print_type=self.print_type)
                return

        fintree = FinalizedTree(evotree, meta)
        fintree.append_tag(tag)
        self.pop_next.append(fintree)

    def plot_gen_performance(self, path_monitoring: Path):
        """
        All monitoring infos
        sfeh den shit in Funktionen aufteilen
        """
        with plt.rc_context(rc={'axes.grid': True}):
            fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]},
                                    sharex='all')  # , figsize=(9, 9)
            plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
            xx = list(self.monitor_df.index)

            axs0 = axs[0]
            axs0.plot(self.monitor_df['fit_avg'], marker='', label='Regression error (average)')
            # sfeh:improvement not just the stderr on both sides...
            try:
                avg = self.monitor_df['fit_avg']
                std = self.monitor_df['fit_var']
                axs0.fill_between(xx, avg - std, avg + std,
                                  alpha=0.2)  # axs0.set_title('Regression Error (average)')  # sfeh not stderr... upper/lower bound?
            except Exception as ex:
                raise Exception(f'Delete this. were there any problems? {ex}')
            # sfeh: the best candidate is the best one in the current population. discussion: best overall?
            axs0.step(x=xx, y=self.monitor_df['fit_best'], linestyle='dashed', marker='', where='post', color='g',
                      label='Best candidate')  # , label=ax_label
            axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

            axs0_twin = axs0.twinx()
            axs0_twin.plot(xx, self.monitor_df['gens_since_last_pareto'], color='tab:gray',
                           label='Generations since last paretofront entry', linestyle='dashed',
                           marker='')  # linestyle='None'
            axs0_twin.tick_params(axis='y', labelcolor='tab:gray')
            try:
                axs0_twin.set_ylim(ymin=0, ymax=max(self.monitor_df['gens_since_last_pareto'].max() or 1, 50))
            except Exception as ex:
                try:
                    print_e(f'damn setting ylim not working sfeh :s {ex}')
                    axs0_twin.set_ylim(ymin=0,
                                       ymax=max(self.monitor_df['gens_since_last_pareto'].notnull().max() or 1, 50))
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
                p_var = self.monitor_df['parsim_var']
                axs1.fill_between(xx, p_avg - p_var, p_avg + p_var, alpha=0.2)  # axs1.set_title('TED (average)')
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
            self.printpl('f', f"monitoring: {path_monitoring.as_posix()}")
            plt.close('all')

    def backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the paretofront front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """
        # sfeh:discuss: saving the yaml is not required
        path_backup_yaml = path_make_dir(self.rootdir / 'backup/config.yaml')
        yaml_dump(path_backup_yaml, self.conf, print_type=self.conf.print_type)

        # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
        run_backup_data = {}, self.conf.gen_id, self.pop_base, self.paretofront, self.monitor_df  # sfeh use this later, help_dict
        path_backup = path_make_dir(self.rootdir / 'backup/backup.pkl')
        pickle_dump(path_backup, run_backup_data)
        # sfeh:debug
        return

    def backup_load(self, argpath_backup):
        """
        If a backup-file is found...
        Loading the state of the run from the pickle file
        """
        path_backup = argpath_backup or self.rootdir / 'backup/backup.pkl'  # sfeh file-load/save is the same...

        if Path.is_file(path_backup):
            self.printpl('g', f'Loading data from backup-file {path_backup}')
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)
            except NotImplementedError as ex:
                raise Exception(f'NotImplementedError: {ex}')
            except EOFError as ex:
                raise Exception(f'EOFError: \n{ex}')

            help_dict, self.conf.gen_id, self.pop_base, self.paretofront, self.monitor_df = run_data  # sfeh use a helping dictt a_helping_dict is used for a useable sldifjsdfsdfg , a_helping_dict
            self.printpl('g', f'Successfully loaded backup file. Generation: {self.conf.gen_id}')

            # except Exception as ex:
            #     raise Exception(f'Even though a backup exists for this run, it could not be loaded, because of\n{ex}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}.')
        return

    def custom_exit_condition(self):
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

    def finalize_tree_get_meta(self, evotree):
        """
        fintree->fintree!
        ===
        # try:
        #     self.check_all()
        # attention: regular hashes may change between python runs. do not save anything on their hash values <.<

        ===
        Very fast eval-version that only computes fitness_train of the train data.
        tree_eval_complete gives more options
        Evaluating the fitness_train of a fintree.
        - extract the expression the fintree is holding
        - sympify the expression
        - (if sympify fails, evaluating does not make sense! Check sympify errors)
        - (sfeh: if sympify fails because of inf or zoo, tf could maybe still work due to save-tf-division)

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
            print_warning('wwww', f'Exception while evaluating: {ex}, fintree: {evotree}.',
                          print_type=self.conf.print_type)
            raise ValueError(ex)

        treeobs = evotree.get_observation_list()
        try:
            fitness = self.kernel.eval_tf(expr_sym, treeobs, only_fitness=True)
            # if fitness != fitness or fitness == float('inf'):
            #     raise Exception(f"fitness_train is: '{fitness}'")  # happens, eg when values are soo wrong that it leaves the float-range
        except Exception as ex:
            raise Exception(f'eval-ex: {ex}')

        meta = TreeMeta(fitness, parsimony, expr_raw, expr_sym)
        self.lut[hash(evotree)] = meta
        return meta

    def plagih_gp_run(self, gen_additionally):
        """
        regular plagih run
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.conf.gen_id + gen_additionally)
            self.printpl('i',
                         f'Adding {gen_additionally} more generations in gen {self.conf.gen_id}, increasing gen_max from {printdummy} to {self.conf.gen_max}.')

        # sfeh check if any roots
        # yaml_dump(self.rootdir / 'used_config.yaml', self.conf, print_type=self.print_type)

        # self.pop.gens_since_last_pareto = 0

        while self.conf.gen_id <= self.conf.gen_max and not self.custom_exit_condition():  # max generation, max time, done...
            self.time_genstart = time.perf_counter()  # sfeh here?

            if self.conf.gen_id == 0:
                self.printpl('gg', f'Preparing to create first Generation. Gen {self.conf.gen_id}.')  # sfeh debug
                self.gen_create_initial()  # sfeh stattdessen einfach checken, ob die letzte population leer ist und info/warnung: neue generation?
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
                    self.gen_next_population()
                    # sfeh check if there are really unique... doubt it.

            self.pareto_from_population()
            self.pop_analyze()

            # ================= #

            self.pass_population()
            self.printpl('ggg',
                         f'Generation {self.conf.gen_id} took a total time of: {time.perf_counter() - self.time_genstart:4.2f}.')
            self.scheduled_io()
            self.conf.gen_id += 1
        else:
            self.printpl('g',
                         f'Done after Generation {self.conf.gen_id}.\nTime since start: {time.perf_counter() - self.time_start:4.2f}s')

        self.backup_save()

        return

    def file_analysis_plots(self):
        """
        Make all plots
        """
        self.plot_gen_performance(self.rootdir / 'monitoring.png')  # largest plot analysing the
        self.pareto_plot(self.rootdir / f'paretofront.pdf', self.conf.name, self.conf.parsimony_max)
        # self.plot_evolve_performance()  # sfeh
        return

    def scheduled_io(self):
        """
        show plots if necessary
        """
        save_gen = int(self.conf.period.get('gen_save', 10))
        plot_gen = int(self.conf.period.get('gen_plots', 10))
        if self.conf.gen_id >= plot_gen and self.conf.gen_id % plot_gen == 0:
            self.file_analysis_plots()

        if self.conf.gen_id >= save_gen and self.conf.gen_id % save_gen == 0 or self.conf.gen_id == 10:  # sfeh extra save at 10 for early feedback while testing
            self.backup_save()


# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['label'], 1/v['xtype_out': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast op_dict:


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
    tree = tree_from_nested_string(nstr)
