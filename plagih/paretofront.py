"""

"""
from plagih.tree_factory import FinalizedTree
from plagih.util import *


class ParetoFront:
    """
    sfeh
    """

    def __init__(self, fitness_compare, origin=None):
        """
        sfeh conf? meh...
        """
        self.paretos = []
        self.fitness_compare = fitness_compare
        self.origin = origin

    def file_pareto(self, path):
        # sfeh where to use? path-> self.conf.rootdir / 'paretofront.yaml',
        yaml_dump(path, self.paretos)  # sfeh this printing is probably shit

    def fitness_compare(self, a, b):
        """
        todo this is the kernel function...
        """
        return a < b

    def insert(self, tree: FinalizedTree, msg=None, print_type=None):
        """
        was pareto_append
        Appending a candidate to the paretofront.
        - append_tree the entry to the paretofront
        - reset gens_since_last_pareto
        - try to add the tree in its sympified version
        """
        if msg:
            printez('a', f"New entry found! ({msg}): {BColors.RESET}{tree.get_parsimony()}, {tree.get_fitness()}:{BColors.RESET} {tree.meta.expr_raw}", print_type=print_type)
        self.paretos.append(tree)
        self.gens_since_last_pareto = 0

        par = tree.get_parsimony()
        fit = tree.get_fitness()

        self.paretos = [x for x in self.paretos[:] if x.get_parsimony() < par or x.get_fitness() < fit or (par == par and x[1] == par)]
        self.pareto_sort()  # as far as I can tell, not really necessary without using iter()

        # tree_sym = copy.deepcopy(tree)
        # # todo
        # try:
        #     # printez('aaa', 'Trying to simplify for paretos entry.')  # simplify the tree and save in paretos once again
        #     tree_sym.evolve_reduce(obs_infos=obs_infos, completely=True)
        #     parsimony = tree_sym.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
        #     if parsimony < tree.meta.parsimony:
        #         # self.printpl('aa', 'Successfully reduced paretos tree!')
        #         sym_fitness = self.eval_tf_fitness(tree_sym)  # sfeh actually not required, delete this
        #         tree_sym.meta.fitness_train = sym_fitness
        #         tree_sym.meta.parsimony = parsimony
        #         self.update_pareto_with_tree(tree_sym)
        # except Exception as ex:
        #     print_warning('www', f'Tree sympification did not work: {ex}', print_type=print_type)
        #
        # else:
        #     printez('aaa', 'Pareto entry was already simplified', print_type=print_type)

    def plot_paretofront(self, path, run_name, parsimony_max):
        """
        Write pyplot with paretos candidates
        """

        run_name = str(run_name).replace('_', '-')  # sfeh asd workaround for latex version

        tuples = [[tree.get_parsimony(), tree.get_fitness()] for tree in self.paretos]
        xx, yy = np.array(tuples).T

        if len(xx) == 0:
            print_e(f'Plotting empty array is not possible! Data={xx, yy}')
            return

        with plt.rc_context(rc=pyplot_rc_tex):
            fig, ax = plt.subplots()
            right = max(max(xx), parsimony_max) * 1.05  # sfeh check this out 1.05  # if set_right:

            # beyond_lines:  # adding a point to the edges to imply that there are no more values (paretos-plot)
            xx = np.concatenate([[xx[0]], xx, [right + 1]])
            yy = np.concatenate([[max(yy) + 1], yy, [yy[-1]]])

            ax.step(xx, yy, linestyle='dashed', marker='.', label=f'{run_name}', where='post')
            ax.set(xlabel='complexity', ylabel='regression error',
                   xlim=(0, right),
                   ylim=(0, (max(yy) - min(min(yy), 0)) * 1.05))

            try:
                fig.savefig(path)
                # self.ui.printpl('f', f"paretofront (pdf): {self.ui.rootdir / f'paretofront.pdf'}")  # todo
            except PermissionError as perm_error:
                print_e(f'Could not save plot: {perm_error}')  # sfeh for everything?

        return

    def append_clean(self, tree: FinalizedTree):
        self.gens_since_last_pareto = 0
        self.paretos.append(tree)
        self.paretos = self.poplike_to_pareto(self.paretos)

    def random_choice(self):
        return np.random.choice(self.paretos)

    def pareto_sort(self):
        """
        sorting the paretos paretos in list for parsimony
        """
        self.paretos.sort(key=lambda x: x.get_parsimony())

    def pareto_txt(self):
        """
        Save all the paretos candidates to a file.
        (Quick feedback that requires little overhead)
        """
        return [f'Parsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{tree.meta.expr_raw}' for (parsim, fitness, tree) in self.paretos]

    def update_pareto_with_tree(self, tree: FinalizedTree):
        """
        inserts a tree into the paretos front
        """
        parsim = tree.get_parsimony()
        fit = tree.get_fitness()

        p_simpler = [p for p in self.paretos if p.get_parsimony() <= parsim]  # all paretos paretos that are less complex

        if len(p_simpler) == 0:  # all other paretos paretos are more complex
            self.insert(tree, msg=f'new simplest entry')
        else:
            best = min(p_simpler, key=lambda p: p[1])  # the fittest of the less complex ones
            if fit < best.get_fitness():
                self.fitness_compare(fit, best.get_fitness())  # if true, at least one insertion
                self.insert(tree, msg=f'old fitness: {best[1]}')

        self.pareto_sort()  # sfeh check if required
        return

    def poplike_to_pareto(self, pop_list):
        """

        """
        pop_list = sorted(pop_list, key=lambda x: (x.get_parsimony(), x.get_fitness()))  # todo -minus fitness

        try:
            best = pop_list[0]
        except Exception as ex:
            raise Exception(f'the pop is empty, i guess: {pop_list}. {ex}')

        best_par = best.get_parsimony()
        best_fit = best.get_fitness()
        pop_pareto = [best]

        for tree in pop_list:
            parsim = tree.get_parsimony()
            if parsim == best_par:
                continue
            else:
                fitness = tree.get_fitness()
                if self.fitness_compare(fitness, best_fit):
                    pop_pareto.append(tree)
                    best_par = parsim
                    best_fit = fitness

        # print(f'The paretos-efficient candidates of the population are: {pop_pareto}')

        return pop_pareto

    def pareto_from_population(self, pop_next):
        """

        """
        pop_pareto = self.poplike_to_pareto(pop_next)

        if len(self.paretos) == 0:
            print('ASD Paretofrunt wird angefangen!!')  # todo debug paretofront wurde erstellt
            self.paretos.append(pop_next[0])

        cmp = lambda a, b: a < b  # todo

        for tree in pop_pareto:
            success = False
            fit = tree.get_fitness()
            par = tree.get_parsimony()
            if all([par < p.get_parsimony() for p in self.paretos]):
                print('Oye found a new, simpler entry')
                self.paretos.append(tree)
            for p in self.paretos:
                if cmp(fit, p.get_fitness()) and par <= p.get_parsimony():
                    self.paretos.remove(p)
                    success = True
            if success:
                self.append_clean(tree)

        # pop_iter = iter(pop_pareto)
        # pareto_iter = iter(self.paretos)
        #
        # p = next(pareto_iter)
        # p_par = p.get_parsimony()
        # t = next(pop_iter)
        #
        # while True:
        #
        #     # lets try the next (or first) tree out!
        #     while t.get_parsimony() < p_par:
        #         print('Found a NEW, simpler entry!')
        #         self.paretos.append_tree(t)  # todo (every entry in this list is already paretoefficient)
        #         t = next(pop_iter)
        #
        #     t_fit = t.get_fitness()
        #     while not cmp(t_fit, p.get_fitness()):
        #         print(f'removing the deprecated entries: Fitness: {p.get_fitness()}')
        #         self.paretos.remove(p)
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
        #         print(f'SHEESH! We kicked out another paretos entry! old: {p_fit}, new fitness: {t_fit}.')
        #         self.paretos.remove(p)
        #         self.paretos.append_tree(t)
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
        #             print('ASD Pareto feddich :3!!')  # todo debug paretofront wurde erstellt
        #             return
        #
        # # todo sort everything
        #
        # # pop_list = sorted(pop_list, key=lambda x: (x.get_parsimony(), x.get_fitness()))
        # # pop_iter = iter(pop_list)
        # # tree = next(pop_iter)
        # # pop_pareto = [tree]
        # # best_par = tree.get_parsimony()
        # # best_fit = tree.get_fitness()
        # #
        # # while True:
        # #     try:
        # #         tree = next(pop_iter)
        # #         parsim = tree.get_parsimony()
        # #         if parsim == best_par:
        # #             continue
        # #         else:
        # #             fitness = tree.get_fitness()
        # #             if self.kernel.fitness_compare(fitness, best_fit):
        # #                 pop_pareto.append_tree(tree)
        # #                 best_par = parsim
        # #                 best_fit = fitness
        # #     except StopIteration:
        # #         break

        self.pareto_sort()  # sfeh check if required
        return

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
    #     for (parsim, fitness_train, tree) in self.paretos:
    #         # agent_name = f'{self.conf.name}_{parsim:.0f}'
    #         agent_name = f'{self.conf.name}_{self.env_vars.action.name}_{parsim:.0f}'
    #         agent_as_python = tree.eval_pycode()
    #         pygents_list.append_tree([parsim, float(fitness_train), agent_name, agent_as_python])
    #
    #     yaml_dump(self.rootdir / 'pycode_list.yaml', pygents_list, print_type=self.ui.print_type)
    #     path = path_make_dir(self.rootdir / 'pycode_list.yaml')
    #     with Path.open(path, 'w') as file:
    #         _ = yaml.dump(pygents_list, file)  # , default_flow_style=False, sort_keys=False)
    #         printez('ff', f'IB pycode-list: {path.as_posix()}', print_type=self.ui.print_type)  # sfeh always the same print structure... just pass the path?
    #
    #     return
    #

    def analyze_pareto(self, cpu_cores=16):  # sfeh 16 cores? nope
        """
        Writing all analysis files after evaluating the paretofront.
        (Currently strongly customized by sfeh for the mountaincar and industrial benchmark)
        """
        # self.printpl('i', f'Analysing the paretos candidates of the run.')
        #
        # dir_benchmarks = Path(__file__).parent.parent.absolute() / 'benchmarks/'
        # path_hist = path_make_dir(self.rootdir / 'histograms/')
        # pareto_agents = {}
        #
        # for (parsim, fitness_train, tree) in self.paretos:
        #     histograms_path = self.plot_agent_histogram(parsim, tree, path_hist)  # sfeh todo
        #
        #     forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest = self.file_pareto_latex(parsim, tree)
        #
        #     pareto_agents[parsim] = {'parsim': parsim,
        #                              'fitness_train': fitness_train,
        #                              'tree': tree,
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
        #         tex_lines.append_tree([f"{int(pp)}",
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
        #     file_dump(self.rootdir / f'analysis_input.tex', paste, print_type=self.ui.print_type)
        #     file_dump(self.rootdir / f'analysis_overview.tex', latex_treeviz_full_document(paste), print_type=self.ui.print_type)
        #
        #     paste_full = ''.join([tex_tabuline(x[:]) for x in tex_lines])
        #     paste_full = f"{str(self.conf.name).replace('_', '-')}  {tex_include_pdf('monitoring.png')}  {tex_include_pdf('sfehs_eval/evaled_overview.pdf')}\n\n" \
        #                  "\\begin{tabular}{lllllllllllll}\n \\hline\n" \
        #                  "dist & error & reward & parsimony & expression \\tabularnewline \\hline\n" \
        #                  f"{paste_full}" \
        #                  "\\hline\n\\end{tabular}\n\n"
        #     file_dump(self.rootdir / f'analysis_overview_plus.tex', latex_treeviz_full_document(paste_full), print_type=self.ui.print_type)
        #
        # elif 'IB' in self.conf.name:
        #
        #     self.file_pareto_listcode()
        #
        #     if self.conf.name[-2:] == '_0':
        #         """
        #         - complexity_sum, [complexities], regression_sum, [regression_errors], real_sum, [formulas]
        #         - plot with paretos candidates
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
    # def file_pareto_latex(self, parsim, tree):
    #     """
    #     Generates latex-file with the computational tree structure of all paretos agents
    #     - build tree from expression
    #     - fill tree meta-data, just in case we want to visualise anything of it
    #     - create latex-forest representation
    #     """
    #
    #     """
    #     whole procedure from tree to forest core
    #     tight_viz:
    #         0: display every node
    #         1: clever tight-visualisation where possible
    #         2: one single mathematical expression
    #     """
    #
    #     tree.set_fix_nodes(self.origin)
    #     tree = tree.get_oldtree()
    #
    #     pl_forest = lambda x: f'\\plforest{{{x}}}\n'
    #
    #     forest_tree_full = None  # todo pl_forest(latex_brackettree(tree))
    #     forest_tree_tight = None  # todo pl_forest(latex_brackettree_tight(latex_tree_semitight(tree)))
    #     # sfeh workaround delete this
    #     tex_expr_raw = f'${tree.export_visualization_latex()}$'  # sfeh dollars
    #     tex_expr_forest = pl_forest(f'[{tex_expr_raw}]')
    #
    #     path_subfolder_tex = path_make_dir(self.rootdir / 'visualisation')  # sfeh running this in every tree seems unneccesary
    #
    #     """
    #     The following lines delete this
    #     """
    #     # sfeh
    #     file_dump(path_subfolder_tex / f'full_{parsim:02d}.tex', forest_tree_full, verbose='ff', print_type=self.ui.print_type)
    #     file_dump(path_subfolder_tex / f'full_{parsim:02d}_tight.tex', forest_tree_tight, verbose='ff', print_type=self.ui.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_input.tex', tex_expr_raw, print_type=self.ui.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_input_forest.tex', tex_expr_forest, verbose='ff', print_type=self.ui.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_doc.tex', latex_treeviz_full_document(forest_tree_full), verbose='ff', print_type=self.ui.print_type)  # delete this
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
    #     # for (parsim, fitness_train, tree) in self.paretos:
    #     #     agent_name = f'{self.conf.name}_{parsim:.0f}'
    #     #
    #     #     agent_as_python = tree.eval_pycode()
    #     #     all_agents.append_tree(f"class {agent_name}:\n{complete_function.format(agent_as_python)}")
    #     #     all_agent_names.append_tree(agent_name)
    #     #     all_more_info.append_tree(f"('{agent_name}', {agent_name}(), {parsim}, {fitness_train})")
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
    #     #     "sys.path.append_tree(str(Path(Path.cwd() / '../../../'))\n" \
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
    #     # pth = path_make_dir(self.rootdir / self.ui.dir_pycode / f"agents.py")
    #     # with Path.open(pth, 'w') as file:
    #     #     file.write(pyc_complete)
    #     #     self.printpl('ff', f'Pycode: {pth.as_posix()}')
    #
    #     return
