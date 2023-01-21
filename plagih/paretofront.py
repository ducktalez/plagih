"""
All paretoefficient GP-candidates, aka the "best" entries for each complexity.
-> updated after each generation

"""
import matplotlib.pyplot as plt

from plagih.tree_factory import *


def pareto_from_pop(pop_list):
    """
    sfeh check this!! op_next mostly has 2 pareto entries??
    """
    pop_list = pareto_sort(pop_list)

    try:
        ref = pop_list[0]
    except Exception as ex:
        raise Exception(f'The list is empty, i guess: {pop_list}. {ex}')

    ref_par = ref.get_parsimony()
    ref_fit = ref.get_fitness()

    pop_pareto = [ref]

    for tree in pop_list:
        parsim = tree.get_parsimony()
        if parsim == ref_par:  # parsim can not be smaller
            continue  # sfeh:discuss does sorted keep the order?
        else:
            fitness = tree.get_fitness()
            if fitness < ref_fit:
                pop_pareto.append(tree)
                ref_fit = fitness

            ref_par = parsim

    return pop_pareto


def pareto_sort(tree_list):
    tree_list = sorted(tree_list, key=lambda x: (x.get_parsimony(), x.get_fitness()))  # keys can be negative (*-1)
    return tree_list


def pareto_export(paretofront):
    """
    Save all the paretofront candidates to a file.
    (Quick feedback that requires little overhead)
    """
    return [f'Parsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{tree.meta.expr_raw}' for (parsim, fitness, tree)
            in paretofront]


def pareto_insert_again_simplified(self, fintree):
    """
    # sfeh:open
    """
    # tree_sym = copy.deepcopy(evotree)
    #
    # try:
    #     # printez('aaa', 'Trying to simplify for paretofront entry.')  # simplify the fintree and save in paretofront once again
    #     tree_sym.evolve_reduce(obs_infos=obs_infos, completely=True)
    #     parsimony = tree_sym.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
    #     if parsimony < evotree.meta.parsimony:
    #         # self.printpl('aa', 'Successfully reduced paretofront fintree!')
    #         sym_fitness = self.eval_tf_fitness(tree_sym)  # sfeh actually not required, delete this
    #         tree_sym.meta.fitness_train = sym_fitness
    #         tree_sym.meta.parsimony = parsimony
    #         self.update_pareto_with_tree(tree_sym)
    # except Exception as ex:
    #     print_warning('www', f'Tree sympification did not work: {ex}')
    #
    # else:
    #     self.printpl('aaa', 'Pareto entry was already simplified')
    pass


def pareto_plot(paretofront, path, name, parsimony_max):
    """
    Write pyplot with paretofront candidates
    """

    tuples = [[tree.get_parsimony(), tree.get_fitness()] for tree in paretofront]
    xx, yy = np.array(tuples).T

    if len(xx) == 0:
        print_e(f'Plotting empty array is not possible! Data={xx, yy}')
        return

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        right = max(max(xx), tb.parsimony_max) * 1.05  # sfeh check this out 1.05  # if set_right:

        # beyond_lines:  # adding a point to the edges to imply that there are no more values (paretofront-plot)
        # xx = np.concatenate([[xx[0]-1.05], xx, [right + 1]])
        # yy = np.concatenate([[max(yy) + 1], yy, [yy[-1]]])
        xx = np.concatenate([xx, [right + 1]])
        yy = np.concatenate([yy, [yy[-1]]])

        run_name_latex = str(name).replace('_', '-')  # workaround for latex version
        ax.step(xx, yy, linestyle='dashed', marker='.', label=f'{run_name_latex}', where='post')
        ax.set(xlabel='complexity', ylabel='regression error',
               xlim=(0, right),
               ylim=(0, (max(yy) - min(min(yy), 0)) * 1.05))

        try:
            path_pdf = path / f'paretofront.pdf'
            fig.savefig(path_pdf)
            printez('f', f"paretofront (.pdf): {path_pdf}")

            path_png = path / f'paretofront.png'
            fig.savefig(path_png)
            printez('f', f"paretofront (.png): {path_png}")
        except PermissionError as perm_error:
            print_e(f'Could not save plot: {perm_error}')  # sfeh for everything?
        except Exception as ex:
            raise ex
    return


def analyze_pareto(cpu_cores=16):  # sfeh 16 cores? nope
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
    # for (parsim, fitness_train, fintree) in paretofront:
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
