"""
All paretoefficient GP-candidates, aka the "best" entries for each complexity.
-> updated after each generation

"""

from plagih.util import *


def pareto_from_pop(pop_list):
    """Return the nondominated candidates with respect to parsimony and fitness."""

    pop_list = sorted(pop_list, key=lambda _x: (_x.get_parsim(), _x.get_fitness()))

    if not pop_list:
        raise Exception("pareto_from_pop called with empty population")

    def dominates(a, b):
        return (
            a.get_parsim() <= b.get_parsim()
            and a.get_fitness() <= b.get_fitness()
            and (a.get_parsim() < b.get_parsim() or a.get_fitness() < b.get_fitness())
        )

    pop_pareto = []
    for candidate in pop_list:
        if any(dominates(existing, candidate) for existing in pop_pareto):
            continue
        pop_pareto = [existing for existing in pop_pareto if not dominates(candidate, existing)]
        pop_pareto.append(candidate)

    return pop_pareto


def plot_paretofront(paretofront, path, parsimony_max):
    """Write pyplot with paretofront candidates"""

    tuples = [[tree.get_parsim(), tree.get_fitness()] for tree in paretofront]
    xx, yy = np.array(tuples).T

    if len(xx) == 0:
        print_caution(f"Plotting empty array is not possible! Data={xx, yy}")
        return

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots()
        right = max(max(xx), parsimony_max) * 1.05
        xx = np.concatenate([xx, [right + 1]])
        yy = np.concatenate([yy, [yy[-1]]])

        run_name_latex = str(path.name).replace("_", "-")  # workaround for latex version
        ax.step(xx, yy, linestyle="dashed", marker=".", label=f"{run_name_latex}", where="post")
        ax.set(
            xlabel="complexity",
            ylabel="regression error",
            xlim=(0, right),
            ylim=(0, (max(yy) - min(min(yy), 0)) * 1.05),
        )

        try:
            path_png = path / "paretofront.png"
            # path_pdf = path / f'paretofront.pdf'
            fig.savefig(path_png)
            printez("f", f"paretofront (.png): {path_png}")
        except PermissionError as perm_error:
            print_caution(f"Could not save plot: {perm_error}")
        except Exception as ex:
            raise ex
    return


def analyze_pareto(cpu_cores=4):
    """
    sfeh:open
    Writing all analysis files after evaluating the paretofront.
    (Currently strongly customized by sfeh for the mountaincar and industrial benchmark)
    """
    return


def analyze_pareto(cpu_cores=4):
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
    #         res_all = combined_lists(self.rootdir.parent, 40, 40, local_yamls=True, cpu_cores=cpu_cores)
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


def plot_performance(monitor_df, save_path):
    """Plots population performance metrics over generations.

    Creates a figure showing fitness and parsimony statistics
    across generations for monitoring GP progress.

    Args:
        monitor_df: DataFrame with generation statistics.
        save_path: Path to save the plot image.
    """
    if len(monitor_df) == 0:
        print_warning("w", "Cannot plot performance: empty monitor_df")
        return

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Fitness over generations
        ax1 = axes[0, 0]
        ax1.plot(monitor_df.index, monitor_df["fit_best"], "b-", label="Best")
        ax1.plot(monitor_df.index, monitor_df["fit_avg"], "g--", label="Average")
        ax1.fill_between(
            monitor_df.index,
            monitor_df["fit_quantile_25"],
            monitor_df["fit_quantile_75"],
            alpha=0.3,
            label="25-75 percentile",
        )
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title("Fitness Evolution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Parsimony over generations
        ax2 = axes[0, 1]
        ax2.plot(monitor_df.index, monitor_df["parsim_best"], "b-", label="Best")
        ax2.plot(monitor_df.index, monitor_df["parsim_avg"], "g--", label="Average")
        ax2.fill_between(
            monitor_df.index,
            monitor_df["parsim_quantile_25"],
            monitor_df["parsim_quantile_75"],
            alpha=0.3,
            label="25-75 percentile",
        )
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Parsimony")
        ax2.set_title("Parsimony Evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Population diversity
        ax3 = axes[1, 0]
        ax3.plot(monitor_df.index, monitor_df["pop_unique"], "r-", label="Unique")
        ax3.plot(monitor_df.index, monitor_df["pop_len"], "b--", label="Total")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Count")
        ax3.set_title("Population Diversity")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Time per generation
        ax4 = axes[1, 1]
        ax4.bar(monitor_df.index, monitor_df["time"], alpha=0.7)
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Time (s)")
        ax4.set_title("Time per Generation")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        try:
            fig.savefig(save_path)
            printez("f", f"Performance plot saved: {save_path}")
        except PermissionError as e:
            print_caution(f"Could not save plot: {e}")
        finally:
            plt.close(fig)


def plot_parsimony_histogram(population, save_path, max_population=100, max_parsimony=50):
    """Plots histogram of parsimony (complexity) distribution in population.

    Args:
        population: List of Candidate objects.
        save_path: Path to save the plot image.
        max_population: Maximum expected population size for y-axis scaling.
        max_parsimony: Maximum expected parsimony for x-axis scaling.
    """
    if not population:
        print_warning("w", "Cannot plot histogram: empty population")
        return

    parsimony_values = [c.get_parsim() for c in population]

    with plt.rc_context(rc=pyplot_rc_tex):
        fig, ax = plt.subplots(figsize=(10, 6))

        bins = range(0, max_parsimony + 2)
        ax.hist(parsimony_values, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Parsimony (Complexity)")
        ax.set_ylabel("Count")
        ax.set_title("Population Parsimony Distribution")
        ax.set_xlim(0, max_parsimony)
        ax.set_ylim(0, max_population * 0.5)
        ax.grid(True, alpha=0.3, axis="y")

        try:
            fig.savefig(save_path)
            printez("f", f"Parsimony histogram saved: {save_path}")
        except PermissionError as e:
            print_caution(f"Could not save plot: {e}")
        finally:
            plt.close(fig)
