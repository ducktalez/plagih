from plagih.util import *
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def plot_performance(monitor_df, name, path_monitoring: Path):
    """
    All monitoring infos
    # fit_best is not necessary
    """
    with plt.rc_context(rc={'axes.grid': True}):
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]},
                                sharex='all')
        plt.subplots_adjust(wspace=0, hspace=0.1)  # left=0, bottom=0, right=1, top=1
        xx = list(monitor_df.index)

        axs0 = axs[0]
        axs0.plot(monitor_df['fit_avg'], marker='', label='regression error (average)')
        # sfeh:improvement not just the stderr on both sides...
        avg = monitor_df['fit_avg']
        std = monitor_df['fit_var']
        fit_quantile_25 = monitor_df['fit_quantile_25']
        fit_quantile_50 = monitor_df['fit_quantile_50']
        fit_quantile_75 = monitor_df['fit_quantile_75']
        parsim_avg = monitor_df['parsim_avg']
        parsim_var = monitor_df['parsim_var']
        parsim_best = monitor_df['parsim_best']
        parsim_quantile_50 = monitor_df['parsim_quantile_50']
        parsim_quantile_25 = monitor_df['parsim_quantile_25']
        parsim_quantile_75 = monitor_df['parsim_quantile_75']

        axs0.fill_between(xx, avg - std, avg + std, alpha=0.2)  # do not use avg in both directions...
        axs0.fill_between(xx, fit_quantile_25, fit_quantile_75, color='b', alpha=0.2)
        # axs0.set_title('regression Error (average)')  # sfeh not stderr... upper/lower bound?
        # sfeh: the best candidate is the best one in the current population. discussion: best overall?
        axs0.step(x=xx, y=monitor_df['fit_best'], linestyle='dashed', marker='', where='post', color='g',
                  label='Best candidate')  # , label=ax_label
        # axs0.step(x=xx, y=fit_quantile_50, linestyle='dashed', marker='', where='post', color='b',
        #           label='Best candidate')
        axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

        axs0_twin = axs0.twinx()
        axs0_twin.plot(xx, monitor_df['gens_since_last_pareto'], color='tab:gray',
                       label='Gen since last pareto entry', linestyle='dashed',
                       marker='')  # linestyle='None'
        axs0_twin.tick_params(axis='y', labelcolor='tab:gray')
        axs0_twin.set_ylim(ymin=0, ymax=max(monitor_df['gens_since_last_pareto'].max() or 1, 50))
        # axs0_twin.set_ylim(ymin=0, ymax=max(monitor_df['gens_since_last_pareto'].notnull().max() or 1, 50))
        # # print(monitor_df['gens_since_last_pareto'].notnull().max())

        axs0_twin.legend(loc='lower right')
        axs1 = axs[1]
        axs1.plot(monitor_df['parsim_avg'], label='Complexity (average)')

        p_avg = monitor_df['parsim_avg']
        p_var = monitor_df['parsim_var']
        axs1.fill_between(xx, p_avg - p_var, p_avg + p_var, alpha=0.2)  # axs1.set_title('TED (average)')
        axs1.set_ylim(ymin=0), axs1.legend(loc='lower left')

        axs2 = axs[2]
        axs2.plot(monitor_df['pop_len'], label='pop_list size')
        axs2.plot(monitor_df['pop_unique'], label='unique')
        axs2.margins(y=0.25), axs2.set_ylim(ymin=0), axs2.legend(loc='lower left')

        axs3 = axs[3]
        between_outliers = monitor_df['time'].between(0, 2 * monitor_df['time'].mean())
        axs3.plot(monitor_df['time'][between_outliers], label='time (s)')  # sfeh could be a better rule...
        axs3.set_ylim(ymin=0), axs3.legend(loc='lower left')

        # Top level style
        axs3.set_xlim(xmin=0, xmax=max(xx)), axs3.set_xlabel('generation')
        axs3.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axs0.set_title(f'monitoring GP generations {name}')  # sfeh
        fig.tight_layout()
        fig.savefig(path_monitoring)
        plt.close('all')
