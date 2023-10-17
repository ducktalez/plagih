from pathlib import Path

from matplotlib import pyplot as plt
from plagih.util import print_e


def plot_performance(monitor_df, name, path_monitoring: Path):
    """
    All monitoring infos
    sfeh den shit in Funktionen aufteilen
    # fit_best is not necessari
    """
    with plt.rc_context(rc={'axes.grid': True}):
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]},
                                sharex='all')
        plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
        xx = list(monitor_df.index)

        axs0 = axs[0]
        axs0.plot(monitor_df['fit_avg'], marker='', label='regression error (average)')
        # sfeh:improvement not just the stderr on both sides...
        try:
            avg = monitor_df['fit_avg']
            std = monitor_df['fit_var']
            axs0.fill_between(xx, avg - std, avg + std, alpha=0.2)
            # axs0.set_title('regression Error (average)')  # sfeh not stderr... upper/lower bound?
        except Exception as ex:
            raise Exception(f'Delete this. were there any problems? {ex}')
        # sfeh: the best candidate is the best one in the current population. discussion: best overall?
        axs0.step(x=xx, y=monitor_df['fit_best'], linestyle='dashed', marker='', where='post', color='g',
                  label='Best candidate')  # , label=ax_label
        axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

        axs0_twin = axs0.twinx()
        axs0_twin.plot(xx, monitor_df['gens_since_last_pareto'], color='tab:gray',
                       label='Gens since last paretofront entry', linestyle='dashed',
                       marker='')  # linestyle='None'
        axs0_twin.tick_params(axis='y', labelcolor='tab:gray')
        try:
            axs0_twin.set_ylim(ymin=0, ymax=max(monitor_df['gens_since_last_pareto'].max() or 1, 50))
        except Exception as ex:
            try:
                print_e(f'damn setting ylim not working sfeh :s {ex}')
                axs0_twin.set_ylim(ymin=0,
                                   ymax=max(monitor_df['gens_since_last_pareto'].notnull().max() or 1, 50))
                # print(monitor_df['gens_since_last_pareto'].notnull().max())
            except Exception as ex2:
                print_e(f'damn setting ylim not working, version 2! {ex2}')
                axs0_twin.set_ylim(ymin=0, ymax=50)

        axs0_twin.legend(loc='lower right')

        axs1 = axs[1]
        axs1.plot(monitor_df['parsim_avg'], label='Complexity (average)')
        # self.conf.complexity_measure

        try:
            p_avg = monitor_df['parsim_avg']
            p_var = monitor_df['parsim_var']
            axs1.fill_between(xx, p_avg - p_var, p_avg + p_var, alpha=0.2)  # axs1.set_title('TED (average)')
        except Exception as ex:
            raise Exception(f'Delete this if no raise since some time. {ex}')
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
        axs3.set_xlim(xmin=0, xmax=max(xx)), axs3.set_xlabel('generations')
        axs0.set_title(f'monitoring gp generations {name}')  # sfeh
        fig.tight_layout()
        fig.savefig(path_monitoring)
        plt.close('all')
