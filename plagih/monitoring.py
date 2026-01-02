from plagih.util import *
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from collections import Counter
from pathlib import Path


def plot_performance(monitor_df, path_monitoring: Path):
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
        axs0.set_title(f'monitoring GP generations {path_monitoring.name}')  # sfeh
        fig.tight_layout()
        fig.savefig(path_monitoring)
        plt.close('all')


def plot_parsimony_histogram(population, path_out: Path, *, title: str | None = None, color: str = 'tab:blue',
                            max_population: int | None = None, max_parsimony: int | None = None):
    """Plottet die Parsimony/Komplexität einer Population als Histogramm.

    Ziel: schnell sehen, welche Tree-Größen in der Population vorkommen.

    Eigenschaften:
    - Bins sind ganzzahlige Parsimony-Werte (ein Bin pro Wert).
    - Balken sind nach Evolution (tag) gruppiert und farbcodiert
    - Keine Abstände zwischen den Balken
    - Feste Skalierung für Vergleichbarkeit über Generationen

    Erwartete Populationseinträge:
    - `Candidate`-Objekte (haben `.parsimony` oder `.get_parsim()` und `.tag`)
    - oder Nodes/Trees, wenn sie ein Attribut `.parsimony` haben (Fallback)

    `path_out` sollte ein `pathlib.Path` sein (z.B. `self.rootdir / 'monitoring_parsimony_histogram.png'`).
    `max_population`: Maximale Populationsgröße für feste X-Achsen-Skalierung
    `max_parsimony`: Maximale Parsimony für feste Y-Achsen-Skalierung
    """

    def _get_parsimony(item):
        if item is None:
            return None
        # Candidate: bevorzugt API
        if hasattr(item, 'get_parsim') and callable(getattr(item, 'get_parsim')):
            return getattr(item, 'get_parsim')()
        if hasattr(item, 'parsimony'):
            return getattr(item, 'parsimony')
        return None

    def _get_tag(item):
        """Extract the evolution tag from a candidate"""
        if hasattr(item, 'tag'):
            return item.tag
        return 'unknown'

    pars = []
    tags = []
    for it in population or []:
        p = _get_parsimony(it)
        if p is None:
            continue
        try:
            pars.append(int(round(float(p))))
            tags.append(_get_tag(it))
        except Exception:
            continue

    if len(pars) == 0:
        # Leeres Plotfile erzeugen (robust für Monitoring-Pipelines)
        with plt.rc_context(rc={'axes.grid': True}):
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_title(title or f'Parsimony histogram ({path_out.name})')
            ax.set_xlabel('parsimony / complexity')
            ax.set_ylabel('count')
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            fig.tight_layout()
            fig.savefig(path_out)
            plt.close('all')
        return

    # Group by tag first, then by parsimony
    from collections import defaultdict
    data_by_tag = defaultdict(lambda: defaultdict(int))
    for p, t in zip(pars, tags):
        data_by_tag[t][p] += 1

    all_tags = sorted(data_by_tag.keys())
    all_parsimony_values = sorted(set(pars))

    # Create a color map for tags
    import matplotlib.cm as cm
    import numpy as np
    colors = cm.tab10(np.linspace(0, 1, len(all_tags)))
    tag_colors = dict(zip(all_tags, colors))

    with plt.rc_context(rc={'axes.grid': True}):
        fig, ax = plt.subplots(figsize=(16, 6))

        # Build bars grouped by evolution
        current_position = 0
        xticks_positions = []
        xticks_labels = []

        for tag in all_tags:
            parsimony_counts = data_by_tag[tag]

            # Plot each parsimony value for this tag
            for parsimony_val in all_parsimony_values:
                count = parsimony_counts.get(parsimony_val, 0)
                if count > 0:  # Only plot if there's data
                    ax.bar(current_position, count, width=1.0, align='edge',
                           color=tag_colors[tag], edgecolor='white', linewidth=0.5, label=tag if parsimony_val == all_parsimony_values[0] else "")
                    current_position += 1

            # Add separator or track position for labels
            if tag != all_tags[-1]:  # Don't add gap after last tag
                current_position += 0.5  # Small visual separator between evolution groups

            # Store midpoint for x-axis label
            xticks_positions.append(current_position - (len([p for p in all_parsimony_values if parsimony_counts.get(p, 0) > 0]) + 0.5) / 2)
            xticks_labels.append(tag)

        ax.set_xlabel('Evolution')
        ax.set_ylabel('count')
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        # Fixed scaling for comparability across generations
        if max_population is not None:
            ax.set_xlim(0, max_population)
        else:
            ax.set_xlim(0, current_position)

        if max_parsimony is not None:
            ax.set_ylim(0, max_parsimony)

        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticks_labels, rotation=45, ha='right')
        ax.set_title(title or f'Parsimony histogram ({path_out.name})')

        # Add legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)

        fig.tight_layout()
        fig.savefig(path_out)
        plt.close('all')
