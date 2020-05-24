import pickle
from plagih.modules.printing import *
import csv
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

try:
    import tikzplotlib
except Exception as ex:
    print_e('Need to install tikzplotlib? matplotlib2tikz is outdated. Exception:\n{}'.format(ex))

example_runs = 'run_examples/'

run_files = 'run_files/'
folder_plots = 'plots/'
folder_steps = 'steps/'
folder_pop_analysis = 'pop_dist/'

file_backup_pickle = 'backup/backup.p'  # backup-version is set here
file_conclusion = 'conclusion.txt'

file_pareto = 'info/pareto.txt'
info_config_yaml = 'info/config.yaml'
file_info_config_json = 'info/config.json'
file_info_evolve_dict_yaml = 'info/evolve_list.yaml'
info_distributions_yaml = 'info/distributions_file.yaml'
env_variables_yaml = 'info/env_variables.yaml'

file_config_yaml = 'run_files/config.yaml'
file_config_json = 'run_files/config.json'
samples_ready_p = 'run_files/samples_ready.p'
file_evolve_functions = 'run_files/evolve_functions.yaml'
samples_csv = 'run_files/samples.csv'
operators_csv = 'run_files/operators_csv.csv'
operators_yaml = 'run_files/operators_csv.yaml'
operators_info = 'run_files/operators_csv.yaml'
distributions_file = 'run_files/distributions_file.yaml'
tree_expr_txt = 'run_files/tree_expr.txt'
tree_labels_csv = 'run_files/tree_labels.csv'
tree_numpy_csv = 'run_files/tree_numpy.csv'

pycode_load = '../../benchmarks/gym_mountaincar/agents/quick_eval.py'  # sfeh make pretty solution

folder_histograms = 'agents/'
trees_tex = 'agents_trees.tex'
file_pycode = 'agents/agents.py'
file_pycode_eval = 'eval_agents.py'

T_num_lines = 15  # sfeh this var is not found otherwise


def folder_make_dir(path):
    """
    Checks if the folders for the specified path exist and creates them otherwise.
    Apparently, this procedure is used often.
    """
    if not Path.is_dir(path):
        Path.mkdir(path)
    return path


def file_make_dir(file_path):
    """
    Creates the folder only knowing the file.
    paff/tuuu/fyle.txe  ->  *mkdir* paff/tuuu/
    """
    p = Path(file_path)
    if not p.parent.is_dir():
        p.parent.mkdir(parents=True)
    return p


def write_file_pareto_text(pareto, root_path):
    """
    Save all the pareto efficient candidates to file
    """

    pth = file_make_dir(root_path / file_pareto)

    with Path.open(pth, 'w') as file:
        for parsim, meta in sorted(list(pareto.items())):
            fitness = meta['fitness_train']
            algo_sym = meta['expr_sym']  # save raw version, not the sympified one
            file.write('\nParsimony: \t{0} Fitness: \t{1} Expr: \t{2}'.format(parsim, fitness, algo_sym))


def open_force_write_text(p, text):
    p = Path(p)
    if not p.parent.is_dir():
        p.parent.mkdir(parents=True)
    p.write_text(text)


def experiment_data(experiment_yaml):
    if Path.is_file(experiment_yaml):  # Load config.yaml
        with Path.open(experiment_yaml, 'r') as file:
            experiment_infos = yaml_load(file)

    wer = {
        'env': {
            'observations': {
                'cartPos': {
                    'use as': 'constant',
                    'type': 'float',
                    'custom label': 'cartPos',
                    'insert min': None,
                    'insert max': None,
                    'value box/shape': None,
                    'time delta': None,
                    'time delta 0 name': None,
                    'description': None
                },
                'cartVel': {
                    'type': 'float',
                },
                'poleAngle': {
                    'type': 'float',
                },
                'poleVel': {'type':
                                'float',
                            },
            },
            'action0': {
                'use': 'result',
                'type': 'float',
            },

            'number of observations': None,
            'number of actions': 1,
        }
    }


def write_file_population_karoo(population, pop_name, path, gen_id, print_type=None):
    """
    Save population_* to disk.

    """
    file_path = file_make_dir(path / 'info/' / 'population_{}.csv'.format(str(pop_name)))
    # sfeh? function to tree_ and append each tree
    with Path.open(file_path, 'w+', newline='') as csv_file:  # instead of w+, this was once a. but, pop_new file gets too big over time.
        target = csv.writer(csv_file, delimiter=',')
        if gen_id != 0:
            target.writerows([''])  # empty row before each generation
        target.writerows([['Plagih GP by Simon Fehrer, inspired by Karoo (Kai Staats)', 'Generation:', str(gen_id)]])

        for ii, tree in enumerate(population):
            target.writerows([''])  # empty row before each Tree
            for row in range(0, T_num_lines):  # increment through each row in the array Tree (+ row 0)
                target.writerows([population[ii][row]])

    printez('f', '{}'.format(file_path), print_type=print_type)

    return


def pickle_load(path):
    """
    loads a data_csv_path file that was already split with the csv reader
    """

    with Path.open(path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data


def pickle_dump(path, data, print_type=None):
    """
    saves prepared plagih data to pickle file
    """

    path = file_make_dir(path)
    with Path.open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        printez('f', '{}'.format(path), print_type=print_type)
    return


def yaml_load(yaml_path):

    with Path.open(yaml_path, 'r') as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_yaml


def yaml_dump(path, data, print_type=None):
    """
    saves prepared plagih data to pickle file
    """

    path = file_make_dir(path)
    with Path.open(path, 'w') as file:
        _ = yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        printez('ff', '{}'.format(path), print_type=print_type)
    return


def plot_styleup(x, y, set_left=None, set_right=None, set_top=None, right_padding=1.05, top_padding=1.05):

    top, bottom, left, right = max(y), min(y), min(x), max(x)
    if set_left:
        left = set_left

    if set_top:
        new_top = set_top
    else:
        new_top = (top - min(bottom, 0)) * top_padding  # top * 1.05 for better style

    if set_right:
        right = max(right, set_right)
    new_right = right * right_padding

    return top, bottom, left, right, new_right, new_top


def plot_end(data_2d, plotname_path,
             plt_title='', plt_curve_label='', plt_x_label='', plt_y_label='', yscale='linear',
             step_where='', plt_xparam='', plt_hist=False,
             linestyle='None',
             marker='',
             set_left=None, set_right=None, set_top=None,
             right_padding=1.05, top_padding=1.05,
             beyond_lines=False,
             save_tikz=False,
             subfolder=None):
    """
    Make all plots in the same style - and also saving space.
    - Makes pyplots


    :param data_2d: array with data, e.g. [[1, 5],[2, 4], [3, 4]]
    :param plotname_path: where to save the result
    :param plt_title:
    :param plt_curve_label: irrelevant for a single curve
    :param plt_x_label: label the x-axis
    :param plt_y_label: label the y-axis
    :param yscale: only 'linear'.
    :param step_where: makes 'step' plots- can be 'post', 'pre' or [pls google]
    :param plt_xparam: not in use, the same adjustment can be done with optional parameters
    :param linestyle: E. g. 'None', 'dashed', '-', ''
    :param set_left: Smallest left value
    :param set_right: E. g. if max_parsimony is 100 -> show complete width, even if entries only go to 40
    :param top_padding: How much padding to the top border
    :param beyond_lines: in step plots, draw the line further to the left and right
    :param save_tikz: Also save the plot as tikzpicture (for Latex)(requires tikzplotlib)
    :param subfolder: save plot in plots/*subfolder*, e.g. if this plot is created in every generation
    :return:

    sfeh: max_height=None,  # when creating a plot in every generation, fix the maximum height and width?
    """

    if len(data_2d) == 0:
        print_e('Plotting empty array is not possible! Data={}'.format(data_2d))
        return

    x, y = [], []
    for a, b in data_2d:
        x.append(a)
        y.append(b)

    # x, y = data_2d.reshape(-1, 2).T  # sfeh this could be a more pythonic way, but tuples can not be reshaped.

    # bottom, top = plt.ylim()
    # left, right = plt.xlim()

    top, bottom, left, right, new_right, new_top = plot_styleup(x, y, set_left=set_left, set_right=set_right, set_top=set_top, right_padding=right_padding, top_padding=top_padding)

    if beyond_lines:
        x = [x[0]] + x + [new_right + 1]
        y = [new_top + 1] + y + [y[-1]]

    # plt.close('all')  # todo
    fig, ax = plt.subplots()  # todo

    # if step_where:
    #     plt.step(x, y, plt_xparam, linestyle=linestyle, marker=marker, label=plt_curve_label, where=step_where)
    # # elif plt_hist:
    # #     plt.hist(x, bins='auto', density=1, alpha=0.75)  # not used: facecolor='blue', bins=20
    # else:
    #     plt.plot(x, y, plt_xparam, linestyle=linestyle, marker=marker, label=plt_curve_label)

    if step_where:
        ax.step(x, y, plt_xparam, linestyle=linestyle, marker=marker, label=plt_curve_label, where=step_where)
    else:
        ax.plot(x, y, plt_xparam, linestyle=linestyle, marker=marker, label=plt_curve_label)

    # # let it start at (0,0) but +5% margin to the top and right
    # plt.yscale(yscale)
    # plt.ylim(min(bottom, 0), new_top)
    # plt.xlim(min(left, 0), new_right)
    # plt.margins(x=0, y=0)
    #
    # plt.xlabel(plt_x_label)
    # plt.ylabel(plt_y_label)
    # plt.title(plt_title)

    ax.set_yscale(yscale)
    ax.set_ylim(min(bottom, 0), new_top)
    ax.set_xlim(min(left, 0), new_right)
    # fig.set_margins(x=0, y=0)
    fig.tight_layout()
    ax.set_xlabel(plt_x_label)
    ax.set_ylabel(plt_y_label)
    ax.set_title(plt_title)

    # plt.legend()
    if subfolder:  #
        plotname_path = folder_make_dir(plotname_path / subfolder)

    if save_tikz:
        try:
            tikzplotlib.save(plotname_path / '{}.tex'.format(plt_title))
        except Exception as ex:
            pass

    plt.tight_layout()
    # plt.title(plt_title)
    plt.savefig(plotname_path / '{}.png'.format(plt_title))
    # plt.close()  # Stackoverflow said that this is too much, clf should be better
    plt.clf()
    return

#
# def plot_histogram(x, plt):
#     """
#     Creating histograms for the agents
#     todo delete?
#     """
#     top, bottom, left, right, new_right, new_top = plot_styleup(x, y, set_left=set_left, set_right=set_right, set_top=set_top, right_padding=right_padding, top_padding=top_padding)
#     n, bins, patches = plt.hist(x, bins='auto', density=1, alpha=0.75)  # not used: facecolor='blue', bins=20
#
#     # import matplotlib.mlab as mlab
#     # # add a 'best fit' line
#     # y = mlab.normpdf( bins, mu, sigma)
#     # l = plt.plot(bins, y, 'r--', linewidth=1)
#
#     plt.show()
