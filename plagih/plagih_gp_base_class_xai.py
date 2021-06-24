"""

Functions, that might be addable in the future:
"""
from plagih.node_labels import *

import math
import random
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from plagih.file_interaction import *
from plagih.plagih_config import *
from plagih.tree_factory import *
from plagih.viz_with_latex import *
import copy

np.set_printoptions(linewidth=320)  # set the terminal to  320 characters before line-wrapping in order to view Trees
"""

"""
import copy

from plagih.node_labels import *

from plagih.paretofront import *
import numpy as np
import sklearn.metrics as skm
import tensorflow
import ast
# from sys import getsizeof
import pickle
from pathlib import Path
from plagih.paretofront import *
import yaml
# no imports here
import pickle
from pathlib import Path

import random
import tensorflow
import ast
import re
import numpy as np

tensorflow.compat.v1.disable_eager_execution()  # sfeh damn what was this line good for?


class BColors:  # sfeh can be deleted
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[39m'

    BLACK2 = '\033[40m'
    RED2 = '\033[41m'


def pickle_dump(path, data):
    """

    """
    path = path_make_dir(path)

    with Path.open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    printez('f', f'Backup: {path.as_posix()}')


def yaml_dump(path, data, print_type=None, default_flow_style=True):
    """
    saves prepared plagih data to pickle file
    - default_flow_style=False for dumping in a block style
    """
    path = path_make_dir(path)
    with Path.open(path, 'w') as file:
        _ = yaml.dump(data, file, default_flow_style=default_flow_style, sort_keys=False)
        printez('ff', f'{path.as_posix()}', print_type=print_type)
        return


def print_warning(self, message_type, text, print_type=None):
    """
    Printing warnings
    """

    if print_type:
        if message_type not in print_type:
            return
    if message_type == 'w':
        print(f'{BColors.WARNING}Warning ({message_type}): {text}{BColors.RESET}')  # completely yellow
    else:
        print(f'{BColors.WARNING}Warning ({message_type}): {text}{BColors.RESET}')  # only "Warning" yellow
    return


def path_make_dir(p: Path):
    """
    sfehfun
    """
    folder = p if len(p.suffix) == 0 else p.parent
    folder.mkdir(parents=True, exist_ok=True)
    return p


def file_dump(path, data, verbose='SKIPsfeh', print_type=None):
    path_make_dir(path)
    with Path.open(path, 'w') as file:
        file.write(data)
        printez(verbose, f'{path.as_posix()}', print_type=print_type)


pyplot_size = (3.6, 2.7)  # default: (6.4, 4.8) S: (4, 3)  XXL: (16, 9)  M: (4.8, 3.6) (4.4, 3.3)
plplot_size_up = (3.6, 3.6)
pyplot_rc_tex = {'figure.autolayout': True,
                 'text.usetex': True,
                 'backend': 'pgf',
                 'figure.figsize': pyplot_size,
                 'axes.labelpad': 0.5,  # padding axis-ticks to axis title
                 'xtick.labelsize': 8, 'xtick.major.size': 1.5, 'xtick.major.pad': 1.5,
                 'ytick.labelsize': 8, 'ytick.major.size': 1.5, 'ytick.major.pad': 1.5,
                 'font.size': 10,
                 'legend.fontsize': 9,
                 'savefig.dpi': 600,
                 # 'savefig.pad_inches': 0,
                 # 'lines.linewidth': 1,
                 # 'lines.markersize': 3,
                 # 'axes.xmargin': 0,
                 # 'axes.ymargin': 0
                 }

# sfeh
pyplot_size2 = (3, 2.1)  # default: (6.4, 4.8) S: (4, 3)  XXL: (16, 9)  M: (4.8, 3.6) (4.4, 3.3)
pyplot_rc_tex2 = {'figure.autolayout': True,
                  'text.usetex': True,
                  'backend': 'pgf',
                  'figure.figsize': pyplot_size2,
                  'axes.labelpad': 0.5,  # padding axis-ticks to axis title
                  'xtick.labelsize': 6, 'xtick.major.size': 1.2, 'xtick.major.pad': 1.2,
                  'ytick.labelsize': 6, 'ytick.major.size': 1.2, 'ytick.major.pad': 1.2,
                  'font.size': 9,
                  'legend.fontsize': 8,
                  'savefig.dpi': 600,
                  # 'savefig.pad_inches': 0,
                  # 'lines.linewidth': 1,
                  # 'lines.markersize': 3,
                  # 'axes.xmargin': 0,
                  # 'axes.ymargin': 0
                  }

# https://www.elsevier.com/authors/policies-and-guidelines/artwork-and-media-instructions/artwork-sizing
plot_ratio = 9 / 16  # classic 16/9 ratio
plot_width_twocol = 9 / 25.4  # main relevant cm->inch by /25.4
pyplot_rc_two_column = {'figure.autolayout': True,
                        'text.usetex': True,
                        'backend': 'pgf',
                        'figure.figsize': (3.5433, 2),
                        'axes.labelpad': 0.4,  # padding axis-ticks to axis title
                        'xtick.labelsize': 7, 'xtick.major.size': 1.2, 'xtick.major.pad': 1.2,
                        'ytick.labelsize': 7, 'ytick.major.size': 1.2, 'ytick.major.pad': 1.2,
                        'font.size': 10,
                        'legend.fontsize': 8,
                        'savefig.dpi': 600,
                        'savefig.pad_inches': 0,
                        'lines.linewidth': 1,
                        'lines.markersize': 3,
                        'axes.xmargin': 0,
                        'axes.ymargin': 0
                        }

rc_pyplot_size = {'figure.figsize': pyplot_size}
# ['text.latex.preamble'=r"\usepackage{lmodern}"]

# pyplot_rc_options = {'font.family': 'serif',
#                      'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
#                      'font.sans-serif': ['Helvetica', 'Avant Garde', 'Computer Modern Sans serif'],
#                      'font.cursive': ['Zapf Chancery'],
#                      'font.monospace': ['Courier', 'Computer Modern Typewriter']}

# def plot_rc_default(self):
#     rc('font', weight='bold')    # bold fonts are easier to see
#     rc('tick', labelsize=15)     # tick labels bigger
#     rc('lines', lw=1, color='k') # thicker black lines
#     rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines

"""
For further options see:
https://matplotlib.org/3.3.2/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files
https://matplotlib.org/3.1.0/api/matplotlib_configuration_api.html#matplotlib.RcParams
"""
# 'backend': 'pgf', 'font.family':'serif',
# pyplot_rc_options2 = {
#     'axes.titlesize': 24,
#     'axes.labelsize': 20,
#     'lines.linewidth': 3,
#     'lines.markersize': 10,
#     'xtick.labelsize': 16,
#     'ytick.labelsize': 16,
# }
# pyplot_rc_options_smallsize = {
#     'axes.titlesize': 12,
#     'axes.labelsize': 10,
#     'lines.linewidth': 1,
#     'lines.markersize': 3,
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'figure.subplot.left': 0.15,
#     'figure.subplot.bottom': 0.16,
#     'figure.subplot.right': 0.99,
#     'figure.subplot.top': 0.97,
# }

"""
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
"""
"""
# sfeh tests
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.2)
s = 1 + np.sin(2 * np.pi * t)

rc_params = {'text.usetex': True, 'figure.figsize': (2.8, 2.1),
    'axes.titlesize': 11,
    'axes.labelsize': 8,
    'lines.linewidth': 1,
    'lines.markersize': 2,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,}
with plt.rc_context(rc=rc_params):
    fig, ax = plt.subplots()
    
    fig.tight_layout()
    plt.tight_layout()
    ax.plot(t, s, marker='x')
    ax.set(ylabel='voltage (mV)', title='About as simple as it gets, folks')
    fig.savefig(f"test-{str(rc_params.values())}.pdf")
    plt.show()

"""


def print_blue(txt):
    print(f"{BColors.CYAN}{txt}{BColors.RESET}")
    return


def print_e(txt):
    """
    Printing errors that are not worth stopping by raising an exception
    BColors.FAIL
    """
    print(f'{BColors.RED}ERROR!\n{BColors.WARNING}{txt}{BColors.RESET}\n')


def pickle_load(path: Path):
    """
    loads a pickle file (usually .p or .pkl)
    """
    with Path.open(path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data


def printez(message_type, text, print_type=None):
    """
    giving prints colours, accessable from everywhere
    """
    if print_type:
        if message_type not in print_type:
            return

    if 'i' in message_type:
        print(f'{BColors.CYAN}Info: {text}{BColors.RESET}')
    elif 'f' in message_type:
        print(f'{BColors.MAGENTA}Writing File: {text}{BColors.RESET}{BColors.RESET}')
    elif 'a' in message_type:
        print(f'{BColors.GREEN}Paretofront: {text}{BColors.RESET}')
    else:
        raise Exception(f'print_type-mode {message_type} not known.')

    return


def yaml_load(path: Path):
    """
    .yaml-file loader (saves two lines that I had to look up all the time)
    Especially the Loader has to be specified.
    """
    with Path.open(path, 'r') as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)  # yaml.safe_load sfeh?

    return loaded_yaml


def data_from_csv(df, action):
    """
    todo
    Loads .csv data files.
    - Reading the .csv-file (with pandas)
    - renaming column headers
    - saving header info for later use

    Information that we need to extract for each column:
    - choose_xtype choosing a random observation for leaf nodes
    - filtering observation index
        is there an index (past values, e.g. velocity_0, velocity_1) -> performing filter-evolve on the variables index (velocity_2 -> velocity_3)
    - evalaction data_train, data_test, is the action for the regression? -> more than one action might be required (IB has three action dimensions)
        - action min max -> for kernel regression bounded. occuring min and max values might not be the theoretical min/max values


    deprecated:

    """

    """
    1. split col name
    - check whether its an observation
    --> check whether there are indizes
    - check whether its an action
    --> check unique valuea, min, max
    - check if it should be ignored (deprecated action, irrelevant column)
    2. 
    """
    # todo remove dat shit

    return


class UserInteraction:
    backup_p = 'backup/backup.p'
    folder_pycode = ''

    # Files that can can (in theory) be used to prepare a runable folder. maybe deprecated now, sfeh.
    use_distributions_file = 'run_files/distributions_file.yaml'

    def absolute(self, path, make_dirs=False):
        p = self.rootdir / path
        if make_dirs and not p.parent.is_dir():
            p.parent.mkdir(parents=True)
        return p

    def __init__(self, rootdir, print_type):
        self.rootdir = Path(rootdir)
        self.print_type = print_type
        # sfeh check stuff?

    def print_initializing(self):
        print(f'\n'
              f'\tInitializing Plagih. \n'
              f'\tName: {BColors.CYAN}{self.rootdir.name}{BColors.RESET}. \n'
              f'\tLocated in: \n'
              f'\t{self.rootdir}\n')

    # def plot_agent_histogram(self, parsim, tree, path_hist):
    #     """
    #     Make histograms for all pareto-efficient candidates
    #     sfeh: based on training data- maybe use test data...
    # 
    #     useful code?
    #     # histogram_data = np.digitize(histogram_data, bins)  # digitize: the bin index the entry belongs to
    #     # histogram_data = np.concatenate((histogram_data.reshape(-1, 1), pairwise_fitness.reshape(-1, 1)), axis=1)
    #     # histogram_data = np.multiply.reduce(histogram_data, axis=1)
    #     # hist, bins = np.histogram(histogram_data, bins=bins, weights=pairwise_fitness)
    #     """
    # 
    #     action_bins = self.kernel.histogram_bins(self.env_vars.eval_action.minmax)
    #     expr_sym = tree.eval_expr_sym()
    #     used_observations = tree.get_observation_list()
    #     pairwise_diff = self.kernel.eval_tf(expr_sym, used_observations)['pairwise_diff']
    # 
    #     with plt.rc_context(rc=pyplot_rc_tex):
    #         fig, ax = plt.subplots()
    #         ax.hist(pairwise_diff, bins=action_bins, histtype="stepfilled", facecolor="none", edgecolor='k')
    #         ax.set(ylim=(0, len(self.data_train)), ylabel='frequency', xlabel='deviation')
    #         histpath = path_hist / f'acthist_{parsim}.pdf'
    #         fig.savefig(histpath)
    #         plt.close('all')
    # 
    #     return histpath

    def backup_save(self):
        """
        automatically saves everything important after a certain amount of time
        - save the pareto front (custom_done)
        - save the last generation (custom_done)
        - Save valuable meta-data_csv_path: current generation (custom_done)
        """

        # help_dict = {'self.monitor_evol': self.monitor_evol,
        #              'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh save complete config?    # sfeh i dont think we need the config
        # run_backup_data = self.gen_id, self.pareto, self.pop_base, self.monitor_df, help_dict  # sfeh use this later, help_dict
        # pickle_dump(self.rootdir / self.backup_p, run_backup_data)

        # run_backup_dict = {'self.gen_id': self.gen_id,
        #                    'self.pareto': self.pareto,
        #                    'self.pop_base': self.pop_base,
        #                    'self.monitor_df': self.monitor_df,
        #                    'help_dict': help_dict}
        #
        # path_backupyyy = path_make_dir(self.rootdir / 'backup/backup.yaml')
        # yaml_dump(path_backupyyy, run_backup_dict, self.print_type=self.print_type)

        return

    def gp_load_oparray(self, path_operators=None, print_type=None):
        """
        Offers the possibility for the user to load a .yaml-file of operators for the gp-process.
        operator_pool is used otherwise.
        The Operator must match its version in the 'op'-Array (alternatively search for op_what)
        The second value in the tuple denotes the probability of choosing an operator

        @:param sfeh_no_crazyops: sfeh's workaround for being too lazy to load a file with the actual operators

        Load all operator_pool ready-to-use from a file
        """

        try:
            operator_pool = yaml_load(Path(path_operators))
        except:
            printez('i', 'Opt-in not specified: Operators-file does not exist.\n'
                         'Creating one with a default list of mathematical operator_pool.', print_type=print_type)
            operator_pool = None
        choose_operators = ChooseOperators(operator_pool=operator_pool)

        yaml_dump(self.rootdir / 'backup/operators_used.yaml', operator_pool, default_flow_style=True)  # delete this??

        return choose_operators

    def check_update(self):
        """
        Update pareto front
        update population (fitness_train, parsimony, etc)
        also, raise if fitness_train problem
        """
        # oldsize = len(self.pareto)
        # pareto_list = self.pareto[:]
        # self.pareto = []
        # for (parsimony, fitness_train, tree) in pareto_list:
        #     fitness_train = round(fitness_train, self.conf.precision)
        #     tree.meta.fitness_train = round(fitness_train, self.conf.precision)
        #     entry = [parsimony, fitness_train, tree]
        #     self.pareto_append(entry)
        # self.printpl('i', f'Updating pareto front from old run. length: {oldsize}, new pareto length: {len(self.pareto)}')

        # sfeh recompute parsimony and fitness_train for every tree (optional?)
        # also rebuild trees?

        poplen = len(self.pop_base)
        pop_base_copy = self.pop_base[:]
        self.pop_base = []
        for tree in pop_base_copy:
            tree.meta.fitness_train = round(tree.meta.fitness_train, self.conf.precision)
            self.pop_base.append(tree)
        self.printpl('i', f'Updating population from old run. length: {poplen}, new population length: {len(self.pop_base)}')

    def backup_load(self, path_load_backup=None):
        """
        If a backup-file is found...
        """
        path_backup = path_load_backup or self.rootdir / self.io.backup_p  # sfeh file-load

        if Path.is_file(path_backup):
            self.print_g('g', f'Loading data from backup-file {path_backup}')
            """
            Loading the state of the run from the pickle file
            """
            try:
                with Path.open(path_backup, 'rb') as file:
                    run_data = pickle.load(file)

            except NotImplementedError as nimp:
                raise Exception(f'NotImplementedError: {nimp}')
            except EOFError as eoferr:
                raise Exception(f'EOFError: \n{eoferr}')

            try:
                self.gen_id, self.pareto, self.pop_base, monitor_pd, a_helping_dict = run_data  # sfeh use a helping dictt a_helping_dict is used for a useable sldifjsdfsdfg , a_helping_dict
                self.monitor_df = monitor_pd  # sfeh
                if 'gens_since_last_pareto' not in self.monitor_df.columns:
                    self.monitor_df['gens_since_last_pareto'] = np.nan
                self.monitor_evol = a_helping_dict.get('self.monitor_evol') or self.monitor_evol
                self.gens_since_last_pareto = a_helping_dict.get('gens_since_last_pareto') or 0  # sfeh
            except:
                self.gen_id, self.pareto, self.pop_base, m = run_data

            self.check_update()
            self.print_g('g', f'Successfully loaded backup file. Generation: {self.gen_id}')

            # except Exception as ex:
            #     raise Exception(f'Even though a backup exists for this run, it could not be loaded, because of\n{ex}')
        else:
            raise FileNotFoundError(f'No backup-file found at {path_backup}.')
        return

    def file_pareto(self, pareto_txt):
        yaml_dump(self.rootdir / 'paretofront.yaml')

    def print_g(self, message_type, text):
        """
        print informations about generations (g) progress.
        Very common print, showing the progress of the generations.
        always printing the time since start. used to be colored blue.
        """
        if message_type in self.print_type:
            # time_passed = time.perf_counter() - self.time_start
            time_now = time.strftime("%d.%m %H:%M", time.localtime())
            print(f'[{time_now}] {text}')
        return


class PtreeMeta:
    """
    sfeh: fitness_train vs. fitness_train
    """

    def __init__(self, fitness_train=None):
        self.hash = None
        self.fitness_train = fitness_train
        self.parsimony = None
        self.expr_raw = None
        self.expr_sym = None

        self.depth = None
        self.complete = None
        self.last_evolution = None

    def __str__(self):
        return f"hash: {self.hash}, fitness_train: {self.fitness_train}, parsimony: {self.parsimony}, {self.depth}, {self.last_evolution}, {self.expr_raw}, {self.expr_sym}"


# @dataclass  # sfeh maybe
class NodeLabel:  # todo
    """
    Kind of abstract class; Dummy-node that holds a nlabel
    """
    nlabel = None
    arity = None
    xtype = None

    tflow = None  # not tf, might be confusing
    pycode = None
    latex = None

    expr_sym = None

    # def __init__(self, *args, **kwargs):
    #     # nlabel=None, arity=0, xtype=(tuple([None]), None), tflow=None, expr_sym=None, pycode=None, latex=None
    #     # self.nlabel = nlabel
    #     # self.arity = arity
    #     # self.xtype = xtype
    #     #
    #     # self.tflow = tflow  # not tf, might be confusing
    #     # self.nlabel = expr_sym
    #     # self.pycode = pycode
    #     # self.latex = latex
    #
    #     # new version?
    #     self.nlabel = kwargs.get('nlabel', None)
    #     self.arity = kwargs.get('arity', None)
    #     self.xtype = kwargs.get('xtype', None)
    #
    #     self.tflow = kwargs.get('tflow', None)  # not tf, might be confusing']
    #     self.nlabel = kwargs.get('expr_sym', None)
    #     self.pycode = kwargs.get('pycode', (tuple([None]), None))
    #     self.latex = kwargs.get('latex', None)

    # def __str__(self):
    # # sfeh currently not (really) working. returns the subclass.
    #     return self.nlabel

    def mutate_filter(self, *args, **kwargs):
        """
        was filter_new_index
         # as default, return own index
        """
        # if self.index_minmax is None:
        pass

    def mutate_point(self):
        """

        """
        # if self.index_minmax is None:
        pass


class BuildNode(NodeLabel):
    """

    """
    pass


class Operator(NodeLabel):
    """
    operator nodes (+, +, *, /, sin(), sign(), ...)
    inner nodes of a tree
    """
    pass


class Terminal(NodeLabel):
    arity = 0

    def mutate_filter(self, *args, **kwargs):
        # todo? ...only for terminal nodes
        pass


class Constant(Terminal):
    arity = 0

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def mutate_filter(self, *args, **kwargs):
        pass


class Observation(Terminal):
    """

    todo discuss: labels should not have a sign (-pos); just pos
    # self.name = nlabel if nlabel[0] != '-' else nlabel[1:]  # sfeh delete?
    """

    # tf_type = tensorflow.float32  # todo yeah...

    def __init__(self, nlabel):
        # todo xtype_out=float
        self.nlabel = nlabel
        self.fam, self.timeindex, _ = observation_get_family_and_time(self.nlabel, none_return=None)  # remove this self.preexpr
        self.xtype = (tuple([]), float)  # todo
        self.sym_str = self.nlabel  # sfeh delete?
        self.index_minmax = None

        latex = f'\\text{{{self.fam}}}'  # remove this {self.preexpr}
        self.latex = (latex, latex)


# class ObservationIndex(Observation):
#     """
#     todo
#     """
#
#     def __init__(self, nlabel, xtype=float, obs_indizes=None):
#         # super().__init__(nlabel, xtype)
#         self.obs_indizes = obs_indizes
#         latex = f'\\text{{{self.fam}}}_{{{self.timeindex}}}'  # remove this {self.preexpr}
#         self.latex = (latex, latex)  # remove this {self.preexpr}
#
#     def mutate_filter(self):
#         new_index = int(max(min(round(random.gauss(self.timeindex, 1)), self.index_minmax[1]), 0))
#         self.timeindex = new_index
#         self.name = f'{self.fam}_{new_index}'


def observation_get_family_and_time(name, re_pattern='_\\d+$', none_return=None):
    """
    When an observation is known, return the family, the time and the SIGN!!
    todo put this function somewhere where it can actually help
    """

    core_expr = re.split(re_pattern, name)[0]
    if core_expr[0] == '-':
        core_expr = core_expr[1::]
        preexpr = '-'
    else:
        preexpr = ''
    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception:
        temp_diff = none_return
    return core_expr, temp_diff, preexpr


class FloatConstant(Constant):
    """
    discuss: how to deal with sign of observations?
    """
    arity = 0
    otype = float
    xtype = (tuple([]), float)

    def __init__(self, nlabel):
        # super().__init__(nlabel)
        self.nlabel = nlabel
        self.latex = (f'{self.nlabel:.3f}', f'{self.nlabel:.3f}')
        self.sym_str = self.nlabel
        self.pycode = self.nlabel

    def mutate_filter(self, filter_type='gaussian_filter', precision=6, *args, **kwargs):  # todo
        """

        """
        if filter_type == 'gaussian_filter':
            if random.choice(['v1', 'v2']) == 'v1' or self.nlabel == 0:
                constant = self.nlabel + np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.nlabel, 0.1)  # sfeh better adjustments?
            self.nlabel = round(constant, precision)  # todo sfeh be careful, might create zero sometimes


class BoolConstant(Constant):
    """
    True/False
    """
    xtype = (tuple([]), bool)
    tf_type = tensorflow.bool

    def __init__(self, expr):
        # super().__init__(nlabel)
        self.nlabel = expr
        self.latex = (f'{self.nlabel}', f'{self.nlabel}')
        self.sym_str = str(self.nlabel)
        self.pycode = str(self.nlabel)

    def mutate_filter(self, *args, **kwargs):
        """
        sfeh: filtering these is kind of nonsense
        """
        pass


class EvalAction:
    """
    todo plabel/labelNode? daut it.
    - minmax for histograms
    - minmax for regression-bounded
    """
    tf_type = tensorflow.float32  # sfeh especiall when the type is integer
    xtype = (None, float)

    def __init__(self, name):
        self.nlabel = name
        self.name = name  # delete this


class Add(Operator):
    nlabel = '+'
    arity = 2
    tflow = tensorflow.add
    latex = ('+', '{}+{}')
    sym_str = '({} + {})'
    pycode = '({}+{})'
    xtype = (tuple([float, float]), float)
    #
    # def __init__(self, *args, **kwargs):
    #     pass
    #     # super().__init__(*args, **kwargs)


class Subtract(Operator):
    """

    """
    nlabel = '-'
    arity = 2
    tflow = tensorflow.subtract
    latex = ('-', '{}-{}')
    sym_str = '({} - {})'
    pycode = '({}-{})'
    xtype = (tuple([float, float]), float)


class Usub(Operator):
    nlabel = 'Usub'
    arity = 1
    tflow = tensorflow.negative
    latex = ('-', '-{}')
    sym_str = '(-{})'
    pycode = '(-{})'
    xtype = (tuple([float]), float)


class Multiply(Operator):
    nlabel = '*'
    arity = 2
    tflow = tensorflow.multiply
    latex = ('\\cdot ', '{}\\cdot {}')
    sym_str = '({} * {})'
    pycode = '({}*{})'
    xtype = (tuple([float, float]), float)


class Divide_no_nan(Operator):
    nlabel = '/'
    arity = 2
    tflow = tensorflow.math.divide_no_nan
    latex = ('\\div ', '\\frac{}{}')
    sym_str = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'
    xtype = (tuple([float, float]), float)


class Power(Operator):
    nlabel = '**'
    arity = 2
    tflow = tensorflow.pow
    latex = ('{{x}}^{{y}}', '{}^{}')
    sym_str = '({}**Round({}))'
    pycode = '({}**round({}))'
    xtype = (tuple([float, float]), float)


class Abs(Operator):
    nlabel = 'abs'
    arity = 1
    tflow = tensorflow.abs
    latex = ('abs', '|{}|')
    expr = 'abs({})'
    pycode = 'abs({})'
    xtype = (tuple([float]), float)


class Sign(Operator):
    nlabel = 'sign'
    arity = 1
    tflow = tensorflow.sign
    latex = ('sign', 'sign({})')
    sym_str = 'sign({})'
    pycode = 'np.sign({})'
    xtype = (tuple([float]), float)


class Round(Operator):
    nlabel = 'Round'
    arity = 1
    tflow = tensorflow.round
    latex = ('round', 'round({})')
    sym_str = 'Round({})'
    pycode = 'round({})'
    xtype = (tuple([float]), float)


class Square(Operator):
    nlabel = 'Square'
    arity = 1
    tflow = tensorflow.square
    latex = ('x^2', '{}^2')
    sym_str = 'Square({})'
    pycode = '({})**2'
    xtype = (tuple([float]), float)


class Sqrt(Operator):
    nlabel = 'sqrt'
    arity = 1
    tflow = tensorflow.sqrt
    latex = ('\\sqrt{x}', '\\sqrt{}')
    sym_str = 'sqrt({})'
    pycode = 'math.sqrt({})'
    xtype = (tuple([float]), float)


class Log(Operator):
    nlabel = 'log'
    arity = 1
    tflow = tensorflow.math.log
    latex = ('\\log()', '\\log{}')
    sym_str = 'log({})'
    pycode = 'math.log({})'
    xtype = (tuple([float]), float)


class Log1p(Operator):
    nlabel = 'log1p'
    arity = 1
    tflow = tensorflow.math.log1p
    latex = ('\\log(1+x)', '\\log(1+{})')
    sym_str = 'log1p({})'
    pycode = 'math.log1p({})'
    xtype = (tuple([float]), float)


class Cos(Operator):
    nlabel = 'cos'
    arity = 1
    tflow = tensorflow.cos
    latex = ('\\cos ', '\\cos({})')
    sym_str = 'cos({})'
    pycode = 'math.cos({})'
    xtype = (tuple([float]), float)


class Sin(Operator):
    nlabel = 'sin'
    arity = 1
    tflow = tensorflow.sin
    latex = ('\\sin ', '\\sin({})')
    sym_str = 'sin({})'
    pycode = 'math.sin({})'
    xtype = (tuple([float]), float)


class Tan(Operator):
    nlabel = 'tan'
    arity = 1
    tflow = tensorflow.tan
    latex = ('\\tan ', '\\tan({})')
    sym_str = 'tan({})'
    pycode = 'math.tan({})'
    xtype = (tuple([float]), float)


class Acos(Operator):
    nlabel = 'acos'
    arity = 1
    tflow = tensorflow.acos
    latex = ('\\acos ', '\\acos({})')
    sym_str = 'acos({})'
    pycode = 'math.acos({})'
    xtype = (tuple([float]), float)


class Asin(Operator):
    nlabel = 'asin'
    arity = 1
    tflow = tensorflow.asin
    latex = ('\\asin ', '\\asin({})')
    sym_str = 'asin({})'
    pycode = 'math.asin({})'
    xtype = (tuple([float]), float)


class Atan(Operator):
    nlabel = 'atan'
    arity = 1
    tflow = tensorflow.atan
    latex = ('\\atan ', '\\atan({})')
    sym_str = 'atan({})'
    pycode = 'math.atan({})'
    xtype = (tuple([float]), float)


class Tanh(Operator):
    nlabel = 'tanh'
    arity = 1
    tflow = tensorflow.tanh
    latex = ('\\tanh ', '\\tanh({})')
    sym_str = 'tanh({})'
    pycode = 'math.tanh({})'
    xtype = (tuple([float]), float)


class And(Operator):
    nlabel = 'Andb'
    arity = 2
    tflow = tensorflow.logical_and
    latex = ('and', '({}\\wedge{})')
    sym_str = 'Andb({}, {})'
    pycode = '({} and {})'
    xtype = (tuple([bool, bool]), bool)


class Or(Operator):
    nlabel = 'Orb'
    arity = 2
    tflow = tensorflow.logical_or
    latex = ('or', '({}\\vee{})')
    sym_str = 'Orb({}, {})'
    pycode = '({} or {})'
    xtype = (tuple([bool, bool]), bool)


class Xor(Operator):
    nlabel = 'Xor'
    arity = 2
    tflow = tensorflow.math.logical_xor
    latex = ('\\oplus', '({}\\oplus{})')
    sym_str = 'Xor({}, {})'
    pycode = '({} ^ {})'
    xtype = (tuple([bool, bool]), bool)


class Not(Operator):
    nlabel = 'Notb'
    arity = 1
    tflow = tensorflow.logical_not
    latex = ('\\neg', '\\neg{}')
    sym_str = 'Notb({})'
    pycode = 'not({})'
    xtype = (tuple([bool]), bool)


class Eq(Operator):
    nlabel = '=='
    arity = 2
    tflow = tensorflow.equal
    latex = ('=', '({}={})')
    sym_str = '({} == {})'
    pycode = '({}=={})'
    xtype = (tuple([bool, bool]), bool)


class Neq(Operator):
    nlabel = '!='
    arity = 2
    tflow = tensorflow.not_equal
    latex = ('\\neq', '({}\\neq{})')
    sym_str = '({} != {})'
    pycode = '({}!={})'
    xtype = (tuple([bool, bool]), bool)


class Lt(Operator):
    nlabel = '<'
    arity = 2
    tflow = tensorflow.less
    latex = ('<', '{}<{}')
    sym_str = '({} < {})'
    pycode = '({}<{})'
    xtype = (tuple([float, float]), bool)


class Le(Operator):
    nlabel = '<='
    arity = 2
    tflow = tensorflow.less_equal
    latex = ('\\leq', '{}\\leq{}')
    sym_str = '({} <= {})'
    pycode = '({}<={})'
    xtype = (tuple([float, float]), bool)


class Gt(Operator):
    nlabel = '>'
    arity = 2
    tflow = tensorflow.greater
    latex = ('>', '{}>{}')
    sym_str = '({} > {})'
    pycode = '({}>{})'
    xtype = (tuple([float, float]), bool)


class Ge(Operator):
    nlabel = '>='
    arity = 2
    tflow = tensorflow.greater_equal
    latex = ('\\geq', '{}\\geq {}')  # sfeh check inserted space
    sym_str = '({} >= {})'
    pycode = '({}>={})'
    xtype = (tuple([float, float]), bool)


class Ifte(Operator):
    nlabel = 'Ifte'
    arity = 3
    tflow = tensorflow.where
    latex = ('\\text{if-then-else}', '\\text{{ if }} ({}) \\text{{ then }} ({}) \\text{{ else }} ({})')  # 'if({} then {} else {})'
    sym_str = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'
    xtype = (tuple([bool, float, float]), float)


class Min(Operator):
    nlabel = 'Mini'
    arity = 2
    tflow = tensorflow.minimum
    latex = ('\\min', '\\min({}, {})')
    sym_str = 'Mini({}, {})'
    pycode = 'min({}, {})'
    xtype = (tuple([float, float]), float)


class Max(Operator):
    nlabel = 'Maxi'
    arity = 2
    tflow = tensorflow.maximum
    latex = ('\\max', '\\max({}, {})')
    sym_str = 'Maxi({}, {})'
    pycode = 'max({}, {})'
    xtype = (tuple([float, float]), float)


op = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': Add,
    ast.Add: Add,
    '-': Subtract,
    ast.Sub: Subtract,
    'Usub': Usub,
    ast.USub: Usub,
    '*': Multiply,
    ast.Mult: Multiply,
    # Division: SAFE division by zero! -->tensorflow.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': Divide_no_nan,
    ast.Div: Divide_no_nan,
    '**': Power,
    ast.Pow: Power,
    'Abs': Abs,
    'sign': Sign,
    'Round': Round,
    'Square': Square,
    'sqrt': Sqrt,
    'log': Log,  # sfeh log/ln?
    'log1p': Log1p,
    'cos': Cos,
    'sin': Sin,
    'tan': Tan,
    'acos': Acos,
    'asin': Asin,
    'atan': Atan,
    'tanh': Tanh,

    # bool->bool
    # DON'T USE tensorflow.bitwise.bitwise_and
    # sympify('Or')->'|', sympify('And')->'&', sympify('Not')->'~'
    'Andb': And,
    ast.And: And,
    'Orb': Or,
    ast.Or: Or,
    'Xor': Xor,
    # ast.BitXor: Xor,
    'Notb': Not,
    ast.Not: Not,

    # float->bool
    '==': Eq,
    ast.Eq: Eq,
    '!=': Neq,
    ast.NotEq: Neq,
    '<': Lt,  # a < b
    ast.Lt: Lt,
    '<=': Le,
    ast.LtE: Le,
    '>': Gt,  # a > b
    ast.Gt: Gt,
    '>=': Ge,  # a >= 1
    ast.GtE: Ge,

    'Ifte': Ifte,  # sfeh essential for evaluation
    'Mini': Min,  # with forced arity-2
    'Maxi': Max,  # with forced arity-2
}


@dataclass
class Node:
    """
    The core is the structure of a plagih gp-tree.
    It recursively holds the nodes of a tree; every tree has a list of potential children.
    Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)

    states?
    [None]: not set
    [0]:    evolution/construction/build mode (potentially missing leaf nodes)
    [1]:    structurally complete/finalized branch (node_depths correct, node_id set, ...)
    [2]:    root-correct structure (todo not relevant?)
    [3]:    including meta-data (fitness_train, complexity)
    """
    # todo
    # arity: int
    # nlabel: str
    # pycode: str

    state = None

    def __init__(self, label: 'NodeLabel' = None, is_fix=False, childs=None, state=0):
        self.label = label
        self.is_fix = is_fix
        self.childs = childs or []

        self.state = state

    def __hash__(self):
        """
        This hash function has currently no use.
        The hash-value of a tree was used as key for the LUT.
        However, the python hash-function has a run-specific salt for security reasons,
        making it impossible to load the LUT table between runs, so just use the __str__ as key.
        """
        return hash(str(self))  # sfeh

    def __str__(self):
        """
        Printing the nodes as nested array structure.
        sfeh: make this statement loadable!
        """
        print_label = self.get_nlabel()  # sfeh or: return the label __str__
        # print_label = str(self.label)  # returns the class, not a string

        if self.childs:
            childstr = ', '.join([str(x) for x in self.childs])
            print_label = f"{print_label}, {childstr}"
        return f"[{print_label}]"

    def loadable_string(self):
        """
        todo
        """
        print_label = self.get_label().nlabel
        if self.is_fix:
            print_label = f'({print_label})'

        if self.childs:
            childstr = ', '.join([str(x) for x in self.childs])
            print_label = f"{print_label}, {childstr}"
        return f"[{print_label}]"

    def __len__(self):
        """
        counting the amount of nodes recursively
        """
        return 1 + sum([len(cc) for cc in self.childs])

    def get_label(self):
        return self.label

    def get_nlabel(self):
        """:param
        todo rename nlabel?
        """
        return self.label.nlabel

    def get_expr_sym(self):
        return self.label.expr_sym

    def get_pycode(self):
        return self.label.pycode

    def get_arity(self):
        return self.label.arity

    def get_xtype(self):
        return self.label.xtype

    def get_xtype_out(self):
        return self.label.xtype[1]

    def is_root(self):
        """
        ==<ROOT
        """
        return self.id == 0

    def get_observation_list(self):
        """
        these are required for the evaluation (are loaded by Tensorflow)
        """
        print('sd', self.get_nlabel())
        obslist = []
        if self.get_arity() > 0:
            obslist.extend(list(itertools.chain(*[cc.get_observation_list() for cc in self.childs])))
        elif isinstance(self.label, Observation):
            obslist.extend([self.get_nlabel()])

        return list(set(obslist))

    def set_label(self, label: 'NodeLabel'):
        """
        all other values are automatically set by assigning the respected node
        """
        self.label = label

    def set_childs(self, childs):
        """

        """
        if len(childs) == self.get_arity():
            self.childs = childs
        return  # ==>STATE?

    def get_nodes_to_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """
        sum_layers=False, get_closest=True, return_all_layers=False
        """
        child_results = []
        if self.depth < goal_depth:
            child_results = sum(
                [child.get_nodes_to_depth(goal_depth, only_mutable=only_mutable, force_depth=get_closest_depth) for
                 child in self.childs], [])

        if only_mutable and self.is_fix or \
                get_closest_depth and self.depth != goal_depth:
            my_result = []
        else:
            my_result = [self]

        return my_result + child_results

    def get_labellist_breath(self):
        """
        Returns all labels in a core node
        Breitensuche im Baum
        """
        label_list = []
        max_depth = self.childs_depth_max
        for depth in range(0, max_depth + 1):
            labels_at_depth = [x.label for x in self.get_nodes_at_depth(depth)]
            label_list.extend(labels_at_depth)
        return label_list

    def get_nodes_at_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """
        Returns a list with mutatable ids which are *goal_depth* layers away from non modifiable nodes
        last_leaves: if you want so save all leave nodes aswell

        sum_layers=False, get_closest=True, return_all_layers=False
        """
        if self.depth < goal_depth:
            return sum(
                [child.get_nodes_at_depth(goal_depth, only_mutable=only_mutable, get_closest_depth=get_closest_depth)
                 for child in self.childs], [])
        else:
            if only_mutable and self.is_fix:
                return []
            if get_closest_depth and self.depth != goal_depth:
                return []

            return [self]

    def eval_expr_sym(self, reducible=None, obs_names=None):
        """
        accumulate and return the complete expression the tree holds recursively
        """
        if self.get_arity() > 0:
            child_expr_list = [cc.get_nlabel(reducible=reducible, obs_names=obs_names) for cc in self.childs]
            # if reducible:
            #     # my_expr = op[self.get_label()]['sym_reduce'] or my_expr
            #     # symloc = sympy_symbol_defaults(obs_names)  # todo solve the problem... new version of sympy?
            #     xxx = plagih_sympify(my_expr.format(*child_expr_list), eval_locals=symloc)  # sfeh the xxx variable
            #     return xxx
            return self.eval_expr_sym().format(*child_expr_list)  # f'cos({})'([33]) does not work. *list makes the list args :D
        else:
            return self.eval_expr_sym()

    def eval_pycode(self):
        """

        """
        if self.get_arity() == 0:
            return f'{self.get_pycode()}'
        else:
            results = [cc.eval_pycode() for cc in self.childs]
            return self.get_pycode().format(*results)  # abs -> lambda a: 'abs({})'.formadt(a) (result1)

    def eval_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
        e.g. {+{Ifte{True}{1}{2}}{3}}
        """
        # sfEh check if this still works as one-liner
        return f"{{{self.get_nlabel()}{''.join([cc.eval_apted_notation() for cc in self.childs])}}}"

    def eval_parsimony(self, parsimony_distance, origin_tree=None, weights=None):
        """
        parsimony_distance: compute the chosen distance by the user.
        #     'tree_node_count': tree_get_size,
        #     'tree_depth': tree_get_depth,
        #     'tree_edit_distance': tree_parsimony_ted,

        # self.meta.parsimony = parsimony  # todo okay where meta? at root node...
        """
        if parsimony_distance == 'tree_node_count':  # number of nodes
            return len(self)  # returns the number of nodes  # sfeh weights
        elif parsimony_distance == 'tree_edit_distance':  # tree_edit_distance, tree-edit-distance
            apted1 = self.eval_apted_notation()
            apted2 = origin_tree.eval_apted_notation()
            distance, mapping = apted_distance(apted1, apted2)  # sfeh the mapping could be handy somewhere
            if weights is None:
                return distance
            else:
                raise
        else:
            raise Exception(f'Complexity measurement not available: {parsimony_distance}')

    def replace_with_branch(self, new_node: 'Node'):
        """
        todo
        was: new_core
        """
        self.set_label(new_node.get_label())
        self.childs = new_node.childs or []  # maybe must be updated recursively
        self.state = 1  # todo ==>state
        # self.is_fix = new_node.is_fix  # debatable

    def eval_mutatable_nodes(self, xtype_out=None, allow_root=True):
        """
        return all nodes that are mutatable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!
        """
        node_list = []
        if not self.is_fix:
            if xtype_out is None or xtype_out == self.get_xtype()[1] and (allow_root or not self.is_root()):
                # crossover requires excluding types that are not matching, and excludes the root node
                node_list.append(self)

        node_list.extend(list(itertools.chain(*[cc.eval_mutatable_nodes(xtype_out=xtype_out, allow_root=allow_root) for cc in self.childs])))
        return node_list

    def evolve_mutate_filter_branch(self, precision=6):
        """
        Recursively filter the nodes in the branch of tree
        sfeh:   random filter all terminal nodes /
                single node /
                nodes in a branch /
                random nodes in a branch /
                intelligent filtering
        """
        if self.get_arity() > 0:
            for cc in self.childs:
                cc.evolve_mutate_filter_branch(precision=precision)
        else:
            self.label.mutate_filter(precision=precision)

    def evolve_mutate_point_ROOTNODE(self, tb):
        """
        ==>todo rootnode
        Mutate a single mutatable point in any Tree.
        """
        # 1. choose a node
        node_list = self.eval_mutatable_nodes()
        node = random.choice(node_list)
        node.evolve_mutate_point(tb)

    def evolve_reduce(self, obs_infos=None, completely=True):
        """
        Reducing a tree to its most basic form with sympify.
        (completely = False: reduce just one branch. if you wanted to have more complexity)
        """
        length_before = len(self)
        if completely:  # reduce the complete tree
            cores_lv0 = self.get_nodes_at_depth(0, only_mutable=True)
            for c in cores_lv0:
                c.evolve_branch_reduce(obs_infos)
        else:
            nodes = self.eval_mutatable_nodes()
            functions = [x for x in nodes if x.arity > 0]
            if functions:
                chosen = random.choice(functions)
                chosen.evolve_branch_reduce(
                    obs_infos)  # sfeh chosen must be set again? or not? test it at least. probably working.
        if length_before < len(self):
            print_e(f'FFS Trees just become larger? {self.get_nlabel()}')
        # self.meta.clear()

    def workaround_remove_tilde(self):
        if isinstance(self.get_label(), Usub):  # tilde '~'
            new_core = self.childs[0]
            self.replace_with_branch(new_core)

        for cc in self.childs:
            cc.workaround_remove_tilde()

    def finalize_set_nodepath(self, nodepath):
        """
        [0,2,1,0,0]
        ==>ROOT
        """
        self.nodepath = nodepath
        for ii, child in enumerate(self.childs):
            nodepath_child = nodepath + [ii]
            child.finalize_set_nodepath(nodepath_child)

    def finalize_set_depth(self, depth=0, recursive=True):
        """
        depth=0 is the root node
        """
        self.depth = depth
        max_depth = depth
        if recursive:
            for cc in self.childs:
                cc_depth = cc.finalize_set_depth(depth=depth + 1)
                max_depth = max(cc_depth, max_depth)

        self.childs_depth_max = max_depth

        return max_depth

    def check_all(self):
        # todo
        #   self.core.workaround_normalize_exponentiation()
        #   Check if a valid tree can be rebuilt from its expression
        #   each parameter in each node.
        #   The expression can include separate '~' (Usub) nodes, which makes expressions not completely equal
        #   ->self.workaround_remove_tilde()
        #   are we in the root_node?
        pass

    # def evolve_branch_reduce(self, obs_infos):
    #     # sfeh asdasdasd reduce me is obviously bullshit crapshit.
    #     #  sympify works with this combination only very few times
    #     #  lets have a new idea.
    #     expr_raw = self.get_nlabel(reducible=True, obs_names=obs_infos.keys())
    #     try:
    #         expr_sym = expr_sympify(expr_raw)
    #     except:
    #         raise Exception(f'Sympify failed. {expr_raw}')
    #
    #     replace_with_branch = [1, 2, 3]  # todo coolcore_from_expr(expr_sym, obs_infos)
    #     if len(replace_with_branch) < len(self):
    #         self.replace_with_branch(replace_with_branch)
    #     elif len(replace_with_branch) > len(self):
    #         raise Exception(
    #             f'Reduced core is even more complex than before  ({len(replace_with_branch)}, {len(self)}). expr_raw: {expr_raw}')  # \nold_core:{self}\nnew_node: {new_node} May happen with sympification and Usub.
    #         # example: Tree sympification did not work: Reduced core is even more complex than before. expr_raw: sign(Mini(((Velocity_2 * -0.790706) - sqrt(Gain_0)), (-0.569271 - Velocity_9)))
    #         # old_core:[sign, [Mini, [-, [*, Velocity_2, -0.790706], [sqrt, Gain_0]], [-, -0.569271, Velocity_9]]]
    #         # new_node: [sign, [Mini, [-, [Usub, [sqrt, Gain_0]], [*, 0.790706, Velocity_2]], [-, -Velocity_9, 0.569271]]]
    #     return


class Kernel:
    """
    The "abstract" Kernel class for the GP process.
    optional: creating another
    """

    # def __init__(self, *args):
    #     self.eval_action = None
    #     pass

    def fitness_compare(self, fitness1, fitness2):
        """
        if fitness_compared is None:
            return True
        else:
            return True/False, DEPEND
        """
        pass

    def relation(self, x, y):
        pass

    def best_fitness_function(self, *args, **kwargs):
        pass

    def tf_get_pairwise_fitness(self, *arg):
        pass

    def conclusion(self, *arg):
        pass


class RegressionKernel(Kernel):

    def fitness_compare(self, fitness, fitness_compared):
        """
        """
        if fitness_compared is None:
            return True
        else:
            return fitness < fitness_compared

    def best_fitness_function(self, *args, **kwargs):
        return min(*args, **kwargs)

    def relation(self, x, y):
        return lambda x, y: x < y

    def __gt__(self, other):
        """
        gt means better?
        """
        pass  # no, greater=better is ambiguous

    def __init__(self, kernel_name, data_train, tf_config, tf_device, eval_action):
        self.np_best_fitness = np.min

        # self.kernel_version_plot_yaxis = f"regression error"
        # for option, plot_axis_string in {'discrete': ', discrete', 'bounded': ', bounded', 'tanhpenalize': ', penalize (tanh)'}.items():
        #     if option in kernel_name:
        #         self.kernel_version_plot_yaxis += plot_axis_string

        self.kname = kernel_name
        self.tf_config = tf_config
        self.tf_device = tf_device
        self.eval_action = eval_action
        self.origin_results = None
        self.data_train = data_train  # sfeh where is the best?

        self.bounded = 'bounded' in kernel_name
        self.discrete = 'discrete' in kernel_name
        self.tanhpenalize = 'tanhpenalize' in kernel_name  # sfeh only makes sense when bounded

        self.MSE = 'MSE' in kernel_name
        self.RMSE = 'RMSE' in kernel_name
        self.MAE = 'MAE' in kernel_name

        self.exploration_risk = 'explun' in kernel_name
        self.origin_results = None  # can only be set after the evaluation of the origin...
        sfeh_help = {'explorate01': 0.1,
                     'explorate05': 0.5}

        self.pen_explorate = 0.1
        for k, v in sfeh_help.items():
            if k in kernel_name:
                self.pen_explorate = v
        return

    def pycode_wrap_result(self, action_min_max):
        wrap = '{}'

        if self.bounded:
            wrap = f'min(max({action_min_max[0]}, {wrap}), {action_min_max[1]})'

        if self.discrete:
            # regression that fits the outputs to a discrete set of actions defined by min and max
            wrap = f'int(math.round({wrap}))'

        return wrap

    def histogram_bins(self, action_minmax):
        act_min, act_max = action_minmax
        act_range = act_max - act_min
        if self.discrete:  # [0, 1, 2] -> 2
            # sfehfun make kernel histogram function?
            action_bins = np.linspace(-0.5 - act_range, 0.5 + act_range, 2 * act_range + 1 + 1)  # for +-0.5 and 0
        else:
            num_bins = 16 + 1  # +1 is extra bin for 0
            breite = 0.5 * (act_range * 2) / num_bins
            action_bins = np.linspace(-(breite + act_range), + (breite + act_range), num_bins + 1)  # sfeh 10 bins?
        return action_bins

    def eval_tf(self, expr_sym, used_observations, only_fitness=False):
        """
        Evaluates an expression using TensorFlow (TF)
        - receives a (string) expression in numpy-style that was reduced with pythons "sympy" (for simplification)
        - uses "ast" to generate a, kind of, python-intern-executable-tree
        - creating a tensorflow graph that is evaluated in an isolated TF session
        """
        tensorflow.compat.v1.reset_default_graph()
        solution = tensorflow.constant(self.data_train[self.eval_action.name])  # tensors[self.eval_action.name]
        tensors = {obs_name: tensorflow.constant(self.data_train[obs_name]) for obs_name in used_observations}  # do not assign dtype here, do this in the pandas df aka data

        results_agent = ast_convert_from_expr(expr_sym, tensors=tensors)  # the actual result from the expression in the agent

        # fit the agents to the possible outcome
        results_kernel = results_agent

        if self.discrete:
            results_kernel = tensorflow.math.round(results_kernel)
        if self.bounded:
            act_min = tensorflow.constant(self.eval_action.minmax[0], dtype=tensorflow.float32)
            act_max = tensorflow.constant(self.eval_action.minmax[1], dtype=tensorflow.float32)
            results_kernel = tensorflow.math.minimum(tensorflow.math.maximum(results_kernel, act_min), act_max)

        # pairwise_fitness = self.tf_get_pairwise_fitness(solution, kernel_result, results_agent)
        pairwise_diff = solution - results_kernel

        if self.MSE or self.RMSE:  # sfeh huber loss! mse, mae, rmse, huber, (log)
            # sfeh remove the fucking RMSE?^^
            tf_error = tensorflow.square
        else:
            tf_error = tensorflow.abs

        regression_errors = tf_error(pairwise_diff)
        # improved_errors = regression_errors  # sfeh not yet required... only one error

        if self.exploration_risk and self.origin_results is not None:
            # tf_error = tensorflow.abs  # sfeh this is required (??)
            # (1 * tensorflow.abs(pairwise_diff)) - # 1 * abs, as the other one is within the error. usually 2*  # sfeh not sure
            exploration_korridor = tensorflow.abs(solution - self.origin_results)  # the complete range that is 'okay' to actually explore here.
            exploration = (self.origin_results - results_kernel)  # the difference to the origin - which we want to "penalize" here
            explore_penalize = tensorflow.maximum(exploration_korridor - exploration, 0)  # removes the above mentioned expected exploration from the penalize process
            penalize_exploration = self.pen_explorate * (tf_error(explore_penalize))  # this should not be weighted as much as the regular expression (0 to 1).
            # Although, even more extreme penalisations are possible. Also, ideas about dummy pen (for no exploration, but no easy improvement) or values >1 for sticking to the origin policy
            # use factor before or after squaring the distance?
            regression_errors += penalize_exploration

            """
            sfeh idea: process is markov chain, but logic seems to correct until the first wrong decision.
            This point could be of large interest, as ir marks the moment where the good policy is lost.
            'correct' is not known, though. (MTC - yes, but IB may be very close)
            """
        else:
            penalize_exploration = tensorflow.no_op()

        mean_error = tensorflow.reduce_mean(regression_errors)
        if self.RMSE:
            mean_error = tensorflow.sqrt(mean_error)

        if self.tanhpenalize:
            """
            for the bounded kernel.
            Values, that are far too high, which get assigned to the action range, should be slightly punished.
            This should hopefully make improvements towards smaller numbers possible without affecting the parsimony.
            (e.g. results_agent = 33.6, but actionminmax[-1, 1] --> kernel_result = +1)

            tanh: closer to 0 is better, but rising steadyly without exceeding max value of 1 (outliers like single points inf become irrelevant)
            factor 1 (0.02) the amplitude. should be small enough to not significantly influence the gp process
            factor 2 (0.1) stretches the tanh function. the largest improvement should be at the points we want to get rid of
            squared distance? -> smooth transition from the area that is considered okay
            """
            penalized_bounds = 0.02 * tensorflow.tanh(tensorflow.square(results_agent - results_kernel) * 0.1)  # sfeh amplitude, stretch, squared
            mean_boundpen = tensorflow.reduce_mean(penalized_bounds)  # sfeh could easily be a reduce_sum
            mean_error += mean_boundpen
        else:
            penalized_bounds = tensorflow.no_op()

        with tensorflow.compat.v1.Session(config=self.tf_config) as sess:  # tensorflow evaluation must be done in a "session". funfact: debugging is not ez
            with sess.graph.device(self.tf_device):  # GPU evaluation in tensorflow
                tf_results = sess.run(
                    {'pairwise_diff': pairwise_diff, 'results_kernel': results_kernel, 'regression_errors': regression_errors, 'mean_error': mean_error, 'penalize_exploration': penalize_exploration})
                # sfeh attention: the dict above returns np-type results, not real floats
        if only_fitness:  # reduced evaluation, only mean_error is returned... (may save memory as only one value gets returned)
            return float(tf_results['mean_error'])
        else:
            return tf_results

    def tree_eval_fitness_offline_train(self, tree: Node):
        """
        Very fast eval-version that only computes fitness_train of the train data.
        tree_eval_complete gives more options
        Evaluating the fitness_train of a tree.
        - extract the expression the tree is holding
        - sympify the expression
        - (if sympify fails, evaluating does not make sense! Check sympify errors)
        - (sfeh: if sympify fails because of inf or zoo, tf could maybe still work due to save-tf-division)

        Returns bool value if we can use the calculated fitness_train
        Fitness values might evaluate to weird stuff
        e.g. 'nan' after dividing by zero or (inf) after 20**1234
        nan: fitness_train == fitness_train -> False
        inf: fitness_train is not float('inf') -> False
        """

        try:
            expr_raw = tree.eval_expr_sym()
            expr_sym = expr_sympify(expr_raw)
        except Exception as evalex:
            raise Exception(f'eval:{evalex}')

        used_observations = tree.get_observation_list()
        fitness_train = round(self.eval_tf(expr_sym, used_observations, only_fitness=True), self.conf.precision)

        if fitness_train != fitness_train or fitness_train == float('inf'):
            raise Exception(f"fitness_train is: '{fitness_train}'")  # happens, eg when values are soo wrong that it leaves the float-range

        return fitness_train

    def conclusion(self, result):
        """
        sfeh this is baad
        """
        # return f"\n\n Regression bounded fitness_train score: {result['fitness_train']}\n Mean Squared Error: {}"
        return


class ClassificationKernel(Kernel):

    def __init__(self, *args):
        pass

    def tf_classify_labels_map(self, result, env_vars):

        """
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the samples-csv.
        Outputs an array of tuples containing the predicted labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.uniques_num / 2) - 1 # '-1' keeps a binary classification splitting over the
            if solution == 0 and result <= 0 - skew; fitness_train = 1: # check for first class (the left-most bin)
            elif solution == self.uniques_num - 1 and result > solution - 1 - skew; fitness_train = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness_train = 1: # check for class bins between first and last
            else: fitness_train = 0 # no class match
        sfeh remove

        """
        uniques_num = env_vars.eval_action.uniques
        skew = (uniques_num / 2) - 1
        label_rules = {uniques_num - 1: (
            tensorflow.constant(uniques_num - 1), tensorflow.constant(f' > {uniques_num - 2 - skew}'))}

        for class_label in range(uniques_num - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tensorflow.cond(cond, lambda: (
                tensorflow.constant(class_label), tensorflow.constant(f' <= {class_label - skew}')), lambda: label_rules[class_label + 1])

        pred_label = tensorflow.cond(result <= 0 - skew, lambda: (tensorflow.constant(0), tensorflow.constant(f' <= {0 - skew}')), lambda: label_rules[1])

        return pred_label

    def fitness_compare(self, fitness1, fitness2):
        """
        todo: replace with pythonic way (__gt__ function above)
        """
        if fitness2 is None:
            return True
        else:
            return fitness1 > fitness2

    def best_fitness_function(self, *args, **kwargs):
        return max(*args, **kwargs)

    def tf_get_pairwise_fitness(self, solution, kernel_result, eval_action):
        """
        Calculates the kernel-specific fitness_train for the solution.
        - classification: dummy
        """

        skew = (eval_action.uniques / 2) - 1

        rule1 = tensorflow.logical_and(
            tensorflow.equal(solution, 0),
            tensorflow.less_equal(kernel_result, 0 - skew))

        rule2 = tensorflow.logical_and(
            tensorflow.equal(solution, eval_action.uniques - 1),
            tensorflow.greater(kernel_result, solution - 1 - skew))

        rule3 = tensorflow.logical_and(
            tensorflow.less(solution - 1 - skew, kernel_result),
            tensorflow.less_equal(kernel_result, solution - skew))

        pairwise_fitness = tensorflow.dtypes.cast(tensorflow.logical_or(tensorflow.logical_or(rule1, rule2), rule3), tensorflow.int32)
        return pairwise_fitness


class MatchKernel(Kernel):
    """
    The match kernel does
    """

    def __init__(self, *args):
        super().__init__(*args)

    def fitness_compare(self, fitness1, fitness2):
        """

        """
        if fitness2 is None:
            return True
        else:
            return fitness1 > fitness2

    def best_fitness_function(self, *args, **kwargs):
        """

        """
        return max(*args, **kwargs)

    def tf_get_pairwise_fitness(self, solution, kernel_result):
        """
        Calculates the kernel-specific fitness_train for the solution.
        - classification: dummy
        """
        """
        This is used for demonstration purposes only.
        """
        # pairwise_fitness = tensorflow.dtypes.cast(tensorflow.equal(solution, result), tensorflow.int32) # breaks due to floating points
        rtol, atol = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
        pairwise_fitness = tensorflow.dtypes.cast(tensorflow.less_equal(tensorflow.abs(solution - kernel_result), atol + rtol * tensorflow.abs(kernel_result)), tensorflow.int32)

        return pairwise_fitness

    def eval_tf(self):
        pass
        # if self.get_predicted_labels:
        #     predicted_labels = tensorflow.map_fn(self.tf_classify_labels_map, kernel_result, dtype=(tensorflow.int32, tensorflow.string), swap_memory=True)
        # else:
        #     predicted_labels = tensorflow.no_op()  # a placeholder, applies only to CLASSIFY kernel
        #  # , 'predicted_labels': predicted_labels    # predicted_labels

    # def conclusion_text(self, result, fitness_control_best):
    #     """
    #
    #     """
    #     elif self.kernel == 'regression':
    #         mse = skm.mean_squared_error(result['agent_result'], result['solution_goal'])
    #         result_str += ('\n\n Regression fitness_train score: {}'.format(result['fitness_train']))
    #         result_str += ('\n Mean Squared Error: {}'.format(mse))
    #
    #     result_str = ''
    #
    #     if self.kernel == 'classification':
    #         result_str += f'\n\n Classification fitness_train score: {fitness_control_best}'
    #         result_str += ('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution_goal'], result['predicted_labels'][0])))
    #         result_str += ('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution_goal'], result['predicted_labels'][0])))
    #
    #     elif self.kernel == 'regression bounded':
    #
    #     elif self.kernel == 'match':
    #         result_str += f"\n\n Matching fitness_train score: {result['fitness_train']}"
    #
    #     else:  # 'regression discrete':
    #         result_str = 'No summary provided for this kernel'
    #
    #     return result_str


def ast_convert_from_expr(expr, tensors=None, build=None):
    """
    Starts the recursive ast-analysis of the expression

    Extract expression tree from the string algo_sym.
    Please provide ONE of the following if you want to get...
    - tensorflow-graph: All variables (observation0, ...) as tensors.
    - build: True
    More information in ast_expr_to()

    """
    ast_tree = ast.parse(expr, mode='eval').body
    graph = ast_expr_to(ast_tree, tensors=tensors, build=build)

    if build:
        graph = labels_from_nestedexpr(graph, [])

    return graph


def labels_from_nestedexpr(labels_nested_list, result_accum):
    """
    Returns a label list from the nested list which ast_expr_to() created
    [+, [a], [/, [b, c]]]]  -> [+, a, /, b, c]
    """

    for x in labels_nested_list:  # all elements, that are not lists themselves
        if type(x) is not list:
            x = str(x)  # labels must be string!
            result_accum.append(x)

    only_lists = [x for x in labels_nested_list if (type(x) == list)]
    if only_lists:
        from itertools import chain
        lists_removed = list(chain(*only_lists))
        result_accum = labels_from_nestedexpr(lists_removed, result_accum)

    return result_accum


def ast_expr_to(node, tensors=None, build=None):
    """
    Returns (recursively) a (tensorflow) graph from a (raw or sympified) math expression.
    please use by calling labels_from_graphlist()

    Used to be for tensorflow only, but was modified to save 'sympified' trees.

    One of [tensors, prnt, build] must be set
    -> tensors: Creates a tensorflow graph for evaluation
    -> prnt: creates a string expression of the tree (I think I tried this before 'build' worked)
    -> build: creates a nested nlabel-list, e.g. a+(b/c) -> [+, [a], [/, [b, c]]]] (at least I think so)
    """

    # Arity 0
    if isinstance(node, ast.Name):  # <tensor_name>
        if build:
            return [node.id]
        else:
            return tensors[node.id]

    elif isinstance(node, ast.Num):  # <number>
        if build:
            return [node.n]
        else:
            # shape = tensors[list(tensors.keys())[0]].get_shape()
            return tensorflow.constant(node.n, dtype=tensorflow.float32)  # , shape=shape

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if build:
            return [node.value]
        else:
            return tensorflow.constant(node.value)
    #
    # Arity 1
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., sin(1), -1
        if build:
            if type(node.op) == ast.USub:  # workaround for ~-problem
                if isinstance(node.operand, ast.Name) or isinstance(node.operand, ast.Num) or isinstance(node.operand, ast.NameConstant):
                    return [f'-{ast_expr_to(node.operand, build=True)[0]}']
                else:
                    return ['Usub', [ast_expr_to(node.operand, build=True)]]
            return [op[type(node.op)]['fun_label'], [ast_expr_to(node.operand, build=True)]]
        else:
            return op[type(node.op)]['tf'](
                ast_expr_to(node.operand, tensors=tensors))

    # Arity 2
    elif isinstance(node, ast.BinOp) or isinstance(node, ast.BitAnd):  # <left> <operator> <right>, e.g., (x + y), (a & True)
        if build:
            return [op[type(node.op)]['fun_label'],
                    [ast_expr_to(node.left, build=True),
                     ast_expr_to(node.right, build=True)]]
        else:
            return op[type(node.op)]['tf'](
                ast_expr_to(node.left, tensors=tensors),
                ast_expr_to(node.right, tensors=tensors))

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        if build:
            return ast_chain_bool(node.values, op[type(node.op)]['fun_label'], build=True)
        else:
            return ast_chain_bool(node.values, op[type(node.op)]['tf'], tensors=tensors)

    elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
        if build:
            return ast_chain_compare([node.left] + node.comparators, node.ops, build=True)
        else:
            return ast_chain_compare([node.left] + node.comparators, node.ops, tensors=tensors)

    # Arity x, all custom functions
    elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)

        if node.func.id == 'Ifte':
            if build:
                return ['Ifte',
                        [ast_expr_to(node.args[0], build=True),
                         ast_expr_to(node.args[1], build=True),
                         ast_expr_to(node.args[2], build=True)]]
            else:
                return op[node.func.id]['tf'](tensorflow.dtypes.cast(
                    ast_expr_to(node.args[0], tensors=tensors), tensorflow.bool),
                    ast_expr_to(node.args[1], tensors=tensors),
                    ast_expr_to(node.args[2], tensors=tensors))

        elif len(node.args) <= 2:
            if build:
                if len(node.args) == 1:
                    return [op[node.func.id]['fun_label'],
                            [ast_expr_to(node.args[0], build=True)]]
                elif len(node.args) == 2:
                    return [op[node.func.id]['fun_label'],
                            [ast_expr_to(node.args[0], build=True),
                             ast_expr_to(node.args[1], build=True)]]
                else:
                    raise Exception('This arity is not supported')
            else:
                return op[node.func.id]['tf'](*[ast_expr_to(arg, tensors=tensors) for arg in node.args])

        else:
            raise Exception('Failed to identify the function. {}'.format(type(node)))
    else:
        raise TypeError('Node type could not be handeled in ast-evaluation: {}'.format(node))


def ast_chain_bool(values, operation, tensors=None, build=False):
    """
    Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.
        a & b
    --> values[0] operation values[1]
    """
    if build:
        x = ast_expr_to(values[0], build=True)
        if len(values) == 2:
            return [operation, [values[0], values[1]]]
        elif len(values) == 1:
            return x
        else:
            raise
    elif tensors:
        x = tensorflow.dtypes.cast(ast_expr_to(values[0], tensors=tensors), tensorflow.bool)
        if len(values) > 1:
            return operation(x, ast_chain_bool(values[1:], operation, tensors=tensors))
        else:
            return x


def ast_chain_compare(comparators, ops, tensors=None, build=False):
    """
    Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

    """

    x = ast_expr_to(comparators[0], tensors=tensors, build=build)
    y = ast_expr_to(comparators[1], tensors=tensors, build=build)

    if len(comparators) > 2:
        print_e('This is usually not used, and-concatenation of multiple chain compares')
        return tensorflow.logical_and(op[type(ops[0])]['tf'](x, y), ast_chain_compare(comparators[1:], ops[1:], tensors=tensors))
    else:
        if build:
            return [op[type(ops[0])]['fun_label'], [x, y]]
        else:
            return op[type(ops[0])]['tf'](x, y)


class ParetoFront:
    """
    sfeh
    """

    def __init__(self, kernel, origin=None):
        self.pareto = []
        self.kernel = kernel
        self.origin = origin

    def pareto_append(self, tree: Node, msg=None, print_type=None):
        """
        Appending a candidate to the paretofront.
        - append the entry to the paretofront
        - reset gens_since_last_pareto
        - try to add the tree in its sympified version
        """
        if msg:
            printez('a', f"New entry found! ({msg}): {BColors.RESET}{tree.get_parsimony()}, {tree.get_fitness()}:{BColors.RESET} {tree.get_expr_raw()}", print_type=print_type)
        self.pareto.append(tree)
        self.gens_since_last_pareto = 0

        par = tree.get_parsimon()
        fit = tree.get_fitness()

        self.pareto = [x for x in self.pareto[:] if x.get_parsimony() < par or x.get_fitness() < fit or (par == par and x[1] == par)]
        self.pareto_sort()  # as far as I can tell, not really necessary without using iter()

        # tree_sym = copy.deepcopy(tree)
        # # todo
        # try:
        #     # printez('aaa', 'Trying to simplify for pareto entry.')  # simplify the tree and save in pareto once again
        #     tree_sym.evolve_reduce(obs_infos=obs_infos, completely=True)
        #     parsimony = tree_sym.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
        #     if parsimony < tree.meta.parsimony:
        #         # self.printpl('aa', 'Successfully reduced pareto tree!')
        #         sym_fitness = self.tree_eval_fitness_offline_train(tree_sym)  # sfeh actually not required, delete this
        #         tree_sym.meta.fitness_train = sym_fitness
        #         tree_sym.meta.parsimony = parsimony
        #         self.update_pareto_with_tree(tree_sym)
        # except Exception as ex:
        #     print_warning('www', f'Tree sympification did not work: {ex}', print_type=print_type)
        #
        # else:
        #     printez('aaa', 'Pareto entry was already simplified', print_type=print_type)

    def append_repair(self, tree: Node):
        self.gens_since_last_pareto = 0
        self.pareto.append(tree)
        self.pareto = self.pop_to_pareto(self.pareto)

    def random_choice(self):
        return np.random.choice(self.pareto)

    def pareto_sort(self):
        """
        sorting the pareto pareto in list for parsimony
        """
        self.pareto.sort(key=lambda x: x[0])

    def pareto_txt(self):
        """
        Save all the pareto candidates to a file.
        (Quick feedback that requires little overhead)
        """
        return [f'Parsimony: \t{parsim} MeanError: \t{fitness} Expr: \t{tree.meta.expr_raw}' for (parsim, fitness, tree) in self.pareto]

    def update_pareto_with_tree(self, tree: Node):
        """
        inserts a tree into the pareto front
        """
        parsim = tree.get_parsimony()
        fit = tree.get_fitness()

        p_simpler = [p for p in self.pareto if p.get_parsimony() <= parsim]  # all pareto pareto that are less complex

        if len(p_simpler) == 0:  # all other pareto pareto are more complex
            self.pareto_append(tree, msg=f'new simplest entry')
        else:
            best = min(p_simpler, key=lambda p: p[1])  # the fittest of the less complex ones
            if tree.get_fitness() < best.get_fitness():
                self.kernel.fitness_compare(tree.get_fitness(), best.get_fitness())  # if true, at least one insertion
                self.pareto_append(tree, msg=f'old fitness: {best[1]}')

        self.pareto_sort()  # sfeh check if required
        return

    def pop_to_pareto(self, poplike):
        """
        inserts a tree into the pareto front
        """

        poplike = sorted(poplike, key=lambda x: (x.get_parsimony(), x.get_fitness()))  # todo -minus fitness
        best = poplike[0]
        best_par = best.get_parsimony()
        best_fit = best.get_fitness()
        pop_pareto = [best]

        for tree in poplike:
            parsim = tree.get_parsimony()
            if parsim == best_par:
                continue
            else:
                fitness = tree.get_fitness()
                if self.kernel.fitness_compare(fitness, best_fit):
                    pop_pareto.append(tree)
                    best_par = parsim
                    best_fit = fitness

        print(f'The pareto-efficient candidates of the population are: {pop_pareto}')

        return pop_pareto

    def pareto_from_population(self, pop):
        """

        """
        pop_pareto = self.pop_to_pareto(pop)

        if len(self.pareto) == 0:
            print('ASD Paretofrunt wird angefangen!!')  # todo debug paretofront wurde erstellt
            self.pareto.append(pop[0])

        cmp = lambda a, b: a < b  # todo

        for tree in pop_pareto:
            success = False
            fit = tree.get_fitness()
            par = tree.get_parsimony()
            if all([par < p.get_parsimony() for p in self.pareto]):
                print('Oye found a new, simpler entry')
                self.pareto.append(tree)
            for p in self.pareto:
                if cmp(fit, p.get_fitness()) and par <= p.get_parsimony():
                    self.pareto.remove(p)
                    success = True
            if success:
                self.append_repair(tree)

        # pop_iter = iter(pop_pareto)
        # pareto_iter = iter(self.pareto)
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
        #         self.pareto.append(t)  # todo (every entry in this list is already paretoefficient)
        #         t = next(pop_iter)
        #
        #     t_fit = t.get_fitness()
        #     while not cmp(t_fit, p.get_fitness()):
        #         print(f'removing the deprecated entries: Fitness: {p.get_fitness()}')
        #         self.pareto.remove(p)
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
        #         print(f'SHEESH! We kicked out another pareto entry! old: {p_fit}, new fitness: {t_fit}.')
        #         self.pareto.remove(p)
        #         self.pareto.append(t)
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
        # # poplike = sorted(poplike, key=lambda x: (x.get_parsimony(), x.get_fitness()))
        # # pop_iter = iter(poplike)
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
        # #                 pop_pareto.append(tree)
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
    #     # pycode_agent = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')
    #
    #     pygents_list = []
    #
    #     for (parsim, fitness_train, tree) in self.pareto:
    #         # agent_name = f'{self.conf.name}_{parsim:.0f}'
    #         agent_name = f'{self.conf.name}_{self.env_vars.eval_action.name}_{parsim:.0f}'
    #         agent_as_python = tree.eval_pycode()
    #         pygents_list.append([parsim, float(fitness_train), agent_name, agent_as_python])
    #
    #     yaml_dump(self.rootdir / 'pycode_list.yaml', pygents_list, print_type=self.print_type)
    #     path = path_make_dir(self.rootdir / 'pycode_list.yaml')
    #     with Path.open(path, 'w') as file:
    #         _ = yaml.dump(pygents_list, file)  # , default_flow_style=False, sort_keys=False)
    #         printez('ff', f'IB pycode-list: {path.as_posix()}', print_type=self.print_type)  # sfeh always the same print structure... just pass the path?
    #
    #     return
    #
    # def analyse_pareto(self, cpu_cores=16):
    #     """
    #     Writing all analysis files after evaluating the paretofront.
    #     (Currently strongly customized by sfeh for the mountaincar and industrial benchmark)
    #     """
    #     self.printpl('i', f'Analysing the pareto candidates of the run.')
    #
    #     dir_benchmarks = Path(__file__).parent.parent.absolute() / 'benchmarks/'
    #     path_hist = path_make_dir(self.rootdir / 'histograms/')
    #     pareto_agents = {}
    #
    #     for (parsim, fitness_train, tree) in self.pareto:
    #         histograms_path = self.plot_agent_histogram(parsim, tree, path_hist)  # sfeh todo
    #
    #         forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest = self.file_pareto_latex(parsim, tree)
    #
    #         pareto_agents[parsim] = {'parsim': parsim,
    #                                  'fitness_train': fitness_train,
    #                                  'tree': tree,
    #                                  'forest_tree_full': forest_tree_full, 'forest_tree_tight': forest_tree_tight, 'tex_expr_raw': tex_expr_raw, 'tex_expr_forest': tex_expr_forest,
    #                                  'histogram': histograms_path,
    #                                  }
    #
    #     tex_include_pdf = lambda x: f"\\includegraphics{{{str(x).replace('.pdf', '')}}}"
    #     tex_tabuline = lambda x: f"{' & '.join(x)}\\tabularnewline\n"
    #     tex_stacklist = lambda x: "\\shortstack[l]{{{}}}".format('\\\\'.join([str(xx) for xx in x]))
    #
    #     if 'MTC' in self.conf.name:
    #         """
    #         complexity, regr. error, real evaluation, decision plot, spiral plot, diff-plot
    #         """
    #         # sfeh i guess not necessary anymore (?)
    #         # self.file_pareto_pycode()
    #         sarsa_agent_steps = 200 if 'MTC200' in self.conf.name else 75 if 'MTC75' in self.conf.name else 'NO_MC_AGENT'
    #         sarsa_agent = pickle_load(dir_benchmarks / f'mc/agents/sarsa_agent_{sarsa_agent_steps}.p')
    #
    #         agent_performance = auto_evaluate_run_end(self.rootdir, sarsa_agent, n=100)
    #         # df_mtc = pd.DataFrame(columns=['complexity', 'regr. error', 'avg. reward', 'fails', 'expression'])
    #         tex_lines = []
    #
    #         for pp, x in agent_performance.items():
    #             pp, fitness_train, avg_reward, fails, path_mcmeshplot, path_mcmeshplot_diff = x
    #
    #             tex_lines.append([f"{int(pp)}",
    #                               f"{pareto_agents[pp]['fitness_train']:.2f}",
    #                               f"{avg_reward:0.1f}",
    #                               f"{pareto_agents[pp]['tex_expr_raw']}",
    #                               f"{tex_include_pdf(f'sfehs_eval/{pp}.pdf')}",  # path_mcmeshplot
    #                               # tex_include_pdf(f'sfehs_eval/diff-{pp}.pdf'),  # path_mcmeshplot_diff),
    #                               tex_include_pdf(f'sfehs_eval/space-{pp}.pdf'),
    #                               # tex_include_pdf(f'histograms/acthist_{pp}'),
    #                               f"{pareto_agents[pp]['forest_tree_full']}",  # forest_tree_full, forest_tree_tight
    #                               f"{pareto_agents[pp]['forest_tree_tight']}",
    #                               f'{fails}'])
    #
    #         paste = ''.join([tex_tabuline(x[:4]) for x in tex_lines])
    #         paste = "\\begin{longtable}[c]{>{\\LTleft}p{5mm}>{\\LTleft}p{6mm}>{\\LTleft}p{10mm}>{\\LTleft}p{102mm}}\n\\hline\n" \
    #                 "dist & error & reward  & expression \\tabularnewline \\hline\n" \
    #                 f"{paste}" \
    #                 "\\hline\n\\end{longtable}\n"
    #         file_dump(self.rootdir / f'analysis_input.tex', paste, print_type=self.print_type)
    #         file_dump(self.rootdir / f'analysis_overview.tex', latex_treeviz_full_document(paste), print_type=self.print_type)
    #
    #         paste_full = ''.join([tex_tabuline(x[:]) for x in tex_lines])
    #         paste_full = f"{str(self.conf.name).replace('_', '-')}  {tex_include_pdf('monitoring.png')}  {tex_include_pdf('sfehs_eval/evaled_overview.pdf')}\n\n" \
    #                      "\\begin{tabular}{lllllllllllll}\n \\hline\n" \
    #                      "dist & error & reward & parsimony & expression \\tabularnewline \\hline\n" \
    #                      f"{paste_full}" \
    #                      "\\hline\n\\end{tabular}\n\n"
    #         file_dump(self.rootdir / f'analysis_overview_plus.tex', latex_treeviz_full_document(paste_full), print_type=self.print_type)
    #
    #     elif 'IB' in self.conf.name:
    #
    #         self.file_pareto_listcode()
    #
    #         if self.conf.name[-2:] == '_0':
    #             """
    #             - complexity_sum, [complexities], regression_sum, [regression_errors], real_sum, [formulas]
    #             - plot with pareto candidates
    #             """
    #             # "\\multicolumn{2}{c}{complexity} & \\multicolumn{2}{c}{regression error} & \\multicolumn{4}{c}{IB reward} & combinations & expression \\tabularnewline\n" \
    #             tex_line_input, tex_line_overview = '', ''
    #
    #             res_all = combined_lists(self.rootdir.parent, 40, 40, local_yamls=True, cpu_cores=cpu_cores)  # sfeh use self.conf.mp_cores
    #
    #             xx = [x['parsim_sum'] for x in res_all]
    #             y_all = [y['experiment'] for y in res_all]
    #             y_safe = [y['experiment_safe'] for y in res_all]
    #             y_all_r50 = [y['experiment_r50'] for y in res_all]
    #             y_safe_r50 = [y['experiment_safe_r50'] for y in res_all]
    #             cnt = [y['cnt'] for y in res_all]
    #
    #             for y in res_all:
    #                 parsims = y['parsims']
    #
    #                 input_agentex = lambda run_ii, prepth: f"\\input{{{prepth}{self.rootdir.parent.name}_{run_ii}/visualisation/{int(parsims[run_ii]):02d}_input.tex}}"  # _forest
    #
    #                 tex_line_overview += tex_tabuline([f"{int(y['parsim_sum'])}",
    #                                                    f"{y['regress_sum']:0.3f}",
    #                                                    f"{y['experiment']:0.0f}",
    #                                                    tex_stacklist([f'{int(x)}' for x in y['parsims']]),
    #                                                    tex_stacklist([input_agentex(x, '') for x in [0, 1, 2]])])
    #
    #                 tex_line_input += tex_tabuline([f"{int(y['parsim_sum'])}",
    #                                                 f"{y['regress_sum']:0.3f}",
    #                                                 f"{y['experiment']:0.0f}",
    #                                                 tex_stacklist([f'{int(x)}' for x in y['parsims']]),
    #                                                 tex_stacklist([input_agentex(x, f'../benchmarks/{self.rootdir.parent.parent.name}/{self.rootdir.parent.name}/') for x in [0, 1, 2]])])
    #
    #                 tex_line_input += tex_tabuline([f"{int(y['parsim_sum'])}",
    #                                                 f"{y['regress_sum']:0.3f}",
    #                                                 f"{y['experiment']:0.0f}",
    #                                                 tex_stacklist([f'{int(x)}' for x in y['parsims']]),
    #                                                 tex_stacklist([input_agentex(x, f'../benchmarks/{self.rootdir.parent.parent.name}/{self.rootdir.parent.name}/') for x in [0, 1, 2]])])
    #
    #             combined_overview = "\\begin{tabular}{llllllllll}\n\\hline \n" \
    #                                 f"{tex_tabuline(['dist', 'error', 'reward', 'dist', 'Agent code'])} \\hline\n" \
    #                                 f"{tex_line_overview}" \
    #                                 f"\\hline\n\\end{{tabular}}\n\n"
    #
    #             combined_input = "\\begin{longtable}[c]{>{\\centering}p{10mm}>{\\centering}p{10mm}>{\\centering}p{12mm}>{\\centering}p{12mm}>{\\centering}p{90mm}} \\hline\n" \
    #                              f"{tex_tabuline(['dist', 'error', 'reward', 'parsimony', 'expressions'])}" \
    #                              f"{tex_line_input}" \
    #                              "\\hline\n\\end{longtable}\n"
    #
    #             combined_fulltrees = "\\begin{longtable}[c]{>{\\centering}p{10mm}>{\\centering}p{10mm}>{\\centering}p{12mm}>{\\centering}p{12mm}>{\\centering}p{90mm}} \\hline\n" \
    #                                  f"{tex_tabuline(['dist', 'error', 'reward', 'parsimony', 'expressions'])}" \
    #                                  f"{tex_line_input}" \
    #                                  "\\hline\n\\end{longtable}\n"
    #
    #             combined_overview = latex_treeviz_full_document(combined_overview)
    #             file_dump(self.rootdir.parent / 'combined_overview.tex', combined_overview)
    #             file_dump(self.rootdir.parent / 'combined_input.tex', combined_input)
    #
    #             with plt.rc_context(rc=pyplot_rc_tex):
    #                 fig, ax = plt.subplots()
    #                 ax.set(xlabel='Pareto complexity sum', ylabel='reward [x1000]', ylim=funny_limits)
    #                 ax.plot(xx, y_all, label='average', marker='.', color='r')
    #                 ax.plot(xx, y_safe, marker='None', color='r', linestyle='dotted')  # , label='low risk'
    #                 ax.plot(xx, y_all_r50, label='randomized start', marker='.', color='b')
    #                 ax.plot(xx, y_safe_r50, marker='None', color='b', linestyle='dotted')  # , label='low risk (randomized)'
    #                 ax.legend(loc='lower right')
    #                 plt.yticks(IB_YICKS[0], IB_YICKS[1])
    #
    #                 # drawing the regression error, but plots seem to be too overloaded
    #                 # axx = ax.twinx()
    #                 # axx.plot(xx, cnt, color='tab:gray', label='regression error', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
    #                 # axx.tick_params(axis='y', labelcolor='tab:gray')
    #                 # # axx.plot(xx, y['regression_sum'])
    #                 #
    #                 # ax2 = ax.twinx()
    #                 # ax2.plot(xx, cnt, color='tab:gray', label='combos', linestyle='dashed', marker='.')  # linestyle='None'  # , legend_loc='best'
    #                 # # ax2.set(ylabel='possible combinations', color='tab:gray')
    #                 # ax2.tick_params(axis='y', labelcolor='tab:gray')
    #                 # ax2.legend(loc='lower left')
    #
    #                 fig.savefig(self.rootdir.parent / f'regression_all.pdf')
    #                 plt.close('all')
    #
    #     else:
    #         raise Exception(f'This should actually never happen right now. name: {self.conf.name}')
    #
    #     return
    #
    # def file_pareto_latex(self, parsim, tree):
    #     """
    #     Generates latex-file with the computational tree structure of all pareto agents
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
    #     file_dump(path_subfolder_tex / f'full_{parsim:02d}.tex', forest_tree_full, verbose='ff', print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'full_{parsim:02d}_tight.tex', forest_tree_tight, verbose='ff', print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_input.tex', tex_expr_raw, print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_input_forest.tex', tex_expr_forest, verbose='ff', print_type=self.print_type)
    #     file_dump(path_subfolder_tex / f'{parsim:02d}_doc.tex', latex_treeviz_full_document(forest_tree_full), verbose='ff', print_type=self.print_type)  # delete this
    #
    #     return forest_tree_full, forest_tree_tight, tex_expr_raw, tex_expr_forest


class Population:

    def __init__(self, kernel: Kernel, origin=None, conf=None):
        self.kernel = kernel
        self.size = 1000
        self.pop_next = []
        self.pop_base = []  # sfeh maybe better names

        self.conf = conf  # todo

        self.lut = {}
        self.origin = origin

    def selection_tournament(self, tourn_size=3):
        """

        """
        tournament_list = [np.random.choice(self.pop_base) for _ in range(tourn_size)]
        tourn_winner = self.kernel.best_fitness_function(tournament_list, key=lambda tree: tree.meta.fitness_train)
        tourn_winner = copy.deepcopy(tourn_winner)
        return tourn_winner

    def finalize(self, tree: Node):
        """
        ==>ROOT (only in root node)
        finalizing the structure
        """

        # try:
        #     self.check_all()
        # except Exception as ex:
        #     logging.warning(f'tree failed the quick check. last-mod: {self.meta.last_evolution}. Reason:\n{ex}')
        #     # print_warning('w', f'tree failed the quick check. last-mod: {self.meta.last_evolution}. Reason:\n{ex}', print_type=self.print_type)

        parsimony = None

        expr_raw = tree.eval_expr_sym()
        expr_sym = expr_sympify(expr_raw)

        # self.meta.expr_raw = self.get_nlabel()  # todo debug
        # self.meta.expr_sym = 'self.eval_expr_sym()'  # todo debug

        # todo evaluate parsimony
        # todo evaluate fitness_train

        # todo: set meta data
        pass

    def append(self, tree: Node):
        """
        was: def pop_append(self, tree: Node):
            # sfeh this check might be important...
        Safely append a tree to the population.
        Even though the raw trees should have everything to display their expression,
        they have gone through a process of changes. Here, the tree is refurbished.
        sfeh: if trees are 100% safely created, tree_check_deep() must not be used. Useful when trying out new gp-operators.
        - Enrich the raw tree for the next generation
        - check if the tree is actually valid
        ->
        """

        meta = self.lut.get(hash(tree), None)  # sfeh str() might even maje more sense? maybe not? idk
        if meta is not None:
            parsimony = meta['parsimony']
            fitness = round(meta['fitness_train'], self.conf.precision)  # sfeh just for now
        else:
            parsimony = tree.eval_parsimony(self.conf.complexity_measure, origin_tree=self.origin)
            if parsimony > self.conf.parsimony_max:
                print_warning('wwww', f'Parsimony too high, last evolution: {tree.meta.last_evolution}', print_type=self.print_type)  # sfeh care about wwww. should not
                return
            try:
                fitness = self.tree_eval_fitness_offline_train(tree)
            except Exception as evalex:
                print_warning('wwww', f'Exception while evaluating: {evalex}, tree: {tree}.', print_type=self.print_type)
                return

        expr_raw = tree.get_nlabel()
        expr_sym = expr_sympify(expr_raw)
        # set fixed nodes

        tree.meta.fitness_train = fitness
        tree.meta.parsimony = parsimony
        tree.meta.expr_raw = expr_raw
        tree.meta.expr_sym = expr_sym
        # tree.set_meta(fitness_train, parsimony, last_evolution, expr_raw, expr_sym)

        self.treelut_tree_add(tree)
        self.pop_next.append(tree)


class ExplainableGP(object):
    """

    """

    def __init__(self, conf: GpConfig, rootdir: Path, path_data, path_origin_tree, mp_cores=1):
        self.conf = conf
        self.mp_cores = mp_cores  # sfeh
        self.time_start = time.perf_counter()

        self.io = UserInteraction(Path(rootdir), self.conf.print_type)
        self.io.print_initializing()

        with Path.open(path_data) as file:
            # self.env_vars, choose_observations = data_from_csv(df, action_name=self.conf.action_name)
            df = pd.read_csv(file, delimiter=',')
            df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P Following design pattern #YOLO
            # todo it is float64, float64, int64 with MTC.. does it work with Tensorflow?

            for ii, header in enumerate(df):
                if header in op:
                    raise Exception(f'Your samples hold a column that matches the potential tree operator {header}.\n'
                                    f'That might end up in confusion, please rename the column.')

            self.conf.action_name = self.conf.action_name or df[len(df.columns) - 1]

            df = df.drop(self.conf.dc, axis=1)  # no need to keep other actions
            printez('i', f'Ignoring columns: {self.conf.dc}')  # , print_type=print_type print_type does not exist yet sfeh
            csv_observations = list(df.columns)
            csv_observations.remove(self.conf.action_name)
            choose_observations = ChooseObservation(csv_observations)

            self.data_train, self.data_control = train_test_split(df, test_size=0.2, random_state=0)  # discussion: random state 0 okay? test_size 0.2?
            kernel = RegressionKernel(self.conf.kernel_name, self.data_train, self.tf_config, "/gpu:0",
                                      self.conf.action_name)  # sfeh Set TF computation backend device (CPU/GPU); gpu:n = 1st, 2nd, or ... GPU device. Is cpu otherwise

        choose_distributions = self.activate_distributions(path_distrib=None)  # asd sfeh path_distrib not None
        choose_ops = self.gp_load_oparray(path_operators=None)  # path_operators sfeh this file from config version1

        self.tb = TreeBuilder(choose_ops, choose_observations, choose_distributions, self.conf.precision)

        # Evaluating kernel (that uses tensorflow)
        self.tf_config = tensorflow.compat.v1.ConfigProto(log_device_placement=self.conf.tf_device_log,
                                                          allow_soft_placement=True)  # TF device usage logging (for debugging) (default false. I lately used it to check if the GPU is used)
        self.tf_config.gpu_options.allow_growth = True

        self.pareto = ParetoFront()  # todo  a list with all pareto candidates. key is complexity, value is tree meta. [[1,344, meta], ...]

        if path_origin_tree:
            # todo make origin class
            self.origin = self.load_origin_tree(path_origin_tree)
            self.origin.meta.last_evolution = 'origin'
            self.origin_is_fix = self.origin.core.is_fix
        else:
            self.origin = None
            self.origin_is_fix = False

            if self.conf.complexity_measure in ['tree_edit_distance']:
                self.conf.complexity_measure = 'tree_node_count'  # sfeh idea
                self.io.print_warning('w', "Complexity measurement 'tree_edit_distance' is not possible without origin!\n"
                                           "Using 'tree_node_count' instead.", print_type=self.print_type)

        # init values with dummies (just to have all self values here for overview)
        self.tree_lut = {}  # LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
        self.population = []
        self.pop_base = []  # population that is taken to the next generation
        self.best_fitness = None  # keeps track of the current best fitness_train
        self.gen_id = 0

        self.monitor_df = pd.DataFrame(columns=['pop_len', 'pop_unique', 'time',
                                                'fit_avg', 'fit_var', 'fit_best',
                                                'complexity_avg', 'complexity_var', 'complexity_stderr',
                                                'gens_since_last_pareto'])

        self.evolve_loop, self.evolve_random = self.make_evolve_rates()
        self.evolve_tags = list(self.evolve_loop.keys()) + list(self.evolve_random.keys())

        # self.monitor_evol = dict.fromkeys(self.evolve_tags, pd.DataFrame(columns=['fitness_train', 'parsimony', 'lentree', 'evolve_num', 'count']))

        self.pop = Population(kernel, origin=self.origin)
        self.kernel = kernel  # sfeh remove

        self.io.print_g('gg', f'Init. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return

    def gen_create_initial(self):
        """
        was: gen_create_initial
        Everything that needs to be custom_done for the first generation
        - Extracts "origin_meta Tree" from file
        - Creates all other trees: origin_meta tree + branch mutation
        - Evaluate the first Generation
        - Monitoring initialisation and monitoring
        """

        if self.origin is not None:
            self.pop.append(self.origin)  # sfeh why not :P
        else:
            total_rate = sum([x['evolve_rate'] for x in self.evolve_random.values()])

            for tag, evolve_specs in self.evolve_random.items():
                evolve_num = int(self.conf.pop_max * (evolve_specs['evolve_rate'] / total_rate))
                call_params = evolve_specs.get('custom_params')

                for nn in range(evolve_num):
                    if self.origin_is_fix:
                        tree = self.tb.pop_random(call_params, from_origin=True)
                    else:
                        tree = self.tb.pop_random(call_params)
                    # tree.meta.last_evolution = tag  # todo
                    self.pop.append(tree)
        return

    def gen_next_population(self):
        """
        Creates all new Generations by applying the evolutions in the evolve-loop.

        brainstorm:
        - the tree can be reproduced (selection), random/new, olymp-reproduction
        - (1 tree) mutations can affect a point, branch, terminal nodes
        - (2 trees) can make a crossover
        """
        # All gp creators: name, function, num of trees from tournament selection

        for tag, evolve_specs in self.evolve_loop.items():  # all selected gp mutations

            evolve_name = evolve_specs['evolve_name']
            evolve_num = evolve_specs['evolve_num']
            tourn_size = evolve_specs['tourn_size']
            call_params = evolve_specs.get('custom_params')

            self.io.print_g('gggg', f'->Evolving \'{tag}\' {evolve_num}x starting...')
            if evolve_name == 'reproduce':
                """
                
                """
                for nn in range(evolve_num):
                    tree = self.pop.selection_tournament(tourn_size=tourn_size)
                    if call_params.get('simplify'):
                        try:
                            pass
                            # self.tb.evolve_reduce(obs_infos=self.env_vars.obs_infos, completely=False)
                            # tree.meta.last_evolution = tag
                        except Exception as ex:
                            self.io.print_warning('www', f'Evolve reproduce failed: {ex}', print_type=self.print_type)

                    self.pop.append(tree)  # append anyways... it was worth a try :P

            elif evolve_name == 'mutate point':
                """
                Point mutation, One point (terminal or function) gets mutated.
                """
                for nn in range(evolve_num):
                    tree = self.pop.selection_tournament(tourn_size=tourn_size)
                    tree = self.tb.evolve_mutate_point(tree)
                    # tree.meta.last_evolution = tag
                    self.pop.append(tree)

            elif evolve_name == 'mutate branch':
                # todo question: tree.copy() or .deepcopy necessary??
                for nn in range(evolve_num):
                    build_spec, size_mode, mean_min_max_var, full_or_grow = self.helper_evolve_params_branch(call_params)
                    full_or_grow = build_spec.get('p_op') or random.choice(['full', 'grow'])
                    tree = self.pop.selection_tournament(tourn_size=tourn_size)
                    cool_build_size = choose_build_size(size_mode, mean_min_max_var, tree=tree)
                    tree.evolve_mutate_branch_depth(cool_build_size, self.tb, size_mode=size_mode, full_or_grow=full_or_grow)
                    # new_tree.meta.last_evolution = tag
                    self.pop.append(tree)

            elif evolve_name == 'crossover branch':
                for nn in range(int(evolve_num / 2)):  # two childs
                    atree = self.pop.selection_tournament(tourn_size=tourn_size)
                    btree = self.pop.selection_tournament(tourn_size=tourn_size)

                    """
                    swap branches of two trees
                    - select parent a and b
                    - select swappable branche for a_parent from b_parent
                        - select a node in a (and crossover here, no matter what)
                    - delete a_parent branch and insert b_parent branch (which tactic?)
                    todo into main tree?
                    """
                    # 1. two parents
                    # 2. search nodes for left and right that can be exchanged. convert_needed
                    atree, btree = self.tb.evolve_crossover(atree, btree)
                    # tree = self.tb.finalize(etree)  # todo ==>state
                    self.pop.append(atree)
                    self.pop.append(btree)

            elif evolve_name == 'filter optimize':

                for nn in range(evolve_num):
                    tree = self.pop.selection_tournament(tourn_size=tourn_size)
                    tree.evolve_mutate_filter_random(call_params, self.tb)
                    tree.meta.last_evolution = tag
                    self.pop.append(tree)

            elif evolve_name == 'revive pareto':

                for nn in range(evolve_num):
                    fitness_train, parsim, tree = self.pareto.random_choice()
                    tree.meta.last_evolution = tag
                    self.pop.append(tree)

            elif evolve_name == 'random trees':

                if self.origin_is_fix:
                    for nn in range(evolve_num):
                        tree = self.tb.pop_random(call_params, from_origin=True)
                        tree.meta.last_evolution = tag
                        self.pop.append(tree)
                else:
                    for nn in range(evolve_num):
                        tree = self.tb.pop_random(call_params)
                        tree.meta.last_evolution = tag
                        self.pop.append(tree)
            else:
                print_e(f"Evolution not known: '{evolve_name}'")

        missing_trees = self.conf.pop_max - len(self.population)
        if missing_trees > 0:
            if missing_trees > 0.05 * self.conf.pop_max:
                self.printpl('ii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            else:
                self.printpl('iii', f'{missing_trees}/{self.conf.pop_max} trees are missing in this population!')
            while len(self.population) < self.conf.pop_max:
                return  # sfeh aka create trees here if desired

        # sfeh automatically fill with random trees (check this at the initiation)
        # total_rate = sum([x['evolve_rate'] for x in self.evolve_list.values()])
        # if total_rate < 0:
        #     self.gen_create_random(int(self.conf.pop_max'] * (1 - total_rate)))

        return

    def activate_distributions(self, path_distrib=None):
        """
        Optional custom distributions specified by the user.
        """
        path_distrib = path_distrib or self.rootdir / self.io.use_distributions_file
        choose_distributions = ChooseConstants(path_distrib, data_train=self.data_train, n_samples=100)
        return choose_distributions

    def make_evolve_rates(self):
        """
        The evolve_dict is converted into a list of the population length.
        The evolve_loop is for the regular evolution, the evolve_random is especially required for the first generation.
        """

        def evolve_safety_update(evolve_dict):
            """
            Updates tournament size and evolve rates

            Example entry of the list could be:
            {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
            'custom_params': {'build_max': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
            """

            for tag, evolve_spec in evolve_dict.items():
                evolve_dict[tag]['tourn_size'] = evolve_spec.get('tourn_size', self.conf.tourn_size)
                evolve_dict[tag]['evolve_num'] = int(evolve_spec.get('evolve_rate') * self.conf.pop_max)
                evolve_spec['custom_params'] = evolve_spec.get('custom_params', {})
            return evolve_dict

        # evolve_loop = self.evolve_list  # todo
        # self.printpl('i', 'Using evolve rates from config')

        class Evolution:
            # todo rate not in here
            def __init__(self, id=None, evolution=None, rate=None, params=None, custom_params=None):
                self.id = id
                self.evolution = evolution
                self.params = params or {}
                self.rate = rate
                self.custom_params = custom_params

        evolve_loop = {
            # Reproduction (10%)
            'Repro': {'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.06, 'custom_params': {}},
            'Rsympy': {'evolve_name': 'reproduce', 'evolve_rate': 0.03, 'custom_params': {'simplify': True}},
            'Pareto': {'evolve_name': 'revive pareto', 'evolve_rate': 0.01, 'custom_params': {}},

            # Mutation (25%)
            'Point': {'evolve_name': 'mutate point', 'evolve_rate': 0.05, 'custom_params': {}},

            'BranchDF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                         'custom_params': {'build_max': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8), 'p_op': 'full'}}},
            'BranchDG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                         'custom_params': {'build_max': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 5, 1), 'p_op': 'grow'}}},
            'BranchNF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'p_op': 'full'}}},
            'BranchNG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'p_op': 'grow'}}},
            'BranchShrink': {'evolve_name': 'mutate branch', 'evolve_rate': 0.0,
                             'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0), 'p_op': 'grow'}}},

            'FilterBO': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                         'custom_params': {'mode': 'branch', 'filter_observations': True}},
            'FilterB': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                        'custom_params': {'mode': 'branch', 'filter_observations': False}},
            'FilterP': {'evolve_name': 'filter optimize', 'evolve_rate': 0.0, 'tourn_size': 5,
                        'custom_params': {'mode': 'point', 'filter_observations': True}},

            # Crossover (35%)
            'Xover': {'evolve_name': 'crossover branch', 'evolve_rate': 0.30, 'custom_params': {}},  # sum 0.70

            # Leftovers are automatically filled with random trees

            # Random (25%)
            'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.05,
                      'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 3, 5, 1), 'p_op': 'full'}}},
            'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.00,
                      'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'p_op': 'grow'}}},
            'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'p_op': 'grow'}}},
            'Rand4': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'p_op': 'full'}}},  # param 'max' can be None
        }

        evolve_loop = evolve_safety_update(evolve_loop)

        if self.origin_is_fix:
            try:
                evolve_random = self.evolve_list_random['from_origin']
            except:
                evolve_random = {'Rand3o': {'evolve_name': 'random trees', 'evolve_rate': 1.00,
                                            'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (10, 3, None, 4), 'p_op': 'full'}}}}
        else:
            try:
                evolve_random = self.evolve_list_random['from_scratch']
            except:
                evolve_random = {'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                           'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (3.5, 2, 5, 1), 'p_op': 'full'}}},
                                 'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                           'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 2, 6, 1), 'p_op': 'grow'}}},
                                 'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.40,
                                           'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 3, None, 4), 'p_op': 'full'}}}
                                 }
        evolve_random = evolve_safety_update(evolve_random)

        return evolve_loop, evolve_random

    def custom_exit_condition(self):
        """
        Special condition to exit the evolve-loop
        1. when >100 generations, no new pareto pareto were found
        """
        try:
            if self.monitor_df['gens_since_last_pareto'].iloc[-1] > 100:
                print('SFEH This condition made your program exit!')
                return True
            else:
                return False
        except Exception:
            return False

    def plagih_gp_run(self, gen_additionally):
        """
        regular plagih run
        """
        if gen_additionally:
            printdummy = copy.deepcopy(self.conf.gen_max)
            self.conf.gen_max = max(self.conf.gen_max, self.gen_id + gen_additionally)
            self.printpl('i', f'Adding new generations, gen_max was {printdummy}, current gen {self.gen_id}. gen_additionally: {gen_additionally}. New max gen: {self.conf.gen_max}')

        # sfeh
        # yaml_dump(self.rootdir / 'used_config.yaml', self.conf, print_type=self.print_type)

        self.gens_since_last_pareto = 0

        while self.gen_id <= self.conf.gen_max and not self.custom_exit_condition():  # max generation, max time, done...
            self.time_genstart = time.perf_counter()

            if self.gen_id == 0:
                self.io.print_g('gg', f'Preparing to create first Generation. Gen {self.gen_id}.')
                self.gen_create_initial()  # sfeh stattdessen einfach checken, ob die letzte population leer ist und info/warnung: neue generation?
            else:
                # This might be a solution for multiprocessing:
                # You can avoid this situation by calling multiprocessing.Process before you load your huge data.
                # Then the additional memory allocations will not be reflected in the child process when you load the data in the parent.
                # sfeh: In python 3.8, this might be availably: multiprocessing.shared_memory https://docs.python.org/3/library/multiprocessing.shared_memory.html
                # sfeh: check memory usage! should not scale with the number of processes, only one pop_base is required, it does not change.

                self.mp_cores = 1  # sfeh wasd
                if self.mp_cores >= 2:
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
                    pass

            self.update_pareto_from_pop_tmp()

            self.pop_analyse()

            self.pop_base = self.population[:]
            self.population = []
            self.io.print_g('ggg', f'Generation {self.gen_id} took a total time of: {time.perf_counter() - self.time_genstart:4.2f}.')
            self.gen_id += 1
        else:
            self.io.print_g('g', f'Done after Generation {self.gen_id}.\nTime since start: {time.perf_counter() - self.time_start:4.2f}s')

        self.io.backup_save()

        return

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
    #     # py_return = self.kernel.pycode_wrap_result(self.env_vars.eval_action.minmax).format('action')
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
    #     # for (parsim, fitness_train, tree) in self.pareto:
    #     #     agent_name = f'{self.conf.name}_{parsim:.0f}'
    #     #
    #     #     agent_as_python = tree.eval_pycode()
    #     #     all_agents.append(f"class {agent_name}:\n{complete_function.format(agent_as_python)}")
    #     #     all_agent_names.append(agent_name)
    #     #     all_more_info.append(f"('{agent_name}', {agent_name}(), {parsim}, {fitness_train})")
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
    #     #     "sys.path.append(str(Path(Path.cwd() / '../../../'))\n" \
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
    #     # pth = path_make_dir(self.rootdir / self.io.folder_pycode / f"agents.py")
    #     # with Path.open(pth, 'w') as file:
    #     #     file.write(pyc_complete)
    #     #     self.printpl('ff', f'Pycode: {pth.as_posix()}')
    #
    #     return

    # def treelut_tree_add(self, tree: Node):
    #     """
    #     update selected values in self.tree_meta
    #     LUT with infos {'parsimony', 'fitness_train', 'expr_sym', 'expr_raw'}
    #     """
    #     # todo
    #     meta = PtreeMeta(tree.fitness_train}
    #             # 'parsimony': tree.meta.parsimony,
    #             # 'expr_raw': tree.meta.expr_raw,
    #             # 'expr_sym': tree.meta.expr_sym}
    #
    #     treehash = hash(tree)  # attention: hashes change between python runs. do not save anything on their hash values <.<
    #
    #     self.tree_lut[treehash] = meta
    #     return

    def helper_evolve_params_branch(self, call_params):
        """
        The call parameters in the evolution file need to be adjusted
        delete if possible
        """
        build_spec = call_params.get('build_max')

        size_mode = build_spec['size_mode']

        mean_min_max_var = build_spec.get('mean_min_max_var')  # (base, min, max, normal_distrib)
        mean_min_max_var = list(mean_min_max_var)
        if 'depth' in size_mode:
            max_dummy = self.conf.tree_depth_max
        elif 'nodes' in size_mode:
            max_dummy = self.conf.parsimony_max
        else:
            raise

        if mean_min_max_var[2] is None:
            mean_min_max_var[2] = max_dummy
        else:
            mean_min_max_var[2] = min(mean_min_max_var[2], self.conf.parsimony_max)
        mean_min_max_var = tuple(mean_min_max_var)

        full_or_grow = build_spec['p_op']

        return build_spec, size_mode, mean_min_max_var, full_or_grow

    def load_origin_tree(self, path_origin_tree):
        """
        The origin tree (which was already loaded) gets activated for its use in the GP-process
        """
        # tree_expr_txt_path = rootdir / 'run_files/tree_expr.txt'
        # tree_numpy_csv_path = rootdir / 'run_files/tree_numpy.csv'
        # tree_labels_csv_path = self.rootdir / 'run_files/tree_labels.csv'

        with Path.open(path_origin_tree, newline='') as file:
            file.read()
        try:
            origin_tree = None  # todo
            expr_sym = 'todo'  # origin_tree.eval_expr_sym()  # todo
        except Exception as sympex:
            raise Exception(f'Loaded origin_tree already failed because of: {sympex}')

        # sfeh, this does not work
        # if not tree_check_is_sympified(tree):
        #     self.io.print_warning('www', 'There is a sympified Version of your raw expression:\nRaw: {}\nSym: {}\n'
        #                          ''.format(expr_raw, expr_sym))

        used_observations = origin_tree.get_observation_list()
        tf_origin_results = self.kernel.eval_tf(expr_sym, used_observations)
        fitness_train = round(tf_origin_results['mean_error'], self.conf.precision)  # fitness_train currently IS the mean error
        if self.kernel.exploration_risk:
            self.kernel.origin_results = tf_origin_results['results_kernel']  # after getting the origin-results, these informations can be updated

        origin_tree.meta.fitness_train = fitness_train
        origin_tree.meta.parsimony = 0

        self.pareto.append([0, fitness_train, origin_tree])  # the origin tree is the only candidate for now -> it is in the pareto front
        self.io.print_g('gg', f'Loading origin tree, regr. error {fitness_train}. Time: {time.perf_counter() - self.time_start:4.2f}s')

        return origin_tree  # self.origin_tree = copy.deepcopy(origin_tree)

    def plot_gen_performance(self):
        """
        All monitoring infos
        sfeh den shit in Funktionen aufteilen
        """
        with plt.rc_context(rc={'axes.grid': True}):
            fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [5, 3, 2, 1]}, sharex='all')  # , figsize=(9, 9)
            plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
            xx = list(self.monitor_df.index)

            axs0 = axs[0]
            axs0.plot(self.monitor_df['fit_avg'], marker='', label='Regression error (average)')
            try:
                avg = self.monitor_df['fit_avg']
                std = self.monitor_df['fit_std']
                axs0.fill_between(xx, avg - std, avg + std, alpha=0.2)  # axs0.set_title('Regression Error (average)')  # sfeh not stderr... upper/lower bound?
            except Exception as ex:
                raise Exception(f'Delete this. were there any problems? {ex}')
            axs0.step(x=xx, y=self.monitor_df['fit_best'], linestyle='dashed', marker='', where='post', color='g', label='Best candidate')  # , label=ax_label
            axs0.set_ylim(ymin=0), axs0.legend(loc='lower left')  # , shadow=True

            axs0_twin = axs0.twinx()
            axs0_twin.plot(xx, self.monitor_df['gens_since_last_pareto'], color='tab:gray', label='Generations since last pareto entry', linestyle='dashed', marker='')  # linestyle='None'
            axs0_twin.tick_params(axis='y', labelcolor='tab:gray')
            try:
                axs0_twin.set_ylim(ymin=0, ymax=max(self.monitor_df['gens_since_last_pareto'].max() or 1, 50))
            except Exception as ex:
                try:
                    print_e(f'damn setting ylim not working sfeh :s {ex}')
                    axs0_twin.set_ylim(ymin=0, ymax=max(self.monitor_df['gens_since_last_pareto'].notnull().max() or 1, 50))
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
                p_std = self.monitor_df['parsim_std']
                axs1.fill_between(xx, p_avg - p_std, p_avg + p_std, alpha=0.2)  # axs1.set_title('TED (average)')
            except Exception as ex:
                raise Exception(f'Delete this if no raise since some time. {ex}')
            axs1.set_ylim(ymin=0), axs1.legend(loc='lower left')

            axs2 = axs[2]
            axs2.plot(self.monitor_df['pop_len'], label='poplike size')
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
            path = self.rootdir / f'monitoring.png'  # -{self.conf.name}
            fig.savefig(path)
            self.printpl('f', f"monitoring: {path.as_posix()}")
            plt.close('all')

    def plot_paretofront(self):
        """
        Write pyplot with pareto candidates
        """
        tuples = [[parsim, fitness] for (parsim, fitness, tree) in self.pareto]
        xx, yy = np.array(tuples).T

        if len(xx) == 0:
            print_e(f'Plotting empty array is not possible! Data={xx, yy}')
            return

        run_name = self.conf.name
        run_name = str(run_name).replace('_', '-')  # sfeh asd workaround for latex version

        with plt.rc_context(rc=pyplot_rc_tex):
            fig, ax = plt.subplots()
            right = max(max(xx), self.conf.parsimony_max) * 1.05  # sfeh check this out 1.05  # if set_right:

            # beyond_lines:  # adding a point to the edges to imply that there are no more values (pareto-plot)
            xx = np.concatenate([[xx[0]], xx, [right + 1]])
            yy = np.concatenate([[max(yy) + 1], yy, [yy[-1]]])

            ax.step(xx, yy, linestyle='dashed', marker='.', label=f'{run_name}', where='post')
            ax.set(xlabel='complexity', ylabel='regression error',
                   xlim=(0, right),
                   ylim=(0, (max(yy) - min(min(yy), 0)) * 1.05))

            try:
                fig.savefig(self.rootdir / f'paretofront.pdf')
                self.printpl('f', f"paretofront (pdf): {self.rootdir / f'paretofront.pdf'}")
            except PermissionError as perm_error:
                print_e(f'Could not save plot: {perm_error}')  # sfeh for everything?

        return

    def plot_evolve_performance(self):
        """
        Plots for each tag in the evolution list
        (too much, i guess)
        sfeh: this should be saved within the trees. Everything else is a waste of memory!
        """
        try:
            with plt.rc_context(rc=pyplot_rc_tex):
                fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 9), sharex='all')  # , gridspec_kw={'height_ratios': [1,1,1]}
                fig.tight_layout()
                for tag in self.evolve_tags:
                    # ['fitness_train', 'parsimony', 'lentree', 'evolve_num', 'count']
                    axs[0].plot(self.monitor_evol[tag]['fitness_train'], label=f'{tag}')
                    axs[1].plot(self.monitor_evol[tag]['parsimony'], label=f'{tag}')
                    axs[2].plot((self.monitor_evol[tag]['lentree'] / self.monitor_evol[tag]['evolve_num']), label=f'{tag}')

                plt.subplots_adjust(wspace=0, hspace=0.1)  # sfeh # left=0, bottom=0, right=1, top=1
                path = self.rootdir / f'monitoring_evolutions.pdf'
                fig.savefig(path)
                self.printpl('f', f"monitoring_evolutions (pdf): {path.as_posix()}")

        except Exception as ex:
            print_e(f'plot_evolution_analysis failed because of: {ex}')

    def file_analysis_plots(self):
        """
        Make all plots
        """

        self.plot_gen_performance()  # largest plot analysing the
        self.plot_paretofront()
        # self.plot_evolve_performance()  # sfeh
        return

    def pop_analyse(self):
        """
        Analysing this generation
        - amount of trees
        - fittest tree
        - average fitness_train
        - average tree parsimony
        """
        gen_id = self.gen_id
        popul = self.population

        if len(popul) == 0:
            raise Exception('Your population isded, its empty. RIP all the computation power used to get here.')

        pop_fitness = [tree.meta.fitness_train for tree in popul]

        # tmp_evol_performance = dict.fromkeys(self.monitor_evol.keys(), pd.DataFrame(columns=['fitness_train', 'parsimony', 'lentree']))
        # for tree in popul:
        #     last_evol = tree.meta.last_evolution
        #     if last_evol in self.evolve_tags:
        #         row = {'fitness_train': tree.meta.fitness_train,
        #                'parsimony': tree.meta.parsimony,
        #                'lentree': len(tree)}
        #         tmp_evol_performance[last_evol].loc[self.gen_id] = row
        #
        # for last_evol, evodata in tmp_evol_performance.items():
        #     if last_evol in self.evolve_loop:
        #         try:
        #             row = {'fitness_avg': evodata['fitness_train'].mean(),
        #                    'parsimony_avg': evodata['parsimony'].mean(),
        #                    'lentree_avg': evodata['lentree'].mean(),
        #                    'evolve_num': self.evolve_loop[last_evol]['evolve_num'],
        #                    'count': len(tmp_evol_performance[last_evol])}
        #             self.monitor_evol[last_evol].append(row, ignore_index=True)  #
        #             # sfeh fitness_train - last fitness_train?
        #         except Exception as ex:
        #             print_e(f'Could not save evol_performance analysis. {ex}')
        #     else:
        #         if last_evol != 'origin' and last_evol != 'Rand3o':
        #             self.io.print_warning('w', f'delete_this, sfeh, okay when the following is origin: {last_evol}')

        pop_parsim = [tree.meta.parsimony for tree in popul]
        pop_treelen = [len(tree) for tree in popul]

        pop_fitness_best = self.kernel.np_best_fitness(pop_fitness)
        try:
            self.best_fitness = self.kernel.best_fitness_function(pop_fitness_best, self.best_fitness)
        except:
            self.best_fitness = pop_fitness_best

        unique_tree_count = len(set([hash(x) for x in popul]))  # sfeh analyze this?

        gen_time = time.perf_counter() - self.time_genstart

        self.monitor_df.loc[gen_id] = {'pop_len': len(popul),
                                       'pop_unique': unique_tree_count,
                                       'fit_avg': np.average(pop_fitness),
                                       'fit_std': np.std(pop_fitness),
                                       'fit_best': self.best_fitness,
                                       'parsim_avg': np.average(pop_parsim),
                                       'parsim_std': np.std(pop_parsim),
                                       'complexity_avg': np.var(pop_treelen),
                                       'time': gen_time,
                                       'gens_since_last_pareto': self.gens_since_last_pareto}  # sfeh version1 delete this shit

        self.io.print_g('gg', f'Created {len(popul)}/{self.conf.pop_max} ({unique_tree_count} unique) in generation {gen_id}. Gen took {gen_time:4.2f}s')
        # sfeh check if there are really unique... doubt it.

        """
        show plots if necessary
        """
        save_gen = int(self.conf.period.get('gen_save', 1))
        plot_gen = int(self.conf.period.get('gen_plots', 1))
        if self.gen_id >= plot_gen and self.gen_id % plot_gen == 0:
            self.file_analysis_plots()

        if self.gen_id >= save_gen and self.gen_id % save_gen == 0 or self.gen_id == 10:  # sfeh extra save at 10 for early feedback while testing
            self.run_backup_save()
        return

    def printpl(self, message_type, message_str):
        """
        Lightweight print function.
        Instead of checking if you should print every time, this is done here.
        message_type options can be found in config
        """
        printez(message_type, message_str, print_type=self.print_type)
        return


# def activate_dataset(path_data, action_name):
#     """
#     loading the data which the GP will be working on.
#     The .csv-file is prepared (loading correct data-type, splitting data, ...)
#     and saved as pickle-file for reloading runs.
#     This is especially important, as the split in training and test-data must be the same.
#
#     separate loading the prepared data into the main class.
#     Why like this? I needed to find a bug in the data_from_csv file and
#     did not want to start the whole stuff everytime
#
#         # self.data_train_panda, self.data_control_panda
#     """
#
#     if path_data.suffix == '.p':
#         data_prepared = pickle_load(path_data)
#     elif path_data.suffix == '.csv':
#         data_prepared = data_from_csv(path_data, action_name=action_name)
#     else:
#         raise FileNotFoundError(f'No data provided? File must be a pickle (.p) or csv (.csv) file. Loaded file: {path_data}')
#
#     env_vars, data_train, data_control = data_prepared  # data_control is data_test
#
#     return env_vars, data_train, data_control


class Selectable:
    """

    """

    def select(self, xtype):
        return


class ChooseOperators(Selectable):
    """

    """

    def __init__(self, operator_pool=None):
        """

        """

        def check_operator_pool(operator_pool):
            """
            Check if the user-specified loaded operators allow closure
            (either float-only/bool only or all 4 types of operators)
            @:param operator_pool: list with operators and their weight of being selected
            """
            # sfeh dunno if that works... 2f not in x
            opxtypes = [oper.xtype for oper in operator_pool.keys()]
            has_2f = any([float == x[1] for x in opxtypes])
            has_2b = any([bool == x[1] for x in opxtypes])
            has_f2b = any([float in x[0] and bool == x[1] for x in opxtypes])
            has_b2f = any([bool in x[0] and float == x[1] for x in opxtypes])
            if not all([has_2f, has_2b, has_f2b, has_b2f]):
                logging.warning(f'Loaded operators do not feature both numeric (float) and Boolean type.')
            if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
                raise Exception(f'Loaded operators do not allow closure!')

        if operator_pool is None:  # quick developer adjustments
            operator_pool = [['+', 2],
                             ['-', 1], ['Usub', 1],
                             ['*', 2], ['/', 1],
                             ['Square', 0.75], ['**', 0.25],
                             ['Abs', 0.5], ['sign', 0.5], ['Round', 0.5],  # sfeh stop chain of arity-1 op in buid method?
                             ['sqrt', 0.25],
                             # ['log', 0.1], ['log1p', 0.1],
                             ['sin', 0.5],  # ['tan', 0.1], ['cos', 0.33], ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
                             ['tanh', 0.2],
                             ['Andb', 1], ['Orb', 1], ['Notb', 0.5], ['Xor', 1],
                             ['==', 1], ['!=', 0.5],
                             ['<', 0.5], ['<=', 0.5], ['>', 0.1], ['>=', 0.1],
                             ['Ifte', 2],
                             ['Mini', 1], ['Maxi', 1]]
            operator_pool = {op[x[0]]: x[1] for x in operator_pool}  # sfeh this maps the actual class to the label

        # if sfeh_no_crazyops:
        #     del operator_pool['**']
        #     # workaround sfeh (delete this)

        check_operator_pool(operator_pool)

        choose_oparray = {
            # all operator_pool with a certain xtype-result
            # None: [[], []],
            float: [[], []],  # 2f
            bool: [[], []],  # 2b
            (tuple([]), float): [[], []],  # todo?  replacing float
            (tuple([]), float): [[], []],  # todo?
            (tuple([float]), float): [[], []],  # x**2, sqrt, log, sin, ...
            (tuple([float, float]), float): [[], []],  # +, -, *, /, **, ...
            (tuple([bool, float, float]), float): [[], []],  # Ifte
            (tuple([float, float]), bool): [[], []],  # <, >, =, >=
            (tuple([bool]), bool): [[], []],  # not
            (tuple([float]), bool): [[], []],  # dummy, currently no such operator
            (tuple([bool, bool]), bool): [[], []],  # and, or, xor, ...
        }
        for label, prob in operator_pool.items():
            # tuple-xtype (point mutations)
            choose_oparray[label.xtype][0].append(label)
            choose_oparray[label.xtype][1].append(prob)
            # float/bool (construction of trees)
            choose_oparray[label.xtype[1]][0].append(label)
            choose_oparray[label.xtype[1]][1].append(prob)

        for o, p in choose_oparray.items():
            # normalizing the probabilities in every case to a sum of 1 (100%)
            # (saving some very little time...)
            if p[0]:
                choose_oparray[o][1] = [x / sum(p[1]) for x in p[1]]
            else:
                pass  # todo delete the entry?

        self.choose_oparray = {}
        for xtype, x in choose_oparray.items():
            # self.choose_oparray[xtype] = lambda: np.random.choice(x[0], p=x[1])  # "seloplam" faster, but less readable version
            self.choose_oparray[xtype] = (x[0], x[1])

    def select(self, xtype):
        """

        """
        return np.random.choice(self.choose_oparray[xtype][0], p=self.choose_oparray[xtype][1])
        # return self.choose_oparray[xtype]()  # "seloplam"


class ChooseConstants(Selectable):
    """

    """
    # sfeh random with numpy?
    distributions = {float: [lambda: random.normalvariate(0, 1),
                             lambda: random.normalvariate(1, 1),
                             lambda: random.normalvariate(10, 5),
                             lambda: random.randint(1, 20)],  # 0 has actually no purpose (except as being an action)
                     bool: [lambda: random.choice([True, False])]}

    def __init__(self, precision=6, path_distrib=Path.cwd(), data_train=None, n_samples=100):
        """
        todo path.cwd() is no good input
        """
        self.precision = precision
        try:
            lambdadist_as_string = yaml_load(path_distrib)
            # todo how should distributions be loaded?
            # e.g. sample_amount = lambdadist_as_string.get('observed_floats')
            self.terminal_distributions = {float: [], bool: []}
            self.terminal_distributions[float].extend([eval(x) for x in lambdadist_as_string[float]]),
            self.terminal_distributions[bool].extend([eval(x) for x in lambdadist_as_string[bool]])

            # self.sample_floats_from_data(obs_infos, data_train, n_samples=n_samples)  # todo
        except Exception:
            logging.info('Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.')

    def sample_floats_from_data(self, obs_infos, data_train, n_samples=100):
        """
        ONLY floats, because ...do you really want to load Boolean True/False samples??
        (okay, it might make sense as it better represents the actual distribution- NO FUCK IT.)
        """
        if obs_infos is not None:
            obsnames = obs_infos.observables[float].keys()
            obs_samples = data_train[obsnames].to_numpy().flatten()
            obs_samples = np.random.choice(obs_samples, size=n_samples)
            self.terminal_distributions[float].extend([lambda: random.choice(obs_samples)]),  # take one

    def select(self, xtype):
        """

        """
        value = random.choice(self.distributions[xtype])()
        if xtype == float:  # sfeh int aswell?
            value = float(round(value, self.precision))
            return FloatConstant(value)
        elif xtype == bool:
            return BoolConstant(value)


class ChooseObservation(Selectable):
    """
    func_list, probability_list = self.operators[xtype]
    return np.random.choice(func_list, p_op=probability_list)
    """

    def select(self, xtype):
        """
        Randomly choosing an operator-label for a given xtype.
        choose_oparray3 must be given, as they are different between runs.
        arity can also be set optionally, e.g. for point mutation
        todo DOUBLE-check if this xtype is chosen correctly... better: replace it
        """
        return self.observables[xtype]()

    def __init__(self, obs_names):
        """
        :param obs_names: list of all observation names (e.g. ['cartVel', 'cartPos'])
        """
        # def observation_select_index(observations, max_hist=10):
        #     """
        #     chooses variables but weighting how old they are.
        #     observations = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4'] -> [0.28, 0.23, 0.19, 0.16, 0.13]
        #     sfeh: what about larger steps?
        #     e.g. [0, 1, 2, 3] is good, but [0, 5, 10, 15] is baaaad
        #     what if variables are not all of same diff?
        #     """
        #     observations = np.delete(observations, np.s_[max_hist:])
        #     x = len(observations)
        #     fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
        #     p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
        #     p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
        #     return np.random.choice(observations, p=p)  # returning a function this time
        #
        # obs_prop = []
        # obs_info = {}
        #
        # for fam in list(set(observation_get_family_and_time(x)[0] for x in obs_names)):
        #     fam_members = sorted([x for x in obs_names if x.fam == fam], key=lambda o: o.timeindex)
        #     if len(fam_members) > 1:
        #         obs_names.extend([x for x in fam_members])
        #         obs_prop.extend(list(observation_select_index(fam_members)))
        #         index_minmax = (fam_members[0].timeindex, fam_members[-1].timeindex)
        #         for obs in fam_members:
        #             obs.index_minmax = index_minmax
        #             # environment.obs_infos[obs.name] = obs  # todo okay do we need this? :s guess we will find out x.D haha
        #             obs_info[obs.name] = obs
        #     else:
        #         obs = fam_members[0]
        #         obs_info[obs.name] = obs
        #         obs_names.append(obs)
        #         obs_prop.append(1)  # just one value

        obs_list = [Observation(x) for x in obs_names]

        self.observables = {float: lambda: np.random.choice(obs_list),  # , p=obs_prop
                            bool: None}  # todo None? no  lambda? yeah, not important but still...


class TreeBuilder:
    # class Choosing(Selectable):  # sfeh was
    """
    Just a class to prevent referencing all the separate shizzle everytime
    todo precision?

    ===
    this was:
    def choose_op():
    choose_oparray3 -> operator
    env_vars.choose_obs -> observation
    choose_distributions -> constant
    """

    def __init__(self, operators: ChooseOperators, observations: ChooseObservation, constants: ChooseConstants, root_xtype):
        self.operators = operators
        self.observations = observations
        self.constants = constants

        self.root_xtype = root_xtype

        # todo
        self.precision = 6

    def choose_any(self, xtype, p_op):
        """

        """
        if random.random() < p_op:
            return self.operators.select(xtype)
        else:
            # sfeh add p_term? 0.5?
            return self.choose_term(xtype)

    def choose_term(self, xtype, p_observation=0.5):
        """
        sfeh: precision not required?
        # sfeh 50% chance observatio    n/value
        """
        if random.random() < p_observation:
            try:
                return self.observations.select(xtype)
            except Exception:
                pass  # just return a constant now, e.g. because there are no boolean observations

        return self.constants.select(xtype)

    def choose_op(self, any_xtype):
        """
        any_xtype can be a tuple, single type or even None
        """
        return self.operators.select(any_xtype)

    def invent_core_depth(self, xtype, depth_max, depth=0, p_op=1):  # todo grow method
        """
        # set path/id? todo
        # set depth? todo
        """
        if depth < depth_max:
            label = self.choose_any(xtype, p_op)
            childs = [self.invent_core_depth(xt, depth_max, depth=depth + 1, p_op=p_op) for xt in label.xtype[0]]
            node = Node(label=label, childs=childs)  # , depth=depth sfeh no depth?
        else:
            label = self.choose_term(xtype)
            node = Node(label=label)
        return node

    def evolve_mutate_filter_random(self, tree):
        """
        Mutates a number of float terminal of a tree
        todo ROOT
        """
        etree = copy.deepcopy(tree)  # todo ==>state

        # mode = call_params['mode']  # point/branch/all  # todo
        # yes_observations = call_params.get('yes_observations')  # sfeh point/branch/all todo
        mutate_filter = 'gaussian_filter'  # sfeh change?  todo

        node = np.random.choice(etree.eval_mutatable_nodes())

        node.evolve_mutate_filter_branch(precision=self.precision)

        # if mode == 'branch':
        #     pass
        # else:
        #     pass
        #     # mode == 'point'
        #     # sfeh delete this? point can always hapen
        #     # node_id.evolve_mutate_point(tb)

        # todo ==>state
        return etree

    def evolve_mutate_point(self, tree: Node):
        """
        Mutate a single mutatable point in any Tree.
        sfeh is the tree a tree copy or the same tree?
        """
        etree = copy.deepcopy(tree)  # todo ==>state

        node = np.random.choice(etree.eval_mutatable_nodes())
        xtype = node.get_xtype()

        if node.get_arity() > 0:
            node.set_label(tb.choose_op(xtype))  # Function is same type, same arity
        else:
            node.set_label(tb.choose_term(xtype[1]))  # 3 -> '2f' -> 5

        # todo ==>state
        return etree

    def evolve_mutate_branch_depth(self, tree, depth_max):
        """
        todo ==>depth only
        currently only one branch
        """
        etree = copy.deepcopy(tree)  # todo ==>state

        node = np.random.choice(etree.eval_mutatable_nodes())
        xtype = node.get_xtype()[1]  # todo todotodo

        branch = self.invent_core_depth(xtype, 3, depth=0, p_op=1)  # todo ==>dummies
        node.replace_with_branch(branch)
        # if node.depth == depth_max:
        #     node.set_label(tb.choose_term(xtype[1]))  # sfeh update node plabel
        # else:
        #     node.childs = [Node(tb.choose_any(xt, p=1)) for xt in node.get_xtype()[0]]  # todo ==>

        # etree.finalize()  # todo ==>state
        return etree

    def evolve_crossover(self, tree1: Node, tree2: Node):
        """
        todo ==>depth only
        currently only one branch
        """
        atree = copy.deepcopy(tree1)  # todo ==>state
        btree = copy.deepcopy(tree2)  # todo ==>state

        try:
            anodes = atree.eval_mutatable_nodes(allow_root=False)
            anode = np.random.choice(anodes)
            xtype = anode.get_xtype_out()
            bnodes = btree.eval_mutatable_nodes(xtype_out=xtype)
            if bnodes:
                bnode = np.random.choice(bnodes)
            else:
                xtype = float if xtype == bool else bool  # the other swap type now
                bnodes = btree.eval_mutatable_nodes(allow_root=False, xtype_out=xtype)
                bnode = np.random.choice(bnodes)
                anode = atree.eval_mutatable_nodes(xtype_out=xtype)
        except:
            raise Exception

        # todo deepcopy required??
        anode_copy = copy.deepcopy(anode)  # todo ==>state
        # bnode = copy.deepcopy(bnode)  # todo ==>state

        anode.replace_with_branch(bnode)
        bnode.replace_with_branch(anode_copy)

        # atree.meta.last_evolution = tag  # todo ==>tree tag
        # btree.meta.last_evolution = tag
        # self.poplike.append(left_parent)
        # self.poplike.append(right_parent)

        return atree, btree

    def pop_random(self, call_params, from_origin=False):
        """
        Creates random trees for the population
        """
        build_spec, size_mode, mean_min_max_var, full_or_grow = helper_evolve_params_branch(call_params)

        if from_origin:
            """
            insert a (random) number of branches at the first possible "layer"
            (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
            - get these nodes, randomly choose a subset of those
            - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
            - split the amount of nodes up (randomly) and add these new branches to the tree
            """

            # layer0_ids = tree_get_mutatable_layer(from_origin, 0)
            layer0_ids = [1, 2, 3]

            # build_split = []
            # if 'depth' in size_mode:
            #     for ii in range(len(layer0_ids)):
            #         build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')
            #         build_split.append(build_size)
            #
            # elif 'nodes' in size_mode:
            #     build_nodes = choose_build_size(size_mode, mean_min_max_var, force='branch')
            #     build_split = randomly_split_range(build_nodes, len(layer0_ids))
            # else:
            #     raise
            #
            # tree = copy.deepcopy(from_origin)
            # for i in range(len(layer0_ids)):  # insert branches! get layer every time (node ids might have changed)
            #     layer0_ids = tree_get_mutatable_layer_lv0(tree)
            #     node_id = layer0_ids[i]
            #     first_xtype = float  # tree_node_get_xtype(tree, node_id)  # todo
            #     old_branch = tree_node_get_branch(tree, node_id, karoo=True)
            #     build_size = build_split[i]
            #
            #     # tree = tree(BuildDummy(float))   # todo deprecated
            #     core = tree.invent_core(size_mode, first_xtype, build_size, full_or_grow)
            #     tree = tree_insert_subtree(tree, core, old_branch, karoo=True)
        else:
            build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')  # depth, in this case
            # todo
            # coolcore.evolve_mutate_branch_depth(build_size, choose_oparray3, env_vars.choose_obs,
            #                                      choose_distributions, precision, size_mode=size_mode, full_or_grow=full_or_grow)
            # coolcore.evolve_random_tree_depth(size_mode, xtype_root, build_size, full_or_grow)

        # return coolcore


# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['label'], 1/v['xtype': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


# def choose_term(xtype_out, choose_obs, choose_distributions, precision):
#     """
#
#     """
#
#     # sfeh 50% chance observation/value
#     if random.choice(['obs', 'distrib']) == 'obs' and choose_obs[xtype_out]:
#         obs = choose_obs[xtype_out]()
#         # print('SAME???', obs.name, obs.label)  # sfeh
#         return obs
#     else:
#         dist_fun = random.choice(choose_distributions[xtype_out])
#         value = dist_fun()
#         if xtype_out == float:  # sfeh int aswell?
#             value = float(round(value, precision))
#             const = FloatConstant(value)
#         elif xtype_out == bool:
#             const = BoolConstant(value)
#         else:
#             raise Exception('ASDASD NOOO WHYY')
#         return const


def helper_evolve_params_branch(call_params, tree_depth_max=10, parsimony_max=30):
    """
    The call parameters in the evolution file need to be adjusted
    delete if possible
    """
    build_spec = call_params.get('build_max')

    size_mode = build_spec['size_mode']

    mean_min_max_var = build_spec.get('mean_min_max_var')  # (base, min, max, normal_distrib)
    mean_min_max_var = list(mean_min_max_var)
    if 'depth' in size_mode:
        max_dummy = tree_depth_max
    elif 'nodes' in size_mode:
        max_dummy = parsimony_max
    else:
        raise

    if mean_min_max_var[2] is None:
        mean_min_max_var[2] = max_dummy
    else:
        mean_min_max_var[2] = min(mean_min_max_var[2], parsimony_max)
    mean_min_max_var = tuple(mean_min_max_var)

    full_or_grow = build_spec['p_op']

    return build_spec, size_mode, mean_min_max_var, full_or_grow


def randomly_split_range(range_max, num_splits):
    """
    todo reuse this
    split a integer range randomly into parts
    [1..100] -> [33, 15, 52] (0 is allowed)
    """

    # tmp_distributions = random.sample(range(1, range_max), num_splits)
    # d_sum = sum(tmp_distributions)
    # d_list = [int(round(range_max*(x/d_sum), 0)) for x in tmp_distributions]
    sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
    d_sum = sum(sample_dist)  # 1.5
    sample_dist = [x / d_sum for x in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
    sample_dist = [x * range_max for x in sample_dist]  # [12, 60, 28] -> for 100 nodes
    sample_dist = [int(round(x, 0)) for x in sample_dist]  # make them useable ints

    # sfeh workaround, this makes exactly the correct range by changing the most extreme entry
    helper_diff = range_max - sum(sample_dist)
    if sum(sample_dist) < range_max:
        smallest = sample_dist.index(min(sample_dist))
        sample_dist[smallest] += helper_diff

    if sum(sample_dist) > range_max:
        greatest = sample_dist.index(max(sample_dist))
        sample_dist[greatest] += helper_diff

    return sample_dist


def choose_build_size(size_mode, mean_min_max_var, tree=None, nodepath=None, force=None):
    """
    delete this?
    Very unified utility function that returns the required tree size from the following parameters
    # branch_nodes, branch_depth, tree_depth, tree_nodes

    It can either return a tree depth or an amount of tree nodes
    """
    mean, size_min, size_max, size_variance = mean_min_max_var
    if size_mode == 'branch_nodes' or size_mode == 'branch_depth' or force == 'branch':
        relative_size = 0
    else:
        if tree and nodepath:
            pass
        else:
            raise Exception('No tree or node is given for computing the relative size')

        if size_mode == 'tree_depth':
            tree_size = tree.core.childs_depth_max
            print('tree_size = tree.core.childs_depth_max:', tree_size)

            node_size = len(nodepath)
        elif size_mode == 'tree_nodes':
            tree_size = len(tree)
            print('len(tree)?:', len(tree))
            node_size = len(tree.get_nodepath(nodepath))
            print('tree.get_nodepath(nodepath)?:', tree.get_nodepath(nodepath))
        else:
            raise Exception('Sizemode not known?')

        relative_size = tree_size - node_size
        print('asdasd', relative_size)

    build_size = int(random.normalvariate(mean, size_variance))
    if size_max is not None:
        build_size = min(size_max - relative_size, build_size)
    build_size = max(size_min, build_size)

    return int(build_size)


def tree_from_labellist(labellist, *args):
    pass


if __name__ == '__main__':
    """
    Alpha tests
    """
    ops = ChooseOperators()
    inputs = ChooseObservation(['a', 'b'])
    consts = ChooseConstants()
    tb = TreeBuilder(ops, inputs, consts, float)
    t1 = tb.invent_core_depth(float, 3, p_op=0.5)
    tree2 = tb.evolve_mutate_point(t1)
    tree3 = tb.evolve_mutate_branch_depth(tree2, depth_max=4)
    tree4 = tb.evolve_mutate_branch_depth(tree3, depth_max=4)
    tree5, tree6 = tb.evolve_crossover(tree3, tree4)
    print('crossover')
    t3 = tb.invent_core_depth(float, 4, p_op=0.5)
    t4 = tb.invent_core_depth(float, 4, p_op=0.8)
    t5 = tb.evolve_mutate_filter_random(t3)
    print('tree3', t3)
    print('huehue', t4, t4.get_observation_list())
    print(inputs.select(float).nlabel)
