import pickle
import time

import yaml
from pathlib import Path

from distutils.spawn import find_executable


DEBUG_DUMMY = True  # Use this to find codeblocks that are just interesting during development
TEST_DUMMY = True  # sfeh: actually use this later! check trees, check if all ops are usable,
DELETE_ME = True  # sfeh:delete this when development phase is over
PRINT_DUMMY = 'wwwwwaaagggggiiifff'  # sfeh:xxx
PRECISION = 6


def printyeah(message_type, message_str):
    """
    sfeh:open
    Lightweight print function.
    Instead of checking if you should print every time, this is done here.
    message_type options can be found in config
    """
    printez(message_type, message_str)
    return


class BColors:
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


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def pickle_dump(path, data):
    """

    """
    path = path_make_dir(path)

    with Path.open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    printez('f', f'Backup: {path}')  # .as_posix()


def yaml_dump(path, data, default_flow_style=True):
    """
    saves prepared plagih data to pickle file
    - default_flow_style=False for dumping in a block style
    """
    path = path_make_dir(path)
    with Path.open(path, 'w') as file:
        _ = yaml.dump(data, file, default_flow_style=default_flow_style, sort_keys=False)
        printez('ff', f'{path}')  # .as_posix()
        return


def print_warning(message_type, text):
    """
    Printing warnings
    """
    try:
        if message_type not in PRINT_DUMMY:
            return
        if message_type == 'w':
            print(f'{BColors.WARNING}Warning ({message_type}): {text}{BColors.RESET}')  # completely yellow
        else:
            print(f'{BColors.WARNING}Warning ({message_type}): {text}{BColors.RESET}')  # only "Warning" yellow
    except Exception as ex:
        print_warning('w', f'Could not print warning: {ex}')
    return


def path_make_dir(p: Path):
    """
    Creates the folder and files according to run specified through naming (E.g. MTC200_MSE_scratch)
    """
    folder = p if len(p.suffix) == 0 else p.parent
    folder.mkdir(parents=True, exist_ok=True)
    return p


pyplot_size = (3.6, 2.7)  # default: (6.4, 4.8) S: (4, 3)  XXL: (16, 9)  M: (4.8, 3.6) (4.4, 3.3)
plplot_size_up = (3.6, 3.6)

pyplot_rc_tex = {'figure.autolayout': True,
                 'text.usetex': find_executable('latex') or False,  # sfeh:debug check if latex is available
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


def printez(message_type, text):
    """
    giving prints colours, accessable from everywhere
    """
    if message_type not in PRINT_DUMMY:
        return

    if 'i' in message_type:
        print(f'{BColors.CYAN}Info: {text}{BColors.RESET}')
    elif 'f' in message_type:
        print(f'{BColors.MAGENTA}Writing File: {text}{BColors.RESET}{BColors.RESET}')
    elif 'a' in message_type:
        print(f'{BColors.GREEN}{text}{BColors.RESET}')  # Paretofront:
    elif 'w' in message_type:
        print_warning(message_type, text)
    elif 'g' in message_type:
        time_now = time.strftime("%d.%m %H:%M", time.localtime())
        print(f'[{time_now}] {text}')
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


if __name__ == '__main__':
    print(f'Testing the plot style')
    import matplotlib.pyplot as plt
    import numpy as np
    x = range(10)
    y = np.sin(np.arange(10)/10) + np.arange(10)/10
    with plt.rc_context(rc=pyplot_rc_two_column):
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='x', label='random values (idk)')
        ax.set(xlabel='some label', ylabel='some other value')
        ax.legend(loc='lower left')
        plt.show()
