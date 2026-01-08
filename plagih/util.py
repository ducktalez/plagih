import pickle
import re
import shutil
import time

import yaml
from pathlib import Path
from matplotlib import pyplot as plt
# from distutils.spawn import find_executable
import numpy as np

PRINT_DUMMY = 'wwaaagggiiifffpp'  # noqa: dummy for print-policy wwwwaaaggggggiiifff
DEBUG_DUMMY = True  # noqa: dummy for debug-policy
FLOAT_PRECISION = 3
PLOTS_INTERVAL = 1
BACKUP_INTERVAL = 10
CHAIN_implement = 'sfeh'  # used as placeholder for implementation-tasks
TREE_MIN_PARSIMONY = 3


class TreeError(Exception):
    """All Tree-specific errors"""
    pass

class TreeLutError(TreeError):
    """Errors regarding lookup-tables for trees"""
    pass

class TreeSizeError(TreeError):
    """Non-important, but errors that often come up, e.g.
    - tree is too small after simplification
    - Tree has too many nodes. This should be covered somewhere though!
    discuss:subclassing value-error?"""
    pass


class SympyError(Exception):
    """Non-important, but errors that often come up
    usually, when imaginary numbers accidentally come up in an expression"""
    pass

class SympyImaginaryNumber(SympyError):
    pass

class CuriosityError(Exception):
    """NeverHappensError/CuriosityError/DeletemeRrror/DebugError
    For code, that should never be reached. Just to check, why."""
    pass


class BColors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    HEADER = '\033[95m'
    FAIL = '\033[91m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BLACK2 = '\033[40m'
    RED2 = '\033[41m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    UNDERLINE_RESET = '\033[0m'
    ITALIC = '\x1B[3m'
    ITALIC_RESET = '\x1B[0m'
    RESET_COLOR = '\033[39m'
    RESET = '\033[0m'

    # Reset
    Color_Off = '\033[0m'  # Text Reset

    # Regular Colors
    Black = '\033[0;30m'  # Black
    Red = '\033[0;31m'  # Red
    Cyan = '\033[0;36m'  # Cyan
    White = '\033[0;37m'  # White

    # Bold
    BBlack = '\033[1;30m'  # Black
    BRed = '\033[1;31m'  # Red
    BCyan = '\033[1;36m'  # Cyan
    BWhite = '\033[1;37m'  # White

    # Underline
    UBlack = '\033[4;30m'  # Black
    URed = '\033[4;31m'  # Red
    UCyan = '\033[4;36m'  # Cyan
    UWhite = '\033[4;37m'  # White

    # Background
    On_Black = '\033[40m'  # Black
    On_Red = '\033[41m'  # Red
    On_Cyan = '\033[46m'  # Cyan
    On_White = '\033[47m'  # White

    # High Intensty
    IBlack = '\033[0;90m'  # Black
    IRed = '\033[0;91m'  # Red
    ICyan = '\033[0;96m'  # Cyan
    IWhite = '\033[0;97m'  # White

    # Bold High Intensty
    BIBlack = '\033[1;90m'  # Black
    BIRed = '\033[1;91m'  # Red
    BICyan = '\033[1;96m'  # Cyan
    BIWhite = '\033[1;97m'  # White

    # High Intensty backgrounds
    On_IBlack = '\033[0;100m'  # Black
    On_IRed = '\033[0;101m'  # Red
    On_ICyan = '\033[0;106m'  # Cyan
    On_IWhite = '\033[0;107m'  # White


def rnd_choice(a):
    return np.random.choice(a)


def xt_self(xtype):
    return xtype[1]


def xt_childs(xtype):
    return xtype[0]


def get_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def pickle_dump(path, data):
    """Saving python data (probably run) in a very small pickle file"""
    path = path_make_dir(path)

    with Path.open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    printez('f', f'Backup: {path}')  # .as_posix()


def yaml_dump(path, data, default_flow_style=True):
    """saves data to yaml file (better than xml/Json)
    - default_flow_style=False for dumping in a block style"""
    path = path_make_dir(path)
    with Path.open(path, 'w') as file:
        _ = yaml.dump(data, file, default_flow_style=default_flow_style, sort_keys=False)
        printez('ff', f'{path}')  # .as_posix()
        return


def remove_trailing_zeroes(x):
    x = re.sub(r'\.0+$|0+$', '', x)
    return x


def term_format(x, cut=False):
    """
    :param x:
    :return:
    sfeh HOW OFTEN IS THIS USED O_ô can be replaced with string_remove_trailing_zeroes()
    """
    try:
        if cut:
            x = float(x)
            if float(x) < 0.001 or float(x) > 1000:
                xstr = f'{x:.3g}'
            else:
                xstr = f'{x:.3f}'
                xstr = re.sub(r'\.0+$|0+$', '', xstr)
        else:
            xstr = remove_trailing_zeroes(x)
        return xstr
    except ValueError:
        return x  # constants/terms


def string_remove_trailing_zeroes(number_string):
    """
    2.00000000000 -> 2
    0.20000000000 -> 0.2
    sfeh: 1.1e+3 is 1100. Looks ugly like this.
    :param number_string:
    :return:
    """
    # Removes unnecessary zeros at the end of decimal numbers
    cleaned_string = re.sub(r'(\.\d*?)0+(?!\d)', r'\1', number_string)
    # Removes the decimal point if there are no decimal places left: keep this point to imply float?
    cleaned_string = re.sub(r'\.(?!\d)', '', cleaned_string)
    return cleaned_string


def printpl(msg_t, message_str):
    """Lightweight print function.
    Instead of checking if you should print every time, this is done here.
    message_type options can be found in config
    """
    printez(msg_t, message_str)
    return


def print_warning(msg_type, text):
    """
    Printing warnings
    """
    try:
        if msg_type not in PRINT_DUMMY:
            return
        if 'w' in msg_type:
            print(f'{BColors.WARNING}Warning ({msg_type}): {text}{BColors.RESET_COLOR}')  # all yellow
        else:
            print(f'{BColors.WARNING}Warning ({msg_type}):{BColors.RESET_COLOR} {text}')  # only "Warning" yellow
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
                 'text.usetex': shutil.which('latex') is not None,  # check if 'latex' is available
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


def blue_string(txt):
    """Was print_blue()"""
    return f"{BColors.CYAN}{txt}{BColors.RESET_COLOR}"


def print_caution(txt):
    """
    Printing errors that are not worth stopping by raising an exception
    BColors.FAIL
    """
    print(f'{BColors.RED}CAUTION! {BColors.WARNING}{txt}{BColors.RESET_COLOR}')


def pickle_load(path: Path):
    """loads a pickle file (usually .p or .pkl)"""
    with Path.open(path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data


def printez(message_type, text):
    """giving prints colours, accessable from everywhere"""
    if message_type not in PRINT_DUMMY:
        return

    if 'i' in message_type:
        txt = f'{BColors.CYAN}Info: {text}{BColors.RESET_COLOR}'
    elif 'f' in message_type:
        txt = f'Writing File: {text}{BColors.RESET_COLOR}{BColors.RESET_COLOR}'
    elif 'a' in message_type:
        txt = f'{BColors.GREEN}{text}{BColors.RESET_COLOR}'  # Paretofront:
    elif 'w' in message_type:
        print_warning(message_type, text)
        return
    elif 'g' in message_type:
        time_now = time.strftime("%d.%m %H:%M", time.localtime())
        txt = f'[{time_now}] {text}'
    elif 'p' in message_type:
        txt = f'Performance: {BColors.CYAN}{text}{BColors.RESET_COLOR}'
    else:
        raise Exception(f'print_type-mode {message_type} not known.')

    print(f'{txt}')

    return


def yaml_load(path: Path):
    """.yaml-file loader (saves two lines that I had to look up all the time)
    Especially the Loader has to be specified."""
    with Path.open(path, 'r') as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)  # yaml.safe_load sfeh?

    return loaded_yaml


if __name__ == '__main__':
    print(f'Testing the plot style')
    import matplotlib.pyplot as plt
    import numpy as np

    x = range(10)
    y = np.sin(np.arange(10) / 10) + np.arange(10) / 10
    with plt.rc_context(rc=pyplot_rc_two_column):
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='x', label='random values (idk)')
        ax.set(xlabel='some label', ylabel='some other value')
        ax.legend(loc='lower left')
        plt.show()
