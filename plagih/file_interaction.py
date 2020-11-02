import pickle
from pathlib import Path
from plagih.printing import *
import yaml


MTC_XTICKS = ((-0.06, -0.03, 0, 0.03, 0.06), ('$-.06$', '$-.03$', '$0$', '$.03$',  '$.06$'))  # , color='k', size=8
MTC_YTICKS = ((-1.2, -0.8, -0.4, 0, 0.4), (r'$-1.2$', r'$-0.8$', r'$-0.4$', r'$0.0$', r'$0.4$'))

IB_YICKS = ((-14000, -12000, -10000, -8000, -6000, -4000), ('$-14$', '$-12$', '$-10$', '$-8$', '$-6$', '$-4$'))

pyplot_size = (3.6, 2.7)  # default: (6.4, 4.8) S: (4, 3)  XXL: (16, 9)  M: (4.8, 3.6) (4.4, 3.3)
plplot_size_up = (3.6, 3.6)
pyplot_rc_tex = {'figure.autolayout': True,
                 'text.usetex': True,
                 'backend': 'pgf',
                 'figure.figsize': pyplot_size,
                 'axes.labelpad': 0.5,  # padding axis-ticks to axis title
                 'xtick.labelsize': 8, 'xtick.major.size': 1.5, 'xtick.major.pad': 1.5,
                 'ytick.labelsize': 8,  'ytick.major.size': 1.5, 'ytick.major.pad': 1.5,
                 # 'savefig.pad_inches': 0,
                 'font.size': 10,
                 'legend.fontsize': 9,
                 'savefig.dpi': 300,
                 # 'lines.linewidth': 1,
                 # 'lines.markersize': 3,
                 # 'axes.xmargin': 0,
                 # 'axes.ymargin': 0
                 }

rc_pyplot_size = {'figure.figsize': pyplot_size}
# ['text.latex.preamble'=r"\usepackage{lmodern}"]

# pyplot_rc_options = {'font.family': 'serif',
#                      'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
#                      'font.sans-serif': ['Helvetica', 'Avant Garde', 'Computer Modern Sans serif'],
#                      'font.cursive': ['Zapf Chancery'],
#                      'font.monospace': ['Courier', 'Computer Modern Typewriter']}
# more_optionsasd = {'savefig.dpi': 300, }
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


# def folder_make_dir(path):
#     """
#     Checks if the folders for the specified path exist and creates them otherwise.
#     Apparently, this procedure is used often.
#     """
#     if not Path.is_dir(path):
#         Path.mkdir(path)
#     return path
#
#
# def file_make_dir(file_path):
#     """
#     Creates the folder only knowing the file.
#     paff/tuuu/fyle.txe  ->  *mkdir* paff/tuuu/
#     """
#     p = Path(file_path)
#     if not p.parent.is_dir():
#         p.parent.mkdir(parents=True)
#     return p


def pickle_dump(path, data, print_type=None):
    """

    """
    path = path_make_dir(path)

    with Path.open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    printez('f', f'Backup: {path.as_posix()}', print_type=print_type)


def pickle_load(path: Path):
    """
    loads a pickle file (usually .p or .pkl)
    """
    with Path.open(path, 'rb') as file:
        pickle_data = pickle.load(file)

    return pickle_data


# def pickle_dump(path, data, print_type=None):
#     """
#     saves prepared plagih data to pickle file
#     """
#
#     path = path_make_dir(path)
#     with Path.open(path, 'wb') as file:
#         pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
#         printez('f', f'{path.as_posix()}', print_type=print_type)
#     return


def yaml_load(yaml_path):
    """
    .yaml-file loader (saves two lines that I had to look up all the time)
    """
    with Path.open(yaml_path, 'r') as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)  # yaml.safe_load sfeh?
    return loaded_yaml


def yaml_dump(path, data, print_type=None):
    """
    saves prepared plagih data to pickle file
    """

    path = path_make_dir(path)
    with Path.open(path, 'w') as file:
        _ = yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        printez('ff', f'{path.as_posix()}', print_type=print_type)
    return
