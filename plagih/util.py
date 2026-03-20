import os
import pickle
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from plagih.config import cfg as _cfg

# ---------------------------------------------------------------------------
# Re-exports from extracted modules (backward compatibility)
# ---------------------------------------------------------------------------
from plagih.exceptions import (  # noqa: F401
    CuriosityError,
    SympyError,
    SympyImaginaryNumber,
    TreeError,
    TreeLutError,
    TreeSizeError,
)
from plagih.logging_utils import (  # noqa: F401
    _flush_progress_line,
    log,
    log_debug,
    log_error,
    log_info,
    log_warning,
    print_caution,
    print_generation_done,
    print_generation_progress,
    print_generation_start,
    print_warning,
    printez,
    printpl,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Backwards-compatible module-level aliases.
#
# Legacy code reads/writes these globals directly, e.g.:
#     from plagih.util import PRINT_DUMMY
#     util.PRINT_DUMMY = "ww"
#
# They are initialised from PlagihConfig and kept in sync.
# For new code, prefer ``from plagih.config import cfg`` directly.
# ---------------------------------------------------------------------------
PRINT_DUMMY = _cfg.verbosity  # w=warning, a=action, g=generation, i=info, f=file, p=performance
TEXT_NEWLINE = "============================================================"
DEBUG_DUMMY = _cfg.debug
FLOAT_PRECISION = _cfg.float_precision
PLOTS_INTERVAL = _cfg.plots_interval
BACKUP_INTERVAL = _cfg.backup_interval
CHAIN_implement = "sfeh"  # used as placeholder for implementation-tasks
TREE_MIN_PARSIMONY = _cfg.tree_min_parsimony


# ---------------------------------------------------------------------------
# CPU utilities
# ---------------------------------------------------------------------------
def cpu_count_physical() -> int:
    """Return the number of *physical* CPU cores (not hyperthreads).

    os.cpu_count() returns logical cores (e.g. 16 on an 8-core CPU with
    hyperthreading). For CPU-bound work like GP evaluation, using more
    workers than physical cores hurts performance because hyperthreads
    share execution units and caches.

    Falls back to os.cpu_count() // 2 if psutil is unavailable.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical is not None:
            return physical
    except ImportError:
        pass
    # Fallback: assume hyperthreading (2 threads per core)
    logical = os.cpu_count()
    if logical is not None:
        return max(1, logical // 2)
    return 4


# ---------------------------------------------------------------------------
# Colors (full set — logging_utils uses a minimal subset internally)
# ---------------------------------------------------------------------------
class BColors:
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    HEADER = "\033[95m"
    FAIL = "\033[91m"
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[39m"
    RESET_COLOR = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def rnd_choice(lst):
    return lst[np.random.randint(0, len(lst))]


def xt_self(xtype):
    return xtype[1]


def xt_childs(xtype):
    return xtype[0]


def get_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from get_subclasses(sub)


def pickle_dump(path, data):
    """Saves data to a pickle file (.p or .pkl)."""
    path = path_make_dir(path)
    with Path.open(path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def yaml_dump(data, path: Path):
    """.yaml file writer."""
    path_make_dir(path)
    with Path.open(path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


def remove_trailing_zeroes(x):
    x = re.sub(r"\.0+$|0+$", "", x)
    return x


def term_format(x, cut=False):
    """Format a terminal value for display.

    Args:
        x: Value to format (string or number).
        cut: If True, use compact scientific notation for very small/large values.
    """
    try:
        if cut:
            x = float(x)
            if float(x) < 0.001 or float(x) > 1000:
                xstr = f"{x:.3g}"
            else:
                xstr = f"{x:.3f}"
                xstr = re.sub(r"\.0+$|0+$", "", xstr)
        else:
            xstr = remove_trailing_zeroes(x)
        return xstr
    except ValueError:
        return x  # constants/terms


def string_remove_trailing_zeroes(number_string: str) -> str:
    """
    2.00000000000 -> 2
    0.20000000000 -> 0.2
    sfeh: 1.1e+3 is 1100. Looks ugly like this.
    """
    cleaned_string = re.sub(r"(\.\d*?)0+(?!\d)", r"\1", number_string)
    cleaned_string = re.sub(r"\.(?!\d)", "", cleaned_string)
    return cleaned_string


def path_make_dir(p: Path):
    """Creates directories as needed for a given path."""
    folder = p if len(p.suffix) == 0 else p.parent
    folder.mkdir(parents=True, exist_ok=True)
    return p


def blue_string(txt):
    """Format string in cyan/blue color."""
    return f"{BColors.CYAN}{txt}{BColors.RESET_COLOR}"


def pickle_load(path: Path):
    """Loads a pickle file (.p or .pkl)."""
    with Path.open(path, "rb") as file:
        pickle_data = pickle.load(file)
    return pickle_data


def yaml_load(path: Path):
    """.yaml file loader."""
    with Path.open(path, "r") as file:
        loaded_yaml = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_yaml


# ---------------------------------------------------------------------------
# Matplotlib plot presets
# ---------------------------------------------------------------------------
pyplot_size = (3.6, 2.7)
plplot_size_up = (3.6, 3.6)

pyplot_rc_tex = {
    "figure.autolayout": True,
    "text.usetex": shutil.which("latex") is not None,
    "backend": "pgf",
    "figure.figsize": pyplot_size,
    "axes.labelpad": 0.5,
    "xtick.labelsize": 8,
    "xtick.major.size": 1.5,
    "xtick.major.pad": 1.5,
    "ytick.labelsize": 8,
    "ytick.major.size": 1.5,
    "ytick.major.pad": 1.5,
    "font.size": 10,
    "legend.fontsize": 9,
    "savefig.dpi": 600,
}

# sfeh
pyplot_size2 = (3, 2.1)
pyplot_rc_tex2 = {
    "figure.autolayout": True,
    "text.usetex": True,
    "backend": "pgf",
    "figure.figsize": pyplot_size2,
    "axes.labelpad": 0.5,
    "xtick.labelsize": 6,
    "xtick.major.size": 1.2,
    "xtick.major.pad": 1.2,
    "ytick.labelsize": 6,
    "ytick.major.size": 1.2,
    "ytick.major.pad": 1.2,
    "font.size": 9,
    "legend.fontsize": 8,
    "savefig.dpi": 600,
}

# https://www.elsevier.com/authors/policies-and-guidelines/artwork-and-media-instructions/artwork-sizing
plot_ratio = 9 / 16
plot_width_twocol = 9 / 25.4
pyplot_rc_two_column = {
    "figure.autolayout": True,
    "text.usetex": True,
    "backend": "pgf",
    "figure.figsize": (3.5433, 2),
    "axes.labelpad": 0.4,
    "xtick.labelsize": 7,
    "xtick.major.size": 1.2,
    "xtick.major.pad": 1.2,
    "ytick.labelsize": 7,
    "ytick.major.size": 1.2,
    "ytick.major.pad": 1.2,
    "font.size": 10,
    "legend.fontsize": 8,
    "savefig.dpi": 600,
    "savefig.pad_inches": 0,
    "lines.linewidth": 1,
    "lines.markersize": 3,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
}

rc_pyplot_size = {"figure.figsize": pyplot_size}


if __name__ == "__main__":
    print("Testing the plot style")
    import matplotlib.pyplot as plt
    import numpy as np

    x = range(10)
    y = np.sin(np.arange(10) / 10) + np.arange(10) / 10
    with plt.rc_context(rc=pyplot_rc_two_column):
        fig, ax = plt.subplots()
        ax.plot(x, y, marker="x", label="random values (idk)")
        ax.set(xlabel="some label", ylabel="some other value")
        ax.legend(loc="lower left")
        plt.show()
