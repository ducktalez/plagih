import pickle
from plagih.printing import *
import yaml
from pathlib import Path

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
        printez('f', f'{path.as_posix()}', print_type=print_type)
    return


def yaml_load(yaml_path):
    """
    .yaml-file loader (saves two lines that I had to look up all the time)
    """
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
        printez('ff', f'{path.as_posix()}', print_type=print_type)
    return


def plot_sexyfy(x, y, set_left=None, set_right=None, set_top=None, right_padding=1.05, top_padding=1.05):
    """

    """
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    if set_left:
        left = (x[0], y[0])

    if set_top:
        new_top = set_top
    else:
        new_top = (top - min(bottom, 0)) * top_padding  # top * 1.05 for better style

    if set_right:
        right = max(right, set_right)
    new_right = right * right_padding

    return top, bottom, left, right, new_right, new_top
