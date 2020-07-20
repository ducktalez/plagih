import sys

sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current root_dir

from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.plagih_data import *
import yaml

# import warnings
# warnings.filterwarnings('error')

# /info/
file_info_evolve_dict_yaml = 'info/evolve_list.yaml'

info_distributions_yaml = 'info/distributions_file.yaml'
env_vars_yaml = 'info/env_vars.yaml'

# /run_files/


def plagih_config_update_from_yaml(config_yaml=Path('config.yaml')):
    """
    The config gets updated
    """
    with Path.open(config_yaml, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def plagih_config_update_from_json(config_json=Path('config.json')):
    """
    The config gets updated
    """
    with Path.open(config_json, 'r') as file:
        config = json.load(file)
    return config


def labellists_from_csv(csv_path):
    """
    from the labellist-csv, loading the label list
    """
    modify_list = []
    with Path.open(csv_path, newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if len(row) > 0:
                if row[0] == 'label_list' or row[0] == 'node_label':
                    label_list = [x.replace(' ', '') for x in row[1:]]
                elif row[0] == 'modify_list' or row[0] == 'node_modify':
                    modify_list = [int(x.replace(' ', '')) for x in row[1:]]
                elif row[0] == '':
                    pass
                else:
                    print_warning('ww', f'Unexpected row start: {row[0]}')

    if label_list is None:
        raise Exception('Labels could not be created from file.')

    return label_list, modify_list


def show_default_config(output_file):
    """
    - config.yaml
    """
    if not Path.is_dir(output_file.parent):
        raise Exception(f'Will not make the path with Path.mkdir(output_file.parent): {output_file}')
    raise Exception('Coming soon! sfeh.')


def show_default_operators(output_file=None):
    """
    - operators.csv
    """

    if not Path.is_dir(output_file.parent):
        raise Exception(f'Will not make the path with Path.mkdir(output_file.parent): {output_file}')
    raise Exception('Coming soon! sfeh.')


def runfolder_exists(root_dir):
    """
    If there is no folder, bad times
    """
    if not Path.is_dir(root_dir):
        raise FileNotFoundError(f'Folder does not exist: {root_dir}.')


def load_label_list(root_dir, user_origin_csv=None):
    """

    """
    tree_expr_txt_path = root_dir / 'run_files/tree_expr.txt'
    tree_labels_csv_path = root_dir / 'run_files/tree_labels.csv'
    tree_numpy_csv_path = root_dir / 'run_files/tree_numpy.csv'
    label_list = None
    modify_list = None
    if user_origin_csv:
        label_list, modify_list = labellists_from_csv(user_origin_csv)
    elif Path.is_file(tree_labels_csv_path):
        label_list, modify_list = labellists_from_csv(tree_labels_csv_path)  # sfeh lol it does just the same?
    elif Path.is_file(tree_numpy_csv_path):  # Load origin tree
        print('SFEH I dont think anyone will want to use this. Create a tree from label_list, ffs.')
        raise
    elif Path.is_file(tree_expr_txt_path):  # karoo_tree_from_expr(expr)
        print('SFEH needs to create an option to make trees from expression')
        with Path.open(tree_expr_txt_path) as txt_file:
            expr = txt_file.read()  # sfeh requires separate handling?
            print('Assuming all variables are floats, sfeh')
            tree = karoo_tree_from_expr(expr, 'sfeh')
            tree_pretty_print(tree)
            tree_save_csv(tree, tree_labels_csv_path)
            raise  # sfeh
    else:
        print_warning('ii', 'No origin-tree file was provided. Continuing.')
    return label_list, modify_list


def load_config(config_path, out_dir=None):
    """

    """
    try:
        file_extension = config_path.suffix
        if file_extension == '.yaml':
            with Path.open(config_path, 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            config = {}  # sfeh test this
    except IOError as ioex:
        raise IOError(f'Config file could not be loaded. Path: {config_path}\nException: {ioex}')

    if out_dir:
        root_dir = out_dir
    else:
        root_dir = config_path.parent

    return root_dir, config


def gp_run(plagih_root, load_backup, config_path, out_dir, force_new_run, eval_action, path_data, cooltree_origin, analyze=False, tf_device_log=False):
    """
    
    """

    root_dir, config = load_config(config_path, out_dir=out_dir)

    config['force_new_run'] = force_new_run
    if eval_action is not None:
        config['eval_action'] = eval_action

    gp = ExplainableGP(plagih_root, root_dir, config, eval_action, path_data=path_data, tf_device_log=tf_device_log)

    label_list, modify_list = load_label_list(root_dir, user_origin_csv=cooltree_origin)
    if label_list is not None and modify_list is not None:
        # env_vars = gp.get_env_vars()

        cooltree_origin = cooltree_from_labellist(label_list, modify_list=modify_list)
        gp.activate_origin_tree(cooltree_origin)

    gp.make_evolve_rates()

    if analyze:
        gp.gp_analyze()
    else:
        gp.make_evolve_rates()
        gp.plagih_gp_run()

    sys.exit()
