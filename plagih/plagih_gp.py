import sys

sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current root_dir

from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.plagih_data import *
from plagih.modules.operators import oparray_from_list
import yaml

# import warnings
# warnings.filterwarnings('error')

# /info/
file_info_evolve_dict_yaml = 'info/evolve_list.yaml'

info_distributions_yaml = 'info/distributions_file.yaml'
env_variables_yaml = 'info/env_variables.yaml'

# /run_files/
file_config_yaml = 'run_files/config.yaml'
file_config_json = 'run_files/config.json'
file_evolve_functions = 'run_files/evolve_functions.yaml'
samples_csv = 'run_files/samples.csv'
distributions_file = 'run_files/distributions_file.yaml'
tree_expr_txt = 'run_files/tree_expr.txt'
tree_labels_csv = 'run_files/tree_labels.csv'
tree_numpy_csv = 'run_files/tree_numpy.csv'


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
                    print_warning('ww', 'Unexpected row start: {}'.format(row[0]))

    if label_list is None:
        raise Exception('Labels could not be created from file.')

    return label_list, modify_list


def show_default_config(output_file):
    """
    - config.yaml
    """
    if not Path.is_dir(output_file.parent):
        raise Exception('Will not make the path with Path.mkdir(output_file.parent): {}'.format(output_file))
    raise Exception('Coming soon! sfeh.')


def show_default_operators(output_file=None):
    """
    - operators.csv
    """

    if not Path.is_dir(output_file.parent):
        raise Exception('Will not make the path with Path.mkdir(output_file.parent): {}'.format(output_file))
    raise Exception('Coming soon! sfeh.')


def runfolder_exists(root_dir):
    """
    If there is no folder, bad times
    """
    if not Path.is_dir(root_dir):
        raise FileNotFoundError('Folder does not exist: {}.'.format(root_dir))


def load_startconfig(root_dir):
    """
    todo
    single file that can specify
    - the files to be loaded (which can be somewhere else)
    - the action for the regression
    - the
    """
    # loading_configuration = root_dir / 'loading.yaml'
    #
    # if Path.is_file(loading_configuration):  # Load config.yaml
    #     with Path.open(loading_configuration, 'r') as file:
    #         config = yaml.load(file, Loader=yaml.FullLoader)
    # else:
    #     print_warning('w', 'You can specify the files that are loaded here blabla sfeh')
    # return config


def load_label_list(root_dir, user_origin_csv=None):
    """

    """
    tree_expr_txt_path = root_dir / tree_expr_txt
    tree_labels_csv_path = root_dir / tree_labels_csv
    tree_numpy_csv_path = root_dir / tree_numpy_csv
    label_list = None
    modify_list = None
    if user_origin_csv:
        label_list, modify_list = labellists_from_csv(user_origin_csv)
    elif Path.is_file(tree_labels_csv_path):
        label_list, modify_list = labellists_from_csv(tree_labels_csv_path)  # sfeh lol it does just the same?
    elif Path.is_file(tree_numpy_csv_path):  # Load origin tree
        print('SFEH I dont think anyone will want to use this. Create a tree from label_list, ffs.')
        raise
    elif Path.is_file(tree_expr_txt_path):  # karoo_ptree_from_expr(expr)
        print('SFEH needs to create an option to make trees from expression')
        with Path.open(tree_expr_txt_path) as txt_file:
            expr = txt_file.read()  # sfeh requires separate handling?
            print('Assuming all variables are floats, sfeh')
            ptree = karoo_ptree_from_expr(expr, 'sfeh')
            tree = ptree.get_uninstanced_tree()
            tree_pretty_print(tree)  # sfeh not working?? todo debug
            tree_save_csv(tree, tree_labels_csv_path)
            raise  # sfeh
    else:
        print_warning('ii', 'No origin-tree file was provided. Continuing.')
    return label_list, modify_list


def load_config(config_path):
    """

    """
    try:
        root_dir = config_path.parent
        file_extension = config_path.suffix
        if file_extension == '.yaml':
            with Path.open(config_path, 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            config = {}  # sfeh test this
    except IOError as ioex:
        raise IOError('Config file could not be loaded. Path: {}\nException: {}'.format(config_path, ioex))

    return root_dir, config


def gp_run(plagih_root, load_backup, config_path, out_dir, force_new_run, eval_action, data_prepared_path, origin_tree):
    """
    
    """

    root_dir, config = load_config(config_path)
    if out_dir:
        root_dir = out_dir

    config['force_new_run'] = force_new_run
    config['eval_action'] = eval_action

    # sfeh , opt_origin_tree_csv=origin_tree, out_dir=out_dir
    gp = ExplainableGP(plagih_root, root_dir, config, user_prepared_path=data_prepared_path)

    label_list, modify_list = load_label_list(root_dir, user_origin_csv=origin_tree)
    if label_list is not None and modify_list is not None:
        # todo beautify
        env_variables = gp.get_env_variables()
        xtype_list = xtypes_from_labels(label_list, env_variables)
        origin_ptree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list)
        gp.activate_origin_tree(origin_ptree)

    gp.prepare_evolve_functions()

    gp.plagih_gp_run()
    sys.exit()


def analyse(plagih_root, load_backup, config_path, out_dir, force_new_run, eval_action, data_prepared_path, origin_tree):
    """
    write all analysing files.
    - pareto (txt, latex_trees, agents)
    - plots (pareto, best)
    """
    # todo the same code is in gp_run...
    root_dir, config = load_config(config_path)
    if out_dir:
        root_dir = out_dir

    config['force_new_run'] = force_new_run
    config['eval_action'] = eval_action

    root_dir, config = load_config(config_path)
    gp = ExplainableGP(plagih_root, root_dir, config, user_prepared_path=data_prepared_path)

    gp.plagih_update_analysis()


if __name__ == "__main__":
    """
    
    """
    runs_dir = Path.cwd() / '../run_examples/'
    root_dir = runs_dir / 'cartpole_v1/'  # todo this is outdated
    gp_run(root_dir, False)
