import sys
sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current root_dir

import plagih.modules.plagih_gp_base_class_xai as plagih
from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.Examples import *
from plagih.modules.plagih_data import *
import yaml

# todo clean up this class... make extra class or folder (!) with all cases to be tested
# todo idee: nach generationen alle mit einem gen sterben lassen! epidemie!
# import warnings
# warnings.filterwarnings('error')


def plagih_config_update_from_yaml(config_yaml='config.yaml'):
    """
    The config gets updated
    """
    with Path.open(config_yaml, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def plagih_config_update_from_yaml(config_yaml='config.yaml'):
    """
    The config gets updated
    """
    with Path.open(config_yaml, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


# create_samples_pickle_prepared(root_dir, CartpoleExamples.files['samples_file'])  # todo outsource
# analyse_old_run(root_dir)

def labellists_from_csv(csv_path):
    modify_list = []
    with Path.open(csv_path, newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            if len(row) > 0:
                if row[0] == 'label_list':
                    label_list = row[1:]
                elif row[0] == 'modify_list':
                    modify_list = [int(x) for x in row[1:]]  # N_modify sfeh
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
    print('Coming soon! sfeh.')


def show_default_operators(output_file=None):
    """
    - operators.csv
    """

    if not Path.is_dir(output_file.parent):
        raise Exception('Will not make the path with Path.mkdir(output_file.parent): {}'.format(output_file))
    print('Coming soon! sfeh.')


def run(root_dir):
    """
    Loads important files in your run-folder
    - load config.yaml
    - load samples_ready.p or samples.csv
    - load operators.csv
    - load tree
    """

    runfiles_dir = root_dir / run_files

    config_yaml_path = runfiles_dir / config_yaml

    samples_ready_path = runfiles_dir / samples_ready
    samples_csv_path = runfiles_dir / samples_csv

    operators_csv = runfiles_dir /operators

    tree_expr_txt_path = runfiles_dir / tree_expr_txt
    tree_labels_csv_path = runfiles_dir / tree_labels_csv
    tree_numpy_csv_path = runfiles_dir / tree_numpy_csv

    if not Path.is_dir(root_dir):
        raise FileNotFoundError('Folder does not exist: {}.'.format(root_dir))

    if Path.is_file(config_yaml_path):  # Load config.yaml
        with open(config_yaml_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print_warning('w', 'You should have a {} here:\n{}'.format(config_yaml, config_yaml_path))
        config = {}

    if Path.is_file(samples_ready_path):
        data_prepared = data_load_pickle(samples_ready_path)
    elif Path.is_file(samples_csv_path):
        data_prepared = data_from_csv(samples_csv_path)
        print('Prepared the raw {} behaviour. Saving for next run.'.format(samples_csv))
        data_save_pickle(data_prepared, samples_ready_path)
    else:
        raise FileNotFoundError('No data provided? Please provide {} or {}.'.format(samples_ready, samples_csv))

    if Path.is_file(operators_csv):  # Load operators.csv
        functions = np.loadtxt(operators_csv, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
        op_array = load_funcarray_from_list(functions)
    else:
        raise FileNotFoundError('File does not exist: {}.'.format(operators_csv))

    origin_tree = None
    if Path.is_file(tree_labels_csv_path):
        label_list, modify_list = labellists_from_csv(tree_labels_csv_path)

        origin_tree = karoo_tree_from_labellist(label_list, modify_list=modify_list)
    elif Path.is_file(tree_numpy_csv_path):  # Load origin tree
        print('SFEH I dont think anyone will want to use this. Create a tree from label_list, ffs.')
        raise
    elif Path.is_file(tree_expr_txt_path):  # karoo_tree_from_expr(expr)
        print('SFEH needs to create an option to make trees from expression')
        raise
    else:
        print_warning('ww', 'No origin-tree file was provided. Continuing.')

    # config, data_prepared, op_array, origin_tree

    gp = ExplainableGP(root_dir, config=config)
    gp.activate_dataset(data_prepared)
    gp.activate_operators(op_array)
    if origin_tree is not None:
        gp.activate_origin_tree(origin_tree)

    gp.plagih_gp_run()
    sys.exit()


def visualize_labellist(csv_file, output_file=None):
    """
    visualize a label list
    e.g. if you want to check, if you made your tree correctly
    """
    if Path.is_file(csv_file):
        label_list, modify_list = labellists_from_csv(csv_file)
        tree = karoo_tree_from_labellist(label_list, modify_list=modify_list)
        forest_input = tree_get_latex_forest(tree)
        latex_full_doc = latex_complete_tree_summary(forest_input)
        if not output_file:
            output_file = csv_file.with_suffix('.tex')
        with Path.open(output_file, 'w') as csv_file:
            csv_file.write(latex_full_doc)
    else:
        print_e('File {}  does not exist!'.format(csv_file))


if __name__ == "__main__":
    runs_dir = Path.cwd() / '../{}'.format(example_runs)
    root_dir = runs_dir / 'cartpole_v1/'
    run(root_dir)
