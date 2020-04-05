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


def runfolder_exists(root_dir):
    """
    If there is no folder, bad times
    """
    if not Path.is_dir(root_dir):
        raise FileNotFoundError('Folder does not exist: {}.'.format(root_dir))


def load_config(root_dir):
    config_yaml_path = root_dir / file_config_yaml
    if Path.is_file(config_yaml_path):  # Load config.yaml
        with Path.open(config_yaml_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        printez('i', 'Loading json-config, yaml-version was not found...')
        config_json_path = root_dir / file_config_json
        if Path.is_file(config_json_path):  # Load config.yaml
            with Path.open(config_yaml_path, 'r') as file:
                json.load(file)
        else:
            print_warning('w', 'You should have a {} here:\n{}'.format(file_config_yaml, config_yaml_path))
            config = {}
    return config


def load_data_prepared(root_dir):
    if Path.is_file(root_dir / samples_ready_p):
        data_prepared = pickle_load(root_dir / samples_ready_p)
    elif Path.is_file(root_dir / samples_csv):
        data_prepared = data_from_csv(root_dir / samples_csv)
        print('Prepared the raw {} behaviour. Saving for next run.'.format(samples_csv))
        pickle_dump(root_dir / samples_ready_p, data_prepared)
        yaml_dump(root_dir / env_variables_yaml, data_prepared)
    else:
        raise FileNotFoundError('No data provided? Please provide {} or {}.'.format(samples_ready_p, samples_csv))

    # if Path.is_file(env_variables_yaml):
    #     env_variables, param_at = yaml_load(env_variables_yaml)
    #     pass
    #     # todo

    return data_prepared


def load_tree_elements(root_dir):

    # Load operators
    operators_csv = root_dir / operators
    if Path.is_file(operators_csv):  # Load operators.csv
        functions = np.loadtxt(operators_csv, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators)
    else:
        # raise FileNotFoundError('File does not exist: {}.'.format(operators_csv))
        print_warning('w', 'Operators-file does not exist. Creating one with a default list of mathematical operators.')
        functions = np.array([['+', 2], ['-', 2], ['*', 2], ['/', 2],['Mini', 2], ['Maxi', 2],['<', 2], ['<=', 2],
                              ['==', 2], ['abs', 2], ['Andb', 2], ['Orb', 2], ['Not', 2], ['sin', 2], ['Ifte', 2]])
        np.savetxt(operators_csv, functions, delimiter=',', fmt='%s')
    choose_oparray = funcarray_from_list(functions)

    # load distributions_file
    distributions_yaml = root_dir / distributions_file
    if Path.is_file(distributions_yaml):
        with Path.open(Path(distributions_yaml), 'r') as file:
            distributions_as_string = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print_warning('w', 'Distributions file does not exist.')
        distributions_as_string = {'2f': ['lambda: np.random.normal(1,2)', 'lambda: np.random.normal(1,1)', 'lambda: np.random.randint(0, 10)'],
                                   '2b': ['lambda: np.random.choice([True, False])']}  # sfeh
        with Path.open(Path(distributions_yaml), 'w') as file:
            _ = yaml.dump(distributions_as_string, file)
    distributions = {}
    distributions['2f'] = [eval(x) for x in distributions_as_string['2f']]
    distributions['2b'] = [eval(x) for x in distributions_as_string['2b']]

    return choose_oparray, distributions


def load_label_list(root_dir):
    tree_expr_txt_path = root_dir / tree_expr_txt
    tree_labels_csv_path = root_dir / tree_labels_csv
    tree_numpy_csv_path = root_dir / tree_numpy_csv
    label_list = None
    modify_list = None
    if Path.is_file(tree_labels_csv_path):
        label_list, modify_list = labellists_from_csv(tree_labels_csv_path)
    elif Path.is_file(tree_numpy_csv_path):  # Load origin tree
        print('SFEH I dont think anyone will want to use this. Create a tree from label_list, ffs.')
        raise
    elif Path.is_file(tree_expr_txt_path):  # karoo_tree_from_expr(expr)
        print('SFEH needs to create an option to make trees from expression')
        raise
    else:
        print_warning('ww', 'No origin-tree file was provided. Continuing.')
    return label_list, modify_list


def analyze(root_dir):
    """
    write all analysing files.
    - pareto (txt, latex_trees, agents)
    - plots (pareto, best)
    """

    config = load_config(root_dir)
    data_prepared = load_data_prepared(root_dir)

    gp = ExplainableGP(root_dir, config=config)
    gp.activate_dataset(data_prepared)
    gp.plagih_update_analysis()


def run(root_dir):
    """
    Loads important files in your run-folder
    - load config.yaml
    - load samples_ready_p.p or samples.csv
    - load operators.csv
    - load tree
    """

    config = load_config(root_dir)
    data_prepared = load_data_prepared(root_dir)
    choose_oparray, distributions = load_tree_elements(root_dir)
    label_list, modify_list = load_label_list(root_dir)

    gp = ExplainableGP(root_dir, config=config)
    gp.activate_dataset(data_prepared)
    gp.activate_operators(choose_oparray, distributions)

    if label_list is not None and modify_list is not None:
        observation_bundle = gp.get_observation_bundle()
        origin_tree = karoo_tree_from_labellist(label_list, observation_bundle, modify_list=modify_list)
        gp.activate_origin_tree(origin_tree)

    gp.plagih_gp_run()
    sys.exit()


def visualize_labellist(csv_file, output_file=None):
    """
    visualize a label list
    e.g. if you want to check, if you made your tree correctly
    """
    print('can not create tree from label list anymore, xtype_list required')
    # if Path.is_file(csv_file):
    #     label_list, modify_list = labellists_from_csv(csv_file)
    #     tree = karoo_tree_from_labellist(label_list, modify_list=modify_list)
    #     forest_input = latex_tree_get_forest(tree)
    #     latex_full_doc = latex_complete_tree_summary(forest_input)
    #     if not output_file:
    #         output_file = csv_file.with_suffix('.tex')
    #     with Path.open(output_file, 'w') as csv_file:
    #         csv_file.write(latex_full_doc)
    # else:
    #     print_e('File {}  does not exist!'.format(csv_file))


if __name__ == "__main__":
    runs_dir = Path.cwd() / '../{}'.format(example_runs)
    root_dir = runs_dir / 'cartpole_v1/'
    run(root_dir)
