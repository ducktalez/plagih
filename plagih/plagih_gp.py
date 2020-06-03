import sys

sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current root_dir

from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.plagih_data import *
from plagih.modules.operators import oparray_from_list
import yaml

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
    - operators_csv.csv
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


def load_config(root_dir):
    config_yaml_path = root_dir / file_config_yaml

    if Path.is_file(config_yaml_path):  # Load config.yaml
        with Path.open(config_yaml_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        config_json_path = root_dir / file_config_json

        if Path.is_file(config_json_path):  # Load config.yaml
            printez('i', 'Loading json-config, yaml-version was not found...')
            with Path.open(config_yaml_path, 'r') as file:
                json.load(file)
        else:
            print_warning('w', 'No config (yaml/json) found in:\n{}'.format(config_yaml_path))
            config = {}
    return config


def load_data_prepared(root_dir, delimiter=',', print_type=None):
    """
    loading the data which the GP will be working on.
    The .csv-file is prepared (loading correct data-type, splitting data, ...)
    and saved as pickle-file for reloading runs.
    This is especially important, as the split in training and test-data must be the same.
    """
    if Path.is_file(root_dir / samples_ready_p):  # maybe the data was already prepared earlier
        data_prepared = pickle_load(root_dir / samples_ready_p)
    elif Path.is_file(root_dir / samples_csv):  # We have to split and check the data
        # preparing the data from raw csv-file
        env_variables, data_train_panda, data_test_panda, data_train_numpy, data_test_numpy = data_from_csv(root_dir / samples_csv, delimiter=delimiter)
        data_prepared = env_variables, data_train_numpy, data_test_numpy  # sfeh version 1.0 remove numpy version
        print('Prepared the raw {} behaviour. Saving for next run.'.format(samples_csv))
        pickle_dump(root_dir / samples_ready_p, data_prepared)
    else:
        raise FileNotFoundError('No data provided? Please provide {} or {}.'.format(samples_ready_p, samples_csv))

    dataspec_file = root_dir / 'run_files/data_specification.yaml'
    if dataspec_file.is_file():
        pass  # sfeh: if you want to load information from extra file, check for this file here
    else:
        # sfeh env_variables, _, _ = data_prepared. anyways, currently loading info via brackets in .csv-file
        # todo only, if these files are meant to be created. (not in gen 0, make special operator for this. wird eh überschrieben)
        yaml_dump(root_dir / env_variables_yaml, data_prepared[0], print_type=print_type)

    return data_prepared


def load_evolve_functions(root_dir, evolve_file=file_evolve_functions):
    """
    Return the gp evolve-rates for the genetic programming process.
    Can be user-defined in the evolve_file, otherwise the default is used...
    """
    if Path.is_file(root_dir / evolve_file):
        evolve_list = yaml_load(root_dir / evolve_file)
    else:
        print_warning('ww', 'Opt-in not specified. Evolve-file for GP evolve functions defined! Trying to choose them for you.')

        evolve_list = [
            # Reproduction (15%)
            {'tag': 'Repro', 'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.10,
             'custom_params': {}},
            {'tag': 'Rsympy', 'evolve_name': 'reproduce', 'evolve_rate': 0.03,
             'custom_params': {'sympify_tree': True}},
            {'tag': 'Pareto', 'evolve_name': 'revive pareto', 'evolve_rate': 0.02,
             'custom_params': {}},

            # Mutation (35%)
            {'tag': 'Point', 'evolve_name': 'mutate point', 'evolve_rate': 0.10,
             'custom_params': {}},  # sum 0.25
            {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
             'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 2, 5, 0.8), 'build_method': 'full'}}},
            {'tag': 'BranchDG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
             'custom_params': {'build_spec': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 2, 6, 1), 'build_method': 'grow'}}},
            {'tag': 'BranchNG', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
             'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (10, 6, 24, 4), 'build_method': 'grow'}}},
            {'tag': 'FilterB', 'evolve_name': 'filter optimize', 'evolve_rate': 0.10,
             'custom_params': {'mode': 'branch'}},
            {'tag': 'FilterP', 'evolve_name': 'filter optimize', 'evolve_rate': 0.0,
             'custom_params': {'mode': 'point'}},

            # Crossover (30%)
            {'tag': 'Xover', 'evolve_name': 'crossover branch', 'evolve_rate': 0.30,  # sum 0.70
             'custom_params': {}},

            # Random (25%)
            {'tag': 'Rand1', 'evolve_name': 'random trees', 'evolve_rate': 0.10,
             'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'build_method': 'full'}}},
            {'tag': 'Rand2', 'evolve_name': 'random trees', 'evolve_rate': 0.10,
             'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 6, 1), 'build_method': 'grow'}}},
            {'tag': 'Rand3', 'evolve_name': 'random trees', 'evolve_rate': 0.05,
             'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (20, 12, 45, 6), 'build_method': 'full'}}},
        ]

        # todo todotodo
        evolve_list_first = {
            'from_origin': [
                {'tag': 'RandO3', 'evolve_name': 'random trees', 'evolve_rate': 1.00,
                 'custom_params': {'build_spec': {'size_mode': 'branch_nodes', 'mean_min_max_var': (20, 10, 45, 6), 'build_method': 'full'}}}
            ],
            'from_scratch': [
                {'tag': 'Rand1', 'evolve_name': 'random trees', 'evolve_rate': 0.30,
                 'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'build_method': 'full'}}},
                {'tag': 'Rand2', 'evolve_name': 'random trees', 'evolve_rate': 0.30,
                 'custom_params': {'build_spec': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 6, 1), 'build_method': 'grow'}}},
                {'tag': 'Rand3', 'evolve_name': 'random trees', 'evolve_rate': 0.40,
                 'custom_params': {'build_spec': {'size_mode': 'tree_nodes', 'mean_min_max_var': (20, 10, 45, 6), 'build_method': 'full'}}}
            ],
        }

    yaml_dump(root_dir / file_info_evolve_dict_yaml, evolve_list)
    # sfeh: if you want to load information from extra file, check for this file here

    return evolve_list


def load_tree_builders(root_dir, data_prepared=None):
    """

    """
    # Load operators_csv
    path_opyaml = root_dir / operators_yaml
    # if Path.is_file(operators_csv):  # sfeh test yaml
    #     functions = np.loadtxt(operators_csv, delimiter=',', skiprows=1, dtype=str)  # load the user defined functions (operators_csv)

    if Path.is_file(path_opyaml):
        operators = yaml_load(path_opyaml)
    else:
        # raise FileNotFoundError('File does not exist: {}.'.format(operators_csv))
        print_warning('ww', 'Opt-in not specified. Operators-file does not exist. Creating one with a default list of mathematical operators_csv.')
        operators = np.array([['+', 3],
                              ['-', 1], ['usub', 2],
                              ['*', 2], ['/', 1],
                              ['Square', 0.75], ['**', 0.25],
                              ['abs', 0.5], ['sign', 0.5],
                              ['sqrt', 0.2],
                              ['log', 0.1], ['log1p', 0.1],
                              ['cos', 0.33], ['sin', 0.1], ['tan', 0.1],  # ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
                              ['tanh', 0.2],
                              ['Andb', 1], ['Orb', 1], ['Xor', 1], ['Notb', 0.5],
                              ['==', 1], ['!=', 0.5],
                              ['<', 0.5], ['<=', 0.5], ['>', 0.5], ['>=', 0.5],
                              ['Ifte', 2],
                              ['Mini', 1],
                              ['Maxi', 1]])

        # np.savetxt(operators_csv, functions, delimiter=',', fmt='%s')

    choose_oparray, choose_oparray2 = oparray_from_list(operators)

    # load distributions_file
    distributions_yaml = root_dir / distributions_file
    if Path.is_file(distributions_yaml):
        with Path.open(Path(distributions_yaml), 'r') as file:
            distributions_as_string = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print_warning('ww', 'Opt-in not specified. Distributions-file (for random leaf-node constants) does not exist. Using default set.')
        distributions_as_string = {'2f': ['lambda: np.random.normal(1,2)',
                                          'lambda: np.random.normal(1,1)',
                                          'lambda: np.random.randint(0, 10)'],
                                   '2b': ['lambda: np.random.choice([True, False])'],
                                   'observed_floats': 100}
        info_file = file_make_dir(root_dir / info_distributions_yaml)
        yaml_dump(info_file, distributions_as_string)

    distributions = {'2f': [], '2b': []}

    if data_prepared and distributions_as_string.get('observed_floats'):
        env_variables, data_train, _ = data_prepared
        action_columns = list(range(len(env_variables['obs_name']), len(data_train[0])))  # remove these
        # non_float_columns = ... # sfeh: data types must be float for this to work, remove non-float values. probably, these do not really exist.
        observ_values = np.delete(data_train, action_columns, 1)
        variables_set = np.random.choice(observ_values.flatten(), distributions_as_string.get('observed_floats'))  # 2nd param is probably '100'
        distributions['2f'].extend([lambda: np.random.choice(variables_set)]),

    distributions['2f'].extend([eval(x) for x in distributions_as_string['2f']]),
    distributions['2b'].extend([eval(x) for x in distributions_as_string['2b']])

    return choose_oparray, choose_oparray2, distributions


def load_label_list(root_dir):
    """

    """
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
    elif Path.is_file(tree_expr_txt_path):  # karoo_ptree_from_expr(expr)
        print('SFEH needs to create an option to make trees from expression')
        with Path.open(tree_expr_txt_path) as txt_file:
            expr = txt_file.read()  # sfeh requires separate handling?
            print('Assuming all variables are floats, ö')
            ptree = karoo_ptree_from_expr(expr, 'ö')
            tree = ptree.get_uninstanced_tree()
            print('ASD', tree)
            # tree_pretty_print(tree)  # sfeh not working??
            print('ASD DONE')
            tree_save_csv(tree, tree_labels_csv_path)
            raise  # sfeh
    else:
        print_warning('ii', 'No origin-tree file was provided. Continuing.')
    return label_list, modify_list


def gp_run(root_dir, force_new_run, eval_action):
    """
    Loads important files in your run-folder
    - load config.yaml
    - load samples_ready_p.p or samples.csv
    - load operators_csv.csv
    - load tree
    """
    # start_configuration = load_startconfig(root_dir)

    config = load_config(root_dir)

    if force_new_run:  # for convenience. Makes restarting runs possible from command line
        config['force_new_run'] = True
    config['eval_action'] = eval_action

    gp = ExplainableGP(root_dir, config=config)

    data_prepared = load_data_prepared(root_dir)
    gp.activate_dataset(data_prepared)

    choose_oparray, choose_oparray2, distributions = load_tree_builders(root_dir, data_prepared=data_prepared)
    gp.activate_operators(choose_oparray, choose_oparray2, distributions)

    label_list, modify_list = load_label_list(root_dir)
    if label_list is not None and modify_list is not None:
        env_variables = gp.get_env_variables()
        xtype_list = xtypes_from_labels(label_list, env_variables)
        origin_ptree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list)
        gp.activate_origin_tree(origin_ptree)

    evolve_list = load_evolve_functions(root_dir)
    gp.activate_evolve_functions(evolve_list)

    gp.plagih_gp_run()
    sys.exit()


def analyse(root_dir):
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


if __name__ == "__main__":
    """
    
    """
    runs_dir = Path.cwd() / '../{}'.format(example_runs)
    root_dir = runs_dir / 'cartpole_v1/'  # todo this is outdated
    gp_run(root_dir, False)
