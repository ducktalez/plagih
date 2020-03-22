import sys
sys.path = ['..'] + sys.path
sys.path.append('modules/')  # add directory 'modules' to the current root_dir

import json
import plagih.modules.plagih_gp_base_class_xai as plagih
from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.Examples import *
from plagih.modules.plagih_data import *

# todo clean up this class... make extra class or folder (!) with all cases to be tested
# todo idee: nach generationen alle mit einem gen sterben lassen! epidemie!
# import warnings
# warnings.filterwarnings('error')


def create_samples_pickle_prepared(path, behaviour_samples_file, pickle_file='prepared_samples.p'):
    """
    Saving your data file
    """
    prepared_data = data_from_csv(path / behaviour_samples_file)
    data_save_pickle(prepared_data, path / pickle_file)
    return


def mountaincar_load_corefiles(config_dict, path):
    gp = plagih.ExplainableGP(config_dict)
    prepared_data = data_load_pickle(path / MountainCarExamples.files['samples_pickle'])
    gp.activate_dataset(prepared_data)
    op_array = load_funcarray_from_csv(path / MountainCarExamples.files['operators_file'])
    gp.activate_operators(op_array)
    return gp


def cartpole_load_corefiles(config_dict, path):
    gp = plagih.ExplainableGP(config_dict)
    prepared_data = data_load_pickle(path / CartpoleExamples.files['samples_pickle'])
    gp.activate_dataset(prepared_data)
    op_array = load_funcarray_from_csv(path / CartpoleExamples.files['operators_file'])
    gp.activate_operators(op_array)
    return gp


def run_mountaincar_v1(config_dict, path):
    description = 'Mountaincar from 2-decision Origin (simple)'
    config_dict['name'] = 'MTC_v1'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.activate_origin_tree(label_list=MountainCarExamples.tree_v1_list, modify_list=MountainCarExamples.tree_v1_modify)
    return gp


def run_mountaincar_v2(config_dict, path):
    config_dict['name'] = 'MTC_v2'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.activate_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def run_mountaincar_v3(config_dict, path):
    config_dict['name'] = 'MTC_v3'
    config_dict['root_dir'] = path
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.activate_origin_tree(label_list=MountainCarExamples.tree_v3_list, modify_list=MountainCarExamples.tree_v3_modify)
    return gp


def run_mountaincar_v4(config_dict, path):
    config_dict['name'] = 'MTC_v4_scratch'
    config_dict['root_dir'] = path
    config_dict['kernel_name'] = 'regression bounded'
    config_dict['evolve_rates']['random from scratch'] += config_dict['evolve_rates']['random from origin_tree']
    config_dict['evolve_rates']['random from origin_tree'] = 0
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.activate_origin_tree(label_list=MountainCarExamples.tree_v1_list)
    return gp


def analyse_old_run(root_dir):
    config_dict = create_config_dict_default()
    mountaincar_update_analysis_files(config_dict, root_dir)


def mountaincar_update_analysis_files(config_dict, path):
    all_runs = ['MTC_v1', 'MTC_v2', 'MTC_v3', 'MTC_v4']

    config_dict['root_dir'] = path
    config_dict['mode'] = 'analyse'

    config_dict['name'] = 'MTC_v4_scratch'
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.plagih_update_analysis()
    return


def run_mountaincar_test(config_dict, path):
    config_dict['name'] = 'MTC_test'
    # test_run = Path.cwd() / example_runs / config_dict['name']
    # print('Test dir is:', test_run)
    config_dict['root_dir'] = path
    config_dict['pop_max'] = 100
    config_dict['gen_max'] = 500
    config_dict['tourn_size'] = 3
    config_dict['kernel_name'] = 'regression bounded'
    config_dict['parsimony_max'] = 50
    config_dict['print_type'] = 'ewaaaggggsiiiivvvtopppttt'    # To print_type absolutely all: ewaaaiiiiggggvvvpppttt
    gp = mountaincar_load_corefiles(config_dict, path)
    gp.activate_origin_tree(label_list=MountainCarExamples.tree_v2_list, modify_list=MountainCarExamples.tree_v2_modify)
    return gp


def plagih_config_update_from_json(config_json='config.json'):
    """
    The config gets updated
    """
    with Path.open(config_json, 'r') as f:
        config = json.load(f)
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


def run(root_dir):
    """
    Loads important files in your run-folder
    """
    run_files_path = root_dir / run_files

    config_json_path = root_dir / config_json
    samples_ready_path = run_files_path / samples_ready
    samples_csv_path = run_files_path / samples_csv
    operators_path = run_files_path /operators
    tree_expr_txt_path = run_files_path / tree_expr_txt
    tree_labels_csv_path = run_files_path / tree_labels_csv
    tree_numpy_csv_path = run_files_path / tree_numpy_csv

    if not Path.is_dir(run_files_path):
        raise FileNotFoundError('Folder does not exist: {}.'.format(run_files_path))

    if Path.is_file(config_json_path):  # Load config.json
        with open(config_json_path, 'r') as f:
            config = json.load(f)
    else:
        print_warning('w', 'You should have a {} here:\n{}'.format(config_json, config_json_path))
        config = {}

    if Path.is_file(samples_ready_path):
        prepared_data = data_load_pickle(samples_ready_path)
    elif Path.is_file(samples_csv_path):
        prepared_data = data_from_csv(samples_csv_path)
        print('Prepared the raw {} behaviour. Saving for next run.'.format(samples_csv))
        data_save_pickle(prepared_data, samples_ready_path)
    else:
        raise FileNotFoundError('No data provided? Please provide {} or {}.'.format(samples_ready, samples_csv))

    if Path.is_file(operators_path):  # Load operators.csv
        op_array = load_funcarray_from_csv(operators_path)
    else:
        raise FileNotFoundError('File does not exist: {}.'.format(operators_path))

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

    gp = ExplainableGP(root_dir, config=config)
    gp.activate_dataset(prepared_data)
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
