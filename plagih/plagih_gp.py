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


def gp_run(plagih_root, load_backup, config_path, out_dir, force_new_run, eval_action, path_data, path_origin_csv, kernel_name, analyze, tf_device_log, pop_max, gen_additionally):
    """
    
    """

    root_dir, config = load_config(config_path, out_dir=out_dir)

    config['force_new_run'] = force_new_run
    if eval_action is not None:
        config['eval_action'] = eval_action

    gp = ExplainableGP(plagih_root, root_dir, config, eval_action, kernel_name=kernel_name, path_data=path_data, tf_device_log=tf_device_log, pop_max=pop_max, path_origin_csv=path_origin_csv, gen_additionally=gen_additionally)

    gp.make_evolve_rates()

    if analyze:
        gp.gp_analyze()
    else:
        gp.make_evolve_rates()
        gp.plagih_gp_run()

    sys.exit()
