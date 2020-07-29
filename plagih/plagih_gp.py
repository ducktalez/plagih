import sys

# sys.path = ['..'] + sys.path
# sys.path.append('modules/')  # add directory 'modules' to the current root_dir

from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.plagih_data import *
import yaml

# import warnings
# warnings.filterwarnings('error')


def gp_run(conf, root_dir, path_data_csv, path_origin_tree, path_load_backup, analyse, force_new_run, gen_additionally, developer_fix=None):
    """
    sfeh YESS this is now gone
    """
    gp = ExplainableGP(conf, root_dir, path_data_csv, path_origin_tree, developer_fix=developer_fix)

    if developer_fix:
        gp.try_load_backup()
    if analyse:
        gp.gp_analyse(path_load_backup)
    else:
        gp.plagih_gp_run(path_load_backup, force_new_run=force_new_run, gen_additionally=gen_additionally)

    sys.exit()
