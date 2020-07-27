import sys

# sys.path = ['..'] + sys.path
# sys.path.append('modules/')  # add directory 'modules' to the current root_dir

from plagih.modules.plagih_gp_base_class_xai import *
from plagih.modules.plagih_data import *
import yaml

# import warnings
# warnings.filterwarnings('error')


def gp_run(conf, path_load_backup, path_data_csv, path_origin_tree, analyse, gen_additionally, force_new_run):
    """
    todo YESS this is now gone
    """
    gp = ExplainableGP(conf, path_load_backup, path_data_csv, path_origin_tree, analyse, gen_additionally, force_new_run)

    gp.make_evolve_rates()

    if analyse:
        gp.gp_analyse()
    else:
        gp.make_evolve_rates()
        gp.plagih_gp_run()

    sys.exit()
