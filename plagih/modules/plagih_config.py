import yaml
from pathlib import Path


class GpConfig:
    """

    """

    # def __init__(self, conf=None):
    #
    #     self.pl_version = PLAGIH_VERSION  # version important when loading old run
    #     self.force_new_run = False  # : False,  # especially for testing, ignores backup files when true
    #     self.time_max = None  # : None,  # int(60 * 60 * 12),  # 60 = 1 min
    #     self.gen_max = 1000  # : 1001,  # Maximum amount of generations
    #
    #     self.pop_max = 1000  # conf['pop_max']  #: 1000,  # amount is never tested
    #     self.tree_depth_max = 10  #: 10,  # maximum Tree depth for entire run
    #     self.tree_depth_min = 1  #: 2,
    #     self.tourn_size = 3  #: 3,  # [7 per 100] number of trees selected for tournament
    #     self.parsimony_mean = 20  #: 20,  # If you wnt your population to be a certain size
    #     self.parsimony_max = 50  #: 50,
    #     self.gen_num_max_parsimony = 50  #: 50,  # Increase tmp_parsim to this generation

    # def update_dict_nested(d, u):
    #     for k, v in u.items():
    #         if isinstance(v, collections.abc.Mapping):
    #             d[k] = update_dict_nested(d.get(k, {}), v)
    #         else:
    #             d[k] = v
    #     return d

    def __init__(self, args):

        self.pl_version = 0.97  # must only update if vital changes were made, version important when loading old run

        try:
            with Path.open(args.load_config, 'r') as file:
                conf = yaml.load(file, Loader=yaml.FullLoader)
                config_dir = args.load_config.parent
        except:
            conf = {}
            config_dir = None
        self.root_dir = Path(args.root_dir or config_dir or Path.cwd())
        self.name = args.name or self.root_dir.resolve().name  # sfeh probably there are better names

        self.print_type = conf.get('print_type', 'wwggaiiff')  # (a)lert, (w)arning, (g)en, (i)nfo, (f)ile written
        if args.print_all:
            conf.print_type = 'wwwwaaaggggiiiff'

        # can be updated from everywhere
        self.pop_max = args.pop_max or int(conf.get('pop_max', 1000))  #: 1000,  # amount is never tested
        self.gen_max = args.gen_max or int(conf.get('gen_max', 1000))  # : 1001,  # Maximum amount of generations
        self.action_name = args.action_name or conf.get('action_name', None)
        self.kernel_name = args.kernel_name or conf.get('kernel_name', 'regression')

        # only from command line
        self.tf_device_log = args.tf_device_log

        # only from config (as noone ever changed it :~P)
        self.tree_depth_max = int(conf.get('tree_depth_max', 10))  #: 10,  # maximum Tree depth for entire run
        self.tourn_size = int(conf.get('tourn_size', 3))  #: 3,  # [7 per 100] number of trees selected for tournament
        self.parsimony_max = conf.get('parsimony_max', 50)  #: 50,

        self.period = conf.get('period', {'gen_plots': 5, 'gen_save': 5})

        # sfeh not here?
        self.evolve_list_random = conf.get('evolve_list_random')  # sfeh
        self.complexity_measure = conf.get('complexity_measure', 'tree_edit_distance')  # sfeh check used origin here? backup loaded origin?

        self.lambdadist_as_string = conf.get('lambdadist_as_string', {'2f': ['lambda: random.normalvariate(0,1)',
                                                                             'lambda: random.normalvariate(1,1)'],  # 'lambda: random.randint(0, 10)',  # not required
                                                                      '2b': ['lambda: random.choice([True, False])'],
                                                                      'observed_floats': 100})

        # sfeh just for the conclusion...
        self.load_backup_path_loaded_dummy = args.load_backup
        self.data_csv_path_loaded_dummy = args.data_csv
        self.origin_tree_path_loaded_dummy = args.origin_tree

        """
        Not used?
        """
        # self.restart_count
        # self.fitness_decimals = int(conf.get('fitness_decimals', 6))  # rounding the fitness
        # self.float_decimals = int(conf.get('float_decimals', 6))  # None or 1-30 decimals
        # self.parsimony_mean = conf.get('parsimony_mean', 15)  #: 20,  # If you wnt your population to be a certain size
        # self.tree_depth_min = conf.get('tree_depth_min', 1)  #: 2,
        # self.swim = 'p'  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool # todo
        # self.gen_num_max_parsimony = conf.get('gen_num_max_parsimony', 50)  #: 50,  # Increase tmp_parsim to this generation
