import yaml
from pathlib import Path


class GpConfig:
    """
    (just to find this with quick search) self.conf self.config
    """

    def __init__(self, args):
        """
        SFEH NEVER try to save paths here. switching between systems is worse than HitlerAIDS
        """
        self.pl_version = 0.97  # must only update if vital changes were made, version important when loading old run
        self.name = args.prepared_run or None  # sfeh

        try:
            with Path.open(args.load_config, 'r') as file:
                conf = yaml.load(file, Loader=yaml.FullLoader)
        except:
            conf = {}

        self.print_type = 'wwwwaaaggggiiiff' if args.print_all else conf.get('print_type', 'wwggaaiiff')  # (a)lert, (w)arning, (g)en, (i)nfo, (f)ile written

        # can be updated from everywhere
        self.pop_max = args.pop_max or int(conf.get('pop_max', 1000))  #: 1000,  # amount is never tested
        self.gen_max = args.gen_max or int(conf.get('gen_max', 1000))  # : 1000,  # Maximum amount of generations
        self.action_name = args.action_name or conf.get('action_name', None)
        self.kernel_name = args.kernel_name or conf.get('kernel_name', 'regression')

        # only from command line
        self.tf_device_log = args.tf_device_log

        # only from config (as noone ever changed it :~P)
        self.tree_depth_max = int(conf.get('tree_depth_max', 10))  #: 10,  # maximum Tree depth for entire run
        self.tourn_size = int(conf.get('tourn_size', 3))  #: 3,  # [7 per 100] number of trees selected for tournament
        self.parsimony_max = conf.get('parsimony_max', 35)
        self.period = conf.get('period', {'gen_plots': 5, 'gen_save': 5})  # todo

        self.float_decimals = conf.get('float_decimals', 6)  # makes the lut more practical - more hits are achieved. be careful with rounding to zero.  # sfeh check this

        # sfeh not here?
        self.evolve_list_random = conf.get('evolve_list_random', None)  # sfeh
        self.complexity_measure = conf.get('complexity_measure', 'tree_edit_distance')  # sfeh check used origin here? backup loaded origin?

        self.lambdadist_as_string = conf.get('lambdadist_as_string', {'2f': ['lambda: random.normalvariate(0,1)',
                                                                             'lambda: random.normalvariate(1,1)',
                                                                             'lambda: random.normalvariate(10,5)',
                                                                             'lambda: random.randint(1, 20)'],  # not required?
                                                                      '2b': ['lambda: random.choice([True, False])'],
                                                                      'observed_floats': 100})

        """
        Not used?
        """
        # self.restart_count
        # self.fitness_decimals = int(conf.get('fitness_decimals', 6))  # rounding the fitness
        # self.float_decimals = int(conf.get('float_decimals', 6))  # None or 1-30 decimals
        # self.parsimony_mean = conf.get('parsimony_mean', 15)  #: 20,  # If you wnt your population to be a certain size
        # self.tree_depth_min = conf.get('tree_depth_min', 1)  #: 2,
        # self.swim = 'p'  # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool # todo
        # self.gen_num_parsim_maxony = conf.get('gen_num_parsim_maxony', 50)  #: 50,  # Increase tmp_parsim to this generation
