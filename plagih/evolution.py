"""

"""


class EvolutionLoop:
    """
    The evolve_dict is converted into a list of the population length.
    The evolve_loop is for the regular evolution, the evolve_random is especially required for the first generation.
    """
    def __init__(self, origin_is_fix, conf):
        self.conf = conf
        evolve_loop = {
            # Reproduction (10%)
            'Repro': {'evolve_name': 'reproduce', 'params': {}, 'evolve_rate': 0.06, 'custom_params': {}},
            'Rsympy': {'evolve_name': 'reproduce', 'evolve_rate': 0.03, 'custom_params': {'simplify': True}},
            'Pareto': {'evolve_name': 'revive paretos', 'evolve_rate': 0.01, 'custom_params': {}},

            # Mutation (25%)
            'Point': {'evolve_name': 'mutate point', 'evolve_rate': 0.05, 'custom_params': {}},

            'BranchDF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                         'custom_params': {'build_max': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 4, 0.8), 'p_op': 1.0}}},
            'BranchDG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.00,
                         'custom_params': {'build_max': {'size_mode': 'branch_depth', 'mean_min_max_var': (2.5, 1, 5, 1), 'p_op': 0.5}}},
            'BranchNF': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'p_op': 1.0}}},
            'BranchNG': {'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
                         'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (7, 1, 12, 3), 'p_op': 0.5}}},
            'BranchShrink': {'evolve_name': 'mutate branch', 'evolve_rate': 0.0,
                             'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (1, 1, 1, 0), 'p_op': 0.5}}},

            'FilterBO': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                         'custom_params': {'mode': 'branch', 'filter_observations': True}},
            'FilterB': {'evolve_name': 'filter optimize', 'evolve_rate': 0.05, 'tourn_size': 5,
                        'custom_params': {'mode': 'branch', 'filter_observations': False}},
            'FilterP': {'evolve_name': 'filter optimize', 'evolve_rate': 0.0, 'tourn_size': 5,
                        'custom_params': {'mode': 'point', 'filter_observations': True}},

            # Crossover (35%)
            'Xover': {'evolve_name': 'crossover branch', 'evolve_rate': 0.30, 'custom_params': {}},  # sum 0.70

            # Leftovers are automatically filled with random trees

            # Random (25%)
            'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.05,
                      'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 3, 5, 1), 'p_op': 1.0}}},
            'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.00,
                      'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4.5, 4, 5, 1), 'p_op': 0.5}}},
            'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'p_op': 0.5}}},
            'Rand4': {'evolve_name': 'random trees', 'evolve_rate': 0.15,
                      'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 1, None, 5), 'p_op': 1.0}}},  # param 'max' can be None
        }

        self.evolve_loop = self.evolve_safety_update(evolve_loop)

        if origin_is_fix:
            evolve_random = {'Rand3o': {'evolve_name': 'random trees', 'evolve_rate': 1.00,
                                        'custom_params': {'build_max': {'size_mode': 'branch_nodes', 'mean_min_max_var': (10, 3, None, 4), 'p_op': 1.0}}}}
        else:
            # try:  # sfeh still not sure about this
            #     evolve_random = self.evolve_list_random['from_scratch']
            # except:
            evolve_random = {'Rand1': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                       'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (3.5, 2, 5, 1), 'p_op': 1.0}}},
                             'Rand2': {'evolve_name': 'random trees', 'evolve_rate': 0.30,
                                       'custom_params': {'build_max': {'size_mode': 'tree_depth', 'mean_min_max_var': (4, 2, 6, 1), 'p_op': 0.5}}},
                             'Rand3': {'evolve_name': 'random trees', 'evolve_rate': 0.40,
                                       'custom_params': {'build_max': {'size_mode': 'tree_nodes', 'mean_min_max_var': (10, 3, None, 4), 'p_op': 1.0}}}
                             }
        self.evolve_random = self.evolve_safety_update(evolve_random)

        self.evolve_tags = list(self.evolve_loop.keys()) + list(self.evolve_random.keys())
        return

    def evolve_safety_update(self, evolve_dict):
        """
        Updates tournament size and evolve rates

        Example entry of the list could be:
        {'tag': 'BranchDF', 'evolve_name': 'mutate branch', 'evolve_rate': 0.05,
        'custom_params': {'build_max': {'size_ref': 'branch_depth', 'mean': 3, 'min': 1, 'max': 5, 'gauss_var': 0.8, 'method': 'full'}}},
        """

        for tag, evolve_spec in evolve_dict.items():
            evolve_dict[tag]['tourn_size'] = evolve_spec.get('tourn_size', self.conf.tourn_size)
            evolve_dict[tag]['evolve_num'] = int(evolve_spec.get('evolve_rate') * self.conf.pop_max)
            evolve_spec['custom_params'] = evolve_spec.get('custom_params', {})
        return evolve_dict

    # evolve_loop = self.evolve_list  # todo
    # self.printpl('i', 'Using evolve rates from config')

    # class Evolution:
    #     # todo rate not in here
    #     # def __init__(self, id=None, evolution=None, rate=None, params=None, custom_params=None):
    #     #     self.id = id
    #     #     self.evolution = evolution
    #     #     self.params = params or {}
    #     #     self.rate = rate
    #     #     self.custom_params = custom_params
