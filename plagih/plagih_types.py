"""
This file takes care of the strong typing in plagih.
'Types' of nodes:
- Terminals: 2f, 2b
- Functions: f2f, f2b, b2b, b2f, b2f2f (last one being if-then-else)
This notation offers the opportunity to actually search for parts, e. g.
> if '2b' in function:  # This returns True if the label evaluates to boolean
Be careful with if-then-else though. This needs boolean and two float inputs to produce one float
"""
import logging
import random

from plagih.file_interaction import yaml_load
from plagih.operators import *
import numpy as np


class Selectable:
    """

    """
    pass


class ChooseOparray3(Selectable):
    """

    """

    def selecting(self, coolxtype):
        """
        # def selecting_slower(self, coolxtype):
        #     oplist = self.slowertodo[coolxtype]
        #     return random.choices(oplist[0], weights=oplist[1])[0]
        """
        return self.choose_oparray[coolxtype]()

    def __init__(self, operator_pool=None, sfeh_no_crazyops=None):
        """

        """

        def check_operator_pool(operator_pool):
            """
            Check if the user-specified loaded operators allow closure
            operator_pool: list with operators and their weight of being selected
            """
            # sfeh dunno if that works... 2f not in x
            opxtypes = [oper.coolxtype for oper in operator_pool.keys()]
            has_2f = any([float == x[1] for x in opxtypes])
            has_2b = any([bool == x[1] for x in opxtypes])
            has_f2b = any([float in x[0] and bool == x[1] for x in opxtypes])
            has_b2f = any([bool in x[0] and float == x[1] for x in opxtypes])
            if not all([has_2f, has_2b, has_f2b, has_b2f]):
                logging.warning(f'Loaded operators do not feature both numeric (float) and Boolean type.')
            if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
                raise Exception(f'Loaded operators do not allow closure!')

        if operator_pool is None:  # quick developer adjustments
            operator_pool = [['+', 2],
                             ['-', 1], ['Usub', 1],
                             ['*', 2], ['/', 1],
                             ['Square', 0.75], ['**', 0.25],
                             ['Abs', 0.5], ['sign', 0.5], ['Round', 0.5],  # sfeh stop chain of arity-1 op in buid method?
                             ['sqrt', 0.25],
                             # ['log', 0.1], ['log1p', 0.1],  # sfeh
                             ['sin', 0.5],  # ['tan', 0.1], ['cos', 0.33], ['acos', 0.33], ['asin', 0.33], ['atan', 0.33],
                             ['tanh', 0.2],
                             ['Andb', 1], ['Orb', 1], ['Notb', 0.5], ['Xor', 1],
                             ['==', 1], ['!=', 0.5],
                             ['<', 0.5], ['<=', 0.5], ['>', 0.1], ['>=', 0.1],
                             ['Ifte', 2],
                             ['Mini', 1], ['Maxi', 1]]
            operator_pool = {op[lp[0]]: lp[1] for lp in operator_pool}  # sfeh this maps the actual class to the label

        # todotodo operator_poolrandom
        if sfeh_no_crazyops:
            del operator_pool['**']
            # workaround sfeh (delete this)

        check_operator_pool(operator_pool)

        choose_oparray = {
            # # all operator_pool (not needed)
            # None: {0: [], 1: [], 2: [], 3: [], None: []},

            # all operator_pool with a certain xtype-result
            None: [[], []],
            float: [[], []],  # 2f
            bool: [[], []],  # 2b
            (tuple([float]), float): [[], []],  # x**2, sqrt, log, sin, ...
            (tuple([float, float]), float): [[], []],  # +, -, *, /, **, ...
            (tuple([bool, float, float]), float): [[], []],  # Ifte
            (tuple([float, float]), bool): [[], []],  # <, >, =, >=
            (tuple([bool]), bool): [[], []],  # not
            (tuple([float]), bool): [[], []],  # dummy, currently no such operator
            (tuple([bool, bool]), bool): [[], []],  # and, or, xor, ...
        }
        for xlabel, probability in operator_pool.items():
            choose_oparray[None][0].append(xlabel)  # all operators #todo delete this? none required?
            choose_oparray[None][1].append(probability)
            choose_oparray[xlabel.coolxtype][0].append(xlabel)  # point mutations
            choose_oparray[xlabel.coolxtype][1].append(probability)
            choose_oparray[xlabel.coolxtype[1]][0].append(xlabel)  # construction of trees
            choose_oparray[xlabel.coolxtype[1]][1].append(probability)

        for o, p in choose_oparray.items():
            # normalizing the probabilities in every case to a sum of 1 (100%)
            # (saving some very little time...)
            choose_oparray[o][1] = [x/sum(p[1]) for x in p[1]]

        # self.slower = {coolxtype: [x[0], x[1]] for coolxtype, x in choose_oparray.items()}
        self.choose_oparray = {coolxtype: lambda: np.random.choice(x[0], p=x[1]) for coolxtype, x in choose_oparray.items()}


class ChooseDistributions(Selectable):
    """

    """
    # todo random with numpy?
    terminal_distributions = {float: [lambda: random.normalvariate(0, 1),
                                      lambda: random.normalvariate(1, 1),
                                      lambda: random.normalvariate(10, 5),
                                      lambda: random.randint(1, 20)],  # 0 has actually no purpose (except as being an action)
                              bool: [lambda: random.choice([True, False])]}
    # can be called with

    def selecting(self, coolxtype_out):
        """

        """
        return random.choice(self.terminal_distributions[coolxtype_out])()

    def __init__(self, path_distrib, env_vars_obs_infos=None, data_train=None, n_samples=100):
        """

        """

        if Path.is_file(path_distrib):
            lambdadist_as_string = yaml_load(path_distrib)

            # todo how should distributions be loaded?
            # e.g. sample_amount = lambdadist_as_string.get('observed_floats')
            self.terminal_distributions = {float: [], bool: []}
            self.terminal_distributions[float].extend([eval(x) for x in lambdadist_as_string[float]]),
            self.terminal_distributions[bool].extend([eval(x) for x in lambdadist_as_string[bool]])
        else:
            logging.info('Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.')

        # self.sample_floats_from_data(env_vars_obs_infos, data_train, n_samples=n_samples)  # todo

    def sample_floats_from_data(self, env_vars_obs_infos, data_train, n_samples=100):
        """
        Why only floats?
        ...do you really, even theoretically, want to load Boolean True/False samples??
        (okay, it might make sense as it better represents the actual distribution- NO FUCK IT.)
        """
        if env_vars_obs_infos is not None:
            obsnames = env_vars_obs_infos.observables[float].keys()
            # todo bool variables aswell?
            obs_samples = data_train[obsnames].to_numpy().flatten()
            obs_samples = np.random.choice(obs_samples, size=n_samples)
            self.terminal_distributions[float].extend([lambda: random.choice(obs_samples)]),  # take one


class ChooseObservation(Selectable):
    """

    """
    observables = {float: lambda: None,
                   bool: None}  # sfeh None? no  lambda? yeah, not important but still...

    def selecting(self, coolxtype_out):
        """

        """
        return self.observables[coolxtype_out]()

    def __init__(self, observations_list):
        """

        """

        def observation_select_index(obs_list, max_hist=10):
            """
            chooses variables but weighting how old they are.
            obs_list = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4']
            -> [0.28, 0.23, 0.19, 0.16, 0.13]
            sfeh: what about larger steps?
            0, 1, 2, 3 is good, but
            0, 5, 10, 15 is worse
            what if variables are not all of same diff?
            """
            obs_list = np.delete(obs_list, np.s_[max_hist:])
            x = len(obs_list)
            fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
            p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
            p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
            lambda_new_obs = np.random.choice(obs_list, p=p)
            return lambda_new_obs  # returning a function this time

        obs_prop = []
        obs_info = {}

        obs_families = list(set(x.family for x in observations_list))
        for fam in obs_families:
            family_meeting = sorted([x for x in observations_list if x.family == fam], key=lambda o: o.obs_index)
            if len(family_meeting) > 1:
                # e.g. pos_1, pos_2, pos_3, ...
                observations_list.extend([x for x in family_meeting])
                obs_prop.extend(list(observation_select_index(family_meeting)))
                index_minmax = (family_meeting[0].obs_index, family_meeting[-1].obs_index)
                for obs in family_meeting:
                    obs.index_minmax = index_minmax
                    # environment.obs_infos[obs.name] = obs  # todo okay do we need this? :s guess we will find out x.D haha
                    obs_info[obs.name] = obs
            else:
                # LOL UMAD? only one family member (probably even more common)
                obs = family_meeting[0]
                obs_info[obs.name] = obs
                observations_list.append(obs)
                obs_prop.append(1)  # just one value
                # todotodo todo
        self.observables = {float: lambda: np.random.choice(observations_list, p=obs_prop),
                            bool: None}  # sfeh None? no  lambda? yeah, not important but still...


class Choosing:
    """
    Just a class to prevent referencing all the separate shizzle everytime
    todo float_decimals?

    ===
    this was:
    def choose_operator():
    choose_oparray3 -> operator
    env_vars.choose_obs -> observation
    choose_distributions -> constant
    """

    operators = {}
    observations = {}
    constants = {}

    def an_operator(self, coolxtype):
        """
        Randomly choosing an operator-label for a given xtype.
        choose_oparray3 must be given, as they are different between runs.
        arity can also be set optionally, e.g. for point mutation
        todo DOUBLE-check if this coolxtype is chosen correctly... better: replace it
        """
        func_list, probability_list = self.operators[coolxtype]
        return np.random.choice(func_list, p=probability_list)

    def __init__(self, operators, observations, constants, float_decimals=6):
        self.operators = operators
        self.observations = observations
        self.constants = constants


def label_get_arity(node_label):
    """
    return arity of a label
    """
    if node_label in op:
        arity_sfehdebug = op[node_label]['arity']
        return arity_sfehdebug
    else:
        return 0


class EnvVars:
    """
    observation
    - take ~100 samples as constants
    - get xtype for buildin tree from expr
    - eval - get tensors with correct tf-type
    - filter index
    eval_action
    """

    def __init__(self):
        # self.obs_krazy = {}  # lookup table with all observations - if an observation is not in here, it is a float # sfeh delete this
        self.obs_infos = {}
        self.eval_action: EvalAction = None
        self.choose_obs = {float: None,
                           bool: None}


def data_from_csv(path_data_csv, action_name, test_size=0.2, delimiter=','):
    """
    Loads .csv data files.
    Information that we need to extract for each column:
    - choose_xtype choosing a random observation for leaf nodes
    - filtering observation index
        is there an index (past values, e.g. velocity_0, velocity_1) -> performing filter-evolve on the variables index (velocity_2 -> velocity_3)
    - evalaction data_train, data_test, is the action for the regression? -> more than one action might be required (IB has three action dimensions)
        - action min max -> for kernel regression bounded. occuring min and max values might not be the theoretical min/max values


    deprecated:

    """

    def is_action(name, role):
        if role is None:
            is_act = any(x in role for x in ['out', 'action', 'act', 'result'])
        else:
            is_act = any(['a_' == name[:2],
                          ii == len(df.columns) - 1,
                          any(x in name for x in ['out', 'action', 'act', 'result'])])
        return is_act

    """
    - Reading the .csv-file (with pandas)
    - renaming column headers
    - saving header info for later use
    """
    with Path.open(path_data_csv) as file:
        df = pd.read_csv(file, delimiter=delimiter)
        # todo it is float64, float64, int64 with MTC.. does it work with Tensorflow?

    actionmain = None
    observations = []
    rename_columns = {}  # cartPos|type=int|... -> cartPos

    """
    1. split col name
    - check whether its an observation
    --> check whether there are indizes
    - check whether its an action
    --> check unique valuea, min, max
    - check if it should be ignored (deprecated action, irrelevant column)
    2. 
    """

    for ii, header in enumerate(df):

        header_split = header.split('|')  # split 1: cartVel|type=float|role=input --> {cartVel, type=float, role=input]
        column_name = header_split[0]  # The first entry is always the column name
        if column_name in op:
            raise Exception(f'Your samples hold a column that matches the potential tree operator {column_name}.\n'
                            f'That might end up in confusion, please rename the column.')

        # todo the column name is actually just the base name
        rename_columns[header] = column_name

        col_infos = {}  # column_name: {'type': 'float', 'role': None, 'colpos': ii}}
        try:
            for colparam in header_split[1:]:
                k, v = colparam.split('=')
                col_infos[k] = v
        except Exception as ex:
            print(f'Could not load samples csv correctly: {ex}')

        coolxtype_out = float

        if is_action(column_name, col_infos.get('role')):
            if column_name == action_name or action_name is None:
                # todo min/max SHOULD be either given manually, be set to [0,1] or [-1,1] or depend on the data
                cmin = col_infos.get('min', df[header].min())
                cmax = col_infos.get('max', df[header].max())
                uniques = df[header].unique()
                actionmain = EvalAction(column_name, coolxtype_out, cmin, cmax, uniques)
            else:
                # drop_actions.append(column_name)  # drop from data (only evaluate ONE action, drop other potential actions) sfeh delete this
                df = df.drop(column_name, axis=1)  # no need to keep other actions
                printez('i', f'Ignoring action {column_name} for this run')  # , print_type=print_type print_type does not exist yet sfeh
        else:
            observations.append(Observation(column_name, coolxtype_out=coolxtype_out))

    df.rename(columns=rename_columns, inplace=True)
    df = df.astype('float32')  # sfeh sheesh, that will NOT work with bool or int data :P Following design pattern #YOLO

    """
    choosing random observations made easy
    """
    environment = EnvVars()
    environment.obs_infos = ChooseObservation(observations)  # todo remove dat shit

    if actionmain:
        environment.eval_action = actionmain
    else:
        raise

    data_train, data_test = skcv.train_test_split(df, test_size=test_size, random_state=0)

    return environment, data_train, data_test


def xtype_equi_outcome(a_xtype, b_xtype):
    """
    Dummy. Returns, whether two xtypes have the same outcome
    """
    return a_xtype[-2:] == b_xtype[-2:]


def choose_term(coolxtype_out, choose_obs, choose_distributions, float_decimals):
    """

    """

    # sfeh 50% chance observation/value
    if random.choice(['obs', 'distrib']) == 'obs' and choose_obs[coolxtype_out]:
        obs = choose_obs[coolxtype_out]()
        # print('SAME???', obs.name, obs.label)  # sfeh
        return obs
    else:
        dist_fun = random.choice(choose_distributions[coolxtype_out])
        value = dist_fun()
        if coolxtype_out == float:  # sfeh int aswell?
            value = float(round(value, float_decimals))
            const = FloatConstant(value)
        elif coolxtype_out == bool:
            const = BoolConstant(value)
        else:
            raise Exception('ASDASD NOOO WHYY')
        return const


def choose_operator(coolxtype_key, choose_oparray3):
    """
    Randomly choosing an operator-label for a given xtype.
    choose_oparray3 must be given, as they are different between runs.
    arity can also be set optionally, e.g. for point mutation
    todoo DOUBLE-check if this coolxtype is chosen correcrtly... better: replace it
    """
    func_list, probability_list = choose_oparray3[coolxtype_key]
    ops = np.random.choice(func_list, p=probability_list)  # [0] as this function can only return lists...
    return ops


def xtypes_from_labels(label_list, obs_infos=None):
    xtype_list = [xtype_get_from_label(label, obs_infos) for label in label_list]
    return xtype_list


def xtype_get_from_label(label, obs_infos=None):
    """
    returns xtype for a label
    if you are not 100% sure that it is a function.
    """

    if label in ['True', 'False']:
        xtype = '2b'
    elif label in op:
        xtype = op[label]['xtype']
    else:
        try:
            label = label[1:] if label[0] == '-' else label
            xtype = obs_infos[label]['xtype']
        except:
            xtype = '2f'

    return xtype


if __name__ == '__main__':
    import timeit

    x = timeit.timeit('lel.selecting(float)', setup='from __main__ import ChooseOparray3\nlel = ChooseOparray3()')
    # y = timeit.timeit('lel.slower(float)', setup='from __main__ import ChooseOparray3\nlel = ChooseOparray3()')
    print(x)
