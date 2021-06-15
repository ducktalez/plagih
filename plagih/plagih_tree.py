import os
from plagih.fitness_kernel import *

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # does not work anyways... delete this
"""
Ptree2 contain a new implementation of trees that we use in genetic programming to display a program.
The old karoo "tree" is replaced with, for now, "cooltreer" in the code.
not all functions can use cooltree for now and some tree-functions require the old "tree"
cooltree splits the karoo tree into the
- meta-info (fitness, parsimony, tree-id, ...) and the
- core (coolcore)
The core of the tree, which "is" the tree, is stored recursively
Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)
"""
import itertools
from collections import deque
from plagih.tree_distances.tree_edit_distance import apted_distance
from plagih.plagih_sympy_extras import expr_sympify
import logging
from plagih.file_interaction import yaml_load
from plagih.operators import *
import numpy as np


class Node:
    """
    The core is the structure of a plagih gp-tree.
    It recursively holds the nodes of a tree; every tree has a list of potential children.
    Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)

    states?
    [None]: not set
    [0]: evolution/construction/build mode (potentially missing leaf nodes)
    [1]: structurally complete/finalized (node_depths correct, node_id set, ...)
    [2]: including meta-data (fitness, complexity)
    """
    state=None

    def __init__(self, plabel: Plabel, is_fix=False, childs=None):
        self.plabel = plabel
        self.is_fix = is_fix
        self.childs = childs or []

    def __hash__(self):
        """
        Hashing the label-list as string should be sufficient.
        Is there a chance for same hash values within oioulations? That would be very bad.

        Problem:
            This old version does not work for the tree LUT when reloading runs.
            >>return hash(','.join([str(x) for x in labellist]))
            the python hash-function has a run-specific salt in to have run-specific hash values.
        """
        return hash(str(self))  # sfeh

    def __str__(self):
        """
        Printing the nodes as nested array structure.
        # only printing the nodes, no meta
        sfeh: used to print a loadable statement aswell!
        """
        print_label = self.plabel.label
        if self.is_fix:
            print_label = f'({print_label})'

        if not self.childs:
            return f"{print_label}"
        else:
            childstr = ', '.join([str(x) for x in self.childs])
            return f"[{print_label}, {childstr}]"

    def __len__(self):
        """
        counting the amount of nodes recursively
        """
        return 1 + sum([len(cc) for cc in self.childs])

    def get_pycode(self):
        """

        """
        if self.plabel.arity == 0:
            return f'{self.plabel}'
        else:
            results = []
            for child in self.childs:
                results.append(child.get_pycode())  # = tree_node_get_label(tree, int(child))
            return self.plabel.pycode.format(*results)  # abs -> lambda a: 'abs({})'.formadt(a) (result1)

    def get_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
        e.g. {+{Ifte{True}{1}{2}}{3}}
        """
        childargs = [cc.get_apted_notation() for cc in self.childs]
        return f"{{{self.plabel.label}{''.join(childargs)}}}"

    def get_expr_raw(self, reducible=None, obs_names=None):
        """
        accumulate and return the complete expression the tree holds recursively
        """
        # if self.expr_raw is None:  # sfeh?
        expr = self.plabel.sym_str
        if self.plabel.arity > 0:
            child_expr_list = [cc.get_expr_raw(reducible=reducible, obs_names=obs_names) for cc in self.childs]
            # if reducible:
            #     # my_expr = op[self.label]['sym_reduce'] or my_expr
            #     # symloc = sympy_symbol_defaults(obs_names)  # todo solve the problem... new version of sympy?
            #     xxx = plagih_sympify(my_expr.format(*child_expr_list), eval_locals=symloc)  # sfeh the xxx variable
            #     return xxx
            expr = expr.format(*child_expr_list)  # f'cos({})'([33]) does not work. *list makes the list args :D
        return expr

    def new_core(self, new_core: 'Node'):
        """

        """
        self.plabel = new_core.plabel
        self.childs = new_core.childs if new_core.childs else []  # maybe must be updated recursively
        # self.is_fix = new_core.is_fix  # debatable
        # self.complete = new_core.complete  # if the node is correct/done/okay
        # depth needs to be fixed?

    def get_mutatable_nodes(self, coolxtype_out=None, allow_root=True):
        """
        return all nodes that are mutatable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!
        """
        coolnode_list = []
        if not self.is_fix:
            if coolxtype_out is None or coolxtype_out == self.plabel.coolxtype[1] and (allow_root or not self.is_root()):
                # crossover requires excluding types that are not matching, and excludes the root node
                coolnode_list.append(self)

        coolnode_list.extend(list(itertools.chain(*[cc.get_mutatable_nodes(coolxtype_out=coolxtype_out, allow_root=allow_root) for cc in self.childs])))
        return coolnode_list

    def evolve_append_layer_depth(self, choose_obs, choose_oparray3, choose_distributions, float_decimals, construct='full'):
        """
        appends a layer of nodes to a tree that is in construction
        returns the
        only_terminals: on the lowest level...
        """
        # go to the current leaf nodes
        if not self.childs and self.plabel.arity > 0:  # alternatively check the amount of input parameters in coolxtype
            # append, if we are on the lowest level and are looking at an operator
            for itype in self.plabel.coolxtype[0]:
                if construct == 'full':  # operators
                    plabel = choose_operator(itype, choose_oparray3)
                elif construct == 'grow':  # 50% chance operator/terminal
                    if random.choice(['term', 'func']) == 'func':
                        plabel = choose_operator(itype, choose_oparray3)
                    else:
                        plabel = choose_term(itype, choose_obs, choose_distributions, float_decimals)
                elif construct == 'term':  # force terminals
                    plabel = choose_term(itype, choose_obs, choose_distributions, float_decimals)
                else:
                    raise Exception('Never happening i hope lel sfeh')

                self.childs.append(Node(plabel=plabel))
            return len(self.childs)
        else:
            appended_nodes = 0
            for cc in self.childs:
                appended_nodes += cc.evolve_append_layer_depth(choose_obs, choose_oparray3, choose_distributions, float_decimals, construct=construct)
            return appended_nodes

    def reduce_me(self, obs_infos):
        # sfeh asdasdasd reduce me is obviously bullshit crapshit.
        #  sympify works with this combination only very few times
        #  lets have a new idea.
        expr_raw = self.get_expr_raw(reducible=True, obs_names=obs_infos.keys())
        try:
            expr_sym = expr_sympify(expr_raw)
        except:
            raise Exception(f'Sympify failed. {expr_raw}')

        new_core = coolcore_from_expr(expr_sym, obs_infos)
        if len(new_core) < len(self):
            self.new_core(new_core)
        elif len(new_core) > len(self):
            raise Exception(
                f'Reduced core is even more complex than before  ({len(new_core)}, {len(self)}). expr_raw: {expr_raw}')  # \nold_core:{self}\nnew_core: {new_core} May happen with sympification and Usub.
            # example: Tree sympification did not work: Reduced core is even more complex than before. expr_raw: sign(Mini(((Velocity_2 * -0.790706) - sqrt(Gain_0)), (-0.569271 - Velocity_9)))
            # old_core:[sign, [Mini, [-, [*, Velocity_2, -0.790706], [sqrt, Gain_0]], [-, -0.569271, Velocity_9]]]
            # new_core: [sign, [Mini, [-, [Usub, [sqrt, Gain_0]], [*, 0.790706, Velocity_2]], [-, -Velocity_9, 0.569271]]]
        return

    def nodepath_insert_branch(self, nodepath, coolbranch):
        """
        inserting a branch into the place of a node
        """
        if len(nodepath) > 1:
            self.childs[nodepath[0]].insert_branch(nodepath[1:], coolbranch)
        else:  # [1] -> set child 1
            self.childs[nodepath[0]] = coolbranch

    def evolve_mutate_branch_depth(self, depths, builder):
        """
        todo other version
        currently only one branch
        """
        # sfeh: making the root node (todo?)
        if depths == 1:
            self.plabel = builder.choose_term(self.plabel.coolxtype[1])  # sfeh update node plabel
        else:
            depths -= 1
            self.childs = [Node(builder.choose_any(xt, p_op=1)) for xt in self.plabel.coolxtype[0]]

    def evolve_mutate_filter(self, choose_oparray3, choose_obs, choose_distributions, float_decimals):
        """
        filtger the nodes in a single tree
        """
        if self.plabel.arity > 0:
            for cc in self.childs:
                cc.evolve_mutate_filter(choose_oparray3, choose_obs, choose_distributions, float_decimals)
        else:
            self.plabel.mutate_filter()

    def evolve_mutate_point(self, choose_oparray3, choose_obs, choose_distributions, float_decimals):
        """
        Mutate a single mutatable point in any Tree.
        """
        if self.plabel.arity > 0:
            self.plabel = choose_operator(self.plabel.coolxtype, choose_oparray3)  # Function is same type, same arity
        else:
            print('hhhhh', self.plabel)
            self.plabel = choose_term(self.plabel.coolxtype[1], choose_obs, choose_distributions, float_decimals)  # 3 -> '2f' -> 5

    def evolve_start(self):
        """
        Before evolving, delete all tree information
        so the tree is not holding "wrong" information about fitness, etc.
        - (append last meta value to history)
        - delete meta info
        - set status to not complete
        """
        self.history.append(self.meta)
        self.meta = self.PtreeMeta()
        self.complete = False

    def evolve_reduce(self, obs_infos=None, completely=True):
        """
        Reducing a tree to its most basic form with sympify.
        (completely = False: reduce just one branch. if you wanted to have more complexity)
        """
        length_before = len(self)
        if completely:  # reduce the complete tree
            coolcores_lv0 = self.get_nodes_at_depth(0, only_mutable=True)
            for coolc in coolcores_lv0:
                coolc.reduce_me(obs_infos)
        else:
            cool_nodes = self.get_mutatable_nodes()
            cool_functions = [x for x in cool_nodes if x.arity > 0]
            if cool_functions:
                chosen = random.choice(cool_functions)
                chosen.reduce_me(
                    obs_infos)  # sfeh chosen must be set again? or not? test it at least. probably working.
        if length_before < len(self):
            print_e(f'FFS Trees just become larger? {self.get_expr_raw()}')
        # self.meta.clear()


    def evolve_mutate_filter_random(self, call_params, choose_oparray3, choose_obs, choose_distributions, float_decimals):
        """
        Mutates a number of float terminal of a tree
        todo
        """
        mode = call_params['mode']  # point/branch/all
        yes_observations = call_params.get('yes_observations')  # point/branch/all
        mutate_filter = 'gaussian_filter'  # sfeh change?

        node_ids = self.get_mutatable_nodes()
        node_id = random.choice(node_ids)  # sfeh should this be completely random?

        if mode == 'branch':
            node_id.evolve_mutate_filter(choose_oparray3, choose_obs, choose_distributions, float_decimals)
        else:
            pass
            # mode == 'point'
            # sfeh delete this? point can always hapen
            # node_id.evolve_mutate_point(choose_oparray3, choose_obs, choose_distributions, float_decimals)


    def evolve_mutate_point_random(self, choose_oparray3, choose_obs, choose_distributions, float_decimals):
        """
        Mutate a single mutatable point in any Tree.
        """
        # 1. choose a node
        node_list = self.get_mutatable_nodes()
        node = random.choice(node_list)
        node.evolve_mutate_point(choose_oparray3, choose_obs, choose_distributions, float_decimals)


    def evolve_mutate_branch_random(self, cool_build_size, builder: TreeBuilder, size_mode='depth', full_or_grow='full'):
        """

        """
        node_list = self.get_mutatable_nodes()
        node = random.choice(node_list)
        node.evolve_mutate_branch_depth()

    def evolve_random_tree_depth(self, depth):
        """
        Expand the branches of a tree to reach a certain depth/layer
        Comment:
        The depth-based function is easier than the grow method
        """
        if self.depth < depth:
            for cc in self.plabel.coolxtype[0]:
                pass
                # todo todotodo
            # choose_operator()
            # self.plabel.coolxtype[0]
        else:
            pass

    def finalize(self):
        """
        (only in root node)
        finalizing the structure
        """
        self.finalize_set_depth()
        self.finalize_set_nodepath([0])
        self.workaround_remove_tilde()


class FinalizedNode(Node):
    """
    sfeh tree age is the length of the hist list. crossover: use maximum
    """

    class PtreeMeta:
        def __init__(self, fitness_train=None):
            self.hash = None
            self.fitness_train = fitness_train
            self.parsimony = None
            self.expr_raw = None
            self.expr_sym = None

            self.depth = None
            self.complete = None
            self.last_evolution = None

        def clear(self):
            # sfeh init/new or so?
            # sfeh parsimony meta, fitness, ...
            self.hash = None
            self.fitness_train = None
            self.parsimony = None
            self.expr_raw = None
            self.expr_sym = None

            self.depth = None
            self.complete = None
            self.last_evolution = None

        def __str__(self):
            return f"hash: {self.hash}, fitness: {self.fitness_train}, parsimony: {self.parsimony}, {self.depth}, {self.last_evolution}, {self.expr_raw}, {self.expr_sym}"

    def export_visualization_latex(self):
        """
        todo
        """
        return None

    def get_nodes_at_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """
        Returns a list with mutatable ids which are *goal_depth* layers away from non modifiable nodes
        last_leaves: if you want so save all leave nodes aswell

        sum_layers=False, get_closest=True, return_all_layers=False
        """
        if self.depth < goal_depth:
            return sum(
                [child.get_nodes_at_depth(goal_depth, only_mutable=only_mutable, get_closest_depth=get_closest_depth)
                 for child in self.childs], [])
        else:
            if only_mutable and self.is_fix:
                return []
            if get_closest_depth and self.depth != goal_depth:
                return []

            return [self]

    def check_all(self):
        """

        """
        if self.plabel.arity != len(self.childs):
            raise
        return True

    def eval_parsimony(self, parsimony_distance, origin_cooltree=None, weights=None):
        """
        parsimony_distance: compute the chosen distance by the user.
        #     'tree_node_count': tree_get_size,
        #     'tree_depth': tree_get_depth,
        #     'tree_edit_distance': tree_parsimony_ted,
        """
        if parsimony_distance == 'tree_node_count':  # number of nodes
            return len(self)  # returns the number of nodes  # sfeh weights
        elif parsimony_distance == 'tree_edit_distance':  # tree_edit_distance, tree-edit-distance
            apted1 = self.get_apted_notation()
            apted2 = origin_cooltree.get_apted_notation()
            distance, mapping = apted_distance(apted1, apted2)  # sfeh the mapping could be handy somewhere
            if weights is None:
                return distance
            else:
                raise
        else:
            raise Exception(f'Complexity measurement not available: {parsimony_distance}')

    def get_nodes_to_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """
        sum_layers=False, get_closest=True, return_all_layers=False
        """
        child_results = []
        if self.depth < goal_depth:
            child_results = sum(
                [child.get_nodes_to_depth(goal_depth, only_mutable=only_mutable, force_depth=get_closest_depth) for
                 child in self.childs], [])

        if only_mutable and self.is_fix or \
                get_closest_depth and self.depth != goal_depth:
            my_result = []
        else:
            my_result = [self]

        return my_result + child_results

    def get_observation_list(self):
        """
        Returns a list with the used observations
        """
        my_return = []
        if self.plabel.arity == 0:
            if terminal_label_is_observation(self.plabel.label):
                my_return = [self.plabel.label]

        for cc in self.childs:
            my_return.extend(cc.get_observation_list())

        return my_return

    # def get_node_from_nodepath(self, target_nodepath):
    #     """
    #     example coordinates: [0, 2, 1] -> node is at depth 2, accessable by going to childs 0 -> 2 -> 1.
    #     (root is always 0)
    #     """
    #     if len(target_nodepath) > len(self.nodepath):
    #         next_child = target_nodepath[len(self.nodepath)]
    #         return self.childs[next_child].get_from_path(target_nodepath)
    #     else:
    #         return self

    def get_labellist_breath(self):
        """
        Returns all labels in a core node
        Breitensuche im Baum
        """
        label_list = []
        max_depth = self.childs_depth_max
        for depth in range(0, max_depth + 1):
            labels_at_depth = [x.label for x in self.get_nodes_at_depth(depth)]
            label_list.extend(labels_at_depth)
        return label_list

    def finalize_set_nodepath(self, nodepath):
        """
        [0,2,1,0,0]
        """
        self.nodepath = nodepath
        for ii, child in enumerate(self.childs):
            nodepath_child = nodepath + [ii]
            child.finalize_set_nodepath(nodepath_child)

    def finalize_structure(self):
        """
        Finalize a single node
        """
        self.finalize_set_depth(depth=self.depth or 0)  # sfeh the starting depth? hmm, why set it none anyways. aye. rewrite

    def __init__(self, plabel:Plabel):
        super().__init__(plabel)
        # self.core = coolcore  # todo sfeh
        self.meta = self.PtreeMeta()
        self.history = deque([], maxlen=10)  # sfeh arbitrary value of 10 historic metainfo of this tree
        self.finalize_structure()

    # def workaround_normalize_exponentiation(self):
    #     self.core.workaround_normalize_exponentiation()
    #     # self.finish_nodes()  #
    #     self.finalize_completely()  #

    def finalize_completely(self):
        """

        """
        self.finalize_structure()
        # self.finalize_meta()  # todo
        self.complete = True

    def finalize_meta(self):
        """

        """
        # sfeh asd does this work?
        self.meta.expr_raw = self.get_expr_raw()
        self.meta.expr_sym = 'self.get_expr_sym()'  # todo
        self.meta.parsimony = None  # save origin apted in root?
        self.meta.fitness = None

    # def evolve_mutate_filter(self, call_params):
    #     """
    #     sfeh wasd
    #     Mutates a number of float terminal of a tree
    #     """
    #     pass
    #     mode = call_params['mode']  # point/branch/all
    #     yes_observations = call_params.get('yes_observations')  # point/branch/all
    #     mutate_filter = 'gaussian_filter'  # sfeh change?
    #
    #     node_ids = self.core.get_mutatable_nodes()
    #     if mode == 'branch':
    #         node_id = random.choice(node_ids)  # sfeh should this be completely random?
    #         node_ids = tree_node_get_branch(tree, node_id)  # select the whole branch
    #
    #     float_nodes = []
    #     obs_nodes = []
    #     for node_id in node_ids:
    #         if tree_node_get_xtype(tree, node_id) == '2f':
    #             try:
    #                 _ = float(tree_node_get_label(tree, node_id))
    #                 float_nodes.append(node_id)
    #             except ValueError:
    #                 obs_nodes.append(node_id)
    #
    #     if mode == 'point':  # if pointmutation, return one nodeid as list
    #         if yes_observations:
    #             filter_id = [random.choice(float_nodes + obs_nodes)]
    #             if filter_id in float_nodes:
    #                 float_nodes = filter_id
    #             else:
    #                 obs_nodes = filter_id
    #         else:
    #             float_nodes = [random.choice(float_nodes)]
    #
    #     if float_nodes:
    #         for node_id in float_nodes:
    #             val = float(tree_node_get_label(tree, node_id))
    #             val = label_constant_mutate(val, term_type=float, float_decimals=float_decimals, filter_type=mutate_filter)
    #             tree = tree_node_set_label(tree, node_id, val)
    #
    #     if obs_nodes and yes_observations:  # 'filtering' variables when they are from different times
    #         for nid in obs_nodes:
    #             obs_label = tree_node_get_label(tree, nid)
    #
    #             is_negative = obs_label[0] == '-'  # workaround for negative labels
    #             if is_negative:
    #                 obs_label = obs_label[1:]
    #
    #             hello_node = env_vars.obs_infos[obs_label]
    #             hello_node.mutate_filter()
    #             obs_label = hello_node.name
    #
    #             new_obs = '-' + obs_label if is_negative else obs_label
    #             tree = tree_node_set_label(tree, nid, new_obs)

    def eval_parsimony(self, parsimony_distance, origin_cooltree=None, weights=None):
        """

        """
        parsimony = super().eval_parsimony(parsimony_distance, origin_cooltree=origin_cooltree, weights=weights)
        self.meta.parsimony = parsimony
        return parsimony

    def get_oldtree(self):
        """
        Helper to create old tree from cooltree
        fitness, complexity, last_evolution, xtypes, modifys, labels
        """
        label_list, xtype_list, modify_list = self.labellist_modifylist_from_coolcore()
        tree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list).get_uninstanced_tree()
        tree = tree_set_fitness(tree, self.meta.fitness_train)
        tree = tree_set_parsimony(tree, self.meta.parsimony)
        tree = tree_set_last_evolution(tree, self.meta.last_evolution)
        return tree

"""
This file takes care of the strong typing in plagih.
'Types' of nodes:
- Terminals: 2f, 2b
- Functions: f2f, f2b, b2b, b2f, b2f2f (last one being if-then-else)
This notation offers the opportunity to actually search for parts, e. g.
> if '2b' in function:  # This returns True if the label evaluates to boolean
Be careful with if-then-else though. This needs boolean and two float inputs to produce one float
"""


class Selectable:
    """

    """
    pass


class EvalTree(Node):


    def finalize_set_depth(self, depth=0, recursive=True):
        """
        depth=0 is the root node
        """
        self.depth = depth
        max_depth = depth
        if recursive:
            for cc in self.childs:
                cc_depth = cc.finalize_set_depth(depth=depth + 1)
                max_depth = max(cc_depth, max_depth)

        self.childs_depth_max = max_depth

        return max_depth


class ChooseOperators(Selectable):
    """

    """

    def select(self, xtype):
        """
        # def selecting_slower(self, coolxtype):
        #     oplist = self.slowertodo[coolxtype]
        #     return random.choices(oplist[0], weights=oplist[1])[0]
        """
        return self.choose_oparray[xtype]()

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


class ChooseConstants(Selectable):
    """

    """
    # todo random with numpy?
    distributions = {float: [lambda: random.normalvariate(0, 1),
                             lambda: random.normalvariate(1, 1),
                             lambda: random.normalvariate(10, 5),
                             lambda: random.randint(1, 20)],  # 0 has actually no purpose (except as being an action)
                     bool: [lambda: random.choice([True, False])]}

    def selecting(self, xtype):
        """

        """
        value = random.choice(self.distributions[xtype])()
        if xtype == float:  # sfeh int aswell?
            value = float(round(value, self.float_decimals))
            return FloatConstant(value)
        elif xtype == bool:
            return BoolConstant(value)

    def __init__(self, float_decimals=6, path_distrib=None, data_train=None, n_samples=100):
        """

        """
        self.float_decimals = float_decimals
        if Path.is_file(path_distrib):
            lambdadist_as_string = yaml_load(path_distrib)

            # todo how should distributions be loaded?
            # e.g. sample_amount = lambdadist_as_string.get('observed_floats')
            self.terminal_distributions = {float: [], bool: []}
            self.terminal_distributions[float].extend([eval(x) for x in lambdadist_as_string[float]]),
            self.terminal_distributions[bool].extend([eval(x) for x in lambdadist_as_string[bool]])

            # self.sample_floats_from_data(env_vars_obs_infos, data_train, n_samples=n_samples)  # todo
        else:
            logging.info('Opt-in not specified: Distributions-file (for random leaf-node constants) does not exist. Using default set.')


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
    func_list, probability_list = self.operators[coolxtype]
    return np.random.choice(func_list, p=probability_list)
    """

    def selecting(self, xtype):
        """
        Randomly choosing an operator-label for a given xtype.
        choose_oparray3 must be given, as they are different between runs.
        arity can also be set optionally, e.g. for point mutation
        todo DOUBLE-check if this coolxtype is chosen correctly... better: replace it
        """
        return self.observables[xtype]()

    def __init__(self, observations_list):
        """
        :param observations_list: list of all observation names (e.g. ['cartVel', 'cartPos'])
        """

        def observation_select_index(obs_list, max_hist=10):
            """
            chooses variables but weighting how old they are.
            obs_list = ['gain_0', 'gain_1', 'gain_2', 'gain_3', 'gain_4'] -> [0.28, 0.23, 0.19, 0.16, 0.13]
            sfeh: what about larger steps?
            e.g. [0, 1, 2, 3] is good, but [0, 5, 10, 15] is baaaad
            what if variables are not all of same diff?
            """
            obs_list = np.delete(obs_list, np.s_[max_hist:])
            x = len(obs_list)
            fairness_bonus = np.log(x) + 1  # raising the opportunity of historic data just a little...
            p = np.geomspace(1 + fairness_bonus, x + fairness_bonus, num=x)[::-1]  # reverse the geometric series
            p = p / np.sum(p)  # the sum must be equal to 1  # not required with choices
            return np.random.choice(obs_list, p=p)  # returning a function this time

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


class TreeBuilder:
    # class Choosing(Selectable):  # sfeh was
    """
    todo delete?
    """

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
    def __init__(self, operators: ChooseOperators, observations: ChooseObservation, constants: ChooseConstants):
        self.operators = operators
        self.observations = observations
        self.constants = constants

    def choose_term(self, xtype, p_observation=0.5):
        """
        sfeh: float_decimals not required?
        """
        # sfeh 50% chance observatio    n/value
        if random.random() < p_observation:
            try:
                return self.observations.selecting(xtype)
            except Exception:
                pass

        return self.constants.selecting(xtype)

    def choose_operator(self, xtype):
        """
        sfeh this is doppelt gemoppelt
        """
        return self.operators.select(xtype)

    def choose_any(self, xtype, p_op):
        """

        """
        if random.random() < p_op:
            return self.operators.select(xtype)
        else:
            # sfeh add p_term? 0.5?
            return self.choose_term(xtype)

    def invent_core_depth(self, xtype, build_depth, depth=0, p_op=1):  # todo grow method
        """

        """
        if depth < build_depth:
            label = self.choose_any(xtype, p_op)
            depth += 1
            childs = [self.invent_core_depth(xt, build_depth, depth=depth, p_op=p_op) for xt in label.coolxtype[0]]
            return EvolveNode(label, is_fix=False, childs=childs)
        else:
            label = self.choose_term(xtype)
            return EvolveNode(label)

        return core
        # set path?
        # set depth?

    def workaround_remove_tilde(self):
        if isinstance(self.plabel, Usub):  # tilde '~'
            new_core = self.childs[0]
            self.new_core(new_core)

        for cc in self.childs:
            cc.workaround_remove_tilde()


def data_from_csv(df, action):
    """
    Loads .csv data files.
    - Reading the .csv-file (with pandas)
    - renaming column headers
    - saving header info for later use

    Information that we need to extract for each column:
    - choose_xtype choosing a random observation for leaf nodes
    - filtering observation index
        is there an index (past values, e.g. velocity_0, velocity_1) -> performing filter-evolve on the variables index (velocity_2 -> velocity_3)
    - evalaction data_train, data_test, is the action for the regression? -> more than one action might be required (IB has three action dimensions)
        - action min max -> for kernel regression bounded. occuring min and max values might not be the theoretical min/max values


    deprecated:

    """

    """
    1. split col name
    - check whether its an observation
    --> check whether there are indizes
    - check whether its an action
    --> check unique valuea, min, max
    - check if it should be ignored (deprecated action, irrelevant column)
    2. 
    """


    """
    choosing random observations made easy
    """
    # todo remove dat shit

    return


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

    x = timeit.timeit('lel.select(float)', setup='from __main__ import ChooseOparray3\nlel = ChooseOparray3()')
    # y = timeit.timeit('lel.slower(float)', setup='from __main__ import ChooseOparray3\nlel = ChooseOparray3()')
    print(x)


# # todo delete this
# #
# def coolcore_from_oldtree(tree, node_id=root_id):
#     """
#     utility function
#     karoo_tree to cooltree version
#     """
#     label, arity, xtype = tree_node_get_lax_v3(tree, node_id)
#     is_fix = False if tree_node_is_modifiable(tree, node_id) else True
#     coolcore = CoolCore(plabel=plabel, arity=arity, xtype=xtype, is_fix=is_fix)
#
#     childs = tree_node_get_childs(tree, node_id)  # [7, 8, 9]
#     for child_id in childs:
#         pchild = coolcore_from_oldtree(tree, node_id=child_id)
#         coolcore.child_append(pchild)
#
#     if node_id == root_id:
#         coolcore.finalize()
#
#     return coolcore


# def cooltree_from_oldtree(tree, node_id=root_id):
#     """
#     fitness, complexity, last_evolution, xtypes, modifys, labels
#     """
#     try:
#         coolcore = coolcore_from_oldtree(tree, node_id=node_id)
#         parsimony = tree_get_parsimony(tree)
#         fitness = tree_get_fitness(tree)
#         last_evolution = tree_get_last_evolution(tree)
#         cooltree = CoolTree(coolcore)
#         cooltree.meta.fitness_train = fitness
#         cooltree.meta.parsimony = parsimony
#         cooltree.meta.last_evolution = last_evolution
#     except Exception as ex:
#         print(f'cooltree_from_oldtree failed: {ex}')
#         cooltree = None
#     return cooltree


def pop_random(call_params, coolxtype_root=float, from_origin=False):  # todo float is wrongg
    """
    Creates random trees for the population
    """
    build_spec, size_mode, mean_min_max_var, full_or_grow = helper_evolve_params_branch(call_params)

    if from_origin:
        """
        insert a (random) number of branches at the first possible "layer"
        (If all nodes are modifiable, it is the root node. Otherwise, it is a list of nodes that are the childs of the last non-modifiable nodes)
        - get these nodes, randomly choose a subset of those
        - get the amount of nodes we are allowed to add. (max nodes without the core-tree and the nodes we are about to delete)
        - split the amount of nodes up (randomly) and add these new branches to the tree
        """

        layer0_ids = tree_get_mutatable_layer(from_origin, 0)

        build_split = []
        if 'depth' in size_mode:
            for ii in range(len(layer0_ids)):
                build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')
                build_split.append(build_size)

        elif 'nodes' in size_mode:
            build_nodes = choose_build_size(size_mode, mean_min_max_var, force='branch')
            build_split = randomly_split_range(build_nodes, len(layer0_ids))
        else:
            raise

        cooltree = copy.deepcopy(from_origin)
        for i in range(len(layer0_ids)):  # insert branches! get layer every time (node ids might have changed)
            layer0_ids = tree_get_mutatable_layer_lv0(cooltree)
            node_id = layer0_ids[i]
            first_xtype = float  # tree_node_get_xtype(tree, node_id)  # todo
            old_branch = tree_node_get_branch(tree, node_id, karoo=True)
            build_size = build_split[i]

            # cooltree = CoolTree(BuildDummy(float))   # todo deprecated
            core = cooltree.invent_core(size_mode, first_xtype, build_size, full_or_grow)
            tree = tree_insert_subtree(tree, core, old_branch, karoo=True)
    else:
        build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')  # depth, in this case
        """
        was invent_core
        Creates a random core.
        "depth": creates a tree with a desired depth
        "nodes": creates a tree with a desired amount of nodes
        """
        coolcore = Node(BuildDummy(coolxtype_root))
        # todo
        # coolcore.evolve_mutate_branch_random(build_size, choose_oparray3, env_vars.choose_obs,
        #                                      choose_distributions, float_decimals, size_mode=size_mode, full_or_grow=full_or_grow)
        # coolcore.evolve_random_tree_depth(size_mode, coolxtype_root, build_size, full_or_grow)
        coolcore.evolve_random_tree_depth(4)  # todo

    return coolcore


def helper_evolve_params_branch(call_params, tree_depth_max=10, parsimony_max=30):
    """
    The call parameters in the evolution file need to be adjusted
    delete if possible
    """
    build_spec = call_params.get('build_spec')

    size_mode = build_spec['size_mode']

    mean_min_max_var = build_spec.get('mean_min_max_var')  # (base, min, max, normal_distrib)
    mean_min_max_var = list(mean_min_max_var)
    if 'depth' in size_mode:
        max_dummy = tree_depth_max
    elif 'nodes' in size_mode:
        max_dummy = parsimony_max
    else:
        raise

    if mean_min_max_var[2] is None:
        mean_min_max_var[2] = max_dummy
    else:
        mean_min_max_var[2] = min(mean_min_max_var[2], parsimony_max)
    mean_min_max_var = tuple(mean_min_max_var)

    full_or_grow = build_spec['full_or_grow']

    return build_spec, size_mode, mean_min_max_var, full_or_grow


def cooltree_from_labellist(label_list, modify_list=None):
    """
    A lazysolution
    """
    xtype_list = xtypes_from_labels(label_list)
    tree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list).get_uninstanced_tree()
    cooltree = FinalizedNode(coolcore_from_oldtree(tree))
    return cooltree


def coolcore_from_expr(expr, obs_infos):
    label_list = ast_convert_from_expr(expr, build=True)
    xtype_list = xtypes_from_labels(label_list, obs_infos)
    array_tree = Ptree_karoo(label_list, xtype_list).get_uninstanced_tree()
    coolcore = coolcore_from_oldtree(array_tree)
    return coolcore


def some_quick_test():
    coollist = '[*, cartPos, [**, [+, cartPos, 1.077166], 2.0]]'
    coollist2 = '[Ifte(fix),[<,cartVel,0],0,2]'
    def coolnode_from_txt(txt):
        if txt[0] == '[' and txt[-1] == ']':  # '[Ifte(fix),[<,cartVel,0],0,2]'
            txt = txt[1:-1]  # 'Ifte(fix),[<,cartVel,0],0,2'
            label = txt.split(',')[0]  # Ifte(fix)
            print(label, txt)


def tree_check_deep(tree):
    """
    Performs all checks that we currently have
    # sfeh do not use this if trees are safely generated
    # sfeh check meta values in separate method? update those aswell?
    """
    if not tree_check_quick(tree):
        tree_works = False
    elif not tree_check_rebuild(tree):
        tree_works = False
    else:
        tree_works = True

    return tree_works


class ObservationNode(Node):

    def __init__(self, name):
        super().__init__(name)
        self.name = name


if __name__ == '__main__':

    tree = ['Ifte',
            ['Orb',
             ['<', ['cartPos', -1]],
             ['Andb',
              ['<', ['cartPos', 0.1]],
              ['<', ['cartVel', -0.05]]]], 2,
            ['Ifte',
             ['Andb',
              ['Andb',
               ['>', ['cartPos', -0.45]],
               ['<', ['cartPos', -0.05]]],
              ['<', ['cartVel', -0.5]]], 0,
             ['Ifte',
              ['<', ['cartVel', 0]], 0, 2]]]

    trexpr = '(Ifte, (Orb, (cartPos < -1), (Andb, (cartPos < 0.1), (cartVel < -0.05))), 2, (Ifte, (Andb, (Andb, (cartPos > -0.45), (cartPos < -0.05)), (cartVel < -0.5)), 0, (Ifte, (cartVel < 0), 0, 2)))'
    trexpr = '(Ifte, (cartVel < 0), 0, 2)'
    # trexpr = plagih_sympify(trexpr)

    coolcore = TreeBuilder()
    coolcore.construct_random_tree_depth(4)  # todo
    print(coolcore)


def randomly_split_range(range_max, num_splits):
    """
    delete this?
    split a integer range randomly into parts
    [1..100] -> [33, 15, 52] (0 is allowed)
    """

    # tmp_distributions = random.sample(range(1, range_max), num_splits)
    # d_sum = sum(tmp_distributions)
    # d_list = [int(round(range_max*(x/d_sum), 0)) for x in tmp_distributions]
    sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
    d_sum = sum(sample_dist)  # 1.5
    sample_dist = [x / d_sum for x in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
    sample_dist = [x * range_max for x in sample_dist]  # [12, 60, 28] -> for 100 nodes
    sample_dist = [int(round(x, 0)) for x in sample_dist]  # make them useable ints

    # sfeh workaround, this makes exactly the correct range by changing the most extreme entry
    helper_diff = range_max - sum(sample_dist)
    if sum(sample_dist) < range_max:
        smallest = sample_dist.index(min(sample_dist))
        sample_dist[smallest] += helper_diff

    if sum(sample_dist) > range_max:
        greatest = sample_dist.index(max(sample_dist))
        sample_dist[greatest] += helper_diff

    return sample_dist


def shuffle_tree_construction(size):
    """
    delete this?
    """
    terms = random.randint(0, size - 1)  # -1 because at least one 'func'
    cons_buf = ['term'] * terms + ['func'] * (size - terms)
    np.random.shuffle(cons_buf)
    return cons_buf


def choose_build_size(size_mode, mean_min_max_var, cooltree=None, nodepath=None, force=None):
    """
    delete this?
    Very unified utility function that returns the required tree size from the following parameters
    # branch_nodes, branch_depth, tree_depth, tree_nodes

    It can either return a tree depth or an amount of tree nodes
    """
    mean, size_min, size_max, size_variance = mean_min_max_var
    if size_mode == 'branch_nodes' or size_mode == 'branch_depth' or force == 'branch':
        relative_size = 0
    else:
        if cooltree and nodepath:
            pass
        else:
            raise Exception('No tree or node is given for computing the relative size')

        if size_mode == 'tree_depth':
            tree_size = cooltree.core.childs_depth_max
            print('tree_size = cooltree.core.childs_depth_max:', tree_size)
            if tree_size is None and 'delete_this':
                raise Exception('ASDASDASD')
            node_size = len(nodepath)
        elif size_mode == 'tree_nodes':
            tree_size = len(cooltree)
            print('len(cooltree)?:', len(cooltree))
            node_size = len(cooltree.get_nodepath(nodepath))
            print('cooltree.get_nodepath(nodepath)?:', cooltree.get_nodepath(nodepath))
        else:
            raise Exception('Sizemode not known?')

        relative_size = tree_size - node_size
        print('asdasd', relative_size)

    build_size = int(random.normalvariate(mean, size_variance))
    if size_max is not None:
        build_size = min(size_max - relative_size, build_size)
    build_size = max(size_min, build_size)

    return int(build_size)


def tree_check_children(tree, karoo=True):
    """
    delete this?
    A method to check if a tree is plausible. aka:
    - do the values in c1, c2, c3 link to correct
    """
    if not karoo:
        tree = tree_convert_pcore_to_karoo(tree)

    id_list = []
    c_list = []
    for n in range(1, len(tree[3])):
        for child in range(0, 3):
            if tree[N_c1 + child][n] != '':
                c_list.append(int(tree[N_c1 + child][n]))
        id_list.append(int(tree[N_id][n]))
    if sum(c_list) == sum(id_list) - 1:
        return True
    else:
        return False


def tree_check_rebuild(tree):
    """
    sfeh
    Check if a valid tree can be rebuilt from its expression
    sfeh: the expression must currently not be equal.
    The expression can include separate '~' (Usub) nodes, which makes expressions not completely equal
    """
    return True


def tree_check_node_label_info(tree, karoo=True):
    """
    sfeh
    A method to check if a tree is type consistant:
    - do the values in c1, c2, c3 link to its parent?
    """
    return True


def tree_check_types(tree, karoo=True):
    """
    sfeh
    A method to check if a tree is type consistant:
    - do the values in c1, c2, c3 link to its parent?
    """
    return True

