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
from plagih.plagih_tree import *
from plagih.tree_distances.tree_edit_distance import apted_distance
from plagih.plagih_sympy_extras import sympy_symbol_defaults, expr_sympify
from plagih.plagih_types import *


class CoolCore:
    """
    The core is the structure of a plagih gp-tree.
    It recursively holds the nodes of a tree; every tree has a list of potential children.
    Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)
    """

    def __init__(self, plabel: Plabel, is_fix=False, childs=None):

        self.plabel = plabel

        self.is_fix = is_fix
        self.childs = childs if childs is not None else []  # maybe must be updateds recursively

        self.complete = False  # if the node is correct/done/okay
        self.nodepath = []  # go to node x (sfeh: this was a deque?) e.g. [0, 1, 0]

        self.depth = None
        self.childs_depth_max = None  # sfeh, is this automatically correct? always?

    def __hash__(self):
        """
        Hashing the label-list as string should be sufficient.
        Is there a chance for same hash values within oioulations? That would be very bad.
        ==
        Note:
            Untill version 1.0, the return value was hashed like this:
            # return hash(','.join([str(x) for x in labellist]))
            However, the hashes change between runs (a python security feature)
        """
        # labellist = self.get_labellist_breath()
        # return ','.join([str(x) for x in labellist])
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

    def finalize_structure(self):
        """
        Finalize a single node
        """
        self.finalize_set_depth(depth=self.depth or 0)  # sfeh the starting depth? hmm, why set it none anyways. aye. rewrite

    def workaround_remove_tilde(self):
        if isinstance(self.plabel, Usub):  # tilde '~'
            new_core = self.childs[0]
            self.new_core(new_core)

        for cc in self.childs:
            cc.workaround_remove_tilde()

    def new_core(self, new_core: 'CoolCore'):
        """

        """
        self.plabel = new_core.plabel
        self.childs = new_core.childs if new_core.childs else []  # maybe must be updated recursively
        # self.is_fix = new_core.is_fix  # debatable
        # self.complete = new_core.complete  # if the node is correct/done/okay
        # depth needs to be fixed?

    def is_root(self):
        """
        delete this?
        todo
        """
        if len(self.nodepath) == 1:
            return True
        else:
            return False

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

    def finalize_set_nodepath(self, nodepath):
        """
        [0,2,1,0,0]
        """
        self.nodepath = nodepath
        for ii, child in enumerate(self.childs):
            nodepath_child = nodepath + [ii]
            child.finalize_set_nodepath(nodepath_child)

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

    def get_tree_depth(self):
        """
        """
        return self.childs_depth_max

    def tree_get_mutatable_layer(self):
        """

        """
        self.get_nodes_at_depth(layer=0)

    def get_lowest_nodes(self):
        """
        todo test
        return the nodes on the lowest level (breath-first).
        Might be operators if the tree is still in construction
        ?? return node-positions or node-objects?
        ===How?===
        exit condition is "not having child nodes"
        """
        if not self.childs:
            return [self.nodepath]  # returning as list, easiert to flatten later with itertools
        else:
            return list(itertools.chain(*[cc.get_lowest_nodes() for cc in self.childs]))

    def construct_append_layer_depth(self, choose_obs, choose_oparray3, choose_distributions, float_decimals, construct='full'):
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

                self.childs.append(CoolCore(plabel=plabel))
            return len(self.childs)
        else:
            appended_nodes = 0
            for cc in self.childs:
                appended_nodes += cc.construct_append_layer_depth(choose_obs, choose_oparray3, choose_distributions, float_decimals, construct=construct)
            return appended_nodes

    def labellist_modifylist_from_coolcore(self):
        """
        breadth-first labellist (karoo)
        """
        label_list = []
        xtype_list = []
        modify_list = []
        max_depth = self.childs_depth_max
        for depth in range(0, max_depth + 1):
            labels_at_depth = [x.label for x in self.get_nodes_at_depth(depth)]
            xtypes_at_depth = [x.xtype for x in self.get_nodes_at_depth(depth)]
            modify_at_depth = [0 if x.is_fix else 1 for x in self.get_nodes_at_depth(depth)]
            label_list.extend(labels_at_depth)
            xtype_list.extend(xtypes_at_depth)
            modify_list.extend(modify_at_depth)
        return label_list, xtype_list, modify_list

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

    def prune_depth(self, max_depth):
        # """
        # reduces the depth of a Tree to depth (in case it is too deep).
        # # sfeh prune node_count?
        #
        # """
        #
        # nodes = []
        #
        # for node_id in range(root_id, len(tree[3])):
        #
        #     node_depth = tree_node_get_depth(tree, node_id)
        #     node_arity = tree_node_get_arity(tree, node_id)
        #     if node_depth == max_depth and node_arity > 0:  # replace this node with terminal
        #         label = tree_node_get_label(tree, node_id)
        #         xtype = xtype_get_from_label(label, obs_krazy)
        #         tree = tree_node_set_arity(tree, node_id, 0)
        #         new_term = choose_term(xtype[-2:], choose_obs, choose_distributions, float_decimals)  # replace label
        #         tree = tree_node_set_label(tree, node_id, new_term)
        #
        #     elif tree_node_get_depth(tree, node_id) > max_depth:  # record nodes deeper than the maximum allowed Tree depth
        #         nodes.append(node_id)
        #
        # tree = np.delete(tree, nodes, axis=1)  # delete nodes deeper than the maximum allowed Tree depth
        # tree = evolve_node_arity_fix(tree)  # fix all node arities
        #
        # return tree
        raise Exception('Sfeh needs to do this')  # sfeh asdasd only relevant when blind crossover? check during build process?

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

    def check_all(self):
        """

        """
        if self.plabel.arity != len(self.childs):
            raise
        return True

    def get_expr_raw(self, reducible=None, obs_names=None):
        """
        accumulate and return the complete expression the tree holds recursively
        """
        # if self.expr_raw is None:  # sfeh?
        if self.plabel.arity == 0:
            return f'{self.plabel.sym_str}'
        else:
            my_expr = self.plabel.sym_str
            child_expr_list = [cc.get_expr_raw(reducible=reducible, obs_names=obs_names) for cc in self.childs]
            # if reducible:
            #     # my_expr = op[self.label]['sym_reduce'] or my_expr
            #     # symloc = sympy_symbol_defaults(obs_names)  # todo solve the problem... new version of sympy?
            #     xxx = plagih_sympify(my_expr.format(*child_expr_list), eval_locals=symloc)  # sfeh the xxx variable
            #     return xxx
            return my_expr.format(*child_expr_list)  # f'cos({})'([33]) does not work. *list makes the list args :D

    def node_insert_width(self, node):
        """
        adds a node to an unfinished tree
        what is this even good for, Simon...
        """
        if len(self.childs) < self.plabel.arity:  #
            self.child_append(node)
            return True
        else:
            for ch in self.childs:
                ch.node_insert_width(node)

        return

    def nodepath_insert_branch(self, nodepath, coolbranch):
        """
        inserting a branch into the place of a node
        """
        if len(nodepath) > 1:
            self.childs[nodepath[0]].insert_branch(nodepath[1:], coolbranch)
        else:  # [1] -> set child 1
            self.childs[nodepath[0]] = coolbranch

    def get_node_from_nodepath(self, target_nodepath):
        """
        example coordinates: [0, 2, 1] -> node is at depth 2, accessable by going to childs 0 -> 2 -> 1.
        (root is always 0)
        """
        if len(target_nodepath) > len(self.nodepath):
            next_child = target_nodepath[len(self.nodepath)]
            return self.childs[next_child].get_from_path(target_nodepath)
        else:
            return self

    def node_add(self, nodepath, child):
        """
        adds a child-node to an existing node
        """
        node = self.get_node_from_nodepath(nodepath)
        node.child_append(child)

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
        return "{{{}{}}}".format(self.plabel.label, ''.join(childargs))

    def child_append(self, child):
        self.childs.append(child)

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

    def set_fix_nodes(self, origin_coolcore: 'CoolCore'):
        """
        Resetting the fix nodes from the cooltree.
        sfeh: this is not required after complete cooltree use
        """
        # sfeh not necessary in the future
        if origin_coolcore.is_fix:
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.set_fix_nodes(origin_coolcore.childs[ii])

    def pretty_format(self):
        """

        """
        layerlabellist = self.get_layer_labellist()
        return '\n'.join([', '.join([str(lbl) for lbl in layer]) for layer in
                          layerlabellist])  # lbl-needed, sometines those are float values

    def evolve_mutate_branch_depth(self, depth_goal, choose_oparray3, choose_obs, choose_distributions, float_decimals, mutate='branch', full_or_grow='full'):
        """
        todo other version
        currently only one branch
        """
        coolxtype = self.plabel.coolxtype

        # sfeh: making the root node (todo?)
        if depth_goal == 1:
            # return just one node
            self.plabel = choose_term(coolxtype[1], choose_obs, choose_distributions, float_decimals)  # sfeh update node plabel
            # delete this, deprecated in class
            # coolcore = CoolCore(plabel=plabel)
            # return coolcore
        else:
            self.plabel = choose_operator(coolxtype[1], choose_oparray3)
            # coolcore = CoolCore(plabel=plabel)  # sfeh delete this

            depth = 1

            while depth < depth_goal - 1:  # depth_goal-1 as terminal nodes get added at the end
                self.construct_append_layer_depth(choose_obs, choose_oparray3, choose_distributions, float_decimals, construct=full_or_grow)
                depth += 1
                # sfeh todo gleichmäßige Verteilung
                # if full_or_grow == 'grow':
                #     coolcore.construct_append_layer_depth(choose_obs, choose_oparray3, choose_distributions, float_decimals, construct)
                # else:
                #     cons_buf = ['func'] * len(oxtype_buffer)

            else:
                # build terminal nodes
                self.construct_append_layer_depth(choose_obs, choose_oparray3, choose_distributions, float_decimals, construct='term')

    def evolve_mutate_filter(self, choose_oparray3, choose_obs, choose_distributions, float_decimals):
        """
        filtger the nodews in a single tree
        """
        self.plabel.mutate_filter()
        if self.plabel.arity > 0:
            for cc in self.childs:
                cc.evolve_mutate_filter(choose_oparray3, choose_obs, choose_distributions, float_decimals)

        self.finalize_structure()

    def evolve_mutate_point(self, choose_oparray3, choose_obs, choose_distributions, float_decimals):
        """
        Mutate a single mutatable point in any Tree.
        """
        if self.plabel.arity > 0:
            self.plabel = choose_operator(self.plabel.coolxtype, choose_oparray3)  # Function is same type, same arity
        else:
            print('hhhhh', self.plabel)
            self.plabel = choose_term(self.plabel.coolxtype[1], choose_obs, choose_distributions, float_decimals)  # 3 -> '2f' -> 5

        self.finalize_structure()

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
        self.finalize_structure()

    def export_visualization_latex(self):
        """
        todo
        """
        return None

    def get_layer_nodelist(self):
        """
        [+,[*,-]]
        """
        max_depth = self.childs_depth_max
        label_layer_list = []
        for depth in range(0, max_depth + 1):
            label_layer_list.append([node_on_lvl for node_on_lvl in self.get_nodes_at_depth(depth)])
        return label_layer_list


class CoolTree(CoolCore):
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

    def __init__(self, plabel:Plabel):
        super().__init__(plabel)
        # self.core = coolcore  # todo sfeh
        self.meta = self.PtreeMeta()
        self.history = deque([], maxlen=10)  # sfeh arbitrary value of 10 historic metainfo of this tree
        self.finalize_structure()

    # def write_histogram(self):

    def finalize(self):
        """
        (only in root node)
        """
        self.finalize_set_depth()
        self.finalize_set_nodepath([0])
        self.workaround_remove_tilde()

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

        self.finalize_structure()

    def evolve_mutate_point_random(self, choose_oparray3, choose_obs, choose_distributions, float_decimals):
        """
        Mutate a single mutatable point in any Tree.
        """
        # 1. choose a node
        node_list = self.get_mutatable_nodes()
        node = random.choice(node_list)
        node.evolve_mutate_point(choose_oparray3, choose_obs, choose_distributions, float_decimals)

        self.finalize_structure()

    def evolve_mutate_branch_random(self, cool_build_size, choose_oparray3, choose_obs, choose_distributions, float_decimals, size_mode='depth', full_or_grow='full'):
        """

        """
        node_list = self.get_mutatable_nodes()
        node = random.choice(node_list)
        node.evolve_mutate_branch_depth(cool_build_size, choose_oparray3, choose_obs, choose_distributions, float_decimals, full_or_grow=full_or_grow)

        self.finalize_structure()


def construct_coolcore_depth(coolxtype, size_mode, mean_min_max_var, choose_obs, choose_oparray3, choose_distributions, float_decimals, full_or_grow=None):
    """
    TODO REPLACE OLD VERSION
    """
    cooltree = CoolTree()
    cooltree.evolve_mutate_branch_depth()


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


def pop_random(call_params, from_origin=False):
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

        tree = from_origin.copy()
        for i in range(len(layer0_ids)):  # insert branches! get layer every time (node ids might have changed)
            layer0_ids = tree_get_mutatable_layer_lv0(tree)
            node_id = layer0_ids[i]
            first_xtype = tree_node_get_xtype(tree, node_id)
            old_branch = tree_node_get_branch(tree, node_id, karoo=True)
            build_size = build_split[i]

            core = self.invent_core(size_mode, first_xtype, build_size, full_or_grow)
            tree = tree_insert_subtree(tree, core, old_branch, karoo=True)
    else:
        action_xtype = self.env_vars.eval_action.xtype
        build_size = choose_build_size(size_mode, mean_min_max_var, force='branch')
        core = self.invent_core(size_mode, action_xtype, build_size, full_or_grow)
        tree = tree_convert_pcore_to_karoo(core)

    return tree


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
    cooltree = CoolTree(coolcore_from_oldtree(tree))
    return cooltree


def coolcore_from_expr(expr, obs_infos):
    label_list = ast_convert_from_expr(expr, build=True)
    xtype_list = xtypes_from_labels(label_list, obs_infos)
    array_tree = Ptree_karoo(label_list, xtype_list).get_uninstanced_tree()
    coolcore = coolcore_from_oldtree(array_tree)
    return coolcore


def cooltree_from_expr(expr, obs_infos):
    coolcore = coolcore_from_expr(expr, obs_infos)
    cooltree = CoolTree(coolcore)
    return cooltree


def some_quick_test():
    coollist = '[*, cartPos, [**, [+, cartPos, 1.077166], 2.0]]'
    coollist2 = '[Ifte(fix),[<,cartVel,0],0,2]'
    def coolnode_from_txt(txt):
        if txt[0] == '[' and txt[-1] == ']':  # '[Ifte(fix),[<,cartVel,0],0,2]'
            txt = txt[1:-1]  # 'Ifte(fix),[<,cartVel,0],0,2'
            label = txt.split(',')[0]  # Ifte(fix)
            print(label, txt)
            


class ObservationNode(CoolCore):

    def __init__(self, name):
        super().__init__(name)
        self.name = name


# def coolcore_from_brackets(coolbrackets):
#     test = ['+', 1, ['-', [2, 3]]]
# # some_quick_test()


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
    trexpr = plagih_sympify(trexpr)


    def funtest(pos, vel):
        if pos < -1 or (pos < 0.1 and vel < -0.05):
            return 2
        else:
            if (pos > -0.45 and pos < -0.05) and vel < 0.02:
                return 0
            else:
                if vel < 0:
                    return 0
                else:
                    return 2


    some_quick_test()
