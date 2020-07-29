from collections import deque
from plagih.modules.plagih_types import *
from plagih.modules.plagih_tree import *
from plagih.tree_distances.tree_edit_distance import apted_distance


class Plabel:
    def __init__(self, label, xtype=None):
        self.label = label

        xtype = xtype if xtype else xtype_get_from_label(label)
        arity = label_get_arity(label)
        type_inputs = float if xtype[-2:] == 'f2' else bool  # sfeh

        # directly related to the label
        self.arity = arity
        self.input_types = [bool, float, float] if label == 'Ifte' else [type_inputs] * arity
        self.type_output = float if xtype[-2:] == '2f' else bool if '2b' == xtype[-2:] else None
        self.xtype = xtype  # obsolete?


class CoolCore:
    """
    x = label_list = ['Ifte', '<', 0, 2, 'cartVel', 0]
    label_list = ['+', 1, 'b']
    x = Pnode('+', childs=[Pnode('1'), Pnode('2')])
    """

    def __init__(self, label=None, is_fix=False, complete=False, arity=None, xtype=None, childs=None, depth=None, nodepath=None):

        xtype = xtype if xtype else xtype_get_from_label(label)
        arity = label_get_arity(label) if arity is None else arity
        type_inputs = float if xtype[-2:] == 'f2' else bool  # sfeh

        self.label = label

        # directly related to the label
        self.arity = arity
        self.input_types = [bool, float, float] if label == 'Ifte' else [type_inputs] * arity
        self.type_output = float if xtype[-2:] == '2f' else bool  # sfeh
        self.xtype = xtype  # obsolete?

        self.is_fix = is_fix

        self.complete = complete  # if the node is correct/done/okay

        self.childs = childs if childs is not None else []  # maybe must be updateds recursively

        # changes after insertion
        self.nodepath = nodepath if nodepath else []  # go to node x (sfeh: this was a deque?)
        self.depth = depth
        self.childs_depth_max = None

        self.label_type = type_inputs

    def finalize(self):
        self.finalize_set_depth()
        self.finalize_set_nodepath([0])
        self.workaround_remove_tilde()

    def workaround_remove_tilde(self):
        if self.label == '~':
            new_core = self.childs[0]
            self.new_core(new_core)

        for cc in self.childs:
            cc.workaround_remove_tilde()

    def new_node(self, new_label, new_arity, new_xtype, childs=None):
        """

        """
        self.label = new_label
        self.arity = new_arity
        self.xtype = new_xtype
        if childs is not None:
            self.childs = childs

    def new_core(self, new_core: 'CoolCore'):
        self.label = new_core.label
        self.arity = new_core.arity
        self.xtype = new_core.xtype
        self.childs = new_core.childs

        type_inputs = float if new_core.xtype[-2:] == 'f2' else bool  # sfeh
        self.arity = label_get_arity(new_core.label) if new_core.arity is None else new_core.arity
        self.input_types = [bool, float, float] if new_core.label == 'Ifte' else [type_inputs] * self.arity
        self.type_output = float if self.xtype[-2:] == '2f' else bool
        self.xtype = new_core.xtype  # obsolete?
        self.label_type = type_inputs

        self.is_fix = new_core.is_fix

        self.complete = new_core.complete  # if the node is correct/done/okay

        self.childs = new_core.childs if new_core.childs else []  # maybe must be updateds recursively

        # changes after insertion
        # self.nodepath = new_core.nodepath if nodepath else []  # go to node x (sfeh: this was a deque?)  # sfeh not possible at the end!
        # self.depth = depth  # sfeh not possible
        # self.childs_depth_max = None  # sfeh not possible

    def finalize_set_nodepath(self, nodepath):
        """
        aka nodepath?
        [0,2,1,0,0]
        """
        self.nodepath = nodepath
        for ii, child in enumerate(self.childs):
            nodepath_child = nodepath + [ii]
            child.finalize_set_nodepath(nodepath_child)

    def get_labellist(self):
        label_list = []
        max_depth = self.childs_depth_max
        for depth in range(0, max_depth + 1):
            labels_at_depth = [x.label for x in self.get_nodes_at_depth(depth)]
            label_list.extend(labels_at_depth)
        return label_list

    def labellist_modifylist_from_coolcore(self):
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
        sum_layers=False, get_closest=True, return_all_layers=False

        """
        if self.depth < goal_depth:
            return sum([child.get_nodes_at_depth(goal_depth, only_mutable=only_mutable, get_closest_depth=get_closest_depth) for child in self.childs], [])
        else:
            if only_mutable and self.is_fix:
                return []
            if get_closest_depth and self.depth != goal_depth:
                return []

            return [self]

    def get_nodes_to_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """
        sum_layers=False, get_closest=True, return_all_layers=False

        """
        child_results = []
        if self.depth < goal_depth:
            child_results = sum([child.get_nodes_to_depth(goal_depth, only_mutable=only_mutable, force_depth=get_closest_depth) for child in self.childs], [])

        if only_mutable and self.is_fix or \
                get_closest_depth and self.depth != goal_depth:
            my_result = []

        else:
            my_result = [self]

        return my_result + child_results

    def reduce_me(self, obs_krazy):
        # todo reduce me is obviously bullshit crapshit. lets have a new idea.
        expr_raw = self.get_expr_raw(reduceable=True)
        try:
            expr_sym = expr_sympify(expr_raw)
        except:
            raise Exception(f'Sympify failed. \n{expr_raw}')

        new_core = coolcore_from_expr(expr_sym, obs_krazy)
        if len(new_core) < len(self):
            self.new_core(new_core)
        elif len(new_core) > len(self):
            raise Exception(f'Reduced core is even more complex than before  ({len(new_core)}, {len(self)}). expr_raw: {expr_raw}')  # \nold_core:{self}\nnew_core: {new_core} May happen with sympification and Usub.
            # example: Tree sympification did not work: Reduced core is even more complex than before. expr_raw: sign(Mini(((Velocity_2 * -0.790706) - sqrt(Gain_0)), (-0.569271 - Velocity_9)))
            # old_core:[sign, [Mini, [-, [*, Velocity_2, -0.790706], [sqrt, Gain_0]], [-, -0.569271, Velocity_9]]]
            # new_core: [sign, [Mini, [-, [Usub, [sqrt, Gain_0]], [*, 0.790706, Velocity_2]], [-, -Velocity_9, 0.569271]]]
        return

    def get_mutatable_nodes(self):
        """

        """
        add_me = [] if self.is_fix else [self]
        return add_me + sum([cc.get_mutatable_nodes() for cc in self.childs], [])

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
        if self.arity != len(self.childs):
            raise

        return True

    def get_expr_raw(self, reduceable=None):
        # if self.expr_raw is None:  # sfeh?
        if self.arity == 0:
            return f'{self.label}'
        else:
            my_expr = op[self.label]['sym_str']
            child_expr_list = [child.get_expr_raw(reduceable=reduceable) for child in self.childs]
            if reduceable:
                my_expr_new = op[self.label]['sym_reduce']
                my_expr = my_expr_new if my_expr_new is not None else my_expr
            return my_expr.format(*child_expr_list)  # f'cos({})'([33]) does not work. *list makes the list args :D

    def node_insert_width(self, node, depth=0):
        """
        adds a node to an unfinished tree
        """

        if len(self.childs) < self.arity:  #
            self.child_append(node)
            return True
        else:
            for ch in self.childs:
                ch.node_insert_width(node)

        return

    def find_me(self, nodepath):
        try:
            self.childs[nodepath[0]].find_me(nodepath[1:])
        except:
            return True

    def insert_branch(self, nodepath, coolbranch):
        """
        inserting a branch into the place of a node
        """

        if len(nodepath) > 1:
            self.childs[nodepath[0]].insert_branch(nodepath[1:], coolbranch)
        else:  # [1] -> set child 1
            self.childs[nodepath[0]] = coolbranch

    def get_pycode(self):

        if self.arity == 0:
            if terminal_label_is_observation(self.label):
                ib_sfeh_dict = {'p': 'SetPoint',
                                'v': 'Velocity',
                                'g': 'Gain',
                                'h': 'Shift',
                                'f': 'Fatigue',
                                'c': 'Consumption'}
                ib_sfeh_rev = {v: k for k, v in ib_sfeh_dict.items()}
                is_negative = self.label[0] == '-'
                use_label = self.label[1:] if is_negative else self.label
                obs_family, obs_time = observation_get_family_and_time(use_label, none_return=None)
                if obs_time is None:
                    pass
                else:
                    geth_name = ib_sfeh_rev[obs_family]
                    return f"{'-' if is_negative else ''}self.get_h('{geth_name}', {obs_time})"

            return f'{self.label}'
        else:
            results = []
            for child in self.childs:
                results.append(child.get_pycode())  # = tree_node_get_label(tree, int(child))
            return op[self.label]['pycode'].format(*results)  # abs -> lambda a: 'abs({})'.formadt(a) (result1)

    def get_apted_notation(self):
        """
        {+{Ifte{True}{1}{2}}{3}}
        """

        childargs = [cc.get_apted_notation() for cc in self.childs]
        return "{{{}{}}}".format(self.label, ''.join(childargs))

    def __str__(self):
        """
        """
        label_info = self.label
        meta_str = []
        if self.is_fix:
            meta_str.append('is_fix: True')
        if meta_str:
            label_info += f"{{{', '.join(meta_str)}}}"

        if not self.childs:
            return f"{label_info}"
        else:
            childstr = ', '.join([str(x) for x in self.childs])
            return f"[{label_info}, {childstr}]"

    def __len__(self):
        """
        """
        return 1 + sum([len(cc) for cc in self.childs])

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

    def node_from_path(self, path_from_here):
        """
        example coordinates: [0, 2, 1] -> node is at depth 2, accessable by going to childs 0 -> 2 -> 1.
        (root is always 0)
        """
        if path_from_here:
            next_node = path_from_here[0]
            path_from_here = path_from_here[1:]
            return self.childs[next_node].get_from_path(path_from_here)
        else:
            return self

    def node_add(self, nodepath, child):
        """
        adds a child-node to an existing node
        """
        node = self.node_from_path(nodepath)
        node.child_append(child)

    def get_observation_list(self):
        my_return = []
        if self.arity == 0:
            if terminal_label_is_observation(self.label):
                my_return = [self.label]

        for cc in self.childs:
            my_return.extend(cc.get_observation_list())

        return my_return

    def set_fix_nodes(self, origin_coolcore: 'CoolCore'):
        # sfeh not necessary in the future
        if origin_coolcore.is_fix:
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.set_fix_nodes(origin_coolcore.childs[ii])


class CoolTree:
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

    def __init__(self, coolcore: CoolCore):
        self.core = coolcore
        self.meta = self.PtreeMeta()
        self.history = deque([], maxlen=10)  # sfeh arbitrary value of 10 historic metainfo of this tree
        self.finalize_structure()

    def __hash__(self):
        """
        Hashing the label-list as string should be sufficient.
        Is there a chance for same hash values within oioulations? That would be very bad.
        """
        core = self.core.get_labellist()
        return hash(','.join([str(x) for x in core]))

    # def set_meta(self, meta):
    #     self.history.append(self.meta)
    #
    # def set_meta(self, fitness_train, parsimony, last_evolution, expr_raw, expr_sym):
    #     self.meta.fitness_train = fitness_train
    #     self.meta.parsimony = parsimony
    #     self.meta.last_evolution = last_evolution
    #     self.meta.expr_raw = expr_raw
    #     self.meta.expr_sym = expr_sym
    #     # set flags if done?

    def set_fix_nodes(self, origin_tree: 'CoolTree'):
        """
        Resetting the fix nodes from the cooltree.
        sfeh: this is not required after complete cooltree use
        """
        if origin_tree is not None:
            self.core.set_fix_nodes(origin_tree.core)
        else:
            return None

    def get_layer_nodelist(self):
        max_depth = self.core.childs_depth_max
        label_layer_list = []
        for depth in range(0, max_depth + 1):
            label_layer_list.append([node_on_lvl for node_on_lvl in self.core.get_nodes_at_depth(depth)])
        return label_layer_list

    def get_layer_labellist(self):
        layerlist = self.get_layer_nodelist()
        labellayerlist = []
        for depth_list in layerlist:
            labellayerlist.append([node_on_lvl.label for node_on_lvl in depth_list])
        return labellayerlist

    def pretty_format(self):
        layerlabellist = self.get_layer_labellist()
        return '\n'.join([', '.join([str(lbl) for lbl in layer]) for layer in layerlabellist])  # lbl-needed, sometines those are float values

    # def workaround_normalize_exponentiation(self):
    #     self.core.workaround_normalize_exponentiation()
    #     # self.finish_nodes()  #
    #     self.finalize_completely()  #

    def finalize_structure(self):
        self.core.finalize_set_depth(depth=self.core.depth)

    def finalize_completely(self):
        self.finalize_structure()
        self.complete = True

    def finalize_meta(self):
        # todo set_meta
        self.meta.expr_raw = self.get_expr_raw()
        self.meta.expr_sym = self.get_expr_sym()
        self.meta.parsimony = None  # save origin apted in root?
        self.meta.fitness = None

    def __len__(self):
        """
        The amount of nodes
        """
        return len(self.core)

    def get_apted_notation(self):
        return self.core.get_apted_notation()

    def check_all(self):
        if not self.core.check_all():
            raise
        return True

    def get_expr_raw(self, symred=None):
        expr_raw = self.core.get_expr_raw(reduceable=symred)
        return expr_raw

    def get_expr_sym(self):
        expr_raw = self.core.get_expr_raw()
        return expr_sympify(expr_raw)

    def get_expr(self, sympified=False, symred=None):
        expr = self.core.get_expr_raw(reduceable=symred)
        if sympified:
            expr = expr_sympify(expr)  # may cause exception
        return expr

    def insert_branch(self, nodepath, coolbranch):
        self.core.complete = False

        if len(nodepath) == 1:  # [1] -> set child 1
            self.core = coolbranch
        else:
            self.core = self.core.insert_branch(nodepath[1:], coolbranch)
        self.finalize_structure()

    def get_pycode(self):
        """
        For generating actual programming code
        """
        return self.core.get_pycode()

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

    def childs_insert_list_width(self, node_list):
        """

        """
        iter_nodes = iter(node_list)
        while True:
            try:
                node = next(iter_nodes)
            except StopIteration:
                break

            self.core.node_insert_width(node)

    def __str__(self):
        """
        Printing the nodes as nested array structure.
        # only printing the nodes, no meta
        """
        return f"{self.core}"

    def get_nodes_at_depth(self, lvl_goal, only_mutable=False, get_closest_depth=False):
        """
        Returns a list with mutatable ids which are *lvl_goal* layers away from non modifiable nodes
        last_leaves: if you want so save all leave nodes aswell
        """

        return self.core.get_nodes_at_depth(lvl_goal, only_mutable=only_mutable, get_closest_depth=get_closest_depth)

    def evolve_reduce(self, obs_krazy=None, completely=True):
        """
            Reducing a tree to its most basic form with sympify.
            (completely = False: reduce just one branch. if you wanted to have more complexity)
            """
        length_before = len(self)
        if completely:  # reduce the complete tree
            coolcores_lv0 = self.get_nodes_at_depth(0, only_mutable=True)
            for coolc in coolcores_lv0:
                coolc.reduce_me(obs_krazy)
        else:
            cool_nodes = self.core.get_mutatable_nodes()
            cool_functions = [x for x in cool_nodes if x.arity > 0]
            if cool_functions:
                chosen = random.choice(cool_functions)
                chosen.reduce_me(obs_krazy)
        if length_before < len(self):
            print_e(f'FFS Trees just become larger? {self.get_expr_raw()}')
        # self.meta.clear()
        self.finalize_structure()

    def eval_parsimony(self, parsimony_distance, origin_cooltree=None, weights=None):
        parsimony = self.core.eval_parsimony(parsimony_distance, origin_cooltree=origin_cooltree, weights=weights)
        self.meta.parsimony = parsimony
        return parsimony

    def get_mutatable_nodes(self):
        return self.core.get_mutatable_nodes()

    def evolve_mutate_point(self, choose_oparray2, choose_obs, choose_distributions):
        """
        Mutate a single mutatable point in any Tree.
        """

        # 1. choose a node
        node_list = self.get_mutatable_nodes()
        node = random.choice(node_list)
        arity = node.arity
        if arity > 0:
            new_label = choose_operator(node.xtype, choose_oparray2, arity=arity)  # Function is same type, same arity
        else:
            new_label = choose_term(node.xtype[-2:], choose_obs, choose_distributions)  # 3 -> '2f' -> 5

        node.label = new_label
        self.finalize_structure()
        # All node info should stay the same. xtype, arity

    def get_oldtree(self):
        """
        Helper to create old tree from cooltree
        fitness, complexity, last_evolution, xtypes, modifys, labels
        """
        label_list, xtype_list, modify_list = self.core.labellist_modifylist_from_coolcore()
        tree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list).get_uninstanced_tree()
        tree = tree_set_fitness(tree, self.meta.fitness_train)
        tree = tree_set_parsimony(tree, self.meta.parsimony)
        tree = tree_set_last_evolution(tree, self.meta.last_evolution)
        return tree

    def tree_get_oldmeta(self):
        """
        Get the meta information from a tree
        ! This does not evaluate fitness or parsimony !
        """
        tree_meta = {}
        parsimony = self.meta.parsimony
        fitness_train = self.meta.fitness_train
        expr_raw = self.get_expr_raw()
        expr_sym = expr_sympify(expr_raw=expr_raw)  # sfeh store algo sym?

        tree_meta['parsimony'] = parsimony
        tree_meta['fitness_train'] = fitness_train
        tree_meta['expr_raw'] = expr_raw
        tree_meta['expr_sym'] = expr_sym
        return tree_meta  # sfeh delete_this

    def get_observation_list(self):
        """
        Returns a list with the used observations
        """
        observation_list = self.core.get_observation_list()
        return [x if x[0] != '-' else x[1:] for x in observation_list]


def coolcore_from_oldtree(tree, node_id=root_id):
    """
    utility function
    karoo_tree to pnode version
    """
    label, arity, xtype = tree_node_get_lax_v3(tree, node_id)
    is_fix = False if tree_node_is_modifiable(tree, node_id) else True
    pnode = CoolCore(label=label, arity=arity, xtype=xtype, is_fix=is_fix)

    childs = tree_node_get_childs(tree, node_id)  # [7, 8, 9]
    for child_id in childs:
        pchild = coolcore_from_oldtree(tree, node_id=child_id)
        pnode.child_append(pchild)

    if node_id == root_id:
        pnode.finalize()

    return pnode


def cooltree_from_oldtree(tree, node_id=root_id):
    """
    fitness, complexity, last_evolution, xtypes, modifys, labels
    """
    try:
        coolcore = coolcore_from_oldtree(tree, node_id=node_id)
        parsimony = tree_get_parsimony(tree)
        fitness = tree_get_fitness(tree)
        last_evolution = tree_get_last_evolution(tree)
        cooltree = CoolTree(coolcore)
        cooltree.meta.fitness_train = fitness
        cooltree.meta.parsimony = parsimony
        cooltree.meta.last_evolution = last_evolution
    except Exception as ex:
        print(f'cooltree_from_oldtree failed: {ex}')
        cooltree = None
    return cooltree


def coolcore_from_treenode(tree, node_id, is_fix=True):
    label = tree_node_get_label(tree, node_id)
    pnode = CoolCore(label, is_fix=is_fix)
    return pnode


def cooltree_from_labellist(label_list, obs_krazy=None, modify_list=None):
    xtype_list = xtypes_from_labels(label_list, obs_krazy=obs_krazy)
    tree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list).get_uninstanced_tree()
    cooltree = CoolTree(coolcore_from_oldtree(tree))
    return cooltree


def coolcore_from_expr(expr, obs_krazy):
    label_list = ast_convert_from_expr(expr, build=True)
    xtype_list = xtypes_from_labels(label_list, obs_krazy)
    p_tree = Ptree_karoo(label_list, xtype_list)
    tree = p_tree.get_uninstanced_tree()
    coolcore = coolcore_from_oldtree(tree)
    return coolcore


def cooltree_from_expr(expr, obs_krazy):
    label_list = ast_convert_from_expr(expr, build=True)
    xtype_list = xtypes_from_labels(label_list, obs_krazy)
    p_tree = Ptree_karoo(label_list, xtype_list)
    tree = p_tree.get_uninstanced_tree()
    coolcore = coolcore_from_oldtree(tree)
    cooltree = CoolTree(coolcore)
    return cooltree


def test_insert_subtree():
    label_list = ['sin', '+', 'cos', 'Ifte', 1, 2, 3, 4]
    cooltree = cooltree_from_labellist(label_list)
    print(cooltree)
    print(cooltree.core)
    print(len(cooltree))
    print(cooltree.core.get_expr_raw())
    print(cooltree.core.get_apted_notation())

    # 1. choose node
    many_nodes = cooltree.core.get_mutatable_nodes()
    chosen_node = random.choice(many_nodes)
    print(f"1 choosing a node: {chosen_node}")
    # 2. check node's depth and size
    print('2 depth and len', chosen_node.depth, len(chosen_node))
    # 3. insert a new subtree
    subtree = cooltree_from_labellist(['Mini', 11, 12]).core
    chosen_node_path = chosen_node.nodepath
    print('PATH???', chosen_node_path)
    print(cooltree)
    cooltree.insert_branch(chosen_node_path, subtree)
    print(cooltree)
    # 4. tree must be correct at all times!


def test_sdf():
    label_list = ['Ifte', '<', '0', '2', 'cartVel', '0']
    modify_list = [0, 1, 0, 0, 1, 1]
    tree = cooltree_from_labellist(label_list, modify_list=modify_list)
    cooltree = CoolTree(coolcore_from_oldtree(tree))
    print(cooltree)
    old_labels = cooltree.core.get_labellist()
    print(old_labels)
    # print([str(x) for x in platree2_from_oldversion(labels, modify_list=allowMods)])


def some_quick_test():
    coollist = '[*, cartPos, [**, [+, cartPos, 1.077166], 2.0]]'
    # test_insert_subtree()

    label_list = ['Ifte', '<', '0', '2', 'cartVel', '0']
    cooltree = cooltree_from_labellist(label_list, modify_list=[0, 1, 0, 0, 1, 1])
    label_list = ['Usub', 'asd']
    cooltree = cooltree_from_labellist(label_list)
    print(cooltree)
    cooltree.evolve_reduce()
    print(cooltree)
    print(cooltree.get_observation_list())


class ObservationNode(CoolCore):

    def __init__(self, name):
        super().__init__(name)
        self.name = name


# some_quick_test()
