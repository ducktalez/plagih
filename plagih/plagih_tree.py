"""
plagih_tree contain a new implementation of trees that we use in genetic programming to display a program.
The old karoo "tree" is replaced with, for now, "treer" in the code.
not all functions can use tree for now and some tree-functions require the old "tree"
tree splits the karoo tree into the
- meta-info (fitness, parsimony, tree-id, ...) and the
The core of the tree, which "is" the tree, is stored recursively
Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)

sfeh: write test that checks all operators for sympificytion (...+branch-combinations, and more?)
sfeh: use function-types (-> 'kommuttative'?)

    sfeh
    Check if a valid tree can be rebuilt from its expression
    sfeh: the expression must currently not be equal.
    sfeh: The expression can include separate '~' (Usub) nodes, which makes expressions not completely equal
    A method to check if a tree is type consistant:
    - do the values in c1, c2, c3 link to correct types?
    - do the values in c1, c2, c3 link to its parent?
"""

from plagih.fitness_kernel import *
from plagih.node_labels import *
from plagih.sympy_extras import expr_sympify
from plagih.tree_distances.tree_edit_distance import apted_distance

from dataclasses import dataclass
import itertools

# lol, lol. https://github.com/tensorflow/tensorflow/issues/27023 these messages are tingeling

latex_inline = ['+', '-', '*', '**', '==', '!=', '<', '<=', '>', '>=', 'Andb', 'Orb', 'Xor']


@dataclass
class Node:
    """
    The core is the structure of a plagih gp-tree.
    It recursively holds the nodes of a tree; every tree has a list of potential children.
    Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)

    states?
    [None]: not set
    [0]:    evolution/construction/build mode (potentially missing leaf nodes)
    [1]:    structurally complete/finalized (node_depths correct, node_id set, ...)
    [2]:    including meta-data (fitness, complexity)
    """
    # todo
    # arity: int
    # sym_expr: str
    # pycode: str

    state = None

    def __init__(self, label: 'NodeLabel' = None, is_fix=False, childs=None):
        self.label = label
        self.is_fix = is_fix
        self.childs = childs or []

        self.state = 0

    def __hash__(self):
        """
        This hash function has currently no use.
        The hash-value of a tree was used as key for the LUT.
        However, the python hash-function has a run-specific salt for security reasons,
        making it impossible to load the LUT table between runs, so just use the __str__ as key.
        """
        return hash(str(self))  # sfeh

    def __str__(self):
        """
        Printing the nodes as nested array structure.
        sfeh: make this statement loadable!
        """
        print_label = self.get_label()
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

    def get_label(self):
        return self.label

    def get_expr(self):
        return self.label.expr

    def get_expr_sym(self):
        return self.label.expr_sym

    def get_arity(self):
        return self.label.arity

    def get_xtype(self):
        return self.label.xtype

    def set_label(self, label: NodeLabel):
        """
        all other values are automatically set by assigning the respected node
        """
        self.label = label

    def eval_parsimony(self, parsimony_distance, origin_tree=None, weights=None):
        """
        parsimony_distance: compute the chosen distance by the user.
        #     'tree_node_count': tree_get_size,
        #     'tree_depth': tree_get_depth,
        #     'tree_edit_distance': tree_parsimony_ted,

        # self.meta.parsimony = parsimony  # todo okay where meta? at root node...
        """
        if parsimony_distance == 'tree_node_count':  # number of nodes
            return len(self)  # returns the number of nodes  # sfeh weights
        elif parsimony_distance == 'tree_edit_distance':  # tree_edit_distance, tree-edit-distance
            apted1 = self.get_apted_notation()
            apted2 = origin_tree.get_apted_notation()
            distance, mapping = apted_distance(apted1, apted2)  # sfeh the mapping could be handy somewhere
            if weights is None:
                return distance
            else:
                raise
        else:
            raise Exception(f'Complexity measurement not available: {parsimony_distance}')

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

    def new_core(self, new_node: 'Node'):
        """
        todo
        """
        self.set_label(new_node.get_label())
        self.childs = new_node.childs or []  # maybe must be updated recursively
        # self.is_fix = new_node.is_fix  # debatable
        # self.complete = new_node.complete  # if the node is correct/done/okay
        # depth needs to be fixed?

    def get_mutatable_nodes(self, xtype_out=None, allow_root=True):
        """
        return all nodes that are mutatable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!
        """
        coolnode_list = []
        if not self.is_fix:
            if xtype_out is None or xtype_out == self.get_xtype()[1] and (allow_root or not self.is_root()):
                # crossover requires excluding types that are not matching, and excludes the root node
                coolnode_list.append(self)

        coolnode_list.extend(list(itertools.chain(*[cc.get_mutatable_nodes(xtype_out=xtype_out, allow_root=allow_root) for cc in self.childs])))
        return coolnode_list

    def evolve_mutate_branch_depth(self, depths, tb):
        """
        todo other version
        currently only one branch
        """
        # sfeh: making the root node (todo?)
        if depths == 1:
            self.set_label(tb.choose_term(self.get_xtype()[1]))  # sfeh update node plabel
        else:
            depths -= 1
            self.childs = [Node(tb.choose_any(xt, p_op=1)) for xt in self.get_xtype()[0]]

    def evolve_mutate_filter(self, tb):
        """
        filtger the nodes in a single tree
        """
        if self.get_arity() > 0:
            for cc in self.childs:
                cc.evolve_mutate_filter(tb)
        else:
            raise  # todo
            # self.mutate_filter()

    def evolve_mutate_point_ROOTNODE(self, tb):
        """
        ==>todo rootnode
        Mutate a single mutatable point in any Tree.
        """
        # 1. choose a node
        node_list = self.get_mutatable_nodes()
        node = random.choice(node_list)
        node.evolve_mutate_point(tb)

    def evolve_mutate_point(self, tb):
        """
        Mutate a single mutatable point in any Tree.
        """
        if self.get_arity() > 0:
            self.set_label(tb.choose_op(self.get_xtype()))  # Function is same type, same arity
        else:
            print('hhhhh', self.get_label())
            self.set_label(tb.choose_term(self.get_xtype()[1]))  # 3 -> '2f' -> 5

    def evolve_start(self):
        """
        todo
        Before evolving, delete all tree information
        so the tree is not holding "wrong" information about fitness, etc.
        - (append last meta value to history)
        - delete meta info
        - set status to not complete
        """
        pass

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
            print_e(f'FFS Trees just become larger? {self.get_expr()}')
        # self.meta.clear()

    def evolve_mutate_filter_random(self, call_params, tb):
        """
        Mutates a number of float terminal of a tree
        todo ROOT
        """
        mode = call_params['mode']  # point/branch/all
        yes_observations = call_params.get('yes_observations')  # point/branch/all
        mutate_filter = 'gaussian_filter'  # sfeh change?

        node_ids = self.get_mutatable_nodes()
        node_id = random.choice(node_ids)  # sfeh should this be completely random?

        if mode == 'branch':
            node_id.evolve_mutate_filter(tb)
        else:
            pass
            # mode == 'point'
            # sfeh delete this? point can always hapen
            # node_id.evolve_mutate_point(tb)

    def evolve_mutate_branch(self, cool_build_size, tb, size_mode='depth', full_or_grow='full'):
        """

        """
        node_list = self.get_mutatable_nodes()
        node = random.choice(node_list)
        node.evolve_mutate_branch_depth(3, tb)  # todo

    def workaround_remove_tilde(self):
        if isinstance(self.get_label(), Usub):  # tilde '~'
            new_core = self.childs[0]
            self.new_core(new_core)

        for cc in self.childs:
            cc.workaround_remove_tilde()

    def finalize(self):
        """
        (only in root node)
        finalizing the structure
        """
        self.finalize_set_depth()
        self.finalize_set_nodepath([0])
        self.workaround_remove_tilde()

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
        if self.get_arity() != len(self.childs):
            raise
        return True

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

    # def __init__(self, plabel: LabelNode):
    #     super().__init__(plabel)
    #     # self.core = coolcore  # todo sfeh
    #     self.meta = self.PtreeMeta()
    #     self.history = deque([], maxlen=10)  # sfeh arbitrary value of 10 historic metainfo of this tree
    #     self.finalize_structure()

    # def workaround_normalize_exponentiation(self):
    #     self.core.workaround_normalize_exponentiation()
    #     # self.finish_nodes()  #
    #     self.finalize_completely()  #

    def finalize_completely(self):
        """

        """
        self.finalize_set_depth(depth=self.depth or 0)  # sfeh the starting depth? hmm, why set it none anyways. aye. rewrite
        # self.finalize_meta()  # todo
        # sfeh asd does this work?
        # self.meta.expr_raw = self.get_expr()
        # self.meta.expr_sym = 'self.get_expr_sym()'  # todo

    def get_expr(self, reducible=None, obs_names=None):
        """
        accumulate and return the complete expression the tree holds recursively
        """
        if self.get_arity() > 0:
            child_expr_list = [cc.get_expr(reducible=reducible, obs_names=obs_names) for cc in self.childs]
            # if reducible:
            #     # my_expr = op[self.get_label()]['sym_reduce'] or my_expr
            #     # symloc = sympy_symbol_defaults(obs_names)  # todo solve the problem... new version of sympy?
            #     xxx = plagih_sympify(my_expr.format(*child_expr_list), eval_locals=symloc)  # sfeh the xxx variable
            #     return xxx
            return self.get_expr_sym().format(*child_expr_list)  # f'cos({})'([33]) does not work. *list makes the list args :D
        else:
            return self.get_expr_sym()

    def get_pycode(self):
        """

        """
        if self.get_arity() == 0:
            return f'{self.get_label()}'
        else:
            results = [cc.get_pycode() for cc in self.childs]
            return self.pycode.format(*results)  # abs -> lambda a: 'abs({})'.formadt(a) (result1)

    def get_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
        e.g. {+{Ifte{True}{1}{2}}{3}}
        """
        # sfEh check if this still works as one-liner
        return f"{{{self.get_label()}{''.join([cc.get_apted_notation() for cc in self.childs])}}}"

    def reduce_me(self, obs_infos):
        # sfeh asdasdasd reduce me is obviously bullshit crapshit.
        #  sympify works with this combination only very few times
        #  lets have a new idea.
        expr_raw = self.get_expr(reducible=True, obs_names=obs_infos.keys())
        try:
            expr_sym = expr_sympify(expr_raw)
        except:
            raise Exception(f'Sympify failed. {expr_raw}')

        new_core = [1, 2, 3]  # todo coolcore_from_expr(expr_sym, obs_infos)
        if len(new_core) < len(self):
            self.new_core(new_core)
        elif len(new_core) > len(self):
            raise Exception(
                f'Reduced core is even more complex than before  ({len(new_core)}, {len(self)}). expr_raw: {expr_raw}')  # \nold_core:{self}\nnew_node: {new_node} May happen with sympification and Usub.
            # example: Tree sympification did not work: Reduced core is even more complex than before. expr_raw: sign(Mini(((Velocity_2 * -0.790706) - sqrt(Gain_0)), (-0.569271 - Velocity_9)))
            # old_core:[sign, [Mini, [-, [*, Velocity_2, -0.790706], [sqrt, Gain_0]], [-, -0.569271, Velocity_9]]]
            # new_node: [sign, [Mini, [-, [Usub, [sqrt, Gain_0]], [*, 0.790706, Velocity_2]], [-, -Velocity_9, 0.569271]]]
        return


if __name__ == '__main__':
    hugo = ['Ifte',
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

    trexpr1 = '(Ifte, (Orb, (cartPos < -1), (Andb, (cartPos < 0.1), (cartVel < -0.05))), 2, (Ifte, (Andb, (Andb, (cartPos > -0.45), (cartPos < -0.05)), (cartVel < -0.5)), 0, (Ifte, (cartVel < 0), 0, 2)))'
    trexpr2 = '(Ifte, (cartVel < 0), 0, 2)'
    # trexpr = plagih_sympify(trexpr)

    # ops = ChooseOperators()
    # inputs = ChooseObservation(['a', 'b'])
    # consts = ChooseConstants()
    #
    # builder = TreeBuilder(ops, inputs, consts, float)
    # tree = builder.invent_core_depth(float, 2)  # todo

    x = Node(label=Max())
    print(x)
