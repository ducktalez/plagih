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
"""

import ast
import itertools
import random
import re

import numpy as np
import tensorflow as tf

from plagih.fitness_kernel import *
from plagih.sympy_extras import expr_sympify
from plagih.tree_distances.tree_edit_distance import apted_distance

tf.compat.v1.disable_eager_execution()  # sfeh damn what was this line good for?

# lol, lol. https://github.com/tensorflow/tensorflow/issues/27023 these messages are tingeling
# import tensorflow.python.util.deprecation as deprecation  # not possible on python 3.6
# deprecation._PRINT_DEPRECATION_WARNINGS = False

latex_inline = ['+', '-', '*', '**', '==', '!=', '<', '<=', '>', '>=', 'Andb', 'Orb', 'Xor']


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
    state = None
    arity = 0
    coolxtype = (tuple([None]), None)
    tf = None
    sym_expr = 'None'
    pycode = 'None'
    latex = ('None', 'None')

    def __init__(self, label=None, is_fix=False, childs=None):
        self.label = label  # todo
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
        print_label = self.label.label
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

    def new_core(self, new_core: 'Node'):
        """

        """
        self.label = new_core.label
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
            if coolxtype_out is None or coolxtype_out == self.label.coolxtype[1] and (allow_root or not self.is_root()):
                # crossover requires excluding types that are not matching, and excludes the root node
                coolnode_list.append(self)

        coolnode_list.extend(list(itertools.chain(*[cc.get_mutatable_nodes(coolxtype_out=coolxtype_out, allow_root=allow_root) for cc in self.childs])))
        return coolnode_list

    def evolve_mutate_branch_depth(self, depths, tb):
        """
        todo other version
        currently only one branch
        """
        # sfeh: making the root node (todo?)
        if depths == 1:
            self.label = builder.choose_term(self.label.coolxtype[1])  # sfeh update node plabel
        else:
            depths -= 1
            self.childs = [Node(builder.choose_any(xt, p_op=1)) for xt in self.label.coolxtype[0]]

    def evolve_mutate_filter(self, tb):
        """
        filtger the nodes in a single tree
        """
        if self.label.arity > 0:
            for cc in self.childs:
                cc.evolve_mutate_filter(tb)
        else:
            self.label.mutate_filter()

    def evolve_mutate_point_ROOTNODE(self, tb):
        """
        todo rootnode
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
        if self.arity > 0:
            self.label = tb.choose_op(self.coolxtype)  # Function is same type, same arity
        else:
            print('hhhhh', self.label)
            self.label = tb.choose_term(self.label.coolxtype[1])  # 3 -> '2f' -> 5

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
        if isinstance(self.label, Usub):  # tilde '~'
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
        if self.label.arity != len(self.childs):
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
        if self.arity > 0:
            child_expr_list = [cc.get_expr(reducible=reducible, obs_names=obs_names) for cc in self.childs]
            # if reducible:
            #     # my_expr = op[self.label]['sym_reduce'] or my_expr
            #     # symloc = sympy_symbol_defaults(obs_names)  # todo solve the problem... new version of sympy?
            #     xxx = plagih_sympify(my_expr.format(*child_expr_list), eval_locals=symloc)  # sfeh the xxx variable
            #     return xxx
            return self.sym_expr.format(*child_expr_list)  # f'cos({})'([33]) does not work. *list makes the list args :D
        else:
            return self.sym_expr

    def get_pycode(self):
        """

        """
        if self.arity == 0:
            return f'{self.label}'
        else:
            results = [cc.get_pycode() for cc in self.childs]
            return self.pycode.format(*results)  # abs -> lambda a: 'abs({})'.formadt(a) (result1)

    def get_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
        e.g. {+{Ifte{True}{1}{2}}{3}}
        """
        # sfEh check if this still works as one-liner
        return f"{{{self.label}{''.join([cc.get_apted_notation() for cc in self.childs])}}}"

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
                f'Reduced core is even more complex than before  ({len(new_core)}, {len(self)}). expr_raw: {expr_raw}')  # \nold_core:{self}\nnew_core: {new_core} May happen with sympification and Usub.
            # example: Tree sympification did not work: Reduced core is even more complex than before. expr_raw: sign(Mini(((Velocity_2 * -0.790706) - sqrt(Gain_0)), (-0.569271 - Velocity_9)))
            # old_core:[sign, [Mini, [-, [*, Velocity_2, -0.790706], [sqrt, Gain_0]], [-, -0.569271, Velocity_9]]]
            # new_core: [sign, [Mini, [-, [Usub, [sqrt, Gain_0]], [*, 0.790706, Velocity_2]], [-, -Velocity_9, 0.569271]]]
        return


class LabelNode(Node):  # todo
    """
    Kind of abstract class; Dummy-node that holds a label
    """
    pass


class Operator(LabelNode):
    pass


class Terminal(LabelNode):

    def mutate_filter(self):
        # todo? ...only for terminal nodes
        pass


class Constant(Terminal):
    pass


class Observation(Terminal):
    """

    todo discuss: labels should not have a sign (-pos); just pos
    # self.name = label if label[0] != '-' else label[1:]  # sfeh delete?
    """
    tf_type = tf.float32  # todo yeah...

    def __init__(self, label, xtype=float):
        # todo coolxtype_out=float
        self.label = label
        self.fam, self.obs_index, _ = observation_get_family_and_time(label, none_return=None)  # remove this self.prelabel
        self.xtype = xtype
        self.sym_str = label  # sfeh delete?
        self.index_minmax = None

        latex = f'\\text{{{self.fam}}}'  # remove this {self.prelabel}
        self.latex = (latex, latex)

    def mutate_filter(self):
        """
        was filter_new_index
         # as default, return own index
        """
        # if self.index_minmax is None:
        return


class ObservationIndex(Observation):
    """

    """
    def __init__(self, label, xtype=float, obs_indizes=None):
        super().__init__(label, xtype)
        self.obs_indizes = obs_indizes
        latex = f'\\text{{{self.fam}}}_{{{self.obs_index}}}'   # remove this {self.prelabel}
        self.latex = (latex, latex)  # remove this {self.prelabel}

    def mutate_filter(self):
        new_index = int(max(min(round(random.gauss(self.obs_index, 1)), self.index_minmax[1]), 0))
        self.obs_index = new_index
        self.name = f'{self.fam}_{new_index}'


def observation_get_family_and_time(name, re_pattern='_\\d+$', none_return=None):
    """
    When an observation is known, return the family, the time and the SIGN!!
    todo put this function somewhere where it can actually help
    """

    core_label = re.split(re_pattern, name)[0]
    if core_label[0] == '-':
        core_label = core_label[1::]
        prelabel = '-'
    else:
        prelabel = ''
    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception:
        temp_diff = none_return
    return core_label, temp_diff, prelabel


class FloatConstant(Constant):
    """
    discuss: how to deal with sign of observations?
    """
    arity = 0
    otype = float
    coolxtype = (tuple([]), float)

    def __init__(self, label):
        self.label = label
        self.latex = (f'{label:.3f}', f'{label:.3f}')
        self.sym_str = label
        self.pycode = label

    def mutate_filter(self, filter_type='gaussian_filter', precision=6):  # todo
        if filter_type == 'gaussian_filter':
            if random.choice(['v1', 'v2']) == 'v1' or self.label == 0:
                self.label += np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.label, 0.1)  # sfeh better adjustments?
                self.label = round(constant, precision)  # sfeh be careful, might create zero sometimes


class BoolConstant(Constant):
    """

    """
    arity = 0
    coolxtype = (tuple([]), bool)
    tf_type = tf.bool

    def __init__(self, label):
        self.latex = (f'{label}', f'{label}')
        self.sym_str = label
        self.pycode = label


class EvalAction:
    """
    todo plabel/labelNode? daut it.
    - minmax for histograms
    - minmax for regression-bounded
    """
    tf_type = tf.float32  # sfeh especiall when the type is integer
    xtype = float  # sfeh todo
    coolxtype = [None, float]

    def __init__(self, name):
        self.plabel = name
        self.name = name  # delete this


class Add(Operator):
    label = '+'
    arity = 2
    tf = tf.add
    latex = ('+', '{}+{}')
    sym_str = '({} + {})'
    pycode = '({}+{})'
    coolxtype = (tuple([float, float]), float)


class Subtract(Operator):
    """

    """
    label = '-'
    arity = 2
    tf = tf.subtract
    latex = ('-', '{}-{}')
    sym_str = '({} - {})'
    pycode = '({}-{})'
    coolxtype = (tuple([float, float]), float)


class Usub(Operator):
    label = 'Usub'
    arity = 1
    tf = tf.negative
    latex = ('-', '-{}')
    sym_str = '(-{})'
    pycode = '(-{})'
    coolxtype = (tuple([float]), float)


class Multiply(Operator):
    label = '*'
    arity = 2
    tf = tf.multiply
    latex = ('\\cdot ', '{}\\cdot {}')
    sym_str = '({} * {})'
    pycode = '({}*{})'
    coolxtype = (tuple([float, float]), float)


class Divide_no_nan(Operator):
    label = '/'
    arity = 2
    tf = tf.math.divide_no_nan
    latex = ('\\div ', '\\frac{}{}')
    sym_str = '({} / {})'
    pycode = '(lambda x, y: x/y if y!=0 else 0)(({}),({}))'
    coolxtype = (tuple([float, float]), float)


class Power(Operator):
    label = '**'
    arity = 2
    tf = tf.pow
    latex = ('{{x}}^{{y}}', '{}^{}')
    sym_str = '({}**Round({}))'
    pycode = '({}**round({}))'
    coolxtype = (tuple([float, float]), float)


class Abs(Operator):
    label = 'abs'
    arity = 1
    tf = tf.abs
    latex = ('abs', '|{}|')
    sym_expr = 'abs({})'
    pycode = 'abs({})'
    coolxtype = (tuple([float]), float)


class Sign(Operator):
    label = 'sign'
    arity = 1
    tf = tf.sign
    latex = ('sign', 'sign({})')
    sym_str = 'sign({})'
    pycode = 'np.sign({})'
    coolxtype = (tuple([float]), float)


class Round(Operator):
    label = 'Round'
    arity = 1
    tf = tf.round
    latex = ('round', 'round({})')
    sym_str = 'Round({})'
    pycode = 'round({})'
    coolxtype = (tuple([float]), float)


class Square(Operator):
    label = 'Square'
    arity = 1
    tf = tf.square
    latex = ('x^2', '{}^2')
    sym_str = 'Square({})'
    pycode = '({})**2'
    coolxtype = (tuple([float]), float)


class Sqrt(Operator):
    label = 'sqrt'
    arity = 1
    tf = tf.sqrt
    latex = ('\\sqrt{x}', '\\sqrt{}')
    sym_str = 'sqrt({})'
    pycode = 'math.sqrt({})'
    coolxtype = (tuple([float]), float)


class Log(Operator):
    label = 'log'
    arity = 1
    tf = tf.math.log
    latex = ('\\log()', '\\log{}')
    sym_str = 'log({})'
    pycode = 'math.log({})'
    coolxtype = (tuple([float]), float)


class Log1p(Operator):
    label = 'log1p'
    arity = 1
    tf = tf.math.log1p
    latex = ('\\log(1+x)', '\\log(1+{})')
    sym_str = 'log1p({})'
    pycode = 'math.log1p({})'
    coolxtype = (tuple([float]), float)


class Cos(Operator):
    label = 'cos'
    arity = 1
    tf = tf.cos
    latex = ('\\cos ', '\\cos({})')
    sym_str = 'cos({})'
    pycode = 'math.cos({})'
    coolxtype = (tuple([float]), float)


class Sin(Operator):
    label = 'sin'
    arity = 1
    tf = tf.sin
    latex = ('\\sin ', '\\sin({})')
    sym_str = 'sin({})'
    pycode = 'math.sin({})'
    coolxtype = (tuple([float]), float)


class Tan(Operator):
    label = 'tan'
    arity = 1
    tf = tf.tan
    latex = ('\\tan ', '\\tan({})')
    sym_str = 'tan({})'
    pycode = 'math.tan({})'
    coolxtype = (tuple([float]), float)


class Acos(Operator):
    label = 'acos'
    arity = 1
    tf = tf.acos
    latex = ('\\acos ', '\\acos({})')
    sym_str = 'acos({})'
    pycode = 'math.acos({})'
    coolxtype = (tuple([float]), float)


class Asin(Operator):
    label = 'asin'
    arity = 1
    tf = tf.asin
    latex = ('\\asin ', '\\asin({})')
    sym_str = 'asin({})'
    pycode = 'math.asin({})'
    coolxtype = (tuple([float]), float)


class Atan(Operator):
    label = 'atan'
    arity = 1
    tf = tf.atan
    latex = ('\\atan ', '\\atan({})')
    sym_str = 'atan({})'
    pycode = 'math.atan({})'
    coolxtype = (tuple([float]), float)


class Tanh(Operator):
    label = 'tanh'
    arity = 1
    tf = tf.tanh
    latex = ('\\tanh ', '\\tanh({})')
    sym_str = 'tanh({})'
    pycode = 'math.tanh({})'
    coolxtype = (tuple([float]), float)


class And(Operator):
    label = 'Andb'
    arity = 2
    tf = tf.logical_and
    latex = ('and', '({}\\wedge{})')
    sym_str = 'Andb({}, {})'
    pycode = '({} and {})'
    coolxtype = (tuple([bool, bool]), bool)


class Or(Operator):
    label = 'Orb'
    arity = 2
    tf = tf.logical_or
    latex = ('or', '({}\\vee{})')
    sym_str = 'Orb({}, {})'
    pycode = '({} or {})'
    coolxtype = (tuple([bool, bool]), bool)


class Xor(Operator):
    label = 'Xor'
    arity = 2
    tf = tf.math.logical_xor
    latex = ('\\oplus', '({}\\oplus{})')
    sym_str = 'Xor({}, {})'
    pycode = '({} ^ {})'
    coolxtype = (tuple([bool, bool]), bool)


class Not(Operator):
    label = 'Notb'
    arity = 1
    tf = tf.logical_not
    latex = ('\\neg', '\\neg{}')
    sym_str = 'Notb({})'
    pycode = 'not({})'
    coolxtype = (tuple([bool]), bool)


class Eq(Operator):
    label = '=='
    arity = 2
    tf = tf.equal
    latex = ('=', '({}={})')
    sym_str = '({} == {})'
    pycode = '({}=={})'
    coolxtype = (tuple([bool, bool]), bool)


class Neq(Operator):
    label = '!='
    arity = 2
    tf = tf.not_equal
    latex = ('\\neq', '({}\\neq{})')
    sym_str = '({} != {})'
    pycode = '({}!={})'
    coolxtype = (tuple([bool, bool]), bool)


class Lt(Operator):
    label = '<'
    arity = 2
    tf = tf.less
    latex = ('<', '{}<{}')
    sym_str = '({} < {})'
    pycode = '({}<{})'
    coolxtype = (tuple([float, float]), bool)


class Le(Operator):
    label = '<='
    arity = 2
    tf = tf.less_equal
    latex = ('\\leq', '{}\\leq{}')
    sym_str = '({} <= {})'
    pycode = '({}<={})'
    coolxtype = (tuple([float, float]), bool)


class Gt(Operator):
    label = '>'
    arity = 2
    tf = tf.greater
    latex = ('>', '{}>{}')
    sym_str = '({} > {})'
    pycode = '({}>{})'
    coolxtype = (tuple([float, float]), bool)


class Ge(Operator):
    label = '>='
    arity = 2
    tf = tf.greater_equal
    latex = ('\\geq', '{}\\geq {}')  # sfeh check inserted space
    sym_str = '({} >= {})'
    pycode = '({}>={})'
    coolxtype = (tuple([float, float]), bool)


class Ifte(Operator):
    label = 'Ifte'
    arity = 3
    tf = tf.where
    latex = ('\\text{if-then-else}', '\\text{{ if }} ({}) \\text{{ then }} ({}) \\text{{ else }} ({})')  # 'if({} then {} else {})'
    sym_str = 'Ifte({}, {}, {})'
    pycode = '({} if {} else {})'
    coolxtype = (tuple([bool, float, float]), float)


class Min(Operator):
    label = 'Mini'
    arity = 2
    tf = tf.minimum
    latex = ('\\min', '\\min({}, {})')
    sym_str = 'Mini({}, {})'
    pycode = 'min({}, {})'
    coolxtype = (tuple([float, float]), float)


class Max(Operator):
    label = 'Maxi'
    arity = 2
    tf = tf.maximum
    latex = ('\\max', '\\max({}, {})')
    sym_str = 'Maxi({}, {})'
    pycode = 'max({}, {})'
    coolxtype = (tuple([float, float]), float)


op = {  # 'f2f': Classical mathematical operators, evaluate from float to float
    '+': Add,
    ast.Add: Add,
    '-': Subtract,
    ast.Sub: Subtract,
    'Usub': Usub,
    ast.USub: Usub,
    '*': Multiply,
    ast.Mult: Multiply,
    # Division: SAFE division by zero! -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented sfeh: is it okay to display this as '/'?
    '/': Divide_no_nan,
    ast.Div: Divide_no_nan,
    '**': Power,
    ast.Pow: Power,
    'Abs': Abs,
    'sign': Sign,
    'Round': Round,
    'Square': Square,
    'sqrt': Sqrt,
    'log': Log,  # sfeh log/ln?
    'log1p': Log1p,
    'cos': Cos,
    'sin': Sin,
    'tan': Tan,
    'acos': Acos,
    'asin': Asin,
    'atan': Atan,
    'tanh': Tanh,

    # bool->bool
    # DON'T USE tf.bitwise.bitwise_and
    # sympify('Or')->'|', sympify('And')->'&', sympify('Not')->'~'
    'Andb': And,
    ast.And: And,
    'Orb': Or,
    ast.Or: Or,
    'Xor': Xor,
    # ast.BitXor: Xor,
    'Notb': Not,
    ast.Not: Not,

    # float->bool
    '==': Eq,
    ast.Eq: Eq,
    '!=': Neq,
    ast.NotEq: Neq,
    '<': Lt,  # a < b
    ast.Lt: Lt,
    '<=': Le,
    ast.LtE: Le,
    '>': Gt,  # a > b
    ast.Gt: Gt,
    '>=': Ge,  # a >= 1
    ast.GtE: Ge,

    'Ifte': Ifte,  # sfeh essential for evaluation
    'Mini': Min,  # with forced arity-2
    'Maxi': Max,  # with forced arity-2
}

# sfeh https://docs.sympy.org/latest/tutorial/manipulation.html

# import tensorflow as tf; import ast; import textwrap
# print(', '.join(['[\'{}\', {:.2f}]'.format(v['label'], 1/v['coolxtype': ([], []), 'c-weight']) for k, v in op_what.items()]))  # retreive a list with all non-ast ops:


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


def randomly_split_range(range_max, num_splits):
    """
    todo reuse this
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
    todo reuse this
    """
    terms = random.randint(0, size - 1)  # -1 because at least one 'func'
    cons_buf = ['term'] * terms + ['func'] * (size - terms)
    np.random.shuffle(cons_buf)
    return cons_buf


def choose_build_size(size_mode, mean_min_max_var, tree=None, nodepath=None, force=None):
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
        if tree and nodepath:
            pass
        else:
            raise Exception('No tree or node is given for computing the relative size')

        if size_mode == 'tree_depth':
            tree_size = tree.core.childs_depth_max
            print('tree_size = tree.core.childs_depth_max:', tree_size)
            if tree_size is None and 'delete_this':
                raise Exception('ASDASDASD')
            node_size = len(nodepath)
        elif size_mode == 'tree_nodes':
            tree_size = len(tree)
            print('len(tree)?:', len(tree))
            node_size = len(tree.get_nodepath(nodepath))
            print('tree.get_nodepath(nodepath)?:', tree.get_nodepath(nodepath))
        else:
            raise Exception('Sizemode not known?')

        relative_size = tree_size - node_size
        print('asdasd', relative_size)

    build_size = int(random.normalvariate(mean, size_variance))
    if size_max is not None:
        build_size = min(size_max - relative_size, build_size)
    build_size = max(size_min, build_size)

    return int(build_size)


def tree_check_rebuild(tree):
    """
    sfeh
    Check if a valid tree can be rebuilt from its expression
    sfeh: the expression must currently not be equal.
    sfeh: The expression can include separate '~' (Usub) nodes, which makes expressions not completely equal
    A method to check if a tree is type consistant:
    - do the values in c1, c2, c3 link to correct types?
    - do the values in c1, c2, c3 link to its parent?
    """
    return True


if __name__ == '__main__':

    for k, v in op.items():
        print(f'{k}\t: {v}')

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

    ops = ChooseOperators()
    inputs = ChooseObservation(['a', 'b'])
    consts = ChooseConstants()

    builder = TreeBuilder(ops, inputs, consts, float)
    tree = builder.invent_core_depth(float, 3)  # todo
    print(tree)
