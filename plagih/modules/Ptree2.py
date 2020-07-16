from collections import deque
from plagih.modules.plagih_types import *
from plagih.modules.plagih_tree import *


class Plabel:
    def __init__(self, label):
        self.label = label


class CoolCore:
    """
    x = label_list = ['Ifte', '<', 0, 2, 'cartVel', 0]
    label_list = ['+', 1, 'b']
    x = Pnode('+', childs=[Pnode('1'), Pnode('2')])
    """

    def __init__(self, label=None, is_fix=False, complete=False, arity=None, xtype=None, childs=None, depth=0, nodepath=None):

        if xtype is None:
            xtype = xtype_get_from_label(label)
        label_type = float if xtype[-2:] == '2f' else bool  # todo
        arity = label_get_arity(label) if arity is None else arity

        self.label = label
        self.expr_raw = None

        self.is_fix = is_fix

        self.complete = complete  # if the node is correct/done/okay
        self.childs = childs if childs is not None else []
        self.depth = depth
        self.childs_depth_max = None
        self.nodepath = nodepath if nodepath else []  # go to node x (sfeh: this was a deque?)
        self.label_type = label_type
        self.arity = arity
        self.input_types = [bool, float, float] if label == 'Ifte' else [label_type] * arity
        self.xtype_old = xtype  # obsolete?

    def finalize(self):
        self.finalize_set_depth()
        self.finalize_set_nodepath([0])
        print('ddddddddddddddddddd')
        print(self)
        print(self.nodepath)
        # fill meta...
        # eval fitness
        # eval parsimony
        # check tree
        self.complete = True

    def finalize_set_nodepath(self, nodepath):
        """
        aka nodepath?
        [0,2,1,0,0]
        """
        self.nodepath = nodepath
        for ii, child in enumerate(self.childs):
            nodepath_child = nodepath + [ii]
            child.finalize_set_nodepath(nodepath_child)

    def get_nodes_at_depth(self, depth):
        if self.depth < depth:
            return sum([child.get_nodes_at_depth(depth) for child in self.childs], [])
        elif self.depth == depth:
            return [self]
        else:
            print_e('shhosfisufnbikudsrnfg')
            return

    def get_mutatable_nodes(self):
        """

        """
        add_me = [] if self.is_fix else [self]
        return add_me + sum([cc.get_mutatable_nodes() for cc in self.childs], [])

    def check_all(self):
        if self.arity != len(self.childs):
            raise

        return True

    def get_expr_raw(self):
        # if self.expr_raw is None:  # sfeh?
        if self.arity == 0:
            return self.label
        else:
            my_expr = op[self.label]['sym_str']
            f_my_expr = lambda *args: my_expr.format(*args)
            child_expr_list = [child.get_expr_raw() for child in self.childs]
            return f_my_expr(*child_expr_list)  # f'cos({})'([33]) does not work. *list makes the list args :D

    def get_nodes_list_modifiable(self):
        """
        not required with new trees?
        """
        pass
        # if self.is_fix:
        #     coords = self.co
        #     return sum([child.get_nodes_list_modifiable() for child in self.childs], [])
        # elif self.depth == depth:
        #     return [self]
        # else:
        #     print_e('fdgdfg')
        #     return

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
        len_nodepath = len(nodepath)
        print('asd nodepath', nodepath)
        print('asd selfchil', self.childs[nodepath[0]])
        if len_nodepath == 1:  # [1] -> set child 1
            self.childs[nodepath[0]] = coolbranch
        elif len_nodepath > 1:
            self.childs[nodepath[0]].insert_branch(nodepath[1:], coolbranch)

        else:
            raise Exception(f'Branch could not be inserted, node was not found')

    def get_pycode(self):

        if self.arity == 0:
            if label_is_observation(self.label):
                ib_sfeh_dict = {'p': 'SetPoint',
                                'v': 'Velocity',
                                'g': 'Gain',
                                'h': 'Shift',
                                'f': 'Fatigue',
                                'c': 'Consumption'}
                ib_sfeh_rev = {v: k for k, v in ib_sfeh_dict.items()}
                obs_family, obs_time = observation_get_family_and_time(self.label, none_return=None)
                if obs_time is None:
                    pass
                else:
                    geth_name = ib_sfeh_rev[obs_family]
                    return f"self.get_h('{geth_name}', {obs_time})"

            return f'{self.label}'
        else:
            results = []
            for child in self.childs:
                results.append(child.get_pycode)  # = tree_node_get_label(tree, int(child))
            return op[self.label]['pycode'](*results)  # abs -> lambda a: 'abs({})'.formadt(a) (result1)

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

    def workaround_normalize_exponentiation(self, normalize_me=False):
        """
        a**b requires b to be discrete.
        """
        # 1. ** should have an int as second number

        if normalize_me:
            try:
                self.label = float(round(float(self.label)))
            except ValueError:
                pass  # todo: implicit round operator in TF-grapf and evaluation?

        if self.label == '**':
            normalize_me = True
        for ii, cc in enumerate(self.childs):
            if ii == 1 and normalize_me:
                cc.workaround_normalize_exponentiation(normalize_me=True)
            else:
                cc.workaround_normalize_exponentiation()
        return

    def set_fix_nodes(self, origin_coolcore: 'CoolCore'):
        # todo not necessary?
        if origin_coolcore.is_fix:
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.set_fix_nodes(origin_coolcore.childs[ii])


class PlaTree2:

    class PtreeMeta:
        def __init__(self, fitness_train=None):
            self.hash = None
            self.fitness_train = fitness_train
            self.complexity = None
            self.node_len = None
            self.depth = None
            self.complete = None
            self.last_evolution = None

        def __str__(self):
            return f"hash: {self.hash}, fitness: {self.fitness_train}, complexity: {self.complexity}"

    def __init__(self, root_node: CoolCore, fitness=None):
        self.core = root_node
        self.meta = self.PtreeMeta(fitness)
        self.history = deque([], maxlen=10)  # sfeh arbitrary value of 10 historic metainfo of this tree
        self.finalize()

    def __hash__(self):
        # todo
        return hash(self)

    def finish_nodes(self, last_evolution, origin_tree: CoolCore = None):
        # todo
        if origin_tree is not None:
            self.core.set_fix_nodes(origin_coolcore=origin_tree)
        self.core.workaround_normalize_exponentiation()
        self.meta.last_evolution = last_evolution  # sfeh: should this be done during the evolve process?

    def __len__(self):
        """
        The amount of nodes
        """
        return len(self.core)

    def check_all(self):
        if not self.core.check_all():
            raise
        return True

    def insert_branch(self, nodepath, coolbranch):
        self.core.complete = False
        self.core.insert_branch(nodepath, coolbranch)
        self.core.finalize()

    def finalize(self):
        self.core.finalize_set_depth(depth=self.core.depth)
        self.complete = True

    def get_pycode(self):
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
        
    def meta_set(self, meta):
        self.history.append(self.meta)

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

        """
        return f"Ptree2: {self.core}. Tree meta: {self.meta}"


def pnode_from_oldtree(tree, node_id=root_id):
    """
    utility function
    karoo_tree to pnode version
    """
    label = tree_node_get_label(tree, node_id)
    pnode = CoolCore(label=label)

    childs = tree_node_get_childs(tree, node_id)  # [7, 8, 9]
    for child_id in childs:
        pchild = pnode_from_oldtree(tree, node_id=child_id)
        pnode.child_append(pchild)

    if node_id == root_id:
        pnode.finalize()

    return pnode


def pnode2_from_treenode(tree, node_id, is_fix=True):
    label = tree_node_get_label(tree, node_id)
    pnode = CoolCore(label, is_fix=is_fix)
    return pnode


def tree_from_ptree2(platree2: PlaTree2):
    label_list = []
    max_depth = platree2.core.childs_depth_max
    print('max depth', max_depth)
    for depth in range(0, max_depth + 1):
        labels_at_depth = [x.label for x in platree2.core.get_nodes_at_depth(depth)]
        label_list.extend(labels_at_depth)
    return label_list


def cooltree_from_labellist(label_list, modify_list=None):
    xtype_list = xtypes_from_labels(label_list)
    tree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list).get_uninstanced_tree()
    cooltree = PlaTree2(pnode_from_oldtree(tree))
    return cooltree


def cooltree_from_expr(expr, env_vars):

    label_list = ast_convert_from_expr(expr, build=True)
    xtype_list = xtypes_from_labels(label_list, env_vars)
    p_tree = Ptree_karoo(label_list, xtype_list)
    tree = p_tree.get_uninstanced_tree()
    coolcore = pnode_from_oldtree(tree)
    return coolcore


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
    cooltree.check_all()


def test_sdf():
    label_list = ['Ifte', '<', '0', '2', 'cartVel', '0']
    modify_list = [0, 1, 0, 0, 1, 1]
    tree = cooltree_from_labellist(label_list, modify_list=modify_list)
    cooltree = PlaTree2(pnode_from_oldtree(tree))
    print(cooltree)
    old_labels = tree_from_ptree2(cooltree)
    print(old_labels)
    # print([str(x) for x in platree2_from_oldversion(labels, modify_list=allowMods)])




test_insert_subtree()
