from collections import deque
from plagih.modules.plagih_types import *
from plagih.modules.plagih_tree import *
import gc


class Plabel:
    def __init__(self, label):
        self.label = label


class Pnode:
    """
    x = label_list = ['Ifte', '<', 0, 2, 'cartVel', 0]
    label_list = ['+', 1, 'b']
    x = Pnode('+', childs=[Pnode('1'), Pnode('2')])
    """

    def __init__(self, label=None, is_fix=False, complete=False, arity=None, xtype=None, childs=None, depth=0, nodepath=0):

        if xtype is None:
            xtype = xtype_get_from_label(label)
        label_type = float if xtype[-2:] == '2f' else bool  # todo
        arity = label_get_arity(label) if arity is None else arity

        self.label = label

        self.is_fix = is_fix

        self.complete = complete  # if the node is correct/done/okay
        self.childs = childs if childs is not None else []
        self.depth = depth
        self.childs_depth_max = None
        self.nodepath = deque([nodepath])  # go to node x
        self.label_type = label_type
        self.arity = arity
        self.input_types = [bool, float, float] if label == 'Ifte' else [label_type] * arity
        self.xtype_old = xtype  # obsolete?

    def finalize(self):
        self.set_depth()

    def get_nodes_at_depth(self, depth):
        if self.depth < depth:
            return sum([child.get_nodes_at_depth(depth) for child in self.childs], [])
        elif self.depth == depth:
            return [self]
        else:
            print_e('shhosfisufnbikudsrnfg')
            return

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

    def __str__(self):
        """
        """
        label_info = self.label
        meta_str = []
        if self.is_fix:
            meta_str.append('is_fix: True')
        if meta_str:
            label_info += '{{{}}}'.format(', '.join(meta_str))

        if not self.childs:
            return '{}'.format(label_info)
        else:
            childstr = ', '.join([str(x) for x in self.childs])
            return '[{}, {}]'.format(label_info, childstr)

    def __len__(self):
        """
        """
        return 1 + sum([len(cc) for cc in self.childs])

    def child_set(self, child, pos):
        self.childs[pos] = child

    def child_append(self, child):
        self.childs.append(child)

    def set_depth(self, depth=0, recursive=True):
        """
        depth=0 is the root node
        """
        self.depth = depth
        max_depth = depth
        if recursive:
            for cc in self.childs:
                cc_depth = cc.set_depth(depth=depth + 1)
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


class PlaTree2:

    class PtreeMeta:
        def __init__(self, fitness=None):
            self.hash = None
            self.fitness = fitness
            self.complexity = None
            self.node_len = None
            self.depth = None
            self.complete = None

        # def __str__(self):
        #     return 'hash: {}, fitness: {}, complexity: {}'.format(self.hash, self.fitness, self.complexity)

    def __init__(self, root_node: Pnode, fitness=None):
        self.root = root_node

        self.meta = self.PtreeMeta(fitness)
        self.history = deque([], maxlen=10)  # sfeh arbitrary value of 10 historic metainfo of this tree
        self.finalize()

    def finalize(self):
        self.root.set_depth(depth=self.root.depth)
        self.complete = True

    def reset_meta(self, update_history=None):
        if update_history:
            self.history.append(self.meta)
        self.meta = self.PtreeMeta()
        self.complete = False

    def childs_insert_list_width(self, node_list):
        iter_nodes = iter(node_list)
        while True:
            try:
                node = next(iter_nodes)
            except StopIteration:
                break

            self.root.node_insert_width(node)
    
    def __str__(self):
        return 'Ptree2: {}. Tree meta: {}'.format(self.root, self.meta)


def pnode_from_oldtree(tree, node_id=root_id):
    label = tree_node_get_label(tree, node_id)
    pnode = Pnode(label=label)

    childs = tree_node_get_childs(tree, node_id)  # [7, 8, 9]
    for child_id in childs:
        pchild = pnode_from_oldtree(tree, node_id=child_id)
        pnode.child_append(pchild)

    return pnode


def pnode2_from_treenode(tree, node_id, is_fix=True):
    label = tree_node_get_label(tree, node_id)
    pnode = Pnode(label, is_fix=is_fix)
    return pnode


def tree_from_ptree2(platree2: PlaTree2):
    label_list = []
    max_depth = platree2.root.childs_depth_max
    print('max depth', max_depth)
    for depth in range(0, max_depth + 1):
        labels_at_depth = [x.label for x in platree2.root.get_nodes_at_depth(depth)]
        label_list.extend(labels_at_depth)
    return label_list

def test_sdf():
    label_list = ['Ifte', '<', '0', '2', 'cartVel', '0']
    modify_list = [0, 1, 0, 0, 1, 1]
    xtype_list = xtypes_from_labels(label_list)
    plagihtree = Ptree_karoo(label_list, xtype_list, modify_list=modify_list)
    tree = plagihtree.get_uninstanced_tree()
    ptree2 = PlaTree2(pnode_from_oldtree(tree))
    print(ptree2)
    old_labels = tree_from_ptree2(ptree2)
    print(old_labels)
    # print([str(x) for x in platree2_from_oldversion(labels, modify_list=allowMods)])

test_sdf()