from apted import APTED, Config
from apted.helpers import Tree as aptree


class CustomConfig(Config):
    def rename(self, node1, node2):
        """Compares attribute .c1 of trees"""
        return 1 if node1.value != node2.value else 0

    def children(self, node):
        """Get left and right children of binary tree"""
        return [x for x in (node.left, node.right) if x]


def apted_distance(expr1, expr2):
    """
    Computes the apted distance
    sfeh: can also compute the actual edit steps
    """

    tree1 = aptree.from_text(expr1)
    tree2 = aptree.from_text(expr2)

    apted = APTED(tree1, tree2)
    ted = apted.compute_edit_distance()
    mapping = apted.compute_edit_mapping()

    # print(ted, '\t', mapping)

    return ted, mapping


# Ifte,<,0,2,observation1,0
# pre1 = '(Ifte(observation1<0),(0),(2)'
# raw1 = '{Ifte{<{0}{2}}{observation1}{0}}'
# raw2 = '{Ifte{<{0}{2}}{observation1}{0}}'
# tree_source = aptree.from_text('{a{b}{c}}')
# tree_orig = aptree.from_text('{a{b{d}}}')
# apted_disstance(tree_source, tree_orig)