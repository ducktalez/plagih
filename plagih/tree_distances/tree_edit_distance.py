from apted import APTED, Config
from apted.helpers import Tree as aptree
from plagih.operators import *


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

    return ted, mapping


def print_apted_tree(tree):
    tree_vis_dic = {}

    def print_apted_tree_helper(subtree, depth=0):
        if tree_vis_dic.get(depth) is None:
            tree_vis_dic[depth] = []
        tree_vis_dic[depth].append(subtree.name)

        depth += 1

        for child in subtree.children:
            print_apted_tree_helper(child, depth)
        return subtree.name

    print_apted_tree_helper(tree)

    for k, v in tree_vis_dic.items():
        print('->', k, v)


def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def is_bool(x):
    try:
        float(x)
        return False
    except ValueError:
        try:
            bool(x)
            return True
        except ValueError:
            return False


def weight_ted_mapping(mapping):
    """
    todo: make clear how this distance is meant to be used...
    """
    weighted_distance = 0
    for map_i in mapping:
        a, b = map_i[0], map_i[1]
        if a is None:
            b_name = b.name
            if b_name not in op:  # weight can be 0, so check for None
                weighted_distance += 1  # all inserted constants
                print('Inserted', b_name, 'weight 1 (non-op dummy)')
            else:
                b_weight = op.get(b_name).get('weight')
                weighted_distance += b_weight
                print('Inserted', b_name, 'weight', b_weight, '(op)')
            continue
        elif b is None:
            print('Deleted, weight 0 (no penalty)')
            continue
        else:
            a_name = a.name
            b_name = b.name
            if a_name == b_name:
                print('No change, no weight.')
            else:

                a_weight = op[a_name]['weight'] if a_name in op else None
                b_weight = op[b_name]['weight'] if b_name in op else None

                if a_weight is None:  # a_name is either env-variable, float, bool

                    if b_weight:
                        weighted_distance += b_weight
                        print('Substituted non-op with op, weight', b_weight)
                    # elif True:  # sfeh
                    #     weighted_distance += 0
                    #     print('dummy exir with weight=0 for constants')
                    elif is_float(a_name) and is_float(b_name):
                        # weight_diff = max(0, min(1, abs(float(a_name) - float(b_name))))
                        # weighted_distance += weight_diff
                        print('Substituted float parameters. weight 0.')
                    elif is_bool(a_name) and is_bool(b_name):
                        weighted_distance += 1
                        print('Substituted bool parameters, weight 1.')
                    else:
                        weighted_distance += 1
                        print('Substituted leaves, weight 1.')

                else:
                    if b_weight is not None:
                        weighted_distance += b_weight
                        print('Substituted non-op with op, weight', b_weight)
                    # elif True:  # sfeh
                    #     weighted_distance += max(0.5, b_weight-(0.5*a_weight))
                    #     print('dummy exit with weight=0 for constants')
                    elif is_float(a_name) and is_float(b_name):
                        weighted_distance += 0.5
                        print('Substituted float parameters, weight 0.1.')
                    elif is_bool(a_name) and is_bool(b_name):
                        weighted_distance += 0.5
                        print('Substituted bool parameters, weight 0.1.')
                    else:
                        weighted_distance += 1
                        print('Substituted leaves, weight 0.1.')

                # constant weight for the occurance of an env-variables?
    return weighted_distance


# tree1 = aptree.from_text('{A{B{X}{Y}{F}}{C}}')
# tree2 = aptree.from_text('{A{C{D}{E}}{F}}')
# tree3, tree4 = aptree.from_text('{a{b}{c}}'), aptree.from_text('{a{b{d}}}')
# tree5, tree6 = aptree.from_text('{a}'), aptree.from_text('{b}')
# tree7, tree8 = aptree.from_text('{a}'), aptree.from_text('{a}')
# tree9, tree10 = aptree.from_text('{Ifte{<{pos}{0}}{0}{2}}'), aptree.from_text('{Ifte{<{+{{vel}{1.02}}{0}}{0}{2}}}')
# print('tree1', tree1)
#
# apted = APTED(tree1, tree2)
#
# ted = apted.compute_edit_distance()
# mapping = apted.compute_edit_mapping()
# print('Distance1:', ted)
# print('Mappinmg1:', mapping)
#
#
# weighted_distance = weight_ted_mapping(mapping)
# print('weighted_distance', weighted_distance)
#
# # apted = APTED(tree9, tree10)
# # ted = apted.compute_edit_distance()
# # mapping = apted.compute_edit_mapping()
# # print('DistanceB:', ted)
# # print('MappinmgB:', mapping)


# Ifte,<,0,2,observation1,0
# pre1 = '(Ifte(observation1<0),(0),(2)'
# raw1 = '{Ifte{<{0}{2}}{observation1}{0}}'
# raw2 = '{Ifte{<{0}{2}}{observation1}{0}}'
# tree_source = aptree.from_text('{a{b}{c}}')
# tree_orig = aptree.from_text('{a{b{d}}}')
# apted_disstance(tree_source, tree_orig)
#
#


def aptree_to_width(curlytree):
    """
    '{+{+{-{4}{5}}{1}}{*{2}{3}}}'
    ->
    [['+'], ['+', '*'], ['-', '1', '2', '3'], ['4', '5']]
    """
    xy = []
    y = -1
    buffer = ''
    for letter in curlytree:
        if letter == '{' or letter == '}':
            if buffer != '':
                xy.append([y, buffer])
                buffer = ''
            y += (1 if letter == '{' else -1)
        else:
            buffer += letter

    max_y = max(xy, key=lambda x: x[0])[0]
    widtree = [[] for _ in range(max_y + 1)]

    print(curlytree)
    print(xy)

    for y, label in xy:
        widtree[y].append(label)

    label_list = sum(widtree, [])

    return label_list


def tree_nodeid_ted_mapping(mapping):

    for ii, map_i in enumerate(mapping):
        a, b = map_i[0], map_i[1]
        if a is None:
            print(f'{ii} Inserted {b.name}')
            pass
        elif b is None:
            print(f'{ii} deleted {a.name}')
        else:
            a_name = a.name
            b_name = b.name
            if a_name == b_name:
                # print(f'{ii} No change, {a.name}')
                pass
            else:
                print(f'{ii} changed {a.name}')

    return


# weirdtree = '{+{+{-{4}{5}}{1}}{*{2}{3}}}'
# weirdtree = '{A{B{C{D}{E}}{F}}{G{H}{I}}}'
# print(aptree_to_width(weirdtree))
