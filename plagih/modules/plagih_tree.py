import sys
import os
import csv
import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skcv
from sympy import sympify, count_ops
from datetime import datetime
import plagih.modules.plagih_gp_pause as menu
# sfeh import the pause later, maybe
# import plagih.modules.plagih_gp_pause as menu
import tensorflow as tf
import ast
import pickle

# PLAGI imports
import re
from pydoc import locate  # convert stringed-type to type. ('float' -> float)
from plagih.modules.plagih_sympy_extras import plagih_sympify
from pprint import pprint
import matplotlib.pyplot as plt
import scipy.stats as st
from plagih.modules.tree_distances.tree_edit_distance import apted_distance
import time
from plagih.modules.dicts import *

### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

TR_ID = 0
TR_type = 1
TR_depth = 2
TRn_id = 3
TRn_depth = 4
TRn_type = 5
TRn_label = 6
TRn_parent = 7
TRn_arity = 8
TRn_c1 = 9
TRn_c2 = 10
TRn_c3 = 11
TR_fitness = 12
TRn_modify = 13
TR_parsimony = 14
TRn_um_lines = 15
P_first_node = 1

f2f, f2b, b2b, b2f, b2f2f = 0, 1, 2, 3, 4

ast_tensor_dict = {ast.Add: tf.add,  # e.g., a + b
                   ast.Sub: tf.subtract,  # e.g., a - b
                   ast.Mult: tf.multiply,  # e.g., a * b
                   ast.Div: tf.divide,  # e.g., a / b
                   ast.Pow: tf.pow,  # e.g., a ** 2
                   ast.USub: tf.negative,  # e.g., -a
                   ast.And: tf.logical_and,  # e.g., a and b
                   ast.Or: tf.logical_or,  # e.g., a or b
                   ast.Not: tf.logical_not,  # e.g., not a
                   ast.Eq: tf.equal,  # e.g., a == b
                   ast.NotEq: tf.not_equal,  # e.g., a != b
                   ast.Lt: tf.less,  # e.g., a < b
                   ast.LtE: tf.less_equal,  # e.g., a <= b
                   ast.Gt: tf.greater,  # e.g., a > b
                   ast.GtE: tf.greater_equal,  # e.g., a >= 1
                   'abs': tf.abs,  # e.g., abs(a)
                   'sign': tf.sign,  # e.g., sign(a)
                   'square': tf.square,  # e.g., square(a)
                   'sqrt': tf.sqrt,  # e.g., sqrt(a)
                   'pow': tf.pow,  # e.g., pow(a, b)
                   'log': tf.math.log,  # e.g., log(a)
                   'log1p': tf.math.log1p,  # e.g., log1p(a)
                   'cos': tf.cos,  # e.g., cos(a)
                   'sin': tf.sin,  # e.g., sin(a)
                   'tan': tf.tan,  # e.g., tan(a)
                   'acos': tf.acos,  # e.g., acos(a)
                   'asin': tf.asin,  # e.g., asin(a)
                   'atan': tf.atan,  # e.g., atan(a)
                   'Ifte': tf.compat.v2.where,  # e.g., Ifte(a, b, c)
                   'Mini': tf.math.minimum,  # if reduce_min does not work...
                   'Maxi': tf.math.maximum,
                   }
op = {
    'float': {'arity': 0, 'xtype': '2f', 'tf': ''},
    'int': {'arity': 0, 'xtype': '2f', 'tf': ''},
    'bool': {'arity': 0, 'xtype': '2b', 'tf': ''},

    '+': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '-': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '*': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '/': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    '**': {'arity': 2, 'xtype': 'f2f', 'tf': ''},
    'abs': {'arity': 1, 'xtype': 'f2f', 'tf': tf.abs},
    'sign': {'arity': 1, 'xtype': 'f2f', 'tf': tf.sign},
    'square': {'arity': 1, 'xtype': 'f2f', 'tf': tf.square},
    'sqrt': {'arity': 1, 'xtype': 'f2f', 'tf': tf.sqrt},
    'log': {'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log},
    'log1p': {'arity': 1, 'xtype': 'f2f', 'tf': tf.math.log1p},
    'cos': {'arity': 1, 'xtype': 'f2f', 'tf': tf.cos},
    'sin': {'arity': 1, 'xtype': 'f2f', 'tf': tf.sin},
    'tan': {'arity': 1, 'xtype': 'f2f', 'tf': tf.atan},
    'acos': {'arity': 1, 'xtype': 'f2f', 'tf': tf.acos},
    'asin': {'arity': 1, 'xtype': 'f2f', 'tf': tf.asin},
    'atan': {'arity': 1, 'xtype': 'f2f', 'tf': tf.atan},

    'And': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Or': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Xor': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Nand': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Xand': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Nor': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Xnor': {'arity': 2, 'xtype': 'b2b', 'tf': ''},
    'Not': {'arity': 1, 'xtype': 'b2b', 'tf': ''},

    '==': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '!=': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '<': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '<=': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '>': {'arity': 2, 'xtype': 'f2b', 'tf': ''},
    '>=': {'arity': 2, 'xtype': 'f2b', 'tf': ''},

    'Ftob': {'arity': 1, 'xtype': 'f2b', 'tf': ''},
    'Btof': {'arity': 1, 'xtype': 'b2f', 'tf': ''},

    'Ifte': {'arity': 3, 'xtype': 'b2f2f', 'tf': tf.compat.v2.where},  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Mini': {'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum},
    'Maxi': {'arity': 2, 'xtype': 'f2f', 'tf': tf.math.maximum},
}
op_xtype_dict = {  # Needs A LOT OF further testing
    'float': '2f',  # these three are dummies
    'int': '2f',  # neede to use the dict for function types aswell
    'bool': '2b',  # so we can "work around" use them

    '+': 'f2f',
    '-': 'f2f',
    '*': 'f2f',
    '/': 'f2f',
    '**': 'f2f',
    'abs': 'f2f',
    'sign': 'f2f',
    'square': 'f2f',
    'sqrt': 'f2f',
    'log': 'f2f',
    'log1p': 'f2f',
    'cos': 'f2f',
    'sin': 'f2f',
    'tan': 'f2f',
    'acos': 'f2f',
    'asin': 'f2f',
    'atan': 'f2f',

    'And': 'b2b',
    'Or': 'b2b',
    'Xor': 'b2b',
    'Nand': 'b2b',
    'Xand': 'b2b',
    'Nor': 'b2b',
    'Xnor': 'b2b',
    'Not': 'b2b',
    'ITE': 'b2b',

    '==': 'f2b',
    '!=': 'f2b',
    '<': 'f2b',
    '<=': 'f2b',
    '>': 'f2b',
    '>=': 'f2b',

    'Ftob': 'f2b',
    'Btof': 'b2f',  # False->0, True->1, dummy-function
    'Btof_extreme': 'b2f',  # False->-1, True->1. Does that make sense?

    'Ifte': 'b2f2f',  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Mini': 'f2f',
    'Maxi': 'f2f',
}
function_arity_dict = {  # Needs A LOT OF further testing
    'float': 0,  # these three are dummies
    'int': 0,  # neede to use the dict for function types aswell
    'bool': 0,  # so we can "workarounded" use them

    '+': 2,
    '-': 2,
    '*': 2,
    '/': 2,
    '**': 2,
    'abs': 1,
    'sign': 1,
    'square': 1,
    'sqrt': 1,
    'log': 1,
    'log1p': 1,
    'cos': 1,
    'sin': 1,
    'tan': 1,
    'acos': 1,
    'asin': 1,
    'atan': 1,

    'And': 2,
    'Or': 2,
    'Xor': 2,
    'Nand': 2,
    'Xand': 2,
    'Nor': 2,
    'Xnor': 2,
    'Not': 1,

    '==': 2,
    '!=': 2,
    '<': 2,
    '<=': 2,
    '>': 2,
    '>=': 2,

    'Ftob': 1,
    'Btof': 1,  # False->0, True->1, dummy-function
    'Btof_extreme': 1,  # False->-1, True->1. Does that make sense?

    'Ifte': 3,  # Note that boolean if's can be realized with boolean operators. (Or ITE())
    'Mini': 2,
    'Maxi': 2,
}

functions_wrap_dict = ['Mini', 'Maxi', 'abs', 'sign', 'square', 'sqrt', 'log', 'log1p', 'cos', 'sin', 'tan', 'acos', 'asin', 'atan']
functions_infix_dict = ['+', '-', '*', '/', '**', '==', '!=', '<', '>', '<=', '>=']

function_infix_to_prefix = {  # currently obsolete
    '+': 'add',
    '-': 'sub',
    '*': 'mult',
    '/': 'div',
    '**': 'power',
    '==': 'eq',
    '!=': 'neq',
    '<': 'lt',
    '<=': 'leq',
    '>': 'gt',
    '>=': 'geq',
}

sympy_dummy = plagih_sympify(1)
np.set_printoptions(linewidth=320)  # set the terminal to print 320 characters before line-wrapping in order to view Trees


def evolve_node_arity_fix(tree):
    """
    In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.

    This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and
    Crossover.

    """

    for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'
        if len(tree[3]) <= 2:
            print('FUCK {}'.format(tree))
        if tree[TRn_type][n] == 'term':  # check for discrepency
            tree[TRn_arity][n] = '0'  # set arity to 0
            tree[9][n] = ''  # wipe 'node_c1'
            tree[10][n] = ''  # wipe 'node_c2'
            tree[11][n] = ''  # wipe 'node_c3'
            tree[TRn_modify][n] = '1'

    return tree


def treegp_mutate_point_evolve(tree, arity='same'):
    """
    Mutate a single mutatable point in any Tree.
    """

    # 1. choose a node
    node_id = evolve_choose_mutatable_node_id(tree, mode='mutate_point')
    label = tree[TRn_label][node_id]
    node_xtype = xtype_label_get_xtype(tree[TRn_label][node_id])  # '>' -> 'f2b'

    if arity == 'same':
        # 2. perform point mutation on that specific node
        if tree[TRn_type][node_id] == 'func':
            tree[TRn_label][node_id] = evolve_func_get_func(label)  # Function is same type, same arity
        elif tree[TRn_type][node_id] == 'term':
            tree[TRn_label][node_id] = xtype_choose_term(node_xtype)  # 3 -> '2f' -> 5
        else:
            printpl('e', 'Operator type is not specified for PLAGIH ("term", "func",...)', tree[TRn_type][node_id])
            raise
    elif arity == 'plagih_switcharoo':
        printpl('e', 'SFEH this is TODO')
    else:
        printpl('e', 'treegp_mutate_point_evolve dies not know this method to handle the arity:', arity)

    return tree, node_id  # 'node' is returned only to be assigned to the 'tourn_trees' record keeping


def tree_branch(root_xtype, max_depth=''):
    """
    Builds a new 'branch'-tree
    TODO max_depth or max_nodes
    """

    tree_id, last_mod = 'x', 'b',
    tree = tree_init_first_column()
    if max_depth:

        label, arity = xtype_choose_func(root_xtype)

        if np.random.choice(['func', 'term']) == 'term':
            tree[TRn_type][top_id] = 'term'
            tree[TRn_label][top_id] = xtype_choose_term(xtype)  # replace with a correct label
            tree = np.delete(tree, branch_ids[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')
            tree = evolve_node_arity_fix(tree)  # fix all node arities (term)
            tree = tree_fix_link_child(tree)  # fix all child links (func)
            tree = evolve_node_renum(tree)  # renumber all 'node_id's

            # 5.2 We insert a function here
            tree = tree_replace_branch_nodelist(tree, branch_ids, tree)  # insert new 'branch' at point of mutation 'branch_top' in tourn_winner 'tree'

        tree = tree_init_first_column()
        tree = tree_add_node_fromnode(tree, node)

        treegp_mutate_branch_terminal_build()  # build the Terminal nodes
        # TODO set tree_depth_base in tree.
        return


def node_init():
    """
    Initialize a node (plagih-tree-node).
    A tree is build up of nodes, of which one this structure represents
    """

    node = {TR_ID: '',
            TR_type: '',
            TR_depth: '',
            TRn_id: '',
            TRn_depth: '',
            TRn_type: '',
            TRn_label: '',
            TRn_parent: '',
            TRn_arity: '',
            TRn_c1: '',
            TRn_c2: '',
            TRn_c3: '',
            TRn_modify: '1'}

    return node


def node_update(label, c1='', c2='', c3='', tree_id='', last_mod=''):
    """
    Initialize a node (plagih-tree-node).
    A tree is build up of nodes, of which one this structure represents
    """

    node = {TR_ID: '',
            TR_type: '',
            TR_depth: '',
            TRn_id: '',
            TRn_depth: '',
            TRn_type: '',
            TRn_label: label,
            TRn_parent: '',
            TRn_arity: '',
            TRn_c1: '',
            TRn_c2: '',
            TRn_c3: '4' if arity == 3 else '',
            TRn_modify: '1'}

    return node


def treegp_mutate_branch_terminal_build():
    """
    Build the Terminal nodes for the tree.

    """

    node[TRn_depth] = node[TR_depth]  # set the final node_depth (same as 'gp.pop[TRn_depth]' + 1)

    for j in range(1, len(tree[TRn_id])):  # go through all nodes
        if int(tree[TRn_depth][j]) == node[TRn_depth] - 1:  # this node is a parent
            for k in range(1, (int(tree[TRn_arity][j]) + 1)):  # increment through each degree of arity for each parent node
                node[TRn_parent] = int(tree[3][j])  # set the parent 'node_id'  ...
                treegp_branch_terminal(op_xtype_dict[tree[TRn_label][j]])  # ... generate a Terminal node

    return


def treegp_branch_terminal(terminal_xtype):
    """
    Generate a single Terminal node.

    """

    xtype_xtype_get_terminal(terminal_xtype)
    node[TRn_c1] = ''
    node[TRn_c2] = ''
    node[TRn_c3] = ''

    tree = tree_node_add_frominstance(tree)  # commit new node to array

    return


def treegp_branch_node(parent_arity_sum, prior_sibling_arity, prior_siblings, xtype):
    """
    Generate a single node (func or term) for

    """

    if np.random.choice(['func', 'term']) == 'func':  # randomly selected as Function
        label, arity = xtype_choose_func(xtype)
        node_function_select(node, xtype)  # retrieve a function, input-reverse the parent-function (f2b -> we need 2f input)
        node = ptree_node_link_child(tree, node, parent_arity_sum, prior_sibling_arity, prior_siblings)  # establish links to children
    else:
        xtype_xtype_get_terminal(xtype)  # was here
        node[TRn_c1] = ''
        node[TRn_c2] = ''
        node[TRn_c3] = ''

    tree = tree_node_add_frominstance(tree)  # commit new node to array
    prior_sibling_arity = prior_sibling_arity + node[TRn_arity]  # sum the arity of prior siblings

    return prior_sibling_arity


def ptree_node_link_child(tree, tr_node, parent_arity_sum, prior_sibling_arity, prior_siblings):
    """
    Fill in the tree-nodes metadata

    """
    if len(tree[3]) < 2:
        print('WHAAT \n{}'.format(tree))
    for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'
        if int(tree[TRn_depth][n]) == tr_node[TRn_depth] - 1:  # find all nodes that reside at the prior (parent) 'node_depth'
            c_buffer = tr_node[TRn_id] + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

            if tr_node[TRn_arity] == 0:  # terminal in a Grow Tree
                tr_node[TRn_c1] = ''
                tr_node[TRn_c2] = ''
                tr_node[TRn_c3] = ''

            elif tr_node[TRn_arity] == 1:  # 1 child
                tr_node[TRn_c1] = c_buffer
                tr_node[TRn_c2] = ''
                tr_node[TRn_c3] = ''

            elif tr_node[TRn_arity] == 2:  # 2 children
                tr_node[TRn_c1] = c_buffer
                tr_node[TRn_c2] = c_buffer + 1
                tr_node[TRn_c3] = ''

            elif tr_node[TRn_arity] == 3:  # 3 children
                tr_node[TRn_c1] = c_buffer
                tr_node[TRn_c2] = c_buffer + 1
                tr_node[TRn_c3] = c_buffer + 2

            else:
                printpl('e', 'tree_build_child_link: pop[TRn_arity] = {}'.format(node[TRn_arity]))

    return tr_node


def tree_subtree_build(tree, branch_ids):
    """
    Given: Tree and a node list
    - checks how far to build down
    - checks the old nodes xtype, etc.
    - checks if we are not too far down the tree
    -

    returns: new tree
    """

    label = tree[TRn_label][branch_ids[0]]
    xtype = xtype_label_get_xtype(label)
    top_depth = tree[TRn_depth][branch_ids[0]]
    top_node_id, bottom_id = int(branch_ids[0]), int(branch_ids[-1])
    top_parent = tree[TRn_parent][branch_ids[0]]

    node = node_init()
    depth_upper_bound = gp['tree_depth_max'] - int(top_depth)

    grow_method = gp['tree_growth']

    if grow_method == 'depth_base_uniform':
        """
        We allow base depth (which is a little lower than max)
        but every node has 0.5 chance to become a terminal
        - iterate over depths
        - fill with as many funcs as possible
        """

        depth_goal = min(gp['tree_depth_base'], depth_upper_bound)

        todo_xtypes = [xtype]
        result_label_list = []
        result_arity_list = []
        next_xtype_list = []

        # Build a list with labels in row, and a list wit their arities
        for depth in range(1, depth_goal + 1):
            for t in todo_xtypes:

                # Randomly choose a new label
                if np.random.choice(['func', 'term']) == 'term':
                    label = xtype_choose_term(t)
                    arity = 0
                else:
                    label, arity = xtype_choose_func(t)

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)

                # create a new xtype-'To-do' list for the next depth
                if label == 'Ifte':
                    next_xtype_list.extend(['2b', '2f', '2f'])
                else:
                    tmp_xtype = xtype_label_get_xtype(label)
                    for n in range(0, arity):  # when arity==2, add 2 times
                        next_xtype_list.append(tmp_xtype)

                # Finally, update the list for the next round
                todo_xtypes = next_xtype_list[:]
        print('Oky {}'.format(result_label_list))

        tree = tree_from_labels(tree, branch_ids, result_label_list)

    elif grow_method == 'old plagih code':
        printpl('e', 'Not yet')

        width_goal = 1
        nodes_cnt = 0  # reset for 'c_buffer' in 'children_link'
        prior_s = 0  # reset for 'c_buffer' in 'children_link'

        # go as far wide as needed
        for count in range(1, width_goal + 1):

            # Check, how many of the lower nodes
            if label == 'Ifte':
                # build up "parent" list
                nodes_cnt = treegp_branch_node(width_goal, nodes_cnt, prior_s, '2b')
                prior_s += 1
                nodes_cnt = treegp_branch_node(width_goal, nodes_cnt, prior_s, '2f')
                prior_s += 1
                nodes_cnt = treegp_branch_node(width_goal, nodes_cnt, prior_s, '2f')
                prior_s += 1
            else:
                node[TRn_parent] = todo_xtypes[count]  # set the nodes parent
                parent_func_xtype = op_xtype_dict[tree[TRn_label][node[TRn_parent]]]  # find parents node
                xtype = parent_func_xtype[:2][::-1]
                nodes_cnt = treegp_branch_node(width_goal, nodes_cnt, prior_s, xtype)
                prior_s += 1

            node[TRn_label] = label
            # needen: label, c1, c2, c3, depth, parent, arity, depth, type, modify

        for count in range(1, len(tree[TRn_depth])):  # increment through all nodes in array 'tree'
            if int(tree[TRn_depth][count]) == node[TRn_depth] - 1:  # find parent nodes which reside at the prior depth
                width_goal = width_goal + int(tree[TRn_arity][count])  # sum arities of all parent nodes at the prior depth

        # how many nodes
        node = node_update()

        node = treegp_branch_functions(tree, node)  # build all the Function nodes

        node = node_asdf(node, label, arity)

        if branch_depth == 0:  # the point of mutation ('branch_top') chosen resides at the maximum allowable depth, so mutate term to term
            tree[TRn_label][top_node_id] = xtype_choose_term(xtype)

        else:
            tree_branch(xtype, max_depth=branch_depth)  # build new Tree ('gp.tree') with a maximum depth which matches 'branch'
    elif grow_method == 'nodes_max_uniform':
        """
        We allow a certain amount of new nodes instead tree depth.
        This could be calculated respectively to the parsimony level
        which the tree might have up his sleeve
        """
        raise
    else:
        printpl('e', 'That did not work')

    return tree


def evolve_subtree_depth_choose(tree, top_id, bottom_id, amount_replaced_nodes, mode='base_depth'):  # sfeh other default
    """
    Return the size of the tree to be inserted.
    Should not be set to maximum to reduce complexity!
    """

    # TODO consider tree size of last tree,
    # TODO consider random tree size,
    # TODO consider always maximum tree size,
    # TODO is this already considered by 50:50 func-term?

    depth_old = int(tree[TRn_depth][bottom_id]) - int(tree[TRn_depth][top_id])  # subtract depth of 'branch_top' from the last in 'branch'
    depth_upper_bound = gp['tree_depth_max'] - int(tree[TRn_depth][top_id])  # = 10 - (node_depth)
    if mode == 'maximum':
        return depth_upper_bound
    elif mode == 'same_length':
        return depth_old
    elif mode == 'base_depth':
        return min(gp['tree_depth_base'], depth_upper_bound)
    elif mode == 'random':
        return min(depth_upper_bound, np.random.randint(0, 1 + max(depth_upper_bound, 3)))  # SFEH random depth, I hope this is enough to guarantee tree size
    else:
        printpl('e', 'evolve_subtree_depth_choose does not accept this mode: {}'.format(mode))
        raise
    return branch_depth


def tree_replace_branch_nodelist(tree, tree_ids, insert_tree):
    """
    This method enables the insertion of tree_branch_node_ids in place of a branch
    tree_branch_node_ids = [5,6,8,9] node that are changed

    The end result is a Tree with a mutated branch.
    """

    branch_top = int(tree_ids[0])
    # tree[TRn_type][branch_top] = 'func'
    tree[TRn_label][branch_top] = insert_tree[TRn_label][1]  # copy node_label from new tree
    tree[TRn_arity][branch_top] = insert_tree[TRn_arity][1]  # copy node_arity from new tree
    tree = np.delete(tree, tree_ids[1:], axis=1)  # delete all nodes beneath point of mutation ('branch_top')

    c_buffer = evolve_c_buffer(tree, branch_top)  # generate c_buffer for point of mutation ('branch_top')
    tree = evolve_subtree_insert_child(tree, branch_top, c_buffer)  # insert a single new node ('branch_top')
    tree = evolve_node_renum(tree)  # renumber all 'node_id's

    ### PART 2 - insert b_branchody from 'gp.tree' into 'tree' ###
    node_count = 2  # set node count for 'gp.tree' to 2 as the new root has already replaced 'branch_top' (above)

    while node_count < len(insert_tree[3]):  # increment through all nodes in the new Tree ('gp.tree'), starting with node 2

        for j in range(1, len(tree[3])):  # increment through all nodes in tourn_winner ('tree')

            if tree[TRn_type][j] == '':
                tree[TRn_type][j] = insert_tree[TRn_type][node_count]  # copy 'node_type' from branch to tree
                tree[TRn_label][j] = insert_tree[TRn_label][node_count]  # copy 'node_label' from branch to tree
                tree[TRn_arity][j] = insert_tree[TRn_arity][node_count]  # copy 'node_arity' from branch to tree

                if tree[TRn_type][j] == 'term':
                    tree = tree_fix_link_child(tree)  # fix all child links
                    tree = evolve_node_renum(tree)  # renumber all 'node_id's

                if tree[TRn_type][j] == 'func':
                    c_buffer = evolve_c_buffer(tree, j)  # generate 'c_buffer' for point of mutation ('branch_top')
                    tree = evolve_subtree_insert_child(tree, j, c_buffer)  # insert new nodes
                    tree = tree_fix_link_child(tree)  # fix all child links
                    tree = evolve_node_renum(tree)  # renumber all 'node_id's

                node_count = node_count + 1  # exit loop when 'node_count' reaches the number of columns in the array 'gp.tree'

    return tree


def evolve_subtree_insert_child(tree, node, c_buffer):
    """
    Insert child node into the copy of a parent Tree.

    """

    if int(tree[TRn_arity][node]) == 0:  # if arity = 0
        printpl('e', 'In evolve_child_insert: node', node, 'has arity 0')
        plagih_pause()  # consider special instructions for this

    elif int(tree[TRn_arity][node]) == 1:  # if arity = 1
        tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
        tree[3][c_buffer] = c_buffer  # node ID
        tree[TRn_depth][c_buffer] = int(tree[TRn_depth][node]) + 1  # node_depth
        tree[7][c_buffer] = int(tree[3][node])  # parent ID

    elif int(tree[TRn_arity][node]) == 2:  # if arity = 2
        tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
        tree[3][c_buffer] = c_buffer  # node ID
        tree[TRn_depth][c_buffer] = int(tree[TRn_depth][node]) + 1  # node_depth
        tree[7][c_buffer] = int(tree[3][node])  # parent ID

        tree = np.insert(tree, c_buffer + 1, '', axis=1)  # insert node for 'node_c2'
        tree[3][c_buffer + 1] = c_buffer + 1  # node ID
        tree[TRn_depth][c_buffer + 1] = int(tree[TRn_depth][node]) + 1  # node_depth
        tree[7][c_buffer + 1] = int(tree[3][node])  # parent ID

    elif int(tree[TRn_arity][node]) == 3:  # if arity = 3
        tree = np.insert(tree, c_buffer, '', axis=1)  # insert node for 'node_c1'
        tree[3][c_buffer] = c_buffer  # node ID
        tree[TRn_depth][c_buffer] = int(tree[TRn_depth][node]) + 1  # node_depth
        tree[7][c_buffer] = int(tree[3][node])  # parent ID

        tree = np.insert(tree, c_buffer + 1, '', axis=1)  # insert node for 'node_c2'
        tree[3][c_buffer + 1] = c_buffer + 1  # node ID
        tree[TRn_depth][c_buffer + 1] = int(tree[TRn_depth][node]) + 1  # node_depth
        tree[7][c_buffer + 1] = int(tree[3][node])  # parent ID

        tree = np.insert(tree, c_buffer + 2, '', axis=1)  # insert node for 'node_c3'
        tree[3][c_buffer + 2] = c_buffer + 2  # node ID
        tree[TRn_depth][c_buffer + 2] = int(tree[TRn_depth][node]) + 1  # node_depth
        tree[7][c_buffer + 2] = int(tree[3][node])  # parent ID

    else:
        printpl('e', 'In evolve_child_insert: node', node, 'arity > 3')
        plagih_pause()  # consider special instructions for this (pause)

    return tree


def tree_choose_branch_ids(tree, node=None):
    """
    chooses a mutatable branch to mutate
    - specify a starting node
    - return all child-nodes as list
    """

    branch = np.array([])  # the array is necessary in order to len(branch) when 'branch' has only one element

    if node:  # Crossover: Option to specify own starting node
        branch_top = node
    else:
        branch_top = evolve_choose_mutatable_node_id(tree, mode='mutate_branch_no_root')  # "2" returns mutable node (except root node)

    # 2. Also return all child nodes
    branch_eval = tree_node_get_childlist(tree, branch_top)  # generate tuple of 'branch_top' and subsequent nodes
    branch_symp = plagih_sympify(branch_eval)  # convert string into something useful # sfeh: simple sympy might be faster
    branch = np.append(branch, branch_symp)  # append list to array
    branch = np.sort(branch)  # sort nodes in branch for Crossover.

    return branch


def evolve_c_buffer(tree, node):
    """
    Generates the c_buffer for a node of a tree

    """

    parent_arity_sum = 0
    prior_sibling_arity = 0
    prior_siblings = 0

    for n in range(1, len(tree[3])):  # increment through all nodes (exclude 0) in array 'tree'

        if int(tree[TRn_depth][n]) == int(tree[TRn_depth][node]) - 1:  # find parent nodes at the prior depth
            if tree[TRn_arity][n] != '':
                parent_arity_sum = parent_arity_sum + int(tree[TRn_arity][n])  # sum arities of all parent nodes at the prior depth

        if int(tree[TRn_depth][n]) == int(tree[TRn_depth][node]) and int(tree[3][n]) < int(tree[3][node]):  # find prior siblings at the current depth
            if tree[TRn_arity][n] != '':
                prior_sibling_arity = prior_sibling_arity + int(tree[TRn_arity][n])  # sum prior sibling arity
            prior_siblings = prior_siblings + 1  # sum quantity of prior siblings

    c_buffer = node + (parent_arity_sum + prior_sibling_arity - prior_siblings)  # One algo to rule the world!

    return c_buffer


def tree_fix_link_child(tree):
    """
    In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

    This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.

    """

    for node in range(1, len(tree[3])):
        c_buffer = evolve_c_buffer(tree, node)  # generate c_buffer for each node
        tree = evolve_fix_link_child_doit(tree, node, c_buffer)  # update child links for each node

    return tree


def evolve_fix_link_child_doit(tree, node, c_buffer):
    """
    Link each parent node to its children.

    """

    if int(tree[3][node]) == 1:
        # SFEH Root can only be ignored, if root was not changed
        c_buffer = c_buffer + 1  # if root (node 1) is passed through this method

    if tree[TRn_arity][node] != '':

        if int(tree[TRn_arity][node]) == 0:  # if arity = 0
            tree[9][node] = ''
            tree[10][node] = ''
            tree[11][node] = ''

        elif int(tree[TRn_arity][node]) == 1:  # if arity = 1
            tree[9][node] = c_buffer
            tree[10][node] = ''
            tree[11][node] = ''

        elif int(tree[TRn_arity][node]) == 2:  # if arity = 2
            tree[9][node] = c_buffer
            tree[10][node] = c_buffer + 1
            tree[11][node] = ''

        elif int(tree[TRn_arity][node]) == 3:  # if arity = 3
            tree[9][node] = c_buffer
            tree[10][node] = c_buffer + 1
            tree[11][node] = c_buffer + 2

        else:
            printpl('e', 'evolve_child_link: node', node, 'has arity', tree[TRn_arity][node])
            raise  # plagih_pause()  # consider special instructions for this (pause)

    return tree


def evolve_fix_link_parent(tree):
    """
    In a given Tree, fix 'parent_id' for all nodes.

    This is automatically handled in all mutations except with Crossover due to the need to copy branches 'a' and
    'b' to their own trees before inserting them into copies of	the parents.

    Technically speaking, the 'node_parent' c1 is not used by any methods. The parent ID can be completely out
    of whack and the expression will work perfectly. This is maintained for the sole purpose of granting the user
    a friendly, makes-sense interface which can be read in both directions.

    Called by: evolve_branch_copy

    Arguments required: tree
    """

    ### THIS METHOD MAY NOT BE REQUIRED AS SORTING 'branch' SEEMS TO HAVE FIXED 'parent_id' ###

    for node in range(1, len(tree[3])):

        if tree[9][node] != '':
            child = int(tree[9][node])
            tree[7][child] = node

        if tree[10][node] != '':
            child = int(tree[10][node])
            tree[7][child] = node

        if tree[11][node] != '':
            child = int(tree[11][node])
            tree[7][child] = node

    return tree


def evolve_node_renum(tree):
    """
    Renumber all 'node_id' in a given tree.

    This is required after a new generation is evolved as the node_id numbers are carried forward from the previous
    generation but are no longer in order.

    """

    for n in range(1, len(tree[3])):
        tree[3][n] = n  # renumber all nodes

    return tree


def evolve_choose_mutatable_node_id(tree, mode='', same_xtype=''):
    """
    Returns a mutatable node for point-mutation
    -> no_root handles
    """
    # TODO only works for 2-array functions

    node_ids = []

    # 1. Build up a list with nodes
    if same_xtype:
        for i, label in enumerate(tree[TRn_label]):
            if tree[TRn_modify][i] == '1':  # also skips node 0
                # TODO make this faster
                node_xtype = xtype_label_get_xtype(tree[TRn_label][i])
                if xtype_outcome_equi_test(node_xtype, same_xtype):
                    node_ids.append(int(tree[3][i]))
    else:
        for i, x in enumerate(tree[TRn_type]):
            if tree[TRn_modify][i] == '1':
                node_ids.append(int(tree[3][i]))

    # 2. Kick out root if it is there?
    if 'no_root' in mode:  # delete root node
        node_ids = [x for x in node_ids if x != 1]

    # 3: return the node. Not safe, could be try-except block.
    # eg: all nodes are not modifiable
    # eg. all nodes are not of correct type
    node_id = np.random.choice(node_ids)
    return node_id


# +++++++++++++++++++++++++++++++++++++++++++++
#   Work with trees                           |
# +++++++++++++++++++++++++++++++++++++++++++++

def tree_node_add_frominstance(tree):
    """
    Commit the values of a new node (root, function, or terminal) to the array 'tree'.
    TODO
    """

    tree = np.append(tree, [[node[TR_ID]],
                            [node[TR_type]],
                            [node[TR_depth]],
                            [node[TRn_id]],
                            [node[TRn_depth]],
                            [node[TRn_type]],
                            [node[TRn_label]],
                            [node[TRn_parent]],
                            [node[TRn_arity]],
                            [node[TRn_c1]],
                            [node[TRn_c2]],
                            [node[TRn_c3]],
                            '',  # [node[TR_fitness]],
                            ['1'],
                            '',  # [node[TR_parsimony]]
                            ], 1)

    node[TRn_id] = node[TRn_id] + 1

    return tree


def tree_node_add_fromvalues(tree, node_id, node_depth,
                             node_type, node_label, node_parent, node_arity, node_c1,
                             node_c2, node_c3):
    np.append(tree,
              ['', '', '', [node_id], [node_depth], [node_type],
               [node_label], [node_parent], [node_arity], [node_c1], [node_c2], [node_c3],
               '', '', ''], 1)
    return tree


def tree_data_load_origin_tree(origin_tree_file_path):
    """
    This loads the 'origin' and evaluates it

    Arguments required: path to csv
    returns: tree
    """

    # Check if the user provided an origin
    if origin_tree_file_path == '':
        # Probably the best idea is to specify the outcome only. e.g. float
        printpl('t', 'No origin provided. Need to rework everything for this case.')
        raise

    # Load origin from file
    with open(origin_tree_file_path, 'r') as csv_file:
        target = csv.reader(csv_file, delimiter=',')
        tree = np.array([[]])
        for row in target:
            if tree.shape[1] == 0:  # looks if tree is empty
                tree = np.append(tree, [row], axis=1)  # append first row to Tree ('tree_id')
            else:
                tree = np.append(tree, [row], axis=0)  # append subsequent rows to Tree
        if tree.shape[0] == TRn_um_lines:  # (+ row 0)
            pass  # print('Origin Tree is: \n' + str(tree))
        else:
            printpl('e', "Tree could not be imported correctly from .csv file.")
            raise
    tree[TR_parsimony][1] = 0  # the distance to it is 0 by definition
    origin_algo_raw = tree_expr_raw(tree, P_first_node)
    origin = {'tree': tree,
              'algo_raw': origin_algo_raw,
              'algo_sym': tree_expr_sympify(algo_raw_str=origin_algo_raw),
              'parsimony': 0}

    origin_hash, origin_meta = tree_store_meta_get_hash(tree)
    origin['fitness_train'] = origin_meta['fitness_train']

    hashtable_fitness_train = {}
    return


def tree_store_meta_get_hash(tree, store_in_tree=True):
    """
    gets all the main tree information
    1. algo_raw
    2. tree_identifier (algo_raw)
    2. algo_sym
    3. parsimony
    4. fitness_train
    """
    # 1. get algo_raw - what is needed to compute the tree identifier
    algo_raw_str = tree_expr_raw(tree, 1)
    tree_ident = hash(algo_raw_str)  # sfeh: potential for improvement- use algo_sym in separate dict as identifier.

    # 2.1 Did we have this tree already? -> Nice, we have everything
    if tree_ident in tree_hash_meta:
        tree_meta = tree_hash_meta[tree_ident]

    # 2.2 New tree, but still Skip fitness eval for complex trees
    else:
        parsimony = tree_parsimony(tree)

        # 3. compute fitness
        if parsimony < parsimony_min_max[1]:
            # 3.1 With tensorflow
            algo_sym = tree_expr_sympify(algo_raw_str=str(algo_raw_str))
            fitness_train = eval_tf(algo_sym, data_train)['fitness']
        else:
            # 3.2 just fill with bad values
            algo_sym = sympy_dummy
            fitness_train = fitness_bad_dummy

        # 4. All the tree-specific meta data into the
        tree_meta = {'algo_raw': str(algo_raw_str), 'tree_ident': tree_ident, 'algo_sym': str(algo_sym), 'parsimony': float(parsimony), 'fitness_train': float(fitness_train)}
        tree_hash_meta[tree_ident] = tree_meta

    # 5. store fitness in 'old' Karoo tree structure
    if store_in_tree:
        tree_store_parsimony(tree, tree_meta['parsimony'])
        tree_store_fitness(tree, tree_meta['fitness_train'])
        # tree_store_meta_lastgen(tree)

        return tree_ident, tree_meta


def tree_modifyable_nodes_set(chosen_tree):
    """
    Sets all the origin core nodes back to non-modifyable
    """
    # Set all nodes to be modifiable (=1)
    for i, tmp in enumerate(chosen_tree[TRn_modify][1:]):
        chosen_tree[TRn_modify][i + 1] = '1'

    # Find no-modifyables in Origin
    non_modifiable_nodes = []
    if origin['tree'][TRn_modify][1] == '0':  # check is modifiable nodes are specified
        non_modifiable_nodes.extend(tree_nomodifyable_nodes_get(1, chosen_tree, 1))

    for non_modifiable in non_modifiable_nodes:
        chosen_tree[TRn_modify][non_modifiable] = '0'

    return chosen_tree


def tree_nomodifyable_nodes_get(origin_node, chosen_tree, chosen_node):
    """
    Returns a list of nodes that are not supposed to be modified
    """

    if origin['tree'][TRn_modify][origin_node] == '0':
        non_modifiables = []
        non_modifiables.append(int(chosen_tree[3][chosen_node]))
        for child in [9, 10, 11]:
            if origin['tree'][child][origin_node] != '':
                next_origin_node = int(origin['tree'][child][origin_node])
                next_chosen_node = int(chosen_tree[child][chosen_node])
                tmp = tree_nomodifyable_nodes_get(next_origin_node, chosen_tree, next_chosen_node)
                if tmp is not None:
                    non_modifiables.extend(tmp)
        return non_modifiables
    else:
        return


def tree_build_type_constant_get(term_type='', mode='float-1to1', uniform_range=''):
    """
    todo random samples
    Returns a constant that fits into the position
    -- term_type = 'float'
    """
    if uniform_range:
        return np.random.uniform(uniform_range[0], uniform_range[1])

    if term_type == 'bool':
        return np.random.choice([True, False])
    elif term_type == 'float':
        if mode == 'float-1to1':
            return np.random.uniform(-1, 1)
        elif mode == 'intTotal_10':
            return np.random.random_integers(-10, 10)
        elif mode == 'random_optimised':
            return np.random.choice([-10, -5, -2, -1, -1, -0.8, -0.6, -0.5, -0.4, -0.2, 0, 10,
                                     5, 2, 1, 1, 0.8, 0.6, 0.5, 0.4, 0.2, 0])
        else:
            # sfeh: gibt viele Verteilungen: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html
            printpl('e', 'You did not take care of the kind of numbers you want to have')
            raise
    elif term_type == 'int':
        # TODO give more opportunities, similar to random floats
        return np.random.random_integers(-10, 10)
    else:
        printpl('w', 'Please specify your desired datatype if possible. Trying to return c1 similar to terminals.')
        printpl('e', 'This term type should not occur, I guess', term_type)
        term_type = np.random.choice(variables_dict['types'])
        return tree_build_type_constant_get(term_type=term_type)


def tree_expr_sympify(algo_raw_str='', tree=''):
    """
    returns the sympifyed expression
    """
    if len(tree) > 0:  # If we got a tree, we generate the expression
        algo_raw_str = str(tree_expr_raw(tree, 1))

    try:
        x = plagih_sympify(algo_raw_str)
        strx = str(x)

        if 'zoo' in strx:
            x = re.sub('zoo', '10', strx)  # TODO how to handle zoo?

        if 'nan' in strx:  # Happens when 0/0 occurs. This tree is worth nothing anyways
            printpl('w', 'We had a "nan"')
            remove_this_tree()
            return str(sympy_dummy)
        else:
            return str(x)
    except:
        printpl('w', 'In sympify. Caused by this raw algorithm: ' + str(algo_raw_str))
        # todo.
        remove_this_tree()
        return str(sympy_dummy)


def remove_this_tree():
    """
    If a tree makes problems, delete it somehow.
    - set parsimony very high?
    """


def tree_expr_raw(tree, node_id):
    """
    Evaluate all or part of a Tree (starting at node_id) and return a raw multivariate expression ('algo_raw').
    sfeh/todo: this can be optimized to create a nicer brackets-styled algorithm
    """
    node_id = int(node_id)

    if tree[TRn_arity, node_id] == '0':  # arity of 0 for the pattern '[term]'
        return '(' + tree[TRn_label, node_id] + ')'  # 'node_label' (function or terminal)

    elif tree[TRn_arity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        return '(' + tree_expr_raw(tree, tree[9, node_id]) + tree[TRn_label, node_id] + ')'

    elif tree[TRn_arity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
        # This if case is for 2-ary ops that is prefix. like Min(a, b)
        if tree[TRn_label, node_id] not in functions_infix_dict:
            return '(' + tree[TRn_label, node_id] + '(' + tree_expr_raw(tree, tree[9, node_id]) + ', ' + tree_expr_raw(tree, tree[10, node_id]) + '))'
        else:
            return '(' + tree_expr_raw(tree, tree[9, node_id]) + tree[TRn_label, node_id] + tree_expr_raw(tree, tree[10, node_id]) + ')'  # Klammern, da sympify sonst abkacnen könnte

    elif tree[TRn_arity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return '(Ifte(' + tree_expr_raw(tree, tree[9, node_id]) + ', ' + tree_expr_raw(tree, tree[10, node_id]) + ', ' + tree_expr_raw(tree, tree[11, node_id]) + '))'


def tree_raw_depth_prefix(tree, node_id):
    """
    Does the same as tree_expr_raw, but evaluates infix functions in prefix notation (functional form)

    """

    node_id = int(node_id)

    if tree[TRn_arity, node_id] == '0':  # arity of 0 for the pattern '[term]'
        return '{' + tree[TRn_label, node_id] + '}'  # 'node_label' (function or terminal)

    elif tree[TRn_arity, node_id] == '1':  # arity of 1 for the explicit pattern 'not [eval]'
        return '{' + tree[TRn_label, node_id] + tree_raw_depth_prefix(tree, tree[9, node_id]) + '}'

    elif tree[TRn_arity, node_id] == '2':  # arity of 2 for the pattern '[eval] [func] [eval]'
        return '{' + tree[TRn_label, node_id] + '' + tree_raw_depth_prefix(tree, tree[9, node_id]) + tree_raw_depth_prefix(tree, tree[10, node_id]) + '' + '}'

    elif tree[TRn_arity, node_id] == '3':  # arity of 3 for the explicit pattern 'Ifte(a, b, c)'
        return '{Ifte' + tree_raw_depth_prefix(tree, tree[9, node_id]) + tree_raw_depth_prefix(tree, tree[10, node_id]) + tree_raw_depth_prefix(tree, tree[11, node_id]) + '' + '}'


def tree_node_get_childlist(tree, node_id):
    """
    return a list of s nodes childs.
    + Evaluate all or part of a Tree and

    This method generates a list of all 'node_id's from the given Node and below. It is used primarily to generate
    'branch' for the multi-generational mutation of Trees.
    TODO what does this exactly?
    """

    node_id = int(node_id)

    if tree[TRn_arity, node_id] == '0':  # arity of 0 for the pattern '[node_id]'
        return tree[3, node_id]  # 'node_id'

    else:
        if tree[TRn_arity, node_id] == '1':  # arity of 1 for the pattern '[node_id], [node_id]'
            return '{}, {}'.format(tree[3, node_id], tree_node_get_childlist(tree, tree[9, node_id]))

        elif tree[TRn_arity, node_id] == '2':  # arity of 2 for the pattern '[node_id], [node_id], [node_id]'
            return '{}, {}, {}'.format(
                tree[3, node_id],
                tree_node_get_childlist(tree, tree[9, node_id]),
                tree_node_get_childlist(tree, tree[10, node_id]))

        elif tree[TRn_arity, node_id] == '3':  # arity of 3 for the pattern '[node_id], [node_id], [node_id], [node_id]'
            return '{}, {}, {}, {}'.format(
                tree[3, node_id],
                tree_node_get_childlist(tree, tree[9, node_id]),
                tree_node_get_childlist(tree, tree[10, node_id]),
                tree_node_get_childlist(tree, tree[11, node_id]))


def tree_parsimony(tree, parsimony_distance='ted'):
    """
    parsimony_distance: compute the chosen distance by the user.

    """
    if parsimony_distance == 'ted':
        return tree_parsimony_ted(origin['tree'], tree)
    elif parsimony_distance == 'total_count_nodes':
        return int(tree[3][-1:])  # returns the tree size
    elif parsimony_distance == 'total_tree_depth':
        return tree[TRn_depth][1]  # returns the tree size
    elif parsimony_distance == 'total_karoo_original':  # do not use with long variable names
        algo_raw_str = str(tree_expr_raw(tree, 1))
        return len(str(algo_raw_str))
    # elif parsimony_distance == 'total_simplified':
    #     algo_sym = tree_expr_sympify(tree=tree)
    #     return count_ops(algo_sym)
    elif parsimony_distance == 'rel_ari_1':  # Does this work?
        return tree_parsimony_relari(tree)
    else:
        printpl('i', 'Parsimony distance not specified! Use default.')
        tree_parsimony(tree)


def tree_parsimony_relari(tree):
    """
    This distance penalizes non-original functions with its arity
    - ignore node[0] [description]
    - look within the subtree if the original function is on origin spot
    """

    # If the new tree is actually less complex than the original one, just return 1
    if len(tree[TRn_label]) < len(origin['tree'][TRn_label]):
        return 1

    distance = 0

    # iterate over every node in the new tree
    for i, arity in enumerate(tree[TRn_arity]):
        if i == 0:  # skip node 0. the description
            continue
        elif i < len(origin['tree'][TRn_label]):  # Make sure we stay within the tree index. <= does not work
            if origin['tree'][TRn_label][i] != tree[TRn_label][i]:  # is it different from the origin?
                distance = distance + int(arity)  # add the nodes arity. double-punishes large trees
        else:
            distance = distance + int(arity)

    return max(distance, 1)  # make sure, it does not return 0


def tree_parsimony_ted(tree1, tree2):
    """
    The Tree Edit distance (TED) ('coolest' distance)
    - the amount of changes that have to be applied to the origin to equality are counted
    """
    # TODO TED soll geänderte Werte ignorieren
    apted_tree1 = tree_raw_depth_prefix(tree1, 1)
    apted_tree2 = tree_raw_depth_prefix(tree2, 1)
    distance, mapping = apted_distance(apted_tree1, apted_tree2)
    # sfeh the mapping could be handy somewhere
    return distance


def tree_store_fitness(tree, fitness):
    """
    Store the fitness within the tree np-array

    """

    fitness = float(fitness)
    fitness = round(fitness, precision)

    tree[TR_fitness][1] = fitness  # store the fitness with each tree

    return


def tree_store_parsimony(tree, parsimony):
    """
    Store the parsimony within the tree np-array
    """
    if parsimony < 0:
        printpl('w', 'Parsimony is:', parsimony)
    tree[TR_parsimony][1] = parsimony


def tree_store_meta_lastgen(tree, modification=''):
    """
    Remove all fitness data from a given tree.

    This is required after a new generation is evolved as the fitness of the same Tree prior to its mutation will
    no longer apply.

    """

    # save information about how good last changes were
    # for i in range(min(tree_depth_min, 5), 2, -1):  # 5,4,3,2
    #     tree[TR_type][i] = tree[TR_type][i-1]    # The last modifications
    #     tree[TR_fitness][i] = tree[TR_fitness][i-1]  # The last fitness
    #     tree[TR_parsimony][i] = tree[TR_parsimony][i-1]  # The last parsimony (TODO) # tree_id,1,a,b,c -> tree_id,1,a,a,b

    # What needs to be assigned later
    # tree[TR_type][2] = modification  # wipe last modification data
    # tree[TR_ID][1] = ''  # -> tree_id,,
    # tree[TR_fitness][1] = ''  # wipe fitness data
    # tree[TR_parsimony][1] = ''  # wipe parsimony data

    return tree


def eval_tf(expr, data, get_pred_labels=False):
    """
    computes gp-tree results and fitness scores.
    - Computes tree expression using TensorFlow (TF)
    - parsing input string 'expression' and converting it into a TF operation graph
    - processing tf graph in an isolated TF session (results and corresponding fitness)

        'tf_device' - controls which device will be used for computations (CPU or GPU).
        'tf_device_log' - controls device placement logging (debug only).

    Args:
        'expr' - a string expression to be computed on the data. Variable -> 'terminals'
        'data' - an 'n by m' matrix of the data points containing n observations like 'terminals'.
        'get_pred_labels' - (Classify Kernel) a boolean flag which controls whether the predicted labels should be
        extracted from the evolved results.

    Returns:
        A dict mapping keys to the following outputs:
            'result'            - array of the results of applying given expression to the data
            'pred_labels'       - (Classify) an array of the predicted labels extracted from the results
            'solution'          - array of the solution values extracted from the data (variable 's' in the dataset)
            'pairwise_fitness'  - array of the element-wise results of applying the fitness kernel function
            'fitness'           - aggregated scalar fitness score

    """

    # Initialize TensorFlow session
    tf.compat.v1.reset_default_graph()  # tf.reset_default_graph()
    config = tf.compat.v1.ConfigProto(log_device_placement=tf_device_log, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        with sess.graph.device(tf_device):

            # 1. data (observations, actions) to tensors
            tensors = {}

            num_terminals = len(variables_dict['all'])
            num_actions = len(actions)

            for i in range(num_terminals):
                var = variables_dict['all'][i]
                if '2f' in xtype_node_get_xtype(var, 'term'):
                    tensors[var] = tf.constant(data[:, i], dtype=tf.float32)  # converts data into vectors
                else:  # '2b'
                    tensors[var] = tf.constant(data[:, i], dtype=tf.bool)

            for i in range(num_actions):
                var = actions[i]
                action_xtype = xtype_node_get_xtype(var, 'term')
                if '2f' in action_xtype:
                    tensors[var] = tf.constant(data[:, num_terminals + i], dtype=tf.float32)  # converts data into vectors
                elif '2b' in action_xtype:  # '2b'
                    printpl('t', 'Currently no kernel available for boolean fitness')
                    tensors[var] = tf.constant(data[:, i], dtype=tf.bool)
                else:
                    printpl('e', 'Kernel not known for: {} which is {}.'.format(var, action_xtype))

            # 2- Transform string expression into TF operation graph
            tf_result = eval_tf_ast_expr(expr, tensors)
            pred_labels = tf.no_op()  # a placeholder, applies only to CLASSIFY kernel

            # TODO currently does only support one label
            solution = tensors['action0']  # solution c1 is assumed to be stored in this terminal
            # 3- Add fitness computation into TF graph
            if kernel == 'c':  # CLASSIFY kernel

                """
                Creates element-wise fitness computation TensorFlow (TF) sub-graph for CLASSIFY kernel.
                - tree-label vs. true label
                This multiclass classifer compares each row of a given Tree to the known solution, comparing predicted labels 
                generated by plagih GP against the true class labels. This method is able to work with any number of class 
                labels, from 2 to n. The left-most bin includes -inf. The right-most bin includes +inf. Those inbetween are 
                by default confined to the spacing of 1.0 each, as defined by:

                    (solution - 1) < result <= solution

                The skew adjusts the boundaries of the bins such that they fall on both the negative and positive sides of the 
                origin. At the time of this writing, an odd number of class labels will generate an extra bin on the positive 
                side of origin as it has not yet been determined the effect of enabling the middle bin to include both a 
                negative and positive result.
                """

                if len(actions) > 1:
                    printpl('e', 'TODO multidimensional input. To be done, there is no solution yet.')

                if get_pred_labels:
                    pred_labels = tf.map_fn(eval_tf_classify_labels_map, tf_result, dtype=(tf.int32, tf.string), swap_memory=True)

                skew = (class_labels / 2) - 1

                rule11 = tf.equal(solution, 0)
                rule12 = tf.less_equal(tf_result, 0 - skew)
                rule13 = tf.logical_and(rule11, rule12)

                rule21 = tf.equal(solution, class_labels - 1)
                rule22 = tf.greater(tf_result, solution - 1 - skew)
                rule23 = tf.logical_and(rule21, rule22)

                rule31 = tf.less(solution - 1 - skew, tf_result)
                rule32 = tf.less_equal(tf_result, solution - skew)
                rule33 = tf.logical_and(rule31, rule32)

                pairwise_fitness = tf.dtypes.cast(tf.logical_or(tf.logical_or(rule13, rule23), rule33), tf.int32)

            elif kernel == 'r':  # REGRESSION kernel

                """
                A very, very basic REGRESSION kernel which is not designed to perform well in the real world. It requires
                that you raise the minimum node count to keep it from converging on the c1 of '1'. Consider writing or 
                integrating a more sophisticated kernel.
                """

                pairwise_fitness = tf.abs(solution - tf_result)

            elif kernel == 'm':  # MATCH kernel

                """
                This is used for demonstration purposes only.
                """

                # pairwise_fitness = tf.dtypes.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
                RTOL, ATOL = 1e-05, 1e-08  # fixes above issue by checking if a float c1 lies within a range of values
                pairwise_fitness = tf.dtypes.cast(tf.less_equal(tf.abs(solution - tf_result), ATOL + RTOL * tf.abs(tf_result)), tf.int32)

            # elif kernel == '[other]': # use others as a template

            else:
                raise Exception('Kernel type is wrong or missing. You entered {}'.format(kernel))

            fitness = tf.reduce_sum(pairwise_fitness)

            # Process TF graph and collect the results
            tf_result, pred_labels, solution, fitness, pairwise_fitness = sess.run([tf_result, pred_labels, solution, fitness, pairwise_fitness])

    # todo delete this
    # printpl('c', ('arity', fitness_compare_better(fitness, origin['fitness_train'])), 'Fitness was better than original fitness')
    # if fitness_compare_better(fitness, origin['fitness_train']):
    #     print('Fitness was better than original fitness:', fitness, ' better than:', origin['fitness_train'])

    return {'result': tf_result, 'pred_labels': pred_labels, 'solution': solution, 'fitness': float(fitness),  # this was changed
            'pairwise_fitness': pairwise_fitness, 'old_fitness': float(fitness)}


def eval_tf_ast_expr(expr, tensors):
    """
    Extract expression tree from the string algo_sym and transform into TensorFlow (TF) graph.

    """
    # print('Current expr:', expr)  # importantprint for debugging failed expressions
    tree = ast.parse(expr, mode='eval').body

    # TODO diesen try-except block entfernen
    debug_warnings = str(expr)
    try:
        return eval_tf_expr_graph(tree, tensors)
    except:
        return fitness_dummy_get()


def eval_tf_expr_graph(node, tensors):
    """
    Recursively transforms parsed expression tree into TensorFlow (TF) graph.

    """

    if isinstance(node, ast.Name):  # <tensor_name>
        return tensors[node.id]

    elif isinstance(node, ast.Num):  # <number>
        shape = tensors[list(tensors.keys())[0]].get_shape()
        return tf.constant(node.n, shape=shape, dtype=tf.float32)

    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>, e.g., x + y
        return ast_tensor_dict[type(node.op)](
            eval_tf_expr_graph(node.left, tensors),
            eval_tf_expr_graph(node.right, tensors))

    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return ast_tensor_dict[type(node.op)](
            eval_tf_expr_graph(node.operand, tensors))

    elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)
        # special case: If-then-else
        if node.func.id == 'Ifte':
            return ast_tensor_dict[node.func.id](tf.dtypes.cast(
                eval_tf_expr_graph(node.args[0], tensors), tf.bool),
                eval_tf_expr_graph(node.args[1], tensors),
                eval_tf_expr_graph(node.args[2], tensors))
        # # This was here for Min and Max. complicated stuff, did not work.
        # if node.func.id in functions_multiparam_dict:
        #     return operator_dict[node.func.id]([eval_tf_expr_graph(arg, tensors) for arg in node.args])

        if node.func.id == 'Ftob':
            printpl('i', 'float was converted to bool in tensorflow')
            return tf.dtypes.cast(
                *[eval_tf_expr_graph(arg, tensors) for arg in node.args], dtype=tf.bool)
        elif node.func.id == 'Btof':
            return tf.dtypes.cast(
                *[eval_tf_expr_graph(arg, tensors) for arg in node.args], dtype=tf.float32)

        if len(node.args) > 2:
            printpl('e', 'This has more than 2 args?', str(node.func.id))
        else:
            try:
                return ast_tensor_dict[node.func.id](*[eval_tf_expr_graph(arg, tensors) for arg in node.args])
            except Exception as ex:
                printpl('w', 'debug warning:', debug_warnings)
                printpl('e', 'node.func.id caused an exception, type:\n', ex,
                        '\nnode.func.id:\n', node.func.id,
                        '\nnode.args:', str(node.args),
                        '\nand expression:\n', debug_warnings)

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        return eval_tf_chain_bool(node.values, ast_tensor_dict[type(node.op)], tensors)

    elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
        return eval_tf_chain_compare([node.left] + node.comparators, node.ops, tensors)

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        return tf.constant(node.value)

    else:
        raise TypeError(node)


def eval_tf_chain_bool(values, operation, tensors):
    """
    Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.

    """

    x = tf.dtypes.cast(eval_tf_expr_graph(values[0], tensors), tf.bool)
    if len(values) > 1:
        return operation(x, eval_tf_chain_bool(values[1:], operation, tensors))
    else:
        return x


def eval_tf_chain_compare(comparators, ops, tensors):
    """
    Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

    Called by: fitness_node_parse

    Arguments required: comparators, ops, tensors
    """

    x = eval_tf_expr_graph(comparators[0], tensors)
    y = eval_tf_expr_graph(comparators[1], tensors)

    if len(comparators) > 2:
        return tf.logical_and(ast_tensor_dict[type(ops[0])](x, y), eval_tf_chain_compare(comparators[1:], ops[1:], tensors))
    else:
        return ast_tensor_dict[type(ops[0])](x, y)
    # sfeh idea: note: we have to convert all values to the action space if not discrete


def eval_tf_classify_labels_map(result):
    """
    For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
    the quantity of true class labels provided in the data .csv. Outputs an array of tuples containing the predicted
    labels based upon the result and corresponding boolean condition triggered.

    For comparison, the original (pre-TensorFlow) cod follows:

        skew = (class_labels / 2) - 1 # '-1' keeps a binary classification splitting over the origin
        if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
        elif solution == class_labels - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
        elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
        else: fitness = 0 # no class match

    Called by: fitness_eval

    Arguments required: result
    """

    skew = (class_labels / 2) - 1
    label_rules = {class_labels - 1: (
        tf.constant(class_labels - 1), tf.constant(' > {}'.format(class_labels - 2 - skew)))}

    for class_label in range(class_labels - 2, 0, -1):
        cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
        label_rules[class_label] = tf.cond(cond, lambda: (
            tf.constant(class_label), tf.constant(' <= {}'.format(class_label - skew))),
                                           lambda: label_rules[class_label + 1])

    pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))),
                         lambda: label_rules[1])

    return pred_label


def op_label_get_terminal(node_label):
    """
    return terminal or function according to the label
    """

    if node_label in op_xtype_dict:
        return 'func'
    else:
        return 'term'


def xtype_xtype_get_terminal(node_xtype):
    """
    Define a single Terminal (variable extracted from the top row of the associated TRAINING data)

    """

    node[TRn_type] = 'term'
    node[TRn_label] = xtype_choose_term(node_xtype)  # get a terminal
    node[TRn_arity] = 0

    return


def evolve_func_get_func(label, mode='same_arity_same_type'):
    """
    returns a function for a function in point mutation
    This only accepts functions as inputs. (point mutation)
    No need to handle terminals
    """

    arity = op[label]['arity']
    xtype = op[label]['xtype']

    if mode == 'same_arity_same_type':

        if xtype == 'f2f':
            return np.random.choice(op_type_arity_array[f2f][arity])
        elif xtype == 'f2b':
            return np.random.choice(op_type_arity_array[f2b][arity])
        elif xtype == 'b2b':
            return np.random.choice(op_type_arity_array[b2b][arity])
        elif xtype == 'b2f':
            return np.random.choice(op_type_arity_array[b2f][arity])
        elif xtype == 'b2f2f':
            return np.random.choice(op_type_arity_array[b2f2f][arity])  # sfeh okay that does not make sense tbh
        else:
            printpl('e', 'Function was not found in function_types_dict', xtype)
            raise

    else:
        printpl('e', 'Mode not known: ', mode)


def xtype_choose_func(xtype):
    """
    This fills in a function that fits the type of the function/terminal before.
    terminal  '2f' -> '_2f', arity
    function 'f2f' -> '_2f', arity
    function 'b2f2f' -> '_2f', arity
    > ->
    """
    if '2f' in xtype:
        label = np.random.choice(xtype_func_dict['2f'])
    elif '2b' in xtype:
        label = np.random.choice(xtype_func_dict['2b'])
    else:
        printpl('e', 'Warning: Function was not found in function_types_dict', xtype)
        raise

    return label, op[str(label)]['arity']


def xtype_node_get_xtype(node_label, node_type):
    """
    input: (+, 'func')
    'term' or 'func'
    """

    if node_type == 'term':
        if 'True' in node_label or 'False' in node_label:
            return '2b'
        elif 'observation' in node_label:
            term_position = variables_dict['all'].index(node_label)
            return op_xtype_dict[variables_dict['types'][term_position]]
        elif 'action' in node_label:
            term_position = actions.index(node_label)
            return op_xtype_dict[action_types[term_position]]
        else:  # only 'float' left
            return '2f'
    elif node_type == 'func':
        return op_xtype_dict[node_label]
    else:
        printpl('e', 'This node_type is not known', node_type)
        raise


def xtype_label_get_xtype(label, node_type=''):
    """
    returns xtype for a label
    todo runtime compared to xtype_node_get_xtype?
    """
    if not node_type:
        node_type = op_label_get_terminal(label)

    node_xtype = xtype_node_get_xtype(label, node_type)

    if node_type == 'term':
        if 'True' in label or 'False' in label:
            return '2b'
        elif 'observation' in label:
            term_position = variables_dict['all'].index(label)
            return op_xtype_dict[variables_dict['types'][term_position]]
        elif 'action' in label:
            term_position = actions.index(label)
            return op_xtype_dict[action_types[term_position]]
        else:  # only 'float' left
            return '2f'
    elif node_type == 'func':
        return op_xtype_dict[label]
    else:
        printpl('e', 'This node_type is not known', node_type)
        raise

    return node_xtype


def xtype_choose_term(node_xtype):
    """
    Returns a terminal of xtype.

    function: f2b -> 2b needed
    terminal:  2f -> 2f needed
    --> check if it is function, aka _2f
    --> check if it is terminal, aka f2

    Modes:
    var_and_const: return randomly (50:50) a variable or a constant
    terminal_only: return                  a variable
    Todo Introduce constants-mode, where the user can give constant types (similar to functions)?

    input options: f2f, f2b, b2f, b2b, f2b2b, 2f, 2b
    """

    # node_xtype == '2f' or 'f2' in node_xtype:
    if '2f' in node_xtype:
        terminals_correct = variables_dict['float']
        the_type = 'float'
    elif '2b' in node_xtype:
        terminals_correct = variables_dict['bool']
        the_type = 'bool'
    else:
        printpl('e', 'Probably, you have to check if your "function" is actually a terminal. xtype', node_xtype)
        raise

    if np.random.choice(['var', 'const']) == 'var':  # our choice is variable
        if terminals_correct:  # Is there an entry in the list?
            return np.random.choice(terminals_correct)  # ...so we return one
    return tree_build_type_constant_get(term_type=the_type)  # otherwise: constant (There are always constants :P)
    #
    # try:
    # except ValueError:
    #     printpl('w', 'Should not happen. Did not find a terminal. Made up a ' + the_type + ' constant.')
    #     return tree_build_type_constant_get(term_type=the_type)


def xtype_outcome_equi_test(a_xtype, b_xtype):
    """
    Dummy. Returns, whether two xtypes are equal
    """
    return a_xtype in b_xtype or b_xtype in a_xtype


def xtype_get_converter(a_xtype, b_xtype):
    """
    convert b-to-a dummy
    """
    if '2b' in a_xtype and '2f' in b_xtype:
        return 'Ftob'
    if '2f' in a_xtype and '2b' in b_xtype:
        return 'Btof'
    else:
        printpl('e', 'One of those two cases should happen', a_xtype, b_xtype)
        raise


def node_function_select(node):
    """
    Returns a function with the same outcome

    """


def tree_add_node_onlylabel(tree, label):
    tree = np.append(tree, [[''], [''], [''], [''],
                            [''], [''], [''], [''],
                            [''], [''], [''], [''],
                            [''], [''], ['']], 1)
    return tree


def tree_add_node_fromnode(tree, node):
    tree = np.append(tree, [[''], [''], [''],
                            [node[TRn_id]],
                            [node[TRn_depth]],
                            [node[TRn_type]],
                            [node[TRn_label]],
                            [node[TRn_parent]],
                            [node[TRn_arity]],
                            [node[TRn_c1]],
                            [node[TRn_c2]],
                            [node[TRn_c3]],
                            [''], [''], ['']], 1)
    return tree


def tree_init_first_column():
    tree = np.array(
        [['tree_id'],
         ['tree_type'],
         ['tree_depth_base'],
         ['node_id'],
         ['node_depth'],
         ['node_type'],
         ['node_label'],
         ['node_parent'],
         ['node_arity'],
         ['node_c1'],
         ['node_c2'],
         ['node_c3'],
         ['fitness'],
         ['node_modify'],
         ['parsimony']])

    return tree


def tree_init_core(node_dummys):
    """
    returns an empty tree with an amount of nodes, auto fills
    """
    tree = np.zeros((TRn_um_lines, node_dummys), dtype=np.dtype('U12'))  # U12: longest is observation1
    # tree = np.concatenate((tree, empty_node_array), axis=1)

    return tree


def tree_core_depth(tree, parent_list=None):
    """
    Automatically filly node depth
    - Tree needs:
        - c1, c2, c3 filled
    """

    if not parent_list:
        parent_list = tree_row_int(tree, TRn_c1)

    tree[TRn_depth][0] = 0  # the root is always here

    for my_id, parent in enumerate(parent_list):

        child_depth = int(tree[TRn_depth][my_id]) + 1
        c1 = tree[TRn_c1][my_id]
        c2 = tree[TRn_c2][my_id]
        c3 = tree[TRn_c3][my_id]
        for c in [c1, c2, c3]:
            if c != '':
                tree[TRn_depth][int(c)] = child_depth
    return tree


def tree_core_parents(tree, arity_list=None):
    """
    Automatically filly the node_parent of a tree
    - arity list in tree or
    """
    if not arity_list:
        arity_list = [int(x) for x in tree[TRn_arity]]

    parent_list = [-1]
    for i, arity in enumerate(arity_list):
        parent_list.extend([i] * arity)
        tree[TRn_parent][i] = parent_list[i]
    return tree, parent_list


def tree_core_c(tree, parent_list=None):
    """
    automaticalls fills c1, c2, c3 for each node
    Needed: node_parent
    """
    if not parent_list:
        parent_list = tree_row_int(tree, TRn_parent)

    c_iter = 0
    last_parent = -1

    # parent_list [-1, 0, 0, 0, 1, 1]
    for i, val in enumerate(parent_list):
        my_id = i  # + 1  # nodeone
        parent_id = val  # - 1  # nodeone
        if val >= 0:  # +1 nodeone

            if val == last_parent:
                c_iter += 1
            else:
                last_parent = val
                c_iter = 0
            tree[TRn_c1 + c_iter][parent_id] = my_id
    return tree


def tree_row_int(tree, row_id):
    row = []
    for x in tree[row_id]:
        row.append(int(x))
    return row


def tree_core_insert(tree, row_id, row):
    for i, x in enumerate(row):
        tree[row_id][i] = x
    return tree


def tree_plusnode(tree, add_or_sub=1, firstrow='1'):
    """
    returns a tree where the nodes start at 1 instead of 0
    """
    nodes = len(tree[1])

    for row_id in [TRn_id, TRn_c1, TRn_c2, TRn_c3, TRn_parent]:
        for value in range(firstrow, nodes):
            if tree[row_id][value] != '':
                tree[row_id][value] = int(tree[row_id][value]) + add_or_sub
    tree[TRn_parent][firstrow] = -1
    return tree


def tree_from_labels(label_list, arity_list, type_list):
    # tree = tree_init_first_column()
    size = len(label_list)
    tree = tree_init_core(size)

    tree = tree_core_insert(tree, TRn_id, [x for x in range(0, size)])
    # tree = tree_core_insert(tree, TRn_id, [x for x in range(1, size + 1)])
    tree = tree_core_insert(tree, TRn_label, label_list)
    tree = tree_core_insert(tree, TRn_arity, arity_list)
    tree = tree_core_insert(tree, TRn_type, type_list)

    tree, parent_list = tree_core_parents(tree)
    tree = tree_core_c(tree)
    tree = tree_core_depth(tree, parent_list)

    return tree


def test_cases(number):
    if number == 0:
        label_list = ['Ifte', '<', '0', '2', 'observation1', '0']
        arity_list = [3, 2, 0, 0, 0, 0]
        type_list = ['func', 'func', 'term', 'term', 'term', 'term']
    elif number == 1:
        label_list = ['+', '+', '+', '+', '1', '2', '3', '4', '5']
        arity_list = [2, 2, 2, 2, 0, 0, 0, 0, 0]
        type_list = ['func', 'func', 'func', 'func', 'term', 'term', 'term', 'term', 'term']
    elif number == 2:
        label_list = ['+', '+', '+', '0', '1', 'Ifte', '2', '3', '+', '4', '5', '6']  # 12 nodes
        arity_list = [2, 2, 2, 0, 0, 3, 0, 0, 2, 0, 0, 0]
        type_list = ['func', 'func', 'func', 'term', 'term', 'func', 'term', 'term', 'func', 'term', 'term', 'term']
    else:
        label_list = ['0']
        arity_list = [0]
        type_list = ['term']
        solution = np.array([['tree_id', '', '', '', '', '', '', '', '', '', '', '', ''],
                             ['tree_type', '', '', '', '', '', '', '', '', '', '', '', ''],
                             ['tree_depth_base', '', '', '', '', '', '', '', '', '', '', '', ''],
                             ['node_id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
                             ['node_depth', '0', '1', '1', '2', '2', '2', '2', '3', '3', '3', '4', '4'],
                             ['node_type', 'func', 'func', 'func', 'term', 'term', 'func', 'term', 'term', 'func', 'term', 'term', 'term'],
                             ['node_label', '+', '+', '+', '0', '1', 'Ifte', '2', '3', '+', '4', '5', '6'],
                             ['node_parent', '-1', '0', '0', '1', '1', '2', '2', '5', '5', '5', '8', '8'],
                             ['node_arity', '2', '2', '2', '0', '0', '3', '0', '0', '2', '0', '0', '0'],
                             ['node_c1', '1', '3', '5', '', '', '7', '', '', '10', '', '', ''],
                             ['node_c2', '2', '4', '6', '', '', '8', '', '', '11', '', '', ''],
                             ['node_c3', '', '', '', '', '', '9', '', '', '', '', '', ''],
                             ['fitness', '', '', '', '', '', '', '', '', '', '', '', ''],
                             ['node_modify', '', '', '', '', '', '', '', '', '', '', '', ''],
                             ['parsimony', '', '', '', '', '', '', '', '', '', '', '', '']])
    return label_list, arity_list, type_list


label_list, arity_list, type_list = test_cases(2)
tree1 = tree_from_labels(label_list, arity_list, type_list)

tree2 = tree_init_first_column()
tree2 = np.concatenate((tree2, tree1), axis=1)
tree2 = tree_plusnode(tree2, add_or_sub=1, firstrow=1)
print(tree2)

