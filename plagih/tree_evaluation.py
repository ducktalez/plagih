"""
tree_evaluation.py

"""
from plagih.node_labels import op_dict
from plagih.util import *

import tensorflow
import ast


def ast_expr_to(node, tensors=None, build=None):
    """
    Returns (recursively) a (tensorflow) graph from a (raw or sympified) math expression.
    please use by calling labels_from_graphlist()

    Used to be for tensorflow only, but was modified to save 'sympified' trees.

    One of [tensors, prnt, build] must be set
    -> tensors: Creates a tensorflow graph for evaluation
    -> prnt: creates a string expression of the fintree (I think I tried this before 'build' worked)
    -> build: creates a nested nlabel-list, e.g. a+(b/c) -> [+, [a], [/, [b, c]]]]
    """

    # Arity 0
    if isinstance(node, ast.Name):  # <tensor_name>
        if build:
            # return node.id
            return [node.id]
        else:
            return tensors[node.id]

    elif isinstance(node, ast.Num):  # <number>
        if build:
            # return node.n
            return [node.n]
        else:
            try:
                shape = tensors[list(tensors.keys())[0]].get_shape()  # sfeh:workaround
                return tensorflow.constant(node.n, dtype=tensorflow.float32, shape=shape)  # , shape=shape
                # ^ValueError: Shapes must be equal rank, but are 0 and 1 for 'Select_1' (op: 'Select') with input shapes
                # => sfeh: in some tf-versions, the constants have to match their shape. data has shape [3423, 0]
                # constants have shape []
            except Exception as ex:
                # Error: # Could not append fintree to population because: eval-ex: list index out of range
                # # ^Problem occurs, when no real constants/variables are in the tree
                # #   =>fintree: [Ifte, [<, [0.0], [0.0]], [0.0], [2.0]]
                return tensorflow.constant(node.n, dtype=tensorflow.float32)  # , shape=shape

    elif isinstance(node, ast.NameConstant):  # <True/False> e.g., <True>
        if build:
            return [node.value]
        else:
            return tensorflow.constant(node.value)
    #
    # Arity 1
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., sin(1), -1
        if build:
            if type(node.op) == ast.USub:  # workaround for ~-problem, sfeh
                if isinstance(node.operand, (ast.Name, ast.Num, ast.NameConstant)):  # 'cartVel', 5, e
                    return [f'-{ast_expr_to(node.operand, build=True)[0]}']  # (!! "un-listed") ['-cartVel'], [-5], [-e]
                else:
                    return ['Usub', ast_expr_to(node.operand, build=True)]  # -> ['Usub', ast()]

            return [op_dict[type(node.op)].nlabel, ast_expr_to(node.operand, build=True)]  # -> ['sin', ast()]
        else:
            return op_dict[type(node.op)].tflow(ast_expr_to(node.operand, tensors=tensors))

    # Arity 2
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>, e.g., (x + y)
        if build:
            return [op_dict[type(node.op)].nlabel,
                    ast_expr_to(node.left, build=True), ast_expr_to(node.right, build=True)]  # e. g. ['+', a, b]
        else:
            return op_dict[type(node.op)].tflow(
                ast_expr_to(node.left, tensors=tensors),
                ast_expr_to(node.right, tensors=tensors))

    elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
        if build:
            values = node.values
            if len(values) == 2:
                return [op_dict[type(node.op)].nlabel,
                        ast_expr_to(values[0], build=True),
                        ast_expr_to(values[0], build=True)]  # -> ['and', ast(), ast()]
            elif len(values) == 1:
                raise  # sfeh:
                # return [op_dict[type(node.op)].nlabel,
                #         ast_expr_to(values[0], build=True)]  # -> ['not', ast()]
            else:
                raise
                # return ast_expr_to(values[0], build=True)
            # return ast_chain_bool(node.values, op_dict[type(node.op)].nlabel, build=True)
        else:
            return ast_chain_bool(node.values, op_dict[type(node.op)].tflow, tensors=tensors)

    elif isinstance(node, ast.Compare):  # ast.Compare(left, ops, comparators)
        if build:
            # NO CHAINRULE YET
            return [op_dict[type(node.ops[0])].nlabel,
                    ast_expr_to(node.left, build=True), ast_expr_to(node.comparators[0], build=True)]
            # return ast_chain_compare([node.left] + node.comparators, node.ops, build=True)
        else:
            return ast_chain_compare([node.left] + node.comparators, node.ops, tensors=tensors)

    # Arity x, all custom functions
    elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x) -> or if(a, b, c) -> or Ftob(a)

        if node.func.id == 'Ifte':
            if build:
                return ['Ifte',
                        ast_expr_to(node.args[0], build=True),
                        ast_expr_to(node.args[1], build=True),
                        ast_expr_to(node.args[2], build=True)]
                # return ['Ifte',
                #         [ast_expr_to(node.args[0], build=True),
                #          ast_expr_to(node.args[1], build=True),
                #          ast_expr_to(node.args[2], build=True)]]
            else:
                return op_dict[node.func.id].tflow(tensorflow.dtypes.cast(
                    ast_expr_to(node.args[0], tensors=tensors), tensorflow.bool),
                    ast_expr_to(node.args[1], tensors=tensors),
                    ast_expr_to(node.args[2], tensors=tensors))

        elif len(node.args) <= 2:
            if build:
                if len(node.args) == 1:
                    return [op_dict[node.func.id].nlabel,
                            ast_expr_to(node.args[0], build=True)]
                    # return [op_dict[node.func.id],  # sfeh:check: remove .nlabel?
                    #         [ast_expr_to(node.args[0], build=True)]]
                elif len(node.args) == 2:
                    return [op_dict[node.func.id].nlabel,
                            ast_expr_to(node.args[0], build=True),
                            ast_expr_to(node.args[1], build=True)]
                else:
                    raise Exception('This arity is not supported')
            else:
                return op_dict[node.func.id].tflow(*[ast_expr_to(arg, tensors=tensors) for arg in node.args])

        else:
            raise Exception('Failed to identify the function. {}'.format(type(node)))
    else:
        raise TypeError('Node type could not be handeled in ast-evaluation: {}'.format(node))


def ast_chain_bool(values, operation, tensors=None, build=False):
    """
    Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.
        a & b
    --> values[0] operation values[1]
    """
    if build:
        raise
        # x = ast_expr_to(values[0], build=True)
        # if len(values) == 2:
        #     return [operation, values[0], values[1]]  # -> ['and', ast(), ast()]
        # elif len(values) == 1:
        #     # return [x]
        #     # raise  # TODO debug
        #     return x  # -> ['not', ast()]
        # else:
        #     raise
    else:
        x = tensorflow.dtypes.cast(ast_expr_to(values[0], tensors=tensors), tensorflow.bool)
        if len(values) > 1:
            return operation(x, ast_chain_bool(values[1:], operation, tensors=tensors))
        else:
            return x


def ast_chain_compare(comparators, ops, tensors=None, build=False):
    """
    Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

    """
    x = ast_expr_to(comparators[0], tensors=tensors, build=build)
    y = ast_expr_to(comparators[1], tensors=tensors, build=build)

    if len(comparators) > 2:
        print_e('This is usually not used, and-concatenation of multiple chain compares. sfeh, bring this back?')
        return tensorflow.logical_and(op_dict[type(ops[0])].tflow(x, y),
                                      ast_chain_compare(comparators[1:], ops[1:], tensors=tensors))
    else:
        if build:
            raise  # delete this
            # return [op_dict[type(ops[0])].nlabel, [x, y]]
        else:
            return op_dict[type(ops[0])].tflow(x, y)


def labels_from_nestedexpr(labels_nested_list, result_accum):
    """
    Returns a label list from the nested list which ast_expr_to() created
    [+, [a], [/, [b, c]]]]  -> [+, a, /, b, c]
    """

    for x in labels_nested_list:  # all elements, that are not lists themselves
        if type(x) is not list:
            x = str(x)  # labels must be string!
            result_accum.append(x)

    only_lists = [x for x in labels_nested_list if (type(x) == list)]
    if only_lists:
        from itertools import chain
        lists_removed = list(chain(*only_lists))
        result_accum = labels_from_nestedexpr(lists_removed, result_accum)

    return result_accum


def ast_convert_from_expr(expr, tensors=None, build=None):
    """
    Starts the recursive ast-analysis of the expression

    Extract expression fintree from the string algo_sym.
    Please provide ONE of the following if you want to get...
    - tensorflow-graph: All variables (observation0, ...) as tensors.
    - build: True
    More information in ast_expr_to()
    """

    ast_tree = ast.parse(expr, mode='eval').body
    graph = ast_expr_to(ast_tree, tensors=tensors, build=build)

    if build:
        graph = str(graph)  # sfeh... necessary to make string?

    return graph
