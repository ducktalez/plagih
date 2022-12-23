"""
Enriching the python-core 'sympy'. Sympy is used to unify and reduce functions to their most basic form.
E. g., it reduces '1+1+a' to 'a+2' and thus saves much computation power.

- implementing missing functions in sympify, e. g. 'if a then b else c'.
- All number-related functions must have set
    is_real = True
    otherwise: '1 < BinaryMax(2, Ifte(1 < a, 1, 1))' will crash. (< operators only work on non-complex - aka real numbers)
    check for is_number if required.

- Classes must currently have the exact same name as their occurance (Ifte -> Ifte, not ifte or so)
    This is because when None is returned, the class name gets replaced at the function. could be solved, but why though :P
The following line is an honorable mention for myself; it was required for
## if ((a == True or a == False) and (b == True or b == False)) == (sympify(a).is_Boolean and sympify(b).is_Boolean):

Useful information:
- These variables are set for every sympy object and thus can be tested, e.g. a.is_Boolean
    # To be overridden with True in the appropriate subclasses

    #sfeh:open combine the nodes with the sympy shizzle
    sfeh xxx input variables as locals? Provide information such as real, integer, positive, range/interval?
    sfeh:open this is probably the reason for the capitalized class names in sympy: return eval(self, a)
    sfeh: I think we should get rid of sympy in the long term. A lot of problems are related to sympy.

    sfeh:sypyunification errors:
        - 'a and b', 'b and a'
        - 'And(a<2, a < 5)'
        - sympy.simplify('sign(-a)') -> -sign(a)

    todo sympy facttor (up/downfactor), so it adds stuff together
    sfeh:discus simplify/unify
"""

import os
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import re

import numpy as np
import sympy

tf.compat.v1.enable_eager_execution()


class NodeLabel:
    """
    Kind of abstract class; Dummy-node that holds a nlabel
    sfeh:discussion
    """
    nlabel = None
    arity = None
    xtype = None
    tflow = None
    insym = None

    expr_sym = None  # todo: does zis

    def __str__(self):
        return self.nlabel

    def mutate_self_filter(self, *args, **kwargs):
        """
         as default, return own index
         sfeh:move to tree
        """
        pass


class Operator(NodeLabel, sympy.Function):
    """
    operator nodes (+, +, *, /, sin(), sign(), ...)
    inner nodes of a fintree
    """
    is_Function = True

    @classmethod
    def eval(cls, *args):
        return cls.insym(*args)  # just if no eval is implemented

    def tflambda(self):
        pass


class ChainableOperator(NodeLabel):
    """
    todo open, mapping operators are
    Add, Mult, Min, Max
    And, Or
    Actually not:
    """

    # @classmethod
    # def eval(self, *args):
    #     return  # args[self.arity:]  # todo

    def tflambda(self):
        pass


class NoSympyClass:
    pass


class Terminal(NodeLabel):
    """
    Terminal nodes are leaf nodes which can not have children. e.g.:
    - constants (e.g. 2.3)
    - observations (e.g. b, aka data input)
    - user-functions (sfeh:open)
    """
    arity = 0

    def mutate_self_filter(self, *args, **kwargs):
        # sfeh:? ...only for terminal nodes
        pass


class Constant(Terminal):
    arity = 0

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def mutate_self_filter(self, *args, **kwargs):
        pass


def observation_get_family_and_time(name, re_pattern='_\\d+$', none_return=None):
    """
    When an observation is known, return the family, the time and the SIGN!!
    sfeh:move this function somewhere else
    """
    core_expr = re.split(re_pattern, name)[0]
    if core_expr[0] == '-':
        core_expr = core_expr[1::]
        preexpr = '-'
    else:
        preexpr = ''
    try:
        re_search = re.search(re_pattern, name)  # re_search => ['_12']
        temp_diff = re_search[0].replace('_', '')  # (only) solution found (at [0]), e.g. '_14'. only keep the digits
        temp_diff = int(temp_diff)
    except Exception as ex:
        temp_diff = none_return
    return core_expr, temp_diff, preexpr


class Observation(Terminal):
    """
    todo sfeh discuss: labels should not have a sign (-pos)
        y though? -> just use a '-'-operator in an additional node.
        also: When reconstructing trees, the sign can appear in observations
    This was used to deal with negative labels
        self.name = nlabel if nlabel[0] != '-' else nlabel[1:]
    """

    def __init__(self, nlabel):
        self.nlabel = nlabel
        self.fam, self.timeindex, _ = observation_get_family_and_time(self.nlabel, none_return=None)
        self.xtype = (tuple([]), float)  # tuple([]) -> workaround for empty tuple
        self.expr_sym = self.nlabel
        self.index_minmax = None


class FloatConstant(Constant):
    """
    todo:discuss: how to deal with sign of observations?
    """
    arity = 0
    otype = float
    xtype = (tuple([]), float)

    def __init__(self, nlabel):
        self.nlabel = nlabel
        self.expr_sym = self.nlabel

    def mutate_self_filter(self, filter_type='gaussian_filter', *args, **kwargs):  # sfeh:open
        """

        """
        if filter_type == 'gaussian_filter':
            if np.random.choice(['v1', 'v2']) == 'v1' or self.nlabel == 0:
                constant = self.nlabel + np.random.normal(0, 0.1)  # sfeh better adjustments?
            else:
                constant = np.random.normal(self.nlabel, 0.1)  # sfeh better adjustments?
            self.nlabel = round(constant, PRECISION)  # sfeh:discussion be careful, might create zero sometimes


class BoolConstant(Constant):
    """
    True/False
    """
    xtype = (tuple([]), bool)
    tf_type = tf.bool

    def __init__(self, expr):
        # super().__init__(nlabel)
        self.nlabel = expr
        self.expr_sym = str(self.nlabel)

    def mutate_self_filter(self, *args, **kwargs):
        """
        sfeh: filtering these is kind of nonsense
        """
        pass


class BinaryAdd(Operator):
    nlabel = 'BinaryAdd'
    arity = 2
    tflow = tf.add
    expr_sym = '({} + {})'  # todo vs. Add()
    insym = sympy.Add
    xtype = (tuple([float, float]), float)

    nargs = 2
    is_real = True

    #
    # def __init__(self, *args, **kwargs):
    #     pass
    #     # super().__init__(*args, **kwargs)

    def backprop(self):
        """
        #:      count >, count <, count =
        #me:    wenn Abweichung am höchsten von allen sum-knoten
        c:      sum(alle Abweichungen) -> schlimmster +-child
        propagate-down: y - ^y, wenn 5 zu hoch -> -5 nach unten
        erst normieren? Also, die avg. Abweichung abziehen?
        """
        pass


class Subtract(Operator):
    """

    """
    nlabel = 'Sub'
    arity = 2
    tflow = tf.subtract
    expr_sym = '({}-{})'
    insym = sympy.Add
    xtype = (tuple([float, float]), float)

    def propagate_down(self):
        """
        defprop:
        # propagation: Anzahl
        propagate-down: negativ-Fehler (also summe, bl,os negativ)
        """
        pass

    @classmethod
    def eval(cls, a, b):
        return a - b  # sfeh check if its sympied


class BinaryMultiply(Operator):
    """

    """
    nlabel = 'BinaryMultiply'
    arity = 2
    tflow = tf.multiply
    expr_sym = '({} * {})'
    insym = sympy.Mul
    xtype = (tuple([float, float]), float)

    def propagate_down(self):
        """
        Idee: Ein weiterer Faktor bestimmt, wie weit man am Ziel vorbei ist.
            Beispielsweise Ziel: 20, aber Ergebnis war 10 -> Lösungsfaktor ist 20/10=2.
            Nun gibt man den eigenen Wert*2 nach unten als Ziel zurück.
            Falls selbst=0, gib differenz zwischen Ziel und 0 nach unten (also: Ziel)

        wenn 0: immer selbst runterpropagieren, wenn anderer auch 0: beide 0.5
        defprop:
        # propagation:
        propagate-down error: (y-^Y)-
        propagate-down itsme: (y-^Y)-
        """
        pass


# class Divide_no_nan(Operator):
#     """
#     # Division: SAFE division by zero!
#     -->tf.math.divide_no_nan -->pycode a/b --> div(a,b) !!pycode requires div_safe() implemented
#     sfeh: is it okay to display this as '/'?
#     xxxx optional to use high value instead of tf-eval to 1
#     """
#     nlabel = 'Divide_no_nan'
#     # classname = 'Divide_no_nan'  # sfeh??
#     arity = 2
#     tflow = tf.math.divide_no_nan
#     expr_sym = 'Div_no_nan({}, {})'
#     insym = None
#     xtype = (tuple([float, float]), float)
#     @classmethod
#     def eval(self, a, b):
#         return a / b


class Div(Operator):
    """
    sfeh:xxx make this available, make a correct version of "Divide_no_nan"
    div
    """
    nlabel = 'Div'
    arity = 2
    tflow = tf.math.divide
    expr_sym = '({} / {})'
    insym = None
    xtype = (tuple([float, float]), float)

    @classmethod
    def eval(cls, a, b):
        return a / b  # is a sympy thing returned?

    def backprop(self):
        """
        Wie bei Mult: Ziel wird ausgerechnet, Faktor für sich selbst wird ausgerechnet. Nach unten propagieren.
        """
        pass


class Pow(Operator):
    """
    AKA Power
    ALERT: Power can create complex numbers, maybe you should use Powerounded
    inline-available
    """
    nlabel = 'Pow'
    arity = 2
    tflow = tf.pow
    expr_sym = '({} ** {})'  # **
    insym = sympy.Pow
    xtype = (tuple([float, float]), float)

    @classmethod
    def eval(self, a, b):
        # return a ** b
        return sympy.Pow(a, b)

    def backprop(self):
        """
        # KILL, wenn Lösung gar nicht erreicht werden kenn (wegen Definitionsbereich). Es ist dann halt einfach so.
        Idee: Ergebnis Differenz ist vermutlich manchmal sehr hoch.
            # Zähle, wie oft Basis & Exponent jeweils drüber/drunter liegen
            -> Nur den Exponenten beachten. Anzahl drüber/drunter im Vergleich zum besten Exponenten (der Ziel erreicht)
        """
        pass


class Powrounded(Operator, NoSympyClass):
    """
    ALERT: This Power Version does not round the results
    inline-available
    sfeh:xxx not yet used, make this available and rewrite Power above
    """
    nlabel = 'Powrounded'
    arity = 2
    tflow = tf.pow
    expr_sym = '({}**Round({}))'
    insym = None
    xtype = (tuple([float, float]), float)

    @classmethod
    def eval(cls, a, b):
        return sympy.Pow(a, sympy.N(b, 0))


class Abs(Operator):
    """

    """
    nlabel = 'Abs'
    arity = 1
    tflow = tf.abs
    expr_sym = 'Abs({})'
    insym = sympy.Abs
    xtype = (tuple([float]), float)

    def backprop(self):
        """
        """
        pass


class Sign(Operator):
    """
    sign
    """
    nlabel = 'sign'
    arity = 1
    tflow = tf.sign
    expr_sym = 'sign({})'  # todo 'sign({})' sympy.simplify('sign(-a)') -> -sign(a)
    insym = sympy.sign
    xtype = (tuple([float]), float)

    # @classmethod
    # def eval(cls, a):
    #     print('SIGN EVAL WAS THERE SYMPY???')
    #     # if a.is_real:
    #     #     return a ** 2  # sfeh:update
    #     # else:
    #     #     return None
    #     return sympy.sign(a)

    def _sympy_(self, *args):
        print('SIGN SYMPY')
        return self.eval(*args)


class Square(Operator):
    nlabel = 'Square'
    arity = 1
    tflow = tf.square
    expr_sym = 'Square({})'
    insym = None
    xtype = (tuple([float]), float)

    nargs = 1
    is_real = True

    @classmethod
    def eval(cls, a):
        # if a.is_real:
        #     return a ** 2  # sfeh:update
        # else:
        #     return None
        return sympy.Pow(a, 2)

    def _sympy_(self, *args):
        return self.eval(*args)


class Sqrt(Operator, NoSympyClass):
    nlabel = 'sqrt'
    arity = 1
    tflow = tf.sqrt
    expr_sym = 'sqrt({})'
    insym = sympy.sqrt
    xtype = (tuple([float]), float)


class Log(Operator):
    """
    sfeh: Log isactually Ln (base e)
    """
    nlabel = 'log'
    arity = 1
    tflow = tf.math.log
    expr_sym = 'log({})'
    insym = sympy.log
    xtype = (tuple([float]), float)


class Log1p(Operator, NoSympyClass):
    nlabel = 'log1p'
    arity = 1
    tflow = tf.math.log1p
    expr_sym = 'log1p({})'
    insym = None
    xtype = (tuple([float]), float)

    @classmethod
    def eval(cls, a):
        return sympy.log(a + 1)  # just if no eval is implemented


class Cos(Operator):
    nlabel = 'cos'
    arity = 1
    tflow = tf.cos
    expr_sym = 'cos({})'
    insym = sympy.cos
    xtype = (tuple([float]), float)


class Sin(Operator):
    nlabel = 'sin'
    arity = 1
    tflow = tf.sin
    expr_sym = 'sin({})'
    insym = sympy.sin
    xtype = (tuple([float]), float)


class Tan(Operator):
    nlabel = 'tan'
    # nlabel = 'tan'
    arity = 1
    tflow = tf.tan
    expr_sym = 'tan({})'
    insym = sympy.tan
    xtype = (tuple([float]), float)


class Acos(Operator):
    nlabel = 'acos'
    arity = 1
    tflow = tf.acos
    expr_sym = 'acos({})'
    insym = sympy.acos
    xtype = (tuple([float]), float)


class Asin(Operator):
    nlabel = 'asin'
    arity = 1
    tflow = tf.asin
    expr_sym = 'asin({})'
    insym = sympy.asin
    xtype = (tuple([float]), float)


class Atan(Operator):
    nlabel = 'atan'
    # nlabel = 'atan'
    arity = 1
    tflow = tf.atan
    expr_sym = 'atan({})'
    insym = sympy.atan
    xtype = (tuple([float]), float)


class Tanh(Operator):
    nlabel = 'tanh'
    arity = 1
    tflow = tf.tanh
    expr_sym = 'tanh({})'
    insym = sympy.tanh
    xtype = (tuple([float]), float)


class Sinh(Operator):
    nlabel = 'sinh'
    arity = 1
    tflow = tf.sinh  # sfeh sinh, asinh
    expr_sym = 'sinh({})'
    insym = sympy.sinh
    xtype = (tuple([float]), float)


class Cosh(Operator):
    nlabel = 'cosh'
    arity = 1
    tflow = tf.cosh  # sfeh acosh
    expr_sym = 'cosh({})'
    insym = sympy.cosh
    xtype = (tuple([float]), float)


class Xor(Operator):
    nlabel = 'Xor'
    arity = 2
    tflow = tf.math.logical_xor
    expr_sym = 'Xor({}, {})'
    insym = sympy.Xor
    xtype = (tuple([bool, bool]), bool)


class Not(Operator):
    nlabel = 'Not'
    arity = 1
    tflow = tf.logical_not
    expr_sym = '~({})'
    insym = sympy.Not
    xtype = (tuple([bool]), bool)

    nargs = 1


class BinaryNot(Operator):
    """
    Not (boolean)
    Problem was:
    - Not(a) evaluates to ~a
    - not(a<2) evaluates to nan
    """
    nlabel = 'BinaryNot'
    arity = 1
    tflow = tf.logical_not
    expr_sym = 'BinaryNot({})'
    insym = sympy.Not
    xtype = (tuple([bool]), bool)

    nargs = 1

    # @classmethod
    # def eval(cls, a):
    #     # if a.is_Boolean:  # sfeh sympify.is_xxx here seems dumb
    #     #     return not a
    #     return sympy.Not(a)

    def _sympy_(self, *args):
        print('BinaryNot SUCKS')
        return self.eval(*args)


class Eq(Operator):
    nlabel = 'Eq'
    arity = 2
    tflow = tf.equal
    expr_sym = '({} == {})'
    insym = sympy.Eq
    xtype = (tuple([float, float]), bool)


class MapxMultiply(Operator):
    """
    sfeh:reduceoperator
    """
    nlabel = 'Mul'
    arity = None
    tflow = tf.multiply
    expr_sym = '({} * {})'
    mapping_expr = '*'.split()
    insym = sympy.Mul
    xtype = (float, float)

    nargs = 2
    is_real = True

    def propagate_down(self):
        """
        Idee: Ein weiterer Faktor bestimmt, wie weit man am Ziel vorbei ist.
            Beispielsweise Ziel: 20, aber Ergebnis war 10 -> Lösungsfaktor ist 20/10=2.
            Nun gibt man den eigenen Wert*2 nach unten als Ziel zurück.
            Falls selbst=0, gib differenz zwischen Ziel und 0 nach unten (also: Ziel)

        wenn 0: immer selbst runterpropagieren, wenn anderer auch 0: beide 0.5
        defprop:
        # propagation:
        propagate-down error: (y-^Y)-
        propagate-down itsme: (y-^Y)-
        """
        pass


class BinaryMin(Operator):
    """
    Minimum function with arity-2.
    min() does not work (for now), as min(min()) get folded to min(), which leads to problems creating the tf-graph
    """
    # todo
    nlabel = 'BinaryMin'
    arity = 2
    tflow = tf.minimum
    expr_sym = 'BinaryMin({}, {})'
    insym = sympy.Min
    xtype = (tuple([float, float]), float)

    nargs = 2
    is_real = True

    # @classmethod
    # def eval(cls, a, b):
    #
    #     # # if (a < b) == True or (a < b) == False: # first solution
    #     #
    #     # if a.is_number and b.is_number:  # must be real for a comparison
    #     #     return a if a <= b else b
    #     # elif a == b:  # recent update for BinaryMin(b, b)
    #     #     return a
    #     # else:
    #     #     return
    #     return sympy.Min(a, b)

    def _sympy_(self, *args):
        print('binMin todo SUCKS')
        return self.eval(*args)


class BinaryMax(Operator):
    """

    """
    nlabel = 'BinaryMax'
    arity = 2
    tflow = tf.maximum
    expr_sym = 'BinaryMax({}, {})'
    insym = sympy.Max
    xtype = (tuple([float, float]), float)

    nargs = 2
    is_real = True

    # @classmethod
    # def eval(cls, a, b):
    #
    #     # # if (a < b) == True or (a < b) == False: # first solution, was working.
    #     #
    #     # if a.is_number and b.is_number:
    #     #     return a if a > b else b
    #     # elif a == b:  # recent update for BinaryMin(b, b)
    #     #     return a
    #     # else:
    #     #     return
    #     return sympy.Max(a, b)

    def _sympy_(self, *args):
        print('binMax SUCKS')
        return self.eval(*args)


class BinaryAnd(Operator):
    """

    """
    nlabel = 'BinaryAnd'
    arity = 2
    tflow = tf.logical_and
    expr_sym = '({} & {})'
    insym = sympy.And
    xtype = (tuple([bool, bool]), bool)

    nargs = 2

    is_Boolean = True

    # @classmethod
    # def eval(cls, *args):
    #
    #     # # if (a == True or a == False) and (b == True or b == False):
    #     # # sfeh xxx
    #     #
    #     # if a.is_Boolean and b.is_Boolean:
    #     #     return a and b
    #     # elif a == b:
    #     #     return a
    #     # else:
    #     #     return None
    #     return sympy.And(*args)

    def _sympy_(self, *args):
        print('BinAnd SUCKS')
        return self.eval(*args)


class BinaryOr(Operator):
    """
    """
    nlabel = 'BinaryOr'
    arity = 2
    tflow = tf.logical_or
    expr_sym = '({} | {})'
    insym = sympy.Or
    xtype = (tuple([bool, bool]), bool)

    nargs = 2

    # @classmethod
    # def eval(cls, *args):
    #
    #     # # if (a == True or a == False) and (b == True or b == False):  # this works guaranteed
    #     #
    #     # if a.is_Boolean and b.is_Boolean:
    #     #     return a and b
    #     # elif a == b:
    #     #     return a
    #     # else:
    #     #     return None
    #     sympy.Or(*args)

    def _sympy_(self, *args):
        return self.eval(*args)


class Ne(Operator):
    nlabel = 'Ne'
    arity = 2
    tflow = tf.not_equal
    expr_sym = '({} != {})'
    insym = sympy.Ne  # sympy.Unequality
    xtype = (tuple([bool, bool]), bool)


class Lt(Operator):
    nlabel = 'Lt'
    arity = 2
    tflow = tf.less
    expr_sym = '({} < {})'
    insym = sympy.Lt  # sympy.StrictLessThan
    xtype = (tuple([float, float]), bool)


class Le(Operator):
    nlabel = 'Le'
    arity = 2
    tflow = tf.less_equal
    expr_sym = '({} <= {})'
    insym = sympy.Le  # sympy.LessThan
    xtype = (tuple([float, float]), bool)


class Gt(Operator):
    nlabel = 'Gt'
    arity = 2
    tflow = tf.greater
    expr_sym = '({} > {})'
    insym = sympy.Gt  # sympy.StrictGreaterThan
    xtype = (tuple([float, float]), bool)


class Ge(Operator):
    nlabel = 'Ge'
    arity = 2
    tflow = tf.greater_equal
    expr_sym = '({} >= {})'
    insym = sympy.GreaterThan
    xtype = (tuple([float, float]), bool)


# class Usub(sympy.Function, Operator, NoSympyClass):
#     """
#     todo idea introduce negative labels as input?
#     """
#     nlabel = 'Usub'
#     arity = 1
#     tflow = tf.negative
#     expr_sym = '(-{})'
#     insym = None
#     xtype = (tuple([float]), float)
#
#     nargs = 1
#     is_Function = True
#     is_real = True
#
#     @classmethod
#     def eval(cls, a):
#         return -a  # sfeh
#
#     def _sympy_(self, *args):
#         return self.eval(*args)


class Ifte(Operator):
    """
    self-expert: opportunity_cost = best_vals - chosen_vals
    self-childs:
    condition: opportunity_cost = chosen_vals - best_vals
    a        : opportunity_cost = when_chosen(chosen_vals - best)
    b        : opportunity_cost = when_chosen(chosen_vals - best)
    #self.childs:
    condition: #correct_decisions
    a        : cond->a, #correct_decisions - #false_decisions
    b        : cond->b, #correct_decisions - #false_decisions
    --> If a node
    sfeh:open update to piecewise
    """
    nlabel = 'Ifte'
    arity = 3
    tflow = tf.where
    expr_sym = 'Ifte({}, {}, {})'
    insym = None
    xtype = (tuple([bool, float, float]), float)

    nargs = 3
    is_real = True

    @classmethod
    def eval(cls, a, b, c):
        # if a.is_Boolean:
        #     return b if a else c  # search for 'gotcha' in https://docs.sympy.org/latest/_modules/sympy/core/relational.html
        # else:
        #     return None  # returns None, is not further evaluated
        return sympy.Piecewise((b, a), (c, True))  # also available: sympy.piecewise_fold

    def _sympy_(self, *args):
        return self.eval(*args)


class MapxAnd(Operator):
    nlabel = 'And'
    arity = 2
    tflow = tf.logical_and
    expr_sym = '({} & {})'
    insym = sympy.And
    xtype = (tuple([bool, bool]), bool)


class MapxMax(Operator):
    nlabel = 'Max'
    arity = 2
    tflow = tf.maximum
    expr_sym = 'Max({}, {})'
    insym = sympy.Max
    xtype = (tuple([float, float]), float)


class MapxAdd(Operator):
    """

    """
    nlabel = 'Add'
    arity = 1
    tflow = tf.negative
    expr_sym = 'Add({})'
    insym = sympy.Add
    xtype = (float, float)

    nargs = None

    is_real = True


class MapxPiecewise(ChainableOperator):
    # todo all the function assumptions!!
    # todo must have a True-case
    nlabel = 'Piecewise'
    arity = None
    tflow = tf.where  # sfeh:open tf.cond https://stackoverflow.com/questions/45517940
    expr_sym = 'Piecewise({})'
    insym = sympy.Piecewise
    xtype = (float, float)  # todo

    # nargs = None  # sfeh:hmmm

    is_real = True


class MapxMin(Operator):
    nlabel = 'Min'
    arity = None  # sfeh:does not have arity?
    tflow = tf.minimum
    expr_sym = 'Min({}, {})'
    insym = sympy.Min
    xtype = (tuple([float, float]), float)


class MapxOr(Operator):
    nlabel = 'Or'
    arity = 2
    tflow = tf.logical_or
    expr_sym = '({} | {})'
    insym = sympy.Or
    xtype = (tuple([bool, bool]), bool)


class Round(Operator, sympy.Function, NoSympyClass):
    """
    sfeh:discussion this does only round to full numbers
    """
    nlabel = 'Round'
    arity = 1
    tflow = lambda a: tf.math.round(a, 1)
    insym = None  # lambda a: sympy.N(a, 1)
    xtype = (tuple([float]), float)

    nargs = 1
    is_real = True

    @classmethod
    def eval(cls, a):
        # if a.is_number:  # sympify(a) evaluates first... but i guess it is evaluated already
        #     return round(a)  # see
        return sympy.N(a, 0)

    def _sympy_(self, *args):
        return self.eval(*args)


def sympy_symbol_defaults(name_list):
    """
    sfeh workaround.
    sympy expressions like 'sign(((a * b) ** 151))' take forever.
    ignoring complex numbers with this trick (use this as locals)

    'sym_reduce': '({} ** {})'
    'sym_reduce': 'sign(re({}))'
    """
    return {str(x): sympy.symbols(str(x), real=True, imaginary=False) for x in name_list}


# attention: exactly same capitals/letters! (gets replaced)
# todo todotodo potential names
# arityfix, custom, structural
# Folding, collective, reduce, chained, fold, map
# todo required, if string is loaded. not required, if sympys are loaded

loadable_ops_dict = {'BinaryAdd': BinaryAdd, 'BinaryMultiply': BinaryMultiply, 'Sub': Subtract,
                     # 'Divide_no_nan': Divide_no_nan, 'Usub': Usub,
                     'Div': Div, 'Pow': Pow, 'Powrounded': Powrounded, 'Abs': Abs, 'sign': Sign, 'Square': Square,
                     'sqrt': Sqrt, 'log': Log, 'log1p': Log1p, 'cos': Cos, 'sin': Sin, 'tan': Tan, 'acos': Acos,
                     'asin': Asin, 'atan': Atan, 'tanh': Tanh, 'cosh': Cosh, 'sinh': Sinh,
                     'Xor': Xor, 'BinaryNot': BinaryNot,
                     'Ne': Ne, 'Lt': Lt, 'Le': Le, 'Gt': Gt, 'Ge': Ge, 'Ifte': Ifte,
                     'Round': Round,
                     'BinaryMin': BinaryMin, 'BinaryMax': BinaryMax, 'BinaryAnd': BinaryAnd, 'BinaryOr': BinaryOr,
                     'Eq': Eq}


# inline_operator_dict2 = {'and': And, 'or': Or, '<>': Ne}


def expr_sympify(expr, evaluate=None, eval_locals=None):
    """
    Returns a simplified expression using sympify.
    - sympify the expression
    - If sympify evaluates to one of these errors: 'zoo', 'inf', '*I', 'nan', stop evaluation

    Sympify is a python core module which reduced mathematical expressions.
    Example: sympify('a+a+a+a') -> a*4
    Note that the sympify was extended in plagih_sympify_extras.py with extra functions

    Sympify fails: The results are, or contain, expressions that should/can not be evaluated
    'zoo': (Complex infinity) E.g. when an int-number is divided by zero
    'inf': (Regular infinity) E.g. when a float-number is divided by zero (...i know, why are there two infinities?)
    '*I': (Complex number) E.g. when putting a number to the power of negative fractals, 1**(-0.5)
    'nan': (Not a number) when Evaluation fails, E.g. types contradict, expression is empty, 'BinaryMin(a, zoo' ...

    Sympy bug #1:
    It is a bug in sympy, read here https://stackoverflow.com/a/58530435/5626139
    Or this issue: https://github.com/sympy/sympy/issues/17785

    Sympy bug #2:
    print(plagih_sympify('a<zoo'))
    throws an exception.
    -> Try-except block for this case

    Lastly, it is recommended that you not use I, E, S, N, C, O, or Q
    """

    loadable_ops_dict.update(eval_locals or {})

    try:
        expr_sym = sympy.sympify(expr, evaluate=evaluate, locals=loadable_ops_dict)
        # try:  # todo
        #     expr_sym2 = expr_sym.factor()
        #     if expr_sym != expr_sym2:
        #         print(f'COMPARE FACTOR:\n{expr_sym}  len {len(expr_sym)}\n{expr_sym2}  len {len(expr_sym2)}')
        # except Exception as ex:
        #     print('fdsfdsahds')
        if expr_sym.has(sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I):
            raise ArithmeticError(f'Simplification failed for expression: {expr_sym}')
        return expr_sym

    except ValueError as ex:
        # return 'nan'  # 'nan' always evaluates to nan. ALl nan bugs should be solved.
        raise ValueError(f'NaN in {ex}')
    except AttributeError as ex:
        # print(f'sfeh: This sympy bug happens, when sympifying "True": {ex}')
        return sympy.true if expr else sympy.false
    # except Exception as ex:
    #     raise Exception(f'sympify_1: {expr} reason: ({ex})')


# sympy_constants = {
#     sympy.numbers.Zero: 0,
#     sympy.numbers.Half: 0.5,
#     sympy.numbers.One: 1,
#     sympy.numbers.NegativeOne: -1,
#     sympy.numbers.Exp1: 2.71828182845904,  # sympy.numbers.Exp1().evalf(16)
#     sympy.numbers.Pi: 3.1415926535897932,  # sympy.numbers.Pi().evalf(17)
#     sympy.numbers.GoldenRatio: 1.61803398874989,  # sympy.numbers.GoldenRatio().evalf(16)
#     sympy.numbers.TribonacciConstant: 1.83928675521416,  # sympy.numbers.TribonacciConstant().evalf(16)
#     sympy.numbers.EulerGamma: 0.577215664901532,  # sympy.numbers.EulerGamma().evalf(16)
# }
# sfeh
#  sympy.numbers.Infinity: tensorflow.constant(np.Infinity),
#  sympy.numbers.NegativeInfinity: tensorflow.constant(-np.Infinity),
#  sympy.numbers.ComplexInfinity: tensorflow.complex(0, np.Infinity),sympy.numbers.ImaginaryUnit,
#  sympy.numbers.NumberSymbol
#  sympy.numbers.Catalan: tensorflow.constant(sympy.numbers.Catalan),
#  sympy.numbers.NaN: tensorflow.constant(sympy.numbers.NaN),


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


# class Integer(sympy.Integer):
#     def get_tf(self, *args):
#         return tensorflow.int32(args[0].get_tf())
#
#
# class Max(sympy.functions.elementary.miscellaneous.Max):
#     def get_tf(self, *args):
#         return tensorflow.maximum(*args)
#     # def __init__(self, *args, **kwargs):
#     #     sympy.Max(*args, **kwargs)
#     # super(Max, self).__new__(*args, **kwargs)


# print(Max(3, 4), sympy.Max(3, 4))

# x = sympy.sympify('Max(a, 4)', locals={'Max': Max.__base__, 'Integer': Integer.__base__})
# x = sympy.sympify('Max(a, 4) + b ** 2')


# class PowRounded(Operator):
# sfeh open
#     nlabel = 'Pown'
#     arity = 2
#     tflow = tf.pow
# #     expr_sym = 'BinaryMax({}, {})'
# #     xtype = (tuple([float, float]), float)

loadable_ops_dict = {'BinaryAdd': BinaryAdd, 'BinaryMultiply': BinaryMultiply, 'Sub': Subtract,
                     # 'Divide_no_nan': Divide_no_nan, 'Usub': Usub,
                     'Div': Div, 'Pow': Pow, 'Powrounded': Powrounded, 'Abs': Abs, 'sign': Sign, 'Square': Square,
                     'sqrt': Sqrt, 'log': Log, 'log1p': Log1p, 'cos': Cos, 'sin': Sin, 'tan': Tan, 'acos': Acos,
                     'asin': Asin, 'atan': Atan, 'tanh': Tanh, 'sinh': Sinh, 'cosh': Cosh, 'Xor': Xor,
                     'BinaryNot': BinaryNot,
                     'Ne': Ne, 'Lt': Lt, 'Le': Le, 'Gt': Gt, 'Ge': Ge, 'Ifte': Ifte,
                     'BinaryMin': BinaryMin, 'BinaryMax': BinaryMax, 'BinaryAnd': BinaryAnd, 'BinaryOr': BinaryOr,
                     'Round': Round}

# TODO_LOADABLE_LATER = { 'Multiply': Multiply, 'And': And, 'Or': Or, 'Eq': Eq, 'Max': Max, 'Min': Min}

loadable_inline_operator_dict = {'+': BinaryAdd, '-': Subtract, '*': BinaryMultiply, '/': Div, '**': Pow,
                                 '==': Eq, '!=': Ne, '<': Lt, '<=': Le, '>': Gt, '>=': Ge,
                                 '&': BinaryAnd, '|': BinaryOr}
loadable_ops_dict.update(loadable_inline_operator_dict)

totf = {
    # sympy.Symbol: 'tf': lambda x: tf.cons
    sympy.Min: tf.minimum,
    sympy.Max: tf.maximum,
    sympy.Add: tf.add,
    sympy.Mul: tf.multiply,
    sympy.Pow: tf.pow,
    sympy.Abs: tf.abs,

    sympy.Not: tf.logical_not,
    sympy.And: tf.logical_and,
    sympy.Or: tf.logical_or,
    sympy.Xor: tf.math.logical_xor,

    sympy.Equality: tf.equal,
    sympy.Unequality: tf.not_equal,
    sympy.GreaterThan: tf.greater_equal,
    sympy.StrictGreaterThan: tf.greater,
    sympy.LessThan: tf.less_equal,
    sympy.StrictLessThan: tf.less,

    sympy.N: tf.math.round,

    sympy.log: tf.math.log,
    sympy.cos: tf.cos,
    sympy.sin: tf.sin,
    sympy.tan: tf.tan,
    sympy.acos: tf.acos,
    sympy.asin: tf.asin,
    sympy.atan: tf.atan,
    sympy.tanh: tf.tanh,

    sympy.sign: tf.sign,

    # BinaryAnd: tf.add,
    # BinaryMin: tf.minimum,
    # BinaryMax: tf.maximum,
    # Round: tf.round,
    # Ifte: tf.where,

    sympy.ITE: tf.where,  # sfeh:test this
    sympy.re: lambda x: tf.convert_to_tensor(x, dtype=tf.dtypes.float32),  # gotcha, comes up rndomly
}

# Sfeh:error "has no attribute 'tflow'" ->
sympy_to_node = {Subtract: Subtract, sympy.div: Div, sympy.Pow: Pow, Powrounded: Powrounded, sympy.Abs: Abs,
                 # Divide_no_nan: Divide_no_nan,
                 sympy.sign: Sign, Square: Square, sympy.sqrt: Sqrt, sympy.log: Log, Log1p: Log1p,
                 sympy.cos: Cos, sympy.sin: Sin, sympy.tan: Tan, sympy.acos: Acos, sympy.asin: Asin,
                 sympy.atan: Atan, sympy.tanh: Tanh, sympy.sinh: Sinh, sympy.cosh: Cosh, sympy.Xor: Xor,
                 sympy.Not: Not, sympy.Ne: Ne, sympy.Lt: Lt,
                 sympy.Le: Le, sympy.Gt: Gt, sympy.Ge: Ge,
                 # Usub: Usub,
                 Ifte: Ifte, sympy.And: BinaryAnd, Eq: Eq, BinaryOr: BinaryOr,
                 BinaryMultiply: BinaryMultiply}
sympy_to_node.update({sympy.Mul: BinaryMultiply, sympy.Max: BinaryMax, sympy.Min: BinaryMin, sympy.Add: BinaryAdd,
                      sympy.Eq: Eq, sympy.Ne: Ne})


# todo sympy_to_node mit mapx
# todo gotcha sympy.re comes up randomly


# todo_sympytonode = {sympy.Add: Add, sympy.Mul: Multiply, sympy.Max: Max, sympy.Min: Min, sympy.Or: Or, sympy.Equality: Eq, Piecewise: Piecewise}


# custom_locals = {
#     'BinaryMin': tf.minimum,  # todo
#     'BinaryMax': tf.maximum,
#     'Round': tf.round,
#     'Ifte': tf.where_v2,
# }


def sympy_to_tensorflow(expr, tensors_dict=None):
    """
    - check terminal-node
    -- check symbol
    --

    Bugs/gotchas:

    sympify('True')             -> True
    sympify('1')==True          ->
    sympify('~(True)')          -> -2
    sympify('~(False)')         -> -1
    sympify('a <= Min(a, b)')   ->

    sympy.logic.boolalg.ITE has only boolean inputs
    evaluate=None/false

    sympy.logic.boolalg.ITE is not If-then-else
    sympy.cosh can emerge out of sin-stuff
    """
    # shape = tensors[list(tensors.keys())[0]].get_shape()  # sfeh:open:workaround:

    # ==Bug-handling==  sympy.sympify('True')->True is no sympy expression anymore
    if isinstance(expr, bool):  # e.g. '1'
        return tf.constant(expr, dtype=tf.dtypes.bool)
    if isinstance(expr, (bool, sympy.logic.boolalg.BooleanAtom)):
        expr = True if isinstance(expr, sympy.logic.boolalg.BooleanTrue) else False  # sfeh:collect sympy bug gotcha bug
        return tf.constant(expr, dtype=tf.dtypes.bool)

    # the following lines are not required, when sympy filters for bad expressions earlier
    # if expr.is_imaginary or expr.is_infinite:
    #     raise ValueError(f'Cannot convert this to Tensorflow: {expr}')

    # ==Terminal nodes==
    elif expr.is_Atom:
        if expr.is_Symbol:
            result = tensors_dict[expr.name]  # sfeh:discuss placeholder
            return result

        else:
            expr_eval = expr.evalf()  # standard 15 digits
            if expr.is_Boolean:
                return tf.constant(expr_eval, dtype=tf.dtypes.bool)
            else:  # float
                return tf.constant(float(expr_eval), dtype=tf.dtypes.float32)

    else:  # Operator # len(expr.args) > 0:  # sfeh: line can be removed or replaced
        if isinstance(expr, sympy.Piecewise):
            _revlist = list(expr.args[::-1])  # tuples to list, reverse: last tuple must be nested the deepest
            _revlist = [[sympy_to_tensorflow(xx, tensors_dict=tensors_dict) for xx in list(x)] for x in _revlist]
            otherwise = _revlist[0][0]  # aka the last "True" condition
            for x in _revlist[1:]:
                otherwise = tf.where(x[1], x[0], otherwise)
            return otherwise
        try:
            tf_fun = totf[type(expr)]
        except KeyError:
            try:
                tf_fun = type(expr).tflow  # sfeh
            except Exception as ex:
                print('aaa', type(expr), expr, type(expr) in sympy_to_node)
                # ignore:
                # -> sympy.conjugate
                tf_fun = expr.tflow  # todo delete? delete case above? ### binarymax,

        tf_args = [sympy_to_tensorflow(x, tensors_dict=tensors_dict) for x in expr.args]
        try:
            result = tf_fun(*tf_args)  # fits, if the arguments match the expected arguments exactly Add(a, b)
        except TypeError:
            result = tf_args.pop()  # only commutative arity-2 functions here (Add, Mul, Max, Min)
            while tf_args:
                # sfeh:optimization
                result = tf_fun(result, tf_args.pop())
        return result


def sympy_to_nestedlist(expr):
    """
    creates a nestedlist, which can be used to create/load a Tree in the GP process
    """

    if isinstance(expr, bool):
        print('dndfjfugjuh')
        me = expr
        raise  # todo

    # ==Terminal nodes==
    elif expr.is_Atom:
        if expr.is_Symbol:
            me = f"'{expr}'"
        else:
            me = expr

    else:
        cc = [sympy_to_nestedlist(x) for x in expr.args]
        cc = ', '.join(cc)
        me = f"'{sympy_to_node[expr.func].nlabel}', {cc}"

    return f'[{me}]'


if __name__ == '__main__':

    ns = {
        'a': sympy.Symbol('a', real=True),
        'b': sympy.Symbol('b', real=True),
        'c': sympy.Symbol('c', bool=True),
        'd': sympy.Symbol('d', bool=True),
    }

    tensors = {
        'a': tf.constant([1.0, 2, 3, 4, 5, 6], dtype=tf.dtypes.float32),
        'b': tf.constant([-1.0, -2, -3, -4, -5, -6], dtype=tf.float32),
        'c': tf.constant([True, False, True, False, True, False], dtype=tf.dtypes.bool),
        'd': tf.constant([True, True, True, True, True, True], dtype=tf.dtypes.bool)
    }
    tst = [
        '5', '1', '0', '0.5', '-1', 'True', 'False',
        'c & True', 'c | False', '~c',
        'a<1', 'a<b', 'a<=b', 'a>=b', 'a>b', 'a==b', 'a!=b', 'a',
        # 'oo', 'zoo', 'I',
        'a + 1', 'a + 2', 'a*2', 'a - 2', 'a/2', 'a < 2', 'a**2', '2/a',
        'a*b*2', 'a+b+a+2+4', 'Min(a, b, 3)', 'Max(a, b, 4, a**2, a+b)', 'a<3',
        'Piecewise((a, c), (b, d), (a+b, True))',
        'Eq(4, 4.0)',
        # 'Square((BinaryMin(-2.176629, b) - Abs(a)))', 'Round(-123.333334234) + Round(b)',
        # 'Ifte(c, 1, 2)', '1 < BinaryMax(2, Ifte(1 < a, 1, 1))', 'BinaryMax(a+1, 2**(5-b))'
    ]
    tst_custom = [
        'Ifte(a, b, c)',
        '(((0.326675 * b_2) - c_9) + (Ifte((-c_9 < b_5), c_7, Ifte((Square(Gain_6) < BinaryMax(a_2, Ifte((c_9 < '
        'c_4), -Gain_3, Gain_5))), c_9, c_4))))',
        'BinaryMax(a, 2)',
        'BinaryMin(b, b)',
        'Ifte(True, a, 3)',
        'sin(asin(b))',
        'And(False, True)',
        'BinaryAdd(-1.490149, 14.0)',
        'Ifte(False, (3+a), 3)',
        'Eq(4, 4.0)',
        'Lt(a, a)',
        'BinaryOr(Ne(False, False), False)',
        'sqrt(5 * a)',
        'BinaryMultiply(log(acos(-0.212976)), asin(2))',
        'BinaryNot(False)',
        'acos(0.5)',
        'Round(1.2345)',
        'BinaryMin(Ifte(True, BinaryMultiply(a, 20.0), acos(-0.5)), 1)',
        'Div(BinaryAdd(13.159398, 19.284178), 1)',
        'Pow(a, b)',
        'BinaryAdd(-2, BinaryMin(Ifte(True, 1, b), 8))',
        '(BinaryOr(True, True) & c)',
        'Ifte(BinaryAnd(False, True), b, 0.046948)',
        'Ifte(Ne(True, Lt(Sub(a, 2), 1)), 1, 2)',
        'cos(tan(Square(BinaryMultiply(BinaryAdd(Round(Ifte(Ne(Ge(b, 15), True), 7, Sub(a, 16.5))), 5), 4))))'
    ]
    todo_problems = ['Ifte(Lt(Ifte(Eq(Min(b, 1), 3), Max(a, b), b), 0), 0, 2)']


    def test_basic_tfconversion():
        # sfeh:open tests

        for t in tst:
            x = sympy.sympify(t, locals=ns)
            x = sympy_to_tensorflow(x, tensors_dict=tensors)

            print(f'{t} \t{x}')


    def test_sympify():
        print('Running sympify example')

        # expr = '-b_0*sign(re(asdW**2)) - 0.004073'
        # expr = 'BinaryMin(-1 - 1 + sqrt(1)'
        # expr = 'BinaryMax(2.202197, (Abs(b) - sqrt(b)))'
        # expr = '(vel + vel)'
        # expr = 'a - 0.4375'

        # obs = ['a', 'b']
        # symloc = {x: sympy.symbols(x, real=True, imaginary=False) for x in obs}
        # sympy_symbol_dict = {'a': sympy.symbols('a', real=True, imaginary=False),
        #                      'b': sympy.symbols('b', real=True, imaginary=False)}
        # sympify('sign(((a * b) ** 151))', symloc)

        # obs = {'b': 0.5, 'a': -0.8}
        # sympex = plagih_sympify(expr, eval_locals=obs)
        for x in tst + tst_custom:
            sx = expr_sympify(x)
            print(sx)


    def test_this():
        x = expr_sympify(
            '(((0.326675 * b_2) - c_9) + (Ifte((-c_9 < b_5), c_7, Ifte((Square(Gain_6) < BinaryMax(a_2, Ifte((c_9 < c_4), -Gain_3, Gain_5))), c_9, c_4))))')
        print(x)


    # test_basic_tfconversion()
    # test_this()
    def print_relevant_subclasses():
        #
        def get_all_subclasses(cls):
            all_subclasses = []

            for subclass in cls.__subclasses__():
                all_subclasses.append(subclass)
                all_subclasses.extend(get_all_subclasses(subclass))

            return all_subclasses

        l = [x.nlabel for x in get_all_subclasses(Operator)]
        print(f'loadable_strings = {l}')
        c = [x.__name__ for x in get_all_subclasses(Operator)]
        print(f'operator_classes = {c}')
        d = dict(zip(l, c))
        d = ', '.join([f"'{k}': {v}" for k, v in d.items()])
        print(f'loadable_ops_dict = {{{d}}}')
        # sfeh: matches sympy/tenorflow
        st = {}
        # todo, todo ignore None as keys
        for x in get_all_subclasses(Operator):
            try:
                st[f'sympy.{x.insym.__name__}'] = x.__name__  # x.tflow.__name__
            except AttributeError as ex:
                print(f'Could not get {x}: {ex}')
                st[x.__name__] = x.__name__
        st = ', '.join([f"{k}: {v}" for k, v in st.items()])
        print(f'sympy_to_node = {{{st}}}')


    test_basic_tfconversion()  # sfeh all tests
    test_sympify()
    # x = expr_sympify('Ifte(True, 1, 2)')
    # print(x)
