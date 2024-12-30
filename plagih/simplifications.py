from plagih.trees import *


class Simplification:
    """
    Different types of simplifications
    1. always possible + safe + sympyable(prio 1)
    - Tree complexity lower/same, readability better
    - E. g.: x**(0.5) -> sqrt(x)
    2. Undoing sympy overcomplexification
    - E. g.: (a/b) -sym-> a*(1/b) => a/b
    3. tree complexity grows, but readability or semantic improvements or pro-evolution
    - E. g.: Pow(a, 4) -> Powrounded(a, 4)
    4. alters tree identity
    - E. g.: approaching Number values (3.14 -> pi)

    Complexity metrics
    -

    Storing multiple versions of oneself
    -
    """
    for_node: type['Node']
    def __new__(cls, node: 'Node', *args, **kwargs):
        cls.for_node = node


class SimplifyExponentiation(Simplification):

    @staticmethod
    def apply(node):
        if isinstance(node, (Pow, PowRounded)):
            base, exp = node.get_childs()
            if isinstance(exp, Number):
                exp_val = exp.get_value()
                if exp_val == 2:
                    return Square(base)
                elif exp_val == 0.5:
                    return Sqrt(base)
        return node

class SimplifyPow(Simplification):
    @staticmethod
    def simplify(node, allow_grow=False):
        base, exp = node.get_childs()
        if isinstance(exp, Number):
            exp_val = exp.get_value()
            if exp_val in [-1, sympy.S.NegativeOne]:
                return DivFraction(base)
            elif exp_val == 2:
                return Square(base)
            elif exp_val in [0.5, sympy.S.Half]:
                return Sqrt(base)
            elif exp_val % 1 == 0:
                return PowRounded(base, exp)
            elif (1 / exp_val) % 1 == 0:
                return DivFraction(PowRounded(base, exp))
            elif exp_val == -2 and allow_grow:
                return DivFraction(Square(base))
        return node

class SimplifyMultiplication:
    @staticmethod
    def apply(node):
        if isinstance(node, Mul):
            children = node.get_childs()
            for child in children:
                if isinstance(child, Number) and child.get_value() == -1:
                    return Usub(Mul(*[c for c in children if c != child]))
        return node

SIMPLIFICATION_PIPELINE = [
    SimplifyExponentiation,
    SimplifyMultiplication,
    # Add more simplifications as needed
]

def apply_simplifications(node, pipeline=SIMPLIFICATION_PIPELINE):
    for strategy in pipeline:
        simplified_node = strategy.apply(node)
        if simplified_node != node:  # If simplification occurred
            return simplified_node  # Apply one simplification at a time
    return node  # No simplifications applied


def tree_simplify_with_pipeline(node, pipeline=SIMPLIFICATION_PIPELINE):
    if node.is_term():
        return node  # Terminal nodes do not require simplification

    # Simplify children
    node.set_childs([tree_simplify_with_pipeline(child, pipeline) for child in node.get_childs()])

    # Simplify current node
    return apply_simplifications(node, pipeline)

if __name__ == '__main__':
    root = Mul(Pow(Symbol('x'), Number(2)), Number(-1))  # Represents: -1 * (x ** 2)
    simplified_tree = tree_simplify_with_pipeline(root)
    print(simplified_tree)  # Output: Usub(Square(Symbol('x')))