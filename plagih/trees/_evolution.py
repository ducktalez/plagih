"""Evolution module: Candidate, NodeSelect, Evolution, and population helpers."""

import copy
import random
import warnings
from collections import deque
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
import sympy
from sympy.utilities.exceptions import ignore_warnings

from plagih.config import cfg as _cfg
from plagih.trees._nodes import *
from plagih.util import *


class Candidate:
    """A finalized individual in the genetic programming population.

    Combines a computation tree with its evaluated fitness and complexity metrics.
    Tracks the evolutionary history through tags indicating which operations
    created or modified this candidate.

    Attributes:
        tree: The computation tree (Node) representing the symbolic expression.
        fitness: The evaluated fitness score (lower is better).
        parsimony: The complexity/size measure of the tree.
        tag: Deque tracking the evolution history (max 10 entries).
    """

    def __init__(self, _tree: Node, fitness, parsimony, tag: str):
        self.tree = _tree
        self.fitness = fitness
        self.parsimony = parsimony
        self.tag = deque([tag], maxlen=10)  # Track which evolution created this candidate

    def append_tag(self, tag: str) -> None:
        self.tag.append(tag)

    def get_tag(self, i_evo: int = -1) -> str:
        # i_evo: -1 is last, -2 is second last, ...
        return self.tag[i_evo]

    def __str__(self):
        """Show the Parsimony and Fitness of a tree"""
        return f"[{self.get_parsim():2.0f}: fit {self.get_fitness():4.2f} ({self.tree.__str__()})]"

    def full_string(self) -> str:
        # Paretofront: Removing obsol ... ))]: \x1b[1msign(Max(c
        # https://stackoverflow.com/questions/62213322/python-3-bug-print-background-color-issue
        return f"{self.__str__()}: {BColors.BOLD}{self.get_evotree().get_sympy_expr()}{BColors.RESET}"

    def get_evotree(self) -> Node:
        return self.tree

    def get_fitness(self) -> float:
        # return self.meta.fitness
        return self.fitness

    def get_parsim(self) -> int:
        return self.parsimony


def selection_tournament(population: List[Candidate], n: int = 3) -> Node:
    """Selects an individual from population using tournament selection.

    Randomly samples n individuals and returns the fittest one's tree.
    Lower fitness is better (minimization).

    Args:
        population: List of Candidate objects to select from.
        n: Tournament size (number of individuals to compare).

    Returns:
        Deep copy of the winning candidate's tree.

    Raises:
        ValueError: If population is empty.
    """
    if not population:
        raise ValueError("Cannot select from empty population")

    # Sample n candidates (with replacement if n > len)
    tournament = random.choices(population, k=min(n, len(population)))

    # Select best (lowest fitness)
    winner = min(tournament, key=lambda c: c.get_fitness())

    # Return deep copy of tree to avoid modifying original
    return copy.deepcopy(winner.get_evotree())


def eval_predict_sympyBatch(sy_expr: sympy.Basic, _df: pd.DataFrame, symbol_list) -> pd.Series:
    """Evaluates a SymPy expression on a DataFrame using lambdify.

    Uses sympy.lambdify for vectorized evaluation with custom NumPy handlers
    for functions like Abs, RoundDummy, Min, Max that need special treatment.

    Args:
        sy_expr: SymPy expression to evaluate.
        _df: DataFrame with input columns.
        symbol_list: List of symbols matching DataFrame columns.

    Returns:
        Series with evaluated results for each row.
    """

    symbol_list_str = [str(s) for s in symbol_list]

    # Required functions for lambdify (poor native handling ofdimensionslity, ...)
    sy_np_handling = {"Abs": Abs.np_fun, "RoundDummy": RoundDummy.np_round_dummy, "Min": Min.np_fun, "Max": Max.np_fun}
    func = sympy.lambdify(symbol_list, sy_expr, modules=[sy_np_handling, "numpy"])

    with warnings.catch_warnings(), ignore_warnings(RuntimeWarning):  # often in ITE-terms? When math errors occur
        with ignore_warnings(DeprecationWarning):  # something 'like use "**" instead of "Pow"'
            df_results = _df.apply(lambda row: func(*[row[s] for s in symbol_list_str]), axis=1)

    return df_results


def check_operator_pool(ops: Dict[Type[BaseOperator], float]) -> None:
    """Validates that the operator pool allows type closure.

    Closure means the system can generate any required type:
    - Operators producing float from float
    - Operators producing bool from bool
    - Operators converting between types (float->bool, bool->float)

    Args:
        ops: Dict mapping operator classes to selection weights.

    Raises:
        Exception: If operators don't allow closure.

    Example (float-only, works):
        dict_operator_pool = {Add: 2, Sub: 1, Mul: 2, Div: 1}
    """

    opxtypes = [oper.xtype for oper in ops]
    has_2f = any([float == i[1] for i in opxtypes])
    has_2b = any([bool == i[1] for i in opxtypes])
    has_f2b = any([float in i[0] and bool == i[1] for i in opxtypes])
    has_b2f = any([bool in i[0] and float == i[1] for i in opxtypes])
    if not all([has_2f, has_2b, has_f2b, has_b2f]):
        log("ww", "Loaded operators do not feature both numeric (float) and bool type.")
    if all([has_2f, has_2b]) and not all([has_f2b or has_b2f]):
        raise Exception("Loaded operators do not allow closure!")


def norm_choices(val_p_tuples: list) -> list:
    """Normalizes a weighted choice list for numpy.random.choice.

    Transforms [['a', 1], ['b', 2]] -> [('a', 'b'), (0.333, 0.666)]
    Probabilities are normalized to sum to 1.

    Args:
        val_p_tuples: List of [value, weight] pairs.

    Returns:
        [values_tuple, probabilities_tuple] for np.random.choice.
    """
    xx = list(zip(*val_p_tuples))
    # normalizing the probabilities in every case to a sum of 1 (100%)
    psum = sum(xx[1])
    xx[1] = [i / psum for i in xx[1]]
    return xx


def operatorpool_to_picks(d_operator_pool: Dict[Type[BaseOperator], float]) -> Tuple[dict, dict]:
    """Converts operator pool to selection dictionaries.

    Creates two lookup structures:
    - pick_op: Operators grouped by output type (float/bool)
    - pick_op_match: Operators grouped by full xtype signature

    Args:
        d_operator_pool: Dict mapping operator classes to weights.

    Returns:
        Tuple of (pick_op, pick_op_match) dictionaries.
    """
    check_operator_pool(d_operator_pool)
    pick_op = {float: [], bool: []}
    pick_op_match = {}
    for _cls, _p in d_operator_pool.items():
        xt = _cls.xtype
        pick_op[xt[1]].append([_cls, _p])
        if pick_op_match.get(xt) is None:
            pick_op_match[xt] = []
        pick_op_match[xt].append([_cls, _p])

    pick_op = {float: norm_choices(pick_op[float]), bool: norm_choices(pick_op[bool])}
    for k_xt in pick_op_match:
        pick_op_match[k_xt] = norm_choices(pick_op_match[k_xt])
    return pick_op, pick_op_match


class NodeSelect:
    """Node selection utility for random tree generation.

    Manages probability distributions for selecting operators, terminals,
    and constants during tree creation and mutation.

    Attributes:
        pick_op: Probability distributions for operators by output type (float/bool).
        pick_op_match: Probability distributions for operators by full xtype signature.
        pick_symbol: Probability distributions for symbol selection.
        pick_constant: Probability distributions for constant generation.
    """

    def __init__(self, operators: dict, symbol_list: List[sympy.Symbol]):
        """Initialize node selector with operator pool and available symbols.

        Args:
            operators: Dict mapping operator classes to their selection weights.
            symbol_list: List of sympy symbols available as input variables.
        """

        self.pick_op, self.pick_op_match = operatorpool_to_picks(operators)
        # sfeh: Acos: 0.33, Asin: 0.33, Atan: 0.33, Tanh: 0.5, Usub: 1, Xor: 1
        # Round: 0.5, Eq: 1,  # Ne: 0.5, #  # Log1p: 0.1, Gt: 0.1, Ge: 0.1,, Tan: 0.1, Sub: 1, Cos: 0.33
        # Powrounded: 0.5

        self.pick_symbol = {
            # float: norm_choices([[symbols_lambda(ii), 1] for ii in symbols]),
            float: norm_choices([[ii, 1] for ii in symbol_list]),
            bool: [],
        }  # NotImplementedError

        # Design: Direct generation instead of lambda-pool.
        # Simpler, picklable (required for ProcessPoolExecutor), equally flexible.
        # Previously used lambdas in a weighted choice list, but the indirection
        # added complexity without benefit. Constants are now generated inline
        # in choose_constant_node().

    def choose_operator_class(self, xt: Union[Type[float], Type[bool]]) -> Type[BaseOperator]:
        """Randomly selects an operator class that produces the given output type.

        Args:
            xt: The required output type (float or bool).

        Returns:
            An operator class (not instance) matching the output type.
        """
        op = np.random.choice(self.pick_op[xt][0], p=self.pick_op[xt][1])  # no (), which would evaluate the op
        return op

    def choose_operator_class_match(self, xtype: tuple) -> Type[BaseOperator]:
        """Selects an operator class matching the exact type signature.

        Args:
            xtype: Full type signature ((input_types), output_type).

        Returns:
            An operator class with matching xtype.
        """
        if CHAIN_implement:
            pass
        op = np.random.choice(self.pick_op_match[xtype][0], p=self.pick_op_match[xtype][1])
        return op

    def choose_terminal_node(self, xt: Union[Type[float], Type[bool]], p_observation: float = 0.5) -> Terminal:
        """Randomly selects a terminal node (Symbol or constant).

        With probability p_observation, tries to select a Symbol (input variable).
        Falls back to constant if no Symbol available for the type.

        Args:
            xt: The required output type (float or bool).
            p_observation: Probability of choosing a Symbol over a constant.

        Returns:
            A Terminal node (Symbol, Number, or Boolean).

        Note:
            Bug fixed: 'expected str|int|long|float|Decimal|Number object but got Node'
        """
        if np.random.random() > p_observation:
            try:
                _v = self.choose_symbol_node(xt)
                return _v  # MUST STAY HERE
            except (TypeError, IndexError):
                # return a constant (E.g. because there are no boolean observations)
                pass

        _v = self.choose_constant_node(xt)

        return _v

    def choose_constant_node(self, xt: Union[Type[float], Type[bool]]) -> Terminal:
        """Randomly generates a constant terminal node.

        For float: 50% normal distribution N(1,1), 50% random integer [1,20].
        For bool: Random True/False.

        Design: Direct generation instead of lambda-pool for picklability
        (required for multiprocessing) and simplicity.

        Args:
            xt: The required output type (float or bool).

        Returns:
            A Number or Boolean terminal node.
        """
        if xt == float:
            # 50/50 chance: normal distribution or random integer
            if random.random() < 0.5:
                _v = round(random.normalvariate(1, 1), _cfg.float_precision)
            else:
                _v = round(random.randint(1, 20), _cfg.float_precision)
            _v = sympy.Float(_v, _cfg.float_precision)  # discuss allow "rational" inputs? 1/3, 3/4, ...
            return Number(_v)  # round float_precision was here
        else:
            _v = random.choice((True, False))
            return Boolean(_v)

    def choose_symbol_node(self, xt: Union[Type[float], Type[bool]]) -> Symbol:
        """Randomly selects a Symbol node from available input variables.

        Similar to choose_terminal_node but always returns a Symbol.

        Args:
            xt: The required output type (float or bool).

        Returns:
            A Symbol terminal node referencing an input column.
        """
        _v = np.random.choice(self.pick_symbol[xt][0], p=self.pick_symbol[xt][1])
        n = Symbol(_v)
        return n


class Evolution:
    """Tree evolution operations for genetic programming.

    Provides methods for creating, mutating, and crossing over computation trees.
    Manages constraints like maximum depth and node count.

    Tree creation strategies:
    - Size measure: depth, node count, weighted node count (parsimony)
    - Architectures: Full, Grow, Ramped Half-and-Half
    - Node selection: Random, weighted random

    Attributes:
        origin_xtype: Output type of the root node (float or bool).
        origin_tree: Optional template tree with fixed structure.
        symbol_list: Available input variables as sympy symbols.
        node_selector: NodeSelect instance for random node generation.
        depth_max: Maximum allowed tree depth.
        nodes_max: Maximum allowed node count.
        complexity_metric: Method for measuring tree complexity.
    """

    operator_presets = {
        "math_simple": {
            Add: 2,
            Mul: 2,
            Scale: 0.5,
            Div: 1,
            Square: 0.75,
            Abs: 0.5,
            Sign: 0.5,
            Sqrt: 0.1,
            Log: 0.1,
            Sin: 0.5,
            Not: 0.5,
            Lt: 0.5,
            Le: 0.5,
            And: 1,
            Or: 1,
            Min: 1,
            Max: 1,
        }
    }

    def __init__(
        self,
        symbol_list=None,
        origin_xtype=float,
        operators=None,
        origin_tree=None,
        depth_max=10,
        nodes_max=100,
        complexity_metric="tree_node_count_fair",
        allow_chain=None,
    ):
        """Initialize evolution with operator pool and constraints.

        Args:
            symbol_list: List of input variable names or sympy symbols.
            origin_xtype: Expected output type (float or bool).
            operators: Dict of operators with weights, or preset name, or list.
            origin_tree: Optional template tree with fixed nodes.
            depth_max: Maximum tree depth.
            nodes_max: Maximum node count.
            complexity_metric: 'tree_node_count_raw', 'tree_node_count_fair',
                'tree_edit_distance', 'tree_python_bytecode_count',
                'tree_python_bytecode_weighted_count', 'tree_cpu_cost_proxy',
                or 'tree_flops_proxy'.
            allow_chain: Whether to allow chained operators.
        """
        self.origin_xtype = origin_xtype
        self.origin_tree = origin_tree

        # operators -> {Add: 1}
        if operators is None:
            operators = self.operator_presets["math_simple"]
        elif isinstance(operators, str):
            operators = self.operator_presets[operators]
        elif isinstance(operators, list):
            operators = {e: 1 for e in list(operators)}
        elif isinstance(operators, dict):
            pass
        else:
            raise NotImplementedError

        if symbol_list is None:
            symbol_list = sympy.symbols("a b", real=True, imaginary=False)  # sympy symbols options
        else:
            symbol_list = [sympy.Symbol(s) if isinstance(s, str) else s for s in symbol_list]
            symbol_list = sorted(symbol_list, key=lambda x: str(x))
        self.symbol_list = symbol_list
        self.symbol_list_str = [str(s) for s in symbol_list]  # -> for df-evaluation (string-keys are expected...)
        self.node_selector = NodeSelect(operators, symbol_list)

        self.complexity_metric = complexity_metric

        self.depth_max = depth_max
        self.nodes_max = nodes_max

        self.allow_a_chain = allow_chain

    def evolve_prune_tree(self, _tree: Node) -> Node:
        """Prunes a tree to meet depth and node count constraints.

        Strategies:
        - Depth pruning: Replaces nodes exceeding max depth with terminals
        - Node pruning: Randomly replaces branches to reduce total count

        Note: Pruning should ideally be handled during creation, as it
        strongly affects tree structure and randomly removes nodes.

        Args:
            _tree: The tree to prune.

        Returns:
            The pruned tree (modified in place).
        """
        nodelist = _tree.list_mutable_nodes()
        for dnode in nodelist:
            if dnode.depth == self.nodes_max and dnode.get_arity() > 0:
                log("wwww", f"Node in fintree is too deep: {dnode.depth}")
                new_node = self.node_selector.choose_terminal_node(dnode.get_xtype_self())
                new_node.depth = dnode.depth
                dnode.set_new_node(new_node)

        # sfeh not as trivial as pruning the max. tree depth: Which nodes to prune randomly?
        #   This strongly affects the tree structure and should thus be decided in the creation process
        #   Pruning strategies:
        #   - Randomly prune nodes until complexity is met
        #   - Prune the deepest nodes first, every depth level completely
        #   - check crossover
        prune_amount = len(_tree) - self.nodes_max
        while prune_amount > 0:
            log("wwww", f"Tree too complex: {len(_tree)} > {self.nodes_max}, pruning {prune_amount}.")
            nodelist = _tree.list_mutable_nodes()
            prune_now = 1 + np.random.randint(prune_amount)  # 19 -> prune branch with 1 to max. 19 nodes

            nodelist = [x for x in nodelist if len(x) >= prune_now]  # only (operator-) nodes
            _tree = random.choice(nodelist)
            new_node = self.node_selector.choose_terminal_node(_tree.get_xtype_self())
            new_node.depth = _tree.depth
            _tree.set_new_node(new_node)
            prune_amount = len(_tree) - self.nodes_max

        return _tree

    def evolve_new_tree_depth(
        self, xt_out: Union[Type[float], Type[bool]], depth_goal: int, p_term: float = 0.0
    ) -> Node:
        """Creates a new random tree with target depth.

        If an origin_tree is set, fills its mutable slots with random branches.
        Otherwise creates a completely random tree.

        Args:
            xt_out: Output type for the root node (float or bool).
            depth_goal: Target maximum depth.
            p_term: Probability of terminating at each node with a terminal.

        Returns:
            A new random tree.
        """

        if self.origin_tree is not None:
            evotree = copy.deepcopy(self.origin_tree)
            layer0 = evotree.get_mutable_rootnodes(extend_lvls=0)

            for ii, nd in enumerate(layer0):  # -> get layer every time (nsted ids might have changed)
                new_subbranch = self.evolve_create_random(
                    nd.get_xtype_self(), depth_goal, num_rest=-1, depth=nd.depth, p_term=p_term
                )
                nd.set_new_node(new_subbranch)

        else:
            evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    def evolve_chained_new_tree_depth(
        self, depth_goal: int, xt_out: Union[Type[float], Type[bool]], p_term: float = 0.0
    ) -> Node:
        """Creates a new random tree with chained operators allowed.

        Args:
            depth_goal: Target maximum depth.
            xt_out: Output type for the root node.
            p_term: Probability of terminating branches early.

        Returns:
            A new random tree potentially using chained operators.
        """

        evotree = self.evolve_create_random(xt_out, depth_goal, depth=0, num_rest=-1, p_term=p_term)

        return evotree

    def evolve_create_random(
        self,
        xt_out: Union[Type[float], Type[bool]],
        depth_max_local: int,
        num_rest: int = -1,
        depth: int = 0,
        p_term: float = 0.0,
    ) -> Node:
        """Recursively creates a random tree/subtree.

        Args:
            xt_out: Required output type for this node.
            depth_max_local: Maximum depth (can be less than self.depth_max).
            num_rest: Remaining node budget (-1 ignores limit).
            depth: Current depth level.
            p_term: Probability of placing a terminal instead of operator.

        Returns:
            A randomly generated subtree.

        Note:
            Node count is not an ideal threshold as it limits depth-spreading.
            Consider pruning at the end and allowing any growth initially.
        """

        # setting a terminal-node if it is required OR p_term is met
        if depth >= min(self.depth_max, depth_max_local) or num_rest == 0 or random.random() < p_term:
            node = self.node_selector.choose_terminal_node(xt_out)
        else:
            node_cls = self.node_selector.choose_operator_class(xt_out)
            child_xts = node_cls.get_child_xts()
            childs = []

            if CHAIN_implement:
                pass  # optional; just add more node here already

            # Scale special case: first child is always a constant (Number),
            # second child is a random expression (operator or symbol, NOT Number).
            if node_cls is Scale:
                scale_factor = self.node_selector.choose_constant_node(float)
                # Second child: recurse normally but avoid a bare Number terminal
                # by setting p_term=0 if depth allows at least one more level.
                sub_p_term = 0.0 if depth + 1 < min(self.depth_max, depth_max_local) else p_term
                scale_expr = self.evolve_create_random(
                    float, depth_max_local, num_rest=max(num_rest - 1, -1), depth=depth + 1, p_term=sub_p_term
                )
                # Fallback: if we still got a Number, wrap it in an operator or use a Symbol
                if isinstance(scale_expr, Number):
                    try:
                        scale_expr = self.node_selector.choose_symbol_node(float)
                    except (TypeError, IndexError):
                        pass  # keep the Number; Scale.__init__ will warn
                node = Scale(scale_factor, scale_expr)
            else:
                nums = randomly_split_range(num_rest - 1, len(child_xts))

                for ii, xt in enumerate(child_xts):
                    cc = self.evolve_create_random(
                        xt, depth_max_local, num_rest=nums[ii], depth=depth + 1, p_term=p_term
                    )
                    childs.append(cc)

                node = node_cls(*childs)

        node.depth = depth

        return node

    def evolve_mutate_filter(self, _tree: Node) -> Node:
        """Applies Gaussian mutation to numeric terminals in a random subtree.

        Prefers Scale nodes as mutation targets — their scaling factor
        (childs[0]) is the ideal parameter for fine-tuning via Gaussian noise.
        Falls back to a random mutable node if no Scale nodes exist.

        Args:
            _tree: The tree to mutate.

        Returns:
            The mutated subtree node.
        """

        # Prefer Scale nodes for targeted constant tuning
        mutable = _tree.list_mutable_nodes()
        scale_nodes = [n for n in mutable if isinstance(n, Scale)]
        if scale_nodes and random.random() < 0.5:
            _nd = random.choice(scale_nodes)
            # Mutate only the scaling factor (childs[0]), not the expression subtree
            if isinstance(_nd.childs[0], Number):
                val = round(random.gauss(_nd.childs[0].get_value(), 0.1), _cfg.float_precision)
                _nd.childs[0].set_value(val)
                return _nd
        _nd = random.choice(mutable)
        _nd.evolve_mutate_filter_gauss()

        return _nd

    def evolve_mutate_point(self, _tree: Node) -> Node:
        """Mutates a single random node while preserving type signature.

        For operators: Replaces with another operator of same arity/type.
        For terminals: Replaces with another terminal of same type.

        Args:
            _tree: The tree to mutate.

        Returns:
            A deep copy of the tree with one node mutated.
        """
        evotree = copy.deepcopy(_tree)

        node = rnd_choice(evotree.list_mutable_nodes())  # debug if ignores chains
        xtype = node.get_xtype_tuple()

        if node.is_operator():
            # allow_chain-option
            new_label = self.node_selector.choose_operator_class_match(xtype)  # Function is same type, same arity
            node = new_label(*node.childs)

        elif is_terminal(node):
            new_node = self.node_selector.choose_terminal_node(xt_self(xtype))
            node.set_new_node(new_node)
        else:
            raise NotImplementedError

        return evotree

    def evolve_mutate_branch_depth(self, tree: Node, depth_goal: int, allow_chain=False, p_term: float = 0.0) -> Node:
        """Replaces a random subtree with a new random branch.

        Args:
            tree: The tree to mutate.
            depth_goal: Target depth for the new branch.
            allow_chain: Whether to allow chained operators.
            p_term: Probability of terminating nodes early.

        Returns:
            The mutated tree (modified in place).
        """
        n_init = len(tree)
        node_list = tree.list_mutable_nodes()
        node = random.choice(node_list)
        xtype_out = node.get_xtype_self()  # ValueError: 'a' cannot be empty unless no samples are taken
        branch = self.evolve_create_random(
            xtype_out, depth_goal, num_rest=self.nodes_max - n_init, depth=0, p_term=p_term
        )
        node.set_new_node(branch)

        return tree

    def evolve_mutate_branch_nodes(self, _tree: Node, nodes_goal, p_term=0.0) -> Node:
        """Replaces a random subtree with a new branch of target node count.

        Args:
            _tree: The tree to mutate.
            nodes_goal: Target number of nodes for the new branch.
            p_term: Probability of terminating the tree at each node.

        Returns:
            The mutated tree.

        Raises:
            NotImplementedError: If tree is None (selection mechanism needed).
        """
        nodes_init = len(_tree)
        if _tree is None:
            raise NotImplementedError("Implement standard selection mechanism")
        nd = _tree.list_mutable_nodes()
        nd = rnd_choice(nd)
        xt_out = nd.get_xtype_self()
        nodes_goal = min(self.nodes_max - (nodes_init - len(nd)), nodes_goal)

        branch = self.evolve_create_random(xt_out, -1, num_rest=nodes_goal, depth=nd.depth, p_term=p_term)
        nd.set_new_node(branch)
        mutated_tree = _tree
        return mutated_tree

    def evolve_crossover(self, aa: Node, bb: Node):
        """Performs subtree crossover between two trees.

        Swaps compatible subtrees between parent trees:
        1. Select a random node in tree aa
        2. Find a compatible node (same output type) in tree bb
        3. Swap the subtrees

        Args:
            aa: First parent tree.
            bb: Second parent tree.

        Returns:
            Tuple (aa, bb) with swapped subtrees, pruned if necessary.

        Raises:
            TreeError: If tree aa has no mutable nodes.
            ValueError: If no compatible nodes can be found.

        Examples::

            # Build two parent trees and cross them over.
            # Use deepcopy so the originals are preserved for comparison.
            import copy
            from plagih.demo_helpers import (
                do_crossover,
                make_evolution,
                make_tree_crossover_parent_a,
                make_tree_crossover_parent_b,
            )

            parent_a = make_tree_crossover_parent_a()  # sin(a) + b
            parent_b = make_tree_crossover_parent_b()  # a * |b|
            child_a, child_b = do_crossover(parent_a, parent_b)

            print(parent_a.get_sympy_expr())  # sin(a) + b
            print(parent_b.get_sympy_expr())  # a*Abs(b)
            print(child_a.get_sympy_expr())  # subtree from parent_b grafted in
            print(child_b.get_sympy_expr())  # subtree from parent_a grafted in

            # See docs/demo.ipynb §2 for the visual side-by-side render.
        """

        a_nds = aa.list_mutable_nodes()
        a_nds = a_nds[1:]  # skip_first ...why actually ignore root node?
        #   -> this shall prevent two trees from just "swapping place" (aka only root nodes are exchanged)
        #   -> this can actually happen quite often, when trees have low complexity

        if len(a_nds) == 0:
            raise TreeError("Crossover tree 1 has no mutable nodes!")

        a_nd = random.choice(a_nds)
        xt_out = a_nd.get_xtype_self()
        b_nds = bb.list_mutable_nodes(xtype=xt_out)

        if len(b_nds) > 0:
            b_nd = random.choice(b_nds)

        else:
            xt_out = float if xt_out == bool else bool  # switching to the other swap type
            b_nds = bb.list_mutable_nodes(xtype=xt_out)
            b_nd = random.choice(b_nds)
            a_nds = [x for x in a_nds if x.get_xtype_self() == xt_out]
            if len(a_nds) == 0:
                raise ValueError("Crossover cant find matching nodes. This Should always be possible.")
            a_nd = random.choice(a_nds)

        cpy = copy.deepcopy(a_nd)  # deepcopy required??

        a_nd.set_new_node(b_nd)
        b_nd.set_new_node(cpy)

        # only required, if pruning is not done in finalize_tree()
        aa = self.evolve_prune_tree(_tree=aa)
        bb = self.evolve_prune_tree(_tree=bb)

        return aa, bb

    def finalize_tree(self, tree):
        """Finalizes a tree after evolution operations.

        Performs:
        - Ensures tree has at least one input variable
        - Prunes if necessary (should be handled in evolution)
        - Repairs depth values in all nodes

        Currently a no-op placeholder for future validation.

        Args:
            tree: The tree to finalize.
        """
        # sfeh:open
        pass


def randomly_split_range(range_max: int, num_splits: int) -> list[int]:
    """Randomly splits an integer range into parts that sum to range_max.

    Example: split_range(100, 3) -> [33, 15, 52]

    Used for distributing node budget among child branches during tree building.
    Zero is allowed as it terminates a branch with a terminal node.

    Args:
        range_max: Total to split (or -1 to ignore limits).
        num_splits: Number of parts to create.

    Returns:
        List of integers summing to range_max.
    """

    if range_max < 0:
        return [-1 for _ in range(num_splits)]

    sample_dist = np.random.rand(num_splits)  # [0.2, 0.8, 0.5] -> random samples
    d_sum = sum(sample_dist)  # 1.5
    sample_dist = [i / d_sum for i in sample_dist]  # [0.12, 0.6, 0.28] -> fittet to sum of 1
    sample_dist = [i * range_max for i in sample_dist]  # [12, 60, 28] -> for 100 nodes
    sample_dist = [int(round(i, 0)) for i in sample_dist]  # int required

    # workaround, this makes exactly the correct range by changing the most "extreme" entry
    imprecise_diff = range_max - sum(sample_dist)  # sfeh: this can be [0, 0, 0], which assigns to the 0th bin...
    # sfeh:discuss: maybe this difference is 2 or larger more often than 1 (->rounding),
    # so maybe while-loop (just check if it happens?)
    if imprecise_diff != 0:
        if sum(sample_dist) < range_max:
            # if relatively empty, this appends to the first bin
            sample_dist[sample_dist.index(min(sample_dist))] += imprecise_diff  # extreme_bin = smallest
        elif sum(sample_dist) > range_max:
            sample_dist[sample_dist.index(max(sample_dist))] += imprecise_diff  # extreme_bin = greatest
        else:
            raise

    return sample_dist


def print_pop(pop):
    """Prints all candidates in a population with colored formatting.

    Alternates between blue and yellow for readability.
    Shows parsimony, fitness, and sympy expression for each tree.

    Args:
        pop: List of Candidate objects to print.
    """
    n = [f"{k.full_string()}" for k in pop]
    n = [f"{BColors.BLUE}{x}" if ii % 2 == 0 else f"{BColors.YELLOW}{x}" for ii, x in enumerate(n)]
    n = [f"{k}\n" if ii % 10 == 9 else f"{k}\t" for ii, k in enumerate(n)]  # stop \n in line 0
    n = "".join(n)
    n = re.sub(r"\n$", "", n)  # remove trailing \n (\t irrelevant)
    n = f"{n}{BColors.RESET_COLOR}"
    print(n)


def pop_analyze(population: List[Candidate], gen_time: float, gens_since_pareto: int, lut_symex: dict) -> dict:
    """Analyzes population statistics for monitoring.

    Computes fitness and parsimony statistics for the current generation.

    Args:
        population: List of Candidate objects.
        gen_time: Time taken for this generation in seconds.
        gens_since_pareto: Generations since last Pareto front update.
        lut_symex: Lookup table for sympy expressions to fitness.

    Returns:
        Dictionary with population statistics.
    """
    if not population:
        return {
            "pop_len": 0,
            "pop_unique": 0,
            "lut_symex_fitness-len": len(lut_symex),
            "time": gen_time,
            "fit_avg": np.nan,
            "fit_var": np.nan,
            "fit_quantile_25": np.nan,
            "fit_quantile_50": np.nan,
            "fit_quantile_75": np.nan,
            "fit_best": np.nan,
            "parsim_avg": np.nan,
            "parsim_var": np.nan,
            "parsim_quantile_25": np.nan,
            "parsim_quantile_50": np.nan,
            "parsim_quantile_75": np.nan,
            "parsim_best": np.nan,
            "gens_since_last_pareto": gens_since_pareto,
        }

    fitnesses = np.array([c.get_fitness() for c in population])
    parsimony = np.array([c.get_parsim() for c in population])

    # Count unique expressions
    unique_exprs = set(str(c.tree.get_sympy_expr()) for c in population)

    return {
        "pop_len": len(population),
        "pop_unique": len(unique_exprs),
        "lut_symex_fitness-len": len(lut_symex),
        "time": gen_time,
        "fit_avg": np.mean(fitnesses),
        "fit_var": np.var(fitnesses),
        "fit_quantile_25": np.percentile(fitnesses, 25),
        "fit_quantile_50": np.percentile(fitnesses, 50),
        "fit_quantile_75": np.percentile(fitnesses, 75),
        "fit_best": np.min(fitnesses),
        "parsim_avg": np.mean(parsimony),
        "parsim_var": np.var(parsimony),
        "parsim_quantile_25": np.percentile(parsimony, 25),
        "parsim_quantile_50": np.percentile(parsimony, 50),
        "parsim_quantile_75": np.percentile(parsimony, 75),
        "parsim_best": np.min(parsimony),
        "gens_since_last_pareto": gens_since_pareto,
    }
