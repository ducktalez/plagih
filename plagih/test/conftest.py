"""
Pytest fixtures for plagih test suite.

Provides common test data, symbols, operators, and helper functions
used across multiple test modules.

Test tiers
----------
* **Classic tests** (default): unit and functional tests that run fast on every
  ``pytest`` invocation.  No special flag required.

* **Performance / extended tests** (opt-in): marked with
  ``@pytest.mark.performance``.  These tests are skipped unless the
  ``--run-perf`` flag is passed::

      pytest - -run - perf

  They include GP-pipeline integration tests against real benchmark data,
  timing benchmarks, and other resource-intensive scenarios.

* **Standalone benchmark scripts** (``benchmarks/bench_*.py``): completely
  excluded from pytest collection.  Run them directly::

      python plagih/test/benchmarks/bench_performance.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy

# ---------------------------------------------------------------------------
# Exclude standalone benchmark scripts from pytest collection.
# They live in plagih/test/benchmarks/ and are meant to be run directly.
# ---------------------------------------------------------------------------
collect_ignore_glob = ["benchmarks/bench_*.py", "benchmarks/demo_*.py"]


# ---------------------------------------------------------------------------
# Performance-test marker: opt-in via --run-perf
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help=(
            "Include extended performance / integration tests "
            "(marked with @pytest.mark.performance). "
            "These are skipped by default because they are resource-intensive."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "performance: extended performance / integration test - skipped by default, run with --run-perf to include",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Skip all @pytest.mark.performance tests unless --run-perf was passed."""
    if config.getoption("--run-perf"):
        return  # nothing to skip

    skip_marker = pytest.mark.skip(reason="extended performance test - pass --run-perf to include")
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_marker)


# Lazy imports to avoid circular dependencies
def _get_tree_classes():
    """Lazy import of tree classes."""
    from plagih.trees import (
        ITE,
        Abs,
        Acos,
        Add,
        And,
        Asin,
        Atan,
        BaseOperator,
        Boolean,
        Candidate,
        ChainableOp,
        Clip,
        Cos,
        Cosh,
        Div,
        DivFraction,
        Eq,
        Evolution,
        Exp,
        ExplainableGP,
        ExprCondPair_Dummy,
        Ge,
        Gt,
        Ifte,
        Le,
        Log,
        LogicOperator,
        Lt,
        MathOperator,
        Max,
        Min,
        Mul,
        Ne,
        Node,
        NodeDummy,
        NodeWithChilds,
        Not,
        NthRoot,
        Number,
        Or,
        Piecewise,
        Pow,
        PowRounded,
        RelationalOperator,
        Round,
        Sign,
        Sin,
        Sinh,
        Sqrt,
        Square,
        Sub,
        Symbol,
        Tan,
        Tanh,
        Terminal,
        Trigonometry,
        Usub,
        Xor,
        eval_parsimony,
        sympy_to_tree,
        tree_simplification,
    )

    return locals()


# =============================================================================
# Symbols and Data Fixtures
# =============================================================================


@pytest.fixture
def float_symbols():
    """Returns sympy symbols for float variables."""
    return [sympy.Symbol("a", real=True), sympy.Symbol("b", real=True)]


@pytest.fixture
def bool_symbols():
    """Returns sympy symbols for boolean variables."""
    return [sympy.Symbol("c"), sympy.Symbol("d")]


@pytest.fixture
def sample_df():
    """Returns a small DataFrame for testing tree evaluation."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [-1.0, 0.5, 2.0, -0.5, 1.0],
            "c": [True, False, True, False, True],
            "d": [False, True, True, False, False],
            "action": [0.0, 1.0, 2.0, 1.0, 0.0],
        }
    )


@pytest.fixture
def large_df():
    """Returns a larger DataFrame for performance testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame(
        {
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.choice([True, False], n),
            "d": np.random.choice([True, False], n),
            "action": np.random.choice([0, 1, 2], n).astype(float),
        }
    )


@pytest.fixture
def cartpole_df():
    """Returns DataFrame similar to cartpole benchmark."""
    return pd.DataFrame(
        {
            "cartPos": [0.0, 0.5, 1.0, -0.5, -1.0, 0.1, 0.2, -0.3, 0.8, -0.9],
            "cartVel": [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.5, -0.4, 0.1, -0.2],
            "action": [1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        }
    )


# =============================================================================
# Operator Fixtures
# =============================================================================


@pytest.fixture
def basic_operator_dict():
    """Returns a minimal operator dictionary for testing."""
    from plagih.trees import Abs, Add, And, Cos, Div, Le, Lt, Max, Min, Mul, Not, Or, Sign, Sin, Sqrt, Square, Sub

    return {
        Add: 2,
        Mul: 2,
        Div: 1,
        Sub: 1,
        Abs: 1,
        Sign: 1,
        Square: 1,
        Sqrt: 0.5,
        Sin: 0.5,
        Cos: 0.5,
        Min: 1,
        Max: 1,
        Lt: 1,
        Le: 1,
        And: 1,
        Or: 1,
        Not: 1,
    }


@pytest.fixture
def full_operator_dict():
    """Returns complete operator dictionary including all operators."""
    from plagih.trees import (
        Abs,
        Acos,
        Add,
        And,
        Asin,
        Atan,
        Cos,
        Cosh,
        Div,
        Eq,
        Exp,
        Ge,
        Gt,
        Ifte,
        Le,
        Log,
        Lt,
        Max,
        Min,
        Mul,
        Ne,
        Not,
        Or,
        Pow,
        PowRounded,
        Round,
        Sign,
        Sin,
        Sinh,
        Sqrt,
        Square,
        Sub,
        Tan,
        Tanh,
        Xor,
    )

    return {
        Add: 2,
        Mul: 2,
        Div: 1,
        Sub: 1,
        Pow: 0.5,
        PowRounded: 0.5,
        Sqrt: 0.5,
        Square: 1,
        Abs: 1,
        Sign: 1,
        Log: 0.3,
        Exp: 0.3,
        Sin: 0.5,
        Cos: 0.5,
        Tan: 0.2,
        Asin: 0.2,
        Acos: 0.2,
        Atan: 0.2,
        Tanh: 0.3,
        Sinh: 0.2,
        Cosh: 0.2,
        Min: 1,
        Max: 1,
        Lt: 1,
        Le: 1,
        Gt: 0.5,
        Ge: 0.5,
        Eq: 0.5,
        Ne: 0.5,
        And: 1,
        Or: 1,
        Xor: 0.5,
        Not: 1,
        Ifte: 1,
        Round: 0.5,
    }


# =============================================================================
# Evolution and GP Fixtures
# =============================================================================


@pytest.fixture
def evolution_instance(float_symbols, basic_operator_dict):
    """Returns a configured Evolution instance."""
    from plagih.trees import Evolution

    return Evolution(symbol_list=float_symbols, operators=basic_operator_dict, depth_max=5, nodes_max=30)


@pytest.fixture
def cartpole_evolution():
    """Returns Evolution configured for cartpole-like symbols."""
    from plagih.trees import (
        Abs,
        Add,
        And,
        Cos,
        Div,
        Evolution,
        Le,
        Lt,
        Max,
        Min,
        Mul,
        Not,
        Or,
        Sign,
        Sin,
        Sqrt,
        Square,
        Sub,
    )

    operator_dict = {
        Add: 2,
        Mul: 2,
        Div: 1,
        Sub: 1,
        Abs: 1,
        Sign: 1,
        Square: 1,
        Sqrt: 0.5,
        Sin: 0.5,
        Cos: 0.5,
        Min: 1,
        Max: 1,
        Lt: 1,
        Le: 1,
        And: 1,
        Or: 1,
        Not: 1,
    }
    return Evolution(symbol_list=["cartPos", "cartVel"], operators=operator_dict, depth_max=5, nodes_max=30)


@pytest.fixture
def temp_output_dir():
    """Creates and yields a temporary directory, cleans up after test."""
    temp_dir = Path(tempfile.mkdtemp(prefix="plagih_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def gp_instance(cartpole_evolution, cartpole_df, temp_output_dir):
    """Returns a configured ExplainableGP instance for testing."""
    from plagih.trees import ExplainableGP

    eval_autocast = lambda x: np.clip(np.asarray(x, dtype=np.float64), 0.0, 2.0)
    eval_error_metric = lambda pred, true: np.sqrt(np.mean((pred - true) ** 2))

    return ExplainableGP(
        evolve=cartpole_evolution,
        df_train=cartpole_df,
        rootdir=temp_output_dir,
        pop_max_size=10,
        gen_end=3,
        eval_autocast=eval_autocast,
        eval_error_metric=eval_error_metric,
        allow_chain=False,
    )


# =============================================================================
# Tree Fixtures
# =============================================================================


@pytest.fixture
def simple_tree():
    """Returns a simple tree: Add(a, 1)"""
    from plagih.trees import Add, Number, Symbol

    return Add(Symbol(sympy.Symbol("a")), Number(1.0))


@pytest.fixture
def complex_tree():
    """Returns a more complex tree: Sin(Add(Mul(a, b), 2))"""
    from plagih.trees import Add, Mul, Number, Sin, Symbol

    return Sin(Add(Mul(Symbol(sympy.Symbol("a")), Symbol(sympy.Symbol("b"))), Number(2.0)))


@pytest.fixture
def boolean_tree():
    """Returns a boolean tree: And(Lt(a, 1), c)"""
    from plagih.trees import And, Boolean, Lt, Number, Symbol

    return And(Lt(Symbol(sympy.Symbol("a")), Number(1.0)), Boolean(True))


@pytest.fixture
def ifte_tree():
    """Returns an if-then-else tree."""
    from plagih.trees import Ifte, Lt, Number, Symbol

    return Ifte(Lt(Symbol(sympy.Symbol("a")), Number(0.0)), Number(-1.0), Number(1.0))


@pytest.fixture
def cartpole_tree():
    """Returns a tree using cartpole symbols."""
    from plagih.trees import Add, Mul, Number, Symbol

    return Add(Symbol(sympy.Symbol("cartPos")), Mul(Number(2.0), Symbol(sympy.Symbol("cartVel"))))


# =============================================================================
# Helper Functions (available to all tests)
# =============================================================================


def get_all_operator_classes():
    """Returns list of all concrete operator classes for testing."""
    from plagih.trees import BaseOperator
    from plagih.util import get_subclasses

    skip_classes = {
        "BaseOperator",
        "MathOperator",
        "LogicOperator",
        "RelationalOperator",
        "Trigonometry",
        "BaseMinMax",
        "NodeWithChilds",
        "NodeDummy",
        "PleaseUsePartnerOp",
        "CustomOperator",
        "NoSymCapitalized",
    }

    all_ops = []
    for cls in get_subclasses(BaseOperator):
        if cls.__name__ in skip_classes:
            continue
        if hasattr(cls, "xtype") and cls.xtype:
            all_ops.append(cls)
    return all_ops


def get_all_terminal_classes():
    """Returns list of all terminal classes."""
    from plagih.trees import Boolean, Number, Symbol

    return [Number, Symbol, Boolean]


def create_random_inputs(xtype_inputs, seed=42):
    """Creates random inputs based on xtype specification."""
    np.random.seed(seed)
    inputs = []
    for t in xtype_inputs:
        if t == float:
            inputs.append(np.random.uniform(-2, 2))
        elif t == bool:
            inputs.append(np.random.choice([True, False]))
        else:
            inputs.append(np.random.uniform(-1, 1))
    return inputs
