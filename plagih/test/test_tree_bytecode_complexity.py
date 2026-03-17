"""Tests for Python-bytecode-based tree complexity measures."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import dis

import pytest
import sympy

from plagih.tree_complexity.python_bytecode_complexity import (
    BytecodeComplexityConfig,
    compile_tree_to_python_lambda,
    count_python_bytecode_instructions,
    count_tree_cpu_cost_proxy,
    count_tree_flops_proxy,
    get_python_bytecode_disassembly,
    get_python_bytecode_instructions,
    get_python_expression_source,
    get_python_lambda_source,
)
from plagih.trees import Add, Boolean, Ifte, Log, Mul, Number, Sin, Symbol, eval_parsimony


def _sym(name: str) -> Symbol:
    return Symbol(sympy.Symbol(name, real=True))


def _num(val: float) -> Number:
    return Number(val)


class TestPythonExpressionSource:
    def test_simple_expression_source(self):
        tree = Add(_sym("x"), _num(1))
        src = get_python_expression_source(tree)
        assert "kwargs['x']" in src
        assert "+" in src

    def test_lambda_source(self):
        tree = Add(_sym("x"), _num(1))
        src = get_python_lambda_source(tree)
        assert src.startswith("lambda **kwargs:")
        assert "kwargs['x']" in src

    def test_compile_lambda(self):
        tree = Add(_sym("x"), _num(1))
        fn = compile_tree_to_python_lambda(tree)
        assert callable(fn)


class TestBytecodeInstructions:
    def test_instruction_list(self):
        tree = Add(_sym("x"), _num(1))
        instructions = get_python_bytecode_instructions(tree)
        assert instructions
        assert all(isinstance(instr, dis.Instruction) for instr in instructions)

    def test_disassembly_contains_return(self):
        tree = Add(_sym("x"), _num(1))
        text = get_python_bytecode_disassembly(tree)
        assert "RETURN_VALUE" in text

    def test_instruction_count_positive(self):
        tree = Add(_sym("x"), _num(1))
        n = count_python_bytecode_instructions(
            tree,
            BytecodeComplexityConfig(method="instruction_count"),
        )
        assert n > 0

    def test_complex_tree_has_higher_bytecode_count(self):
        simple = Add(_sym("x"), _num(1))
        complex_tree = Sin(Add(Mul(_sym("x"), _sym("y")), _num(2)))
        n_simple = count_python_bytecode_instructions(
            simple,
            BytecodeComplexityConfig(method="instruction_count"),
        )
        n_complex = count_python_bytecode_instructions(
            complex_tree,
            BytecodeComplexityConfig(method="instruction_count"),
        )
        assert n_complex > n_simple

    def test_weighted_count_positive(self):
        tree = Log(Add(Mul(_sym("x"), _sym("y")), _num(1)))
        n = count_python_bytecode_instructions(
            tree,
            BytecodeComplexityConfig(method="weighted_instruction_count"),
        )
        assert n > 0


class TestProxyMeasures:
    def test_cpu_cost_proxy_positive(self):
        tree = Add(_sym("x"), _num(1))
        assert count_tree_cpu_cost_proxy(tree) > 0

    def test_flops_proxy_logic_is_low(self):
        logic_tree = Ifte(Boolean(True), _num(1), _num(2))
        math_tree = Log(Add(Mul(_sym("x"), _sym("y")), _num(1)))

        f_logic = count_tree_flops_proxy(logic_tree)
        f_math = count_tree_flops_proxy(math_tree)

        assert f_logic >= 0
        assert f_math > f_logic

    def test_flops_proxy_positive_for_math_tree(self):
        tree = Sin(Add(Mul(_sym("x"), _sym("y")), _num(2)))
        assert count_tree_flops_proxy(tree) > 0

    def test_method_dispatch_cpu_proxy(self):
        tree = Add(_sym("x"), _num(1))
        n = count_python_bytecode_instructions(
            tree,
            BytecodeComplexityConfig(method="cpu_cost_proxy"),
        )
        assert n == count_tree_cpu_cost_proxy(tree)

    def test_method_dispatch_flops_proxy(self):
        tree = Add(_sym("x"), _num(1))
        n = count_python_bytecode_instructions(
            tree,
            BytecodeComplexityConfig(method="flops_proxy"),
        )
        assert n == count_tree_flops_proxy(tree)


class TestEvalParsimonyIntegration:
    def test_eval_parsimony_python_bytecode(self):
        tree = Add(_sym("x"), _num(1))
        val = eval_parsimony(tree, "tree_python_bytecode_count")
        assert val > 0

    def test_eval_parsimony_python_bytecode_weighted(self):
        tree = Add(_sym("x"), _num(1))
        val = eval_parsimony(tree, "tree_python_bytecode_weighted_count")
        assert val > 0

    def test_eval_parsimony_cpu_cost_proxy(self):
        tree = Add(_sym("x"), _num(1))
        val = eval_parsimony(tree, "tree_cpu_cost_proxy")
        assert val > 0

    def test_eval_parsimony_flops_proxy(self):
        tree = Add(_sym("x"), _num(1))
        val = eval_parsimony(tree, "tree_flops_proxy")
        assert val >= 0

    def test_more_complex_tree_has_higher_cpu_proxy(self):
        simple = Add(_sym("x"), _num(1))
        complex_tree = Sin(Add(Mul(_sym("x"), _sym("y")), _num(2)))

        p_simple = eval_parsimony(simple, "tree_cpu_cost_proxy")
        p_complex = eval_parsimony(complex_tree, "tree_cpu_cost_proxy")

        assert p_complex > p_simple


class TestConfigValidation:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            BytecodeComplexityConfig(method="invalid")

    def test_invalid_execution_model_raises(self):
        with pytest.raises(ValueError):
            BytecodeComplexityConfig(execution_model="parallel_critical_path")
