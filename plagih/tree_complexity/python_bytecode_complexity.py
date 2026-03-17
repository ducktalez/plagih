"""Python-bytecode-based tree complexity measures.

This module implements proof-of-concept complexity measures for plagih trees
based on a Python expression generated from a Node tree.

Implemented measures
--------------------
- instruction_count:
    Count CPython bytecode instructions of a compiled lambda expression.
- weighted_instruction_count:
    Same as above, but with rough opcode-class weights.
- cpu_cost_proxy:
    Tree-level heuristic cost proxy for execution effort.
- flops_proxy:
    Tree-level heuristic proxy for floating-point operations.
    Logic and relational operators typically contribute 0 FLOPs.

Important caveats
-----------------
- This is a **proof of concept**, not a hardware-stable execution-cost model.
- CPython bytecode depends on the Python version (e.g. 3.11 / 3.12).
- This is **not** real CPU assembly and **not** a stable machine-code metric.
- Constant folding / compiler normalization may slightly change counts.

Open tasks
----------
- TODO: Add a true Numba/LLVM/ASM backend for JIT-generated machine code.
- TODO: Add a parallel critical-path model:
        in ideal parallel execution, complexity is dominated by the longest
        dependency path, not the sum of all sub-branches.
- TODO: Add branch-sensitive execution models for Ifte / Piecewise:
        only one branch may be executed at runtime.
"""

from __future__ import annotations

import dis
import io
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from plagih.trees import Node


@dataclass
class BytecodeComplexityConfig:
    """Configuration for Python-bytecode-based complexity measures.

    Attributes:
        method:
            - "instruction_count"
            - "weighted_instruction_count"
            - "cpu_cost_proxy"
            - "flops_proxy"
        ignore_opnames:
            Opcodes ignored in the plain/weighted bytecode counts.
            ``RESUME`` and ``CACHE`` are version/compiler artefacts and usually
            do not reflect tree complexity.
        count_return_value:
            Whether ``RETURN_VALUE`` contributes to the count.
        execution_model:
            Currently only ``"sequential_full"`` is implemented.

            Future ideas:
            - ``"parallel_critical_path"``
            - ``"branch_runtime_path"``
    """

    method: str = "instruction_count"
    ignore_opnames: Tuple[str, ...] = ("RESUME", "CACHE")
    count_return_value: bool = True
    execution_model: str = "sequential_full"

    def __post_init__(self) -> None:
        valid_methods = (
            "instruction_count",
            "weighted_instruction_count",
            "cpu_cost_proxy",
            "flops_proxy",
        )
        if self.method not in valid_methods:
            raise ValueError(f"Unknown bytecode complexity method: {self.method!r}")

        valid_models = ("sequential_full",)
        if self.execution_model not in valid_models:
            raise ValueError(
                f"Unsupported execution_model {self.execution_model!r}. Currently supported: {valid_models}"
            )


# ---------------------------------------------------------------------------
# Tree -> Python expression source
# ---------------------------------------------------------------------------


_MATH_FUN_MAP: Dict[str, str] = {
    "Sin": "math.sin",
    "Cos": "math.cos",
    "Tan": "math.tan",
    "Asin": "math.asin",
    "Acos": "math.acos",
    "Atan": "math.atan",
    "Sinh": "math.sinh",
    "Cosh": "math.cosh",
    "Tanh": "math.tanh",
    "Log": "math.log",
    "Exp": "math.exp",
    "Sqrt": "math.sqrt",
    "Abs": "abs",
    "Round": "round",
    "Min": "min",
    "Max": "max",
}


def _terminal_to_python_expr(node: Node) -> str:
    """Translate a terminal node to Python source."""
    cls_name = type(node).__name__
    value = node.get_value()

    if cls_name == "Boolean":
        return "True" if bool(value) else "False"

    if cls_name == "Number":
        try:
            return repr(float(value))
        except Exception:
            return repr(value)

    if cls_name == "Symbol":
        return f"kwargs[{str(value)!r}]"

    return repr(value)


def _xor_fold(children: Sequence[str]) -> str:
    if not children:
        return "False"
    expr = children[0]
    for child in children[1:]:
        expr = f"(({expr}) != ({child}))"
    return expr


def _piecewise_to_python_expr(node: Node) -> str:
    """Translate Piecewise / ExprCondPair_Dummy into nested Python ternaries."""
    pairs = node.get_childs()
    if not pairs:
        return "None"

    rendered_pairs: List[Tuple[str, str]] = []
    for pair in pairs:
        expr_node, cond_node = pair.get_childs()
        rendered_pairs.append((_node_to_python_expr(expr_node), _node_to_python_expr(cond_node)))

    # Last pair is treated as default branch.
    default_expr, _default_cond = rendered_pairs[-1]
    expr = default_expr
    for then_expr, cond_expr in reversed(rendered_pairs[:-1]):
        expr = f"(({then_expr}) if ({cond_expr}) else ({expr}))"
    return expr


def _node_to_python_expr(node: Node) -> str:
    """Recursively translate a plagih tree node to Python expression source."""
    cls_name = type(node).__name__

    if node.is_term():
        return _terminal_to_python_expr(node)

    childs = node.get_childs()
    child_exprs = [_node_to_python_expr(child) for child in childs]

    # Chainable / infix arithmetic
    if cls_name == "Add":
        return "(" + " + ".join(child_exprs) + ")"
    if cls_name == "Mul":
        return "(" + " * ".join(child_exprs) + ")"
    if cls_name == "Sub":
        return f"({child_exprs[0]} - {child_exprs[1]})"
    if cls_name == "Div":
        return f"({child_exprs[0]} / {child_exprs[1]})"
    if cls_name == "DivFraction":
        return f"(1.0 / ({child_exprs[0]}))"
    if cls_name == "Pow":
        return f"(({child_exprs[0]}) ** ({child_exprs[1]}))"
    if cls_name == "PowRounded":
        return f"(({child_exprs[0]}) ** round({child_exprs[1]}))"
    if cls_name == "NthRoot":
        return f"(({child_exprs[0]}) ** (1.0 / ({child_exprs[1]})))"
    if cls_name == "Usub":
        return f"(-({child_exprs[0]}))"
    if cls_name == "Scale":
        return f"(({child_exprs[0]}) * ({child_exprs[1]}))"

    # Logic / comparisons
    if cls_name == "And":
        return "(" + " and ".join(child_exprs) + ")"
    if cls_name == "Or":
        return "(" + " or ".join(child_exprs) + ")"
    if cls_name == "Xor":
        return _xor_fold(child_exprs)
    if cls_name == "Not":
        return f"(not ({child_exprs[0]}))"

    if cls_name == "Eq":
        return f"(({child_exprs[0]}) == ({child_exprs[1]}))"
    if cls_name == "Ne":
        return f"(({child_exprs[0]}) != ({child_exprs[1]}))"
    if cls_name == "Lt":
        return f"(({child_exprs[0]}) < ({child_exprs[1]}))"
    if cls_name == "Le":
        return f"(({child_exprs[0]}) <= ({child_exprs[1]}))"
    if cls_name == "Gt":
        return f"(({child_exprs[0]}) > ({child_exprs[1]}))"
    if cls_name == "Ge":
        return f"(({child_exprs[0]}) >= ({child_exprs[1]}))"

    # Branching
    if cls_name in ("Ifte", "ITE"):
        return f"(({child_exprs[1]}) if ({child_exprs[0]}) else ({child_exprs[2]}))"
    if cls_name == "Piecewise":
        return _piecewise_to_python_expr(node)

    # Named functions
    if cls_name in _MATH_FUN_MAP:
        fun = _MATH_FUN_MAP[cls_name]
        if cls_name in ("Min", "Max"):
            return f"{fun}(" + ", ".join(child_exprs) + ")"
        return f"{fun}({', '.join(child_exprs)})"

    if cls_name == "Exp2":
        return f"(2.0 ** ({child_exprs[0]}))"

    if cls_name == "Clip":
        # expected as Clip(x, lo, hi)
        return f"min(max(({child_exprs[0]}), ({child_exprs[1]})), ({child_exprs[2]}))"

    # Fallback:
    # This still compiles even if the name is not defined, because the lambda
    # body is not executed when we only inspect bytecode.
    return f"{cls_name}(" + ", ".join(child_exprs) + ")"


def get_python_expression_source(tree: Node) -> str:
    """Return a Python expression string for the tree."""
    return _node_to_python_expr(tree)


def get_python_lambda_source(tree: Node) -> str:
    """Return a `lambda **kwargs: ...` source string for the tree."""
    expr = get_python_expression_source(tree)
    return f"lambda **kwargs: ({expr})"


def compile_tree_to_python_lambda(tree: Node):
    """Compile the tree to a Python lambda object for bytecode inspection."""
    src = get_python_lambda_source(tree)
    code = compile(src, "<plagih_tree_bytecode>", "eval")
    return eval(
        code,
        {
            "math": math,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
        },
        {},
    )


# ---------------------------------------------------------------------------
# Bytecode inspection
# ---------------------------------------------------------------------------


def get_python_bytecode_instructions(
    tree: Node,
    config: Optional[BytecodeComplexityConfig] = None,
) -> List[dis.Instruction]:
    """Return filtered CPython bytecode instructions for the compiled tree."""
    if config is None:
        config = BytecodeComplexityConfig()

    fn = compile_tree_to_python_lambda(tree)

    try:
        instructions = list(dis.get_instructions(fn, show_caches=False, adaptive=False))
    except TypeError:
        instructions = list(dis.get_instructions(fn))

    filtered: List[dis.Instruction] = []
    for instr in instructions:
        if instr.opname in config.ignore_opnames:
            continue
        if not config.count_return_value and instr.opname == "RETURN_VALUE":
            continue
        filtered.append(instr)
    return filtered


def get_python_bytecode_disassembly(
    tree: Node,
    config: Optional[BytecodeComplexityConfig] = None,
) -> str:
    """Return a human-readable disassembly string for the compiled tree."""
    if config is None:
        config = BytecodeComplexityConfig()

    fn = compile_tree_to_python_lambda(tree)
    buf = io.StringIO()
    try:
        dis.dis(fn, file=buf, show_caches=False, adaptive=False)
    except TypeError:
        dis.dis(fn, file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tree-level heuristic proxies
# ---------------------------------------------------------------------------


_FLOPS_WEIGHTS: Dict[str, int] = {
    # low/basic arithmetic
    "Add": 1,
    "Sub": 1,
    "Mul": 1,
    "Div": 1,
    "DivFraction": 1,
    "Scale": 1,
    "Usub": 0,
    "Abs": 0,
    "Min": 0,
    "Max": 0,
    "Clip": 0,
    "Round": 0,
    # medium/high numeric effort
    "Pow": 4,
    "PowRounded": 4,
    "NthRoot": 4,
    "Sqrt": 4,
    "Log": 8,
    "Exp": 8,
    "Exp2": 4,
    "Sin": 8,
    "Cos": 8,
    "Tan": 8,
    "Asin": 8,
    "Acos": 8,
    "Atan": 8,
    "Sinh": 8,
    "Cosh": 8,
    "Tanh": 8,
    "Sign": 0,
    # logic / branching / comparisons ≈ 0 FLOPs
    "Eq": 0,
    "Ne": 0,
    "Lt": 0,
    "Le": 0,
    "Gt": 0,
    "Ge": 0,
    "And": 0,
    "Or": 0,
    "Xor": 0,
    "Not": 0,
    "Ifte": 0,
    "ITE": 0,
    "Piecewise": 0,
    "ExprCondPair_Dummy": 0,
}


_CPU_COST_WEIGHTS: Dict[str, int] = {
    # light
    "Number": 0,
    "Symbol": 0,
    "Boolean": 0,
    "Eq": 1,
    "Ne": 1,
    "Lt": 1,
    "Le": 1,
    "Gt": 1,
    "Ge": 1,
    "And": 1,
    "Or": 1,
    "Xor": 1,
    "Not": 1,
    "Add": 2,
    "Sub": 2,
    "Mul": 2,
    "Abs": 1,
    "Min": 1,
    "Max": 1,
    "Usub": 1,
    "Sign": 1,
    # moderate
    "Div": 3,
    "DivFraction": 3,
    "Scale": 2,
    "Round": 2,
    "PowRounded": 3,
    "Clip": 2,
    "Ifte": 2,
    "ITE": 2,
    "Piecewise": 2,
    "ExprCondPair_Dummy": 0,
    # heavy / transcendentals
    "Pow": 5,
    "NthRoot": 5,
    "Sqrt": 4,
    "Log": 5,
    "Exp": 5,
    "Exp2": 4,
    "Sin": 5,
    "Cos": 5,
    "Tan": 5,
    "Asin": 5,
    "Acos": 5,
    "Atan": 5,
    "Sinh": 5,
    "Cosh": 5,
    "Tanh": 5,
}


def _sum_tree_weight(node: Node, weights: Dict[str, int]) -> int:
    total = weights.get(type(node).__name__, 1)
    if node.has_childs():
        total += sum(_sum_tree_weight(child, weights) for child in node.get_childs())
    return total


def count_tree_flops_proxy(tree: Node) -> int:
    """Return a rough FLOPs proxy for a tree.

    Logic operators and comparisons are counted as 0 FLOPs.
    This is a heuristic score, not a real hardware FLOPs measurement.
    """
    return _sum_tree_weight(tree, _FLOPS_WEIGHTS)


def count_tree_cpu_cost_proxy(tree: Node) -> int:
    """Return a rough CPU-cost proxy for a tree.

    This is a heuristic score, not a measured runtime or hardware cycle count.
    """
    return _sum_tree_weight(tree, _CPU_COST_WEIGHTS)


# ---------------------------------------------------------------------------
# Weighted bytecode count
# ---------------------------------------------------------------------------


def _opcode_weight(instr: dis.Instruction) -> float:
    op = instr.opname

    if op in {"LOAD_FAST", "LOAD_CONST", "LOAD_GLOBAL", "LOAD_DEREF", "COPY", "SWAP"}:
        return 0.25
    if op in {"PUSH_NULL", "PRECALL"}:
        return 0.25
    if op in {"RETURN_VALUE"}:
        return 0.25
    if op in {"UNARY_NEGATIVE", "UNARY_POSITIVE", "UNARY_NOT"}:
        return 0.5
    if op in {"BINARY_OP", "COMPARE_OP"}:
        return 1.0
    if "JUMP" in op or op.startswith("POP_JUMP"):
        return 0.75
    if "CALL" in op:
        return 2.0

    return 1.0


def count_python_bytecode_instructions(
    tree: Node,
    config: Optional[BytecodeComplexityConfig] = None,
) -> int:
    """Return a configured Python-bytecode-based complexity score."""
    if config is None:
        config = BytecodeComplexityConfig()

    if config.method == "cpu_cost_proxy":
        return int(count_tree_cpu_cost_proxy(tree))

    if config.method == "flops_proxy":
        return int(count_tree_flops_proxy(tree))

    instructions = get_python_bytecode_instructions(tree, config)

    if config.method == "instruction_count":
        return len(instructions)

    if config.method == "weighted_instruction_count":
        return round(sum(_opcode_weight(instr) for instr in instructions))

    raise ValueError(f"Unsupported bytecode complexity method: {config.method!r}")
