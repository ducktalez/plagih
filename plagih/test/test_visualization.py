"""
Tests for visualization renderers (tree_renderer, latex_renderer).

Validates that:
- All concrete node classes carry the required _viz_* rendering attributes.
- get_node_style_for_type works for every node type without isinstance fallback.
- get_expr_latex produces valid output for every operator class.
- latex_brackettree doesn't crash on any tree structure.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import sympy

from plagih.trees import (
    Abs,
    Add,
    And,
    BaseOperator,
    Boolean,
    Cos,
    Div,
    Ifte,
    LogicOperator,
    MathOperator,
    Max,
    Min,
    Mul,
    Node,
    Not,
    Number,
    Or,
    Piecewise,
    Pow,
    Sign,
    Sin,
    Sqrt,
    Sub,
    Symbol,
    Terminal,
    Xor,
)
from plagih.util import get_subclasses

# =============================================================================
# Helpers
# =============================================================================

_SYM_A = sympy.Symbol("a", real=True)
_SYM_B = sympy.Symbol("b", real=True)
_SYM_C = sympy.Symbol("c")


def _all_concrete_node_classes():
    """Return all leaf node classes (no further subclasses)."""
    return [cls for cls in get_subclasses(Node) if not cls.__subclasses__()]


def _make_sample_tree(cls):
    """Build a minimal valid tree for *cls* based on its xtype."""
    if not hasattr(cls, "xtype") or not cls.xtype:
        return None
    inputs, _ = cls.xtype
    if not inputs:
        return None

    children = []
    for t in inputs:
        if t is float:
            children.append(Number(sympy.Float(1.5)))
        elif t is bool:
            children.append(Boolean(True))
        else:
            children.append(Number(sympy.Float(1.0)))
    try:
        return cls(*children)
    except Exception:
        return None


# =============================================================================
# _viz_* attribute completeness
# =============================================================================


class TestNodeVizAttributes:
    """Every concrete node class must have _viz_color/border/text/shape."""

    REQUIRED_ATTRS = ("_viz_color", "_viz_border", "_viz_text", "_viz_shape")

    def test_all_concrete_classes_have_viz_attrs(self):
        missing = []
        for cls in _all_concrete_node_classes():
            for attr in self.REQUIRED_ATTRS:
                if not hasattr(cls, attr):
                    missing.append(f"{cls.__name__}.{attr}")
        assert not missing, f"Missing viz attributes: {missing}"

    def test_viz_colors_are_hex_strings(self):
        for cls in _all_concrete_node_classes():
            for attr in ("_viz_color", "_viz_border", "_viz_text"):
                val = getattr(cls, attr, None)
                assert isinstance(val, str), f"{cls.__name__}.{attr} is not str: {val!r}"
                assert val.startswith("#"), f"{cls.__name__}.{attr} is not hex color: {val!r}"

    def test_viz_shape_is_valid(self):
        valid_shapes = {"ellipse", "rounded", "diamond"}
        for cls in _all_concrete_node_classes():
            shape = getattr(cls, "_viz_shape", None)
            assert shape in valid_shapes, f"{cls.__name__}._viz_shape={shape!r} not in {valid_shapes}"

    def test_base_class_hierarchy_colors(self):
        """Ensure base classes define distinct color palettes."""
        assert MathOperator._viz_color != Node._viz_color
        assert LogicOperator._viz_color != MathOperator._viz_color
        assert Number._viz_color != Symbol._viz_color
        assert Boolean._viz_color != Number._viz_color

    def test_terminal_classes_are_ellipse(self):
        for cls in (Number, Symbol, Boolean):
            assert cls._viz_shape == "ellipse", f"{cls.__name__} should be ellipse"

    def test_logic_operator_is_diamond(self):
        assert LogicOperator._viz_shape == "diamond"


# =============================================================================
# tree_renderer: get_node_style_for_type
# =============================================================================


class TestTreeRendererStyles:
    """get_node_style_for_type should return valid styles for all node types."""

    def test_style_for_number(self):
        from plagih.visualization.tree_renderer import get_node_style_for_type

        node = Number(sympy.Float(3.14))
        style = get_node_style_for_type(node)
        assert style.fill_color == Number._viz_color
        assert style.border_color == Number._viz_border

    def test_style_for_symbol(self):
        from plagih.visualization.tree_renderer import get_node_style_for_type

        node = Symbol(_SYM_A)
        style = get_node_style_for_type(node)
        assert style.fill_color == Symbol._viz_color

    def test_style_for_math_operator(self):
        from plagih.visualization.tree_renderer import get_node_style_for_type

        node = Add(Number(sympy.Float(1.0)), Number(sympy.Float(2.0)))
        style = get_node_style_for_type(node)
        assert style.fill_color == MathOperator._viz_color

    def test_style_for_logic_operator(self):
        from plagih.visualization.tree_renderer import get_node_style_for_type

        node = And(Boolean(True), Boolean(False))
        style = get_node_style_for_type(node)
        assert style.fill_color == LogicOperator._viz_color

    def test_all_concrete_classes_return_style(self):
        """get_node_style_for_type must not crash for any node type."""
        from plagih.visualization.tree_renderer import NodeStyle, get_node_style_for_type

        for cls in _all_concrete_node_classes():
            tree = _make_sample_tree(cls)
            if tree is None:
                # Terminal or abstract — test with a terminal
                if issubclass(cls, Terminal):
                    if issubclass(cls, Boolean):
                        tree = cls(True)
                    elif issubclass(cls, Number):
                        tree = cls(sympy.Float(1.0))
                    elif issubclass(cls, Symbol):
                        tree = cls(_SYM_A)
                if tree is None:
                    continue
            style = get_node_style_for_type(tree)
            assert isinstance(style, NodeStyle), f"{cls.__name__} did not return NodeStyle"


# =============================================================================
# latex_renderer: get_expr_latex / latex_brackettree
# =============================================================================


class TestLatexRenderer:
    """LaTeX rendering must work for all node types."""

    def test_terminal_number(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        result = get_expr_latex(Number(sympy.Float(3.14)))
        assert result.startswith("3.14"), f"Expected '3.14…', got {result!r}"

    def test_terminal_symbol(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        assert get_expr_latex(Symbol(_SYM_A)) == "a"

    def test_terminal_bool(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        result = get_expr_latex(Boolean(True))
        assert result in ("True", "true")

    def test_add_inline(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Add(Symbol(_SYM_A), Number(sympy.Float(2.0)))
        result = get_expr_latex(tree)
        assert "+" in result
        assert "a" in result

    def test_mul_inline(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Mul(Symbol(_SYM_A), Symbol(_SYM_B))
        result = get_expr_latex(tree)
        assert r"\cdot" in result

    def test_pow_format(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Pow(Symbol(_SYM_A), Number(sympy.Float(3.0)))
        result = get_expr_latex(tree)
        assert "^" in result
        assert "a" in result

    def test_abs_format(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Abs(Symbol(_SYM_A))
        result = get_expr_latex(tree)
        assert r"\left" in result
        assert "|" in result, f"Abs LaTeX should contain pipe char: {result!r}"

    def test_sqrt_format(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Sqrt(Symbol(_SYM_A))
        result = get_expr_latex(tree)
        assert r"\sqrt" in result

    def test_min_two_children(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Min(Symbol(_SYM_A), Symbol(_SYM_B))
        result = get_expr_latex(tree)
        assert r"\min" in result
        assert "a" in result
        assert "b" in result, f"Min should show both children: {result!r}"

    def test_max_two_children(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Max(Symbol(_SYM_A), Symbol(_SYM_B))
        result = get_expr_latex(tree)
        assert r"\max" in result
        assert "a" in result and "b" in result

    def test_logic_operators(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        assert r"\wedge" in get_expr_latex(And(Boolean(True), Boolean(False)))
        assert r"\vee" in get_expr_latex(Or(Boolean(True), Boolean(False)))
        assert r"\oplus" in get_expr_latex(Xor(Boolean(True), Boolean(False)))

    def test_sub_div_inline(self):
        from plagih.visualization.latex_renderer import get_expr_latex

        sub = Sub(Symbol(_SYM_A), Symbol(_SYM_B))
        assert "-" in get_expr_latex(sub)

        div = Div(Symbol(_SYM_A), Symbol(_SYM_B))
        assert "/" in get_expr_latex(div)

    def test_fallback_function_style(self):
        """Nodes without latex_fmt/latex_inline should use showme(children)."""
        from plagih.visualization.latex_renderer import get_expr_latex

        tree = Sign(Symbol(_SYM_A))
        result = get_expr_latex(tree)
        # Should fall back to function-style: Sign(a) or sign(a)
        assert "a" in result
        assert "(" in result

    def test_nested_expression(self):
        """Complex nested tree should render without crashing."""
        from plagih.visualization.latex_renderer import get_expr_latex

        # sqrt(min(a, b)) * (a + 2)^3
        tree = Mul(
            Sqrt(Min(Symbol(_SYM_A), Symbol(_SYM_B))),
            Pow(Add(Symbol(_SYM_A), Number(sympy.Float(2.0))), Number(sympy.Float(3.0))),
        )
        result = get_expr_latex(tree)
        assert r"\sqrt" in result
        assert r"\min" in result
        assert r"\cdot" in result
        assert "^" in result

    def test_brackettree_force_node(self):
        from plagih.visualization.latex_renderer import latex_brackettree

        tree = Add(Symbol(_SYM_A), Number(sympy.Float(1.0)))
        result = latex_brackettree(tree, force_node=True)
        assert result.startswith("[")
        assert result.endswith("]")
        assert "$" in result  # LaTeX math mode markers

    def test_brackettree_single_expression(self):
        from plagih.visualization.latex_renderer import latex_brackettree

        tree = Add(Symbol(_SYM_A), Number(sympy.Float(1.0)))
        result = latex_brackettree(tree, force_node=False)
        assert result.startswith("[$ ")
        assert result.endswith(" $]")

    def test_all_operators_render_latex(self):
        """get_expr_latex must not crash for any operator with valid children."""
        from plagih.visualization.latex_renderer import get_expr_latex

        for cls in _all_concrete_node_classes():
            tree = _make_sample_tree(cls)
            if tree is None:
                if issubclass(cls, Boolean):
                    tree = cls(True)
                elif issubclass(cls, Number):
                    tree = cls(sympy.Float(1.0))
                elif issubclass(cls, Symbol):
                    tree = cls(_SYM_A)
                if tree is None:
                    continue
            result = get_expr_latex(tree)
            assert isinstance(result, str), f"{cls.__name__}: expected str, got {type(result)}"
            assert len(result) > 0, f"{cls.__name__}: empty LaTeX output"


# =============================================================================
# Digit truncation
# =============================================================================


class TestTexTruncDigits:
    def test_long_decimal(self):
        from plagih.visualization.latex_renderer import _tex_trunc_digits

        assert _tex_trunc_digits("1.23456789") == "1.234"

    def test_very_small(self):
        from plagih.visualization.latex_renderer import _tex_trunc_digits

        assert _tex_trunc_digits("0.00000123") == "0.001"

    def test_integer_unaffected(self):
        from plagih.visualization.latex_renderer import _tex_trunc_digits

        assert _tex_trunc_digits("42") == "42"
