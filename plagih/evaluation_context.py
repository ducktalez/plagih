"""
Unified Evaluation Context System for Plagih GP Trees.

This module provides an OPTIONAL and BACKWARD-COMPATIBLE evaluation framework
that unifies the three existing evaluation methods:
1. SymPy evaluation (get_sympy_expr) - symbolic, precise
2. NumPy eager evaluation (eval_predict_numpy_now) - fast, debug-friendly
3. NumPy lambda evaluation (eval_np_lambdas) - graph-based, reusable

WICHTIG: Die bisherigen Implementierungen in trees.py bleiben UNVERÄNDERT!
Dieses System ist nur eine zusätzliche Option für vereinheitlichte Evaluation.

Usage:
    >>> from plagih.evaluation_context import EvaluationContext
    >>> context = EvaluationContext(modes=["numpy_eager"], use_lut=True)
    >>> result = context.evaluate(tree, df)

    # Multiple modes at once:
    >>> context = EvaluationContext(modes=["sympy", "numpy_eager", "numpy_lambda"])
    >>> results = context.evaluate(tree, df)

Author: Generated with assistance
Date: 2026-01-27
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import sympy

if TYPE_CHECKING:
    from plagih.trees import Node


# =============================================================================
# Enums and Data Classes
# =============================================================================


class EvalMode(str, Enum):
    """Available evaluation modes."""

    SYMPY = "sympy"
    NUMPY_EAGER = "numpy_eager"
    NUMPY_LAMBDA = "numpy_lambda"


@dataclass
class EvaluationResult:
    """Container for evaluation results from multiple modes.

    Attributes:
        sympy: SymPy expression result (if mode was enabled)
        numpy_eager: NumPy array result from eager evaluation
        numpy_lambda: Callable lambda function for lazy evaluation
        errors: Dict of mode -> error message for failed evaluations
    """

    sympy: Optional[sympy.Basic] = None
    numpy_eager: Optional[np.ndarray] = None
    numpy_lambda: Optional[Callable[[pd.DataFrame], np.ndarray]] = None
    errors: Dict[str, str] = field(default_factory=dict)

    def get(self, mode: str) -> Any:
        """Get result for specific mode."""
        return getattr(self, mode.replace("-", "_"), None)

    def has_error(self, mode: str) -> bool:
        """Check if mode had an error."""
        return mode in self.errors

    def successful_modes(self) -> List[str]:
        """Get list of modes that succeeded."""
        modes = []
        if self.sympy is not None:
            modes.append("sympy")
        if self.numpy_eager is not None:
            modes.append("numpy_eager")
        if self.numpy_lambda is not None:
            modes.append("numpy_lambda")
        return modes


# =============================================================================
# Main EvaluationContext Class
# =============================================================================


class EvaluationContext:
    """Unified evaluation context for parallel evaluation modes.

    This class provides a single interface to evaluate trees in multiple modes
    simultaneously, with optional LUT caching and statistics tracking.

    BACKWARD-COMPATIBLE: The existing methods (get_sympy_expr, eval_predict_numpy_now,
    eval_np_lambdas) continue to work unchanged. This is an ADDITIONAL option.

    Features:
    - Evaluate in one or multiple modes simultaneously
    - Optional LUT caching per mode
    - Statistics tracking (evaluations, cache hits)
    - Prepared for future gradient tracking (backpropagation)

    Example:
        >>> # Single mode (most common usage)
        >>> context = EvaluationContext(modes=["numpy_eager"])
        >>> result = context.evaluate(tree, df)
        >>> assert isinstance(result, np.ndarray)

        >>> # Multiple modes at once
        >>> context = EvaluationContext(modes=["sympy", "numpy_eager", "numpy_lambda"])
        >>> results = context.evaluate(tree, df)
        >>> assert "sympy" in results and "numpy_eager" in results

        >>> # With LUT disabled (for memory-constrained environments)
        >>> context = EvaluationContext(modes=["numpy_eager"], use_lut=False)
    """

    # Valid mode names
    VALID_MODES = {"sympy", "numpy_eager", "numpy_lambda"}

    def __init__(
        self,
        modes: Optional[List[str]] = None,
        use_lut: bool = True,
        track_gradients: bool = False,
        df: Optional[pd.DataFrame] = None,
    ):
        """Initialize evaluation context.

        Args:
            modes: List of evaluation modes to use. Options:
                - 'sympy': SymPy symbolic evaluation
                - 'numpy_eager': Direct NumPy evaluation (eager, debug-friendly)
                - 'numpy_lambda': NumPy lambda (lazy, graph-based) evaluation
                Default: ['numpy_eager']
            use_lut: Whether to cache results in context-local LUT.
                Set to False for memory-constrained environments.
            track_gradients: Whether to track gradients for later backpropagation.
                (Future feature - requires JAX or PyTorch, not yet implemented)
            df: Optional DataFrame to store for evaluation. If provided,
                numpy_eager evaluations can be called without passing df.
        """
        if modes is None:
            modes = ["numpy_eager"]

        # Validate modes
        for mode in modes:
            if mode not in self.VALID_MODES:
                raise ValueError(f"Invalid mode '{mode}'. Valid modes: {self.VALID_MODES}")

        self.modes: Set[str] = set(modes)
        self.use_lut = use_lut
        self.track_gradients = track_gradients
        self.df = df

        # Per-mode LUT caches: mode → {tree_id → result}
        self._lut: Dict[str, Dict[str, Any]] = {mode: {} for mode in self.modes}

        # Statistics tracking
        self._stats = {
            "evaluations": {mode: 0 for mode in self.modes},
            "cache_hits": {mode: 0 for mode in self.modes},
            "cache_misses": {mode: 0 for mode in self.modes},
            "errors": {mode: 0 for mode in self.modes},
        }

        # Gradient storage (for future backprop support)
        self._gradients: Dict[str, Any] = {}

    # =========================================================================
    # Context Configuration (Fluent Interface)
    # =========================================================================

    def with_modes(self, modes: List[str]) -> EvaluationContext:
        """Create a new context with different modes (LUT is shared for common modes).

        Args:
            modes: New list of modes

        Returns:
            New EvaluationContext with specified modes
        """
        new_ctx = EvaluationContext(modes=modes, use_lut=self.use_lut, track_gradients=self.track_gradients, df=self.df)
        # Share LUT for modes that exist in both
        for mode in modes:
            if mode in self._lut:
                new_ctx._lut[mode] = self._lut[mode]
        return new_ctx

    def with_df(self, df: pd.DataFrame) -> EvaluationContext:
        """Create a new context with a specific DataFrame.

        Args:
            df: DataFrame to use for numpy evaluations

        Returns:
            New EvaluationContext with specified DataFrame
        """
        new_ctx = EvaluationContext(
            modes=list(self.modes), use_lut=self.use_lut, track_gradients=self.track_gradients, df=df
        )
        new_ctx._lut = self._lut  # Share LUT
        return new_ctx

    def with_lut(self, use_lut: bool) -> EvaluationContext:
        """Create a new context with LUT enabled/disabled.

        Args:
            use_lut: Whether to use LUT caching

        Returns:
            New EvaluationContext with specified LUT setting
        """
        new_ctx = EvaluationContext(
            modes=list(self.modes), use_lut=use_lut, track_gradients=self.track_gradients, df=self.df
        )
        if use_lut:
            new_ctx._lut = self._lut  # Share LUT if enabled
        return new_ctx

    # =========================================================================
    # LUT Management
    # =========================================================================

    def get_cached(self, mode: str, tree_id: str) -> Optional[Any]:
        """Get cached result for a tree in a specific mode.

        Args:
            mode: Evaluation mode
            tree_id: Tree identifier (from node.get_lut_id())

        Returns:
            Cached result or None if not cached
        """
        if not self.use_lut:
            return None
        return self._lut.get(mode, {}).get(tree_id)

    def set_cached(self, mode: str, tree_id: str, result: Any) -> None:
        """Cache a result for a tree in a specific mode.

        Args:
            mode: Evaluation mode
            tree_id: Tree identifier
            result: Result to cache
        """
        if self.use_lut and mode in self._lut:
            self._lut[mode][tree_id] = result

    def clear_cache(self, mode: Optional[str] = None) -> None:
        """Clear LUT cache for one or all modes.

        Args:
            mode: Specific mode to clear, or None for all modes
        """
        if mode is None:
            for m in self._lut:
                self._lut[m].clear()
        elif mode in self._lut:
            self._lut[mode].clear()

    def get_cache_size(self) -> Dict[str, int]:
        """Get number of cached entries per mode.

        Returns:
            Dict mapping mode name to cache size
        """
        return {mode: len(cache) for mode, cache in self._lut.items()}

    # =========================================================================
    # Statistics
    # =========================================================================

    def _record_evaluation(self, mode: str, cache_hit: bool = False) -> None:
        """Record an evaluation for statistics (internal).

        Args:
            mode: Evaluation mode
            cache_hit: Whether this was a cache hit
        """
        self._stats["evaluations"][mode] = self._stats["evaluations"].get(mode, 0) + 1
        if cache_hit:
            self._stats["cache_hits"][mode] = self._stats["cache_hits"].get(mode, 0) + 1
        else:
            self._stats["cache_misses"][mode] = self._stats["cache_misses"].get(mode, 0) + 1

    def _record_error(self, mode: str) -> None:
        """Record an evaluation error for statistics (internal).

        Args:
            mode: Evaluation mode that errored
        """
        self._stats["errors"][mode] = self._stats["errors"].get(mode, 0) + 1

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get evaluation statistics.

        Returns:
            Dict with 'evaluations', 'cache_hits', 'cache_misses', 'errors' per mode
        """
        return {k: dict(v) for k, v in self._stats.items()}

    def get_cache_hit_rate(self, mode: str) -> float:
        """Get cache hit rate for a specific mode (0.0 to 1.0).

        Args:
            mode: Evaluation mode

        Returns:
            Cache hit rate as float between 0 and 1
        """
        total = self._stats["evaluations"].get(mode, 0)
        hits = self._stats["cache_hits"].get(mode, 0)
        return hits / total if total > 0 else 0.0

    def summary(self) -> str:
        """Get a summary string of context statistics.

        Returns:
            Human-readable summary string
        """
        lines = ["EvaluationContext Statistics:"]
        lines.append(f"  Modes: {sorted(self.modes)}")
        lines.append(f"  LUT enabled: {self.use_lut}")

        for mode in sorted(self.modes):
            evals = self._stats["evaluations"].get(mode, 0)
            hits = self._stats["cache_hits"].get(mode, 0)
            errors = self._stats["errors"].get(mode, 0)
            hit_rate = self.get_cache_hit_rate(mode)
            cache_size = len(self._lut.get(mode, {}))
            lines.append(f"  {mode}:")
            lines.append(f"    Evaluations: {evals}, Cache hits: {hits} ({hit_rate:.1%})")
            lines.append(f"    Errors: {errors}, Cache size: {cache_size}")

        return "\n".join(lines)

    # =========================================================================
    # Main Evaluation Entry Point
    # =========================================================================

    def evaluate(
        self, node: Node, df: Optional[pd.DataFrame] = None, single_mode: Optional[str] = None
    ) -> Union[Any, Dict[str, Any]]:
        """Evaluate a node tree in one or multiple modes.

        This is the main entry point for unified evaluation.
        WICHTIG: Diese Methode DELEGIERT zu den bestehenden Methoden in Node!

        Args:
            node: The tree node to evaluate
            df: DataFrame with input data (required for numpy_eager mode)
            single_mode: If specified, only evaluate this mode and return
                the result directly (not wrapped in dict)

        Returns:
            If single_mode specified or only one mode active:
                The evaluation result directly (sympy.Basic, np.ndarray, or Callable)
            If multiple modes:
                Dict mapping mode name to result

        Raises:
            ValueError: If numpy_eager mode requested but no df provided

        Example:
            >>> ctx = EvaluationContext(modes=["numpy_eager"])
            >>> result = ctx.evaluate(tree, df)  # Returns np.ndarray

            >>> ctx = EvaluationContext(modes=["sympy", "numpy_eager"])
            >>> results = ctx.evaluate(tree, df)  # Returns dict
            >>> results["sympy"]  # sympy.Basic
            >>> results["numpy_eager"]  # np.ndarray
        """
        # Use stored df if not provided
        if df is None:
            df = self.df

        # Determine which modes to evaluate
        modes_to_eval = [single_mode] if single_mode else list(self.modes)

        # Validate df for numpy_eager mode
        if "numpy_eager" in modes_to_eval and df is None:
            raise ValueError(
                "DataFrame (df) is required for numpy_eager evaluation. "
                "Pass df to evaluate() or use context.with_df(df)."
            )

        results = {}
        tree_id = node.get_lut_id()

        for mode in modes_to_eval:
            # Check cache first
            cached = self.get_cached(mode, tree_id)
            if cached is not None:
                self._record_evaluation(mode, cache_hit=True)
                results[mode] = cached
                continue

            # Evaluate based on mode - DELEGATING to existing methods!
            try:
                if mode == "sympy":
                    result = node.get_sympy_expr()
                elif mode == "numpy_eager":
                    result = node.eval_predict_numpy_now(df)
                elif mode == "numpy_lambda":
                    result = node.eval_np_lambdas()
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Cache and record
                self.set_cached(mode, tree_id, result)
                self._record_evaluation(mode, cache_hit=False)
                results[mode] = result

            except Exception:
                self._record_error(mode)
                # Re-raise - caller decides how to handle
                raise

        # Return single value if only one mode
        if len(results) == 1:
            return list(results.values())[0]
        return results

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def eval_sympy(self, node: Node) -> sympy.Basic:
        """Convenience: Evaluate only in sympy mode.

        Args:
            node: Tree node to evaluate

        Returns:
            SymPy expression
        """
        return self.evaluate(node, single_mode="sympy")

    def eval_numpy(self, node: Node, df: pd.DataFrame) -> np.ndarray:
        """Convenience: Evaluate only in numpy_eager mode.

        Args:
            node: Tree node to evaluate
            df: Input DataFrame

        Returns:
            NumPy array with results
        """
        return self.evaluate(node, df=df, single_mode="numpy_eager")

    def eval_lambda(self, node: Node) -> Callable[[pd.DataFrame], np.ndarray]:
        """Convenience: Evaluate only in numpy_lambda mode.

        Args:
            node: Tree node to evaluate

        Returns:
            Callable that takes DataFrame and returns np.ndarray
        """
        return self.evaluate(node, single_mode="numpy_lambda")

    def eval_all(self, node: Node, df: pd.DataFrame) -> EvaluationResult:
        """Evaluate in all three modes and return structured result.

        This is useful for comparison/debugging purposes.

        Args:
            node: Tree node to evaluate
            df: Input DataFrame

        Returns:
            EvaluationResult with all three results (or errors)
        """
        result = EvaluationResult()

        # SymPy
        try:
            result.sympy = node.get_sympy_expr()
        except Exception as e:
            result.errors["sympy"] = str(e)

        # NumPy Eager
        try:
            result.numpy_eager = node.eval_predict_numpy_now(df)
        except Exception as e:
            result.errors["numpy_eager"] = str(e)

        # NumPy Lambda
        try:
            result.numpy_lambda = node.eval_np_lambdas()
        except Exception as e:
            result.errors["numpy_lambda"] = str(e)

        return result

    # =========================================================================
    # Future: Gradient Tracking for Backpropagation
    # =========================================================================

    def enable_gradient_tracking(self) -> EvaluationContext:
        """Enable gradient tracking for backpropagation.

        NOTE: This is a placeholder for future implementation.
        Requires integration with JAX or PyTorch for autodiff.

        Returns:
            Self for method chaining
        """
        warnings.warn(
            "Gradient tracking is not yet implemented. This is a placeholder for future JAX/PyTorch integration.",
            FutureWarning,
        )
        self.track_gradients = True
        return self

    def get_gradients(self) -> Dict[str, Any]:
        """Get tracked gradients (future feature).

        Returns:
            Dict of gradients (empty until backprop is implemented)
        """
        return self._gradients.copy()


# =============================================================================
# Utility Functions
# =============================================================================


def create_context(
    mode: str = "numpy_eager", use_lut: bool = True, df: Optional[pd.DataFrame] = None
) -> EvaluationContext:
    """Factory function for creating EvaluationContext with a single mode.

    This is a convenience function for the common case of evaluating
    in a single mode.

    Args:
        mode: Evaluation mode ('sympy', 'numpy_eager', or 'numpy_lambda')
        use_lut: Whether to enable LUT caching
        df: Optional DataFrame for numpy evaluations

    Returns:
        Configured EvaluationContext

    Example:
        >>> ctx = create_context("numpy_eager", use_lut=True)
        >>> result = ctx.evaluate(tree, df)
    """
    return EvaluationContext(modes=[mode], use_lut=use_lut, df=df)


def evaluate_tree(
    node: Node, df: Optional[pd.DataFrame] = None, mode: str = "numpy_eager", use_lut: bool = False
) -> Any:
    """One-shot tree evaluation without creating a context explicitly.

    This is the simplest way to use the unified evaluation system.
    For repeated evaluations, prefer creating an EvaluationContext.

    Args:
        node: Tree node to evaluate
        df: DataFrame with input data (required for numpy_eager mode)
        mode: Evaluation mode
        use_lut: Whether to cache (usually False for one-shot)

    Returns:
        Evaluation result (type depends on mode)

    Example:
        >>> result = evaluate_tree(tree, df, mode="numpy_eager")
        >>> lambda_fn = evaluate_tree(tree, mode="numpy_lambda")
    """
    ctx = EvaluationContext(modes=[mode], use_lut=use_lut, df=df)
    return ctx.evaluate(node)


# =============================================================================
# Integration Helper for Node class
# =============================================================================


def add_unified_evaluation_to_node(node_class: type) -> None:
    """Add the evaluate_unified method to a Node class.

    This function can be called to add the unified evaluation method
    to the Node class without modifying trees.py directly.

    Usage:
        >>> from plagih.trees import Node
        >>> from plagih.evaluation_context import add_unified_evaluation_to_node
        >>> add_unified_evaluation_to_node(Node)
        >>> # Now all nodes have evaluate_unified method
        >>> result = tree.evaluate_unified(context, df)

    Args:
        node_class: The Node class to extend
    """

    def evaluate_unified(
        self, context: EvaluationContext, df: Optional[pd.DataFrame] = None
    ) -> Union[Any, Dict[str, Any]]:
        """Evaluate this node using the unified EvaluationContext.

        This is an OPTIONAL method that complements the existing
        evaluation methods. The old methods still work unchanged.

        Args:
            context: EvaluationContext with modes and settings
            df: DataFrame for numpy evaluations

        Returns:
            Evaluation result(s) based on context modes
        """
        return context.evaluate(self, df=df)

    # Add method to class
    node_class.evaluate_unified = evaluate_unified


# =============================================================================
# Module-level Default Context
# =============================================================================

_default_context: Optional[EvaluationContext] = None


def get_default_context() -> EvaluationContext:
    """Get or create the default evaluation context.

    Returns:
        Default EvaluationContext (numpy_eager mode, LUT enabled)
    """
    global _default_context
    if _default_context is None:
        _default_context = EvaluationContext(modes=["numpy_eager"])
    return _default_context


def set_default_context(context: EvaluationContext) -> None:
    """Set the default evaluation context.

    Args:
        context: New default context
    """
    global _default_context
    _default_context = context
