"""
Exception classes for the plagih GP framework.

Extracted from util.py to break import chains — these are stable and
imported by almost every module.
"""


class TreeError(Exception):
    """All Tree-specific errors."""

    pass


class TreeLutError(TreeError):
    """Errors regarding lookup-tables for trees."""

    pass


class TreeSizeError(TreeError):
    """Tree size constraint violations (too small after simplification,
    too many nodes, etc.)."""

    pass


class SympyError(Exception):
    """SymPy evaluation errors (imaginary numbers, zoo, inf, nan)."""

    pass


class SympyImaginaryNumber(SympyError):
    """Imaginary number appeared in a SymPy expression."""

    pass


class CuriosityError(Exception):
    """Code that should never be reached — kept for debugging."""

    pass
