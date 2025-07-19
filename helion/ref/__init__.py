"""Ref - Runtime replacement approach for ref mode execution.

This package provides a minimal runtime replacement for helion.language that
preserves the original kernel syntax while enabling ref mode execution.
"""

from . import runtime

__all__ = ['runtime']