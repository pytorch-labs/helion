"""Device print support for Helion kernels."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import create
from .._compiler.ast_extension import expr_from_string
from . import _decorators

if TYPE_CHECKING:
    import torch

    from .._compiler.inductor_lowering import CodegenState


@has_side_effect
@_decorators.api()
def device_print(prefix: str, *values: torch.Tensor) -> None:
    """
    Print values from device code.

    :param prefix: A string prefix for the print statement
    :param values: Tensor values to print
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(device_print)
def _(prefix: str, *values: object) -> None:
    return None


@_decorators.codegen(device_print)
def _(state: CodegenState) -> None:
    if len(state.proxy_args) < 1:
        raise ValueError("device_print requires at least a prefix argument")

    prefix = state.proxy_arg(0)
    if not isinstance(prefix, str):
        raise TypeError(f"device_print prefix must be a string, got {type(prefix)}")

    call_args = [create(ast.Constant, value=prefix)]

    # Handle varargs - they come as a tuple in the second argument due to *args
    # TODO NOW: make this look nicer and cleaner
    if len(state.proxy_args) > 1:
        # The corresponding AST nodes should be in ast_args[1]
        if len(state.ast_args) > 1:
            ast_varargs = state.ast_args[1]

            # If ast_varargs is a single tuple containing the AST nodes
            if (
                isinstance(ast_varargs, tuple)
                and len(ast_varargs) == 1
                and isinstance(ast_varargs[0], tuple)
            ):
                # Unwrap the nested tuple
                ast_nodes = ast_varargs[0]
                for ast_node in ast_nodes:
                    if isinstance(ast_node, ast.AST):
                        call_args.append(ast_node)
            # If ast_varargs is directly the tuple of AST nodes
            elif isinstance(ast_varargs, tuple):
                for ast_node in ast_varargs:
                    if isinstance(ast_node, ast.AST):
                        call_args.append(ast_node)

    call_expr = create(
        ast.Call,
        func=expr_from_string("tl.device_print"),
        args=call_args,
        keywords=[],
    )
    stmt = create(ast.Expr, value=call_expr)
    state.add_statement(stmt)
