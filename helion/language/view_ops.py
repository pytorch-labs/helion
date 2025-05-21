from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import torch  # noqa: TC002

from .. import exc
from .._compiler.ast_extension import expr_from_string
from ..exc import NotInsideKernel
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState


@_decorators.api(tiles_as_sizes=True)
def subscript(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    """
    Equivalent to tensor[index] where tensor is a kernel-tensor (not a host-tensor).

    Can be used to add dimensions to the tensor, e.g. tensor[None, :] or tensor[:, None].
    """
    raise NotInsideKernel


@_decorators.register_fake(subscript)
def _(tensor: torch.Tensor, index: list[object]) -> torch.Tensor:
    input_size = collections.deque(tensor.size())
    output_size = []
    for val in index:
        if val is None:
            output_size.append(1)
        elif isinstance(val, slice) and repr(val) == "slice(None, None, None)":
            output_size.append(input_size.popleft())
        elif isinstance(val, torch.Tensor) and val.ndim == 1:
            input_size.popleft()
            output_size.append(val.size(0))
        else:
            raise exc.InvalidIndexingType(repr(val))
    assert len(input_size) == 0
    return tensor.new_empty(output_size)


@_decorators.codegen(subscript)
def _(state: CodegenState) -> ast.AST:
    output_keys = []
    # pyre-ignore[16]
    for val in state.proxy_arg(1):
        if val is None:
            output_keys.append("None")
        elif isinstance(val, slice) and repr(val) == "slice(None, None, None)":
            output_keys.append(":")
        else:
            raise exc.InvalidIndexingType(repr(val))
    return expr_from_string(
        f"base[{', '.join(output_keys)}]",
        base=state.ast_arg(0),
    )
