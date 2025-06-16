from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState


@has_side_effect
@_decorators.api(tiles_as_sizes=True)
def wait(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    op: str = "ld",
    sem: str = "acquire",
    scope: str = "sys",
) -> None:
    """
    Wait for a signal before accessing the data tensor.
    Args:
        signal_pad: The signal tensor to wait on
        index: Indices into signal_pad tensor for which signal to wait for
        signal: the signal to wait for
        update: update the signal_pad after acquiring the signal.
        sem: The memory op for acquring the lock (default: 'ld.acquire')

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait)
def _(*args: object, origin: object, **kwargs: object) -> object:
    from .._compiler.type_propagation import NoType

    return NoType(origin=origin)


@_decorators.prepare_args(wait)
def _(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    op: str = "ld",
    sem: str = "acquire",
    scope: str = "sys",
) -> tuple[torch.Tensor, object, int, int | None, str, str]:
    from helion._compiler.tile_index_proxy import TileIndexProxy

    print("in wait prepare_args")

    valid_ops = {"ld", "atomic_add", "atomic_cas"}
    valid_sems = {"relaxed", "acquire", "acq_rel"}
    valid_scopes = {"sys", "gpu"}

    if op not in valid_ops:
        raise ValueError(f"Invalid Wait op '{op}'. Must be one of {valid_ops}. ")

    if sem == "release":
        raise ValueError(
            f"Do not use '{sem}' for wait patterns. Wait sem must be one of {valid_sems}."
        )

    if sem not in valid_sems:
        raise ValueError(
            f"Invalid memory semantic '{sem}'. Must be one of {valid_sems}."
        )

    if op == "atomic_cas" and not update:
        raise ValueError(
            f"{op} without an update value. Do you want to use 'ld' instead? "
        )

    if op == "ld":
        assert update is None
        update = 0

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    print("index:", index)

    index = TileIndexProxy.prepare_index(index)
    print("index prepare_index", index)
    index = TileIndexProxy.tiles_to_sizes(index)

    print("tiles_to_sizes", index)

    print("tiles_to_sizes", type(index[0]))

    return (signal_pad, index, signal, update, op, sem)


@_decorators.register_fake(wait)
def _(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    update: int | None = None,
    op: str = "ld",
    sem: str = "acquire",
    scope: str = "sys",
) -> None:
    return None


def get_lock_spin_ptx(name: str, op: str, sem: str, scope: str):
    ptx_acquire_list = {
        "ld": f"ld.global.{sem}.{scope}.u32 $0, [$1];",
        "atomic_cas": f"atom.global.{sem}.{scope}.cas.b32 $0, [$1], $2, $3;",
        "atomic_add": f"atom.global.{sem}.{scope}.add.u32 $0, [$1], $2;",
    }

    acquire_lock_expr = ptx_acquire_list[op]
    ptx_template = f'''
tl.inline_asm_elementwise("""
    {{
        .reg .u32   %tmp32_<1>;
        .reg .pred  %p<2>;

        // calculate tid assuming tid.y=tid.z=1. TODO: get this from Triton
        mov.u32 %tmp32_0, %tid.x;
        setp.eq.s32 %p1, %tmp32_0, 0;

        mov.u32 $0, 0;
        // initialize tmp_0 to 0
        wait_block:
            @%p1 {acquire_lock_expr}
            setp.ne.u32 %p0, $0, $2;
            and.pred %p0, %p0, %p1;
            @%p0 bra wait_block;
        bar.sync 0;
    }}
    """,
    "=r, l, r, r",
    [{name} + offset, signal, update],
    dtype={name}.dtype.element_ty,
    is_pure=False,
    pack=1,
)
'''
    print("ptx_template", ptx_template)
    return ptx_template


@_decorators.codegen(wait)
def _(state: CodegenState) -> ast.AST:
    import ast

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.indexing_strategy import SubscriptIndexing

    signal_pad = state.proxy_arg(0)
    index = state.proxy_arg(1)
    signal = state.proxy_arg(2)
    update = state.proxy_arg(3)
    op = state.proxy_arg(4)
    sem = state.proxy_arg(5)
    scope = state.proxy_arg(6)

    assert isinstance(signal_pad, torch.Tensor)
    assert isinstance(index, (list))

    print(index, "index")
    indices = SubscriptIndexing.create(state, signal_pad, index)
    print("indices", indices)
    signal_pad_name = state.device_function.tensor_arg(signal_pad).name

    signal_expr = ast.Constant(value=signal)
    update_expr = ast.Constant(value=update)

    lock_spin_ptx = get_lock_spin_ptx(signal_pad_name, op, sem, scope)

    return expr_from_string(
        lock_spin_ptx,
        offset=indices.index_expr,
        signal=signal_expr,
        update=update_expr,
    )


@has_side_effect
@_decorators.api(tiles_as_sizes=True)
def signal(
    signal_pad: torch.Tensor,
    index: list[object],
    signal: int = 1,
    sem: str = "ld.acquire",
) -> None:
    """
    Wait for a signal before accessing the data tensor.
    Args:
        signal_pad: The signal tensor to wait on
        index: Indices into signal_pad tensor for which signal to wait for
        signal: the signal to wait for
        update: update the signal_pad after acquiring the signal.
        sem: The memory op for acquring the lock (default: 'ld.acquire')

    Returns:
        None
    """
    raise exc.NotInsideKernel
