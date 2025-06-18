from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState
    from .loops import Tile


@_decorators.api(tiles_as_sizes=True)
def tile_index(tile: Tile) -> torch.Tensor:
    """
    Retrieve the index (a 1D tensor containing offsets) of the given tile.
    This can also be written as: `tile.index`.

    Example usage:

        @helion.kernel
        def arange(length: int, device: torch.device) -> torch.Tensor:
            out = torch.empty(length, dtype=torch.int32, device=device)
            for tile in hl.tile(length):
               out[tile] = tile.index
            return out
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_index)
def _(tile: torch.SymInt) -> torch.Tensor:
    assert isinstance(tile, torch.SymInt)
    env = CompileEnvironment.current()
    assert env.get_block_id(tile) is not None
    return torch.empty([tile], dtype=env.settings.index_dtype, device=env.device)


@_decorators.codegen(tile_index)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    return expr_from_string(state.codegen.index_var(index))


@_decorators.api(tiles_as_sizes=True)
def tile_begin(tile: Tile) -> int:
    """
    Retrieve the start offset of the given tile.
    This can also be written as: `tile.begin`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_begin)
def _(tile: torch.SymInt) -> torch.SymInt:
    _disable_flatten_get_tile(tile)  # update config spec if needed
    return CompileEnvironment.current().create_unbacked_symint()


def _disable_flatten_get_tile(tile: object) -> int:
    """Helper to extract tile index from state."""
    assert isinstance(tile, torch.SymInt), (type(type), tile)
    env = CompileEnvironment.current()
    index = env.get_block_id(tile)
    assert index is not None
    # The functions in this file can't be used in flattened loops.
    env.config_spec.flatten_loops.disable_block_id(index)
    return index


@_decorators.codegen(tile_begin)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    return expr_from_string(state.codegen.offset_var(index))


@_decorators.api(tiles_as_sizes=True)
def tile_end(tile: Tile) -> int:
    """
    Retrieve the end offset of the given tile.
    For the first 0 to N-1 tiles, this is equivalent to `tile.begin + tile.block_size`.
    For the last tile, this is the end offset passed to `hl.tile()`.
    This can also be written as: `tile.end`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_end)
def _(tile: torch.SymInt) -> torch.SymInt:
    _disable_flatten_get_tile(tile)  # update config spec if needed
    return CompileEnvironment.current().create_unbacked_symint()


@_decorators.codegen(tile_end)
def _(state: CodegenState) -> ast.AST:
    index = _disable_flatten_get_tile(state.proxy_arg(0))
    offset_var = state.codegen.offset_var(index)
    block_size_var = state.device_function.block_size_var(index)
    if block_size_var is None:
        block_size_var = "1"
    naive_exp = f"{offset_var} + {block_size_var}"
    if state.codegen.mask_var(index) is not None:
        # if masking is used, we must update the end bound of the last tile
        end_var = (
            state.codegen.active_device_loops[index][-1]
            .block_id_to_info[index]
            .end_var_name
        )
        return expr_from_string(f"tl.minimum({naive_exp}, {end_var})")
    # If we don't have a mask, we can simply return the offset + block size
    return expr_from_string(naive_exp)


@_decorators.api(tiles_as_sizes=True)
def tile_block_size(tile: Tile) -> int:
    """
    Retrieve block size of a given tile, usually set the autotuner.
    This can also be written as: `tile.block_size`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_block_size)
def _(tile: torch.SymInt) -> torch.SymInt:
    return tile


# since we return tile above, no codegen is needed for this function.
# codegen is handled in _get_symnode()


@_decorators.api(tiles_as_sizes=True)
def tile_id(tile: Tile) -> int:
    """
    Retrieve tile_id of a given tile or list of tiles.
    This is equivalent to `tile.begin // tile.block_size`.
    This can also be written as: `tile.id`.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(tile_id)
def _(tile: torch.SymInt) -> torch.SymInt:
    assert isinstance(tile, torch.SymInt)
    return CompileEnvironment.current().create_unbacked_symint()


@_decorators.codegen(tile_id)
def _(state: CodegenState) -> ast.AST:
    t = state.proxy_arg(0)
    env = CompileEnvironment.current()
    assert isinstance(t, torch.SymInt)
    index = env.get_block_id(t)
    assert index is not None
    # disable_flatten:
    # The functions in this file can't be used in flattened loops.
    env.config_spec.flatten_loops.disable_block_id(index)
    offset = state.codegen.offset_var(index)

    block_size = state.device_function.block_size_var(index)
    if block_size is None:
        expr_str = offset
    else:
        expr_str = f"{offset} // {block_size}"
    return expr_from_string(expr_str)
