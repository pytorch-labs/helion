"""Runtime module for ref - acts as a drop-in replacement for helion.language.

This module provides all the functions and classes that would normally be in
helion.language (hl.*) but with ref mode implementations.
"""

# Import torch for namespace availability (kernels use torch.* directly)
import torch

# Import utilities from tensor_ops
from .tensor_ops import (
    mixed_precision_patch_context,
)

# Import all hl APIs
from .hl_apis import (
    # Creation ops (part of hl.* API) - these are the hl.zeros, hl.full etc
    zeros, full, arange,
    # Control flow
    tile, grid, 
    # Reductions
    reduce, associative_scan, cumsum, cumprod,
    # Type casting
    cast,
    # Specialization
    specialize, constexpr,
    # Tuning
    register_block_size, register_reduction_dim, register_tunable,
    # Signal/wait
    signal, wait,
    # Tile operations
    tile_begin, tile_end, tile_block_size, tile_id, tile_index,
    # Device print
    device_print,
    # Memory operations
    load, store, atomic_add,
)

# Print function that works in ref mode
print = print  # Use built-in print

# Make sure all exports are available
__all__ = [
    # Memory operations from tensor_ops
    'load', 'store', 'atomic_add',
    # Creation ops from hl_apis (part of hl.* API)
    'zeros', 'full', 'arange',
    # Control flow from hl_apis
    'tile', 'grid',
    # Reductions from hl_apis
    'reduce', 'associative_scan', 'cumsum', 'cumprod',
    # Type casting from hl_apis
    'cast',
    # Specialization from hl_apis
    'specialize', 'constexpr',
    # Tuning from hl_apis
    'register_block_size', 'register_reduction_dim', 'register_tunable',
    # Signal/wait from hl_apis
    'signal', 'wait',
    # Tile operations from hl_apis
    'tile_begin', 'tile_end', 'tile_block_size', 'tile_id', 'tile_index',
    # Device print from hl_apis
    'device_print',
    # Namespace for kernels to use torch.* directly
    'torch',
    # Other
    'print'
]