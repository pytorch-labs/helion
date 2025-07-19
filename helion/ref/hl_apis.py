"""Helion language APIs (hl.*) for ref mode."""

from typing import Any, Optional, Union, Tuple, List, Iterator, Callable
import torch
from torch import Tensor
import itertools

# Import builtins
import builtins
_builtin_int = builtins.int
_builtin_list = builtins.list
_builtin_tuple = builtins.tuple
_builtin_range = builtins.range
_builtin_min = builtins.min
_builtin_max = builtins.max
_builtin_any = builtins.any

# No special tensor wrapper needed - we use monkey patching instead


# For torch.compile compatibility, we just use regular slices
# The .index attribute won't work with torch.compile anyway
slice = slice


# Creation ops (part of hl.* API)
def zeros(shape: List[Union[int, slice, slice]], dtype: torch.dtype = torch.float32) -> Tensor:
    """Create a zeros tensor, compatible with hl.zeros API."""
    # Convert slices to sizes
    processed_shape = []
    for s in shape:
        if isinstance(s, slice):
            # Calculate size from slice
            processed_shape.append(s.stop - s.start)
        else:
            processed_shape.append(s)
    # Return regular tensor
    return torch.zeros(processed_shape, dtype=dtype, device='cuda')


def full(shape: List[Union[int, slice, slice]], value: float, dtype: torch.dtype = torch.float32) -> Tensor:
    """Create a tensor filled with a value, compatible with hl.full API."""
    # Convert slices to sizes
    processed_shape = []
    for s in shape:
        if isinstance(s, slice):
            # Calculate size from slice
            processed_shape.append(s.stop - s.start)
        else:
            processed_shape.append(s)
    # Return regular tensor
    return torch.full(processed_shape, value, dtype=dtype, device='cuda')


def arange(*args: int, dtype: Optional[torch.dtype] = None, **kwargs) -> Tensor:
    """Create an arange tensor, compatible with hl.arange API."""
    if dtype is None:
        dtype = torch.int64  # Default index dtype
    return torch.arange(*args, dtype=dtype, device='cuda', **kwargs)


def tile(sizes: Union[int, List[int]], block_sizes: Optional[Union[int, List[int]]] = None, 
         block_size: Optional[Union[int, List[int]]] = None) -> Iterator[Union[slice, Tuple[slice, ...]]]:
    """Generate tiles for the given sizes.
    
    In eager mode with torch.compile, we use large tiles (full tensor dimensions)
    for better optimization opportunities.
    
    Args:
        sizes: Single size or list of sizes for each dimension
        block_sizes: Optional block sizes for tiling
        block_size: Alias for block_sizes (for compatibility)
        
    Yields:
        Single slice or tuple of slices for multi-dimensional tiling
    """
    # Support both block_size and block_sizes
    if block_size is not None and block_sizes is None:
        block_sizes = block_size
    
    # Handle the hl.tile(start, end) case - this is used for nested loops
    if block_sizes is not None and isinstance(sizes, (_builtin_int, torch.Tensor)) and isinstance(block_sizes, (_builtin_int, torch.Tensor)):
        # This is tile(start, end) syntax
        start = sizes
        end = block_sizes
        # Convert tensors to int
        if isinstance(start, torch.Tensor):
            start = int(start.item()) if start.numel() == 1 else int(start)
        if isinstance(end, torch.Tensor):
            end = int(end.item()) if end.numel() == 1 else int(end)
        # Generate a single slice for the range
        if start < end:
            yield slice(start, end)
        return
    
    # Normalize inputs
    if isinstance(sizes, _builtin_int):
        sizes = [sizes]
    elif isinstance(sizes, torch.Tensor) and sizes.numel() == 1:
        # Handle scalar tensor
        sizes = [int(sizes.item())]
    elif hasattr(sizes, '__iter__'):
        sizes = _builtin_list(sizes)  # Convert torch.Size or other iterables to list
    else:
        sizes = [sizes]
    
    # In eager mode, we use the full dimension as the block size
    # This effectively processes the entire tensor as a single tile
    if block_sizes is None:
        # Use full dimension size as block size for each dimension
        block_sizes = sizes
    elif isinstance(block_sizes, _builtin_int):
        block_sizes = [block_sizes] * len(sizes)
    
    # Generate tiles for each dimension
    dim_tiles = []
    for idx, (size, block_size) in enumerate(zip(sizes, block_sizes)):
        # Convert to int to handle torch scalar tensors
        size = _builtin_int(size)
        # Handle None block_size (means use full dimension)
        if block_size is None:
            block_size = size
        else:
            block_size = _builtin_int(block_size)
        
        # In eager mode, we use the full dimension as a single tile
        # This is equivalent to slice(0, size)
        tiles = [slice(0, size)]
        dim_tiles.append(tiles)
    
    # Generate all combinations for multi-dimensional tiling
    if len(dim_tiles) == 1:
        # Single dimension
        for t in dim_tiles[0]:
            yield t
    else:
        # Multi-dimensional - generate cartesian product
        for tile_combo in itertools.product(*dim_tiles):
            yield tile_combo


def grid(*args):
    """Simple grid implementation that returns range for single dimension.
    
    In eager mode, we just iterate over the range of values.
    """
    # Convert all arguments to integers
    int_args = []
    for arg in args:
        # Direct tensor handling
        
        if hasattr(arg, 'item'):
            int_args.append(arg.item())  # Convert tensor to Python int
        elif hasattr(arg, '__index__'):
            int_args.append(arg.__index__())  # For numpy integers etc
        else:
            int_args.append(_builtin_int(arg))
    
    if len(int_args) == 1:
        # Single dimension grid
        return _builtin_range(int_args[0])
    else:
        # Multi-dimensional grid - return cartesian product
        ranges = [_builtin_range(arg) for arg in int_args]
        return itertools.product(*ranges)


def reduce(op: Callable, value: Tensor, axis: Optional[int] = None, keep_dims: bool = False) -> Tensor:
    """Reduction operation.
    
    Args:
        op: Reduction operation (e.g., torch.sum, torch.max)
        value: Tensor to reduce
        axis: Optional axis to reduce along
        keep_dims: Whether to keep reduced dimensions
        
    Returns:
        Reduced tensor
    """
    if axis is None:
        result = op(value)
        if keep_dims:
            # Keep all dimensions as 1
            shape = [1] * len(value.shape)
            result = result.reshape(shape)
    else:
        result = op(value, dim=axis, keepdim=keep_dims)
    
    return result


def cast(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    """Cast tensor to a different dtype."""
    return tensor.to(dtype)


def specialize(value: Any) -> Any:
    """Mark a value for specialization (no-op in eager mode)."""
    return value


def constexpr(value: Any) -> Any:
    """Mark a value as constant expression (no-op in eager mode)."""
    return value


def register_block_size(*args, **kwargs):
    """Register block size for tiling (no-op in eager mode).
    
    In compiled mode, this provides hints to the compiler about preferred
    block sizes. In eager mode, we ignore these hints and use default tiling.
    """
    # No-op in eager mode
    pass


def register_reduction_dim(dim: int):
    """Register a dimension as a reduction dimension (no-op in eager mode).
    
    In compiled mode, this marks a dimension for special handling during
    reduction operations. In eager mode, reductions work without this hint.
    """
    # Return the dimension value for use in assertions
    return dim


def register_tunable(name: str, values: Any, default: Any = None):
    """Register a tunable parameter (no-op in eager mode).
    
    In compiled mode, this allows runtime tuning of parameters.
    In eager mode, we always use the default value or first value.
    """
    # Return the default if provided
    if default is not None:
        return default
    
    # Handle different types of values
    if hasattr(values, 'default'):
        # PowerOfTwoFragment and other ConfigSpecFragment types
        return values.default()
    elif hasattr(values, '__getitem__') and hasattr(values, '__len__'):
        # List-like objects
        if len(values) > 0:
            return values[0]
    
    # Return a sensible default
    return 1


def associative_scan(combine_fn: Callable, input_data: Union[Tensor, Tuple[Tensor, ...]], dim: int, reverse: bool = False) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Associative scan operation (prefix sum generalization).
    
    Args:
        combine_fn: Binary associative function to combine elements
        input_data: Input tensor or tuple of tensors
        dim: Dimension along which to scan
        reverse: Whether to scan in reverse order
        
    Returns:
        Scanned tensor or tuple of tensors
    """
    # Handle tuple input
    if isinstance(input_data, _builtin_tuple):
        # Convert to list of tensors for easier manipulation
        tensors = _builtin_list(input_data)
        
        # Get the shape along the scan dimension
        scan_size = tensors[0].shape[dim]
        
        # Initialize result tensors with same shape as input
        results = [torch.empty_like(t) for t in tensors]
        
        # Perform the scan
        for i in _builtin_range(scan_size):
            if reverse:
                idx = scan_size - 1 - i
            else:
                idx = i
                
            # Create index tuple for accessing the current position
            indices = [slice(None)] * len(tensors[0].shape)
            indices[dim] = idx
            indices = _builtin_tuple(indices)
            
            if (i == 0 and not reverse) or (i == scan_size - 1 and reverse):
                # First element - just copy
                for j, tensor in enumerate(tensors):
                    results[j][indices] = tensor[indices]
            else:
                # Get previous accumulated value
                if reverse:
                    prev_idx = idx + 1
                else:
                    prev_idx = idx - 1
                prev_indices = _builtin_list(indices)
                prev_indices[dim] = prev_idx
                prev_indices = _builtin_tuple(prev_indices)
                
                # Get current values
                current_vals = _builtin_tuple(t[indices] for t in tensors)
                prev_vals = _builtin_tuple(r[prev_indices] for r in results)
                
                # Apply combine function - unpack tuples as arguments
                combined = combine_fn(*prev_vals, *current_vals)
                
                # Store results
                if isinstance(combined, _builtin_tuple):
                    for j, val in enumerate(combined):
                        results[j][indices] = val
                else:
                    # Single result
                    results[0][indices] = combined
        
        return _builtin_tuple(results)
    else:
        # Single tensor input
        result = torch.empty_like(input_data)
        scan_size = input_data.shape[dim]
        
        for i in _builtin_range(scan_size):
            if reverse:
                idx = scan_size - 1 - i
            else:
                idx = i
                
            # Create index tuple for accessing the current position
            indices = [slice(None)] * len(input_data.shape)
            indices[dim] = idx
            indices = _builtin_tuple(indices)
            
            if (i == 0 and not reverse) or (i == scan_size - 1 and reverse):
                # First element - just copy
                result[indices] = input_data[indices]
            else:
                # Get previous accumulated value
                if reverse:
                    prev_idx = idx + 1
                else:
                    prev_idx = idx - 1
                prev_indices = _builtin_list(indices)
                prev_indices[dim] = prev_idx
                prev_indices = _builtin_tuple(prev_indices)
                
                # Apply combine function
                result[indices] = combine_fn(result[prev_indices], input_data[indices])
        
        return result


# Additional scan ops
def cumsum(x, axis: int, reverse: bool = False, keep_dims: bool = False):
    """Cumulative sum along an axis."""
    return associative_scan(x, axis, "add", reverse=reverse, keep_dims=keep_dims)


def cumprod(x, axis: int, reverse: bool = False, keep_dims: bool = False):
    """Cumulative product along an axis."""
    return associative_scan(x, axis, "mul", reverse=reverse, keep_dims=keep_dims)


# Signal/wait operations (no-op in eager mode)
def signal(handle, signal_id):
    """Signal operation - no-op in eager mode."""
    pass


def wait(handle, signal_id, signal=1, update=None, op="ld", scope="gpu", sem="acquire"):
    """Wait operation - no-op in eager mode."""
    pass


# Tile operations
def tile_begin(tile: slice) -> int:
    """Get the beginning index of a tile."""
    return tile.start


def tile_end(tile: slice) -> int:
    """Get the ending index of a tile."""
    return tile.stop


def tile_block_size(tile: slice) -> int:
    """Get the block size of a tile."""
    return tile.stop - tile.start


def tile_id(tile: slice) -> int:
    """Get the ID of a tile."""
    # For slices, we don't have an ID, so return 0
    return 0


def tile_index(tile: slice) -> Tensor:
    """Get the index tensor for a tile or slice."""
    # For slices, create an index tensor with the range
    return torch.arange(tile.start, tile.stop, device='cuda')


# Device print (for debugging)
def device_print(prefix: str, *args):
    """Print from device - in eager mode, just print normally."""
    print(prefix, *args)


# Memory operations
def load(tensor: Tensor, indices: Any, mask: Optional[Tensor] = None, other: Optional[Any] = None, extra_mask: Optional[Any] = None) -> Tensor:
    """Load from tensor with optional masking.
    
    Args:
        tensor: Source tensor
        indices: Indices to load from (can be tiles or regular indices)
        mask: Optional mask for conditional loading
        other: Optional fill value for masked elements
        extra_mask: Additional mask to apply (combined with mask via AND)
        
    Returns:
        Loaded tensor values
    """
    # Direct tensor operations
    
    # Handle different indexing patterns
    if isinstance(indices, (slice, slice)):
        # Simple slice indexing
        begin = indices.start
        end = _builtin_min(indices.stop, tensor.shape[0])
        if end > begin:
            result = tensor[begin:end]
        else:
            result = torch.empty(0, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    elif isinstance(indices, (_builtin_list, _builtin_tuple)):
        # Complex indexing - could be tiles, tensors, or mixed
        if len(indices) == 1 and isinstance(indices[0], torch.Tensor):
            # Advanced indexing with single tensor - this is the jagged case
            # indices[0] contains the computed indices for gathering
            idx_tensor = indices[0]
            
            # Check if we have a mask to apply
            if extra_mask is not None:
                # Apply extra_mask to filter valid indices
                # For jagged tensors, we need to handle the case where some indices might be out of bounds
                # Create a safe version of indices that won't cause out of bounds errors
                valid_mask = extra_mask.flatten()
                flat_indices = idx_tensor.flatten()
                
                # Clamp indices to valid range
                clamped_indices = torch.clamp(flat_indices, 0, tensor.shape[0] - 1)
                
                # Gather values using clamped indices
                gathered = tensor[clamped_indices]
                
                # Create output tensor with proper shape
                output_shape = idx_tensor.shape
                result = torch.zeros(output_shape, dtype=tensor.dtype, device=tensor.device)
                
                # Only copy valid values using where instead of boolean indexing
                result_flat = result.flatten()
                result_flat = torch.where(valid_mask, gathered, result_flat)
                result = result_flat.reshape(output_shape)
            else:
                # No mask - use regular advanced indexing but clamp to avoid out of bounds
                flat_indices = idx_tensor.flatten()
                clamped_indices = torch.clamp(flat_indices, 0, tensor.shape[0] - 1)
                result = tensor[clamped_indices].reshape(idx_tensor.shape)
        elif _builtin_any(isinstance(idx, (slice, slice)) for idx in indices):
            # Contains tiles - convert to slices or handle tensor indices
            # First, determine the expected output shape based on the indices
            expected_shape = []
            actual_indices = []
            tensor_shape = tensor.shape
            
            for i, idx in enumerate(indices):
                if isinstance(idx, (slice, slice)):
                    # For slice, the output size is the tile size
                    expected_shape.append(idx.stop - idx.start)
                    # Convert to slice, ensuring bounds are valid
                    begin = _builtin_max(0, idx.start)
                    end = _builtin_min(idx.stop, tensor_shape[i] if i < len(tensor_shape) else idx.stop)
                    actual_indices.append(slice(begin, end))
                elif isinstance(idx, torch.Tensor):
                    # For tensor indices, output shape matches the tensor shape
                    expected_shape.extend(idx.shape)
                    # Clamp indices to valid range to avoid errors
                    clamped_idx = torch.clamp(idx, 0, tensor_shape[i] - 1 if i < len(tensor_shape) else 0)
                    actual_indices.append(clamped_idx)
                else:
                    actual_indices.append(idx)
            
            # Try to index, but if we get an error or wrong shape, create a zero tensor
            try:
                result = tensor[_builtin_tuple(actual_indices)]
                # If the result shape doesn't match expected (due to clamping), pad with zeros
                if extra_mask is not None and result.shape != _builtin_tuple(expected_shape):
                    # Create a zero tensor of the expected shape
                    padded_result = torch.zeros(expected_shape, dtype=tensor.dtype, device=tensor.device)
                    # Copy valid data
                    if result.numel() > 0:
                        # Figure out the valid region
                        slices = []
                        for i, s in enumerate(result.shape):
                            slices.append(slice(0, s))
                        padded_result[_builtin_tuple(slices)] = result
                    result = padded_result
            except (RuntimeError, IndexError):
                # If indexing fails (e.g., due to negative indices), create zeros
                result = torch.zeros(expected_shape, dtype=tensor.dtype, device=tensor.device)
        else:
            # Regular indexing
            result = tensor[indices]
    else:
        # Direct indexing
        result = tensor[indices]
    
    # Combine masks if both provided
    if mask is not None and extra_mask is not None:
        combined_mask = mask & extra_mask
    elif extra_mask is not None:
        combined_mask = extra_mask
    elif mask is not None:
        combined_mask = mask
    else:
        combined_mask = None
    
    # Apply combined mask if any
    if combined_mask is not None:
        if other is None:
            other = 0
        # Ensure result and mask have compatible shapes
        # This handles cases where indexing produces different shapes than expected
        if result.shape != combined_mask.shape:
            # Try to broadcast or handle shape mismatch
            if combined_mask.numel() == 0 or result.numel() == 0:
                # Empty result - return zeros with mask shape
                result = torch.zeros(combined_mask.shape, dtype=result.dtype, device=result.device)
            else:
                # Attempt to broadcast - this will raise an error if shapes are incompatible
                result = torch.where(combined_mask, result, other)
        else:
            result = torch.where(combined_mask, result, other)
    
    return result


def store(tensor: Tensor, indices: Any, value: Tensor, mask: Optional[Tensor] = None) -> None:
    """Store value into tensor with optional masking.
    
    Args:
        tensor: Destination tensor
        indices: Indices to store to (can be tiles or regular indices)
        value: Value to store
        mask: Optional mask for conditional storing
    """
    # Handle tile-based storing
    if isinstance(indices, (slice, slice)):
        if mask is not None:
            current = tensor[indices.start:indices.stop]
            tensor[indices.start:indices.stop] = torch.where(mask, value, current)
        else:
            tensor[indices.start:indices.stop] = value
    elif isinstance(indices, (_builtin_list, _builtin_tuple)) and _builtin_any(isinstance(idx, (slice, slice)) for idx in indices):
        # Convert tiles to slices
        actual_indices = []
        for idx in indices:
            if isinstance(idx, (slice, slice)):
                actual_indices.append(slice(idx.start, idx.stop))
            else:
                actual_indices.append(idx)
        
        if mask is not None:
            current = tensor[_builtin_tuple(actual_indices)]
            tensor[_builtin_tuple(actual_indices)] = torch.where(mask, value, current)
        else:
            tensor[_builtin_tuple(actual_indices)] = value
    else:
        # Regular indexing
        if mask is not None:
            current = tensor[indices]
            tensor[indices] = torch.where(mask, value, current)
        else:
            tensor[indices] = value


def atomic_add(tensor: Tensor, indices: Any, value: Tensor) -> None:
    """Atomic add operation (simulated in eager mode).
    
    Note: In eager mode, atomic operations are not truly atomic but are
    simulated for correctness testing.
    """
    # Note: warnings.warn is not supported by torch.compile, so we skip it
    # In eager mode, atomic operations are not thread-safe but this is expected
    
    # Handle different indexing patterns
    if isinstance(indices, (_builtin_list, _builtin_tuple)):
        # Multi-dimensional indexing
        if len(indices) == 2:
            idx0, idx1 = indices
            # Convert tiles to appropriate indices
            if isinstance(idx1, slice):
                idx1 = slice(idx1.start, idx1.stop)
            
            # Handle the scatter-add pattern for segment reduction
            if isinstance(idx0, torch.Tensor) and isinstance(idx1, (slice, slice)):
                # This is the pattern: output[idxs, tile_f] += segment_vals
                # For 2D indexing with tensor and slice, we need special handling
                if isinstance(idx1, (slice, slice)):
                    # Extract the slice range
                    start = idx1.start if idx1.start is not None else 0
                    stop = idx1.stop if idx1.stop is not None else tensor.shape[1]
                    # Create a temporary view and use scatter_add
                    tensor_view = tensor[:, start:stop]
                    # Use index_add along dim 0
                    tensor_view.index_add_(0, idx0, value)
            else:
                # Regular indexing
                tensor[idx0, idx1] += value
        else:
            # Convert tiles to slices
            actual_indices = []
            for idx in indices:
                if isinstance(idx, slice):
                    actual_indices.append(idx)
                else:
                    actual_indices.append(idx)
            tensor[_builtin_tuple(actual_indices)] += value
    else:
        # Regular indexing
        tensor[indices] += value
