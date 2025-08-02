from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

import torch
import triton

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclasses.dataclass
class DimSize:
    """Stores physical and logical size information for a tensor dimension.

    Can be used both as input (to request power-of-2 allocation) and
    internally to track dimension information.
    """

    logical_size: int
    physical_size: int | None = None

    def __post_init__(self) -> None:
        """Compute the physical size if not provided."""
        if self.physical_size is None:
            # Auto-compute power-of-2 physical size
            self.physical_size = triton.next_power_of_2(self.logical_size)

    def __int__(self) -> int:
        """Return the physical size when converted to int."""
        assert self.physical_size is not None
        return self.physical_size

    def has_custom_logical_size(self) -> bool:
        """Check if this dimension has a custom logical size (different from physical)."""
        return self.logical_size != self.physical_size

    def __repr__(self) -> str:
        return f"DimSize(logical={self.logical_size}, physical={self.physical_size})"


class RefTensor(torch.Tensor):
    """A tensor subclass that tracks physical vs logical dimension sizes.

    This allows tensors to have a physical allocation size (e.g., power of 2)
    while maintaining knowledge of their logical size for operations.
    """

    _dim_size_map: dict[int, DimSize]

    def __new__(
        cls,
        data: torch.Tensor,
        dim_size_map: dict[int, DimSize],
    ) -> Self:
        instance = data.as_subclass(cls)
        instance._dim_size_map = dim_size_map
        return instance

    _DUNDER_IN_PLACE_OPS: ClassVar[set[Callable[..., Any]]] = {
        torch.Tensor.__iadd__,
        torch.Tensor.__isub__,
        torch.Tensor.__imul__,
        torch.Tensor.__itruediv__,
    }

    @classmethod
    def _is_mutable_op(cls, func: Callable[..., Any]) -> bool:
        # Check known dunder in-place ops
        if func in cls._DUNDER_IN_PLACE_OPS:
            return True

        # Check if name ends with underscore (PyTorch in-place convention)
        # but exclude special methods like __getitem__, __setitem__
        if hasattr(func, "__name__"):
            name = func.__name__
            if name.endswith("_") and not (
                name.startswith("__") and name.endswith("__")
            ):
                return True

        # Check schema if available
        if hasattr(func, "_schema"):
            schema = func._schema  # type: ignore[attr-defined]
            return getattr(schema, "is_mutable", False)
        return False

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        kwargs = kwargs or {}

        # Check for special handlers
        # 1. In-place operations
        if cls._is_mutable_op(func):
            return cls._handle_mutable_op(func, args, kwargs)

        # 2. Item access operations
        if func is torch.Tensor.__getitem__:
            return cls._handle_getitem(func, args, kwargs)
        if func is torch.Tensor.__setitem__:
            return cls._handle_setitem(func, args, kwargs)

        # Default handling for all other operations
        return cls._execute_with_unwrapped_args(func, args, kwargs)

    @classmethod
    def _process_indices_for_logical_sizes(
        cls, indices: object, dim_size_map: dict[int, DimSize]
    ) -> object:
        """Process indices to respect logical sizes."""
        if isinstance(indices, tuple):
            new_indices = []
            for dim, idx in enumerate(indices):
                if (
                    isinstance(idx, slice)
                    and idx == slice(None)
                    and dim in dim_size_map
                ):
                    # Only modify if logical != physical
                    if dim_size_map[dim].has_custom_logical_size():
                        idx = slice(0, dim_size_map[dim].logical_size)
                new_indices.append(idx)
            return tuple(new_indices)
        return indices

    @classmethod
    def _handle_getitem(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Handle getitem operations to respect logical sizes."""
        tensor, indices = args

        # Always process indices - it's a no-op when logical == physical
        if isinstance(tensor, RefTensor):
            indices = cls._process_indices_for_logical_sizes(
                indices, tensor._dim_size_map
            )

        # Call regular getitem
        return tensor.as_subclass(torch.Tensor)[indices]  # type: ignore[index]

    @staticmethod
    def _get_logical_slice_indices(
        dim_size_map: dict[int, DimSize],
    ) -> tuple[slice, ...] | slice:
        """Get slice indices for logical dimensions."""
        if not dim_size_map:
            return slice(None)

        max_dim = max(dim_size_map.keys())
        indices = []
        has_custom = False

        for dim in range(max_dim + 1):
            if dim in dim_size_map and dim_size_map[dim].has_custom_logical_size():
                indices.append(slice(0, dim_size_map[dim].logical_size))
                has_custom = True
            else:
                indices.append(slice(None))

        # If no custom dimensions, return simple slice
        if not has_custom:
            return slice(None)

        return indices[0] if len(indices) == 1 else tuple(indices)

    @classmethod
    def _get_value_as_logical_view(
        cls, value: object, target_dim_size_map: dict[int, DimSize] | None
    ) -> object:
        """Convert a value to its logical view if needed for operations with RefTensor.

        Args:
            value: The value to convert (could be RefTensor, Tensor, or scalar)
            target_dim_size_map: The dimension size info of the target tensor (for slicing regular tensors)

        Returns:
            The logical view of the value
        """
        if isinstance(value, RefTensor):
            indices = cls._get_logical_slice_indices(value._dim_size_map)
            # Use direct tensor indexing to avoid going through __torch_function__
            return value.as_subclass(torch.Tensor)[indices]
        if isinstance(value, torch.Tensor) and target_dim_size_map:
            # Slice regular tensor to match target's logical size
            logical_slice = cls._get_logical_slice_indices(target_dim_size_map)
            return value[logical_slice]
        return value

    @classmethod
    def _handle_setitem(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Handle setitem operations to respect logical sizes."""
        tensor, indices, value = args

        # Convert value to logical view if needed
        value = cls._get_value_as_logical_view(value, None)

        # Always process indices - it's a no-op when logical == physical
        if isinstance(tensor, RefTensor):
            indices = cls._process_indices_for_logical_sizes(
                indices, tensor._dim_size_map
            )

        # Use PyTorch's setitem directly without going through __torch_function__
        base_tensor = tensor.as_subclass(torch.Tensor)
        base_tensor[indices] = value  # type: ignore[index,assignment]
        return None

    @classmethod
    def _handle_mutable_op(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> RefTensor:
        """Handle in-place operations with logical size adjustments."""
        tensor, other = args[:2]

        if isinstance(tensor, RefTensor):
            # Get logical view of tensor
            tensor_logical_view = cls._get_value_as_logical_view(tensor, None)

            # Get appropriate logical view of other
            other_logical_view = cls._get_value_as_logical_view(
                other, tensor._dim_size_map
            )

            # Apply operation on the logical views
            func(tensor_logical_view, other_logical_view, *args[2:], **kwargs)
            return tensor

        # Fall back to regular operation
        cls._execute_with_unwrapped_args(func, args, kwargs)
        return args[0]

    @classmethod
    def _unwrap_args(cls, args: object) -> object:
        """Convert RefTensor instances to regular tensors."""
        if isinstance(args, RefTensor):
            return args.as_subclass(torch.Tensor)
        if isinstance(args, (list, tuple)):
            return type(args)(cls._unwrap_args(arg) for arg in args)
        if isinstance(args, dict):
            return {k: cls._unwrap_args(v) for k, v in args.items()}
        return args

    @classmethod
    def _execute_with_unwrapped_args(
        cls,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object:
        """Execute a function with unwrapped RefTensor arguments."""
        new_args = cls._unwrap_args(args)
        new_kwargs = cls._unwrap_args(kwargs)
        assert isinstance(new_args, tuple), "Expected args to be tuple"
        return func(*new_args, **new_kwargs)  # type: ignore[arg-type]

    def __repr__(self, *, tensor_contents: None = None) -> str:
        base_repr = super().__repr__()

        # Build dimension info for custom mappings only
        custom_dims = [
            f"dim{d}: {info.logical_size}/{info.physical_size}"
            for d, info in sorted(self._dim_size_map.items())
            if info.has_custom_logical_size()
        ]

        if not custom_dims:
            return f"RefTensor({base_repr})"

        dim_str = ", ".join(custom_dims)
        return f"RefTensor({base_repr}, dims=[{dim_str}])"
