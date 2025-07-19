"""Tensor operations for ref mode - mixed precision patches for PyTorch operations."""

import torch

# Store original torch functions
_original_addmm = None
_original_baddbmm = None
_original_matmul = None

# Import the E5M2 matmul kernel to register the custom op
try:
    from .triton_e5m2_matmul import e5m2_matmul_op
    _e5m2_matmul_available = True
except ImportError:
    _e5m2_matmul_available = False


def _patched_addmm(input, mat1, mat2, *, beta=1, alpha=1):
    """Patched addmm that handles dtype mismatches."""
    # Convert mat1 and mat2 to match input dtype if needed
    if mat1.dtype != input.dtype:
        mat1 = mat1.to(input.dtype)
    if mat2.dtype != input.dtype:
        mat2 = mat2.to(input.dtype)
    # Call original function
    return _original_addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def _patched_baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    """Patched baddbmm that handles dtype mismatches."""
    # Convert batch1 and batch2 to match input dtype if needed
    if batch1.dtype != input.dtype:
        batch1 = batch1.to(input.dtype)
    if batch2.dtype != input.dtype:
        batch2 = batch2.to(input.dtype)
    # Call original function
    return _original_baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)


def _patched_matmul(input, other):
    """Patched matmul that handles dtype mismatches, especially for FP8."""
    # Check if either input is FP8
    input_is_fp8 = hasattr(input.dtype, 'is_floating_point') and 'float8' in str(input.dtype)
    other_is_fp8 = hasattr(other.dtype, 'is_floating_point') and 'float8' in str(other.dtype)
    
    # For FP8 inputs, check specific formats
    input_is_e4m3 = input_is_fp8 and 'e4m3' in str(input.dtype)
    other_is_e4m3 = other_is_fp8 and 'e4m3' in str(other.dtype)
    input_is_e5m2 = input_is_fp8 and 'e5m2' in str(input.dtype)
    other_is_e5m2 = other_is_fp8 and 'e5m2' in str(other.dtype)
    
    # For E4M3 inputs, use torch._scaled_mm 
    if input_is_e4m3 and other_is_e4m3 and input.dim() == 2 and other.dim() == 2:
        # torch._scaled_mm requires the second matrix to be column-major (Fortran order)
        # For a 2D tensor, column-major means stride[0] == 1 and stride[1] == rows
        # The standard PyTorch way to get column-major is: tensor.t().contiguous().t()
        other_col_major = other.t().contiguous().t()
        
        # Use torch._scaled_mm with unity scales
        scale_a = torch.tensor(1.0, device=input.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=other.device, dtype=torch.float32)
        
        # Determine output dtype - use float32 for accuracy
        out_dtype = torch.float32
        
        # Use torch._scaled_mm for FP8 E4M3 GEMM
        result = torch._scaled_mm(input, other_col_major, scale_a, scale_b, 
                                 out_dtype=out_dtype, use_fast_accum=False)
        return result
    
    # For E5M2 inputs, use custom Triton kernel
    elif input_is_e5m2 and other_is_e5m2 and input.dim() == 2 and other.dim() == 2:
        if not _e5m2_matmul_available:
            raise RuntimeError(
                "FP8 E5M2 matmul requires custom Triton kernel but it's not available. "
                "Ensure triton_e5m2_matmul module is properly installed."
            )
        # Call our custom E5M2 matmul kernel
        result = torch.ops.helion.e5m2_matmul(input, other)
        return result
    
    # Error out for unsupported FP8 cases
    if input_is_fp8 or other_is_fp8:
        # Determine which formats we have
        formats = []
        if input_is_fp8:
            formats.append(f"input: {input.dtype}")
        if other_is_fp8:
            formats.append(f"other: {other.dtype}")
        
        # Build error message
        if input_is_fp8 and other_is_fp8 and (input.dim() != 2 or other.dim() != 2):
            raise RuntimeError(
                f"FP8 matmul only supports 2D matrices, got {input.dim()}D x {other.dim()}D. "
                f"Dtypes: {', '.join(formats)}"
            )
        elif (input_is_fp8 and not other_is_fp8) or (other_is_fp8 and not input_is_fp8):
            raise RuntimeError(
                f"Mixed FP8/non-FP8 matmul not supported. Got input: {input.dtype}, other: {other.dtype}"
            )
        else:
            # This covers any other FP8 cases we don't handle
            raise RuntimeError(
                f"Unsupported FP8 matmul configuration. {', '.join(formats)}"
            )
    
    # Handle non-FP8 dtype mismatches
    if input.dtype != other.dtype:
        # Determine target dtype (use higher precision)
        if input.dtype == torch.float64 or other.dtype == torch.float64:
            target_dtype = torch.float64
        elif input.dtype == torch.float32 or other.dtype == torch.float32:
            target_dtype = torch.float32
        elif input.dtype == torch.float16 or other.dtype == torch.float16:
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32  # Default to float32
        
        input = input.to(target_dtype)
        other = other.to(target_dtype)
    
    # Call original function
    return _original_matmul(input, other)


class MixedPrecisionPatchContext:
    """Context manager for patching torch functions to handle mixed precision."""
    
    def __init__(self):
        self._patch_count = 0
    
    def __enter__(self):
        """Apply monkey patches on entry."""
        global _original_addmm, _original_baddbmm, _original_matmul
        
        if self._patch_count == 0:
            # First entry - save original methods and apply patches
            _original_addmm = torch.addmm
            _original_baddbmm = torch.baddbmm
            _original_matmul = torch.matmul
            
            # Apply patches
            torch.addmm = _patched_addmm
            torch.baddbmm = _patched_baddbmm
            torch.matmul = _patched_matmul
        
        self._patch_count += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original methods on exit."""
        global _original_addmm, _original_baddbmm, _original_matmul
        
        self._patch_count -= 1
        
        if self._patch_count == 0:
            # Last exit - restore original methods
            if _original_addmm is not None:
                torch.addmm = _original_addmm
                _original_addmm = None
            
            if _original_baddbmm is not None:
                torch.baddbmm = _original_baddbmm
                _original_baddbmm = None
            
            if _original_matmul is not None:
                torch.matmul = _original_matmul
                _original_matmul = None
        
        # Don't suppress exceptions
        return False


# Create a global instance for mixed precision patching
mixed_precision_patch_context = MixedPrecisionPatchContext()