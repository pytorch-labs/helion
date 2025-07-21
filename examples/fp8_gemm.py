from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(static_shapes=True)
def fp8_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """FP8 General Matrix Multiplication (GEMM).

    This kernel demonstrates FP8 computation in Helion.
    When lowered to Triton, the tl.dot operation will handle
    FP8 inputs natively and accumulate to FP32.

    Args:
        x: Input tensor of shape [m, k] in FP8 format
        y: Input tensor of shape [k, n] in FP8 format

    Returns:
        Output tensor of shape [m, n] in FP16 format
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"

    # Output is in FP16 to match tritonbench behavior
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        # Accumulate in FP32 for accuracy
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            # Load FP8 tiles directly - no conversion needed
            x_tile = x[tile_m, tile_k]
            y_tile = y[tile_k, tile_n]

            # Use torch.matmul which will be lowered to tl.dot
            # When the inputs are FP8, tl.dot handles them natively
            # The result needs to be converted to FP32 for accumulation
            result = torch.matmul(x_tile, y_tile).to(torch.float32)
            acc = acc + result
        out[tile_m, tile_n] = acc.to(torch.float16)

    return out


def reference_fp8_gemm_pytorch(
    x_fp8: torch.Tensor, y_fp8: torch.Tensor
) -> torch.Tensor:
    """Reference implementation using torch._scaled_mm."""
    # torch._scaled_mm requires column-major for second operand
    y_fp8_t = y_fp8.T.contiguous().T
    scale_a = torch.tensor(1.0, device=x_fp8.device)
    scale_b = torch.tensor(1.0, device=x_fp8.device)
    return torch._scaled_mm(
        x_fp8, y_fp8_t, scale_a, scale_b, use_fast_accum=False, out_dtype=torch.float16
    )


def fp8_gemm_tritonbench(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for TritonBench compatibility."""
    return fp8_gemm(a, b)


def check(m: int, k: int, n: int) -> None:
    """Test the FP8 GEMM implementation."""
    # Create FP8 tensors
    x = torch.randn([m, k], device="cuda", dtype=torch.float32)
    y = torch.randn([k, n], device="cuda", dtype=torch.float32)

    # Convert to FP8 format (e4m3fn is commonly used for forward pass)
    x_fp8 = x.to(torch.float8_e4m3fn)
    y_fp8 = y.to(torch.float8_e4m3fn)

    run_example(fp8_gemm, reference_fp8_gemm_pytorch, (x_fp8, y_fp8))


def main() -> None:
    # Test with different sizes
    check(256, 256, 256)
    check(512, 512, 512)
    check(1024, 1024, 1024)


if __name__ == "__main__":
    main()
