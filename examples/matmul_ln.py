from __future__ import annotations

import torch

import helion
import helion.language as hl


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul_ln(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        acc = torch.nn.functional.layer_norm(acc, [n], weight.to(torch.float32), bias.to(torch.float32))
        out[tile_m, :] = acc
    return out


def matmul_ln_pytorch(x: torch.Tensor,
                      y: torch.Tensor,
                      weight: torch.Tensor,
                      bias: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch equivalent of the Helion `matmul_ln` kernel.
    Shapes
        x : [m, k]
        y : [k, n]
        weight, bias : [n]
    Returns
        out : [m, n]   dtype = promote_types(x.dtype, y.dtype)
    """
    # 1. Matmul (m × k) @ (k × n)  ->  (m × n)
    #    Keep the intermediate in FP32 to match the kernel’s `acc` buffer.
    prod_fp32 = torch.matmul(x.to(torch.float32), y.to(torch.float32))

    # 2. Row-wise LayerNorm across the n dimension.
    #    LayerNorm expects the normalized_shape as a *tuple*.
    ln_fp32 = F.layer_norm(
        prod_fp32,
        normalized_shape=(prod_fp32.shape[-1],),
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
    )

    # 3. Cast back to the promoted dtype that the kernel wrote to `out`.
    out = ln_fp32.to(torch.promote_types(x.dtype, y.dtype))
    return out


def check(m: int, k: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    weight = torch.randn([n], device="cuda", dtype=torch.float16)
    bias = torch.randn([n], device="cuda", dtype=torch.float16)
    result = matmul_ln(x, y, weight, bias)
    torch.testing.assert_close(result, x @ y, rtol=1e-2, atol=1e-1)
    sec = do_bench(lambda: matmul_ln(x, y, weight, bias))
    baseline_sec = do_bench(lambda: matmul_ln_pytorch(x, y, weight, bias))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


if __name__ == "__main__":
    check(256, 512, 1024)
