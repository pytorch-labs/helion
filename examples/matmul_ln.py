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
    """
    Problem:
    the : in y[tile_k, :] creates a reduction dim with symsize u2, which causes a mismatch with acc's second dim which is a concrete int 1024:
    `helion.exc.InternalError: RuntimeError: The size of tensor a (u2) must match the size of tensor b (1024) at non-singleton dimension 1). a: (u0, u2), b: torch.Size([u0, 1024])`
    I wonder is there a way for Helion compiler to infer that u2 == 1024? from the program, we know that n is the size of y's second dim (so `:` should just have the range n),
    and also we know that n is the size of acc's second dim. So it feels that maybe there is a way to tie these two together.

    Comment from Jason:
    - For this one, one issue you will have with:
    `hl.zeros([tile_m, n], dtype=torch.float32)`
    is if n is not a power of 2 you will get invalid code.
    The `y[tile_k, :]` is implicitly rounding the : up to a power of two.
    So the sizes are n and next_power_of_2(n), which is a real mismatch.
    - I think the way to fix this is to make the hl.zeros([tile_m, n], dtype=torch.float32) allocation
    implicitly round up to a power of 2 the same way `:` does and introduce a reduction dimension.

    NOTE: this should fix Driss's flashattn repro too: https://www.internalfb.com/phabricator/paste/view/P1815504920
    """
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            mm = torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
            acc = acc + mm
        acc = torch.nn.functional.layer_norm(acc, [acc.size(1)], weight.to(torch.float32)[tile_n], bias.to(torch.float32)[tile_n])
        out[tile_m, tile_n] = acc
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
