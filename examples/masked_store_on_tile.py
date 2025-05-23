"""Example: in-place tile update with an explicit *mask*.

The example demonstrates the pattern requested in the user prompt::

    x[tile_m, tile_n][mask] = y

where

* ``tile_m`` / ``tile_n`` come from ``hl.tile`` and describe the block that is
  processed by the current kernel instance;
* ``mask`` is a run-time boolean tensor that selects *rows* within the tile;
* ``y`` provides the replacement values.

For simplicity the mask is defined per-row (it is a 1-D tensor of the same
length as ``tile_m``).  This aligns with Helion's current indexing rules which
support 1-D boolean / integer tensors as advanced indices.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# -----------------------------------------------------------------------------
# Kernel
# -----------------------------------------------------------------------------


@helion.kernel()
def scatter_masked_rows_on_tile(
    x: torch.Tensor,  # [M, N]
    y: torch.Tensor,  # [M, N]
    row_mask: torch.Tensor,  # [M]  – boolean mask *per row*
) -> torch.Tensor:
    """Replace the rows of *x* where ``row_mask`` is **True** with the
    corresponding rows from ``y`` and return the updated tensor.

    The logic uses ``x[tile_m, tile_n][mask] = y`` as requested.  Within each
    2-D tile the mask selects a *subset of the rows* to be written.
    """

    M, N = x.size()

    out = torch.empty_like(x)

    for tile_m in hl.tile(M):
        mask_rows = row_mask[tile_m]  # shape: [tile_m]

        for tile_n in hl.tile(N):
            rows = tile_m[mask_rows]
            out[rows, tile_n] = y[rows, tile_n]

    return out


# -----------------------------------------------------------------------------
# Quick correctness / performance check
# -----------------------------------------------------------------------------


def _check(M: int = 1024, N: int = 1024) -> None:  # pragma: no cover – example
    from triton.testing import do_bench

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(M, N, device=device, dtype=torch.float16)
    y = torch.randn(M, N, device=device, dtype=torch.float16)
    row_mask = torch.randint(0, 2, (M,), device=device, dtype=torch.bool)

    # Reference result (plain PyTorch)
    ref = x.clone()
    ref[row_mask] = y[row_mask]

    # Helion kernel
    res = scatter_masked_rows_on_tile(x, y, row_mask)

    torch.testing.assert_close(res, ref, rtol=1e-2, atol=1e-1)

    helm_sec = do_bench(lambda: scatter_masked_rows(x, y, row_mask))
    torch_sec = do_bench(lambda: ref[row_mask])  # just access cost baseline

    print(
        f"Helion time: {helm_sec:.4f}s, torch time: {torch_sec:.4f}s, "
        f"speed-up: {torch_sec / helm_sec:.2f}x",
    )


if __name__ == "__main__":
    _check()
