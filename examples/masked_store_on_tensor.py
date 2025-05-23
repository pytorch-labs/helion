"""
Matmul-of-Experts (MoE) with Outer-Gather-Scatter (OGS)
"""

import helion
import helion.language as hl
import torch


@helion.kernel(static_shapes=False)
def scatter_masked_rows_on_tensor(
    A: torch.Tensor,                          # [B, K]
    W: torch.Tensor,                          # [E, K, N]
    expert_token_counts: torch.Tensor,        # [E]
    expert_token_offsets: torch.Tensor,       # [E + 1]
    sorted_to_orig_token_idx: torch.Tensor,   # [B]
    T_max_tensor: torch.Tensor,               # [T_max]
):
    B, K = A.shape
    E, _, N = W.shape
    T_max = T_max_tensor.numel()

    C = torch.empty(
        B, N,
        dtype=torch.promote_types(A.dtype, W.dtype),
        device=A.device,
    )

    row_ids = torch.arange(T_max, device=A.device, dtype=torch.int32)

    for e_idx in hl.grid(E):
        start       = expert_token_offsets[e_idx]
        num_tokens  = expert_token_counts[e_idx]
        for tile_t, tile_n in hl.tile([T_max, N]):
            local_rows   = row_ids[tile_t].squeeze(0).squeeze(0)     # shape [BLOCK_T]
            row_valid    = (local_rows < num_tokens).squeeze(0).squeeze(0)        # bool [BLOCK_T]
            orig_rows_idxes = start + local_rows               # [1, BLOCK_T]
            orig_rows       = sorted_to_orig_token_idx[orig_rows_idxes.squeeze(0)]     # [BLOCK_T]
            acc = hl.zeros([tile_t, tile_n], dtype=torch.float32)

            # scatter masked rows
            C[orig_rows[row_valid], tile_n] = acc[row_valid, tile_n]

    return C


def masked_store_on_tensor(
    A: torch.Tensor,  # [B, K]
    W: torch.Tensor,  # [E, K, N]
    top1_expert_per_token: torch.Tensor,  # [B]
) -> torch.Tensor:
    B = A.size(0)
    E = W.size(0)
    device = A.device

    sorting = torch.argsort(top1_expert_per_token, stable=True).to(torch.int32)  # [B]
    expert_token_counts = torch.bincount(
        top1_expert_per_token, minlength=E
    ).to(torch.int32)  # [E]

    expert_token_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)  # [E+1]
    expert_token_offsets[0] = 0
    expert_token_offsets[1:] = torch.cumsum(expert_token_counts, 0, dtype=torch.int32)

    T_max = int(expert_token_counts.max().item())
    C = scatter_masked_rows_on_tensor(
        A,
        W,
        expert_token_counts,
        expert_token_offsets,
        sorting,
        torch.empty(T_max, device=device),
    )

    return C

def check() -> None:
    from triton.testing import do_bench

    B = 1024   # tokens / rows
    K = 512    # hidden size
    N = 256    # output size
    n_experts = 32
    dtype = torch.float16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    top1_expert_per_token = torch.randint(n_experts, (B,), device=device)
    A = torch.randn(B, K, device=device, dtype=dtype)
    W = torch.randn(n_experts, K, N, device=device, dtype=dtype)

    C_helion = masked_store_on_tensor(A, W, top1_expert_per_token)
    # C_ref = moe_ref(A, W, top1_expert_per_token)
    # torch.testing.assert_close(C_helion, C_ref, atol=1e-2, rtol=1e-2)

    sec = do_bench(lambda: masked_store_on_tensor(A, W, top1_expert_per_token))
    # baseline_sec = do_bench(lambda: moe_ref(A, W, top1_expert_per_token))
    # print(f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}s, speed-up: {baseline_sec/sec:.2f}x")

if __name__ == "__main__":
    check()
