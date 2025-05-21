"""
Matmul-of-Experts (MoE) with Outer-Gather-Scatter (OGS)
"""

import helion
import helion.language as hl
import torch


@helion.kernel(static_shapes=True)
def _moe_matmul_ogs_maxT(
    A: torch.Tensor,  # [B, K]
    W: torch.Tensor,  # [E, K, N]
    expert_token_counts: torch.Tensor,    # [E]  number of tokens per expert  (TODO: can we remove this?)
    expert_token_offsets: torch.Tensor,  # [E+1] cumulative sum of token counts
    sorted_to_orig_token_idx: torch.Tensor,  # [B] mapping from sorted -> orig idx
    T_max: hl.constexpr,  # maximum #tokens any expert may receive (compile-time constant)
):
    """Compute `C = MoE(A, W)` using OGS with fixed max # tokens per expert `T_max`.

    The *dispatch layout* is described by `expert_token_offsets` such that tokens
    belonging to expert `e` live in the half-open slice

        [ expert_token_offsets[e] : expert_token_offsets[e + 1] )

    inside the *expert-sorted* permutation of the batch.  All rows are stored in
    the original tensor `A` where we still operate in *original* (unsorted)
    order - the indirection is handled explicitly via `sorted_to_orig_token_idx`.

    Each expert is processed independently.  They all iterate over *exactly* the
    same number of rows (`T_max`).  Rows whose relative index is `>= #tokens`
    for that expert are considered padding.  During the computation we
    1. mask-out their contribution by zeroing out the corresponding input rows
       of `A`, and
    2. skip the scatter write-back for those rows so the output tensor `C`
       remains untouched for them.
    """

    B = A.size(0)  # number of tokens in the *whole* batch
    K = A.size(1)  # hidden size
    N = W.size(2)  # output dimension
    E = W.size(0)  # number of experts

    C = torch.empty(B, N, dtype=torch.promote_types(A.dtype, W.dtype), device=A.device)
    row_ids = torch.arange(T_max, device=A.device, dtype=torch.int32)

    for e_idx in hl.grid(E):
        start = expert_token_offsets[e_idx]
        num_tokens = expert_token_counts[e_idx]

        # Process this expert only if it received at least one token.
        if num_tokens != 0:
            for tile_t, tile_n in hl.tile([T_max, N]):
                # Determine which rows inside this (expert-local) tile correspond to
                # *real* tokens and which ones are just padding.
                local_rows = row_ids[tile_t]  # [BLOCK_T]
                row_valid = local_rows < num_tokens  # bool[BLOCK_T]
                # Skip tiles that consist only of padding rows.
                # if torch.any(row_valid):  # TODO(yf225): make torch.any work
                # Map (expert-local) row indices to original batch order.  For
                # rows that are padding we *saturate* the offset to the last
                # valid element so we stay in-bounds while their results will be
                # discarded later via `row_valid`.
                safe_offset = torch.where(
                    row_valid,
                    local_rows,
                    num_tokens - 1,  # any valid row inside the expert slice
                )
                orig_rows = sorted_to_orig_token_idx[start + safe_offset]  # [BLOCK_T]

                acc = hl.zeros([tile_t, tile_n], dtype=torch.float32)

                for tile_k in hl.tile(K):
                    # Gather the relevant slice of A.
                    A_frag = A[orig_rows, tile_k]
                    # Mask out padding rows by zeroing their inputs
                    A_frag = A_frag * (row_valid[:, None].to(A_frag.dtype))
                    # Expert weights for the current (K, N) subtile.
                    W_frag = W[e_idx, tile_k, tile_n].view(tile_k, tile_n)
                    # Core matmul on the tile
                    acc = torch.addmm(acc, A_frag, W_frag)

                # Scatter the results back – only for the *real* rows.
                C[orig_rows[row_valid], tile_n] = acc[row_valid, :]

    return C


def moe_matmul_ogs(
    A: torch.Tensor,  # [B, K]
    W: torch.Tensor,  # [E, K, N]
    top1_expert_per_token: torch.Tensor,  # [B]
    variant: str,
) -> torch.Tensor:
    """Top-level helper that prepares the dispatch metadata and launches the
    Helion kernel.

    Parameters
    ----------
    A : Tensor[ B, K ]
        Input activations (one row per token).
    W : Tensor[ E, K, N ]
        Expert weight matrices.
    top1_expert_per_token : Tensor[ B ] (int32 / int64)
        Routing decisions - *which* expert each token is sent to.
    variant : str
        Variant of the OGS kernel to use.
        - "raggedT" : ragged-T variant
        - "maxT" : max-T variant
    """

    assert variant in ["raggedT", "maxT"]

    B = A.size(0)
    E = W.size(0)
    device = A.device

    sorting = torch.argsort(top1_expert_per_token, stable=True).to(torch.int32)  # [B]
    print(f"sorting.shape: {sorting.shape}")

    expert_token_counts = torch.bincount(
        top1_expert_per_token, minlength=E
    ).to(torch.int32)  # [E]

    expert_token_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)  # [E+1]
    expert_token_offsets[0] = 0
    expert_token_offsets[1:] = torch.cumsum(expert_token_counts, 0, dtype=torch.int32)

    if variant == "maxT":
        T_max = int(expert_token_counts.max().item())
        C = _moe_matmul_ogs_maxT(
            A,
            W,
            expert_token_counts,
            expert_token_offsets,
            sorting,
            T_max,
        )
    # elif variant == "raggedT":
    #     C = _moe_matmul_ogs_raggedT(
    #         A,
    #         W,
    #         expert_token_counts,
    #         expert_token_offsets,
    #         sorting,
    #     )

    return C


def moe_ref(A: torch.Tensor, W: torch.Tensor, top1_expert_per_token: torch.Tensor) -> torch.Tensor:
    B, K = A.shape
    N = W.size(2)
    device, dtype = A.device, torch.promote_types(A.dtype, W.dtype)

    C = torch.empty(B, N, device=device, dtype=dtype)
    n_experts = W.size(0)

    for e in range(n_experts):
        token_idx = (top1_expert_per_token == e).nonzero(as_tuple=True)[0]
        if token_idx.numel() == 0:
            continue
        C[token_idx] = A[token_idx] @ W[e]  # [Ne, K] @ [K, N]

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

    def _check_variant(variant: str) -> None:
        C_helion = moe_matmul_ogs(A, W, top1_expert_per_token, variant)
        C_ref = moe_ref(A, W, top1_expert_per_token)
        torch.testing.assert_close(C_helion, C_ref, atol=1e-2, rtol=1e-2)

        sec = do_bench(lambda: moe_matmul_ogs(A, W, top1_expert_per_token, variant))
        baseline_sec = do_bench(lambda: moe_ref(A, W, top1_expert_per_token))
        print(f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}s, speed-up: {baseline_sec/sec:.2f}x")

    # _check_variant("raggedT")
    _check_variant("maxT")


if __name__ == "__main__":
    check()
