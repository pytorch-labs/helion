"""
Matmul-of-Experts (MoE) with Outer-Gather-Scatter (OGS)
"""

import helion
import helion.language as hl
import torch


@helion.kernel(static_shapes=False)
def _moe_matmul_ogs_maxT(
    A: torch.Tensor,                          # [B, K]
    W: torch.Tensor,                          # [E, K, N]
    expert_token_counts: torch.Tensor,        # [E]
    expert_token_offsets: torch.Tensor,       # [E + 1]
    sorted_to_orig_token_idx: torch.Tensor,   # [B]
    T_max_tensor: torch.Tensor,               # [T_max]
):
    """
    Implements the same algorithm as the Triton reference kernel
    __moe_matmul_ogs_maxT_kernel(), including every masking /
    alias-avoidance trick used there.
    """

    B, K = A.shape
    E, _, N = W.shape
    T_max = T_max_tensor.numel()

    C = torch.empty(
        B, N,
        dtype=torch.promote_types(A.dtype, W.dtype),
        device=A.device,
    )

    # ──────────────────────────────────────────────────────────────────
    # pre-materialise [0 … T_max-1] so we can index it tile-wise
    # ──────────────────────────────────────────────────────────────────
    row_ids = torch.arange(T_max, device=A.device, dtype=torch.int32)
    k_ids   = torch.arange(K, device=A.device, dtype=torch.int32)

    for e_idx in hl.grid(E):

        # ----------------------------------------------------------------
        # 1️⃣  SAME “OGS” METADATA AS IN TRITON
        # ----------------------------------------------------------------
        start       = expert_token_offsets[e_idx]
        num_tokens  = expert_token_counts[e_idx]

        if num_tokens != 0:                                   # v_1

            for tile_t, tile_n in hl.tile([T_max, N]):

                # ----------------------------------------------------------------
                # 2️⃣  BUILD PER-TILE “ROW VALID” MASK  (v_3 in Triton)
                # ----------------------------------------------------------------
                local_rows   = row_ids[tile_t].squeeze(0)     # shape [BLOCK_T]
                row_valid    = local_rows < num_tokens        # bool [BLOCK_T]

                # ----------------------------------------------------------------
                # 3️⃣  SAFE ROW OFFSET WITHOUT **ALIASING** REAL TOKENS
                #     The Triton variant points dummy rows at `num_tokens`
                #     (one *past* the slice) instead of `num_tokens-1`.        v_5
                # ----------------------------------------------------------------
                safe_offset  = torch.where(
                    row_valid,
                    local_rows,
                    num_tokens - 1,
                )

                orig_rows_idxes = start + safe_offset               # [1, BLOCK_T]
                orig_rows       = sorted_to_orig_token_idx[
                    orig_rows_idxes.squeeze(0)
                ]                                                  # [BLOCK_T]

                # ----------------------------------------------------------------
                # 4️⃣  ACCUMULATOR
                # ----------------------------------------------------------------
                acc = hl.zeros([tile_t, tile_n], dtype=torch.float32)

                # ----------------------------------------------------------------
                # 5️⃣  INNER K LOOP  (offset_3 in Triton)
                #     We replicate Triton’s “masked load” so that
                #     no data from padding rows is ever fetched.                A
                # ----------------------------------------------------------------
                for tile_k in hl.tile(K):

                    col_ids = k_ids[tile_k].squeeze(0)          # numeric col indices
                    col_valid = col_ids < K                     # bool [BLOCK_K]

                    load_mask = row_valid[:, None] & col_valid[None, :]

                    A_frag = A[orig_rows[:, None].squeeze(1), tile_k].masked_fill(~load_mask, 0).squeeze(0)

                    W_frag = W[e_idx, tile_k, tile_n]               # [BLOCK_K, BLOCK_N]

                    acc = torch.addmm(acc, A_frag, W_frag)          # TF32 is on by default

                # # ─ after the inner K loop ends ─────────────────────────────────────
                # #
                # # • orig_rows      : 1-D Long[BLOCK_T]  (batch-order row ids)
                # # • row_valid      : 1-D Bool[BLOCK_T]  (True ⇔ real token row)
                # # • acc            : 2-D Float[BLOCK_T, BLOCK_N]  (the tile result)

                # block_N = acc.size(1)                             # concrete, no .numel on tile_n
                # valid_store_mask = row_valid[:, None].expand(-1, block_N)   # Bool[BT, BN]

                # # 1. Slice exactly the same tile from C as we have in `acc`
                # C_tile = C[orig_rows, :][:, tile_n]                   # view into C, no copy

                # # 2. Scatter only the elements that correspond to *real* rows
                # C_tile.masked_scatter_(valid_store_mask, acc[valid_store_mask])

                # C[orig_rows[row_valid], tile_n] = acc[row_valid.squeeze(0), :]

                block_N = acc.size(1)                       # concrete
                flat_C   = C[orig_rows, tile_n].flatten()         # view
                flat_acc = acc.flatten()
                flat_mask = row_valid[:, None].expand(-1, block_N).flatten()

                flat_C[flat_mask] = flat_acc[flat_mask]      # 1-D bool mask → OK in Helion

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
            torch.empty(T_max, device=device),
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
