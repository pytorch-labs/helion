from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion


@helion.jit(
    config=helion.Config(
        block_sizes=[24],
        num_warps=32,
        indexing="pointers",
    ),
    static_shapes=True,
)
def one_shot_all_reduce_kernel(
    buffer_ptr_addrs,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output = torch.empty_like(x)
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequenceMemAccess=True
    )

    pid = tl.program_id(axis=0)
    buffer_ptr_addrs = buffer_ptr_addrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.bfloat16))
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        # Each thread processes 128 bits.
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.bfloat16)
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptr_addrs + i).to(tl.pointer_type(tl.bfloat16))
            tl.multiple_of(buffer_ptr, 16)
            x = tl.load(buffer_ptr + offsets, mask=mask)
            acc += x
        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )


def one_shot_all_reduce(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    config = {
        "max_num_blocks": kwargs.get("max_num_blocks", 24),
        "num_warps": kwargs.get("num_warps", 32),
        "BLOCK_SIZE": kwargs.get("BLOCK_SIZE", 8192),
    }

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert tensor.numel() % 8 == 0, "The number of elements must be 128-bit aligned."
    assert config["BLOCK_SIZE"] % (config["num_warps"] * 32) == 0, (
        "BLOCK_SIZE must be a multiple of num_warps * 32"
    )

    num_blocks = min(
        triton.cdiv(tensor.numel(), config["BLOCK_SIZE"]), config["max_num_blocks"]
    )

    symm_mem_hdl = symm_mem.rendezvous(tensor, group=dist.group.WORLD)
    output = torch.empty_like(tensor)

    signal_pads = tuple(
        [
            symm_mem_hdl.get_signal_pad(i, dtype=torch.int32)
            for i in range(symm_mem_hdl.world_size)
        ]
    )

    one_shot_all_reduce_kernel[(num_blocks, 1, 1)](
        symm_mem_hdl.buffer_ptrs_dev,
        signal_pads,
        output,
        numel=tensor.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=config["BLOCK_SIZE"],
        num_warps=config["num_warps"],
    )

    return output
