from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Symmemtric Memory Helpers
from torch.utils.cpp_extension import load_inline

import helion
import helion.language as hl

from_blob_cpp = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


at::Tensor from_blob(uint64_t data_ptr, c10::IntArrayRef sizes, py::object dtype) {

    at::Tensor tensor = at::for_blob((void*)data_ptr, sizes)
             .deleter([](void *ptr) {
               ;
             })
             .options(at::device(at::kCUDA).dtype(((THPDtype*)dtype.ptr())->scalar_type))
             .make_tensor();

    return tensor;
}
"""

cpp_mod = load_inline(
    "cpp_mod", cpp_sources=from_blob_cpp, with_cuda=True, functions=["from_blob"]
)


def dev_array_to_tensor_short(
    dev_array_ptr: int, shape: tuple[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return cpp_mod.from_blob(dev_array_ptr, shape, dtype)  # pyright: ignore[reportAttributeAccessIssue]


@helion.jit(
    config=helion.Config(
        block_sizes=[8192],
        num_warps=32,
    ),
    static_shapes=True,
)
def one_shot_all_reduce_kernel(
    signal_pad_addrs: torch.Tensor,
    local_signal_pad: torch.Tensor,
    a_shared_tuple: tuple[torch.Tensor, ...],
    my_rank: hl.constexpr,
) -> torch.Tensor:
    _, world_size = local_signal_pad.size()
    world_size = hl.specialize(world_size)
    out = torch.empty_like(a_shared_tuple[0])
    N = out.size(0)

    for tile_n in hl.tile(N):
        ptr_tile = signal_pad_addrs[:]
        multicast_signalpad = hl.multicast_like(local_signal_pad, ptr_tile)
        hl.signal(
            multicast_signalpad,
            [tile_n.id, my_rank],
            signal=1,
            wait_for=0,
            scope="sys",
            hasPreviousMemAccess=False,
        )

        for world in hl.tile(world_size, block_size=world_size):
            hl.wait(
                local_signal_pad,
                [tile_n.id, world],
                signal=1,
                update=0,
                scope="sys",
            )

        acc = hl.zeros(
            [tile_n], dtype=a_shared_tuple[0].dtype, device=local_signal_pad.device
        )

        for a in a_shared_tuple:
            acc += a[tile_n]

        out[tile_n] = acc

        hl.signal(
            multicast_signalpad, [tile_n.id, my_rank], signal=1, wait_for=0, scope="sys"
        )

        for world in hl.tile(world_size, block_size=world_size):
            hl.wait(
                local_signal_pad,
                [tile_n.id, world],
                signal=1,
                update=0,
                scope="sys",
                hasSubsequentMemAccess=False,
            )
    return out


def helion_one_shot_all_reduce(a_shared: torch.Tensor) -> torch.Tensor:
    assert dist.group.WORLD is not None

    symm_mem_hdl = symm_mem.rendezvous(a_shared, group=dist.group.WORLD)

    a_shared_tuple = tuple(
        [
            symm_mem_hdl.get_buffer(i, tuple(a_shared.shape), a_shared.dtype)
            for i in range(symm_mem_hdl.world_size)
        ]
    )

    local_signal_pad = symm_mem_hdl.get_signal_pad(
        symm_mem_hdl.rank, dtype=torch.int32
    ).view(-1, symm_mem_hdl.world_size)

    signal_pad_addrs = dev_array_to_tensor_short(
        symm_mem_hdl.signal_pad_ptrs_dev,
        (symm_mem_hdl.world_size,),
        dtype=torch.uint64,
        device=a_shared.device,
    )

    return one_shot_all_reduce_kernel(
        signal_pad_addrs,
        local_signal_pad,
        a_shared_tuple,
        my_rank=symm_mem_hdl.rank,
    )


def test(N: int, device: torch.device, dtype: torch.dtype) -> None:
    dist_group = dist.group.WORLD
    assert dist_group is not None

    world_size = dist.get_world_size()
    a_shared = symm_mem.empty(N // world_size, dtype=dtype, device=device).normal_()

    a_shared_clone = symm_mem.empty(
        a_shared.shape,
        dtype=a_shared.dtype,
        device=a_shared.device,
    )
    symm_mem.rendezvous(a_shared_clone, dist_group.group_name)
    a_shared_clone.copy_(a_shared)

    a_out = helion_one_shot_all_reduce(a_shared)

    gloden_o = torch.ops.symm_mem.one_shot_all_reduce(
        a_shared_clone, "sum", dist_group.group_name
    )

    torch.testing.assert_close(a_out, gloden_o, rtol=1e-1, atol=1e-1)


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test(16384, device, torch.bfloat16)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 examples/all_reduce.py
    """
    main()
