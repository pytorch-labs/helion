from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl


@helion.kernel(
    static_shapes=True,
    config=helion.Config(
        block_sizes=[64, 64], num_warps=8, num_stages=4, indexing="block_ptr"
    ),
)
def wait_and_copy_kernel(x: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    # TODO: call proper API to auto generate it based on tilesize & tensor shape/stride.
    """Test Spinning on global memory signal pad."""
    m, n = x.size()
    # block_m = hl.register_block_size(m)
    # block_n = hl.register_block_size(n)

    # print(block_m)

    # tiles_m = (m + block_m - 1) // block_m # cdiv
    # tiles_n = (n + block_n - 1) // block_n # cdiv

    print("progress size:", progress.size())
    progress = progress.view(-1, 128)
    print("progress shape", progress.size(), progress.stride())

    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        # index_m, index_n = hl.get_tile_index([tile_m, tile_n])
        hl.wait(
            progress,
            [tile_m.begin, tile_n.begin],
            signal=1,
            update=None,
            op="ld",
            scope="gpu",
            sem="acquire",
        )
        out[tile_m, tile_n] = x[tile_m, tile_n]

    return out


@helion.kernel(static_shapes=True)
def atomic_add_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n]
        hl.atomic_add(out, [tile_m, tile_n], 1)
    return out


def test_tile_id():
    @helion.kernel(
        static_shapes=True,
        config=helion.Config(
            block_sizes=[
                16,
            ],
            num_warps=8,
            num_stages=4,
            indexing="block_ptr",
        ),
    )
    def test_tile_id_access(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.int32)
        for tile in hl.tile(x.size(0)):
            out[tile] = tile.id
        return out

    x = torch.randn([64], device=DEVICE)
    result = test_tile_id_access(x)
    print(result)


def test_tile_id_indexing():
    @helion.kernel(
        static_shapes=True,
        config=helion.Config(
            block_sizes=[16, 16],
        ),
    )
    def test_tile_id_atomic_add(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.int32)
        for tile_m, tile_n in hl.tile(x.size()):
            hl.atomic_add(out, [tile_m.id, tile_n.id], 1)
        return out

    x = torch.randn([64, 64], device=DEVICE)
    result = test_tile_id_atomic_add(x)
    print(result)


if __name__ == "__main__":
    # test_tile_id()
    test_tile_id_indexing()
    # m = 4096
    # n = 16384
    # x = torch.randn([m, n], device="cuda", dtype=torch.float32)
    # progress = torch.zeros(4096, device="cuda", dtype=torch.int32)
    # wait_and_copy_kernel(x, progress)

    # atomic_add_kernel(x)
