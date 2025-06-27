from __future__ import annotations

import unittest

import helion
import helion.language as hl
import torch

from expecttest import TestCase
from helion._testing import code_and_output, DEVICE


class TestWait(TestCase):
    def test_basic_wait(self):
        @helion.kernel
        def gmem_wait_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(signal_pad)
            (n,) = signal_pad.shape
            for i in hl.grid(n):
                hl.wait(signal_pad, [i], signal=1)
                out[i] = i

            return out

        signal_pad = torch.ones(4, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gmem_wait_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.arange(4, device=DEVICE, dtype=torch.int32)
        )
        self.maxDiff = None
        self.assertIn(
            "from helion import _triton_ext as hl_ext", code
        )  # Import hl_ext.
        self.assertIn(
            """\
@triton.jit
def _gmem_wait_kernel_kernel(signal_pad, out, out_stride_0, signal_pad_stride_0):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    hl_ext._triton_wait_signal(addr=signal_pad + offset_0 * signal_pad_stride_0, expect=1, update=0, sem='acquire', scope='gpu', op='ld', skip_sync=False)
    tl.store(out + offset_0 * out_stride_0, offset_0, None)""",
            code,
        )

    def test_2d_tile_wait(self):
        @helion.kernel
        def wait_for_2d_tile_kernel(
            signal_pad: torch.Tensor, x: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            (n, m) = x.shape
            for tile_n, tile_m in hl.tile([n, m]):
                hl.wait(signal_pad, [tile_n.id, tile_m.id], signal=1)
                out[tile_n, tile_m] = x[tile_n, tile_m]
            return out

        signal_pad = torch.ones([4, 4], device=DEVICE, dtype=torch.int32)
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        code, result = code_and_output(
            wait_for_2d_tile_kernel,
            (signal_pad, x),
            block_size=[16, 16],
        )

        torch.testing.assert_close(result, x)
        self.assertIn(
            """\
@triton.jit
def _wait_for_2d_tile_kernel_kernel(signal_pad, x, out, out_stride_0, out_stride_1, signal_pad_stride_0, signal_pad_stride_1, x_stride_0, x_stride_1, n, m, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr):
    num_blocks_0 = tl.cdiv(n, _BLOCK_SIZE_0)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < n
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < m
    tile_id = offset_0 // _BLOCK_SIZE_0
    tile_id_1 = offset_1 // _BLOCK_SIZE_1
    hl_ext._triton_wait_signal(addr=signal_pad + (tile_id * signal_pad_stride_0 + tile_id_1 * signal_pad_stride_1), expect=1, update=0, sem='acquire', scope='gpu', op='ld', skip_sync=False)
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_0[:, None] & mask_1[None, :], other=0)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), load, mask_0[:, None] & mask_1[None, :])

def wait_for_2d_tile_kernel(signal_pad: torch.Tensor, x: torch.Tensor):
    out = torch.empty_like(x)
    n, m = x.shape
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    _wait_for_2d_tile_kernel_kernel[triton.cdiv(n, _BLOCK_SIZE_0) * triton.cdiv(m, _BLOCK_SIZE_1),](signal_pad, x, out, out.stride(0), out.stride(1), signal_pad.stride(0), signal_pad.stride(1), x.stride(0), x.stride(1), n, m, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)
    return out

def _wait_for_2d_tile_kernel_make_precompiler(signal_pad: torch.Tensor, x: torch.Tensor):
    out = torch.empty_like(x)
    n, m = x.shape
    _BLOCK_SIZE_0 = 16
    _BLOCK_SIZE_1 = 16
    from helion.runtime.precompile_shim import make_precompiler
    return make_precompiler(_wait_for_2d_tile_kernel_kernel)(signal_pad, x, out, out.stride(0), out.stride(1), signal_pad.stride(0), signal_pad.stride(1), x.stride(0), x.stride(1), n, m, _BLOCK_SIZE_0, _BLOCK_SIZE_1, num_warps=4, num_stages=3)""",
            code,
        )


if __name__ == "__main__":
    unittest.main()
