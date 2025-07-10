from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


class TestWait(TestCase):
    def test_wait_basic(self):
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
        self.assertExpectedJournal(code)

    def test_wait_2d_tile(self):
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
        self.assertExpectedJournal(code)

    def test_wait_multi_bar(self):
        @helion.kernel
        def gmem_wait_multi_bar_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (N,) = signal_pad.shape
            n = hl.register_block_size(N)
            out = torch.empty(n, dtype=torch.int32, device=DEVICE)

            for tile in hl.tile(N, block_size=n):
                hl.wait(signal_pad, [tile], signal=1)
                out[tile.id] = tile.id

            return out

        signal_pad = torch.ones(16, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gmem_wait_multi_bar_kernel, (signal_pad,), block_size=[4]
        )
        torch.testing.assert_close(
            result, torch.arange(4, device=DEVICE, dtype=torch.int32)
        )
        self.maxDiff = None
        self.assertExpectedJournal(code)

    def test_wait_multi_bar_cas(self):
        @helion.kernel
        def gmem_wait_multi_bar_kernel_cas(signal_pad: torch.Tensor) -> torch.Tensor:
            (N,) = signal_pad.shape
            n = hl.register_block_size(N)

            for tile in hl.tile(N, block_size=n):
                hl.wait(signal_pad, [tile], signal=1, update=2, op="atomic_cas")

            return signal_pad

        signal_pad = torch.ones(16, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gmem_wait_multi_bar_kernel_cas, (signal_pad,), block_size=[4]
        )
        torch.testing.assert_close(
            result, torch.full((16,), fill_value=2, device=DEVICE, dtype=torch.int32)
        )
        self.maxDiff = None
        self.assertExpectedJournal(code)

    def test_signal_basic(self):
        @helion.kernel
        def gmem_signal_scalar_bar_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for i in hl.grid(n):
                hl.signal(signal_pad, [i], signal=1)
            return signal_pad

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gmem_signal_scalar_bar_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.ones(4, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_signal_cas(self):
        @helion.kernel
        def gmem_signal_cas_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for i in hl.grid(n):
                hl.signal(signal_pad, [i], signal=1, wait_for=0, op="atomic_cas")
            return signal_pad

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(gmem_signal_cas_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.ones(4, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_signal_multiple(self):
        @helion.kernel
        def gmem_signal_tensor_bar_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for tile in hl.tile(n):
                hl.signal(signal_pad, [tile], signal=1)
            return signal_pad

        signal_pad = torch.zeros(16, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gmem_signal_tensor_bar_kernel,
            (signal_pad,),
            block_size=[4],
        )
        torch.testing.assert_close(
            result, torch.ones(16, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_signal_multiple_cas(self):
        @helion.kernel
        def gmem_signal_tensor_bar_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for tile in hl.tile(n):
                hl.signal(signal_pad, [tile], wait_for=0, signal=1, op="atomic_cas")
            return signal_pad

        signal_pad = torch.zeros(16, device=DEVICE, dtype=torch.int32)
        code, result = code_and_output(
            gmem_signal_tensor_bar_kernel,
            (signal_pad,),
            block_size=[4],
        )
        torch.testing.assert_close(
            result, torch.ones(16, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_send_recieve_cta(self):
        @helion.kernel
        def gmem_signal_n_wait_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            (n,) = signal_pad.shape
            for i in hl.grid(n):  # first N ctas sends signal
                hl.signal(signal_pad, [i], signal=1)
            for i in hl.grid(n):  # last N ctas waits for signal
                hl.wait(signal_pad, [i], signal=1)
            return signal_pad

        signal_pad = torch.zeros(4, device=DEVICE, dtype=torch.int32)

        code, result = code_and_output(gmem_signal_n_wait_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.ones(4, device=DEVICE, dtype=torch.int32)
        )
        self.assertIn("helion.runtime.triton_send_signal", code)
        self.assertIn("helion.runtime.triton_wait_signal", code)

    def test_global_sync(self):
        @helion.kernel
        def gmem_multi_bar_sync_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            M, N = signal_pad.shape
            assert M == N
            for i in hl.grid(N):
                for tile in hl.tile(N, block_size=N):
                    hl.signal(signal_pad, [tile, i], signal=1, skip_sync=True)
                    hl.wait(signal_pad, [i, tile], signal=1)
            return signal_pad

        signal_pad = torch.zeros(4, 4, device=DEVICE, dtype=torch.int32)

        code, result = code_and_output(gmem_multi_bar_sync_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.ones(4, 4, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_global_sync_cas(self):
        @helion.kernel
        def gmem_multi_bar_sync_kernel(signal_pad: torch.Tensor) -> torch.Tensor:
            M, N = signal_pad.shape
            assert M == N
            for i in hl.grid(N):
                for tile in hl.tile(N, block_size=N):
                    hl.signal(
                        signal_pad,
                        [tile, i],
                        signal=1,
                        wait_for=0,
                        skip_sync=True,
                        op="atomic_cas",
                    )
                    hl.wait(signal_pad, [i, tile], signal=1, update=2, op="atomic_cas")
            return signal_pad

        signal_pad = torch.zeros(4, 4, device=DEVICE, dtype=torch.int32)

        code, result = code_and_output(gmem_multi_bar_sync_kernel, (signal_pad,))
        torch.testing.assert_close(
            result, torch.full((4, 4), fill_value=2, device=DEVICE, dtype=torch.int32)
        )
        self.assertIn("atomic_cas", code)

    def test_wait_pointers(self):
        @helion.kernel
        def gmem_wait_pointers_kernel(
            signal_pad_ptrs: torch.Tensor, pad_shape: hl.constexpr
        ) -> torch.Tensor:
            out = torch.empty(
                pad_shape, device=signal_pad_ptrs.device, dtype=torch.int32
            )
            N = signal_pad_ptrs.size(0)
            for i in hl.grid(pad_shape):
                offset = i * 4  # number of btypes of each pointer (torch.int32)
                for multicast_tile in hl.tile(N, block_size=N):
                    signal_pads = (
                        signal_pad_ptrs[multicast_tile] + offset
                    )  # Load the pointers, and broadcast the offset
                    hl.wait(signal_pads, signal=1, as_ptrs=True)
                    out[i] = i
            return out

        signal_pad_list = [
            torch.ones(4, device=DEVICE, dtype=torch.int32) for _ in range(4)
        ]
        signal_pad_ptrs = torch.as_tensor(
            [p.data_ptr() for p in signal_pad_list], device=DEVICE, dtype=torch.uint64
        )
        code, result = code_and_output(gmem_wait_pointers_kernel, (signal_pad_ptrs, 4))
        torch.testing.assert_close(
            result, torch.arange(4, device=DEVICE, dtype=torch.int32)
        )
        self.assertExpectedJournal(code)

    def test_signal_pointers(self):
        @helion.kernel
        def gmem_signal_pointers_kernel(
            signal_pad_ptrs: torch.Tensor, pad_shape: hl.constexpr
        ) -> torch.Tensor:
            N = signal_pad_ptrs.size(0)
            for i in hl.grid(pad_shape):
                offset = i * 4  # number of btypes of each pointer (torch.int32)
                for multicast_tile in hl.tile(N, block_size=N):
                    signal_pads = signal_pad_ptrs[multicast_tile] + offset
                    hl.signal(signal_pads, signal=1, as_ptrs=True)
            return signal_pad_ptrs

        signal_pad_list = [
            torch.zeros(4, device=DEVICE, dtype=torch.int32) for _ in range(4)
        ]
        signal_pad_ptrs = torch.as_tensor(
            [p.data_ptr() for p in signal_pad_list], device=DEVICE, dtype=torch.uint64
        )
        code, result = code_and_output(
            gmem_signal_pointers_kernel, (signal_pad_ptrs, 4)
        )

        for tensor in signal_pad_list:
            torch.testing.assert_close(
                tensor, torch.ones(4, device=DEVICE, dtype=torch.int32)
            )
        self.assertExpectedJournal(code)

    def test_global_sync_on_pointers(self):
        @helion.kernel
        def gmem_multi_bar_sync_ptrs_kernel(
            signal_pad_ptrs: torch.Tensor,
            pad_shape: hl.constexpr,
        ) -> torch.Tensor:
            N = hl.specialize(signal_pad_ptrs.size(0))
            for i in hl.grid(pad_shape):
                signal_offset = (
                    i * 4
                )  # 4 = number of btypes of each pointer (torch.int32)
                for multicast_tile in hl.tile(N, block_size=N):
                    hl.signal(
                        signal_pad_ptrs[multicast_tile] + signal_offset,
                        signal=1,
                        skip_sync=True,
                        as_ptrs=True,
                    )
                wait_offsets = torch.arange(N, device=DEVICE, dtype=torch.uint64) * 4
                wait_ptrs = signal_pad_ptrs[i] + wait_offsets
                hl.wait(wait_ptrs, signal=1, as_ptrs=True)

        signal_pad_list = [
            torch.zeros(4, device=DEVICE, dtype=torch.int32) for _ in range(4)
        ]
        signal_pad_ptrs = torch.as_tensor(
            [p.data_ptr() for p in signal_pad_list], device=DEVICE, dtype=torch.uint64
        )

        code, result = code_and_output(
            gmem_multi_bar_sync_ptrs_kernel, (signal_pad_ptrs, 4)
        )
        for tensor in signal_pad_list:
            torch.testing.assert_close(
                tensor, torch.ones(4, device=DEVICE, dtype=torch.int32)
            )

    def test_global_sync_on_pointers_cas(self):
        @helion.kernel
        def gmem_multi_bar_sync_ptrs_kernel(
            signal_pad_ptrs: torch.Tensor,
            pad_shape: hl.constexpr,
        ) -> torch.Tensor:
            N = hl.specialize(signal_pad_ptrs.size(0))
            for i in hl.grid(pad_shape):
                signal_offset = (
                    i * 4
                )  # 4 = number of btypes of each pointer (torch.int32)
                for multicast_tile in hl.tile(N, block_size=N):
                    hl.signal(
                        signal_pad_ptrs[multicast_tile] + signal_offset,
                        wait_for=0,
                        signal=1,
                        op="atomic_cas",
                        skip_sync=True,
                        as_ptrs=True,
                    )
                wait_offsets = torch.arange(N, device=DEVICE, dtype=torch.uint64) * 4
                wait_ptrs = signal_pad_ptrs[i] + wait_offsets
                hl.wait(wait_ptrs, signal=1, update=0, op="atomic_cas", as_ptrs=True)

        signal_pad_list = [
            torch.zeros(4, device=DEVICE, dtype=torch.int32) for _ in range(4)
        ]
        signal_pad_ptrs = torch.as_tensor(
            [p.data_ptr() for p in signal_pad_list], device=DEVICE, dtype=torch.uint64
        )

        code, result = code_and_output(
            gmem_multi_bar_sync_ptrs_kernel, (signal_pad_ptrs, 4)
        )
        for tensor in signal_pad_list:
            torch.testing.assert_close(
                tensor, torch.zeros(4, device=DEVICE, dtype=torch.int32)
            )


if __name__ == "__main__":
    unittest.main()
