from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
import helion.language as hl


@helion.kernel
def broadcast_fn(a, b):
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    for tile0, tile1 in hl.tile(out0.size()):
        out0[tile0, tile1] = a[tile0, tile1] + b[tile0, None]
        out1[tile0, tile1] = a[tile0, tile1] + b[None, tile1]
    return out0, out1


def broadcast_fn_ref(a, b):
    out0 = a + b[:, None]
    out1 = a + b[None, :]
    return out0, out1


def _check_broadcast_fn(**config):
    args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
    code, (out0, out1) = code_and_output(broadcast_fn, args, **config)
    ref0, ref1 = broadcast_fn_ref(*args)
    torch.testing.assert_close(out0, ref0)
    torch.testing.assert_close(out1, ref1)
    return code


class TestBroadcasting(TestCase):
    def test_broadcast_no_flatten(self):
        args = [torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE)]
        assert not broadcast_fn.bind(args).config_spec.flatten_loops

    def test_broadcast1(self):
        code = _check_broadcast_fn(
            block_sizes=[16, 8],
        )
        self.assertExpectedJournal(code)

    def test_broadcast2(self):
        code = _check_broadcast_fn(block_size=[16, 8], loop_order=(1, 0))
        self.assertExpectedJournal(code)

    def test_broadcast3(self):
        code = _check_broadcast_fn(
            block_sizes=[64, 1],
        )
        self.assertExpectedJournal(code)

    def test_broadcast4(self):
        code = _check_broadcast_fn(
            block_sizes=[1, 64],
        )
        self.assertExpectedJournal(code)

    def test_broadcast5(self):
        code = _check_broadcast_fn(
            block_sizes=[32, 32],
            indexing="block_ptr",
        )
        self.assertExpectedJournal(code)

    def test_constexpr_index(self):
        @helion.kernel
        def fn(a, idx1):
            out0 = torch.empty_like(a)
            out1 = torch.empty_like(a)
            out2 = torch.empty_like(a)
            idx0 = 11
            for tile0, tile1 in hl.tile(out0.size()):
                out0[tile0, tile1] = a[tile0, tile1] + a[tile0, 3, None]
                out1[tile0, tile1] = a[tile0, tile1] + a[idx0, tile1][None, :]
                out2[tile0, tile1] = a[tile0, tile1] + a[tile0, idx1, None]
            return out0, out1, out2

        args = (torch.randn(512, 512, device=DEVICE), 123)
        code, (out0, out1, out2) = code_and_output(fn, args, block_sizes=[16, 16])
        torch.testing.assert_close(out0, args[0] + args[0][:, 3, None])
        torch.testing.assert_close(out1, args[0] + args[0][11, None, :])
        torch.testing.assert_close(out2, args[0] + args[0][:, args[1], None])
        self.assertExpectedJournal(code)

    def test_implicit_broadcast(self):
        @helion.kernel
        def fn(a, b):
            out = torch.empty_like(a)
            for tile0, tile1 in hl.tile(a.size()):
                out[tile0, tile1] = a[tile0, tile1] + b[tile1]
            return out

        args = (torch.randn(512, 512, device=DEVICE), torch.randn(512, device=DEVICE))
        code, out = code_and_output(fn, args, block_sizes=[16, 16])
        torch.testing.assert_close(out, sum(args))
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
