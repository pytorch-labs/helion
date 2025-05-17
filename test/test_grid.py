from __future__ import annotations

from pathlib import Path
import unittest

from expecttest import TestCase
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import import_path
import helion.language as hl


class TestGrid(TestCase):
    def test_grid_1d(self):
        @helion.kernel
        def grid_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
            for i in hl.grid(b):
                out[i] = torch.mm(x[i], y)
            return out

        def grid_1d_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            b, m, k = x.size()
            k2, n = y.size()
            assert k == k2, f"size mismatch {k} != {k2}"
            out = torch.empty(b, m, n, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
            for i in range(b):
                out[i] = torch.mm(x[i], y)
            return out

        args = (torch.randn([8, 16, 32], device=DEVICE), torch.randn([32, 4], device=DEVICE))
        code, result = code_and_output(grid_1d, args)
        torch.testing.assert_close(result, grid_1d_pytorch(args[0], args[1]))

if __name__ == "__main__":
    unittest.main()
