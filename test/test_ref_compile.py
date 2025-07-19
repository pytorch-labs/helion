#!/usr/bin/env python3
"""Test reference implementation in torch.compile mode using HELION_REF_COMPILE=1.
"""

import os
import unittest
import pytest

# Import the module to get test methods without importing the class
from . import test_examples

# Import test utilities for the print tests
import torch
import helion
import helion.language as hl
from helion._testing import TestCase


class TestExamplesRefCompile(test_examples.TestExamples):
    """Run all TestExamples tests in reference torch.compile mode via HELION_REF_COMPILE=1."""
    
    # NOTE: All tests in TestExamples are run in torch.compile(fullgraph=True) mode by default in this test file.
    # Below, we override specific tests with smaller input sizes to make torch.compile mode tests run faster.
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var
        self._original_compile = os.environ.get("HELION_REF_COMPILE")
        # Set ref compile mode
        os.environ["HELION_REF_COMPILE"] = "1"
    
    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_compile is not None:
            os.environ["HELION_REF_COMPILE"] = self._original_compile
        elif "HELION_REF_COMPILE" in os.environ:
            del os.environ["HELION_REF_COMPILE"]
    
    def test_add(self):
        """Override test_add to verify ref compile mode execution."""
        from torch._dynamo.utils import counters
        
        # Clear counters before running the test
        counters.clear()
        
        # Run the original test
        super().test_add()
        
        # In ref compile mode, torch.compile SHOULD be called
        # Check for compilation-related counters
        frames_total = counters.get('frames', {}).get('total', 0)
        aot_total = counters.get('aot_autograd', {}).get('total', 0)
        
        self.assertGreater(frames_total, 0, 
                          f"torch.compile should be invoked in ref compile mode. frames.total={frames_total}")
        self.assertGreater(aot_total, 0,
                          f"AOT autograd should be invoked in ref compile mode. aot_autograd.total={aot_total}")
        
        # Also check for specific compile indicators
        stats_captured = counters.get('stats', {}).get('calls_captured', 0)
        self.assertGreater(stats_captured, 0,
                          f"Compilation should capture calls in ref compile mode. stats.calls_captured={stats_captured}")
        
        print(f"\n✓ Ref compile mode verified: torch.compile execution detected")
        print(f"  - frames.total: {frames_total}")
        print(f"  - aot_autograd.total: {aot_total}")
        print(f"  - stats.calls_captured: {stats_captured}")
    
    def test_bmm(self):
        # Use more relaxed tolerances for float16 operations in torch.compile mode
        # The slight accuracy difference is expected due to different accumulation strategies
        super().test_bmm(atol=1e-3, rtol=1e-3)

    @pytest.mark.skip(reason="torch.compile doesn't support data-dependent branching (if num_tokens != 0)")
    def test_moe_matmul_ogs(self):
        super().test_moe_matmul_ogs()


class TestKernelRefCompileParam(TestCase):
    """Test @helion.kernel(ref_compile=True) parameter functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var to ensure clean state
        self._original_compile = os.environ.get("HELION_REF_COMPILE")
        # Remove env var to test parameter-only behavior
        if "HELION_REF_COMPILE" in os.environ:
            del os.environ["HELION_REF_COMPILE"]
    
    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_compile is not None:
            os.environ["HELION_REF_COMPILE"] = self._original_compile
    
    def test_ref_compile_param_simple(self):
        """Test that ref_compile=True parameter works for simple kernels."""
        @helion.kernel(ref_compile=True)
        def add_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b
        
        # Test execution
        a = torch.tensor([1.0, 2.0], device='cuda')
        b = torch.tensor([3.0, 4.0], device='cuda')
        result = add_kernel(a, b)
        expected = a + b
        torch.testing.assert_close(result, expected)
    
    def test_ref_compile_param_complex(self):
        """Test that ref_compile=True parameter works for complex kernels."""
        @helion.kernel(ref_compile=True)
        def matmul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, k = a.shape
            k2, n = b.shape
            assert k == k2
            out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
            
            for tile_m, tile_n in hl.tile([m, n]):
                acc = torch.zeros_like(out[tile_m, tile_n])
                for tile_k in hl.tile(k):
                    acc += a[tile_m, tile_k] @ b[tile_k, tile_n]
                out[tile_m, tile_n] = acc
            return out
        
        # Test execution
        a = torch.randn(8, 12, device='cuda')
        b = torch.randn(12, 16, device='cuda')
        result = matmul_kernel(a, b)
        expected = a @ b
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    
    def test_ref_compile_param_overrides_env(self):
        """Test that ref_compile=False parameter overrides env var."""
        # Set env var to enable ref compile mode
        os.environ["HELION_REF_COMPILE"] = "1"
        
        # Create kernel with ref_compile=False (should override env var)
        @helion.kernel(ref_compile=False)
        def mul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a * b
        
        # Note: We can't easily test that Helion compilation happens without
        # inspecting internal state, but we can verify it executes correctly
        a = torch.tensor([2.0, 3.0], device='cuda')
        b = torch.tensor([4.0, 5.0], device='cuda')
        result = mul_kernel(a, b)
        expected = a * b
        torch.testing.assert_close(result, expected)
    
    def test_ref_compile_param_with_config(self):
        """Test that ref_compile=True works with kernel configs."""
        from helion import Config
        
        # Create a simple config
        config = Config(block_sizes=[128])
        
        @helion.kernel(ref_compile=True, config=config)
        def configured_kernel(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0
        
        # Test execution
        x = torch.randn(256, device='cuda')
        result = configured_kernel(x)
        expected = x * 2.0
        torch.testing.assert_close(result, expected)
    
    def test_ref_compile_with_torch_compile_verification(self):
        """Test that ref_compile=True actually uses torch.compile."""
        from torch._dynamo.utils import counters
        
        # Clear counters before running the test
        counters.clear()
        
        @helion.kernel(ref_compile=True)
        def compile_test(x: torch.Tensor) -> torch.Tensor:
            return x + x * 2.0
        
        # Test execution
        x = torch.randn(128, device='cuda')
        result = compile_test(x)
        expected = x + x * 2.0
        torch.testing.assert_close(result, expected)
        
        # In ref compile mode, torch.compile SHOULD be called
        frames_total = counters.get('frames', {}).get('total', 0)
        aot_total = counters.get('aot_autograd', {}).get('total', 0)
        
        self.assertGreater(frames_total, 0, 
                          f"torch.compile should be invoked with ref_compile=True. frames.total={frames_total}")
        self.assertGreater(aot_total, 0,
                          f"AOT autograd should be invoked with ref_compile=True. aot_autograd.total={aot_total}")


if __name__ == "__main__":
    unittest.main()
