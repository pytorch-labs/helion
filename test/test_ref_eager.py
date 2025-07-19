#!/usr/bin/env python3
"""Test reference implementation in ref mode using HELION_REF_EAGER=1.
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


class TestExamplesRefEager(test_examples.TestExamples):
    """Run all TestExamples tests in reference ref mode via HELION_REF_EAGER=1."""
    
    # NOTE: All tests in TestExamples are run in ref mode by default in this test file.
    # Below, we override specific tests with smaller input sizes to make ref mode tests run faster.
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var
        self._original_eager = os.environ.get("HELION_REF_EAGER")
        # Set ref eager mode
        os.environ["HELION_REF_EAGER"] = "1"
    
    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_eager is not None:
            os.environ["HELION_REF_EAGER"] = self._original_eager
        elif "HELION_REF_EAGER" in os.environ:
            del os.environ["HELION_REF_EAGER"]
    
    def test_add(self):
        """Override test_add to verify ref eager mode execution."""
        from torch._dynamo.utils import counters
        
        # Clear counters before running the test
        counters.clear()
        
        # Run the original test
        super().test_add()
        
        # In ref eager mode, torch.compile should NOT be called
        # So there should be no compilation-related counters
        self.assertEqual(counters.get('frames', {}).get('total', 0), 0, 
                        "torch.compile should not be invoked in ref eager mode")
        self.assertEqual(counters.get('aot_autograd', {}).get('total', 0), 0,
                        "AOT autograd should not be invoked in ref eager mode")
        
        print("\n✓ Ref eager mode verified: No torch.compile execution detected")
    
    def test_bmm(self):
        # Use more relaxed tolerances for float16 operations in ref mode
        # The slight accuracy difference is expected due to different accumulation strategies
        super().test_bmm(atol=1e-3, rtol=1e-3)

class TestExamplesRefEagerPrint(TestCase):
    """Test print functionality specific to reference eager mode."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var
        self._original_eager = os.environ.get("HELION_REF_EAGER")
        # Set ref eager mode
        os.environ["HELION_REF_EAGER"] = "1"
    
    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_eager is not None:
            os.environ["HELION_REF_EAGER"] = self._original_eager
        elif "HELION_REF_EAGER" in os.environ:
            del os.environ["HELION_REF_EAGER"]
    
    def test_print_ref_eager(self):
        """Test print functionality in reference eager mode."""
        @helion.kernel
        def print_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                print("tensor value: ", val)
                print("multiple values: ", val, val * 2)
                out[tile_m, tile_n] = val * 2
            return out
        
        # Create predictable input
        x = torch.ones([2, 2], device='cuda', dtype=torch.float32) * 42.0
        expected = x * 2
        
        # Kernel should automatically run in reference eager mode due to HELION_REF_EAGER=1
        result = print_kernel(x)
        
        # Check accuracy
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        
        # Get the generated code for verification
        bound = print_kernel.bind((x,))
        ref_code = bound._ref_code if hasattr(bound, '_ref_code') else ""
        
        # Since we're using the original code, verify it contains the print statements
        self.assertIn('print("tensor value: ", val)', ref_code)
        self.assertIn('print("multiple values: ", val, val * 2)', ref_code)

    def test_print_with_computation_ref_eager(self):
        """Test print with computed values in reference eager mode."""
        @helion.kernel
        def print_compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                x_val = x[tile_m, tile_n]
                y_val = y[tile_m, tile_n]
                sum_val = x_val + y_val
                print("x: ", x_val)
                print("y: ", y_val)
                print("sum: ", sum_val)
                out[tile_m, tile_n] = sum_val
            return out
        
        # Create predictable inputs
        x = torch.ones([2, 2], device='cuda', dtype=torch.float32) * 10.0
        y = torch.ones([2, 2], device='cuda', dtype=torch.float32) * 5.0
        expected = x + y
        
        # Kernel runs in reference eager mode automatically
        result = print_compute_kernel(x, y)
        
        # Check accuracy
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        
        # Get and check generated code
        bound = print_compute_kernel.bind((x, y))
        ref_code = bound._ref_code if hasattr(bound, '_ref_code') else ""
        
        # Verify print statements are in the generated code
        self.assertIn('print("x: ", x_val)', ref_code)
        self.assertIn('print("y: ", y_val)', ref_code)
        self.assertIn('print("sum: ", sum_val)', ref_code)

    def test_print_in_conditional_ref_eager(self):
        """Test print inside conditional statements in reference eager mode."""
        @helion.kernel
        def print_conditional_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                # Use element-wise conditional to match expected behavior
                mask = val > 0
                if torch.any(mask):
                    print("tile has positive values: ", val)
                else:
                    print("tile has no positive values: ", val)
                out[tile_m, tile_n] = torch.where(mask, val * 2, val * 0.5)
            return out
        
        # Create input with mixed values
        x = torch.tensor([[1.0, -1.0], [2.0, -2.0]], device='cuda', dtype=torch.float32)
        
        # Calculate expected output
        expected = torch.where(x > 0, x * 2, x * 0.5)
        
        # Run in reference eager mode
        result = print_conditional_kernel(x)
        
        # Check accuracy
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        
        # Get and check generated code
        bound = print_conditional_kernel.bind((x,))
        ref_code = bound._ref_code if hasattr(bound, '_ref_code') else ""
        
        # Verify conditional print statements are in the generated code
        self.assertIn('if torch.any(mask):', ref_code)
        self.assertIn('print("tile has positive values: ", val)', ref_code)
        self.assertIn('print("tile has no positive values: ", val)', ref_code)

    def test_incomplete_kernel_with_print_ref_eager(self):
        """Test that print works even in incomplete/wrong kernels in reference eager mode."""
        @helion.kernel
        def incomplete_kernel(x: torch.Tensor) -> torch.Tensor:
            # This kernel is intentionally incomplete/wrong
            # It doesn't properly initialize the output tensor
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                print("processing tile: ", val)
                # Intentionally not doing anything with the output
                # This would cause issues in a real kernel
                pass
            # Return input unchanged (wrong behavior)
            return x
        
        # Create input
        x = torch.ones([2, 2], device='cuda', dtype=torch.float32) * 3.14
        
        # Skip the compilation check - just test reference eager mode behavior
        # The test is about reference eager mode working even with incomplete kernels
        
        # Now run in reference eager mode - should not crash despite being incomplete
        result = incomplete_kernel(x)
        
        # Result should be same as input (since kernel is incomplete)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)
        
        # Get and check generated code
        bound = incomplete_kernel.bind((x,))
        ref_code = bound._ref_code if hasattr(bound, '_ref_code') else ""
        
        # Verify print statement is still in the generated code
        self.assertIn('print("processing tile: ", val)', ref_code)


class TestKernelRefEagerParam(TestCase):
    """Test @helion.kernel(ref_eager=True) parameter functionality."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Save original env var to ensure clean state
        self._original_eager = os.environ.get("HELION_REF_EAGER")
        # Remove env var to test parameter-only behavior
        if "HELION_REF_EAGER" in os.environ:
            del os.environ["HELION_REF_EAGER"]
    
    def tearDown(self):
        """Restore original environment."""
        super().tearDown()
        # Restore original env var
        if self._original_eager is not None:
            os.environ["HELION_REF_EAGER"] = self._original_eager
    
    def test_ref_eager_param_simple(self):
        """Test that ref_eager=True parameter works for simple kernels."""
        @helion.kernel(ref_eager=True)
        def add_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b
        
        # Test execution
        a = torch.tensor([1.0, 2.0], device='cuda')
        b = torch.tensor([3.0, 4.0], device='cuda')
        result = add_kernel(a, b)
        expected = a + b
        torch.testing.assert_close(result, expected)
    
    def test_ref_eager_param_complex(self):
        """Test that ref_eager=True parameter works for complex kernels."""
        @helion.kernel(ref_eager=True)
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
    
    def test_ref_eager_param_overrides_env(self):
        """Test that ref_eager=False parameter overrides env var."""
        # Set env var to enable ref eager mode
        os.environ["HELION_REF_EAGER"] = "1"
        
        # Create kernel with ref_eager=False (should override env var)
        @helion.kernel(ref_eager=False)
        def mul_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a * b
        
        # Note: We can't easily test that compilation happens without
        # inspecting internal state, but we can verify it executes correctly
        a = torch.tensor([2.0, 3.0], device='cuda')
        b = torch.tensor([4.0, 5.0], device='cuda')
        result = mul_kernel(a, b)
        expected = a * b
        torch.testing.assert_close(result, expected)
    
    def test_ref_eager_param_with_config(self):
        """Test that ref_eager=True works with kernel configs."""
        from helion import Config
        
        # Create a simple config
        config = Config(block_sizes=[128])
        
        @helion.kernel(ref_eager=True, config=config)
        def configured_kernel(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0
        
        # Test execution
        x = torch.randn(256, device='cuda')
        result = configured_kernel(x)
        expected = x * 2.0
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
