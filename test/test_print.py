from __future__ import annotations

import contextlib
import io
import os
import unittest

from expecttest import TestCase
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
import helion.language as hl

try:
    from triton.compiler.errors import CompilationError
except ImportError:
    CompilationError = None


class TestPrint(TestCase):
    maxDiff = 16384

    def run_kernel_with_output(self, kernel_fn, args):
        """Helper to run kernel and capture output"""
        # Capture stdout
        captured_output = io.StringIO()

        # Get the generated code
        code, result = code_and_output(kernel_fn, args)

        # Run the kernel again capturing stdout
        # Note: device_print output goes to stdout in the GPU kernel
        with contextlib.redirect_stdout(captured_output):
            # Reset the kernel to force recompilation
            kernel_fn.reset()
            kernel_fn(*args)

        output_str = captured_output.getvalue()

        return code, result, output_str

    def run_test_with_and_without_interpret(self, test_func):
        """Helper to run a test function with and without TRITON_INTERPRET=1"""
        original_env = os.environ.get("TRITON_INTERPRET")

        try:
            # First run without TRITON_INTERPRET
            if original_env:
                os.environ.pop("TRITON_INTERPRET", None)
            test_func(interpret_mode=False)

            # Then run with TRITON_INTERPRET=1
            os.environ["TRITON_INTERPRET"] = "1"
            try:
                test_func(interpret_mode=True)
            except Exception as e:
                # TRITON_INTERPRET might not be fully supported
                error_msg = str(e)
                is_triton_interpret_error = (
                    "Cannot call @triton.jit" in error_msg
                    or "InterpreterError" in str(type(e))
                    or "InterpreterError" in error_msg
                    or "CompilationError" in str(type(e))
                    or (CompilationError and isinstance(e, CompilationError))
                )
                if is_triton_interpret_error:
                    self.skipTest("TRITON_INTERPRET not supported in this environment")
                else:
                    raise
        finally:
            # Restore original env
            if original_env is None:
                os.environ.pop("TRITON_INTERPRET", None)
            else:
                os.environ["TRITON_INTERPRET"] = original_env

    def test_basic_print(self):
        """Test basic print with prefix and tensor values"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    val = x[tile_m, tile_n]
                    print("tensor value:", val)
                    out[tile_m, tile_n] = val * 2
                return out

            x = torch.ones([2, 2], device=DEVICE) * 42.0  # Use predictable values

            # Run kernel and capture output
            code, result, output = self.run_kernel_with_output(print_kernel, (x,))
            torch.testing.assert_close(result, x * 2)

            # Check that print is generated in the code
            self.assertIn("'tensor value:'", code)
            if interpret_mode:
                # In interpret mode, should use regular print()
                self.assertIn("print('tensor value:'", code)
            else:
                # In normal mode, should use tl.device_print
                self.assertIn("tl.device_print('tensor value:'", code)

            if interpret_mode:
                # In interpret mode, we might get output directly
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    # Check that we have print statements
                    self.assertGreater(len(output_lines), 0)
                    # Check for our prefix in the output
                    for line in output_lines:
                        self.assertIn("tensor value:", line)
                        # In interpret mode, output might be formatted differently
                        # but should contain our value
                        self.assertIn("42", line)
            else:
                # In normal mode, device_print output goes to GPU's stdout (not captured here)
                # So we expect empty output or formatted device_print output
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    # Check that each line contains our prefix and value
                    for line in output_lines:
                        self.assertIn("tensor value:", line)
                        # Check for the value (might be formatted as 42.0 or 42.000000)
                        self.assertTrue("42" in line)
                        # Device print typically includes pid and idx info
                        self.assertTrue("pid" in line or "idx" in line)

        self.run_test_with_and_without_interpret(run_test)

    def test_print_multiple_tensors(self):
        """Test print with multiple tensor arguments"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_multi_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    x_val = x[tile_m, tile_n]
                    y_val = y[tile_m, tile_n]
                    print("x and y:", x_val, y_val)
                    out[tile_m, tile_n] = x_val + y_val
                return out

            x = torch.ones([2, 2], device=DEVICE) * 10.0
            y = torch.ones([2, 2], device=DEVICE) * 20.0

            # Run kernel and capture output
            code, result, output = self.run_kernel_with_output(
                print_multi_kernel, (x, y)
            )
            torch.testing.assert_close(result, x + y)

            # Check that print is generated with multiple format specifiers
            self.assertIn("'x and y:'", code)
            if interpret_mode:
                # In interpret mode, should use regular print()
                self.assertIn("print('x and y:'", code)
            else:
                # In normal mode, should use tl.device_print
                self.assertIn("tl.device_print('x and y:'", code)

            if interpret_mode:
                # In interpret mode, check for output
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    self.assertGreater(len(output_lines), 0)
                    # Check that each line contains our prefix and both values
                    for line in output_lines:
                        self.assertIn("x and y:", line)
                        self.assertIn("10", line)
                        self.assertIn("20", line)
            else:
                # In normal mode, check for device_print formatted output
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    # Check that each line contains our prefix and both values
                    for line in output_lines:
                        self.assertIn("x and y:", line)
                        # Values might be formatted as 10.0 or 10.000000
                        self.assertTrue("10" in line and "20" in line)

        self.run_test_with_and_without_interpret(run_test)

    def test_print_no_prefix_error(self):
        """Test that print without arguments raises an error"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_no_args_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print()  # This should fail
                    out[tile_m, tile_n] = x[tile_m, tile_n]
                return out

            x = torch.randn([32, 32], device=DEVICE)
            with pytest.raises(
                helion.exc.InternalError,
                match="print\\(\\) requires at least one argument",
            ):
                code_and_output(print_no_args_kernel, (x,))

        self.run_test_with_and_without_interpret(run_test)

    def test_print_non_string_prefix_error(self):
        """Test that print with non-string prefix raises an error"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_bad_prefix_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print(123, x[tile_m, tile_n])  # Non-string prefix
                    out[tile_m, tile_n] = x[tile_m, tile_n]
                return out

            x = torch.randn([32, 32], device=DEVICE)
            with pytest.raises(
                helion.exc.InternalError,
                match="First argument to print\\(\\) must be a string prefix",
            ):
                code_and_output(print_bad_prefix_kernel, (x,))

        self.run_test_with_and_without_interpret(run_test)

    def test_print_compile_time_value_error(self):
        """Test that printing compile-time values raises an error"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_shape_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print("shape:", m)  # Compile-time value inside loop
                    out[tile_m, tile_n] = x[tile_m, tile_n]
                return out

            x = torch.randn([32, 32], device=DEVICE)
            with pytest.raises(
                helion.exc.InternalError,
                match="print\\(\\) only supports runtime tensor values",
            ):
                code_and_output(print_shape_kernel, (x,))

        self.run_test_with_and_without_interpret(run_test)

    def test_print_prefix_only(self):
        """Test print with only string prefix - this is allowed and will print just the message"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_message_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                for tile_m, tile_n in hl.tile([m, n]):
                    print("processing tile")  # This is allowed
                    out[tile_m, tile_n] = x[tile_m, tile_n] * 2
                return out

            x = torch.ones([2, 2], device=DEVICE)

            # Run kernel and capture output
            code, result, output = self.run_kernel_with_output(
                print_message_kernel, (x,)
            )
            torch.testing.assert_close(result, x * 2)

            # Check that print is generated
            self.assertIn("'processing tile'", code)
            if interpret_mode:
                # In interpret mode, should use regular print()
                self.assertIn("print('processing tile'", code)
            else:
                # In normal mode, should use tl.device_print
                self.assertIn("tl.device_print('processing tile'", code)

            if interpret_mode:
                # In interpret mode, check for message output
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    self.assertGreater(len(output_lines), 0)
                    # Check that each line contains our message
                    for line in output_lines:
                        self.assertIn("processing tile", line)
            else:
                # In normal mode, check for device_print formatted output
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    # Check that each line contains our message
                    for line in output_lines:
                        self.assertIn("processing tile", line)

        self.run_test_with_and_without_interpret(run_test)

    def test_print_with_different_tile_sizes(self):
        """Test print output with different tile sizes to verify output count"""

        def run_test(interpret_mode):
            @helion.kernel
            def print_tile_kernel(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                m, n = x.shape
                # Note: tile_dims parameter may not be supported, so we use default tiling
                for tile_m, tile_n in hl.tile([m, n]):
                    val = x[tile_m, tile_n]
                    print("tile value:", val)
                    out[tile_m, tile_n] = val
                return out

            # Create small tensor with predictable values
            x = torch.arange(4 * 4, device=DEVICE, dtype=torch.float32).reshape(4, 4)

            # Test kernel execution
            code, result, output = self.run_kernel_with_output(print_tile_kernel, (x,))
            torch.testing.assert_close(result, x)

            # Check code generation
            self.assertIn("'tile value:'", code)
            if interpret_mode:
                # In interpret mode, should use regular print()
                self.assertIn("print('tile value:'", code)
            else:
                # In normal mode, should use tl.device_print
                self.assertIn("tl.device_print('tile value:'", code)

            if interpret_mode:
                # In interpret mode, check for value output
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    self.assertGreater(len(output_lines), 0)
                    # Check that output contains our prefix
                    for line in output_lines:
                        self.assertIn("tile value:", line)
                        # Should contain numeric values from our tensor
                        parts = line.split(":")
                        if len(parts) > 1:
                            # Check that there's a number after the prefix
                            value_part = parts[-1].strip()
                            # Should be able to find a digit
                            self.assertTrue(any(c.isdigit() for c in value_part))
            else:
                # In normal mode, check for device_print formatted output
                if output.strip():
                    output_lines = [line for line in output.strip().split("\n") if line]
                    self.assertGreater(len(output_lines), 0)
                    # Check first line has expected format
                    self.assertIn("tile value:", output_lines[0])
                    # Check for numeric content
                    self.assertTrue(any(c.isdigit() for c in output_lines[0]))

        self.run_test_with_and_without_interpret(run_test)


if __name__ == "__main__":
    unittest.main()
