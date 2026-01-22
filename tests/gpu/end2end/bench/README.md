# Benchmark Code Pattern Tests

This directory contains code patterns from real benchmarking implementations. These tests are used to guard against regressions in code generation and compilation for complex, performance-critical patterns.

## Purpose

The files in this directory are **NOT** intended to be run as actual benchmarks. Instead, they serve as:

1. **Regression tests** - Ensure that complex code patterns continue to compile correctly
2. **Pattern preservation** - Maintain real-world usage patterns that exercise the compiler
3. **Code generation validation** - Verify that the generated code is correct for performance-critical kernels

## Running Tests

These files are compiled as part of the standard test suite to ensure they continue to compile without errors. They are not executed as functional tests.
