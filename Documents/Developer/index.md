# Croqtile Developer Guide

This documentation is for **developers of the Croqtile compiler and toolchain** -- those modifying the compiler source, adding new targets, extending the test infrastructure, or contributing to the runtime.

For the **language reference** (syntax, semantics, and usage of `.co` files), see the [Croqtile Language Reference](../Documentation/index.md).

For a **hands-on tutorial**, see the [Croqtile Tutorial](https://codes1gn.github.io/croktile-tutorial/) (level-0).

---

## Table of Contents

### Getting Started

1. [Build and Test](build-and-test.md) -- prerequisites, build targets, running tests, compiler flags
2. [Coding Style](coding-style.md) -- formatting, naming conventions, code patterns, OSS compliance

### Compiler Internals

3. [Compilation Passes](compilation-passes.md) -- pass pipeline (SEMA -> CODEGEN), debugging with `-pa` and `-sa`
4. [Value Numbering](value-numbering.md) -- shape inference foundation, simplification, scope rules
5. [Memory Reuse](memory-reuse.md) -- automatic buffer lifetime analysis, static/dynamic reuse, alignment
6. [DMA Resource Allocation](dma-resource-allocation.md) -- liveness-driven allocation of DMA futures and events, interval coloring, HeapSimulator

### Extending the Compiler

7. [Developing a Target](developing-a-target.md) -- adding a new compilation backend, codegen visitor, target registration
8. [Lit Test Runner](lit-test-runner.md) -- `lit.sh` directives, `%` substitutions, hooks, `include_dir`, per-target configs

### Target-Specific References

9. [Target-Specific Calls](target/calls.md) -- calling conventions for specific backends (e.g., `extern "C"` for Factor)
10. [Target Environment Setup](target/env_setting_up.md) -- target SDK and hardware configuration
