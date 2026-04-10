# Croqtile Language Reference

## About This Document

This document is the **language reference** for Croqtile, a C++ embedded domain-specific language (eDSL) for orchestrating data movement in heterogeneous hardware. It covers the syntax, semantics, and type system in depth -- the language as a programming language, not as a tool.

**Audience.** Programmers with a background in programming languages and/or compilers who want to understand *how the language works and why* it is designed the way it is. Prior GPU experience is helpful but not required; familiarity with type systems, data-flow semantics, and domain-specific language design will be the most useful background.

**Relationship to the Croqtile Tutorial.** The [Croqtile Tutorial](https://codes1gn.github.io/croktile-tutorial/) (level-0) provides a hands-on introduction starting from installation, walking through increasingly complex kernels, and demonstrating performance tuning workflows. This document (level-2) assumes you have seen the basics and focuses on:

- Precise syntax and semantic rules for every construct
- The type system: scalars, shapes, spanned data, bounded variables, futures
- Shape inference and how the compiler reasons about your program
- Advanced features such as view operations, MMA primitives, and TMA

If you are new to Croqtile and want to build your software quickly, start with the [Croqtile Tutorial](https://codes1gn.github.io/croktile-tutorial/) first.

For compiler internals, build instructions, and target development, see the [Developer Guide](../Developer/index.md).

## How to Read

The chapters are ordered from shallow to deep:

| Part | Topic | What You Learn |
|------|-------|----------------|
| **A** | Foundations | Program model, types, shapes, integers |
| **B** | Data and Memory | Buffers, storage qualifiers, dynamic shapes, view operations |
| **C** | Control Flow | Parallelism, iteration, masking |
| **D** | Data Movement | DMA basics, tiling, advanced patterns, TMA |
| **E** | Compute Primitives | MMA operations, numeric builtins |
| **F** | Synchronization | Events, futures, async execution model |
| **G** | C++ Interop | Calling device code, I/O, macros |
| **H** | Advanced Topics | Assertions |
| **Appendix** | Quick Reference | Syntax table, operator gallery, optimization patterns |

Each chapter introduces the concept with a minimal example, then layers on semantic rules, type constraints, and edge cases. Code examples are drawn from the test suite and samples in the repository.

For building the compiler from source and running tests, see the [Developer Guide -- Build and Test](../Developer/build-and-test.md). For a quick-start installation, see the [Croqtile Tutorial -- Ch 0: Installation](https://codes1gn.github.io/croktile-tutorial/).
