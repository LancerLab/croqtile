# Croqtile Language Reference

Welcome to the **Croqtile Language Reference** -- the comprehensive syntax and semantics guide for Croqtile, a C++ embedded domain-specific language (eDSL) for orchestrating DMA data movement in heterogeneous hardware.

**For a hands-on tutorial and performance tuning guide, see the [Croqtile Tutorial](https://codes1gn.github.io/croktile-tutorial/) (level-0).**

**For compiler development (building, testing, adding targets), see the [Developer Guide](../Developer/index.md).**

This reference (level-2) is for readers who want to understand the language in depth -- its type system, semantic rules, and advanced features.

---

## Table of Contents

### Part A -- Foundations

1. [Introduction](intro.md) -- Audience, relationship to level-0, how to read
2. [Program Structure](program-structure.md) -- Host/device/tileflow model, function markers, compilation pipeline
3. [Type System](type-system.md) -- Scalar types, fundamental types, mdspan, spanned, bounded, futures, immutability
4. [Shapes and mdspan](shapes-and-mdspan.md) -- Defining shapes, derivation, arithmetic, concatenation
5. [Integers and I-Tuples](integers-and-ituples.md) -- Integer variables, i-tuples, operations, bounded preview

### Part B -- Data and Memory

6. [Spanned Data](spanned-data.md) -- Declaring buffers, fundamental types, storage qualifiers, initialization, parameters
7. [Symbolic and Dynamic Shapes](symbolic-and-dynamic.md) -- Anonymous `?`, symbolic dimensions, runtime checks, limitations
8. [View Operations](view-operations.md) -- `chunkat`, `span_as`, `view.from`, `subspan.at`, `subspan.step.at`

### Part C -- Control Flow and Parallelism

9. [Parallel-By](parallel-by.md) -- SPMD parallelism, multi-level, sub-levels, annotations, host/device boundary
10. [Iteration](iteration.md) -- `with-in`, `foreach`, `while`, `break`/`continue`, range expressions
11. [Thread Masking](thread-masking.md) -- `inthreads`, async masking, multi-level, implicit masking

### Part D -- Data Movement

12. [DMA Basics](dma-basics.md) -- Sync/async, operation types, futures, `wait`
13. [DMA Tiling](dma-tiling.md) -- View operations in DMA, ubound operator, multi-variable tiling
14. [DMA Advanced](dma-advanced.md) -- `dma.any`, `swap`/`select`, multi-buffering, swizzle
15. [TMA](tma.md) -- Tensor Memory Accelerator, bulk copy, SM90+

### Part E -- Compute Primitives

16. [MMA](mma.md) -- Matrix multiply-accumulate: `fill`, `load`, `row.col`, `op`, `store`
17. [Numerics](numerics.md) -- Builtin functions, transcendentals, type promotion, float variants

### Part F -- Synchronization and Async

18. [Events](events.md) -- `event`, `trigger`, `wait`, deadlock avoidance
19. [Futures and Async](futures-and-async.md) -- Future semantics, async model, non-blocking chains

### Part G -- C++ Interop

20. [Calling Device Code](call-and-device.md) -- `call`, `__co_device__`, template calls
21. [I/O and Linking](io-and-linking.md) -- Parameters, `make_spanned`, return values, streams, linking
22. [Macros](macros.md) -- `#define`, conditional compilation, preprocessing order

### Part H -- Advanced Topics

23. [Assertions](assertions.md) -- `assert`, compile-time vs runtime evaluation, hoisting, `--runtime-check`

### Appendix

24. [Syntax Quick Reference](syntax-quick-ref.md) -- Lookup tables for all constructs
25. [Operator Gallery](operator-gallery.md) -- Curated index of example `.co` files
26. [Tileflow Optimization Patterns](tileflow-opt.md) -- Matmul and optimization case studies
