# TMA (Tensor Memory Accelerator)

## Overview

Starting with NVIDIA's Hopper architecture (SM90), the **Tensor Memory Accelerator (TMA)** provides hardware-accelerated bulk data transfers between global memory and shared memory. Croqtile exposes TMA through the `tma.copy` operation.

TMA transfers are **always between global and shared memory** -- global-to-shared or shared-to-global. Direct global-to-global transfers are not supported by the hardware.

## Basic Syntax

```choreo
tma.copy input.chunkat(p, idx) => shared;
```

TMA operations accept any sub-view expression as the source: `chunkat`, `subspan.at`, `span_as`, `view.from`, and other view operations all work with `tma.copy`.

```choreo
tma.copy input.subspan(tile_m, tile_k).at(m_idx, k_idx) => shared;
tma.copy input.chunkat(p, idx) => shared;
tma.copy input.view(ext).from(off) => shared;
```

## TMA vs DMA

| Aspect | `dma.copy` | `tma.copy` |
|--------|-----------|------------|
| Mechanism | Software-orchestrated or hardware async DMA | Dedicated TMA hardware unit |
| Requires | Any CUDA GPU | SM90+ (Hopper and later) |
| Direction | Any memory hierarchy transfer | Global <-> shared only |
| Typical use | SM86 and earlier, or arbitrary transfers | SM90 high-throughput bulk transfers |

## Synchronization

TMA operations are asynchronous by nature. The compiler manages the synchronization needed to ensure data is available before use. The `wait` statement marks the point where the program needs the transferred data to be ready:

```choreo
parallel p by grid_size : block {
  parallel q by block_size : thread {
    with {m, k} in [M / tile_m, K / tile_k] {
      foreach m, k {
        fa = tma.copy lhs.chunkat(m, k) => shared;
        fb = tma.copy rhs.chunkat(k, n) => shared;
        wait fa, fb;
        // data is now available in shared memory
      }
    }
  }
}
```

## Store Back: Shared to Global

TMA also supports writing from shared memory back to global memory:

```choreo
tma.copy shared_result => output.subspan(tile_m, tile_n).at(m_idx, n_idx);
```

The same global <-> shared constraint applies: the source or destination must be in shared memory, and the other must be in global memory.

## FP8 and FP6/FP4 Support

TMA supports low-precision data types for modern AI workloads:

```choreo
tma.copy fp8_input.subspan(...).at(...) => shared;
tma.copy fp6_input.subspan(...).at(...) => shared;
```

*(Reference: `tests/gpu/end2end/tma_fp8.co`, `tests/gpu/end2end/tma_fp6_fp4.co`)*

## Architecture Requirements

TMA operations require SM90 or later. Test files use the `REQUIRES` directive:

```
// REQUIRES: TARGET-SM_90
```

When targeting earlier architectures, use `dma.copy` instead.

*(Reference: `tests/gpu/codegen/cute/tma_barrier_init.co`)*
