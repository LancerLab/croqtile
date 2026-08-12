# Cooperative Kernel Launch

> **Target-specific feature.** Cooperative launch is only available on
> targets that expose a cooperative launch API. See
> [Target Support](#target-support) below for the supported targets.

## Overview

By default, CUDA kernel launches provide no ordering or synchronization
guarantees between blocks. Blocks execute independently and may be scheduled
in any order. When a kernel needs blocks to synchronize through global memory,
the application must use cooperative launch, which guarantees all blocks of
the grid run concurrently on the device.

Choreo supports cooperative launch through a `block<cooperative>` annotation
on the `parallel-by` construct. When this annotation is present, the compiler
emits `cuLaunchCooperativeKernel` in the generated host stub and enables
grid-level `sync.barrier` operations.

```choreo
parallel bx by 2 : block<cooperative> {
  a.at(bx) = bx + 1;       // each block writes to global memory
  sync.barrier : block;    // grid-level barrier
  b.at(bx) = a.at(1 - bx); // read neighbor's value
}
```

Without cooperative launch, block 0 might read `a[1]` before block 1 writes
it. With the cooperative annotation, `sync.barrier : block` ensures all
blocks reach the barrier before any proceeds, making cross-block data exchange
through global memory safe.

## Enabling Cooperative Launch

Add `<cooperative>` to the block-level annotation:

```choreo
parallel bx by N : block<cooperative> {
  // kernel body -- blocks execute concurrently
  // and grid-level barriers are available
}
```

The annotation applies only to the `block` level. The compiler validates that
the target supports cooperative launch and emits the appropriate host launch
API.

## Grid-Level Barrier

Inside a cooperative block, use `sync.barrier : block` to synchronize all
blocks in the grid:

```choreo
sync.barrier : block;
```

This is a grid-scoped barrier -- every block in the launch must reach it
before any block proceeds. It is distinct from `sync.barrier : thread`, which
synchronizes threads within a single block.

Grid barriers are only valid inside `block<cooperative>` regions. Using
`sync.barrier : block` without the cooperative annotation is a semantic error.

## Target Support

Cooperative launch requires hardware and driver support:

| Target | Typical Architectures | Notes |
|--------|----------------------|-------|
| CUDA GPU (`-t cute`, `-t gpu`) | SM 86+ (e.g., RTX 30 series+) | Requires cooperative launch support in the CUDA driver |
| AMDGPU (`-t hip`) | gfx1030+ (RDNA 2+) | Emits `hipLaunchCooperativeKernel` in host stubs |

The compiler reports a warning if cooperative launch is requested on a target
that does not support it.

## Complete Example

```choreo
__co__ s32[2] coop_grid_sync() {
  s32[2] a;
  s32[2] b;

  parallel bx by 2 : block<cooperative> {
    a.at(bx) = bx + 1;
    sync.barrier : block;
    b.at(bx) = a.at(1 - bx);
  }
  return b;
}

int main() {
  auto res = coop_grid_sync();
  choreo::choreo_assert(res[0] == 2,
    "block 0 should see block 1's value (2)");
  choreo::choreo_assert(res[1] == 1,
    "block 1 should see block 0's value (1)");
}
```

Blocks exchange values through the global `a` array. Block 0 writes `1` and
reads `a[1]`; block 1 writes `2` and reads `a[0]`. The grid barrier ensures
both writes complete before either block reads.

## Relationship to Other Synchronization Primitives

Cooperative launch provides **block-level** synchronization. For
**thread-level** and **warpgroup-level** synchronization, use:

| Primitive | Scope | Mechanism |
|-----------|-------|-----------|
| `sync.barrier : thread` | Threads within a block | `__syncthreads()` |
| `event` / `trigger` / `wait` | Producer-consumer paths within a CTA | mbarrier or named barrier |
| `sync.barrier : block` | Grid (all blocks) | Grid-level barrier |

For cross-warpgroup synchronization patterns, use the
`xwg` execution model with named barriers (see the cross-warpgroup
barrier design document in `Documents/design/`).

## Code Generation

When the `block<cooperative>` annotation is present, the compiler:

1. Validates cooperative launch support in the target backend
2. Emits `cuLaunchCooperativeKernel` (CUDA) or
   `hipLaunchCooperativeKernel` (HIP) in host stubs instead of the
   standard launch API
3. Enables grid-scoped barrier validation in device code

The cooperative launch API requires the CUDA driver to verify that all blocks
can reside concurrently on the device. If the kernel uses too many resources
(shared memory, registers), the launch may fail even though a non-cooperative
launch of the same kernel would succeed.
