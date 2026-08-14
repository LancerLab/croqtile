# Explicit Buffer Mapping

> **Target-specific feature.** Explicit buffer mapping is only available on
> targets that expose hardware support for mapping a region of global memory
> directly into a processing element's local address space. See
> [Target Support](#target-support) below for details.

## Overview

A `dma.copy` moves data by copying it between memory spaces. Some
architectures additionally provide a memory-mapping unit that can expose a
contiguous region of global memory directly inside the local address space of
a processing element (PE), without moving the data. Choreo exposes this
capability through explicit buffer mapping: the `buffer.map` and
`buffer.remap` statements.

When a buffer is mapped, the program gains a `local` view of a global buffer.
Reads and writes through that view alias the global buffer directly (zero
copy). This is useful when a processing element must operate on a large,
regular region of global data and a full DMA copy into local memory would be
unnecessary or too expensive.

The mapped result is a *view*, not a copy: it does not own storage, and no
separate buffer is allocated for it.

## Syntax

```choreo
mapped = src.map<local>(offset, size);
```

`src` is a dense (contiguous) global buffer, `offset` is the element offset
into the buffer, and `size` is the number of elements to map. The destination
storage qualifier (`<local>`) selects the memory space the region is mapped
into; only the local memory space is supported.

`buffer.remap` re-points the existing mapping of the same source buffer to a
different window, without tearing down and re-establishing the mapping:

```choreo
m0 = src.map<local>(0, 32);
m1 = src.remap<local>(32, 64);
```

There is no explicit `unmap` statement. The compiler inserts an unmap
automatically when the mapped region's scope ends (auto-unmap).

## Result and Data Access

A buffer mapping statement introduces a new symbol whose value is a data
future wrapping a `local` mdspan. Access the mapped data through `.data`:

```choreo
mapped = input.map<local>(0, |input.span|);
foreach {i} in [|mapped.span|] {
  mapped.data.at(i) = mapped.data.at(i) + 1;
}
```

## Zero-Copy Aliasing

The mapped result aliases the source buffer rather than copying it. In
liveness terms the result is a no-storage alias of the source: uses of the
result are redirected to the source, and the result does not receive its own
live range or local allocation. Only the source buffer is defined; the mapped
result simply names a window into it.

## Storage Mapping

Explicit buffer mapping supports a single direction: from global memory into
the processing element's local memory space. The source must be a global
buffer (or a function parameter that resolves to global memory), and the
destination must be the local memory space. Any other source/destination
combination is rejected.

## Dense Source Requirement

The source of a `buffer.map` or `buffer.remap` must be dense (contiguous). A
strided view is rejected:

```
error: buffer.map/remap requires a dense (contiguous) source buffer, but the source is non-contiguous.
```

Use `dma.copy` for non-contiguous regions.

## Target Support

Explicit buffer mapping is only available on targets that expose the
underlying hardware capability:

| Target | Explicit buffer mapping | Notes |
|--------|------------------------|-------|
| Architectures with global-to-local memory-mapping hardware | Yes | `buffer.map` / `buffer.remap` supported |
| CUDA GPU (`-t cute`, `-t gpu`) | No | Rejected; use `dma.copy` |
| AMDGPU (`-t hip`) | No | Rejected; use `dma.copy` |
| CPU (`-t cc`) | No | Rejected; use `dma.copy` |

On a target without this capability, the compiler rejects the construct
during early semantic analysis, before any code generation:

```
error: explicit memory mapping is not supported by this target.
```

## CoIR Representation

During lowering, explicit buffer mapping becomes the `coir.buffer.map`,
`coir.buffer.remap`, and `coir.buffer.unmap` operations. A module lowered for
a buffer-map-capable target carries a `coir.has_buffer_map` attribute; the
operations are rejected by the CoIR verifier when that attribute is absent.

## Complete Example

```choreo
#include "choreo.h"

__co__ s32 [128] buffer_map_add_two(s32 [128] input) {
  s32[input.span] output;

  parallel p by 1 {
    // Map: aliases global input into local address space.
    // This avoids an explicit DMA copy.
    mapped_input = input.map<local>(0, |input.span|);

    local s32[|mapped_input.span|] local_out;
    foreach {i} in [|mapped_input.span|] {
      local_out.at(i) = mapped_input.data.at(i) + 2;
    }
    dma.copy local_out => output;
  }

  return output;
}
```

The mapped region `mapped_input` provides zero-copy read access to `input` in
the local address space. The results are then copied back to the global
`output` buffer with a normal DMA copy.
