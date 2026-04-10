# Memory Reuse

## Automatic Buffer Reuse

The Croqtile compiler analyzes buffer lifetimes and automatically reuses storage when lifetimes do not overlap. This minimizes the memory footprint of tileflow programs without programmer intervention.

```choreo
parallel p by 4 {
  with index in [8] {
    foreach index {
      f = dma.copy input.chunkat(p, index) => shared;
      // 'f' buffer is reused across iterations if possible
      call kernel(f.data);
    }
  }
}
```

The compiler determines that the local buffer allocated for `f` can be reused in each loop iteration, since the previous iteration's buffer is no longer live.

## Static vs Dynamic Memory Reuse

### Static Reuse

When buffer sizes are known at compile time, the compiler performs **static memory reuse** -- computing a fixed memory layout that maps multiple buffers to overlapping address ranges.

```choreo
// Compiler may map buf1 and buf2 to the same physical memory
// if their lifetimes don't overlap
f1 = dma.copy a.chunkat(p) => shared;
call kernel(f1.data);
f2 = dma.copy b.chunkat(p) => shared;
call kernel(f2.data);
```

*(Reference: `tests/gpu/end2end/memreuse-static.co`)*

### Dynamic Reuse

For dynamically-shaped buffers, the compiler generates runtime code to manage memory allocation and reuse.

*(Reference: `tests/gpu/end2end/memreuse-dynamic.co`)*

## Memory Alignment

The `align` attribute controls buffer alignment for optimal memory access patterns:

```choreo
shared f32 [128, 64] buf;  // compiler chooses default alignment
```

Alignment is particularly important for:
- **TMA operations**: Require specific alignment for bulk transfers.
- **Vectorized access**: Aligned loads/stores are faster.
- **Bank conflict avoidance**: Proper alignment in shared memory.

The compiler automatically selects appropriate alignment based on the target platform and DMA operation types. In performance-critical cases, the programmer can influence alignment through buffer declarations.

*(Reference: `tests/infer/align.co`)*

## Programmer Guidelines

- Let the compiler manage buffer lifetimes -- avoid manual reuse unless profiling shows a need.
- Use the compiler's diagnostic flags (`-pa=LIVENESS`) to inspect buffer lifetime analysis.
- For maximum reuse, minimize the live range of buffers by placing `wait` statements as late as possible and `call` statements as early as possible.
