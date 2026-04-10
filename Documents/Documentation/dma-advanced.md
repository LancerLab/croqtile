# Advanced DMA Patterns

## `dma.any` -- Dummy Futures

`dma.any` creates a placeholder future with no actual DMA operation:

```choreo
lfB = dma.any;
rfB = dma.any;
```

This is needed when a future variable must be declared before the loop iteration that assigns it a real DMA operation. The dummy future is later replaced by an actual DMA in the loop body.

## `swap` and `rotate` -- Buffer Exchange

**`swap`** exchanges two futures, enabling double-buffering:

```choreo
swap lfA, lfB;
swap rfA, rfB;
```

After `swap`, `lfA` refers to what was `lfB` and vice versa. Both futures must have identical DMA operation types.

**`rotate`** generalizes `swap` for more than two buffers (triple-buffering, etc.).

## `select` -- Conditional Future

`select` chooses between two futures based on a condition:

```choreo
lf = select(#y % 2, lfB, lfA);
```

If `#y % 2` is non-zero, `lf` becomes `lfB`; otherwise `lfA`. This handles odd/even edge cases in multi-buffering epilogues.

## Multi-Buffering Pattern (Double Buffering)

The canonical double-buffering pattern in Croqtile:

```choreo
parallel p by 6 {
  with index = {x, y} in [17, 4] {
    foreach x {
      // Prologue: pre-load first chunk into buffer A
      lfA = dma.copy.async lhs.chunkat(p, x, y) => shared;
      rfA = dma.copy.async rhs.chunkat(p, x, y) => shared;
      lfB = dma.any;
      rfB = dma.any;

      // Body: overlap load B with compute on A
      foreach y(1:) {
        lfB = dma.copy.async lhs.chunkat(p, x, y) => shared;
        rfB = dma.copy.async rhs.chunkat(p, x, y) => shared;
        wait lfA, rfA;
        call kernel(lfA.data, rfA.data, l_out, |lfB.span|);
        dma.copy l_out => output.chunkat(p, x, y - 1);
        swap lfA, lfB;
        swap rfA, rfB;
      }

      // Epilogue: process last chunk
      lf = select(#y % 2, lfB, lfA);
      rf = select(#y % 2, rfB, rfA);
      wait lf, rf;
      call kernel(lf.data, rf.data, l_out, |lfB.span|);
      dma.copy l_out => output.chunkat(p, x, y(-1));
    }
  }
}
```

Key techniques:
- **`foreach y(1:)`**: Skip the first iteration (handled by prologue).
- **`y - 1`**: Store to the previous chunk's position.
- **`y(-1)`**: Access the last value of `y` within its bound.
- **`select(#y % 2, ...)`**: Handle odd/even upper bounds for correct epilogue.

## Swizzle DMA

Swizzle DMA rearranges data layout during transfer to optimize bank conflict patterns in shared memory:

```choreo
dma.copy.swiz input => shared;
```

Swizzle is a cooperative feature: the DMA transfer writes data in a swizzled layout, and the subsequent MMA operation must read it in the matching swizzled layout. Both sides must agree on the swizzle pattern for correctness. The exact swizzle configuration depends on the target platform.

*(Reference: `tests/gpu/codegen/cute/dma_copy_swizzle_syntax.co`)*

