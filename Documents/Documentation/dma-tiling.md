# DMA Tiling

Any [view operation](view-operations.md) can be used on either side of a DMA statement to describe tiled data transfers. In practice `chunkat` is the most common form, so this chapter focuses on it, but `subspan`, `view.from`, and `span_as` are equally valid.

This chapter builds on three earlier chapters:

- [View Operations](view-operations.md) (ch 8) introduced `chunkat`, `subspan`, `view.from`, and `span_as`.
- [Parallel-By](parallel-by.md) (ch 9) introduced `parallel` blocks that provide the bounded `p` variables used as tiling factors.
- [Iteration](iteration.md) (ch 10) introduced `with-in` / `foreach` that provide the bounded index variables swept by the loop.

The examples below combine all three to show the complete pattern.

## The Ubound Operator

Before diving into tiling, recall the **ubound** operator `#`, which extracts the upper bound of a bounded variable:

```choreo
with size in [32] {
  shared u8 [#size, 16] buf;  // shape [32, 16]
  shared u8 [size, 16] bad;   // shape [0, 16] -- current value, not upper bound!
}
```

The `#` prefix is essential for shape computations involving bounded variables.

## ChunkAt Expression

The `chunkat` expression divides spanned data into equal-sized chunks, driven by bounded variables. It is the most common way to express tiled data movement in Croqtile.

### Basic Example

```choreo
__co__ void foo(f32 [36] input) {
  parallel p by 6 {
    dma.copy input.chunkat(p) => shared;
  }
}
```

`input.chunkat(p)` is shorthand for `input.chunk(#p).at(p)`:

- **Tiling factor**: `#p = 6` (upper bound of `p`).
- **Chunk shape**: `[36] / {6} = [6]`.
- **Chunk index**: `p` (0 through 5), selecting which chunk to access.

Each of the 6 parallel threads copies a different 6-element chunk to shared memory.

### Multiple Bounded Variables

```choreo
__co__ void foo(f32 [36, 14, 8] input) {
  parallel p by 6 {
    with index in [7, 4] {
      foreach index {
        dma.copy input.chunkat(p, index) => shared;
      }
    }
  }
}
```

The bounded variables are concatenated: upper bounds `{#p, #index} = {6, 7, 4}`, current values `{p, index}`.

- Chunk shape: `[36, 14, 8] / {6, 7, 4} = [6, 2, 2]`.
- 6 parallel threads, each iterating 7x4 times, each iteration copying a `[6, 2, 2]` chunk.

## Source vs Destination ChunkAt

`chunkat` can appear on the source side (dividing input into chunks) or the destination side (assembling output from chunks):

```choreo
__co__ auto foo(f32 [36] input) {
  f32 [36] output;
  parallel p by 6 {
    f0 = dma.copy input.chunkat(p) => shared;     // tile source
    dma.copy f0.data => output.chunkat(p);          // assemble destination
  }
  return output;
}
```

Using `chunkat` on both source and destination simultaneously is rarely supported by target platforms.

## Composing ChunkAt with Other View Ops

`chunkat` can be chained after `span_as` for reshape-then-tile patterns:

```choreo
f = dma.copy a.span_as([16, 9, 36]).chunkat(x, p, y) => shared;
```

And with futures:

```choreo
dma.copy f.data.span_as([12, 9]) => output.chunkat(x, y);
```

## How the Compiler Uses Chunk Information

The Croqtile compiler uses the bounded variable information to:

1. Compute the chunk shape (data shape / upper bounds).
2. Compute the offset for each chunk (current values * chunk shape).
3. Generate the appropriate DMA configuration (stride, offset, size) in the target code.

This eliminates the manual offset arithmetic that C++ programmers typically write, reducing errors and improving readability.
