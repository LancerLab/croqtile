# View Operations

## Overview

Croqtile provides a family of **view operations** for selecting, reshaping, and striding into spanned data. These operations do not copy data -- they create new views (sub-regions or reinterpretations) of existing buffers. View operations are central to expressing tiling patterns in DMA statements.

> **Note on examples:** The examples below use `parallel-by` (ch 9) and `foreach` (ch 10), which are introduced in Part C. The view operations themselves are independent of control flow -- they operate on spanned data and bounded variables. Readers encountering `parallel` and `foreach` for the first time can treat them as "run this code in parallel" and "loop over this variable" respectively; the full semantics are covered in the following chapters.

The operations fall into two levels:

**High-level (regular tiling):** assume the data divides evenly into equal-sized blocks. Most kernel code lives here.

| Operation | Syntax | Best For |
|-----------|--------|----------|
| `chunkat` | `data.chunkat(p, index)` | Natural partitioning driven by bounded variables, iterating in order |
| `subspan.at` | `data.subspan(extents).at(coords)` | Fixed tile extents at any computed anchor |
| `subspan.step.at` | `data.subspan(extents).step(strides).at(coords)` | Fixed tiles with spacing between them |
| `span_as` | `data.span_as([new_shape])` | Reshape / reinterpret the shape before tiling |

**Low-level (arbitrary window):** the most general form; does not assume even partitioning.

| Operation | Syntax | Best For |
|-----------|--------|----------|
| `view.from` | `data.view(extents).from(offsets)` | Explicit window with explicit per-dimension strides |

## `chunkat` -- Natural Partition (in order)

The most common view operation. It assumes the data divides evenly into equal-sized blocks and accesses them in the natural order imposed by the bounded loop variables. The upper bounds of the variables determine the tile shape; the current values index into the sequence.

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

`input.chunkat(p, index)` is equivalent to `input.chunk({#p, #index}).at({p, index})`:

- The chunk shape is `[36, 14, 8] / {6, 7, 4} = [6, 2, 2]`.
- The chunk index `{p, index}` selects which chunk to access.

The `#` (ubound) operator extracts the upper bound of a bounded variable. See [DMA Tiling](dma-tiling.md) for a deeper treatment.

## `span_as` -- Reshape

`span_as` reinterprets the total element count of a buffer under a new shape. The product of the new dimensions must equal the product of the original dimensions.

```choreo
__co__ auto foo(s32 [32, 18, 9] a) {
  parallel p by 3 {
    with index = {x, y} in [4, 4] {
      foreach index {
        f = dma.copy a.span_as([16, 9, 36]).chunkat(x, p, y) => shared;
      }
    }
  }
}
```

Here `a` has shape `[32, 18, 9]` (5184 elements). `a.span_as([16, 9, 36])` reinterprets it as `[16, 9, 36]` (also 5184 elements). The `chunkat` then tiles the reshaped view.

`span_as` can also be applied to futures:

```choreo
f = dma.copy a.chunkat(p, index) => shared;
dma.copy f.span_as([12, 9]) => output.chunkat(x, y);
```

*(Reference: `tests/infer/span_as.co`)*

## `view(...).from(...)` -- Explicit Window (low-level)

This is the most general view form and the only one with explicit per-dimension stride control. Unlike `subspan` and `chunkat`, it makes no assumption that the data partitions evenly. Use it for sliding windows, non-unit strides, and any access pattern that does not fit a regular tiling.

```choreo
__co__ void foo(f32 [64, 32] a) {
  parallel p by 2 {
    b = a.view(16, 8).from(17, 18);
    dma.copy a.view(16, 4).from(14, 13) => shared;
    dma.copy a.view(17, 3 : 7, 2).from(11, 10) => shared;
  }
}
```

The syntax `a.view(extents).from(offsets)`:

- **`view(16, 8)`** -- window of shape `[16, 8]` with unit strides.
- **`view(17, 3 : 7, 2)`** -- window of shape `[17, 3]` with explicit strides `[7, 2]`. The `:` separates extent from stride per dimension.
- **`from(17, 18)`** -- places the window origin at `(17, 18)` in the source.

**Stride vs. step:** The strides here are memory-layout strides -- they describe how elements are laid out in the source buffer (i.e., how many elements to skip per step in each dimension). This is a low-level concept distinct from `subspan(...).step(...)`, which describes the spacing between adjacent tiles in a regular tiling.

**`span_as` limitation:** `span_as` requires contiguous layout. If `view.from` produces a non-contiguous view (i.e., strides do not match a compact layout), `span_as` cannot be applied to it.

*(Reference: `tests/infer/view_from.co`)*

## `subspan(...).at(...)` -- Fixed Tile at Any Anchor

`subspan` and `chunkat` both assume even partitioning, but they differ in how the tile is selected. `chunkat` accesses tiles in the natural order of its loop variables. `subspan(...).at(...)` lets you pick any tile by supplying explicit anchor coordinates -- the tile does not have to correspond to a loop-variable iteration:

```choreo
__co__ auto foo(s32 [32, 18, 9] a) {
  parallel p by 3 {
    with index = {x, y} in [4, 4] {
      foreach index {
        f = dma.copy a.subspan(2, 2, 1).at(x, p, y) => shared;
        g = dma.copy a.subspan(bshape, 9).at(index, p) => shared;
      }
    }
  }
}
```

`a.subspan(2, 2, 1).at(x, p, y)`:

- The tile shape is explicitly `[2, 2, 1]`.
- The anchor coordinate `(x, p, y)` is computed from bounded variables.
- The tile is a contiguous region of size `[2, 2, 1]` starting at the anchor.

The extents can include i-tuples and expressions:

```choreo
bshape = {16, 18};
cshape = {bshape, a.span(2) / #p};
dma.copy b.subspan(cshape).at(x, y, p) => shared;
```

Use `_` (wildcard) to take the full extent of a dimension:

```choreo
h = dma.copy f.subspan(_, 2, _).at(0, 0, 0) => shared;
```

*(Reference: `tests/infer/subspan.co`)*

## `subspan(...).step(...).at(...)` -- Tiles with Explicit Spacing

`step` extends `subspan` for cases where adjacent tiles are not immediately adjacent in memory -- the spacing between tile starts differs from the tile size itself (e.g., swizzled, persistent, or interleaved layouts). This is different from `view.from` strides: `step` operates at the tile level (tile index * step = start address), while `view.from` strides operate at the element level (element offset within the source layout).

```choreo
dma.copy a.subspan(tile_m, tile_k).step(step_m, step_k).at(m_idx, k_idx) => shared;
```

- **`subspan(tile_m, tile_k)`** -- the tile shape.
- **`step(step_m, step_k)`** -- the spacing between the starts of adjacent tiles along each dimension.
- **`at(m_idx, k_idx)`** -- tile index (multiplied by step to reach the anchor, not a raw element coordinate).

Like `subspan.at`, this form lets you access any tile by index. `chunkat` is the simpler alternative when tiles are accessed in the natural loop order and spacing equals tile size.

*(Reference: `tests/parse/stride.co`, `benchmark/performance/matmul/matmul_f16_dyn_persis_sta_sm90.co`)*

## Choosing the Right Operation

```
Is the data divided into equal-sized tiles?
  NO  --> view(...).from(...) [low-level; explicit element strides; most general]
  YES --> Does the access pattern follow natural loop order?
    YES --> chunkat [simplest; tile shape and index from bounded variables]
    NO  --> subspan(...).at(...) [any tile by explicit anchor]
              + .step(...) if tile spacing != tile size

Need to reshape before tiling?
  --> span_as [requires contiguous layout; cannot follow a non-contiguous view.from]
```

## Shape Inference Implications

Different view operations push the compiler through different shape and stride inference paths. If shape inference reports errors or unexpected results, try rewriting the access with a different view form before changing declared shapes. A practical order to try:

1. `chunkat` -- natural partitions in loop order
2. `subspan(...).at(...)` -- any fixed-extent tile at an explicit anchor
3. `subspan(...).step(...).at(...)` -- when tile spacing differs from tile size
4. `view(...).from(...)` -- when even partitioning cannot be assumed or stride control is needed
