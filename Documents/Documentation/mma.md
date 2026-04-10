# MMA (Matrix Multiply-Accumulate)

## Overview

Modern GPU hardware provides dedicated **Matrix Multiply-Accumulate (MMA)** units that perform small matrix multiplications in hardware. The underlying hardware operation is:

```
D = A * B + C
```

Croqtile exposes MMA through a simplified interface: a family of `mma.*` operations where the programmer specifies the operand layouts, and the **compiler infers the data types and tile shapes** to select the appropriate hardware MMA instruction automatically (WMMA, mma.sync, WGMMA, or their sparse variants). If the data shapes do not satisfy a valid MMA tile configuration, the compiler reports an error.

In most usage, the accumulator register serves as both C (input) and D (output), yielding the familiar `C += A * B` pattern.

## MMA Operations

### `mma.fill` -- Initialize Accumulator

```choreo
mc = mma.fill 0;
mc = mma.fill 0.0f;
```

Creates an accumulator matrix initialized to the given value. The accumulator is an opaque register-level object managed by the MMA hardware.

A variant re-initializes an existing accumulator:

```choreo
mma.fill mc, 0.0f;
```

### `mma.load` -- Load Operand

```choreo
ma = mma.load lhs.chunkat(m, k);
mb = mma.load rhs.chunkat(k, n);
```

Loads a tile of data from a spanned view into an MMA-compatible register fragment. The view operation (`chunkat`, `subspan`, etc.) must produce a tile whose dimensions are compatible with a hardware MMA shape. The compiler checks this and reports an error if the tile does not match any available MMA configuration.

### `mma.row.col` / `mma.row.row` -- Execute MMA

```choreo
mma.row.col mc, ma, mb;    // mc = ma * mb + mc  (A row-major, B col-major)
mma.row.row mc, ma, mb;    // mc = ma * mb + mc  (A row-major, B row-major)
```

Executes the matrix multiply-accumulate. The layout suffixes (`.row.col`, `.row.row`, `.col.row`, `.col.col`) specify the memory layout of operands A and B. The accumulator `mc` is both the input C and the output D -- effectively `mc += ma * mb`.

All four layout combinations are supported:

| Syntax | A Layout | B Layout |
|--------|----------|----------|
| `mma.row.col` | Row-major | Column-major |
| `mma.row.row` | Row-major | Row-major |
| `mma.col.row` | Column-major | Row-major |
| `mma.col.col` | Column-major | Column-major |

The compiler inspects the element types and tile shapes of `ma` and `mb`, then selects the matching hardware MMA instruction. If no instruction matches (unsupported type combination or tile shape), the compiler emits a diagnostic.

### `mma.store` -- Store Result

```choreo
mma.store mc, output.chunkat(m, n);
```

Stores the accumulator contents back to a spanned data view.

## Complete MMA Example

A basic matmul using MMA:

```choreo
#define M 128
#define N 256
#define K 64

__co__ auto matmul(f16 [M, K] lhs, f16 [K, N] rhs) {
  f16 [M, N] output;
  int MMA_M = 16, MMA_N = 16, MMA_K = 16;

  parallel {m, n} by [M / MMA_M / 4, N / MMA_N] : block {
    mc = mma.fill 0.0;
    parallel {g0, g1} by [4, 1] : group {
      mma.fill mc, 0.0f;
      foreach k in K / MMA_K {
        ma = mma.load lhs.chunkat(m#g0, k);
        mb = mma.load rhs.chunkat(k, n#g1);
        mma.row.col mc, ma, mb;   // mc += ma * mb
      }
      mma.store mc, output.chunkat(m#g0, n#g1);
    }
  }
  return output;
}
```

Note the `m#g0` syntax, which composes the block-level index `m` with the group-level index `g0` to form a hierarchical tile coordinate.

*(Reference: `tests/gpu/end2end/mma.co`, `tests/parse/mma.co`)*

## Hardware Mapping

The compiler automatically selects the hardware MMA instruction based on the operand types, tile shapes, and target architecture:

| Target | Hardware Instruction | How Selected |
|--------|---------------------|--------------|
| Volta+ | `wmma` | Type and shape match WMMA tile sizes |
| Ampere+ | `mma.sync` | Type and shape match PTX MMA tiles (m8n8k16, m16n8k8, ...) |
| Hopper+ | `wgmma` | Warpgroup-level types and shapes |
| Sparse | `mma.sync.sp` / `wgmma.sp` | Sparse operand detected (2:4 sparsity) |

The programmer does **not** select the instruction variant manually. Croqtile infers the correct instruction from the data types and shapes, making the MMA interface portable across GPU generations.

## Supported Types

MMA operations support a wide range of precision combinations:

| A Type | B Type | C Type | Notes |
|--------|--------|--------|-------|
| `f16` | `f16` | `f16`/`f32` | Standard half-precision |
| `bf16` | `bf16` | `f32` | Brain float |
| `tf32` | `tf32` | `f32` | TensorFloat-32 |
| `f8_e4m3` | `f8_e4m3` | `f32` | FP8 (Hopper+) |
| `f8_e5m2` | `f8_e5m2` | `f32` | FP8 (Hopper+) |
| `s8` | `s8` | `s32` | Integer MMA |
| `f64` | `f64` | `f64` | Double precision |

*(Reference: `tests/gpu/end2end/wmma/`, `tests/gpu/end2end/ptx_mma/`)*

## MMA with Fusion

Scalar operations can be fused with MMA:

```choreo
mma.row.col.scale mc, ma, mb, scale_factor;
```

*(Reference: `tests/gpu/end2end/mma_fusion_scalar.co`)*

## stmatrix

For Hopper targets, the `stmatrix` instruction provides efficient store from registers to shared memory. Croqtile can generate `stmatrix` when the MMA store pattern matches.

*(Reference: `tests/gpu/codegen/cute/stmatrix_codegen_enable.co`)*
