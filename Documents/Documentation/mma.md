# MMA (Matrix Multiply-Accumulate)

## Overview

Choreo maps the `mma.*` family to WMMA, `mma.sync`, WGMMA, and sparse MMA
instructions. The compiler selects the hardware atom from operand types,
shapes, layouts, storage, and target architecture.

The mathematical operation is:

```
D = A * B + C
```

The accumulator normally represents both C and D.

## Initialize an Accumulator

```choreo
acc = mma.fill.f32 0.0f;
mma.fill acc, 0.0f;
```

The first form creates an accumulator. The second reinitializes an existing
accumulator.

## Operand Forms

### Register fragments

`mma.load` materializes a register operand for WMMA, `mma.sync`, WGMMA RS mode,
or sparse metadata where the selected atom requires one:

```choreo
a_frag = mma.load lhs.chunkat(m, k);
mma.row.col acc, a_frag, b_frag;
```

### Direct WGMMA shared operands

For WGMMA SS mode, reference shared-memory views directly:

```choreo
mma.row.row acc,
    q_shared.subspan(64, 128).at(consumer, 0),
    k_shared;
```

The compiler constructs and hoists the required shared-memory descriptors.
There is no source-level `mma.desc` operation. Loading a WGMMA shared B operand
into a register fragment is rejected because it obscures the actual hardware
operand mode.

## Layouts

The layout suffix names A and B memory layouts:

| Syntax | A | B |
|--------|---|---|
| `mma.row.col` | row-major | column-major |
| `mma.row.row` | row-major | row-major |
| `mma.col.row` | column-major | row-major |
| `mma.col.col` | column-major | column-major |

Example:

```choreo
mma.row.col acc, a, b;
```

## Asynchronous MMA and Operation Futures

Assignment form with `.async` creates an operation future:

```choreo
qk = mma.row.row.async scores, q_shared, k_shared;
```

The accumulator remains the result object. `qk` represents completion only.

Wait before local code consumes the accumulator:

```choreo
wait qk;
reduce_max(scores_max, scores, 1);
```

WGMMA has no native completion-event operand. To release another asynchronous
path, wait for the operation future and then trigger the event directly:

```choreo
wait qk;
trigger k_empty;
```

Local work can be placed between that wait and publication:

```choreo
wait qk;
// Local work derived from scores.
trigger qk_done;
```

`trigger event after qk` is rejected rather than being lowered to a hidden
WGMMA wait. The compiler infers WGMMA commit and wait depth from explicit
future waits. There is no source-level `mma.wait` statement.

## Fragment Reductions

`reduce_sum` and `reduce_max` reduce a logical tensor dimension and write a
lower-rank fragment:

```choreo
reduce_sum(row_sum, scores, 1);
reduce_max(row_max, scores, 1);
```

`all_reduce_sum` instead preserves a replicated 1D fragment. It combines the
partial value held by each cooperating owner and broadcasts the sum back to
all of them:

```choreo
apply {i, j} in scores.span
  row_sum.at(i) = row_sum.at(i) + scores.at(i, j);
all_reduce_sum(row_sum);
```

The compiler infers the participating width from the fragment layout. No
thread count, lane mask, or hardware shuffle width is exposed in source. The
operand must have a replicated 1D layout, normally inferred from an `apply`
over an MMA accumulator.

## K-Tile Issue Schedule

A dense auto-split WGMMA can request a semantic K-tile permutation:

```choreo
[[schedule(k_tiles(7, 6, 5, 4, 0, 1, 2, 3))]]
pv = mma.row.col.async output, probabilities, v_shared;
```

The list must be a permutation containing exactly one entry for every
auto-split K tile. The annotation controls issue order, not a named CUDA helper
or a descriptor representation. The backend may select a specialized lowering
for a recognized order or generate the general sequence.

The schedule applies only to the following MMA execution statement. Sparse
WGMMA schedules are not currently supported.

## Store an Accumulator

```choreo
mma.store acc, output_shared;
```

The backend can select `stmatrix` when the target, accumulator layout, and
destination match. A later TMA operation can copy the shared tile to global
memory.

## Hardware Selection

| Target | Typical instruction |
|--------|---------------------|
| Volta+ | WMMA |
| Ampere+ | `mma.sync` |
| Hopper+ | WGMMA |
| Sparse | `mma.sync.sp` or WGMMA sparse |

The programmer does not encode atom names, K sizes, descriptor order, or
generated helper function names. Those are inferred backend details.

## Supported Types

Common combinations include:

| A/B | Accumulator | Notes |
|-----|-------------|-------|
| `f16` | `f16` or `f32` | Half precision |
| `bf16` | `f32` | Brain floating point |
| `tf32` | `f32` | TensorFloat-32 |
| FP8 | `f32` | Hopper+ |
| `s8` | `s32` | Integer MMA |
| `f64` | `f64` | Double precision |

Exact shape and type support depends on the selected target atom. Invalid
combinations produce a compile-time diagnostic.
