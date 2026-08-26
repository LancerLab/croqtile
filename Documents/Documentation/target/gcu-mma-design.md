# GCU MMA Interface Design: acore/VACC Lowering

## Overview

This document describes the design for enabling Choreo's `mma.*` syntax on the
GCU topscc target, lowering to `acore::matmul` on VACC hardware. The goal is a
unified programming model where the same `.co` GEMM source compiles to both
GPU (CUTE mma.sync/WGMMA) and GCU (acore/VACC).

## Scope

**In scope:** `mma.fill`, `mma.load`, `mma.row.col`/`mma.row.row`,
`mma.store` on GCU via `MMAType::UKERNEL` lowering. K-loop accumulation state
machine. Out-of-line device stubs (ICALL workaround). Performance parity with
hand-tuned acore patterns.

**Out of scope:** `mma.scale`, sparse MMA on GCU, and lowering GPU-style MMA
operation futures on GCU. The GCU implementation remains synchronous.

## Architecture

```
.co source: mma.fill / mma.load / mma.row.col / mma.store
    |
    v
Parser (unchanged) -> AST::MMA node
    |
    v
ShapeInference / TypeInference / SemaChecker (unchanged, target-agnostic)
    |
    v
CodegenPrepare -> MMAInfo (unchanged, target-agnostic)
    |
    v
GCUAdaptor (NEW) -> validates shapes, sets MMAType::UKERNEL
    |
    v
TopsccCodeGen::Visit(MMA&) (NEW) -> emits acore::matmul via device stubs
```

## Semantic Mapping

| Choreo MMA op          | GCU Lowering                                      |
|-------------------------|----------------------------------------------------|
| `mc = mma.fill 0`      | Register accumulator state; first exec gets acc=0  |
| `ma = mma.load buf`    | No-op; validate buf is Storage::LOCAL              |
| `mma.row.col mc,ma,mb` | `acore::matmul<M, MK_KN>(...)` via stub            |
| `mma.row.row mc,ma,mb` | `acore::matmul<M, MK_NK>(...)` via stub            |
| `mma.store mc, out`    | Final acore call with store_flag=1                 |

## Performance-Critical Design Decisions

### P0: VACC K-accumulation

The compiler tracks a per-accumulator state machine:
- `mma.fill`: set first_exec=true
- First `mma.exec`: acc_flag=0, store_flag=0, launch_times=0
- Subsequent `mma.exec`: acc_flag=1, store_flag=0, launch_times=1
- `mma.store`: acc_flag=1, store_flag=1, launch_times=1

This keeps partial sums in VACC registers across K tiles, matching the 3.08+
TFLOPS hand-tuned pattern. The existing `__lib_gemm` emits acc=0,store=1 every
call, which is ~K_tiles times slower.

### P1: No implicit barriers

`mma.exec` codegen emits only the acore call. No waits, no barriers. DMA/compute
overlap is the user's responsibility via existing `dma.copy.async` + `wait`.

### P2: Out-of-line device stubs

Codegen emits acore calls inside `__co_device__` wrapper functions to avoid the
ICALL relocation bug. Overhead is <1% of compute time (confirmed by tuning).

### P3: vab_off for multi-accumulator

When multiple accumulators are live, compiler assigns distinct vab_off values
(multiples of 256) from the 4096-register VACC budget.

## Acore Config Table

Supported configurations are transcribed from `extern/include/common/matmul.h`
and encoded in `lib/Target/GCU/lower_libcall.hpp` (`kAcoreMatmulPatterns`,
`GetAcoreAlignments`, `IsAcoreSupportedStaticM`). The table is arch-agnostic:
the same constraints apply to gcu300, gcu400, and every other GCU arch, because
MMA lowers to the single software `acore::matmul` path with no per-arch branch.

- Static M: {1, 16, 32, 49, 64, 96, 128, 256}
- Dynamic M: 64-aligned, for f16/bf16/f32
- Formats: MK_KN (row.col), MK_NK (row.row)
- Input types: f16, bf16, f32, s8 (char). The accumulator/output type is
  modeled separately: f16/bf16 inputs may accumulate in float, and char input
  is a quantization op producing int output.

Format and shape restrictions (per matmul.h):

- M=49: MK_NK only.
- M=96: MK_KN only.
- M=256: MK_KN only.
- f32 M=1: MK_KN only; f32 M=128: MK_NK only.
- f16/bf16 float-accumulator rows exist only for the sizes listed in matmul.h
  (M=96/256 have no float-accumulator rows).
- M=512/1024 belong to `acore::addmm`, not `acore::matmul`; such sizes are
  served by the 64-aligned dynamic-M overload.

K/N alignment varies by (M, input type, output type, format) — see
`GetAcoreAlignments`.

## Async MMA Status

GPU async MMA uses an operation future:

```choreo
f = mma.row.col.async mc, ma, mb;
wait f;
```

The GPU backend infers the hardware wait depth from future dependencies. GCU
does not yet lower this form; its MMA state machine remains synchronous.
