# Parallel Index Codegen (virtual thread/group indices)

## Overview

A `parallel ... by ...` block does not map one-to-one onto a CUDA builtin.
The compiler emits **virtual index preludes** -- named locals derived from
`threadIdx.x` -- that translate each Choreo parallel variable into a lane or
group index. This page documents that mapping, the naming convention, and a
class of bug where the wrong index is chosen when a `thread` parallel is
nested inside a `group`.

This applies to both GPU codegen backends:

| Backend | Binary | Source |
|---------|--------|--------|
| Native CuTe | `choreo` | `lib/Target/GPU/cute_codegen.cpp` |
| CoIR (MLIR) | `cocc` | `tools/coir/lib/CodeGen/GPU/EmitCUDA.cpp` |

## Parallel Level Hierarchy

Choreo normalizes every parallel structure into a fixed nesting of levels.
The enum order (smallest scope first) is:

```
THREAD -> GROUP -> GROUPx4 -> BLOCK -> CLUSTER -> DEVICE -> SEQ
```

- `THREAD`: one thread; index within the enclosing group (a "lane").
- `GROUP`: a warp of 32 threads.
- `GROUPx4`: a warp group of 128 threads.
- `BLOCK` / `CLUSTER` / `DEVICE`: coarser scopes.

The level enum is generated from `tools/coir/include/Dialect/CoIR/CoIRAttrs.td`
as `ParallelLevel` (`THREAD = 0 ... SEQ = 6`).

## Virtual Index Names

Both backends emit block-scoped locals with the prefix `__choreo_v`
(`vid_pfx` in `cute_codegen.cpp:196`):

| Level | Name | Definition |
|-------|------|------------|
| THREAD | `__choreo_vtid_x` (also `_y`, `_z`) | lane within the enclosing **enforced** group: `threadIdx.x % 128` (GROUPx4), `threadIdx.x % 32` (GROUP), or raw `threadIdx.x` (no enforced group) |
| GROUP | `__choreo_vgid_x` (also `_y`, `_z`) | `threadIdx.x / 32` |
| GROUPx4 | `__choreo_vg4id_x` (also `_y`, `_z`) | `threadIdx.x / 128` |
| BLOCK | `blockIdx.x` etc. | builtin |

Multi-dimensional bounds are decomposed row-major: `gid` (or `g4id`) is the
flat group index, then each dimension is `gid / (product of trailing bounds)
% bound`.

These are codegen-emitted locals, **not** runtime helpers -- there is no
`__choreo_v*` symbol in `runtime/`.

## Normalization Invariant: THREAD always sits directly under GROUP

`ParaByFiller` in `lib/normalize.hpp` (`FillMissingLevels`,
`FillMissingBetweens`, `FillMissingInners`, `FillMissingOuters`) inserts every
intermediate parallel level that the user omitted. The inserted levels are
**non-enforced** (`SetEnforced(false)`).

Consequences:

- A `thread` parallel is always nested directly inside a `group` parallel.
  The `group` is **enforced** only when the user wrote it explicitly (e.g.
  `parallel ... : group`); otherwise it is auto-inserted (non-enforced) with
  size 1, and the `thread` parallel effectively spans the whole block.
- A standalone `parallel ... by [N]` with no explicit level is expanded to
  `block -> group -> thread` (the `group` is auto-inserted with size 1).

The thread lane width therefore depends on whether the enclosing `group` is
*enforced*: `% 32` under an enforced `group`, `% 128` under an enforced
`group4`, and raw `threadIdx.x` when the enclosing `group` is non-enforced (or
absent). See the next section.

```co
// source -- enforced group: thread is a lane within the warp
parallel q0 by in1.span(1) : group {
  parallel {q1} by [in1.span(2)]      // implicit thread
    ...
}
```

emits (CoIR):

```cuda
// parallel level=group bounds=[1]     // enforced
{
  [[maybe_unused]] auto __choreo_vgid_x = threadIdx.x / 32;
  // parallel level=thread bounds=[32]
  {
    [[maybe_unused]] auto __choreo_vtid_x = threadIdx.x % 32;
    ...
  }
}
```

```co
// source -- no explicit group: auto-inserted (non-enforced) group of size 1
parallel q0 by 64 { ... }             // thread spans the whole block
```

emits (CoIR):

```cuda
// parallel level=group bounds=[1]     // non-enforced
{
  [[maybe_unused]] auto __choreo_vgid_x = threadIdx.x / 32;
  // parallel level=thread bounds=[64]
  {
    [[maybe_unused]] auto __choreo_vtid_x = threadIdx.x;   // NOT % 32
    ...
  }
}
```

*(Reference: `tests/gpu/end2end/copy.co` for the enforced case,
`tests/gpu/end2end/add-shared.co` for the standalone case.)*

## Enforced-level tracking (native and CoIR)

The lane width is decided by the innermost **enforced** group level in scope:

- Native (`cute_codegen.cpp`) tracks a `bdim_level` member, set **only** for
  *enforced* GROUP/GROUPx4 (lines ~968) and reset to THREAD at the end of the
  thread parallel (line ~1389). `EmitDeviceVirtualIndices` (line ~9773) then
  switches on it: `% 128` under GROUPx4, `% 32` under GROUP, raw `threadIdx.x`
  standalone.
- CoIR (`EmitCUDA.cpp`) mirrors this with a `bdimLevel` member of type
  `coir::ParallelLevel`. It is set in the GROUP/GROUPx4 branch of
  `emitParallel` when the op carries the `coir.enforced` attribute, and
  restored on scope exit (the save/restore mirrors the recursive traversal).

### The `coir.enforced` attribute

The CoIR `ParallelOp` (in `CoIROps.td`) has **no** enforced flag in its
operand/attribute list -- the AST's enforced bit is dropped during lowering.
To let `EmitCUDA` distinguish an explicit `: group` from an auto-inserted one,
`ASTCoIRGen::Visit(AST::ParallelBy&)` now records the flag as a free
`coir.enforced` boolean attribute on the op:

```mlir
coir.parallel (%arg) in [1] level = #coir.level<group> attributes {coir.enforced = true} {
```

`emitParallel` reads it via `op->getAttrOfType<mlir::BoolAttr>("coir.enforced")`.

### Regression note

An earlier fix hard-coded `% 32` for every THREAD parallel, assuming the
enclosing level is always an enforced group. That broke standalone thread
parallels (e.g. `add-shared.co`, which launches `<<<6, 64>>>` with a bare
`parallel q by 64`): indices `32..63` were truncated to `0..31`, producing wrong
results and a `choreo_assert` trap. The correct behavior is the three-way
switch above; both backends must derive the lane width from the *enforced*
group level, not from the mere presence of a group ancestor.

## The `[[maybe_unused]]` requirement

Virtual index preludes must be declared `[[maybe_unused]]`. A prelude whose
variable is not referenced in a particular kernel triggers nvcc
`variable declared but never referenced` warnings, which pollute stdout and
break `FileCheck --match-full-lines` in `tests/lit.sh`. The native backend
does the same via the `mu` prefix in `EmitDeviceVirtualIndices`.

## Reproduce / verify

```bash
# rebuild the CoIR backend after editing EmitCUDA.cpp
ninja -C build cocc

# dump generated CUDA (no compile) and inspect virtual indices
./cocc -t cute -es tests/gpu/end2end/copy.co | grep -E '__choreo_v|parallel level='

# run the lit test
bash tests/lit.sh tests/gpu/end2end/copy.co
```

`tests/gpu/end2end/copy.co` is the smallest runnable verification on sm_86;
`fp6_fp4_copy.co` and `fp8_ops.co` exercise the same nested pattern but require
SM_90+ (`REQUIRES: TARGET-GPU` + arch guard) and skip locally.

## Known remaining issue

The wmma `store_matrix_sync` store-conversion path in `EmitCUDA.cpp` still
contains two hard-coded `threadIdx.x / 32` / `threadIdx.x % 32` expressions
instead of using the enclosing group's virtual index. They should be derived
from the enclosing `ParallelOp` rather than hard-coded. Tracked in the
upstream issue tracker (see `scripts/oss` docs for the public repo URL).
