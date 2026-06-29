# Foreach iter_args Design: Loop-Carried Values in CoIR

Internal developer reference for how `coir.foreach` iter_args are generated,
lowered, and emitted -- particularly for the swap/double-buffering pattern.

---

## Overview

MLIR's SSA form requires explicit mechanisms for loop-carried values. A variable
that is assigned inside a loop and read in the next iteration (or after the loop)
must be threaded through `iter_args` + `yield`. This document describes how the
Choreo AST-to-CoIR lowering handles this.

---

## The Foreach iter_args Contract

```mlir
%results:N = coir.foreach %iv in %bound
    iter_args(%arg_0 = %init_0, ..., %arg_N = %init_N)
    : type_0, ..., type_N {
  // loop body -- may mutate values
  coir.yield %final_0, ..., %final_N : type_0, ..., type_N
}
// %results#i = last iteration's yield value for position i
```

**Semantics** (same as `scf.for` in upstream MLIR):
- On first iteration: `%arg_i = %init_i`
- On subsequent iterations: `%arg_i = previous iteration's yield value at position i`
- After the loop: `%results#i = final iteration's yield value at position i`

---

## What Gets Added to iter_args

The `ASTCoIRGen::Visit(AST::ForeachBlock)` collects three categories of names:

1. **Accumulator names** (`accNames`) -- values modified by `+=`, `-=`, etc.
2. **Scalar names** (`scalarNames`) -- scalar variables reassigned inside the loop
3. **Rotate names** (`rotateNames`) -- all identifiers appearing in `swap(...)` or
   `rotate(...)` calls within the loop body

For each rotate name (e.g. `fa`, `fb`):
- The async token value is added as an iter_arg
- If `<name>.data` exists and is a `!coir.tensor` type, it's also added

**Important**: iter_args is a SUPERSET of what's strictly needed. There is NO
liveness analysis. All collected names are yielded unconditionally, even if their
post-loop results are never referenced. Unused results are simply dead SSA values
(removable by DCE but harmless if left).

---

## The Yield Mechanism

The yield is emitted implicitly when the foreach body scope closes
(`AfterVisit(AST::ForeachBlock)`):

```cpp
for (auto &[name, _] : pendingYields) {
    auto val = LookupValue(name);
    if (val) yieldVals.push_back(val);
}
```

It looks up the **current** mapping of each name at the end of the loop body.
This means:
- Whatever the last `UpdateValue(name, ...)` set is what gets yielded
- The ordering of mutations within the body doesn't matter -- only the final state
- `swap(fa, fb)` works because it calls `UpdateValue("fa", rotated_result_0)` and
  `UpdateValue("fb", rotated_result_1)` before the yield is emitted

After the foreach, all results are mapped back to their names:

```cpp
for (unsigned i = 0; i < pendingYields.size(); ++i)
    resultMappings.push_back({pendingYields[i].first, foreach_.getResult(i)});
// ... later:
for (auto &[name, val] : resultMappings)
    UpdateValue(name, val);
```

This ensures post-loop code (e.g. `wait fa`) resolves to the foreach result.

---

## AsyncUndefOp: The Placeholder for Uninitialized Futures

When a rotate name (e.g. `fb`) has no initial value (declared as `dma.any`),
there is no SSA value to provide as the initial iter_arg. `coir.async.undef`
fills this role:

```mlir
%6 = coir.async.undef : !coir.async   // fb's initial value
%7:3 = coir.foreach ... iter_args(..., %arg6 = %6) { ... }
```

**Why it exists**: MLIR requires every iter_arg to have an initial value. Without
`async.undef`, there's no `!coir.async` value to bind the block argument.

**Safety contract**: The initial `async.undef` must NEVER be waited on before the
loop body assigns a real token to that slot. Choreo's double-buffering idiom
guarantees this -- only `fa` (which has a real initial DMA) is waited in
iteration 0. The `fb` slot receives its first real token from `dma.copy.async`
inside the loop body before it could ever be reached by a wait.

**Emission**: The topscc emitter generates `choreo::future __fut_N;` (a
default-constructed future with no DTE context). This is effectively a no-op
declaration.

---

## Double-Buffer Example: swap(fa, fb)

Source:
```
fa = dma.copy.async input.chunkat(0) => shared;
fb = dma.any;
foreach i(1:) {
    fb = dma.copy.async input.chunkat(i) => shared;
    wait fa;
    dma.copy fa.data => output.chunkat(i - 1);
    swap(fa, fb);
}
wait fa;
dma.copy fa.data => output.chunkat(i(-1));
```

### iter_args setup:

| Position | Name    | Type            | Initial Value               |
|----------|---------|-----------------|-----------------------------|
| 0        | fa      | !coir.async     | DMA invoke result (real)    |
| 1        | fa.data | !coir.tensor    | shared buffer alloc (%1)    |
| 2        | fb      | !coir.async     | coir.async.undef            |

### Inside the loop body:

1. `fb = dma.copy.async ...` -> produces `%14` (invoke token), maps `fb -> %14`
2. `wait fa` -> `coir.wait %arg4`
3. `dma.copy fa.data => output...` -> uses `%arg5` (fa.data iter_arg)
4. `swap(fa, fb)` -> `coir.async.rotate %arg4, %14` produces `%20#0, %20#1`
   - `UpdateValue("fa", %20#0)` -- new fa = old fb's token = %14
   - `UpdateValue("fb", %20#1)` -- new fb = old fa's token = %arg4
   - `.data` rotation: `UpdateValue("fa.data", fb_data)`, `UpdateValue("fb.data", fa_data)`

### Yield (implicit, at block end):

```
LookupValue("fa")      -> %20#0  (rotated: old fb)
LookupValue("fa.data") -> %5     (rotated: fb's buffer)
LookupValue("fb")      -> %20#1  (rotated: old fa)
```

Result: `coir.yield %20#0, %5, %20#1`

### After the loop:

```
foreach_result#0 -> mapped to "fa"      (last iteration's new-fa token)
foreach_result#1 -> mapped to "fa.data" (last iteration's fa buffer)
foreach_result#2 -> mapped to "fb"      (last iteration's new-fb token)
```

`wait fa` after the loop -> `coir.wait %result#0` (waits the last DMA issued)

---

## DMA Lowering Pipeline

After IRGen, the DMA lowering decomposes `coir.dma.copy` into a descriptor
pipeline. The passes run in order:

1. **ClassifyCopies** -- validates TMA usage against target capability
2. **LowerDMADesc** -- decomposes ALL global<->shared DmaCopyOp/TmaCopyOp:
   - `dma.copy %tile to %buf` becomes:
     - `dma.const.desc %base_src, %base_dst` (geometry, loop-invariant)
     - `dma.prefetch.desc` (cache hint)
     - `dma.runtime.desc offsets(...)` (per-iteration tile indices)
     - `dma.invoke` (trigger, produces `!coir.async` token)
   - TensorTileOp is consumed: base -> const.desc, indices -> runtime.desc
   - Non-global-shared copies (e.g. global->local) are NOT decomposed
3. **HoistDMAConfig** -- LICM hoists `const.desc` + `prefetch` above loops
   when their operands are loop-invariant
4. **LowerCopy** -- annotates remaining (non-decomposed) copy ops with metadata

### Hoisting behavior for double-buffer:

The `const.desc` for the fb DMA inside the loop takes `%arg0` (kernel arg) and
`%5` (shared alloc) -- both defined outside the loop. So it's hoisted:

```mlir
// Before loop (hoisted):
%7 = coir.dma.const.desc %arg0, %5 {kind=slice}
%8 = coir.dma.prefetch.desc %7

// Inside loop (stays -- offset is loop-variant):
%13 = coir.dma.runtime.desc %8 offsets(%arg3)
%14 = coir.dma.invoke %13
```

The output copy's `const.desc` uses `%arg5` (a loop-carried block arg) so it
is NOT hoisted -- it stays inside the loop.

---

## Topscc Emission for Descriptor Pipeline

| CoIR Op             | Topscc Output                                   |
|---------------------|-------------------------------------------------|
| dma.const.desc      | `future.configure(dst_mdspan, src_mdspan)`      |
| dma.prefetch.desc   | (propagates context name, no output)            |
| dma.runtime.desc    | `future.set_offset(dim, value)`                 |
| dma.invoke          | `future.trigger_only()` (async) or `.trigger_and_wait()` |
| async.undef         | `choreo::future __fut_N;` (default-constructed) |
| async.rotate        | `choreo::rotate(fut_a, fut_b)`                  |

The emitter maps `DMAInvokeOp` results to future variable names via the
`asyncFutures` map. This map is propagated through foreach iter_args: when a
block argument corresponds to an iter_arg whose initial value has an
`asyncFutures` entry, the block argument inherits that mapping.

---

## Design Decisions and Trade-offs

1. **No liveness analysis on iter_args**: Simpler implementation, avoids
   phase-ordering issues. Cost: slightly larger yield/result tuples. Could be
   optimized later with a dead-arg-elimination pass.

2. **async.undef vs ub.poison**: We use a custom op rather than MLIR's `ub.poison`
   to avoid pulling in the UB dialect dependency and to give a clear semantic name
   specific to async tokens.

3. **All global<->shared DMAs decomposed**: Previously tiled non-TMA copies were
   skipped (emitter handled them directly). Now all are decomposed uniformly.
   Benefit: hoisting works for all cases. Cost: the emitter for raw DmaCopyOp
   still exists for non-global-shared copies (e.g. global->local).

4. **const.desc inside loops (for output copy)**: When the source tensor is
   loop-carried (`%arg5` alternates between buffers), the const.desc cannot be
   hoisted. This means a fresh `configure()` per iteration for the output copy.
   A future optimization could factor this into two pre-configured descriptors
   and select between them.
