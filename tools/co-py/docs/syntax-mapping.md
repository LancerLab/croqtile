# Choreo `.co` -> CroqPy Syntax Mapping

This document maps every Choreo `.co` construct to its CroqPy Python equivalent.

## Function Declaration

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `__co__ s32[6,64] ele_add(s32[2,3,64] lhs, s32[2,3,64] rhs)` | `@croq.co` with type annotations | Done |
| `global f16[M, N] output` (parameter storage) | `output: croq.Global(croq.f16[M, N])` | Done |
| Return type `s32[M, N]` | `-> croq.s32[M, N]` annotation or `@croq.co(ret=...)` | Done |
| `void` return (no return) | omit `return` or `return None` | Done |

```python
@croq.co
def ele_add(lhs: croq.s32[2,3,64], rhs: croq.s32[2,3,64]) -> croq.s32[6,64]:
    ...
```

## Control Flow

### `parallel ... by ...`

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `parallel p by 6, q by 64 { ... }` | `for p, q in croq.parallel(p=6, q=64):` | Done |
| `parallel p by 6 : block { ... }` | `for p in croq.parallel(p=6, scope=croq.BLOCK):` | Done |
| `parallel t by 128 : thread { ... }` | `for t in croq.parallel(t=128, scope=croq.THREAD):` | Done |
| `parallel p by 2 : group { ... }` | `for p in croq.parallel(p=2, scope=croq.GROUP):` | Done |
| `parallel p by 2 : group-4 { ... }` | `for p in croq.parallel(p=2, scope="group-4"):` | Done |

```python
for p in croq.parallel(p=6, scope=croq.BLOCK):
    for t in croq.parallel(t=128, scope=croq.THREAD):
        output[p, t] = lhs[p, t] + rhs[p, t]
```

Short alias: `croq.pb(...)`.

### `foreach ... in ...`

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `foreach k in K / MMA_K { ... }` | `for k in croq.foreach(k=K // MMA_K):` | Done |
| `foreach {m, n, k} in [M, N, K]` | `for m, n, k in croq.foreach(m=M, n=N, k=K):` | Done |
| `foreach k(1:) in ...` (staged) | `for _ in croq.foreach_staged(k, start=1):` | Done |

```python
for m, n, k in croq.foreach(m=128, n=256, k=256):
    out[m, n] = out[m, n] + lhs[m, k] * rhs[k, n]
```

Short alias: `croq.fe(...)`. Staged: `croq.fs(var, start=1)`.

### `with ... in ...`

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `with tile_k in 16 { ... }` | `with croq.with_in(tile_k=16) as tile_k:` | Done |
| `with tile_m in 8, tile_n in 8 { ... }` | `with croq.with_in(tile_m=8, tile_n=8) as (tile_m, tile_n):` | Done |

```python
with croq.with_in(tile_k=16) as tile_k:
    ma = croq.mma.load(lhs_s.chunkat(tile_k))
    mb = croq.mma.load(rhs_s.chunkat(tile_k))
    mc = croq.mma.exec(mc, ma, mb)
```

Short alias: `croq.wi(...)`. Also supports `for tile_k in croq.wi(tile_k=16):` syntax.

## Variable Declarations

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `s32[6, 64] output` (local) | `output = croq.declare(croq.s32[6,64], "output")` | Done |
| `shared f16[128, 64] buf` | `buf = croq.declare(..., "buf", storage=croq.SHARED)` | Done |
| `local s32[8] tmp` | `tmp = croq.declare(..., "tmp", storage=croq.LOCAL)` | Done |
| `f32[M, N] output = {0}` | `output = croq.declare(..., "output", init=0)` | Done |
| `int TILE_M = 64` | `TILE_M = croq.declare_int("TILE_M", 64)` | Done |

## Expressions

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `a + b`, `a - b`, `a * b`, `a / b` | `a + b`, `a - b`, `a * b`, `a / b` | Done |
| `a % b` | `a % b` | Done |
| `a < b`, `a > b`, `a <= b`, `a >= b` | `a < b`, `a > b`, `a <= b`, `a >= b` | Done |
| `a == b`, `a != b` | `a == b`, `a != b` | Done |
| `-a` (unary negation) | `-a` | Done |
| `m # g0` (hierarchical composition) | `m @ g0` (`@` operator) | Done |
| `(cond) ? a : b` (ternary) | `croq.select(cond, a, b)` | Done |
| `cdiv(M, N)` (ceiling division) | `croq.cdiv(M, N)` | Done |

## Data Access

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `output[p, q]` (read) | `output[p, q]` | Done |
| `output[p, q] = expr` (write) | `output[p, q] = expr` | Done |
| `output.at(p, q)` | `output[p, q]` | Done |
| `var.chunkat(i, j)` | `var.chunkat(i, j)` | Done |
| `var.chunkat(i, _)` (full slice) | `var.chunkat(i, croq.FULL)` | Done |
| `var.data` (unwrap future) | `var.data` | Done |
| `var.subspan(M, K).at(bm, bk)` | `var.subspan(M, K).at(bm, bk)` | Done |
| `var.view(...).from(...)` | `var.view(shape).from_(offsets)` | Done |
| `var.subspan(M, K).step(M, K).at(bm, bk)` | `var.subspan(M, K).step(M, K).at(bm, bk)` | Done |
| `var.subspan(M, K).at(s, 0).chunkat(_, iv)` | `var.subspan(M, K).at(s, 0).chunkat(croq.FULL, iv)` | Done |
| `output[p, q] += expr` (compound assign) | `output[p, q] += expr` | Done |
| `mc = var` (plain assignment) | `croq.assign(mc, mc + bias)` | Done |
| `var.span` / `var.span(i)` | Not yet supported | Planned |

## DMA (Direct Memory Access)

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `f = dma.copy src => dst` | `f = croq.dma.copy(src, dst)` | Done |
| `f = dma.copy.async src => shared dst` | `f = croq.dma.copy_async(src, dst, to=croq.SHARED)` | Done |
| `dma.copy.swiz<N> src => dst` | `croq.dma.copy(src, dst, swizzle=N)` | Done |
| `dma.transp<...> src => dst` | Not yet supported | Planned |
| `dma.pad<...> src => dst` | Not yet supported | Planned |
| `dma.any(f1, f2, ...)` | `croq.dma.any(f1, f2, ...)` | Done |

## TMA (Tensor Memory Access)

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `f = tma.copy src => dst` | `f = croq.tma.copy(src, dst)` | Done |
| `f = tma.copy.async src => shared dst` | `f = croq.tma.copy_async(src, dst, to=croq.SHARED)` | Done |
| `tma.copy.swiz<N> src => dst` | `croq.tma.copy(src, dst, swizzle=N)` | Done |
| `tma.copy.async<event[stage]>` | Not yet supported (event tokens) | Planned |
| `tma.copy.multicast` | Not yet supported | Planned |
| `tma.any(...)` | Not yet supported | Planned |

## MMA (Matrix Multiply-Accumulate)

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `mc = mma.fill 0.0f` | `mc = croq.mma.fill(0.0)` | Done |
| `mma.fill.f32 mc, 0.0f` | `mc = croq.mma.fill(mc, 0.0, dtype=croq.f32)` | Done |
| `ma = mma.load a_s.chunkat(k)` | `ma = croq.mma.load(a_s.chunkat(k))` | Done |
| `mma.load.swiz<N> ...` | `croq.mma.load(..., swizzle=N)` | Done |
| `mc = mma.row.col mc, ma, mb` | `mc = croq.mma.exec(mc, ma, mb, method="row.col")` | Done |
| `mma.store mc, output_s.chunkat(...)` | `croq.mma.store(mc, output_s.chunkat(...))` | Done |
| `mma.store.transp mc, ...` | `croq.mma.store(mc, ..., transpose=True)` | Done |
| `mma.commit` | `croq.mma.commit()` | Done |
| `frag mc[2][3] {0.0}` | `mc = croq.mma.frag("mc", [2, 3], 0.0)` | Done |
| `mma.fill mc[0][1], 0.0f` | `croq.mma.fill(mc[0][1], 0.0)` | Done |
| `mma.row.col mc[0][1], ma, mb` | `croq.mma.exec(mc[0][1], ma, mb, method="row.col")` | Done |
| `mma.store mc[0][1], ...` | `croq.mma.store(mc[0][1], ...)` | Done |
| `mma.row.row.sp ...` (sparse) | Not yet supported | Planned |
| `mma.row.row.scale ...` (block-scale) | Not yet supported | Planned |

## Synchronization

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `wait f` | `croq.wait(f)` | Done |
| `wait f1, f2` | `croq.wait(f1, f2)` | Done |
| `swap(a, b)` | `croq.swap(a, b)` | Done |
| `sync.shared` | `croq.sync("shared")` | Done |
| `wait full[stage]` (event-based) | Not yet supported | Planned |
| `trigger full[stage]` | Not yet supported | Planned |
| `shared event full[N]` | Not yet supported | Planned |

## Types

| Choreo `.co` | CroqPy Python |
|---|---|
| `s32`, `u32`, `s16`, etc. | `croq.s32`, `croq.u32`, `croq.s16` |
| `f32`, `f16`, `bf16`, `tf32` | `croq.f32`, `croq.f16`, `croq.bf16`, `croq.tf32` |
| `f8_e4m3`, `f8_e5m2` | `croq.f8_e4m3`, `croq.f8_e5m2` |
| `f64`, `s64`, `u64` | `croq.f64`, `croq.s64`, `croq.u64` |
| `f6_e2m3`, `f6_e3m2`, `f4_e2m1` | `croq.f6_e2m3`, `croq.f6_e3m2`, `croq.f4_e2m1` |
| `stream` | `croq.stream` |
| `s32[6, 64]` (tensor type) | `croq.s32[6, 64]` (subscript syntax) |

## Host Code

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `int main() { ... }` | `croq.host("""...""")` | Done |
| `#include <...>` | embedded in `croq.host(...)` | Done |

## Storage Qualifiers

| Choreo `.co` | CroqPy Python |
|---|---|
| `global` | `croq.GLOBAL` |
| `shared` | `croq.SHARED` |
| `local` | `croq.LOCAL` |

## Scope Annotations

| Choreo `.co` | CroqPy Python |
|---|---|
| `: block` | `scope=croq.BLOCK` |
| `: group` | `scope=croq.GROUP` |
| `: group-4` | `scope="group-4"` |
| `: thread` | `scope=croq.THREAD` |

## Math Builtins

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `__sqrt(x)` | `croq.sqrt(x)` | Done |
| `__rsqrt(x)` | `croq.rsqrt(x)` | Done |
| `__sin(x)` | `croq.sin(x)` | Done |
| `__cos(x)` | `croq.cos(x)` | Done |
| `__tan(x)` | `croq.tan(x)` | Done |
| `__sinh(x)` | `croq.sinh(x)` | Done |
| `__cosh(x)` | `croq.cosh(x)` | Done |
| `__tanh(x)` | `croq.tanh(x)` | Done |
| `__asin(x)` | `croq.asin(x)` | Done |
| `__acos(x)` | `croq.acos(x)` | Done |
| `__atan(x)` | `croq.atan(x)` | Done |
| `__exp(x)` | `croq.exp(x)` | Done |
| `__expm1(x)` | `croq.expm1(x)` | Done |
| `__log(x)` | `croq.log(x)` | Done |
| `__log1p(x)` | `croq.log1p(x)` | Done |
| `__pow(x, y)` | `croq.pow(x, y)` | Done |
| `__atan2(x, y)` | `croq.atan2(x, y)` | Done |
| `__ceil(x)` | `croq.ceil(x)` | Done |
| `__floor(x)` | `croq.floor(x)` | Done |
| `__round(x)` | `croq.round_(x)` | Done |
| `__sign(x)` | `croq.sign(x)` | Done |
| `__gelu(x)` | `croq.gelu(x)` | Done |
| `__sigmoid(x)` | `croq.sigmoid(x)` | Done |
| `__softplus(x)` | `croq.softplus(x)` | Done |
| `__isfinite(x)` | `croq.isfinite(x)` | Done |

## Device Printing

| Choreo `.co` | CroqPy Python | Status |
|---|---|---|
| `println(a, b)` | `croq.println(a, b)` | Done |
| `print(a, b)` | Not yet (only println) | Planned |

## Not Yet Supported

These constructs are identified in the GPU end-to-end tests but not yet implemented in CroqPy.
Sorted by priority (how many tests use them):

### High Priority -- All Done

- ~~**Dynamic parallel bounds**~~: Done. `parallel(m=croq.cdiv(M, WARP_M), scope=croq.BLOCK)`.
- ~~**`global` parameter storage**~~: Done. `output: croq.Global(croq.f16[M, N])`.
- ~~**Staged foreach**~~: Done. `croq.foreach_staged(k, start=1)`.
- ~~**View/From**~~: Done. `var.view(shape).from_(offsets)`.
- ~~**DMA/TMA swizzle**~~: Done. `croq.dma.copy(src, dst, swizzle=128)`.
- ~~**Device `if`**~~: Done. `with croq.device_if(cond):`.
- ~~**Compound `+=`**~~: Done. `output[m, n] += expr`.
- ~~**SubSpan/Step**~~: Done. `var.subspan(M, K).step(M, K).at(bm, bk)`.
- ~~**SubSpan+ChunkAt chaining**~~: Done. `var.subspan(M, K).at(s, 0).chunkat(_, iv)`.
- ~~**sync.shared**~~: Done. `croq.sync("shared")`.
- ~~**Plain assignment**~~: Done. `croq.assign(mc, mc + bias)`.
- ~~**Math builtins**~~: Done. `croq.sqrt(x)`, `croq.pow(x, y)`, etc. -- 25 math intrinsics.
- ~~**FP6/FP4 types**~~: Done. `croq.f6_e2m3`, `croq.f6_e3m2`, `croq.f4_e2m1`.
- ~~**Frag arrays**~~: Done. `croq.mma.frag("mc", [2, 3], 0.0)` with indexing `mc[0][1]`.
- ~~**println**~~: Done. `croq.println(a, b)`.
- ~~**stream type**~~: Done. `croq.stream` as parameter type.
- **Relative indexing**: `k(-1)` -- last iteration value in a `with` block.

### Medium Priority (used in 5-20 tests)

- **TMA event tokens**: `tma.copy.async<event[stage]>` -- async dependency chains for warpspec kernels.
- **Event system**: `shared event full[N]`, `trigger full[stage]`, `wait full[stage]` -- event-based synchronization.
- **`inthreads.async`**: Warp-specialization predicated code blocks.

### Lower Priority (used in <5 tests)

- **DMA transpose**: `dma.transp<1,0,2>` -- DMA with dimension permutation.
- **DMA pad**: `dma.pad<{L1,L2},{H1,H2},{0,0},V>` -- DMA with padding.
- **TMA multicast**: `tma.copy.multicast` -- multi-SM broadcast.
- **Sparse MMA**: `mma.row.row.sp` -- structured sparsity.
- **Block-scale MMA**: `mma.row.row.scale` -- block-scale WGMMA.
- **Span iteration**: `foreach {x, y} in lhs_s.span` -- shape-driven loops.
- **`var.span(i)` dynamic bounds**: Parallel bounds from tensor shape dimensions.
- **`ituple<N>`**: Bounded index tuple types.
- **`with idx in [tuple]`**: Iteration over index tuples.
