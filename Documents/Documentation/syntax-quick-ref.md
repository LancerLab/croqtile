# Syntax Quick Reference

## Function Markers

| Marker | Description |
|--------|-------------|
| `__co__` | Tileflow function (Croqtile DSL) |
| `__cok__ { ... }` | Separate-source device code block |
| `__co_device__` | Croqtile-style device function |
| `__device__` | Target-native device function (e.g., CUDA) |

## Choreo Function Templates

| Syntax | Description |
|--------|-------------|
| `template <typename T, int N, bool B> __co__ ...` | Primary template |
| `template __co__ foo<f16, 64, true>;` | Explicit concrete instance |
| `template <> __co__ ... foo<f16, 64, true>(...)` | Full specialization |
| `if constexpr (condition) { ... } else { ... }` | Compile-time branch |
| `__is_same<T, f16>` | Compare Choreo scalar types |
| `static_assert(condition, "message");` | Reject an invalid instance |

## Scalar Types

| Type | Description |
|------|-------------|
| `int` | 32-bit signed integer (program control) |
| `bool` | Boolean |
| `mutable int` | Mutable integer |

## Fundamental Types (Element Types)

| Type | Bits | | Type | Bits |
|------|------|-|------|------|
| `u8` | 8 | | `s8` | 8 |
| `u16` | 16 | | `s16` | 16 |
| `u32` | 32 | | `s32` | 32 |
| `f16` | 16 | | `s64` | 64 |
| `bf16` | 16 | | `f32` | 32 |
| `f64` | 64 | | `f8_e4m3` | 8 |
| `f8_e5m2` | 8 | | | |

## Shape (mdspan) Syntax

| Syntax | Example | Result |
|--------|---------|--------|
| Define | `sp : [7, 8]` | mdspan<2> |
| Rank annotation | `mdspan<3> sp : [a, b, c]` | Rank-checked |
| Element access | `sp(0)` | First dimension value |
| Arithmetic | `sp + 1`, `sp / 4` | Element-wise |
| Concatenation | `[sp, 6]` | Append dimension |
| Sugar derivation | `sp [(0)/2, (1)+1]` | Derive from base |
| Element count | `\|sp\|` | Product of dimensions |
| From spanned | `input.span` | Associated shape |

## I-Tuple Syntax

| Syntax | Example |
|--------|---------|
| Define | `t = {1, 2, 3}` |
| With keyword | `ituple<3> t = {a, b, c}` |
| Element access | `t(0)` |
| Concatenation | `{t, 4, 5}` |
| Arithmetic | `t + 1`, `shape / t` |

## Spanned Data

| Syntax | Example |
|--------|---------|
| Declare | `f32 [32, 16] buf` |
| With storage | `shared f16 [64] tile` |
| Initialize | `shared s32 [8] b {0}` |
| From shape | `f32 [input.span] out` |
| Parameter | `__co__ void f(f32 [M,K] x)` |

## Parallel-By

| Syntax | Description |
|--------|-------------|
| `parallel p by 6 { ... }` | Basic parallel block |
| `parallel by 2 { ... }` | Anonymous parallel variable |
| `parallel p by 2, q by 12 { ... }` | Multi-level (comma) |
| `parallel {x,y} by [3, 4] { ... }` | Sub-level with ituple |
| `parallel p by 6 : block { ... }` | With annotation |

## With-In and Foreach

| Syntax | Description |
|--------|-------------|
| `with x in 128 { ... }` | Bounded integer |
| `with index in [10, 10] { ... }` | Bounded ituple |
| `with {x,y} in [10, 10] { ... }` | Named elements |
| `with index = {x,y} in [10, 10] { ... }` | Named ituple + elements |
| `foreach x { ... }` | Iterate bounded variable |
| `foreach x in 128 { ... }` | Sugar: with + foreach |
| `foreach y(1::) { ... }` | Range expression |
| `foreach y(1:-1:2) { ... }` | Full range: start:end:stride |

## DMA Operations

| Syntax | Description |
|--------|-------------|
| `dma.copy src => dst` | Sync copy |
| `dma.copy.async src => shared` | Async copy to storage |
| `dma.pad<low, high, mid, val> src => dst` | Pad during transfer |
| `dma.transp<perm> src => dst` | Transpose during transfer |
| `dma.any` | Placeholder future |
| `dma.copy.async src => dst after f` | Chained DMA |
| `tma.load src => shared` | Synchronous global-to-shared TMA load (SM90+) |
| `tma.copy shared => dst` | Synchronous shared-to-global TMA copy (SM90+) |
| `f = tma.load.async src => shared` | Async TMA data future |
| `f = tma.load.async.evict_first src => shared` | TMA load with an L2 eviction hint |

## View Operations

| Syntax | Description |
|--------|-------------|
| `data.chunkat(p, index)` | Partition by bounded vars |
| `data.span_as([new_shape])` | Reshape |
| `data.view(ext).from(off)` | Explicit window |
| `data.view(ext : stride).from(off)` | Window with stride |
| `data.subspan(ext).at(coord)` | Fixed tile extents |
| `data.subspan(ext).step(s).at(c)` | Strided tile |

## MMA Operations

| Syntax | Description |
|--------|-------------|
| `mc = mma.fill 0` | Initialize accumulator |
| `mma.fill mc, 0.0f` | Re-initialize |
| `ma = mma.load data.chunkat(...)` | Load operand |
| `mma.row.row mc, shared_a, shared_b` | Direct WGMMA shared operands |
| `mma.row.col mc, ma, mb` | D = A * B + C (A row, B col) |
| `mma.row.row mc, ma, mb` | D = A * B + C (A row, B row) |
| `mma.col.row mc, ma, mb` | D = A * B + C (A col, B row) |
| `mma.col.col mc, ma, mb` | D = A * B + C (A col, B col) |
| `f = mma.row.row.async mc, a, b` | Async MMA operation future |
| `[[schedule(k_tiles(...))]]` | Set auto-split WGMMA K issue order |
| `mma.store mc, output.chunkat(...)` | Store result |

## Synchronization

| Syntax | Description |
|--------|-------------|
| `wait f` | Wait for async future |
| `wait f0, f1` | Wait for multiple |
| `swap(f1, f2)` | Exchange futures |
| `select(cond, f_t, f_f)` | Conditional future |
| `shared event e` | Declare event |
| `shared event e[N] = ready` | Declare initially ready generation zero |
| `trigger e` | Publish readiness immediately |
| `trigger e after f0, f1` | Bind native future completion to an event (no wait) |
| `wait e` | Wait for the current event generation |
| `wait e.at(order)` | Wait for a cyclic event generation |
| `trigger e.at(order++)` | Trigger the current generation, then advance order |
| `sync.shared` | Thread synchronization |

## Thread Masking

| Syntax | Description |
|--------|-------------|
| `inthreads (p < 2) { ... }` | Divergent block |
| `inthreads.async (p == 0) { ... }` | Async divergent |
## C++ Interop

| Syntax | Description |
|--------|-------------|
| `call func(args)` | Call device function |
| `call func<T>(args)` | Template call |
| `#pragma croq intrinsic prefix p` | Register intrinsic call prefix |
| `#pragma croq intrinsic namespace n` | Register intrinsic call namespace |
| `nullptr` | Null pointer argument |
| `__to<type>(expr)` | Explicit type conversion (validated) |
| `__to<"type">(expr)` | Foreign type cast (verbatim) |
| `#define NAME value` | Object-like macro |
| `#ifdef NAME ... #endif` | Conditional compilation |

## Inline Assembly

| Syntax | Description |
|--------|-------------|
| `__asm__("nop")` | Basic statement, no operands |
| `__asm__ volatile("nop" : : : "memory")` | Volatile with memory clobber |
| `__asm__("mov %0, %1" : "=r"(r) : "r"(v))` | Positional output + input operands |
| `__asm__("mov %0, %0" : "+r"(x))` | Read-write operand |
| `__asm__("mov %[d], %[s]" : [d] "=r"(r) : [s] "r"(v))` | Named (symbolic) operands |

## Assertions

| Syntax | Description |
|--------|-------------|
| `assert(cond, "msg")` | Compile-time if static, runtime if dynamic |

## Compiler Flags

| Flag | Description |
|------|-------------|
| `-e` | Dump AST |
| `-i` / `-ii` | Show type inference |
| `-vn` | Print value numbering |
| `-pa=PASS` | Print AST after pass |
| `-sa=PASS` | Stop after pass |
| `-es` | Emit target source only |
| `-gs` | Generate work-script |
| `-t TARGET` | Set target platform |
| `-arch=ARCH` | Set architecture |
| `--device-namespace=NS` | Wrap generated device kernels in `namespace NS` (empty = global) |
| `--runtime-check=LEVEL` | Set assertion level (none/entry/low/medium/high/all) |
| `--disable-runtime-check` | Disable all runtime assertions |
