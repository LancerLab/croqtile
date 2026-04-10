# Choreo `.co` reference

This inventory is derived from observed `.co` files in `tests/`, `samples/`, and `benchmark/`, plus project rules for preferred new syntax. Use it to match existing syntax, then verify against nearby files in the same directory and backend.

## Top-level structures

- Test and validation comments: `// REQUIRES:`, `// RUN:`, `// CHECK:`, `// CHECK-NOT:`, `// CHKINF:`, `// VALNO:`, `// GDB:`
- Host/C++ preamble is valid in many `.co` files: `#include`, `#define`, `template<...>`, `extern "C"`, `struct`, `main()`
- Choreo entry form: `__co__ ... { ... }`
- Hybrid kernel sections also appear: `__cok__ { ... }` with `__co_device__` helpers

## Common function and type spellings

- Returns: `__co__ auto foo(...)`, `__co__ void foo(...)`, `__co__ f32[M, N] foo(...)`
- Scalars observed often: `f32`, `f16`, `bf16`, `s32`, `u32`, `int`, `float`, `bool`
- Less-common spellings still present in tests: `half`, `double`, `f8`, `bfp16`
- Tensor and mdspan forms:
  - `f32[M, N]`
  - `f32 [M, N]`
  - `global f16[M, K] lhs`
- `f32 mdspan<2>`
- Storage qualifiers: `global`, `shared`, `local`

## Project rules for new performance kernels

- For new GPU GEMM-like and blockscaled GEMM code, prefer `mma.op` as the compute primitive.
- `mma.op` is the unified MMA primitive. Lowering chooses `wgmma`, `wmma`, `mma.sync`, `mma.sync.sp`, or `wgmma.sp` from the MMA shape plus the target GPU hardware.
- Use `benchmark/performance/matmul/` and `benchmark/performance/blockscale_gemm/` as the primary performance references for pipeline structure, staging, swizzle, and event patterns.
- Prefer programming with data tiles through `tma.copy` / `dma.copy`, `chunkat`, `subspan`, `span_as`, and `view(...).from(...)` rather than scalarized `.at()` hot paths.
- Treat `.at()` as the most conservative access form. Reserve it for edge handling, scalar glue, or places where tiled copy/view forms cannot express the access.
- Existing performance files still contain legacy explicit MMA forms such as `mma.row.row` and `mma.row.row.scale`; reuse their surrounding pipeline structure, but prefer `mma.op` in new GEMM-like code unless preserving a legacy test or benchmark exactly is the goal.

## Control and tiling forms

- `parallel by 12 {}`
- `parallel p by 6 {}`
- `parallel {p, q} by [128, 64] {}`
- `parallel bdim = {x, y, z} by [2, 4, 8] {}`
- Chained or nested forms: `parallel g by 2, b by 12 { ... }`
- Role-qualified forms: `parallel ... : block`, `parallel ... : group-4`
- Async form: `parallel.async by 3 ;`
- `with index in [2, 3] { ... }`
- `with index = {x, y} in [32, 16] { ... }`
- `with {x, y} in span_v4 { ... }`
- `where index2 <-> index3`
- `foreach index { ... }`
- `foreach idx, m { ... }`
- Slice controls: `foreach m(1:-1)`, `foreach m(2:)`, `foreach m(:-2)`, `foreach m(:)`
- `yield;`: make the device function return (without any safety check!)

## Shapes and bounded expressions

- `.span`, `.span(0)`, `|input.span|`
- Indexed span reorder: `input.span[(2), (3), (0), (4), (1), (5)]`
- Decomposition: `a#b`, `p#x`, `b#f`
- Shape declarations: `ndims: [3, b];`, `tile_factor : [32, 16];`
- Derived shapes: `f32[a.span]`, `local f32[batch/#p, seq/#q, hidden]`
- Dynamic markers observed in samples: `?`, `??`, symbolic dims such as `M`, `N`, `K`

## Access-shaping primitives

- `chunkat(...)` partitions a tensor/span into logical chunks driven by bounded indices, ubounds, or `#p`-style decomposition.
- `view(...)` defines an explicit window shape and, when provided, explicit strides.
- `from(...)` anchors a `view(...)` at explicit origin offsets.
- `subspan(...)` declares the tile extents to select from a larger tensor/span.
- `step(...)` declares the spacing between repeated `subspan(...)` tiles.
- `at(...)` anchors an already shaped view or subspan at concrete coordinates; by itself it is the most conservative access form.

## Data movement and view operations

- `dma.copy src => dst;`
- `f = dma.copy.async src => local;`
- Chained future: `... => local after i_shared;`
- `dma.transp<{1, 0, 2}> src => dst;`
- `tma.copy src => dst;`
- `tma.copy.async src => shared;`
- Swizzled variants: `tma.copy.swiz<128> ...`, `tma.copy.async<full[stage]>.swiz<MATMUL_SWIZ> ...`
- Preferred hot-path style: move data as tiles with `tma.copy` / `dma.copy`, then compute on tile-shaped buffers.
- Observed helpers:
  - `chunkat(...)`
  - `subspan(...)`
  - `span_as(...)`
  - `view(...).from(...)`
  - `step(...)`
  - `.at(...)` (conservative fallback, not the default performance style)

## Common copy-shaping patterns for `dma.copy` / `tma.copy`

- `chunkat(...)`
  - Feature: natural chunk decomposition from bounded indices and parallel partitioning.
  - Difference: tile sizes are mostly implicit from the source span and ubounds, so it is often the simplest expression for regular tiled data-parallel access.
  - Examples: `samples/topscc/elementwise/add.co`, `tests/infer/view_from.co`
- `view(...).from(...)`
  - Feature: explicit window extents, optional explicit strides, and explicit starting offsets.
  - Difference: better when the access is a shaped window rather than a partition-derived chunk, and often useful when you want to make shape/stride information obvious to inference.
  - Examples: `tests/infer/view_from.co`, `tests/gpu/end2end/add-shared.co`
- `subspan(...).at(...)`
  - Feature: explicit tile extents plus an anchor coordinate.
  - Difference: better when tile size is known and repeated at different anchors; more explicit than `chunkat(...)` about the tile extents being copied.
  - Examples: `tests/infer/subspan.co`, `tests/gpu/end2end/matmul/matmul_f16_dyn_sm90.co`
- `subspan(...).step(...).at(...)`
  - Feature: explicit tile extents, explicit tile spacing, and an anchor coordinate.
  - Difference: best for staged, swizzled, persistent, or non-unit-stride tile layouts where adjacent tiles are not expressed cleanly by plain `subspan(...).at(...)`.
  - Examples: `tests/parse/stride.co`, `benchmark/performance/matmul/matmul_f16_dyn_persis_sta_sm90.co`

## Shape-inference troubleshooting

- These access forms are not only different surface syntax. They can push the compiler through different shape and stride inference paths.
- If shape inference is hard, first try rewriting the copy with another access-shaping pattern before changing declared shapes or assuming the current shape math is wrong.
- A practical order is: `chunkat(...)` for natural partitions, `view(...).from(...)` for explicit windows and strides, `subspan(...).at(...)` for fixed tile extents, then `subspan(...).step(...).at(...)` when spacing must be explicit.
- Only stop and change the shapes first when the compiler clearly reports an actual inconsistency such as mismatched size, explicit non-contiguity/contiguity failure, or out-of-bounds.

## Futures and control primitives

- `dma.any`
- `wait f;`
- `select(cond, a, b)` and multi-way `select(...)`
- `swap(f4, f);`
- Future data and shape access: `f.data`, `f.span`

## MMA and TMA forms

- Preferred new GEMM primitive: `mma.op`
- `mc = mma.fill 0;`
- `mc = mma.fill.f16 0.0f;`
- `mc = mma.fill.f32 0.0f;`
- `ma = mma.load ...;`
- `ma = mma.load.swiz<MATMUL_SWIZ> ...;`
- `mma.row.col mc, ma, mb;`
- `mma.row.row mc, ma, mb;`
- `mma.row.row.scale mc, ma, mb, scale_a, scale_b;`
- `mma.row.col.sp mc, ma, mb;`
- `mma.commit;`
- `mma.store mc, output_s;`
- `mma.store.transp mc, output_s;`

## Printing and calls

- `call kernel(...);`
- `print(...)`, `println(...)`
- Compile-time print forms: `print!(...)`, `println!(...)`

## Representative files

- `tests/parse/parallel_by.co` - `parallel` variants, named/dimensioned bounds, async form
- `tests/parse/with_statement.co` - `with`, destructuring, `where`
- `tests/parse/with_foreach.co` - `foreach` controls, futures, `wait`
- `tests/parse/span_as.co` - `span_as`, future `.data`
- `tests/parse/select.co` - `select(...)`, local buffers, chained async copies
- `tests/infer/view_from.co` - `view(...).from(...)`
- `tests/infer/futures.co` - `dma.any`, `select`, `swap`
- `tests/gpu/end2end/tma.co` - `tma.copy(.async)` and `subspan`
- `tests/gpu/end2end/matmul/matmul_f16_dyn_sm90.co` - `tma`, `mma`, `: block`, `: group-4`
- `samples/factor/span_as_static0.co` - `span_as`, `dma.transp`, decomposed indices
- `samples/topscc/elementwise/add.co` - hybrid `__cok__`, `__co_device__`, host `main()`

## Performance GEMM references

- `benchmark/performance/matmul/matmul_f16_dyn_sm90.co` - baseline FP16 tensor-core GEMM structure with TMA, swizzle, shared tiles, and `group-4`
- `benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co` - staged warp-specialized GEMM with events and `mma.commit`
- `benchmark/performance/matmul/matmul_e4m3_dynamic_sm90.co` - FP8/e4m3 GEMM constraints and tile choices
- `benchmark/performance/blockscale_gemm/blockscale_gemm_e4m3_dynamic_sm90.co` - blockscaled GEMM with scale operands and tensor-core pipeline
- `benchmark/performance/blockscale_gemm/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co` - blockscaled warp-specialized GEMM with event staging

## Ask instead of guess

- Backend target or directory style is mixed.
- The file should be pure DSL but nearby samples are hybrid C++/host files.
- Async pipeline behavior affects correctness: `dma.copy.async`, `wait`, `after`, `swap`.
- Storage target is unclear: `local` vs `shared` vs `global`.
- The `mma.op` shape, accumulator type, transpose/sparsity semantics, or scale operands are unclear.
- It is unclear whether a hot path can stay tile-based or truly needs conservative scalar `.at()` handling.
- Test intent is unclear: parser/infer/check vs end-to-end execution.
- Tiling or index decomposition would be invented rather than copied from nearby examples.

## Practical search hints

- Start in the same directory as the target file.
- Then search globally for the exact primitive you need, such as `mma.load.swiz`, `tma.copy.async`, `span_as`, or `view(...).from(...)`.
- Prefer examples that share the same backend marker in comments or path, such as `gpu`, `gcu`, `topscc`, or `sm90`.
