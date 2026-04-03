---
name: choreo-syntax
description: Reference Choreo `.co` syntax, primitives, and repo patterns. Use before creating, editing, or reviewing any `.co` source file, especially when choosing DMA/TMA/MMA forms, storage qualifiers, tiling, test directives, or hybrid host/device structure.
---

# Choreo syntax

Use this skill whenever the task touches a `.co` file. Treat it as the first syntax pass before editing, then pair it with `compile-and-test` for build/run validation or `develop-feature` / `develop-compiler` for broader implementation work.

## When this skill should trigger

- The user asks to create, edit, refactor, fix, or explain a `.co` file.
- The task mentions Choreo syntax, primitives, tiling, DMA/TMA/MMA, `parallel`, `with`, `foreach`, or parser behavior for `.co` code.
- The agent is about to modify `.co` tests under `tests/`, samples under `samples/`, or benchmarks under `benchmark/`.

## Core behavior

- Read nearby `.co` files first and match their local dialect, backend, and comment conventions.
- Use `choreo-syntax-reference.md` as the cross-repo inventory of observed syntax; treat it as evidence-based guidance, not a license to invent new syntax.
- Prefer an existing repo pattern over a novel spelling when both could work.
- Preserve hybrid `.co` structure when present: `#include`, macros, `main()`, `__cok__`, and `__co_device__` blocks are part of the file format in some areas.
- If validation is requested, also load `compile-and-test`.

## Performance-critical GPU rule

- For new GPU GEMM-like or blockscaled GEMM kernels, prefer `mma.op` as the tensor-core compute primitive.
- `mma.op` is the unified MMA primitive. Lowering chooses `wgmma`, `wmma`, `mma.sync`, `mma.sync.sp`, or `wgmma.sp` from the MMA shape plus the target GPU hardware.
- Do not ask the user which hardware MMA mnemonic to use for performance GEMM work; choose the right MMA shape and let lowering align it to the GPU.
- Use `benchmark/performance/matmul/` and `benchmark/performance/blockscale_gemm/` as the first performance references for tiling, staging, `tma.copy(.async)`, swizzle, `group-4`, events, and `mma.commit` patterns.
- Prefer programming with data tiles via `tma.copy` / `dma.copy` and tile-shaped buffers or views. Treat `.at()` as the most conservative access form and avoid making it the default in hot paths when tiled copy/view forms can express the work.
- Many current benchmark files still use legacy explicit forms such as `mma.row.row` and `mma.row.row.scale`; use them as structural references, but prefer `mma.op` for new GEMM-like code unless the task explicitly preserves legacy syntax.

## Copy-pattern selection

- The same shaping patterns can appear around both `dma.copy` and `tma.copy`; choose the copy primitive for backend/hardware, and choose the shaping pattern for the access semantics and inference behavior.
- Prefer `chunkat(...)` for natural chunks driven by bounded indices or `#p`-style partitioning.
- Prefer `view(...).from(...)` when the window shape, stride, or starting offset must be stated explicitly.
- Prefer `subspan(...).at(...)` when the tile extents are known and only the anchor changes.
- Prefer `subspan(...).step(...).at(...)` when tiles repeat with explicit spacing or staged/swizzled layout, especially in persistent or warp-specialized kernels.
- These are not just cosmetic rewrites. They can take different shape/stride inference paths and sometimes bypass non-trivial shape constraints.
- If a copy is hard to type or infer, first try rewriting it with another shaping pattern before assuming the declared shape is wrong, unless the compiler clearly reports inconsistent size, contiguity failure, or out-of-bounds.

## File mapping

- `tests/parse/`, `tests/infer/`, `tests/check/`: parser, inference, and diagnostic-oriented syntax examples.
- `tests/gpu/`, `samples/cuda/`, `benchmark/` GPU kernels: CUDA/CuTe-oriented `tma`, `mma`, swizzle, and role-qualified `parallel` forms.
- `tests/gcu/`, `samples/topscc/`: GCU/TOPSCC-oriented syntax, often with hybrid host/device code.
- `samples/factor/`, `samples/cse/`: transformation-heavy examples for `span_as`, `subspan`, multibuffering, and named indices.

## Editing workflow

1. Classify the `.co` file as parser test, infer/check test, end-to-end kernel, benchmark, or hybrid sample.
2. Find 2-3 nearby files in the same directory/backend and mirror their spelling.
3. Reuse observed forms for types, qualifiers, index decomposition, and primitives.
4. If copy typing or shape inference is hard, first try `chunkat`, `view(...).from(...)`, `subspan(...).at(...)`, or `subspan(...).step(...).at(...)` before changing declared shapes.
5. If a needed form is still unclear, search the wider `.co` corpus before guessing.
6. Keep the change minimal and stylistically local to the target file.

## Ask instead of guessing

Use a single inline choice question (`question`) when one answer would materially change the code shape:

- backend or dialect: generic frontend, CUDA/CuTe, or GCU/TOPSCC
- pure DSL vs hybrid host/device `.co`
- sync vs async copy/future pipeline
- `dma` vs `tma` vs `mma` family
- `local`, `shared`, or `global` storage destination
- tiling/index decomposition or named-dim layout
- which copy-shaping pattern should carry the access: `chunkat`, `view(...).from(...)`, `subspan(...).at(...)`, or `subspan(...).step(...).at(...)`
- the `mma.op` shape, accumulator type, transpose/sparsity semantics, or scale operands
- whether a hot path should remain tile-based with `tma.copy` / `dma.copy` or needs conservative scalar `.at()` handling only for edges
- parser/infer/check test vs executable sample expectations

When asking:

- do all non-blocked work first
- offer concise choices with the recommended default first
- state what changes based on the answer
- keep the conversation moving instead of stopping for open-ended clarification

## Guardrails

- Preserve existing `RUN`, `REQUIRES`, `CHECK`, `CHECK-NOT`, `CHKINF`, `VALNO`, and `GDB` conventions.
- Files under `lib/`, `runtime/`, `tools/`, and `tests/` must remain ASCII-only unless the file already intentionally contains non-ASCII content.
- After non-trivial `.co` or compiler-adjacent edits, run `make test-debug` before commit unless user-approved skip or clearly minor low-risk scope.
- Do not normalize a file from one backend style into another just because both are valid Choreo.
- For tests, prefer syntax already exercised nearby so diagnostics stay stable.
- For new performance GEMM-like GPU kernels, prefer `mma.op` even if nearby older files use explicit `mma.row.row`-style spellings.
- Prefer tile-level `tma.copy` / `dma.copy` programming over scalar `.at()` in performance-sensitive code; use `.at()` as the conservative fallback, not the default strategy.
- If copy typing is difficult, try changing `chunkat` / `view(...).from(...)` / `subspan(...).at(...)` / `subspan(...).step(...).at(...)` before concluding the declared shape is wrong.
- If the task is specifically about parser/codegen coverage for a legacy MMA primitive or scalar access pattern, preserve the exact syntax already used by that test.
- If a construct does not appear in nearby files or `choreo-syntax-reference.md`, say that and ask instead of inventing it.

## Supporting reference

- `choreo-syntax-reference.md` inventories common constructs, primitive families, type spellings, representative files, and ambiguity checkpoints.

## Related skills

- `compile-and-test` for running or debugging `.co` files.
- `develop-feature` for broader multi-file feature work that includes `.co` changes.
- `develop-compiler` when compiler sources under `lib/` or `tools/` also change.
