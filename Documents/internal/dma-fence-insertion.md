# Automatic DMA Memory-Visibility Fence Insertion

## Status

Implemented on `main` (all six stages of the fence-insertion plan: Stages
1-4 GCU fence insertion, Stage 5 the COIR path, and Stage 6 the
explicit-fence `order` axis). See the "What is and is not implemented"
section at the end.

## Overview

A DMA transfer moves data between two memory spaces, but "the copy finished"
does not mean "the copy is visible to the consumer". On hardware with
multiple cache levels and multiple execution entities (compute threads, DMA
engines, MMA accelerators), the producer and consumer of a buffer sit on
different sides of a cache boundary and need an explicit memory fence to
publish (release) or observe (acquire) each other's accesses.

Before this feature, every such fence was written by hand. Choreo kernels
carried ~250 explicit `tcle::fence<...>()` calls whose placement was a manual,
error-prone consequence of the surrounding DMA. This feature removes that
burden: the compiler now derives the required fences from the DMA edges that
are already in the IR and emits them automatically.

A user writes only:

```choreo
f1 = dma.copy.async global_ptr => shared;
// ... independent work ...
wait f1;
// ... consume shared ...
```

and the compiler inserts the `(SHARED, DMA, RELEASE)` fence at the `wait`
site with no user input.

## The fence model

The core type is `FenceKind` (`lib/types.hpp`), a four-fold request
`(space, entity, order, scope)`:

- `space`  (`Storage`)       -- which memory level the fence flushes/orders
                                (`LOCAL` / `SHARED` / `GLOBAL` / `NODE` / `NONE`).
- `entity` (`FenceEntity`)   -- which agent's accesses are ordered:
                                `THREADS` (the executing threads), `DMA` /
                                `TMA` (data-movement engines), `MMA` (the
                                matrix accelerator), or `ALL`.
- `order`  (`FenceOrder`)    -- the direction and strength:
                                `RELEASE` (publish the writer's stores),
                                `ACQUIRE` (order/invalidate the reader's
                                loads), `ACQ_REL` (full barrier), or
                                `SEQ_CST` (sequentially-consistent barrier).
                                The parser maps a missing `.acq`/`.rel`/
                                `.acq_rel`/`.seq_cst` suffix to `FenceOrder::DEFAULT`,
                                which semantic analysis resolves to `SEQ_CST`
                                on targets that support it and `ACQ_REL`
                                otherwise.
- `scope`  (`ParallelLevel`) -- the coherence domain the ordering is made
                                visible to. `kAutoScope` (`ParallelLevel::NONE`)
                                means "derive from `space`" via
                                `Target::GetDefaultFenceMemory`.

`FenceSelection` bundles the two sides of a DMA edge:

```cpp
struct FenceSelection {
  std::vector<FenceKind> producer; // emitted at the source site, before the DMA reads src
  std::vector<FenceKind> consumer; // emitted at the destination site, before the reader reads dst
};
```

Vector order is emission order; empty means "no fence at that site".

`FenceKind::Name()` renders the wire form `SPACE_ENTITY_ORDER[_SCOPE]`
(e.g. `shared_DMA_RELEASE`, `local_THREADS_RELEASE`). `FenceKindFromName`
parses it back, and `ParseFenceKinds` splits a comma-joined list. These
serialize/parse helpers are the round-trip format the analysis and the
codegen sink use to exchange fence annotations (see "Annotation and
serialization" below).

## Pipeline

Fence insertion is a **def-use walk over DMA edges**, split across two
pipeline phases (`lib/pipeline.cpp`):

1. `BufferAccessAnalyzer` runs in `PlanSemanticRoutine`, right before
   `SemaChecker`. It walks the normalized AST and records, per buffer, an
   ordered `(READ | WRITE, stmt, storage, entity)` log into
   `CompilationContext::GetBufferAccessLog()` (`lib/buffer_access.hpp`).
   A WRITE is a DEF/REDEF of the buffer; a READ is a USE.

2. `FenceInsertion` runs in `PlanCodeGenRoutine`, after `CodegenPrepare`.
   It consumes the access log and, for every `AST::DMA`, looks up the fence
   requirements from the target's `SelectDMAFences` table, elides fences
   already covered by same-engine in-order execution, and **annotates** the
   tree (see below). It does not emit anything itself.

3. The codegen sink (`TopsccCodeGen` for the GCU/topscc path) reads the
   annotation and emits the actual `tcle::fence<...>()` calls.

The decision runs once on the AST; no backend re-derives it.

## The edge model

The fence decision is a def-use walk classified by the *entities* at the two
endpoints of each DMA edge. The `.data` member exposes the DMA-to-DMA edge as
a direct symbol link (the parser strips `.data` to the underlying symbol), so
no alias analysis is needed.

| Edge | Producer | Consumer | Fence needed |
|------|----------|----------|--------------|
| compute -> DMA | threads | engine | producer (`LOCAL_THREADS_RELEASE`) |
| DMA -> compute | engine | threads | consumer (`SHARED_DMA_RELEASE`) |
| DMA -> DMA | engine | engine | depends on engine/queue; same in-order queue elides |
| ordering only (`after`) | -- | -- | none; `after` orders issuance, not visibility |

The two queries the decision needs, reduced to the access log
(`FenceInsertion::PrevWrite` / `NextRead` in `lib/fence_insertion.cpp`):

- "Who wrote `src` before this DMA read it?" -- previous WRITE of `src`.
- "Who reads `dst` after this DMA wrote it?" -- next READ of `dst` after
  the DMA's wait.

`FenceInsertion` is deliberately **not** built on `LivenessAnalyzer`:
liveness is an allocation-lifetime analysis that records WRITE as a `use`
and injects synthetic uses for loop ends and barriers; the fence decision
needs a faithful data-access log. The two only share the symbol-resolution
primitives (alias/binding/no-storage-alias/future-buffer), extracted into a
shared `AccessTracker` so they cannot drift on what a buffer *is*.

### Elision: same-engine in-order execution

Before emitting, `FenceInsertion` elides fences already subsumed by
same-engine in-order execution. In `Visit(AST::DMA)`, if the previous writer
of the source buffer is the *same* DMA/TMA engine entity, the producer fence
is dropped; if the next reader of the destination buffer is the same entity,
the consumer fence is dropped. A chain of DMAs on one engine needs no fence
between its own segments.

## GCU fence table

`SelectGCUDMAFences(src, dst)` in `lib/Target/GCU/gcu_fence.hpp` is the
release-only table keyed on the DMA's storage direction:

```cpp
if (src == GLOBAL && dst == SHARED)  return {{}, {SHARED, DMA, RELEASE}};
if (src == SHARED && dst == LOCAL)   return {{LOCAL, DMA, RELEASE}, {LOCAL, DMA, RELEASE}};
if (src == LOCAL  && dst == GLOBAL)  return {{LOCAL, THREADS, RELEASE}, {}};
return {};
```

Only RELEASE fences are produced: GCU's spaces (local VDMEM and shared L2)
are strongly ordered on the read side, so the ACQUIRE side is always elided.
Note the `LOCAL -> GLOBAL` producer is `THREADS` (not `DMA`) because the
source `l1_out` was written by the compute thread, not a DMA engine.

The default `Target::SelectDMAFences` returns `{}` (no-op), so targets with
no fence requirement are unaffected.

## topscc emission

`EmitAutoFences` in `lib/Target/GCU/topscc_codegen.cpp` reads the
comma-joined `FenceKind` note and maps each kind to a GCU `FenceType`:

| `space` | base `FenceType` | `RELEASE` | `ACQUIRE` | `ACQ_REL` |
|---------|------------------|-----------|-----------|-----------|
| `LOCAL`  | `L1_VDMEM` | `L1_VDMEM_STORE` | `L1_VDMEM_LOAD` | `L1_VDMEM` |
| `SHARED` | `L2_MEM`   | `L2_MEM_STORE`   | `L2_MEM_LOAD`   | `L2_MEM` |
| `GLOBAL` | `L3_MEM`   | `L3_MEM_STORE`   | `L3_MEM_LOAD`   | `L3_MEM` |

The `entity` axis has no distinct GCU `FenceType`, so it is ignored; a kind
with an unmapped space (e.g. `NONE`) is skipped.

The hooks emit at the semantically correct positions:

- `Visit(AST::DMA)`: producer fence **before** `DMACodeGen()` (publish the
  source writer's stores before the engine reads `src`), consumer fence
  **after** `DMACodeGen()` (for synchronous DMAs).
- `Visit(AST::Wait)`: consumer fence **after** the wait loop (publish the
  engine's writes before the threads read `dst`).

The `wait` fence is *not* redundant with the wait itself: `future.wait()`
guarantees the engine *retired* the copy, but the engine and the compute
threads are different entities with separate store paths, so completion does
not imply visibility. The `_STORE` fence publishes the engine's writes into
the shared space.

## Annotation and serialization

`FenceInsertion` attaches the decision to the AST as string notes:

- `dma_fence_producer` -- on the `AST::DMA` node (producer site).
- `dma_fence_consumer` -- on the `AST::DMA` node (sync DMA) or deferred to
  the matching `AST::Wait` (async DMA, via `consumer_by_future_` keyed on the
  scoped future name).

Each note holds `JoinKinds(sel)` -- the comma-joined `FenceKind::Name()`
list -- and the sink parses it back with `ParseFenceKinds`. This is a
serialize-then-parse round-trip through a flat string; a typed side-table on
`CompilationContext` would be cleaner, but the note keeps the annotation
colocated with the node it modifies.

## Options and observability

| Option | Default | Effect |
|--------|---------|--------|
| `--insert-dma-fences` | `true` | Gate the whole feature; `=false` skips both sinks and the user writes fences explicitly. |
| `--dump-fence-insertion` | `false` | Print each fence annotation as `[dma-fence] producer <src>-><dst> : <kinds>` / `[dma-fence] consumer ...`. |

Inserted fences are also counted into `AssessmentStats::FenceStats`
(producer/consumer counts, per-kind histogram, elided count), surfaced via
`PrintAssessmentStats`.

## Key files

| File | Role |
|------|------|
| `lib/types.hpp` | `FenceKind`, `FenceSelection`, `FenceEntity`, `FenceOrder`, `FenceKindFromName`, `ParseFenceKinds` |
| `lib/target.hpp` | `SelectDMAFences` / `GetDefaultFenceMemory` virtuals |
| `lib/buffer_access.hpp` | `BufferAccessEvent`, `BufferAccessLog`, `AccessEntity` |
| `lib/fence_insertion.{hpp,cpp}` | `FenceInsertion`: edge walk, elision, annotation |
| `lib/pipeline.cpp` | registers `BufferAccessAnalyzer` and `FenceInsertion` |
| `lib/codegen_utils.hpp` / `lib/command_line.cpp` | `insert_dma_fences`, `dump_fence_insertion` options |
| `lib/Target/GCU/gcu_fence.hpp` | `SelectGCUDMAFences` release-only table |
| `lib/Target/GCU/topscc_codegen.cpp` | `EmitAutoFences` + `Visit(AST::DMA)` / `Visit(AST::Wait)` hooks |
| `tests/gcu/codegen/topscc/dma-fence.co` | topscc codegen test with `// CHECK` lines |

## What is and is not implemented

Implemented (on `main`):

- Stage 1: the `FenceKind` / `FenceSelection` model, the
  `Target::SelectDMAFences` virtual, the GCU release-only table, the
  `--insert-dma-fences` / `--dump-fence-insertion` options, and the stats.
- Stage 2-3: the `AccessTracker` symbol-resolution extraction, the
  `BufferAccessAnalyzer`, and the `FenceInsertion` decision pass.
- Stage 4: the topscc (GCU) emission sink.
- Stage 5: the COIR path -- lower the annotation through `ASTCoIRGen` into a
  `coir.fence` op / wait attribute, plus `FenceElision`.
- Stage 6: an explicit `order` axis on `sync.fence` (`.acq` / `.rel` /
  `.acq_rel` / `.seq_cst`) and the corresponding COIR `order` attribute.

The `order` axis is honored by every target fence emitter to the extent its
hardware allows:

- GCU (topscc) is fully order-aware (`_STORE` / `_LOAD` / bare fence). It
  does not support `.seq_cst`; an explicit `.seq_cst` is rejected in semantic analysis.
- CPU (cc) emits `memory_order_release` / `acquire` / `acq_rel` / `seq_cst`.
- HIP (AMDGPU) emits directional `__builtin_amdgcn_fence` for
  `.rel` / `.acq`, full `__threadfence*` for `.acq_rel`, and
  `__builtin_amdgcn_fence(__ATOMIC_SEQ_CST, ...)` for `.seq_cst`.
- CUDA (cute) collapses `.rel` / `.acq` / `.acq_rel` to the full
  `__threadfence*` and lowers `.seq_cst` to the PTX `fence.sc.{cta,gpu}`; the
  standalone `fence` instruction only offers `.acq_rel` and `.sc` (no
  release-only or acquire-only form).

When no `.acq`/`.rel`/`.acq_rel`/`.seq_cst` suffix is given, the default is
`SEQ_CST` on targets that support it (CPU, AMDGPU, CUDA) and `ACQ_REL`
otherwise (GCU), per `Target::SupportsSeqCstFence`.

## Verification

```bash
make test-debug
./tests/lit.sh tests/gcu/codegen/topscc/dma-fence.co
```

The test exercises a `global -> shared` async copy followed by a thread read
(consumer fence at the wait) and a `local -> global` async copy (producer
fence at the DMA), and asserts the emitted `tcle::fence` lines plus their
absence when `--insert-dma-fences=false`.
