# DMA Resource Allocation

## Status

Implemented. The shared interval-coloring allocator and its target
integration are in place:

- `lib/heap_simulator.hpp` extracts the interval-coloring core (previously
  private to `MemReuse`) as a reusable `HeapSimulator`.
- `lib/resource_allocator.{hpp,cpp}` adds `DmaResourceAllocation`, a pass that
  runs `LivenessAnalyzer` and colors `FUTURE` and `EVENT` handles per device
  function into a `DmaResourcePlan`.
- The GPU backend (`lib/Target/GPU/cute_codegen.cpp`) replaces the greedy
  named-barrier pop with the event's colored slot indexed into
  `available_named_barrier_ids_`, keeping the mbarrier fallback for overflow.
- The pass is gated by two user options, both on by default:
  `-fdma-alloc` (FUTURE coloring) and `-fevent-alloc=<mode>` (EVENT coloring;
  `off`/`simple`/`full`, default `simple`; `full` reserves the upcoming
  scalar-replacement pass). Disabling either makes the corresponding codegen
  fall back to monotonic slot assignment.

Remaining work is tracked under "Known gaps to close" below (SALA binding for
events, and explicit loop-carried reuse scoping).

## Motivation

Croqtile programs DMA transfers that consume finite hardware resources on the
target device. Today these resources are allocated ad hoc, without real
liveness analysis:

- The GPU backend (`lib/Target/GPU/cute_codegen.cpp`) hands out copy atoms and
  TMA mbarriers by monotonic counter (`dma_count_`, `tma_future_count_`) and
  named-barrier IDs by popping a fixed pool (`available_named_barrier_ids_`,
  IDs 14..8), falling back to mbarriers when the pool is exhausted.

The result is that resource pressure is not minimized, and programs that issue
many concurrent DMAs can exhaust a fixed hardware budget even when most of the
transfers have already completed and their slots could be reused.

The compiler already has the machinery to do better: `LivenessAnalyzer`
(`lib/liveness_analysis.{hpp,cpp}`) computes live ranges for buffers *and*
futures, and `MemReuse` (`lib/mem_reuse.cpp`) already turns buffer ranges into
an interference-graph coloring for scratch memory. This design extends the
same model to DMA resources.

## Background: what the hardware forces us to allocate

A DMA resource, at the hardware level, is a **completion / ordering slot**. Its
lifetime is always the same shape:

```
acquire at async issue  ->  release at wait
```

The GPU (NVIDIA) instantiations are:

| Resource | Budget source | Lifetime | Handle in Choreo |
|---|---|---|---|
| mbarrier | 64 per CTA (sm_90+) | TMA issue -> wait (future) or event set -> wait | per-TMA-site, or event |
| named barrier (`bar.sync`) | 16 per CTA; Choreo uses IDs 14..8 | event set -> wait | event lowering |
| shared memory | arch table in `gpu_target.hpp` | buffer live range | already via `MemReuse` |
| TMA descriptor | static (resides in shared/const) | static | per-site (already minimal) |

`InitializeNamedEventLowering` reserves barrier 15 for warpspec handshake and
0..7 for CUDA/CUTLASS, leaving 14..8 for Choreo events; when that 7-slot pool
is exhausted it falls back to an mbarrier.

## Resource taxonomy

Every finite resource in the compiler belongs to exactly one class:

| Class | Members | Liveness today | Allocation |
|---|---|---|---|
| `BUFFER` | shared/local scratch | fully tracked (`VarRanges`) | `MemReuse` (done) |
| `FUTURE` | DMA completion slots (GPU TMA mbarrier, and analogous slots on other backends) | fully tracked | **new** (this design) |
| `EVENT` | GPU named barrier, scalar mbarrier | **partial** (use at `Wait`, no def) | **new** (this design) |
| `SITE` | GPU cp.async copy atom / TMA descriptor | n/a (static, already minimal) | leave as-is |
| `DELEGATED` | registers, tmem, rodata | n/a | leave to backend compiler |

Only `FUTURE` and `EVENT` need the new allocator. `BUFFER` is solved. `SITE`
is already `O(distinct DMA sites)` and must not be regressed. `DELEGATED` is
owned by the factor/ptxas backend compiler and must not be duplicated.

## The model

One interval allocator, parameterized by resource class:

```
ResourceClass in { FUTURE, EVENT }
```

For a device function:

1. Collect the handles of that class (futures, or events).
2. Read their live ranges from `LivenessAnalyzer::VarRanges()`.
3. Two handles *interfere* iff their ranges overlap (respecting the SALA
   `HBGraph` so provably non-overlapping async phases may share).
4. Greedy interval / graph coloring produces a `handle -> slot` map and a
   minimal slot count.

The allocator emits, per device function, a `handle -> slot` map and the
minimal slot count for that class:

- `FUTURE` handles map to DMA completion slots on backends that pool such
  slots; the pool size is the number of colors.
- On the GPU backend, `EVENT` handles map to named-barrier IDs; the lowering
  falls back to scalar mbarriers when `colors > available IDs` (7 today).

This is exactly the `MemReuse` pattern (`ProtoType` / `AnalyzeMemOffset` /
interference matrix) generalized from buffers to DMA handles.

## Known gaps to close

1. **Event defs.** *(closed)* `LivenessAnalyzer::Visit(AST::DMA)` now records a
   def for `dma->Event()` at issue, and `Visit(AST::NamedVariableDecl)` registers
   the event's scoped name, so `EVENT` handles get ranges symmetric to futures.

2. **SALA binding for events.** Futures are added to the `HBGraph` via
   `AddBinding`. Events need the same so the allocator does not over-serialize
   GPU events that provably cannot overlap. Still open.

3. **Loop-carried reuse.** A future defined and waited inside the same loop has
   a self-overlapping interval and cannot share a slot with itself across
   iterations. The allocator must make the scoping rule explicit (key intervals
   by scoped name, as `VarRanges()` already does).

4. **Event-array scalar replacement.** Event arrays (`event[N]`) are keyed by
   base symbol in liveness (`GetEventName` collapses `ElemOf` refs), so the
   whole array is a single handle. GPU named-barrier lowering rejects
   multi-element arrays (`element_count != 1`) so they fall back to one mbarrier
   array -- arrays never consume the restricted named-barrier pool. Modeling an
   array as a contiguous size-N block would reserve named-barrier slots no
   codegen reads and starve scalar events out of the pool (a regression). The
   current implementation therefore leaves event arrays uncolored; per-element
   reuse is tracked in a GitHub issue
   (https://github.com/LancerLab/croqtile/issues/6). Two scalar-replacement
   routes are under consideration:

   - **Route A -- virtual per-element naming.** Keep the array in the AST, but
     give each element a distinct virtual scoped name and live range in
     liveness (e.g. `ev[0]`..`ev[N-1]`). Codegen lowers `ev[i]` to
     `barrier[base + i]`, where `base` is the array's colored starting slot.
     Handles runtime-indexed `ev[i]`; requires per-element codegen lowering.
   - **Route B -- AST-level scalar replacement.** A scalar-replacement pass
     rewrites `event[N]` into N real scalar symbols (`ev0`..`evN-1`), turning
     the declaration and every constant-indexed `ElemOf` reference into a
     distinct symbol. Downstream liveness and codegen see real scalar events and
     need no change. This is the classic IR-level scalar-replacement action;
     only constant indices can be split (a runtime `ev[i]` needs guards or falls
     back to the array).

## Switches

The two liveness-driven allocator switches are target-agnostic:

| Switch | Default | Resource | Consumed by |
|---|---|---|---|
| `-fdma-alloc` | `true` | `FUTURE` | backends that pool DMA completion slots (GPU TMA mbarrier is not yet a consumer) |
| `-fevent-alloc=<mode>` | `simple` | `EVENT` | GPU named barriers |

`off` is a real, observable fallback on both switches:

- `-fdma-alloc=false` falls back to monotonic per-future slot assignment.
- `-fevent-alloc=off` restores the monotonic `named_event_lowerings_.size()`
  slot index in `cute_codegen.cpp`.
- `-fevent-alloc=full` is currently equivalent to `simple` (it reserves the
  planned event-array scalar-replacement pass; see "Known gaps").

## Ordering, fences, and optimizer interaction

Slot assignment is a compile-time constant (`handle -> slot` map). It can only
be invalidated if a later optimization extends one handle's live range over
another's -- that is, hoists a `Wait`/`Trigger` earlier or sinks a `DMA` later
across a synchronization edge. In practice this cannot happen:

- Named-barrier arrive/sync and the explicit compiler barriers are emitted as
  `asm volatile("..." ::: "memory")`, which is opaque to the backend compiler,
  so nothing reorders across them.
- DMA producers and consumers are separated by release/acquire fences
  (auto-inserted by `FenceInsertion`, emitted as `__threadfence*` / directional
  fences), pinning the data-movement order.
- A `Wait` reads the future the `DMA` wrote (RAW); a `Trigger`/`Wait` on an event
  is the same pattern. The compiler cannot hoist a read above its write.

So the compiler may only reorder within the freedom the synchronization edges
already encode, which is exactly the freedom liveness (and the SALA `HBGraph`)
compute. The slot map and the emitted code stay consistent by construction.

This is deliberately conservative. MLIR models memory effects over a single
flat memory space, but this target has leveled storage (register / local /
shared / global) plus finite hardware resources (named barriers, mbarriers,
copy atoms). A correct model must treat each storage level and each resource
pool as a distinct ordering and aliasing domain rather than one flat memory.
Relaxing the current conservative fences is future work, and any relaxation
must be gated on a per-level, per-resource model -- not a single-space
side-effect annotation.

## Integration points

- `lib/liveness_analysis.{hpp,cpp}`: close the event-def gap; optionally add a
  `ResourceClass` filter over `VarRanges()`.
- `lib/mem_reuse.cpp`: extract/reuse the interval-coloring core so both the
  buffer allocator and the DMA allocator share it.
- `lib/Target/GPU/cute_codegen.{hpp,cpp}`: replace the greedy
  `available_named_barrier_ids_` pop with the precomputed `event -> barrier_id`
  map, keeping the mbarrier fallback for overflow.
- `-fdma-alloc` / `-fevent-alloc` are the on/off switches (see Status).

## Non-goals

- Allocating registers, tensor memory, or rodata in Croqtile (backend-owned).
- Changing `SITE`-scoped resources (copy atoms, TMA descriptors); their current
  per-site allocation is already minimal and correct.

## Open questions

- Should the allocator run once per device function in a dedicated pass, or
  inline during codegen over `VarRanges()`?
- What is the exact threshold for GPU named-barrier fallback per arch (7 is
  hard-coded today for sm_90+; sm_70/80 may differ)?
- Is `SITE`-scoped mbarrier sharing (mutually-exclusive DMA sites collapsing
  onto one mbarrier) worth a future `SiteRanges()` keyed by `stmt2id[&dma]`?
  Deferred: the win is narrow and mbarrier phase state complicates sharing.
