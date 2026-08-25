# Design: Liveness-Driven DMA Resource Allocation

## Status

Implemented. The shared interval-coloring allocator and its two target
integrations are in place:

- `lib/heap_simulator.hpp` extracts the interval-coloring core (previously
  private to `MemReuse`) as a reusable `HeapSimulator`.
- `lib/resource_allocator.{hpp,cpp}` adds `DmaResourceAllocation`, a pass that
  runs `LivenessAnalyzer` and colors `FUTURE` and `EVENT` handles per device
  function into `DmaResourcePlan`.
- GCU (`lib/Target/GCU/topscc_codegen.cpp`) replaces the greedy first-seen /
  `-fdte-merge` slot assignment with the precomputed plan; the DTE pool size is
  the number of colors.
- GPU (`lib/Target/GPU/cute_codegen.cpp`) replaces the named-barrier
  front/pop with the event's colored slot indexed into
  `available_named_barrier_ids_`, keeping the mbarrier fallback for overflow.
- The pass is gated by two user options, both on by default:
  `-fdma-alloc` (FUTURE coloring; replaces `-fdte-merge`) and
  `-fevent-alloc=<mode>` (EVENT coloring; `off`/`simple`/`full`, default
  `simple`; `full` reserves the upcoming scalar-replacement pass). Disabling
  either makes the corresponding codegen fall back to monotonic slot
  assignment.

Remaining work is tracked under "Known gaps to close" below (SALA binding for
events, and explicit loop-carried reuse scoping).

## Motivation

Choreo programs DMA transfers that consume finite hardware resources on the
target device. Today these resources are allocated ad hoc, without real
liveness analysis:

- GCU (`lib/Target/GCU/topscc_codegen.cpp`) assigns each async DMA a fixed
  DTE-pool slot (`--use-dte-pool`, default on). The old `-fdte-merge` was a
  single-pass greedy approximation that reused a *waited* future's slot; it
  has been replaced by `-fdma-alloc` (the liveness-driven pass, on by
  default).
- GPU (`lib/Target/GPU/cute_codegen.cpp`) hands out copy atoms and TMA
  mbarriers by monotonic counter (`dma_count_`, `tma_future_count_`) and
  named-barrier IDs by popping a fixed pool (`available_named_barrier_ids_`,
  IDs 14..8), falling back to mbarriers when the pool is exhausted.

The result is that resource pressure is not minimized, and in the GCU case
over-allocation manifests as hardware "SIP asserts" when init/destroy cycles
exceed the device's virtual-channel budget.

The compiler already has the machinery to do better: `LivenessAnalyzer`
(`lib/liveness_analysis.{hpp,cpp}`) computes live ranges for buffers *and*
futures, and `MemReuse` (`lib/mem_reuse.cpp`) already turns buffer ranges into
an interference-graph coloring for scratch memory. This design extends the
same model to DMA resources.

## Background: what the hardware actually forces us to allocate

A DMA resource, at the hardware level, is a **completion / ordering slot**. Its
lifetime is always the same shape:

```
acquire at async issue  ->  release at wait
```

The target-specific instantiations are:

### GCU (TOPS)

| Resource | Budget source | Lifetime | Handle in Choreo |
|---|---|---|---|
| CDTE VC | `cdte_vc_count[4]` in `tops_amos_sip_interface` (`__tops_implicit_params.h`) | DMA issue -> wait | the `future.ctx` (DTE context) |
| SDTE VC | `sdte_vc_count` + runtime `sdte_bitset` | issue -> wait | the `future.ctx` |
| EDTE VC | `__TOPS_EDTE_ENGINE_COUNT` (gcu400+) | issue -> wait | the `future.ctx` |
| SIP mailbox VC | `sip_mbx_vc_count`, `mailbox_bitset_*`, `mbx_unique_id` (1..127) | bound to a DTE's completion, freed at `event.wait()` | the event's bound mailbox |

Two facts from `__tops_dte.h` and `__tops_event.h` shape the model:

1. The CDTE VC is derived from the DTE context's **memory address**
   (`engine_id = id % __TOPS_CDTE_ENGINE_COUNT`). Instantiating a DTE context
   *is* consuming a VC. So the future's live range *is* the VC's live range.
2. `event::wait()` "blocks ... for the SIP mailbox bound to it". The DTE
   context and its SIP mailbox are allocated as a **pair**; the future/event
   holds a `ctx`, and waiting resolves to the mailbox. We do not allocate the
   mailbox separately.

Engine counts: `__TOPS_CDTE_ENGINE_COUNT` is 4 (gcu200/210) and 1 (gcu300+);
`__TOPS_EDTE_ENGINE_COUNT` is 1 (gcu400+).

### GPU (NVIDIA)

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
| `FUTURE` | GCU DTE VC (+ mailbox), GPU TMA mbarrier | fully tracked | **new** (this design) |
| `EVENT` | GPU named barrier, scalar mbarrier, GCU gsync/sync-point | **partial** (use at `Wait`, no def) | **new** (this design) |
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

The allocator emits, per device function:

- GCU: `future -> dte_pool slot`; `dte_pool_size = colors`.
- GPU: `event -> named_barrier_id`; fall back to scalar mbarrier when
  `colors > available IDs` (7 today).

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
   iterations. The current `dte_pool_slots` logic already special-cases this;
   the new allocator must make the scoping rule explicit (key intervals by
   scoped name, as `VarRanges()` already does).

4. **Event-array scalar replacement.** Event arrays (`event[N]`) are keyed by
   base symbol in liveness (`GetEventName` collapses `ElemOf` refs), so the
   whole array is a single handle. This is safe today: GCU events are plain
   `bool` arrays with no fixed pool, and GPU named-barrier lowering rejects
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

## Integration points

- `lib/liveness_analysis.{hpp,cpp}`: close the event-def gap; optionally add a
  `ResourceClass` filter over `VarRanges()`.
- `lib/mem_reuse.cpp`: extract/reuse the interval-coloring core so both the
  buffer allocator and the DMA allocator share it.
- `lib/Target/GCU/topscc_codegen.{hpp,cpp}`: replace `dte_pool_slots` /
  `waited_futures` / the `-fdte-merge` greedy logic with the precomputed
  `future -> slot` map; `dte_pool_size = colors`. Keep `-fnamed-dte` /
  `-fraw-dte` / `-fno-future` as *emission* variants over the same coloring.
- `lib/Target/GPU/cute_codegen.{hpp,cpp}`: replace the greedy
  `available_named_barrier_ids_` pop with the precomputed `event -> barrier_id`
  map, keeping the mbarrier fallback for overflow.
- `-fdma-alloc` / `-fevent-alloc` are the on/off switches (see Status).

## Option interactions

DMA resource allocation is spread across two layers of switches whose scopes
are deliberately narrow. The two liveness-driven allocator switches are
target-agnostic; the DTE-pool switches are GCU emission knobs. Keeping the
layers distinct is what prevents an "off" switch from silently doing nothing.

### The allocator switches (target-agnostic)

| Switch | Default | Resource | Consumed by |
|---|---|---|---|
| `-fdma-alloc` | `true` | `FUTURE` | GCU DTE pool only (today). GPU TMA mbarrier is not yet a consumer. |
| `-fevent-alloc=<mode>` | `simple` | `EVENT` | GPU named barriers only (today). GCU events are plain `bool` arrays with no fixed pool, so this is a no-op on GCU. |

`off` is a real, observable fallback on both switches:

- `-fdma-alloc=false` falls back to monotonic per-future slot assignment (more
  DTEs on the pooled path), asserted by the `NOMERGE` checks in
  `tests/gcu/codegen/topscc/dte-merge.co`.
- `-fevent-alloc=off` restores the monotonic `named_event_lowerings_.size()`
  slot index in `cute_codegen.cpp`.
- `-fevent-alloc=full` is currently equivalent to `simple` (it reserves the
  planned event-array scalar-replacement pass; see "Known gaps").

### The GCU DTE-pool emission switches

`--use-dte-pool` (default `true`) enables the persistent SDTE pool. Disabling
it (`--use-dte-pool=false`) routes every future to its own per-DMA CDTE
context (`choreo_topscc_ctxN`), so there is no shared slot for `-fdma-alloc`
to color and the flag becomes a no-op on GCU. This is the intended coupling:
`-fdma-alloc` only takes effect on the pooled path. The
`use_dte_pool && dma_alloc_mode` guard at the plan-consumption site in
`topscc_codegen.cpp` encodes exactly this.

| Switch | Default | Requires pool | Effect |
|---|---|---|---|
| `--use-dte-pool` | `true` | - | enable the persistent SDTE pool |
| `-fno-future` | `false` | yes (fail-fast) | do not emit `future` handles (emit raw DTE + `tops::event`) |
| `-fnamed-dte` | `false` | yes (fail-fast) | emit named `private_dte` symbols instead of a `__choreo_dte_pool__` array |
| `-fraw-dte` | `false` | implicit (pool-gated) | raw DTE refs for anonymous sync DMA; subsumed by `-fno-future` |

`-fno-future`, `-fnamed-dte`, and `-fraw-dte` are *emission variants* over the
same colored slots: they do not change which futures share a slot, only how the
slot is spelled in the generated code. `-fno-future` and `-fnamed-dte` are
guarded by a `choreo_unreachable` that fails fast when requested without
`--use-dte-pool`, because neither is meaningful outside the pooled path.

### Why no "off" switch is a silent no-op

- `-fdma-alloc=false` (pooled path) yields strictly more DTEs than the default,
  asserted by the `MERGE`/`NOMERGE` differential in `dte-merge.co`.
- `--use-dte-pool=false` yields per-DMA CDTE contexts, asserted by the
  `POOL`/`NOPOOL` differential in `tests/gcu/codegen/topscc/dte_pool.co`.
- `-fevent-alloc=off` restores the monotonic named-barrier slot index on GPU.
- `-fno-future` / `-fnamed-dte` fail fast without the pool instead of silently
  doing nothing.

## Non-goals

- Allocating registers, tensor memory, or rodata in Choreo (backend-owned).
- Changing `SITE`-scoped resources (copy atoms, TMA descriptors); their current
  per-site allocation is already minimal and correct.
- Emitting the mailbox independently of its DTE context; the pair is allocated
  as a unit.

## Open questions

- Should the allocator run once per device function in a dedicated pass (like
  GPU's `DMAPlan`), or inline during codegen over `VarRanges()`?
- What is the exact threshold for GPU named-barrier fallback per arch (7 is
  hard-coded today for sm_90+; sm_70/80 may differ)?
- Is `SITE`-scoped mbarrier sharing (mutually-exclusive DMA sites collapsing
  onto one mbarrier) worth a future `SiteRanges()` keyed by `stmt2id[&dma]`?
  Deferred: the win is narrow and mbarrier phase state complicates sharing.

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
shared / global) plus finite hardware resources (DTE virtual channels, named
barriers, mbarriers, copy atoms). A correct model must treat each storage level
and each resource pool as a distinct ordering and aliasing domain rather than
one flat memory. Relaxing the current conservative fences is future work, and
any relaxation must be gated on a per-level, per-resource model -- not a
single-space side-effect annotation.
