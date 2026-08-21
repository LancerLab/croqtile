# Design: Liveness-Driven DMA Resource Allocation

## Status

Proposal. No implementation yet. This document defines the problem, the
resource taxonomy, and the allocation model. The implementation plan is a
separate follow-up.

## Motivation

Choreo programs DMA transfers that consume finite hardware resources on the
target device. Today these resources are allocated ad hoc, without real
liveness analysis:

- GCU (`lib/Target/GCU/topscc_codegen.cpp`) assigns each async DMA a fixed
  DTE-pool slot (`--use-dte-pool`, default on). `-fdte-merge` is a single-pass
  greedy approximation that reuses a *waited* future's slot, but it is not a
  real interval analysis and it is off by default.
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

1. **Event defs.** `LivenessAnalyzer::Visit(AST::DMA)` (`liveness_analysis.cpp:1337`)
   records a `def` for the future but *not* for `dma->Event()`. `Visit(AST::Wait)`
   (`:1421`) records a `use` for the event. As a result events have a use but no
   def and never enter `var_ranges`. Add `AddDef` for the event at the DMA (and
   at event decl/assign sites) so `EVENT` handles get ranges symmetric to
   futures.

2. **SALA binding for events.** Futures are added to the `HBGraph` via
   `AddBinding`. Events need the same so the allocator does not over-serialize
   GPU events that provably cannot overlap.

3. **Loop-carried reuse.** A future defined and waited inside the same loop has
   a self-overlapping interval and cannot share a slot with itself across
   iterations. The current `dte_pool_slots` logic already special-cases this;
   the new allocator must make the scoping rule explicit (key intervals by
   scoped name, as `VarRanges()` already does).

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
