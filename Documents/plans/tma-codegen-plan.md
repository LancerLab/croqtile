# TMA Codegen Overhaul Plan

Date: 2026-06-11
Status: APPROVED -- all design questions resolved, ready to implement

------------------------------------------------------------------------

## 1. Problem Statement

The current TMA + barrier codegen in `cute_codegen.cpp` grew organically
through multiple iterations, resulting in:

1. **Wrong instruction ordering**: `cde::cp_async_bulk_tensor` (the TMA
   instruction) is emitted BEFORE `arrive_and_expect_tx`. Hardware
   requires the mbarrier expected-tx count to be registered before the
   TMA begins writing. Reference kernels (a2.cu, TileLang) do it right.

2. **Fragile barrier type punning**: Raw mbar events are
   `ClusterTransactionBarrier*`, but `cde::cp_async_bulk_tensor` wants
   `cuda::barrier<thread_scope_block>&`. The generated code uses
   `reinterpret_cast` -- works by accident on identical 8-byte layout
   but couples two incompatible ABIs.

3. **Scattered ad-hoc logic**: TMA tx-byte tracking, barrier selection,
   phase variables, trigger suppression, and arrival counting are
   spread across 6+ code paths controlled by `--ptx-barrier`,
   `--event-arrive-tx`, `IsWarpSpecRawMbar()`, `raw_mbar_tma_arrived_`,
   `recent_tma_tx_bytes`, and `event_arrive_tx_events_`. Each is a
   band-aid for a different scenario.

4. **No unified plan**: DMAPlan knows `use_tma=true` but nothing about
   barriers, tx bytes, event binding, or protocol. All protocol
   decisions are made inline in CuteCodeGen.

------------------------------------------------------------------------

## 2. Design Decisions (Confirmed)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **PTX mbarrier everywhere** -- drop `cuda::barrier` for TMA events entirely | Single barrier backend; matches TileLang/FA3; lower overhead; eliminates reinterpret_cast |
| D2 | **Index-derived phase** -- compute phase from loop induction variable, not local `__phase` vars | Matches a2.cu `((bn & 3) >> 1)` and TileLang; fewer registers; no per-event state |
| D3 | **Remove `--ptx-barrier` flag** -- PTX mbarrier is the only path | Simplification; flag was "obsolete" per user |
| D4 | **Remove `--event-arrive-tx` flag** -- auto-compile arrive-tx from TMA/event analysis | No user choice needed; compiler determines protocol |
| D5 | **Keep `trigger` after `tma.copy.async` mandatory in .co source** -- serves as semantic validation checkpoint; codegen emits nothing for TMA-bound triggers | Source-level safety (compiler can warn on missing trigger); codegen handles protocol entirely at DMA node; trigger populates init count analysis |
| D6 | **Use PTX TMA load helpers for all ranks** -- 2D and 3D variants in `choreo_cute.h` | Avoids `cde::` + reinterpret_cast; direct PTX gives full control |

------------------------------------------------------------------------

## 3. Architecture Overview

```
     DMAPlan pass (new TMA stage)
            |
            v
   DMALoweringDecision.tma_protocol
            |
            +-- barrier_kind (PTX_MBAR)
            +-- tx_bytes (precomputed)
            +-- event_base_name
            +-- effective_rank (after swizzle split)
            +-- is_multicast
            +-- bound_trigger_ast_id (links trigger -> DMA)
            |
            v
   CuteCodeGen::Visit(DMA)
            |
            |  if tma_protocol.event_managed:
            |    1. arrive_and_expect_tx(tx_bytes)
            |    2. tma_load_Nd_mbarrier(dst, map, bar, coords...)
            |    3. (no tx_bytes pushed to deque)
            |
            v
   CuteCodeGen::Visit(Trigger)
            |
            |  if trigger is TMA-bound: no-op (DMA already did arrival)
            |  if trigger is consumer-release: .arrive()
            |
            v
   CuteCodeGen::Visit(Wait)
            |
            |  mbarrier_wait_parity(&bar, phase)
            |  phase computed from loop index (not __phase var)
```

------------------------------------------------------------------------

## 4. Implementation Phases

### Phase 1: Extend DMALoweringDecision (dma_plan.hpp)

Add `TMAProtocol` struct to `DMALoweringDecision`:

```cpp
struct TMAProtocol {
  bool event_managed = false;       // tma.copy.async<event>
  int tx_bytes = 0;                 // tile bytes for arrive_and_expect_tx
  std::string event_base_name;      // "kf", "vf", "qf"
  int effective_rank = 0;           // 2 or 3 (after swizzle split)
  bool is_multicast = false;        // cluster multicast path
  bool is_cluster_scope = false;    // shared::cluster vs shared::cta
};
```

No barrier kind field needed -- PTX mbarrier is the only path (D1).

### Phase 2: TMA Protocol Analysis (dma_plan.cpp)

In `DMAPlan::Visit(AST::DMA&)`, after existing TMA validation:

1. Detect event-managed: `n.HasEvent() && n.IsTMA() && n.is_async`
2. Compute `tx_bytes`: `box.ElementCount() * sizeof(elem_type)`
3. Extract `event_base_name` from event expression AST
4. Set `effective_rank` (check `tma_inner_splits_` from codegen prepare
   -- may need to hoist this analysis into DMAPlan or pass it forward)
5. Detect cluster/multicast from launch config

### Phase 3: TMA-Trigger Binding (dma_plan.cpp or new mini-pass)

Build a mapping: `trigger AST node -> bound DMA AST node`

Algorithm:
- Walk AST in source order within each `inthreads` scope
- When visiting `tma.copy.async<event>`: record
  `pending[event_name] = &dma_node`
- When visiting `trigger event_name`:
  - If `pending[event_name]` exists: mark trigger as TMA-bound
    (store in static map: `tma_bound_triggers_[trigger_ast_id] = dma_ast`)
  - Clear `pending[event_name]`
  - If not in pending: this is a consumer-release trigger (no binding)

This can be a post-processing step after DMAPlan's main visitor, or
a lightweight AST walk in DMAPlan::Visit(ParallelBy) before descending.

#### Init Count Derivation: Full vs Empty Events

Two event roles, each with a deterministic init count:

| Event Role | Examples | Who signals | Init count | How determined |
|------------|---------|-------------|------------|----------------|
| **Full event** (TMA-fill) | `kf`, `vf`, `qf` | 1 elect thread via `arrive_and_expect_tx` | **1** | Event appears in `tma.copy.async<event>` |
| **Empty event** (signal/release) | `ke`, `ve` | All consumer threads via `.arrive()` | **num_consumers** | Event only used in `trigger` (no TMA binding) |

**Auto-detection algorithm** (in DMAPlan or codegen_prepare):
1. Scan all DMA nodes: if `tma.copy.async<ev>` found -> mark `ev` as
   **full event** -> `init(1)`
2. Remaining events with triggers but no TMA binding -> **empty event**
   -> `init(trigger_thread_count)` from `GetEventTriggerParticipation`

This eliminates the heuristic `(blockDim.x - 128) + 1` fallback.

**Robustness**: As a safety net, also record event trigger usage from
the DMA node itself in `codegen_prepare.hpp` (when visiting an
event-managed TMA). This way even if trigger is accidentally missing,
init count is still correct. If both TMA and trigger record usage,
the masks are OR'd (same thread -> still 1).

### Phase 4: Event Declaration and Init Codegen

**Barrier declaration** (for TMA events in warp-spec):

Generated code (current, keep but simplify):
```cpp
__shared__ __align__(8) uint64_t kf__mem[2];
```

Init: single thread inits with count derived from event role:
- **Full events** (`kf`, `vf`, `qf`): `init(1)` -- 1 elect thread
  does arrive_and_expect_tx; TMA hardware completes the transaction
- **Empty events** (`ke`, `ve`): `init(num_consumer_threads)` -- all
  consumer threads call `.arrive()` to signal buffer release

Event role determined automatically by checking if the event appears
in a `tma.copy.async<event>` (full) or only in `trigger` (empty).
See Phase 3 for the auto-detection algorithm.

**Phase variable elimination (D2)**:

Instead of:
```cpp
int kf__phase[2] = {0, 0};
```

Compute inline from loop induction variable. For double-buffer
(STAGES=2):
```cpp
int phase = (loop_idx & 3) >> 1;  // == ((loop_idx / STAGES) & 1)
```

This requires knowing which loop variable drives the pipeline stage.
DMAPlan or CuteCodeGen can derive this from the `stage = bn % STAGES`
pattern in the producer loop.

**Implementation approach for phase derivation:**

Option A (simple): Keep __phase variables but compute from loop index
```cpp
int kf__phase = (bn & 3) >> 1;   // recomputed each iteration
```

Option B (optimal): Inline the expression directly into wait/arrive
```cpp
mbar.wait((bn & 3) >> 1);        // no phase variable at all
```

Recommend Option B: matches a2.cu exactly, zero register overhead.

**Phase derivation from stage variable (confirmed by user Q2)**:
The `.co` source always has `stage = id % STAGES`. Phase can be
derived from this:
- For STAGES=2: `phase = (id & 3) >> 1` or equivalently
  `phase = (stage_iteration_count & 1)` where the iteration count
  tracks how many times we've cycled through all stages.
- More generally: `phase = ((id / STAGES) & 1)`

For the initial pipeline priming (iteration 0, no loop yet), phase is 0.

### Phase 5: TMA Load Codegen (Visit(DMA))

For event-managed TMA loads (`tma_protocol.event_managed == true`):

```
// Step 1: arrive_and_expect_tx BEFORE TMA
//   Emitted inside single-thread guard (elect thread only)
<event_expr>.arrive_and_expect_tx(<tx_bytes>);

// Step 2: PTX TMA load with mbarrier
choreo::tma_load_<rank>d_shared_cta_global_mbarrier(
    (void*)dst, (const void*)&tensor_map,
    (uint64_t*)&<event_expr>,
    coord0, coord1 [, coord2]);
```

Key change: `arrive_and_expect_tx` is emitted AT the DMA visit, not
deferred to trigger. The barrier argument is `(uint64_t*)&event` -- a
raw pointer to the mbarrier, no type punning.

**No `recent_tma_tx_bytes` push** -- the DMA node owns its tx bytes.

**Trigger is no-op**: The DMA node self-contains the complete mbarrier
protocol (arrive + expect_tx + TMA load). The mandatory `trigger`
after the TMA generates zero instructions -- it exists only as a
source-level semantic checkpoint for validation and init count
derivation (see Phase 3).

For non-event TMA loads (future-based): keep existing TMAAtom path
for now (separate cleanup later).

### Phase 6: PTX TMA Load Helpers (choreo_cute.h)

**Keep**:
- `tma_to_shared_u32` -- address conversion utility
- `tma_mbarrier_init` -- PTX mbarrier init
- `tma_mbarrier_wait_parity` -- CTA-scope phase-aware wait
- `tma_load_2d_shared_cta_global_mbarrier` -- 2D CTA-scope load
- `tma_load_2d_shared_cluster_global_mbarrier` -- 2D cluster-scope load
- `tma_load_2d_shared_cluster_global_mbarrier_multicast` -- multicast
- `tma_mbarrier_arrive_cluster` -- cluster remote arrive
- `tma_mbarrier_wait_parity_cluster` -- cluster-scope wait
- `tma_cluster_rank`, `tma_cluster_dim`, `tma_cluster_sync` -- cluster
  utility

**Add**:
- `tma_load_3d_shared_cta_global_mbarrier` -- 3D CTA-scope load (was
  removed; re-add with correct implementation)

**Review and potentially remove**:
- `tma_mbarrier_expect_tx` -- was for non-arrive expect-tx; still used
  in future/TMAAtom path. Keep for now, mark for future removal.
- `tma_mbarrier_expect_tx_noarrive` -- currently unused in codegen.
  **Remove**.
- `tma_mbarrier_arrive` (non-cluster) -- check if used. If only in
  cluster path, may be redundant with Cutlass `.arrive()`.

**CTA vs Cluster mbarrier -- key difference**:

| Aspect | CTA scope | Cluster scope |
|--------|-----------|---------------|
| PTX qualifier | `shared::cta` | `shared::cluster` |
| Address space | CTA-local shared mem | Cluster-wide shared mem (DSMEM) |
| TMA instruction | `cp.async.bulk.tensor.Nd.shared::cta.global` | `cp.async.bulk.tensor.Nd.shared::cluster.global` |
| Barrier init | `mbarrier.init.shared::cta.b64` | (same, but visible across cluster) |
| Barrier arrive | `mbarrier.arrive.release.cta.shared::cta` | `mbarrier.arrive.shared::cluster` with `mapa` (remote CTA) |
| Wait | `mbarrier.try_wait.parity.shared::cta` | `mbarrier.try_wait.parity.shared::cta` (local CTA waits on local copy) |
| When to use | Single-CTA kernels (most common, e.g. flash attention) | Multi-CTA cluster kernels (rare, e.g. cluster-aware collectives) |

For flash attention v2 (no cluster), **CTA scope is correct**. The
cluster variants are only needed when `launch.cluster_count > 1`.
DMAPlan can set `is_cluster_scope` based on launch config.

### Phase 7: Trigger Codegen (Visit(Trigger))

```cpp
// Lookup: is this trigger bound to a TMA?
auto* bound_dma = LookupTMABoundTrigger(trigger_ast_id);

if (bound_dma) {
  // TMA-bound trigger: no-op (semantic check only, no codegen)
  // The arrive_and_expect_tx was already emitted at DMA visit.
  // The trigger remains mandatory in .co source for validation but
  // generates no instructions.
} else {
  // Consumer-release trigger: emit raw mbarrier arrive
  ds << d_indent << event_expr << ".arrive();\n";
}
```

This replaces the entire `recent_tma_tx_bytes`-based trigger logic.

**Semantic validation** (in semacheck or codegen_prepare): If an
event-managed TMA (`tma.copy.async<ev>`) has no matching `trigger ev`
in the same scope, emit a warning. The trigger is required as a
source-level contract even though codegen ignores it.

### Phase 8: Wait Codegen (Visit(Wait))

```cpp
// Phase from loop index (D2):
std::string phase_expr = ComputePhaseFromLoopIndex(event, scope);

// Raw mbarrier wait:
ds << d_indent << "choreo::tma_mbarrier_wait_parity("
   << "(uint64_t*)&(" << event_expr << "), " << phase_expr << ");\n";
```

`ComputePhaseFromLoopIndex`:
- For events inside a pipeline loop with `stage = iter % STAGES`:
  Use `((iter & (2*STAGES - 1)) >> log2(STAGES))` xored as needed
- For one-shot events (like `qf`): phase = 0 (always first wait)
- Fallback: keep __phase variable if loop structure not recognized

### Phase 9: Remove Legacy Mechanisms

| Remove | From | Replaced by |
|--------|------|-------------|
| `recent_tma_tx_bytes` deque | `cute_codegen.hpp/cpp` | TMA protocol tx_bytes in DMALoweringDecision |
| `raw_mbar_tma_arrived_` flag | `cute_codegen.hpp/cpp` | Already removed |
| `event_arrive_tx_events_` set | `cute_codegen.hpp/cpp` | TMA-trigger binding in DMAPlan |
| `--ptx-barrier` option | `codegen_utils.cpp/hpp` | Always PTX mbar (D3) |
| `--event-arrive-tx` option | `codegen_utils.cpp/hpp` | Auto-compiled (D4) |
| `reinterpret_cast<cuda::barrier>` | codegen TMA path | PTX load helpers take `uint64_t*` |
| `cde::cp_async_bulk_tensor` for event TMA | codegen | PTX load helpers |
| `tma_mbarrier_expect_tx_noarrive` | `choreo_cute.h` | Unused |
| `__phase` local variables (per event) | codegen event decl | Index-derived phase (D2) |
| `IsWarpSpecRawMbar()` guard | codegen trigger/wait | All warpspec events are raw mbar |

### Phase 10: Non-Event TMA (future-based) -- Later

The future/TMAAtom path (`has_future`, non-event-only) is a separate
code path that currently works. Leave it unchanged in this round.
It can be unified later once the event-managed path is solid.

------------------------------------------------------------------------

## 5. File Change Summary

| File | Changes |
|------|---------|
| `lib/Target/GPU/dma_plan.hpp` | Add `TMAProtocol` to `DMALoweringDecision` |
| `lib/Target/GPU/dma_plan.cpp` | Compute TMA protocol; add trigger binding analysis |
| `lib/Target/GPU/cute_codegen.hpp` | Remove `recent_tma_tx_bytes`, `event_arrive_tx_events_`; add `tma_bound_triggers_` lookup |
| `lib/Target/GPU/cute_codegen.cpp` | Rewrite event-managed TMA load, trigger, wait, event decl/init; remove legacy flags |
| `lib/codegen_utils.cpp` | Remove `--ptx-barrier`, `--event-arrive-tx` option defs |
| `lib/codegen_utils.hpp` | Remove option externs |
| `runtime/choreo_cute.h` | Add 3D CTA-scope PTX load; remove unused helpers |

------------------------------------------------------------------------

## 6. Validation Plan

1. **Compile v2_manual_s2_1p2c_tma.co** with modified compiler
2. **Inspect generated .cu**: verify arrive_and_expect_tx before TMA,
   no reinterpret_cast, no __phase variables
3. **Diff against a2.cu**: producer loop should match pattern
4. **PTXAS warnings**: no C7514, no C7510
5. **Correctness**: run B1S4096, B2S8192 (note: pre-existing ~17%
   fail rate on B1S4096 is from .co kernel, not compiler)
6. **Performance**: benchmark against a2.cu and TileLang baselines
7. **Regression**: compile non-flash-attention TMA tests to ensure
   future-based TMA path still works

------------------------------------------------------------------------

## 7. Design Questions (All Resolved)

### Q1: 3D PTX TMA helper -- RESOLVED

Re-added with same PTX pattern as 2D helper and a2.cu. Verified
compiling with nvcc. The suspected bug was the ordering of
arrive_and_expect_tx (now fixed: arrive before TMA, not after).
Status: **Done** -- helper in choreo_cute.h, tested.

### Q2: Phase derivation -- RESOLVED

Always derive from loop index when recognizable. The `.co` source
has `stage = id % STAGES`; phase = `((id / STAGES) & 1)`.
For unrecognized patterns, keep `__phase` variable as fallback.
Consumer uses same loop range as producer, so same formula works.

### Q3: Multi-TMA-before-one-trigger -- RESOLVED

Each trigger is bound to its corresponding TMA via event name matching.
Each event has exactly one TMA source, so 1:1 binding is unambiguous.

### Q4: Non-warpspec TMA uses PTX mbarrier -- RESOLVED

Confirmed: TMAAtom always uses `uint64_t` PTX mbarrier. Simplify
TMAAtom to PTX-only (drop cuda::barrier members). Non-warpspec
codegen follows the same PTX pattern as warpspec.

For non-event TMA (TMAAtom path), keep `tma_mbarrier_expect_tx`
as the arrival mechanism -- it wraps `mbarrier.arrive.expect_tx`
which is the combined arrive+expect, equivalent to
`arrive_and_expect_tx`. This works with the PTX TMA load helpers.

### Q5: No future for TMA in warpspec -- RESOLVED

Confirmed: warpspec TMA is always event-driven. Add guard in
`claimFuture` to skip future creation for warpspec TMA.

### Q6: cuda::barrier for arch < SM90 -- RESOLVED

Keep `cuda::barrier` as fallback for arch < 90. But TMA is only
supported on SM90+ (Hopper), so TMA paths are always PTX mbarrier.
Non-TMA shared events on arch < 90 can keep cuda::barrier.

### Q7: Non-warpspec shared events -- RESOLVED

Non-warpspec shared events also use PTX mbarrier (same as warpspec).
This unifies the barrier type across all SM90+ codegen.

------------------------------------------------------------------------

## 8. Dead Code Audit Report

Comprehensive review of TMA/trigger/wait/event codegen. Items marked
REMOVE are dead or redundant; SIMPLIFY items can be consolidated.

### A. Runtime Helpers (choreo_cute.h) -- REMOVE

| Function | Reason |
|----------|--------|
| `tma_mbarrier_expect_tx_noarrive` | Zero call sites in codegen or runtime |
| `tma_mbarrier_arrive` (non-cluster, line ~303) | Zero call sites; triggers use `__choreo_Barrier::arrive()` or `cuda::barrier::arrive()` |
| `tma_mbarrier_wait_parity_cluster` (line ~275) | Zero call sites; cluster waits use CTA-scope `tma_mbarrier_wait_parity` |

### B. Runtime TMAAtom (choreo_cute.h) -- SIMPLIFY

| Item | Action |
|------|--------|
| `TMAAtom::bar` (cuda::barrier* member) | Remove -- PTX mbarrier only (D1) |
| `TMAAtom::tok` (arrival_token member) | Remove -- no token protocol |
| `TMAAtom::use_ptx_mbarrier` flag | Remove -- always true |
| `TMAAtom::IsPTXMBarrier()` | Remove -- always true |
| `TMAAtom::barrier()` / `TMAAtom::token()` | Remove -- unused after PTX-only |
| `TMAAtom::EnablePTXMBarrier()` | Replace with constructor `TMAAtom(uint64_t* bar)` |
| `future::wait_impl` dual branch | Remove cuda::barrier branch; always PTX wait |

### C. Compiler Options (codegen_utils.cpp/hpp) -- REMOVE

| Option | Reason |
|--------|--------|
| `--ptx-barrier` | PTX mbarrier is the only path (D3); flag becomes meaningless |
| `--event-arrive-tx` | Auto-compiled from TMA/event analysis (D4) |

### D. cute_codegen.cpp -- G2S TMA Load Paths -- REMOVE/SIMPLIFY

| Code Path | Lines (approx) | Reason |
|-----------|----------------|--------|
| `use_ptx_barrier_for_desc == false` branch in block TMA init | ~3653-3692 (else branch) | Only PTX init; remove cuda::barrier init + `TMAAtom{&bar}` |
| `use_ptx_tma_sync == false` + `has_future` (CuTe cde:: + barrier_arrive_tx) | ~4596-4647 | Replace with PTX load; cde:: path is dead |
| `use_ptx_tma_sync == false` + `event_only` + `raw_mbar_event_tma` (reinterpret_cast hack) | ~4596-4606 | Replace with PTX load; no more type punning |
| `use_ptx_tma_sync == false` + `!has_future && !event_only` (direct TMAAtom) | ~4609-4630 | Unreachable -- claimFuture always creates future for non-event G2S |
| Non-issuing thread `else` branch (`.arrive()` on cuda barrier) | ~4637-4647 | Only needed for cuda::barrier; PTX path uses single-thread expect_tx |
| Sync TMA wait via `atom.barrier().wait(token)` | ~4669-4683 | Replace with PTX wait |
| `future.trigger()` for TMA async | ~4658 | Runtime no-op for is_tma; remove codegen emission |

### E. cute_codegen.cpp -- Future Creation -- SIMPLIFY

| Item | Action |
|------|--------|
| `claimFuture` for TMA in warpspec mode | Add guard: warpspec TMA should never create futures; always event-only |
| `event_arrive_tx && S2G` skip in claimFuture | Remove -- S2G never creates futures anyway |
| Placeholder path (`FutureType` TMA PH) in warpspec | Add warpspec guard or assert |

### F. cute_codegen.cpp -- Event Declaration -- SIMPLIFY

| Item | Action |
|------|--------|
| `cuda::barrier<thread_scope_block>` event declaration branch | Remove for SM90+ warpspec; only raw mbar |
| `cuda::barrier::arrival_token __tok` declaration | Remove with `--event-arrive-tx` |
| `__phase` local variables | Replace with index-derived phase (D2) |
| Init count `(blockDim.x - 128) + 1` heuristic | Replace with full/empty event auto-detection |

### G. cute_codegen.cpp -- Trigger Codegen -- REMOVE/SIMPLIFY

| Item | Lines (approx) | Action |
|------|----------------|--------|
| `recent_tma_tx_bytes` deque mechanism | entire deque | Replace with DMAPlan TMA-trigger binding |
| `event_arrive_tx` scalar token path (D4) | ~7398-7410 | Remove with option |
| `event_arrive_tx_events_` set | hpp + cpp | Remove with option |
| cuda::barrier `barrier_arrive_tx` paths | ~7276-7307 | Remove; only raw mbar `arrive_and_expect_tx` or `.arrive()` |
| Non-warpspec array trigger dropping tx bytes | ~7311 | Remove; always use arrive_and_expect_tx or arrive |
| Cluster trigger not clearing `recent_tma_tx_bytes` | ~7229 | Fix or remove deque entirely |
| `conditional_tx` / `__CHOREO_GROUPX4_SINGLE__` ternary | ~7276-7296 | Remove; single-thread guard is sufficient |

### H. cute_codegen.cpp -- Wait Codegen -- REMOVE/SIMPLIFY

| Item | Action |
|------|--------|
| `cuda::barrier .wait(.arrive())` paths | Remove for SM90+; only PTX parity wait |
| `event_arrive_tx` token `.wait(std::move(__tok))` path | Remove with option |
| `BeginEventCritical`/`EndEventCritical` for shared events | Remove for SM90+; raw mbar wait needs no guard |
| `__phase` toggle `^= 1` | Replace with index-derived phase |

### I. cute_codegen.hpp -- State Variables -- REMOVE

| Variable | Replaced by |
|----------|-------------|
| `recent_tma_tx_bytes` (deque) | DMAPlan TMA protocol tx_bytes |
| `event_arrive_tx_events_` (set) | Removed with `--event-arrive-tx` |
| `raw_mbar_tma_arrived_` (bool) | Already removed |

### J. Failing Tests (need update, not rollback)

| Test | Issue |
|------|-------|
| `tests/gpu/codegen/cute/dma_event.co` | Expects `cde::cp_async_bulk_tensor_2d_global_to_shared(... full[stage])` without reinterpret_cast; needs update to expect PTX TMA load or raw mbar cast |

### K. Verified Working

| Item | Status |
|------|--------|
| `tma_load_3d_shared_cta_global_mbarrier` (re-added) | Compiles with nvcc; same PTX pattern as a2.cu |
| `tests/gpu/codegen/cute/ptx_barrier_codegen_toggle.co` | 3/3 PASS |
| `tests/gpu/codegen/cute/tma_barrier_init.co` | 1/1 PASS |
| v2_manual_s2_1p2c_tma.co compilation | PASS (no PTXAS errors) |

------------------------------------------------------------------------

## 9. Implementation Order

| Step | Description | Est. Effort |
|------|-------------|-------------|
| 1 | Add 3D PTX TMA load helper back to choreo_cute.h (test in isolation) | 30 min |
| 2 | Add TMAProtocol to DMALoweringDecision, compute in DMAPlan | 1-2 hr |
| 3 | Add TMA-trigger binding analysis | 1 hr |
| 4 | Rewrite Visit(DMA) for event-managed TMA: arrive_and_expect_tx + PTX load | 2 hr |
| 5 | Rewrite Visit(Trigger) with TMA-bound check | 1 hr |
| 6 | Implement index-derived phase in Visit(Wait) + event decl | 1-2 hr |
| 7 | Remove --ptx-barrier, --event-arrive-tx, legacy mechanisms | 1 hr |
| 8 | Remove unused helpers from choreo_cute.h | 30 min |
| 9 | Build + test + benchmark | 1-2 hr |
| **Total** | | **~10 hr** |
