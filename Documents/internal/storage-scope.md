# Choreo Storage Scope Model

This document defines Choreo's memory-hierarchy naming model and pins down the
semantics of the three on-chip storage tiers. It is the reference for the
`shared<group>` (group-shared) feature and supersedes the earlier "per-lane DMA
fix" that papered over the mismatch between Choreo `local` semantics and the
gcu400 `__local__` address space.

## 1. The three-tier on-chip ladder

Choreo programs run inside a nested parallel structure. On-chip buffers are
declared with one of three storage keywords that name the *scope* the buffer is
visible within:

| Storage keyword | Scope ("visible within ...") | Intuition |
|-----------------|------------------------------|-----------|
| `shared`        | a `BLOCK`                    | block-shared scratchpad |
| `shared<group>` | a `GROUP`                    | shared by the threads of one group |
| `local`         | a `THREAD`                   | private to one thread |

The wording rule that keeps the model consistent:

> The argument to `shared<...>` and the meaning of `local` always name a
> **scope** ("within X"), never a **sharer** ("between threads").

So `shared<group>` means "shared *within* a group" (one copy per group), not
"shared *between* groups". `local` means "local *to* a thread" (one copy per
thread). `shared` (bare) means "shared *within* a block".

## 2. Why `shared<group>` exists

On targets with a GROUP level between BLOCK and THREAD (gcu400 and later), the
on-chip hierarchy has one more rung than the two keywords `shared` and `local`
can express:

- `shared`  = block-shared (DSM / `__shared__`).
- `local`   = per-THREAD (per-lane) private.
- `shared<group>` = per-GROUP, shared by the THREADs (lanes) of one group.

Without a group tier, a `local` buffer is silently re-interpreted as
group-shared, which is the root cause of the "stale" per-lane workaround
described in Section 6.

On targets with no GROUP level (gcu300, GPU), `shared<group>` is rejected: the
group tier does not exist.

## 3. Vendor mapping (gcu400)

Choreo levels map onto gcu400 hardware as documented in `AGENTS.md`:

| Choreo level | Hardware unit | Builtin |
|--------------|---------------|---------|
| `BLOCK`      | cluster / grid | `__tops_bid_*()` |
| `GROUP`      | thread          | `__tops_tid_*()` |
| `THREAD`     | SIMT lane | `__tops_stid_*()` |

One GROUP runs four THREADs (SIMT lanes) in lockstep.

The storage keywords lower to vendor address spaces and DTE contexts as follows:

| Choreo storage | gcu400 declaration modifier | gcu400 DTE context | mdspan space | sync | fence |
|----------------|-----------------------------|--------------------|--------------|------|-------|
| `shared`       | `__shared__`                | `shared_dte`       | `tops::Shared` | `__syncthreads()` | `L2_MEM` |
| `shared<group>`| `__local__ __valigned__`    | `local_dte`        | `tops::Local`  | `__syncsubthreads()` | `L1_VDMEM` |
| `local`        | `__local__ __private__ __valigned__` | `private_dte` | `tops::Private` | none (per-lane) | `L1_VDMEM` |

Fence domain note: `shared<group>` and `local` both live in VDMEM and both use
the `L1_VDMEM` fence domain (`tcle::FenceType::L1_VDMEM`, with `_STORE`/`_LOAD`
release/acquire variants). gcu400 supports this domain today — `tests/gcu/.../sync.co`
(`CHECK-GCU4`) and `dma-fence.co` already emit `L1_VDMEM` and `L1_VDMEM_STORE`.

Notes:

- The `tops` mdspan `AddrSpace` enum is `{ Global, Shared, Private, Local }`.
  Choreo maps `local` -> `tops::Private` (per-lane) and `shared<group>` ->
  `tops::Local` (per-group). This is the "name-shift by one" relative to the
  CUDA mental model: Choreo `local` is the *private* lane address space.
- All three gcu400 DTE contexts are CDTE (`DTEType::CDTE`). SDTE is a
  gcu300-only concept; gcu300's `private_dte` is SDTE, while gcu400 `local_dte`,
  `private_dte`, and `shared_dte` are all CDTE.
- `private_dte` does per-lane **partition** (not replication) via
  `id = static_id * lane_count + lane_id`, so each lane owns its own
  slice. This is why per-lane `local` buffers need no manual replication.

## 4. Allocation model

There are exactly two on-chip allocation pools:

| Pool | Members | Replicated? | Driven by |
|------|---------|-------------|-----------|
| DSM (shared) | `shared` | no | host (`extern __shared__` + `shared_spm_size` at launch) |
| VDMEM (local) | `local`, `shared<group>` | yes, `replicas_per_pool` per GROUP | device, static |

`local` and `shared<group>` live in the **same** VDMEM pool and the **same**
interference graph. They must NOT be treated as disjoint live ranges, because a
group-shared buffer and a thread-private buffer can legitimately share the same
physical VDMEM region at different points in time (they are never live
simultaneously across the same region when the graph allows it). Merging them
into one heap is what keeps the allocator correct.

`shared` stays in its own host-driven pool, exactly as today.

## 5. Budget model

The joint on-chip budget check (compile time and runtime) becomes:

    LOCAL elems * replicas_per_pool * max_group_dim
  + GROUP_SHARED elems * replicas_per_pool
  + SHARED elems
  <= pool_bytes

Where `GetLocalSharedPool()` returns `{ aliased = true, replicas_per_pool = 8,
pool_bytes = 7 MB }` and `GetMaxGroupDim()` returns 4 for gcu400 (8 GROUPs x
896 KB VDMEM, 4 THREADs per GROUP). A `local` buffer is private to a THREAD
(lane), so it is replicated `max_group_dim` times per GROUP; a
`shared<group>` buffer has one copy per GROUP. Per-GROUP VDMEM capacity is
`GetMemCapacity(Storage::LOCAL) == GetMemCapacity(Storage::GROUP_SHARED) ==
0xE0000` (896 KB).

## 6. What the "stale" fix did and why it is removed

The previous approach treated `local` as group-shared and emulated per-lane
behavior at the codegen level:

- `DMATypeSTR` switched `Storage::LOCAL` between `local_dte` (BLOCK/GROUP
  level) and `private_dte` (THREAD level) based on a `block_level` flag.
- `HandleSharedLocal` tagged a THREAD-level DMA result buffer with a
  `thread_replicate` note, enlarged the scratchpad by the THREAD count, and
  offset each THREAD by its linear index.

This conflated two distinct scopes in one `Storage` value and encoded the
distinction in codegen special cases. It is replaced by the proper
`Storage::GROUP_SHARED` category:

- `shared<group>` carries the group-shared semantics natively (`__local__` +
  `local_dte`), always, at any parallel level.
- `local` carries per-lane semantics natively (`__private__` + `private_dte`),
  always, at any parallel level.

The `block_level` switching, the `thread_replicate` note, the enlarged
scratchpad, and the per-THREAD offset arithmetic are all deleted.
