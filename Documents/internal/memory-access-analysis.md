# Design: A Unified Memory-Access Analysis for Reuse, Allocation, and Synchronization

## Status

Proposal. No implementation yet. This document frames a single analysis that
would subsume three currently-separate compiler concerns. It reframes DMA
completion-slot allocation ("output 2" below) as one instance of this analysis
and adds a third output: automatic fence/barrier insertion. A paper is not a
requirement; if the analysis turns out to be precise and minimal, it may be a
natural write-up.

## Motivation

Choreo compiles a high-level DMA EDSL to heterogeneous hardware (NVIDIA GPUs,
and an internal accelerator). Three distinct things are currently computed by
three disjoint, mostly ad hoc mechanisms:

1. **Scratch memory reuse** -- `MemReuse` (`lib/mem_reuse.cpp`) colors buffer
   live ranges into a shared scratch-pad memory (SPM). This is the one
   mechanism that is a real interval analysis.
2. **DMA completion-slot allocation** -- GCU assigns fixed DTE-pool slots
   (`--use-dte-pool` + `-fdma-alloc`); GPU hands out TMA mbarriers / copy
   atoms by monotonic counter and named-barrier IDs by popping a fixed pool.
3. **Synchronization / fence placement** -- there is no automatic placement.
   Fences and barriers are explicit source constructs (`Fence`, `Barrier`,
   `Trigger`) that each target merely *lowers* (`cute_codegen.cpp:7491`,
   `topscc_codegen.cpp` fence visit). Some proxy fences are hard-coded inline at
   TMA/cp.async sites.

All three are, in fact, the **same problem viewed at different granularities**:
given the accesses to a memory location, decide how much storage to reserve and
what ordering to enforce. This document proposes computing all three from one
analysis.

## Background: the three mechanisms today

### Memory reuse (the model to generalize)

`LivenessAnalyzer` (`lib/liveness_analysis.{hpp,cpp}`) computes, per statement:

- `use` and `def` sets, plus bindings/aliases (`AddUse`/`AddDef`/`AddBinding`);
- per-variable live ranges in `VarRanges()`;
- a signal-aware happens-before graph in `HBGraphs()` (SALA).

`MemReuse::Initialize` reads `VarRanges()` and builds, per device function, a
list of `Buffer{size, ranges, buffer_id}`. `MemReuse::ProtoType` /
`AnalyzeMemOffset` then builds an interference graph (`CanOverlap`, optionally
refined by `HBGraphs()`) and colors buffers into an SPM offset layout.

Two facts are important for the generalization:

- `VarRanges()` already contains ranges for **buffers and futures** (futures get
  `AddDef(..., is_buffer_or_future=true)` in `Visit(AST::DMA)`).
- `MemAnalyzer` already knows about **events** (`event_vars`), but excludes them
  from reuse because shared events carry `__volatile__`.

### Completion-slot allocation (ad hoc)

The future/event is a handle to a finite hardware completion object (DTE VC on
GCU, mbarrier / named barrier on GPU), and its live range is exactly the
handle's `issue -> wait` range. The current allocation ignores `VarRanges()`
and uses counters / greedy pools.

### Fence placement (absent)

The fence/barrier AST nodes (`lib/ast.hpp:3796` `Barrier`, `:3814` `Fence`,
`:3629` `Trigger`) carry only a `ParallelLevel` (scope) and a `Storage` (memory
level). They are lowered, not derived. Separately, `cute_codegen.cpp` emits
proxy fences *inline* at specific sites because Hopper async transfers write
through a different proxy than normal loads:

- `cde::fence_proxy_async_shared_cta()` (`cute_codegen.cpp:1992, 4897`),
- `cp_async_bulk_wait_group_read<0>()` (`:5053`),
- `warpgroup_fence_operand(...)` (`:494`) for WGMMA accumulators.

These inline fences are the strongest evidence that GPU needs *proxy-aware*
ordering that the current explicit-fence model does not express.

## The unified model

### One graph, two edge kinds

Model a device function as a graph whose nodes are **memory accesses** and whose
edges are of two kinds:

1. **Allocation-conflict edges.** Two accesses conflict if they touch the same
   *allocatable resource* with overlapping liveness. Resources are: scratch
   buffers (bytes), and completion slots (DTE VC / mbarrier / named barrier).
   These edges drive outputs 1 and 2.

2. **Ordering-hazard edges.** A `write -> read` (or `write -> write`,
   `read -> write`) on one location across an async boundary requires an
   ordering edge. These edges drive output 3.

### The access tuple

An access is more than a buffer name. To place a fence correctly we must know:

```
Access = ( direction: read | write
         , agent:      thread | warpgroup | CTA | async-engine
         , level:      register | local | shared | global
         , proxy:      generic | async        (GPU; GCU uses L1/L2/L3 instead)
         , scope:      thread | group | block | device )
```

`direction` is the currently-missing dimension: `LivenessAnalyzer` records
`use`/`def` but not whether a DMA `use` is a read (src) or a write (dst). The
header comment is explicit that src/dst are both "use".

### SALA as the precision layer

`HBGraph` phases (`HBPhase`) already record `buffers_accessed`, `is_async`,
`multi_instance`, and `signal_in`/`signal_out`, and expose `CanOverlap`. SALA is
the mechanism that lets us prove two accesses *cannot* overlap and therefore:

- share a completion slot (output 2), and
- skip a fence that would otherwise be conservatively required (output 3).

The proposed extension adds `direction` (and, where relevant, `agent`/`proxy`)
to the phase access records so both the allocator and the fence placer can
exploit the same precision.

## Three outputs

### Output 1 -- scratch memory reuse (exists)

Unchanged in behavior; re-derived as "color allocation-conflict edges on byte
resources." `MemReuse` becomes a client of the unified graph.

### Output 2 -- completion-slot allocation

Color the future/event handles over the finite hardware slot universe. Same
interval machinery as output 1, different resource class.

### Output 3 -- fence/barrier insertion (new)

For each ordering-hazard edge `A -> B` where `A` and `B` execute on different
agents (or through different proxies), emit the minimal ordering operation:

- **completion** (wait the future / mbarrier / event) if the producer is async;
- **visibility** (fence) if the agents differ but share a memory level;
- **proxy** fence (`fence_proxy_async_*`) if the producer wrote via the async
  proxy and the consumer reads the generic proxy;
- **scope** selection (`__threadfence_block` vs `__threadfence`) from the
  consumer's level.

The target-specific part is a *fence lattice*:

- GPU: generic vs async proxy, CTA vs device scope.
- GCU (anonymized in any public write-up): three memory levels
  (`tcle::fence<FenceType::L1_VDMEM|L2_MEM|L3_MEM>`).

The placement pass consults the lattice to pick the cheapest fence that
dominates every hazard edge, and elides fences whose hazards SALA proves
unreachable.

## Soundness and minimality

- **Soundness**: a fence is required for every hazard edge that survives SALA's
  non-overlap proof. No manual fence is silently removed unless the analysis can
  prove its hazards are covered by an earlier dominating fence or are
  unreachable.
- **Minimality**: the goal is the fewest fences / smallest slot pool, not just
  correctness. This is the differentiator versus "emit fences everywhere."

The explicit source fences (`sync.fence`, `sync.barrier`, `trigger`) remain as
*assertions* / *hints*: the compiler must at least preserve them, and may warn
if an explicit fence is provably redundant.

## What is genuinely new vs. incremental

Honest accounting, to decide later whether this is paper material:

- Interval coloring for memory (output 1) -- known, done.
- Interval coloring for completion slots (output 2) -- "register allocation on
  mbarriers/DTE VCs," incremental but with a non-obvious hardware fact (VC
  derived from context address) on the GCU side.
- **Proxy + scope + completion-aware fence insertion driven by async
  producer/consumer analysis (output 3)** -- the most open of the three. Hopper
  async-proxy fences are recent and under-addressed in compiler literature, and
  "minimal, signal-aware fence placement" is not something standard compilers
  do today (they place fences for correctness, not minimally, and often treat
  the async proxy as an opaque hazard).
- **The unification** -- one analysis yielding storage layout, slot assignment,
  and synchronization -- is the conceptual claim, not the algorithmic one.

The strongest potential paper contribution is the *directed, agent/proxy-aware
extension of SALA* and its use for minimal fence placement.

## Non-goals

- Register allocation, tensor-memory, or rodata placement (owned by the backend
  compiler).
- Re-deriving memory-consistency theory; we consume the hardware's fence
  primitives, we do not model a memory model from scratch.
- Removing the ability to write explicit fences/barriers.

## Open questions

1. Does the `HBGraph` need a full `direction` field, or can direction be
   recovered at the DMA node (`FromSymbol` = read, `ToSymbol` = write) and
   attached to the existing `buffers_accessed` records?
2. For GPU, is the async-proxy hazard better modeled as a `proxy` dimension on
   the access tuple, or as a special fence kind emitted whenever a TMA/cp.async
   site is a producer?
3. Should fence placement run as a dedicated pass (like GPU `DMAPlan`), or
   inline during codegen over the graph, mirroring how `MemReuse` runs before
   codegen?
4. What is the fence lattice for GCU, and can it be expressed behind a
   `Target` virtual (so the placement pass is target-agnostic)?
5. Interaction with `Trigger` (cluster-scope event publication) -- does the
   hazard model need to treat cluster-scope events as an additional agent?

## Relationship to other documents

- `Documents/Developer/memory-reuse.md` -- documents output 1 as shipped.
- `Documents/design/cross-wg-barriers.md` -- the named-barrier pool is one of
  the finite resources that output 2 allocates, and one of the agents that
  output 3 orders against.
