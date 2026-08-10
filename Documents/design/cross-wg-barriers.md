# Design: Cross-Warpgroup Named Barriers for Choreo

## Motivation

Flash Attention on Hopper achieves higher throughput by overlapping computation
across consumer warpgroups using PTX named barriers (`bar.sync` / `bar.arrive`).
Choreo exposes named-barrier builtins for this explicit scheduler pattern. This
is separate from async operation completion, which uses futures and events.

The goal: let a fast WG start PV WGMMA while the slow WG is still in softmax,
achieving CUDA core + Tensor Core concurrency across warpgroups.

## Background

### PTX Named Barriers

Hopper provides 16 CTA-scoped named barriers (IDs 0-15). Each supports:

- `bar.sync ID, COUNT`: Arrive and wait until COUNT threads have arrived.
- `bar.arrive ID, COUNT`: Arrive without waiting (split arrive/wait).
- `bar.red.popc ID, COUNT, pred`: Arrive with reduction (not needed here).

Thread count must be a multiple of warp size (32). Barriers are CTA-wide --
any thread in the CTA can participate regardless of warpgroup.

### FA3 Scheduler Barrier Pattern

FA3 assigns one barrier per consumer WG (ring topology):

```
Consumer WG0 uses barrier_id = 1 (WarpSchedulerWG1)
Consumer WG1 uses barrier_id = 2 (WarpSchedulerWG2)
```

Each iteration:
1. WG_k finishes QK -> `bar.arrive (k+1)%N_wgs, 256` (signal next WG)
2. WG_k does softmax (no barrier involvement)
3. WG_k before PV -> `bar.sync k, 256` (wait for prev WG's arrive)
4. WG_k does PV WGMMA

The arrival count is 256 (= 2 warpgroups * 128 threads), meaning BOTH WGs must
participate. This creates a ring: each WG's PV gate requires the other WG's QK
completion signal.

### Choreo's Current State

- `shared event` publishes future completion to one or more consumers.
- Async MMA returns an operation future; the compiler derives WGMMA waits.
- Barrier ID 15 is reserved for internal GROUPx4 synchronization.
- `__bar_arrive` and `__bar_sync` provide explicit named-barrier scheduling.

## Interface

### Named-Barrier Builtins

Add two new compiler-recognized builtin functions:

```co
__bar_arrive(int barrier_id, int thread_count);
__bar_sync(int barrier_id, int thread_count);
```

These emit raw PTX and are available only in device code within `inthreads.async`
blocks. No type checking beyond integer arguments.

### Codegen Output

```cpp
// __bar_arrive(2, 256)
asm volatile("bar.arrive %0, %1;" :: "r"(2), "r"(256) : "memory");

// __bar_sync(1, 256)
asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(256) : "memory");
```

### Reserved Barrier IDs

Choreo reserves barrier IDs for internal use:
- ID 15: GROUPx4 warpspec sync (existing)

CUTLASS reserves IDs 1-7 (EpilogueBarrier, TransposeBarrier, etc.) but since
we don't use CUTLASS runtime (only vendored headers for WGMMA atoms), these
don't conflict.

Convention for user code:
- IDs 0-7: User-available for cross-WG scheduling
- ID 15: Reserved (internal warpspec sync)
- IDs 8-14: Future Choreo use

### Semantic Validation

The compiler should warn (not error) if:
- `barrier_id` is a compile-time constant == 15 (conflicts with internal sync)
- Used outside `inthreads.async` block (meaningless on host)
- `thread_count` is not a multiple of 32 (PTX requirement)

### Usage Example (Flash Attention Cross-WG Stagger)

```co
inthreads.async (p > 0) {
  cid = p - 1;  // consumer WG index: 0 or 1
  int my_barrier = cid + 1;       // barrier 1 or 2
  int next_barrier = 2 - cid;     // barrier 2 or 1

  foreach {bn} in [kv_bound] {
    // QK WGMMA
    qk = mma.row.row.async acc_s, q_shared, k_buf[stage];
    wait qk;

    // Signal: my QK is done
    __bar_arrive(next_barrier, 256);

    // Softmax on acc_s (CUDA cores, overlaps with other WG's PV)
    // ... softmax computations ...

    // Wait: other WG's QK must be done before I do PV
    __bar_sync(my_barrier, 256);

    // PV WGMMA
    pv = mma.row.col.async acc_o, acc_s_cast, v_buf[stage];
    wait pv;
  }
}
```

### Combined with IntraWG Overlap

The cross-WG stagger composes with the existing QK/PV overlap (iter036):

```co
    // Issue PV[n]
    pv = mma.row.col.async acc_o, acc_s_cast, v_buf[stage];

    __bar_arrive(next_barrier, 256);  // Signal: PV issued, QK next

    // Issue QK[n+1] while PV[n] in flight
    if (bn + 1 < kv_bound) {
      wait kvf[(bn + 1) % STAGES];
      acc_s = mma.fill.f32 0.0f;
      qk = mma.row.row.async acc_s, q_shared, k_buf[(bn+1) % STAGES];

      __bar_sync(my_barrier, 256);  // Wait for other WG before next softmax
      wait qk, pv;                  // Both PV[n] and QK[n+1] are done
    }
```

## Implementation Notes

### Phase 1: Parser + Scanner (scanner.l, parser.yy)

1. Recognize `__bar_arrive` and `__bar_sync` as builtin calls.
2. Require two integer arguments.

### Phase 2: Codegen (cute_codegen.cpp)

1. Detect `__bar_arrive` and `__bar_sync` during target code generation.
2. Emit the appropriate PTX instruction.
3. Validate the argument count.

### Phase 3: Sema (optional warnings)

1. Warn if barrier_id == 15 (reserved)
2. Warn if used in host context

### Phase 4: Test

1. Add codegen test: verify correct PTX emission
2. Add end-to-end test: simple 2-WG kernel with stagger

### Phase 5: Flash Attention Integration

1. Modify `best.co` to use cross-WG barriers
2. Benchmark against current best (452 TFLOPS)
3. Measure impact

## Alternatives Considered

### Named barrier as first-class type

```co
shared named_barrier sched[2] : 256;  // type with participation count
sched[next].arrive();
sched[my].sync();
```

Rejected for now: Adds parser complexity (new type, method syntax) for a feature
that may only be used in advanced warp-specialized kernels. The builtin approach
is simpler and equally expressive.

### Completion fanout through `shared event`

```co
shared event<named, 256> sched[2];  // named barrier mode
trigger sched[next];                // -> bar.arrive
wait sched[my];                     // -> bar.sync
```

Async completion should use an ordinary event:

```co
qk = mma.row.row.async acc_s, q_shared, k_buf[stage];
wait qk;
trigger qk_done;
```

WGMMA has no native completion-event binding, so its wait remains explicit.
The event backend supports multiple consumers after the direct trigger.
Explicit named barriers remain available for scheduler topology and fixed
participant-count control; they are not operation futures.

### Direct asm() escape

```co
__asm__("bar.arrive %0, %1;" :: "r"(id), "r"(count));
```

Rejected: Choreo has no inline assembly support. Adding general asm() is much
more work than targeted builtins.

## Expected Performance Impact

Based on FA3 vs HPC-Ops comparison:
- HPC-Ops (no overlap, no stagger): ~equivalent to our pre-iter036 baseline
- FA3 (IntraWG overlap + cross-WG stagger): achieves peak performance

Our current kernel already has IntraWG overlap (iter036). Adding cross-WG
stagger should provide an additional ~5-15% improvement by allowing one WG's
PV to run on Tensor Cores while the other WG's softmax uses CUDA FP32 cores.

However, with only 2 consumer WGs, the benefit may be limited since both WGs
tend to be in the same loop phase simultaneously (they process the same tile
sequence). The stagger is most effective when WGs drift apart due to
thread-level timing jitter.

## References

- FA3: `hopper/named_barrier.hpp`, `mainloop_fwd_sm90_tma_gmma_ws.hpp:914-930`
- CUTLASS: `cutlass/arch/barrier.h:167-307`
- PTX ISA 8.x: Section 9.7.12 "Parallel Synchronization and Communication"
- HPC-Ops: `src/attention/prefill/kernels.cuh:578` (epilogue WG sync)
