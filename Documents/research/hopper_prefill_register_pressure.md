# Register Pressure Management in Hopper Prefill MHA/GQA Kernels

Research comparing HPC-Ops (Tencent) and FlashAttention-3 (FA3) register
pressure strategies for SM90a (Hopper) prefill attention.

---

## 1. The Hardware Budget

SM90 (H100) provides **65,536 registers per SM**, shared among all threads of
all resident blocks. With `__launch_bounds__(N, 1)` (1 block/SM), the entire
register file is available to one CTA.

Key constraint: `setmaxnreg` (PTX `setmaxnreg.inc/dec.sync.aligned`) lets
warpgroups dynamically **reallocate** registers at runtime -- producer warpgroups
shrink their budget, consumers expand theirs.

---

## 2. Warpgroup Register Reallocation

### 2.1 The Pattern

Both implementations split threads into **producer** (TMA loads) and **consumer**
(MMA + softmax) roles at kernel entry:

```
if (thread_idx >= consumer_threads) {
    warpgroup_reg_dealloc<LOAD_REGS>();   // PTX setmaxnreg.dec
    // ... TMA load loop ...
} else {
    warpgroup_reg_alloc<MMA_REGS>();      // PTX setmaxnreg.inc
    // ... MMA + softmax loop ...
}
```

This is the **single most important** register pressure optimization. Without it,
all threads share a uniform register budget, wasting producer capacity.

### 2.2 Concrete Values

#### HPC-Ops

| Kernel Variant              | Producer dealloc | Consumer alloc | Block threads |
|-----------------------------|------------------|----------------|---------------|
| BF16 prefill (warp-spec)    | **32**           | **168**        | 384           |
| BF16 KV-cache (warp-spec)   | **24**           | **168**        | 384           |
| FP8 KV-cache (warp-spec)    | **24**           | **192**        | 384           |
| Multi-stage (all)           | (none)           | (none)         | 128           |

#### FA3

| NumMmaWGs | Use_TMA_KV | Producer dealloc | Consumer alloc | Block threads |
|-----------|------------|------------------|----------------|---------------|
| 1         | either     | **56**           | **256**        | 256           |
| 2         | TMA        | **24**           | **240**        | 384           |
| 2         | cp.async   | **40**           | **232**        | 384           |
| 3         | either     | **32**           | **160**        | 512           |

### 2.3 Total Register Accounting

```
Total_regs = LoadRegs x ProducerThreads + MmaRegs x ConsumerThreads
```

| Config             | Calculation              | Total  | Blocks/SM |
|--------------------|--------------------------|--------|-----------|
| HPC BF16 warp-spec | 32x128 + 168x256         | 45,056 | 1         |
| HPC FP8 warp-spec  | 24x128 + 192x256         | 52,224 | 1         |
| FA3 2WG TMA        | 24x128 + 240x256         | 64,512 | 1         |
| FA3 3WG            | 32x128 + 160x384         | 65,536 | 1 (exact) |

**Key insight:** FA3 3WG config uses **exactly** 65,536 registers -- the full SM
register file. FA3 generally pushes register budgets higher than HPC-Ops (240 vs
168 per consumer thread for 2WG).

### 2.4 Why Different Values?

The consumer register budget is dictated by:

1. **Accumulator fragment sizes** (WGMMA C registers)
2. **Number of simultaneously live accumulators** (QK scores + output + softmax)
3. **RS vs SS PV GEMM** (RS keeps P in registers)
4. **IntraWGOverlap** (overlapping QK[n] with PV[n-1] keeps both live)
5. **FP8 extras** (dual float/fp8 P buffers, byte permute temporaries)

The producer budget is dictated by:
1. **TMA vs cp.async** (TMA needs minimal state; cp.async needs pointer regs)
2. **Loop bookkeeping** (stage indices, barrier pointers, block IDs for paged KV)

---

## 3. `__launch_bounds__` Interaction

Both use `__launch_bounds__(MaxThreads, 1)`:

- **`MaxThreads`**: 256/384/512 depending on config
- **`minBlocksPerSM = 1`**: Tells compiler each SM holds at most 1 block

This is critical because `minBlocks=1` tells the compiler it can allocate up to
`65536 / MaxThreads` registers per thread **before** the dynamic `setmaxnreg`
reallocation. Without this hint, the compiler might assume 2+ blocks/SM and
cap registers lower, conflicting with the explicit `warpgroup_reg_alloc` to
168-256.

HPC-Ops multi-stage kernels (128 threads) do **not** declare launch bounds --
they rely on default compiler heuristics since no warpgroup reconfig is needed.

---

## 4. What Consumes Registers: Fragment Inventory

### 4.1 WGMMA Accumulator Fragments

Per consumer thread, the main register consumers are:

| Fragment  | WGMMA Atom              | Elements/Thread | Type  | Regs  |
|-----------|-------------------------|-----------------|-------|-------|
| `tSrS`   | QK: 64x64x16 SS (BF16)  | 32 FP32         | Accum | ~32   |
| `tOrO`   | PV: 64x128x16 RS (BF16) | 64 FP32         | Accum | ~64   |
| `tOrP`   | PV A-operand (RS mode)   | 4 uint32        | Input | ~4    |
| Q desc   | SS smem descriptor       | 1 uint64        | Input | ~2    |
| K desc   | SS smem descriptor       | 1 uint64        | Input | ~2    |
| V desc   | SS smem descriptor       | 1 uint64        | Input | ~2    |

**tOrO (output accumulator)** is the largest single register consumer at ~64
FP32 registers per thread.

### 4.2 Softmax State (Per Row)

| Variable       | Count  | Type  | Regs |
|----------------|--------|-------|------|
| `row_max`      | kNRows | FP32  | 2-4  |
| `row_sum`      | kNRows | FP32  | 2-4  |
| `scores_scale` | kNRows | FP32  | 2-4  |

Where `kNRows = 2 * (2 * kBlockM / NumMmaThreads)`. For 2WG, 128x128 tile:
kNRows = 2 * (2 * 128 / 256) = 2. So softmax adds ~6-12 registers total.

### 4.3 Peak Live Register Windows

**Without IntraWGOverlap:**
```
Peak = max(tSrS + softmax_temps, tOrO + tOrP)
     = max(32 + 12, 64 + 4)
     = 68 regs (FP32 equivalent)
```
P can reuse tSrS memory after conversion (alias via `make_tensor(tSrS.data(),...)`).

**With IntraWGOverlap (FA3):**
```
Peak = tSrS(new) + tOrP(prev) + tOrO + softmax_temps
     = 32 + 16 + 64 + 12
     = ~124 regs (FP32 equivalent)
```
Both QK scores and previous P must be live simultaneously.

### 4.4 FP8 Additional State

FP8 adds ~24 registers/thread vs BF16 (explaining HPC-Ops 192 vs 168):

| Extra State            | Purpose                         | Regs  |
|------------------------|---------------------------------|-------|
| `tAttr_fp8`            | FP8 copy of P for RS PV A-op   | ~8    |
| FP32->FP8 conversion   | `__byte_perm` shuffle temps     | ~4    |
| `tQS[kM]`, `kscale`   | Per-row Q/K descale factors     | ~4    |
| V transpose temps      | LDSM.T + byte_perm + STSM      | ~8    |

---

## 5. RS vs SS for PV GEMM: The Key Register/SMEM Trade-off

This is the most explicit register pressure design decision in both codebases.

### RS PV (Register-Side P)

- P stays in registers as WGMMA A-operand (`uint32[4]` after bf16 conversion)
- **Saves smem**: No `smem_P` buffer (saves `kBlockM x kBlockN x 2` bytes)
- **Costs registers**: P conversion buffer + RS A fragment live during PV GEMM
- Used by: HPC-Ops always; FA3 for hdim >= 96 (BF16), all FP8

### SS PV (Shared-Memory P)

- P written to smem via STSM (`SM90_U32x4_STSM_N`), then read as SS descriptor
- **Saves registers**: P dies after STSM, freeing those regs before PV GEMM
- **Costs smem**: Additional `kBlockM x kBlockN x sizeof(Element)` shared memory
- Used by: FA3 for hdim <= 96 (BF16), LargeHeadDimV (>256)

### Decision Matrix (FA3)

| Head dim | MmaPV_is_RS | Reason                                    |
|----------|-------------|-------------------------------------------|
| <= 64    | depends     | SS for large V dim (512); RS otherwise    |
| <= 96    | **false**   | Enough smem, save registers               |
| <= 128   | **true**    | Tight smem budget, RS saves critical bytes |
| <= 192   | **true**    | Same                                      |
| > 192    | **true**    | Same                                      |
| All FP8  | **true**    | Required (static_assert)                  |

**HPC-Ops always uses RS** -- simpler design, accepts higher register pressure.

---

## 6. IntraWGOverlap: Overlapping GEMM-I and GEMM-II

This is an FA3-only optimization that **doubles peak register pressure** but
hides WGMMA latency.

### How It Works

```
for each KV tile n:
  1. Issue QK GEMM for tile n         (tSrS[n] live)
  2. While QK runs, issue PV GEMM     (tOrP[n-1] + tOrO live)
     for PREVIOUS tile n-1
  3. Wait for QK[n], do softmax       (tSrS[n] -> tOrP[n])
  4. Loop back: PV[n] will overlap with QK[n+1]
```

### Register Impact

- **Without overlap**: Peak = `max(tSrS, tOrO)` because P reuses S data
- **With overlap**: Peak = `tSrS + tOrP + tOrO` -- all three live simultaneously
- FA3 AI tuning doc confirms: "with overlap peak = regs_S + regs_P + regs_O"

### When Disabled

FA3 disables IntraWGOverlap for:
- LargeHeadDimV (>256) -- not enough registers
- Some paged KV + FP8 configs at large head dims

HPC-Ops does **not** implement IntraWGOverlap at all.

---

## 7. Compiler Pragmas and Hints

### `#pragma unroll 1` on Outer Loops (Critical)

Both codebases prevent the compiler from unrolling the outer KV iteration loop:

```cpp
#pragma unroll 1
for (int itile_kv = 0; itile_kv < num_tiles_kv; ++itile_kv) { ... }
```

Without this, the compiler would duplicate the entire loop body, multiplying
live ranges for tSrS, tOrO, and all softmax temporaries. This is one of the
most impactful register pressure controls.

### `#pragma unroll` on Inner MMA Loops (Intentional)

The inner K-reduction loop (WGMMA across head dim) IS fully unrolled:

```cpp
#pragma unroll
for (int ik = 0; ik < size<2>(tQr); ++ik) {
    gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik), tSrS);
}
```

This is safe because the trip count is small (8 for hdim=128, k=16) and the
accumulator (tSrS) is the same across iterations.

FA3 adds a spill guard: if `kNumKIters > 16`, splits into two loops to avoid
compiler storing smem addresses in registers and causing spills.

### `__forceinline__` on Softmax/Utility Functions

Both codebases inline softmax functions to avoid function call overhead, at the
cost of potentially larger live ranges. This is intentional -- the softmax
computation is interleaved with WGMMA fences.

### `warpgroup_fence_operand`

Not a register hint per se, but it creates **live range boundaries** for
WGMMA operands:

```cpp
warpgroup_fence_operand(tSrS);     // mark accumulator stable
warpgroup_arrive();                  // begin WGMMA batch
for (...) gemm(..., tSrS);          // tSrS MUST stay live
warpgroup_commit_batch();            // end batch
warpgroup_wait<0>();                 // wait for completion
warpgroup_fence_operand(tSrS);     // fence again before reuse
```

Between `warpgroup_arrive` and `warpgroup_wait`, the accumulator registers
are **pinned** -- the compiler cannot spill them.

---

## 8. Fallback Strategy: Multi-Stage (No Warp Specialization)

Both codebases provide a simpler kernel for low-occupancy scenarios:

| Aspect           | Multi-stage        | Warp-spec           |
|------------------|--------------------|---------------------|
| Threads          | 128 (1 WG)         | 384 (3 WG)          |
| Register reconfig| None               | setmaxnreg          |
| Tile size        | 64x64              | 128x64 or larger    |
| Pipeline stages  | 1                  | 2                   |
| Load/compute     | Fused (1 elected)  | Dedicated producer   |
| When used        | Few tiles (< 2xSM) | Many tiles           |

The multi-stage path **avoids register reconfig entirely** by running fewer
threads with a smaller tile. The default register allocator handles everything.
The cost is loss of load/compute overlap and lower throughput per block.

HPC-Ops additionally saves smem in multi-stage by reusing the Q/K/V buffer
for output Y (`shm_y = shm_data`).

---

## 9. Shared Memory as Register Pressure Relief

### What Lives in SMEM to Reduce Registers

| Data                | Purpose                               | Both/FA3/HPC |
|---------------------|---------------------------------------|--------------|
| Q, K, V tiles       | WGMMA operands (SS mode)              | Both         |
| Barriers (mbarrier) | Pipeline synchronization              | Both         |
| Batch metadata      | seqlens, block IDs (KV-cache)         | Both         |
| P (SS PV mode)      | Softmax probs after STSM              | FA3 only     |
| FP8 V transpose     | LDSM.T -> byte_perm -> STSM staging  | Both         |
| Softmax scales      | Cross-WG scale sharing (LargeHeadDimV)| FA3 only     |

### What Stays in Registers (Critical Decision)

| Data                | Why in registers                         |
|---------------------|------------------------------------------|
| tOrO (output accum) | Streaming across all KV tiles, too large |
| row_max, row_sum    | Updated every tile, small                |
| tOrP (RS PV)        | Avoids smem write+read round-trip        |
| tSrS (QK scores)    | In-place softmax, short-lived per tile   |

---

## 10. Summary: Three-Layer Register Pressure Strategy

### Layer 1: Architectural (Hardware-Level)

- **`setmaxnreg.inc/dec`**: Warpgroup register reallocation (producer 24-56,
  consumer 160-256)
- **`__launch_bounds__(N, 1)`**: Allow full register file for 1 block/SM
- **Hopper-specific**: Only SM90+ supports per-warpgroup register budgets

### Layer 2: Algorithmic (Design-Level)

- **RS vs SS PV**: Trade smem for registers (or vice versa) based on head dim
- **IntraWGOverlap**: Accept higher peak regs for latency hiding (FA3 only)
- **In-place accumulator reuse**: tSrS -> tOrP conversion aliases memory
- **Streaming softmax**: row_max/sum updated in-place, O rescaled without copy
- **LargeHeadDimV split**: QK and PV on separate WGs to reduce per-WG pressure
- **Multi-stage fallback**: Avoid reconfig entirely for small workloads

### Layer 3: Compiler (Pragma-Level)

- **`#pragma unroll 1`**: Prevent outer loop unroll (saves live range duplication)
- **`#pragma unroll`**: Inner MMA loops unrolled (small, safe)
- **`__forceinline__`**: Keep softmax inlined to avoid call overhead
- **`warpgroup_fence_operand`**: Create explicit live range boundaries
- **Spill guard**: Split large K-loops to avoid compiler spill heuristic failures

### The Hierarchy

```
setmaxnreg (hard ceiling per warpgroup)
  |
  +-- RS/SS choice (shifts pressure between regs and smem)
  |
  +-- IntraWGOverlap (trades peak regs for latency)
  |
  +-- Accumulator aliasing (reuse tSrS data for tOrP)
  |
  +-- #pragma unroll 1 (prevent compiler from blowing up live ranges)
  |
  +-- warpgroup_fence (explicit live range boundaries for WGMMA)
```

---

## 11. Implications for Choreo

The Choreo compiler needs to generate code that:

1. **Emits `setmaxnreg`** for producer/consumer warpgroup roles
2. **Honors `__launch_bounds__(N, 1)`** when targeting 1 block/SM
3. **Supports RS vs SS PV GEMM selection** as a code generation option
4. **Uses `#pragma unroll 1`** on outer KV loops (or equivalent LLVM hint)
5. **Generates `warpgroup_fence_operand`** around WGMMA sequences
6. **Manages accumulator aliasing** when converting FP32 scores to bf16/fp8 P

The register budget is the binding constraint for tile size selection:
- 2 WG (128 M-rows): ~240 regs/thread, ~64K total -- fills the SM
- 3 WG (192 M-rows): ~160 regs/thread, ~65K total -- exactly fills the SM
- Larger tiles require fewer regs/thread, limiting accumulator sizes

This means tile size, warp count, and register budget must be co-designed --
they are not independent parameters.

---

## 12. GEMM / Softmax Pipeline Overlap (Tensor Core vs CUDA Core)

On Hopper, WGMMA uses the **Tensor Core** units while softmax (exp2f, fmaxf,
FMA reductions, shuffles) uses the **CUDA FP32 ALU** units. These are
**different hardware pipes**, so they can theoretically execute concurrently
if scheduled on different warpgroups -- or even within the same warpgroup if
the WGMMA is issued asynchronously.

### 12.1 The Three Overlap Strategies

| Strategy | Used by | Overlap type |
|----------|---------|--------------|
| **Intra-WG overlap** (QK[n] + PV[n-1]) | FA3 (`IntraWGOverlap=true`) | Within same WG: two WGMMA batches in flight + softmax between waits |
| **Cross-WG scheduler barriers** | FA3 (`IntraWGOverlap=false`) | Across WGs: one WG's PV WGMMA overlaps other WG's softmax |
| **No overlap** (sequential) | HPC-Ops (all variants) | `warpgroup_wait<0>` after every WGMMA; no scheduler barriers |

### 12.2 FA3 IntraWGOverlap = true (Within-WG Pipeline)

When enabled, each consumer WG's inner loop looks like:

```
// fwd_step(n_block):
consumer_wait(pipeline_k)
flash::gemm QK[n]        (wg_wait=-1)   -> WGMMA issued, NOT waited
[RescaleOBeforeGemm: rescale_o here]     -> FP32 CUDA core (while QK[n] in flight)
consumer_wait(pipeline_v)
flash::gemm PV[n-1]      (wg_wait=-1)   -> second WGMMA issued
warpgroup_wait<1>                        -> QK[n] complete (1 batch remaining = PV[n-1])
pipeline_k.consumer_release
causal_mask(S[n])                        -> CUDA core
online_softmax(S[n])                     -> CUDA core (exp2f, reductions)
                                            WHILE PV[n-1] still in flight on Tensor Cores!
warpgroup_wait<0>                        -> PV[n-1] complete
pipeline_v.consumer_release
convert P[n] to bf16                     -> CUDA core
```

**Key insight:** Between `warpgroup_wait<1>` and `warpgroup_wait<0>`, softmax
runs on CUDA FP32 units while PV[n-1] WGMMA is still executing on Tensor
Cores. This is **explicit intra-WG Tensor Core / CUDA core overlap**.

The cost: peak register pressure = `tSrS(new) + tOrP(prev) + tOrO + softmax`,
because both QK scores and previous P must be live simultaneously.

### 12.3 FA3 IntraWGOverlap = false (Cross-WG Scheduler Barriers)

When IntraWGOverlap is disabled (e.g., hdim64/v512, some paged KV configs),
FA3 uses a **different** overlap mechanism via `NamedBarrier` scheduler:

```
// Non-overlap fwd_step (each WG runs this independently):
consumer_wait(pipeline_k)
flash::gemm QK[n]        (wg_wait=-1)   -> WGMMA issued
warp_scheduler_barrier_arrive()          -> signal OTHER WG: "my QK is issued"
warpgroup_wait<0>                        -> QK[n] complete
pipeline_k.consumer_release
causal_mask + online_softmax             -> CUDA core
rescale_o                                -> CUDA core
consumer_wait(pipeline_v)
warp_scheduler_barrier_sync()            -> wait for OTHER WG's QK arrive
flash::gemm PV[n]        (wg_wait=-1)   -> PV WGMMA issued
warpgroup_wait<0>                        -> PV complete
pipeline_v.consumer_release
```

**How cross-WG overlap emerges:**

The `warp_scheduler_barrier_sync()` before PV only requires that the **other**
WG has **issued** its QK (called `arrive` after QK issue). It does NOT wait
for the other WG's softmax to complete.

So if WG1 finishes softmax before WG2:
- WG1's pre-PV sync passes (WG2 already arrived after QK issue)
- WG1 enters PV WGMMA (Tensor Cores)
- WG2 is still in softmax (CUDA FP32 cores)
- **WG1 Tensor Cores + WG2 CUDA cores run concurrently**

This overlap is **implicit** (not role-assigned) and **dynamic** (whichever
WG finishes softmax first gets the overlap). The NamedBarrier mechanism:

```
WG1 arrive() -> signals WarpSchedulerWG2  (unblocks WG2's pre-PV sync)
WG2 arrive() -> signals WarpSchedulerWG1  (unblocks WG1's pre-PV sync)
```

### 12.4 HPC-Ops: No Overlap (Sequential Pipeline)

HPC-Ops uses **`warpgroup_wait<0>`** after every WGMMA, creating a strictly
sequential pipeline within each WG:

```
// HPC-Ops KV loop body (each WG):
wait_barrier(readable_k)
warpgroup_fence(tAttr)
warpgroup_arrive()
for ik: QK WGMMA                         [Tensor Core]
warpgroup_commit_batch()
warpgroup_wait<0>()                      <-- HARD WAIT: QK must finish
warpgroup_fence(tAttr)
arrive_barrier(writable_k)               [early K release to producer]
causal_mask                              [CUDA core]
online_softmax                           [CUDA core]
P -> bf16 conversion                     [CUDA core]
wait_barrier(readable_v)
warpgroup_fence(tYr)
warpgroup_arrive()
PV WGMMA                                [Tensor Core]
warpgroup_commit_batch()
warpgroup_wait<0>()                      <-- HARD WAIT: PV must finish
arrive_barrier(writable_v)
```

**No cross-WG stagger either:** There are no scheduler barriers between WG0
and WG1. Both WGs run identical code with no synchronization except the
shared `writable_k/v` mbarriers (count=2, ensuring both WGs finish before
buffer reuse). This means:

- Both WGs likely stay near **lockstep** (symmetric work on same-sized tiles)
- Both use Tensor Cores at the same time (QK), both use CUDA cores at the
  same time (softmax)
- No cross-phase overlap where one WG's WGMMA overlaps another's softmax

**What HPC-Ops DOES overlap:** Producer TMA loads with consumer compute.
The early `arrive_barrier(writable_k)` after QK (before softmax) lets the
producer start loading the next K tile while consumers run softmax. This is
**TMA vs compute overlap**, not WGMMA vs softmax.

### 12.5 Comparison Summary

```
HPC-Ops (no overlap):
  WG0: [QK_WGMMA][wait][softmax][PV_WGMMA][wait]  [QK...
  WG1: [QK_WGMMA][wait][softmax][PV_WGMMA][wait]  [QK...
                                                     (lockstep)

FA3 non-overlap (cross-WG stagger via scheduler barriers):
  WG1: [QK_WGMMA][wait][softmax    ][sync][PV_WGMMA][wait]  [QK...
  WG2: [QK_WGMMA][wait][softmax][sync][PV_WGMMA   ][wait]  [QK...
                                   ^    ^
                          WG2 faster, enters PV while WG1 still in softmax
                          Tensor Core (WG2) + CUDA core (WG1) concurrent

FA3 IntraWGOverlap (within-WG dual WGMMA + softmax):
  WG1: [QK[n] + PV[n-1] issued][wait<1>][softmax[n] + PV[n-1] in flight][wait<0>]
  WG2: [QK[n] + PV[n-1] issued][wait<1>][softmax[n] + PV[n-1] in flight][wait<0>]
        ^^^^^^^^^^^^^^^^^^^^^            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Two WGMMA batches concurrent     CUDA core + Tensor Core concurrent (same WG!)
```

### 12.6 Hardware Implications on H100/H800/H20

All three are Hopper (SM90a) with the same compute unit layout:

| Unit | Function | Shared? |
|------|----------|---------|
| Tensor Core (WGMMA) | Matrix multiply-accumulate | Per-SM, 4th gen |
| FP32 CUDA Core | Softmax (exp2f, FMA, shuffles) | Per-SM |
| TMA Unit | Async bulk memory copy | Per-SM |

On Hopper, WGMMA is truly asynchronous: once issued via `wgmma.mma_async`,
the warpgroup can proceed to execute CUDA core instructions. The
`wgmma.wait_group.sync N` instruction blocks until at most N WGMMA groups
remain pending.

- `wait<0>`: Block until ALL WGMMA complete (no overlap)
- `wait<1>`: Block until at most 1 WGMMA group pending (the older one done)
- `wait<-1>` (FA3 convention): Don't emit any wait (full overlap)

The key difference between H100/H800 and H20:

| GPU | SMs | HBM BW | Tensor Core TFLOPS (FP16) |
|-----|-----|--------|---------------------------|
| H100 SXM | 132 | 3.35 TB/s | 989 |
| H800 | 132 | 3.35 TB/s | 989 |
| H20 | 132 | 4.0 TB/s | 148 |

H20 has **~6.7x fewer Tensor Core TFLOPS** but higher memory bandwidth. This
means on H20, the Tensor Cores are the bottleneck, making GEMM/softmax
overlap even more valuable -- softmax on CUDA cores is essentially "free"
when Tensor Cores are the constraint. On H100/H800, the balance shifts
toward memory bandwidth, but overlap still helps hide softmax latency.

### 12.7 Why HPC-Ops Might Not Need It

HPC-Ops targets a narrower design space (head_dim=128 only, fixed tiles).
Possible reasons for omitting cross-WG overlap:

1. **Simplicity**: Sequential pipeline is easier to reason about and debug
2. **Producer overlap sufficiency**: TMA-vs-compute overlap may provide
   enough latency hiding for their target workloads
3. **Register pressure**: Scheduler barriers add complexity that could
   interact with register allocation (though FA3 manages it fine)
4. **Occupancy trade-off**: With 1 block/SM and 384 threads, the SM is
   already fully utilized; overlap might not change the throughput
   significantly if Tensor Cores are not the bottleneck

However, FA3's approach suggests there IS measurable performance benefit
from GEMM/softmax overlap, especially for the IntraWGOverlap path which
FA3 enables by default for most tile configurations.
