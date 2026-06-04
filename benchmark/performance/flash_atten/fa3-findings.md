# Hopper Prefill MHA/GQA Optimization Findings

Research comparing HPC-Ops (Tencent, `~/dev/hpc-ops/`) and FlashAttention-3
(FA3, `~/dev/flash-attention/`) for SM90a (Hopper) prefill attention.

Repos studied:
- HPC-Ops: `src/attention/prefill/` (config.h, kernels.cuh, prefill.cc)
- FA3: `hopper/` (flash_fwd_kernel_sm90.h, mainloop_fwd_sm90_tma_gmma_ws.hpp,
  tile_size.h, softmax.h, pack_gqa.h, named_barrier.hpp, utils.h)

---

## 1. Kernel Architecture

Both use warp-specialized kernels built on CuTe + CUTLASS 3.x:
1 producer warp group (TMA loads) + N consumer warp groups (WGMMA + softmax).

| | HPC-Ops | FA3 |
|---|---|---|
| Producer threads | 128 (1 WG, 1 elected TMA) | 32 (1 warp) or 128 (PackGQA) |
| Consumer threads | 256 (2 WGs) | 256 (2 WGs) or 384 (3 WGs) |
| Total block | 384 | 288-512 |
| Launch bounds | `(384, 1)` | `(MaxThreads, 1)` via CUTLASS |
| Occupancy | 1 block/SM | 1 block/SM |

Ref: HPC `__launch_bounds__`: `kernels.cuh:245,599,995`.
FA3: `flash_fwd_kernel_sm90.h:76-77` (MaxThreads, MinBlocks=1),
applied via `cutlass/device_kernel.h:116`.

---

## 2. Tiling Strategy

### HPC-Ops (fixed, head_dim=128 only)

| Variant | BLOCK_M | BLOCK_N | Stages | MMA WGs |
|---|---|---|---|---|
| Warp-spec (persistent) | 128 | 64 | 2 | 2 |
| Multi-stage (low occupancy) | 64 | 64 | 1 | 1 |

Ref: `config.h:22-27` (tile params), `warp_spec_dim128.cu:48` (128x64),
`multi_stage_dim128.cu:48` (64x64).

### FA3 (adaptive, head_dim 64-256)

| Head dim | Causal? | BLOCK_M | BLOCK_N | RS? | WGs |
|---|---|---|---|---|---|
| 64 | no | 192 | 192 | yes | 3 |
| 64 | yes | 192 | 128 | yes | 3 |
| 128 | no | 128 | 176 | yes | 2 |
| 128 | yes | 128 | 128 | yes | 2 |
| 192 | no | 128 | 128 | yes | 2 |
| 256 | yes | 128 | 64 | yes | 2 |

Ref: `tile_size.h:10` (`tile_size_fwd_sm90` function).

---

## 3. TMA Configuration

| | HPC-Ops | FA3 |
|---|---|---|
| Q load | TMA (always) | TMA or cp.async (PackGQA) |
| K load | TMA with mbarrier | TMA with pipeline barrier |
| V load | TMA (MN-major layout) | TMA (transposed for FP8) |
| O store | TMA store | TMA store or vectorized gmem |
| Multicast | No | 2-CTA cluster (non-causal, hdim>=128) |
| Cache hints | No | EVICT_FIRST (Q), EVICT_LAST (K/V) |
| Per-batch update | Separate patch kernel | Via scheduler/seqlen info |

Ref: HPC `config.h:76` (make_tma_copy Q/K/V/Y), `config.h:164` (KV-cache).
FA3 `mainloop:461-477` (tma_load_Q/K/V creation), `mainloop:742`
(multicast mask), `flash_fwd_launch_template.h:212` (cluster enable gate).

---

## 4. WGMMA Atoms

| GEMM | HPC-Ops | FA3 |
|---|---|---|
| QK (GEMM-I) | `SM90_64x64x16_F32BF16BF16_SS` (K,K) | `ss_op_selector` -- always SS |
| PV (GEMM-II) | `SM90_64x128x16_F32BF16BF16_RS` (K,MN) | SS or RS via `MmaPV_is_RS` |
| FP8 QK | `SM90_64x64x32_F32E4M3E4M3_SS_TN` | `ss_op_selector` FP8 |
| Accumulator | FP32 | FP32 |

---

## 5. Online Softmax

Both use the same algorithm: running max + sum with exp2 (maps to FFMA).

```
scale = 1/sqrt(dim) * log2(e)
For each KV tile:
  S = QK^T
  apply causal mask (invalid -> -inf)
  new_max = max(old_max, row_max(S))
  scores_scale = exp2((old_max - new_max) * scale)
  P = exp2(S * scale - new_max * scale)
  row_sum = row_sum * scores_scale + sum(P)
  O = O * scores_scale + P @ V
Final: O /= row_sum
```

HPC-Ops: `warp_4lane_reduce_max_xor` for 4-lane reductions.
FA3: `quad_allreduce_` for final reduction. FP8 subtracts max by 8.0.

Ref: HPC `kernels.cuh:59` (online_softmax), `utils.cuh:378`
(warp_4lane_reduce_max_xor).
FA3 `softmax.h:102` (max_get_scale), `softmax.h:127` (online_softmax).

---

## 6. GQA Optimization

### HPC-Ops: Simple Head Index Mapping

```cpp
int kv_group = num_head_q / num_head_kv;
int ihead_kv = ihead_q / kv_group;  // FastDivmod
```

Each block processes one Q head. Multiple blocks sharing a KV head load it
independently.

Ref: `warp_spec_dim128.cu:72` (FastDivmod head_kv_divmod).

### FA3: PackGQA -- Fuse Q Heads into M Dimension

Multiple Q heads sharing one KV head are packed along M:
- Q shape becomes `(qhead_per_khead * seqlen, dim)` per KV head
- One block processes multiple Q heads against single K/V load
- Trade-off: TMA cannot load packed Q -> falls back to cp.async

Heuristic:
```
nopack_eff = seqlen_q / round_up(seqlen_q, blockM)
pack_eff = seqlen_q * qhead_per_khead / round_up(seqlen_q * qhead_per_khead, blockM)
use_pack = nopack_eff < 0.9 * pack_eff
```

Ref: `mainloop:52,56` (PackGQA, Use_TMA_Q definitions),
`heuristics.h:9` (should_pack_gqa).

---

## 7. Pipeline / Staging

| | HPC-Ops | FA3 |
|---|---|---|
| K/V stages | 2 (warp-spec) / 1 (multi-stage) | 2 (always) |
| Q buffering | Single, phase flip | Single, barrier phase |
| IntraWG overlap | No | Yes (QK[n] + PV[n-1]) |
| Barrier type | Hopper mbarrier pairs | CUTLASS PipelineTmaAsync (custom) |

---

## 8. Causal Masking

### Block-level tile pruning (both)

```
n_block_max = ceil_div(m_idx_max + seqlen_k - seqlen_q, kBlockN)
```

### Within-tile masking

- HPC-Ops: Direct `icol > irow` check -> -inf
- FA3: Three-phase loop: (1) causal-masked, (2) fully unmasked, (3) local.
  Avoids mask overhead on unmasked tiles.

Ref: FA3 `mask.h:89` (causal mask application).

---

## 9. Persistent vs Non-Persistent Dispatch

### HPC-Ops: Occupancy-based

```
if (num_tiles < 2 * SM_count): multi_stage (1 WG, 64x64)
else: warp_spec persistent (2 WGs, 128x64)
```

Ref: `prefill.cc:27` (multi-stage gate), `prefill.cc:36` (warp-spec).

### FA3: Always persistent for SM90 warp-spec

Reverse-linear work-stealing scheduler. Grid = SM_count blocks.

---

## 10. Shared Memory Budget

| | HPC-Ops (warp-spec BF16) | FA3 (hdim128 causal BF16) |
|---|---|---|
| Q | 32 KiB | 32 KiB |
| K (x2 stages) | 32 KiB | 32 KiB |
| V (x2 stages) | 32 KiB | 32 KiB |
| P (softmax probs) | 0 (RS) | 0 (RS) or 32 KiB (SS) |
| O (output) | 32 KiB (overlaps Q) | overlaps Q |
| Total | ~128 KiB | ~128-224 KiB |

FA3 uses up to 224 KiB on H100 (228 KiB hardware limit).
All swizzle patterns use 128B atoms for bank-conflict-free WGMMA access.

HPC-Ops multi-stage saves smem by reusing Q/K/V buffer for output Y
(`shm_y = shm_data`).

---

## 11. Register Pressure Management

(See also: Documents/research/hopper_prefill_register_pressure.md for
detailed analysis with code references and line numbers.)

### Warpgroup Register Reallocation (setmaxnreg)

Both split threads into producer (low regs) and consumer (high regs):

| | HPC-Ops | FA3 |
|---|---|---|
| Producer dealloc | 24-32 | 24-56 |
| Consumer alloc | 168 (BF16), 192 (FP8) | 160-256 |
| Total/SM | 45K-52K | 64K-65K |

FA3 pushes budgets much higher, using nearly the full 65,536 register file.
FA3 3WG config uses exactly 65,536 registers.

Ref: HPC `kernels.cuh:364` (dealloc<32>), `kernels.cuh:428` (alloc<168>),
`kernels.cuh:1236` (FP8 alloc<192>).
FA3 `flash_fwd_kernel_sm90.h:82-83` (Load/MmaRegisterRequirement),
`flash_fwd_kernel_sm90.h:309` (dealloc), `flash_fwd_kernel_sm90.h:361`
(alloc).

### Fragment Inventory (per consumer thread)

| Fragment | Elements | Type | Regs |
|---|---|---|---|
| tSrS (QK scores) | 32 FP32 | Accumulator | ~32 |
| tOrO (output) | 64 FP32 | Accumulator | ~64 |
| tOrP (PV RS A-op) | 4 uint32 | Input | ~4 |
| Softmax state | 6-12 FP32 | row_max/sum/scale | ~12 |

### RS vs SS for PV GEMM

| | RS PV | SS PV |
|---|---|---|
| P storage | Registers | Shared memory |
| Saves | smem | registers |
| Used by | HPC-Ops always, FA3 hdim>=96 | FA3 hdim<=96, LargeHeadDimV |

Ref: HPC `kernels.cuh:531` (tAttAbf16 conversion), `kernels.cuh:541`
(PV RS gemm). FA3 `mainloop:1079` (write_P_to_smem lambda for SS path).

### Compiler Pragmas

- `#pragma unroll 1` on outer KV loop (prevents live range duplication)
- `#pragma unroll` on inner MMA K-loop (small, safe)
- `warpgroup_fence_operand` for WGMMA live range boundaries
- `__launch_bounds__(N, 1)` for 1 block/SM register allocation

---

## 12. GEMM / Softmax Pipeline Overlap

On Hopper, WGMMA (Tensor Cores) and softmax (CUDA FP32 cores) are different
hardware units that can run concurrently.

### FA3: Two Overlap Mechanisms

**IntraWGOverlap = true** (default for most configs):

```
Issue QK[n]  (wg_wait=-1)     -> WGMMA issued, NOT waited       mainloop:1177
Issue PV[n-1] (wg_wait=-1)    -> second WGMMA issued            mainloop:1182
warpgroup_wait<1>              -> QK[n] done, PV[n-1] in flight  mainloop:1184
softmax on S[n]                -> CUDA cores WHILE PV[n-1] on Tensor Cores
warpgroup_wait<0>              -> PV[n-1] done
```

Ref: `mainloop:1138` (IntraWGOverlap path entry), `mainloop:1170`
(fwd_step lambda). `flash::gemm` template: `utils.h:254` (wg_wait param),
`utils.h:313` (conditional warpgroup_wait dispatch).

**IntraWGOverlap = false** (cross-WG scheduler barriers):

```
QK issued -> arrive (signal other WG) -> wait<0> -> softmax -> sync -> PV
```

Pre-PV sync only requires other WG's post-QK arrive, NOT softmax completion.
Faster WG enters PV WGMMA while slower WG still in softmax.

Ref: `mainloop:1258` (non-overlap fwd_step lambda),
`mainloop:352` (UseSchedulerBarrier definition),
`mainloop:915` (warp_scheduler_barrier_sync),
`mainloop:922` (warp_scheduler_barrier_arrive),
`named_barrier.hpp:57-58` (PFull/PEmpty barriers for LargeHeadDimV).

### HPC-Ops: No Overlap

```
QK WGMMA -> warpgroup_wait<0> -> softmax -> PV WGMMA -> warpgroup_wait<0>
```

Hard `wait<0>` after every WGMMA. No scheduler barriers between consumer WGs.
Both WGs run near-lockstep. Only overlap: TMA (producer) vs compute (consumer).

Ref: `kernels.cuh:494` (warpgroup_wait<0> after QK -- the key no-overlap
evidence), `kernels.cuh:498` (early K barrier release before softmax),
`kernels.cuh:351,353` (writable_k/v barrier init with count=2).

### Comparison Timeline

```
HPC-Ops (no overlap):
  WG0: [QK_WGMMA][wait][softmax][PV_WGMMA][wait]  [QK...
  WG1: [QK_WGMMA][wait][softmax][PV_WGMMA][wait]  [QK...
                                                     (lockstep)

FA3 non-overlap (cross-WG stagger):
  WG1: [QK][wait][softmax    ][sync][PV_WGMMA][wait]  [QK...
  WG2: [QK][wait][softmax][sync][PV_WGMMA   ][wait]  [QK...
                              WG2 PV on Tensor Cores + WG1 softmax on CUDA cores

FA3 IntraWGOverlap:
  WG: [QK[n]+PV[n-1] issued][wait<1>][softmax[n] + PV[n-1] in flight][wait<0>]
      Two WGMMA batches concurrent    CUDA core + Tensor Core concurrent (same WG)
```

---

### LargeHeadDimV Path (FA3 only, kHeadDimV > 256)

A third overlap strategy where WGs have different roles:
- WG1: QK + softmax + P write to smem (signals PFull barrier)
- WG2+: PV WGMMA only (waits on PFull, signals PEmpty)

This is explicit cross-WG overlap: WG2 runs Tensor Core PV while WG1
does CUDA core softmax + P write. Coordinated by PFull/PEmpty barriers.
Used for MLA-like configs (hdim=64, hdim_v=512).

Ref: `flash_fwd_kernel_sm90.h:420-430` (LargeHeadDimV dispatch:
mma vs mma_pv), `mainloop:1356` (mma_pv function entry).

---

## 13. Feature Comparison Summary

| Feature | HPC-Ops | FA3 |
|---|---|---|
| Head dims | 128 only | 64, 96, 128, 192, 256 |
| Tile sizes | Fixed | Adaptive per config |
| GQA | Simple head mapping | PackGQA (fuse Q heads in M) |
| Cluster multicast | No | Yes (2-CTA) |
| PV GEMM mode | Always RS | RS or SS |
| IntraWG overlap | No | Yes |
| Cross-WG stagger | No | Yes (NamedBarrier) |
| FP8 prefill | Paged-KV only | Full |
| Consumer regs | 168-192 | 160-256 |
| Causal mask | Per-element | Three-phase skip |
| TMA cache hints | No | Yes |
