# SALA Cross-Compiler Evaluation Summary

## Overview

This document summarizes the cross-compiler evaluation of SALA (Signal-Aware Liveness Analysis) across four GPU compiler/framework ecosystems.

## Compiler Landscape

### Capability Matrix

| Capability                          | Choreo (XComp) | CUTLASS   | Triton    | TileLang   |
|-------------------------------------|----------------|-----------|-----------|------------|
| Concurrent warp groups              | Yes (inthreads.async) | Yes (WG roles) | No  | Partial (T.ws, sequential) |
| Explicit signal/barrier             | Yes (trigger/wait) | Yes (PipelineTmaAsync) | Implicit (async_wait) | Partial (mbarrier API) |
| Shared memory epilogue buffer       | Yes            | Yes       | No        | Yes (T.alloc_shared) |
| Automatic liveness analysis         | Yes (SALA)     | No (manual) | Yes (conservative) | Yes (conservative) |
| Mainloop/epilogue SMEM overlap      | Yes (SALA)     | Manual (union for non-persistent) | N/A | Yes (sequential only) |
| Multi-consumer warp-spec            | Yes (1P2C, 1P3C) | Yes (pingpong, cooperative) | No | No |

### Key Insight

**Choreo, CUTLASS, and Tawa can express the concurrent warp-group pattern** where multiple warp groups execute simultaneously with signal-mediated buffer access. This is exactly the pattern where SALA provides its unique value -- analyzing signal semantics to determine safe buffer overlap.

- **Triton**: No concurrent warp groups. All threads execute the same code path. Pipeline is loop-internal (async_copy + commit + wait). Liveness analysis is conservative but sufficient since there's no inter-group signal ordering to exploit.

- **TileLang**: Has `T.ws()` for warp specialization, but generates **sequential** warp-specialized code with `__syncthreads()` between groups. Its compiler already handles simple sequential overlap (e.g., A_shared and C_shared at offset 0 when lifetimes don't overlap). However, it cannot express Choreo's concurrent `inthreads.async` pattern.

- **CUTLASS**: Has concurrent warp groups with explicit pipeline barriers (`PipelineTmaAsync`, `OrderedSequenceBarrier`, `MathWarpGroupOrderBarrier`, `NamedBarrier`). However, CUTLASS relies on **manual C++ engineering** (union vs struct) to manage SMEM layout -- no automatic liveness analysis.

- **Tawa**: Has concurrent warp groups with `aref`-based signal semantics (`put`/`get`/`consumed`), which map directly to SALA's happens-before chain. However, Tawa's epilogue writes from registers directly to GMEM -- no epilogue SMEM buffer exists, so SALA's physical overlap is currently vacuous. If Tawa adopted `stmatrix`-based output staging, SALA would be immediately applicable via the existing `aref` HB graph.

## Evaluation Results

### 1. Liveness Interval Precision (Primary Metric)

Measures how much SALA shrinks buffer liveness intervals compared to conservative (alloc-to-dealloc) analysis.

#### Triton Kernels (standalone analyzer on TTGIR)

| Kernel                     | Buffers | Conv Span | SALA Span | Shrinkage |
|----------------------------|---------|-----------|-----------|-----------|
| GEMM 128x128x64 (f16)     | 2       | 20        | 14        | 30.0%     |
| GEMM 256x128x64 (f16)     | 2       | 20        | 14        | 30.0%     |
| GEMM 128x256x64 (f16)     | 2       | 20        | 14        | 30.0%     |
| GEMM 128x128x32 (f32)     | 2       | 20        | 14        | 30.0%     |
| GEMM 128x128x64 (bf16)    | 2       | 20        | 14        | 30.0%     |
| FlashAttention (64 head)   | 3       | 44        | 21        | 52.3%     |
| FlashAttention (128 head)  | 3       | 50        | 24        | 52.0%     |
| FlashAttention (256 head)  | 3       | 40        | 19        | 52.5%     |

**Interpretation**: Even without epilogue SMEM buffers, SALA identifies 30-52% liveness interval shrinkage in Triton kernels. This represents dead-but-allocated SMEM that would be reclaimable if Triton adopted SMEM-based output staging or additional pipeline buffers.

#### Synthetic Warp-Spec TTGIR (mimicking CUTLASS/Choreo patterns)

| Kernel                     | Buffers | Conv Span | SALA Span | Shrinkage | SMEM Reduction |
|----------------------------|---------|-----------|-----------|-----------|----------------|
| warpspec_gemm_1p1c         | 3       | 26        | 6.5       | 75.0%     | 37.5%          |
| warpspec_gemm_1p2c         | 4       | 31        | 5.8       | 81.2%     | 37.5%          |
| unsafe_overlap (negative)  | 2       | 17        | 11.5      | 32.4%     | 0% (correct)   |

**Interpretation**: When epilogue output buffers exist (the concurrent warp-spec pattern), SALA achieves 75-81% interval shrinkage and 37.5% physical SMEM reduction. The unsafe_overlap test confirms SALA correctly rejects unsound overlaps.

### 2. CUTLASS Static Analysis

CUTLASS manages SMEM layout manually via C++ template specialization:
- **Non-persistent kernels**: Uses `union TensorStorage` to overlap mainloop and epilogue (safe because they execute sequentially). SALA matches this.
- **Persistent kernels**: Uses `struct TensorStorage` -- mainloop and epilogue are allocated separately because they may be concurrent. SALA can improve this.

| CUTLASS Schedule               | Tile         | Stages | CUTLASS | SALA (conservative) | Savings |
|--------------------------------|--------------|--------|---------|---------------------|---------|
| Non-persistent (f16)           | 128x128x64   | 4      | 128 KB  | 128 KB              | 0%      |
| Cooperative persistent (f16)   | 128x128x64   | 4      | 256 KB  | 240 KB              | 6.2%    |
| PingPong persistent (f16)      | 128x256x64   | 4      | 448 KB  | 416 KB              | 7.1%    |
| PingPong persistent large (f16)| 256x128x64   | 3      | 272 KB  | 229 KB              | 15.7%   |
| Cooperative persistent (fp8)   | 128x128x128  | 4      | 192 KB  | 184 KB              | 4.2%    |
| Cooperative ReuseC (f16)       | 128x128x64   | 4      | 192 KB  | 176 KB              | 8.3%    |

**Interpretation**: For persistent warp-specialized kernels (the dominant schedule on H100), CUTLASS allocates 4-16% more SMEM than necessary because it cannot automatically analyze signal ordering. SALA would reclaim this overhead. The savings increase with tile size and when ReuseSmemC is enabled.

### 3. TileLang Analysis

TileLang's compiler already handles the simple case:
- Sequential warp groups: `A_shared` and `C_shared` placed at offset 0 (overlap)
- This is equivalent to CUTLASS's non-persistent union

However, TileLang **cannot express concurrent warp groups** with explicit signal coordination. Its `T.ws()` generates `if (threadIdx.x < 128)` / `else` blocks with `__syncthreads()`, not concurrent `inthreads.async` execution.

Generated CUDA from TileLang warp-spec GEMM:
```
A_shared = (void*)((char*)buf_dyn_shmem + 0);     // offset 0
C_shared = (void*)((char*)buf_dyn_shmem + 0);     // offset 0 (overlapped!)
B_shared = (void*)((char*)buf_dyn_shmem + 49152); // separate
```

**Interpretation**: TileLang's LA already handles the easy sequential case. SALA's contribution is for the hard concurrent case that TileLang cannot express.

### 4. Tawa (aref MLIR) Analysis

Tawa occupies a unique position in the landscape: it has concurrent warp groups with the richest signal formalism (`aref`), but its epilogue bypasses SMEM.

| IR Model                    | Buffers | Interval Shrinkage | SMEM Reduction (hypothetical) |
|-----------------------------|---------|--------------------|-----------------------------|
| Tawa 1P1C GEMM (D=3)       | 3       | 80.0%              | 25% (128 KB → 96 KB)       |
| Tawa cooperative (D=2)     | 4       | 85.5%              | 33% (96 KB → 64 KB)        |

**Key insight**: Tawa's `aref` operations (`put`/`get`/`consumed`) provide exactly the happens-before chain SALA needs. Integration would require:
1. Constructing the HB graph from `aref` operations (~50 LOC)
2. Adding a non-interference predicate to the SMEM allocation pass (~60 LOC)
3. Adding `stmatrix`-based epilogue output staging (the harder part -- architecture decision)

The numbers above assume step 3 is complete; steps 1-2 are minimal given `aref`'s explicit signal semantics.

### 5. Choreo (XComp) End-to-End Results

(These are from the existing paper evaluation -- not repeated here.)

SALA achieves 37-50% SMEM reduction on real warp-specialized GEMM kernels with concurrent producer-consumer warp groups, leading to occupancy improvements (1→2 CTAs/SM in several cases) and 5-15% throughput gains.

## Cross-Compiler Narrative

The evaluation reveals a clear hierarchy:

1. **Compilers with no concurrent warp groups** (Triton, TileLang sequential mode):
   - SALA provides interval shrinkage (30-52%) identifying dead SMEM
   - No physical overlap possible because there's no separate epilogue buffer
   - SALA's value here is *diagnostic*: it quantifies the LA quality gap

2. **Compilers with concurrent warp groups but no epilogue SMEM** (Tawa):
   - Has the richest signal formalism (`aref`) -- SALA integration is trivial
   - SALA interval shrinkage is 80-85% (hypothetical)
   - Physical overlap would yield 25-33% if `stmatrix` output staging were added
   - SALA's value here is *latent*: it quantifies what would become possible

3. **Compilers with concurrent warp groups but manual SMEM management** (CUTLASS):
   - SALA provides 4-16% conservative SMEM savings on persistent schedules
   - Currently CUTLASS engineers manually decide union vs struct
   - SALA would automate this decision with compiler-verified correctness

4. **Compilers with concurrent warp groups and automatic LA** (Choreo with SALA):
   - Full 37-50% SMEM reduction with proven correctness
   - The only system where signal-aware LA is actually implemented

## Soundness Validation

All SALA analysis results pass the soundness checker:
- Every buffer access falls within the SALA-computed liveness interval
- No two overlapped buffers have temporally intersecting SALA intervals within the same phase
- The unsafe_overlap test case correctly rejects co-live buffers

## Files

| File | Description |
|------|-------------|
| `tools/sala_analyzer.py` | Standalone SALA analyzer for TTGIR |
| `tools/test_ttgir/warpspec_gemm_1p1c.ttgir` | Synthetic 1P1C warp-spec TTGIR |
| `tools/test_ttgir/warpspec_gemm_1p2c.ttgir` | Synthetic 1P2C warp-spec TTGIR |
| `tools/test_ttgir/unsafe_overlap.ttgir` | Negative test (co-live buffers) |
| `tools/test_ttgir/tawa_gemm_aref.ttgir` | Synthetic Tawa 1P1C GEMM with aref |
| `tools/test_ttgir/tawa_gemm_cooperative.ttgir` | Synthetic Tawa cooperative GEMM |
| `tools/sala_data/cutlass_analysis/cutlass_smem_analysis.py` | CUTLASS SMEM layout analysis |
| `tools/sala_data/triton_ttgir/` | Generated Triton TTGIRs |
| `tools/sala_data/tilelang_cuda/` | Generated TileLang CUDA sources |
