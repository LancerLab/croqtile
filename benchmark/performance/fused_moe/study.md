# Blockscale GEMM Study: Choreo vs DeepGEMM

## Overview

This document summarizes the comparison between Choreo's current `blockscale_gemm` implementation and DeepGEMM's `sm90_fp8_gemm_1d1d.cuh` reference implementation.

## Test Cases Examined

### Choreo Source Files
- `wip_kernels/blockscale_gemm_e4m3_dyn_sm90.co`
- `wip_kernels/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co`
- `wip_kernels/blockscale_gemm_e4m3_dyn_sm90_tileN.co`
- `tests/gpu/end2end/matmul/blockscale_gemm_e4m3_dyn_sm90_N48.co`
- `tests/gpu/end2end/matmul/matmul_e4m3_dynamic_sm90.co`
- `tests/gpu/end2end/matmul/matmul_e4m3_dynamic_mwg_sm90.co`
- `tests/gpu/end2end/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co`

### Generated CUDA Files (via `-es`)
All generated to `build/investigate/`:
- `blockscale_gemm_e4m3_dyn_sm90.cu`
- `blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.cu`
- `blockscale_gemm_e4m3_dyn_sm90_tileN.cu`
- `blockscale_gemm_e4m3_dyn_sm90_N48.cu`
- `matmul_e4m3_dynamic_sm90.cu`
- `matmul_e4m3_dynamic_mwg_sm90.cu`
- `matmul_f16_dyn_sm90_warpspec_1p1c.cu`

---

## Key Findings

### 1. Scale Data Path

**Choreo (Current):**
- Performs MMA first (`MMA_64x128x32_F16E4M3E4M3`), stores to temporary `mc_scale_frag`
- Then calls `scale_accumulator` helper (in `runtime/choreo_cute.h:2400`)
- The helper reads scale directly from **global memory** using `__ldg()`

```cpp
// Generated code: build/investigate/blockscale_gemm_e4m3_dyn_sm90.cu:245-248
cute::SM90::GMMA::MMA_64x128x32_F16E4M3E4M3_SS_TN<>::fma(...);
float* mc_scale_a_ptr = (float*)((DIV_BLK_K * blockIdx.x * 64 + __iv_iv_k + scale_lhs));
float mc_scale_b_val = static_cast<float>(*((float*)scale_rhs + ...));
scale_accumulator<f16, float, 128>(...);
```

**DeepGEMM (Reference):**
- Uses TMA to also load `sfa` (scale A) and `sfb` (scale B) to **shared memory**
- Math warpgroup reads scales from shared memory, performs scaling **within the pipeline**
- Full `sfa/sfb` pipeline integrated with A/B pipeline

---

### 2. Accumulator / Output Type

**Choreo (Current):**
- Accumulator type: `f16` (half)
- Output type: `f16` (half)
- See: `build/investigate/blockscale_gemm_e4m3_dyn_sm90.cu:198` (accumulators), `.cu:255` (output store)

**DeepGEMM (Reference):**
- Accumulator type: `float` for both intermediate and final accumulation
- Uses `float final_accum[WGMMA::kNumAccum]` for epilogue scaling
- `cd_dtype_t` is constrained to `float`
- This is critical for FP8 blockscale precision

---

### 3. Producer/Consumer Topology

**Choreo (Non-warpspec):**
- Single CTA with 128 threads
- Same warpgroup does both TMA load and WGMMA compute
- No explicit producer/consumer separation

**Choreo (Warpspec - 1p1c):**
- CTA with 256 threads
- Explicit 1 producer + 1 consumer warpgroup structure
- See: `build/investigate/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.cu:200,218`

**DeepGEMM (Reference):**
- Explicitly supports `1 TMA warpgroup + N math warpgroups`
- Parameters: `kNumTMAThreads` and `kNumMathThreads`
- Can have multiple consumer warpgroups working in parallel

---

### 4. Pipeline Depth

**Choreo (Baseline):**
- "Issue TMA → Wait → Compute" (single-buffered)
- No pipeline overlap

**Choreo (Warpspec):**
- 2-stage ping-pong (see `wip_kernels/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co:20`)
- Limited pipeline parallelism

**DeepGEMM (Reference):**
- Configurable `kNumStages` (tunable parameter)
- Pipeline unroll control: `kNumPipelineUnrolls`
- TMA and math register budget: `kNumTMARegisters`, `kNumMathRegisters`
- These are core tuning parameters

---

### 5. Scheduling Strategy

**Choreo (Current):**
- Regular grid tile scheduling: simple `block_m x block_n` grid
- See: `build/investigate/blockscale_gemm_e4m3_dyn_sm90.cu:351`
```cpp
dim3 __blockscale_gemm_gdims0(((M + 63) / 64), ((N + 127) / 128), 1);
dim3 __blockscale_gemm_bdims0(128, 1, 1);
```

**DeepGEMM (Reference):**
- Persistent kernel with `Scheduler<...>` class
- Supports grouped contiguous K and tensor map prefetch switching
- Can handle complex tiling patterns

**Note:** Choreo has persistent kernel examples in regular FP8 matmul:
- `benchmark/performance/matmul/matmul_e4m3_dyn_persis_sta_sm90.co`

---

### 6. Memory System Optimizations

**DeepGEMM implements:**
- TMA descriptor prefetch
- Shared/Global tensor map double buffering
- Cluster Transaction Barrier (`cutlass::arch::ClusterTransactionBarrier`)
- TMA multicast support
- `warpgroup_reg_alloc/dealloc` for register reconfiguration
- `__launch_bounds__` kernel attribute
- TMA reduce store (`SM90_TMA_REDUCE_ADD_2D`)

**Choreo (Current):**
- Basic TMA copy with barriers
- Warpspec version has `full/empty` barriers
- Dynamic shared memory setup
- See: `build/investigate/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.cu:184,361`

---

## Hyperparameter Comparison

| Aspect | Choreo (Current) | DeepGEMM |
|--------|------------------|----------|
| BLOCK_K | 128 (fixed) | 128 (template param) |
| BLOCK_M | 64 (fixed) | Template param |
| BLOCK_N | 32/48/64/128 (limited trials) | Template param |
| Swizzle | 128 (fixed) | Template param (`kSwizzleAMode`, `kSwizzleBMode`) |
| Stages | 2-4 (static) | Template param (`kNumStages`) |
| Thread count | 128 or 256 (CTA-level) | `kNumTMAThreads` + `kNumMathThreads` (separate) |
| Multicast | Not implemented | Template param (`kNumTMAMulticast`) |
| SM count | Not explicit | Template param (`kNumSMs`) |
| GEMM type | Single mode | Template param (`GemmType`) |

---

## Architecture Summary

### Current Choreo Blockscale = "FP8 MMA + Global Scale Post-process"

The current implementation is conceptually:
1. Load A/B from global → shared via TMA
2. Perform FP8 WGMMA → accumulator (f16)
3. After all K iterations, multiply accumulator by scale from global memory

This is NOT the same as DeepGEMM's approach where scale is part of the pipeline.

### DeepGEMM = "Scale-Aware Pipelined MMA"

DeepGEMM treats scale as first-class pipeline citizen:
1. Load A/B from global → shared via TMA
2. Load sfa/sfb from global → shared via TMA (in parallel)
3. In math warpgroup: read sfa/sfb from shared, multiply during MMA
4. Final output with epilogue scaling

---

## Recommended Implementation Roadmap

### Priority 1: Scale Data Path
- Lift scale loading to TMA pipeline (global → shared)
- Change accumulator from `f16` to `float`
- Integrate scale multiply within WGMMA loop

### Priority 2: Producer/Consumer Topology
- Extend from `1p1c` to `1pN` (e.g., 1 producer + 2-3 consumers)
- Leverage existing templates in:
  - `tests/gpu/end2end/matmul/matmul_e4m3_dynamic_mwg_sm90.co`
  - `benchmark/performance/matmul/matmul_e4m3_dyn_sm90_warpspec_1p1c.co`

### Priority 3: Hyperparameter Tuning
- After fixing P1/P2, tune:
  - BLOCK_N (shape)
  - Stage count
  - Swizzle mode
  - TMA multicast

---

## Warnings Found During Generation

All blockscale cases showed TMA dimension warnings:
```
warning: Dimensions could be inconsistent between DMA...
```

This indicates tile/descriptor constraint expressions need improvement.

---

## Files Generated

All generated CUDA files are located in:
```
build/investigate/
├── blockscale_gemm_e4m3_dyn_sm90.cu
├── blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.cu
├── blockscale_gemm_e4m3_dyn_sm90_tileN.cu
├── blockscale_gemm_e4m3_dyn_sm90_N48.cu
├── matmul_e4m3_dynamic_sm90.cu
├── matmul_e4m3_dynamic_mwg_sm90.cu
└── matmul_f16_dyn_sm90_warpspec_1p1c.cu
```

---

## References

- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- Reference file: `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`
- Choreo runtime: `runtime/choreo_cute.h` (contains `scale_accumulator` at line 2400)
