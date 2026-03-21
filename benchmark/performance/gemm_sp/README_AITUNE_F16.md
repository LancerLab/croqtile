# gemm_sp F16 AI-Tune Results

Performance progression of structured-sparse GEMM (F16, 2:4 sparsity) on H800 PCIe.

**Problem size**: M=4096, N=8192, K=8192 (configurable via `SPMM_DEFAULT_M/N/K`)
**Baseline (main)**: `trial000128` — ~460 TFLOPS (30.4% HW eff)
**Peak achieved**: `iter233` — ~678 TFLOPS (44.8% HW eff, +47% over baseline)

---

## Outperforming Kernels

### iter130 — 477 TFLOPS (F16 Accumulator) `.co`

**Optimization**: Switched accumulator from F32 to F16, reducing register pressure
and enabling higher WGMMA throughput.

```bash
# Build and run (from repo root):
./choreo -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
    benchmark/performance/gemm_sp/gemm_sp_f16_aitune_iter130.co \
    -o /tmp/gemm_sp_f16_aitune_iter130.cute.result
CUDA_VISIBLE_DEVICES=1 bash /tmp/gemm_sp_f16_aitune_iter130.cute.result --execute
```

---

### iter188 — 621 TFLOPS (TMA Metadata Staging) `.cu`

**Optimization**: Replaced inline metadata decode with dedicated TMA-based metadata
staging into shared memory, using a separate `meta_full` barrier for early metadata
availability. Reduced consumer warp stalls by 35%.

```bash
# Build (from repo root):
nvcc -std=c++17 -O2 --use_fast_math -ftz=true -arch=sm_90a \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
    -I runtime \
    -I extern/cutlass/include \
    -I extern/cutlass/tools/util/include \
    -I extern \
    benchmark/performance/gemm_sp/aitune_iter188/gemm_sp_f16_aitune_iter188.cu \
    -o /tmp/gemm_sp_f16_aitune_iter188 \
    -lcuda

# Run:
/tmp/gemm_sp_f16_aitune_iter188

# Run (skip verification, timing only):
/tmp/gemm_sp_f16_aitune_iter188 --skip-verify
```

---

### iter214 — 655 TFLOPS (Batch-4 WGMMA + CTA Swizzle) `.cu`

**Optimization**: Restructured consumer loop to issue all 4 WGMMA operations per
K-step in a single warpgroup batch, combined with M-first CTA rasterization for
improved L2 cache locality of RHS data.

```bash
# Build (from repo root):
nvcc -std=c++17 -O2 --use_fast_math -ftz=true -arch=sm_90a \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
    -I runtime \
    -I extern/cutlass/include \
    -I extern/cutlass/tools/util/include \
    -I extern \
    benchmark/performance/gemm_sp/aitune_iter214/gemm_sp_f16_aitune_iter214.cu \
    -o /tmp/gemm_sp_f16_aitune_iter214 \
    -lcuda

# Run:
/tmp/gemm_sp_f16_aitune_iter214

# Run (skip verification, timing only):
/tmp/gemm_sp_f16_aitune_iter214 --skip-verify
```

---

### iter233 — 678 TFLOPS (Split Metadata Barrier) `.cu` **CURRENT BEST**

**Optimization**: Separated metadata TMA loads onto their own `meta_full` barrier,
decoupled from the main `full` barrier used for LHS+RHS data. This allows the
consumer to begin metadata decode as soon as metadata arrives, overlapping with
the slower LHS/RHS TMA transfers. Achieved 44.8% of H800 theoretical sparse peak.

```bash
# Build (from repo root):
nvcc -std=c++17 -O2 --use_fast_math -ftz=true -arch=sm_90a \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
    -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
    -I runtime \
    -I extern/cutlass/include \
    -I extern/cutlass/tools/util/include \
    -I extern \
    benchmark/performance/gemm_sp/aitune_iter233/gemm_sp_f16_aitune_iter233.cu \
    -o /tmp/gemm_sp_f16_aitune_iter233 \
    -lcuda

# Run:
/tmp/gemm_sp_f16_aitune_iter233

# Run (skip verification, timing only):
/tmp/gemm_sp_f16_aitune_iter233 --skip-verify
```

---

## Notes

- All `.cu` subfolders include a local copy of `choreo.h` for reference.
  The build commands use `-I runtime` to pick up the repo's canonical header.
- CUTLASS headers are required from `extern/cutlass/include/`.
- Timing uses 10 warmup + 500 measurement iterations by default.
  Override with `CHOREO_TIMING_WARMUP` and `CHOREO_TIMING_REPEAT` env vars.
- All measurements taken on NVIDIA H800 PCIe (SM90, 81GB HBM3).
- GPU thermal state affects results; cool GPU reads ~5-10% higher than sustained.
