# MoE GEMM FP8→BF16 AI-Tune Results (2026-03-27)

## Summary

| Kernel | TFLOPS | Regs | Blocks/SM | Occupancy | Key Optimization |
|--------|--------|------|-----------|-----------|-----------------|
| Baseline (v1, WARP_N=128) | 5.76 | 150 | 3 | 18.75% | — |
| **iter012 (WARP_N=64)** | **6.22** | **72** | **7** | **43.75%** | `__launch_bounds__(128, 7)` eliminates `--maxrregcount` flag |

**Problem size**: M=384 (192 tokens × TOPK=2), N=512, K=2048, 256 experts, Poisson routing (seed=42).
**Architecture**: SM90a (H100/H800).
**Improvement**: +8% TFLOPS over baseline.

## Shipped Kernels

### 1. `.co` source (Choreo DSL)

**File**: `moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co`

```bash
EXTRA_TARGET_CFLAGS="--maxrregcount=72 -Xptxas --allow-expensive-optimizations=true" \
  ./choreo -gs -t cute -arch=sm_90a \
  benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co \
  -o /tmp/moe_aitune.cute.result

bash /tmp/moe_aitune.cute.result --execute
```

### 2. `.cu` self-contained (with `__launch_bounds__`)

**Folder**: `moe_gemm_fp8_bf16_aitune_2026-03-27_iter012/`

The `.cu` file has `__launch_bounds__(128, 7)` on the kernel function, making
`--maxrregcount=72` unnecessary. Compile and run in one step:

```bash
bash benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012/run.sh --execute
```

No external flags required — register limit is embedded via `__launch_bounds__`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_TIMING_WARMUP` | 10 | Warmup iterations |
| `CHOREO_TIMING_REPEAT` | 500 | Timing iterations |
| `CHOREO_DISABLE_TIMING` | 0 | Set to 1 to skip timing |
| `CHOREO_SKIP_VERIFY` | 0 | Set to 1 to skip verification |
| `CUDA_VISIBLE_DEVICES` | (all) | Select GPU |

## Optimization History

1. **Baseline** (5.76 TFLOPS): 256 experts, FP8 WGMMA 64×128×32, DMA+TMA loads, 150 regs/thread, 3 blocks/SM. DRAM-bound at 71% HBM bandwidth.
2. **iter010** (6.05 TFLOPS, +5%): `--maxrregcount=128` reduced registers 150→128, enabling 4 blocks/SM.
3. **iter012** (6.22 TFLOPS, +8%): WARP_N=64 halves accumulator registers (64→32 fp32), combined with `--maxrregcount=72` achieves 72 regs, 0 spill, 7 blocks/SM. Near-optimal for this problem size (72% HBM efficiency).

### Why WARP_N=64 with BLOCK_SIZE_N=128 (Qwen 3.5 compatibility)

The weight quantization uses `weight_block_size={128, 128}`, meaning one `scale_b` value covers a 128×128 block. With WARP_N=64, two adjacent N-tiles share one scale factor. The kernel handles this via:

```
sc_b = scale_b.at(eid * cdiv(N, BLOCK_SIZE_N) + block_n / SCALE_N_RATIO, iv_k);
```

where `SCALE_N_RATIO = BLOCK_SIZE_N / MATMUL_WARP_N = 128/64 = 2`.

## Source Branch

Full experiment history: `ai-tune/2026-03-27/moe_gemm` and `ai-tune/2026-03-27/moe_gemm-resume-1`
