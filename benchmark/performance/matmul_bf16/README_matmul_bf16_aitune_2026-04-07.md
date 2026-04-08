# matmul_bf16 AI-Tune Results — 2026-04-07

## Summary

| Kernel | TFLOPS | Eff % | Improvement | Config |
|--------|--------|-------|-------------|--------|
| Baseline (`matmul_bf16_tuned.co`) | 304.2 | 20.1% | — | 1p2c N=152 S2 |
| **`matmul_bf16_aitune_2026-04-07_iter024.co`** | **351** | **23.2%** | **+16%** | 1p1c N=160 S3 |

- **Target**: 2048×2048×2048 bf16 GEMM with f32 accumulator
- **Architecture**: SM90a (H100/H800)
- **Session**: 50 iterations across Choreo-level and CUDA-level optimizations

## Shipped Kernel

### `matmul_bf16_aitune_2026-04-07_iter024.co`

1-producer, 1-consumer warp-specialized matmul with WARP_N=160, STAGES=3.

**Key parameters:**
- `WARP_M=64`, `WARP_N=160`, `TILE_M=64`, `TILE_K=64`, `WARP_K=16`
- `SWIZ=128`, `STAGES=3`, `CONSUMER_COUNT=1`
- 2 warpgroups (256 threads/block), 2 blocks/SM, ~100KB smem/block

**Build & run:**
```bash
./choreo -gs -t cute -arch=sm_90a \
  benchmark/performance/matmul_bf16/matmul_bf16_aitune_2026-04-07_iter024.co \
  -o /tmp/matmul_bf16_iter024.cute.result

CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500 \
  bash /tmp/matmul_bf16_iter024.cute.result --execute
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_TIMING_WARMUP` | 5 | Warmup iterations before timing |
| `CHOREO_TIMING_REPEAT` | 20 | Timed iterations for average |
| `CHOREO_DISABLE_TIMING` | 0 | Set to `1` to skip timing entirely |
| `CHOREO_SKIP_VERIFY` | 0 | Set to `1` to skip numerical verification |

## Key Optimization: 1p2c → 1p1c

The decisive breakthrough was switching from 1-producer/2-consumer (1p2c) to
1-producer/1-consumer (1p1c) warp specialization at iteration 21:

- **1p2c** uses 3 warpgroups (384 threads) → only 1 block per SM
- **1p1c** uses 2 warpgroups (256 threads) → **2 blocks per SM**

The doubled occupancy more than compensates for each block processing half the
M-tiles. The pipeline remains well-balanced: STAGES=3 with 100KB smem per block
fits two blocks within the 228KB SM90 shared memory limit.

## Optimization History (50 iterations)

### Phase 1: Initial exploration (iter000–iter012)
- Tested 1p2c/1p3c patterns, N={64,128,256}, STAGES={2–5}, persistent kernels
- Best: iter010 (1p2c N=128 S4) at 240.2 TFLOPS

### Phase 2: User baseline adoption (iter015)
- User's hand-tuned `matmul_bf16_tuned.co` (1p2c N=152 S2) at 304.2 TFLOPS
- Became the new baseline for comparison

### Phase 3: Tile-width sweep (iter017–iter024)
- Swept WARP_N from 128 to 256 in both 1p2c and 1p1c configurations
- **iter021**: 1p1c N=160 S2 → 340.8 TFLOPS (first 1p1c breakthrough)
- **iter024**: 1p1c N=160 S3 → 348–353 TFLOPS (best Choreo kernel)

### Phase 4: Micro-optimizations (iter025–iter050)
- CUDA-level: register cap, fast math, L2 promotion, launch bounds, fence hoist
- Choreo-level: different N values (128–192), TILE_K=32, TBC, MWG, global stores
- Barrier protocol experiments: leader-only, delayed empty, WGMMA pipelining
- **iter045**: CUDA fence hoist gave consistent +0.23% (353.6 vs 352.8 avg)
- All other modifications either matched or degraded performance

### Remaining bottlenecks (architectural)
1. **4-way smem bank conflicts** on `mma.store` epilogue (f32→bf16 store pattern
   with stride 160 maps every 2nd row to the same banks; incompatible with TMA
   output padding)
2. **58% partial wave tail** (416 blocks / 264 active slots per wave)

## Source Branch

Full experiment history: `ai-tune/2026-04-07/matmul_bf16`
