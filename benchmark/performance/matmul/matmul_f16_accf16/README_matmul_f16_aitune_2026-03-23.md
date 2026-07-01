# matmul_f16 AI-Tune Results (2026-03-23)

Dense GEMM FP16 kernel optimization on H800 PCIe (SM90a, 114 SMs).
Multiple problem sizes tested (2048^3, 4096^3, 8192^3).

## Shipped Kernels

| Kernel | TFLOPS | Problem Size | Key Optimization |
|--------|--------|-------------|------------------|
| iter048 | 354.1 | 2048^3 | 1p1c WN=176 STAGES=3 |
| iter050 | ~375 | 4096^3 | 1p2c split-output WN=128 STAGES=2 |
| iter057 | 382.5 | 8192^3 | 1p2c split-output WN=152 non-persistent |
| iter061 | 380.6 | 8192^3 | 1p2c split-output WN=160 K-unrolled (100.5% cuBLAS at 2048^3) |

Baseline on main: **208.7 TFLOPS** (1p1c, WN=128, STAGES=4, 8192^3).
Best at 8192^3: **382.5 TFLOPS** (+83% over baseline).

## Build & Run

All shipped kernels are `.co` files compiled with the Choreo compiler.
Requires `./choreo` to be built first (`make build`).

```bash
# iter061 (best at 8192^3, 100.5% cuBLAS at 2048^3)
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter061_1p2c_so_wn160_kunroll.co \
  -o /tmp/iter061.cute.result && bash /tmp/iter061.cute.result --execute

# iter057 (382.5 TFLOPS at 8192^3)
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter057_1p2c_so_wn152_nonpersis.co \
  -o /tmp/iter057.cute.result && bash /tmp/iter057.cute.result --execute

# iter050 (1p2c split-output at 4096^3)
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter050_1p2c_splitout.co \
  -o /tmp/iter050.cute.result && bash /tmp/iter050.cute.result --execute

# iter048 (WN=176 3-stage at 2048^3)
./choreo -gs -t cute -arch=sm_90a --use-warpspec --stmatrix --hoist-offset --hoist-scale --ptx-barrier --tma-cluster-aware \
  benchmark/performance/matmul/matmul_f16_aitune_2026-03-23_matmul_f16_iter048_s3_wn176_best.co \
  -o /tmp/iter048.cute.result && bash /tmp/iter048.cute.result --execute
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_TIMING_WARMUP` | 10 | Warmup iterations before timing |
| `CHOREO_TIMING_REPEAT` | 500 | Number of timed iterations |
| `CHOREO_DISABLE_TIMING` | 0 | Set to `1` to disable timing |
| `CHOREO_SKIP_VERIFY` | 0 | Set to `1` to skip verification |

## Optimization History

### Phase 1 (iter001-038): 1p1c Warpspec Exploration at 2048^3

- Baseline: 204 TFLOPS (1p1c WN=128 STAGES=4)
- iter004: WN=256 STAGES=2 — 208.9 TFLOPS (reduced SMEM for better occupancy)
- iter023: +ptx-barrier +stmatrix +subspan — 214.3 TFLOPS (+5%)

### Phase 2 (iter043-057): 1p2c Split-Output, Multi-Size

- iter046: 1p1c WN=176 STAGES=2 — 242 TFLOPS at 2048^3 (+13%)
- iter048: 1p1c WN=176 STAGES=3 — 354.1 TFLOPS at 2048^3 (3-stage sweet spot)
- iter050: 1p2c split-output WN=128 — ~375 TFLOPS at 4096^3 (separate SMEM per consumer)
- iter057: 1p2c split-output WN=152 non-persistent — 382.5 TFLOPS at 8192^3

### Phase 3 (iter061-065): WN Sweep at 8192^3

- iter061: WN=160 K-unrolled — 380.6 TFLOPS at 8192^3 (80.7% cuBLAS, 100.5% cuBLAS at 2048^3)
- Compiler: added `--wgmma-wait-depth=N` flag for configurable pipeline depth
- Discovery: WN=168 causes occupancy cliff (SMEM > 228KB, 1 CTA/SM)

### Key Architectural Insights

- 1p2c split-output > shared output for large tiles (eliminates SMEM contention)
- Non-persistent > persistent for 8192^3 (wave quantization is acceptable)
- WN=160 is occupancy-optimal (2 CTAs/SM, 114.7KB SMEM fits in 228KB)
- 3-stage helps at 2048^3 but hurts at 8192^3 (SMEM pressure)

## Source Branch

Full experiment history with all 65 iterations:
`origin/ai-tune/2026-03-23/matmul_f16`
