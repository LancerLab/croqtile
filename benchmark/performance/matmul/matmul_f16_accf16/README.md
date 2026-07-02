# matmul_f16_accf16 -- FP16 GEMM Benchmark Variants

Dense GEMM: `C[M,N] = A[M,K] * B[K,N]`, FP16 input, FP16 accumulation.
All kernels target SM90a (Hopper) using WGMMA via CuTe backend.

## Performance Summary

Measured on NVIDIA H800 PCIe (SM90a, 132 SMs).
Default timing: 10 warmup, 500 timed iterations.

### Non-Warpspec Kernels (no producer/consumer specialization)

| Variant | Tile (MxNxK) | Stages | Feature | Size | TFLOPS | ms |
|---------|-------------|--------|---------|------|-------:|---:|
| dyn | 64x128x64 | -- | Baseline WGMMA | 2048^3 | 254.8 | 0.067 |
| dyn_mwg | 128x64x64 | -- | Multi-warpgroup (2 WG) | 2048^3 | 239.4 | 0.072 |
| dyn_transout | 64x128x64 | -- | Transposed output | 2048^3 | 256.7 | 0.067 |
| dyn_persis_colmajor | 64x128x64 | -- | Persistent, col-major scheduling | 2048^3 | 108.7 | 0.158 |
| dyn_persis_hilbert | 64x128x64 | -- | Persistent, Hilbert-curve scheduling | 2048^3 | 125.5 | 0.137 |
| dyn_persis_swizzle | 64x128x64 | -- | Persistent, swizzle scheduling | 2048^3 | 124.5 | 0.138 |

### Warpspec Kernels (producer/consumer warp specialization)

| Variant | Tile (MxNxK) | Stages | Consumers | Feature | Size | TFLOPS | ms |
|---------|-------------|--------|-----------|---------|------|-------:|---:|
| warpspec_1p1c | 64x128x64 | 4 | 1 | Base 1P1C | 2048^3 | 319.7 | 0.054 |
| warpspec_1p1c_persis_sta | 64x128x64 | 4 | 1 | 1P1C + persistent | 2048^3 | 243.4 | 0.071 |
| warpspec_1p2c | 128x128x64 | 3 | 2 | Base 1P2C | 2048^3 | 320.1 | 0.054 |
| warpspec_1p2c_n192_regctrl | 128x192x64 | 2 | 2 | WN=192, register control | 8192^3 | 353.7 | 3.109 |
| warpspec_1p3c | 192x128x64 | 2 | 3 | Base 1P3C | 2048^3 | 295.6 | 0.058 |
| warpspec_1p3c_persis_sta | 192x128x64 | 2 | 3 | 1P3C + persistent | 2048^3 | 294.4 | 0.058 |

### AI-Tuned Kernels (aitune 2026-03-23)

| Variant | Tile (MxNxK) | Stages | Consumers | Feature | Size | TFLOPS | ms |
|---------|-------------|--------|-----------|---------|------|-------:|---:|
| iter048_s3_wn176_best | 64x176x64 | 3 | 1 | WN=176, 3-stage | 2048^3 | 345.8 | 0.050 |
| iter050_1p2c_splitout | 128x128x64 | 2 | 2 | Split-output SMEM | 4096^3 | 443.5 | 0.310 |
| iter061_1p2c_so_wn160_kunroll | 128x160x64 | 2 | 2 | WN=160, K-unrolled | 8192^3 | 431.6 | 2.547 |

## Build & Run

All kernels compile with no extra flags:

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| CHOREO_TIMING_WARMUP | 10 | Warmup iterations before timing |
| CHOREO_TIMING_REPEAT | 500 | Number of timed iterations |
| CHOREO_DISABLE_TIMING | 0 | Set to 1 to skip timing |
| CHOREO_SKIP_VERIFY | 0 | Set to 1 to skip correctness check |
