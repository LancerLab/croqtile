# blockscale_gemm -- Block-Scaled FP8 GEMM Benchmarks

FP8 (E4M3) GEMM with per-block scaling factors. All kernels target SM90a (Hopper).
Tile: 64x128xK128, Swizzle 128.

## Performance Summary

Measured on NVIDIA H800 PCIe (SM90a, 114 SMs).
Default problem: M=2048, N=2048, K=2048.
H800 PCIe peak F8: 3026 TFLOPS.

| Variant | Feature | TFLOPS | ms | HW% |
|---------|---------|-------:|---:|----:|
| dyn_sm90 | Baseline (no warpspec) | 289.8 | 0.059 | 9.6 |
| warpspec_1p1c | 1P1C warpspec | 341.8 | 0.050 | 11.3 |
| warpspec_1p1c_rhs_scale_dma_smem | 1P1C + RHS scale via DMA to smem | 258.7 | 0.066 | 8.6 |
| warpspec_persis_1p1c | 1P1C + persistent | 236.4 | 0.073 | 7.8 |
| warpspec_persis_1p1c_rhs_scale_dma_smem | 1P1C + persistent + RHS scale smem | 185.7 | 0.093 | 6.1 |

## Build & Run

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```
