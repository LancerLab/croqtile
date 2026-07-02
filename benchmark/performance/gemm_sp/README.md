# gemm_sp -- Structured Sparse GEMM Benchmarks

Sparse GEMM with 2:4 structured sparsity (SM90 sparse WGMMA).
LHS is 2:4 sparse (prepacked), RHS is dense. All kernels target SM90a (Hopper).

## Performance Summary

Measured on NVIDIA H800 PCIe (SM90a, 114 SMs).
Default problem: M=4096, N=8192, K=8192.

### E4M3 Sparse GEMM (FP8 input, FP16 output)

H800 PCIe peak F8: 3026 TFLOPS.

| Variant | Tile (MxNxK) | Swizzle | Feature | TFLOPS | ms | HW% |
|---------|-------------|---------|---------|-------:|---:|----:|
| e4m3 | 64x256x128 | 64/128 | Baseline (no prepack) | 434.7 | -- | 14.4 |
| e4m3_aitune_iter001 | 64x256x128 | 64/128 | 1P1C + TMA metadata staging | 766.3 | 0.717 | 25.3 |
| e4m3_aitune_iter016 | 64x256x128 | 64/128 | 1P1C + early empty + merged barrier | 782.8 | 0.702 | 25.9 |
| e4m3_1p1c_swiz128_regctrl | 64x256x128 | 128/128 | 1P1C warpspec + regctrl + prepack | 656.7 | 0.837 | 21.7 |
| e4m3_1p1c_swiz64_regctrl | 64x256x128 | 64/128 | 1P1C warpspec + regctrl + prepack (view/from) | 655.8 | 0.838 | 21.7 |

### F16 Sparse GEMM (FP16 input, FP16 output)

H800 PCIe peak F16: 1513 TFLOPS.

| Variant | Tile (MxNxK) | Swizzle | Feature | TFLOPS | ms | HW% |
|---------|-------------|---------|---------|-------:|---:|----:|
| f16 | 64x128x64 | 32/64 | Baseline (no prepack) | 289.4 | -- | 19.1 |
| f16_dyn_swiz64_128_prepack | 64x128x128 | 64/128 | Dynamic + prepack | 298.4 | 1.842 | 19.7 |
| f16_1p1c_swiz128_prepack_v2 | 64x128x128 | 128/128 | 1P1C warpspec + prepack v2 | 296.7 | 1.853 | 19.6 |
| f16_1p2c_swiz128_prepack_v2 | 128x128x128 | 128/128 | 1P2C warpspec + prepack v2 | 311.9 | 1.763 | 20.6 |
| f16_aitune_iter120 | 128x128x64 | 64/128 | 1P2C + 3-stage (aitune best) | 463.6 | 1.186 | 30.6 |

## Build & Run

Most kernels need `--use-prepack` (prepack variants and aitune kernels):

```bash
./choreo -gs -t cute -arch=sm_90a --use-prepack <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```

Baseline kernels (gemm_sp_e4m3.co, gemm_sp_f16.co) need no extra flags:

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```
