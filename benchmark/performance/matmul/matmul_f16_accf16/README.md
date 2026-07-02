# matmul_f16_accf16 -- FP16 Matrix Multiply Benchmarks

FP16 input, FP16 accumulator GEMM variants. All kernels target SM90a (Hopper).

## Performance Summary

Measured on NVIDIA H800 PCIe (SM90a, 114 SMs).
Default problem: M=4096, N=4096, K=4096.
H800 PCIe peak F16: 1513 TFLOPS.

| Variant | Feature | TFLOPS | ms | HW% |
|---------|---------|-------:|---:|----:|
| dyn | Baseline dynamic | 173.5 | 0.099 | 11.5 |
| dyn_mwg | Multi-warp-group | 169.1 | 0.102 | 11.2 |
| dyn_transout | Transposed output | 174.7 | 0.098 | 11.5 |
| dyn_persis_colmajor | Persistent + col-major | 72.2 | 0.238 | 4.8 |
| dyn_persis_hilbert | Persistent + Hilbert curve | 83.4 | 0.206 | 5.5 |
| dyn_persis_swizzle | Persistent + swizzle | 84.7 | 0.203 | 5.6 |
| dyn_warpspec_1p1c | 1P1C warpspec | 218.4 | 0.079 | 14.4 |
| dyn_warpspec_1p1c_persis_sta | 1P1C + persistent + static | 173.9 | 0.099 | 11.5 |
| dyn_warpspec_1p2c | 1P2C warpspec | 221.8 | 0.077 | 14.7 |
| dyn_warpspec_1p2c_n192_regctrl | 1P2C + N192 + regctrl | 282.5 | 3.892 | 18.7 |
| dyn_warpspec_1p3c | 1P3C warpspec | 204.2 | 0.084 | 13.5 |
| dyn_warpspec_1p3c_persis_sta | 1P3C + persistent + static | 201.8 | 0.085 | 13.3 |
| aitune_iter048_s3_wn176_best | aitune: 3-stage, wn176 | 246.3 | 0.070 | 16.3 |
| aitune_iter050_1p2c_splitout | aitune: 1P2C + split output | 318.3 | 0.432 | 21.0 |
| aitune_iter061_1p2c_so_wn160_kunroll | aitune: 1P2C + wn160 + K-unroll | 370.1 | 2.971 | 24.5 |

## Build & Run

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```
