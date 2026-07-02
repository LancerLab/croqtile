# matmul_e4m3_accf16 -- FP8 E4M3 Matrix Multiply Benchmarks

FP8 (E4M3) input, FP16 accumulator GEMM variants. All kernels target SM90a (Hopper).

## Performance Summary

Measured on NVIDIA H800 PCIe (SM90a, 114 SMs).
Default problem: M=4096, N=4096, K=4096.
H800 PCIe peak F8: 3026 TFLOPS.

| Variant | Feature | TFLOPS | ms | HW% |
|---------|---------|-------:|---:|----:|
| dyn | Baseline dynamic | 298.7 | 0.058 | 9.9 |
| dyn_mwg | Multi-warp-group | 294.8 | 0.058 | 9.7 |
| dyn_persis_colmajor | Persistent + col-major | 130.8 | 0.131 | 4.3 |
| dyn_persis_hilbert | Persistent + Hilbert curve | 151.5 | 0.113 | 5.0 |
| dyn_persis_sta | Persistent + static | 130.7 | 0.131 | 4.3 |
| dyn_persis_swizzle | Persistent + swizzle | 150.3 | 0.114 | 5.0 |
| dyn_warpspec_1p1c | 1P1C warpspec | 357.7 | 0.048 | 11.8 |

## Build & Run

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```
