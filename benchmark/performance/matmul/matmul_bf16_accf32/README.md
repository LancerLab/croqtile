# matmul_bf16_accf32 -- BF16 Matrix Multiply Benchmarks

BF16 input, FP32 accumulator GEMM. All kernels target SM90a (Hopper).

## Performance Summary

Measured on NVIDIA H800 PCIe (SM90a, 114 SMs).
Default problem: M=4096, N=4096, K=4096.
H800 PCIe peak F16: 1513 TFLOPS.

| Variant | Feature | TFLOPS | ms | HW% |
|---------|---------|-------:|---:|----:|
| aitune_2026-04-11_iter015 | aitune best | 253.0 | 0.068 | 16.7 |

## Build & Run

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
bash /tmp/out.cute.result --execute
```
