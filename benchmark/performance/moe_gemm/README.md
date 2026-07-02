# moe_gemm -- Mixture-of-Experts GEMM Benchmarks

FP8 (E4M3) to BF16 MoE GEMM kernels for expert-parallel inference.
All kernels target SM90a (Hopper).

## Performance Summary

Measured on NVIDIA H800 PCIe (SM90a, 114 SMs).
Default problem: M=4096 (512 tokens x 8 topk), N=512, K=2048.
H800 PCIe peak F8: 3026 TFLOPS.

| Variant | Feature | TFLOPS | ms | HW% | Status |
|---------|---------|-------:|---:|----:|--------|
| fp8_bf16 (v1) | Baseline MoE GEMM | 43.9 | 0.196 | 1.4 | PASS |
| fp8_bf16_shared_out (v2) | Shared output accumulation | -- | -- | -- | BROKEN (outputs zero) |

## Known Issues

- `moe_gemm_fp8_bf16_shared_out.co`: Pre-existing correctness bug. All output values are zero.
  This is a codegen issue with the shared output accumulation pattern.

## Build & Run

```bash
./choreo -gs -t cute -arch=sm_90a <file>.co -o /tmp/out.cute.result
CHOREO_MAX_VERIFY=1024 bash /tmp/out.cute.result --execute
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| CHOREO_MAX_VERIFY | all | Max elements to verify (use 1024 for faster runs) |
| CHOREO_TIMING_WARMUP | 10 | Warmup iterations |
| CHOREO_TIMING_REPEAT | 500 | Timed iterations |
| CHOREO_DISABLE_TIMING | 0 | Set to 1 to skip timing |
