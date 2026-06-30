# Benchmark Suite

Performance benchmarks for the Choreo compiler, organized by kernel type.

## Directory Structure

### `performance/` - GPU Kernel Performance Benchmarks

High-performance GPU kernel implementations measuring TFLOPS and hardware
efficiency. Each subdirectory targets a specific compute pattern:

| Directory | Description | Target |
|-----------|-------------|--------|
| `matmul/` | Dense GEMM (f16, e4m3, bf16). Includes tutorials (v0-v4) and best-of-breed warpspec variants | SM90 |
| `gemm_sp/` | Sparse GEMM with 2:4 structured sparsity (f16, e4m3) | SM90 |
| `blockscale_gemm/` | Block-scaled FP8 GEMM with per-block scale factors | SM90 |
| `bmm/` | Batched matrix multiplication (f16, fp8) | SM90 |
| `flash_atten/` | Flash Attention variants (prefill, decode, causal, GQA, KV-cache) | SM90 |
| `fused_moe/` | Fused Mixture-of-Experts kernels | SM90 |
| `moe_gemm/` | MoE grouped GEMM (v1-v4 progression) | SM90 |
| `gemv/` | Matrix-vector multiply | SM90 |
| `topk/` | TopK selection for MoE routing | SM90 |

### `cmp-perf/` - Compiler Performance Tests

311 operator-level benchmarks measuring compile-time and scheduling quality
across standard DL operators (batch_norm, concat, conv2d, elemwise_add,
embedding, gelu, layer_norm, matmul, max_pool2d, reduce_mean, relu, reshape,
sigmoid, softmax, transpose).

## Running Benchmarks

### Compile and run a .co benchmark:

```bash
./choreo -gs -t cute -arch=sm_90a [flags] path/to/kernel.co \
  -o /tmp/kernel.cute.result
bash /tmp/kernel.cute.result --execute
```

### Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_TIMING_WARMUP` | 10 | Warmup iterations |
| `CHOREO_TIMING_REPEAT` | 500 | Timed iterations |
| `CHOREO_DISABLE_TIMING` | 0 | Set to 1 to skip timing |
| `CHOREO_SKIP_VERIFY` | 0 | Set to 1 to skip verification |

## Archive

Intermediate AI-tuning iterations (with full .cu/.co sources and binaries) are
preserved on branch `archive/benchmark-aitune-iterations`. The README files in
each subdirectory reference the original experiment branches for full history.
