# gemm_sp e4m3 AI-Tune Results (2026-03-21)

Sparse GEMM (2:4 structured sparsity) with FP8 E4M3 inputs, FP16 accumulation/output.
Target: SM90a (H800 PCIe). Problem size: M=4096, N=8192, K=8192.
H800 PCIe peak F8 TFLOPS: 3026.

## Summary

| Iter | TFLOPS | HW Eff% | vs Baseline | Type | Verification | Key Optimization |
|------|--------|---------|-------------|------|--------------|------------------|
| baseline | 671 | 22.2% | — | .co | ✓ | 1p1c, swizzle128/128, prepack, 2-stage |
| **iter001** | 759 | 25.1% | +13.1% | .co | ✓ | TMA metadata staging |
| **iter016** | 772 | 25.5% | +15.1% | .co | ✓ | early empty signal + merged barrier |
| **iter023** | 811 | 26.8% | +20.9% | .cu | ✓ | SW pipelined consumer loop + warpgroup_wait\<1\> |
| **iter036** | 897 | 29.6% | +33.7% | .cu | ✓ | 1-producer/2-consumer warp specialization |
| **iter040** | 1090 | 36.0% | +62.4% | .cu | ✓ | 3-stage pipeline (from 2-stage), 1p2c |
| **iter068** | 1127 | 37.2% | +67.9% | .cu | ✓ | early empty arrive + all prior optimizations (BEST) |

All `.cu` kernels verified with `choreo::verify_spmm_sampled()` (512 samples,
base_tol=0.5, rel_tol=0.01).

## Build & Run

Each `.cu` subfolder has a `run.sh` that compiles and runs in one step:

### iter001 — TMA metadata staging (.co)

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
  benchmark/performance/gemm_sp/gemm_sp_e4m3_aitune_2026-03-21_iter001.co \
  -o /tmp/e4m3_iter001.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/e4m3_iter001.cute.result --execute
```

### iter016 — early empty + merged barrier (.co)

```bash
./choreo -gs -t cute -arch=sm_90a --use-warpspec --use-prepack \
  benchmark/performance/gemm_sp/gemm_sp_e4m3_aitune_2026-03-21_iter016.co \
  -o /tmp/e4m3_iter016.cute.result
CUDA_VISIBLE_DEVICES=0 bash /tmp/e4m3_iter016.cute.result --execute
```

### iter023 — SW pipeline + warpgroup_wait\<1\> (.cu)

```bash
bash benchmark/performance/gemm_sp/e4m3_aitune_2026-03-21_iter023/run.sh
```

### iter036 — 1-producer/2-consumer warp specialization (.cu)

```bash
bash benchmark/performance/gemm_sp/e4m3_aitune_2026-03-21_iter036/run.sh
```

### iter040 — 3-stage pipeline breakthrough (.cu)

```bash
bash benchmark/performance/gemm_sp/e4m3_aitune_2026-03-21_iter040/run.sh
```

### iter068 — final best: early empty arrive (.cu) **WINNER**

```bash
bash benchmark/performance/gemm_sp/e4m3_aitune_2026-03-21_iter068/run.sh
```

### Common Options

```bash
# Skip verification (timing only):
bash <subfolder>/run.sh --skip-verify

# Skip timing (verify only):
CHOREO_DISABLE_TIMING=1 bash <subfolder>/run.sh
```

## Benchmark Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_TIMING_WARMUP` | 10 | Warmup iterations |
| `CHOREO_TIMING_REPEAT` | 500 | Measurement iterations |
| `CHOREO_DISABLE_TIMING` | 0 | Set to `1` to skip timing, verify only |
| `CHOREO_SKIP_VERIFY` | 0 | Set to `1` to skip correctness check |

## Optimization History

1. **iter001**: Moved sparse metadata loading from register-based to TMA-based shared memory staging, reducing L1/TEX scoreboard pressure.
2. **iter016**: Merged metadata and data barriers into a single `full[]` event, and added early `empty[]` signaling before `mma.commit` to let the producer start loading while the consumer finishes WGMMAs.
3. **iter023**: Restructured the consumer K-loop with software pipelining and `warpgroup_wait<1>()` to overlap WGMMA execution with barrier operations.
4. **iter036**: Changed from 1-producer/1-consumer to 1-producer/2-consumer warp specialization, halving per-consumer RHS bandwidth via shared RHS tile access.
5. **iter040**: Expanded from 2-stage to 3-stage pipeline, increasing producer-consumer overlap by 50% and delivering a +23.8% jump.
6. **iter068**: Moved `empty[stage].arrive()` between the two WGMMA instructions (before `mma.commit`) to signal the producer earlier, enabling better overlap. Combined with all prior optimizations for the best result.

## Source Branch

Full experiment history: `ai-tune/2026-03-21/gemm_sp_e4m3`
