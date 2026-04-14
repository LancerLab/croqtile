# matmul_bf16 AI-Tune Results — 2026-04-11

## Summary

This session continued from the currently shipped 2026-04-07 matmul_bf16 winner on main and focused on the remaining epilogue shared-store bottleneck in the BF16 input, FP32 accumulate, row-row Hopper kernel.

Historical experiment results on the ai-tune branch showed two verified winners over the 2026-04-07 shipment. After preparing ship-form `.co` files and rerunning them on the current branch tip, both kernels still beat the currently shipped kernel when all accompanying branch-side compiler/runtime changes are included, but the absolute TFLOPS numbers regressed materially relative to the original experiment ledger.

| Kernel | Historical experiment TFLOPS | Current branch-tip verify TFLOPS | Key optimization |
|--------|------------------------------|----------------------------------|------------------|
| Main shipped winner (`matmul_bf16_aitune_2026-04-07_iter024.co`) | 351.000 | 335.792 | 1p1c, WARP_N=160, STAGES=3 |
| `matmul_bf16_aitune_2026-04-11_iter015.co` | 352.171 | 347.669 | Flattened staged lhs/rhs buffers with explicit stage subspan views |
| `matmul_bf16_aitune_2026-04-11_iter027.co` | 352.504 | 347.000 | Added a 16-column pad to the shared-output epilogue layout |

## Ship Status

- These two kernels are the only verified winners from this branch that beat the current main shipment once revalidated on the current branch toolchain.
- They depend on the compiler-backed `--stmatrix` BF16 shared-output store path developed on `ai-tune/2026-04-11/matmul_bf16`.
- The branch-tip reruns show an absolute performance regression across the family versus the historical experiment logs, so this summary should be treated as a local wrap-up and not as a clean push-ready shipment.
- Nothing from this summary was pushed to any remote.

## Build & Run

### `matmul_bf16_aitune_2026-04-11_iter015.co`

```bash
./choreo -gs -t cute -arch=sm_90a --stmatrix \
  benchmark/performance/matmul_bf16/matmul_bf16_aitune_2026-04-11_iter015.co \
  -o build/agent_tmp/matmul_bf16_aitune_2026-04-11_iter015.cute.result

CUDA_VISIBLE_DEVICES=0 CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 \
  bash build/agent_tmp/matmul_bf16_aitune_2026-04-11_iter015.cute.result --execute
```

### `matmul_bf16_aitune_2026-04-11_iter027.co`

```bash
./choreo -gs -t cute -arch=sm_90a --stmatrix \
  benchmark/performance/matmul_bf16/matmul_bf16_aitune_2026-04-11_iter027.co \
  -o build/agent_tmp/matmul_bf16_aitune_2026-04-11_iter027.cute.result

CUDA_VISIBLE_DEVICES=0 CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 \
  bash build/agent_tmp/matmul_bf16_aitune_2026-04-11_iter027.cute.result --execute
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_TIMING_WARMUP` | 5 | Warmup iterations before timing |
| `CHOREO_TIMING_REPEAT` | 20 | Timed iterations for average |
| `CHOREO_DISABLE_TIMING` | 0 | Set to `1` to skip timing |
| `CHOREO_SKIP_VERIFY` | 0 | Set to `1` to skip sampled verification |

## Optimization History

### `matmul_bf16_aitune_2026-04-11_iter015.co`

This was the first verified winner above the existing main shipment in the historical experiment logs. It kept the 1p1c, WARP_N=160, STAGES=3 structure but flattened the staged lhs/rhs shared-memory tensors and accessed them through explicit stage subspan views. The change preserved the new compiler-backed stmatrix epilogue and lifted throughput while keeping the same basic occupancy regime. On the current branch tip it is also the faster of the two prepared ship-form kernels.

### `matmul_bf16_aitune_2026-04-11_iter027.co`

This was the best verified kernel from the historical experiment branch. Source-correlated profiling showed the dominant residual issue had moved into the epilogue shared-store path, so the winning follow-up changed only the shared-output row stride by padding the output tile with 16 extra columns. That small layout perturbation gave the best measured result during the experiment, although the branch-tip rerun dropped slightly below iter015.

## Rejected Follow-Ups

- Larger output padding (`iter029`, `iter030`) regressed.
- Transposed epilogues (`iter031`) were slower even after compiler support was added.
- Nonzero-base or padded-transposed shared-output views (`iter032`, `iter033`) compiled but failed verification and should not be shipped.

## Verification

The prepared ship kernels use `choreo::verify_matmul_row_row_sampled()` with:

- `num_samples = 512`
- `base_tol = 1.0f`
- `rel_tol = 0.01f`

## Source Branch

Full experiment history and compiler/runtime support work: `ai-tune/2026-04-11/matmul_bf16`