# Blockscale GEMM E4M3 AI-Tune Winners (2026-03-22)

Shipped optimized blockscale GEMM FP8 E4M3 kernels from
`ai-tune/2026-03-22/blockscale_gemm_v2`. All kernels target SM90a (H800 PCIe).

## Summary Table

| Iter | TFLOPS @2k | TFLOPS @4k | Eff% @4k | Key Optimization | Tile |
|------|-----------|-----------|----------|------------------|------|
| baseline | 314.2 | 397.9 | 13.2% | — | M64N128K32 |
| **iter049** | **380** | — | — | TMA overlap: issue next TMA after WGMMA wait, before scale_accumulator | M64N128K32 |
| iter051 | 372 | **602** | 19.9% | N256 WGMMA: M64N256K32 doubles compute per tile | M64N256K32 |
| iter053 | — | **610** | 20.2% | N256 + L2 256B promotion on RHS TMA tensor | M64N256K32 |
| **iter066** | — | **621** | 20.5% | N256 + L2 + prefetch scale_a before WGMMA | M64N256K32 |

**Best @2048^3**: iter049 (380 TFLOPS, +21.0% over baseline)
**Best @4096^3**: iter066 (621 TFLOPS, +56.1% over baseline)

## Build & Run

Each shipped kernel is a self-contained subfolder with `run.sh` that compiles
and runs in one step. No local `choreo.h` copy — uses `$REPO_ROOT/runtime`.

```bash
# iter049: best N128 kernel @2048^3 (380 TFLOPS)
bash benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_aitune_2026-03-22_iter049/run.sh --disable-timing

# iter051: first N256 breakthrough @4096^3 (602 TFLOPS)
bash benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_aitune_2026-03-22_iter051/run.sh --disable-timing

# iter053: N256 + L2 promotion @4096^3 (610 TFLOPS)
bash benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_aitune_2026-03-22_iter053/run.sh --disable-timing

# iter066: BEST overall @4096^3 (621 TFLOPS)
bash benchmark/performance/blockscale_gemm_v2/blockscale_gemm_e4m3_aitune_2026-03-22_iter066/run.sh --disable-timing
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CHOREO_TIMING_WARMUP` | Number of warmup iterations (default: 100) |
| `CHOREO_TIMING_REPEAT` | Number of timed iterations (default: 1000) |
| `CHOREO_DISABLE_TIMING` | Set to `1` to skip timing |
| `CHOREO_SKIP_VERIFY` | Set to `1` to skip verification |

## Verification

All kernels use **sampled verification** with 512 coprime-stride samples
over the M×N output. Each sample computes a full FP32 reference dot product
with blockscale factors. Tolerances: base_tol=0.5, rel_tol=0.01 (FP8 E4M3
input, FP16 accumulator).

## Optimization History

1. **iter049** (overlap_tma_scale): Issue next K-block's TMA loads immediately
   after WGMMA wait completes, overlapping TMA latency with scale_accumulator
   computation. +21% @2k from hiding TMA latency.

2. **iter051** (n256_wgmma): Switch from M64N128K32 to M64N256K32 WGMMA,
   doubling compute per CTA tile. Requires 40KB SMEM. Best at 4096^3 where
   the reduced grid (256 vs 512 CTAs) doesn't hurt; worse at 2048^3.

3. **iter053** (n256_l2rhs): Enable `CU_TENSOR_MAP_L2_PROMOTION_L2_256B`
   on the RHS TMA tensor map, improving L2 hit rate on larger RHS transfers.

4. **iter066** (n256_prefetch): Prefetch per-row `scale_a` values via `__ldg`
   into registers before the WGMMA loop body, hiding `__ldg` latency behind
   WGMMA execution. Combined with iter053 L2 promotion for best-ever result.

## Source Branch

Full experiment history with 71 iterations: `ai-tune/2026-03-22/blockscale_gemm_v2`
