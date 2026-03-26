# gemm_sp_f16 AI-Tune Results (2026-03-25)

Sparse GEMM FP16 kernel optimization on H800 PCIe (SM90a).
Problem size: M=4096, N=8192, K=8192 (structured 2:4 sparsity).

## Shipped Kernels

| Kernel | TFLOPS | HW Eff. | Key Optimization |
|--------|--------|---------|------------------|
| iter137 | 543 | 35.9% | 1p2c 3-stage + vec2 metadata + hoisted __ldg + unroll24 + ftz |
| iter143 | 655 | 43.3% | TK128 + TMA metadata staging + split RHS TMA + B128 LHS swizzle + 4 WGMMAs/batch |

Baseline on main: **368 TFLOPS** (1p1c, swizzle64, TK64, 2-stage).
Best result: **655 TFLOPS** (+78% over baseline, 43.3% HW efficiency).

## Build & Run

Both shipped kernels are `.cu` files with self-contained `run.sh` scripts.
Each `run.sh` compiles and runs in one step:

```bash
# iter137 (543 TFLOPS) — organic optimization milestone
bash benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-25_iter137/run.sh

# iter143 (655 TFLOPS) — best result
bash benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-25_iter143/run.sh
```

### Options

Pass arguments after `run.sh`:
- `--disable-timing` — run once without timing loop
- `--skip-verify` — skip correctness verification

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_TIMING_WARMUP` | 10 | Warmup iterations before timing |
| `CHOREO_TIMING_REPEAT` | 500 | Number of timed iterations |
| `CHOREO_DISABLE_TIMING` | 0 | Set to `1` to disable timing |
| `CHOREO_SKIP_VERIFY` | 0 | Set to `1` to skip verification |

## Optimization History

### iter137 (543 TFLOPS) — Organic Optimization Milestone

Built incrementally from the baseline through 137 iterations of optimization:

1. **warpgroup_wait<1>** (iter023, +4%): WGMMA pipeline overlap across K iterations
2. **--wgmma-split-batch** (iter052, +5%): Compiler flag for 2-batch WGMMA with wait<2>
3. **__ldg metadata** (iter074, +0.5%): Read-only cache for metadata loads
4. **1p2c + 3-stage pipeline** (iter120, +9%): Two consumer warpgroups share RHS bandwidth, 3-stage pipeline hides TMA latency
5. **1p2c + 4-stage + stmatrix** (iter126, +2%): Deeper pipeline + stmatrix for bank conflict reduction in output store
6. **L2 128B promotion** (iter129, +0.7%): TMA L2 promotion for LHS and RHS
7. **uint2 vectorized metadata** (iter130, +8%): Halves uncoalesced global transactions for metadata
8. **3-stage + vec2 + L2 + stmatrix** (iter133, +2.5%): Combination with 3-stage uses less SMEM, freeing L1 for metadata cache
9. **Hoisted metadata __ldg** (iter135, +7%): Metadata load issued before full barrier wait, overlapping with TMA transfer
10. **unroll 24** (iter136, +3%): Loop unroll factor sweep found optimal at 24
11. **-ftz=true** (iter137, +0.4%): Flush-to-zero for denormal FP16 operations

### iter143 (655 TFLOPS) — Adapted Reference Kernel

Adapted from `origin/ai-tune/2026-03-21/gemm_sp_f16` reference branch (iter233), with full correctness verification:

- **TILE_K=128**: Larger K tile for more compute per TMA load
- **TMA metadata staging**: Metadata loaded via TMA into shared memory with dedicated `meta_full` barrier
- **Split RHS TMA**: Two `{64,256}` TMA boxes for RHS instead of one
- **B128 LHS swizzle**: 128-byte swizzle for LHS packed data
- **4 WGMMAs per batch**: 4 WGMMA operations per commit group (2 K-halves × 2 consumers)
- **PTX mbarrier**: Direct PTX barrier instructions for TMA synchronization
- **Output padding**: SPMM_OUTPUT_PAD=8 for shared memory bank conflict avoidance

## Source Branch

Full experiment history with all 153 iterations:
`origin/ai-tune/2026-03-25/gemm_sp_f16_opus_max`
