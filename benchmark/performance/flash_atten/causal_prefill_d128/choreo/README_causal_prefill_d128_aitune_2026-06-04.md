# Causal Prefill D=128 AI-Tune Results (2026-06-04)

## Summary

Optimized the causal Flash Attention prefill kernel (D=128, bf16, SM90a/H100)
from **279 TFLOPS to 446 TFLOPS** (+60% total improvement, 29.5% HW efficiency).

| Metric | Baseline | Final Winner |
|--------|----------|--------------|
| TFLOPS (B=2 H=16 SEQ=8192) | 279.25 | 446 (stable 443-449) |
| HW Efficiency | 18.5% | 29.5% |
| Registers/thread | 160 | 157 |
| SMEM/CTA | 164 KB | 164 KB |
| Occupancy | 1 CTA/SM | 1 CTA/SM |

## Winning Optimizations (cumulative)

### 1. iter027: Skip Causal Mask on Full Tiles (+10%)

**279 -> 323 TFLOPS**

For tiles where the entire BLOCK_N column range falls below the causal diagonal
(all positions are valid), skip the per-element mask `frag.apply` entirely.
Only apply masking on the boundary tile where the diagonal intersects.

### 2. iter036: QK/PV WGMMA Overlap via mma.commit Pipelining (+37%)

**323 -> 446 TFLOPS**

Restructure the KV loop to issue QK[n+1] WGMMA while PV[n] is still in the
tensor core pipeline. After committing PV[n], immediately fill acc_s=0 and
issue QK[n+1] before calling mma.wait<0>. This allows the WGMMA hardware to
execute both operations concurrently since they use independent accumulator
registers (acc_o vs acc_s).

Key insight: PV writes to acc_o, QK writes to acc_s -- they don't conflict.
By committing both before waiting, the hardware pipelines them.

## Architecture

- 1 producer + 2 consumer warpgroups (1p2c)
- 2-stage TMA async pipeline for K/V
- 128B swizzle for all shared memory buffers
- Combined KV events (single event per stage for both K and V)
- Online softmax with AllReduce<4,1> cross-thread butterfly reduction

## Build and Run

```bash
cd benchmark/performance/flash_atten/causal_prefill_d128/choreo
bash bench.sh --kernel causal_prefill_d128_aitune_2026-06-04_best.co
```

## Key Findings and Limitations

### What Worked
- **WGMMA overlap (iter036)**: The single biggest win. Explicit mma.commit/wait
  pipelining lets the hardware execute QK and PV concurrently.
- **Conditional mask skip (iter027)**: Simple branch avoidance for causal attention.
- **Combined KV events (iter009)**: Halving barrier count gave ~4% gain.
- **launch_bounds(1) (iter008)**: Minor register allocation hint.

### What Did NOT Work (explored but discarded)
- **BLOCK_N=64**: 2x more loop iterations always outweigh register savings
- **STAGES=3**: Consumer-bound kernel; extra buffering doesn't help
- **STAGES=1**: Producer must run ahead; single-buffer too slow
- **1p3c / 1p1c**: Wrong parallelism granularity for this workload
- **IntraWG overlap (FA3-style)**: Register pressure too high (need 2 full
  accumulator sets simultaneously; C7511 insufficient registers)
- **TMA output store**: Compiler limitation -- rank mismatch on 4D tensor
- **--stmatrix flag**: Added staging overhead exceeds coalescing benefit
- **Pre-scaling Q / fused exp+cast**: Compiler already optimizes via FMA
- **mma.wait<1> pipelining**: Extra sync points degraded performance
- **maxrregcount**: WGMMA accumulators can't be spilled below ~157 regs
- **Various compiler flags** (--hoist-wgmma-arrive, --event-arrive-tx,
  --use-fast-math, -Xptxas --opt-level=4): All within noise

### Remaining Bottlenecks (ceiling at ~446 TFLOPS)
1. **Low occupancy (1 CTA/SM)**: Fixed by WGMMA register requirements (157 regs)
   + SMEM (164KB). Cannot be reduced with current algorithm.
2. **WGMMA serialization from AllReduce**: The `AllReduce<MaxOp,4,1>` butterfly
   shuffle crosses a "function boundary" in ptxas, preventing full WGMMA pipeline.
3. **Serial softmax dependency chain**: Online softmax between QK and PV is
   algorithmically required. The overlap (iter036) mitigates but can't eliminate.

### What Would Help (future work)
- Compiler: inline AllReduce templates to avoid function boundary serialization
- Compiler: configurable WGMMA wait depth for deeper overlap
- Algorithm: persistent kernel for better SM utilization (currently 18 waves)
- Algorithm: split-K or CTA cooperation for higher occupancy

## Source Branch

Full experiment history: `ai-tune/2026-06-04/causal_prefill_d128`
(38 iterations, 5 KEEP, 33+ DISCARD/ABANDON)

## Verification

All KEEP iterations verified with sampled element-wise comparison:
- max_abs_err <= 0.0051 (bf16 precision)
- fail_rate = 0 across 512 samples
- Tested configs: B=2/H=16/SEQ=8192 and B=1/H=16/SEQ=4096
