# Causal Prefill D=128 AI-Tune Results (2026-06-04)

## Summary

Optimized the causal Flash Attention prefill kernel (D=128, bf16, SM90a/H100)
from **279 TFLOPS to 458.5 TFLOPS** (+64% total improvement, 30.3% HW efficiency).

| Metric | Baseline | Final Winner (iter055) |
|--------|----------|------------------------|
| TFLOPS (B=2 H=16 SEQ=8192) | 279.25 | 458.5 (peak ~475) |
| HW Efficiency | 18.5% | 30.3% |
| Registers/thread | 160 | 158 |
| SMEM/CTA | 164 KB | 164 KB |
| Occupancy | 1 CTA/SM | 1 CTA/SM |

## Shipped Artifacts

### 1. `causal_prefill_d128_aitune_2026-06-04_best.co` (446 TFLOPS)

The winning `.co` kernel incorporating all Choreo-level optimizations (iter008 through
iter036). This is the pure Choreo source that can be compiled with `./choreo`.

```bash
bash bench.sh --kernel causal_prefill_d128_aitune_2026-06-04_best.co
```

### 2. `causal_prefill_d128_aitune_2026-06-04_iter055/` (458.5 TFLOPS)

The `.cu`-level modification of the best.co generated code, adding a stmatrix epilogue
with vectorized global stores. Self-contained subfolder with `run.sh`.

```bash
bash causal_prefill_d128_aitune_2026-06-04_iter055/run.sh
```

## Winning Optimizations (cumulative)

### 1. iter008: launch_bounds(1) (+0.4%)

**279.25 -> 280.27 TFLOPS**

Hint to the compiler that only 1 CTA runs per SM, enabling better register allocation.

### 2. iter009: Combined K/V Events (+3.6%)

**280.27 -> 290.35 TFLOPS**

Replace separate K and V pipeline events with a single combined event per stage,
halving barrier overhead.

### 3. iter012: Reorder Cast Before Rescale (+0.9%)

**290.35 -> 293.01 TFLOPS**

Cast `acc_s` to bf16 before rescaling `acc_o`, reducing register pressure on the
critical path.

### 4. iter027: Skip Causal Mask on Full Tiles (+10.4%)

**293.01 -> 323.47 TFLOPS**

For tiles where the entire BLOCK_N column range falls below the causal diagonal
(all positions are valid), skip the per-element mask `frag.apply` entirely.
Only apply masking on the boundary tile where the diagonal intersects.

### 5. iter036: QK/PV WGMMA Overlap via mma.commit Pipelining (+37.4%)

**323.47 -> 444.60 TFLOPS**

Restructure the KV loop to issue QK[n+1] WGMMA while PV[n] is still in the
tensor core pipeline. After committing PV[n], immediately fill acc_s=0 and
issue QK[n+1] before calling mma.wait<0>. PV writes to acc_o, QK writes to
acc_s -- they don't conflict.

### 6. iter055: stmatrix Epilogue + Vectorized Stores (+3.1%)

**444.60 -> 458.5 TFLOPS** (`.cu`-level modification)

Replace the default per-element epilogue store with:
1. `stmatrix.sync.aligned.m8n8.x4.b16` to write f32 WGMMA accumulators as bf16
   directly to shared memory (reusing V buffer, separate per consumer)
2. Vectorized `int4` copy (128-bit) from shared memory to global memory

This eliminates uncoalesced global stores (reduced LG throttle stalls from 10%+ to ~0%),
confirmed by NCU profiling showing 75% -> 50% uncoalesced store reduction.

## Architecture

- 1 producer + 2 consumer warpgroups (1p2c)
- 2-stage TMA async pipeline for K/V
- 128B swizzle for all shared memory buffers
- Combined KV events (single event per stage for both K and V)
- Online softmax with AllReduce<4,1> cross-thread butterfly reduction

## Key Findings and Limitations

### What Worked
- **WGMMA overlap (iter036)**: The single biggest win (+37%). Explicit mma.commit/wait
  pipelining lets the hardware execute QK and PV concurrently.
- **Conditional mask skip (iter027)**: Simple branch avoidance (+10%) for causal attention.
- **stmatrix epilogue (iter055)**: Coalesced stores via shared memory staging (+3%).
- **Combined KV events (iter009)**: Halving barrier count gave ~4% gain.
- **Reorder cast (iter012)**: Scheduling optimization for register pressure.

### What Did NOT Work (57 iterations explored, 51 discarded/abandoned)
- **BLOCK_N=64**: 2x more loop iterations always outweigh register savings
- **STAGES=3 / STAGES=1**: Consumer-bound; extra buffering doesn't help
- **1p3c / 1p1c**: Wrong parallelism granularity for this workload
- **IntraWG overlap (FA3-style)**: Register pressure too high (C7511)
- **TMA output store**: Compiler limitation -- rank mismatch on 4D tensor
- **maxrregcount (80-144)**: SMEM=164KB prevents 2 CTAs regardless of reg count
- **setmaxnreg dynamic register reallocation**: ptxas ignores (C7507)
- **L2 cache promotion hints on TMA**: No measurable effect
- **Explicit FMA in softmax**: Compiler already contracts with --use-fast-math
- **Coalesced warp-level stores (uint2/uint4)**: stmatrix SMEM layout incompatible
- **Persistent kernel**: TMA descriptors bound to grid dims, can't do dynamic scheduling
- **nanosleep in producer**: No effect on barrier stalls
- **Various compiler flags** (--hoist-wgmma-arrive, --event-arrive-tx, -Xptxas --opt-level=4): All within noise

### Remaining Bottlenecks (NCU profile of iter055)
1. **Wait stalls (18.1%)**: WGMMA pipeline / event waits
2. **Barrier stalls (15.5%)**: Warpgroup synchronization
3. **Long Scoreboard (10.5%)**: Global memory latency
4. **MIO Throttle (9%)**: Memory I/O pipe saturation
5. **LG Throttle (6.9%)**: Remaining uncoalesced stores (boundary tiles)
6. **Low occupancy (1 CTA/SM)**: Fixed by 158 regs + 164KB SMEM

### What Would Help (future work)
- Compiler: inline AllReduce templates to avoid function boundary WGMMA serialization
- Compiler: native stmatrix epilogue support (--stmatrix flag with vectorized global copy)
- Compiler: configurable WGMMA wait depth for deeper overlap
- Algorithm: split-K or CTA cooperation for higher occupancy
- Algorithm: 3-phase causal loop (full tiles / boundary / skip) to reduce divergence

## Source Branch

Full experiment history: `ai-tune/2026-06-04/causal_prefill_d128`
(57 iterations, 6 KEEP, 51 DISCARD/ABANDON)

## Verification

All KEEP iterations verified with sampled element-wise comparison:
- max_abs_err <= 0.0051 (bf16 precision)
- fail_rate = 0 across 512 samples
- Tested configs: B=2/H=16/SEQ=8192 and B=1/H=16/SEQ=4096
