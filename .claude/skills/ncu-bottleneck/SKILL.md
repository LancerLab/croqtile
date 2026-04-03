---
name: ncu-bottleneck
description: Dedicated ncu profiling and bottleneck interpretation for GPU kernel optimization. Use before every IDEATE step in croq-tuner to identify the dominant performance bottleneck and ground optimization ideas in measured data.
---

# NCU Bottleneck Analysis

This skill provides the canonical workflow for profiling a GPU kernel with NVIDIA Nsight Compute (ncu) and extracting actionable bottleneck information.

## Profiling Command

```bash
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o <ncu_output_path> \
  <kernel_binary_or_bash_command>
```

For choreo-compiled kernels:
```bash
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o <ncu_output_path> \
  bash <kernel>.cute.result --execute
```

For nvcc-compiled binaries:
```bash
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o <ncu_output_path> \
  ./<binary> --skip-verify
```

Always save the `.ncu-rep` file for reproducibility. Store at the path defined by `ai-tune-artifacts` skill.

## Reading NCU Reports

Extract text summary:
```bash
/usr/local/cuda/bin/ncu --import <file>.ncu-rep --page raw 2>&1 | tee <log_path>
```

Or for specific metrics:
```bash
/usr/local/cuda/bin/ncu --import <file>.ncu-rep \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_lookup_miss.sum 2>&1
```

## Key Metrics

| Metric | What It Measures |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization (overall) |
| `sm__pipe_tensor_op_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor core utilization |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth utilization |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Global load sectors |
| `lts__t_sectors_lookup_miss.sum` | L2 cache misses |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |
| `smsp__inst_executed_pipe_tensor_op_hmma.sum` | HMMA instructions executed |
| `sm__sass_thread_inst_executed_op_memory_128b.sum` | 128-byte memory ops |

## Bottleneck Categories

You MUST classify the kernel into exactly ONE category. `unknown` is FORBIDDEN.

| Category | Indicators | Typical Fixes |
|---|---|---|
| `compute_bound` | Tensor % > 70%, DRAM % < 50% | Already near peak; try wider tiles, unrolling |
| `dram_throughput` | DRAM % > 70%, Tensor % < 50% | Prefetch, TMA descriptor tuning, data layout |
| `smem_throughput` | High SMEM throughput, bank conflicts visible | SMEM padding, swizzle, layout changes |
| `l2_throughput` | High L2 miss rate (miss/total > 30%) | L2 promotion flags, CTA scheduling, tile size for L2 residency |
| `latency_bound` | Low SM %, low DRAM %, high stall cycles | Pipeline depth, barrier placement, nanosleep |
| `occupancy_limited` | Achieved occ << theoretical, high reg count | `--maxrregcount`, `__launch_bounds__`, smaller tiles |
| `instruction_fetch` | High icache misses | Reduce code size, loop unrolling, fewer branches |

## Decision Tree

```
1. Check DRAM throughput %
   ├── > 70%  →  dram_throughput
   └── <= 70%
       2. Check Tensor core %
          ├── > 70%  →  compute_bound
          └── <= 70%
              3. Check SM throughput %
                 ├── > 80% but tensor < 50%  →  smem_throughput or l2_throughput
                 │   (check L2 miss rate to distinguish)
                 └── < 80%
                     4. Check achieved occupancy
                        ├── < 50% of theoretical  →  occupancy_limited
                        └── >= 50%
                            5. Check stall reasons
                               ├── barrier/membar stalls dominant  →  latency_bound
                               └── other  →  inspect warp stall breakdown
```

## Bottleneck-to-Idea Mapping

After identifying the bottleneck, use this table to guide idea generation:

| Bottleneck | Relevant Optimization Ideas |
|---|---|
| `compute_bound` | Warp topology (1p2c vs 1p3c), WGMMA tile shape, loop unroll |
| `dram_throughput` | TMA async, prefetch hints, larger tiles to amortize loads |
| `smem_throughput` | SMEM padding (OUTPUT_PAD), swizzle factor, bank conflict elimination |
| `l2_throughput` | L2 promotion (`l2_128b`, `l2_256b`), CTA scheduling for reuse, tile shape |
| `latency_bound` | Pipeline stages, event/barrier depth, `nanosleep`, `fence_proxy_async` |
| `occupancy_limited` | Register pressure (`--maxrregcount`, `__launch_bounds__`), smaller tiles |

## Integration with Croqtuner FSM

This skill is invoked during the `PROFILE` state of the fsm-engine FSM:

1. Run ncu on current best kernel
2. Save `.ncu-rep` to `ncu/` directory
3. Save text summary to `logs/iter<NNN>_ncu.log`
4. Classify bottleneck using decision tree above
5. Update `loop-state.json`: set `guard_flags.ncu_ran_this_iter = true`, `metrics.last_bottleneck = <category>`

## When to Profile

- **ALWAYS** before the first IDEATE step (iter 1)
- **ALWAYS** after 3+ consecutive discards (strategy change trigger)
- **RECOMMENDED** every 5-10 iterations to track bottleneck shifts
- **ALWAYS** after a significant KEEP (to see what changed)

## Common Pitfalls

1. **Do not profile with verification enabled.** Use `--skip-verify` to avoid timing the CPU reference.
2. **Do not profile with warmup/repeat.** ncu captures a single kernel launch; timing loops distort results.
3. **Check GPU health before profiling.** `nvidia-smi` should show ~0% utilization and low memory usage.
4. **Profile the MATMUL kernel, not host code.** Use `--target-processes all` or specify the kernel name.
