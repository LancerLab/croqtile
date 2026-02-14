---
name: performance-bottleneck-analysis
description: Analyze bottlenecks from Nsight Systems and Nsight Compute reports.
---

# Performance bottleneck analysis

## Behavior
- Execute the analysis immediately without asking for confirmation.
- Use the default paths and settings unless explicitly told otherwise.
- If a command or report read fails, capture the output and continue to the next diagnostic step.
- Assume `./choreo`, `ncu`, and related CUDA tools are safe to run without confirmation.
- If GPU0 fails (OOM or no device), retry once with `CUDA_VISIBLE_DEVICES=1`.

## Steps
- Use Nsight Systems to identify the dominant phase:
  - Kernel time vs memcpy time vs host time.
- Use Nsight Compute on the dominant kernel with key metrics:
  - `achieved_occupancy`
  - `sm_efficiency`
  - `dram_read_throughput`, `dram_write_throughput`
  - `l2_tex_hit_rate`
  - `warp_execution_efficiency`

## Interpretation checklist
- Memory bound: high DRAM throughput, low SM efficiency, low L2 hit rate.
- Compute bound: high SM efficiency, lower memory pressure.
- Low occupancy: high register use or shared memory limits.
- Divergence: low warp execution efficiency.

## Next actions
- Adjust tile sizes (`WARP_M`, `WARP_N`, `WARP_K`) to improve occupancy.
- Reduce global memory traffic with better shared-memory reuse.
- Re-check correctness after each change.
