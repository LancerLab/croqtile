# Tuning Playbook

## Macro Spec Format

- Sweep space: `--sweep MACRO=v1,v2,v3`
- Fixed override: `--set MACRO=value`
- Problem size: `--mnk M,N,K`

## Typical Tunables

- Tile shape: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- Warp shape: `WARP_M`, `WARP_N`, `WARP_K`
- Pipeline: `PIPE_STAGES`
- Warp specialization split: `PRODUCER_WARPS`, `CONSUMER_WARPS`
- Cluster/cga shape: `CLUSTER_M`, `CLUSTER_N`
- Swizzle mode: `SWIZZLE_MODE`

Use names that exactly match `#define` macros in the kernel source.

## Example

```bash
.codex/skills/tune-choreo-kernel/scripts/tune_kernel_macros.py \
  --input-co benchmark/performance/blockscale_gemm/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  --mnk 2048,8192,4096 \
  --sweep BLOCK_M=64,128 \
  --sweep BLOCK_N=64,128 \
  --sweep PIPE_STAGES=2,3,4 \
  --set SPLIT_K=1 \
  --compile-arg -t --compile-arg cute \
  --compile-arg -arch=sm_90a --compile-arg --use-warpspec
```

## Output

- Candidate files and logs under `build/skill-logs/tune-choreo-kernel/run_<timestamp>/`
- Ranked table: `results.csv`
- Best kernel copy: `<input_stem>_m<M>_n<N>_k<K>.co`

## Benchmark Timing

- During tuning, benchmark runs should use the executable's default timing
  configuration.
- Do not inject `CHOREO_TIMING_WARMUP` or `CHOREO_TIMING_REPEAT` unless the
  user explicitly asks for those overrides.
