---
name: tune-choreo-kernel
description: Fine-tune Choreo kernel macro-defined parameters for a fixed problem size, then pick and export the best-performing kernel variant by measured TFLOPS after compile+verification. Use when requests ask to tune one kernel type (for example warpspec/persistent/swizzle variants) against specific M/N/K and keep the best configuration.
---

# Tune Choreo Kernel

Tune a single `.co` kernel by sweeping compile-time macro definitions (typically `#define` constants), verifying correctness for each candidate, and ranking by TFLOPS.

## Workflow

1. Confirm scope.
- Use one base kernel file.
- Use one target problem size `M,N,K`.
- Pick only macros that directly affect tiling/pipeline/cluster shape.

2. Define search space.
- Express tunable macros with repeated `--sweep` flags:
  `--sweep MACRO=v1,v2,v3`
- Use `--set` for fixed macro overrides that should not be searched.
- The tuner auto-overrides `MATMUL_DEFAULT_M/N/K` to the target problem size when those macros exist.

3. Run exhaustive tuning.
- Use script:
  `.codex/skills/tune-choreo-kernel/scripts/tune_kernel_macros.py`
- The script generates candidate `.co` files, calls:
  `.codex/skills/verify-choreo-kernel/scripts/verify_choreo_kernel.sh`
  then parses `TFLOPS` from benchmark logs.
- For the benchmark phase, never inject `CHOREO_TIMING_WARMUP` or
  `CHOREO_TIMING_REPEAT`. Use the kernel/program default timing configuration.

4. Export best kernel.
- The script writes a ranked CSV and copies the best candidate to:
  `<original_name>_m<M>_n<N>_k<K>.co`
- The output filename always includes the problem size.

## Command Template

```bash
.codex/skills/tune-choreo-kernel/scripts/tune_kernel_macros.py \
  --input-co benchmark/performance/blockscale_gemm/blockscale_gemm_e4m3_dyn_sm90_warpspec_1p1c.co \
  --mnk 4096,4096,4096 \
  --sweep MMA_M=64,128 \
  --sweep MMA_N=64,128 \
  --sweep PIPE_STAGES=2,3,4 \
  --compile-arg -t --compile-arg cute \
  --compile-arg -arch=sm_90a
```

## Notes

- Prefer small, hardware-safe search spaces first, then expand.
- For GPU execution in Codex harness, run compile+runtime steps outside sandbox (escalated permissions).
- If no TFLOPS is parsed for a candidate, treat that candidate as failed and keep logs for debugging.
- When reporting tune results, treat benchmark numbers as coming from the
  executable's default warmup/repeat settings.

## Resources

- Script: `scripts/tune_kernel_macros.py`
- Reference examples: `references/tuning-playbook.md`
