---
name: verify-choreo-kernel
description: Compile Choreo `.co` kernels into runnable binaries, execute correctness verification, and collect targeted debug diagnostics when compile or runtime fails. Use when requests ask to run/verify kernel correctness, reproduce runtime failures, or triage validation errors after kernel changes.
---

# Verify Choreo Kernel

Compile `.co` to executables, run correctness verification first, then run a large-size benchmark for performance numbers.

## Workflow

0. Run GPU steps outside sandbox.
- For this skill, always execute compile+run commands with escalated permissions (`sandbox_permissions: "require_escalated"`), so CUDA/NVML can access real GPUs.
- If a run was started inside sandbox and logs show `CUDA failure: 304` or `Failed to initialize NVML`, rerun the same command outside sandbox before debugging kernel logic.

1. Build executable (not object-only).
- Use `./build/choreo <input.co> -o <binary>` so output can be executed.
- If no target is specified and input file name contains `sm90`, default to `-t cute -arch=sm_90a`.
- Warpspec mode is auto-detected by the compiler; do not pass `--use-warpspec` (the flag has been removed).
- If input file name contains `prepack`, append `--use-prepack` automatically unless already provided.

2. Run quick verification.
- Execute with timing disabled by default: `--disable-timing`.
- Set `CHOREO_DISABLE_TIMING=1`, `CHOREO_TIMING_WARMUP=0`, `CHOREO_TIMING_REPEAT=1`.
- Prefer a smaller compile-time matmul shape for fast verification (`MATMUL_DEFAULT_M/N/K`, default `256,256,256`).
- Select an idle GPU automatically before runtime unless GPU is explicitly pinned.
- Enforce runtime timeout (`--timeout-sec`, default `120`) with kill grace period to avoid deadlock hangs.
- Treat output containing `Test Passed` as verification success.
- This `0/1` timing setup is only for the quick verify step, not for the
  post-verify benchmark step.

3. Run benchmark after verify.
- Compile and run a bench binary after verification succeeds.
- Use bench shape `2048,2048,2048` by default for throughput measurement.
- Pass `--skip-verify` and keep timing enabled to report `Timing avg ms`, `TFLOPS`, and efficiency.
- Never override benchmark warmup/repeat counts in this workflow; always let the kernel/program default timing configuration apply.
- Enforce bench timeout (`--bench-timeout-sec`, default `180`) to avoid deadlock hangs.

4. Debug on failure.
- For compile failures: capture compile log and run focused frontend diagnostics (`-i`).
- For runtime failures: capture run log, detect CUDA/NVML environment issues, and provide clear hints.
- If runtime log reports OOM and `CHOREO_VERIFY_OOM_GPU` is set, retry once with `CUDA_VISIBLE_DEVICES=<CHOREO_VERIFY_OOM_GPU>`.
- Emit log files under `build/skill-logs/verify-choreo-kernel/`.

## Script

Use:

```bash
.codex/skills/verify-choreo-kernel/scripts/verify_choreo_kernel.sh <input.co> [output] [options] [-- extra compile args]
```

Execution requirement:
- Run the command outside sandbox (`require_escalated`) when using Codex tools, because the script performs real GPU runtime verification and benchmarking.

Examples:

```bash
.codex/skills/verify-choreo-kernel/scripts/verify_choreo_kernel.sh \
  benchmark/performance/matmul/matmul_e4m3_dyn_persis_sta_sm90.co

.codex/skills/verify-choreo-kernel/scripts/verify_choreo_kernel.sh \
  benchmark/performance/matmul/matmul_e4m3_dyn_persis_sta_sm90.co \
  build/matmul_e4m3_verify -- --use-prepack
```

Options:
- `--timeout-sec <n>`: runtime timeout in seconds (default `120`).
- `--bench-timeout-sec <n>`: bench runtime timeout in seconds (default `180`).
- `--run-arg <arg>`: append one runtime argument (repeatable).
- `--no-disable-timing`: do not inject `--disable-timing`.
- `--no-bench`: skip post-verify benchmark run.
- `--no-auto-target`: disable automatic `sm90` target flags.
- `--no-debug-on-fail`: disable extra diagnostics.

Environment:
- `CHOREO_VERIFY_GPU`: force runtime on a specific GPU (`CUDA_VISIBLE_DEVICES`).
- `CHOREO_VERIFY_AUTO_GPU`: auto-pick an idle GPU when `CHOREO_VERIFY_GPU` is unset (default `1`).
- `CHOREO_VERIFY_GPU_MAX_USED_MB`: memory-used threshold for idle GPU selection (default `2048` MB).
- `CHOREO_VERIFY_GPU_MAX_UTIL`: utilization threshold for idle GPU selection (default `20`).
- `CHOREO_VERIFY_OOM_GPU`: on OOM, retry once on the specified GPU (`CUDA_VISIBLE_DEVICES` fallback).
- `CHOREO_VERIFY_TIMEOUT_KILL_SEC`: grace period before force-kill after timeout (default `10`).
- `CHOREO_VERIFY_USE_SMALL`: enable/disable small-shape compile override (default `1`).
- `CHOREO_VERIFY_SMALL_MNK`: small matmul shape as `M,N,K` for compile override (default `256,256,256`).
- `CHOREO_VERIFY_RUN_BENCH`: run post-verify bench (default `1`).
- `CHOREO_VERIFY_BENCH_MNK`: bench shape as `M,N,K` (default `2048,2048,2048`).
- `CHOREO_VERIFY_BENCH_TIMEOUT_SEC`: override bench timeout seconds.

## Debug Reference

For common failures and next actions, read:
- `references/debug-playbook.md`
