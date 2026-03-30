---
name: benchmark-regression-check
description: Run benchmark/performance kernels against the current Choreo commit, preserve raw and summarized results in CSV and TSV files, and detect correctness or performance regressions against a prior baseline. Use when the user asks to benchmark the repo, batch-check benchmark/performance after compiler changes, or verify that a new commit did not regress benchmark behavior.
disable-model-invocation: true
argument-hint: "optional benchmark folders, kernel regex, or baseline CSV path"
---

# Benchmark Regression Check

Use this skill for repeatable regression validation of kernels under `benchmark/performance`.

This workflow is side-effectful: it may rebuild the compiler, run many kernels, and write CSV/TSV artifacts. Keep the user informed before starting expensive runs.

## Related Skills

- Use `compile-and-test` for one-off `.co` execution, quick build validation, or targeted debugging.

This skill is intentionally narrow: its job is to detect whether regression happened, and to name which benchmark cases regressed.

## Response Language Alignment

- If the user prompt is in Chinese, respond in Chinese.
- If the user prompt is in English, respond in English.
- If the prompt is mixed, default to the language of the latest user message.

## Primary Goal

For the requested benchmark scope, produce two durable artifacts:

1. A raw per-run CSV with one row per kernel.
2. A cumulative TSV ledger that records whether each kernel passed, failed, or regressed versus baseline.

Do not overwrite prior benchmark history. Append new entries and preserve logs.

## Canonical Artifacts

- Raw run CSV: `performance/<run_tag>.csv`
- Cumulative ledger: `performance/regressions.tsv`
- Batch logs: `build/kernel-batch/<run_tag>/logs/`

`<run_tag>` should be `<commit_time>_<commit_id>` when possible.

## Existing Repo Support

The repo already provides a batch runner:

- Actual script path: `scripts/bash_compile_and_profile_kernels.sh`
- Default benchmark folders inside that script:
  - `benchmark/performance/blockscale_gemm`
  - `benchmark/performance/blockscale_gemm_v2`
  - `benchmark/performance/gemm_sp`
  - `benchmark/performance/matmul`

Note: the script usage banner says `batch_compile_and_profile_kernels.sh`, but the real file in the repo is `bash_compile_and_profile_kernels.sh`. Use the real file path.

## Workflow

### 1. Define Scope

Decide the benchmark scope from the user request.

- If the user names folders, use only those folders.
- If the user gives a regex or kernel name, pass `--pattern`.
- If the user gives no scope, use the script defaults.
- If the run would be very large, offer a smaller smoke subset first, but keep the default full-run path available.

### 2. Ensure the Compiler Build Is Valid

- All build operations must go through the top-level `Makefile`.
- Do not run `cmake` or `ninja` directly.
- If compiler sources under `lib/`, `runtime/`, or `tools/` changed, rebuild before benchmarking.
- Prefer `make` or `make release` for the benchmarked build unless the user explicitly wants debug behavior.

### 3. Capture Run Identity

Before execution, collect:

- commit timestamp
- commit short SHA
- selected GPU index
- GPU model if available
- benchmark folders and pattern filter
- timeout settings

Use these values consistently in the CSV and TSV outputs.

## 4. Run the Batch Benchmark

Prefer the existing batch script instead of inventing ad hoc loops.

Example command shape:

```bash
scripts/bash_compile_and_profile_kernels.sh \
  --output-csv performance/<run_tag>.csv \
  --timeout-sec 300 \
  [--gpu <id>] \
  [--pattern '<regex>'] \
  [benchmark/performance/<folder> ...]
```

Rules:

- Always set an explicit output CSV.
- Keep the default timeout unless the user asks for slower kernels.
- Do not delete prior CSV files.
- Treat the generated log paths inside the CSV as part of the benchmark record.

### 5. Establish the Baseline

Baseline selection logic:

1. If the user supplies a baseline CSV, use it.
2. Otherwise use the most recent prior `performance/*.csv` that matches the same GPU and relevant arch.
3. If no trustworthy baseline exists, mark the run as `no_baseline` rather than inventing a comparison.

Do not compare across clearly different GPU models unless the user explicitly approves that tradeoff.

### 6. Detect Regressions

For each kernel, classify three things:

- `correctness_status`
- `performance_status`
- `overall_status`

#### Correctness Regression

Mark as a correctness regression when any of the following happen in the current run:

- `compile_status != ok`
- `run_status != ok`
- `verify_passed != 1`
- raw CSV `status` is not `ok`

#### Performance Regression

Compare only against a baseline row for the same logical kernel identity:

- folder
- kernel file name
- arch
- same GPU class when possible

Default provisional threshold for regression:

- TFLOPS drops by more than 5%, or
- average milliseconds increase by more than 10%

If both TFLOPS and time are missing, classify as `unknown_perf` and explain why in notes.

Because benchmark devices may be busy, treat the first detected performance regression as provisional until it is rechecked on the same GPU device.

#### Performance Regression Confirmation

If a kernel crosses the performance regression threshold:

- record the first comparison as a suspected regression
- rerun that same kernel on the same GPU device
- keep the same benchmark folder, kernel identity, timeout, and GPU index
- do not switch to another GPU for confirmation

Confirmation rule:

- if the rerun still crosses the threshold, keep `performance_regression`
- if the rerun no longer crosses the threshold, downgrade to `investigate`
- note in `notes` that the initial regression was not reproduced and that GPU contention or environment noise is suspected

If the confirmation rerun fails for correctness reasons, classify it under `correctness_regression` and preserve both logs.

#### Overall Status

- `pass`: no correctness regression and no performance regression
- `correctness_regression`: compile, run, or verification failure
- `performance_regression`: correctness passed but performance crossed threshold
- `no_baseline`: current run succeeded but no valid comparison existed
- `investigate`: data was incomplete or ambiguous

### 7. Persist the TSV Ledger

Append one row per kernel to `performance/regressions.tsv`.

Create the file with this header if it does not exist:

```tsv
run_tag	commit_time	commit_id	gpu_index	gpu_name	folder	kernel	arch	baseline_commit	baseline_tflops	current_tflops	tflops_delta_pct	baseline_avg_ms	current_avg_ms	avg_ms_delta_pct	compile_status	run_status	verify_passed	correctness_status	performance_status	overall_status	csv_path	compile_log	run_log	notes
```

Rules:

- Append only; never rewrite history unless the user explicitly asks for cleanup.
- Keep enough metadata to reproduce the run.
- Use `notes` for missing baseline, noisy measurements, timeouts, or suspected environment issues.

### 8. When Regressions Happen

If correctness regressed:

- Keep the failing CSV and logs.
- Identify the exact kernels that changed from pass to fail.

If performance regressed:

- Keep the CSV and logs.
- Report the worst regressions first.
- Include both absolute values and percentage deltas.
- Re-run suspected performance regressions on the same GPU device before calling them real regressions.
- If the rerun confirms the slowdown, report it as `performance_regression`.
- If the rerun does not confirm the slowdown, report it as `investigate` and note likely measurement noise or GPU contention.
- Do not diagnose root cause in this workflow; only identify which benchmark cases regressed and preserve the evidence.

## Completion Criteria

The task is complete only when all of the following are true:

- The requested benchmark scope was executed or a concrete blocker was reported.
- A raw CSV was produced and its path was reported.
- `performance/regressions.tsv` was created or appended.
- Each kernel was classified as pass, regression, no baseline, or investigate.
- The final summary names the affected kernels, not just aggregate counts.

## Final Report Format

Summarize with:

1. benchmark scope
2. raw CSV path
3. TSV ledger path
4. pass count
5. correctness regression count
6. performance regression count
7. no-baseline count
8. worst offending kernels with deltas and log paths

If nothing regressed, say so explicitly.

If any suspected regressions were cleared by the confirmation rerun, say that explicitly too.

## Example Prompts

- `Check benchmark/performance for regressions after my latest compiler changes.`
- `Run a benchmark regression sweep for benchmark/performance/matmul and benchmark/performance/gemm_sp.`
- `Benchmark this commit and compare against the latest CSV in performance/.`
- `Run only sm90 matmul kernels and append the result to the regression ledger.`