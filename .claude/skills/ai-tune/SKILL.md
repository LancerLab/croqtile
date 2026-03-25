---
name: ai-tune
description: Launch an infinite AI-driven kernel optimization loop for Choreo GPU kernels. Use when the user asks to "ai-tune", "optimize", or "auto-tune" a kernel folder under benchmark/performance/. Reads .claude/program.md, creates a dated branch, profiles the baseline, and iterates ncu-guided optimizations indefinitely until interrupted.
argument-hint: <kernel-folder-path e.g. benchmark/performance/gemm_sp/>
disable-model-invocation: true
---

# AI-Tune: Infinite Kernel Optimization Loop

Kick off an autonomous, ncu-guided optimization experiment on the kernel folder: `$ARGUMENTS`.

## Pre-flight

1. **Read the program**: Load `.claude/program.md` in full - it defines the loop protocol, mandatory rules, and constraints. Follow it exactly.

2. **Read the syntax reference**: Load the `choreo-syntax` skill before editing any `.co` file.

3. **Identify the target folder**: The user specifies a kernel folder (e.g. `benchmark/performance/gemm_sp/`, `benchmark/performance/blockscale_gemm_v2/`). All kernel `.co` files live there.

4. **Determine the kernel mnemonic**: Derive a short mnemonic from the folder name (e.g. `gemm_sp`, `gemm_sp_e4m3`, `blockscale_gemm_v2`). The user may specify a more specific mnemonic if the folder contains multiple kernel families.

## Branch Setup

1. **Generate the branch name**: `ai-tune/<today's date YYYY-MM-DD>/<kernel-mnemonic>`.
2. **Check uniqueness**: Run `git branch -a | grep ai-tune` and verify no branch with the same name exists. If a collision exists, append a numeric suffix (e.g. `-2`, `-3`).
3. **Create and switch**:
   ```bash
   git checkout -b ai-tune/<date>/<mnemonic>
   ```

## Baseline Discovery

Before any optimization, find the current best kernel(s) in the target folder:

1. **List all `.co` files** in the folder and identify candidates by name patterns (warpspec, prepack, stages, swizzle, etc.).
2. **Compile and benchmark the top 2-3 candidates** to find the actual best performer. Use:
   ```bash
   ./choreo -gs -t cute -arch=sm_90a [flags] <kernel>.co -o /tmp/<kernel>.cute.result
   CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 CUDA_VISIBLE_DEVICES=<free-gpu> \
     bash /tmp/<kernel>.cute.result --execute
   ```
   Choose appropriate choreo flags for the kernel type (e.g. `--use-warpspec --use-prepack` for sparse GEMM).
3. **Record the baseline** in `results.tsv` with the full `run_command`.
4. **Select the best** as the starting genome for optimization.

## Optimization Loop (Infinite)

Follow the loop defined in `.claude/program.md` Steps 1-5:

1. **Step 1 - Profile**: Run `ncu --set full` on the current best. Identify the dominant bottleneck.
2. **Step 2 - Raise an idea**: Propose ONE targeted optimization grounded in ncu data.
3. **Step 3 - Implement**: Create a new versioned `.co` file. Compile and verify correctness.
   - **Prefer `.co` modification + choreo compile**. This is the primary path.
   - If the optimization requires a choreo compiler change, implement it in `lib/` first, rebuild `./choreo`, then use the new feature in the `.co` file. Treat both as ONE atomic change.
   - If the optimization is too complex for `.co` (e.g. inline PTX, manual barrier reordering), modify the generated `.cu` file directly. Track the `.cu` artifact alongside the `.co` base.
4. **Step 4 - Profile and decide**: Compare TFLOPS. KEEP if better, DISCARD if not.
5. **Step 5 - Commit**: On KEEP, commit with descriptive message and update `results.tsv`.

**Repeat from Step 1. NEVER STOP unless the user interrupts or the network drops.**

## Mandatory Verification (NEVER SKIP)

**Every iteration MUST pass verification before it can be KEPT.** A kernel that produces wrong results is worthless regardless of TFLOPS.

### Using `verify_matmul_row_row_subset` or `verify_matmul_row_col_subset` (in `runtime/choreo.h`)

```cpp
size_t M = res.shape()[0];
size_t N = res.shape()[1];

// Sample rate based on problem size (applied to total MxN output)
// 4096x4096 @ 1% => 167,772 samples; 2048x2048 @ 2% => 83,886 samples; 1024x1024 @ 4% => 41,943 samples
float rate = (M >= 4096 && N >= 4096) ? 0.01f :
             (M >= 2048 && N >= 2048) ? 0.02f :
             0.04f;

size_t total_samples = static_cast<size_t>(rate * M * N);
size_t max_i = static_cast<size_t>(std::sqrt(total_samples));
size_t max_j = max_i;

verify_matmul_row_row_subset(lhs, rhs, res, base_tol, rel_tol, max_i, max_j);
```

### Tolerance Guidelines

| Precision | base_tol | rel_tol | Notes |
|-----------|----------|---------|-------|
| FP16 input, FP32 accum | 1.0 | 0.01 | Standard |
| FP16 input, FP16 accum | 16.0 | 0.05 | Higher error from FP16 accumulation |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 | FP8 inputs are lower magnitude |

### Requirements

- Host code MUST include verification block when `skip_verify` is false.
- **NEVER print "Test Passed" without actual numerical comparison.** Running kernel twice is NOT verification.
- **Keep `lhs_dense_h` alive** in host code - needed for CPU reference.

## Iteration File Naming

- `.co` iterations: `<kernel-base>_iter<NNN>_<brief-tag>.co` (e.g. `blockscale_gemm_e4m3_iter001_tma_meta.co`)
- `.cu` iterations (when modifying generated CUDA): `<kernel-base>_iter<NNN>_<brief-tag>.cu`
- Keep ALL iteration files (even discarded ones) for traceability.

## Artifact Management

- **Every KEEP iteration**: `git add` the new `.co` (and `.cu` if applicable), `results.tsv`, and any compiler changes. Commit with: `iter<NNN>: <description> - TFLOPS: X -> Y (KEEP)`
- **Every DISCARD**: Note in `results.tsv` but do not commit the failed kernel file. Or commit with `(DISCARD)` tag for traceability.
- **Periodic pushes**: Push to the remote branch every 5-10 iterations or after significant wins.
- **On context compaction**: Before the context window fills, commit all pending work, push, and note the current state in the commit message. After compaction, re-read `.claude/program.md` and `results.tsv` to resume.

## Critical Rules (from program.md)

- Profile before EVERY idea (Rule 1)
- Hill-climb from ONE current best (Rule 2)
- Diverse optimizations, not just macro sweeps (Rule 3)
- Never repeat failed combinations (Rule 4)
- Abandon stuck ideas after 3 attempts (Rule 5)
- Understand the kernel before mutating (Rule 6)
- **NEVER SKIP verification** - every KEEP must pass verification (Rule 7)
- **NEVER STOP the loop** - run indefinitely until user interrupts

## Related Skills

- `choreo-syntax` - DSL reference for `.co` editing
- `compile-and-test` - build/run workflows
- `develop-compiler` - when compiler changes are needed
- `profiling` - ncu invocation and metric interpretation
- `performance-bottleneck-analysis` - interpreting ncu reports
- `ai-tune-summary` - when the user wants to stop and ship results to main
