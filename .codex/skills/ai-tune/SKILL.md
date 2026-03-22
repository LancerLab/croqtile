---
name: ai-tune
description: Launch an infinite AI-driven kernel optimization loop for Choreo GPU kernels. Use when the user asks to "ai-tune", "optimize", or "auto-tune" a kernel folder under benchmark/performance/. Reads .claude/program.md, creates a dated branch, profiles the baseline, and iterates ncu-guided optimizations indefinitely until interrupted.
argument-hint: <kernel-folder-path e.g. benchmark/performance/gemm_sp/>
disable-model-invocation: true
---

# AI-Tune: Infinite Kernel Optimization Loop

Kick off an autonomous, ncu-guided optimization experiment on the kernel folder: `$ARGUMENTS`.

## Pre-flight

1. **Read the program**: Load `.claude/program.md` in full — it defines the loop protocol, mandatory rules, and constraints. Follow it exactly.

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

1. **Step 1 — Profile**: Run `ncu --set full` on the current best. Identify the dominant bottleneck.
2. **Step 2 — Raise an idea**: Propose ONE targeted optimization grounded in ncu data.
3. **Step 3 — Implement**: Create a new versioned `.co` file. Compile and verify correctness.
   - **Prefer `.co` modification + choreo compile**. This is the primary path.
   - If the optimization requires a choreo compiler change, implement it in `lib/` first, rebuild `./choreo`, then use the new feature in the `.co` file. Treat both as ONE atomic change.
   - If the optimization is too complex for `.co` (e.g. inline PTX, manual barrier reordering), modify the generated `.cu` file directly. Track the `.cu` artifact alongside the `.co` base.
4. **Step 4 — Profile and decide**: Compare TFLOPS. KEEP if better, DISCARD if not.
5. **Step 5 — Commit**: On KEEP, commit with descriptive message and update `results.tsv`.

**Repeat from Step 1. NEVER STOP unless the user interrupts or the network drops.**

## Iteration File Naming

- `.co` iterations: `<kernel-base>_iter<NNN>_<brief-tag>.co` (e.g. `blockscale_gemm_e4m3_iter001_tma_meta.co`)
- `.cu` iterations (when modifying generated CUDA): `<kernel-base>_iter<NNN>_<brief-tag>.cu`
- Keep ALL iteration files (even discarded ones) for traceability.

## Artifact Management

- **Every KEEP iteration**: `git add` the new `.co` (and `.cu` if applicable), `results.tsv`, and any compiler changes. Commit with: `iter<NNN>: <description> — TFLOPS: X -> Y (KEEP)`
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
- **NEVER STOP the loop** — run indefinitely until user interrupts

## Related Skills

- `choreo-syntax` — DSL reference for `.co` editing
- `compile-and-test` — build/run workflows
- `develop-compiler` — when compiler changes are needed
- `profiling` — ncu invocation and metric interpretation
- `performance-bottleneck-analysis` — interpreting ncu reports
- `ai-tune-summary` — when the user wants to stop and ship results to main
