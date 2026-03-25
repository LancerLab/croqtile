---
name: ai-tune-continue
description: Continue a halted AI-tune experiment from where it left off. Use when the user wants to resume an interrupted ai-tune session due to connection issues, priority changes, or other interruptions. Verifies branch status, reads experiment history, and resumes the optimization loop exactly as defined in program.md.
argument-hint: [optional: specific ai-tune branch name to continue]
disable-model-invocation: true
---

# AI-Tune Continue: Resume Interrupted Optimization Loop

Resume an AI-tune experiment that was halted due to external factors. **NEVER** modifies the original `ai-tune/<date>/<kernel>` branch directly - always forks to a new `ai-tune/<date>/<kernel>/resume-<id>` branch.

## Pre-flight: Branch Detection, Selection, and Safe Fork

### Step 1: Identify the source experiment branch

**If on `main` or non-ai-tune branch - ASK the user which experiment to continue.**

```bash
git branch -a | grep -E "ai-tune|ai-tune-det" | grep -v resume
```

Present the list and ask:
> You are on `main`. Which ai-tune experiment would you like to continue?
>
> Available branches:
> 1. `ai-tune/2026-03-22/blockscale_gemm_v2`
> 2. `ai-tune-det/2026-03-21/gemm_sp_e4m3`
> ...
>
> Or provide a branch name manually.

**If already on an `ai-tune/<date>/<kernel>` or `ai-tune-det/<date>/<kernel>` branch** - use that as the source.

**If already on a `resume-<N>` or `resume-<N>-det` branch** - ask whether to continue on this branch or fork a fresh one.

**If argument provided** - use that as the source branch.

### Step 2: Fork with resume suffix (MANDATORY - never skip)

**CRITICAL: You MUST create a new branch. NEVER work directly on the source branch.**

This compact skill uses `resume-<N>` suffix (no `-det`).

```bash
SOURCE_BRANCH=<ai-tune or ai-tune-det>/<YYYY-MM-DD>/<kernel-mnemonic>

# List existing resume branches for this experiment
git branch -a | grep "${SOURCE_BRANCH}/resume-" | sort

# Find next available resume ID
# If no resume branches exist: resume-1
# If resume-1 exists: resume-2
# etc.

git checkout -b ${SOURCE_BRANCH}/resume-<N> ${SOURCE_BRANCH}
```

**Branch naming examples:**
- Source: `ai-tune/2026-03-22/blockscale_gemm_v2` or `ai-tune-det/2026-03-22/blockscale_gemm_v2`
- Resume (compact): `.../resume-1`, `.../resume-2`
- Resume (detailed): `.../resume-1-det`, `.../resume-2-det`

**If checking out from remote:**
```bash
git fetch origin
git checkout -b ${SOURCE_BRANCH} origin/${SOURCE_BRANCH>
git checkout -b ${SOURCE_BRANCH}/resume-<N> ${SOURCE_BRANCH>
```

### Step 3: Confirm branch state

```bash
git branch --show-current
# Should show: <ai-tune or ai-tune-det>/<YYYY-MM-DD>/<kernel>/resume-<N>

git log --oneline -5
# Should show the last commits from the source branch
```

## Step 0: Experiment Recovery

**Read these files IN THIS ORDER before doing anything else.**

1. **`.claude/program.md`** - Load in full. Defines the loop protocol, mandatory rules, and constraints. Follow it exactly.

2. **`choreo-syntax` skill** - Load before editing any `.co` file.

3. **`results.tsv`** - Parse to determine:
   - **Current iteration count**: Highest `iter<N>` number
   - **Current best kernel**: Row with highest TFLOPS where decision is KEEP
   - **All tried optimizations**: Every `idea_summary` - avoid repetition (Rule 4)
   - **Bottleneck history**: What was targeted in each iteration
   - **Problem size**: M/N/K used (must remain constant)
   - **Architecture**: `arch` column (e.g., `sm_90a`)

4. **Identify current best kernel file**: From `results.tsv` KEEP row with highest TFLOPS. Verify file exists in `benchmark/performance/<kernel-folder>/`.

5. **Determine next iteration number**: `current_max_iter + 1`. Do NOT reuse or skip.

6. **Verify current best compiles and runs**:
   ```bash
   # For .co
   ./choreo -gs -t cute -arch=<ARCH> <current_best>.co -o <current_best>.cute.result
   bash <current_best>.cute.result --execute

   # For .cu
   nvcc -arch <ARCH> -O2 -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
     -I"$(pwd)/runtime" -I"$(pwd)/extern/cutlass/include" -I"$(pwd)" \
     -L/usr/local/cuda/lib64 -lcuda \
     <current_best>.cu -o iter<NNN>_<tag>
   ./iter<NNN>_<tag> --execute
   ```

## Resume: Optimization Loop (Infinite)

Follow the loop defined in `.claude/program.md` Steps 1-5:

1. **Step 1 - Profile**: Run `ncu --set full` on the current best. Identify the dominant bottleneck.
2. **Step 2 - Raise an idea**: Propose ONE targeted optimization grounded in ncu data. Avoid repetitions from `results.tsv`.
3. **Step 3 - Implement**: Create a new versioned `.co` (or `.cu`) file. Compile and verify correctness.
   - **Prefer `.co` modification + choreo compile**. This is the primary path.
   - If the optimization requires a choreo compiler change, implement it in `lib/` first, rebuild `./choreo`, then use the new feature in the `.co` file. Treat both as ONE atomic change.
   - If the optimization is too complex for `.co` (e.g. inline PTX, manual barrier reordering), modify the generated `.cu` file directly. Track the `.cu` artifact alongside the `.co` base.
4. **Step 4 - Profile and decide**: Compare TFLOPS. KEEP if better, DISCARD if not.
5. **Step 5 - Commit**: On KEEP, commit with descriptive message and update `results.tsv`.

**Repeat from Step 1. NEVER STOP unless the user interrupts or the network drops.**

## Mandatory Verification (NEVER SKIP)

**Every iteration MUST pass verification before it can be KEPT.**

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

| Precision | base_tol | rel_tol |
|-----------|----------|---------|
| FP16 input, FP32 accum | 1.0 | 0.01 |
| FP16 input, FP16 accum | 16.0 | 0.05 |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 |

### Requirements

- Host code MUST include verification block when `skip_verify` is false.
- **NEVER print "Test Passed" without actual numerical comparison.** Running kernel twice is NOT verification.
- **Keep `lhs_dense_h` alive** in host code - needed for CPU reference.

## Iteration File Naming

- `.co` iterations: `<kernel-base>_iter<NNN>_<brief-tag>.co`
- `.cu` iterations: `<kernel-base>_iter<NNN>_<brief-tag>.cu`
- Keep ALL iteration files (even discarded ones) for traceability.

## Artifact Management

- **Every KEEP iteration**: `git add` the new `.co` (and `.cu` if applicable), `results.tsv`. Commit with: `iter<NNN>: <description> - TFLOPS: X -> Y (KEEP)`
- **Every DISCARD**: Note in `results.tsv`. Commit with `(DISCARD)` tag for traceability.
- **Periodic pushes**: Push to remote branch every 5-10 iterations or after significant wins.
- **On context compaction**: Commit all pending work, push, note current state. After compaction, re-read `.claude/program.md` and `results.tsv` to resume.

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

- `ai-tune` - original optimization loop (start new experiment)
- `ai-tune-summary` - finalize and ship results to main
- `choreo-syntax` - DSL reference
- `compile-and-test` - build/run workflows
- `develop-compiler` - when compiler changes needed
- `profiling` - ncu invocation
- `performance-bottleneck-analysis` - interpreting ncu reports

## Usage Examples

Continue a compact experiment with compact skill:
```
/ai-tune-continue ai-tune/2026-03-20/gemm_sp
```
Creates `ai-tune/2026-03-20/gemm_sp/resume-1`

Continue a detailed experiment with compact skill:
```
/ai-tune-continue ai-tune-det/2026-03-20/gemm_sp
```
Creates `ai-tune-det/2026-03-20/gemm_sp/resume-1`

When on main, the skill asks which experiment to continue.

IMPORTANT: DO NOT STOP UNTIL USER MANUALLY DO OR NETWORK ISSUE, KEEP TRYING THE EXPERIMENT LOOP.
