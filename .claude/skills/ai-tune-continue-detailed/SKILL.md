---
name: ai-tune-continue
description: Continue a halted AI-tune experiment from where it left off. Use when the user wants to resume an interrupted ai-tune session due to connection issues, priority changes, or other interruptions. Verifies branch status, reads experiment history, and resumes the optimization loop exactly as defined in program.md.
argument-hint: [optional: specific ai-tune branch name to continue]
disable-model-invocation: true
---

# AI-Tune Continue: Resume Interrupted Optimization Loop

Resume an AI-tune experiment that was halted due to external factors (connection issues, user interruption, context compaction, etc.). This skill ensures the experimental behavior continues **exactly** as defined in the original `ai-tune` skill and `.claude/program.md`.

**SAFETY**: This skill NEVER modifies the original `ai-tune/<date>/<kernel>` branch directly. It always forks to a new `ai-tune/<date>/<kernel>/resume-<id>` branch to preserve the original experiment state.

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

This detailed skill uses `resume-<N>-det` suffix to distinguish from compact version.

```bash
SOURCE_BRANCH=<ai-tune or ai-tune-det>/<YYYY-MM-DD>/<kernel-mnemonic>

# List existing resume branches for this experiment (both compact and detailed)
git branch -a | grep "${SOURCE_BRANCH}/resume-" | sort

# Find next available resume ID for detailed version
# If no resume-det branches exist: resume-1-det
# If resume-1-det exists: resume-2-det
# etc.

git checkout -b ${SOURCE_BRANCH}/resume-<N>-det ${SOURCE_BRANCH}
```

**Branch naming examples:**
- Source: `ai-tune/2026-03-22/blockscale_gemm_v2` or `ai-tune-det/2026-03-22/blockscale_gemm_v2`
- Resume (compact): `.../resume-1`, `.../resume-2`
- Resume (detailed): `.../resume-1-det`, `.../resume-2-det`

**If checking out from remote:**
```bash
git fetch origin
git checkout -b ${SOURCE_BRANCH} origin/${SOURCE_BRANCH}
git checkout -b ${SOURCE_BRANCH}/resume-<N>-det ${SOURCE_BRANCH>
```

### Step 3: Confirm branch state

After checkout, verify:
```bash
git branch --show-current
# Should show: <ai-tune or ai-tune-det>/<YYYY-MM-DD>/<kernel>/resume-<N>-det

git log --oneline -5
# Should show the last commits from the source branch
```

Present the list and ask:
> You are on `main`. Which ai-tune experiment would you like to continue?
>
> Available branches:
> 1. `ai-tune/2026-03-22/blockscale_gemm_v2`
> 2. `ai-tune/2026-03-21/gemm_sp_e4m3`
> ...
>
> Or provide a branch name manually.

**If already on an `ai-tune/<date>/<kernel>` branch** - use that as the source.

**If already on an `ai-tune/<date>/<kernel>/resume-<id>` branch** - the user is resuming an existing resume. Ask whether to continue on this branch or fork a fresh one.

**If argument provided** - use `ai-tune/<argument>` as the source branch.

### Step 2: Fork with resume suffix (MANDATORY - never skip)

**CRITICAL: You MUST create a new branch. NEVER work directly on the source `ai-tune/<date>/<kernel>` branch.**

```bash
# Define source branch (the one to resume from)
SOURCE_BRANCH=ai-tune/<YYYY-MM-DD>/<kernel-mnemonic>

# Determine resume ID
# List existing resume branches for this experiment
git branch -a | grep "${SOURCE_BRANCH}/resume-" | sort

# Find next available resume ID
# If no resume branches exist: resume-1
# If resume-1 exists: resume-2
# If resume-2 exists: resume-3
# etc.

# Create and switch to the resume branch
git checkout -b ${SOURCE_BRANCH}/resume-<N> ${SOURCE_BRANCH}
```

**Branch naming examples:**
- Source: `ai-tune/2026-03-22/blockscale_gemm_v2`
- Resume 1: `ai-tune/2026-03-22/blockscale_gemm_v2/resume-1`
- Resume 2: `ai-tune/2026-03-22/blockscale_gemm_v2/resume-2`
- Resume 3: `ai-tune/2026-03-22/blockscale_gemm_v2/resume-3`

**If checking out from remote:**
```bash
# Ensure remote refs are up-to-date
git fetch origin

# Create local tracking branch from remote source
git checkout -b ${SOURCE_BRANCH} origin/${SOURCE_BRANCH}

# Then fork to resume
git checkout -b ${SOURCE_BRANCH}/resume-<N> ${SOURCE_BRANCH}
```

### Step 3: Confirm branch state

After checkout, verify:
```bash
git branch --show-current
# Should show: ai-tune/<YYYY-MM-DD>/<kernel>/resume-<N>

git log --oneline -5
# Should show the last commits from the source branch
```

## Step 0: Experiment Recovery - Read Everything Before Continuing

**CRITICAL: Read these files IN THIS ORDER before doing anything else.**

1. **`.claude/program.md`** - Load in full. This defines the loop protocol, mandatory rules, constraints, and the exact optimization workflow. Follow it exactly. This is the single source of truth.

2. **`choreo-syntax` skill** - Load before editing any `.co` file.

3. **`results.tsv`** - Parse to determine:
   - **Current iteration count**: Highest `iter<N>` number (e.g., `iter066`)
   - **Current best kernel**: Row with highest TFLOPS where decision is KEEP
   - **Current best TFLOPS**: The performance baseline for comparison
   - **All tried optimizations**: Every `idea_summary` - to avoid repetition (Rule 4)
   - **Bottleneck history**: `bottleneck_before` column - understand what was targeted
   - **Decision history**: Which ideas were KEEP vs DISCARD
   - **Problem size**: M/N/K used (must remain constant unless explicitly changed)
   - **Architecture**: `arch` column (e.g., `sm_90a`)

4. **Identify the current best kernel file**:
   - From results.tsv, find the KEEP row with highest TFLOPS
   - The `kernel` column gives the filename
   - Verify the file exists in `benchmark/performance/<kernel-folder>/`
   - For `.co` kernels: `<kernel-base>_iter<NNN>_<tag>.co`
   - For `.cu` kernels: `<kernel-base>_iter<NNN>_<tag>.cu`

5. **Determine the next iteration number**: `current_max_iter + 1`
   - Do NOT reuse iteration numbers
   - Do NOT skip ranges
   - Rule 9: Track iteration counter monotonically

6. **Verify the current best kernel compiles and runs**:
   ```bash
   # For .co kernels
   ./choreo -gs -t cute -arch=<ARCH> <current_best>.co -o <current_best>.cute.result
   bash <current_best>.cute.result --execute

   # For .cu kernels
   nvcc -arch <ARCH> -O2 -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
     -I"$(pwd)/runtime" -I"$(pwd)/extern/cutlass/include" -I"$(pwd)" \
     -L/usr/local/cuda/lib64 -lcuda \
     <current_best>.cu -o iter<NNN>_<tag>
   ./iter<NNN>_<tag> --execute
   ```
   If it fails, the source branch may be in a broken state. Report to user and ask for guidance.

## Resume: Optimization Loop (Continue from Step 1)

**Continue the loop exactly as defined in `.claude/program.md` Steps 1-5.**

### Step 1 - Profile Current Best

**NOTE: The ncu and nvcc commands below are examples. You may flexibly use or adapt the profiling and compilation tools to achieve your optimization ideas. The key is to gather bottleneck data and verify correctness.**

```bash
KERNEL=benchmark/performance/<kernel-folder>/<current_best>.co   # or .cu
ARCH=<from results.tsv>
KERNEL_OUT=benchmark/performance/<kernel-folder>/<current_best>

# Compile if .co file
./choreo -gs -t cute -arch=$ARCH $KERNEL -o ${KERNEL_OUT}.cute.result
# May need flags like --use-warpspec --use-prepack (check results.tsv run_command)

# If .cu file, compile with nvcc
nvcc -arch $ARCH -O2 -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
  -I"$(pwd)/runtime" -I"$(pwd)/extern/cutlass/include" -I"$(pwd)" \
  -L/usr/local/cuda/lib64 -lcuda \
  $KERNEL -o ${KERNEL_OUT}_iter<N>

# Profile with ncu
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o ${KERNEL_OUT}_ncu_iter<next_N> \
  bash ${KERNEL_OUT}.cute.result --execute   # or run the binary directly for .cu
```

**Read ncu metrics:**

| Metric | Meaning |
|--------|---------|
| `sm__throughput.avg.pct_of_peak_sm` | SM utilization |
| `sm__throughput.avg.pct_of_peak_tensor` | Tensor-core utilization |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Global load throughput |
| `dram__bytes.sum` | Memory bandwidth |
| `lts__t_sectors_lookup_miss.sum` | L2 cache miss rate |
| `wgmma...` | WGMMA-specific metrics on Hopper |
| Achieved occupancy vs single-issue occupancy | Occupancy gap |

**Bottleneck categories:**

| Bottleneck | ncu indicator | Typical fix |
|------------|---------------|-------------|
| SM underutilized | sm__throughput.avg.pct < 80% | Increase CTAs, check occupancy |
| Memory bound | sm__pipe_tensor_op_pct < 50%, high l1tex ld | Larger tiles, TMA async, more stages |
| L2 thrash | lts__lookup_miss.sum high vs total | Change tile shape (M/N/K ratio) |
| WGMMA latency hidden | low wgmma__ops issue rate | Better producer/consumer overlap, more stages |
| Register pressure | sm__occ_pct_of_peak_smem_per_block_at_block_limit low | Reduce tile size, fewer threads per CTA |

### Step 2 - Raise an Optimization Idea

Based on ncu data from Step 1, propose ONE targeted optimization grounded in specific metrics.

**Ideas may involve:**
- Tuning `#define` constants (WARP_N, STAGES, etc.)
- Changing tile shapes, adding/removing pipeline stages
- Warp-specialization ratio changes (1p1c ↔ 1p3c)
- Switching from sync to async TMA
- Persistent CTA vs static CTA
- Swizzle factor, K-tiling depth, register blocking
- Adding compiler flags (`--hoist-offset`, `--hoist-scale`, `--stmatrix`)
- Changing output store pattern (shared padding, transpose)
- Modifying pipeline structure (event placement, commit placement)
- Adding explicit inline asm intrinsics (nanosleep, fence_proxy_async)
- Changing copy-shaping patterns (`view().from()`, `subspan().step().at()`)
- Implementing new DSL features (requires `develop-compiler` skill)

**What NOT to do:**
- Do not change host harness (main, timing, verification)
- Do not change problem size (M/N/K) unless explicitly asked
- Do not disable verification or timing
- Do not repeat ideas already in results.tsv (Rule 4)
- Do not guess without ncu data (Rule 1)

### Step 3 - Implement and Debug

Create a new versioned candidate file. Do NOT mutate the current best in place.

```bash
# Create new iteration file
cp <current_best>.co <kernel-base>_iter<NNN>_<tag>.co   # or .cu

# Edit
$EDITOR <kernel-base>_iter<NNN>_<tag>.co

# Compile
./choreo -gs -t cute -arch=$ARCH <kernel-base>_iter<NNN>_<tag>.co \
  -o <kernel-base>_iter<NNN>_<tag>.cute.result

# Verify (MUST pass before profiling)
bash <kernel-base>_iter<NNN>_<tag>.cute.result --execute
```

**Hard debugging protocol**: If after 3 distinct fix attempts the kernel still fails compilation or verification, ABANDON the idea. Revert to current best. Go to Step 2.

### Step 4 - Profile and Decide

```bash
# Profile new iteration
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o ${KERNEL_OUT}_ncu_iter<NNN> \
  bash <kernel-base>_iter<NNN>_<tag>.cute.result --execute

# Measure TFLOPS
bash <kernel-base>_iter<NNN>_<tag>.cute.result --execute
```

**Decision rule:**
- TFLOPS > current best => **KEEP**
- TFLOPS ≤ current best => **DISCARD**

### Step 5 - Commit and Update

**On KEEP:**
```bash
git add <kernel-base>_iter<NNN>_<tag>.co results.tsv
# Also add compiler changes if applicable
git commit -m "iter<NNN>: <description> - TFLOPS: X -> Y (KEEP)"
```

**On DISCARD:**
- Note in results.tsv
- Do NOT commit the failed kernel file (or commit with DISCARD tag for traceability)
- Revert to current best

**Update results.tsv:**
```bash
echo -e "iter<NNN>\t<KERNEL>\t<ARCH>\t<TFLOPS>\t<EFF%>\t<BOTTLENECK_CATEGORY>\t<RUN_COMMAND>\t<IDEA_SUMMARY>" >> results.tsv
```

**Repeat from Step 1. NEVER STOP.**

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

### Verification Requirements

- **For `.co` kernels**: Host code MUST include verification block calling `verify_matmul_row_row_subset` (or `verify_matmul_row_col_subset`) when `skip_verify` is false. Must print `Test Passed` only after verification succeeds.
- **For `.cu` kernels**: Same. `.cu` host code must compute CPU reference from `lhs_dense_h` and compare with GPU result.
- **NEVER print "Test Passed" without actual numerical comparison.** Running kernel twice and copying result back is NOT verification.
- **Keep `lhs_dense_h` alive** in host code - needed for CPU reference dot products.

### Verification Pattern for Sparse GEMM

```cpp
if (skip_verify) {
  std::cout << "Test Passed (verify skipped)\n" << std::endl;
  return 0;
}

auto lhs_dense_v = lhs_dense_h.view();
auto rhs_v = rhs_h.view();
auto res_v = res_h.view();

size_t M = res.shape()[0];
size_t N = res.shape()[1];

float rate = (M >= 4096 && N >= 4096) ? 0.01f :
             (M >= 2048 && N >= 2048) ? 0.02f : 0.04f;

size_t total_samples = static_cast<size_t>(rate * M * N);
size_t max_i = static_cast<size_t>(std::sqrt(total_samples));
size_t max_j = max_i;

float base_tol = <see tolerance table>;
float rel_tol = <see tolerance table>;

verify_matmul_row_row_subset(lhs_dense_v, rhs_v, res_v, base_tol, rel_tol, max_i, max_j);

std::cout << "Test Passed\n" << std::endl;
```

## Iteration File Naming

- `.co` iterations: `<kernel-base>_iter<NNN>_<brief-tag>.co`
- `.cu` iterations: `<kernel-base>_iter<NNN>_<brief-tag>.cu`
- Keep ALL iteration files (even discarded ones) for traceability

## Artifact Management

- **Every KEEP iteration**: `git add` the new `.co` (and `.cu` if applicable), `results.tsv`, and any compiler changes. Commit with: `iter<NNN>: <description> - TFLOPS: X -> Y (KEEP)`
- **Every DISCARD**: Note in `results.tsv` but do not commit the failed kernel file. Or commit with `(DISCARD)` tag for traceability.
- **Periodic pushes**: Push to remote branch every 5-10 iterations or after significant wins.
  ```bash
  git push -u origin HEAD
  ```
- **On context compaction**: Before context window fills, commit all pending work, push, note current state in commit message. After compaction, re-read `.claude/program.md` and `results.tsv` to resume.

## Critical Rules (from program.md)

These rules are ABSOLUTE. Violating any of them renders the experiment worthless.

### Rule 1: PROFILE BEFORE EVERY IDEA - no exceptions
Before proposing ANY optimization idea, run ncu on the current best and read the report. The `bottleneck_before` column in results.tsv MUST contain a real bottleneck category. `unknown` is FORBIDDEN.

### Rule 2: ONE current best, hill-climb from it - no random base-hopping
At any point there is exactly ONE file designated as the "current best". Every new candidate MUST be derived from that file.

### Rule 3: DIVERSE optimizations - macro sweeps alone are not optimization
After at most 2 consecutive macro-only changes, MUST try a STRUCTURAL change.

### Rule 4: NEVER repeat a failed combination
Before every iteration, read results.tsv and check whether the exact combination has been tried. If yes, choose different idea.

### Rule 5: ABANDON stuck ideas after 3 attempts
If an idea fails to compile or pass verification after 3 distinct fix attempts, ABANDON it entirely.

### Rule 6: UNDERSTAND the kernel before mutating
Before any change, read and understand:
1. The current best `.co` kernel source code
2. The generated `.cute.result` output (at least the kernel launch signature)
3. The ncu profiling data from Step 1

### Rule 7: COMMIT messages must encode the optimization
Every commit and results.tsv entry MUST include:
1. What was changed
2. Why it was expected to help
3. The measured TFLOPS result
4. KEEP or DISCARD decision

### Rule 8: USE the compile+run workflow from program.md
Do NOT delegate to external wrapper scripts as black boxes.

### Rule 9: TRACK iteration counter monotonically
Each iteration gets a unique, monotonically increasing number. Do not reuse or skip ranges.

### Rule 10: Kernel-specific constraints
For gemm_sp (sparse GEMM) f16 kernels on SM90:
- `SPMM_WARP_M` MUST be 64 (WGMMA constraint - never change)
- `SPMM_WARP_K` MUST be 32 for f16 (never change)
- `SPMM_TILE_K` MUST equal `2 * SPMM_PACKED_TILE_K`
- `SPMM_META_TILE_COLS` MUST equal `SPMM_TILE_K / 32`
- Changing `SPMM_WARP_N` is allowed but ONLY as part of a broader structural change

### Anti-pattern checklist (verify before every iteration)

- [ ] Changing only macros without structural change (after 2 consecutive)
- [ ] Using `unknown` as the bottleneck category
- [ ] Repeating a combination already in results.tsv
- [ ] Starting from a different base kernel than the current best
- [ ] Submitting a raw command line as the idea_summary
- [ ] Skipping ncu profiling

## Handling Context Compaction

If context window was full during interruption:
1. Commit all pending work before proceeding
2. Push to remote branch
3. Note current state in commit message
4. After resuming, re-read `.claude/program.md` and `results.tsv` to continue
5. Continue from exact step where left off

## Stop Conditions

- User manually stops the loop
- 10 consecutive discarded ideas (stuck in local minimum - report to user)
- Compiler crashes repeatedly (likely a Choreo bug, escalate to user)
- No more kernel files remain to try as baselines

## Related Skills

- `ai-tune` - original optimization loop skill (start new experiment)
- `ai-tune-summary` - for finalizing and shipping results to main
- `choreo-syntax` - DSL reference for `.co` editing
- `compile-and-test` - build/run/debug workflows
- `develop-compiler` - when compiler changes are needed
- `profiling` - ncu invocation and metric interpretation
- `performance-bottleneck-analysis` - interpreting ncu reports

## Usage Examples

Continue a compact experiment with detailed skill:
```
/ai-tune-continue-detailed ai-tune/2026-03-20/gemm_sp
```
Creates `ai-tune/2026-03-20/gemm_sp/resume-1-det`

Continue a detailed experiment with detailed skill:
```
/ai-tune-continue-detailed ai-tune-det/2026-03-20/gemm_sp
```
Creates `ai-tune-det/2026-03-20/gemm_sp/resume-1-det`

When on main, the skill asks which experiment to continue.
