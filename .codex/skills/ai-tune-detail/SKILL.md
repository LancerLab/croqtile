---
name: ai-tune-detail
description: Launch an infinite AI-driven kernel optimization loop for Choreo GPU kernels (detailed version). Use when the user asks to "ai-tune", "optimize", or "auto-tune" a kernel folder under benchmark/performance/. Reads .claude/program.md, creates a dated branch, profiles the baseline, and iterates ncu-guided optimizations indefinitely until interrupted.
argument-hint: <kernel-folder-path e.g. benchmark/performance/gemm_sp/>
disable-model-invocation: true
---

# AI-Tune Detail: Infinite Kernel Optimization Loop (Detailed)

Kick off an autonomous, ncu-guided optimization experiment on the kernel folder: `$ARGUMENTS`.

This is the **detailed version** of `ai-tune`. It provides expanded guidance for profiling, optimization ideas, implementation, and verification.

## Pre-flight

1. **Read the program**: Load `.claude/program.md` in full - it defines the loop protocol, mandatory rules, and constraints. Follow it exactly.

2. **Read the syntax reference**: Load the `choreo-syntax` skill before editing any `.co` file.

3. **Identify the target folder**: The user specifies a kernel folder (e.g. `benchmark/performance/gemm_sp/`, `benchmark/performance/blockscale_gemm_v2/`). All kernel `.co` files live there.

4. **Determine the kernel mnemonic**: Derive a short mnemonic from the folder name (e.g. `gemm_sp`, `gemm_sp_e4m3`, `blockscale_gemm_v2`). The user may specify a more specific mnemonic if the folder contains multiple kernel families.

## Branch Setup

1. **Generate the branch name**: `ai-tune-det/<today's date YYYY-MM-DD>/<kernel-mnemonic>`.
2. **Check uniqueness**: Run `git branch -a | grep ai-tune-det` and verify no branch with the same name exists. If a collision exists, append a numeric suffix (e.g. `-2`, `-3`).
3. **Create and switch**:
   ```bash
   git checkout -b ai-tune-det/<date>/<mnemonic>
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

### Step 1 - Profile Current Best

```bash
KERNEL=benchmark/performance/<kernel-folder>/<current_best>.co   # or .cu
ARCH=sm_90a   # or from results.tsv
KERNEL_OUT=benchmark/performance/<kernel-folder>/<current_best>

# Compile if .co file
./choreo -gs -t cute -arch=$ARCH $KERNEL -o ${KERNEL_OUT}.cute.result
# May need flags like --use-warpspec --use-prepack

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

**NOTE: The ncu and nvcc commands above are examples. You may flexibly use or adapt the profiling and compilation tools to achieve your optimization ideas. The key is to gather bottleneck data and verify correctness.**

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

**NOTE: The compile and run commands above are examples. You may flexibly adapt the workflow to your specific kernel type and optimization approach.**

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

- `choreo-syntax` - DSL reference for `.co` editing
- `compile-and-test` - build/run/debug workflows
- `develop-compiler` - when compiler changes are needed
- `profiling` - ncu invocation and metric interpretation
- `performance-bottleneck-analysis` - interpreting ncu reports
- `ai-tune-summary` - when the user wants to stop and ship results to main

## Usage Examples

Start a new ai-tune experiment:
```
/ai-tune-detail benchmark/performance/gemm_sp/
```
