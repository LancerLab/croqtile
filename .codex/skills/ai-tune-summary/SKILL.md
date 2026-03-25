---
name: ai-tune-summary
description: Summarize and ship AI-tune optimization results to main. Use when the user asks to "summarize", "ship results", "wrap up ai-tune", or "finalize the experiment". Commits all artifacts, identifies winning kernels, and ships them to main with proper naming, build scripts, and documentation.
argument-hint: [optional: specific branch name to summarize]
disable-model-invocation: true
---

# AI-Tune Summary: Ship Winning Kernels to Main

Finalize an AI-tune experiment session and ship winning kernels to the main branch.

## Workflow

### Phase 1 - Finalize Current Branch

1. **Identify the experiment branch**: Should be on `ai-tune/<date>/<kernel>`. If not, check out the branch specified in `$ARGUMENTS` or ask the user.
2. **Commit all pending artifacts**:
   ```bash
   git add -A benchmark/performance/<kernel-folder>/ results.tsv
   # Also add any compiler changes in lib/ if applicable
   git commit -m "<kernel>: final experiment artifacts"
   ```
3. **Push to remote**:
   ```bash
   git push -u origin HEAD
   ```

### Phase 2 - Analyze Results on Main

1. **Switch to main and sync**:
   ```bash
   git checkout main && git pull origin main
   ```
2. **Read experiment records**: Use `git show <branch>:results.tsv` to read all experiment iterations without switching branches. Understand:
   - The baseline TFLOPS on main
   - Each iteration's TFLOPS, whether it was KEEP or DISCARD
   - The optimization idea behind each KEEP iteration
3. **Identify winners**: List all iterations that outperform main's current best. These are the candidates to ship.

### Phase 3 - Ship Winners to Main

For each winning iteration, ship it to main using the naming convention below. Process them **sequentially** (lowest iter first) and **verify each one** before proceeding.

#### Naming Convention

The naming must include the kernel family, `aitune` marker, date, and iteration ID to avoid collisions across branches:

- **`.co` kernels**: `<kernel-family>_aitune_<YYYY-MM-DD>_iter<NNN>.co`
  - Example: `gemm_sp_e4m3_aitune_2026-03-21_iter016.co`
  - Place directly in the kernel folder.
  - Add the full choreo compile+run command as a comment at the top of the file.

- **`.cu` kernels** (modified CUDA, not from `.co`): Create a self-contained subfolder:
  - Folder: `<kernel-folder>/<kernel-family>_aitune_<YYYY-MM-DD>_iter<NNN>/`
  - Contents:
    - The `.cu` source file
    - A `run.sh` script (compile + run in one command)
    - **NO local `choreo.h` copy** - use `-I"$REPO_ROOT/runtime"` to pick up the canonical header
  - The subfolder must be **self-contained and runnable** with:
    ```bash
    bash <subfolder>/run.sh
    ```

#### run.sh Template (for .cu kernels)

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/<kernel-binary-name>"

nvcc -arch sm_90a -std=c++17 \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
  -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
  -Xcompiler -static-libstdc++ -O2 --use_fast_math \
  -I"$REPO_ROOT/runtime" -I"$REPO_ROOT/extern/cutlass/include" -I"$REPO_ROOT" \
  -L/usr/local/cuda/lib64 -lcuda \
  -o "$BIN" \
  "$SCRIPT_DIR"/<kernel-source>.cu

echo "Built: $BIN"
"$BIN" "$@"
```

Key points about `run.sh`:
- Uses `$REPO_ROOT/runtime` for `choreo.h` - never bundle a local copy
- Compiles AND runs in one invocation; pass `--skip-verify` or
  `--disable-timing` as arguments
- Binary is placed inside the subfolder (not `/tmp`)
- Must be `chmod +x`

#### Shipping Steps (per winner)

1. **Extract the kernel** from the experiment branch:
   ```bash
   git show <branch>:<path-to-iter-file> > <destination-on-main>
   ```
2. **For `.co` kernels**:
   - Copy the `.co` file with the aitune naming.
   - Verify: compile with choreo, run **without** `--skip-verify`, confirm
     `Test Passed` appears AFTER numerical verification.
3. **For `.cu` kernels**:
   - Create the subfolder.
   - Copy the `.cu` file.
   - Create `run.sh` using the template above.
   - `chmod +x run.sh`.
   - Verify: run `bash run.sh`, confirm `Test Passed` appears AFTER
     numerical verification.
4. **If the iteration requires compiler changes**:
   - Cherry-pick or manually apply the compiler changes to main first.
   - Rebuild `./choreo`.
   - Then ship the kernel.

### Phase 4 - Create README

Create or update `README_<kernel-family>_aitune_<date>.md` in the kernel folder with:

1. **Summary table**: All shipped iterations with TFLOPS, efficiency, and key optimization.
2. **Build & run commands**: For each shipped kernel:
   - `.co` kernels: the choreo compile command + bash execute command.
   - `.cu` kernels: `bash <subfolder>/run.sh` (and note that run.sh compiles
     + runs in one step).
3. **Environment variables**: Document `CHOREO_TIMING_WARMUP`, `CHOREO_TIMING_REPEAT`, `CHOREO_DISABLE_TIMING`, `CHOREO_SKIP_VERIFY`.
4. **Optimization history**: Brief description of each shipped optimization.
5. **Source branch**: Reference the experiment branch for full history.

### Phase 5 - Commit and Push

```bash
git add benchmark/performance/<kernel-folder>/
git commit -m "(bench): ship <kernel-family> aitune winners to main

Ship N optimized kernels from ai-tune/<date>/<kernel>.
Performance: X -> Y TFLOPS (+Z% over baseline).
All kernels verified on SM90a."

git push origin main
```

## Verification Protocol

**MANDATORY - every shipped kernel must pass sampled verification.**

### Using `choreo::SampledVerifier`

All kernels must use `choreo::verify_spmm_sampled()` (or the equivalent
`verify_matmul_row_row_sampled` / `verify_matmul_row_col_sampled`) from
`runtime/choreo.h`. These utilities sample the MxN output at a coprime stride
and compare each sample against a CPU FP32 reference.

```cpp
choreo::SampledVerifierConfig vcfg;
vcfg.num_samples = 512;
vcfg.base_tol = <see tolerance table>;
vcfg.rel_tol = <see tolerance table>;
choreo::verify_spmm_sampled(lhs_dense_v, rhs_v, res_v, vcfg);
```

### Tolerance Guidelines

| Precision | base_tol | rel_tol |
|-----------|----------|---------|
| FP16 input, FP32 accum | 1.0 | 0.01 |
| FP16 input, FP16 accum | 16.0 | 0.05 |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 |

### Verification Checklist

- [ ] Every `.co` kernel: choreo compile succeeds, `--execute` prints `Test Passed` after real numerical comparison
- [ ] Every `.cu` kernel: `bash run.sh` succeeds (exit 0), prints `Test Passed` after real numerical comparison
- [ ] **NEVER print "Test Passed" without actual element-wise comparison** - running the kernel twice is not verification
- [ ] After commit: `git status` is clean for the shipped files
- [ ] After push: `git log --oneline -1` matches the commit

### What NOT to Ship

- Kernels where verification was removed or always prints "Test Passed"
- Kernels that bundle a local `choreo.h` copy (stale headers cause subtle bugs)
- `.cu` subfolders without a `run.sh`

## Related Skills

- `ai-tune` - the optimization loop that produces the results being shipped
- `compile-and-test` - build/run workflows for verification
