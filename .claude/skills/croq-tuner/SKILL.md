---
name: croq-tuner
description: Unified AI-guided GPU kernel optimization entry point. Handles tuning, resuming, summarizing, and shipping -- dispatched by user prompt. Use this for any ai-tune, kernel tuning, experiment finalization, or result shipping request.
---

# CroqTuner -- Unified Kernel Optimization Skill

This skill is the single entry point for all kernel optimization workflows. It reads the user prompt to determine intent and dispatches to the correct behavior.

## Prompt-Driven Dispatch

Read the user's message and classify intent into one of these modes:

### Mode A: TUNE (default)

**Trigger phrases**: "tune", "optimize", "ai-tune", "improve perf", "boost TFLOPS", kernel file path, or any request mentioning a kernel to optimize.

**Action**: Set up or resume a tuning session, then enter the FSM loop.

### Mode B: SUMMARIZE

**Trigger phrases**: "summarize", "ship", "finalize", "wrap up", "merge winners", "send to main", "ship results".

**Action**: Analyze experiment results and ship winning kernels to main.

### Mode C: STATUS

**Trigger phrases**: "status", "where are we", "progress", "what iteration", "show results".

**Action**: Read `loop-state.json` and `results.tsv`, report current state.

### Mode D: RESUME

**Trigger phrases**: "continue", "resume", "keep going", "next iteration", or when no explicit intent but an active FSM exists on the current branch.

**Action**: Same as TUNE, but skip branch setup -- read state and resume from current FSM position.

If ambiguous, default to TUNE. If the current branch is `ai-tune/*` and a `loop-state.json` exists, default to RESUME.

---

## Mode A & D: TUNE / RESUME

### Step 1: Parse Request

Extract from the user request:
- **kernel_family**: e.g., `gemm_sp`, `matmul`, `moe_gemm`, `blockscale_gemm_v2`
- **seed_kernel**: the starting kernel file path (or "best on main")
- **target_arch**: e.g., `sm_90a` (default)
- **precision**: e.g., `f16`, `f8_e4m3`
- **max_iterations**: default 30

### Step 2: Determine Session

Session ID format: `<DATE>_<kernel_mnemonic>`
- DATE = today's date YYYY-MM-DD
- kernel_mnemonic = `<kernel_family>_<precision>` or similar

Branch: `ai-tune/<DATE>/<kernel_mnemonic>`

### Step 3: Check for Existing Session

```bash
EXPERIMENT_DIR="benchmark/performance/<kernel_family>/aitune/<session_id>"
STATE_FILE="$EXPERIMENT_DIR/loop-state.json"

if [ -f "$STATE_FILE" ]; then
    echo "RESUMING existing session"
else
    echo "STARTING new session"
fi
```

### Step 4: Branch Management

**New session:**
```bash
git checkout main && git pull
git checkout -b ai-tune/<DATE>/<kernel_mnemonic>
```

**Existing session (same branch):**
```bash
git checkout ai-tune/<DATE>/<kernel_mnemonic>
git pull origin ai-tune/<DATE>/<kernel_mnemonic> 2>/dev/null || true
```

NO `-resume-N` suffixes. ONE branch per experiment. Continue stacking iterations.

### Step 5: Read Sub-Skills and Dispatch

1. Read `ai-tune-artifacts` skill -- understand path conventions
2. Dispatch to `fsm-engine` skill -- enter FSM loop

The FSM engine handles: INIT -> BASELINE -> PROFILE -> IDEATE -> IMPLEMENT -> MEASURE -> DECIDE -> STORE, in a loop until `max_iterations`.

### Step 6: When FSM Reaches DONE

Automatically proceed to Mode B (SUMMARIZE) below.

---

## Mode B: SUMMARIZE

### Phase 1 -- Finalize Current Branch

1. Confirm the experiment branch (`ai-tune/<date>/<kernel>`).
2. Read `loop-state.json` to confirm FSM is in `DONE` state or user explicitly requests early summarization.
3. Commit all pending artifacts:
   ```bash
   git add -A benchmark/performance/<kernel-family>/aitune/<session_id>/
   git commit -m "<kernel>: final experiment artifacts"
   ```
4. Push to remote:
   ```bash
   git push -u origin HEAD
   ```

### Phase 2 -- Analyze Results on Main

1. Switch to main:
   ```bash
   git checkout main && git pull origin main
   ```
2. Read experiment records via `git show <branch>:benchmark/performance/<family>/aitune/<session>/results.tsv`.
3. Identify winners: all iterations with decision=KEEP that outperform main's current best.
4. Winners are at `iters/iter<NNN>_<tag>.<ext>` in the experiment dir.

### Phase 3 -- Ship Winners to Main

Process winners sequentially (lowest iter first). Verify each before proceeding.

**Naming Convention** (see `ai-tune-artifacts` for authoritative paths):

- `.co` kernels: `<family>_aitune_<YYYY-MM-DD>_iter<NNN>.co` placed in `benchmark/performance/<family>/`
- `.cu` kernels: subfolder `<family>_aitune_<YYYY-MM-DD>_iter<NNN>/` containing the `.cu` source and `run.sh`

**Shipping Steps (per winner):**
1. Extract from experiment branch: `git show <branch>:<path> > <destination>`
2. For `.co`: compile with choreo, verify `Test Passed`
3. For `.cu`: create subfolder, write `run.sh` using template from `ai-tune-artifacts`, `chmod +x`, verify `bash run.sh` passes
4. If compiler changes needed: cherry-pick to main first, rebuild `./choreo`

**run.sh Template:**
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PATH=/usr/local/cuda/bin:$PATH

BIN="$SCRIPT_DIR/<kernel_binary_name>"

nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
  -D__USE_CUDA_TYPE__ -D__CHOREO_DMA_DIAGNOSIS__ \
  -Xcompiler -static-libstdc++ -O3 --use_fast_math -ftz=true \
  -I"$REPO_ROOT/runtime" -I"$REPO_ROOT/extern/cutlass/include" \
  -L/usr/local/cuda/lib64 -lcuda \
  -o "$BIN" \
  "$SCRIPT_DIR/<kernel_source>.cu"

echo "Built: $BIN"
"$BIN" "$@"
```

### Phase 4 -- Create README

Create `README_<family>_aitune_<date>.md` in the kernel folder:
1. Summary table: shipped iterations, TFLOPS, efficiency, optimization
2. Build & run commands per kernel
3. Environment variables: `CHOREO_TIMING_WARMUP`, `CHOREO_TIMING_REPEAT`, `CHOREO_DISABLE_TIMING`, `CHOREO_SKIP_VERIFY`
4. Optimization history
5. Source branch reference

### Phase 5 -- Commit and Push

```bash
git add benchmark/performance/<kernel-folder>/
git commit -m "(bench): ship <family> aitune winners to main

Ship N optimized kernels from ai-tune/<date>/<kernel>.
Performance: X -> Y TFLOPS (+Z% over baseline).
All kernels verified on SM90a."

git push origin main
```

---

## Mode C: STATUS

1. Read `loop-state.json` and report:
   - Current FSM state, iteration number, max iterations
   - Baseline TFLOPS, current best TFLOPS, improvement %
   - Consecutive discards, last bottleneck
2. Read `results.tsv` and show a formatted table of all iterations
3. Read `idea-log.jsonl` and report ideas tried

---

## Verification Protocol

**MANDATORY for both TUNE and SUMMARIZE.**

### Tolerance Guidelines

| Precision | base_tol | rel_tol |
|---|---|---|
| FP16 input, FP32 accum | 1.0 | 0.01 |
| FP16 input, FP16 accum | 16.0 | 0.05 |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 |

### Checklist

- Every `.co` kernel: choreo compile succeeds, `--execute` prints `Test Passed` after numerical comparison
- Every `.cu` kernel: `bash run.sh` exits 0, prints `Test Passed` after numerical comparison
- NEVER print "Test Passed" without actual element-wise comparison
- NEVER bundle a local `choreo.h` copy
- NEVER ship `.cu` subfolders without a `run.sh`

---

## Quick Reference

### Choreo (.co) Build
```bash
./choreo -gs -t cute -arch=sm_90a [flags] kernel.co -o output.cute.result
bash output.cute.result --execute
```

### CUDA (.cu) Build
```bash
nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -D__CHOREO_TARGET_CUTE__ \
  -D__USE_CUDA_TYPE__ -Xcompiler -static-libstdc++ -O3 \
  -I runtime -I extern/cutlass/include \
  -o binary kernel.cu
```

### Timing
```bash
CHOREO_TIMING_WARMUP=5 CHOREO_TIMING_REPEAT=50 <command>
```

### NCU
```bash
/usr/local/cuda/bin/ncu --set full --target-processes all -o <out>.ncu-rep <command>
```

---

## Sub-Skills Reference

| Skill | Purpose | When to Read |
|---|---|---|
| `ai-tune-artifacts` | Naming, paths, storage rules | Always (before creating any files) |
| `ncu-bottleneck` | NCU profiling and bottleneck interpretation | During PROFILE step |
| `fsm-engine` | FSM-driven tuning loop engine | During TUNE/RESUME |
| `choreo-syntax` | Choreo DSL reference | When editing .co files |
| `compile-and-test` | Build and run workflows | When compiling/verifying |
