---
name: ai-tune-artifacts
description: Unified naming, path, locating, and storage rules for all ai-tune experiment artifacts. Read this skill before creating any files during ai-tune experiments. Ensures nothing is lost across iterations, branches, compactions, and future revisiting.
---

# AI-Tune Artifact Management

This skill defines the single source of truth for how ALL ai-tune artifacts are named, stored, and organized. Every other ai-tune skill defers to this document for path decisions.

## Directory Layout

All ai-tune artifacts live under `benchmark/performance/<kernel_family>/`. Each kernel family (e.g., `gemm_sp`, `matmul`, `moe_gemm`, `blockscale_gemm_v2`) is a top-level folder.

### Per-Kernel-Family Structure

```
benchmark/performance/<kernel_family>/
├── <kernel_family>_<variant>.co               # Development kernel .co files (not ai-tune)
├── <kernel_family>_aitune_<DATE>_iter<NNN>.co  # Shipped .co winners (on main)
├── <kernel_family>_aitune_<DATE>_iter<NNN>/    # Shipped .cu winners (on main)
│   ├── <kernel_binary_name>.cu
│   └── run.sh
├── README_<kernel_family>_aitune_<DATE>.md     # Shipped experiment summary
└── aitune/                                     # Active experiment workspace
    └── <DATE>_<mnemonic>/                      # One experiment session
        ├── loop-state.json                     # FSM state (fsm-engine)
        ├── results.tsv                         # Iteration log
        ├── idea-log.jsonl                      # Idea dedup log
        ├── compaction-summary.md               # Compaction-safe resume state
        ├── iters/                              # ALL iteration kernel sources
        │   ├── iter000_baseline.co             # or .cu
        │   ├── iter001_<tag>.co                # or .cu
        │   ├── iter002_<tag>.cu
        │   └── ...
        ├── logs/                               # Build/run/ncu logs per iter
        │   ├── iter000_compile.log
        │   ├── iter000_timing.log
        │   ├── iter001_compile.log
        │   ├── iter001_ncu.log
        │   ├── iter001_timing.log
        │   └── ...
        └── ncu/                                # ncu report files
            ├── iter000_baseline.ncu-rep
            ├── iter001_<tag>.ncu-rep
            └── ...
```

## Naming Conventions

### Experiment Session ID

Format: `<DATE>_<mnemonic>`
- `DATE`: `YYYY-MM-DD` (e.g., `2026-04-03`)
- `mnemonic`: short kernel identifier (e.g., `gemm_sp_f16`, `matmul_f16`, `moe_gemm_fp8`)

Example: `2026-04-03_gemm_sp_f16`

### Branch Name

Format: `ai-tune/<DATE>/<mnemonic>`

Examples:
- `ai-tune/2026-04-03/gemm_sp_f16`
- `ai-tune/2026-04-03/matmul_f16`

Rules:
- ONE branch per experiment. No `-resume-N` suffixes. Continue on same branch.
- If branch exists, check it out and continue from where it left off.
- Iteration counter stacks monotonically. Read `results.tsv` / `loop-state.json` for the latest iter.

### Iteration File Naming

Format: `iter<NNN>_<tag>.<ext>`
- `NNN`: 3-digit zero-padded iteration number (e.g., `001`, `023`, `143`)
- `tag`: brief optimization descriptor (e.g., `warpn128`, `stages3`, `l2_promo`, `baseline`)
- `ext`: `.co` or `.cu`

Examples:
- `iter000_baseline.co`
- `iter001_warpn128.co`
- `iter015_inline_ptx_fence.cu`

Rules:
- iter000 is ALWAYS the baseline/seed kernel (the starting point)
- KEEP all iteration files, even discarded ones. They are your experiment history.
- Monotonically increasing. Never reuse or skip numbers.

### Log File Naming

Format: `iter<NNN>_<type>.log`

Types:
- `compile` -- choreo/nvcc compile output
- `timing` -- benchmark timing output (TFLOPS, efficiency)
- `ncu` -- ncu profiling summary (text extract from ncu report)
- `verify` -- verification output

### Shipped Kernel Naming (on main)

When shipping winners to main (via croq-tuner SUMMARIZE mode):
- `.co` kernels: `<family>_aitune_<DATE>_iter<NNN>.co` placed directly in `<kernel_family>/`
- `.cu` kernels: subfolder `<family>_aitune_<DATE>_iter<NNN>/` containing:
  - The `.cu` source
  - `run.sh` (self-contained compile + run)
  - NO local `choreo.h` copy -- use `$REPO_ROOT/runtime`

## run.sh Template

Every shipped `.cu` kernel folder MUST contain a self-contained `run.sh`:

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

Adjust `REPO_ROOT` depth based on actual folder nesting.

## results.tsv Format

Tab-separated, one row per iteration:

```
iter	kernel	arch	tflops	eff_pct	decision	bottleneck	idea_summary	run_command
```

- `iter`: `iter000`, `iter001`, etc.
- `kernel`: filename relative to iters/ (e.g., `iter001_warpn128.co`)
- `arch`: e.g., `sm_90a`
- `tflops`: measured TFLOPS (float)
- `eff_pct`: hardware efficiency percentage
- `decision`: `BASELINE`, `KEEP`, `DISCARD`, `DISCARD_COMPILE_FAIL`, `DISCARD_INCORRECT`
- `bottleneck`: ncu bottleneck category before this iteration
- `idea_summary`: human-readable description of the optimization
- `run_command`: exact command to reproduce the measurement

## idea-log.jsonl Format

Append-only, one JSON object per line:

```json
{"iter": 5, "idea": "L2_256B on both TMA descriptors", "category": "macro", "bottleneck_before": "l2_throughput", "justification": "ncu shows 78% L2 hit rate", "result": "DISCARD", "tflops": 415.3, "tflops_delta": -6.2}
```

Categories: `macro`, `structural`, `choreo`, `ncu_micro`

## Path Resolution

All paths are relative to repo root. Given experiment session `<DATE>_<mnemonic>` in kernel family `<family>`:

| Artifact | Path |
|---|---|
| Experiment root | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/` |
| FSM state | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/loop-state.json` |
| Results log | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/results.tsv` |
| Idea log | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/idea-log.jsonl` |
| Iter N kernel | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/iters/iter<NNN>_<tag>.<ext>` |
| Iter N compile log | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/logs/iter<NNN>_compile.log` |
| Iter N timing log | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/logs/iter<NNN>_timing.log` |
| Iter N ncu report | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/ncu/iter<NNN>_<tag>.ncu-rep` |
| Compaction summary | `benchmark/performance/<family>/aitune/<DATE>_<mnemonic>/compaction-summary.md` |

## Verification Tolerances

| Precision | base_tol | rel_tol |
|---|---|---|
| FP16 input, FP32 accum | 1.0 | 0.01 |
| FP16 input, FP16 accum | 16.0 | 0.05 |
| FP8 E4M3 input, FP16 accum | 0.5 | 0.01 |

## Rules

1. **NEVER place iteration files in /tmp.** All artifacts go under the experiment directory.
2. **NEVER delete iteration files.** Even discarded kernels are valuable history.
3. **NEVER bundle a local `choreo.h` copy.** Always use `$REPO_ROOT/runtime`.
4. **ALWAYS capture logs.** Redirect compile/run output to log files. Tee if you need stdout.
5. **ALWAYS use the exact naming conventions above.** No ad-hoc names.
6. **Git commit after every KEEP iteration.** Include the kernel file, results.tsv, idea-log.jsonl.
