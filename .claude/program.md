# Choreo Kernel Auto-Tuner

This is an experiment to have an AI agent autonomously improve high-performance GPU kernels
compiled via the Choreo EDSL. It modifies `.co` source files, profiles with Nsight Compute,
keeps or discards based on measured TFLOPS, and repeats indefinitely.

## Architecture

Unlike training-research AutoRA (which optimizes stochastic ML objectives), this is a
deterministic numerical optimization: each mutation is either faster or it is discarded.
The "fitness signal" is **wall-clock TFLOPS** and **hardware efficiency %** measured by
`ncu`. The "genome" is the `.co` Choreo source file under `benchmark/performance/matmul/`.

### Three roles

| Role | What it does | Can the agent modify it? |
|---|---|---|
| **Fixed infrastructure** | Choreo compiler, CUDA toolkit (ncu, nvcc), benchmark harness (main, timing, verification), project build system | **No** — ever |
| **Mutable genome** | The `.co` kernel source file | **Yes — only this** |
| **program.md** | This file: context, loop protocol, constraints | Human only |

## Setup

Work with the user to agree on:

1. **Run tag**: propose `YYYY-MM-DD/kernel-name` (e.g. `2026-03-20/matmul`). The branch
   `ai-tune/<tag>` must not already exist.
2. **Baseline kernel**: the `.co` file chosen as the starting genome. Must already exist
   under `benchmark/performance/matmul/`.
3. **Target architecture**: the SM (e.g. `sm_90a`, `sm_86`). Affects compile `-arch` flag.
4. **Iteration cap**: maximum attempts per loop before giving up on a stuck idea.
5. **Experiment logging contract**: for every measured result, append the exact
   compile/profile/run command line used for that data point into `results.tsv`
   (`run_command` column) so the measurement is fully reproducible.

Once confirmed:

```
git checkout -b ai-tune/<tag>
```

Then proceed directly to the Experimentation section. Do NOT write a plan, do NOT ask
for confirmation — just start the loop.

## Context every agent needs to know

### How Choreo compiles

```
.co file  ──[choreo compiler]──►  .cute.result (CUTE/CUDA source)
                                     │
                              [nvcc / copp wrapper]
                                     │
                                    binary
```

- `./choreo` is the main compiler CLI. Run it from repo root.
- `./copp` is the C preprocessor wrapper used internally.
- The generated `.cute.result` is human-readable C++/CUTE code.
- Choreo targets `cute` architecture and emits CUDA C++ with CuTe library abstractions.
- Tensor-core primitives lower to `wgmma`, `mma.sync`, `wmma`, or `wgmma.sp` depending on
  SM and MMA shape.

### Choreo GEMM kernel anatomy

Every matmul `.co` under `benchmark/performance/matmul/` has two parts:

**1. Choreo kernel (`__co__ void matmul(...)`)** — the genome you edit:
- `parallel {block_m, block_n} by [...] : block` — CTA-level tiling.
- `parallel p1 by N : group-4` — warpgroup (warp-specialization): p1==0 is producer TMA
  loader, p1>0 are consumer MMA warps.
- `shared f16 [...]` — shared-memory staging buffers.
- `shared event full[N], empty[N]` — producer/consumer event synchronization.
- `tma.copy(.async)` / `dma.copy(.async)` — shared-memory load.
- `mma.load(.swiz)` / `mma.row.row` / `mma.commit` — tensor-core compute.
- `mma.store` / `tma.copy` — store result to global.

**2. Host harness (`main()`)** — fixed infrastructure:
- `choreo::make_spanview` / `choreo::make_spandata` — managed buffers.
- `choreo::timing(...)` — wall-clock measurement with warmup/repeat.
- Prints `Timing avg ms`, `TFLOPS`, `HW efficiency %`.
- `--disable-timing` skips measurement and correctness check.
- `--skip-verify` skips verification only.
- `--flops=<float>` overrides FLOP count for TFLOPS calculation.
- `CHOREO_DISABLE_TIMING=1`, `CHOREO_SKIP_VERIFY=1` env vars.

### Tunable parameters in existing kernels

The following are the canonical degrees of freedom. All are `#define` macros:

| Macro | Typical values | Effect |
|---|---|---|
| `MATMUL_WARP_M` | 64 | Warp-group height (blockscale M) |
| `MATMUL_WARP_N` | 64, 128, 192, 256 | Warp-group width |
| `MATMUL_TILE_M` | 64, 128, 192 | CTA tile height |
| `MATMUL_TILE_K` | 64 | CTA tile K-depth |
| `MATMUL_WARP_K` | 16 | WGMMA K-step per MMA instruction |
| `MATMUL_SWIZ` | 32, 64, 128 | TMA swizzle factor = 2*MATMUL_TILE_K |
| `MATMUL_STAGES` | 2, 3, 4 | Producer/consumer pipeline depth |
| `NUM_SMS` | 114 (H800) | CTA count for persistent kernels |
| `MATMUL_DEFAULT_M/N/K` | 2048 (or other) | Problem size |

### Available MMA / copy primitives (from Choreo syntax)

```
# Data movement
dma.copy src => dst;
f = dma.copy.async src => shared after i_shared;
tma.copy.swiz<N> src => dst;
tma.copy.async<event>.swiz<N> src => dst;

# Compute
mma.fill.f16 0.0f;
mma.load.swiz<N> buffer;
mma.row.row  accumulator, a, b;
mma.row.row.scale accumulator, a, b, scale_a, scale_b;
mma.row.col.sp accumulator, a, b;    # sparse MMA
mma.op <shape> accumulator, a, b;  # preferred unified form
mma.commit;
mma.store accumulator, output_s;
mma.store.transp accumulator, output_s;

# Control
parallel {p, q} by [a, b] : block;
parallel p by n : group-4;
inthreads.async (condition) { ... };
foreach {i} in [N] { ... };
wait event;
trigger event;
```

### Copy-shaping patterns

```
chunkat(...)              # natural partition from bounded indices
view(...).from(...)       # explicit window + optional strides + origin
subspan(...).at(...)      # fixed tile extents + anchor
subspan(...).step(...).at(...)  # repeated tiles with spacing (for staged/swizzled layout)
```

### Relevant reference files (read these before editing a `.co`)

- `benchmark/performance/matmul/matmul_f16_dyn_sm90.co` — baseline FP16 GEMM with TMA+swizzle.
- `benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co` — 1-producer/1-consumer warp-specialized GEMM with staged events.
- `benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` — 1-producer/3-consumer warp-specialized GEMM (wider CTA tile).
- `benchmark/performance/matmul/matmul_f16_dyn_persis_sta_sm90.co` — persistent CTA with `.step().at()` tile iteration.
- `benchmark/performance/blockscale_gemm/blockscale_gemm_e4m3_dynamic_sm90.co` — FP8 blockscaled GEMM with scale operands.

## Build and run workflow

### Prepare — initialize reproducible logging (do once per run tag)

Before baseline measurement, set the run tag in `results.tsv`, ensure the header includes
the `run_command` column, and log the exact benchmark command used to produce each row.
This is mandatory for every iteration and every measured data point (baseline and mutants).
Use the literal command line (including env vars and flags) that generated the reported
TFLOPS so future reruns can reproduce the same numbers.

### Step 0 — Baseline measurement (do once)

```bash
# 1. Identify the baseline kernel file and arch
KERNEL=benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co
ARCH=sm_90a
KERNEL_OUT=benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c

# 2. Compile
./choreo -gs -t cute -arch=$ARCH $KERNEL -o ${KERNEL_OUT}.cute.result

# 3. Baseline timing (this is your reference TFLOPS)
bash ${KERNEL_OUT}.cute.result --execute

# 4. Baseline ncu profiling (one-time bottleneck diagnosis)
#    Run this BEFORE making ANY changes so you know where time goes.
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o ${KERNEL_OUT}_ncu_baseline \
  bash ${KERNEL_OUT}.cute.result --execute

# Read ncu output:
#   - sm__throughput.avg.pct_of_peak_sm  → SM utilization
#   - sm__throughput.avg.pct_of_peak_tensor  → Tensor-core utilization
#   - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  → global load throughput
#   - dram__bytes.sum  → memory bandwidth
#   - lts__t_sectors_lookup_miss.sum  → L2 cache miss rate
#   - wgmma...  → WGMMA-specific metrics on Hopper
#   - Any "achieved occupancy" vs "single-issue occupancy" gap
# Bottleneck categories:
#   - Low SM% + high tensor%  → compute-bound (good, near peak)
#   - High SM% + low tensor%  → memory-bound (try TMA, larger tiles, more stages)
#   - High L2 miss rate  → cache-bound (try larger tiles, different tile shape)
```

### Step 1 — ncu bottleneck analysis of CURRENT kernel

Profile the current best kernel with ncu. Read the report to identify the bottleneck.
If you made changes in a previous iteration, ALWAYS re-profile from this step.

```bash
KERNEL=benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co
ARCH=sm_90a
KERNEL_OUT=<path_to_current_best>

# Always recompile first if source changed
./choreo -gs -t cute -arch=$ARCH $KERNEL -o ${KERNEL_OUT}.cute.result
you may need --use-warpspec --use-prepack or other optional switches to enable feature

# ncu profiling
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o ${KERNEL_OUT}_ncu_iter<N> \
  bash ${KERNEL_OUT}.cute.result --execute
```

Read the `.ncu-rep` file (either via `ncu --import <file>` output or open in NSight Compute GUI).
Key metrics to interpret:

| Bottleneck symptom | ncu indicator | Typical fix |
|---|---|---|
| SM underutilized | sm__throughput.avg.pct < 80% | Increase CTAs, check occupancy |
| Memory bound | sm__pipe_tensor_op_pct < 50%, high l1tex ld | Larger tiles, TMA async, more stages |
| L2 thrash | lts__lookup_miss.sum high vs total | Change tile shape (M/N/K ratio) |
| WGMMA latency hidden | low wgmma__ops issue rate | Better producer/consumer overlap, more stages |
| Register pressure | sm__occ_pct_of_peak_smem_per_block_at_block_limit low | Reduce tile size, fewer threads per CTA |

### Step 2 — Raise an optimization idea

Based on the ncu analysis from Step 1, identify ONE concrete bottleneck and propose ONE
targeted optimization. The idea must be specific: not "make it faster" but "reduce X
bottleneck by doing Y because Z".

**Brainstorming principles:**
- Ideas come from ncu data, not from guessing.
- Each iteration should test ONE idea. Multiple ideas in one commit make it impossible
  to know what worked.
- If an idea was tried before (see results.tsv), do NOT repeat the same change.
- Ideas may involve: tuning #define constants, changing tile shapes, adding/removing
  pipeline stages, warp-specialization ratio changes (1p1c → 1p3c), switching from
  sync to async TMA, persistent CTA vs static CTA, blockscale vs non-blockscale,
  swizzle factor, K-tiling depth, register blocking, smem bank冲突, etc.
- Ideas may also involve Choreo DSL syntax: a new primitive form that doesn't exist
  yet in the compiler (e.g., a new MMA shape, a new copy form). If so, you must also
  implement the compiler change in `lib/` and rebuild `./choreo` before the kernel can
  use it. Treat the compiler change and the kernel change as ONE atomic optimization.
- If the idea requires a compiler change, load the `develop-compiler` and
  `choreo-syntax` skills, implement the DSL extension first, rebuild, then use it in
  the kernel.

**What NOT to do:**
- Do not change the host harness (main, timing, verification) — it is infrastructure.
- Do not change problem size (M/N/K) unless explicitly asked — measurement must be
  comparable across iterations.
- Do not disable verification or timing to "cheat" a better score.
- Do not submit ideas that are identical to something already in results.tsv.
- Do not guess without ncu data. Profile first, hypothesize second.

### Step 3 — Implement and debug

For each iteration, create a new versioned candidate `.co` file and edit that candidate only.
Do not mutate the current best kernel file in place. Keep every tried candidate file for traceability.
Compiler+kernel ideas still remain one atomic change when required.

```bash
# Edit the kernel
$EDITOR benchmark/performance/matmul/<your_kernel>.co

# Compile
./choreo -gs -t cute -arch=$ARCH benchmark/performance/matmul/<your_kernel>.co \
  -o benchmark/performance/matmul/<your_kernel>.cute.result

# Run functional test (no timing)
bash benchmark/performance/matmul/<your_kernel>.cute.result --execute
```

If it fails to compile:
- Read the choreo compiler error. Fix the `.co` source.
- Common issues: shape mismatch in `subspan(...).step().at()`, wrong swizzle factor,
  WGMMA constraint violations (WARP_M must be 64, WARP_K must be 16 for FP16 on SM90),
  invalid event indexing in staged pipeline.

If it compiles but fails verification:
- Read the error. Common issues: wrong tile shape causing out-of-bounds, accumulator
  precision loss (try WARP_N=128 instead of 256), race condition in producer/consumer.
- The verification samples only `[0:128, 0:256]` of the output — edge tiles may have
  different behavior. Do not blindly relax the tolerance.
- If the bug is in the Choreo DSL itself (not your kernel), load the `develop-compiler`
  skill and fix the compiler first.

If it compiles and passes verification:
- Proceed to Step 4.

**Hard debugging protocol:** If after 3 distinct fix attempts within the same idea
the kernel still doesn't work, ABANDON the idea. Do not spend more than 3 iterations
debugging one idea. Discard the broken code, revert to last known-good state, and go
back to Step 2 with a new idea.

### Step 4 — Profile and decide

```bash
KERNEL=benchmark/performance/matmul/<your_kernel>.co
KERNEL_OUT=benchmark/performance/matmul/<your_kernel>

# Profile
/usr/local/cuda/bin/ncu --set full --target-processes all \
  -o ${KERNEL_OUT}_ncu_iter<N> \
  bash ${KERNEL_OUT}.cute.result --execute

# Timing run
bash ${KERNEL_OUT}.cute.result --execute
```

Extract `TFLOPS` and `HW efficiency %` from the output.

**Decision rule:**
- If TFLOPS > current best TFLOPS: **KEEP** — commit, update best, go to Step 1.
- If TFLOPS ≤ current best: **DISCARD** — revert to last best, go to Step 2.

### Step 5 — Commit and iterate

After a successful (kept) optimization:

```bash
# Commit on ai-tune/<tag> branch
git add benchmark/performance/matmul/<kernel>.co
# Also add any compiler changes if applicable
git add lib/ ...
git commit -m "iter <N>: <brief description of change> — TFLOPS: X -> Y"

# Update results.tsv (append one row; include reproducible run command)
echo -e "iter<N>\t<KERNEL>\t<ARCH>\t<TFLOPS>\t<EFF%>\t<BOTTLENECK_CATEGORY>\t<RUN_COMMAND>\t<IDEA_SUMMARY>" >> results.tsv
```

Then **repeat from Step 1** using the improved kernel as the new baseline.

## Results tracking

Create and maintain `results.tsv` in repo root with columns:

```
iter	kernel	arch	tflops	eff%	bottleneck_before	run_command	idea_summary
```

Append one row per iteration. This is the experiment log — it is the agent's memory.
Before raising a new idea, consult this file to avoid repeating ideas.

## Stop conditions

- User manually stops the loop.
- 10 consecutive discarded ideas (stuck in local minimum — report to user).
- Compiler crashes repeatedly (likely a Choreo bug, escalate to user).
- No more kernel files remain to try as baselines.

## Constraints summary

| Allowed | Forbidden |
|---|---|
| Edit ONE `.co` kernel file per iteration | Change host harness, main(), timing, verification |
| Edit Choreo compiler (`lib/`) as part of a new DSL feature | Modify `./choreo` or `./copp` binary wrappers |
| Change `#define` macro values | Change problem size (M/N/K) mid-experiment |
| Add/remove pipeline stages, tiles, warpgroups | Disable timing or verification |
| Implement new MMA/copy forms if they need compiler support | Break the build |
| Profile with ncu, iterate | Guess without data |

## Skill references

Before touching `.co` files, the agent should load:
- `choreo-syntax` — for DSL grammar, primitive forms, and copy patterns.
- `profiling` — for ncu/nsys invocation and metric interpretation.
- `compile-and-test` — for build/run/debug workflows.
- `develop-compiler` — only if a new DSL feature needs compiler support.
- `performance-bottleneck-analysis` — for interpreting ncu reports.

## Practical workflow summary (copy-paste reference)

```bash
# Baseline
KERNEL=benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c.co
ARCH=sm_90a
OUT=benchmark/performance/matmul/matmul_f16_dyn_sm90_warpspec_1p1c

./choreo -gs -t cute -arch=$ARCH $KERNEL -o ${OUT}.cute.result
/usr/local/cuda/bin/ncu --set full --target-processes all -o ${OUT}_ncu_baseline bash ${OUT}.cute.result --execute
bash ${OUT}.cute.result --execute   # get TFLOPS

# Iter loop
# Step 1: ncu profile current best
/usr/local/cuda/bin/ncu --set full --target-processes all -o ${OUT}_ncu_iterN bash ${OUT}.cute.result --execute

# Step 2: propose idea (in head, from ncu data)

# Step 3: edit kernel, compile, test
$EDITOR $KERNEL
./choreo -gs -t cute -arch=$ARCH $KERNEL -o ${OUT}.cute.result
bash ${OUT}.cute.result --execute   # must pass verification

# Step 4: profile and compare
/usr/local/cuda/bin/ncu --set full --target-processes all -o ${OUT}_ncu_iterN bash ${OUT}.cute.result --execute
bash ${OUT}.cute.result --execute   # extract TFLOPS

# Step 5: keep or discard
# keep: git commit, update results.tsv, repeat from Step 1
# discard: git checkout -- $KERNEL, repeat from Step 2
```

IMPORTANT: NEVER STOP THE LOOP UNTIL USER MANUALLY STOPPED, USER MAY LEAVE THE EXPERIMENT OVER NIGHT, SO DO NOT ATTEMPT TO ASK USER, DO ALL DECISION BY YOURSELF. DO NOT ATTEMPT TO SUMMARY WHEN CONTEXT WINDOW FULL, JUST COMPACTION AND READ THIS FILE AGAIN TO RESUME THE PROGRESS. LOOP INFINITELY.
