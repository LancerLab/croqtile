---
name: croqtile-developer
disable-model-invocation: true
description: |
  Unified skill for Croqtile (Choreo) compiler development. Use when building
  the compiler, running or debugging .co files, modifying lib/ or tools/ sources,
  troubleshooting compiler passes, or editing Choreo syntax. Covers make, ./choreo,
  lit.sh, GPU execution, cuda-gdb, and the full compiler option reference.
---

### Section 1: Overview & Behavior

This skill is the single reference for **Croqtile** (formerly Choreo): the public, OSS CUDA/CuTe GPU compiler. It covers building the compiler and tools, running and testing `.co` sources, understanding the compiler pass pipeline and options, developing compiler changes under `lib/` and `tools/`, debugging both compiler and generated GPU executables, and editing `.co` files with correct syntax and repo patterns.

This repository targets **GPU (CUDA/CuTe) only**. The codebase uses **C++17**, **LLVM-style clang-format** (see `Documents/Developer/coding-style.md` and `.clang-format`: IndentWidth 2, ColumnLimit 80, PointerAlignment Left). Naming: PascalCase for classes, types, and namespaces; camelCase for functions; snake_case for variables. Main entrypoints: **`make`** (build), **`./choreo`** (compiler, symlink into `build/`), **`./copp`** (preprocessor), **`tests/lit.sh`** (test runner).

### Behavior rules

- Execute build, test, and debug commands immediately without asking for confirmation unless Section 11 applies.
- Use default paths and settings unless the user specifies otherwise.
- If a command fails, capture the output and continue with the next diagnostic step.
- Assume `./choreo`, `nvcc`, `cuda-gdb`, and related CUDA tools are safe to run without confirmation.

### Response language alignment

- If the user prompt is in Chinese, respond in Chinese.
- If the user prompt is in English, respond in English.
- If the prompt is mixed, default to the language of the latest user message.

### Related files

- `compiler-options.md` - complete compiler option reference (all categories A-F)
- `debug-scenarios.md` - 10 debug scenarios with symptom-to-command mappings, decision trees, CUDA debugging
- `choreo-syntax-reference.md` - .co syntax inventory with access-shaping patterns and representative files
- For C++ style, review checklists, and static analysis: see `croqtile-code-style` skill

---

### Section 2: Building the Compiler

### Mandatory build protocol

**All build, test, and clean commands MUST be run from the project root directory using `make`.** This is the only supported workflow. The Makefile orchestrates CMake, Ninja, and all tooling internally.

**PROHIBITED actions (never do these):**

- Do NOT run `cmake` directly (e.g. `cmake -S . -B build`, `cmake --build build`).
- Do NOT run `ninja` or `make -C build/` directly.
- Do NOT `cd` into `build/`, `build-debug/`, or `build-release/` to invoke build tools.
- Do NOT hand-roll compiler invocations (e.g. `g++ -std=c++17 ...`) for building the project.
- Do NOT write generated or intermediate files under `/tmp`. Place temporary outputs under the workspace (e.g. `build/`).

### Build commands (from project root, always)

| Command | Description |
|---|---|
| `make` | Default build (Release, cmake+ninja) |
| `make -jN` | Parallel build (N = core count) |
| `make debug` | Debug build → `build-debug/` |
| `make release` | Release build → `build-release/` |
| `make clean` | Clean all build artifacts |
| `make test` | Run full test suite |
| `make test-debug` | Test against debug build |
| `make format` | Format C++ sources (run before committing) |

After a successful build, `./choreo` and `./copp` symlinks are created/updated automatically.

### Environment

- CUDA: commonly `/usr/local/cuda` (`CUDA_HOME`).
- CuTe/CUTLASS: `extern/cutlass` (`CUTE_HOME`). Override if needed.

### Rebuild decision

- If `./choreo` exists and points at a valid binary, you often do not need to rebuild.
- If `.cpp`/`.hpp` files under `lib/` or `tools/` changed → run `make`.
- On compile errors or broken symlinks → `make clean && make`.

---

### Section 3: Compiler Architecture

Internally, Croqtile has two major flows: semantic analysis and codegen. Each pass has a unique uppercase name usable in options such as `-dv`, `-tv`, `-pa`, `-sa`, and `-ds`.

### Full pass pipeline

```
Preprocess => Parse => SEMA (Early Semantic) => NORM (Normalization) => VALNO (Value Numbering) => INFER (Type Inference)
                                      +-- COMP_N
                                      +-- PBFILL
                                      \-- HINT-CHECKER
=> LATENORM (Late Normalization) => BUFGEN => BUFFER-INFO-COLLECT
=> MEMORYREUSE => LIVENESS => MEMANLZ => MEMREUSE
=> CHECK (Semantic Check) => CODEGEN => CG_INFO => FUTANLY
```

### Pass quick reference

| Pass Name | Description | Best Debug Use |
|---|---|---|
| `SEMA` | Early semantic analysis | Variable declarations, scopes, basic semantic errors |
| `NORM` / `COMP_N` | AST normalization / desugaring | Loop structures, parallel syntax expansion |
| `PBFILL` | Parallel-by hierarchy fill | block/thread/group hierarchy assignment |
| `HINT-CHECKER` | Compiler hint validation | Annotation/property legality |
| `VALNO` | Value Numbering | Constant propagation, shape aliasing, span inference |
| `INFER` | Type inference | Types for variables/expressions/function signatures |
| `LATENORM` | Late normalization | DMA buffer handling, future semantics |
| `BUFGEN` | Buffer generation | Future buffer allocation |
| `BUFFER-INFO-COLLECT` | Buffer info collection | Buffer attributes/sizes |
| `LIVENESS` | Liveness analysis | Variable lifetime analysis |
| `MEMREUSE` / `MEMANLZ` | Memory reuse / analysis | Local memory optimization |
| `CHECK` | Full semantic checking | DMA legality, type consistency |
| `CODEGEN` / `CG_INFO` / `FUTANLY` | Code generation | Target-code generation behavior |

---

### Section 4: Compiler Options Reference

See `compiler-options.md` for the complete option reference. Key categories:

- Compilation mode: `-t`, `-gs`, `-es`, `-c`, `-o`, `-arch`, `-E`, `-s`
- AST/structure: `-e`, `-pa`, `-pb`, `-paa`, `-pnt`
- Type/symbol: `-i`, `-l`, `-ds`
- Value numbering: `-vn`, `--debug-visit=valno`
- Pass-level: `-dv`, `-tv`, `-d`, `-sp`, `-sa`, `-dp`
- Features: `-sr`, `-dvec`, `-dd`, `--dma-verbose`, `-u`, `--save-temps`

---

### Section 5: Running `.co` Files

### GPU end-to-end execution

The standard flow to compile and run a `.co` file on GPU:

```bash
./choreo -gs -t cute -arch=sm_90a path/to/file.co -o path/to/file.cute.result
bash path/to/file.cute.result --execute
```

To select an idle GPU before execution:

```bash
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
CUDA_VISIBLE_DEVICES=$FREE_GPU bash path/to/file.cute.result --execute
```

If `scripts/run_co_auto_gpu.sh` is present in the tree, it wraps these steps (generate script, pick GPU, run):

```bash
scripts/run_co_auto_gpu.sh path/to/file.co --arch sm_90a --disable-timing
```

On CUDA OOM or device failures, do not stop after the first failure: inspect `nvidia-smi`, pick a less busy GPU, and retry (see Section 6).

### Files with `RUN:` directives (`tests/` and parts of `benchmark/`)

These files embed directives consumed by `tests/lit.sh`.

**Typical `RUN:` patterns:**

```
// RUN: choreo -gs -t cute %s -o %s.cute.result && bash %s.cute.result --execute | FileCheck --match-full-lines %s
// RUN: choreo -es -t cute -arch=sm_86 %s -o - | FileCheck %s
// RUN: not choreo -gs %s 2>&1 | FileCheck %s
// RUN: choreo %s --debug-visit=latenorm -sa=latenorm 2>&1 | FileCheck %s
```

`%s` is replaced with the current file path. `not` is `tests/not.sh` (added to `PATH` by `lit.sh`) and inverts the expected exit code.

Run with `lit.sh`:

```bash
./tests/lit.sh tests/gpu/end2end/add.co
./tests/lit.sh tests/check/
./tests/lit.sh -j4 tests/
./tests/lit.sh -l tests/
```

Suite entrypoints:

```bash
make test
make test-debug
make test-release
```

For quick standalone GPU execution without FileCheck, use the manual flow from the top of this section.

### Files without `RUN:` directives

Standard manual flow:

```bash
./choreo -gs -t cute path/to/file.co -o build/output.cute.result
bash build/output.cute.result --execute

./choreo -es -t cute path/to/file.co -o -

./choreo -gs -t cute path/to/file.co -o build/output.cute.result
bash build/output.cute.result --compile-link
```

### Category quick reference

| Directory | File traits | Run method |
|---|---|---|
| `tests/check/` | Compiler semantic checks via FileCheck | `lit.sh` or manual `choreo ... \| FileCheck` |
| `tests/gpu/codegen/` | Codegen checks, often `-es` + FileCheck | `lit.sh` or manual |
| `tests/gpu/end2end/` | Full tests, `-gs` + `--execute` | `lit.sh`; requires GPU |
| `tests/parse/`, `tests/norm/`, etc. | Frontend checks | `lit.sh` |
| `benchmark/performance/` | Performance work; some files have `RUN:` | `lit.sh` or manual `-gs -t cute -arch=sm_XX` |
| `benchmark/shapeinfer/` | Shape-inference benchmarks | `lit.sh` or manual |
| `samples/cuda/` | CUDA-oriented samples when present | Manual `-gs -t cute` + `--execute` |

### Practical shortcuts

Build:

```bash
make build
```

Run a single test:

```bash
./tests/lit.sh tests/gpu/end2end/add.co
```

Manual GPU end-to-end:

```bash
./choreo -gs -t cute -arch=sm_90a path/to/file.co -o build/run.cute.result
bash build/run.cute.result --execute
```

Benchmarks:

```bash
./tests/lit.sh benchmark/performance/hgemm_rr/
```

Full suite:

```bash
make test
make JOBS=4 test
```

---

### Section 6: GPU Device Management

When running GPU `.co` files (`-t cute`, or `// REQUIRES: TARGET-GPU`):

- GPU 0 may be busy. On OOM or unavailable-device errors, try another idle GPU.

### Check GPU status

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

### Select a less-used GPU

Pick the least-used GPU and run:

```bash
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
CUDA_VISIBLE_DEVICES=$FREE_GPU bash build/output.cute.result --execute
```

### Common errors

1. **CUDA OOM**  
   Capture `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`. Retry on the least-used GPU or a specific idle index using `CUDA_VISIBLE_DEVICES=X`.

2. **CUDA device not available**  
   Verify the driver with `nvidia-smi`. Ensure `CUDA_HOME` is set (often `/usr/local/cuda`). Try a non-zero GPU index.

3. **nvcc failure (architecture mismatch)**  
   Read actual capability: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`. Match `-arch=sm_XX` to hardware (examples: RTX 3070 => `sm_86`, H100/H800 => `sm_90a`, A100 => `sm_80`).

---

### Section 7: Developing the Compiler

### Workflow

- Edit sources under `lib/` and `tools/`.
- Rebuild Release:

```bash
make
```

- Or Debug:

```bash
make debug
```

### Pre-commit checklist

- Run **`make format`** after editing C++ sources (clang-format per `.clang-format`).
- Follow **`Documents/Developer/coding-style.md`** for naming and structure.

### Validation depth

- **Quick sanity** on the compiler (example):

```bash
./choreo -gs -t cute -arch=sm_90a wip_code.co -o wip_code.cute.result
```

- For real `.co` workloads, follow Section 5 with GPU selection from Section 6.

- Optional full check:

```bash
make test
```

### Regression debugging for end-to-end failures

Build two variants when isolating regressions:

```bash
make debug
```

```bash
make release
```

Then emit generated CUDA for comparison (adjust options and paths as needed):

```
build-debug/choreo <options> <co_file> -es -o fail.cu
build-release/choreo <options> <co_file> -es -o pass.cu
```

Diffing `fail.cu` and `pass.cu` often localizes codegen regressions.

---

### Section 8: Debug Scenarios

See `debug-scenarios.md` for the full scenario guide, decision trees, and CUDA debugging workflow.

Quick symptom lookup:

| Symptom pattern | Scenario | First command |
|---|---|---|
| syntax error, unexpected token | 1 | `./choreo -E file.co` |
| type inconsistent | 2 | `./choreo -i file.co` |
| invalid span, chunkat | 3 | `./choreo -vn file.co` |
| parallel/block/thread | 4 | `./choreo -pa=norm -s file.co` |
| DMA/future/wait | 5 | `./choreo --debug-visit=latenorm -sa=latenorm file.co` |
| memory/buffer/reuse | 6 | `./choreo --debug-visit=liveness -s file.co` |
| nvcc/codegen errors | 7 | `./choreo -es -t cute file.co -o -` |
| unknown which pass | 8 | `./choreo -sp -s file.co` |
| vectorize/foreach | 9 | `./choreo -dvec -s file.co` |
| symbol replacement | 10 | `./choreo -sr -s file.co` |

---

### Section 9: Choreo `.co` Syntax Reference

Use this section whenever you create, edit, refactor, or review `.co` files, or when work touches DMA/TMA/MMA forms, storage qualifiers, tiling, test directives, or host/device structure.

### When to apply it

- Tasks mention `.co` sources under `tests/`, `benchmark/`, or future `samples/` trees.
- Questions cover Choreo syntax, `parallel`, `with`, `foreach`, or parser behavior.

### Core behavior

- Read nearby `.co` files first and match local dialect, target comments, and conventions.
- Use `choreo-syntax-reference.md` in this same directory as an evidence-based inventory; it is guidance to align with the corpus, not a license to invent syntax.
- Prefer an existing pattern over a novel spelling when both could work.
- Preserve hybrid structure when files already mix `#include`, macros, `main()`, `__co__`, `__cok__`, and `__co_device__` blocks.
- For build or run validation, use Sections 2 and 5 above, and compiler-options.md for compiler flags.

### Performance rules on GPU

- For new GEMM-like or blockscaled GEMM kernels, prefer **`mma.op`** as the tensor-core compute primitive. Lowering maps it to `wgmma`, `wmma`, `mma.sync`, `mma.sync.sp`, or `wgmma.sp` from shape and hardware; do not ask the user to pick a mnemonic for routine performance work.
- Use `benchmark/performance/matmul/` and `benchmark/performance/blockscale_gemm/` as primary references for tiling, staging, `tma.copy(.async)`, swizzle, `group-4`, events, and `mma.commit`.
- Prefer tile-level **`tma.copy` / `dma.copy`** with tile-shaped buffers or views. Treat `.at()` as the conservative access form; avoid making it the default hot-path choice when a copy or view pattern can express the access.
- Older files may use legacy spellings such as `mma.row.row` or `mma.row.row.scale`; use them as structural references, but prefer `mma.op` for new GEMM-like code unless a test must pin legacy syntax.

### Copy-pattern selection

- The same shaping patterns can wrap `dma.copy` or `tma.copy`; pick the **copy primitive** for the GPU generation (e.g. TMA on sm_90+) and the **shaping pattern** for semantics and inference.
- Prefer `chunkat(...)` for chunks driven by bounded indices or `#p`-style partitioning.
- Prefer `view(...).from(...)` when window shape, stride, or base offset must be explicit.
- Prefer `subspan(...).at(...)` when tile extents are fixed and only the anchor moves.
- Prefer `subspan(...).step(...).at(...)` when tiles repeat with explicit spacing or staged/swizzled layouts (persistent or warp-specialized kernels).
- These choices are not cosmetic: they can follow different inference paths and satisfy different constraints.
- If a copy is hard to type, try another shaping pattern before assuming declared shapes are wrong, unless the compiler reports clear size, contiguity, or out-of-bounds errors.

### File mapping (GPU-focused tree)

- `tests/parse/`, `tests/infer/`, `tests/check/`: parser, inference, and diagnostic examples.
- `tests/gpu/`, `benchmark/`: CUDA/CuTe kernels with `tma`, `mma`, swizzle, and role-qualified `parallel` forms.
- `tests/norm/` and related: transformation-heavy `span_as`, `subspan`, and multibuffer patterns.

### Editing workflow

1. Classify the file: parser test, infer/check test, end-to-end kernel, benchmark, or hybrid sample.
2. Find two or three neighbors in the same directory and mirror their spelling.
3. Reuse observed forms for types, qualifiers, index decomposition, and primitives.
4. If copy typing is difficult, try `chunkat`, `view(...).from(...)`, `subspan(...).at(...)`, or `subspan(...).step(...).at(...)` before changing global shapes.
5. If still unclear, search the wider `.co` corpus before guessing.
6. Keep edits minimal and stylistically local.

### Guardrails

- Preserve `RUN`, `REQUIRES`, `CHECK`, `CHECK-NOT`, `CHKINF`, `VALNO`, and `GDB` conventions in tests.
- Do not rewrite a file from one GPU kernel style into another solely because both are valid Choreo.
- For tests, prefer syntax already used nearby so diagnostics stay stable.
- For new performance GEMM-like GPU code, prefer `mma.op` even when neighbors use legacy explicit MMA forms.
- If the task is specifically about parser or codegen test coverage for a legacy MMA primitive or scalar access pattern, preserve the exact syntax already used by that test.
- Prefer tile-level `tma.copy` / `dma.copy` over scalar `.at()` on hot paths; use `.at()` as a fallback.
- If a construct does not appear nearby or in `choreo-syntax-reference.md`, say so and ask instead of inventing it.

### Ask instead of guessing

Ask (inline, concise) when one answer would materially change the approach:

- Generic frontend versus CUDA/CuTe GPU usage.
- Pure DSL versus hybrid host/device `.co`.
- Synchronous versus async copy/future pipelines.
- `dma` versus `tma` versus `mma` family choices when ambiguity remains after reading neighbors.
- `local`, `shared`, or `global` destination when unclear.
- Tiling or named-dim layout when examples conflict.
- Which copy-shaping pattern should own the access: `chunkat`, `view(...).from(...)`, `subspan(...).at(...)`, or `subspan(...).step(...).at(...)`.
- `mma.op` shape, accumulator type, transpose/sparsity, or scale operands when not fixed by context.
- Whether a hot path can stay tile-based with `tma.copy` / `dma.copy` or truly needs scalar `.at()` only at edges.
- Parser/infer/check expectations versus executable sample behavior.

When asking: finish unblocked work first; offer short choices with a recommended default; state what changes with each option; avoid open-ended stalls.

---

### Section 10: `lit.sh` / `lit.cfg` Conventions

### Framework overview

`tests/lit.sh` is the shared test driver. It manages generic variables and delegates GPU-specific behavior to `lit.cfg` files in each test directory.

Core globals managed by `lit.sh`:

| Variable | Purpose |
|---|---|
| `device_type` | Device kind (set by the GPU `hw_detect` hook, or `"none"`) |
| `mach` | Detected GPU architecture string |
| `simulator` | Simulator identifier (or `"none"`) |

GPU-specific arch variables, `%`-token expansions, skip conditions, and capability flags belong in the GPU `lit.cfg` and its registered hooks, not in `lit.sh` itself.

### Where responsibilities belong

| What | Where | Example |
|---|---|---|
| GPU arch variable init | GPU `lit.cfg` top | `gpu_arch="none"` |
| Hardware detection | `hw_detect` hook in GPU `lit.cfg` | sets arch, `device_type`, `mach` |
| `%` token expansion | `target_cmd` hook in GPU `lit.cfg` | expands `%gpu_arch` and `%target` |
| Skip rules | `target_noskip` hook in GPU `lit.cfg` | GPU capability checks |
| Cached HW summary | `lit.sh` core | only `device_type`, `mach`, `simulator` |

### Adding new GPU test tokens

1. Initialize the variable in the GPU `lit.cfg`.
2. Set it inside `hw_detect`.
3. Expand it in `target_cmd`.
4. Do **not** add it to `lit.sh` core or the HW cache.

---

### Section 11: Interaction Guidelines

Ask when uncertainty would waste work or pick the wrong workflow:

1. **Unknown GPU architecture** after reading `RUN:` lines and paths: ask which `-arch=sm_XX` to use.
2. **No `RUN:` and unclear intent** Ask: compile-only, codegen-only (`-es`), or full compile-and-run?
3. **GPU selection** Show brief `nvidia-smi` context and ask which device to bind when automation is inappropriate.
4. **Compile failure** Ask whether to run `make clean` and rebuild when the failure suggests a corrupted tree versus a code bug.
5. **Debug direction** Ask whether to prioritize AST dumps, type inference, value numbering, or generated CUDA when multiple paths remain plausible.

Combine with Section 9 "Ask instead of guessing" for syntax-level decisions.
