---
name: compile-and-test
description: Build, test, run, and debug workflows for the Choreo compiler project. Use this first when asked to compile, run, benchmark, or debug any `.co` file, or when troubleshooting test/runtime failures.
---

# Choreo Project Build and Test Skill

> Consolidation note: this skill is the **single source of truth** for the former `run-code` behavior. The `run-code` skill now acts as a thin compatibility entry and delegates here.

You are a build, test, and debugging expert for the Choreo compiler project. Choreo is an orchestration-language compiler for heterogeneous computing, supporting GPU (CUDA/CuTe) backend.

---

## Response Language Alignment

- If the user prompt is in Chinese, respond in Chinese.
- If the user prompt is in English, respond in English.
- If the prompt is mixed, default to the language of the latest user message.

---

## Trigger Priority

- If the request mentions a `.co` file, this skill should be loaded immediately before other Choreo workflow skills.
- If the task is "modify compiler code and verify with a `.co` file", use `develop-compiler` for source changes and this skill for build/run validation.
- Do not skip this skill for benchmark or end-to-end `.co` execution just because the compiler source is also being edited.

---

## Project Build Modes

### Core Rules
- **All build operations must go through the top-level `Makefile` entrypoint**; do not run `cmake` or `ninja` directly.
- Build artifacts `./choreo` and `./copp` are **symlinks** to binaries in `build/`.
- The default target is `cute` (CUDA CuTe).
- **Never write generated/intermediate files to `/tmp`**. Always place temporary outputs under the current build workspace (for example: `build/`).
- **All execution commands must set an explicit timeout**. Use a conservative timeout by default (at least 30 seconds for single-run validation), and increase timeout for heavier compile/run tasks.

### Build Command Reference

| Command | Description |
|---|---|
| `make` or `make build` | Default Release build (cmake + ninja) |
| `make debug` | Debug build (`build-debug/`) |
| `make release` | Release build (`build-release/`) |
| `make clean` | Clean all build outputs |

After build, `./choreo` and `./copp` symlinks are created/updated automatically.

### Decide Whether Rebuild Is Needed
- If the `./choreo` symlink exists and points to a valid binary, you usually **do not need to rebuild**.
- If the user modified `.cpp` / `.hpp` files under `lib/`, run `make build` again.
- If there are compile errors or broken links, run `make clean && make build`.
- Before running, use `ls -la ./choreo` to quickly verify symlink validity.

---

## Compiler Pass Pipeline (Core Architecture)

Internally, Choreo has two major flows: Semantic analysis and CodeGen. Each pass has a unique uppercase name, which can be used in options such as `-dv`, `-tv`, `-pa`, `-sa`, and `-ds`.

### Full Pass Pipeline

```
Preprocess => Parse => SEMA (Early Semantic) => NORM (Normalization) => VALNO (Value Numbering) => INFER (Type Inference)
                                      ├ COMP_N
                                      ├ PBFILL
                                      └ HINT-CHECKER
=> LATENORM (Late Normalization) => BUFGEN => BUFFER-INFO-COLLECT
=> MEMORYREUSE => LIVENESS => MEMANLZ => MEMREUSE
=> CHECK (Semantic Check) => CODEGEN => CG_INFO => FUTANLY
```

### Pass Quick Reference

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

## Complete Choreo Compiler Option Reference

`./choreo` is the compiler main entrypoint.

### Compilation Mode Options
| Option | Description |
|---|---|
| `-t <target>` | Set compilation target: `cute` (GPU) |
| `-gs` / `--generate-script` | Generate bash compile/execute script (main end-to-end method) |
| `-es` | Generate target source only (do not compile target code; inspect codegen output) |
| `-c` / `--compile` | Compile but do not link |
| `-o <file>` | Set output file (`-o -` outputs to stdout) |
| `-arch=<arch>` | Set GPU architecture (e.g., `sm_86`, `sm_90a`) |
| `-n` / `--remove-comments` | Remove comments (useful with FileCheck) |
| `-E` | Preprocess only (inspect macro-expanded `.co` code) |
| `-s` / `--no-codegen` | No codegen (frontend analysis only; often combined with debug flags) |

### Debug Options by Purpose

#### A. AST / Program Structure Inspection
| Option | Description | Output Example |
|---|---|---|
| `-e` / `--dump-ast` | Dump AST tree right after parsing | Tree-form function/parameter/statement structure |
| `-pa=<PASS>` | Print AST **after** the specified pass | Shows AST transformed by that pass |
| `-pb=<PASS>` | Print AST **before** the specified pass | Compare before/after behavior |
| `-paa` / `--print-after-all` | Print AST after every pass | Trace full AST evolution (large output) |
| `-pba` / `--print-before-all` | Print AST before every pass | Same as above |
| `-pnt` / `--print-node-type` | Include node type annotations in AST dump | Shows `<{type}>` after nodes |

#### B. Type / Symbol Information
| Option | Description | Output Example |
|---|---|---|
| `-i` / `--infer-types` | Show type inference results (abort after INFER pass) | `Parameter: ::fn::lhs, Type: s32 mdspan<3> [2, 3, 64]` |
| `-l` / `--dump-symbol` | Dump full symbol table after LATENORM | `symbol: ::fn::var, type: f32 mdspan<2> [M, N]` |
| `-ds=<PASS>` | Dump symbol table after a specific pass | Same as `-l`, but at any pass point |

#### C. Value Numbering Debug
| Option | Description | Output Example |
|---|---|---|
| `-vn` / `--print-valno` | Trace full value-numbering process (abort after INFER) | `New VN #5: '#2,#3,#4'`; `Alias "::fn::lhs.span" -> #5` |
| `--debug-visit=valno` | Detailed per-node debug for VALNO pass | VN/MDS/UB state changes per AST node |

#### D. Pass-Level Debug
| Option | Description | Output Example |
|---|---|---|
| `-dv=<PASS>` / `--debug-visit=<PASS>` | Full node-level debug for one pass | State on every node enter/exit |
| `-tv=<PASS>` / `--trace-visit=<PASS>` | Trace only which nodes were visited by a pass | `AST::Program`, `AST::Parameter`, ... |
| `-d` / `--debug` | Enable debug for **all** passes (including parser bison debug) | Extremely verbose, starting from parser tokens |
| `-sp` / `--show-passes` | Show full pass pipeline | Prints pipeline pass list |
| `-sa=<PASS>` / `--stop-after=<PASS>` | Stop compilation after specified pass | No final output; runs until that pass only |
| `-dp=<PASS>` / `--disable-visit=<PASS>` | Disable one pass | Isolate issues introduced by a specific pass |

#### E. Feature-Specific Debug
| Option | Description |
|---|---|
| `-sr` / `--print-sym-replace` | Trace symbol replacement behavior |
| `-dvec` / `--debug-vectorize` | Debug loop vectorization |
| `-dd` / `--diag-dma` | Runtime DMA diagnostics (enabled in generated code) |
| `--dma-verbose` | Print detailed DMA runtime logs |
| `--liveness` | Analyze variable liveness |
| `--mem-reuse` | Analyze memory reuse |
| `-u` / `--visualize` | Visualize DMA data movement |
| `--save-temps` | Keep temporary compilation files |
| `-v` / `--verbose` | Show external programs invoked by compiler |
| `-vf` / `--verify` | Verify AST validity after every pass |

#### F. Other Hidden Options
| Option | Description |
|---|---|
| `-sfv` / `--simplify-fp-valno` | Simplify floating-point value numbering (experimental) |
| `-kt` / `--use-kernel-template` | Allow instantiation of C++ template functions |
| `-ht` / `--use-hetero-tileflow` | Implicit tileflow optimization in heterogeneous scenarios (experimental) |
| `-bn` / `--branch-norm` | Normalize if-else branches |
| `-ln` / `--loop-norm` | Normalize loops |
| `-nm` / `--no-vectorize` | Disable automatic vectorization |
| `-npp` / `--no-preprocess` | Skip preprocessor |
| `-adf` / `--analyze-device-functions` | Analyze device functions |
| `-fno-show-source-location` | Hide source locations |
| `-tos` / `--target-options` | Extra options passed to target compiler |
| `-fmax-local` | Max per-thread local memory size (bytes) |
| `-api=<mode>` | API mode: `cffi` or `sglang` |

---

## Debug Scenario Quick Guide (AI Agent Decision Reference)

Choose the best compiler-option combination based on the issue type.

### Scenario 1: Syntax / Parse Errors
**Symptoms**: `syntax error`, `unexpected token`, parse-phase failure
```bash
./choreo -E file.co
./choreo -e file.co
./choreo -d -sa=sema file.co 2>&1 | head -100
```

### Scenario 2: Type Inference Issues
**Symptoms**: `type inconsistent`, `unable to apply to the types`, type mismatch
```bash
./choreo -i file.co
./choreo --debug-visit=infer -s file.co
./choreo -ds=infer -s file.co
./choreo -pnt -pa=infer -s file.co
```

### Scenario 3: Shape / Span Inference Issues
**Symptoms**: `invalid span`, shape mismatch, `chunkat` out-of-bounds
```bash
./choreo -vn file.co
./choreo --debug-visit=valno -s file.co
./choreo -pa=sema -s file.co
```

### Scenario 4: Parallel Decomposition Issues
**Symptoms**: wrong parallel-by hierarchy, block/thread/group assignment anomalies
```bash
./choreo -pa=norm -s file.co
./choreo -pa=pbfill -s file.co
./choreo -paa -sa=pbfill file.co
```

### Scenario 5: DMA / Data-Movement Issues
**Symptoms**: DMA semantic error, future/wait mismatch, `non-async future can not be waited`
```bash
./choreo --debug-visit=latenorm -sa=latenorm file.co
./choreo -ds=latenorm -s file.co
./choreo --debug-visit=check -s file.co
./choreo -u file.co
```

### Scenario 6: Memory Reuse / Liveness Issues
**Symptoms**: abnormal memory allocation, buffer lifetime issues
```bash
./choreo --debug-visit=liveness -s file.co
./choreo --debug-visit=memreuse -s file.co
./choreo --debug-visit=memanlz -s file.co
```

### Scenario 7: CodeGen Issues
**Symptoms**: incorrect generated CUDA/C++ code, nvcc compilation errors
```bash
./choreo -es -t cute file.co -o -
./choreo -es -t cute -arch=sm_86 file.co -o -
./choreo --save-temps -gs -t cute file.co -o /tmp/out.cute.result
./choreo -pa=check -s file.co
```

### Scenario 8: Full Pipeline Tracing
**Symptoms**: pass introducing issue is unknown
```bash
./choreo -sp -s file.co
./choreo -paa -s file.co
./choreo -sa=sema file.co
./choreo -sa=valno file.co
./choreo -sa=infer file.co
./choreo -sa=latenorm file.co
./choreo -sa=check file.co
./choreo -dp=memreuse -s file.co
```

### Scenario 9: Vectorization Issues
**Symptoms**: abnormal foreach vectorization behavior
```bash
./choreo -dvec -s file.co
./choreo -nm -s file.co
```

### Scenario 10: Symbol Replacement Tracing
**Symptoms**: unexpected variable replacement, symbol changes after macro expansion
```bash
./choreo -sr -s file.co
```

---

## How to Run `.co` Files

### Preferred Script (GPU end-to-end / benchmark first choice)

For GPU end-to-end and benchmark execution, **prefer the helper script**:

```bash
scripts/run_co_auto_gpu.sh path/to/file.co --arch sm_90a --disable-timing
```

This script will:
- run `choreo -gs -t cute ...` to generate the execute script,
- select the least-used GPU via `nvidia-smi` (or use `--gpu`),
- execute with `CUDA_VISIBLE_DEVICES=<gpu>`.

When validating a `.co` file, prefer this helper unless you are debugging the generated script itself.
On CUDA OOM or device failures, do not stop after the first failure: inspect `nvidia-smi`, identify a less-busy GPU, and retry.

Use manual commands only when debugging script internals or when a custom flow is required.

### 1) Files with `RUN:` directives (tests/ and part of benchmark/, samples/)

These files embed execution directives parsed/executed by the `lit.sh` test driver.

**Typical `RUN:` patterns:**
```
// RUN: choreo -gs -t cute %s -o %s.cute.result && bash %s.cute.result --execute | FileCheck --match-full-lines %s
// RUN: choreo -es -t cute -arch=sm_86 %s -o - | FileCheck %s
// RUN: not choreo -gs %s 2>&1 | FileCheck %s
// RUN: choreo %s --debug-visit=latenorm -sa=latenorm 2>&1 | FileCheck %s
```

`%s` is replaced by `lit.sh` with the current file path. `not` is `extern/not.sh` and inverts exit code.

#### Run with `lit.sh`
```bash
./tests/lit.sh tests/gpu/end2end/add.co
./tests/lit.sh tests/check/
./tests/lit.sh -j4 tests/
./tests/lit.sh -l tests/
```

For quick standalone GPU execution (without FileCheck), prefer:
```bash
scripts/run_co_auto_gpu.sh tests/gpu/end2end/add.co --arch sm_90a --disable-timing
```

#### Run test suites with make
```bash
make test
make test-debug
make test-release
```

### 2) Files without `RUN:` directives (part of samples/ and benchmark/)

Preferred:
```bash
scripts/run_co_auto_gpu.sh path/to/file.co --arch sm_90a --disable-timing
```

Fallback manual:
```bash
./choreo -gs -t cute path/to/file.co -o /tmp/output.cute.result
bash /tmp/output.cute.result --execute

./choreo -es -t cute path/to/file.co -o -

./choreo -gs -t cute path/to/file.co -o /tmp/output.cute.result
bash /tmp/output.cute.result --compile-link
```

### 3) Category Quick Reference

| Directory | File Traits | Run Method |
|---|---|---|
| `tests/check/` | Compiler semantic checks, validated by FileCheck | `lit.sh` or manual `choreo ... \| FileCheck` |
| `tests/gpu/codegen/` | Codegen checks, source via `-es` + FileCheck | `lit.sh` or manual |
| `tests/gpu/end2end/` | Full end-to-end tests, `-gs` + `--execute` | `lit.sh`, requires GPU |
| `tests/parse/`, `tests/norm/`, etc. | Frontend checks | `lit.sh` |
| `benchmark/performance/` | Performance benchmarks, some have RUN: | `lit.sh` or manual `-gs -t cute -arch=sm_XX` |
| `benchmark/shapeinfer/` | Shape-inference benchmarks | `lit.sh` or manual |
| `samples/cuda/` | CUDA sample code | Manual `-gs -t cute` + `--execute` |

---

## GPU Device Management (Critical)

### Automatically Select an Idle GPU
When running GPU-related `.co` files (target = `cute`, or `REQUIRES: TARGET-GPU`), note:
- GPU 0 may be occupied by other users/processes.
- If OOM or unavailable-device issues occur, try another idle GPU.

### Check GPU Status
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

### Select the Least-Used GPU
```bash
scripts/run_co_auto_gpu.sh path/to/file.co --arch sm_90a --disable-timing
```

Manual fallback:
```bash
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
CUDA_VISIBLE_DEVICES=$FREE_GPU bash /tmp/output.cute.result --execute
```

### Common Error Handling

1. **CUDA OOM (out of memory)**
   - Immediately capture `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`.
   - Retry on the least-used GPU, or a user-requested GPU such as GPU 1 if it is clearly idle.
   - Prefer `scripts/run_co_auto_gpu.sh ... --gpu X` for the retry so the rerun path matches normal `.co` execution.

2. **CUDA device not available**
   - Ensure CUDA driver is installed: `nvidia-smi`.
   - Ensure `CUDA_HOME` is set (usually `/usr/local/cuda`).
   - Try a non-zero GPU index.

3. **nvcc compile failed (arch mismatch)**
   - Check real GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.
   - Ensure `-arch=sm_XX` matches hardware.
   - RTX 3070 => `sm_86`, H100/H800 => `sm_90a`, A100 => `sm_80`.

---

## AI Agent Debug Decision Tree

### When compiling and testing a `.co` file
```
1. Validate ./choreo
   ├─ Invalid => make build
   └─ Valid => continue

2. Analyze .co file
   ├─ Contains "// RUN:"?
   │   ├─ Yes => run with lit.sh (or execute RUN commands manually)
   │   └─ No  => determine target and build mode manually
   ├─ Contains "// REQUIRES: TARGET-GPU"?
   │   ├─ Yes => needs GPU
   │   └─ No  => likely frontend-only check
   └─ Content clues:
├─ has main() / CHECK: => end-to-end test (`-gs + --execute`)
        └─ only __co__ functions => codegen check or library code

3. Determine target
├─ path contains gpu/cuda/cute => cute
    ├─ RUN: contains -t XXX        => use XXX
    └─ uncertain                   => ask user inline

4. During GPU execution
   ├─ check GPU status with `nvidia-smi`
   ├─ select idle GPU
   ├─ on OOM/device error, inspect `nvidia-smi` again before concluding
   └─ prefer `scripts/run_co_auto_gpu.sh ...` (manual `CUDA_VISIBLE_DEVICES=X` as fallback)
```

### When debugging compiler issues
```
"syntax error" / "unexpected"            => Scenario 1
"type inconsistent" / "unable to apply" => Scenario 2
"invalid span" / "chunkat"               => Scenario 3
"parallel" / "block" / "thread"        => Scenario 4
"DMA" / "future" / "wait"               => Scenario 5
"memory" / "buffer" / "reuse"           => Scenario 6
nvcc/codegen errors                        => Scenario 7
unknown pass                               => Scenario 8
"vectorize" / foreach                     => Scenario 9

Default to `-s` (no codegen) in debugging commands unless codegen output is required.
For long output, use `| head -N` or `2>&1 | grep KEY`.
```

---

## Interaction Guidelines

For uncertain cases, ask via inline interaction (`ask_questions`) and keep conversation flow:

1. **Unknown target** => ask: "Please choose target: cute"
2. **No RUN and unclear operation** => ask: "Compile-only / codegen-only / compile-and-run?"
3. **GPU selection** => show GPU state and ask which GPU to use
4. **After compile failure** => ask whether to run `make clean` and rebuild
5. **Unclear debug direction** => ask whether to inspect AST / type inference / codegen / value numbering

---

## Practical Command Reference

### Build Project
```bash
make build
```

### Run a Single Test
```bash
./tests/lit.sh tests/gpu/end2end/add.co
```

### Manually Compile and Run GPU End-to-End Test
```bash
scripts/run_co_auto_gpu.sh path/to/file.co --arch sm_90a --disable-timing
```

### Common Fast Debug Commands
```bash
./choreo -e file.co
./choreo -i file.co
./choreo -vn file.co
./choreo -l -s file.co
./choreo -sp -s file.co
./choreo -pa=sema -s file.co
./choreo -pa=infer -pnt -s file.co
./choreo -paa -s file.co
./choreo --debug-visit=valno -s file.co
./choreo --debug-visit=infer -s file.co
./choreo -ds=infer -s file.co
./choreo --trace-visit=sema -s file.co
./choreo -es -t cute file.co -o -
./choreo -E file.co
```

### Debug the Regressions of End-to-End Failures
It is convenient to build two versions when regressions happen for end-to-end tests:

To build a 'fail' choreo that triggers regressions, which can be done with debug build:

```bash
make debug
```

To build a 'pass' choreo without regressions,  which can be done with release build (you can stash changes to get the 'pass' version normally):

```bash
make release
```

then generating the CUDA code for both versions:

```
build-debug/choreo <options> <co_file> -es -o fail.cu
build-release/choreo <options> <co_file> -es -o pass.cu
```

A comparison between the two versions helps to identify the root cause.

### Run Benchmarks
```bash
./tests/lit.sh benchmark/performance/hgemm_rr/
scripts/run_co_auto_gpu.sh benchmark/performance/hgemm_rr/hgemm_v1_warptiling.co --arch sm_90a --disable-timing
```

### Run Full Test Suite
```bash
make test
make JOBS=4 test
```
