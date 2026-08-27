# AGENTS.md - Choreo Project Developer Guide

This file provides essential information for agentic coding agents operating in the Choreo repository.

## Project Overview

Choreo is a low-level Embedded Domain Specific Language (EDSL) for C++ that programs data movement entities (DMAs) in heterogeneous hardware like GPUs.

---

## Build Commands

### Primary Build Targets

| Command | Description |
|---------|-------------|
| `make` or `make build` | Default Release build with CMake + Ninja |
| `make debug` | Debug build (outputs to `build-debug/`) |
| `make release` | Release build (outputs to `build-release/`) |
| `make clean` | Clean all build outputs |

### BAD BUILD COMMANDS: AVOID to use cmake directly if make commands can do the thing

### CMake Project and Target Selection

Use `CROQ_PROJECT` to select which projects to build (semicolon-separated,
like LLVM's `LLVM_ENABLE_PROJECTS`):

```bash
cmake -DCROQ_PROJECT="choreo;coir" ...   # default
cmake -DCROQ_PROJECT="coir" ...          # coir only
cmake -DCROQ_PROJECT="choreo;co-mock" ...
```

Available projects: `choreo`, `coir`, `co-mock`, `co-web`.

Use `CROQ_TARGET` to select which target backends to build:

```bash
cmake -DCROQ_TARGET="all" ...            # default: all backends
cmake -DCROQ_TARGET="gpu;cpu" ...        # GPU + CPU only
cmake -DCROQ_TARGET="cpu" ...            # CPU only
```

Available targets: `all`, plus any directory under `lib/Target/` (e.g.
`gpu`, `cpu`, `amdgpu`, `hetero`). Each backend's `target.aliases` file
defines extra accepted names (e.g. `cute` and `nvptx` map to the GPU
backend, `hip` maps to AMDGPU, `cc` maps to CPU). Unknown names are
silently ignored.

Toolchain dependencies (CUDA/CUTLASS, HIP/ROCm) are only probed when the
corresponding target is enabled (GPU needs CUDA/CUTLASS, AMDGPU needs ROCm).

All sub-projects share the Choreo core libraries (parser, codegen, targets).
Software dependencies (flex/bison, LLVM/MLIR, CUTLASS) are auto-downloaded
by CMake when not present in `extern/`. Dependency URLs and versions are
configured in `cmake/deps.conf` (branch-specific, see `oss_conflict_paths.txt`).

### CoIR (MLIR-based IR tooling)

| Command | Description |
|---------|-------------|
| `make coir` | Build CoIR tools (`coir-opt`, `co2ir`, `cocc`); auto-downloads LLVM/MLIR if missing |
| `make coir-test` | Run CoIR lit tests |
| `make coir-clean` | Remove CoIR build artifacts |

### After Build
- `./choreo` - Main compiler symlink (points to `build/choreo`)
- `./copp` - Preprocessor symlink (points to `build/copp`)

---

## Test Commands

### Run Full Test Suite

```bash
make test              # Run all tests (default build)
make test-debug       # Run tests with debug build
make test-release     # Run tests with release build
make JOBS=4 test      # Run tests with parallel jobs
```

### Run a Single Test

```bash
# Using lit.sh (preferred for tests with RUN: directives)
./tests/lit.sh tests/gpu/end2end/add.co
./tests/lit.sh tests/check/

# For GPU end-to-end tests
scripts/run_co_auto_gpu.sh path/to/file.co --arch sm_90a --disable-timing
```

### Standalone Tests

```bash
cd tests/standalone/ && make test
```

---

## Code Style Guidelines

### Formatting

The project uses **ClangFormat** with an LLVM-based configuration (see `.clang-format`):

```yaml
BasedOnStyle: LLVM
IndentWidth: 2
ColumnLimit: 80
PointerAlignment: Left
AlwaysBreakTemplateDeclarations: Yes
AllowShortFunctionsOnASingleLine: true
AllowShortLoopsOnASingleLine: true
AllowShortIfStatementsOnASingleLine: true
```

**Format the code before committing:**
```bash
make format
```

### Text Content Style (MANDATORY)

- Use ASCII-only characters in scripts, skills, and docs unless there is a
   strong reason not to.
- Do not use decorative separator-only lines (for example, lines made only of
   box-drawing characters). Use plain ASCII separators if needed.

### Commit Message Conventions (MANDATORY)

- Write messages that describe **what kind of change** was made, not the
  specific implementation details (file names, variable names, function
  names — these change over time). Future readers tracing history need to
  understand the intent, not the mechanics.
- Start with a conventional prefix: `fix:`, `feat:`, `refactor:`, `cleanup:`,
  `docs:`, `test:`, `chore:`, `build:`.
- Keep the subject line under 72 characters.
- Use bullet points for body details describing categories of changes, not
  specific files or symbols.
- Example — GOOD: "fix: make check tests target-agnostic and relax arch
  validation" | BAD: "fix: change hetero_target.cpp IsArchSupported to
  return true".

### C++ Standard

- **C++17** is required
- Compiler: GCC 9.0+ or Clang 5.0+

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes/Types | PascalCase | `class MultiNodes` |
| Functions | camelCase | `void accept(Visitor& v)` |
| Variables | snake_case | `auto& sub`, `init_expr` |
| Constants | snake_case | `max_buffer_size` |
| Namespaces | PascalCase | `Choreo::AST::` |
| Member variables | snake_case with optional `_` suffix | `values_` or `values` |

### File Organization

| Directory | Purpose |
|----------|---------|
| `lib/` | Compiler source code |
| `runtime/` | Runtime headers |
| `tools/` | Tool binaries (choreo, copp) |
| `tests/` | Test files |
| `samples/` | Sample `.co` files |
| `benchmark/` | Performance benchmarks |

### Import/Include Order

1. Project headers (local)
2. External libraries
3. Standard library

```cpp
#include "visitor.hpp"           // Local project header
#include <vector>                // Standard library
#include "extern/cutlass/..."    // External library
```

### Error Handling

- Use `assert()` for internal invariants and programming errors
- Use descriptive error messages with context
- The compiler uses pass-based architecture; errors should indicate the failing pass

### Code Patterns

**Namespaces:**
```cpp
namespace Choreo {
namespace AST {
class MultiNodes { ... };
} // namespace AST
} // namespace Choreo
```

**Visitor Pattern:**
```cpp
void MultiNodes::accept(Choreo::Visitor& v) {
  v.BeforeVisit(*this);
  for (auto& sub : values) sub->accept(v);
  v.Visit(*this);
  v.AfterVisit(*this);
}
```

**Getters/Setters:**
```cpp
class Expr {
public:
  auto GetR() const { return value_r; }
};
```

---

## Compiler Options Reference

### Common Debug Options

| Option | Description |
|--------|-------------|
| `-e` / `--dump-ast` | Dump AST tree after parsing |
| `-i` / `--infer-types` | Show type inference results |
| `-vn` / `--print-valno` | Trace value-numbering process |
| `-pa=<PASS>` | Print AST after specified pass |
| `-sa=<PASS>` / `--stop-after=<PASS>` | Stop after specified pass |
| `-es` | Generate target source only (no compile) |
| `-t <target>` | Set target: `cute` |

### Pass Names

- `SEMA` - Early semantic analysis
- `NORM` / `COMP_N` - AST normalization
- `VALNO` - Value Numbering
- `INFER` - Type inference
- `LATENORM` - Late normalization
- `CHECK` - Full semantic checking
- `CODEGEN` - Code generation

---

## Git Push and OSS Sync Rules (MANDATORY)

### Never push without explicit instruction (one-time permission, never assumed)

**Do NOT run `git push` to any remote unless the user explicitly asks for
that specific operation.** `git commit` (local) is fine when the user asks
to commit. `git push` requires the user to say "push", "push to origin",
"push to remote", etc.

**Permission is ALWAYS one-time.** Even if the user has asked you to push
earlier in the same conversation, you must get a fresh, explicit push
instruction every time. Never assume prior permission carries over. Stop
after committing and ask the user whether they want to push.

### Every main commit needs an oss/main counterpart

This repo maintains an `oss/main` branch for open-source sync. When you commit
on `main`, you MUST also run `bash scripts/oss/oss-push.sh <sha>` to create a
corresponding oss/main commit. If all files in the commit are excluded (internal-
only), the script skips automatically -- but you must still run it.

See the `/oss-merge` skill for the full workflow.

### OSS Violation Awareness (MANDATORY when developing on main)

All code committed to `main` that is NOT excluded in `scripts/oss/oss_exclude_paths.txt`
will be synced to the public `oss/main` branch. Before committing, you MUST ensure:

1. **No forbidden keywords**: Run `make oss-scan` (or `bash scripts/oss/oss-scan.sh --staged`)
   to verify no internal company names, product codenames, or internal URLs leak into
   non-excluded files. The keyword list is in `scripts/oss/os_kw.txt`.

2. **No non-ASCII in synced files**: The scan also flags non-ASCII characters in files
   that will reach oss/main.

3. **Exclude-aware development**: If you are adding a new file or directory that should
   remain internal, add it to `scripts/oss/oss_exclude_paths.txt` FIRST.

4. **Build integrity**: All `configure_file()` inputs referenced in `CMakeLists.txt`
   must exist on oss/main. The `oss-push.sh` script verifies this automatically.

5. **Quick check**: `make oss-scan` runs the full tree scan. Use `make oss-scan-staged`
   to scan only staged changes before committing.

6. **OSS script regression gate** (MANDATORY when editing `scripts/oss/*.sh`):
   - For any change to OSS sync scripts, run regression tests to prevent breakage:
   
   ```bash
   # Quick smoke test (baseline/catchup logic):
   bash scripts/oss/tests/test-oss-pull-baseline.sh --quick
   
   # For option parsing or filtering changes:
   bash scripts/oss/tests/test-oss-scripts.sh --quick
   
   # Full validation before merge (if modifying core logic):
   bash scripts/oss/tests/test-oss-pull-baseline.sh
   bash scripts/oss/tests/test-oss-scripts.sh
   ```
   
   Tests run in isolated sandboxes and clean up automatically (~47s quick, ~120s full).
   See `scripts/oss/tests/README.md` and the `/oss-merge` skill for detailed guidance.

---

## OSS Test Suites

Three comprehensive test suites validate OSS script functionality. They live in
`scripts/oss/tests/` (see `scripts/oss/tests/README.md` for design and usage).

| Test File | Purpose | Coverage | Runtime |
|-----------|---------|----------|---------|
| `scripts/oss/tests/test-oss-pull-baseline.sh` | Core pull/catchup logic | Baseline, catchup, filtering, commit detection | ~56s (full), ~47s (quick) |
| `scripts/oss/tests/test-oss-scripts.sh` | Individual script options | push, pull, sync options; roundtrip workflows | ~4min (full), ~2min (quick) |
| `scripts/oss/tests/test-oss-workflow.sh` | End-to-end integration | push/pull/scan roundtrip; diverged sync; author preservation | ~1min |

**When to run:**
- Before ANY commit touching `scripts/oss/*.sh`
- When troubleshooting sync behavior
- When validating option changes or fixes
- Required gate for merge requests on OSS sync scripts

---

## Project-Specific Conventions

### .co Files (Choreo Source)

- Custom DSL files with `.co` extension
- Can include `// RUN:` directives for test execution
- Use `// REQUIRES: TARGET-GPU` for GPU-dependent tests

### Generated Files

- Parser: `lib/parser.yy` (Bison) -> `build/parser.tab.cc`
- Scanner: `lib/scanner.l` (Flex) -> `build/scanner.yy.cc`

### Build Outputs

- Never write generated files to `/tmp`; use `build/` directory
- Build artifacts are symlinked to repo root

### GCU SIMT Parallel-Level Mapping (gcu400+)

The gcu400 target uses a SIMT model whose naming is the reverse of the CUDA
mental model, and this frequently confuses agents. The canonical mapping:

| Choreo level | Hardware unit          | Index builtin     | Builtin var  |
|--------------|------------------------|-------------------|--------------|
| `BLOCK`      | cluster / grid         | `__tops_bid_*()`  | `blockIdx`   |
| `GROUP`      | thread / SIP           | `__tops_tid_*()`  | `threadIdx`  |
| `THREAD`     | subthread / SIMT lane  | `__tops_stid_*()` | `subThreadIdx`|

Key facts:

- Choreo `GROUP` = the hardware **thread (SIP)** = `threadIdx`. A GROUP runs
  `DEFAULT_SUBTHREAD_NUM == 4` subthreads in SIMT lockstep.
- Choreo `THREAD` = the hardware **subthread (lane)** = `subThreadIdx`
  (`__tops_stid_*()`). It is NOT `threadIdx`.
- The DTE (data-transfer engine) context is **thread-shared per SIP** for
  `LOCAL`/`SHARED` (`tops::local_dte` / `tops::shared_dte`); only
  `private_dte` is per-subthread. See `DMATypeSTR` in
  `lib/Target/GCU/topscc_codegen.cpp`.
- Guard semantics (see `runtime/choreo.h`): `__CHOREO_BLOCK_SINGLE__` selects
  a single **GROUP** (`threadIdx == 0`; its subthreads run in lockstep);
  `__CHOREO_GROUP_SINGLE__` selects a single **THREAD** (`subThreadIdx == 0`).
  A DMA issued at `THREAD` level (per-lane slice/deslice) is **unguarded** and
  uses `tops::private_dte` (per-subthread) instead of the thread-shared
  `local_dte`.

### Target Development: Supporting Explicit Type Conversions

When implementing a new `Target` subclass, override `SupportedScalarTypes(arch)` to declare which scalar types your target supports for explicit `__to<type>(expr)` conversions. Return the set of `BaseType` values valid for the given architecture. The compiler validates both source and target types in early semantic analysis. Optionally override `IsCastSupported(arch, from, to)` to restrict specific conversion pairs; the default allows all conversions between supported types. See `lib/target.hpp` for the interface and existing targets for reference.

### Foreign Type Casts (`__to<"type">`)

For casting to types not recognized by Choreo (compiler extensions, library types), use
`__to<"type_string">(expr)`. The quoted type string is emitted verbatim as a C-style cast
in generated code. Choreo skips type validation and assigns `AddrType` to the result,
which passes the `call` argument whitelist. This is useful when C++ template argument
deduction requires a concrete pointer type instead of `nullptr_t`.

---

## Skills

The project includes specialized skills for Claude Code:

| Skill | Purpose |
|-------|---------|
| `/skill-creator` | Create or refine project-specific skills with Choreo conventions |
| `/choreo-syntax` | Reference `.co` syntax, primitives, and editing patterns before changing `.co` files |
| `/build-and-install` | Build the Choreo compiler and tools |
| `/compile-and-test` | Build, test, run, and debug .co files |
| `/develop-compiler` | Modify and rebuild the Choreo compiler |
| `/develop-feature` | End-to-end feature development workflow |
| `/profiling` | Profile kernels and runtime behavior |
| `/debugging` | Debug generated executables |
| `/performance-bottleneck-analysis` | Analyze bottlenecks from profiling reports |
| `/oss-merge` | Unified OSS sync workflow: push/pull, compliance scanning, testing, troubleshooting |
| `/oss-scan` | (LEGACY) OSS compliance scanning reference (merged into oss-merge) |

---

## Opening Issues (GitLab / GitHub)

API tokens are available as environment variables (never commit or print them):

- `GITLAB_TOKEN` -- internal GitLab personal access token (`glpat-...`)
- `GITHUB_TOKEN` -- GitHub personal access token (`ghp_...`)

### Internal GitLab

The internal GitLab is `git.enflame.cn` (resolves to `172.16.11.21`). It
exposes SSH (port 22) and the REST API over HTTP (port 80). HTTPS (443) is
refused, so always use the `http://` endpoint.

- Project path: `xiaofeng.guan/choreo`
- Project id: `10205`

```bash
# Resolve the project
curl -sS --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  "http://git.enflame.cn/api/v4/projects/xiaofeng.guan%2Fchoreo"

# Open an issue (for multiline bodies, write to a file and pass description@<file>)
curl -sS --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
  --data-urlencode "title=Summary of the issue" \
  --data-urlencode "description=Long form description..." \
  "http://git.enflame.cn/api/v4/projects/10205/issues"
```

Internal issues may use internal terminology (gcu, topscc, DTE, etc.) freely.

### GitHub (public)

The public repo is `LancerLab/croqtile`.

```bash
curl -sS -X POST -H "Authorization: token $GITHUB_TOKEN" \
  -d '{"title":"...","body":"..."}' \
  "https://api.github.com/repos/LancerLab/croqtile/issues"
```

GitHub issues are public and MUST be OSS-compliant:

- No forbidden keywords (see `scripts/oss/os_kw.txt`): gcu, topscc, DTE,
  enflame, etc.
- ASCII-only, no internal URLs or IPs.
- When in doubt, open the issue on internal GitLab instead.

---

## Additional Resources

- [Language Reference](./Documents/Documentation/index.md)
- [Developer Guide](./Documents/Developer/index.md)
