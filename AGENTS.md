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

### Never push without explicit instruction

**Do NOT run `git push` to any remote unless the user explicitly asks.**
`git commit` (local) is fine when the user asks to commit. `git push` requires
the user to say "push", "push to origin", "push to remote", etc.

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

---

## Project-Specific Conventions

### .co Files (Choreo Source)

- Custom DSL files with `.co` extension
- Can include `// RUN:` directives for test execution
- Use `// REQUIRES: TARGET-GPU` for GPU-dependent tests

### Generated Files

- Parser: `lib/parser.yy` (Bison) → `build/parser.tab.cc`
- Scanner: `lib/scanner.l` (Flex) → `build/scanner.yy.cc`

### Build Outputs

- Never write generated files to `/tmp`; use `build/` directory
- Build artifacts are symlinked to repo root

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

---

## Additional Resources

- [Getting Started Guide](./Documents/Documentation/getting-started-with-choreo.md)
- [Choreo Language Documentation](./Documents/Documentation/)
