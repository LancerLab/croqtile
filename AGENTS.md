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

### After Build
- `./choreo` - Main compiler symlink (points to `build/choreo`)
- `./copp` - Preprocessor symlink (points to `build/copp`)

### Direct CMake (if needed)
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

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
