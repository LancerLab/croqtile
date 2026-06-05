# CoIR Coding Style Guide

This document defines coding conventions for the CoIR tooling under `tools/coir/`.
CoIR straddles two ecosystems -- Choreo (the host compiler) and MLIR/LLVM (the IR
framework) -- so the style blends both where appropriate.

## Language Standard

- **C++17** is required. Do not use C++20/23 features.
- Prefer Choreo's utility templates (`dyn_cast<>`, `cast<>`, `isa<>`) over
  `std::dynamic_pointer_cast` or raw RTTI for shared-pointer downcasts.

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes / Types | PascalCase | `ASTCoIRGen`, `CUDAEmitter` |
| Member Functions | PascalCase | `LowerBaseType()`, `EmitExpr()` |
| Free / Static Functions | PascalCase | `EvalToInt()` |
| Variables | snake_case | `expr_stack`, `value_stack` |
| Constants | snake_case or UPPER_SNAKE | `max_dims` |
| Namespaces | PascalCase | `CoIR::` |
| Enum values | UPPER_SNAKE (matching Choreo) | `ParallelLevel::BLOCK` |
| MLIR Op classes | PascalCase (TableGen convention) | `KernelOp`, `ParallelOp` |
| File names | PascalCase for library sources | `ASTCoIRGen.cpp`, `EmitCUDA.cpp` |
| Executables | kebab-case | `coir-gen`, `coir-opt` |

### Rationale

CoIR is part of Choreo's codegen pipeline, so member/free functions follow
Choreo's PascalCase convention rather than MLIR's camelCase. TableGen-generated
code retains MLIR's style since it is auto-generated.

## File Organization

```
tools/coir/
  include/Dialect/CoIR/   -- TableGen + generated headers
  lib/ASTIRGen/           -- AST-to-CoIR lowering (uses Choreo AST)
  lib/Opt/                -- CoIR MLIR passes (optimization, lowering)
  lib/CodeGen/GPU/        -- GPU target emission (CUDA/CUTE)
  lib/CodeGen/<Target>/   -- Additional target backends (internal)
  tests/
    irgen/                -- .co -> CoIR MLIR generation tests
    opt/                  -- Dialect roundtrip / parse verification tests
    transform/            -- MLIR pass transformation tests (lowering)
    gpu/emit/             -- CoIR MLIR -> CUDA source string tests
    gpu/e2e/              -- Full GPU pipeline (.co -> compile -> execute)
    <target>/emit/        -- CoIR MLIR -> target source tests (internal)
    <target>/e2e/         -- Full target pipeline (internal, requires device)
  docs/                   -- Design documents and gap analysis
```

## Include Order

1. Local project headers (`"ASTCoIRGen.hpp"`, `"MLIRUtility.hpp"`)
2. CoIR dialect headers (`"Dialect/CoIR/CoIROps.h"`)
3. MLIR headers (`"mlir/IR/Builders.h"`)
4. LLVM headers (`"llvm/Support/raw_ostream.h"`)
5. Standard library headers (`<vector>`)

## Type Casting

Use Choreo's cast utilities for shared_ptr types:

```cpp
// Correct: Choreo's dyn_cast returns ptr<T> (shared_ptr<T>)
if (auto sty = dyn_cast<SpannedType>(symType))
  argTypes.push_back(TranslateSpannedType(sty));

// Avoid: std::dynamic_pointer_cast
// argTypes.push_back(TranslateSpannedType(
//     std::dynamic_pointer_cast<SpannedType>(inTy)));
```

For MLIR types, use `mlir::dyn_cast<>` (newer API) or the member `.dyn_cast<>()`
(deprecated but still supported in MLIR 18).

## Enum Conventions

CoIR enums mirror Choreo's enums for consistency:

```cpp
// CoIR ParallelLevel aligns with Choreo::ParallelLevel
enum class ParallelLevel { THREAD, GROUP, GROUPx4, BLOCK, CLUSTER, DEVICE, SEQ };
```

## Formatting

Follow `.clang-format` at the repo root (LLVM-based, 80 column limit, 2-space
indent). Run `make format` before committing.

## Test Conventions

- All test files include `// RUN:` directives compatible with lit.sh
- IRGen tests use `.co` extension and verify `coir-gen` output
- Opt tests use `.mlir` extension for dialect roundtrip verification
- Transform tests use `.mlir` extension for lowering pass tests
- Emit tests use `.mlir` extension and verify generated target source
- E2E tests use `.co` extension with `coir-codegen -gs` (script-based)
  - GPU e2e: `coir-gen | coir-codegen -gs` -> self-contained bash script
  - Aux target e2e: `coir-gen | coir-codegen --target=<name> -gs` -> self-contained script
  - Scripts auto-detect GPU arch and include test harness
  - Scripts can be scp'd to remote machines for execution
- Use `FileCheck` for output verification
- Tests should be self-contained (no external file dependencies)

## Emission Modes

| Mode | Command | Output |
|------|---------|--------|
| Source only (`-es`, default) | `coir-codegen` | CUDA/CUTE C++ source |
| Script (`-gs`) | `coir-codegen -gs` | Bash script with embedded source |

The `-gs` mode generates a script with `--execute` flag for running:
```bash
coir-gen kernel.co | coir-codegen -gs - > test.sh
bash test.sh --execute   # compile + run on local GPU
scp test.sh remote: && ssh remote bash test.sh --execute  # remote execution
```
