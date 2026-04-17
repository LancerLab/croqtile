# Developing a New Target

This guide explains how to add a new compilation target (backend) to the Croqtile compiler. A target maps Croqtile's intermediate representation to a specific hardware platform's code.

---

## Architecture Overview

The Croqtile compiler is target-agnostic through its pass pipeline. Target-specific code lives in self-contained directories under `lib/Target/`:

```
lib/Target/
+-- GPU/              # CUDA/CuTe target
|   +-- cute_codegen.hpp
|   +-- cute_codegen.cpp
|   +-- gpu_adapt.hpp
|   +-- gpu_target.hpp
+-- <OtherTarget>/    # other target backends
|   +-- ...
+-- NewTarget/        # your new target
    +-- newtarget_codegen.hpp
    +-- newtarget_codegen.cpp
    +-- newtarget_target.hpp
```

Each target registers itself with the compiler's target registry, providing:
1. A **codegen visitor** that emits target-specific source code
2. Optional **target-specific passes** (transforms, checks, adaptations)
3. A **target descriptor** with metadata (name, supported features, arch list)

---

## Step 1: Create the Target Directory

```bash
mkdir lib/Target/NewTarget
```

Create a `target.mk` makefile fragment if the target has special build rules (extra libraries, custom compilation steps):

```makefile
# lib/Target/NewTarget/target.mk
NEWTARGET_SRCS := $(wildcard lib/Target/NewTarget/*.cpp)
```

The build system discovers `lib/Target/*/target.mk` files automatically.

---

## Step 2: Implement the Target Descriptor

Create a target header that inherits from the base `Target` class:

```cpp
// lib/Target/NewTarget/newtarget_target.hpp
#pragma once
#include "target.hpp"

namespace Choreo {

class NewTarget : public Target {
public:
  std::string name() const override { return "newtarget"; }

  bool supportsAsync() const override { return true; }
  bool supportsTMA() const override { return false; }
  bool supportsMMA() const override { return true; }

  void registerPasses(Pipeline& pipeline) override;
  void codegen(AST::Node* root, std::ostream& out) override;
};

} // namespace Choreo
```

### Target Registration

Register the target in `lib/target_registry.hpp` so the compiler can find it via `-t newtarget`:

```cpp
#include "Target/NewTarget/newtarget_target.hpp"

// In the registry initialization:
registry.add("newtarget", std::make_unique<NewTarget>());
```

---

## Step 3: Implement Code Generation

The codegen visitor traverses the AST after all passes complete and emits target source code. Implement it by inheriting from the base `Visitor`:

```cpp
// lib/Target/NewTarget/newtarget_codegen.hpp
#pragma once
#include "visitor.hpp"
#include <ostream>

namespace Choreo {

class NewTargetCodegen : public Visitor {
  std::ostream& out_;
public:
  explicit NewTargetCodegen(std::ostream& out) : out_(out) {}

  void Visit(AST::DMA& node) override;
  void Visit(AST::ParallelBy& node) override;
  void Visit(AST::MMA& node) override;
  // ... other node types
};

} // namespace Choreo
```

Key mappings to implement:

| Croqtile Construct | What to Emit |
|---------------------|--------------|
| `parallel p by N` | Kernel launch / thread block mapping |
| `dma.copy src => dst` | Memory transfer API calls |
| `dma.copy.async` | Async transfer with synchronization |
| `mma.row.col` | Matrix multiply-accumulate (D = A * B + C) |
| `tma.copy` | Bulk copy API (if supported) |
| `wait f` | Synchronization barriers |
| `inthreads (pred)` | Thread predication / masking |

### Existing Targets as Reference

Study the GPU target (`lib/Target/GPU/cute_codegen.cpp`) for a complete implementation. It maps:
- `parallel-by` -> CUDA kernel launches with CuTe thread layouts
- `dma.copy` -> CuTe `copy` operations with tiled layouts
- `mma.row.col` / `mma.row.row` -> CuTe MMA atoms
- `inthreads` -> `if (threadIdx.x < N)` guards

---

## Step 4: Add Target-Specific Passes (Optional)

If the target needs transformations before codegen (e.g., adapting generic DMA patterns to hardware-specific constraints), register them in `registerPasses()`:

```cpp
void NewTarget::registerPasses(Pipeline& pipeline) {
  pipeline.addAfter("CHECK", "NEWTARGET_ADAPT",
                    std::make_unique<NewTargetAdapt>());
}
```

Common target-specific passes:
- **Adaptation** -- rewriting generic constructs into target-specific forms
- **Verification** -- checking target-specific constraints (e.g., alignment, tile sizes)
- **Vectorization** -- mapping operations to vector instructions

---

## Step 5: Add Test Infrastructure

### Create test directory

```bash
mkdir tests/newtarget
```

### Create `lit.cfg` for the test runner

The `lit.cfg` file must start with `# co-lit` and register hooks for hardware detection and command substitution:

```bash
# co-lit
newtarget_arch="none"

newtarget_detect() {
  # Detect hardware; set device_type and mach if found
  local _dev="$(lspci | grep -i 'NewVendor' | head -1)"
  [[ -z "$_dev" ]] && return
  newtarget_arch="nt100"
  device_type="newtarget"
  mach="${newtarget_arch}"
}

newtarget_command() {
  declare -n cmd_ref="$1"
  cmd_ref="${cmd_ref//%newtarget_arch/-arch ${newtarget_arch}}"
  cmd_ref="${cmd_ref//%target/-t newtarget}"
}

register_hook "hw_detect"  "newtarget_detect"
register_hook "target_cmd" "newtarget_command"

# Optionally share tests/check/ under this target's hooks:
# include_dir "../check"
```

### Write tests

```co
// REQUIRES: TARGET-NT100
// RUN: choreo %target -es %s -o - | FileCheck %s

__co__ auto add(f32 [M] a, f32 [M] b) {
  f32 [M] c;
  parallel p by 4 {
    dma.copy a.chunkat(p) => shared;
    // ...
  }
  return c;
}

// CHECK: expected_output_pattern
```

See [Lit Test Runner](lit-test-runner.md) for the full reference on `RUN:`, `REQUIRES:`, `XFAIL:`, and hook registration.

---

## Step 6: Environment Setup (Optional)

If the target requires special driver or SDK setup, document it in `Documents/Developer/target/`:

```markdown
# NewTarget Environment Setup

## SDK Installation
...

## Driver Configuration
make setup-newtarget
```

Add corresponding Makefile targets for environment setup if needed.

---

## Step 7: Declare Library Call Support (Optional)

Choreo provides a generic `__lib_*` builtin mechanism that lets `.co` programs
call abstract library operations (GEMM, element-wise ops, reductions, etc.)
without knowing the target's API. The compiler lowers these to target-specific
library calls at codegen time.

### How It Works

1. **Scanner/Parser** -- The scanner matches any `__lib_<id>` token generically
   as `LIB_CALL`. No per-operation token is needed.
2. **Early Sema** -- Validates the call against the target's
   `IsLibCallSupported()` and `LibCallArgRange()` methods. If the target
   doesn't support a given name, early sema rejects it with a clear error.
3. **Target Check** -- Optional per-target checks (alignment, storage, etc.)
   in the target check pass.
4. **Codegen** -- The target's codegen dispatches on `func_name` and emits
   either the target-library call or a general fallback.

### Declaring Support

Override these methods in your `Target` subclass:

```cpp
class NewTarget : public Target {
public:
  // Declare which __lib_* names this target supports
  bool IsLibCallSupported(const std::string& name) const override {
    static const std::set<std::string> supported = {
        "__lib_gemm", "__lib_add", "__lib_relu",
    };
    return supported.count(name) > 0;
  }

  // Declare expected argument counts {min, max} for early sema
  std::pair<int, int>
  LibCallArgRange(const std::string& name) const override {
    if (name == "__lib_gemm") return {5, 6};  // out, A, B, [bias,] K, N
    if (name == "__lib_add") return {4, 4};   // dst, lhs, rhs, num
    if (name == "__lib_relu") return {3, 3};  // dst, src, num
    return {-1, -1};
  }

  // Whether target-library lowering is enabled by default
  // (user can override with --use-target-lib / --use-target-lib=false)
  bool DefaultUseTargetLib() const override { return true; }
};
```

### Handling in Codegen

In your codegen visitor's `Visit(AST::Call&)`, add dispatch for `__lib_*`
builtins:

```cpp
if (PrefixedWith(func_name, "__lib_")) {
  if (func_name == "__lib_gemm") {
    EmitLibGemm(n, os, indent);
  } else if (func_name == "__lib_add") {
    // When --use-target-lib is on: emit target library call
    if (CCtx().UseTargetLib()) {
      os << indent << "my_accel::add(dst, lhs, rhs, num);\n";
    } else {
      // General fallback
      os << indent << "for (int i = 0; i < num; ++i)\n";
      os << indent << "  dst[i] = lhs[i] + rhs[i];\n";
    }
  }
  return true;
}
```

### Standard Interface Conventions

| Pattern | Arguments | Examples |
|---------|-----------|---------|
| Unary   | `(dst, src, num)` | `__lib_relu`, `__lib_abs`, `__lib_sigmoid` |
| Binary  | `(dst, lhs, rhs, num)` | `__lib_add`, `__lib_mul`, `__lib_max` |
| GEMM    | `(out, A, B, K, N)` or `(out, A, B, bias, K, N)` | `__lib_gemm` |
| Reduce  | `(dst, src, num, rdim, nred)` | `__lib_reduce_sum` |

Following these conventions ensures portability across targets.

### Adding a New Library Operation

To add a new `__lib_*` operation:

1. **No scanner/parser changes needed** -- the `__lib_<id>` pattern is matched
   generically
2. Add the name to your target's `IsLibCallSupported()` and `LibCallArgRange()`
3. Add the codegen dispatch in your target's `Visit(AST::Call&)` handler
4. Add a test in `tests/<target>/codegen/` that verifies both the library path
   and the fallback path

---

## Checklist

- [ ] `lib/Target/NewTarget/` directory with codegen and target descriptor
- [ ] Target registered in `target_registry.hpp`
- [ ] Codegen visitor handles all relevant AST node types
- [ ] `tests/newtarget/lit.cfg` with `# co-lit` marker and hooks
- [ ] Basic test cases covering DMA, parallel, and (optionally) MMA
- [ ] `target.mk` if build requires extra rules
- [ ] Run `make test` to verify no regressions

---

## See Also

- [Compilation Passes](compilation-passes.md) -- pass pipeline and target pass registration
- [Lit Test Runner](lit-test-runner.md) -- test infrastructure and hook system
- [Build and Test](build-and-test.md) -- building the compiler
- [Target-Specific Calls](target/calls.md) -- target-specific calling conventions
- [Target Environment Setup](target/env_setting_up.md) -- hardware driver configuration
