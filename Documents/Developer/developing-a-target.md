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
