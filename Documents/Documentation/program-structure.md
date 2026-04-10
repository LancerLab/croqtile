# Program Structure

## The Three-Part Model

A Croqtile-C++ program is composed of three parts:

| Part | Marker | Runs On | Written In |
|------|--------|---------|------------|
| **Host program** | (none) | CPU | Standard C++ |
| **Device program** | `__device__` or `__cok__{}` | Accelerator | Target C++ (CUDA, etc.) |
| **Tileflow program** | `__co__` | Transpiled | Croqtile DSL |

The host program is the entry point. It prepares data, calls tileflow functions, and consumes results. The device program defines compute kernels that run on the accelerator. The tileflow program -- the core of Croqtile -- orchestrates data movement between storage levels and dispatches device kernels.

```choreo
__device__ void kernel(float* a, float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

__co__ s32 [6, 17, 128] ele_add(s32 [6, 17, 128] lhs, s32 [6, 17, 128] rhs) {
  s32 [lhs.span] output;
  parallel p by 6 {
    with index in [17, 4] {
      foreach index {
        lhs_load = dma.copy lhs.chunkat(p, index) => shared;
        rhs_load = dma.copy rhs.chunkat(p, index) => shared;
        shared s32 [lhs_load.span] l1_out;
        call kernel(lhs_load.data, rhs_load.data, l1_out, |lhs_load.span|);
        dma.copy l1_out => output.chunkat(p, index);
      }
    }
  }
  return output;
}

int main() {
  choreo::s32 a[6][17][128] = {0};
  choreo::s32 b[6][17][128] = {0};
  std::fill_n(&a[0][0][0], sizeof(a)/sizeof(a[0][0][0]), 1);
  std::fill_n(&b[0][0][0], sizeof(b)/sizeof(b[0][0][0]), 2);
  auto res = ele_add(choreo::make_spanview<3>(&a[0][0][0], {6, 17, 128}),
                     choreo::make_spanview<3>(&b[0][0][0], {6, 17, 128}));
  // ... verify res ...
}
```

## Function Markers

### `__co__` -- Tileflow Functions

Any function prefixed with `__co__` is a **Croqtile function**. The compiler transpiles its body into target host and device code. Croqtile functions cannot call other Croqtile functions.

```choreo
__co__ auto foo(f32 [M, K] input) { ... }
```

### `__cok__` -- Separated Device Code Block

For targets that use the *separate source compilation model* (e.g., OpenCL-style), device code must be wrapped in a `__cok__` block. The compiler extracts this block into a separate compilation unit.

```choreo
__cok__ {
  extern "C" void device_fn(float* data, int n) { ... }
}

__co__ void bar(f32 [128] input) {
  parallel p by 4 { call device_fn(input, 128); }
}
```

For *single source* targets like CUDA/CuTe, device functions use the target's own annotations (e.g., `__device__`) and appear outside `__cok__` blocks.

### `__co_device__` -- Croqtile-Style Device Functions

Croqtile also supports writing device-side compute functions in a Croqtile-like syntax with `__co_device__`. These functions run on the device but are written using Croqtile scalar and loop constructs rather than raw C++.

```choreo
__co_device__ void layernorm(f32 [N] input, f32 [N] output) {
  // scalar loops over elements
}
```

## Compilation Pipeline

The Croqtile compiler works in three stages:

```
Source (.co) --> Preprocessing --> Transpilation --> Target Compilation
```

1. **Preprocessing** handles `#define`, `#if`/`#ifdef`, and file inclusion via the `copp` preprocessor.
2. **Transpilation** converts the `__co__` function body into target source code (e.g., CUDA/CuTe C++). Host and device code outside `__co__` blocks passes through unchanged.
3. **Target Compilation** invokes the platform compiler (e.g., `nvcc`) to produce the final binary.

The `-es` flag stops after transpilation (emit source). The `-gs` flag generates a work-script that drives target compilation and execution.

## Compilation Models

| Model | Description | Example Target |
|-------|-------------|----------------|
| Single source | Host, device, and tileflow code in one file | CUDA/CuTe |
| Separate source | Device code extracted to a separate file | OpenCL, Factor |

The `__cok__` block enables the separate source model. The Croqtile compiler handles the split automatically.

## Data Flow: Host to Tileflow to Device

The typical call chain is:

```
main() --> __co__ function --> call device_kernel()
   host        tileflow             device
```

The tileflow function receives shaped data from the host via `choreo::make_spanview` (or `choreo::make_spanned`), moves it through the memory hierarchy using DMA statements, calls device kernels for computation, and returns results to the host.

Host-side data is wrapped with shape information using the Croqtile runtime API:

```cpp
#include "choreo.h"
auto view = choreo::make_spanview<3>(raw_ptr, {6, 17, 128});
auto result = choreo_function(view);
```

The return type of a Croqtile function is `choreo::spanned_data`, which owns its buffer. In contrast, `choreo::spanned_view` is a non-owning reference used for passing input data.
