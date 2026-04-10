# Input/Output and Linking

## Function Parameters

Croqtile functions accept two types of parameters:

- **Spanned data**: Multi-dimensional shaped buffers.
- **Integers**: Scalar control values.

```choreo
__co__ void foo(f32 [7] a, int n) { ... }
__co__ void bar(f32 [7] a, int) { ... }  // unnamed int parameter
```

By default, spanned data parameters represent references to **host-side** data. The compiler handles the host-to-device transfer automatically.

### `global` Parameters -- Device Memory Input

When the `global` storage qualifier is placed before a parameter, it indicates that the caller passes a **device global memory pointer** directly, bypassing the automatic host-to-device transfer:

```choreo
__co__ void matmul(global f16 [M, K] lhs, global f16 [N, K] rhs,
                   global f16 [M, N] output) {
  // lhs, rhs, output are already in device global memory
  ...
}
```

On the host side, `global` parameters are passed by wrapping device pointers with `make_spanview`:

```cpp
half *a_d, *b_d, *c_d;
cudaMalloc(&a_d, M * K * sizeof(half));
cudaMalloc(&b_d, N * K * sizeof(half));
cudaMalloc(&c_d, M * N * sizeof(half));

// wrap device pointers with shape information
auto lhs_d = choreo::make_spanview<choreo::f16, 2>(a_d, {M, K});
auto rhs_d = choreo::make_spanview<choreo::f16, 2>(b_d, {N, K});
auto res_d = choreo::make_spanview<choreo::f16, 2>(c_d, {M, N});

matmul(lhs_d, rhs_d, res_d);
```

This is the typical pattern for high-performance kernels where the caller manages device memory explicitly.

| Parameter Style | Memory Location | Host-Side Type | Transfer |
|----------------|-----------------|----------------|----------|
| `f32 [M, K] x` | Host | `make_spanned` / `.view()` | Automatic |
| `global f32 [M, K] x` | Device global | `make_spanview` on device pointer | None (already on device) |

*(Reference: `tests/gpu/end2end/matmul_f16_dynamic.co`)*

## Host-Side Data Wrapping

### `make_spanned` / `make_spanview`

Raw C++ pointers must be wrapped with shape information before passing to Croqtile functions:

```cpp
#include "choreo.h"

__co__ void foo(f32 [7, 16] input, f32 [7, 16] output) { ... }

void entry(float* input, float* output) {
  foo(choreo::make_spanned<2>(input, {7, 16}),
      choreo::make_spanned<2>(output, {7, 16}));
}
```

The template parameter is the rank. The shape initializer list provides dimension values.

Dynamic dimensions are supported -- the Croqtile runtime checks them against the function signature at entry:

```cpp
void entry(float* input, int M, int N) {
  foo(choreo::make_spanned<2>(input, {M, N}));
  // Croqtile checks M == 7 and N == 16 at runtime
}
```

If the dimensions don't match, the program terminates with an assertion failure.

### `spanned_view` vs `spanned_data`

| Type | Ownership | Use |
|------|-----------|-----|
| `choreo::spanned_view<T, Rank>` | Non-owning | Input parameters |
| `choreo::spanned_data<T, Rank>` | Owning (`unique_ptr`) | Return values |

Both support C-style array indexing (`res[i][j]`) and `.shape()` for dimension queries.

## Return Values

### Return Type Inference

Croqtile supports `auto` return type deduction:

```choreo
__co__ auto foo() {
  f32 [7, 16] result;
  // ...
  return result;
}
```

The compiler infers the return type as `f32 [7, 16]`.

### Return Semantics

Returns use move semantics (no copy):

```cpp
auto result = foo();  // result is choreo::spanned_data
// result owns the buffer -- no copy happened
```

The returned `spanned_data` uses `std::unique_ptr` internally for zero-copy return value optimization.

## CUDA Stream Integration

For GPU targets, Croqtile functions can accept a CUDA stream for asynchronous kernel execution:

```choreo
__co__ void foo(f32 [64] input, stream s) {
  parallel p by 4 { ... }
}
```

*(Reference: `tests/gpu/end2end/stream.co`)*

## Linking and Name Mangling

Croqtile functions follow the C++ calling convention with standard C++ name mangling. This means:

- Croqtile functions can be linked with standard C++ linkers.
- `extern "C"` is **not supported** for Croqtile functions (they use template-based parameter types).
- Function overloading follows C++ rules.

### Current Limitations

- **Forward declarations**: Croqtile function declarations without definitions are not yet supported.
- **`inline`**: Croqtile functions cannot be declared `inline` for header use.

*(Reference: `tests/parse/co_func.co`)*
