# Calling Device Code

## The `call` Statement

Inside a tileflow function, the `call` keyword invokes C++ device functions:

```choreo
void bar(float* a, float* b, int n) { ... }

__co__ void foo(f32 [64] input) {
  parallel p by 4 {
    f = dma.copy input.chunkat(p) => shared;
    call bar(f.data, f.data, |f.span|);
  }
}
```

`call` is required (not optional) -- Croqtile functions cannot call C++ functions without it.

## Semantics

- The callee must be a C++ function (device or host, depending on context).
- `call` can only appear inside a `parallel-by` block (device code region).
- The Croqtile compiler does **not** verify the callee's signature at transpile time. Errors are caught during target compilation.
- Arguments are passed by value. Spanned data arguments are implicitly decayed to pointers.

## `__co_device__` Functions

Croqtile supports writing device-side compute functions in Croqtile-like syntax:

```choreo
__co_device__ void layernorm(f32 [N] input, f32 [N] output) {
  // scalar loops, conditionals, etc.
}

__co__ void foo(f32 [256] data) {
  parallel p by 4 {
    f = dma.copy data.chunkat(p) => shared;
    call layernorm(f.data, f.data);
  }
}
```

`__co_device__` functions are transpiled into device code but use Croqtile scalar and loop constructs rather than raw C++.

## Argument Types

The following types can be passed through `call`:

| Croqtile Expression | Passed As |
|-------------------|-----------|
| `f.data` (future data) | Pointer to buffer |
| `int` variable | Integer value |
| `\|f.span\|` (element count) | Integer value |
| Compile-time constant | Literal value |
| `nullptr` | Null pointer literal |
| `__to<type>(expr)` | Explicit type conversion (validated) |
| `__to<"type">(expr)` | Foreign type cast (verbatim, unvalidated) |

Spanned data is **decayed to a typed pointer** when passed to device functions. The device function sees it as a flat pointer without shape information.

### Foreign Type Casts

When a C++ template function requires a concrete pointer type for template argument deduction, bare `nullptr` may cause deduction failures. Use `__to<"type">` with a quoted string to provide the expected type:

```choreo
// Without cast: template deduction fails on nullptr_t
call my_lib::matmul<TM, my_lib::MK_NK>(data, nullptr, nullptr);

// With __to<"type">: provides concrete __fp16* for deduction
call my_lib::matmul<TM, my_lib::MK_NK>(data, __to<"__fp16*">(nullptr), __to<"__fp16*">(nullptr));
```

See [Type System - Foreign Type Cast](type-system.md#foreign-type-cast-with-__to%22type%22) for full details.

## Template Calls

Device functions can be C++ templates. Template arguments are passed in `<>`:

```choreo
__cok__ {
  template <int M, int N>
  void matmul_kernel(float* a, float* b, float* c) { ... }
}

__co__ void foo() {
  parallel p by 1 {
    call matmul_kernel<16, 32>(a, b, c);
  }
}
```

Template arguments must be compile-time constants. The Croqtile compiler generates explicit template specializations in the target code.

## Restrictions

- Croqtile functions **cannot call other Croqtile functions**. Only device/host C++ functions are callable.
- `call` must be inside a `parallel-by` block.
- No return value capture -- `call` is a statement, not an expression.

*(Reference: `tests/parse/call_stmt.co`, `tests/check/illegal_calls.co`)*
