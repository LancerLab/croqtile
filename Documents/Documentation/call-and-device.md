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

`call` is required (not optional) for ordinary device functions -- Croqtile
functions cannot call C++ functions without it. (The one exception is
[intrinsic passthrough](#intrinsic-passthrough), which lets registered target
intrinsics pass through verbatim without the `call` keyword.)

## Semantics

- The callee must be a C++ function (device or host, depending on context).
- `call` can only appear inside a `parallel-by` block (device code region).
- The Croqtile compiler does **not** verify the callee's signature at transpile time. Errors are caught during target compilation.
- Arguments are passed by value. Spanned data arguments are implicitly decayed to pointers.
- `call` returns the callee's value, which can be captured by assignment
  (`mutable int a = call bar(a, b);`) or tested in an `if` condition.

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

## Intrinsic Passthrough

Some target backends expose low-level hardware or compiler intrinsics that
have no Croqtile-language equivalent and must be emitted verbatim into the
generated device code. Croqtile supports this through *intrinsic passthrough*:
register a call prefix or a namespace with a `#pragma`, and any bare call (no
`call` keyword) that matches is passed through unchanged.

Two pragmas are available:

```choreo
#pragma croq intrinsic prefix <prefix>
#pragma croq intrinsic namespace <namespace>
```

- `prefix` registers a leading token. Any identifier that *starts with* the
  prefix becomes an intrinsic call; for example, prefix `hw_` matches
  `hw_transfer(...)`. A namespace-style prefix such as `vendor::` is also
  supported.
- `namespace` registers a scope qualifier. Any call of the form
  `namespace::name(...)` is treated as an intrinsic call; for example,
  namespace `arch` matches `arch::device_init(...)`.

```choreo
#pragma croq intrinsic prefix hw_
#pragma croq intrinsic namespace arch

__co__ auto example(s32 [32, 4] a) {
  parallel by 1 {
    arch::device_init();   // emitted verbatim
    hw_transfer(0, 0, 0);  // emitted verbatim
  }
  return 0;
}
```

### Scope

Intrinsic registrations are scoped:

| Declaration site | Visibility |
|------------------|------------|
| File scope (outside any function) | All functions in the file |
| Function scope | Within the declaring function only |
| Block scope (e.g. inside `parallel-by`) | Within the enclosing block only |

Registrations do not leak past their scope: a prefix declared in one function
is not visible in a later function, and a prefix declared inside a block is
not visible after that block ends. Outer (file-scope) registrations remain in
effect after an inner scope ends.

### Semantics

- Intrinsic calls are emitted **verbatim**; Croqtile does not verify the
  callee's name, signature, or return type. Errors are caught during target
  compilation.
- Matching is name-based only: any bare call whose name begins with a
  registered prefix, or is qualified by a registered namespace, passes through
  verbatim. This also applies to device code you have written yourself -- for
  example, a `__co_device__` function or a host-side C++ device function whose
  name happens to match a registered prefix or namespace is emitted verbatim,
  bypassing normal `call` resolution and signature checking.
- Intrinsic calls must appear in a device-code region (inside `parallel-by`),
  like ordinary `call` statements.
- Arguments follow the same passing rules as `call` (see
  [Argument Types](#argument-types)).
- The namespace `croq` (and prefixes beginning with `croq::`) are reserved for
  built-in functions and cannot be used for intrinsic registration.

## Restrictions

- Croqtile functions **cannot call other Croqtile functions**. Only device/host C++ functions are callable.
- `call` must be inside a `parallel-by` block.
- `call` is not a general expression: its result can be captured by
  assignment, but it cannot be nested inside arithmetic or other expressions.

*(Reference: `tests/parse/call_stmt.co`, `tests/check/illegal_calls.co`)*
