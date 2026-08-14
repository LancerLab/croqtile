# Macros and Preprocessing

## Object-Like Macros

Croqtile's preprocessor supports **object-like macros** for text substitution across both host and tileflow code:

```choreo
#define M 256
#define N 32
#define K 64

__co__ auto matmul(f32 [M, K] lhs, f32 [K, N] rhs) { ... }

void foo() {
  choreo::f32 a[M][K];
  auto res = matmul(choreo::make_spanview<2>(a, {M, K}), ...);
}
```

Macros are expanded before Croqtile compilation, producing statically-shaped tileflow functions. This keeps host and tileflow code consistent.

### Function-Like Macros

Function-like macros (with parameters) are supported and expanded during
Croqtile preprocessing, including nested definitions where one macro's body
references another:

```choreo
#define ADD(a, b) (a + b)
#define SUB(a, b) (a - b)
#define NESTED_ADD(a, b) ADD(ADD(a, b), b)

__co__ void foo(f32 [128] a) {
  foreach {i} in [a.span(0)] {
    a.at(i) = NESTED_ADD(a.at(i), 1.0f);
  }
}
```

Arguments are substituted textually into the macro body before Croqtile
compilation. As in C/C++, parenthesize parameters and bodies to avoid
precedence surprises after expansion.

## Comments

Croqtile supports both C-style comment forms:

```choreo
// single-line comment
/* multi-line
   comment */
```

## Conditional Compilation

Standard conditional directives work across host and tileflow code:

```choreo
#define PATH0

__co__ void foo() {
#ifdef PATH0
  // code for PATH0
#else
  // alternative code
#endif
}

#ifdef PATH0
// host code for PATH0
#endif
```

Supported directives: `#define`, `#undef`, `#if`, `#ifdef`, `#ifndef`,
`#elif`, `#else`, `#endif`.

`#include` is **not** expanded by Croqtile preprocessing. Quote-style
`#include "..."` lines are passed through to the C++ preprocessor; Croqtile
only follows them to collect `__cok__` device-kernel blocks from the included
files. Angle-bracket `#include <...>` is not followed at all.

## Preprocessing Order

Croqtile preprocessing runs **before** C++ preprocessing:

```
Croqtile preprocessing -> Croqtile compilation -> C++ preprocessing -> C++ compilation
```

The Croqtile preprocessor only processes code inside tileflow functions (`__co__` blocks). Other code is left for the C++ preprocessor. This ordering means:

- Macros defined for Croqtile are expanded in tileflow code first.
- C++ preprocessor macros from system headers are not available during Croqtile preprocessing.
- The same macro name can be used in both Croqtile and C++ contexts.

## Pre-Defined Target Macros

The Croqtile preprocessor defines target-specific macros through each target's
`ChoreoMacros` hook. Only targets that override it provide macros:

| Macro | Defined When |
|-------|-------------|
| `__CHOREO_TARGET_AMDGPU__` | Targeting AMDGPU/HIP |
| `__CHOREO_AMDGPU_ARCH__` | Targeting AMDGPU (set to the arch name) |

These can be used in tileflow code for target-specific paths.

Macros such as `__CUDA__`, `__CUDA_ARCH__`, and `__CHOREO_TARGET_CUTE__` are
**not** defined by the Croqtile preprocessor. They are provided later by the
target toolchain: nvcc defines `__CUDA__`/`__CUDA_ARCH__`, and the code
generator passes `__CHOREO_TARGET_CUTE__` to nvcc as a `-D` flag.

*(Reference: `tests/pp/`, `tests/pp/ifdef.co`, `tests/pp/cond_define0.co`)*
