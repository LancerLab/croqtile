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

Function-like macros (with parameters) are **not supported**:

```cpp
#define MAX(a, b) (((a) > (b)) ? (a) : (b))  // will not work in Croqtile
```

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

Supported directives: `#if`, `#ifdef`, `#ifndef`, `#else`, `#endif`, `#define`.

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

The Croqtile preprocessor defines target-specific macros to support conditional compilation:

| Macro | Defined When |
|-------|-------------|
| `__CUDA__` | Targeting CUDA/CuTe |
| `__CUDA_ARCH__` | Inside CUDA device code compilation |

These can be used in both tileflow and host code for target-specific paths.

*(Reference: `tests/pp/`, `tests/pp/ifdef.co`, `tests/pp/cond_define0.co`)*
