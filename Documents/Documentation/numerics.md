# Numeric Operations

## Overview

Croqtile supports a set of numeric built-in operations for device-side computation. These are available within `__co_device__` functions and through `call` to device kernels.

## Scalar Arithmetic

Standard C++ arithmetic operators work on Croqtile scalar types (`int`, `bool`):

- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`
- Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`

## Built-in Numeric Functions

Croqtile provides built-in numeric functions accessible in device code. These map to the target platform's optimized math library (e.g., CUDA fast math intrinsics).

### Exponential and Logarithmic

| Function | Description |
|----------|-------------|
| `__exp(x)` | Exponential (e^x) |
| `__expm1(x)` | exp(x) - 1 |
| `__log(x)` | Natural logarithm |
| `__log1p(x)` | log(1 + x) |
| `__pow(x, y)` | Power (x^y) |

### Trigonometric

| Function | Description |
|----------|-------------|
| `__sin(x)` | Sine |
| `__cos(x)` | Cosine |
| `__tan(x)` | Tangent |
| `__asin(x)` | Inverse sine |
| `__acos(x)` | Inverse cosine |
| `__atan(x)` | Inverse tangent |
| `__atan2(y, x)` | Two-argument inverse tangent |
| `__sinh(x)` | Hyperbolic sine |
| `__cosh(x)` | Hyperbolic cosine |
| `__tanh(x)` | Hyperbolic tangent |

### Algebraic and Rounding

| Function | Description |
|----------|-------------|
| `__sqrt(x)` | Square root |
| `__rsqrt(x)` | Reciprocal square root (1/sqrt(x)) |
| `__sign(x)` | Sign function (-1, 0, or 1) |
| `__ceil(x)` | Ceiling |
| `__floor(x)` | Floor |
| `__round(x)` | Round to nearest |
| `__isfinite(x)` | Finiteness test |
| `__alignup(x, a)` | Round up to alignment boundary |
| `__aligndown(x, a)` | Round down to alignment boundary |

### ML Activations

| Function | Description |
|----------|-------------|
| `__gelu(x)` | Gaussian Error Linear Unit |
| `__sigmoid(x)` | Sigmoid activation |
| `__softplus(x)` | Softplus activation |

*(Reference: `tests/gpu/end2end/numerics_core6.co`)*

## Type Promotion

When mixing fundamental types in expressions, Croqtile follows promotion rules similar to C++:

- Narrower types promote to wider types.
- Integer types promote to floating-point types when mixed.
- `f16`/`bf16` operations may accumulate in `f32` depending on the target.

## ElementCount Operator

The `|mdspan|` operator returns the total element count as an integer:

```choreo
int count = |sp|;           // product of all dimensions
call kernel(f.data, |f.span|);
```

This is frequently used to pass buffer sizes to device kernels.

## Float Type Variants

Croqtile supports multiple floating-point formats for AI workloads:

| Type | Bits | Exponent | Mantissa | Use Case |
|------|------|----------|----------|----------|
| `f16` | 16 | 5 | 10 | Standard half-precision |
| `bf16` | 16 | 8 | 7 | ML training/inference |
| `f32` | 32 | 8 | 23 | General purpose |
| `f64` | 64 | 11 | 52 | High-precision |
| `f8_e4m3` | 8 | 4 | 3 | Inference (Hopper+) |
| `f8_e5m2` | 8 | 5 | 2 | Training (Hopper+) |

*(Reference: `tests/gpu/end2end/float_types.co`, `tests/gpu/end2end/add_fp8_e4m3_to_f16.co`)*
