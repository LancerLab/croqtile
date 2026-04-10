# Symbolic and Dynamic Shapes

## Static vs Dynamic Shapes

Most high-performance kernels operate on data with **fixed shapes** known at compile time, enabling aggressive optimization. However, real-world deployments often require **dynamic shapes** -- dimensions determined at runtime by configuration files, framework shape inference, or user input.

Croqtile supports both static and dynamic dimensions in a unified framework.

## Anonymous Dimensions (`?`)

The simplest way to declare a dynamic dimension is the **anonymous dimension** using `?`:

```choreo
__co__ auto foo(s32 [?, 1, 2] input) { ... }
```

The `?` represents a compile-time unknown positive integer. The actual value is provided at runtime through the host-side API:

```cpp
void bar(int* data, int dim0) {
  foo(choreo::make_spanview<3>(data, {dim0, 1, 2}));
}
```

Anonymous dimensions are sufficient for simple cases but lack the ability to express relationships between shapes.

## Symbolic Dimensions (Named)

**Symbolic dimensions** are named dynamic dimensions that express relationships explicitly:

```choreo
__co__ auto Matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
  int tile_m = 32, tile_n = 8, tile_k = 16;
  s32 [M / tile_m, K / tile_k] tiled_lhs;
  s32 [K / tile_k, N / tile_n] tiled_rhs;
  // ...
}
```

Here `M`, `K`, `N` are symbolic dimension names. The compiler can reason about their relationships -- for instance, `K` appears in both `lhs` and `rhs`, so the compiler knows these dimensions must match at runtime.

## Why Symbolic Dimensions are Better

### Readability

Symbolic names make shape relationships self-documenting. Compare:

```choreo
// Symbolic: relationship between lhs and rhs is clear
__co__ auto Matmul(s32 [M, K] lhs, s32 [K, N] rhs) { ... }

// Anonymous: no visible relationship
__co__ auto Matmul(s32 [?, ?] lhs, s32 [?, ?] rhs) { ... }
```

### Safety Through Additional Checks

The Croqtile compiler generates additional runtime checks for symbolic dimensions. When dimension `K` appears in both parameters, the transpiled host code verifies consistency:

```cpp
croq_assert(lhs.shape()[1] == rhs.shape()[0],
              "dimension 'K' is not consistent.");
```

With anonymous dimensions, this cross-parameter check is impossible -- the compiler only verifies that individual dimensions are positive.

### Compile-Time Shape Algebra

Symbolic dimensions participate in shape arithmetic and derivation, preserving relationships through the inference pipeline:

```choreo
__co__ auto foo(f32 [M, K] input) {
  s32 [M / 4, K / 8] tiled;   // compiler tracks M/4 and K/8
}
```

## Runtime Checks

For both anonymous and symbolic dimensions, Croqtile generates runtime assertions in the transpiled host code:

- **Zero-dimension check**: Every derived shape must have positive dimensions.
- **Consistency check** (symbolic only): Dimensions sharing a symbolic name must have the same runtime value.

These checks execute once at function entry with negligible overhead.

```cpp
void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,
                                choreo::span_view<2, choreo::s32> rhs) {
  croq_assert(lhs.shape()[1] == rhs.shape()[0],
                "dimension 'K' is not consistent.");
  croq_assert(lhs.shape()[0] / 32 > 0,
                "dimension would produce zero-sized buffer.");
  // ...
}
```

## Limitations

1. **Parameters only**: Dynamic dimensions (both `?` and symbolic) can only appear on spanned data in the parameter list. Local buffer declarations inside the function must have statically-determined shapes:

```choreo
__co__ void foo() {
  f32 [M] d;    // error: dynamic dimension in local buffer
}
```

2. **Conservative compile-time analysis**: Dynamic shapes limit the compiler's ability to perform static checks. Some checks that would be compile-time errors with fixed shapes become runtime assertions instead.

3. **No negative dimensions**: Both `?` and symbolic dimensions must resolve to positive integers at runtime.

## Recommendations

- Use **symbolic dimensions** over `?` whenever possible for better safety and readability.
- Use **fixed shapes** for performance-critical inner buffers where the size is known.
- Add `croq_assert` statements when the compiler's automatic checks are insufficient for your constraints.
