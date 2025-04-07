## Overview
n this section, you will learn about the *anonymous dimension* and *symbolic dimension* in Choreo and how they support dynamic shapes required by certain systems.

## Fixed and Dynamic Shape
Data with fixed shapes is commonly used in high-performance computing kernels, allowing for aggressive optimization with known dimensions. However, certain scenarios require handling shapes with runtime dimensions, known as dynamic shapes. This necessitates kernel generality to manage inputs of varying shapes, a feature highlighted by many machine learning frameworks.

## *mdspan* with **Anonymous Dimension**
Choreo supports shape dimensions with unknown values, similar to many high-level machine learning languages. The simplest method is to use **Anonymous Dimension** in `mdspan`:

```choreo
__co__ auto foo(s32 [?, 1, 2] input) { ... }
```

Here, the question mark `?` represents a compile-time unknown value. Any positive integer could be assumed here. However, the value must be provided at runtime, as the following example shows:

```cpp
void bar(int * data, int dim0) {
  foo(choreo::make_spanview<3>(data, {dim0, 1, 2}));
}
```

In this example, the first dimension of `input` is provided by `foo`'s *spanned* input, where the value is retrieved from `bar`'s caller at execution. You may assume the value be determined by reading a configuration file, through shape derivation in a machine learning framework, or etc..

The *anonymous dimension* can work properly for this case. However, in certain scenarios, it is not sufficient.

## Step Further: **Symbolic Dimension**

Another approach in Choreo to support dynamic shape is to use **Symbolic Dimension**. *Symbolic dimension*s are named but as well unknown at compile-time. Here's an example:

```choreo
__co__ auto Matmul(s32 [M, K] lhs, s32 [K, N] rhs) {
  int tile_m = 32, tile_m = 8, tile_k = 16;

  s32 [M / tile_m, K / tile_k] tiled_lhs;
  s32 [K / tile_k, N / tile_n] tiled_rhs;

  // ...
}
```
In the code, each dimension is given a symbolic name (`M, K, N` in this case) for the *spanned* inputs. Compared to *anonymous dimension*, *symbolic dimension* approach clearly describes the relationship between shapes. The code looks intuitive and is easy to maintain. Furthermore, it is possible to improve the code safety with the additional information provided. Let us dive deeper for this.

## Improved Code Safety
The reason for improved code safety of using *symbolic dimension* is that more comprehensive code checks can be applied by Choreo compiler. For example, consider an *anonymous dimension* version of the `Matmul` function:

```choreo
__co__ auto Matmul(s32 [?, ?] lhs, s32 [?, ?] rhs) {
  int tile_m = 32, tile_m = 8, tile_k = 16;

  s32 [lhs.span / {tile_m, tile_k}] tiled_lhs;  // 'mdspan' and 'ituple': rank must be same
  s32 [rhs.span / {tile_k, tile_n}] tiled_rhs;  // 'tiled_rhs' can not have a zero dimension

  // ...
}
```

Choreo performs both compile-time and runtime checks to ensure safety:

- **Compile-time Checks**: Choreo verifies rank consistency. In this example, `lhs.span` has a ranked of `2`, matching the rank of `{tile_m, tile_k}`, so there are no issues.
- **Runtime Checks**: Choreo generates code to validate dimensions at runtime. For instance, in the declaration of `tiled_lhs`, its shape must not have a dimension of `0`, which would result in an invalid zero-sized buffer. Choreo's *transpilation* process generates the following *target host code*:

```cpp
void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,
                                choreo::span_view<2, choreo::s32> rhs) {
  choreo_assert(lhs.shape()[0] / 32 > 0,
                "the 0 dimension of 'lhs' may result in spanned data "
                "with a dimension value of 0.");
  // ...
}
```

Here, the `choreo_abort` function aborts program execution if the condition is not met. This early check helps Choreo detect issues promptly.

For the *symbolic dimension* version, Choreo compiler generates additional checks in the *target host code*:

```cpp
void __choreo_transpiled_Matmul(choreo::span_view<2, choreo::s32> lhs,
                                choreo::span_view<2, choreo::s32> rhs) {
  // additional check happens only when using symbolic dimensions
  choreo_assert(lhs.shape()[1] == rhs.shape()[0],
                "dimension 'K' is not consistent.");

  choreo_assert(lhs.shape()[0] / 32 > 0,
                "the 0 dimension of 'lhs' may result in spanned data "
                "with a dimension value of 0.");
  // ...
}
```

In addition to the checks performed in the *anonymous dimension* version, the *symbolic dimension* version verifies the consistency of dimensions between `lhs` and `rhs`, resulting in safer code.

Therefore, it is recommended that programmers use symbolic dimensions not only for ease of use but also to enhance code safety.

## Cost and Limitations
To support dynamic shapes, Choreo compile has to be conservative at compile time. This limitation restricts certain compile-time checks and necessitates some checks to be performed at runtime instead. Although the implementation aims to minimize overhead by placing these checks in the host code, they **remain more conservative compared to code with fixed-sized shapes**. Programmers should be mindful of this situation and may need to add user-level assertions to enhance code safety when necessary.

A significant limitation of both *anonymous dimension* and *symbolic dimension* is that they can **only be applied to the *spanned data* in parameter lists**. Dimensions of buffers declared within Choreo functions cannot be made dynamic. For example, the following code will result in a compile-time error:

```choreo
__co__ void foo() { f32 [M] d;} // compile-time error: can not applied to the spanned buffer
```

## Quick Summary
In this section, we discussed *anonymous dimension* and *symbolic dimension* in Choreo, which support dynamic shapes required in certain scenarios. Compared to *anonymous dimension*, using *symbolic dimension* enhance code safety through additional compile-time checks, making them the recommended choice for ease of use and improved safety.
