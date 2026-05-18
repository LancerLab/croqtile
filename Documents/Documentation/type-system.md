# Type System

## Type Categories

Croqtile has four fundamental type categories, each serving a distinct role:

| Category | Purpose | Examples |
|----------|---------|---------|
| **Scalar** | Program control (loop bounds, flags) | `int`, `bool` |
| **Spanned** | Shaped multi-dimensional data for computation | `f32 [64, 32]` |
| **I-Tuple** | Groups of integer values (indices, tiling factors) | `{4, 8, 16}` |
| **Bounded** | Variables with known ranges (loop variables, thread IDs) | `parallel p by 6` |

Additionally, **future** types arise from DMA statements and **mdspan** is a partial type representing shapes alone.

## Scalar Types

Croqtile provides two scalar types:

- **`int`** -- 32-bit signed integer, used for program control. Supports all standard arithmetic, shift, and logical operations with C++ syntax.
- **`bool`** -- Boolean type, identical to C++ `bool`. Conversion between `int` and `bool` follows C++ rules.

Croqtile does not provide unsigned scalar integers or other bit widths (8, 16, 64) as scalar types. These are not needed for program control in the tileflow domain. For data element types, see *fundamental types* below.

### Immutability

By default, all scalar variables in Croqtile are **immutable** -- they must be initialized at declaration and cannot be reassigned:

```choreo
int a = 5;
a = 10;       // error: reassignment not allowed
int b;        // error: declaration without initialization
```

When mutation is required (e.g., accumulators in reduction patterns), use the `mutable` keyword:

```choreo
mutable int sum = 0;
// sum can be reassigned inside loops
```

## Fundamental Types (Element Types)

Fundamental types describe the element type of spanned data. They can declare scalar variables (mutable by default) or be paired with a shape to form spanned data. They cannot carry a storage qualifier.

| Type | Description | Bits |
|------|-------------|------|
| `u8` | Unsigned integer | 8 |
| `u16` | Unsigned integer | 16 |
| `u32` | Unsigned integer | 32 |
| `s8` | Signed integer | 8 |
| `s16` | Signed integer | 16 |
| `s32` | Signed integer | 32 |
| `s64` | Signed integer | 64 |
| `f16` | IEEE-754 half-precision float | 16 |
| `bf16` | Brain floating-point | 16 |
| `f32` | IEEE-754 single-precision float | 32 |
| `f64` | IEEE-754 double-precision float | 64 |
| `f8_e4m3` | FP8 (4-bit exponent, 3-bit mantissa) | 8 |
| `f8_e5m2` | FP8 (5-bit exponent, 2-bit mantissa) | 8 |

Note: `s32` and `int` are distinct. `s32` is a fundamental type for data elements; `int` is a scalar type for program control. They cannot be interconverted.

```choreo
s32 a;          // error: fundamental type cannot declare a standalone variable
s32 [10] buf;   // OK: fully-typed spanned data
int x = 5;      // OK: scalar for control
```

## The mdspan Partial Type

An `mdspan` (multi-dimensional span) represents a shape without associated data. It is a *partial type* -- you cannot allocate storage with an mdspan alone, but you can define, manipulate, and pass shapes independently.

```choreo
mdspan sp : [7, 8];         // 2D shape
s3 : [7, 8, 9];             // type-inferred mdspan
```

The mdspan is Croqtile's mechanism for first-class shape manipulation. See [Shapes and mdspan](shapes-and-mdspan.md) for full details.

## Spanned Types (Complete Types)

A **spanned type** combines a fundamental type with an mdspan to form a complete type that can declare data with storage:

```choreo
f32 [32, 16] data;            // 32x16 buffer of f32
shared f16 [128, 64] tile;    // shared memory tile
```

Spanned types can carry a **storage qualifier** (`global`, `shared`, `local`) to place data in different memory hierarchy levels. See [Spanned Data](spanned-data.md).

## Bounded Types

Bounded types represent values with a known range `[0, upper_bound)`. They arise from `parallel-by` and `with-in` statements:

- **Bounded integer**: `parallel p by 6` makes `p` a bounded integer with range `[0, 6)`.
- **Bounded ituple**: `with index = {x, y} in [10, 20]` makes `index` a bounded ituple with range `[10, 20]`.

Bounded variables are used to select subviews of spanned data -- they determine which tile or partition of a buffer is accessed within a given loop iteration or parallel thread. The upper-bound is always available (via `#`) for shape computations, while the current value is used in view operations like `chunkat` and `subspan.at`. See [Parallel-by](parallel-by.md) and [Iteration](iteration.md).

## Future Types

A **future** is produced by a DMA statement. It represents both the handle of an (optionally asynchronous) operation and a reference to the destination buffer:

```choreo
f = dma.copy.async input => shared;
wait f;
call kernel(f.data, |f.span|);
```

Futures provide `.data` (the destination buffer) and `.span` (the destination shape). See [DMA Basics](dma-basics.md).

## Type Inference

Croqtile aggressively infers types from initialization expressions. The explicit type keyword can often be omitted:

```choreo
a = 5;                 // inferred as int
b = {3, 4};            // inferred as ituple<2>
sp : [7, 8];           // inferred as mdspan<2>
f = dma.copy x => shared;  // inferred as future
```

The compiler also infers shapes through DMA operations, `chunkat` expressions, and arithmetic on mdspans. The [Value Numbering](../Developer/value-numbering.md) chapter in the Developer Guide explains the inference pipeline.

## Explicit Type Conversion with `__to`

When working with different fundamental types, you often need to convert between them explicitly. Croqtile provides the `__to` builtin:

```choreo
__to<target_type>(expression)
```

The target type inside `<...>` can be any fundamental type supported by the current compilation target (e.g. `f32`, `f16`, `bf16`, `s8`, `s16`, `s32`, `u8`, `u16`, `u32`). The source expression type is inferred automatically.

```choreo
__co__ auto example(int i, float f) {
  f32 x = __to<f32>(i);      // int -> f32
  bf16 y = __to<bf16>(f);    // f32 -> bf16
  s32 z = __to<s32>(f);      // f32 -> s32
  u8 b = __to<u8>(i);        // int -> u8
}
```

### Why Use `__to` Instead of Implicit Conversion?

Croqtile inserts implicit conversions automatically when types mismatch. However, implicit conversions that lose precision emit compiler **warnings**:

```choreo
mutable s32 a = 3.14f;   // warning: implicit conversion may lose precision
```

Use `__to` to suppress the warning and make the intent explicit:

```choreo
s32 a = __to<s32>(3.14f);  // no warning -- conversion is deliberate
```

| Scenario | Implicit (assignment) | Explicit (`__to`) |
|----------|----------------------|--------------------|
| Value-preserving (e.g. `s16` -> `s32`) | No warning | No warning |
| Lossy (e.g. `f32` -> `bf16`) | Warning emitted | No warning |
| Reinterpretive (e.g. `s32` -> `u32`) | Warning emitted | No warning |

### Target-Specific Type Support

Not all targets support all types. For instance, `u64`/`s64` may not be available on every architecture. If you attempt an unsupported conversion, the compiler reports an error during semantic analysis:

```
error: the target type 'u64' is not supported by the current target ...
```

Consult your target's documentation for the set of supported scalar types.

## Foreign Type Cast with `__to<"type">`

The `__to` syntax also supports casting to types that Croqtile does not recognize (compiler extensions like `__fp16`, `__nv_bfloat16*`, or library-defined types). Pass the target type as a **quoted string** instead of a fundamental type keyword:

```choreo
__to<"target_type_string">(expression)
```

The type string inside `<"...">` is passed **verbatim** to the generated C++ code. Croqtile performs no validation on the type name -- it is the programmer's responsibility to ensure the type is valid in the target compiler.

```choreo
__co__ auto example(f16 [128] input) {
  parallel by 1 {
    // Cast nullptr to a concrete pointer type for template deduction
    call my_lib::matmul<16, my_lib::MK_NK>(
        input.data,
        __to<"__fp16*">(nullptr),
        __to<"__fp16*">(nullptr)
    );
  }
}
```

This generates:

```cpp
my_lib::matmul<16, my_lib::MK_NK>(input, ((__fp16*)nullptr), ((__fp16*)nullptr));
```

### When to Use `__to<"type">`

| Scenario | Use |
|----------|-----|
| Template argument deduction needs a concrete pointer type | `__to<"__fp16*">(nullptr)` |
| Passing data to a function expecting a non-Croqtile type | `__to<"const float*">(ptr)` |
| Compiler extension types (`__fp16`, `__nv_bfloat16`, etc.) | `__to<"__nv_bfloat16*">(expr)` |

### Summary: `__to` with Fundamental Type vs String

- `__to<type>(expr)` -- converts between Croqtile **fundamental types** (e.g., `f32`, `bf16`). The compiler validates both source and target types.
- `__to<"type">(expr)` -- emits a C-style cast to an **arbitrary C++ type string**. No type validation is performed by the compiler.

*(Reference: `tests/check/foreign_cast.co`, `tests/check/explicit_cast.co`)*

## Immutability as Default

All Croqtile variables are immutable by default. This is a deliberate design choice:

- **Integers**: initialized once, cannot be reassigned
- **I-Tuples**: initialized once, cannot be reassigned
- **mdspans**: initialized once, cannot be reassigned
- **Spanned data**: buffer contents are modified only through DMA statements, not direct assignment
- **Bounded variables**: range is fixed at the `parallel-by` or `with-in` declaration

The `mutable` keyword is the escape hatch for the few cases requiring reassignment. This default immutability enables aggressive compile-time analysis and shape inference.
