# Spanned Data and Memory

## What is Spanned Data?

In Croqtile, the primary objects of data movement are **spanned data** -- multi-dimensional arrays associated with a shape (mdspan) and element type. Both input/output parameters and internal buffers are spanned data. The term "spanned" reflects that every piece of data carries its shape information.

## Declaring Spanned Data

A declaration requires a fundamental type and a shape:

```choreo
f32 [32, 16] data;              // 32x16 buffer of f32
```

The shape can come from a named mdspan or be derived:

```choreo
shape : [72, 14];
s32 [shape] data0;              // [72, 14]
u32 [shape(0), shape(1)] data1; // same shape
u16 [shape + 6] data2;          // [78, 20]
s16 [shape, 8] data3;           // [72, 14, 8]
```

## Fundamental Types

The element type of spanned data is called the **fundamental type**:

| Category | Types |
|----------|-------|
| Unsigned integers | `u8`, `u16`, `u32` |
| Signed integers | `s8`, `s16`, `s32`, `s64` |
| Floating-point | `f16`, `bf16`, `f32`, `f64` |
| FP8 variants | `f8_e4m3`, `f8_e5m2` |

Fundamental types can declare scalar variables (mutable by default) or be paired with a shape to form spanned data. They cannot carry a storage qualifier.

```choreo
s32 a;          // OK: scalar variable (mutable, equivalent to `mutable int a`)
s32 [10] buf;   // OK: spanned data with shape [10]
shared s32 a;   // error: storage qualifier not allowed on scalar
```

Note: `s32` (fundamental type for data) and `int` (scalar type for control) are distinct and non-interconvertible.

## Storage Qualifiers

Spanned data can be placed in different memory hierarchy levels using **storage qualifiers**:

| Qualifier | Description | Typical Use |
|-----------|-------------|-------------|
| (none) / `global` | Host or device global memory, largest capacity, highest latency | Large inputs/outputs |
| `shared` | Shared memory within a compute unit (e.g., GPU block) | Intermediate tiles, reused across threads |
| `local` | Thread-private memory, smallest, fastest | Per-thread scratch buffers |

```choreo
global f32 [100, 200] matrix;    // explicit global
shared f32 [10, 10] tile_data;   // shared within a block (typical GPU tile)
shared f16 [5] stage;            // another shared buffer (common for DMA staging)
local f16 [4] thread_scratch;    // per-thread scratch (still supported)
f32 [100, 100] d0;               // implicit global (default)
```

### Platform-Specific Semantics

Storage qualifier semantics depend on the target platform. For CUDA/CuTe:

- `global` maps to device global memory.
- `shared` maps to `__shared__` memory (per-block).
- `local` maps to thread-private registers or local memory.

Other targets may define these differently. The Croqtile compiler enforces placement rules: `shared` and `local` buffers can only be declared inside `parallel-by` blocks (device code), while `global` buffers can be declared anywhere.

## Initialization

Buffers can be initialized at declaration with a uniform value:

```choreo
shared s32 [17, 6] b1 {0};         // all elements 0
shared f32 [128, 16] b2 {3.14f};  // all elements 3.14
```

The `{value}` syntax after the variable name sets every element to the given constant.

## Parameters and Return Values

Spanned data is how the host program communicates with tileflow functions:

```choreo
__co__ f16 [7, 8] foo(f32 [16, 17, 5] input) { ... }
```

Parameters cannot have storage qualifiers or initializers. The return type specifies the shape and element type of the output.

### The `.span` Member

The `.span` member returns the mdspan of a spanned parameter, enabling shape-relative buffer declarations:

```choreo
__co__ auto foo(f32 [16, 17, 5] input) {
  f32 [input.span] same_shape;
  f32 [input.span / {4, 1, 5}] tiled;
}
```

### Host-Side API

From C++ host code, raw pointers are wrapped with shape information using the Croqtile runtime:

```cpp
#include "choreo.h"
auto view = choreo::make_spanview<3>(ptr, {6, 17, 128});
auto result = choreo_function(view);  // returns spanned_data (owning)
```

- `choreo::spanned_view` -- non-owning reference for passing input data.
- `choreo::spanned_data` -- owning buffer returned from Croqtile functions.

Both support C-style array indexing and `.shape()` for querying dimensions.

## Buffer Lifetime Management

The Croqtile compiler manages buffer lifetimes automatically. It reuses storage when lifetimes do not overlap, minimizing memory footprint. Programmers do not need to manually free buffers.

## The `.data` Accessor

Futures from DMA operations provide `.data` to access the destination buffer as spanned data:

```choreo
f = dma.copy input.chunkat(p) => shared;
call kernel(f.data, |f.span|);
```

See [DMA Basics](dma-basics.md) for the full future interface.
