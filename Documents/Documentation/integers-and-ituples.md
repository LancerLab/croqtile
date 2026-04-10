# Integers and I-Tuples

## Integers

In Croqtile tileflow programs, integers are primarily used for **shape composition** -- defining tile sizes, computing dimension extents, and deriving shapes from other shapes. They may also appear in program control expressions. Loop control itself is delegated entirely to bounded variables (see [Iteration](iteration.md)). Croqtile provides a single integer type: the **32-bit signed integer**.

### Definition

Integer variables follow C/C++-like syntax:

```choreo
int a = 1;
a = 1;            // type inferred from initializer
int x = 1, y = 2, z = 3;
```

### Immutability

Integers are immutable by default -- they must be initialized at declaration and cannot be reassigned:

```choreo
int a;        // error: no initializer
int b = 1;
b = 3;        // error: reassignment
```

When mutable integers are needed (e.g., accumulators), use the `mutable` keyword:

```choreo
mutable int sum = 0;
```

### As Parameters

Integers can be passed between host and tileflow programs, and from tileflow to device kernels:

```choreo
__co__ void foo(int a) {
  parallel p by 4 { call kernel(a, 3); }
}
```

Note that mdspan-typed shapes cannot be passed as function parameters, but integers can.

## I-Tuples: Groups of Integers

An **integer-tuple** (i-tuple) groups multiple integer values. Common uses include tiling factors, multi-dimensional indices, and shape deltas.

### Defining I-Tuples

```choreo
ituple a = {1, 2, 3};      // explicit keyword
b = {4, 5, 6};              // type inferred (uses = and {})
```

To distinguish from mdspan definitions:

| | I-Tuple | mdspan |
|---|---------|--------|
| Initializer | `= {values}` | `: [values]` |
| Braces | `{}` | `[]` |

### Rank Checking

The optional `<N>` annotation enforces rank at compile time:

```choreo
ituple<3> a = {1, 2};    // error: rank mismatch
```

### Operations

I-tuples support element access, derivation, concatenation, and element-wise arithmetic:

```choreo
a = {3, 4};
b = {a(0), 1, a(1)};     // element access with ()
c = a {(0), (1), 2};     // sugar form
d = {a, 5, 6};           // concatenation: {3, 4, 5, 6}
e = a + 1;               // element-wise: {4, 5}
```

I-tuples are immutable -- they must be initialized and cannot be reassigned.

### With mdspan

I-tuples commonly interact with shapes for tiling and padding:

```choreo
shape : [7, 18, 28];
tiling_factors = {1, 2, 4};
tiled_shape : shape / tiling_factors;     // [7, 9, 7]
padded_shape : shape + {2, 0, 2};        // [9, 18, 30]
```

Rank must match between operands:

```choreo
shape : [7, 8, 9] + {1, 2};  // error: rank 3 vs rank 2
```

## Evaluation

Both integers and i-tuples are evaluated at compile time whenever possible, incurring zero runtime cost. When they depend on dynamic values (e.g., symbolic dimensions passed as function parameters), evaluation is likely to happen at function entry in the generated host code.

## Bounded I-Tuples (Preview)

I-tuples become much more powerful when *bounded* -- associated with an mdspan that defines their range. Bounded i-tuples are essential for `chunkat` operations and `foreach` loops. They are created by `parallel-by` and `with-in` statements, covered in [Parallel-by](parallel-by.md) and [Iteration](iteration.md).
