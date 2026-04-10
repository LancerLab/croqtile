# Shapes and mdspan

## Shape as First-Class Citizen

Croqtile's primary function is to manage data movement across memory hierarchies. In most C++ environments, data is handled as flat pointers or fixed arrays without native shape representation. Croqtile takes a different approach: **every piece of data must be associated with a shape**. This makes shape a first-class entity that can be defined, derived, passed, and checked independently of the data it describes.

## Defining Shapes

An `mdspan` (multi-dimensional span) defines a shape. Each dimension specifies an upper bound:

```choreo
mdspan s0 : [7, 8];       // 2D: 7 rows, 8 columns
mdspan<1> s1 : [3];        // 1D: explicit rank annotation
s2 : [64, 32, 16];         // 3D: rank inferred from initializer
```

The syntax elements:

- The `mdspan` keyword is optional when the compiler can infer the type from the `:` initializer.
- The optional `<N>` rank annotation triggers a compile-time rank consistency check.
- Dimensions are comma-separated inside `[]`.
- The `:` symbol introduces the initialization expression (distinguishing mdspan from ituple, which uses `=`).

```choreo
mdspan<3> bad : [64, 32];  // error: rank mismatch (declared 3, got 2)
```

### Rank

The **rank** of an mdspan is its number of dimensions. A rank-2 mdspan `[7, 8]` represents a 2D region with bounds `[0,7) x [0,8)`.

## Accessing Dimensions

The element-of operator `()` retrieves individual dimension values as integers:

```choreo
sp : [7, 8];
int total = sp(0) + sp(1);   // 15
int first = sp(0);            // 7
```

The `.span` member on spanned data returns the associated mdspan:

```choreo
__co__ auto foo(f32 [16, 17, 5] input) {
  f32 [input.span] buffer;           // same shape as input
  int dim0 = input.span(0);          // 16
}
```

## Deriving New Shapes

New shapes are commonly derived from existing ones through arithmetic and concatenation. This is essential for tiling, padding, and reshaping operations.

### Dimension-Wise Derivation (Sugar Form)

```choreo
shape : [128, 64];
tiled : shape [(0) / 2, (1) / 4, 1];     // [64, 16, 1]
padded : shape [(1) + 2, (0) / 16];       // [66, 8]
```

Inside `[]`, the notation `(i)` refers to dimension `i` of the base shape. This is syntactic sugar for the explicit form:

```choreo
tiled : [shape(0) / 2, shape(1) / 4, 1]; // equivalent
```

### Whole-Shape Arithmetic

Arithmetic operators apply dimension-wise to the entire mdspan:

```choreo
shape : [32, 72];
same : shape;           // [32, 72]
grown : shape + 1;      // [33, 73]
halved : shape / 4;     // [8, 18]
```

`shape + 1` is equivalent to `shape [(0) + 1, (1) + 1]`.

### Concatenation

Placing an mdspan inside another's initializer concatenates dimensions:

```choreo
shape : [32, 72];
extended : [shape, 6];           // [32, 72, 6]
prefix : [1, shape];             // [1, 32, 72]
```

### I-Tuple Operations

Shapes and i-tuples interact through arithmetic. The operands must have matching rank:

```choreo
shape : [128, 64, 32];
factors = {4, 2, 8};
tiled : shape / factors;          // [32, 32, 4]
padded : shape + {2, 0, 4};       // [130, 64, 36]
```

```choreo
shape : [7, 8, 9] + {1, 2};      // error: rank mismatch (3 vs 2)
```

Supported mdspan-ituple operations: `/`, `+`, `%`, `*`, `-`.

## The ElementCount Operator

The `|mdspan|` operator returns the total number of elements (product of all dimensions):

```choreo
sp : [6, 10, 100];
int count = |sp|;           // 6000
call kernel(f.data, |f.span|);
```

## Evaluation Semantics

Shapes with constant dimensions are evaluated at **compile time** with zero runtime overhead. When symbolic or dynamic dimensions are involved (see [Symbolic and Dynamic Shapes](symbolic-and-dynamic.md)), dimension values are likely evaluated at function entry in the generated host code.

## Immutability

An mdspan variable must be initialized at its declaration and **cannot be reassigned or modified**. This enables the compiler to track shapes through the entire program for inference and checking.
