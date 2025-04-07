## Overview
In Choreo, shape is a first-class citizen. In this section, you will learn how to program shapes in Choreo code.

## First-Class Citizen: Shape
Choreo's primary function is to manage data movements, which are crucial for efficiently organizing and processing large datasets, especially in machine learning and high-performance computing scenarios. However, most C++ programming environments handle data in a casual manner —either as flat (pointers) or hierarchical structures (arrays), without a native representation of associated shapes.

In contrast, Choreo enforces code safety and simplifies the programming of shaped data by requiring that **any data declared or used must be associated with a shape**. This motivates Choreo to treat shape as a first-class citizen.


## Defining Shapes with `mdspan`

In Choreo, shapes can be defined with the `mdspan` keyword, which stands for **Multi-Dimensional Span**. This keyword represents multi-dimensional data for computations.

You can explicitly define `mdspan` variables as follows:

```choreo
mdspan s0 : [7, 8]; // Defines a 2D shape with dimensions [7, 8]
mdspan<1> s1 : [3]; // Defines a 1D shape with dimensions [3]
```

In this example, the leading keyword `mdspan` indicates the declaration of a *mdspan* variable, followed by the user-provided variable name. Each *mdspan* variable is initialized with an **initialization expression**, which consists of comma-separated integer values enclosed by `[]`. The symbol `:`, which immediately follows the variable name, introduces the *initialization expression*.
Therefore, `s0` is defined as a 2D shape with `7` rows and `8` columns, while `s1` is a 1D shape with a dimension of `3`.

It is possible to optionally specify the **rank** of an *mdspan* variable by placing `<>` after the `mdspan` keyword. If the rank is explicitly specified, it informs the Choreo compiler to check for rank consistency. If the rank value differs from the corresponding *initialization expression*, it triggers a failure at compilation. Here is an example:

```choreo
mdspan<3> s2 : [64, 32]; // error: the rank of mdspan is inconsistent
```

In addition to explicit mdspan declarations, the Choreo compiler can infer the type of `mdspan` variable from its *initialization expression*, eliminating the need for the explicit `mdspan` keyword:

```choreo
s3 : [7, 8, 9]
```

In this code, `s3` is an `mdspan` of rank `3` with dimensions `7, 8, 9`. Since Choreo **requires mdspan to always be initialized within declarations**, the type inference version is preferred in programming practice.

## Deriving *mdspan*s
In practical coding, it is common to derive a new shape from an existing one. For example, you might want to perform data **tiling** or **blocking**, which requires dividing the dimensions of a shape. Alternatively, you might want to **pad** specific dimensions, which involves adding delta to the shape dimensions.

In Choreo, such shape derivations can be easily accomplished using arithmetic operations on *mdspan*. The following code showcases an example:

```choreo
shape : [128, 64]; // initial shape 
new-shape0 : shape [(0) / 2, (1) / 4, 1];  // tile and reshape: [1, 64, 16]
new-shape1 : shape [(1) + 2, (0) / 16];    // pad and reshape: [66, 8]
```

In this example, the `new-shape0` is derived from `shape`, with dimension 0 divided by `2`, dimension 1 divided by `4`. This corresponds to *tiling* operation in high-level semantics. Additionally, the code adds a new dimension to `new-shape`. In high-level semantics, this operation is often referred to as *reshaping*.

In Choreo, the definition of `new-shape0` is equivalent to:

```
new-shape0: [shape(0) / 2, shape(1) / 4, 1];
```

Here, the initial `shape` is explicitly listed element-wise rather than specified outside `[]`. But similar to the prior version, The **element-of** operation, which is annotated as `()`, is used on top of existing shape to retrieve dimension values. Obviously, this approach requires more code but yields the same result. Thus, the prior version can be considered *syntactic sugar* for the complete *initialization expression* of the new shape.

In the code example, `new-shape1` is also derived from `shape`, it pads dimension 1 by `2` and swaps the dimensions in the derived shape.

Furthermore, you may use the `mdspan` as a whole for derivations:

```choreo
shape : [32, 72]
new-shape0 : shape;  // [32, 72]
new-shape1 : shape + 1;  // [33, 73]
new-shape2 : shape / 4;  // [8, 18]
new-shape3 : [shape, 6]; // [32, 72, 6]
```

Note that arithmetic operations on an *mdspan* variable are applied dimensionally. Thus, the statement

`new-shape1 : shape + 1;`

is equivalent to

`new-shape1 : shape [(0) + 1, (1) + 1];`

The derived definition of `new-shape3` demonstrates that using an *mdspan* variable in an *mdspan initialization expression* results in **concatenation** behavior. Thus, the declaration

`new-shape3 : [shape, 6];`

is equivalent to

`new-shape3 : shape [(0), (1), 6];`

The prior version can be as well deemed as a *syntactical sugar* of a complete definition.

## Evaluation of *mdspan*
So far, we have seen mdspan with constant values, which is sufficient for many scenarios, as high-performance device kernels often require fine-tuning based on fixed input data shapes. In these cases, the `mdspan`s are evaluated at compile-time, meaning their values do not incur extra execution time or storage overhead.

However, in some scenarios, a runtime shape (where some dimensions are determined at execution) is required for building the kernel. Choreo supports this with **Symbolic Dimensions** for `mdspan`, which will be introduced later. In such scenarios, runtime evaluation of dimension values may be necessary. Fortunately, Choreo manages this evaluation immediately after entering a choreo function in the Choreo-generated host code, resulting in negligible startup cost. **Therefore, programmers can ignore overheads related to _mdspan_ normally**.
