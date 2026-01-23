## Overview
Choreo *tileflow program* describes how to move data around. Therefore, declaring or defining the data or buffer are fundamental. In this section, you will learn related syntax.

## *Spanned*: Data and Buffers
The primary focus of a Tileflow program is to manipulate large datasets by moving them around. In some terminology, the input and output of a Choreo function are classified as "input data" and "output data," respectively. Meanwhile, any other storage is referred to as "buffers." This conceptually distinguishes between "external" and "internal" memory from a function's perspective. However, both "data" and "buffer" refer to storage positions. In Choreo, both are typed as **spanned**, reflecting that they are data/buffers associated with an
*mdspan*-represented shape. And in the following context, we call it *spanned data*, or simply *spanned*.

## Defining a *Spanned Data*
Defining a *spanned data* requires both the element type and its shape. For example,

```
f32 [32, 16] data;
```

This defines a buffer with a shape of `[32, 16]`, where the element type is `f32`, which is the *IEEE-754* single-precision float. It is also possible to use a named `mdspan` to define the buffer:

```
shape : [72, 14];
s32 [shape] data0;              // using the named mdspan `shape`
u32 [shape(0), shape(1)] data1; // same shape as the above
u16 [shape + 6] data2;          // shaped as [78, 20]
s16 [shape, 8] data3;           // shaped as [72, 14, 8]
```

Here we used the named *mdspan* `shape` to define a 2D buffer named `data0`, where the elements are 32-bit integers. Note that the named *mdspan* `shape` is placed inside `[]`. Additionally, it is allowed to use *element-of*, arithmetics, or concatenation on *mdspan*s to specify the buffer's shape.

In Choreo terminology, the element type is referred to as the **fundamental type**. Choreo supports multiple fundamental types, including:

- **Unsigned Integers**: `u8`, `u16`, `u32`
- **Signed Integers**: `s8`, `s16`, `s32`
- **Floating Points**: `f16`, `bf16`, `f32`

The prefixes `u` and `s` stand for "unsigned" and "signed", respectively, while the number suffix indicates the bit width of the type. `f16` refers to the *IEEE-754* standard *half-float*, and `bf16` refers to the 16-bit *binary float*, commonly used in machine learning scenario.

Note `s32` is different from `int`. Although they occupy the same amount of storage, in Choreo, `s32` can not be used alone to define programming entities. For example:

```choreo
s32 a;  // error: the fundamental type can not be used for variable definition alone
```
This results in a compile-time failure. Consequently, `s32` cannot be used for program control like `int`, and type conversion between them is not possible.

## Storage Specifier
In practical code, some buffer declarations must specify a **storage specifier**, like so:

```choreo
global f32 [32, 7, 2] a;
shared u8 [512, 144] b;
local u8 [72, 1024] c;
```

Since Choreo handles storage in a heterogenous context, a buffer definition without a storage specifier defaults to the storage type of the host program, i.e., CPU memory. Other storage specifiers are defined by the target. For example, *Cuda/Cute* (for GPU hardware) supports:

- *global*: Refers to the device's global storage.
- *shared*: Refers to the device block's shared storage
- *local*: Refers to the thread-private storage.

Other targets may have different definitions. Furthermore, buffers with different storage types have limitations in their declarations. We will explore this in later chapters.

## Initialization
It is possible to initialize a buffer at the declaration site. For example:

```choreo
local s32 [17, 6] b1 {0};       // elements are initialized to 0
shared f32 [128, 16] b2 {3.14f}; // elements are initialized to 3.14f
```

The syntax for the *initialization expression* of a *spanned data* is straightforward: it encloses an initial value inside brackets following the variable name. However, its functionality is limited - it always set all elements to a fixed value.

## Declaring the Parameters and Return Values
A *spanned data* is passed between host and tileflow programs. Therefore, a choreo function can have *spanned data*s as its parameters. The following code showcases an example:

```choreo
__co__ f16 [7, 8] foo(f32 [16, 17, 5] input) {...}
```

The syntax is similar to variable definitions, except that operations on an existing *mdspan* is not possible. Additionally, no storage specifier is allowed or initialization is allowed.

One useful built-in member function for the *spanned* parameters is `.span`, which provides the associated *mdspan* of the *spanned data*. The following code demonstrates how to use it for a buffer declaration:

```choreo
__co__ auto foo(f32 [16, 17, 5] input) {
  f32 [input.span / {4, 1, 5} ] buffer;  // declare a buffer with tiled shape
  // ...
}
```

## Buffer Lifetime Management
The lifetime of storage allocated inside a *choreo function* is managed by the Choreo compiler for efficient use. The compiler attempts to reuse buffers as much as possible if their lifetimes do not overlap. Therefore users of Choreo are not necessary to do buffer management.

## Quick Summary
This section covered defining and managing data/buffers in Choreo tileflow programs, including declaration syntax, initialization, storage specifiers, and efficient buffer lifetime management by the compiler. This enables us to step further to the next essential topic: the dynamic shape support.
