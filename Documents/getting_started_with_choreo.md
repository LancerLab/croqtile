# Getting started with Choreo Programming

## Introduction
Choreo is a Domain Specific Language (DSL) designed for streamlining the manipulation of data transmission within accelerator hardware. It aims to simplify the daily tasks of engineers responsible for crafting high-performance kernels. This includes navigating the complexities of data transmission, such as tiling strategies and hardware capabilities. By leveraging Choreo, the programming process becomes more accessible, allowing engineers to optimize these critical processes with greater ease and efficiency.

## Choreo Function within C++ code
Choreo is designed as an extension of C++ code.

## Variables and Data Types
In Choreo, it introduces 3 basic types: integer-type, spanned-type, and integer-tuple-type (ituple-type). These types serve for different purposes.

- Integer Type. It is designed to fullfil the logic of program control. For example, the parallel-factor, the dimension upper bound are all integers.
- Spanned Type. This type represents the data for computation. In addition to the reference to the raw data, it also associate data with multi-dimensional ranges, which is useful for tiling purposes, etc. 
- Integer Tuple Type. This type serves as the indices for the multi-dimensional data.

### Integer Types
Integer types are 
```
int a;
int foo(int b);
```

### Spanned Types
A spanned type is a composite type. It consists of a fundamental type, and a multi-dimensional-span(mdspan) type.

#### Partial-typing: mdspan
Unlike most type systems, the mdspan type, which comes as an partial entity for typing, can be defined as a standalone. Furthermore, some computations are also defined upon such an incomplete type. Such a design is rarely seen, but can facilitate the work like tiling/blocking.

To define a mdspan, simply use '[' and ']' to enclose the integer dimension values. I.e.
```
mdspan sp : [7, 8];           // defined a mdspan of 2-dimensions
mdspan<4> mds : [a, 3, 4, 1]; // 'a' is an existing integer
mdsp : [c, 4, 28];
```

#### Fully Typing of Data
However, a mdspan can not be applied alone to define the data for computation. 

#### Memory Attributes
A Spanned data is usually large. For such large data, it could appears in different memory hierachy of accelerator hardware since choreo moves the data across different software managed memory. (It is known as scratchpad memory, SPM)

In choreo, we defines three memory attributes to annotate the data being defined, including: *global*, *shared*, and *local*. By default, when no memory attribute shows, the data defined is from *global* memory.

### I-Tuple Types
An integer tuple is an unordered set of integers. As described, it is normally used as a (subscription) index.

To define an i-tuple, simply enclose its elements within '{' and '}' braces. I.e.

```
ituple index = {5, 4, 3, 2, 1};  // It defines a tuple of 5 elements
index = {a, b};  // 'a' and 'b' are existing integers
```

## Control Structures


### The 'parallel-by' Block

### The 'with-in' and 'foreach' Block

## Function Calls
