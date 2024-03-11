# Getting Started with Choreo Programming

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

#### Partial-Typing: **mdspan**
Unlike most type systems, *mdspan*, though comes as an partial entity for typing, can be defined alone. Such design is based on the oberservation that loop tiling/blocking cares nothing but the shape of multiple dimension data. Therefore, Choreo make *mdspan* a partial-type to make the daily work easy. In addition, it also makes the *mdspan* a part of type to allow more type checking to happen, which is likely to reduce possiblity of code error as early (ahead of execution, sometimes) as possble.

To define a mdspan, simply use '[' and ']' to enclose the integer dimension values. I.e.
```
mdspan sp : [7, 8];           // defined a mdspan of 2-dimensions
mdspan<4> mds : [a, 3, 4, 1]; // 'a' is an existing integer
d : [c, 4, 28];               // 'd' is not explicit annotated.
```
As illustrated, a 'mdspan' can be explicitly defined with *mdspan* partial-type keyword anotated, optionally with a '<' '>' enclosed total dimension value, or even implicit defined.

When a mdspan is defined, it is possible to get the integer dimension value using operator '()' over the mdspan.
```
sp : [7, 8];
int b = sp(0) + sp(1);   // 'b' equals to 15 (7 + 8)
```

Conseqeuntly, you may derive a mdspan from existing one as the below code snippet:
```
sp : [6, 8];
spn : [1, sp(0)/2, sp(1)/4]; // define a new mdspan from the existing one.
```
This is very convinient to apply tiling in code. Considering that tiling operation is normally a must in constructing the high-performance kernels, Choreo has provided syntax suger to make the work even easier:
```
sp : [6, 8];
spn : sp [1, (0)/2, (1)/4];  // spn is defined as [1, 3, 2]
```
This code works the same way as the previous example, but in much simpler syntax.

**Note, a mdspan can only be defined. No modification to an existing mdspan is allowed. Further, a mdspan can only be defined once.**

In Choreo, it allows some arithmetic operations over mdspan. We will introduce such operations when more detail of *i-tuple* is revealed.


#### Fully-Typing to Define Data
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

### Sepcial Operations over *mdspan* and *i-tuple*
In Choreo, we allow special operation over *i-tuple* and *mdspan*. Below is an example to apply a fixed tiling over an mdspan:
```
sp : [6, 8];
tiling_factor = {3, 2};
spn : sp / tiling_factor;   // spn is defined as [2, 4];

```


## Control Structures


### The 'parallel-by' Block

### The 'with-in' and 'foreach' Block

## Function Calls
