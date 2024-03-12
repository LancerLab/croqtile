# Getting Started with Choreo Programming

## Introduction
Choreo is a Domain Specific Language (DSL) designed for streamlining the manipulation of data transmission within accelerator hardware. It aims to simplify the daily tasks of engineers responsible for crafting high-performance kernels. This includes navigating the complexities of data transmission, such as tiling strategies, which is subjected to the hardware capabilities. By leveraging Choreo, the programming process becomes more accessible, allowing engineers to optimize these critical processes with greater ease and efficiency.

## Embedding Choreo within C++
Choreo is the DSL code embedded within C++. The Choreo compiler performs the source-to-source translation of the Choreo function to be C++ code. And with the inclusion of "choreo.h", the translated code can work properly with other C++ code. The below code snippet showcases an example.

```
#include <choreo.h>   // Necessary for C++ code interaction
// some C++ code
__co__ void choreo_function() {
  // choreo code
}
// another C++ code
void foo() {
  choreo_function();
}
```
Note, a Choreo function is prefixed with "__co__" keyword. All code in the function scope is translated by the Choreo compiler. Other C++ code calls the Choreo function by its name as declared.

In addition, certain Choreo-specific types are introduced to ensure consistency between Choreo function arguments and their respective callers. The detail will be shown as soon as the Choreo data types are explained.

## Variables and Data Types
In Choreo, it introduces 3 type categories: integer-type, spanned-type, and integer-tuple-type (ituple-type). These types serve for different purposes.

- **Integer Type**. It is designed to fullfil the requirement of program control.  For instance, the factor of parallelization, and dimension value are all integers.
- **Spanned Type**. It represents the data type for computation. Apart from referencing the raw data, it also associates data with multi-dimensional ranges, which proves useful for tiling purposes, among others. 
- **Integer Tuple (I-Tuple) Type**. It represents a group of integer values. A common usage of *i-tuple* is to index multi-dimensional data.

### Integer Types
Integer types in Choreo is similar to C++ type 'int' or 'int32_t'. It is an signed value which takes 32-bits, ranging from -2^31 ~ 2^31 - 1. The below code illustrates its usage for defining the data and function declaration.
```
int a;
__co__ int foo(int b);
```
In Choreo, we neither provide equivalence of unsigned scalar integers, nor equivalence of 8-bits, 16-bits, 64-bits scalar integers. The reason is simple: these types are not necessary for program control purpose.

### Spanned Types
A spanned type is a composite type. It always consists of a fundamental type, and a multi-dimensional-span(mdspan) type. However, interestingly, in Choreo, the *mdspan* could be manipulated alone.

#### Partial-Typing: **mdspan**
Unlike most type systems, *mdspan*, though comes as an partial entity for typing, can be defined alone. Such design is based on the oberservation that loop tiling/blocking cares nothing but the shape of multiple dimension data. Therefore, Choreo allows programmers to manipulate *mdspan* regardless the fundanmental type it associated with, in order to make the daily work easy. Meanwhile, it still makes the *mdspan* a part of type to allow more type checking to happen. In this way, it expects to reveal code errors as early (ahead of execution, when applies) as possble.

To define a mdspan, simply use '[' and ']' to enclose the integer dimension values. I.e.
```
mdspan sp : [7, 8];           // defined a mdspan of 2-dimensions
mdspan<4> mds : [a, 3, 4, 1]; // 'a' is an existing integer
d : [c, 4, 28];               // 'd' is not explicit annotated. Type is deduced.
```
As illustrated, a 'mdspan' can be explicitly defined with *mdspan* partial-type keyword as annotated, optionally with a '<' '>' enclosed total dimension value. It is also possible to define a mdspan without type annotation. As shown in definition of 'd' above.

For C++ programmers, you may think a mdspan is the trait of a multiple dimensional array, where it described a multi-level-range. For example, 'sp' in the above example defines two-level of ranges, ranging from 0 to 6, and 0 to 7 in seperate.

When a mdspan is defined, it is possible to get the integer dimension value using operator '()' over the mdspan.
```
sp : [7, 8];
int b = sp(0) + sp(1);   // 'b' equals to 15 (7 + 8)
```
In the above example, the expression 'sp(0)' reasons about the first dimension value of mdspan 'sp'. The dimension values are in essence integer values, which are suitable for integer arithmetics.

Conseqeuntly, you may derive a mdspan from existing one as the below code snippet:
```
sp : [6, 8];
spn : [1, sp(0)/2, sp(1)/4]; // define a new mdspan from the existing one.
```
With such a facility, it is very convinient to apply tiling over multiple-dimensional spans in your Choreo code. Considering that tiling operation is always required in constructing the high-performance kernels, Choreo has provided syntax suger to make the work even easier:
```
sp : [6, 8];
spn : sp [1, (0)/2, (1)/4];  // spn is defined as [1, 3, 2]
```
This code works the same way as the previous one, but obviously in a much simpler syntax. This follows one of Choreo's design philosophy - To enable functionalities with minimal code whenever possible.

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
In Choreo, programmer can define a mdspan using such operations. The supported operations includes:

- *mdspan* / *i-tuple*
- *mdspan* + *i-tuple*
- *mdspan* % *i-tuple*
- *mdspan* * *i-tuple*
- *mdspan* - *i-tuple*

Note, most of the operations can be archieved with *mdspan* dimension-wise operations. However, the above operations could help programmer write more meaning full code.

## Control Structures
Choreo follows C++ to include 'if-else' statement to handle the branches. However, it has significant difference with C++ on parallelization, loop, etc.

### Parallel Region: the 'parallel-by' Block
In systems like CPU, it allows of asynchonized thread to realize the parallel execution. However, in Choreo, it employs the Single Instruction Multiple Data (SPMD) model as it way to perform parallelization. This is similar to some OpenMP parallel directive and OpenCL/CUDA. However, the syntax is different and more C-like. In Choreo, it encloses the code for parallel execution within the 'parallel-by' block.

```
parallel p by 6 {
  // SPMD code
}
```
The above code snippet illustrates the method to create a parallel region with Choreo keyword 'parallel' and 'by'. Here, we assume we have conceptually 6 processing elements. Each of the processing element execute the same SPMD code but with a different 'p' value. If you are familiar with programming CUDA, you may think 'p' is a equivalence of 'thread index'. Alternatively, if you are more familiar with sequential C/C++ programming, you may consider 'p' as an iteration variable of a loop of 6 iterations.

Note here is one implication, 'p' is an integer associated with its bound [0, 6). In Choreo, we call 'p' a **bounded integer** instead of a simple integer. In some special operations like 'chunkat', it requires the *bounded-integer* to work properly since the bound is essential for related computation.

### The 'with-in' and 'foreach' Block

### Special Async Operation: the DMA statement
Execept for parallel execution, Choreo allows 



## Function Calls
