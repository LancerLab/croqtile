# Getting Started with Choreo Programming

## Introduction
Choreo is an Embedded Domain Specific Language (eDSL) designed for streamlining the manipulation of data transmission within accelerator hardware. It aims to simplify the daily tasks of engineers responsible for crafting high-performance kernels. This includes navigating the complexities of data transmission, such as tiling strategies, which is subjected to the hardware capabilities. By leveraging Choreo, the programming process becomes more accessible, allowing engineers to optimize these critical processes with greater ease and efficiency.

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
Note, a Choreo function is prefixed with the "\__co\__" keyword. All code in the function scope is translated by the Choreo compiler. Other C++ code calls the Choreo function by its name as declared.

In addition, certain Choreo-specific types are introduced to ensure consistency between Choreo function arguments and their respective callers. Considering that there are some important programming elements missing, we will defer the introduction of such details until the data types are explained.

## Variables and Types
Choreo introduces 3 type categories: integer-type, spanned-type, and integer-tuple-type (ituple-type). These types serve for different purposes.

- **Scalar Type**. It is designed to fullfil the requirement of program control. It consist of *Integer Type* and *Boolean Type*.
- **Spanned Type**. It represents the data type (normally the tensor) for computation. Apart from referencing the raw data, a *spanned type* also associates data with a shape representing by multi-dimensional ranges. The 'multi-dimensional ranges' is named 'mdspan'. It is useful for tiling, etc, which is introduced soon.
- **Integer Tuple (I-Tuple) Type**. It represents a group of integer values. A common usage of *i-tuple* is to index multi-dimensional data.
- **Bounded (Integer/ITuple) Type**. This is the special type that is used to simplify data (sub-zone) reference.

Among the four type categories, *scalar* and *ituple* could be well accepted consider it maps to elements of existing general purpose programming languages. However, the *spanned type*, and *bounded type* are specific to the domain. The following sections will show the detail.

### Scalar Types
As described, Scalar types in Choreo includes the *Integer Type* and *Boolean Type*. The *Integer Type* is similar to C++ type 'int' or 'int32_t'. It is an signed value which takes 32-bits, ranging from -2^31 ~ 2^31 - 1. The below code illustrates its usage for defining the data and function declaration.
```
int a;
__co__ int foo(int b);
```
Operations like integer arithmetics, shifting are all supported. And the syntax is indential to C++ builtin operations.
In Choreo, we neither provide equivalence of unsigned scalar integers, nor equivalence of 8-bits, 16-bits, 64-bits scalar integers. The reason is simple: these types are not essential for program control purposes. And normally a 32-bits signed integer is enough for such work.

And the *Boolean Type* in Choreo is similar to C++ type 'bool'. The operations on top of Boolean, and conversion between Integers, are identical to these of C++.

### Spanned Types
A spanned type is a **Composite Type**. It consists of a **fundamental type**, and a **multi-dimensional-span(mdspan) type**. Neither the *fundamental type* nor the *mdspan* is a *complete type*. It implies that neither of them can type a data alone. In Choreo, we name them as *partial types*.

The reason for making the type *partial* is to specifically manipulate *mdspan* alone. This is crucial in AI scenarios where handling multi-dimensional data is common. In the following sections, we will demonstrate how to define such partial types and how to compose partial types into a complete type for declaring/defining multi-dimensional data purpose.

#### The Partial Type: **mdspan**
Unlike most type systems, *mdspan*, comes as an partial entity for typing. Here we claims it as *partial* because Choreo program is unable to define data as a *mdspan* type. However, *mdspan* itself can be defined solely. Such design is based on the oberservation that loop tiling/blocking cares nothing but the shape of multiple dimension data. Therefore, Choreo allows programmers to manipulate *mdspan* regardless the fundanmental type it associated with, in order to make data shape manipulation easy. Meanwhile, as *mdspan* is a part of a full type, the compiler could apply type checking with it. In this way, it expects to reveal code errors as early (ahead of execution, when applies) as possble.

To define a mdspan, simply use '[' and ']' to enclose the integer dimension values. I.e.
```
mdspan sp : [7, 8];           // defined a mdspan of 2-dimensions
mdspan<4> mds : [a, 3, 4, 1]; // 'a' is an existing integer
d : [c, 4, 28];               // 'd' is not explicit annotated. Type is deduced.
```
As illustrated, a 'mdspan' can be explicitly defined with *mdspan* partial-type keyword as annotated, optionally with a '<' '>' enclosed total dimension value. It is also possible to define a mdspan without type annotation. As shown in definition of 'd' above. (Note: 'mds' and 'd' are of the **dependent type**, considering the type depends on the evaluation of 'a' and 'b'. Programers may provides 'a' and 'b' with runtime values. However, in this way, some runtime check are paid consequently)

For C++ programmers, you may think a mdspan as the trait of a multiple dimensional array, where it described a multi-level-range. For example, 'sp' in the above example defines two-level of ranges, ranging from 0 to 6, and 0 to 7 in seperate.

When a mdspan is defined, it is possible to get the integer dimension value using operator '()' over the mdspan.
```
sp : [7, 8];
int b = sp(0) + sp(1);   // 'b' equals to 15 (7 + 8)
```
In the above example, the expression 'sp(0)' reasons about the first dimension value of mdspan 'sp'. The dimension values are in essence integer values, which are suitable for integer arithmetics.

As a conseqeunce, you may derive a mdspan from existing one as the below code snippet:
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

Except for defining a mdspan in the above *dimension-wise* way, Choreo also support to define mdspan definition with other methods. We will introduce such methods when *i-tuple* is revealed.

#### Fully-Typing
A *mdspan* can not be applied alone to define the data for computation. In Choreo function, a data definition must be fully-typed, which consists of a fundamental type and a *mdspan*. The below code showcases how it works.
```
ndims : [20, 15];
f32 [10, 10] d0;
f16 [ndims] d1;
```
There are fundamental types that Choreo has supported, including:

- Unsigned Integers: *u8/u16/u32*
- Signed Integers: *s8/s16/s32*
- Floating-points: *f16/bf16/f32*

Note 's32' and 'int' are different in Choreo. 's32' is a fundamental type, which can not be applied for a fully typing.

#### The Storage Qualifier
A spanned-typed data in Choreo is usually large. For such large data, programmers could move it across different memory hierachy of accelerator to best utilize hardware resource.

In choreo, we defines three storage qualifiers to annotate the data being defined, including:

- **global**,
- **shared**,
- and **local**.

The below code showcases the usage.
```
ndims : [20, 15];
local f32 [10, 10] d0;
shared f16 [ndims] d1;
```

By default, when no storage qualifier appears, the data defined is considered as from the *global* memory.

### I-Tuple Types
An integer tuple is an unordered set of integers. As described, it is normally used as a (subscription) index.

To define an i-tuple, simply enclose its elements within '{' and '}' braces. I.e.

```
ituple index = {5, 4, 3, 2, 1};  // It defines a tuple of 5 elements
index = {a, b};  // 'a' and 'b' are existing integers
```

### Operations over *mdspan* and *i-tuple*
In Choreo, we allow special operation over *i-tuple* and *mdspan*. Below is an example to apply a fixed tiling over an mdspan:
```
sp : [6, 8];
tiling_factor = {3, 2};
spn : sp / tiling_factor;   // spn is defined as [2, 4];

```
In Choreo, mdspan can be defined with such **Tuple-Span Operations". The supported operations includes:

- *mdspan* / *i-tuple*
- *mdspan* + *i-tuple*
- *mdspan* % *i-tuple*
- *mdspan* * *i-tuple*
- *mdspan* - *i-tuple*

Essentially, these operations can be achieved through mdspan *dimension-wise* definition. However, *tuple-span operations* aid programmers in writing more readable code. This is also the objective that Choreo aims to achieve.

### Bounded Types
Bounded types consists of **Bounded Scalar** and **Bounded ITuple**. Bounded Scalar takes a range of [0, ub], where is the 'ub' represents its upper bound. Therefore, if an integer 'p' is set as bounded, it should also be associated with a specific upper bound. To establish such associations, programmers must code explicitly inside the *Control Structures* of 'parallel-by' and 'with-in', which will be introduced later.

Similarly, since *ITuple* is a group of *Integer*s, it can also be associated with a group of bounds. Specifically, in Choreo, the *Bounded ITuple* is associated with a *mdspan* value, where a group of upper bounds are settled. In later sections, we shall illustrate the detailed syntax.

## Control Structures
Choreo follows C++ to involve 'if-else' blocks to handle branches inside programs. However, it has significant difference with C++ on parallelization, loop, etc.

### Parallel Region: the 'parallel-by' Block
In systems like CPU, it allows of asynchonized thread to realize the parallel execution. However, in Choreo, it employs the Single Instruction Multiple Data (SPMD) model as it way to realize parallelization. This is similar to some OpenMP parallel directive, and some parallel programming language like OpenCL/CUDA.

However, the syntax of constructing a parallel region is quite different. It employs the C-style bracket and encloses the code for parallel execution within the 'parallel-by' block.

```
parallel p by 6 {
  // SPMD code
}
```

The above code snippet illustrates the method to create a parallel region with Choreo keyword 'parallel' and 'by'. Here, it assume there are 6 execution threads. Each of the thread execute the same SPMD code inside but with a different 'p' value. If you are familiar with programming CUDA, you may think 'p' is a equivalence of 'thread index'. Alternatively, if you are more familiar with sequential C/C++ programming, you may consider 'p' as the iteration variable of a loop with 6 iterations. (But any iteration may go first to be executed!)

Despite parallelism, there is one more implication of 'parallel-by'. In the statement, 'p' is an integer associated with its bound [0, 6). We name 'p' as a **bounded integer** instead of a simple integer. In some special operations like 'chunkat' (explain later), it requires the *bounded-integer* to work properly since the bound is essential for its computation.

### The 'with-in' Block and 'requires' Clause
Similar to 'parallel-by', 'with-in' statement can also bind *i-tuples* to a *mdspan*. The below code shows an example.
```
with index in [10, 10] {
  // index is ituple with 2 elements
}
```
Here, 'index' is a *i-tuple* with 2 elements. Sometimes programmers perfer that the 2 elements being named. This is possible by using the below syntax.
```
with {x, y} in [10, 10] {...}
```
Or even to name both the *i-tuple* and its elements.
```
with index = {x, y} in [10, 10] {...}
```
We name 'index' as a **bounded ituple** in such scenarios.

You may think 'with-in' statement is similar to 'parallel-by'. However, it is not true. One significant difference is 'with-in' statement does not have implication for parallelism. The code block inside 'with-in' statement is sequentially executed. It does nothing more than creating the *bounded-ituple*.

Neverthless, programmers could append a 'requires' clause. For example,
```
with {m, n} in [M, N], {n_p, k} in [N_P, K] requires n_p <-> n {
  // matmul implements with m,n,K. n_p is no long useful.
}
```
The code snippet requires 'n' and 'n_p' to have an identical value in all iterations. Thus inside the 'with-in' block, it is possible to replace'n' whenever 'n_p' is required, or the opposite. Such a facility is useful to program many AI kernels. Programmers should use operation '<->' to establishs such relations.

### The 'foreach' Block
Once the *bounded-ituple* is defined by the 'with-in' clause, programmers can loop over the bounded-ituples/bounded-integers. In Choreo, this is simple.

```
with x in [10] {
  foreach x {
    // do something with each x
  }
}
```

### The 'inbound' Operation

### Async Operation: the DMA Statement
Execept for parallel execution, Choreo allows one fixed form of async operation: the DMA statement.

Conceptually, a DMA statement is executed asynchronously with the SPMD code. It works quite similar to CPU async thread, except its behavior is limited by the DMA configuration. (CPU allows to program the async thread as will)

The below code showcases one basic DMA statement.
```
global f32 [10] g_data;
local f32 [10] l_data;
f = dte.linear g_data => l_data;
// ... async operations
wait f;         // explicit wait
```
Here, we utilize the data transfer engine (DTE) to invoke a linear copy, which moves the 'data' residing in global memory to 'data1' in local memory. You might observe the assignment of 'f'. This is the handler for the asynchronous DMA entity. In Choreo, we refer to it as the 'future' of the DMA operation. Programmers have the option to code the explicit synchronization statement 'wait' to pause the current thread until the 'future' arrives. Otherwise, the thread continues execution in parallel with the DMA operation.

At times, programmers may find it tedious to explicitly define temporal data. Choreo offers an even simpler syntax:
```
global f32 [10] data;
f = dte.linear data => local;
wait f;
... f.data;  // retrieve the 'local' data from the future
```
Here, we do not need to specify the exact target location where the DMA transfers data. Instead, we only specify the destination memory type. This is advantageous in many scenarios. Programmers often prefer to avoid dealing with scratchpad memory (SPM) management. In this code, the local memory allocation is left to the compiler. And to retrieve the transferred data, we simply invoke the 'data' member function of 'future'.

DMA operations entail intricate details that demand careful programming. Programmers should refer to the DMA manual to make informed decisions for their code. Nonetheless, Choreo compiler provides plenty of static and runtime checks to assist programmers in avoiding potential errors in this aspect.

### Bounded-ituple/integer and 'chunkat' Opertion
'chunkat' is an operation performed on spanned data. It creates a new *mdspan* over the existing data. Thus in certain systems, it is referred to as 'subview'. However, as 'chunkat' accepts bounded-ituple and bounded-integer as parameters, it is named differently in Choreo.
```
global f32 [6, 10, 100] data;
parallel p by 6 {
  with index = [x, y] in [10, 10] {
    // for every data move, the stride into 'data' is
    //    stride = p*1000 + x * 100 + y * 10
    //
    // for each chunk, the dimensioned size for the movement is {1, 1, 10}
    f = dte.linear data.chunkat(p, index) => local;
  }
}
```
The above example showcases one typical usage of 'chunkat'. Here we have a spanned data with its type is 'f32 [6, 10, 100]'. The chuckat operation receive two parameter, *integer* 'p' and *ituple* 'index', it assumes to divide the data into 6 * 10 * 10 pieces, which is the size of associated ranges relating to 'p' and 'index'. Then 1 piece is fetched for transference to the local data. That is 10 elements of data, which is consecutive considering it is the quotient is 10 for the least significant dimension.


## Function Calls and Call Choreo Function
Choreo functions are not allowed to call another Choreo function. However, inside a Choreo function, it is normal to have function calls to C++ kernels.

```
void bar() {...}    // C++ kernel function
__co__ void foo() {
  parallel p by 6 {
    call bar();     // Call the C++ function
  }
}

```
In the above example, program calls the existing C++ function 'bar' using Choreo keyword 'call', which is intuitive.

## Parameters Passing between Choreo and C++ Function
Passing arguments to Choreo or opposite requires inclusion of Choreo header file: choreo.h. Normally, programmers combine the raw C++ pointer and associated dimensions info to construct Choreo spanned data. The below code domenstrate how it works.

```
#include "choreo.h"

void bar(const float* data, unsigned size) {}

__co__ void foo(f32 mdspan<2> d) {
  parallel p by 6 {
    call bar(d.data, |d|);     // Call the C++ function
  }
}

void foobar(float* a) {
  foo(choreo::make_spanned<2>(a, {1, 2}));
}
```
In the example, we make use of choreo utility function (template) 'make_spanned' to make the spanned data. And in Choreo function 'foo', the parameter 'f32 mdspan<2>' is the correspondance entity. To get the raw pointer of the spanned data, simply use 'data' operation over the spanned parameter. And operation '|d|' obtains the total size of the spanned data 'd'. These are used for calling C++ function 'bar'.

Similarly, *Scalar Type* data can also be passed from/to Choreo function. Neverthless, ituple is only used inside Choreo function.

## Summary
Choreo introduces a novel approach to SPMD programming. It favors C++-style coding and is embedded within C++. However, its primary focus is on alleviating the burden of low-level programming details, particularly those related to data manipulation across various memory layers through DMA operations. At times, it is also referred to as the dataflow programming DSL. We developed this tool to support the daily task of constructing high-performance kernels. Our aim is to enable programmers to focus less on the intricacies of language construction and more on higher-level conceptual thinking.

We wish you find it functions as expected. And we are looking forward to any feedback for continuous improvement.
