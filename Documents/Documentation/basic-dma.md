## Overview
The data movement is abstracted as the most complicated statement in Choreo. This section will introduce you  the basic *DMA statement* structure and the future variable they produce.

## Data Movement Statement

Choreo aims to address multiple levels of data movement. Currently, the primary focus is on data movement between heterogeneous hardware and across memory hierarchies. In hardware terminology, such movements are known as **Direct Memory Access (DMA)**. Choreo adopts this terminology and abstracts data movement as **DMA Operations**.

### Basic Syntax

A DMA statement in Choreo can be an asynchronous entity, meaning that the code following an **asynchronous DMA** executes in parallel with the DMA statement. For a *asynchronous DMA*, explicit synchronization is required before using the DMA result.

To support these features, the syntax is organized as follows:

```choreo
future = dma-op src-expr => dst-expr;
```

The statement defines a **future**-typed variable, which appears on the left side of the operator `=`. In Choreo, **a *future* represents both the handle of the asynchronous execution instance and the DMA result**. Therefore, the definition of *future* can be ignored for *synchronized DMA* statements.

The right-hand side of the **DMA operation** includes the **operation type** (`dma-op`), the **data expression** for the operation source (`src-expr`), and the *data expression* for the destination (`dst-expr`). The source and destination are separated by the symbol `=>`, which indicates the direction of data flow.

### Synchronous and Asynchronous

**Asynchronous DMA**, or **Async-DMA**, is also known as **Non-Blocking DMA** because it does not block the execution of subsequent code. Conversely, **Synchronous DMA**, or **Sync-DMA**, is referred to as **Blocking DMA** because the code does not proceed until the DMA operation is complete. These terms will be used interchangeably in the following text.

In Choreo, the `.async` suffix in a **DMA operation** indicates that the operation is asynchronous. The following code provides an example:

```choreo
f0 = dma.copy.async data0 => shared; // non-blocking
f1 = dma.copy data1 => shared;       // blocking
...
wait f0;                             // synchronization
```

In this example, the DMA operation that produces the future `f0` is asynchronous, allowing the program to continue executing while the data transfer occurs in the background. The DMA operation that produces `f1` is synchronous, meaning the program will pause until the data is fully transferred from `data1` to `shared` memory. Meanwhile, the data copy of `data0` might still be in progress.

Executing the statement `wait f0;` causes the program to check if `data0` has been fully transferred to `shared` memory. This synchronizes the main execution flow with the asynchronous DMA of `f0`. Note, if the *Async-DMA* is not waited, it typically results in unexpected hardware issues. Therefore programmers must make sure there is no "dangling future* left in your code.

### Operation Type

Choreo's DMA statement is an abstraction designed to support modern hardware. Beyond simple linear memory copies, advanced hardware such as the **Tensor Memory Accelerator (TMA)** can transfer shaped data and apply shape transformations in-flight. Choreo maps these functionalities at the software level as different **DMA Operation Types**. The supported operations include:

- `dma.copy`: Copies flat memory directly.
- `dma.pad`: Pads shaped data while transferring the data.
- `dma.transp`: Transposes shaped data while transferring the data.

The configurations, such as padding and transposing details, are programmed as parameters of the DMA operation. For example:

```choreo
global f32 [32, 16, 9] input;
dma.transp<0, 2, 1> input => shared;  // Result shape [32, 9, 16]
dma.pad<{1, 0, 3}, {0, 1, 2}, {0, 0, 0}, 0.1f> input => shared; // Result shape [33, 17, 14]
```

Here, the **DMA configuration**s are enclosed by `<>`. The configuration varies according to different operations. The detailed configuration syntax and limitations for a specific platform, such as *CUDA/Cute*, are listed below as an example:

Programmers should note that support for DMA types other than `dma.copy` varies by platform. For instance, on the *CUDA/Cute* platform, it could be either a mapping of TMA or an orchestration of load instructions from multiple threads. It is also possible to implement advanced DMA operations using software-only methods or software-hardware cooperation, though these are not yet supported.

### Data Expression

So far, we have only used simple data expressions, either a defined **spanned data** or a **storage location**. The **storage location** is used solely for the **destination buffer** declaration, as it requires the Choreo compiler to allocate memory for it. The following code provides an example:

```choreo
global f32 [32, 16] input;
shared f32 [32, 16] output;
dma.copy input => output;  // Copy to a user-declared buffer
dma.copy input => shared;  // Same, but to a compiler-allocated buffer
```

In this example, the destination of the second DMA operation is specified as the storage location `shared`. This requires the compiler to allocate storage accordingly, making it equivalent to the first DMA operation. In practice, we recommend programmers use storage locations as destinations, as Choreo can deduce the destination shape from the **DMA statement**.

However, the use of simple **data expressions** is rare in practice. In the next section, we will introduce the **ChunkAt** expression to demonstrate how to implement tiling in data expressions.

### Future and Wait

Both *Sync-DMA* and *Async-DMA* operations can be used to define a *future*. As seen in earlier examples, a *future* acts as a handle for asynchronous operations and is typically used as a parameter of the `wait` statement to synchronize an *Async-DMA*.

Choreo allows the `wait` statement to take multiple *future*s. However, using a *future* from a **Sync-DMA** in a `wait` statement will result in an error. The following code provides an example:

```choreo
f0 = dma.copy.async input0 => shared;
f1 = dma.copy.async input1 => shared;
f2 = dma.copy input2 => shared;
...
wait f0, f1;  // Multiple wait
wait f2;      // Error: cannot wait on a sync-dma
```

In Choreo, another use of a *future* is to retrieve the destination buffer of the associated DMA statement. This is also the reason why **Sync-DMA** is allowed to define a future. Choreo provides two built-in member functions for *future* variables:

- `.span` to retrieve the **mdspan** of the DMA destination buffer.
- `.data` to retrieve the reference of the DMA destination buffer (spanned data).

For example:

```choreo
f0 = dma.copy.async input0 => shared;
f1 = dma.copy input2 => shared;
...
local f32 [f0.span] buffer;
call device_kernel(f1.data, |f1.span|);
```

In this example, we use `.span` built-in member function to retrieve the shape of `f0`'s destination buffer, and `.data` to retrieve the *spanned* data of `f1`'s destination. In addition, the **ElementCount** operation on the *mdspan* of the destination buffer of `f1`. The **ElementCount** operation returns the number of elements in an **mdspan**, which is useful in many scenarios.

## Quick Summary
In this section, we learned how Choreo simplifies data movement using *DMA statements*, making it easier to work with modern hardware. We explored both synchronous and asynchronous operations, and saw how Choreo supports advanced data handling with various *DMA Operation Types*. Finally, we discovered how futures help manage async operations and retrieve buffers, ensuring smooth data flow and synchronization.

In the next section, we will step further to the *Data Expression* to see how to make data tiling/blocking happen.
