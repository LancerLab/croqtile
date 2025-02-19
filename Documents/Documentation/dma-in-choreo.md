## Overiew
In this section, you will learn the basic data movement statements and the future variable they produce.

## Data Movement Statement

Choreo intends to address mulitple-levels of data movement. However, till now the primary focus of Choreo is the data movement between heterogeous hardware, and across memory hiararchy. In hardware terminology, such movement are named the **Direct Memory Access (DMA)**. Choreo follows this naming and abstract the data movement as **DMA Operations**.

### Basic Syntax
The DMA statement is the most complicated statement in Choreo. A DMA statement in Choreo could be an asynchronous entity, which means that the code following an asynchronous DMA execute in parallel with the DMA statement. In some terminology, the *asynchronous DMA* is called the **None-Blocking DMA**, since it would not block the code following from execution, while the *synchronous DMA* is named the **Blocking DMA**, as the code would not step further until the DMA has finished. For the *Asynchronous DMA*, an explicit synchronization is required before using the DMA result.

To support such features, the *DMA statement* in Choreo is organized as below:

```choreo
future = dma-op src-expr => dst-expr;
```

The statement defines a **future**-typed variable, which appears on the left of operator "=". In Choreo, a *future* represent **both the handle of asynchrous execution instance and the DMA result**. Thus, the definition of *future* could be ignored for synchronized *DMA statement*s.

The right-hand-side of the *DMA operation* includes the operation type (`dma-op`), expression for the operation source (`src-expr`), and the expression for the destination (`dst-expr`). The source and destination is seperated with the symbol "=>", which indicates how the data flow.

### Operation Type

Choreo's DMA statement is an abstraction that is extended to support morden hardware. Despite linear memory copies, advanced hardware, such as *Data Transfer Engine (DTE)* of GCU hardware and *Tensor Memory Accelerator (TMA)*, are capable to transfer shaped data and apply shape transformations inflight Therefore, currently Choreo supports operations including:

- `dma.copy`: it copies the flat memory directly.
- `dma.pad`: it *pad*s shaped data while transferring the data.
- `dma.transp`: it *transpose*s the shaped data while transferring the data.


Note the support of the operation types may differ according to the underlying software and hardware. If not support, the compiler will emit error to warn programmers.

In addtion to the operation type, an `.async` suffix could be appended to the operation to indicate the operation is asynchronous. For example:

```
dma.copy from => data;           // synchronous
f = dma.copy.async from => data; // asynchronous
f = dma.pad.async<1,1 from => data; // asynchronous
```

### Synchronous DMA Operations
In a synchronous DMA operation, the program execution will block until the DMA operation is completed. For example:

```choreo
load = dma.copy rhs.chunkat(k_tile, q) => shared;
```
Here, the program will pause until the data is fully transferred from `rhs.chunkat(k_tile, q)` to shared memory. The program cannot proceed until the copy operation completes. This type of DMA operation is simple and predictable but may lead to performance bottlenecks in applications that require overlapping computation with data transfers.

### Asynchronous DMA Operations
An asynchronous DMA operation, on the other hand, does not block the program. The program continues executing while the data transfer happens in the background. When the transfer is finished, a future object is returned to track the operation's completion. For example:

```choreo
load = dma.copy.async rhs.chunkat(k_tile, q) => shared;
```
In this case, the program continues execution without waiting for the data transfer to complete. The load object represents a future, which holds the status of the DMA operation. To ensure the data is available before further operations, you need to explicitly wait for the future:

```choreo
wait load;
```
This wait operation blocks the program until the DMA operation completes. The key point here is that the initial dma.copy.async operation is non-blocking, but wait load will block the program execution until the DMA operation is finished.

### Full Non-Blocking DMA Mode (Chain Mode in Choreo)
In Choreo, it's possible to perform a full non-blocking DMA by chaining multiple asynchronous DMA operations and using event-based notifications with after. This enables complete non-blocking execution, where one DMA operation is triggered only after the completion of a prior one. Here’s an example of such a setup:

```choreo
out_store = dma.copy.async l2_out => output.chunkat(m_tile, n_tile) after out_store_s;
```
In this example:

- `out_store` is the asynchronous DMA operation that transfers data from l2_out to output.chunkat(m_tile, n_tile).
- `out_store_s` is another DMA operation or event that must complete before out_store can proceed.
- The after out_store_s syntax specifies that `out_store` should only start after the completion of `out_store_s`, which ensures that there is no blocking in the main thread.

In this case, neither `out_store` nor `out_store_s` will block the main program flow. The program continues executing while these DMA operations are handled in the background. The key difference here is that the completion of `out_store_s` triggers the start of `out_store`, creating an event-driven dependency between the two DMA operations. This model enables highly efficient and non-blocking memory transfers.
