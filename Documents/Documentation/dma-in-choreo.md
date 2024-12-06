# DMA in Choreo

In Choreo, Direct Memory Access (DMA) operations are designed to manage data transfers between various memory regions, such as from host memory to device memory or between different memory hierarchies. DMA operations in Choreo are triggered using the dma keyword, followed by the specific operation type (e.g., dma.copy, dma.pad, dma.transp). These operations play a key role in optimizing memory access and are essential for achieving high performance in parallel computing environments.

## DMA Operations in Choreo
A typical DMA operation in Choreo follows this structure:

```choreo
future = dma-op src-operand => dst-operand;
```
Where:

- `dma-op`: The DMA operation type, such as dma.copy, dma.pad, or dma.transp, specifies the type of data transfer operation (e.g., copying data, padding data, or transposing data).

- `src-operand`: The source operand can be an identifier (a variable or array) or a chunkat (a slice or chunk of a memory block). It refers to the memory from which data will be transferred.

- `dst-operand`: The destination operand can also be an identifier or a chunkat. This specifies the location to which data will be transferred. Additionally, it can include a memory specifier (such as shared, local, or global), which indicates the memory space where the data will reside.

- `future`: A future represents the result of the DMA operation. It serves as a handle to track the operation's completion. If the DMA operation is asynchronous, you can use the future object to synchronize the program execution using wait.

## Synchronous vs. Asynchronous DMA Operations
DMA operations in Choreo can be synchronous or asynchronous, each providing different behavior in terms of blocking and non-blocking execution.

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
