
# Memory in Choreo

In Choreo, the memory system is designed to handle large datasets effectively, which is crucial for high-performance computing scenarios, especially in machine learning and scientific computing. To achieve optimal resource utilization, Choreo employs a **spanned type** with **storage qualifiers** that allow programmers to manage how data is distributed and accessed across various levels of memory hierarchy.

## Spanned Types in Choreo

A **spanned type** is a composite type that includes both the fundamental data type (such as `f32`, `s32`, or `f16`) and a **multi-dimensional span (mdspan)** that describes the shape and organization of the data. The spanned type allows Choreo to define large arrays or matrices with an efficient layout, while also enabling the programmer to manipulate the shape (i.e., dimensions) independently of the data type. This flexibility is essential for managing multi-dimensional datasets that are frequently encountered in parallel computation and machine learning workloads.

However, the spanned type by itself does not allocate storage or memory for the data. Instead, it serves as a blueprint for defining how the data will be organized in memory, allowing for runtime flexibility and optimizations. When combined with **storage qualifiers**, the spanned type defines where the data will reside in the memory hierarchy, such as global, shared, or local memory spaces. This distinction enables more efficient memory management, especially when working with accelerators (e.g., GPUs or specialized hardware).

## Storage Qualifiers in Choreo

In Choreo, **storage qualifiers** annotate the memory allocation and specify which memory region the data will reside in. The storage qualifiers help guide the placement of data across the different levels of memory hierarchy, improving memory access patterns and performance. Choreo defines three primary storage qualifiers:

### 1. `global`
- The default memory space for data in Choreo, if no storage qualifier is specified. 
- Data defined in the global memory space is typically the most abundant and has the largest capacity. However, accessing global memory can be slower compared to more localized memory spaces. 
- Global memory is suitable for storing large datasets that need to be shared across multiple processing units or components, such as matrices in deep learning applications or large scientific datasets.

**Example:**
```choreo
global f32 [100, 200] matrix;
```

### 2. `shared`
The shared memory space is typically used for fast, high-throughput access by processing units within a single compute node (e.g., threads in a GPU block or a CPU core).
Data stored in shared memory is generally smaller in size than global memory but offers much faster access times. It's ideal for storing intermediate data, which needs to be accessed frequently during computation, but is not shared across different compute units.
Shared memory allows for efficient data transfer within compute units and can be crucial for optimizing algorithms like matrix multiplication, where intermediate results are reused multiple times.
Example:
```choreo
shared f32 [10, 10] tile_data;
```

### 3. `local`
Local memory is the smallest and fastest memory space, typically used for storing very small data that only needs to be accessed by a specific thread or computation unit.
It is particularly useful when each thread requires a small amount of data, such as scalar values or small arrays that do not need to be shared with other threads.
Local memory is often used in parallel computing scenarios where data needs to be private to each thread and accessed very quickly, but it is not meant for large datasets.
Example:
```choreo
local f16 [5] thread_local_data;
```
## Default Behavior and Usage
By default, when a storage qualifier is not specified, the data is considered to be in global memory. This default behavior simplifies memory management for users who do not need to explicitly define where their data should be stored, but it also means that accessing such data could incur higher latency, especially when working with large arrays or parallel processing units.

For example, the following code defines an array d0 without any storage qualifier, implying it resides in global memory:
```choreo
f32 [100, 100] d0;
```
If you want to place data in a different memory space (shared or local), you need to specify the appropriate storage qualifier. For example, the following code places the matrix d0 in local memory, and the matrix d1 in shared memory:
```choreo
local f32 [10, 10] d0;     // Small data, local to the thread
shared f16 [ndims] d1;     // Intermediate data, shared between threads in a block
```
### Example Code: Using Storage Qualifiers
To better understand how storage qualifiers work, consider the following code example:
```choreo
ndims : [20, 15];
local f32 [10, 10] d0;        // Local memory for thread-specific data
shared f16 [ndims] d1;        // Shared memory for intermediate data across threads
global f32 [100, 200] matrix; // Global memory for large dataset

// Perform computations on these arrays, utilizing the appropriate storage space
dma.copy d0 => d1;            // Copy from local to shared memory
dma.copy d1 => matrix;        // Copy from shared to global memory
```
In this code:

d0 is a small, thread-specific data array placed in local memory.
d1 is an intermediate array placed in shared memory to be used by multiple threads within a block.
matrix is a large dataset placed in global memory, where it can be accessed by all threads.
