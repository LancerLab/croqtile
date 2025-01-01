# Parallelisation and Iterations

Parallelisation and iterations are essential construct in programming laugagues. Parallelisation in choreo follows a Single Programs Multiple Data (SPMD) paradigm, which is abstracted explicitly with the **parallel-by** block. Iterations (loops) is the most utilized method to code sequential code. In choreo, programmers can construct an iteration within the **with-in + foreach** block. These two constructs help Choreo developers express parallel and sequential execution efficiently, enabling better resource utilization on modern hardware.

To understand the parallelism model in Choreo, it’s crucial to first explore the physical architecture of the system on which the code will run. Let’s take **GCU3** (a typical hardware accelerator) as an example, which consists of **clusters** and **SIPs (Single Instruction Processors)** within each cluster. Choreo uses these hardware structures to abstract parallel execution at both the **physical** and **virtual** levels.

## Physical Parallelism: Clusters and SIPs in GCU3

In hardware like GCU3, parallelism is organized into physical entities such as **clusters** and **SIPs**. Each cluster consists of several SIPs, and each SIP executes instructions independently, although they may share data or resources. This physical parallelism is inherent in the system and can be leveraged through the **parallel-by** construct in Choreo.

## Virtual Parallelism: `parallel-by` vs. `with-in + foreach`

Choreo abstracts two types of parallelism:
1. **Physical Parallelism**: Realized through the `parallel-by` block, which allows asynchronous execution using multiple threads, similar to the way hardware clusters work in parallel.
2. **Virtual Parallelism**: Achieved with the `with-in` and `foreach` blocks, which provide more flexible and virtualized constructs for iteration-based parallelism. These constructs are more about organizing and binding data, rather than explicitly handling threads or physical resources.

Let’s explore both approaches in more detail.

## `parallel-by`: Physical Parallelism Abstraction

The **`parallel-by`** block in Choreo is designed to express parallel execution in a manner that reflects physical parallelism on hardware. It uses the **Single Program, Multiple Data (SPMD)** model, which is common in high-performance computing scenarios like OpenMP, CUDA, or OpenCL. This model allows multiple threads to execute the same instruction on different pieces of data simultaneously.

### Syntax of `parallel-by`

The basic syntax of a `parallel-by` block looks like this:

```choreo
parallel p by 6 {
  // SPMD code
}
```

In this example:

- **`parallel`**: Marks the start of the parallel execution block.
- **`p`**: An index that refers to the current thread executing the block. Each thread will execute the same code but with a different value of `p`.
- **`by 6`**: Specifies that there are 6 parallel threads (or executions) in this block, each running the code with a different `p` value.

### How `parallel-by` Works

- **Execution**: In `parallel-by`, each thread (indexed by `p`) executes the same code, but with a different index `p` (e.g., the first thread runs with `p=0`, the second with `p=1`, and so on).
- **Bounded Integer**: The variable `p` in the example above is a bounded integer, meaning it has a defined range of values (from 0 to 5 in this case). This bounded nature is crucial because certain operations, such as `chunkat`, depend on knowing the range of values for proper computation.
- **Asynchronous Execution**: While `parallel-by` allows for asynchronous execution of threads, the threads themselves are synchronized within the block, meaning that the code runs concurrently but may have dependencies that need to be managed through synchronization primitives or other mechanisms.
  
In summary, `parallel-by` directly maps to physical parallelism, and the `p` variable serves as an index, similar to a thread index in CUDA or OpenMP. It provides a way to explicitly define parallel regions that correspond to the hardware’s physical execution threads.

### `with-in` + `foreach`: Virtual Parallelism Abstraction

The `with-in` and `foreach` constructs are designed for more flexible, virtualized parallelism. Unlike `parallel-by`, these constructs are not tied directly to the physical execution model but are used to abstract the data flow and control how elements in a multi-dimensional space are processed.

#### `with-in` Block: Binding Data

The `with-in` block allows you to bind i-tuples (index tuples) to `mdspan` objects, which are multi-dimensional data structures in Choreo. The block allows you to define index ranges and create these bindings.

Example:

```choreo
with index in [10, 10] {
  // index is an i-tuple with 2 elements
}
```

- **index**: Represents an i-tuple, which is a tuple of multiple indices.
- **[10, 10]**: Defines the range of indices for the i-tuple, meaning the index will have 2 elements, both ranging from 0 to 9.

You can also name the elements of the i-tuple for clarity:
```choreo
with {x, y} in [10, 10] {
  // x and y are now explicitly named elements of the i-tuple
}

```

Alternatively, you can give the entire i-tuple a name and refer to its elements by name:
```choreo
with index = {x, y} in [10, 10] {
  // x and y can be used within the block
}
```

#### `where` clause for loop constraints
You can append a where clause to impose constraints between indices, which can be useful in cases where certain indices need to have specific relationships.

```choreo
with {m, n} in [M, N], {n_p, k} in [N_P, K] 
where n_p <-> n {
  // matmul implements with m, n, K. n_p is no longer useful.
}

```
In this example:

- The where clause defines that n_p and n must have the same value in all iterations.
- `<->` is the notation used to indicate that n_p is related to n in some way.

The with-in block by itself does not imply parallelism, unlike parallel-by. It is used primarily to bind data, iterate over ranges, and define relationships between indices. The where clause further refines how these indices interact within the block.

#### `foreach` Block: Iterating Over Bounded Elements

Once the i-tuple is defined in a with-in block, you can use the foreach block to iterate over these bounded elements.

In this case, the foreach block will iterate over each element of the index x in the range [0, 9] and execute the corresponding code for each value of x.

Example:
```choreo
with x in [10] {
  foreach x {
    // do something with each x
  }
}

```

#### Quick summary
- `with-in`: Binds indices or i-tuples to a data structure (e.g., an mdspan). It is used to create a virtualized iteration space but does not directly imply parallelism.
- `foreach`: Used in combination with with-in to iterate over the bound indices or i-tuples, performing operations on each element. It provides a convenient way to loop over multidimensional data.

### Key Differences Between `parallel-by` and `with-in` + `foreach`

| Feature               | `parallel-by`                        | `with-in` and `foreach`                |
|-----------------------|--------------------------------------|----------------------------------------|
| **Parallelism Type**   | Physical parallelism (SPMD model)    | Virtualized parallelism (used for data binding and iteration) |
| **Synchronization**    | Threads are asynchronous but synchronized within the block | Sequential execution; no parallel execution implied |
| **Indexing**           | Index `p` is bounded, corresponds to thread index | Index or i-tuple binds to `mdspan`, can define relationships with `where` clause |
| **Usage**              | Directly for parallel execution on multiple threads | For binding and iterating over data in virtual parallel regions |

