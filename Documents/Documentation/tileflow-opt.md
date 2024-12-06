# Tiling Optimisation in Choreo

In this section, we will extend our discussion on tiling optimization in Choreo by examining some practical examples.

## Example 1: `matmul`
The first example is matrix multiplication on a device program. This example demonstrates how Choreo’s tiling and memory management mechanisms work together to optimize matrix operations, including how DMA (Direct Memory Access) is leveraged for efficient data movement.

---

### Example Code: Matrix Multiplication with Tiling

```choreo
__co__ s32 [512, 1024] matmul(s32 [512, 1024] lhs, s32 [1024, 1024] rhs) { 
  // Device program
  s32[lhs.span(0), rhs.span(1)] output; // Use same shape as lhs

  parallel p by 1 {
    with index = {m_tile, k_tile, n_tile} in [1, 1, 1] {
      foreach m_tile, n_tile {
        shared s32[512, 1024] l2_out;
        foreach k_tile {
          // Initiate DMA operations
          lhs_load = dma.copy.async lhs.chunkat(m_tile, k_tile) => shared;
          rhs_load = dma.copy.async rhs.chunkat(k_tile, n_tile) => shared;

          parallel q by 1 {
            with index = {m_tile_s, k_tile_s, n_tile_s} in [4, 4, 8] {
              // Local buffers for tiling
              local s32 [128, 256] lhs_load_s_buffer_0 {0}, lhs_load_s_buffer_1 {0};
              local s32 [256, 128] rhs_load_s_buffer_0 {0}, rhs_load_s_buffer_1 {0};
              foreach index {
                // Add "after" primitive to chain up paired DMA operations
                lhs_load_s = dma.copy.async lhs_load.data.chunkat(m_tile_s, k_tile_s) => select(k_tile_s % 2, lhs_load_s_buffer_0, lhs_load_s_buffer_1) after lhs_load;
                rhs_load_s = dma.copy.async rhs_load.data.chunkat(k_tile_s, n_tile_s) => select(k_tile_s % 2, rhs_load_s_buffer_0, rhs_load_s_buffer_1) after rhs_load;
                out_load_s = dma.copy.async l2_out.chunkat(m_tile_s, n_tile_s) => local;
                wait out_load_s;

                // Perform the matrix multiplication (dot product) operation
                call dot_general_fp32(lhs_load_s.data, rhs_load_s.data, 256, 256, 256, 0, 8, out_load_s.data);

                // Store the result back to l2_out
                out_store_s = dma.copy.async out_load_s.data => l2_out.chunkat(m_tile_s, n_tile_s); 
              }
            }
          }

          // Add "after" primitive to chain up DMA operations for the final output
          out_store = dma.copy.async l2_out => output.chunkat(m_tile, n_tile) after out_store_s;
        }
      }
    }
  }
  return output;
}
```
---

### Memory Tiling with `chunkat`

In the matrix multiplication example, **memory tiling** is achieved using the `chunkat` method, which slices the input matrices `lhs` (left-hand side) and `rhs` (right-hand side) into smaller chunks to be processed in parallel.

##### Tiling via `chunkat`:

- The `lhs` and `rhs` matrices are tiled using the `chunkat` method. The chunks are defined by `m_tile`, `k_tile`, and `n_tile`, which represent different dimensions of the matrices involved in the multiplication.
- The chunks for both matrices are loaded asynchronously into the **shared memory** (`lhs_load` and `rhs_load`) with the `dma.copy.async` operation. This operation loads a portion of the matrix from **global memory** to **shared memory**, which is a common optimization in large matrix operations to improve data locality.

##### Bounded Variables:

- The variables `m_tile`, `k_tile`, and `n_tile` are **bounded variables**. They are defined in the context of `parallel-by` and `with-in` constructs, which define the iteration space for the tiling.
- These bounded variables allow the chunks to be computed in parallel, where each thread works on a different slice of the matrix. This parallelization enables efficient computation of matrix multiplications across multiple threads.


## Example 2: rgb2gray
The second example is the RGB to Grayscale conversion algorithm. The algorithm converts a color (RGB) image to a grayscale image using a weighted average method. This example shows how physical parallelism work together with virtual parallelism.

---
### Example Code

```choreo
__co__ auto rgb2gray(f32 [N, 3, H, W] input) {
  f32 [N, H, W] out;
  parallel q by 6 {
    with index={n, h, w} in [N, H, W]/{#q, 16, 512} {
      foreach n, h, w {
        // _ means no tiling
        input_L1_A = dma.copy input.chunkat(q#n, _, h, w) => local;
        dims : input_L1_A.span[(0), (2), (3)];
        local f32 [dims] out_L1;
        call rgb2gray_kernel_fp32_fp32(input_L1_A.data, out_L1, |out_L1|, 1);
        dma.copy out_L1 => out.chunkat(q#n, h, w);
      }
    }
  }
  return out;
}
```
---
### Compose Two Bounded Variables for Tiling with `#`

In the example above, there is a need to tile the matrices in a manner that involves **two bounded variables**. This is where the **`#` operator** comes into play, allowing the program to compute the **Cartesian product** of two bounded variables (**not commutative**).

##### The `#` Operator:

- The code defines the tiling parameters with four bounded variables: `q`, `n`, `h`, and `w`, whose upper bounders are `6`, `N/6`, `H/16`, and `W/512`.
- Suppose `N` is 18, `q` represents physical thread index. We want each thread to process data that is contiguous in the first dimension. That is, the first thread handles input[0~2][xxx], the second thread handles input[3~5][xxx], and so on.
- `q # n` will result in a new bounded variable implicitly, whose upper bound is `N`: bound of `q` multiply bound of `n`.
- For every data move, the stride of each dimension is 1, 3, 16, 512. In the last dimension, the index is `w`. In the first dimension, the index is `q * (bound of n) + n`. 


##### Why this is Important:

- The combination of `q` and `n` ensures that physical and virtual parallelism can work together efficiently.
- This helps optimize **memory access patterns**, especially when working with large matrices, where efficient memory use is crucial.
- Choreo’s approach here is more aligned with **physical memory hierarchies** (e.g., local/shared memory on GPUs or accelerators). Unlike higher-level loop scheduling techniques like in TVM, which abstract away the actual data movement, Choreo explicitly links **virtual loops** to **physical DMA operations**. This results in more efficient data movement and computation.


## Virtual Parallelism and Physical DMA Mapping

In traditional systems like **TVM** (Tensor Virtual Machine), scheduling and loop transformations are abstracted to optimize computation. For example, a typical **TVM schedule** for a matrix multiplication operation might look like the following:

```python
# Example of TVM-style loop schedule
s = tvm.create_schedule(A.op)
xo, xi = s[A].split(A.op.axis[0], factor=64)
yo, yi = s[A].split(A.op.axis[1], factor=64)
s[A].bind(xo, tvm.thread_axis("blockIdx.x"))
s[A].bind(yo, tvm.thread_axis("blockIdx.y"))
s[A].bind(xi, tvm.thread_axis("threadIdx.x"))
s[A].bind(yi, tvm.thread_axis("threadIdx.y"))
```

Here, the computation of a matrix A is split into multiple loops with the axis split by some factor (e.g., 64). The bind method binds loop variables to hardware threads, providing an abstraction of parallelism. This approach works well in a variety of cases, but it comes with some limitations when dealing with specific memory hierarchies or when performance is tightly coupled with the underlying hardware behaviour.

#### The Limitation of Virtual Parallelism in TVM-style Schedules
In TVM’s loop schedule, virtual parallelism is represented via abstraction layers like threadIdx and blockIdx, which map high-level parallelism to physical hardware threads. However, the physical behavior of data movement, particularly in the case of complex memory hierarchies (e.g., local, shared, or global memory), is not explicitly represented. TVM abstracts away the details of data movement, leaving some of these behaviors to be automatically determined by the system, either through rules or optimizations.

The main limitation here is that the physical behavior and memory access patterns may not be fully optimized because they are hidden behind the abstraction. As a result, the system may perform well in general cases, but it may not always exploit the full potential of the hardware, especially when fine-grained control over memory access is required (e.g., in high-performance computing scenarios).

#### Freedom Left to the System
In this abstraction, much of the decision-making about memory and parallelism is left to the automatic optimization process in the system. This includes decisions about:

- Memory layout (whether data should be placed in local, shared, or global memory),
- DMA operations (how data should be transferred across different levels of memory),
- Parallel thread mapping (how virtual loops are mapped to physical threads).

While this freedom allows for a broad set of optimizations, it hides some of the underlying physical behavior from the user. This can be both a strength and a weakness:

- Strength: The system can automatically make optimizations based on heuristics or rules, allowing it to adapt to different hardware architectures or runtime conditions.
- Weakness: The user has less control over how parallelism is mapped to hardware, which may lead to suboptimal performance in certain cases, especially for complex memory and data movement patterns.

#### The Importance of Explicit Physical Mapping in Choreo
In contrast to the abstracted approach in TVM, Choreo explicitly represents both virtual parallelism (e.g., through parallel-by, with-in, and foreach constructs) and physical memory operations (e.g., using DMA operations, chunkat, and memory qualifiers).

By explicitly modeling the mapping of virtual loops to physical DMA operations and memory hierarchies, Choreo ensures that memory access patterns are optimized for the underlying hardware. This allows for fine-grained control over memory management, which can lead to better performance, especially on hardware accelerators like GPUs or custom accelerators.

#### How Choreo Works to address this limitation?
In Choreo, the virtual parallelism is clearly defined through bounded variables and loops (e.g., parallel by, foreach).
The physical behavior (e.g., data transfer between memory hierarchies via DMA) is directly controlled and mapped.
This explicit coupling between virtual parallelism and physical data movement gives users more control over performance optimizations, especially in complex scenarios where manual tuning is required to exploit specific hardware features like memory hierarchies, cache optimizations, and data locality.

In **Choreo**, **DMA operations** explicitly represent data movement between different memory levels, such as from **DRAM** (main memory) to **SRAM** (on-chip memory) and from **SRAM** to **registers**. This explicit control over data movement is crucial for optimizing memory access patterns, particularly on hardware accelerators like GPUs. When performing **tiling**, the **# operator** is used to calculate the **Cartesian product** of bounded variables, which enables the tiling of multiple loop dimensions and directly binds them to **DMA operations**. This provides a tight coupling between **virtual loop dimensions** (e.g., m_tile, k_tile, n_tile) and the actual **physical memory transactions** involved in data movement. 

The **# operator** combines multiple bounded variables into a **composite variable**, defining a tiling pattern for the data. The resulting tiling improves cache locality and reduces memory access latency by optimizing the movement of data across memory spaces. Once the tiling is defined, **DMA operations** move data from **DRAM** to **SRAM** and from **SRAM** to **local registers** at the correct times. In matrix multiplication, for instance, **dma.copy.async** is used to load chunks of the matrix into **shared memory (SRAM)**, and later from **SRAM** into **registers** for computation. Each **DMA operation** is tied to the tiling pattern defined by the **bounded variables**, ensuring efficient memory access during computation.

The **# operator** for tiling allows multiple **virtual loop dimensions** to be mapped directly to **memory operations**, offering a clear advantage over systems like **TVM**, which abstract away the details of memory operations and parallelism. This explicit mapping of **virtual parallelism** to **physical DMA operations** ensures that data is moved efficiently across the memory hierarchy, improving performance. 

By using the **# operator** to define **multi-dimensional tiling** and explicitly mapping it to **DMA operations**, Choreo gives programmers **fine-grained control** over memory access, leading to improved cache locality and reduced latency. It aligns virtual parallelism with the physical hardware memory hierarchy, providing **better performance** on accelerators by optimizing data movement. This approach of **explicit memory tiling** and **DMA control** ensures that Choreo can leverage the full potential of the hardware, unlike higher-level abstractions that may limit the ability to directly control data movement.

This compact version integrates all the important points without using subheadings, maintaining a more concise, flow-like structure for easier reading.
