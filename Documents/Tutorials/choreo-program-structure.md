## Choreo Program Structure

A typical **Choreo** program is composed of three main parts, each serving a specific purpose and allowing for efficient data orchestration and parallel computation. These components can be integrated into a single program, making it easier to manage the flow of data and execution. Let's dive deeper into each part and its role within a Choreo program.

**Host Program**: The **Host Program** serves as the central control unit of the Choreo program. Written in standard **C++**, the host program is responsible for managing the overall flow, including data preparation, kernel launches, and synchronization with the target device (e.g., CPU, GPU, or accelerator). The host program coordinates the execution of kernels and the orchestration of data movement between the CPU and the device.

**Kernel Program**: In Choreo, kernels are often written in a structured form, wrapped in the `__cok__` block. The **Kernel Program** defines the inner-most kernel code executed on the target device. It is responsible for processing the data in parallel, leveraging the full computational power of the target hardware, such as using `2d intructions`. Specifically, you can use arbitrary programming tools (other than choreo) to write this part of code as long as this is supported by Choreo to collaborate with. Currently, Choreo supports kernel code written in **Intrinsics**, **Tcle**, and will support **Primo++** later on.

**Choreo Function (Tileflow Program)**: The **Choreo Function** (also called the Tileflow Program) is the heart of the Choreo programming model. It is responsible for orchestrating the movement and computation of data between the host and the kernel, ensuring that data is processed in manageable chunks (tiles) and moved efficiently between different memory spaces. In essence, the Choreo function provides a high-level abstraction that simplifies the process of orchestrating data flow and parallel computation, making it easier for developers to write high-performance applications.

Notably, **Choreo** programs integrate all three parts—**Host Program**, **Kernel Program**, and **Choreo Function**—into a single source file. This design simplifies the coding process by reducing the mental overhead required to manage multiple files simutaneously and jump back and forth during coding. By keeping all parts of the program in one place, Choreo encourages a compact and cohesive code structure. The feasibility of this organization relies on the overall brevity of the program, where the **lines of code (LOC)** are significantly reduced. This is achieved by making Choreo programs highly information-dense, enabling developers to express complex operations with minimal code. In subsequent chapters, we will gradually introduce these design characteristics and demonstrate how Choreo achieves both simplicity and expressiveness through its unique abstractions.

### Host Program

The **host program** is the entry point of the Choreo program. It is typically written in standard C++ and serves as the control center for launching kernel programs and managing data transfers. A simple host program looks like this:

```cpp
int main() {
  // Define data arrays
  choreo::s32 a[6][17][128] = {0};
  choreo::s32 b[6][17][128] = {0};
  
  // Fill arrays with data
  std::fill_n(&a[0][0][0], sizeof(a) / sizeof(a[0][0][0]), 1);
  std::fill_n(&b[0][0][0], sizeof(b) / sizeof(b[0][0][0]), 2);

  // Call Choreo function (data transfer and kernel execution)
  auto res = ele_add(choreo::make_spanview<3, choreo::s32>((int*)a, {6, 17, 128}),
                     choreo::make_spanview<3, choreo::s32>((int*)b, {6, 17, 128}));

  // Print the result shape
  std::cout << res.shape()[0] << std::endl;
  std::cout << res.shape()[1] << std::endl;
  std::cout << res.shape()[2] << std::endl;

  // Verification: check correctness of results
  for (size_t i = 0; i < res.shape()[0]; ++i)
    for (size_t j = 0; j < res.shape()[1]; ++j)
      for (size_t k = 0; k < res.shape()[2]; ++k) {
        std::cout << "i: " << i << ", j: " << j << ", k: " << k << "\n";
        std::cout << "a: " << a[i][j][k] << ", b: " << b[i][j][k] << ", res: " << res[i][j][k] << "\n";
        if (a[i][j][k] + b[i][j][k] != res[i][j][k]) {
          assert(a[i][j][k] + b[i][j][k] == res[i][j][k]);
        }
      }

  std::cout << "Test Passed\n" << std::endl;
}

```

Key responsibilities of the Host Program include:

- **Initializing data arrays**: The host program begins by defining and initializing the data arrays that will be processed by the kernel. In the given example, two 3D arrays `a` and `b` of dimensions `[6][17][128]` are created and initialized with values `1` and `2`, respectively. These arrays are prepared for subsequent operations.
  
```cpp
choreo::s32 a[6][17][128] = {0}; // Initialize array a
choreo::s32 b[6][17][128] = {0}; // Initialize array b

// Fill arrays with values
std::fill_n(&a[0][0][0], sizeof(a) / sizeof(a[0][0][0]), 1);
std::fill_n(&b[0][0][0], sizeof(b) / sizeof(b[0][0][0]), 2);
```
  

- **Transferring data to the device**: In a typical Choreo program, the data arrays initialized by the host program would need to be transferred to the target device (CPU, GPU, or accelerator). Although this step is abstracted in the given code (via the Choreo framework), the host program is responsible for ensuring that the necessary data is moved to the right memory locations for kernel execution. Choreo handles these data transfers behind the scenes using efficient memory management techniques such as DMA (Direct Memory Access).
  
- **Kernel execution**: After the data has been initialized, the host program calls the **Choreo function** (`ele_add` in this case), which orchestrates data movement, invokes the kernel, and processes the data. The `ele_add` function is responsible for performing the actual computation of adding the elements of arrays `a` and `b` in parallel. The host program calls this function with the necessary arguments (e.g., arrays `a` and `b` converted to Choreo spanviews), triggering the kernel execution on the target device.
  
```cpp
auto res = ele_add(choreo::make_spanview<3, choreo::s32>((int*)a, {6, 17, 128}),
                    choreo::make_spanview<3, choreo::s32>((int*)b, {6, 17, 128}));

```
  
- **Result verification**: You can just use trivial C++ code to write arbitrary checks/verifications as you like.
  
```cpp
for (size_t i = 0; i < res.shape()[0]; ++i)
  for (size_t j = 0; j < res.shape()[1]; ++j)
    for (size_t k = 0; k < res.shape()[2]; ++k) {
      std::cout << "i: " << i << ", j: " << j << ", k: " << k << "\n";
      std::cout << "a: " << a[i][j][k] << ", b: " << b[i][j][k] << ", res: " << res[i][j][k] << "\n";
      if (a[i][j][k] + b[i][j][k] != res[i][j][k]) {
        assert(a[i][j][k] + b[i][j][k] == res[i][j][k]);
      }
    }
```

The host programs are embedded DSL with **C/C++** and are thus designed to work seamlessly with other C/C++ programming models. They can be compiled as standard C++ functions and assembled into libraries, enabling developers to use the rich functionality of Choreo alongside regular C++ code. The core difference between standard C++ and Choreo is the provided utilities, which enable you to integrate Choreo elements like special types and memory management abstractions into your code.

### **Kernel Program (`__cok__`)**

The **kernel program** defines the computational logic that will be executed on the target device (e.g., GPU, CPU). This part of the code is wrapped within the `__cok__` block, and it contains the actual computation. The kernel is designed to operate on input data, process it in parallel, and produce the output.

Example kernel program inside `__cok__`:

```cpp
__cok__ { /// Kernel program 

extern "C" void kernel(int * a, int * b, int * c, int n) { 
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i]; 
}

} /// End of kernel declaration
```

In this kernel:

- The kernel function `kernel` takes in three arrays `a`, `b`, and `c` and performs an element-wise addition for each element in the arrays.
- This kernel is a simple example and may be run on a CPU for testing. In real-world scenarios, more complex logic can be written here to leverage the target device's computational power.

Choreo’s kernel programming model allows for different backends to be used. For example, you can use:

- **TCLE (Target Compiler Language Extension)**: Specialized kernel language for target devices.
- **Intrinsic kernels**: Custom hardware-accelerated kernels for specific tasks.
- **Standard C++**: For mock execution on the CPU.

The `__cok__` block wraps the kernel logic and is executed in parallel on the target device.

### **Choreo Function (`__co__`)**

The **Choreo function** is where the **data orchestration** happens. It manages the movement of data between the host and the target device and ensures that data is copied correctly across different memory spaces. This part of the program defines **data flows** and **synchronization** between devices.

Here is a minimal example and corresponding explainations:

```cpp
__co__ s32 [6, 17, 128] ele_add(s32 [6, 17, 128] lhs, s32 [6, 17, 128] rhs) {  /// Device program
  s32[lhs.span] output; // Use same shape as lhs
  
  parallel p by 6 {  // p is sip_index
    with index = {x, y} in [17, 4] {  // Declare your tiled spans
      foreach x {  // Alternative: foreach index, index.x, index.y
        foreach y {
          lhs_load = dma.copy.async lhs.chunkat(p, x, y) => local; // Tiling factor
          rhs_load = dma.copy.async rhs.chunkat(p, x, y) => local;
          wait lhs_load, rhs_load;

          local s32[lhs_load.span] l1_out;

          // Call kernel with loaded data
          call kernel(lhs_load.data, rhs_load.data, l1_out, |lhs_load.span|);

          // Store result back to output
          out_store = dma.copy.async l1_out => output.chunkat(p, x, y);
          wait out_store;
        }
      }
    }
  }
  return output;
}
```

As the core part of a Choreo program, `__co__` function has following roles:

- **Tiling and partitioning**: The function divides the data into tiles (`lhs.chunkat(p, x, y)`) to ensure that the computation is performed on smaller, manageable pieces.
- **DMA transfers**: The `dma.copy.async` operation is used to asynchronously copy data from the device memory to local buffers for computation. This is a crucial feature for optimizing memory access patterns in high-performance computing.
- **Parallelism**: The `parallel p by 6` construct allows for parallel execution across different data chunks, improving throughput and utilizing the target device efficiently.
- **Kernel invocation**: Once the data is loaded, the `call kernel()` invokes the computational kernel (defined in the `__cok__` block) to perform the element-wise addition.
- **Synchronization**: The `wait` statements ensure proper synchronization between memory transfers and computation.
