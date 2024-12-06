# Choreo - The C++ DSL for TileFlow Programming
Choreo is a low-level Embedded Domain Specific Language (EDSL) for C++ specifically engineered to program data movement entities like DMAs, which traditionally focus on hardware configuration rather than the data itself. This language facilitates the programming of data flow across a hardware’s memory hierarchy, where at each level, data is partitioned into smaller chunks that are positioned closer to the processing elements. By making data tiling and flowing more manageable, Choreo pioneers the 'TileFlow' programming paradigm. This new approach aims to significantly streamline the engineering tasks involved, particularly for professionals developing AI operations on specific hardware.

## Choreo and Device-Level C++
Choreo has close relationship to the device-level C++ programming, which is usually provided by the hardware vendor. In current implementation, Choreo performs source-to-source translation to convert *choreo-c++* programs to vendor-supported (like factor, topscc, CUDA) C++ language entities and APIs. Despite of turning higher level abstraction of *TileFlow* functions into corresponding low level C++API calls, Choreo also glues the kernels, the host code into executables. Consequently, using Choreo compiler is easy when the vendor-provided device-level C++ compiler is properly configured.

## Features
### Productivity
One of the standout features of Choreo by Design is its ability to **enhance productivity**, particularly through the **mind-set saving** in **data tiling** tasks. This is achieved by introducing programming entities such as *mdspan*/*ituple*, which simplify data shape manipulation to a level comparable to *Python*. For instance: For example:
```cpp
__co__ f32 mdspan<2> my_function(f32 [8, 4, 12] d) {
    block_shape : d.span { (0)/ 2, (1)/ 4, 1, (2)};
}
```
With this code, programmers can effortlessly create a shape with a tiling factor of {2, 4, 1} from data 'd' and even add an extra dimension to the tiled 'block_shape', all in a single line. Additionally, programmers can describe a DMA operation within a single line of code:
```cpp
  dma.copy input.chunkat(tiling_factors) => shared;
```
This code transfers a data chunk with specified tiling factors to a storage location named 'shared' via DMA. The explicit tiling in the code is easy to maintain, and the complexities of DMA configurations, index calculations, and storage management are handled by the Choreo compiler. This allows programmers to concentrate on high-level strategies for building high-performance kernels.

For more productivity-enhancing designs and features, please refer to our tutorials.


### Code Safety
Another primary design goals of Choreo is to **ensure code safety** by catching errors at compile-time or as early as possible at runtime. To achieve this, Choreo employs **compile-time checks** and **runtime-check** based on the shapes and rules inferred from the Choreo code.

Bugs related to DMA are typically challenging to diagnose. However, with Choreo's safety checks, programmers can significantly reduce debugging efforts, thereby shortening the overall development cycle.

### Dynamic/Symbolic Shapes
Choreo could be the first programming languages to support **symbolic shape dimensions**, where the values are determined at runtime but can still be checked statically or dynamically. This dynamic capability is crucial for building many machine learning kernels. The example below illustrates this feature:

```
__co__ auto matmul(f32 [M, K] lhs, f32 [N, K] rhs) { ... }
```
This feature introduces a natural way to program shaped inputs, such as tensors. Additionally, for enhanced code safety, Choreo generates assertions by leveraging the relationships (e.g., dimensional equivalence) inferred from the code. This eliminates the need for many trivial, explicitly programmed assertions, thereby reducing boilerplate code.

### Visualization
**Analytic and visualization** is another compelling feature of Choreo, designed to help newcomers understand DMA behaviors. For instance, consider the following DMA statement:

`f1 = dma.copy a.chunkat(p, x, y) => local;`

With Choreo's visualization capability, programmers can observe the data movement resulting from this statement. The visualization might look something like this:

![visualizing the DMA statement](./images/simple_dma.png)

This is helpful for programmers escpecially for novices as visualization gives clear projection about the tiling behavior. Or else, programmers have to visualize in their mind, which is more error-prone.

# Documentation for Reference
Consult the [Choreo Documentation and Tutorials](http://10.31.50.149:8000/) document for information on building Choreo and the detailed usage.

