# Choreo - The C++ DSL for TileFlow Programming
Choreo is a low-level Embedded Domain Specific Language (**EDSL**) for C++ specifically engineered to program data movement entities like Direct-Memory-Accesses (DMA). 

Traditionally, programming DMA has focused on hardware configuration rather than the data itself. In modern heterogeneous hardware like GPUs, programmers often need to move smaller chunks of data to faster memory to enhance performance. This requirement can make programming more complex and sometimes results in hard-to-maintain code.

To address these challenges, Choreo is designed to simplify DMA programming by introducing a novel paradigm called **'TileFlow' programming**. It has already shown significant improvement on productivity, safety, and adaptibility over existing design, and is proven effective for building **machine learnig computing kernels** on heterogeneous hardware.

## Features and Design Targets
### Productivity
One of the standout features of Choreo by design is its ability of **mind-set saving** in **data tiling** tasks. This is achieved by introducing domain specific types, which simplify data **shape manipulation** to a level comparable to *Python*. For instance: 
```cpp
  f32 [8, 4, 12] shaped_data;
  new_shape : shaped_data.span { (0)/ 2, (1)/ 4, 1, (2)};
```
With this code, programmers can effortlessly create a shape with a tiling factor of {2, 4, 1} from data 'd' and even add an extra dimension to the 'new_shape', all in a single line. Compared with corresponding C++ code, which has to build array and apply trivial arithmetics, Choreo spares programmers from having to combine low-level abstractions. 

Furthermore, as Choreo simplify operations of data movement, it provides the high-level abstraction of tiled data movement:
```cpp
  dma.copy input.chunkat(tiling_factors) => shared;
```
This code moves a data chunk of 'input' with specified tiling factors to a storage location named 'shared'. The code is usually observed in programs with hardware DMA support. Choreo compiler hides the complexities of DMA configurations, index calculations, and storage management with easy-to-maintain semantics. Therefore, it allows programmers to concentrate on high-level strategies for building computing kernels, which are normally essential for building ML applications.

### Code Safety
Another primary design goal of Choreo is to **ensure code safety** by catching errors at compile-time or as early as possible at runtime. To achieve this, Choreo employs **compile-time checks** and instruments **runtime-check** based on the shapes and rules inferred from the *tileflow code*.

Bugs related to DMA are typically challenging to diagnose. However, with Choreo's safety checks, programmers can significantly reduce debugging efforts, thereby shortening the overall development cycle.

### Dynamic Shapes
Dynamic shape support is crucial for building many ML kernels. Choreo enhances the dynamic shape support via the **symbolic dimension** feature. Programmers can utilize the feature easily like the below code:

```
__co__ auto matmul(f32 [M, K] lhs, f32 [N, K] rhs) { ... }
```
'M', 'N' and 'K' are the symbolic shape dimensions. Programs program shaped inputs, such as tensors, in such a natural way. Such a design priors any existing systems (late 2024). Additionally, symbolic dimensions are also checked to ensure safety. As it is automatic, and systematic, it eliminates the need for non-systematic, explicitly programmed assertions by the users, thereby reducing boilerplate code.

### Visualization
**Analytic and visualization** is another compelling feature of Choreo, designed to help programmers understand tiling behaviors. For instance, consider the following data movement statement:

`f1 = dma.copy a.chunkat(p, x, y) => local;`

With Choreo's visualization capability, it renders figures like:

![visualizing the DMA statement](./images/simple_dma.png)

Programmers is easy to find the projection of the tiling and data movement behavior from this visualization. Such assistance can significantly reduce user erorrs when being properly used.

# Documentation for Reference
Consult the [Getting Started With Choreo](./Documents/Documentation/getting-started-with-choreo.md) to build and install Choreo.
Consult the [Choreo Tutorials](http://10.31.50.149:8000/) document for information on building Choreo and the detailed usage.

