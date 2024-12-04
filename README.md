# Choreo - An eDSL for TileFlow Programming
Choreo is a low-level embedded Domain Specific Language (eDSL) specifically engineered to program data movement entities like DMAs, which traditionally focus on hardware configuration rather than the data itself. This language facilitates the programming of data flow across a hardware’s memory hierarchy, where at each level, data is partitioned into smaller chunks that are positioned closer to the processing elements. By making data tiling and flowing more manageable, Choreo pioneers the 'TileFlow' programming paradigm. This new approach aims to significantly streamline the engineering tasks involved, particularly for professionals developing AI operations on specific hardware.

## Choreo and Device-Level C++
Choreo has close relationship to the device-level C++ programming, which is usually provided by the hardware vendor. In current implementation, Choreo performs source-to-source translation to convert *choreo-c++* programs to vendor-supported(like factor, topscc) C++ language entities and APIs. Despite of turning higher level abstraction of *TileFlow* functions into corresponding low level C++API calls, Choreo also glues the kernels, the host program into executables. Consequently, using Choreo compiler is easy when the vendor-provided device-level C++ compiler is properly configured.

## Features
### Productivity
One of the attempting feature of Choreo by design is the **improve productivity**. Specifically, it is about **the mind-set saving** when dealing with **data tiling** work. This is done by introducing programming entities including *mdspan*/*ituple*, which makes the manipulation of data shapes as easy as if it is in *Python*-level. For example:
```
__co__ f32 mdspan<2> my_function(f32 [8, 4, 12] d) {
    block_shape : d.span { (0)/ 2, (1)/ 4, 1, (2)};
}
```
Programmers can easily make a shape with a tiling factor of {2, 4, 1} from data 'd''s shape. In addition, the programmer adds one dimension to the tiled 'block_shape'. All these work are done in one line of code. It saves mind-set since neither the trivial computation is necessarily done by human, nor tidious C++ programming is required.

There are still many different designs to improve productivity, including tiling control, loop control, parallelization control, etc. Choreo tries to provide feasible and simple abstractions to make it as simple as possible.


### Program Safety
One of the primary design goals of Choreo is to ensure code safety by catching errors at compile-time rather than at runtime. To achieve this, Choreo introduces **compile-time checks**. This feature includes domain-specific types that are associated with corresponding shapes. These shaped types, along with other programming entities, are checked statically. This allows programmers to identify and address code issues as early as possible during the development process.

### Dynamic/Symbolic Shapes
Choreo now supports dynamic shapes, a feature essential for machine learning kernels. To facilitate this, the input parameters of Choreo could be made **symbolic**, as shown in the example below:

```
__co__ auto matmul(f32 [M, K] lhs, f32 [N, K] rhs) { ... }
```
This feature introduces a natural way to program shaped inputs, such as tensors. Furthermore, Choreo generates runtime assertions for these dynamic shapes by leveraging the relationships (e.g., dimensional equivalence) inferred from the code. This enhancement increases the safety of runtime code and eliminates the need for many trivial explicit assertions, reducing boilerplate code.


### Visualization
Another prominent feature of Choreo is its **analytic and visualization functionality**. For example, for a DMA statement

`f1 = dma.copy a.chunkat(p, x, y) => local;`

programmers can view the data movement using Choreo's visualization capability. The result is like the image shows:

![visualizing the DMA statement](./images/simple_dma.png)

This is helpful for programmers escpecially for novices as visualization gives clear projection about the tiling behavior. Or else, programmers have to visualize in their mind, which is more error-prone.

# Documentation for Reference
Consult the [Choreo Documentation and Tutorials](http://10.31.50.149:8000/) document for information on building Choreo and the detailed usage.

