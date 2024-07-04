# Choreo - An eDSL for TileFlow Programming
Choreo is a low-level embedded Domain Specific Language (eDSL) specifically engineered to program data movement entities like DMAs, which traditionally focus on hardware configuration rather than the data itself. This language facilitates the programming of data flow across a hardware’s memory hierarchy, where at each level, data is partitioned into smaller chunks that are positioned closer to the processing elements. By making data tiling and flowing more manageable, Choreo pioneers the 'TileFlow' programming paradigm. This new approach aims to significantly streamline the engineering tasks involved, particularly for professionals developing AI operations on specific hardware.

## Choreo and Device-Level C++
Choreo has close relationship to the device-level C++ programming, which is usually provided by the hardware vendor. In current implementation, Choreo performs source-to-source translation to convert *choreo-c++* programs to vendor-supported(like factor, topscc) C++ language entities and APIs. Despite of turning higher level abstraction of *TileFlow* functions into corresponding low level C++API calls, Choreo also glues the kernels, the host program into executables. Consequently, using Choreo compiler is easy when the vendor-provided compiler is properly configured.

## Features
### Productivity
One of the attempting feature of Choreo by design is the **improve productivity**. Specifically, it is about **the mind-set saving** when dealing with **data tiling** work. This is done by introducing programming entities including *mdspan*/*ituple*, which makes the manipulation of data shapes as easy as if it is in *Python*-level. For example:
```
__co__ f32 mdspan<2> my_function(f32 [8, 4, 12] d) {
    block_shape : d.span { (0)/ 2, (1)/ 4, 1, (2)};
}
```
Programmers can easily make a shape with a tiling factor of {2, 4, 1} from data 'd''s shape. In addition, the programmer adds one dimension to the tiled 'block_shape'. All these work are done in one line of code. It saves mind-set since neither the trivial computation is necessarily done by human, nor tidious C++ programming is required.

### Visualization
Another prominent feature of Choreo is its **analytic and visualization functionality**. For example, for a DMA statement

`f1 = dma.copy a.chunkat(p, x, y) => local;`

programmers can view the data movement using Choreo's visualization capability. The result is like the image shows:

![visualizing the DMA statement](./images/simple_dma.png)

This is helpful for programmers escpecially for novices as visualization gives clear projection about the tiling behavior. Or else, programmers have to visualize in their mind, which is more error-prone.

### Program Safty
And one of the most important feature about **code quality** is the **compile-time check** of code. In Choreo, we introduce different types, bound with corresponding shapes. The types, shapes, together with other programming entities are checked statically at compile time, which allows programmers to discover code issues as early as possible.

# Getting Started
## How to Program with Choreo
It is good to read [Getting Started with Choreo](./Documents/getting_started_with_choreo.md). The document demonstrates the skeleton to program with Choreo.


## Build Choreo from Scratch
### Prerequisitions
- GCC >=8.1 or clang >=6, where c++17 is fully supported.
- Bison >=3.8, Flex >=2.6.4, where c++ features are supported.

### Environment
To streamline the configuration of the Choreo build environment, developers can currently utilize the following command:
```
make setup
```
This command retrieves the essential software prerequisites, such as the flex executable, bison executable (version 3.8 or higher), FileCheck utility, and gtest source code, among others. This process enables the building and testing of Choreo.

Once the environment configuration is complete, Choreo can be built using the command:
```
make
```
Given that Choreo is still in development, running unit tests is crucial to prevent using corrupted version:
```
make test
```
In the event of any issues, the test should halt and report errors.

To run the compiled Choreo program on real hardware like GCU, the current Makefile can assist with the environment configuration. The command:
```
make setup-gcu2
```
sets up the GCU-2.x compiler and runtime environment. Additionally,
```
make kmd-gcu2
```
helps configure the GCU-2.x hardware driver.

Similarly, the commands:
```
make setup-gcu3
```
and
```
make kmd-gcu3
```
assist in setting up the GCU-3.x compiler, runtime, and hardware driver.


## Compile Choreo-C++ Program
Once choreo is built, developers can compile Choreo program with the following command:
```
choreo your_program.co
```
Despite of generating target source code, in current implementation, choreo also generates scripts to drive further low-level compilation and execution invokation.
