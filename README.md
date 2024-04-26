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

`f1 = dte.linear a.chunkat(p, x, y) => local;`

programmers can view the data movement using Choreo's visualization capability. The result is like the image shows:

![visualizing the DMA statement](./images/simple_dma.png)

This is helpful for programmers escpecially for novices as visualization gives clear projection about the tiling behavior. Or else, programmers have to visualize in their mind, which is more error-prone.

### Program Safty
And one of the most important feature about **code quality** is the **compile-time check** of code. In Choreo, we introduce different types, bound with corresponding shapes. The types, shapes, together with other programming entities are checked statically at compile time, which allows programmers to discover code issues as early as possible. 

## Getting Started
### How to Program with Choreo
It is good to read [Getting Started with Choreo](./Documents/getting_started_with_choreo.md). The document demonstrates the skeleton to program with Choreo.


### Build Choreo from Scratch
To build choreo, it requires to intialize the build environment first. The below command is provided:
```
make setup
```
to fetches the required bison executable (version >=3.8), FileCheck utility etc.

And if nothing is missing, build Choreo is done by single command:
```
make
```
Since Choreo is under development, performing unit tests could beneficial to avoid using incorrect version:
```
make test
```
All the unit tests are executed. And if any issue happens, the test should stop to report errors.

### Compile Choreo-C++ Program
Compile Choreo program is easy:
```
choreo your_program.co
```
