# Build Choreo From Source Code and Compile Choreo Program
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
Once Choreo is built, developers can compile Choreo-C++ programs into various output forms, including:

- Target source code
- Target object module
- Target executable binary/module
- Target assembly

The availability of these output forms depends on the target platform's support and limitations.

The usage of the Choreo-C++ compiler is similar to that of _gcc_ or _clang_. For example:
```
choreo your_program.co
```
This command generates an _a.out_ executable file. The _-o <filename>_ option can be used to specify the output filename when needed. Compiler options like _-c_ and _-S_ work similarly to those in C++ compilers, provided they are supported by the target platform.

The _-t <platform>_ option allows you to specify the target platform for the compilation. Additionally, the _-es_ option generates target source code without performing the compilation. For instance:
```
choreo -t cude your_program.co -es -o cuda_source.co
```
This command produces CUDA C++ source code, which can be useful for specific development tasks.

Notably, Choreo includes the _-E_ option to support Choreo-only preprocessing. The Choreo preprocessor handles simple macros and preprocessor directives such as #if, #ifdef, #ifndef, #else, and #endif, enabling Choreo functions to be integrated with other C++ code. The _-E_ option outputs the preprocessed code, for example, removing code within #if 0 and #endif directives inside Choreo functions.

Furthermore, in development scenarios, Choreo can generate scripts (using the -gs option) to drive further low-level compilation and execution. This facilitates the development process, as many scripts are integrated for easy debugging.

Lastly, options _--help_ and _--help-hidden_ are avaiable for listing the full option set. Programmers and users can check the list to find their appropriate usage.

