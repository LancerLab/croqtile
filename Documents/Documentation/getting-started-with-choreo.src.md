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

!INCLUDE_IF_EXISTS "target/env_setting_up.md"

For development and testing purposes, the Makefile provides additional utilities:
```
make help
```
displays available build and test targets. To test elementwise operators in the samples:
```
make sample-test
```
runs all elementwise operator tests, while:
```
make sample-test-operator OPERATOR=add
```
tests a specific operator (e.g., add, mul, relu, sigmoid, softmax, tanh).


## Compile Choreo-C++ Program
In the current implementation, Choreo performs **source-to-source translation** (or **transpilation**) to convert *Choreo-C++* programs into vendor-supported C++ language code and APIs (such as CUDA/Cute, and more).

However, since Choreo integrates lower-level *target compiler* in its compilation process, it appears as an **end-to-end compiler** when the vendor-provided device-level C++ compiler is properly configured.

Therefore, Once Choreo is built, developers can compile Choreo-C++ programs into various output forms, including:

- Target source code
- Target object module
- Target executable binary/module
- Target assembly
- Work-script

Species except for *target source code* and *work-script* (introduced later) are similar to those of `gcc` and `clang`. However, the availability of these output forms depends on the target platform's support and limitations.

The usage of the Choreo-C++ compiler is similar to that of `gcc` or `clang`. For example:
```
choreo your_program.co
```
This command generates an `a.out` executable file. The `-o <filename>` option can be used to specify the output filename when needed. Compiler options like `-c` and `-S` work similarly to those in C++ compilers, provided they are supported by the target platform.

The `-t <platform>` option allows you to specify the target platform for the compilation. Additionally, the `-es` option generates *target source code* without performing the "target compilation*. For instance:
```bash
choreo -t cuda your_program.co -es -o cuda_source.co
```
This command produces CUDA C++ source code, which can be useful for specific development tasks.

Notably, Choreo allows the `-E` option to support Choreo-only preprocessing. The Choreo preprocessor handles simple macros and preprocessor directives such as `#if`, `#ifdef`, `#ifndef`, `#else`, and `#endif`, enabling Choreo functions to be integrated with other C++ code. The `-E` option outputs the preprocessed code, for example, removing code within `#if 0` and `#endif` directives inside Choreo functions.

Furthermore, in development scenarios, Choreo can generate *work-script* (using the `-gs` option) to drive further low-level compilation and execution. This facilitates the development process, as many scripts are integrated for easy debugging.

Lastly, options `--help` and `--help-hidden` are available for listing the full option set. Programmers and users can check the list to find their appropriate usage.

Here is a example output of `--help`:

```bash
// --help
Usage: choreo [options] file...
Options:
  --help                    Display this information.
  --help-hidden             Display hidden options.
  -e, --dump-ast            Dump the Abstract Syntax Tree (AST) after parsing.
  -i, --infer-types         Show the result of type inference.
  -bf16n, --native-bf16     Utilize native bf16 type when target platform support.
  -f16n, --native-f16       Utilize native f16 type when target platform support.
  -n, --remove-comments     Remove all comments in non-choreo code. (Useful for FileCheck)
  -t, --target <platform>   Set the compilation target. Use '--help-target' to show current supported targets.
  -v, --verbose             Display the programs invoked by the compiler.
  -E                        Preprocess only; do not compile.
  -arch=<processor>         Set the architecture to execute the binary code.
  -c                        Compile choreo code and the generated target code; Without linking.
  -es                       Emit target source file without target source compilation.
  -gs                       Generate target script.
  -o <file>                 Place the output into <file>.
```
