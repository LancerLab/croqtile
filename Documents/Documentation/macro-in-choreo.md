## Overview
Macros have long been a fundamental feature in C/C++ programming. To enable seamless manipulation of both C++ and *tileflow* programs, macro expansion is essential. This section demonstrates Choreo's macro processing capabilities.

## Object-Like Macros
Choreo preprocessor supports C/C++ **object-like macros** to connect both host code and tileflow code. The *object-like macro* performs pure text replacement and accepts no parameters. The below code gives an example:

```choreo
#define M 256
#define N 32
#define K 64

__co__ auto matmul(f32 [M, N] lhs, f32 [N, K] rhs) { /*...*/ }

void foo() {
  choreo::f32 a[M][K];
  choreo::f32 b[N][K];
  // ...
  auto res = matmul(choreo::make_spanview<2>(a, {M, K}),
                    choreo::make_spanview<2>(b, {N, K}));
}

```
In the code snippet, the inputs of choreo function `matmul` are not dynamically shaped. Instead, Choreo pre-processor substitutes `M`, `N`, `K` with values `256`, `32`, `64` ahead of Choreo compilation, which produces tileflow function with statically shaped inputs. In this way, programmers makes different code consistent and easy to manipulate.

Note that, till now Choreo does only support macros with parameters. It means code like:
```cpp
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
```
would not work.

## Comments
Choreo support C-style comments, either `/*...*/` or `//...`, leveraging the capability choreo preprocessor has provided.

## Conditional Compilation
Choreo also support conditional compilation which C/C++ programs used a lot. This includes `#if`/`#ifdef`/`#ifndef`/`#else`/`#endif`. The below code snippet showcases the usage:

```choreo
#define PATH0
// some host code
__co__ foo() {
#ifdef PATH0
// some code related to PATH0
#else
// other code
#endif
}

// host control
#ifdef PATH0
// ...
#else
// ...
#endif
```
In this way, it also make host and choreo code be controlled within the same preprocess method.

Note that, the capability of choreo preprocessor is still enhancing. But it is likely that we would not implement full C preprocessing support. Choreo would not pick up existing C features unless people find it is necessary.

## Difference with C++ pre-processing
One important notices about choreo preprocessing is that it is triggered much earlier than c++ preprocessing. The workflow of choreo compilation is as shown below:

```
chore-preprocessing -> choreo compilation -> c/c++ preprocessing -> c/c++ compilation
```

The primary target of choreo preprocessing is to make host/device macros work as a whole, but such a workflow makes it possible sometimes different. From an implementation perspective, Choreo pre-processor only substitute/conditionally-compile code inside the tileflow function, while leaving other pre-processing to the C++ preprocessor. That could restrict Choreo pre-processing in a limited scope.

## Pre-defined Macros
To mimic a target native compilation, choreo preprocess also takes the builtin macros from the target. For example, `__CUDA__` is globally defined to generate CUDA/Cute code, while `__CUDA_ARCH__` is only set for CUDA/Cute device code compilation.
Consequently, these macros can be utilized inside tileflow functions as well as the host code.
