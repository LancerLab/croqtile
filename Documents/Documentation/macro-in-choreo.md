# Macros and Preprocessing in Choreo
Choreo allows simple C/C++ macros be effective both for host code and for choreo code. One essential preprocessor capability choreo has provided is to pass the value of value-based macros defined in host code into the choreo function.

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
In the above code snippet, the inputs of choreo function 'matmul' are not dynamically shaped since the choreo pre-processor substituate 'M', 'N', 'K' to be the values of '256', '32', '64' ahead of choreo compilation. The benefit of passing host macros into choreo code is obvious. The data used in host code can be easily made consistent with the user choreo function.

However, till now choreo only support 'simple' valued macros. Code like:
```cpp
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
```
is not supported by choreo preprocessor.

In addition, choreo support C-style comments, either '/*...*/' or '//...', leveraging the capability choreo preprocessor has provided. Moreover, C preprocessing directives including '#if/#ifdef/#ifndef/#else/#endif' are also supported by choreo preprocessor. Code snippet in the below showcases the usage.

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
In this way, it also make host and choreo code be controled within the same preprocess method.

Note that, the capability of choreo preprocessor is still enhancing. But it is likely that we would not implement full C preprocessing support. Choreo would not pick up existing C features unless people find it is necessary.
