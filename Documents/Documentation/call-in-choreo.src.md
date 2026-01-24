## Overview
In current implementation of Choreo functions, it invokes *device function* to implement the specified target computations. In this section, we will learn the detail about the *call statement*s in Choreo.

## Call Device Functions

To call a device function in the Choreo function, it requires the call statement explicitly led by `call`.

### Basic Syntax of Call Statements

The general syntax for a `call` statement in Choreo is:

```choreo
call func-name <optional-template-args> (arguments);
```
Here, the keyword `call` is followed by a function name `func-name`, an optional `<>` enclosed template arguments, which are comma-separated. The `arguments` are also comma-separated and listed inside `()` like normal C/C++ functions.

Note that **Choreo transpilation process would not apply any check between the callers and the callees**, including the function existence, function signature consistency, and parameter consistency. It delegates such duties to the target compilation process. Programmers must be careful about the conventions between Choreo function and device function consequently.

Beside that, in all the currently supported platform, such calls must be made inside `parallel-by`, since it calls device function which only runs on heterogeneous hardware. Therefore, in the following example, `call bar();` in `mou` is illegal code since it tries to call device function in a code location which is assumed to be host code area.

```choreo
__co__ void foo() {
  parallel p by 1 {
    call bar();  // ok
  }
}
__co__ void mou() {
  call bar();  // error: not able to call device function from host
}
```

### The Convention: Allowed Argument Types
In current implementation, the argument must be either:

- *Spanned Data* type, or
- Scalar *Integer* type, or
- Scalar *Floating-Point* type.

Here, the scalar *integer* type only includes `int`, whereas the scalar *floating-point type* includes `float`, `double`, `half`, `bfp16`, and `half8`. However, since the target-platform differs in floating-point support, the *floating-point type* parameter also varies for different target platforms.

In addition, **operations on floating-point are not supported** by Choreo till now, considering the floating is not useful for tileflow programs. Instead, they are typically values for device computation, which the *tileflow program*s pass such values directly to device kernels.

The below code showcases an example:

```choreo
__device__ void bar(float *p, int m, int n) {}
__device__ void foo(float *, int, unsigned, double, float) {}

__co__ void foobar(f32 [M, 24] input, int N, float padding) {
  parallel p by 1 {
    shared f32 [14, 7] buffer;
    call bar(input.data, M, buffer.span(0));
    call foo(buffer.data, N, 3, 3.14, padding);
  }
}
```

In this example, it passes different data types as arguments from Choreo function to the device functions, which matches the device parameters exactly. Note that in the device function, the **corresponding parameter type of the *spanned data* argument is simply the pointer of its _element type_**, where the shape information is dropped. For example, the `foo`'s argument `p` is `float*`, which corresponds the *spanned data* argument `input.data`. In Choreo, we names the parameter as a **decayed** pointer of the *spanned data*. If using other form like `float p[]`, it will trigger failure in target compilation stage.

For some types like `f16`, and `bf16`, there may not be native target support of such types, it is possible to utilize `choreo::f16` and `choreo::bf16` to handle such types.

## Instantiate the Function Template for Call

### Trigger C++ Template Instantiation

It is possible for Choreo code to make function call to a template function. For example:

```choreo
template<int M, int N, int K>
__device__ void matmul_kernel(int *lhs, int* rhs, int* output) {}

__co__ auto matmul(s32 [96, 72] lhs, s32 [72, 24] rhs) {
  s32 [lhs.span(0), rhs.span(1)] output;
  parallel p by 6 {
    shared s32 [output.span / #p] buffer;   // shape: [16, 4]
    lhs_load = dma.copy lhs.chunkat(p, _) => shared;
    rhs_load = dma.copy rhs.chunkat(_, p) => shared;
    call matmul_kernel<buffer.span(0), buffer.span(1), 72>(lhs_load.data, rhs_load.data, buffer);
  }
  return output;
}
```
In this case, it defines device function template named `matmul_kernel`, which takes three template parameters. As the caller specified the template function arguments, it triggers the instantiation of the function template, which results in a template function `matmul<16, 4, 72>` for the call.

So in Choreo, calling a template function is similar to those in C++. However, **the template argument passed must be able to be inferred as compile time constant value** by Choreo compiler. Therefore, any runtime values can result in error.

```choreo
__co__ void foo(int M) {
  parallel p by 1 {
    call bar<M>();  // error: 'M' is a runtime value
  }
}
```

Choreo compiler could inference values as much as possible at compile time. If the template argument can not be inferred, the compilation will abort and error will be emitted.

!INCLUDE_IF_EXISTS "target/calls.md"

### Quick Summary
In this section, we learned the syntax to invoke device functions, including normal ones and those triggers function template instantiations. Programmers must code in cautious since no check is applied at transpilation time, which may result in more confusing error reporting since the check is applied at target compile time with generated functions.
