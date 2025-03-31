## Overview

## Choreo Function Input
### Input Parameters
For a Choreo function, it only accepts two types of input parameters: the *spanned data*, and the *integer*.

Similar to any C++ program, the parameter may either be named or unnamed:

```choreo
__co__ void foo(f32 [7] a, int) {...}
```

In this example, the second parameter is unnamed, which makes it unable to be reference in the Choreo function. However, this is still the valid code.


### C++ Host: Data Argument with Shape
One obvious gap between C/C++ pointers/arrays and Choreo *spanned data* is that *spanned data* is shaped. Therefore, to call a Choreo function, it is necessary to convert pointers or arrays.

One direct method is to use the Choreo API provided in `choreo.h`. Though programs does not necessary to explicitly include `choreo.h` in Choreo compilation environment, it is included for every Choreo-C++ programs. 

```choreo
__co__ void foo(f32 [7, 16] input, f32 [7, 16] output) {...}

void entry(float * input, float* output) {
  // foo(input, output); <-- result in error since missing shape

  foo(choreo::make_spanned<2>(input, {7, 16}),
      choreo::make_spanned<2>(output, {7, 16}));
}
```

In the above example, we make use of `choreo::make_spanned` function template, providing the rank as template parameter, and typed pointer together with shape list as the function parameters. In this way, a shape is attached with the pointer, which makes a valid Choreo function parameter. As you may figure out from `choreo.h`, it results in `choreo::spanned_view` object, which simply wraps the C/C++ pointer/array and the array-based list. To make it easy to understand, we list simplified definition is like:

```cpp
template <typename T, size_t Rank>
class spanned_view {
  T* ptr;                 // typed pointer to the data
  size_t dims[Rank];      // ranked dimensions
};
```

It is also possible to have variable as dimensions to construct the shaped data, like following:

```choreo
__co__ void foo(f32 [7, 16] input) {...}

void entry(float * input, float* output, int M, int N) {
  foo(choreo::make_spanned<2>(input, {M, N})),
}
```
In this example, the `input` shape is runtime variable. But do not worry if wrong parameter is provided. Choreo will check if `M == 7` and `N == 16` when entering into the tileflow program. **If not, the program will terminate immediately** since all the assumptions about shapes in the Choreo function does not hold.

One important notice for the *spanned data* inputs is that they are **references to the input data/buffers**. That implies no data copy happens at invoking Choreo function, though somebody may think from the language perspective it look like copy-semantics.

## Choreo Function Output
### Return Type and Type Deduction
Similar to the input, *spanned data* and *integer* are valid output types. The *void* type is also allowed if nothing is returned.

In addition to the explicit return type annotation, Choreo support type inference on returns. For example:

```choreo
__co__ auto foo() {
  f32 [7, 16] result;
  // ...
  return result;
}
```
In this code, the return type is annotated as `auto`, which is the same keyword as C++. Choreo is capable of inferring return type as `f32 [7, 16]`. Therefore, it is recommended to use `auto` whenever possible.

### C++ Host: Shape Return

The C++ host code that receives the output of Choreo function actually receives a buffer. Semantics, the return must be copied than reference, or else it returns a reference to an object allocated in the called function. For example:

```choreo
__co__ auto foo() {
  f32 [7, 16] output;
  // ...
  return output;  // semantically copy
}

void entry() {
  auto result = foo(); // move the 'spanned_data' to caller
  // ...
}
```

In this example, the `output` is semantically copied to the caller. However, in Choreo's implementation, no copy happens. The returned type is `choreo::spanned_data`, where its simplified definition is as following:

```cpp
template <typename T, size_t Rank>
class spanned_data {
  std::unique_ptr<T[]> ptr; // move only pointers
  size_t dims[Rank];        // ranked dimensions
};
```

In contrary to `choreo::spanned_view`, `choreo::spanned_data` owns the buffer it points to. Thus there is **no copy actually happen when returning from a Choreo function**. In compiler's term, this is called the *Return Value Optimization*.

You may use member function like `.rank()`, `.shape()` to query shape related information of Choreo return. This allows you able to work without knowing the too much detail about Choreo function.

## More Questions
Like C++ functions, Choreo functions can be defined anyway in the source code. one common raised question is about the **function signature**, as long as programmers may want to link the module with Choreo function definition.

The answer is that Choreo function follows the C++ calling convention, which makes it standard C++ function with C++ name mangling. Note that **`extern "C"`** is not possible for a Choreo function, since it utilize C++ template-based object as its parameter type (spanned data).

However, there are some features yet to support:

- Choreo function declaration without definition. This allows calling Choreo function from another module.
- Choreo function with `inline`. This allows the Choreo function be used in headers.

