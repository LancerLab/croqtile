In Choreo, the `call` statement is used to invoke kernel functions defined within a kernel program (i.e., `cok`). It provides a convenient way to trigger kernel execution, passing the necessary arguments that the kernel function requires for computation. Below, we will break down the structure of a call statement and explain its key components.

### Syntax of Call Statements

The general syntax for a `call` statement in Choreo is:

```choreo
call krn-func-name <template-args> (arguments);
```


#### 1. **krn-func-name** (Kernel Function Name)
- This is the name of the kernel function that is being called.
- It corresponds to a function defined within a kernel program (or `cok`).
  
#### 2. **template-args** (Template Arguments)
- The template arguments specify the types and shapes that are passed to the kernel function. 
- These arguments are provided inside `<` and `>` brackets. In Choreo, template arguments are crucial for specifying the shapes and sizes of the data that the kernel will process.
- For example, in the statement `call matmul_kernel_sm_stationary<output.span(0)/#p, output.span(1)/#q, 32, 16, 16>(lhs_load.data, rhs_load.data, l2_out);`, `output.span(0)/#p` and `output.span(1)/#q` are template arguments defining the dimensions of the output matrix, while `32`, `16`, and `16` are fixed dimension sizes used for tiling.

#### 3. **arguments** (Kernel Arguments)
- The arguments are the actual values or data that are passed into the kernel function. 
- These arguments can be of primitive types (such as integers or floating-point values), or they can be **data types** (which represent memory references or views into data).
  
In the example call:
```choreo
call matmul_kernel_sm_stationary<output.span(0)/#p, output.span(1)/#q, 32, 16, 16>(lhs_load.data, rhs_load.data, l2_out);
```
where `lhs_load.data`, `rhs_load.data`, and `l2_out` are the arguments passed into the kernel. These are references to data stored in memory, and each of them must resolve to a data type (not a future handle).

### Arguments allowed types
Basically, kernel function can takes primitive types (s32 as int32_t, u32 as unsigned int, and so on), and data types in Choreo.
If one of the arguments involves a future handle, the syntax to extract the underlying data would be:
Instead of passing a future handle directly to the kernel, use .data to access the data contained in the future.
```choreo
lhs_load.data
```
In this case, lhs_load is a future handle (from a previous dma operation), and .data is used to access the actual data that the kernel will operate on.

### Kernel templates in Choreo
Choreo provides powerful support for kernel function templates, which allows for highly flexible and efficient execution on different hardware backends, even with backends that does not support template call (such as **Factor**).

#### Template Parameters and Requirements

When defining a kernel function, you can use template parameters to enable specialization based on the data shape, dimensions, or other factors. The template parameters in Choreo can be expressions, and the kernel can be specialized based on these parameters at compile-time. This flexibility can significantly reduce code complexity and enhance optimization.

Example Kernel Definition:
```choreo
template <typename T, int P, int Q, int M, int N, int K>
void matmul_kernel_sm_stationary(T* lhs, T* rhs, T* output) {
    // Kernel logic for matrix multiplication
}
```
In this case:

- P, Q, M, N, and K are template parameters that specify the dimensions of the matrices.
- These parameters can be evaluated symbolically at compile-time, and they allow the kernel to be highly specialized for different shapes.

### Expression support in template arguments

Choreo's template system is more flexible than standard C++ templates in the sense that it does not require constexpr or const for the template parameters. Instead, Choreo allows symbolic expressions that can be evaluated at compile-time based on the context of the program.

For instance, while traditional C++ template parameters require constexpr or constant values, Choreo allows dynamic dimensions to be part of the template parameters, as long as the dimensions are constraint-free at compile-time.

This feature is built upon Choreo comprehensive compile-time evaluation system for any expressions in program.
Choreo permits symbolic expressions that can be resolved during compilation, making it easier to handle cases where the dimensions of data are not known at the time of writing but can be inferred from the program's structure

To enable Choreo's template parameter functionality, you need to use the `-kt` compiler option, which allows the kernel function to be written as a standard C++ template function. This option removes the need for extern "C" decoration and enables more advanced template functionality, such as symbolic evaluation of template parameters.

When the `-kt` option is enabled, the kernel function should be written in standard C++ template form, without the extern "C" linkage specification.

```choreo
int M = 32;
int N = 16;
call matmul_kernel_sm_stationary<output.span(0)/#M, output.span(1)/#N, 64, 16, 16>(lhs_load.data, rhs_load.data, l2_out);
```
