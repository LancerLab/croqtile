### Data in Choreo

In Choreo, **data**  are tightly integrated with the shape system. To efficiently manage data in high-performance environments, Choreo uses **composite types** that combine fundamental types and shapes (mdspan). This allows for precise control over multi-dimensional data, which is particularly important for performance-sensitive applications like machine learning and scientific computing.

#### 1. Type of Data

In Choreo, **types** of data or so-called **data types** are composed of two parts: the **fundamental type** and the **mdspan** (multi-dimensional span). A **spanned type** is a **composite type**, consisting of a fundamental type (e.g., `f32` for floating-point numbers) and an `mdspan` that describes the shape of the data.

Neither the fundamental type nor the mdspan alone can represent complete data with storage; together, however, they form a **partial type** that provides more control over data representation.

The decision to use composite types (fundamental types + mdspan) stems from the need to manipulate multi-dimensional arrays efficiently. Data in fields like machine learning is often organized in high-dimensional arrays, and efficient handling of these shapes is critical for performance.

The **mdspan** type allows you to manipulate the shape of data independently from its contents, making it easier to work with complex data structures, such as tiled or blocked data. This separation of concerns ensures that programmers can focus on optimizing shapes without worrying about the underlying memory storage.

#### 2. Example of a Spanned Type in Choreo

Here is how you define a composite type in Choreo:

```choreo
mdspan sp : [M, N]; // Symbolic dimensions 
f32 [10, 10] d0; // 2D array with fixed shape 
f16 [ndim] d1; // Array where ndim is a symbolic value
```

In this example:

- `f32 [10, 10] d0` defines a 2D array of `f32` elements with a fixed shape.
- `f16 [ndim] d1` defines a 1D array where the number of elements depends on the symbolic value `ndim`.

Such a design allows for **flexible** and **efficient** management of multi-dimensional data while maintaining type safety and clarity.

#### 3. Defining Data in Choreo

To define data for computation in Choreo, you must fully specify both the fundamental type and the shape. This ensures that data types are well-structured and optimized for performance.

```choreo
ndims : [20, M, N];
f32 [10, 10] d0;
f16 [ndims] d1; // Uses the defined mdspan as dimension
```

In this case:

- `d0` is a 2D array of `f32` elements.
- `d1` is a 3D array of `f16` elements, with dimension [20, M, N].

Choreo supports multiple fundamental types, including:

- **Unsigned Integers**: `u8`, `u16`, `u32`
- **Signed Integers**: `s8`, `s16`, `s32`
- **Floating Points**: `f16`, `bf16`, `f32`

#### 4. Full Typing and Memory Management

In Choreo, **memory management** is tightly coupled with the type system. The **mdspan** type is not used in isolation; it always works in conjunction with a fundamental type. This ensures that both shape and memory layout are well-defined and optimized for computation.

This integration of shape and memory management is crucial for high-performance applications where memory locality, tiling, and data movement across storage hierarchies (e.g., from main memory to cache) play a significant role in optimizing performance.

## *mdspan* in Choreo Function Definition
As described earlier, Choreo accept data as its parameters. However, in constrast to C++ program which accepts raw pointers or array, Choreo requires data parameters to be shaped. Therefore, the shape is necessary for an Choreo function inputs and output.

```choreo
__co__ auto shaped_inputs(f32 [
```
