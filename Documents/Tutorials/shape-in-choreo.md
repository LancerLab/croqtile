## Abstractions and Syntax in Choreo Function

### Shape in Choreo

In Choreo, shapes are primarily represented using the `mdspan` type, which is a fundamental concept for handling multi-dimensional data. The **shape** in Choreo is defined by the dimensions of data, and this is crucial for efficiently organizing and processing large datasets, especially in high-performance computing scenarios.

#### 1. Defining Shapes

In Choreo, shapes are defined using the `mdspan` keyword, which can represent multi-dimensional data. The `mdspan` itself is a **partial type**, meaning it cannot store data by itself but rather describes the shape of the data. A shape can be a single dimension or multiple dimensions, depending on how the data is structured.

For example:

```cpp
mdspan sp : [7, 8]; // Defines a 2D shape with dimensions [7, 8]
```

In this example, `sp` is a 2D `mdspan` that represents a data structure with 7 rows and 8 columns.

#### 2. Operations on Shapes

Once a shape is defined using `mdspan`, you can manipulate it using various operations. For instance, you can derive new shapes by applying arithmetic operations on the dimensions. This is useful for operations like **tiling** or **blocking**, which are common in high-performance computing.

For example, you can derive a new shape from an existing one:

```cpp
mdspan sp : [M, N]; // Initial shape 
mdspan spn : sp [1, M / 2, N / 4]; // Derived shape, [1, M/2, N/4]
```

Here, the new shape `spn` is derived from `sp`, with each dimension calculated symbolically based on `M` and `N`. This allows for the easy application of tiling or other data reorganization patterns.

#### 3. Type of Shape

In Choreo, **shape** is closely tied to the concept of `mdspan`, which is a core element of choreo's **type system**, Rather than directly describing an array's size using traditional dimensions like `[7][8]`, the shape is abstracted into a more flexible form using `mdspan`. This allows developers to define complex data shapes in a highly modular and expressive way.

Specifically, an `mdspan` (multi-dimensional span) is a type that **describes the shape** of multi-dimensional data but **does not itself hold any data**. Instead, it provides a view of the data, specifically focusing on its dimensionality and layout. This allows Choreo programs to operate on data shapes independently of the fundamental data type, facilitating high-level optimizations like tiling and data movement across different memory hierarchies. The shape is tightly coupled with the fundamental type of the data, such as `f32`, `s32`, or `f16`. While `mdspan` alone is a **partial type**, it is used in combination with a fundamental type to form a **complete type**, such as `f32[7, 8]`, representing an array of `f32` with shape `[7, 8]`.

This integration ensures that the shape and type of the data are explicitly defined, improving clarity and enabling more effective optimizations by the compiler. This separation of shape and data type allows the program to focus on the organization and structure of data, especially in scenarios like **loop tiling** and **memory hierarchy optimization**.

#### 4. **Symbolic Dimensions in Choreo: A Better Approach to Dynamic Shapes**

One of the key features of Choreo’s shape system is the ability to define **symbolic dimensions**. These are dimensions that depend on runtime values and can be constrained by expressions, providing more flexibility than static or unknown dimensions (e.g., `?` or `-1`).

For instance:

```cpp
int M = 6, N = 8; // Runtime values 
__co__ func(int M, int N) {
    mdspan sp : [M, N]; // Define a 2D shape with symbolic dimensions M and N
    mdspan spn : sp [1, M / 2, N / 4]; // Derived shape is [1, 3, 2]
}
```

Here, the dimensions of `sp` depend on the values of `M` and `N`, which are determined during runtime. This is more expressive than using `?` or `-1` as dynamic dimensions, as it allows for better control over the relationship between dimensions.

In Choreo, symbolic dimensions provide a robust and flexible mechanism for working with dynamic shapes in multi-dimensional arrays. Instead of relying on placeholders like `?` or `-1` commonly used in other systems to represent dynamic dimensions, Choreo uses **symbolic variables** to directly express **runtime-dependent shapes**. This offers significant advantages in terms of clarity, expressiveness, and type safety. Let’s explore how symbolic dimensions work in Choreo and why they are preferable to traditional approaches using `?` or `-1`.

Choreo’s approach eliminates this ambiguity by allowing **symbolic dimensions** to be defined using runtime values. These symbolic dimensions can then be constrained and manipulated within the program.

In this example, `M` and `N` are symbolic dimensions that depend on runtime values. This means that the shape of the `mdspan` `sp` is determined at runtime based on these values, and the new shape `spn` is derived from `sp` using symbolic expressions. Notice how `M / 2` and `N / 4` directly constrain the new shape, making it clear how the dimensions relate to each other and how they will evolve at runtime.

#### **5. Why Symbolic Dimensions are Better Than `?` or `-1`**

##### **a. Improved Readability and Clarity**

Using symbolic dimensions like `M` and `N` explicitly in the shape definition makes the program **easier to read** and understand. Instead of seeing a `?` or `-1` and wondering what the dimension is supposed to represent, the reader can immediately grasp that `M` and `N` are variables that depend on runtime values. This improves **maintainability** and reduces the cognitive load for developers who are reading and modifying the code.

For example, if a program uses `?` for dynamic dimensions, it's unclear whether it represents an uninitialized value, a wildcard dimension, or something else. This can lead to confusion when trying to reason about the program. In contrast, symbolic dimensions are explicit and show exactly how the shape is determined:

```cpp
// Using symbolic dimensions 
mdspan sp : [M, N]; 
mdspan spn : sp [1, M / 2, N / 4]; 
// Clear how the new shape is derived 

// Using `?` or `-1` 
mdspan sp : [? , ?];
```

The latter version is ambiguous and lacks the clarity that symbolic dimensions provide.

##### **b. Type Safety and Compile-Time Checking**

One of the major benefits of symbolic dimensions in Choreo is that they are **type-safe** and **checked at compile-time**. Choreo allows you to declare symbolic dimensions and even manipulate them algebraically (e.g., `M / 2`, `N / 4`), ensuring that they adhere to the expected types and constraints. This eliminates the need for runtime checks, which can often lead to errors or performance penalties.

For example, using `?` or `-1` as dynamic dimensions in other systems often leads to runtime checks for consistency, such as verifying that the dimensions are properly initialized before use. These checks can introduce overhead and make the program harder to optimize. In contrast, symbolic dimensions allow Choreo to **check constraints at compile-time**, ensuring that shape expressions are valid before execution:

```cpp
mdspan sp : [M, N]; 
// M and N are symbolic dimensions 

mdspan spn : sp [1, M / 2, N / 4]; 
// Compiler ensures validity of shape expressions`
```

This ensures type consistency and reduces the potential for errors during runtime.

##### **c. Expressiveness and Flexibility**

Symbolic dimensions enable a **richer** set of expressions for describing data shapes. Instead of using arbitrary placeholders like `?` or `-1`, which are limited to basic dimensionality, symbolic dimensions can be **constrained** using complex arithmetic and relationships. For example, you can directly express how one dimension depends on another, or how it is scaled or tiled, in a much more natural way.

In contrast, using `?` or `-1` often leads to brittle code, where developers need to manually calculate or infer the actual size at runtime, introducing potential errors and making the code harder to reason about.

For example, you can easily express relationships between dimensions using symbolic variables:

```cpp
mdspan sp : [M, N]; // Symbolic dimensions 
mdspan spn : sp [1, M / 2, N / 4]; // Derived shape based on symbolic expressions
```

If you were using `?` or `-1`, expressing the same relationships would require convoluted workarounds or additional runtime logic.

#### **6. Compile-Time Optimization**

One of the significant benefits of symbolic dimensions is that they can help with **compile-time optimizations**. Since Choreo allows symbolic dimensions to be evaluated at compile time, it can potentially optimize memory layouts and kernel execution plans based on these dimensions. This can result in more efficient code generation and better performance compared to using `?` or `-1`, which may require runtime evaluation and dynamic memory allocation, hindering optimization opportunities.

### Data in Choreo

In Choreo, **data**  are tightly integrated with the shape system. To efficiently manage data in high-performance environments, Choreo uses **composite types** that combine fundamental types and shapes (mdspan). This allows for precise control over multi-dimensional data, which is particularly important for performance-sensitive applications like machine learning and scientific computing.

#### 1. Type of Data

In Choreo, **types** of data or so-called **data types** are composed of two parts: the **fundamental type** and the **mdspan** (multi-dimensional span). A **spanned type** is a **composite type**, consisting of a fundamental type (e.g., `f32` for floating-point numbers) and an `mdspan` that describes the shape of the data.

Neither the fundamental type nor the mdspan alone can represent complete data with storage; together, however, they form a **partial type** that provides more control over data representation.

The decision to use composite types (fundamental types + mdspan) stems from the need to manipulate multi-dimensional arrays efficiently. Data in fields like machine learning is often organized in high-dimensional arrays, and efficient handling of these shapes is critical for performance.

The **mdspan** type allows you to manipulate the shape of data independently from its contents, making it easier to work with complex data structures, such as tiled or blocked data. This separation of concerns ensures that programmers can focus on optimizing shapes without worrying about the underlying memory storage.

#### 2. Example of a Spanned Type in Choreo

Here is how you define a composite type in Choreo:

```cpp
mdspan sp : [M, N]; // Symbolic dimensions 
f32 [10, 10] d0; // 2D array with fixed shape 
f16 [ndims] d1; // Array where ndims is a symbolic value
```

In this example:

- `f32 [10, 10] d0` defines a 2D array of `f32` elements with a fixed shape.
- `f16 [ndims] d1` defines a 1D array where the number of elements depends on the symbolic value `ndims`.

Such a design allows for **flexible** and **efficient** management of multi-dimensional data while maintaining type safety and clarity.

#### 3. Defining Data in Choreo

To define data for computation in Choreo, you must fully specify both the fundamental type and the shape. This ensures that data types are well-structured and optimized for performance.

```cpp
ndims : [20, 15]; f32 [10, 10] d0; f16 [ndims] d1; // Uses symbolic dimension `ndims`
```

In this case:

- `d0` is a 2D array of `f32` elements.
- `d1` is a 1D array of `f16` elements, with the number of elements determined by the symbolic dimension `ndims`.

Choreo supports multiple fundamental types, including:

- **Unsigned Integers**: `u8`, `u16`, `u32`
- **Signed Integers**: `s8`, `s16`, `s32`
- **Floating Points**: `f16`, `bf16`, `f32`

#### 4. Full Typing and Memory Management

In Choreo, **memory management** is tightly coupled with the type system. The **mdspan** type is not used in isolation; it always works in conjunction with a fundamental type. This ensures that both shape and memory layout are well-defined and optimized for computation.

This integration of shape and memory management is crucial for high-performance applications where memory locality, tiling, and data movement across storage hierarchies (e.g., from main memory to cache) play a significant role in optimizing performance.
