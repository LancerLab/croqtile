## **Why Symbolic Dimensions are Better Than `?` or `-1`**

### **a. Improved Readability and Clarity**

Using symbolic dimensions like `M` and `N` explicitly in the shape definition makes the program **easier to read** and understand. Instead of seeing a `?` or `-1` and wondering what the dimension is supposed to represent, the reader can immediately grasp that `M` and `N` are variables that depend on runtime values. This improves **maintainability** and reduces the cognitive load for developers who are reading and modifying the code.

For example, if a program uses `?` for dynamic dimensions, it's unclear whether it represents an uninitialized value, a wildcard dimension, or something else. This can lead to confusion when trying to reason about the program. In contrast, symbolic dimensions are explicit and show exactly how the shape is determined:

```choreo
// Using symbolic dimensions 
mdspan sp : [M, N]; 
mdspan spn : sp [1, M / 2, N / 4]; 
// Clear how the new shape is derived 

// Using `?` or `-1` 
mdspan sp : [? , ?];
```

The latter version is ambiguous and lacks the clarity that symbolic dimensions provide.

### **b. Type Safety and Compile-Time Checking**

One of the major benefits of symbolic dimensions in Choreo is that they are **type-safe** and **checked at compile-time**. Choreo allows you to declare symbolic dimensions and even manipulate them algebraically (e.g., `M / 2`, `N / 4`), ensuring that they adhere to the expected types and constraints. This eliminates the need for runtime checks, which can often lead to errors or performance penalties.

For example, using `?` or `-1` as dynamic dimensions in other systems often leads to runtime checks for consistency, such as verifying that the dimensions are properly initialized before use. These checks can introduce overhead and make the program harder to optimize. In contrast, symbolic dimensions allow Choreo to **check constraints at compile-time**, ensuring that shape expressions are valid before execution:

```choreo
mdspan sp : [M, N]; 
// M and N are symbolic dimensions 

mdspan spn : sp [1, M / 2, N / 4]; 
// Compiler ensures validity of shape expressions`
```

This ensures type consistency and reduces the potential for errors during runtime.

### **c. Expressiveness and Flexibility**

Symbolic dimensions enable a **richer** set of expressions for describing data shapes. Instead of using arbitrary placeholders like `?` or `-1`, which are limited to basic dimensionality, symbolic dimensions can be **constrained** using complex arithmetic and relationships. For example, you can directly express how one dimension depends on another, or how it is scaled or tiled, in a much more natural way.

In contrast, using `?` or `-1` often leads to brittle code, where developers need to manually calculate or infer the actual size at runtime, introducing potential errors and making the code harder to reason about.

For example, you can easily express relationships between dimensions using symbolic variables:

```choreo
mdspan sp : [M, N]; // Symbolic dimensions 
mdspan spn : sp [1, M / 2, N / 4]; // Derived shape based on symbolic expressions
```

If you were using `?` or `-1`, expressing the same relationships would require convoluted workarounds or additional runtime logic.

### **d. Compile-Time Optimization**

One of the significant benefits of symbolic dimensions is that they can help with **compile-time optimizations**. Since Choreo allows symbolic dimensions to be evaluated at compile time, it can potentially optimize memory layouts and kernel execution plans based on these dimensions. This can result in more efficient code generation and better performance compared to using `?` or `-1`, which may require runtime evaluation and dynamic memory allocation, hindering optimization opportunities.


