## Overview
In this section, you will learn how to define *integer*s and integer-tuples (*i-tuple*s) in Choreo code and understand their usage.

## Integers for Program Control
In high-performance computation kernels, integers are typically used for program control rather than computation. In Choreo *tileflow programs*, integers are used exclusively for loop control and array indexing.  Given the limited value ranges in these scenarios, Choreo has provided a single integer type -- the *32-bit signed integer* -- to simplify the design for this domain-specific use.

In Choreo, an **integer**-typed variable is defined similar to C/C++:

```choreo
int a = 1;
```

However, since Choreo can infer the type from the *initialization expression*, you can also define an integer variable in another style:

```choreo
a = 1;
```

Similar to C/C++, you can define multiple integers in a single line:

```choreo
int a = 1, b = 2, c = 3;
d = 4, e = 5;
```

In Choreo, the *mdspan*-typed shape can not be used as the function parameter. However, integers can be passed from the host to the Tileflow program, and from the Tileflow program to the device:

```choreo
__co__ void foo(int a) {
  // ...
  call kernel(a, 3);
}
```

Unlike C/C++, Choreo does not allow the declaration of integers without initialization or the reassignment of an integer. For example:

```choreo
int a;  // error: declaration without initialization
int b = 1;
b = 3;  // error: re-assignment
```

This enforces that **integers must be initialized at declaration and cannot be reassigned**. In some contexts, these integers are referred to as "**immutable**" or "**constant**".

## Group of Ranges and Group of Integers
In Choreo, *mdspan* can be used to represent dimensional shapes. However, as introduced, *mdspan* stands for **Multi-Dimensional Span**. The elements inside are *Dimensional Spans*, which imply ranges of values.

For example, an *mdspan* `[7, 14]` represents a group of two ranges `[0, 7), [0, 14)`. Therefore, an *mdspan* does not represent a list of integer values. However, grouping integer values is useful in certain scenarios:

- When considering the delta of two mdspans, multiple integers are needed. In high-level semantics, the delta may reflect padding differences when adjusting shape dimensions.
- When tiling a shape, the multiple tiling factors are integers, not ranges.

These situations motivate Choreo to have a type for grouping integer values.

## Group Integers as Integer-Tuple (I-Tuple)
### Defining I-Tuples
In Choreo, multiple integers can be grouped into an **integer-tuple**, or **i-tuple** (or **ituple**). The keyword `ituple` can be used to define an *i-tuple*. The below code showcases the usage:

```
ituple a = {1, 2, 3};
b = {4, 5, 6};  // utilize the type inference
```

Since Choreo can infer types from the *initialization expression*, programmers can often omit the `ituple` keyword. However, without explicit type annotation, an *ituple* variable definition might look similar to an *mdspan* definition if you are not yet familiar with Choreo. To distinguish them:

- The *initialization expression* of an *ituple* follows an assignment operation `=` (like integers), whereas an *mdspan* is initialized after `:`;
- The *initialization expression* of an *ituple* is enclosed by `{}`, not `[]` as for *mdspan*.

Similar to `mdspan`, you may enforce rank check for *ituple*s at compile time:

```
ituple<3> a = {1, 2};  // error: inconsistent rank
```

### Operations on I-Tuples
Operations on *i-tuple*s are similar to those on *mdspan*. You can either use the *element-of* operation `()` to retrieve the element values or use *ituple* as a whole:

```choreo
a = {3, 4};
b = {a(0), 1, a(1)};  // '()' to retrieve the element value
c = a {(0), (1), 2};  // syntax sugar
d = {a, 5, 6};        // concatenate
e = a + 1;            // addition is applied elementwise
```

Note that, similar to *integer* and *mdspan*, **an *ituple* variable must be initialized and cannot be reassigned**.

In practice, *ituple*s are often used with *mdspan*s. For example:

```choreo
shape : [7, 18, 28];
tiling_factors = {1, 2, 4};
tiled_shape : shape / tiling_factors;
padded_shape : shape + {2, 0, 2};
```

In this code, a mdspan-typed shape is divided by an ituple-typed tiling factor to derive a `tiled_shape`. Additionally, a `padded_shape` is derived from the addition of the initial `shape` and an anonymous *ituple*. Note that when a *mdspan* is added by an *ituple*, their rank must be consistent. Or else, an error will occur.

```choreo
shape : [7, 8, 9] + {1, 2}; // error: inconsistent rank
```

### Evaluation of Integers and I-Tuples
Similar to mdspan, both *integer*s and *i-tuple*s incur minimal runtime cost. Their values are evaluated at compile time whenever possible, so programmers can generally ignore their cost.

### Look-Ahead: Bounded I-Tuples
*I-tuples* are not frequently used in real Choreo code. However, their variants, **bounded i-tuple**s, which bind an *i-tuple* with an *mdspan* to indicate its possible range, are essential for constructing Choreo loops. We will cover *bounded i-tuples* in more detail in later chapters.

## Quick Summary
In this section, we learned how to define *integer*s and *i-tuple*s in Choreo. The *Integer*s follow C/C++ syntax. *i-tuples* follow a syntax similar to *mdspan*, but they can operate on *mdspan* to derive new values.

Both types **require an _initialization expression_ and cannot be reassigned**, making them appear as **constant** or **immutable** values. In fact, all variables defined in Choreo tileflow programs are immutable, except for those of specific types. Therefore, **the assignment operator `=` is used for variable initialization in most cases**, which programs must be aware of.
