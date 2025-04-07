## Overview

In this section, you will learn about constructing *bounded variable*s using the `with-in` statement and creating loops with the `foreach` statement.

## Loops in Choreo

In C++, loops are typically constructed using `for`, `while`, and `do-while` blocks. These loops include conditions for termination and usually involve variable manipulations related to loop control. Programmers have the freedom to define exit conditions, loop stepping, and other aspects, allowing for flexible loop structures.

In contrast, Choreo, being a domain-specific language for data movement, supports a more constrained loop structure. The current loop construction in Choreo is as follows:

```choreo
foreach index {
  // loop body
}
```

Here, `index` is a *bounded variable* (introduced in the previous section). If `index` has an upper bound of `6`, the above code is equivalent to:

```cpp
for (int index = 0; index < 6; ++index) {
  // loop body
}
```

While *bounded variables* may seem restrictive, they are sufficient for data movement tasks, as you typically won't move data with a negative or out-of-bound index.

In Choreo, *bounded variables* defined within a `parallel-by` statement are immutable and cannot be used in `foreach` statements. Instead, programmers should use the `with-in` statement to define the *bounded variables* suitable for loops.

## The **With-In** Block

### Defining *Bounded Variable*s:

A `with-in` statement allows you to define either a *bounded integer* or a *bounded ituple*. Syntactically, it resembles the `parallel-by` statement but only defines *bounded variable*s without invoking multiple instances for execution. Below are examples:

```choreo
with x in 128 {
  // 'x' is a bounded integer
}
with y in [512] {
  // 'y' is a bounded ituple
}
with index in [10, 10] {
  // 'index' is a bounded ituple
}
```

Here, `x` represents a bounded integer with an upper bound of `128`. Both `y` and `index` represent bounded ituples whose upper bounds are `512` and `10, 10` respectively. Similar to the `parallel-by` statement, you can name the elements of the ituple for clarity or declare both the bounded ituple and the associated bounded integers, as shown below:

```choreo
with {x, y} in [10, 10] {
  // x and y are explicitly named elements of the ituple
}
with index = {x, y} in [10, 10] {
  // 'x', 'y', and 'index' can be used within the block
}
```

Like the `parallel-by` statement, multiple `with-in` declarations can be combined using a comma-separated syntax, as demonstrated below:

```choreo
with index = {x, y} in [10, 10], idx in [100, 10] { }
```

In this way, two bounded ituples are defined. Note that either `index` or `idx` can only be referenced within the following `with-in` block.

## The `foreach` Block

### Basic Syntax

The `with-in` statement defines *bounded variable*s, which can then be iterated over using the `foreach` statement. The basic syntax is as follows:

```choreo
with index in [6] {
  foreach index {
    // Perform operations with each index
  }
}
```

In this example, the `foreach` block iterates 6 times, with the value of the **iteration variable** `index` ranging from `0` to `5` incrementally.

It is also possible to iterate over *bounded ituple*s. For example:

```choreo
with index = {x, y} in [6, 17] {
  foreach index { }
}
```

This code defines a *bounded ituple* `index`. Iterating over `index` is equivalent to a nested loop structure:

```cpp
for (int x = 0; x < 6; ++x)
  for (int y = 0; y < 17; ++y) { }
```

Notice the ordering of the loop nesting: the **left-to-right** order of the *bounded variable*s within a *bounded ituple* corresponds to the **outer-to-inner** nesting of the loops. This ordering is essential for correct code behavior constructing sometimes.

Furthermore, This rule also applies to the *comma-separated bounded variable list* that follows the `foreach` statement:

```choreo
with index = {x, y} in [6, 17], iv in [128] {
  foreach iv, index { }
}
```
In this code, two bounded variables are defined. The following `foreach` block is equivalent to a multi-level nested loop:

```cpp
for (int iv = 0; iv < 128; ++iv)
  for (int x = 0; x < 6; ++x)
    for (int y = 0; y < 17; ++y) { }
```

### Syntactic Sugar


```choreo
foreach x in 128 { }
foreach idx in [10, 20] { }
foreach {y, z} in [8, 16] { }
```

The code above is equivalent to the following code:

```choreo
with x in 128 {
  foreach x { }
}

with idx in [10, 20] {
  foreach idx { }
}

with {y, z} in [8, 16] {
  foreach y, z { }
}
```

### Deriving the Loop From a Bounded Integer

In certain scenarios, such as pipelining data movement, it may be necessary to modify loop iterations. In Choreo, this can be achieved by deriving a loop from a *bounded integer* within the `foreach` statement. For example:

```choreo
with {x, y} in [6, 17] {
  foreach x, y(1::) { }
}
```

In this case, the **Range Expression** `y(1::)` is used to derive the loop. This results in equivalent C/C++ code:

```cpp
for (int x = 0; x < 6; ++x)
  for (int y = 1; y < 17; ++y) { }
```

As seen in the code, the `y`-loop starts at `1`. The *range expression* consists of a *bounded variable*, followed by braced enclosing three colon-separated integer values, like the below form:

```
  bounded-variable(lower-offset:upper-offset:stride)
```

This derives a loop from the `bounded-variable`, where the `bounded-variable` serves as the *iteration variable* of the loop. Specifically:

- The initial value of the *iteration variable* is the `lower-offset` plus the lower bound of the `bounded-variable`. Since all bounded integers have a lower bound of `0` in Choreo, the `lower-offset` sets the initial value of the loop.
- The loop terminates when the *iteration variable* is equal to or greater than the upper bound of the `bounded-variable` plus `upper-offset`. Negative values are typically used for `upper-offset`.
- The *iteration variable* increments by `stride` at the end of each iteration.

Thus, `y(1:-1:2)` results in a loop like `for (y = 0 + 1; y < 17 - 1; y += 2)` in the example above. If any field of the *range expression* is not specified, it results in default values: `0` for `lower-offset` and `upper-offset`, and `1` for `stride`.

Note that *range expression*s only apply to the *bounded variable*s. *Range expression* over the *bounded ituple* triggers an error at compile time.

## Values of Bounded Variables

Understanding the values associated with a bounded variable is crucial. A bounded variable has two key values:

- **Current Value**:
    - Within a `foreach` statement, the current value is determined by the loop iteration, as the bounded variable serves as the iteration variable.
    - Outside of a `foreach` statement, the bounded variable always has a value of zero.

- **Upper-Bound Value**: This is specified in the `with-in` or `parallel-by` statements.

Programmers may encounter issues when using the current value of a bounded variable outside of a `foreach` statement, particularly after the loop has completed. By definition, the current value of a bounded variable is immutable except within a `foreach` loop. The following code illustrates this:

```choreo
with x in 6 {
  // x's current value is 0
  foreach x {
    // x's current value is either 0, 1, 2, ..., 5
  }
  // x's current value is 0, NOT 6
}
```

## Quick Summary
In this section, we explain the use of `with-in` and `foreach` statements in Choreo to define and iterate over *bounded variable*s, which are essential for data movement tasks. We introduced the syntax and behavior of these constructs, including how to apply range operations to modify loop iterations and the importance of understanding the current and upper-bound values of bounded variables.

The loop deriving part is important for implementing multi-buffering data movement, which is essential for building high-performance kernels and will be introduced in optimization chapters later.
