# Iteration: With-In, Foreach, and While

## Loop Model

Croqtile provides three loop constructs:

1. **`with-in`** -- defines bounded variables with known ranges.
2. **`foreach`** -- iterates over those bounded variables.
3. **`while`** -- loops on a runtime condition for statically-unknown iteration counts.

The `with-in` / `foreach` pair covers the common case where loop bounds are explicit and statically analyzable, which is essential for the compiler to reason about tiling, shape inference, and DMA scheduling. The `while` statement complements them for irregular loops that cannot be expressed with a compile-time bound.

## The `with-in` Block

### Defining Bounded Variables

`with-in` creates bounded variables (integers or i-tuples) without any parallelism implication -- the code inside executes sequentially.

```choreo
with x in 128 {
  // x: bounded integer, range [0, 128)
}
with y in [512] {
  // y: bounded ituple of rank 1, range [0, 512)
}
with index in [10, 10] {
  // index: bounded ituple of rank 2, ranges [0, 10) x [0, 10)
}
```

### Naming Elements

I-tuple elements can be individually named:

```choreo
with {x, y} in [10, 10] { ... }
with index = {x, y} in [10, 10] {
  // 'index', 'x', and 'y' all usable
}
```

### Multiple Declarations

Comma-separated declarations define multiple bounded variables:

```choreo
with index = {x, y} in [10, 10], iv in [100, 10] { ... }
```

## The `foreach` Block

### Basic Iteration

`foreach` iterates over bounded variables created by `with-in`:

```choreo
with index in [6] {
  foreach index { ... }
}
```

Equivalent C++ loop: `for (int index = 0; index < 6; ++index) { ... }`

### I-Tuple Iteration

Iterating over a bounded i-tuple produces nested loops. The **left-to-right** element order maps to **outer-to-inner** loop nesting:

```choreo
with index = {x, y} in [6, 17] {
  foreach index { ... }
}
```

Equivalent:

```cpp
for (int x = 0; x < 6; ++x)
  for (int y = 0; y < 17; ++y) { ... }
```

### Comma-Separated Iteration

Multiple bounded variables in `foreach` are nested outer-to-inner left-to-right:

```choreo
with index = {x, y} in [6, 17], iv in [128] {
  foreach iv, index { ... }
}
```

Equivalent:

```cpp
for (int iv = 0; iv < 128; ++iv)
  for (int x = 0; x < 6; ++x)
    for (int y = 0; y < 17; ++y) { ... }
```

### Syntactic Sugar

`foreach` can combine `with-in` and iteration in one statement:

```choreo
foreach x in 128 { ... }
foreach idx in [10, 20] { ... }
foreach {y, z} in [8, 16] { ... }
```

These expand to the corresponding `with-in` + `foreach` pairs.

## Range Expressions

Loop bounds can be modified with **range expressions** to adjust the starting point, ending point, and stride:

```choreo
with {x, y} in [6, 17] {
  foreach x, y(1::) { ... }
}
```

The range expression syntax is `bounded_var(lower_offset:upper_offset:stride)`:

- **`lower_offset`**: Added to the lower bound (default 0). `y(1::)` starts at 1 instead of 0.
- **`upper_offset`**: Added to the upper bound (default 0). `y(::-1)` or `y(:-1:)` ends one iteration early.
- **`stride`**: Increment per iteration (default 1). `y(::2)` steps by 2.

Example: `y(1:-1:2)` generates `for (y = 1; y < 17 - 1; y += 2)`.

Omitted fields take their defaults. Range expressions apply only to bounded integers/variables, not bounded i-tuples.

## Values of Bounded Variables

A bounded variable has two values:

- **Current value**: Determined by the loop iteration inside `foreach`. Outside `foreach`, the current value is always **0**.
- **Upper-bound value**: The bound specified in `with-in` or `parallel-by`, accessed via `#`.

```choreo
with x in 6 {
  // x == 0 here
  foreach x {
    // x ranges 0..5
  }
  // x == 0 here (NOT 6)
}
```

The upper-bound is always available regardless of context:

```choreo
with x in 6 {
  shared f32 [#x] buf;    // shape [6], using upper-bound
}
```

## `while` -- Irregular Loops

For loops whose iteration count is not statically known, Croqtile provides a `while` statement. The condition is an expression (typically an event predicate) that is evaluated each iteration:

```choreo
event e;
while (e) {
  // loop body; runs until the event predicate becomes false
}
```

`while` loops are commonly used with events for spin-waiting or producer/consumer synchronization, and for any iteration pattern whose bound cannot be expressed as a compile-time constant.

```choreo
while (a && b[0]) {
  // compound event predicate
}
```

*(Reference: `tests/parse/while_event.co`)*

## `break` and `continue`

Both `break` and `continue` are supported inside `foreach` and `while` loops:

- **`break`** exits the innermost enclosing loop immediately.
- **`continue`** skips the remainder of the current iteration and advances to the next.

```choreo
while (e) break;                       // exit immediately
foreach a in [3] if (a == 2) break;    // early exit from foreach
```

*(Reference: `tests/parse/break.co`)*
