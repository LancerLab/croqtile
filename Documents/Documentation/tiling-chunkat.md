## Overview

## More on Bounded Variables
In last chapter, we learnt that a *bounded variable* defined by `parallel-by` or `with-in` statement is either an *integer* or an *ituple* with (a) bound(s). As the *bounded variable* has a *current value* together with a *upper-bound value*, we demonstrated how to use it to consturct loops in `foreach` blocks.

In Choreo, you may use the **ubound** operation to retrieve the upper bound of a bounded variable. *ubound* is an unary operation that prefix symbol `#` ahead of a bounded varable. For example:

```choreo
 with size in [32] {
   local u8 [#size, 16] a; // buffer shape: [32, 16]
   local u8 [size, 16] b; //  error in shape: [0, 16]
 }
```

Here, the local buffer `a` is shaped as `[32, 16]`, since `#size` is evaluated the upper-bound of `size`. However, as the declaration of `b` utilize the *current value* of size, it results in incorrect shape definition.


## Use `chunkat` to Tile Data
In all the examples of last section, the *source expression* and *destination expression* all utilize *spanned* data directly. However, in practice, the data tiling is required.
