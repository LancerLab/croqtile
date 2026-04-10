# Parallel-By

## Parallel Execution Blocks

Croqtile uses the `parallel-by` construct to express SPMD (Single Program, Multiple Data) parallelism. Multiple instances of the same code execute simultaneously, each identified by a unique parallel variable.

### Basic Syntax

```choreo
parallel p by 6 {
  // SPMD code: 6 threads, p ranges from 0 to 5
}
```

- **`parallel`** -- keyword initiating the parallel execution block.
- **`p`** -- the **parallel variable**, a bounded integer identifying the current thread.
- **`by 6`** -- total thread count. `p` ranges over `[0, 6)`.

The parallel variable can be omitted for simple constructs where thread identity is not needed:

```choreo
parallel by 2 { ... }
```

### Bounded Variable Semantics

The parallel variable `p` in `parallel p by 6` is a **bounded integer** with range `[0, 6)`. This means:

- Its **current value** differs per thread (0 through 5).
- Its **upper bound** (`#p = 6`) is available for shape computations.
- It can be used in `chunkat` expressions, where the upper bound determines tiling factors.

## Multiple Parallel Levels

Nested `parallel-by` blocks define a hierarchical parallel structure:

```choreo
parallel p by 2 {
  // parallel-level-0 (outer)
  parallel q by 12 {
    // parallel-level-1 (inner)
  }
}
```

Total threads: `2 x 12 = 24`. Each thread is identified by the pair `(p, q)`.

For CUDA targets:
- Level-0 maps to the **grid** level (blocks).
- Level-1 maps to the **block** level (threads within a block).

The comma syntax creates equivalent nesting without intervening code:

```choreo
parallel p by 2, q by 12 { ... }  // same as nested form
```

## Sub-Level Parallelism

A single parallel level can be subdivided into up to three sub-levels using i-tuples and mdspans, matching CUDA's 3D thread/block indexing:

```choreo
parallel {px, py, pz} by [1, 1, 2] {
  parallel t_index = {qx, qy} by [3, 4] { ... }
}
```

The parallel variables in an i-tuple are orthogonal -- each one independently indexes its own dimension and they carry no relative significance ordering.

## Parallel Level Annotations

Parallel levels can be annotated with `: block`, `: group`, or `: thread` to explicitly bind them to hardware hierarchy levels:

```choreo
parallel p by 6 : block {
  parallel q by 4 : thread { ... }
}
parallel r by 8 : group { ... }
```

These annotations guide the compiler's mapping to the target hardware's parallel hierarchy. Without annotations, the compiler applies default mapping rules.

*(Reference: `tests/norm/pb_norm_group.co`)*

## Heterogeneity: Host/Device Boundary

For targets like CUDA/CuTe, the `parallel-by` block defines the boundary between host and device code:

```choreo
__co__ void foo(...) {
  // host code
  parallel p by 6 {
    // device code (kernel launch)
  }
  // host code
}
```

This is analogous to a CUDA kernel launch. The Croqtile compiler handles the split automatically.

## Storage Qualifier Restrictions (CUDA target)

The host/device boundary implies constraints on storage. The exact rules are target-specific; for the **CUDA target**:

- **Outside `parallel-by`**: Only `global` or default (host) storage.
- **Inside `parallel-by`**: `shared` and `local` storage are available.
- **Lifetime**: `shared` and `local` buffers live only for the duration of the kernel launch; they cannot be referenced outside the `parallel-by` block.

```choreo
__co__ void foo(f32 [64] input) {
  shared f32 [32] buf;       // error: shared outside parallel block
  parallel p by 4 {
    shared f32 [16] tile;    // OK
    shared f32 [4] scratch;   // OK
  }
}
```

Other targets may define different storage qualifier semantics and restrictions.
