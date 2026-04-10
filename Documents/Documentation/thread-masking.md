# Thread Masking

## SPMD and Divergence

Croqtile's `parallel-by` block creates SPMD code where all threads execute the same program. However, real kernels often require **divergent** execution -- different threads taking different code paths. Croqtile supports this through explicit and implicit masking mechanisms.

From the programming model's perspective, code is either:

- **Uniform**: All threads execute the same path.
- **Divergent**: Only a subset of threads execute a path.

## `inthreads`: Explicit Masking

The `inthreads` keyword creates a divergent code block guarded by a thread-identity predicate:

```choreo
__co__ void foo() {
  parallel p by 6 {
    // uniform code (all 6 threads)
    inthreads (p <= 2) {
      // divergent: threads 0, 1, 2 only
    }
    // uniform code (all threads synchronized here)
  }
}
```

### Semantics

- The predicate must be a comparison involving parallel variables.
- After the `inthreads` block, **all threads synchronize** before continuing.
- Only the outermost `parallel-by` variables can appear in the predicate.

### Asynchronous `inthreads`

The `.async` suffix removes the implicit synchronization, enabling MPMD (Multiple Program, Multiple Data) execution:

```choreo
parallel p by 6 {
  inthreads.async (p <= 2) {
    // path A: threads 0, 1, 2
  }
  inthreads.async (p > 2) {
    // path B: threads 3, 4, 5
  }
  sync.shared;   // explicit sync point -- both paths must complete
}
```

Without `sync.shared`, the two paths would race. Only the **outermost** `inthreads` can be declared async; inner `inthreads.async` triggers a compile error.

## Multi-Level Masking

Thread masks can involve multiple parallel levels:

```choreo
parallel p by 3, q by 4 {
  inthreads (p < 2 && q == 0) {
    // threads (0,0) and (1,0) only
  }
  inthreads (q == 1) {
    // threads (0,1), (1,1), (2,1)
  }
}
```

The predicate can combine conditions on any parallel variable using `&&` and `||`.

## Implicit Masking

Code between nested `parallel-by` levels is implicitly masked. Only one thread from the inner level executes the outer-level code:

```choreo
parallel p by 6 {
  // effectively: inthreads (q == 0)
  parallel q by 2 {
    // all p*q threads execute here
  }
  // effectively: inthreads (q == 0)
}
```

In the code outside `parallel q by 2`, only the thread with `q == 0` executes. This is because the target hardware (e.g., CUDA) physically launches `p * q` threads, but the outer-only code should run once per outer group.

*(Reference: `tests/check/illegal_inthreads.co`, `tests/gpu/check/active_threads_basic.co`)*
