# Futures and Async Execution

## The Future Type

A **future** in Croqtile is the result type of a DMA statement. It serves dual purposes:

1. **Async handle**: For `dma.copy.async`, the future is a synchronization handle that must be `wait`-ed before the destination data can be used.
2. **Buffer reference**: For both sync and async DMA, the future provides `.data` and `.span` to access the destination buffer.

```choreo
f = dma.copy.async input => shared;
// ... computation overlaps with DMA ...
wait f;
call kernel(f.data, |f.span|);
```

## Wait Semantics

The `wait` statement blocks until the specified async operations complete:

```choreo
wait f0, f1;     // wait for both f0 and f1
```

Rules:
- Only async DMA futures can be `wait`-ed. Waiting on a sync DMA future is a compile error.
- After `wait`, the destination buffer is safe to read.
- An unwait-ed async future is a programming error (undefined behavior on hardware).
- `wait` can accept multiple futures in a comma-separated list.

## Future Members

| Member | Type | Description |
|--------|------|-------------|
| `.data` | Spanned data reference | The destination buffer |
| `.span` | mdspan | Shape of the destination buffer |

These are valid after the DMA completes (immediately for sync, after `wait` for async).

## `dma.any` -- Placeholder Futures

`dma.any` creates a future with no backing DMA operation. This is used for multi-buffering patterns where a future variable must exist before the loop that assigns it:

```choreo
lfB = dma.any;           // placeholder
foreach y(1:) {
  lfB = dma.copy.async data.chunkat(p, y) => shared;
  wait lfA;
  // ...
  swap(lfA, lfB);
}
```

## Async `parallel-by`

The `parallel-by` construct itself represents a transition from host to device (a kernel launch), which is inherently asynchronous from the host's perspective. Multiple kernel launches can overlap with proper synchronization.

## Async `inthreads`

As covered in [Thread Masking](thread-masking.md), `inthreads.async` enables concurrent execution of divergent code paths without implicit synchronization:

```choreo
inthreads.async (p <= 2) { /* path A */ }
inthreads.async (p > 2)  { /* path B */ }
sync.shared;
```

## The Async Execution Model

Croqtile's async model has three levels:

1. **DMA-level**: Individual DMA operations run asynchronously via `.async`. Synchronized with `wait`.
2. **Thread-level**: `inthreads.async` creates divergent execution paths within a parallel block. Synchronized with `sync.shared`.
3. **Kernel-level**: `parallel-by` launches are asynchronous from the host. Synchronized at the end of the `parallel-by` block.

These levels compose: async DMA within async `inthreads` within an async kernel launch, enabling deep pipelining of data movement and computation.

## Non-Blocking DMA Chains

The `after` keyword creates dependency chains between async DMAs without blocking the main thread:

```choreo
store = dma.copy.async result => output.chunkat(m, n) after load;
```

`store` starts only after `load` completes. Neither blocks the main execution flow. See [DMA Advanced](dma-advanced.md).

## `select` and `swap`

These operations manipulate futures for multi-buffering:

- **`swap(f1, f2)`**: Exchanges two futures.
- **`select(cond, f_true, f_false)`**: Chooses a future based on a condition.

See [DMA Advanced](dma-advanced.md) for the full multi-buffering pattern.
