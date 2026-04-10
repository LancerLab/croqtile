# DMA Basics

## Built-in Data Movement for Heterogeneous Workloads

Heterogeneous hardware (CPU + GPU, multi-level memory) is performance-sensitive to data movement. Croqtile provides **built-in DMA (Direct Memory Access) operations** that describe shaped data transfers between memory hierarchy levels, supporting both synchronous (blocking) and asynchronous (non-blocking) execution.

## Statement Syntax

```
future = dma-op src-expr => dst-expr;
```

| Component | Description |
|-----------|-------------|
| `future` | Handle for the operation and reference to the destination buffer |
| `dma-op` | Operation type with optional `.async` suffix |
| `src-expr` | Source data expression |
| `=>` | Data flow direction |
| `dst-expr` | Destination data expression or storage keyword |

## Synchronous vs Asynchronous

The `.async` suffix makes a DMA operation non-blocking:

```choreo
f0 = dma.copy.async data0 => shared;  // non-blocking
f1 = dma.copy data1 => shared;        // blocking
// ... code runs while f0 is in flight ...
wait f0;                                // wait for f0 to complete
```

- **Async DMA** (`dma.copy.async`): Execution continues immediately. The programmer must `wait` on the future before using the result.
- **Sync DMA** (`dma.copy`): Execution blocks until the transfer completes. No `wait` is needed (or allowed).

An async DMA without a corresponding `wait` is a programming error that leads to undefined behavior on hardware.

## Operation Types

Croqtile provides several DMA operation types. The `dma.copy` implementation may use software-orchestrated multi-thread cooperative copies or hardware-accelerated async DMA (e.g., CUDA's `cp.async`) depending on the target and context:

| Operation | Description | Configuration |
|-----------|-------------|---------------|
| `dma.copy` | Flat memory copy | (none) |
| `dma.pad` | Pad shaped data during transfer | `<low, high, mid, pad_value>` |
| `dma.transp` | Transpose dimensions during transfer | `<perm>` |

### Configuration Syntax

Parameters are enclosed in `<>`:

```choreo
global f32 [32, 16, 9] input;
dma.transp<0, 2, 1> input => shared;
// Result shape: [32, 9, 16] (dims 1 and 2 swapped)

dma.pad<{1, 0, 3}, {0, 1, 2}, {0, 0, 0}, 0.1f> input => shared;
// Result shape: [33, 17, 14] (padded per dimension)
```

Support for operations beyond `dma.copy` varies by target platform.

## Destination Expressions

The destination can be either an explicit buffer or a **storage keyword**:

```choreo
global f32 [32, 16] input;
shared f32 [32, 16] output;
dma.copy input => output;    // to explicit buffer
dma.copy input => shared;    // compiler allocates shared buffer
```

Using a storage keyword as the destination lets the compiler allocate and manage the buffer. The compiler deduces the destination shape from the source expression and DMA configuration.

## Futures

Every DMA statement (sync or async) can produce a **future**:

```choreo
f = dma.copy.async input => shared;
```

Futures provide access to the destination:

| Member | Returns |
|--------|---------|
| `.data` | Reference to the destination buffer (spanned data) |
| `.span` | The mdspan of the destination buffer |

```choreo
f = dma.copy input => shared;
shared f32 [f.span] buffer;         // use shape
call kernel(f.data, |f.span|);     // use data and element count
```

### `wait` Statement

`wait` synchronizes one or more async futures:

```choreo
f0 = dma.copy.async a => shared;
f1 = dma.copy.async b => shared;
f2 = dma.copy c => shared;
wait f0, f1;    // wait for multiple async futures
wait f2;        // error: cannot wait on sync DMA future
```

### `dma.any` -- Dummy Future

`dma.any` creates a **placeholder future** with no associated DMA operation. It is used in multi-buffering patterns where a future variable must exist before the actual DMA is issued:

```choreo
lfB = dma.any;    // placeholder
// ... later ...
lfB = dma.copy.async data.chunkat(p, y) => shared;
```

See [Multi-Buffering Patterns](dma-advanced.md) for the full pattern.

## Source and Destination Shape Rule

When both source and destination are explicit spanned data (not a storage keyword), the Croqtile compiler requires them to have **the same shape and element type**. Shape transformations (padding, transposing) modify the destination shape according to the DMA configuration.
