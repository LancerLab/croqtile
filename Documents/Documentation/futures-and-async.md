# Futures and Async Execution

## Overview

A Choreo future names the completion of an operation. Futures are local
producer-side handles. Events publish completion to another asynchronous path.

There are two future categories:

- A data future is produced by DMA or TMA. It also refers to the destination
  data through `.data` and `.span`.
- An operation future is produced by an asynchronous operation that updates an
  existing object. Asynchronous WGMMA is the first operation-future producer.

```choreo
load = tma.copy.async input => tile;
qk = mma.row.row.async scores, q_tile, k_tile;
```

`load` is a data future. `qk` is an operation future and does not own a data
buffer; the updated accumulator is `scores`.

## Waiting Locally

`wait` observes completion on the current control path:

```choreo
wait load;
wait qk;
wait lhs_load, rhs_load;
```

For a data future, the destination can be read after the wait. For an MMA
operation future, the accumulator result can be consumed after the wait. The
compiler derives the required WGMMA wait depth from the dependency graph; there
is no source-level wait-depth control.

Each dynamic future instance has one completion path. Waiting consumes that
producer-local instance; later cross-path notification is an ordinary direct
event trigger:

```choreo
qk = mma.row.row.async scores, q_tile, k_tile;
wait qk;
// Local work that consumes scores.
trigger qk_done;
```

Waiting or binding the same dynamic future instance more than once is an error.
A data-future symbol may still appear at multiple static wait sites in a
swap/rotation pipeline, where each execution observes a different dynamic
instance. If several consumers need one completion, publish one event and let
them wait on that event.

## Publishing Completion to an Event

`trigger event after future` binds an event to an operation's native completion
mechanism:

```choreo
trigger event_name after future_name;
```

One event can join several operations:

```choreo
lhs_load = tma.copy.async lhs => lhs_s;
rhs_load = tma.copy.async rhs => rhs_s;
trigger operands_full after lhs_load, rhs_load;
```

This form is non-blocking and never implies or inserts `wait future`. It is
valid only when the backend can hand the event directly to the operation. The
CUDA backend currently supports async global-to-shared TMA futures, whose
transaction byte counts are joined on the event's mbarrier.

WGMMA has no native completion-event operand. Make its completion point
explicit instead:

```choreo
qk = mma.row.row.async scores, q_tile, k_tile;
wait qk;
trigger k_empty;
```

Likewise, `wait qk; trigger k_empty after qk;` is invalid: after an explicit
wait there is no future left to bind, and the direct trigger states the order
without a redundant dependency.

Use an event for cross-path communication:

```choreo
parallel g by 2 : group-4 {
  inthreads.async (g == 0) {
    load = tma.copy.async input => tile;
    trigger full after load;
  }
  inthreads.async (g == 1) {
    wait full;
    // Consume tile.
  }
}
```

## Cyclic Event Generations

A future has no stage. A cyclic event array derives its slot and generation
from the logical pipeline order:

```choreo
slot = order % 2;
wait empty.at(order);
load = tma.copy.async input.at(order) => tile[slot];
trigger full.at(order) after load;
```

For `event[N]`, `.at(order)` selects slot `order % N` and generation
`order / N`. Ordinary data arrays still use an explicit physical slot. This
prevents a reused event slot from being confused with an older logical
iteration without introducing a separate stage type.

An event is pending by default. Use `= ready` when generation zero must be
available immediately:

```choreo
shared event full[2], empty[2] = ready;
```

See [Events](events.md) for event generations and fanout.

## Data Future Members

Only data futures expose buffer members:

| Member | Description |
|--------|-------------|
| `.data` | Destination data reference |
| `.span` | Destination shape |

These members are available immediately for a synchronous copy and after
completion for an asynchronous copy. Operation futures do not expose either
member.

## Placeholder Data Futures

`dma.any` creates a placeholder data future for legacy buffer-rotation
patterns:

```choreo
next = dma.any;
foreach k in tiles {
  next = dma.copy.async input.chunkat(k) => shared;
  wait current;
  swap(current, next);
}
```

Placeholder data futures do not replace operation futures or events.

## Operation-to-Operation DMA Chains

The DMA `after` clause orders one DMA operation after another without
publishing an event:

```choreo
store = dma.copy.async result => output after load;
```

This is distinct from `trigger event after future`: the former creates an
operation-to-operation edge, while the latter binds native completion to an
event without waiting.

## Async Execution Levels

Choreo composes three kinds of asynchronous execution:

1. An async operation produces a future.
2. `inthreads.async` lets predicated thread groups execute independently.
3. `parallel.async` launches independent kernel work on a stream.

Futures express operation completion. Events express communication between
asynchronous paths. `event.at(order)` expresses cyclic event generations. No
`pipeline` keyword is required.
