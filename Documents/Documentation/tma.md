# TMA (Tensor Memory Accelerator)

## Overview

NVIDIA Hopper and later GPUs provide the Tensor Memory Accelerator (TMA) for
bulk transfers between global and shared memory. Choreo exposes it with
`tma.load` for global-to-shared transfers and `tma.copy` for
shared-to-global transfers, and generates tensor-map descriptors automatically.

TMA supports global-to-shared and shared-to-global transfers. It does not
provide a direct global-to-global operation.

## Synchronous Form

```choreo
tma.load input.subspan(tile_m, tile_k).at(m, k) => shared_tile;
tma.copy shared_result => output.subspan(tile_m, tile_n).at(m, n);
```

The synchronous source form includes the synchronization needed before the
statement completes.

## Asynchronous Form

Assign an asynchronous TMA operation to a data future:

```choreo
load = tma.load.async.swiz<128>
    input.subspan(128, 128).at(tile_id, 0) => tile;
```

Wait locally when the same path consumes the data:

```choreo
wait load;
```

Publish completion to another path with an event:

```choreo
load = tma.load.async input => tile;
trigger full after load;
```

For a global-to-shared load, the compiler binds the event directly to the
native TMA transaction barrier. `trigger ... after ...` is non-blocking and
does not insert a wait. There is no inline event argument on
`tma.load.async`.

## Joining TMA Operations

One event can represent several transfers:

```choreo
lhs_load = tma.load.async lhs_tile => lhs_s;
rhs_load = tma.load.async rhs_tile => rhs_s;
trigger operands_full after lhs_load, rhs_load;
```

The consumer sees the event only after both transactions complete.

## Multi-Stage Pipeline

Use the logical pipeline order for event generations and an explicit physical
slot for the shared buffer:

```choreo
shared f16[128, 128] tile[2];
shared event full[2], empty[2] = ready;

slot = order % 2;
wait empty.at(order);
load = tma.load.async.swiz<128> input.at(order) => tile[slot];
trigger full.at(order) after load;
```

For `event[N]`, the backend derives slot `order % N` and generation
`order / N`. A future itself has no stage.

## L2 Eviction Hints

L2 cache policy is an issue-time modifier on a TMA load:

```choreo
q_load = tma.load.async.evict_first.swiz<128> q => q_s;
kv_load = tma.load.async.evict_last.swiz<128> kv => kv_s;
```

The modifier belongs to the transaction itself; it neither waits for nor
mutates the returned future. It must appear after `.async` and before
`.swiz<N>`. Eviction modifiers are not valid on `tma.copy` or multicast loads.
Omitting the modifier selects the backend's default L2 policy.

## Shared-to-Global Stores

Shared-to-global TMA uses bulk-group completion rather than an mbarrier, so it
cannot currently bind `trigger event after future`. Use a synchronous store
and trigger the event after the statement completes:

```choreo
tma.copy output_s => output_tile;
trigger stored;
```

For warp-specialized consumers, the backend determines whether one elected
thread stores a jointly produced tile or each consumer warp-group stores its
own tile. This follows operand dependence and does not require a hardware hint
in source code.

## Descriptor Management

Tensor-map and WGMMA shared-memory descriptors are backend details. Choreo
creates, hoists, and reuses descriptors from the referenced views. Programs
refer directly to global and shared views; they do not declare descriptor
objects.

## TMA vs DMA

| Aspect | `dma.copy` | `tma.load` / `tma.copy` |
|--------|------------|-------------------------|
| Mechanism | Software copy or target async copy | Tensor Memory Accelerator |
| Target | General | SM90+ |
| Direction | General memory transfers | Global <-> shared |
| Descriptor | Not required | Generated automatically |
| Completion | Data future | Native event binding for async loads |

Use DMA for unsupported targets or transfer directions. Use TMA for large,
regular Hopper global/shared tiles.

`tma.copy` remains the shared-to-global spelling and is also accepted for
existing global-to-shared source. New global-to-shared code should use
`tma.load`.
