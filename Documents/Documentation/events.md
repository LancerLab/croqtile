# Events

## Overview

Events communicate readiness between asynchronous Choreo paths. A future is a
producer-local completion handle; an event is the published synchronization
point that one or more consumers can wait on.

```choreo
shared event full;

load = tma.copy.async input => tile;
trigger full after load;
```

Native operation completion can be bound to an event with
`trigger event after future`.

## Declaration and Initial State

Events use ordinary storage and array syntax:

```choreo
shared event ready_for_compute;
shared event full[2], empty[2] = ready;
```

Generation zero is pending by default. `= ready` marks generation zero ready,
which is useful for initially empty pipeline slots.

The backend chooses the hardware synchronization primitive from the event's
producer, consumers, scope, and participation. Source code does not select an
mbarrier, named barrier, or transaction count.

## Immediate Trigger

`trigger` without dependencies publishes readiness at that point:

```choreo
trigger empty;
trigger empty0, empty1;
```

This form is useful for releasing an empty buffer after its consumer is done
or for explicit initialization. It does not represent completion of an
earlier async operation unless that operation has already been waited on.

## Trigger After Future

Use a dependency when readiness comes from an async operation:

```choreo
load = tma.copy.async input => tile;
trigger full after load;
```

Several futures can be joined into one event:

```choreo
q_load = tma.copy.async q => q_s;
k_load = tma.copy.async k => k_s;
trigger qk_inputs_full after q_load, k_load;
```

`after` is a binding clause, not an implicit wait. The statement does not block
the producer path, and the compiler must not synthesize a wait. It is currently
supported for global-to-shared TMA futures, whose hardware mbarrier completion
can directly publish the event.

Operations without native event binding use an explicit wait and a direct
trigger. WGMMA is one example:

```choreo
qk = mma.row.row.async scores, q_s, k_s;
wait qk;
trigger qk_done;
```

Writing `wait qk; trigger qk_done after qk;` is rejected because the future was
already consumed by the explicit wait. Inline event operands on DMA/TMA and
special MMA wait-trigger statements are not part of the language.

## Waiting

`wait` blocks the current path until the selected event generation is ready:

```choreo
wait full;
wait lhs_full, rhs_full;
```

Events are generation-based. A completed wait advances that path to the next
generation. Repeated producer/consumer handoffs therefore reuse the event
without a source-level reset operation.

## Cyclic Event Generations

Use `.at(order)` when an event array is a cyclic pipeline ring:

```choreo
slot = order % 2;
wait empty.at(order);
load = tma.copy.async input.at(order) => tile[slot];
trigger full.at(order) after load;
```

The consumer uses the same logical order:

```choreo
slot = order % 2;
wait full.at(order);
// Consume tile[slot].
trigger empty.at(order);
```

For `event[N]`, the compiler derives the physical event slot as `order % N`
and its barrier phase as `(order / N) % 2`. The argument may be a monotonically
increasing logical order or an equivalent order maintained modulo `2 * N`.
For example, `(order + 1) & 3` is sufficient for `event[2]`. Reducing the
argument modulo `N` is not sufficient because it discards the phase. A cyclic
event ring is currently one-dimensional.

`event[index]` remains ordinary physical array indexing. Its index is not
interpreted as a logical order, so it is not an alias for `event.at(order)`.

Shared data arrays remain ordinary arrays. Their physical slot is therefore
written explicitly, as in `tile[slot]`; event generation and data indexing do
not share a special source-level type.

## Multiple Consumers

Publish a future once, then use an event when multiple paths need the result:

```choreo
trigger tile_full after load;

inthreads.async (consumer_a) { wait tile_full; }
inthreads.async (consumer_b) { wait tile_full; }
```

The compiler derives event participation from the active scopes. Do not
publish the same operation future separately for each consumer.

For independent generations or independently released buffers, use an event
array rather than making unrelated consumers race on one generation.

## Producer-Consumer Example

```choreo
shared f16[128, 128] tile[2];
shared event full[2], empty[2] = ready;

parallel g by 2 : group-4 {
  inthreads.async (g == 0) {
    foreach {order} in [tile_count] {
      slot = order % 2;
      wait empty.at(order);
      load = tma.copy.async input.at(order) => tile[slot];
      trigger full.at(order) after load;
    }
  }

  inthreads.async (g == 1) {
    foreach {order} in [tile_count] {
      slot = order % 2;
      wait full.at(order);
      // Compute with tile[slot].
      trigger empty.at(order);
    }
  }
}
```

This is the complete pipeline vocabulary: async operations produce futures,
events publish readiness, and `.at(order)` selects cyclic generations.
