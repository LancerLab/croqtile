# Events

## Overview

Events provide communication between asynchronous code blocks in Croqtile. They are the primary mechanism for ordering execution across `inthreads` paths and between different parallel levels.

## Event Basics

An event is a binary state variable: either **SET** or **UNSET**. Events are declared with a storage qualifier:

```choreo
shared event e;           // single event
shared event e[4], e1;    // event array + single event
```

Events follow the same storage rules as spanned data -- `shared` events live inside `parallel-by` blocks, `global` events at the tileflow level.

## `trigger` and `wait`

**`trigger`** changes an event from UNSET to SET:

```choreo
trigger e, e1;    // trigger multiple events
```

**`wait`** blocks until the event is SET, then **auto-resets** the event to UNSET:

```choreo
wait e;           // blocks if UNSET, then resets to UNSET
```

The auto-reset behavior means each `trigger` unblocks exactly one `wait`.

## Chaining Asynchronous Code

Events chain the execution order of async `inthreads` blocks:

```choreo
parallel p by 2 {
  shared event e;
  inthreads.async (p == 1) { wait e; }      // waits for trigger
  inthreads.async (p == 0) { trigger e; }   // triggers e
  sync.shared;
}
```

Thread 0 triggers `e`, which unblocks thread 1's `wait`. This ensures thread 0's `inthreads` completes before thread 1 proceeds.

## Event Instances

The number of event instances depends on the storage qualifier and parallel structure:

```choreo
__co__ void foo() {
  global event ge;          // 1 instance

  parallel p by 2 {
    shared event se;        // 2 instances (1 per block)
    parallel q by 6 {
      local event le;       // 12 instances (6 per block)
    }
  }
}
```

Each parallel group at the storage's level gets its own event instance.

## Deadlock Avoidance

Since `wait` auto-resets the event, multiple threads waiting on the same event can cause deadlocks:

```choreo
// DANGEROUS: deadlock possible
parallel p by 1 {
  shared event e;
  trigger e;
  parallel q by 2 {
    wait e;      // thread 0 may reset before thread 1 sees SET
  }
}
```

Use event arrays to give each thread its own event:

```choreo
qc = 2;
parallel p by 1 {
  shared event e[qc];
  trigger e;              // triggers e[0] and e[1]
  parallel q by qc {
    wait e[q];            // each thread waits its own event
  }
}
```

## Events in Practice

Events are most commonly used in:

- **Warp specialization**: Ordering producer and consumer warp groups.
- **Software pipelining**: Signaling that a buffer is ready for consumption.
- **Multi-stage pipelines**: Coordinating stages that overlap DMA and compute.

*(Reference: `tests/parse/events.co`, `tests/infer/events.co`, `tests/gpu/check/event_validation.co`)*
