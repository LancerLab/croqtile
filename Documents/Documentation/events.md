## Overview
As `inthreads` introduce MPMD programming pattern in Choreo, it is necessary to synchornize between different asynchronous code. We can use event in Choreo to achieve that.

## Event Basics
There are different definitions of `event` in different programming language. In Choreo, the **Event** is used to handle communication and interaction between asynchronous code blocks at different level.

An event in Choreo is either *SET* or *UNSET*. When declaring an event, it is initialized with the *UNSET* state. For example, the following code initialize a `shared` event:

```
shared event e;
```

Note that since it is a `shared` event, it can only be declared inside `parallel-by` block and used for parallel threads. Similarly, `global` and `local` events can be defined as well. There is normally restrictions syntactially for either the declarations and value references according to the target platforms.

In addition, you may declare an event array following C-style syntax:

```
shared event e[4], e1;
```

Note it is allowed to use comma-separated expression to define either event and event array in a single line.

In Choreo, the `trigger` statements changes the `event` from *UNSET* to *SET* state:

```
shared event e[4], e1;

trigger e, e1;
```

The `trigger` statement triggers each event in an event array `e`, as well as the single event `e1`, as showed in the comma-separated list.

Similarly, the `wait` statement can wait for an event in a blocking mannar. The wait statement blocks currentthreads for execution if the event is *UNSET*, until the event state is changed to *SET*.

Therefore, triggering an event is used to unblocking code for the *wait statement*. For example,

```
parallel p by 2 {
  shared event e;                          // init 'e' to 'UNSET'
  inthreads.async (p == 1) { wait e; }     // wait-thread
                                           // once unblocked, change 'e' to 'UNSET'
  inthreads.async (p == 0) { trigger e; }  // trigger-thread
                                           // convert 'e' to 'SET'
  sync.shared;                             // sync point
}
```

We have observed similar code in last section, except for that there are `wait` and `trigger` statements. In this code snippet, once the event `e` is triggered by the *trigger-thread*, the *wait-thread* is no longer blocking. As a consequent, the two asynchronous threads followed an order to reach the synchronization point defined by `sync.shared`, where thread `0` completes its `inthreads` block before thread `1`. From another perspective, the events here **chains the asynchronous `inthreads`**. This is useful in many scenarios.

Note that, in Choreo semantics, **an event is always auto-reset by its `wait` statement**. That means when a `wait` statement is unblocked, the events that are waited are changed to *UNSET* state.

## Event Instances
In Choreo, events are associated with the storage specifier, either `global`, `shared`, and `local`. Such storage specifier specifies the event instances.

```
__co__ void foo() {
  global event ge;     // single instance in the tileflow program

  parallel p by 2 {    // shared level
    shared event se;   // two instances, 1 for each shared instance
    parallel q by 6 {  // local level
      local event le;  // 12 instances in total, 6 for each shared instance
    }
  }
}
```

The comments in the above code explain the event instances. The mechanism is same to the spanned data declarations, where storage specifier is also required. From implementation perspective, event can be implement as the leveled boolean data within the specified storage. Therefore, it also takes the storage of this level. However, in concept, the event belongs to a specific level. Therefore, its instance count coresponds to its parallel level.

## DeadLocks
By default, the event is lock-free for high-performance synchoronization between different asynchronous blocks. That implies it is risky when multiple threads try to update the event, especially for `wait`.

For example,

```
parallel p by 1 {
  shared event e;
  trigger e;   // one thread set e;
  parallel q by 2 {
    wait e;    // two threads can reset e;
  }
}
```

In this code, two threads wait a single event `e`. However, since `wait` resets the event `e`, it is possible that `thread-0` resets the event before `thread-1` find the `e` has been triggered. As a result, `thread-1` is never unblocked and deadlocking happens.

Though such deadlocks are less possible for GPU-like SIMT hardware, where both threads execute in lock-step, it is not safe code for highly asynchronous executing like threading in CPUs. Since Choreo supports multiple levels of asynchronous execution, including different hardware/software, a more robust way is to assign each thread an event for `wait` statements.

```
qc = 2;
parallel p by 1 {
  shared event e[qc];
  trigger e;   // one thread set e[0] and e[1];
  parallel q by qc {
    wait e[q]; // no dead lock
  }
}
```

In this code, we utilize an event array to guarantee each event is waited and reset by a single thread.

## Quick Summary
In this section, we learnt 'event' that can ochestrate asynchronous `inthreads` blocks. The `trigger` statements can set events and event arrays to *SET* state, while `wait` statements block execution, and reset events to *UNSET* until they are triggered.

Programmers must be careful to use Choreo events, or else it is possible to result in deadlocks, which is hard to debug.
