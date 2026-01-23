
## Overview

## Parallelization and Masking

It is usually necessary to mask off some *parallel thread* in programming SPMD code. Programmers may invoke fixed number of parallel threads where the tasks can not evenly divided. Thus masking is necessary to close this gap: it makes only parts of the threads to work, while leaving the other threads wait the working threads..

In Choreo code, the code may be implicitly masked, or explicit masked. Let us dive into the detail.

### Explicit Masking/Condition
Thread masking is a concept when programs have a full view of all instances of SPMD code. However, from the SPMD coding practise, which program is limited in single thread view, the thread masking appears as a boolean expression which typically is a comparison result related to thread identifiers.

For now, Choreo experimentally supports explicit masking of the `foreach` block, which mask off loop code as a whole. For example:

```choreo
__co__  void foo() {
  parallel p by 6 {
    with q in [2] {
      foreach q if (p == 0) {
        // this iteration only work for parallel thread 0
      }
    }
  }
}
```

### Implicit Masking
There are some code structures in Choreo that implies implicit masking. One typical situation is the multi-level `parallel-by` block. The below code showcases an example:

```
__co__  void foo() {
  parallel p by 6 {
    // parallel-level-0:
    //   code here is as if implicitly masked 'if (q == 0)'
    parallel q by 2 {
      // parallel-level-1:
      //   code here is for all 'q' threads
    }
    // parallel-level-0:
    //   code here is as if implicitly masked 'if (q == 0)'
  }
}
```

In this example, there exists two-level of parallelization. The code inside `parallel p by 6` block but outside `parallel q by 2` blocks is only for *parallel-level-0*. However, for target like *CUDA/Cute*, it could invoke 2 parallel threads in practise to execute either code inside *parallel-level-0* or *parallel-level-1*. Therefore, the *parallel-level-0*-only part is as if guarded with a C++ block `if (q == 0)`.

 Conditional `foreach` Block
Since
It is possible to guard the `foreach` loop with conditions. One direct

## Experimental: Optional *Where-Binding* for `within`

You can append a *where-clause* to impose **Where-Binding** constraints among different bounded variables defined in a `with-in` statement. This allows different bounded variables to be treated as aliases for each other. Below is an example:

```choreo
with {m, n} in [M, N], {n_p, k} in [N_P, K]
where n_p <-> n {
  // matmul implements with m, n, K. 'n_p' always refers to the same value as 'n'.
}
```

Syntactically, the `<->` operation establishes the *where-binding* between two bounded variables. Within the `with-in` block, any reference to `n_p` refers to `n`, and vice versa.

