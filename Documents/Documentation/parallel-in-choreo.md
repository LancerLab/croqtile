## Overview
To fully utilize parallel hardware, it is crucial to write parallel code in Choreo. This section will guide you through using the `parallel-by` block to implement parallel code in Choreo.


## Parallel Execution Block
In Choreo, the code run in parallel are placed into the *parallel execution block*, which showcasing there are multiple instances of the same code that are executed simultaneously.

### Basic Syntax: `parallel-by`

The basic syntax for a `parallel-by` block is as follows:

```choreo
parallel p by 6 {
  // SPMD (Single Program, Multiple Data) code
}
```
We name the code inside `{}` as the SPMD-style **parallel execution block**, and the instances of the code as the **parallel thread**.
In this example:

- **`parallel`**: This keyword initiates the *parallel execution block*.
- **`p`**: It is the **parallel variable**. It contains an integer value identifying the current *parallel thread*. Each *parallel thread* runs the same code but with a different value of *parallel variable* `p`.
- **`by 6`**: It indicates that there are totally 6 *parallel thread*s. Thus, the *parallel variable* `p` ranges from `0` to `5` across different threads.

In some scenarios, programers may not require an explicit *parallel variable* for a simple parallel construct. It is possible to omit it:

```choreo
parallel by 2 { ... }
```

This invokes two parallel threads to execute. However, it is not possible for the two threads to work on different data, which is normally required by SPMD programs for a data-parallel processing purpose.

### Multiple-Level Parallelism

In Choreo, you can define multiple levels of parallelism. Here's an example:

```choreo
parallel p by 2 {
  // parallel-level-0
  parallel q by 12 {
    // parallel-level-1
  }
}
```

This code defines two levels of parallelism with values `2` and `12`, respectively. This means a total of `2 x 12 = 24` parallel threads are invoked for the task. Each thread is identified by a unique combination of `p` and `q`. In Choreo, we refer to the level of `p` as *parallel-level-0* and the level of `q` as *parallel-level-1*. For those familiar with CUDA, *parallel-level-0* corresponds to the CUDA **grid** level, and *parallel-level-1* corresponds to the CUDA **block** level.

The concept of multiple-level parallelism originates from hardware architecture with hierarchical memory. One key difference observable by programmers is the **synchronization cost**. For instance, in CUDA, threads within a **block** have a much lower synchronization cost compared to those across different **grids**. This hierarchical design encourages programmers to place the most divergent code (that requires synchronization) within the **block** level to enhance overall performance.

In Choreo syntax, you can define multiple-level parallelism in a single line:

```choreo
parallel p by 2, q by 12 { ... }
```

This achieves the same parallelization as the previous example. It uses the **comma-separated** *parallel-by* expression, assuming there is no code in the between two parallel levels.

### Sub-Level Parallelism

Targets like *CUDA* allow a single parallelism level to be divided into up to three sub-levels. In Choreo, `ituple` and `mdspan` are used for this syntax:

```choreo
parallel {px, py, pz} by [1, 1, 2] {
  // higher synchronization cost
  // ...
  parallel t_index = {qx, qy} by [3, 4] {
    // lower synchronization cost
    // ...
  }
  // ...
}
```

In this example, two levels of parallelism are defined. The first level (*level-0*) is subdivided into three sub-levels, annotated by the `{px, py, pz}` `ituple`. The second level (*level-1*) is subdivided into two sub-levels, annotated by the `{qx, qy}` `ituple`, or `t_index` as a whole. For each sub-level, the parallelization count corresponds to the respective value inside the `mdspan`.

The concept of sub-levels originates from *CUDA*'s GPU workload management, which often involves data with 2D or 3D elements. Note that the first element inside the `ituple` represents the **least significant parallel variable**. Choreo adopts this ordering to align with *CUDA* developers' conventions, rather than for internal consistency.

### Bounded Variable

In a `parallel-by` block, the variables defined after the `parallel` keyword are referred to as **Bounded Variables**. Specifically:

- If the variable is an *integer*, it is called a **bounded integer**.
- If the variable is an *ituple*, it is called a **bounded ituple**.

This terminology indicates that the variable not only has a value but also has defined bounds. For example, in the statement `parallel p by 6`, `p` is a *bounded integer* with a bound `[0, 6)`, where `6` is the exclusive upper bound.

The concept of bounded variables will be explored further in later sections, where their usage will be detailed.

## The Implications

### Heterogeneity: Transpiling to Kernel Launch

For targets like *CUDA/Cute*, the `parallel-by` block not only specifies the number of parallel threads but also defines the boundary between host and device code, leveraging Choreo's ability to manage heterogeneity. Specifically:

- Code **outside** the `parallel-by` block is transpiled into **host code**.
- Code **inside** the block is transpiled into **device code**, as illustrated below:

```choreo
__co__ void foo(...) {
  // Generate host code
  // ...
  parallel p by 6 { // Kernel launch
    // Generate device code
    // ...
  }
  // Generate host code
  // ...
}
```

For those familiar with *CUDA*, this is analogous to a **Kernel Launch** occurring at the `parallel-by` statement. However, Choreo abstracts these details with the unified SPMD parallelism model, allowing developers to focus on their algorithms without worrying about the underlying heterogeneity.

### Restrictions on Storage Specifiers

Since the `parallel-by` construct in Choreo is related to hardware, including heterogeneity and memory hierarchy, it is subject to the constraints of the target platform.

For example, for *CUDA/Cute* targets, a *spanned* buffer must be annotated with the appropriate storage specifier within the *parallel execution block*. This is because the target platform allows only *scratchpad memory*, such as *shared* and *local* memory, to be allocated by device code. Conversely, code outside the *parallel execution block* can only declare *global* or default host memory, which the target platform allows the host code to manage.

Additionally, *shared* and *local* memory types have a lifetime limited to the duration of the kernel launch. Therefore, referencing them outside the *parallel execution block* is not possible. This constraint ensures proper lifetime management of these memory types.

### Quick Summary

In this section, we explored Choreo's approach to parallelism in heterogeneous computing environments, focusing on the `parallel-by` block syntax and the management of multiple and sub-level parallelism. We also discussed how Choreo abstracts kernel launches and imposes memory management constraints, enabling us to leverage parallel hardware effectively.

However, while parallelism enhances efficiency, loops remain essential. In the next section, we will introduce how to build loops in Choreo.
