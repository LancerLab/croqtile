## Sequentialism and Parallelism

Repetitive tasks are common in programming. These tasks can be executed either iteratively, using loops, or in parallel, which requires both hardware and software support for parallelization.

Currently, the most widely used parallelization paradigm is **Single Program, Multiple Data (SPMD)**. In this paradigm, code is written once but executed as multiple instances simultaneously. Each instance operates independently unless synchronization is explicitly implemented. Choreo adopts this paradigm and abstracts parallelism using the **parallel-by** block.

Looping, on the other hand, is a more straightforward concept and is the most common method for writing sequential code. In Choreo, programmers can construct loops using the **with-in + foreach** block. This structure supports *do-while* style iterations, which are particularly useful in scenarios involving multiple buffers, given Choreo's primary focus on orchestrating data movement.

In the following sections, we will delve into the details of constructing loops and implementing parallelism in Choreo.
