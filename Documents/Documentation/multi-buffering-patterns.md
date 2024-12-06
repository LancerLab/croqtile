# Multi-Buffering Optimization with Choreo

To aggressively optimize the high-performance computation kernels, one effective method is to overlap computation with DMA (Direct Memory Access) data movement. This approach, known as **multi-buffering optimization**, typically requires programmers to manage multiple buffers.

In Choreo, this method requires two key programming primitives: the 'dummy' future **dma.any**, and the **swap/rotate** function. Let's illustrate by an example:

```choreo
__co__ s32 [6, 17, 128] ele_add(s32 [6, 17, 128] lhs,
                                s32 [6, 17, 128] rhs) {
  s32[lhs.span] output;
  parallel p by 6 {
    with index = {x, y} in [17, 4] {
      local s32 [rhs.span / {#p, #x, #y}] l_out;
      foreach x { // first bunch
        lfA = dma.copy.async lhs.chunkat(p, x, y) => local;
        rfA = dma.copy.async rhs.chunkat(p, x, y) => local;
        lfB = dma.any;  // dummy DMA used for multi-buffering
        rfB = dma.any;
        foreach y(1:) {
          lfB = dma.copy.async lhs.chunkat(p, x, y) => local;
          rfB = dma.copy.async rhs.chunkat(p, x, y) => local;
          wait lfA, rfA;  // wait another bunch
          call kernel(lfA.data, rfA.data, l_out, |lfB.span|);
          dma.copy l_out => output.chunkat(p, x, y - 1);
          swap(lfA, lfB); // exchange futures
          swap(rfA, rfB);
        }
        lf = select(#y % 2, lfB, lfA);
        rf = select(#y % 2, rfB, rfA);
        wait lf, rf;  // handle the last bunch
        call kernel(lf.data, rf.data, l_out, |lfB.span|);
        dma.copy l_out => output.chunkat(p, x, y(-1));
      }
    }
  }
  return output;
}
```

In choreo function `ele_add`, there are two input parameters: 'lhs' and 'rhs', both double-buffered in local storage. We refer to these buffers as 'A' and 'B'. While buffer A is used for computation, buffer B is filled with data via a DMA operation. Once the kernel function completes computation on buffer A, the roles of A and B are swapped: A becomes the buffer for loading the next data chunk, and B becomes the buffer for computation. This process iterates until all data chunks are consumed.

The implementation divides data movements and computations into three stages, including:

 - the **prologue**(line 8-11). It (pre-)load the first chunk of data. In this example, buffer A is pre-loaded with data.
 - the **body**(line 12-20). It processes the data pre-loaded in last iteration or in prologue, and pre-load the data for next iteration.
 - and the **epilogue**(line 21-25). It processes the last data chunk only.

In the code, futures 'lfB' and 'rfB' are declared as dummies. These dummy futures serve as placeholders and are replaced by DMA statements at lines 13-14. The reason for declaring dummy futures is that the futures ('lfB' and 'rfB') of the DMA invoked in the body stage are used in the epilogue stage (lines 21-22). However, from a lexical scope perspective, defining futures inside the foreach-block (lines 12-20) would not extend their lifetime to their last uses. Therefore, it is necessary to declare 'lfB' and 'rfB' early.

The **swap** statements at lines 18-19 exchange futures to ensure the **wait** statement at line 15 functions correctly. Note that only futures with identical DMA operations can be swapped. Additionally, the iteration range of the *foreach* statement at line 12 is adjusted. `foreach y(1:)` means the loop over the *bounded variable* 'y' starts from the second value up to the upper bound. In line 17, the *chunkat* expression is also tuned. `y - 1` results in the current value of 'y' minus 1, setting up the body stage to work properly with the prologue stage. This adjustment is necessary because the iteration count must be reduced by 1, and the pre-load operation should fetch the 'next' chunk while computation is conducted on the 'current' chunk of data. Alternatively, you can use `foreach y(:-1)` and adjust the chunkat expression of the pre-load statements at lines 13-14 to achieve the same effect.

In line 25, the expression `y(-1)` retrieves the 'last value' of 'y' within its bound.

Note that the **select** expressions in line 21-22 handle both odd and even upper-bound cases for the bounded variable 'y'. In either case, it obtains the future of the last chunk.

If the programmer want to enable more than two buffers, **rotate** statements is required to replace **swap**.
