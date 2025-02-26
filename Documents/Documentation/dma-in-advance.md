## Overview
The data movement is essential for *tileflow programs*. In this section, you will learn more advanced syntax of *DMA statements* to support different scenarios.

## Advances in *Data Expression*
### Reshape with `.span_as`


### Dimension Composition inside `chunkat`

### Full Non-Blocking DMA Mode (Chain Mode in Choreo)
In Choreo, it's possible to perform a full non-blocking DMA by chaining multiple asynchronous DMA operations and using event-based notifications with after. This enables complete non-blocking execution, where one DMA operation is triggered only after the completion of a prior one. Here’s an example of such a setup:

```choreo
out_store = dma.copy.async l2_out => output.chunkat(m_tile, n_tile) after out_store_s;
```
In this example:

- `out_store` is the asynchronous DMA operation that transfers data from l2_out to output.chunkat(m_tile, n_tile).
- `out_store_s` is another DMA operation or event that must complete before out_store can proceed.
- The after out_store_s syntax specifies that `out_store` should only start after the completion of `out_store_s`, which ensures that there is no blocking in the main thread.

In this case, neither `out_store` nor `out_store_s` will block the main program flow. The program continues executing while these DMA operations are handled in the background. The key difference here is that the completion of `out_store_s` triggers the start of `out_store`, creating an event-driven dependency between the two DMA operations. This model enables highly efficient and non-blocking memory transfers.

## Dynamic Shape

### The Placeholders


## Mutate the Current Value


