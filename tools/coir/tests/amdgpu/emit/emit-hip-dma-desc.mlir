// RUN: coir-opt --coir-emit-hip %s 2>&1 | FileCheck %s

module attributes { "coir.has_tma" = false, "coir.has_dma" = false, "coir.mma_target" = "" } {

// Test: DMA descriptor pipeline emits actual cooperative copy on HIP
// CHECK: __global__ void __desc_copy_kernel__
coir.kernel @desc_copy(%src: !coir.tensor<128xf32, global>,
                       %dst: !coir.tensor<128xf32, shared>) {
  %desc = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<copy>}
      : !coir.tensor<128xf32, global>,
        !coir.tensor<128xf32, shared> -> !coir.desc
  coir.dma.check %desc : !coir.desc
  %rt = coir.dma.prefetch.desc %desc : !coir.desc -> !coir.desc.rt
  %token = coir.dma.invoke %rt : !coir.desc.rt

  // The invoke should emit a cooperative copy loop, not just syncthreads
  // CHECK: for (size_t __i = threadIdx.x; __i < 128; __i += blockDim.x)
  // CHECK: __syncthreads

  coir.wait %token : !coir.async
}

} // module
