// RUN: coir-opt %s | FileCheck %s

module attributes { gpu.container_module } {
func.func @kernel(%A: memref<128x128xf16, 1>, %S: memref<128x128xf16, 3>,
                  %coords: index, %mb: i32) {
  %cst_shape = arith.constant dense<[128, 128]> : tensor<2xi64>
  %cst_stride = arith.constant dense<[128, 1]> : tensor<2xi64>
  %d0 = coir.dma.const.desc %A, %cst_shape, %cst_stride : memref<128x128xf16, 1>, tensor<2xi64>, tensor<2xi64> -> !coir.desc
  coir.dma.check %d0 : !coir.desc
  %d1 = coir.dma.prefetch.desc %d0 : !coir.desc -> !coir.desc.rt
  %d2 = coir.dma.runtime.desc %d1, %coords, %S, %mb : !coir.desc.rt, index, memref<128x128xf16, 3>, i32 -> !coir.desc.rt
  %t  = coir.dma.invoke %d2 : !coir.desc.rt
  async.await %t : !async.token
  func.return
}
}

// CHECK: coir.dma.const.desc
// CHECK: coir.dma.check
// CHECK: coir.dma.prefetch.desc
// CHECK: coir.dma.runtime.desc
// CHECK: coir.dma.invoke
