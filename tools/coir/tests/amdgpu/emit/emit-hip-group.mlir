// RUN: coir-opt --coir-emit-hip %s 2>&1 | FileCheck %s

module attributes { "coir.has_tma" = false, "coir.has_dma" = false, "coir.mma_target" = "" } {

// CHECK: __global__ void __group_kernel_kernel__
coir.kernel @group_kernel(%arg0: !coir.tensor<128xf32>, %arg1: !coir.tensor<128xf32>) {
  coir.parallel (%bid) in [2] level = #coir.level<block> {
    coir.parallel (%gid) in [2] level = #coir.level<group> {
      coir.parallel (%tid) in [32] level = #coir.level<thread> {
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %g_off = arith.muli %gid, %c32 : index
        %b_off = arith.muli %bid, %c64 : index
        %off1 = arith.addi %b_off, %g_off : index
        %idx = arith.addi %off1, %tid : index
        %v = coir.tensor.load_elem %arg0[%idx] : !coir.tensor<128xf32> -> f32
        coir.tensor.store_elem %v, %arg1[%idx] : f32, !coir.tensor<128xf32>
      }
    }
  }
}

// GROUP: warp ID from threadIdx.x / 32
// CHECK: parallel level=group
// CHECK: parallel level=thread
// CHECK: threadIdx.x / 32
// CHECK: threadIdx.x % 32

// Host wrapper: 2 groups * 32 threads = 64 block size
// CHECK: __group_kernel_kernel__<<<2, 64>>>

} // module
