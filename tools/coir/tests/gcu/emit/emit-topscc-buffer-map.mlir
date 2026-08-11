// RUN: coir-opt --coir-emit-topscc %s 2>&1 | FileCheck %s

// Test: GCU codegen emits tops::map_mem_m / tops::unmap_mem_m calls for
// buffer.map and buffer.unmap ops with explicit memory mapping.

// CHECK: tops::map_mem_m(
// CHECK: tops::unmap_mem_m(

module attributes { "coir.has_buffer_map" = true } {

coir.kernel @test_buffer_map(
    %src: !coir.tensor<128xf32>,
    %off: index,
    %sz: index) {
  %mapped = coir.buffer.map %src[%off] size(%sz)
      : !coir.tensor<128xf32> -> !coir.tensor<64xf32, local>
  coir.buffer.unmap %mapped, %sz : !coir.tensor<64xf32, local>
  coir.return
}

}
