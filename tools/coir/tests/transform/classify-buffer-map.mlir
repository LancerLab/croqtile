// RUN: coir-opt %s -verify-diagnostics

// Test: without coir.has_buffer_map, buffer.map/remap are rejected by op verifier.
module {
  coir.kernel @test_buffer_map_rejected(
      %src: !coir.tensor<128xf32>,
      %off: index,
      %sz: index) {
    // expected-error@+1 {{requires explicit memory mapping support but target does not provide it}}
    %mapped = coir.buffer.map %src[%off] size(%sz)
        : !coir.tensor<128xf32> -> !coir.tensor<64xf32, local>
    coir.return
  }
}
