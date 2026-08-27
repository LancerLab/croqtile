// RUN: coir-opt --coir-emit-topscc %s | FileCheck %s
//
// Test memory-visibility fence emission on topscc:
//   - a directional coir.fence (release) emits a store fence
//   - a directional coir.fence (acquire) emits a load fence
//   - the full acq-rel fence emits the plain memory fence

module attributes { "coir.target" = "topscc", "coir.arch" = "gcu300" } {

coir.kernel @fence_emit(
    %src: !coir.tensor<64xf32, shared>) -> !coir.tensor<64xf32, local> {

  %dst = coir.tensor.alloc : !coir.tensor<64xf32, local>

  // CHECK: tcle::fence<tcle::FenceType::L1_VDMEM_STORE>();
  coir.fence <local> <dma> <release> <none>
  %tok = coir.dma.copy %src to %dst
    : !coir.tensor<64xf32, shared> -> !coir.tensor<64xf32, local>

  coir.wait %tok : !coir.async

  // CHECK: tcle::fence<tcle::FenceType::L2_MEM_STORE>();
  coir.fence <shared> <dma> <release> <none>

  // CHECK: tcle::fence<tcle::FenceType::L3_MEM_LOAD>();
  coir.fence <global> <dma> <acquire> <none>

  // CHECK: tcle::fence<tcle::FenceType::L3_MEM>();
  coir.fence <global> <all> <acq_rel> <block>

  coir.return %dst : !coir.tensor<64xf32, local>
}

}
