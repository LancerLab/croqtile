// RUN: coir-opt --coir-lower-mma %s | FileCheck %s
// RUN: coir-opt --coir-lower-mma=target-arch=sm_80 %s | FileCheck %s --check-prefix=SM80
// RUN: coir-opt --coir-lower-mma=target-arch=ukernel %s | FileCheck %s --check-prefix=UKERNEL

// Test: full pipeline - SM90 gets wgmma target
// CHECK-LABEL: coir.kernel @test_pipeline_sm90
// CHECK: coir.mma.fill {{.*}} {lowered}
// CHECK: coir.mma.load {{.*}} {lowered}
// CHECK: coir.mma.load {{.*}} {lowered}
// CHECK: coir.mma.exec {{.*}} lowered{{.*}}target = "wgmma"
// CHECK: coir.mma.store {{.*}} {lowered}
coir.kernel @test_pipeline_sm90(
    %a_tile: !coir.tensor<16x16xf16, shared>,
    %b_tile: !coir.tensor<16x16xf16, shared>,
    %c_tile: !coir.tensor<16x16xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  %a_frag = coir.mma.load %a_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %b_frag = coir.mma.load %b_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %res = coir.mma.exec %acc, %a_frag, %b_frag {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
  coir.mma.store %res, %c_tile : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

// Test: SM80 pipeline gets mma_sync target
// SM80-LABEL: coir.kernel @test_pipeline_sm80
// SM80: coir.mma.exec {{.*}} target = "mma_sync"
coir.kernel @test_pipeline_sm80(
    %a_tile: !coir.tensor<16x16xf16, shared>,
    %b_tile: !coir.tensor<16x16xf16, shared>,
    %c_tile: !coir.tensor<16x16xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  %a_frag = coir.mma.load %a_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %b_frag = coir.mma.load %b_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %res = coir.mma.exec %acc, %a_frag, %b_frag {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
  coir.mma.store %res, %c_tile : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

// Test: non-GPU architecture gets ukernel target via --target-arch fallback
// UKERNEL-LABEL: coir.kernel @test_pipeline_sm90
// UKERNEL: coir.mma.exec {{.*}} target = "ukernel"

// Test: accumulate loop with lowered MMA
// CHECK-LABEL: coir.kernel @test_accumulate
// CHECK: coir.mma.fill {{.*}} {lowered}
// CHECK: coir.foreach
// CHECK:   coir.mma.exec {{.*}} target = "wgmma"
// CHECK: coir.mma.store {{.*}} {lowered}
coir.kernel @test_accumulate(
    %a: !coir.tensor<128x64xf16, shared>,
    %b: !coir.tensor<64x128xf16, shared>,
    %c: !coir.tensor<128x128xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  %c4 = arith.constant 4 : index
  %init_acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  %final = coir.foreach %k in %c4 iter_args(%acc = %init_acc) : !coir.mma_frag<16x16xf32> {
    %at = coir.tensor.tile %a[%k] : !coir.tensor<128x64xf16, shared> -> !coir.tensor<16x16xf16, shared>
    %bt = coir.tensor.tile %b[%k] : !coir.tensor<64x128xf16, shared> -> !coir.tensor<16x16xf16, shared>
    %af = coir.mma.load %at : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
    %bf = coir.mma.load %bt : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
    %r = coir.mma.exec %acc, %af, %bf {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
    coir.yield %r : !coir.mma_frag<16x16xf32>
  }
  coir.mma.store %final, %c : !coir.mma_frag<16x16xf32>, !coir.tensor<128x128xf32, shared>
}
