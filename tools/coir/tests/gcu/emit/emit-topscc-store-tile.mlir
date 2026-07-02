// RUN: coir-opt --coir-emit-topscc %s | FileCheck %s
//
// Test coir.tensor.store_tile emission on topscc:
//   - Tile from tensor.alloc: emits flat copy loop (writeback)
//   - Tile from tensor.tile:  no-op (pointer alias)

module attributes { "coir.target" = "topscc", "coir.arch" = "gcu300" } {

// Case 1: tile from tensor.alloc -> copy loop
coir.kernel @store_tile_alloc(
    %src: !coir.tensor<128xf32, global>) -> !coir.tensor<128xf32, global> {

  %alloc = coir.tensor.alloc : !coir.tensor<64xf32, local>

  %c32 = arith.constant 32 : index
  coir.tensor.store_tile %alloc, %src[%c32] : !coir.tensor<64xf32, local>, !coir.tensor<128xf32, global>

  // CHECK: for (int [[IDX:i[0-9]+]] = 0; [[IDX]] < 64; ++[[IDX]])
  // CHECK-NEXT: {{.+\[.+ \+ }}[[IDX]]{{.*\] = .+\[}}[[IDX]]{{\]}}

  coir.return %src : !coir.tensor<128xf32, global>
}

// Case 2: tile from tensor.tile -> no-op (pointer alias writeback is implicit)
// CHECK-LABEL: store_tile_alias
coir.kernel @store_tile_alias(
    %src: !coir.tensor<128xf32, global>) -> !coir.tensor<128xf32, global> {

  %c0 = arith.constant 0 : index
  %tile = coir.tensor.tile %src[%c0] : !coir.tensor<128xf32, global> -> !coir.tensor<64xf32, global>

  coir.tensor.store_tile %tile, %src[%c0] : !coir.tensor<64xf32, global>, !coir.tensor<128xf32, global>

  // CHECK-NOT: for (int {{.*}} = 0;

  coir.return %src : !coir.tensor<128xf32, global>
}

}
