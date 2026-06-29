// Full double-buffer DMA pipeline verification test.
// Verifies the design from Documents/internal/foreach-iterargs-design.md:
//   1. LowerDMADesc decomposes ALL global<->shared copies (including tiled)
//   2. HoistDMAConfig hoists loop-invariant const.desc + prefetch
//   3. Loop-variant runtime.desc + invoke stay inside the loop
//   4. async.undef passes through lowering untouched
//   5. Post-loop wait uses foreach results (no liveness pruning)
//
// RUN: coir-opt --coir-lower-dma-desc --coir-hoist-dma-config %s | FileCheck %s

// CHECK-LABEL: coir.kernel @double_buffer_full
coir.kernel @double_buffer_full(
    %input: !coir.tensor<1024xf32, global>,
    %output: !coir.tensor<1024xf32, global>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  %buf_a = coir.tensor.alloc : !coir.tensor<256xf32, shared>
  %buf_b = coir.tensor.alloc : !coir.tensor<256xf32, shared>

  // fa = dma.copy input => buf_a (non-tiled, global->shared)
  %fa_init = coir.dma.copy %input to %buf_a
    : !coir.tensor<1024xf32, global> -> !coir.tensor<256xf32, shared>

  // fb = dma.any (placeholder)
  %fb_init = coir.async.undef : !coir.async

  // --- Design 1: fa initial copy decomposed ---
  // CHECK: %[[BUF_A:.*]] = coir.tensor.alloc : !coir.tensor<256xf32, shared>
  // CHECK: %[[BUF_B:.*]] = coir.tensor.alloc : !coir.tensor<256xf32, shared>
  // CHECK: %[[FA_DESC:.*]] = coir.dma.const.desc %{{.*}}, %[[BUF_A]] {kind = #coir.dma_kind<copy>}
  // CHECK: %[[FA_PRE:.*]] = coir.dma.prefetch.desc %[[FA_DESC]]
  // CHECK: %[[FA_TOK:.*]] = coir.dma.invoke %[[FA_PRE]]

  // --- Design 2: async.undef survives lowering ---
  // CHECK: %[[FB_TOK:.*]] = coir.async.undef : !coir.async

  // --- Design 3: fb's const.desc + prefetch HOISTED above loop ---
  // CHECK: %[[FB_DESC:.*]] = coir.dma.const.desc %{{.*}}, %[[BUF_B]] {kind = #coir.dma_kind<slice>}
  // CHECK: %[[FB_PRE:.*]] = coir.dma.prefetch.desc %[[FB_DESC]]

  // --- Design 3b: output copy's const.desc + prefetch HOISTED above loop ---
  // CHECK: %[[OUT_DESC:.*]] = coir.dma.const.desc %[[BUF_A]], %{{.*}} {kind = #coir.dma_kind<slice>}
  // CHECK: %[[OUT_PRE:.*]] = coir.dma.prefetch.desc %[[OUT_DESC]]

  // --- Design 4: foreach with iter_args carrying both tokens ---
  // CHECK: %[[RES:.*]]:2 = coir.foreach %[[IV:.*]] in %{{.*}} iter_args(%[[AFA:.*]] = %[[FA_TOK]], %[[AFB:.*]] = %[[FB_TOK]]) : !coir.async, !coir.async
  %res:2 = coir.foreach %iv in %c4
      iter_args(%arg_fa = %fa_init, %arg_fb = %fb_init) : !coir.async, !coir.async {

    // fb = dma.copy.async input.chunkat(i) => buf_b (tiled, global->shared)
    %tile_in = coir.tensor.tile %input[%iv]
      : !coir.tensor<1024xf32, global> -> !coir.tensor<256xf32, global>
    %fb_new = coir.dma.copy %tile_in to %buf_b
      : !coir.tensor<256xf32, global> -> !coir.tensor<256xf32, shared>

    // --- Design 5: inside loop only runtime.desc + invoke for fb ---
    // CHECK: %[[FB_RT:.*]] = coir.dma.runtime.desc %[[FB_PRE]] offsets(%[[IV]])
    // CHECK: %[[FB_INV:.*]] = coir.dma.invoke %[[FB_RT]]

    // wait fa
    coir.wait %arg_fa : !coir.async
    // CHECK: coir.wait %[[AFA]]

    // output: buf_a -> output.chunkat(i-1) (tiled, shared->global)
    %off = arith.subi %iv, %c1 : index
    %tile_out = coir.tensor.tile %output[%off]
      : !coir.tensor<1024xf32, global> -> !coir.tensor<256xf32, global>
    %out_tok = coir.dma.copy %buf_a to %tile_out
      : !coir.tensor<256xf32, shared> -> !coir.tensor<256xf32, global>
    coir.wait %out_tok : !coir.async

    // --- Design 6: output copy runtime.desc + invoke stay inside loop ---
    // CHECK: coir.dma.runtime.desc %[[OUT_PRE]] offsets(%{{.*}})
    // CHECK: coir.dma.invoke
    // CHECK: coir.wait

    // swap(fa, fb)
    %rot:2 = coir.async.rotate %arg_fa, %fb_new : !coir.async

    // --- Design 7: rotate consumes iter_arg + invoke, yield carries both ---
    // CHECK: %[[ROT:.*]]:2 = coir.async.rotate %[[AFA]], %[[FB_INV]]
    // CHECK: coir.yield %[[ROT]]#0, %[[ROT]]#1 : !coir.async, !coir.async
    coir.yield %rot#0, %rot#1 : !coir.async, !coir.async
  }

  // --- Design 8: post-loop waits both results (no dead-arg elimination) ---
  // CHECK: coir.wait %[[RES]]#0
  // CHECK: coir.wait %[[RES]]#1
  coir.wait %res#0 : !coir.async
  coir.wait %res#1 : !coir.async
}
