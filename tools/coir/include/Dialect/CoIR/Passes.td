//===- Passes.td - CoIR dialect passes ----------------------*- tablegen -*-===//
//
// Pass definitions for CoIR dialect transformations.
//
//===----------------------------------------------------------------------===//

#ifndef COIR_PASSES_TD
#define COIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ClassifyCopies : Pass<"coir-classify-copies"> {
  let summary = "Validate copy ops against target capabilities";
  let description = [{
    Validates that coir.tma.copy ops are supported by the target.
    If the target lacks TMA (coir.has_tma = false or absent), any
    coir.tma.copy op triggers a compile-time error.

    coir.dma.copy and coir.element.copy pass through unchanged.

    Target capabilities are read from module attributes set by the
    driver via Target::HasTMA() / Target::HasDMA():

    - "coir.has_tma" (bool) -- target has TMA hardware
    - "coir.has_dma" (bool) -- target has DMA engine for cross-level
      transfers (global<->shared, global<->local, shared<->local)
  }];

  let constructor = "::coir::createClassifyCopiesPass()";

  let dependentDialects = ["::coir::CoIRDialect"];
}

def LowerMMA : Pass<"coir-lower-mma"> {
  let summary = "Lower coir.mma ops to target-specific MMA instructions";
  let description = [{
    Marks coir.mma.fill/load/exec/store as lowered and annotates
    mma.exec with the target MMA strategy:

    - "wgmma"    : SM90+ warp-group MMA
    - "mma_sync" : SM80 cooperative MMA
    - "ukernel"  : library-based micro-kernel MMA (non-GPU targets)

    The MMA target is read from the module attribute "coir.mma_target"
    (set by the driver via Target::MMATargetName()). For standalone
    coir-opt tests, embed coir.mma_target in the test IR.
  }];

  let constructor = "::coir::createLowerMMAPass()";

  let dependentDialects = [
    "::coir::CoIRDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def LowerCopy : Pass<"coir-lower-copy"> {
  let summary = "Lower specialized copy ops to target transfer patterns";
  let description = [{
    Expands coir.dma.copy, coir.tma.copy, and coir.element.copy
    into the corresponding low-level transfer and synchronization
    sequences:

    - element.copy -> element-copy annotation
    - dma.copy     -> cp.async / DTE annotation
    - tma.copy     -> TMA descriptor annotation
  }];

  let constructor = "::coir::createLowerCopyPass()";

  let dependentDialects = [
    "::coir::CoIRDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def LowerDMADesc : Pass<"coir-lower-dma-desc"> {
  let summary = "Decompose DMA copy ops into descriptor pipeline";
  let description = [{
    Target-driven decomposition of coir.dma.copy inside coir.foreach loops into the DMA
    descriptor pipeline:

    1. coir.dma.const.desc    -- captures loop-invariant geometry
    2. coir.dma.prefetch.desc -- materializes the descriptor
    3. coir.dma.runtime.desc  -- binds per-iteration offsets
    4. coir.dma.invoke         -- triggers the transfer

    The pass reads the module attribute "coir.has_dma" to decide
    whether to apply descriptor decomposition:

    - has_dma=true: applies descriptor pipeline
    - has_dma=false: skips (dma.copy stays for direct emission)
    - No attr: applies (assumed standalone test)

    The const.desc and prefetch.desc are placed before the loop by
    construction, enabling the hoist pass to verify placement.
  }];

  let constructor = "::coir::createLowerDMADescPass()";

  let dependentDialects = [
    "::coir::CoIRDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def PlanDMACopy : Pass<"coir-plan-dma-copy"> {
  let summary = "Plan cooperative tiled copy strategy for DMA invoke ops";
  let description = [{
    Analyzes DMAInvokeOp instances that use the non-TMA descriptor pipeline
    and determines whether cooperative tiled copy is profitable. When it is,
    the pass stamps thr_layout, val_layout, copy_atom, need_pred, prediction,
    and swizzle attributes on the DMAInvokeOp. The EmitCUDA backend reads
    these to emit `choreo::tiled_copy<...>` instead of `choreo::naive_copy`.

    The planning logic considers:
    - Participating thread count (from enclosing parallel ops or launch_bounds)
    - Tile box shape (flattened to 2D: M rows x N cols)
    - Memory alignment for vectorization factor selection
    - Copy atom selection (cp.async for G2S with warp-multiple threads,
      AutoVectorizingCopy otherwise)
    - Predication when box_M is not divisible by thread_M

    Runs after coir-hoist-dma-config and before coir-lower-copy.
  }];

  let constructor = "::coir::createPlanDMACopyPass()";

  let dependentDialects = ["::coir::CoIRDialect"];
}

def HoistDMAConfig : Pass<"coir-hoist-dma-config"> {
  let summary = "Hoist loop-invariant DMA descriptor setup above loops";
  let description = [{
    Performs loop-invariant code motion (LICM) on DMA descriptor ops:

    - Moves coir.dma.const.desc and coir.dma.prefetch.desc above
      coir.foreach loops when their operands are loop-invariant
    - Keeps coir.dma.runtime.desc inside the loop when offsets
      depend on the induction variable
    - If runtime.desc offsets are also invariant, hoists it too

    This is a safety-net pass: the lower-dma-desc pass already places
    const.desc outside loops by construction.
  }];

  let constructor = "::coir::createHoistDMAConfigPass()";

  let dependentDialects = ["::coir::CoIRDialect"];
}

def HoistAssertions : Pass<"coir-hoist-assertions"> {
  let summary = "Hoist runtime assertions to earliest safe emission site";
  let description = [{
    Performs LICM-style hoisting on coir.assert ops to reduce their
    runtime cost by moving them out of hot loops when possible:

    - If all values feeding the condition are kernel arguments
      (loop-invariant at kernel level), promote to site=ENTRY and
      move to the kernel entry block.
    - If the condition is loop-invariant w.r.t. an enclosing
      coir.foreach / scf.for, hoist to just before that loop and
      set site=HOIST.
    - Assertions inside conditional regions (scf.if) are NOT hoisted
      past the conditional boundary.

    ENTRY assertions are emitted by backends as host-side checks
    before kernel launch; HOIST/USE assertions become device-side
    checks. This is target-agnostic and applies to all backends.
  }];

  let constructor = "::coir::createHoistAssertionsPass()";
  let dependentDialects = ["::coir::CoIRDialect"];
}

def EstimateAssertCost : Pass<"coir-estimate-assert-cost"> {
  let summary = "Estimate cost and filter runtime assertions by threshold";
  let description = [{
    Walks all coir.assert ops (after hoisting) and stamps three
    attributes on each:

    - estimated_cost (I64): raw trip-count product from enclosing
      loops. coir.foreach uses static bounds when available (else
      100); scf.while / coir.while use 100 conservatively.
    - cost_class (AssertCost enum): classified tier derived from
      estimated_cost (1->ENTRY, 2-49->LOW, 50-499->MEDIUM, >=500->HIGH).
    - enabled (Bool): true iff the tier passes the cost-threshold.

    The cost-threshold option maps to Choreo's --runtime-check / -rtc
    levels: 0=none, 1=entry, 2=low, 3=medium, 4=high/all.

    Run after coir-hoist-assertions so cost reflects the final
    position.
  }];

  let constructor = "::coir::createEstimateAssertCostPass()";
  let dependentDialects = ["::coir::CoIRDialect"];
  let options = [
    Option<"costThreshold", "cost-threshold", "int", /*default=*/"4",
           "Maximum AssertCost tier to enable"
           " (0=none, 1=entry, 2=low, 3=medium, 4=high/all)."
           " Assertions whose tier >= threshold are disabled.">
  ];
}

def CollectAssertStats : Pass<"coir-collect-assert-stats"> {
  let summary = "Collect assertion statistics from stamped cost attributes";
  let description = [{
    Walks all coir.assert ops and accumulates per-tier, per-usage,
    and enabled/disabled counts into the compilation context's
    AssessmentStats (accessible via CCtx()). This pass does not
    modify the IR.

    Run after coir-estimate-assert-cost so that the cost attributes
    are present.
  }];

  let constructor = "::coir::createCollectAssertStatsPass()";
  let dependentDialects = ["::coir::CoIRDialect"];
}

def EmitCUDA : Pass<"coir-emit-cuda", "::mlir::ModuleOp"> {
  let summary = "Emit CUDA/C++ source from lowered CoIR IR";
  let description = [{
    Walks the module and emits CUDA kernel source text to stdout
    or a specified output file. Expects the IR to have been lowered
    through classify-copies, lower-mma, and lower-copy passes.

    The emitter handles:
    - coir.kernel -> __global__ void kernel(...)
    - coir.parallel -> grid/block dimension mapping
    - coir.foreach -> for loop
    - coir.mma.* -> CUDA WMMA/WGMMA API calls
    - coir.{dma,tma,element}.copy -> memcpy_async/TMA/element copy
    - coir.barrier -> __syncthreads() / barrier
  }];

  let constructor = "::coir::createEmitCUDAPass()";
  let dependentDialects = ["::coir::CoIRDialect"];
}

def EmitHIP : Pass<"coir-emit-hip", "::mlir::ModuleOp"> {
  let summary = "Emit HIP/C++ source from lowered CoIR IR";
  let description = [{
    Walks the module and emits HIP kernel source text to stdout
    or a specified output file. Expects the IR to have been lowered
    through classify-copies and lower-copy passes.

    The emitter handles:
    - coir.kernel -> __global__ void kernel(...)
    - coir.parallel -> grid/block dimension mapping
    - coir.foreach -> for loop
    - coir.element.copy -> cooperative thread loop
    - coir.barrier -> __syncthreads()
    - MMA ops are rejected (unsupported on AMD RDNA)
  }];

  let constructor = "::coir::createEmitHIPPass()";
  let dependentDialects = ["::coir::CoIRDialect"];
}

def EmitCC : Pass<"coir-emit-cc", "::mlir::ModuleOp"> {
  let summary = "Emit portable C++ source from CoIR IR for CPU execution";
  let description = [{
    Walks the module and emits C++ source for CPU execution.
    Parallel levels become nested for-loops, DMA becomes std::memcpy,
    and barriers are no-ops.
  }];

  let constructor = "::coir::createEmitCCPass()";
  let dependentDialects = ["::coir::CoIRDialect"];
}

#endif // COIR_PASSES_TD
