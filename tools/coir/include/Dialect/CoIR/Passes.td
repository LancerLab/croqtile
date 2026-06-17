//===- Passes.td - CoIR dialect passes ----------------------*- tablegen -*-===//
//
// Pass definitions for CoIR dialect transformations.
//
//===----------------------------------------------------------------------===//

#ifndef COIR_PASSES_TD
#define COIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ClassifyCopies : Pass<"coir-classify-copies"> {
  let summary = "Lower coir.data.copy to specialized copy ops";
  let description = [{
    Analyzes source and destination memory spaces of coir.data.copy
    operations and replaces them with the appropriate specialized op:

    - global <-> shared with TMA: coir.tma.copy
    - global <-> shared without TMA: coir.dma.copy
    - shared -> register / register -> shared: coir.thread.copy
    - shared -> shared (same level): coir.thread.copy

    TMA capability is read from the module attribute "coir.has_tma"
    (set by the driver). When absent, --target-arch is used as fallback.

    Async coir.data.copy ops produce tokens that are threaded through
    to the specialized async ops.
  }];

  let constructor = "::coir::createClassifyCopiesPass()";

  let options = [
    Option<"targetArch", "target-arch", "std::string",
           /*default=*/"\"sm_86\"",
           "Fallback architecture when module lacks coir.has_tma">
  ];

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
    (set by the driver). When absent, --target-arch is used as fallback.
  }];

  let constructor = "::coir::createLowerMMAPass()";

  let options = [
    Option<"targetArch", "target-arch", "std::string",
           /*default=*/"\"sm_86\"",
           "Fallback architecture when module lacks coir.mma_target">
  ];

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
    Expands coir.dma.copy, coir.tma.copy, and coir.thread.copy
    into the corresponding low-level transfer and synchronization
    sequences:

    - thread.copy -> scf.for element-copy loop
    - dma.copy    -> cp.async bulk copy + barrier
    - tma.copy    -> TMA descriptor pipeline
  }];

  let constructor = "::coir::createLowerCopyPass()";

  let options = [
    Option<"targetArch", "target-arch", "std::string",
           /*default=*/"\"sm_86\"",
           "Fallback architecture (reserved for future use)">
  ];

  let dependentDialects = [
    "::coir::CoIRDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def LowerDMADesc : Pass<"coir-lower-dma-desc"> {
  let summary = "Decompose copy ops into DMA descriptor pipeline";
  let description = [{
    Rewrites coir.data.copy (and classified dma.copy) operations inside
    coir.foreach loops into the DMA descriptor pipeline:

    1. coir.dma.const.desc   -- captures loop-invariant geometry
    2. coir.dma.prefetch.desc -- materializes the descriptor
    3. coir.dma.runtime.desc -- binds per-iteration offsets
    4. coir.dma.invoke        -- triggers the transfer

    The const.desc and prefetch.desc are placed before the loop by
    construction, enabling the hoist pass to verify placement.
  }];

  let constructor = "::coir::createLowerDMADescPass()";

  let dependentDialects = [
    "::coir::CoIRDialect",
    "::mlir::arith::ArithDialect"
  ];
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
    - coir.{dma,tma,thread}.copy -> memcpy_async/TMA/element copy
    - coir.barrier -> __syncthreads() / barrier
  }];

  let constructor = "::coir::createEmitCUDAPass()";
  let dependentDialects = ["::coir::CoIRDialect"];
}

#endif // COIR_PASSES_TD
