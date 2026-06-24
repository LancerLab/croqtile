// CoIROps.td -- Operation definitions for the CoIR dialect
#ifndef COIR_OPS_TD
#define COIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Async/IR/AsyncTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"

include "CoIR/CoIRDialect.td"
include "CoIR/CoIRAttrs.td"
include "CoIR/CoIRTypes.td"

// Base class for CoIR ops
class CoIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<CoIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// DMA descriptor pipeline ops
//
// Target-neutral ops for DMA configuration hoisting. The pipeline:
//   dma.const.desc  -- capture loop-invariant transfer geometry
//   dma.prefetch.desc -- materialize descriptor (prefetch or no-op per target)
//   dma.runtime.desc -- bind per-iteration offsets / coordinates
//   dma.invoke       -- execute the transfer
//
// Target backends lower these ops to hardware-specific APIs.
// Example mappings:
//   const.desc  -> descriptor/context configuration
//   runtime.desc -> per-iteration offset binding
//   invoke       -> transfer trigger
//===----------------------------------------------------------------------===//

def DMAConstDescOp : CoIR_Op<"dma.const.desc", [NoMemoryEffect]> {
  let summary = "Create compile-time DMA descriptor from source/dest tensors";
  let description = [{
    Captures the loop-invariant transfer geometry (shapes, strides, element
    types, memory spaces) from source and destination tensors. The `kind`
    attribute selects the transfer operation: copy, slice, transpose, or pad.
    The resulting `!coir.desc` handle is target-neutral and can be hoisted
    above loops.

    Example:
      %d = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<copy>}
             : !coir.tensor<128x64xf32, global>,
               !coir.tensor<128x64xf32, local> -> !coir.desc
  }];
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    CoIR_TensorTypeC:$dest,
    CoIR_DMAKindAttr:$kind
  );
  let results = (outs CoIR_DMADescType:$out);
  let hasCustomAssemblyFormat = 1;
}

def DMACheckOp : CoIR_Op<"dma.check", [NoMemoryEffect]> {
  let summary = "Optional compile-time descriptor validation";
  let arguments = (ins CoIR_DMADescType:$in);
  let assemblyFormat = "$in attr-dict `:` type($in)";
}

def DMADescPrefetchOp : CoIR_Op<"dma.prefetch.desc", []> {
  let summary = "Materialize descriptor for execution";
  let description = [{
    Converts a compile-time descriptor to a runtime-ready descriptor.
    On some targets this prefetches the descriptor; on others it is a no-op.
  }];
  let arguments = (ins CoIR_DMADescType:$in);
  let results   = (outs CoIR_FinalDMADescType:$out);
  let hasCanonicalizer = 1;
  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def DMADescRuntimeOp : CoIR_Op<"dma.runtime.desc", []> {
  let summary = "Bind per-iteration offsets to a DMA descriptor";
  let description = [{
    Binds runtime offset values to a materialized descriptor. The variadic
    `offsets` list specifies per-dimension coordinates that vary per loop
    iteration. Target backends emit the appropriate offset-binding calls
    (e.g., dimension offset setters or tile coordinate binding).

    Example:
      %d1 = coir.dma.runtime.desc %d0 offsets(%k, %j)
              : !coir.desc.rt -> !coir.desc.rt
  }];
  let arguments = (ins CoIR_FinalDMADescType:$in,
                       Variadic<Index>:$offsets);
  let results   = (outs CoIR_FinalDMADescType:$out);
  let hasCustomAssemblyFormat = 1;
}

def DMAInvokeOp : CoIR_Op<"dma.invoke",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Execute the configured DMA transfer";
  let description = [{
    Triggers the DMA transfer described by a runtime-bound descriptor.
    Returns a completion token for synchronization.
  }];
  let arguments = (ins CoIR_FinalDMADescType:$desc);
  let results   = (outs CoIR_AsyncTokenTypeC:$done);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Tier 1: Kernel and parallelism ops
//===----------------------------------------------------------------------===//

def KernelOp : CoIR_Op<"kernel", [
    IsolatedFromAbove, Symbol, SingleBlock, NoTerminator
]> {
  let summary = "Top-level device kernel function";
  let description = [{
    Defines a device kernel with typed tensor arguments and results.
    Carries launch grid metadata as attributes.

    Example:
      coir.kernel @matmul(%a: !coir.tensor<128x64xf16>) -> !coir.tensor<128x256xf16> {
        ...
      }
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

def KernelReturnOp : CoIR_Op<"return", [Terminator, HasParent<"KernelOp">]> {
  let summary = "Return from a kernel";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def ParallelOp : CoIR_Op<"parallel", [SingleBlock, NoTerminator]> {
  let summary = "Parallel execution region at a given hierarchy level";
  let description = [{
    Models parallel execution at block/warpgroup/warp/thread level.
    Block arguments are the parallel iteration variables.
    Bounds specify the number of parallel instances per dimension.

    Example:
      coir.parallel (%m, %n) in [2, 16] level = #coir.level<block> {
        ...
      }
  }];

  let arguments = (ins
    CoIR_ParallelLevelAttr:$level,
    DenseI64ArrayAttr:$bounds
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

def ForeachOp : CoIR_Op<"foreach",
    [SingleBlock, DeclareOpInterfaceMethods<LoopLikeOpInterface,
      ["getLoopRegions", "getLoopInductionVars",
       "getLoopLowerBounds", "getLoopUpperBounds", "getLoopSteps",
       "getInitsMutable", "getRegionIterArgs",
       "getYieldedValuesMutable", "getLoopResults"]>]> {
  let summary = "Tiled sequential loop";
  let description = [{
    Sequential iteration over a tile dimension.
    The body region has one block argument (the induction variable)
    plus optional iter_args for loop-carried values.

    Example:
      %res = coir.foreach %k in 4 iter_args(%acc = %init) {
        ...
        coir.yield %next : !coir.mma_frag<16x16xf16>
      } : !coir.mma_frag<16x16xf16>
  }];

  let arguments = (ins
    Index:$upper_bound,
    Variadic<AnyType>:$iter_args
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

def YieldOp : CoIR_Op<"yield", [Terminator]> {
  let summary = "Yield values from a foreach or parallel region";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Tier 1: Tensor ops
//===----------------------------------------------------------------------===//

def TensorAllocOp : CoIR_Op<"tensor.alloc", [NoMemoryEffect]> {
  let summary = "Allocate a tensor";
  let description = [{
    Allocates a tensor with the specified type.
    Memory space is determined by the tensor type.
  }];
  let results = (outs CoIR_TensorTypeC:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TensorTileOp : CoIR_Op<"tensor.tile", [NoMemoryEffect]> {
  let summary = "Extract a tile from a tensor at given coordinates";
  let description = [{
    Extracts a sub-tensor (tile) from a larger tensor using
    the provided index coordinates. The result type determines
    the tile shape.
  }];
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    Variadic<Index>:$indices
  );
  let results = (outs CoIR_TensorTypeC:$result);
  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
}

def TensorStoreTileOp : CoIR_Op<"tensor.store_tile", []> {
  let summary = "Store a tile back into a tensor";
  let arguments = (ins
    CoIR_TensorTypeC:$tile,
    CoIR_TensorTypeC:$dest,
    Variadic<Index>:$indices
  );
  let assemblyFormat = "$tile `,` $dest `[` $indices `]` attr-dict `:` type($tile) `,` type($dest)";
}

//===----------------------------------------------------------------------===//
// Tier 1: Scalar compute (for simple matmul.co-style kernels)
//===----------------------------------------------------------------------===//

def TensorLoadElemOp : CoIR_Op<"tensor.load_elem", [NoMemoryEffect]> {
  let summary = "Load a scalar element from a tensor at indices";
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    Variadic<Index>:$indices
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
}

def TensorStoreElemOp : CoIR_Op<"tensor.store_elem", []> {
  let summary = "Store a scalar element to a tensor at indices";
  let arguments = (ins
    AnyType:$value,
    CoIR_TensorTypeC:$dest,
    Variadic<Index>:$indices
  );
  let assemblyFormat = "$value `,` $dest `[` $indices `]` attr-dict `:` type($value) `,` type($dest)";
}

def TensorReduceElemOp : CoIR_Op<"tensor.reduce_elem", []> {
  let summary = "Accumulate (reduce) a value into a tensor element";
  let description = [{
    Performs dest[indices] += value. Used for matmul-style accumulation.
  }];
  let arguments = (ins
    AnyType:$value,
    CoIR_TensorTypeC:$dest,
    Variadic<Index>:$indices
  );
  let assemblyFormat = "$value `,` $dest `[` $indices `]` attr-dict `:` type($value) `,` type($dest)";
}

//===----------------------------------------------------------------------===//
// Tier 1/2: MMA ops (defined here, fully used in Phase 2)
//===----------------------------------------------------------------------===//

def MMAFillOp : CoIR_Op<"mma.fill", [NoMemoryEffect]> {
  let summary = "Initialize MMA accumulator fragment with a value";
  let arguments = (ins AnyType:$value);
  let results = (outs CoIR_MMAFragTypeC:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def MMALoadOp : CoIR_Op<"mma.load", [NoMemoryEffect]> {
  let summary = "Load MMA operand from a tensor tile";
  let arguments = (ins CoIR_TensorTypeC:$source);
  let results = (outs CoIR_MMAFragTypeC:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def MMAExecOp : CoIR_Op<"mma.exec", [NoMemoryEffect]> {
  let summary = "Execute matrix multiply-accumulate";
  let arguments = (ins
    CoIR_MMAFragTypeC:$accumulator,
    CoIR_MMAFragTypeC:$lhs,
    CoIR_MMAFragTypeC:$rhs,
    CoIR_MMALayoutAttr:$layout
  );
  let results = (outs CoIR_MMAFragTypeC:$result);
  let hasCustomAssemblyFormat = 1;
}

def MMAStoreOp : CoIR_Op<"mma.store", []> {
  let summary = "Store MMA result to a tensor tile";
  let arguments = (ins
    CoIR_MMAFragTypeC:$fragment,
    CoIR_TensorTypeC:$dest
  );
  let assemblyFormat = "$fragment `,` $dest attr-dict `:` type($fragment) `,` type($dest)";
}

//===----------------------------------------------------------------------===//
// Tier 2: Copy and synchronization ops (defined here, used in Phase 3)
//===----------------------------------------------------------------------===//

def DataCopyOp : CoIR_Op<"data.copy", []> {
  let summary = "Generic data copy (disambiguated to dma/tma/thread in lowering)";
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    CoIR_TensorTypeC:$dest,
    OptionalAttr<UnitAttr>:$async
  );
  let results = (outs Optional<CoIR_AsyncTokenTypeC>:$token);
  let hasCustomAssemblyFormat = 1;
}

def BarrierOp : CoIR_Op<"barrier", []> {
  let summary = "Synchronization barrier at the given scope";
  let arguments = (ins CoIR_ParallelLevelAttr:$scope);
  let assemblyFormat = "$scope attr-dict";
}

def WaitOp : CoIR_Op<"wait", []> {
  let summary = "Wait for an async operation to complete";
  let arguments = (ins CoIR_AsyncTokenTypeC:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

//===----------------------------------------------------------------------===//
// Tier 3: Specialized copy ops (lowered from data.copy by classify-copies)
//===----------------------------------------------------------------------===//

def DmaCopyOp : CoIR_Op<"dma.copy", []> {
  let summary = "Asynchronous DMA copy between memory hierarchies";
  let description = [{
    Copy data asynchronously using the DMA engine (e.g. global -> shared).
    Always returns an async token for subsequent synchronization.
  }];
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    CoIR_TensorTypeC:$dest
  );
  let results = (outs CoIR_AsyncTokenTypeC:$token);
  let hasCustomAssemblyFormat = 1;
}

def TmaCopyOp : CoIR_Op<"tma.copy", []> {
  let summary = "Tensor Memory Accelerator (TMA) bulk copy";
  let description = [{
    TMA-based bulk copy for SM90+. Uses hardware tensor descriptors
    for efficient global <-> shared memory transfers.
  }];
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    CoIR_TensorTypeC:$dest
  );
  let results = (outs CoIR_AsyncTokenTypeC:$token);
  let hasCustomAssemblyFormat = 1;
}

def ElementCopyOp : CoIR_Op<"element.copy", []> {
  let summary = "Element-wise data copy";
  let description = [{
    Element-wise copy between tensors. Used when no hardware-assisted
    copy engine (DMA/TMA) is available or applicable for the given
    memory space pair. Lowered to target-specific patterns:

    - GPU: cooperative per-thread copy (element.parallel.copy)
    - CPU: sequential iteration (element.iteration.copy)
  }];
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    CoIR_TensorTypeC:$dest
  );
  let assemblyFormat = "$source `to` $dest attr-dict `:` type($source) `->` type($dest)";
}

//===----------------------------------------------------------------------===//
// Runtime safety assertions
//
// Target-agnostic: every backend emits the appropriate check API.
//   ENTRY  -> host-side check before kernel launch
//   HOIST  -> device-side check at hoisted site
//   USE    -> device-side check at original use site
//
// Backends emit the check as the appropriate host/device assertion macro.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// While loop with early exit (break/continue)
//
// coir.while models loops that may contain break/continue statements.
// It has two regions: a condition region and a body region, each receiving
// iter_args as block arguments.
//
// The condition region is terminated by coir.while.cond which tests the
// loop predicate and forwards values to the body.
// The body region uses coir.continue to yield next-iteration values and
// coir.break for early exit.
//===----------------------------------------------------------------------===//

def CoIRWhileOp : CoIR_Op<"while", [RecursiveMemoryEffects]> {
  let summary = "While loop with optional break/continue support";
  let description = [{
    A while loop with two regions: condition and body.
    Supports iter_args for SSA loop-carried values.
    The condition region is terminated by coir.while.cond.
    The body region uses coir.continue and coir.break for control flow.
  }];
  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$cond_region,
                        SizedRegion<1>:$body_region);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def CoIRWhileCondOp : CoIR_Op<"while.cond", [Terminator,
    HasParent<"CoIRWhileOp">]> {
  let summary = "Condition terminator for coir.while condition region";
  let description = [{
    Tests the loop predicate. If true, forwards values to the body region.
    If false, the while loop exits and the forwarded values become the
    while op's results.
  }];
  let arguments = (ins I1:$condition, Variadic<AnyType>:$args);
  let assemblyFormat = [{
    `(` $condition `)` ($args^ `:` type($args))? attr-dict
  }];
}

def CoIRBreakOp : CoIR_Op<"break", []> {
  let summary = "Break out of a coir.while loop";
  let description = [{
    Exits the enclosing coir.while loop. Operands are the final values
    of the loop-carried variables. May appear inside scf.if to model
    conditional break.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def CoIRContinueOp : CoIR_Op<"continue", [Terminator]> {
  let summary = "Continue to next iteration of a coir.while loop";
  let description = [{
    Jumps to the next iteration of the enclosing coir.while loop.
    Operands become the next iteration's iter_args.
    Must be the terminator of the coir.while body region.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Runtime safety assertions
//===----------------------------------------------------------------------===//

def AssertOp : CoIR_Op<"assert", []> {
  let summary = "Runtime safety assertion";
  let description = [{
    Emits a runtime assertion that `condition` is true. When the
    condition is false at runtime, the assertion handler aborts with
    the diagnostic `message`.

    The `site` attribute determines where the check is emitted:
      - entry: host wrapper, before kernel launch (cheapest)
      - hoist: device code, at a hoisted site above inner loops
      - use:   device code, at the original use site

    The `usage` attribute classifies the check for filtering and
    diagnostics (shape compatibility, element access bounds, loop
    bounds, hardware constraints, etc.).

    The HoistAssertions pass promotes assertions from USE -> HOIST
    or ENTRY based on SSA def-chain analysis, analogous to Choreo's
    AssertSite pass.
  }];
  let arguments = (ins
    I1:$condition,
    StrAttr:$message,
    CoIR_AssertSiteAttr:$site,
    CoIR_AssertUsageAttr:$usage
  );
  let assemblyFormat = [{
    $condition `,` $message $site $usage attr-dict
  }];
}

#endif // COIR_OPS_TD
