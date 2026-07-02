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

    When `tma` is set, the descriptor uses the target's hardware tensor-move
    accelerator path (requires target support). Without `tma`, the descriptor
    uses the standard asynchronous DMA or cooperative-copy mechanism.

    Example:
      %d = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<copy>, tma}
             : !coir.tensor<128x64xf32, global>,
               !coir.tensor<128x64xf32, shared> -> !coir.desc
  }];
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    CoIR_TensorTypeC:$dest,
    CoIR_DMAKindAttr:$kind,
    UnitAttr:$tma
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
    Returns an async handle for synchronization.

    When the PlanDMACopy pass determines that a tiled (cooperative) copy is
    profitable, it stamps optional attrs that EmitCUDA uses to emit
    `choreo::tiled_copy<...>` instead of `choreo::naive_copy`:
      - thr_layout: [ThrRows, ThrCols] thread tile
      - val_layout: [ValRows, ValCols] value tile per thread
      - copy_atom: CuTe copy atom type name string
      - need_pred: whether predicated copy is required
      - prediction: [pred_dim0, pred_dim1] shape for predication
      - swizzle: whether to apply shared-memory swizzle
  }];
  let arguments = (ins
    CoIR_FinalDMADescType:$desc,
    Variadic<Index>:$dyn_dims,
    OptionalAttr<DenseI64ArrayAttr>:$thr_layout,
    OptionalAttr<DenseI64ArrayAttr>:$val_layout,
    OptionalAttr<StrAttr>:$copy_atom,
    OptionalAttr<BoolAttr>:$need_pred,
    OptionalAttr<DenseI64ArrayAttr>:$prediction,
    OptionalAttr<BoolAttr>:$swizzle,
    OptionalAttr<DenseI64ArrayAttr>:$copy_shape
  );
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
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<CoIR_LaunchBoundsAttr>:$launchBounds,
    OptionalAttr<CoIR_MaxNregAttr>:$maxNreg
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

def KernelReturnOp : CoIR_Op<"return", []> {
  let summary = "Return from a kernel";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
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
    DenseI64ArrayAttr:$bounds,
    OptionalAttr<StrAttr>:$stream,
    OptionalAttr<BoolAttr>:$is_async
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

    For tensors with dynamic dimensions (`?`), one `index` operand must be
    provided for each `?` dimension, in left-to-right order, following
    MLIR's tensor dialect convention.

    An optional `init` attribute specifies a scalar fill value
    (e.g. `init = -5 : i32`) that the runtime uses to memset
    the buffer immediately after allocation.

    Example:
      %0 = coir.tensor.alloc : !coir.tensor<128x64xf16, shared>
      %1 = coir.tensor.alloc(%K, %N) : !coir.tensor<?x?xi8, shared>
      %2 = coir.tensor.alloc init -5 : i32 : !coir.tensor<32x4xi32, local>
  }];
  let arguments = (ins Variadic<Index>:$dynamicDims,
                       OptionalAttr<TypedAttrInterface>:$init,
                       OptionalAttr<StrAttr>:$reuse_spm,
                       OptionalAttr<I64Attr>:$reuse_offset);
  let results = (outs CoIR_TensorTypeC:$result);

  let hasCustomAssemblyFormat = 1;
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
// Atomic read-modify-write
//===----------------------------------------------------------------------===//

def AtomicOp : CoIR_Op<"atomic", []> {
  let summary = "Atomic read-modify-write on a tensor element";
  let description = [{
    Performs an atomic read-modify-write at dest[indices].
    The `kind` attribute selects the operation (add, sub, exch, min, max,
    and, or, xor, cas). For CAS, `compare` provides the expected value.

    Example:
      coir.atomic <add> %val, %tensor[%i, %j]
        : f32, !coir.tensor<128x128xf32, global>
  }];
  let arguments = (ins
    CoIR_AtomicKindAttr:$kind,
    AnyType:$value,
    CoIR_TensorTypeC:$dest,
    Variadic<Index>:$indices,
    OptionalAttr<AnyAttr>:$compare
  );
  let assemblyFormat = [{
    $kind $value `,` $dest `[` $indices `]` attr-dict
    `:` type($value) `,` type($dest)
  }];
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
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    OptionalAttr<I64Attr>:$k_dim
  );
  let results = (outs CoIR_MMAFragTypeC:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def MMAExecOp : CoIR_Op<"mma.exec", []> {
  let summary = "Execute matrix multiply-accumulate";
  let arguments = (ins
    CoIR_MMAFragTypeC:$accumulator,
    CoIR_MMAFragTypeC:$lhs,
    CoIR_MMAFragTypeC:$rhs,
    CoIR_MMALayoutAttr:$layout,
    OptionalAttr<I64Attr>:$k_dim,
    OptionalAttr<StrAttr>:$mma_atom_name,
    OptionalAttr<I64Attr>:$reg_num_a,
    OptionalAttr<I64Attr>:$reg_num_b,
    OptionalAttr<I64Attr>:$reg_num_d
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

def BarrierOp : CoIR_Op<"barrier", []> {
  let summary = "Synchronization barrier at the given scope";
  let description = [{
    Emits a scope-appropriate synchronization barrier:
    - BLOCK: block-level thread sync
    - GROUP: sub-thread group sync
    - DEVICE: host device synchronize
  }];
  let arguments = (ins CoIR_ParallelLevelAttr:$scope);
  let assemblyFormat = "$scope attr-dict";
}

def WaitOp : CoIR_Op<"wait", []> {
  let summary = "Wait for an async operation to complete";
  let arguments = (ins CoIR_AsyncTokenTypeC:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

//===----------------------------------------------------------------------===//
// Event synchronization ops (bool-event model)
//===----------------------------------------------------------------------===//

def EventTriggerOp : CoIR_Op<"event.trigger", []> {
  let summary = "Signal an event (set to true)";
  let description = [{
    Signals a software event by setting it to true. On device targets,
    events are volatile bools in shared/local memory. The `name` attr
    identifies the event variable for codegen.
  }];
  let arguments = (ins
    StrAttr:$event_name,
    OptionalAttr<StrAttr>:$subscript
  );
  let assemblyFormat = "$event_name (`[` $subscript^ `]`)? attr-dict";
}

def EventWaitOp : CoIR_Op<"event.wait", []> {
  let summary = "Wait on an event (spin until true, then auto-reset)";
  let description = [{
    Waits for a software event by spin-waiting until the event is true,
    then resets it to false (auto-reset semantics). Used for inter-thread
    signaling within a kernel.
  }];
  let arguments = (ins
    StrAttr:$event_name,
    OptionalAttr<StrAttr>:$subscript
  );
  let assemblyFormat = "$event_name (`[` $subscript^ `]`)? attr-dict";
}

def AsyncUndefOp : CoIR_Op<"async.undef", [NoMemoryEffect]> {
  let summary = "Placeholder async token for uninitialized futures";
  let description = [{
    Produces an undefined async token value.  Used as the initial iter_arg
    for loop-carried futures that are assigned inside the loop body before
    first use (e.g. the second buffer in a double-buffering swap pattern).
  }];
  let results = (outs CoIR_AsyncTokenTypeC:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def FutureRotateOp : CoIR_Op<"async.rotate", []> {
  let summary = "Left-rotate async handles (swap when arity=2)";
  let description = [{
    Exchanges the underlying async handles among the operands via a
    left rotation.  For two operands this is equivalent to swap.
    The operation rebinds the runtime future objects so that subsequent
    waits target different physical DMA transfers.
  }];
  let arguments = (ins Variadic<CoIR_AsyncTokenTypeC>:$futures);
  let results   = (outs Variadic<CoIR_AsyncTokenTypeC>:$results);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Tier 3: Specialized copy ops (from AST-to-CoIR generation)
//===----------------------------------------------------------------------===//

def DmaCopyOp : CoIR_Op<"dma.copy", []> {
  let summary = "Asynchronous DMA copy between memory hierarchies";
  let description = [{
    Copy data asynchronously using the DMA engine (e.g. global -> shared).
    Always returns an async handle for subsequent synchronization.
  }];
  let arguments = (ins
    CoIR_TensorTypeC:$source,
    CoIR_TensorTypeC:$dest,
    DefaultValuedAttr<CoIR_DMAKindAttr, "coir::DMAKind::Copy">:$kind,
    OptionalAttr<DenseI64ArrayAttr>:$pad_low,
    OptionalAttr<DenseI64ArrayAttr>:$pad_high,
    OptionalAttr<AnyAttrOf<[I64Attr, F64Attr]>>:$pad_value,
    OptionalAttr<DenseI64ArrayAttr>:$transpose_perm
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
    CoIR_AssertUsageAttr:$usage,
    OptionalAttr<I64Attr>:$estimated_cost,
    OptionalAttr<CoIR_AssertCostAttr>:$cost_class,
    OptionalAttr<BoolAttr>:$enabled
  );
  let assemblyFormat = [{
    $condition `,` $message $site $usage attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Library / device function call
//===----------------------------------------------------------------------===//

def CallOp : CoIR_Op<"call", []> {
  let summary = "Call a library or device function";
  let description = [{
    Represents a call to a library function (__lib_*) or a
    user-defined device function.

    `callee` is the fully-qualified function name (e.g. "__lib_gemm",
    "my_kernel", or a target-specific library call).

    `operands_` are the call arguments (tensor pointers, scalars).
    `template_args` holds compile-time template parameters as strings.
    `is_lib_call` marks __lib_* builtins for target-specific lowering.
    `is_bif` marks compiler built-in functions (assert, print, arith, lib).
    `is_expr` marks calls whose return value is used.
  }];
  let arguments = (ins
    StrAttr:$callee,
    Variadic<AnyType>:$operands_,
    OptionalAttr<StrArrayAttr>:$template_args,
    OptionalAttr<BoolAttr>:$is_lib_call,
    OptionalAttr<BoolAttr>:$is_bif,
    OptionalAttr<BoolAttr>:$is_expr
  );
  let results = (outs Optional<AnyType>:$result);
}

//===----------------------------------------------------------------------===//
// InThreads predicated block
//===----------------------------------------------------------------------===//

def InThreadsOp : CoIR_Op<"inthreads", [SingleBlock, NoTerminator]> {
  let summary = "Predicated thread partition block";
  let description = [{
    Represents an `inthreads (pred) { ... }` block. The body executes
    only on threads where `predicate` is true.

    If `async` is true, no implicit __syncthreads() is emitted at end.
    If `outer` is true, a __syncthreads() is emitted at block close
    (when not async).
  }];
  let arguments = (ins
    I1:$predicate,
    OptionalAttr<BoolAttr>:$async,
    OptionalAttr<BoolAttr>:$outer
  );
  let regions = (region SizedRegion<1>:$body);
}

#endif // COIR_OPS_TD
