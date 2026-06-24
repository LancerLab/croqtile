// CoIRTypes.td -- Type definitions for the CoIR dialect
#ifndef COIR_TYPES_TD
#define COIR_TYPES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "CoIR/CoIRDialect.td"

//===----------------------------------------------------------------------===//
// Existing DMA descriptor types
//===----------------------------------------------------------------------===//

def CoIR_DMADescType : TypeDef<CoIR_Dialect, "DMADesc"> {
  let mnemonic = "desc";
  let summary = "Compile-time DMA descriptor";
  let description = [{
    An opaque handle representing a compile-time descriptor for DMA.
  }];
}

def CoIR_FinalDMADescType : TypeDef<CoIR_Dialect, "FinalDMADesc"> {
  let mnemonic = "desc.rt";
  let summary = "Runtime-bound DMA descriptor";
  let description = [{
    An opaque handle representing a runtime-bound descriptor for DMA.
  }];
}

//===----------------------------------------------------------------------===//
// Tier 1: High-level tensor type
//===----------------------------------------------------------------------===//

def CoIR_TensorType : TypeDef<CoIR_Dialect, "Tensor"> {
  let mnemonic = "tensor";
  let summary = "N-D typed tensor with element type, shape, strides, and memory space";
  let description = [{
    Represents a multi-dimensional tensor in the CoIR dialect.
    Carries an element type, a static shape, optional strides, and an optional
    memory space (global, shared, local, register). Defaults to global if
    unspecified. If strides are empty, the tensor is assumed dense contiguous
    (row-major).

    Syntax:
      !coir.tensor<128x64xf16>                             // dense, global
      !coir.tensor<16x16xf32, shared>                      // dense, shared
      !coir.tensor<128x64xf16, global, strides: [64, 1]>  // explicit strides
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape,
    DefaultValuedParameter<"int32_t", "-1">:$memorySpace,
    OptionalArrayRefParameter<"int64_t">:$strides
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool isDenseContiguous() const;
  }];
}

//===----------------------------------------------------------------------===//
// Tier 1: MMA fragment type (for Phase 2, defined early for forward compat)
//===----------------------------------------------------------------------===//

def CoIR_MMAFragType : TypeDef<CoIR_Dialect, "MMAFrag"> {
  let mnemonic = "mma_frag";
  let summary = "MMA fragment handle (accumulator or operand)";
  let description = [{
    Represents an MMA fragment: a register-resident tile used by
    matrix multiply-accumulate operations. Carries element type and shape.

    Syntax:
      !coir.mma_frag<16x16xf16>
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Async completion token
//===----------------------------------------------------------------------===//

def CoIR_AsyncTokenType : TypeDef<CoIR_Dialect, "AsyncToken"> {
  let mnemonic = "async";
  let summary = "Async completion handle for copy/DMA operations";
  let description = [{
    An opaque handle representing the completion of an asynchronous operation.
    When __CHOREO_DMA_DIAGNOSIS__ is enabled, unwaited async handles trigger
    runtime errors via choreo::future's destructor check.
  }];
}

//===----------------------------------------------------------------------===//
// Type constraints for use in op definitions
//===----------------------------------------------------------------------===//

def CoIR_TensorTypeC : Type<
  CPred<"$_self.isa<coir::TensorType>()">,
  "CoIR tensor type">;

def CoIR_MMAFragTypeC : Type<
  CPred<"$_self.isa<coir::MMAFragType>()">,
  "CoIR MMA fragment type">;

def CoIR_AsyncTokenTypeC : Type<
  CPred<"$_self.isa<coir::AsyncTokenType>()">,
  "CoIR async handle type">;

def CoIR_TensorOrMemRef : AnyTypeOf<[CoIR_TensorTypeC, AnyMemRef]>;

#endif // COIR_TYPES_TD
