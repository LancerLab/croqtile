//===- CoIRKernelLowering.h - Shared CoIR->GPU kernel lowering ----*- C++ -*-===//
//
// Base class for lowering CoIR kernel ops to MLIR GPU + memref + arith + SCF.
// Shared by target-specific ConvertTo* passes (NVPTX, etc.).
//
//===----------------------------------------------------------------------===//

#ifndef COIR_CODEGEN_COIRKERNELLOWERING_H
#define COIR_CODEGEN_COIRKERNELLOWERING_H

#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

namespace coir {

/// Context passed through the recursive convertOp dispatch.
struct KernelConvertCtx {
  mlir::IRMapping &mapping;
  llvm::DenseMap<mlir::Value, mlir::Value> &returnAllocMap;
  /// SPM pool base allocations.  Keyed by reuse_spm name.
  /// First reuse alloc for a pool creates the base memref.alloc;
  /// subsequent allocs create subview + reinterpret_cast into it.
  llvm::DenseMap<llvm::StringRef, mlir::Value> spmPools;
};

struct LaunchDims {
  llvm::SmallVector<int64_t, 3> gridDims = {1, 1, 1};
  llvm::SmallVector<int64_t, 3> blockDims = {1, 1, 1};
};

/// Base class providing shared kernel lowering logic.
/// Target-specific passes inherit this and override virtual hooks.
class CoIRKernelLoweringBase {
public:
  virtual ~CoIRKernelLoweringBase() = default;

protected:
  // -- Shared helpers --
  static mlir::gpu::Dimension dimFromIndex(unsigned i);
  static LaunchDims collectLaunchDims(KernelOp kernel);

  // -- Target-specific hooks (must override) --
  virtual mlir::MemRefType convertTensorType(TensorType tty) = 0;

  // -- Kernel conversion --
  mlir::LogicalResult convertKernel(mlir::ModuleOp module,
                                    KernelOp kernel);

  /// Hook called after gpu.module is created but before the kernel body
  /// is converted. Use to pre-scan for shared-memory allocs, set module
  /// attributes, etc.
  virtual void preScanKernel(KernelOp /*kernel*/, mlir::OpBuilder &/*builder*/,
                             mlir::gpu::GPUModuleOp /*gpuModule*/) {}

  // -- Op conversion --
  virtual void convertOp(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Operation &op, KernelConvertCtx &ctx);

  /// Hook for target-specific ops. Return true if handled.
  virtual bool convertTargetOp(mlir::OpBuilder &/*builder*/,
                               mlir::Location /*loc*/,
                               mlir::Operation &/*op*/,
                               KernelConvertCtx &/*ctx*/) {
    return false;
  }

  // -- Shared op converters --
  virtual void convertAlloc(mlir::OpBuilder &builder, mlir::Location loc,
                            TensorAllocOp alloc, KernelConvertCtx &ctx);
  virtual void convertLoadElem(mlir::OpBuilder &builder, mlir::Location loc,
                               TensorLoadElemOp op, mlir::IRMapping &mapping);
  virtual void convertStoreElem(mlir::OpBuilder &builder, mlir::Location loc,
                                TensorStoreElemOp op, mlir::IRMapping &mapping);
  virtual void convertReduceElem(mlir::OpBuilder &builder, mlir::Location loc,
                                 TensorReduceElemOp op,
                                 mlir::IRMapping &mapping);

  // -- Rank-mismatch helpers --
  mlir::Value flattenIfNeeded(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value memref, unsigned numIndices);
  mlir::Value getBaseMemRef(mlir::Value memref);
  mlir::Value extractOffset(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value memref);
  void convertElementCopy(mlir::OpBuilder &builder, mlir::Location loc,
                          ElementCopyOp op, mlir::IRMapping &mapping);
  virtual void convertDmaCopy(mlir::OpBuilder &builder, mlir::Location loc,
                              DmaCopyOp op, KernelConvertCtx &ctx);
  virtual void emitFlatCopyLoop(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value src, mlir::Value dst);
  void convertForeach(mlir::OpBuilder &builder, mlir::Location loc,
                      ForeachOp op, KernelConvertCtx &ctx);
  void convertParallel(mlir::OpBuilder &builder, mlir::Location loc,
                       ParallelOp op, KernelConvertCtx &ctx);
};

} // namespace coir

#endif // COIR_CODEGEN_COIRKERNELLOWERING_H
