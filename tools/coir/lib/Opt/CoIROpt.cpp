#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/Passes.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::async::AsyncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<coir::CoIRDialect>();

  coir::registerCoIRPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "coir-opt\n", registry));
}
