#ifndef __COIR_SESSION_HPP__
#define __COIR_SESSION_HPP__

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include <iostream>

namespace CoIR {

class IRSession {
public:
  static IRSession& Get() {
    static IRSession s;
    return s;
  }

  mlir::MLIRContext& Context() { return *ctx; }
  mlir::ModuleOp Module() { return *module_ref; }

  void ResetModule(llvm::StringRef module_filename = "module") {
    auto file_loc = mlir::FileLineColLoc::get(ctx.get(), module_filename, 0, 0);
    module_ref = mlir::ModuleOp::create(file_loc);
  }

private:
  IRSession() {
    registry.insert<mlir::func::FuncDialect>();
    ctx = std::make_unique<mlir::MLIRContext>(registry);
    ctx->loadDialect<mlir::func::FuncDialect>();
    ctx->loadDialect<mlir::memref::MemRefDialect>();
    ctx->loadDialect<mlir::scf::SCFDialect>();
    ctx->loadDialect<mlir::arith::ArithDialect>();
    ctx->loadDialect<coir::CoIRDialect>();

    ResetModule();
  }

  IRSession(const IRSession&) = delete;
  IRSession& operator=(const IRSession&) = delete;

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref;
};

} // end namespace CoIR

#endif // __COIR_SESSION_HPP__
