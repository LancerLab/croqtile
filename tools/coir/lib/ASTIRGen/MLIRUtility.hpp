#ifndef __COIR_MLIR_UTILITY_HPP__
#define __COIR_MLIR_UTILITY_HPP__

#include "loc.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <string>

namespace CoIR {

inline std::ostream& operator<<(std::ostream& os, mlir::ModuleOp mod) {
  std::string buffer;
  llvm::raw_string_ostream llvmOS(buffer);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*pretty=*/true);
  mod.print(llvmOS, flags);
  llvmOS.flush();
  os << buffer;
  return os;
}

inline const mlir::Location ToMLIRLoc(mlir::MLIRContext& ctx,
                                      const Choreo::location& cloc) {
  // NOTE: Only pass the begin positon. Improve it as a range later?
  auto attr = mlir::StringAttr::get(&ctx, cloc.begin.filename);
  return mlir::FileLineColLoc::get(&ctx, attr, cloc.begin.line,
                                   cloc.begin.column);
}
} // namespace CoIR

#endif // __COIR_MLIR_UTILITY_HPP__
