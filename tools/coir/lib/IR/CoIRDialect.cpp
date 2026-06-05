// lib/CoIRDialect.cpp
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "llvm/ADT/TypeSwitch.h"

#include "CoIR/CoIRDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "CoIR/CoIRAttrs.cpp.inc"
#include "CoIR/CoIREnums.cpp.inc"

#define GET_OP_CLASSES
#include "CoIR/CoIROps.cpp.inc"

using namespace mlir;
using namespace coir;

void CoIRDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "CoIR/CoIROps.cpp.inc"
  >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "CoIR/CoIRAttrs.cpp.inc"
  >();

  getContext()->loadDialect<mlir::async::AsyncDialect>();
}
