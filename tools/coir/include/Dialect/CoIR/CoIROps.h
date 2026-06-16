#ifndef COIR_OPS_H
#define COIR_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
#include "CoIRDialect.h"
#include "CoIRTypes.h"
#include "CoIRAttrs.h"

namespace coir {
class CoIRDialect;
} // namespace coir

#define GET_OP_CLASSES
#include "CoIR/CoIROps.h.inc"

#endif // COIR_OPS_H
