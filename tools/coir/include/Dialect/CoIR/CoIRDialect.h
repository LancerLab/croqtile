#ifndef COIR_DIALECT_H
#define COIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace coir {
class CoIRDialect;
} // namespace coir

#include "CoIR/CoIRDialect.h.inc"

#endif // COIR_DIALECT_H
