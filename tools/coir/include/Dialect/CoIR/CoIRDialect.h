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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "CoIR/CoIRDialect.h.inc"
#pragma GCC diagnostic pop

#endif // COIR_DIALECT_H
