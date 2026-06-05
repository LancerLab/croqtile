#ifndef COIR_TYPES_H
#define COIR_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"

namespace coir {
class CoIRDialect;
} // namespace coir

#define GET_TYPEDEF_CLASSES
#include "CoIR/CoIRTypes.h.inc"

#endif // COIR_TYPES_H
