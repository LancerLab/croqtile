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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "CoIR/CoIRTypes.h.inc"
#pragma GCC diagnostic pop

#endif // COIR_TYPES_H
