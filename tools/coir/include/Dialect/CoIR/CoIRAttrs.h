#ifndef COIR_ATTRS_H
#define COIR_ATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace coir {
class CoIRDialect;
} // namespace coir

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "CoIR/CoIREnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "CoIR/CoIRAttrs.h.inc"
#pragma GCC diagnostic pop

#endif // COIR_ATTRS_H
