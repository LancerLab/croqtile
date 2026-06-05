// lib/CoIRTypes.cpp
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper: parse shape like "128x64x" (consumes trailing 'x')
//===----------------------------------------------------------------------===//
static ParseResult parseShape(AsmParser &parser,
                              llvm::SmallVectorImpl<int64_t> &shape) {
  int64_t dim;
  auto res = parser.parseOptionalInteger(dim);
  while (res.has_value() && succeeded(*res)) {
    shape.push_back(dim);
    if (failed(parser.parseXInDimensionList()))
      return failure();
    res = parser.parseOptionalInteger(dim);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// coir::TensorType custom parser/printer
//===----------------------------------------------------------------------===//

mlir::Type coir::TensorType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return {};

  llvm::SmallVector<int64_t> shape;
  if (failed(parseShape(parser, shape)))
    return {};

  mlir::Type elemType;
  if (parser.parseType(elemType))
    return {};

  int32_t memSpace = -1;
  if (succeeded(parser.parseOptionalComma())) {
    llvm::StringRef msStr;
    if (parser.parseKeyword(&msStr))
      return {};
    if (msStr == "global")
      memSpace = static_cast<int32_t>(coir::TensorMemorySpace::Global);
    else if (msStr == "shared")
      memSpace = static_cast<int32_t>(coir::TensorMemorySpace::Shared);
    else if (msStr == "local")
      memSpace = static_cast<int32_t>(coir::TensorMemorySpace::Local);
    else if (msStr == "register")
      memSpace = static_cast<int32_t>(coir::TensorMemorySpace::Register);
    else {
      parser.emitError(parser.getCurrentLocation(),
                       "unknown memory space: " + msStr);
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return coir::TensorType::get(parser.getContext(), elemType, shape, memSpace);
}

void coir::TensorType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  int32_t ms = getMemorySpace();
  if (ms >= 0) {
    printer << ", ";
    switch (static_cast<coir::TensorMemorySpace>(ms)) {
    case coir::TensorMemorySpace::Global:   printer << "global";   break;
    case coir::TensorMemorySpace::Shared:   printer << "shared";   break;
    case coir::TensorMemorySpace::Local:    printer << "local";    break;
    case coir::TensorMemorySpace::Register: printer << "register"; break;
    }
  }
  printer << ">";
}

//===----------------------------------------------------------------------===//
// coir::MMAFragType custom parser/printer
//===----------------------------------------------------------------------===//

mlir::Type coir::MMAFragType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return {};

  llvm::SmallVector<int64_t> shape;
  if (failed(parseShape(parser, shape)))
    return {};

  mlir::Type elemType;
  if (parser.parseType(elemType))
    return {};

  if (parser.parseGreater())
    return {};

  return coir::MMAFragType::get(parser.getContext(), elemType, shape);
}

void coir::MMAFragType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Generated type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "CoIR/CoIRTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

void coir::CoIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "CoIR/CoIRTypes.cpp.inc"
  >();
}
