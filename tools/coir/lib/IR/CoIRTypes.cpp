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

static int32_t parseMemorySpaceKeyword(llvm::StringRef msStr) {
  if (msStr == "default")
    return -1;
  if (msStr == "global")
    return static_cast<int32_t>(coir::TensorMemorySpace::Global);
  if (msStr == "shared")
    return static_cast<int32_t>(coir::TensorMemorySpace::Shared);
  if (msStr == "local")
    return static_cast<int32_t>(coir::TensorMemorySpace::Local);
  if (msStr == "register")
    return static_cast<int32_t>(coir::TensorMemorySpace::Register);
  return -2; // invalid
}

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
  llvm::SmallVector<int64_t> strides;

  if (succeeded(parser.parseOptionalComma())) {
    llvm::StringRef kw;
    if (parser.parseKeyword(&kw))
      return {};

    if (kw == "strides") {
      // strides without memspace: parse ": [...]"
      if (parser.parseColon() || parser.parseLSquare())
        return {};
      int64_t s;
      auto res = parser.parseOptionalInteger(s);
      if (res.has_value() && succeeded(*res)) {
        strides.push_back(s);
        while (succeeded(parser.parseOptionalComma())) {
          if (parser.parseInteger(s))
            return {};
          strides.push_back(s);
        }
      }
      if (parser.parseRSquare())
        return {};
    } else {
      int32_t ms = parseMemorySpaceKeyword(kw);
      if (ms == -2) {
        parser.emitError(parser.getCurrentLocation(),
                         "unknown memory space: " + kw);
        return {};
      }
      memSpace = ms;

      // Check for optional strides after memspace
      if (succeeded(parser.parseOptionalComma())) {
        llvm::StringRef stKw;
        if (parser.parseKeyword(&stKw) || stKw != "strides")
          return {};
        if (parser.parseColon() || parser.parseLSquare())
          return {};
        int64_t s;
        auto res2 = parser.parseOptionalInteger(s);
        if (res2.has_value() && succeeded(*res2)) {
          strides.push_back(s);
          while (succeeded(parser.parseOptionalComma())) {
            if (parser.parseInteger(s))
              return {};
            strides.push_back(s);
          }
        }
        if (parser.parseRSquare())
          return {};
      }
    }
  }

  if (parser.parseGreater())
    return {};

  return coir::TensorType::get(parser.getContext(), elemType, shape, memSpace,
                               strides);
}

void coir::TensorType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  int32_t ms = getMemorySpace();
  printer << ", ";
  if (ms < 0) {
    printer << "default";
  } else {
    switch (static_cast<coir::TensorMemorySpace>(ms)) {
    case coir::TensorMemorySpace::Global:   printer << "global";   break;
    case coir::TensorMemorySpace::Shared:   printer << "shared";   break;
    case coir::TensorMemorySpace::Local:    printer << "local";    break;
    case coir::TensorMemorySpace::Register: printer << "register"; break;
    }
  }
  auto stridesArr = getStrides();
  if (!stridesArr.empty()) {
    printer << ", strides: [";
    llvm::interleaveComma(stridesArr, printer.getStream());
    printer << "]";
  }
  printer << ">";
}

bool coir::TensorType::isDenseContiguous() const {
  auto stridesArr = getStrides();
  if (stridesArr.empty()) return true;
  auto shapeArr = getShape();
  int64_t expected = 1;
  for (int i = (int)shapeArr.size() - 1; i >= 0; --i) {
    if (stridesArr[i] != expected) return false;
    expected *= shapeArr[i];
  }
  return true;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "CoIR/CoIRTypes.cpp.inc"
#pragma GCC diagnostic pop

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

void coir::CoIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "CoIR/CoIRTypes.cpp.inc"
  >();
}
