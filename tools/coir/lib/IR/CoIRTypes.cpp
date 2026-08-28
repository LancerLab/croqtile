// lib/CoIRTypes.cpp
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper: parse shape like "128x64x" or "?x128x" (consumes trailing 'x')
//===----------------------------------------------------------------------===//
static ParseResult parseShape(AsmParser& parser,
                              llvm::SmallVectorImpl<int64_t>& shape) {
  while (true) {
    if (succeeded(parser.parseOptionalQuestion())) {
      shape.push_back(mlir::ShapedType::kDynamic);
      if (failed(parser.parseXInDimensionList())) return failure();
      continue;
    }
    int64_t dim;
    auto res = parser.parseOptionalInteger(dim);
    if (!res.has_value() || failed(*res)) break;
    shape.push_back(dim);
    if (failed(parser.parseXInDimensionList())) return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// coir::TensorType custom parser/printer
//===----------------------------------------------------------------------===//

static int32_t parseMemorySpaceKeyword(llvm::StringRef msStr) {
  if (msStr == "default") return -1;
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

mlir::Type coir::TensorType::parse(mlir::AsmParser& parser) {
  if (parser.parseLess()) return {};

  llvm::SmallVector<int64_t> shape;
  if (failed(parseShape(parser, shape))) return {};

  mlir::Type elemType;
  if (parser.parseType(elemType)) return {};

  int32_t memSpace = -1;
  bool isUnsigned = false;
  llvm::SmallVector<int64_t> strides;

  // Optional comma-separated suffixes: memspace keyword, "unsigned", and
  // "strides: [...]" (in any order).
  while (succeeded(parser.parseOptionalComma())) {
    llvm::StringRef kw;
    if (parser.parseKeyword(&kw)) return {};

    if (kw == "unsigned") {
      isUnsigned = true;
    } else if (kw == "strides") {
      if (parser.parseColon() || parser.parseLSquare())
        return {};
      int64_t s;
      auto res = parser.parseOptionalInteger(s);
      if (res.has_value() && succeeded(*res)) {
        strides.push_back(s);
        while (succeeded(parser.parseOptionalComma())) {
          if (parser.parseInteger(s)) return {};
          strides.push_back(s);
        }
      }
      if (parser.parseRSquare()) return {};
    } else {
      int32_t ms = parseMemorySpaceKeyword(kw);
      if (ms == -2) {
        parser.emitError(parser.getCurrentLocation(),
                         "unknown tensor suffix: " + kw);
        return {};
      }
      memSpace = ms;
    }
  }

  if (parser.parseGreater()) return {};

  return coir::TensorType::get(parser.getContext(), elemType, shape, memSpace,
                               isUnsigned, strides);
}

void coir::TensorType::print(mlir::AsmPrinter& printer) const {
  printer << "<";
  for (auto dim : getShape()) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << "?x";
    else
      printer << dim << "x";
  }
  printer << getElementType();
  int32_t ms = getMemorySpace();
  printer << ", ";
  if (ms < 0) {
    printer << "default";
  } else {
    switch (static_cast<coir::TensorMemorySpace>(ms)) {
    case coir::TensorMemorySpace::Global: printer << "global"; break;
    case coir::TensorMemorySpace::Shared: printer << "shared"; break;
    case coir::TensorMemorySpace::Local: printer << "local"; break;
    case coir::TensorMemorySpace::Register: printer << "register"; break;
    }
  }
  if (getIsUnsigned())
    printer << ", unsigned";
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
    if (mlir::ShapedType::isDynamic(shapeArr[i])) return false;
    if (stridesArr[i] != expected) return false;
    expected *= shapeArr[i];
  }
  return true;
}

//===----------------------------------------------------------------------===//
// coir::MMAFragType custom parser/printer
//===----------------------------------------------------------------------===//

mlir::Type coir::MMAFragType::parse(mlir::AsmParser& parser) {
  if (parser.parseLess()) return {};

  llvm::SmallVector<int64_t> shape;
  if (failed(parseShape(parser, shape))) return {};

  mlir::Type elemType;
  if (parser.parseType(elemType)) return {};

  if (parser.parseGreater()) return {};

  return coir::MMAFragType::get(parser.getContext(), elemType, shape);
}

void coir::MMAFragType::print(mlir::AsmPrinter& printer) const {
  printer << "<";
  for (auto dim : getShape()) printer << dim << "x";
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
