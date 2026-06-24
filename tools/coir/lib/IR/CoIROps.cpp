// lib/CoIROps.cpp -- Custom method implementations for CoIR ops
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Dialect/Async/IR/Async.h"

using namespace mlir;
using namespace coir;

//===----------------------------------------------------------------------===//
// DMAConstDescOp
//===----------------------------------------------------------------------===//

// Format: %d = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<copy>}
//           : !coir.tensor<...>, !coir.tensor<...> -> !coir.desc
ParseResult DMAConstDescOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcType, dstType, outType;
  if (parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(dst) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(srcType) ||
      parser.parseComma() || parser.parseType(dstType) ||
      parser.parseArrow() || parser.parseType(outType))
    return failure();
  if (parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands))
    return failure();
  result.addTypes(outType);
  return success();
}

void DMAConstDescOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", " << getDest();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << ", " << getDest().getType()
          << " -> " << getOut().getType();
}

//===----------------------------------------------------------------------===//
// DMADescRuntimeOp
//===----------------------------------------------------------------------===//

// Format: %d1 = coir.dma.runtime.desc %d0 offsets(%k, %j)
//           : !coir.desc.rt -> !coir.desc.rt
ParseResult DMADescRuntimeOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand inOperand;
  Type inType, outType;
  if (parser.parseOperand(inOperand))
    return failure();

  llvm::SmallVector<OpAsmParser::UnresolvedOperand> offsetOperands;
  if (succeeded(parser.parseOptionalKeyword("offsets"))) {
    if (parser.parseLParen() ||
        parser.parseOperandList(offsetOperands) ||
        parser.parseRParen())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(inType) ||
      parser.parseArrow() || parser.parseType(outType))
    return failure();

  if (parser.resolveOperand(inOperand, inType, result.operands))
    return failure();

  auto indexType = IndexType::get(parser.getContext());
  for (auto &off : offsetOperands) {
    if (parser.resolveOperand(off, indexType, result.operands))
      return failure();
  }

  result.addTypes(outType);
  return success();
}

void DMADescRuntimeOp::print(OpAsmPrinter &printer) {
  printer << " " << getIn();
  auto offsets = getOffsets();
  if (!offsets.empty()) {
    printer << " offsets(";
    llvm::interleaveComma(offsets, printer);
    printer << ")";
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getIn().getType() << " -> " << getOut().getType();
}

//===----------------------------------------------------------------------===//
// DMAInvokeOp
//===----------------------------------------------------------------------===//

void DMAInvokeOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get());
}

void DMADescPrefetchOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet & /*results*/,
    mlir::MLIRContext * /*context*/) {}

ParseResult DMAInvokeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand desc;
  Type descType;
  if (parser.parseOperand(desc) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(descType))
    return failure();
  if (parser.resolveOperand(desc, descType, result.operands))
    return failure();
  result.addTypes(coir::AsyncTokenType::get(parser.getContext()));
  return success();
}

void DMAInvokeOp::print(OpAsmPrinter &printer) {
  printer << " " << getDesc();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getDesc().getType();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  llvm::SmallVector<OpAsmParser::Argument> args;
  llvm::SmallVector<Type> argTypes;
  llvm::SmallVector<Type> resultTypes;

  if (parser.parseLParen())
    return failure();

  while (true) {
    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (!args.empty() && parser.parseComma())
      return failure();
    OpAsmParser::Argument arg;
    Type argType;
    if (parser.parseArgument(arg) || parser.parseColonType(argType))
      return failure();
    arg.type = argType;
    args.push_back(arg);
    argTypes.push_back(argType);
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseTypeList(resultTypes))
      return failure();
  }

  auto fnType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(fnType));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  return success();
}

void KernelOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());

  auto fnType = getFunctionType();
  printer << "(";
  auto &body = getBody();
  if (!body.empty()) {
    auto args = body.getArguments();
    for (unsigned i = 0; i < args.size(); ++i) {
      if (i > 0)
        printer << ", ";
      printer.printRegionArgument(args[i]);
    }
  }
  printer << ")";

  auto resultTypes = fnType.getResults();
  if (!resultTypes.empty()) {
    printer << " -> ";
    if (resultTypes.size() > 1)
      printer << "(";
    llvm::interleaveComma(resultTypes, printer);
    if (resultTypes.size() > 1)
      printer << ")";
  }

  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getSymNameAttrName(), getFunctionTypeAttrName()});

  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::Argument> ivs;

  if (parser.parseLParen())
    return failure();
  while (true) {
    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (!ivs.empty() && parser.parseComma())
      return failure();
    OpAsmParser::Argument arg;
    if (parser.parseArgument(arg))
      return failure();
    arg.type = IndexType::get(parser.getContext());
    ivs.push_back(arg);
  }

  if (parser.parseKeyword("in"))
    return failure();

  llvm::SmallVector<int64_t> bounds;
  if (parser.parseLSquare())
    return failure();
  while (true) {
    if (succeeded(parser.parseOptionalRSquare()))
      break;
    if (!bounds.empty() && parser.parseComma())
      return failure();
    int64_t b = 0;
    if (parser.parseInteger(b))
      return failure();
    bounds.push_back(b);
  }

  result.addAttribute("bounds",
                       DenseI64ArrayAttr::get(parser.getContext(), bounds));

  if (parser.parseKeyword("level") || parser.parseEqual())
    return failure();

  coir::ParallelLevelAttr levelAttr;
  if (parser.parseCustomAttributeWithFallback(levelAttr))
    return failure();
  result.addAttribute("level", levelAttr);

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, ivs))
    return failure();

  return success();
}

void ParallelOp::print(OpAsmPrinter &printer) {
  printer << " (";
  auto &body = getBody();
  if (!body.empty()) {
    auto args = body.getArguments();
    llvm::interleaveComma(args, printer, [&](BlockArgument arg) {
      printer.printRegionArgument(arg, {}, /*omitType=*/true);
    });
  }
  printer << ") in [";
  auto bounds = getBounds();
  llvm::interleaveComma(bounds, printer,
                        [&](int64_t b) { printer << b; });
  printer << "] level = ";
  printer.printAttribute(getLevelAttr());

  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                           {"bounds", "level"});

  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ForeachOp -- LoopLikeOpInterface
//===----------------------------------------------------------------------===//

SmallVector<Region *> ForeachOp::getLoopRegions() { return {&getBody()}; }

std::optional<SmallVector<Value>> ForeachOp::getLoopInductionVars() {
  return SmallVector<Value>{getBody().getArgument(0)};
}

std::optional<SmallVector<OpFoldResult>> ForeachOp::getLoopLowerBounds() {
  OpBuilder b(getContext());
  return SmallVector<OpFoldResult>{b.getIndexAttr(0)};
}

std::optional<SmallVector<OpFoldResult>> ForeachOp::getLoopUpperBounds() {
  return SmallVector<OpFoldResult>{getUpperBound()};
}

std::optional<SmallVector<OpFoldResult>> ForeachOp::getLoopSteps() {
  OpBuilder b(getContext());
  return SmallVector<OpFoldResult>{b.getIndexAttr(1)};
}

MutableArrayRef<OpOperand> ForeachOp::getInitsMutable() {
  return getIterArgsMutable();
}

Block::BlockArgListType ForeachOp::getRegionIterArgs() {
  auto args = getBody().getArguments();
  return args.drop_front(1);
}

std::optional<MutableArrayRef<OpOperand>> ForeachOp::getYieldedValuesMutable() {
  auto &block = getBody().front();
  auto *term = block.getTerminator();
  if (!term || term->getNumOperands() == 0)
    return std::nullopt;
  return term->getOpOperands();
}

std::optional<ResultRange> ForeachOp::getLoopResults() {
  return getResults();
}

//===----------------------------------------------------------------------===//
// ForeachOp -- parsing / printing
//===----------------------------------------------------------------------===//

ParseResult ForeachOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::Argument iv;
  iv.type = IndexType::get(parser.getContext());
  if (parser.parseArgument(iv, /*allowType=*/false))
    return failure();

  if (parser.parseKeyword("in"))
    return failure();

  OpAsmParser::UnresolvedOperand ubOperand;
  if (parser.parseOperand(ubOperand))
    return failure();
  if (parser.resolveOperand(ubOperand, IndexType::get(parser.getContext()),
                            result.operands))
    return failure();

  llvm::SmallVector<OpAsmParser::Argument> iterArgs;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> iterInits;

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseLParen())
      return failure();
    while (true) {
      if (succeeded(parser.parseOptionalRParen()))
        break;
      if (!iterArgs.empty() && parser.parseComma())
        return failure();
      OpAsmParser::Argument iterArg;
      OpAsmParser::UnresolvedOperand initOperand;
      if (parser.parseArgument(iterArg, /*allowType=*/false) ||
          parser.parseEqual() || parser.parseOperand(initOperand))
        return failure();
      iterArgs.push_back(iterArg);
      iterInits.push_back(initOperand);
    }
  }

  llvm::SmallVector<Type> resultTypes;
  if (parser.parseOptionalColonTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);

  for (unsigned i = 0; i < iterInits.size(); ++i) {
    if (i < resultTypes.size())
      iterArgs[i].type = resultTypes[i];
    if (parser.resolveOperand(iterInits[i], resultTypes[i], result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  llvm::SmallVector<OpAsmParser::Argument> blockArgs;
  blockArgs.push_back(iv);
  for (auto &ia : iterArgs)
    blockArgs.push_back(ia);

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs))
    return failure();

  return success();
}

void ForeachOp::print(OpAsmPrinter &printer) {
  auto &body = getBody();
  auto args = body.getArguments();

  printer << " ";
  if (!args.empty())
    printer.printRegionArgument(args[0], {}, /*omitType=*/true);
  printer << " in " << getUpperBound();

  auto iterArgs = getIterArgs();
  if (!iterArgs.empty()) {
    printer << " iter_args(";
    for (unsigned i = 0; i < iterArgs.size(); ++i) {
      if (i > 0)
        printer << ", ";
      printer.printRegionArgument(args[i + 1], {}, /*omitType=*/true);
      printer << " = " << iterArgs[i];
    }
    printer << ")";
  }

  if (!getResults().empty()) {
    printer << " : ";
    llvm::interleaveComma(getResultTypes(), printer);
  }

  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs());

  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// MMAExecOp
//===----------------------------------------------------------------------===//

// Format: %res = coir.mma.exec %acc, %lhs, %rhs
//           {layout = #coir.mma_layout<row_col>}
//           : (!coir.mma_frag<MxNxT>, ...) -> !coir.mma_frag<MxNxT>
ParseResult MMAExecOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand acc, lhs, rhs;
  if (parser.parseOperand(acc) || parser.parseComma() ||
      parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  FunctionType fnType;
  if (parser.parseType(fnType))
    return failure();

  if (fnType.getNumInputs() != 3)
    return parser.emitError(parser.getNameLoc(),
                            "expected 3 operand types in mma.exec signature");

  if (parser.resolveOperand(acc, fnType.getInput(0), result.operands) ||
      parser.resolveOperand(lhs, fnType.getInput(1), result.operands) ||
      parser.resolveOperand(rhs, fnType.getInput(2), result.operands))
    return failure();

  result.addTypes(fnType.getResults());
  return success();
}

void MMAExecOp::print(OpAsmPrinter &printer) {
  printer << " " << getAccumulator() << ", " << getLhs() << ", " << getRhs();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : (" << getAccumulator().getType() << ", "
          << getLhs().getType() << ", " << getRhs().getType() << ") -> "
          << getResult().getType();
}

//===----------------------------------------------------------------------===//
// DataCopyOp
//===----------------------------------------------------------------------===//

ParseResult DataCopyOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcType, dstType;

  if (parser.parseOperand(src) || parser.parseKeyword("to") ||
      parser.parseOperand(dst))
    return failure();

  bool isAsync = succeeded(parser.parseOptionalKeyword("async"));
  if (isAsync)
    result.addAttribute("async", UnitAttr::get(parser.getContext()));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon() || parser.parseType(srcType) ||
      parser.parseArrow() || parser.parseType(dstType))
    return failure();

  if (parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands))
    return failure();

  if (isAsync) {
    Type tokenType;
    if (parser.parseComma() || parser.parseType(tokenType))
      return failure();
    result.addTypes(tokenType);
  }

  return success();
}

void DataCopyOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << " to " << getDest();
  if (getAsync())
    printer << " async";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"async"});
  printer << " : " << getSource().getType() << " -> " << getDest().getType();
  if (getToken())
    printer << ", " << getToken().getType();
}

//===----------------------------------------------------------------------===//
// DmaCopyOp / TmaCopyOp -- shared helper for "src to dst : srcT -> dstT"
//===----------------------------------------------------------------------===//

static ParseResult
parseAsyncCopyOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcType, dstType;
  if (parser.parseOperand(src) || parser.parseKeyword("to") ||
      parser.parseOperand(dst) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(srcType) ||
      parser.parseArrow() || parser.parseType(dstType))
    return failure();
  if (parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands))
    return failure();
  result.addTypes(coir::AsyncTokenType::get(parser.getContext()));
  return success();
}

static void printAsyncCopyOp(OpAsmPrinter &printer, Operation *op,
                              Value source, Value dest) {
  printer << " " << source << " to " << dest;
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : " << source.getType() << " -> " << dest.getType();
}

ParseResult DmaCopyOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAsyncCopyOp(parser, result);
}
void DmaCopyOp::print(OpAsmPrinter &printer) {
  printAsyncCopyOp(printer, *this, getSource(), getDest());
}

ParseResult TmaCopyOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAsyncCopyOp(parser, result);
}
void TmaCopyOp::print(OpAsmPrinter &printer) {
  printAsyncCopyOp(printer, *this, getSource(), getDest());
}

//===----------------------------------------------------------------------===//
// CoIRWhileOp
//===----------------------------------------------------------------------===//

// Format:
//   %res = coir.while (%arg = %init) : (type) -> type {
//   ^cond(%c: type):
//     ...
//     coir.while.cond(%pred) %c : type
//   ^body(%b: type):
//     ...
//     coir.continue %next : type
//   }
ParseResult CoIRWhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  SmallVector<OpAsmParser::Argument> condBlockArgs;
  SmallVector<Type> argTypes;

  if (parser.parseLParen())
    return failure();

  // Parse iter_args list: (%arg = %init, ...)
  if (parser.parseOptionalRParen()) {
    do {
      OpAsmParser::Argument regionArg;
      OpAsmParser::UnresolvedOperand initVal;
      if (parser.parseArgument(regionArg) || parser.parseEqual() ||
          parser.parseOperand(initVal))
        return failure();
      condBlockArgs.push_back(regionArg);
      initOperands.push_back(initVal);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  // Parse `: (types) -> (types)`
  FunctionType funcType;
  if (parser.parseColon() || parser.parseType(funcType))
    return failure();

  argTypes = llvm::to_vector(funcType.getInputs());
  result.addTypes(funcType.getResults());

  // Set types on region args
  for (unsigned i = 0; i < condBlockArgs.size(); ++i)
    condBlockArgs[i].type = argTypes[i];

  // Resolve init operands
  if (parser.resolveOperands(initOperands, argTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Parse condition region (with implicit block args from iter_args)
  auto *condRegion = result.addRegion();
  if (parser.parseRegion(*condRegion, condBlockArgs))
    return failure();

  // Parse body region (its own block args)
  auto *bodyRegion = result.addRegion();
  if (parser.parseRegion(*bodyRegion))
    return failure();

  return success();
}

void CoIRWhileOp::print(OpAsmPrinter &printer) {
  auto inits = getInits();
  auto &condBlock = getCondRegion().front();
  auto &bodyBlock = getBodyRegion().front();

  printer << " (";
  llvm::interleaveComma(
      llvm::zip(condBlock.getArguments(), inits), printer,
      [&](auto pair) {
        printer << std::get<0>(pair) << " = " << std::get<1>(pair);
      });
  printer << ") : (";
  llvm::interleaveComma(
      condBlock.getArgumentTypes(), printer,
      [&](Type ty) { printer << ty; });
  printer << ") -> (";
  llvm::interleaveComma(
      getResultTypes(), printer,
      [&](Type ty) { printer << ty; });
  printer << ") ";

  printer.printRegion(getCondRegion(), /*printEntryBlockArgs=*/false);
  printer.printRegion(getBodyRegion());
}

LogicalResult CoIRWhileOp::verify() {
  auto &condRegion = getCondRegion();
  auto &bodyRegion = getBodyRegion();

  if (condRegion.empty() || bodyRegion.empty())
    return emitOpError("requires non-empty condition and body regions");

  auto numInits = getInits().size();
  auto &condBlock = condRegion.front();
  auto &bodyBlock = bodyRegion.front();

  if (condBlock.getNumArguments() != numInits)
    return emitOpError("condition region block argument count (")
           << condBlock.getNumArguments()
           << ") must match init count (" << numInits << ")";

  if (bodyBlock.getNumArguments() != numInits)
    return emitOpError("body region block argument count (")
           << bodyBlock.getNumArguments()
           << ") must match init count (" << numInits << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// ElementCopyOp (uses declarative format, no custom parse/print needed)
//===----------------------------------------------------------------------===//
