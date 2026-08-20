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
  if (parser.parseOperand(desc))
    return failure();

  llvm::SmallVector<OpAsmParser::UnresolvedOperand> dynDims;
  if (succeeded(parser.parseOptionalKeyword("dyn_dims"))) {
    if (parser.parseLParen() ||
        parser.parseOperandList(dynDims) ||
        parser.parseRParen())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(descType))
    return failure();
  if (parser.resolveOperand(desc, descType, result.operands))
    return failure();
  auto indexType = mlir::IndexType::get(parser.getContext());
  for (auto &dyn : dynDims)
    if (parser.resolveOperand(dyn, indexType, result.operands))
      return failure();
  result.addTypes(coir::AsyncTokenType::get(parser.getContext()));
  return success();
}

void DMAInvokeOp::print(OpAsmPrinter &printer) {
  printer << " " << getDesc();
  if (!getDynDims().empty()) {
    printer << " dyn_dims(";
    llvm::interleaveComma(getDynDims(), printer);
    printer << ")";
  }
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
// KernelReturnOp
//===----------------------------------------------------------------------===//

LogicalResult KernelReturnOp::verify() {
  Operation *parent = (*this)->getParentOp();
  while (parent) {
    if (isa<KernelOp>(parent))
      return success();
    parent = parent->getParentOp();
  }
  return emitOpError("expects ancestor op 'coir.kernel'");
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
  printer << " " << getSource() << " to " << getDest();
  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (getKind() == coir::DMAKind::Copy)
    elidedAttrs.push_back("kind");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << " : " << getSource().getType() << " -> " << getDest().getType();
}

ParseResult TmaCopyOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAsyncCopyOp(parser, result);
}
void TmaCopyOp::print(OpAsmPrinter &printer) {
  printAsyncCopyOp(printer, *this, getSource(), getDest());
}

LogicalResult TmaCopyOp::verify() {
  // Lenient when the attribute is absent: this verifier runs during ASTIRGen
  // op construction, before StampTargetOnModule stamps "coir.has_tma".
  // verify-each re-runs it afterward, when a `false` value correctly fails.
  auto module = (*this)->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return success();
  auto attr = module->getAttrOfType<BoolAttr>("coir.has_tma");
  if (!attr || attr.getValue())
    return success();
  return emitOpError(
      "requires TMA support but target does not provide it; "
      "use dma.copy instead or target a TMA-capable architecture");
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
// FutureRotateOp
//===----------------------------------------------------------------------===//

// Format: %r0, %r1 = coir.async.rotate %a, %b : !coir.async
ParseResult FutureRotateOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  Type tokenTy;
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(tokenTy))
    return failure();
  SmallVector<Type> types(operands.size(), tokenTy);
  if (parser.resolveOperands(operands, types, parser.getNameLoc(),
                             result.operands))
    return failure();
  result.addTypes(types);
  return success();
}

void FutureRotateOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperands(getFutures());
  printer.printOptionalAttrDict((*this)->getAttrs());
  if (!getFutures().empty())
    printer << " : " << getFutures().front().getType();
}

//===----------------------------------------------------------------------===//
// TensorAllocOp
//===----------------------------------------------------------------------===//

// Format:
//   %0 = coir.tensor.alloc : !coir.tensor<128x64xf16, shared>
//   %1 = coir.tensor.alloc(%K, %N) : !coir.tensor<?x?xi8, shared>
//   %2 = coir.tensor.alloc init -5 : i32 : !coir.tensor<32x4xi32, local>
ParseResult TensorAllocOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> dynDims;
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseOperandList(dynDims) || parser.parseRParen())
      return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    Attribute initAttr;
    if (parser.parseAttribute(initAttr))
      return failure();
    result.addAttribute("init", initAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();
  Type resultType;
  if (parser.parseType(resultType))
    return failure();
  result.addTypes(resultType);
  auto indexTy = parser.getBuilder().getIndexType();
  SmallVector<Type> dynDimTypes(dynDims.size(), indexTy);
  if (parser.resolveOperands(dynDims, dynDimTypes, parser.getNameLoc(),
                             result.operands))
    return failure();
  return success();
}

void TensorAllocOp::print(OpAsmPrinter &printer) {
  auto dynDims = getDynamicDims();
  if (!dynDims.empty()) {
    printer << "(";
    printer.printOperands(dynDims);
    printer << ")";
  }
  if (auto initAttr = getInit()) {
    printer << " init ";
    printer.printAttribute(*initAttr);
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"init", "spm"});
  printer << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// TensorBindDimsOp
//===----------------------------------------------------------------------===//

/// Parse the dimensions of a tensor type in bind_dims context.
/// Dimensions are separated by commas (`,`), not `x`, to avoid ambiguity
/// with SSA value names that may contain `x`.
/// Accepts: integer (static) or `%name` (bound SSA).
/// Returns the shape (with kDynamic for dynamic dims) and collects SSA dims.
static ParseResult parseBindDimsShape(
    OpAsmParser &parser, SmallVectorImpl<int64_t> &shape,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ssaDims) {
  // Parse the first dimension (required).
  bool isFirst = true;
  while (true) {
    if (!isFirst) {
      if (parser.parseComma())
        return failure();
    }
    isFirst = false;

    OpAsmParser::UnresolvedOperand ssaOp;
    // Try parsing %ssa_ref
    auto ssaRes = parser.parseOptionalOperand(ssaOp);
    if (ssaRes.has_value() && succeeded(*ssaRes)) {
      shape.push_back(ShapedType::kDynamic);
      ssaDims.push_back(ssaOp);
      continue;
    }
    // Try parsing integer
    int64_t dim;
    auto intRes = parser.parseOptionalInteger(dim);
    if (intRes.has_value() && succeeded(*intRes)) {
      shape.push_back(dim);
      continue;
    }
    // No more dimensions -- break (caller will parse elemType).
    break;
  }
  return success();
}

/// Parse a memory-space keyword.  Returns -1 (default) or -2 (invalid).
static int32_t parseMemSpace(llvm::StringRef kw) {
  if (kw == "default")  return -1;
  if (kw == "global")   return (int32_t)TensorMemorySpace::Global;
  if (kw == "shared")   return (int32_t)TensorMemorySpace::Shared;
  if (kw == "local")    return (int32_t)TensorMemorySpace::Local;
  if (kw == "register") return (int32_t)TensorMemorySpace::Register;
  return -2;
}

ParseResult TensorBindDimsOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // ---- parse the (dynamic-dims) prefix                          ---- //
  SmallVector<OpAsmParser::UnresolvedOperand> prefixDims;
  if (parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen())) {
    // Empty parens -- dynamic dims will be parsed from the type.
  } else {
    if (parser.parseOperandList(prefixDims) || parser.parseRParen())
      return failure();
  }

  // ---- parse source operand                                     ---- //
  OpAsmParser::UnresolvedOperand source;
  if (parser.parseOperand(source))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon())
    return failure();

  // ---- parse !coir.tensor<shape x elemTy, memSpace>             ---- //
  // We manually parse the tensor type so that SSA value references
  // (%name) inside the shape are recognised.
  StringRef tensorKw;
  if (parser.parseKeyword(&tensorKw))
    return failure();
  if (tensorKw != "tensor") {
    parser.emitError(parser.getCurrentLocation(),
                     "expected tensor type for bind_dims result");
    return failure();
  }
  if (parser.parseLess())
    return failure();

  // shape (with optional SSA dim refs, comma-separated)
  SmallVector<int64_t> shape;
  SmallVector<OpAsmParser::UnresolvedOperand> typeSsaDims;
  if (failed(parseBindDimsShape(parser, shape, typeSsaDims)))
    return failure();

  // element type
  Type elemType;
  if (parser.parseType(elemType))
    return failure();

  // optional memory space + strides
  int32_t memSpace = -1;
  SmallVector<int64_t> strides;
  if (succeeded(parser.parseOptionalComma())) {
    StringRef kw;
    if (parser.parseKeyword(&kw))
      return failure();
    if (kw == "strides") {
      // strides without memspace
      if (parser.parseColon() || parser.parseLSquare())
        return failure();
      int64_t s;
      auto res = parser.parseOptionalInteger(s);
      if (res.has_value() && succeeded(*res)) {
        strides.push_back(s);
        while (succeeded(parser.parseOptionalComma())) {
          if (parser.parseInteger(s))
            return failure();
          strides.push_back(s);
        }
      }
      if (parser.parseRSquare())
        return failure();
    } else {
      memSpace = parseMemSpace(kw);
      if (memSpace == -2) {
        parser.emitError(parser.getCurrentLocation(),
                         "unknown memory space: " + kw);
        return failure();
      }
      if (succeeded(parser.parseOptionalComma())) {
        StringRef stKw;
        if (parser.parseKeyword(&stKw) || stKw != "strides")
          return failure();
        if (parser.parseColon() || parser.parseLSquare())
          return failure();
        int64_t s;
        auto res2 = parser.parseOptionalInteger(s);
        if (res2.has_value() && succeeded(*res2)) {
          strides.push_back(s);
          while (succeeded(parser.parseOptionalComma())) {
            if (parser.parseInteger(s))
              return failure();
            strides.push_back(s);
          }
        }
        if (parser.parseRSquare())
          return failure();
      }
    }
  }

  if (parser.parseGreater())
    return failure();

  // ---- build the result type (with kDynamic sentinels)          ---- //
  auto resultType = TensorType::get(parser.getContext(), elemType, shape,
                                    memSpace, strides);
  result.addTypes(resultType);

  // ---- resolve operands                                         ---- //
  auto indexTy = parser.getBuilder().getIndexType();

  // Use type-parsed SSA dims if present; otherwise fall back to prefix.
  auto &finalDims = typeSsaDims.empty() ? prefixDims : typeSsaDims;
  SmallVector<Type> dimTypes(finalDims.size(), indexTy);
  if (!finalDims.empty() &&
      parser.resolveOperands(finalDims, dimTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  if (parser.resolveOperand(source, resultType, result.operands))
    return failure();

  return success();
}

void TensorBindDimsOp::print(OpAsmPrinter &printer) {
  // prefix: (dynamic-dim SSA values)
  printer << "(";
  printer.printOperands(getDynamicDims());
  printer << ") ";
  printer.printOperand(getSource());
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : ";

  // Print tensor type with SSA values inlined for dynamic dims.
  auto resultTy = mlir::cast<TensorType>(getResult().getType());
  auto shape = resultTy.getShape();
  auto dynDims = getDynamicDims();

  printer << "tensor<";
  unsigned dynIdx = 0;
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (i > 0)
      printer << ",";
    if (ShapedType::isDynamic(shape[i])) {
      printer.printOperand(dynDims[dynIdx++]);
    } else {
      printer << shape[i];
    }
  }
  printer << "," << resultTy.getElementType();

  int32_t ms = resultTy.getMemorySpace();
  printer << ", ";
  if (ms < 0) {
    printer << "default";
  } else {
    switch (static_cast<TensorMemorySpace>(ms)) {
    case TensorMemorySpace::Global:   printer << "global";   break;
    case TensorMemorySpace::Shared:   printer << "shared";   break;
    case TensorMemorySpace::Local:    printer << "local";    break;
    case TensorMemorySpace::Register: printer << "register"; break;
    }
  }
  auto stridesArr = resultTy.getStrides();
  if (!stridesArr.empty()) {
    printer << ", strides: [";
    llvm::interleaveComma(stridesArr, printer.getStream());
    printer << "]";
  }
  printer << ">";
}

//===----------------------------------------------------------------------===//
// AsmOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AsmOp::verify() {
  // outConstraints and outOperands must match in length
  if (getOutConstraints().size() != getOutOperands().size())
    return emitOpError()
           << "number of output constraints (" << getOutConstraints().size()
           << ") must match number of output operands ("
           << getOutOperands().size() << ")";

  // outSymbolicNames and outOperands must match in length
  if (getOutSymbolicNames().size() != getOutOperands().size())
    return emitOpError()
           << "number of output symbolic names ("
           << getOutSymbolicNames().size()
           << ") must match number of output operands ("
           << getOutOperands().size() << ")";

  // inConstraints and inOperands must match in length
  if (getInConstraints().size() != getInOperands().size())
    return emitOpError()
           << "number of input constraints (" << getInConstraints().size()
           << ") must match number of input operands ("
           << getInOperands().size() << ")";

  // inSymbolicNames and inOperands must match in length
  if (getInSymbolicNames().size() != getInOperands().size())
    return emitOpError()
           << "number of input symbolic names ("
           << getInSymbolicNames().size()
           << ") must match number of input operands ("
           << getInOperands().size() << ")";

  // results count must match output operands count
  if (getResults().size() != getOutOperands().size())
    return emitOpError()
           << "number of results (" << getResults().size()
           << ") must match number of output operands ("
           << getOutOperands().size() << ")";

  // Each output operand's type must match the corresponding result type
  for (unsigned i = 0; i < getOutOperands().size(); ++i) {
    if (getOutOperands()[i].getType() != getResults()[i].getType())
      return emitOpError()
             << "output operand " << i << " type ("
             << getOutOperands()[i].getType()
             << ") must match result type ("
             << getResults()[i].getType() << ")";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ElementCopyOp (uses declarative format, no custom parse/print needed)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BufferMapOp
//===----------------------------------------------------------------------===//

void BufferMapOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get());
  effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

ParseResult BufferMapOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source, offset, size;
  Type sourceType, resultType;
  if (parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperand(offset) || parser.parseRSquare())
    return failure();
  if (parser.parseKeyword("size") || parser.parseLParen() ||
      parser.parseOperand(size) || parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(sourceType) ||
      parser.parseArrow() || parser.parseType(resultType))
    return failure();
  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperand(offset, parser.getBuilder().getIndexType(),
                            result.operands) ||
      parser.resolveOperand(size, parser.getBuilder().getIndexType(),
                            result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void BufferMapOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << "[" << getOffset() << "] size("
          << getSize() << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

LogicalResult BufferMapOp::verify() {
  auto module = (*this)->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return success();
  bool hasBufferMap = false;
  if (auto attr =
          module->getAttrOfType<BoolAttr>("coir.has_buffer_map"))
    hasBufferMap = attr.getValue();
  if (!hasBufferMap)
    return emitOpError(
        "requires explicit memory mapping support but target does not "
        "provide it; use dma.copy instead or target a "
        "buffer-map-capable architecture");
  return success();
}

//===----------------------------------------------------------------------===//
// BufferRemapOp
//===----------------------------------------------------------------------===//

void BufferRemapOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get());
}

ParseResult BufferRemapOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand existing, source, offset, size;
  Type existingType, resultType;
  if (parser.parseOperand(existing) || parser.parseComma() ||
      parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperand(offset) || parser.parseRSquare())
    return failure();
  if (parser.parseKeyword("size") || parser.parseLParen() ||
      parser.parseOperand(size) || parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(existingType) ||
      parser.parseArrow() || parser.parseType(resultType))
    return failure();
  if (parser.resolveOperand(existing, existingType, result.operands) ||
      parser.resolveOperand(source, parser.getBuilder().getIndexType(),
                            result.operands) ||
      parser.resolveOperand(offset, parser.getBuilder().getIndexType(),
                            result.operands) ||
      parser.resolveOperand(size, parser.getBuilder().getIndexType(),
                            result.operands))
    return failure();
  result.addTypes(resultType);
  return success();
}

void BufferRemapOp::print(OpAsmPrinter &printer) {
  printer << " " << getExisting() << ", " << getSource() << "["
          << getOffset() << "] size(" << getSize() << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getExisting().getType() << " -> "
          << getResult().getType();
}

LogicalResult BufferRemapOp::verify() {
  auto module = (*this)->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return success();
  bool hasBufferMap = false;
  if (auto attr =
          module->getAttrOfType<BoolAttr>("coir.has_buffer_map"))
    hasBufferMap = attr.getValue();
  if (!hasBufferMap)
    return emitOpError(
        "requires explicit memory mapping support but target does not "
        "provide it; use dma.copy instead or target a "
        "buffer-map-capable architecture");
  return success();
}

//===----------------------------------------------------------------------===//
// BufferUnmapOp
//===----------------------------------------------------------------------===//

void BufferUnmapOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get());
  effects.emplace_back(mlir::MemoryEffects::Free::get());
}

ParseResult BufferUnmapOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand local, size;
  Type localType;
  if (parser.parseOperand(local) || parser.parseComma() ||
      parser.parseOperand(size))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(localType))
    return failure();
  if (parser.resolveOperand(local, localType, result.operands) ||
      parser.resolveOperand(size, parser.getBuilder().getIndexType(),
                            result.operands))
    return failure();
  return success();
}

void BufferUnmapOp::print(OpAsmPrinter &printer) {
  printer << " " << getLocal() << ", " << getSize();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getLocal().getType();
}
