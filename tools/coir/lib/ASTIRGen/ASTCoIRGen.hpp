#ifndef __AST_COIR_GEN_HPP__
#define __AST_COIR_GEN_HPP__

#include "ast.hpp"
#include "codegen.hpp"
#include "MLIRUtility.hpp"
#include "Session.hpp"
#include "types.hpp"

#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "mlir/IR/Builders.h"

using namespace Choreo;

namespace CoIR {

struct ASTCoIRGen : public CodeGenerator {
  bool suppress_output = false;

private:
  mlir::ModuleOp IRModule() { return IRSession::Get().Module(); }
  mlir::MLIRContext &IRContext() { return IRSession::Get().Context(); }
  mlir::OpBuilder builder;

  const mlir::Location ToLoc(const Choreo::location &cloc) {
    return ToMLIRLoc(IRContext(), cloc);
  }
  const mlir::Location Loc(const Choreo::AST::Node &n) {
    return ToLoc(n.LOC());
  }

  mlir::Type LowerBaseType(BaseType bt);
  coir::TensorType LowerSpannedType(const ptr<SpannedType> &sty);
  coir::ParallelLevelAttr LowerParallelLevel(ParallelLevel pl);

  llvm::SmallVector<llvm::StringMap<mlir::Value>> value_stack;
  void PushScope() { value_stack.push_back({}); }
  void PopScope() { value_stack.pop_back(); }
  void MapValue(llvm::StringRef name, mlir::Value val) {
    value_stack.back()[name] = val;
  }
  mlir::Value LookupValue(llvm::StringRef name) {
    for (auto it = value_stack.rbegin(); it != value_stack.rend(); ++it)
      if (auto found = it->find(name); found != it->end())
        return found->second;
    return nullptr;
  }

  llvm::SmallVector<mlir::Value> expr_stack;
  void PushExpr(mlir::Value v) { expr_stack.push_back(v); }
  mlir::Value PopExpr() {
    if (expr_stack.empty()) return nullptr;
    auto v = expr_stack.back();
    expr_stack.pop_back();
    return v;
  }

  mlir::Value EmitExpr(AST::Node &n);
  void CreateKernelOp(AST::ChoreoFunction &cf);

public:
  ASTCoIRGen()
      : CodeGenerator("ast-coir-gen"),
        builder(&IRContext()) {}

  bool BeforeVisitImpl(AST::Node &) override;
  bool InMidVisitImpl(AST::Node &) override;
  bool AfterVisitImpl(AST::Node &) override;

  // Nodes with real translation logic
  bool Visit(AST::Program &) override;
  bool Visit(AST::ChoreoFunction &) override;
  bool Visit(AST::ParallelBy &) override;
  bool Visit(AST::ForeachBlock &) override;
  bool Visit(AST::Assignment &) override;
  bool Visit(AST::Return &) override;
  bool Visit(AST::NamedVariableDecl &) override;
  bool Visit(AST::FunctionDecl &) override;

  // Stub overrides for all other pure-virtual Visit methods
  bool Visit(AST::MultiNodes &) override { return true; }
  bool Visit(AST::MultiValues &) override { return true; }
  bool Visit(AST::NoValue &) override { return true; }
  bool Visit(AST::IntLiteral &) override { return true; }
  bool Visit(AST::FloatLiteral &) override { return true; }
  bool Visit(AST::StringLiteral &) override { return true; }
  bool Visit(AST::BoolLiteral &) override { return true; }
  bool Visit(AST::Expr &) override { return true; }
  bool Visit(AST::CastExpr &) override { return true; }
  bool Visit(AST::AttributeExpr &) override { return true; }
  bool Visit(AST::MultiDimSpans &) override { return true; }
  bool Visit(AST::NamedTypeDecl &) override { return true; }
  bool Visit(AST::IntTuple &) override { return true; }
  bool Visit(AST::DataAccess &) override { return true; }
  bool Visit(AST::IntIndex &) override { return true; }
  bool Visit(AST::DataType &) override { return true; }
  bool Visit(AST::Identifier &) override { return true; }
  bool Visit(AST::Parameter &) override { return true; }
  bool Visit(AST::ParamList &) override { return true; }
  bool Visit(AST::WhereBind &) override { return true; }
  bool Visit(AST::WithIn &) override { return true; }
  bool Visit(AST::WithBlock &) override { return true; }
  bool Visit(AST::Memory &) override { return true; }
  bool Visit(AST::SpanAs &) override { return true; }
  bool Visit(AST::DMA &) override;
  bool Visit(AST::MMA &) override { return true; }
  bool Visit(AST::ChunkAt &) override { return true; }
  bool Visit(AST::Wait &) override { return true; }
  bool Visit(AST::Trigger &) override { return true; }
  bool Visit(AST::Break &) override { return true; }
  bool Visit(AST::Continue &) override { return true; }
  bool Visit(AST::Yield &) override { return true; }
  bool Visit(AST::Call &) override { return true; }
  bool Visit(AST::Rotate &) override { return true; }
  bool Visit(AST::Synchronize &) override { return true; }
  bool Visit(AST::Select &) override { return true; }
  bool Visit(AST::LoopRange &) override { return true; }
  bool Visit(AST::InThreadsBlock &) override { return true; }
  bool Visit(AST::WhileBlock &) override { return true; }
  bool Visit(AST::IfElseBlock &) override { return true; }
  bool Visit(AST::CppSourceCode &) override;
  bool Visit(AST::DeviceFunctionDecl &) override { return true; }
};

} // namespace CoIR

#endif // __AST_COIR_GEN_HPP__
