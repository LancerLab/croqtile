#ifndef __CHOREO_TYPE_INFERENCE_HPP__
#define __CHOREO_TYPE_INFERENCE_HPP__

#include <iostream>

#include "typeresolve.hpp"
#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

struct TypeInference : public VisitorWithScope {
private:
  TypeConstraints type_equals;

private:
  ptr<Type> cur_type = nullptr;
  std::vector<ptr<Type>> cur_param_types;
  BaseType dma_fmty = BaseType::UNKNOWN;
  Storage dma_mem = Storage::NONE;
  bool allow_named_dim = false;   // named dimensions (mdspan param only)
  bool in_template_param = false; // we are visiting template parameter list

  bool BeforeBeforeVisit(AST::Node&) override;
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool AssignSymbolWithType(const location&, const std::string&,
                            const ptr<Type>&);
  ptr<Type> GetSymbolType(const location&, const std::string&);
  bool ModifySymbolType(const location&, const std::string&, const ptr<Type>&);
  void SetNodeType(AST::Node& n, const ptr<Type>& ty) {
    n.SetType(ty);
    if (debug_visit) {
      dbgs() << " |- node type: `";
      n.InlinePrint(dbgs());
      dbgs() << "' -> '" << PSTR(n.GetType()) << "'\n";
    }
  }

  void TraceEachVisit(const AST::Node& n) {
    if (trace_visit)
      dbgs() << n.TypeNameString() << ":\n";
    else if (debug_visit) {
      dbgs() << "[" << n.TypeNameString() << "] ";
      n.InlinePrint(dbgs());
      dbgs() << " | cur_type: " << PSTR(cur_type) << "\n";
    }
  }

public:
  TypeInference() : VisitorWithScope("infer"), type_equals(this) {
    type_equals.SetTypeReport(CCtx().ShowInferredTypes());
  }

  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::FloatLiteral&) override;
  bool Visit(AST::StringLiteral&) override;
  bool Visit(AST::BoolLiteral&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::CastExpr&) override;
  bool Visit(AST::MultiDimSpans&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::IntTuple&) override;
  bool Visit(AST::DataAccess&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::IntIndex&) override;
  bool Visit(AST::DataType&) override;
  bool Visit(AST::Identifier&) override;
  bool Visit(AST::Parameter&) override;
  bool Visit(AST::ParamList&) override;
  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::WhereBind&) override;
  bool Visit(AST::WithIn&) override;
  bool Visit(AST::WithBlock&) override;
  bool Visit(AST::Memory&) override;
  bool Visit(AST::SpanAs&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::MMA&) override;
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Trigger&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Synchronize&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::InThreadsBlock&) override;
  bool Visit(AST::IfElseBlock&) override;
  bool Visit(AST::IncrementBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::Program&) override;

private:
  bool SetAsCurrentType(AST::Node&, const std::string&);
};

} // end namespace Choreo

#endif // __CHOREO_TYPE_INFERENCE_HPP__
