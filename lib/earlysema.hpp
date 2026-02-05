#ifndef __CHOREO_EARLY_SEMANTICS_CHECK_HPP__
#define __CHOREO_EARLY_SEMANTICS_CHECK_HPP__

// This apply the type check and symbol table generation

#include <unordered_set>

#include "derivation.hpp"
#include "typeresolve.hpp"
#include "visitor.hpp"

namespace Choreo {

struct EarlySemantics : public VisitorWithScope {
private:
  TypeConstraints type_equals{this, true};

  AttributeDeriver mutables{this, "mutables"};
  AttributeDeriver diverges{this, "diverges"};
  std::vector<ptr<AST::DeviceFunctionDecl>> device_functions;

private:
  bool in_decl =
      false; // we need context to judge if it is declaration or reference
  bool allow_named_dim = false;   // tolerate same symbols (mdspan param only)
  bool in_template_param = false; // we are visiting template parameter list
  std::string assign_id;
  bool requires_return =
      false; // only void function does not require return value
  bool found_return = false;
  bool return_deduction = false;
  int pl_depth = 0;
  std::vector<int> pl_depths;
  bool explicit_pl = false;
  std::stack<ParallelLevel> explicit_pl_stk;
  bool allow_auto_threading = false;
  bool inside_loop = false;

  std::vector<int> inthreads_levels;

  std::unordered_set<std::string>
      with_syms; // symbol defined in with-in statement

  FutureBufferInfo& FBInfo() { return FCtx(fname).GetFutureBufferInfo(); }

private:
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool ReportErrorWhenUseBeforeDefine(const location&, const std::string&);
  bool ReportErrorWhenViolateODR(const location&, const std::string&,
                                 const char*, int,
                                 const ptr<Type>& = MakeUnknownType());

  void MutateNodeType(AST::Node& n, const ptr<Type>& ty, bool mutate) {
    if (mutate)
      n.SetType(MutateType(ty));
    else
      n.SetType(ty);
    if (debug_visit)
      VST_DEBUG(dbgs() << "Set type of "; n.InlinePrint(dbgs());
                dbgs() << " as " << PSTR(n.GetType()) << "\n");
  }
  void SetNodeType(AST::Node& n, const ptr<Type>& ty) {
    n.SetType(ty);
    VST_DEBUG(dbgs() << "Set type of "; n.InlinePrint(dbgs());
              dbgs() << " as " << PSTR(n.GetType()) << "\n");
  }
  void ModifySymbolType(const std::string& n, const ptr<Type>& ty) {
    SSTab().ModifySymbolType(n, ty);
    VST_DEBUG(dbgs() << "Modify type of " << n << " as " << PSTR(ty) << "\n");
  }

  virtual void TraceEachVisit(AST::Node& n, bool detail = false,
                              const std::string& m = "") const {
    if (!trace_visit) return;
    dbgs() << m;
    if (detail)
      n.InlinePrint(dbgs());
    else
      dbgs() << n.TypeNameString();
    dbgs() << "\n";
  }

private:
  // shared routine for declarations inside NameVariableDecl and Assignment
  bool CheckInitializerType(const ptr<Type>&, const std::string&,
                            const location&);
  using DeviceTemplateParam = AST::DeviceFunctionDecl::DeviceTemplateParam;
  bool ParseTemplateParams(std::string input,
                           std::vector<DeviceTemplateParam>& template_params);

public:
  EarlySemantics() : VisitorWithScope("sema") {
    //    if (trace_visit) debug_visit = true; // force debug when tracing
    if (debug_visit) {
      trace_visit = true;
      type_equals.SetDebug(true);
    }
  }
  ~EarlySemantics() {}

  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::FloatLiteral&) override;
  bool Visit(AST::StringLiteral&) override;
  bool Visit(AST::BoolLiteral&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::CastExpr&) override;
  bool Visit(AST::AttributeExpr&) override;
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
  bool Visit(AST::Break&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Synchronize&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::InThreadsBlock&) override;
  bool Visit(AST::WhileBlock&) override;
  bool Visit(AST::IfElseBlock&) override;
  bool Visit(AST::IncrementBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::DeviceFunctionDecl&) override;
  bool Visit(AST::Program&) override;
};
} // end namespace Choreo

#endif // __CHOREO_EARLY_SEMANTICS_CHECK_HPP__
