#ifndef __CHOREO_EARLY_SEMANTICS_CHECK_HPP__
#define __CHOREO_EARLY_SEMANTICS_CHECK_HPP__

// This apply the type check and symbol table generation

#include <unordered_set>

#include "typeresolve.hpp"
#include "visitor.hpp"

namespace Choreo {

struct EarlySemantics : public VisitorWithScope {
private:
  std::ostream& os;
  size_t error_count = 0;

  TypeConstraints type_equals{this};

private:
  bool in_decl =
      false; // we need context to judge if it is declaration or reference
  bool allow_named_dim = false; // tolerate same symbols (mdspan param only)

  bool requires_return =
      false; // only void function does not require return value
  bool found_return = false;
  bool return_deduction = false;
  int parallel_level = 0;
  bool allow_auto_threading = false;

  std::unordered_set<std::string>
      with_syms; // symbol defined in with-in statement

private:
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool ReportErrorWhenUseBeforeDefine(const location&, const std::string&);
  bool ReportErrorWhenViolateODR(const location&, const std::string&,
                                 const char*, int,
                                 const ptr<Type>& = MakeUnknownType());

  void SetNodeType(AST::Node& n, const ptr<Type>& ty) {
    n.SetType(ty);
    if (debug_visit)
      os << "Set type of " << STR(n) << " as " << PSTR(n.GetType()) << "\n";
  }
  void ModifySymbolType(const std::string& n, const ptr<Type>& ty) {
    SSTab().ModifySymbolType(n, ty);
    if (debug_visit)
      os << "Modify type of " << STR(n) << " as " << PSTR(ty) << "\n";
  }

  virtual void TraceEachVisit(AST::Node& n, bool detail = false,
                              const std::string& m = "") const {
    if (!trace_visit) return;
    if (detail)
      os << m << STR(n) << "\n";
    else
      os << m << n.TypeNameString() << "\n";
  }

public:
  EarlySemantics(std::ostream& o = outs(),
                 const Choreo::Target& tgt = Choreo::Target::Factor)
      : VisitorWithScope("sema"), os(o) {
    if (trace_visit) debug_visit = true; // force debug when tracing
    if (tgt == Choreo::Target::CUDA) allow_auto_threading = true;
    if (debug_visit) type_equals.SetDebug(true);
  }
  ~EarlySemantics() {}

  bool Visit(AST::MultiNodes&) override;
  bool Visit(AST::MultiValues&) override;
  bool Visit(AST::IntLiteral&) override;
  bool Visit(AST::Boolean&) override;
  bool Visit(AST::Expr&) override;
  bool Visit(AST::MultiDimSpans&) override;
  bool Visit(AST::NamedTypeDecl&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::IntTuple&) override;
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
  bool Visit(AST::ChunkAt&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::Rotate&) override;
  bool Visit(AST::Select&) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::LoopRange&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::FunctionDecl&) override;
  bool Visit(AST::ChoreoFunction&) override;
  bool Visit(AST::CppSourceCode&) override;
  bool Visit(AST::Program&) override;

  bool HasError();
};

} // end namespace Choreo

#endif // __CHOREO_EARLY_SEMANTICS_CHECK_HPP__
