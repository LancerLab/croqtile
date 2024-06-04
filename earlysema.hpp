#ifndef __CHOREO_EARLY_SEMANTICS_CHECK_HPP__
#define __CHOREO_EARLY_SEMANTICS_CHECK_HPP__

// This apply the type check and symbol table generation

#include <unordered_set>

#include "visitor.hpp"

namespace Choreo {

struct EarlySemantics : public Visitor {
 private:
  std::ostream &os;
  bool trace_visit = false;  // for debugging purpose only
  size_t error_count = 0;

 private:
  bool in_decl =
      false;  // we need context to judge if it is declaration or reference

  bool requires_return =
      false;  // only void function does not require return value
  bool found_return = false;
  bool return_deduction = false;
  int parallel_level = 0;

  std::unordered_set<std::string>
      with_syms;  // symbol defined in with-in statement

 private:
  bool BeforeVisit(AST::Node &) override;
  bool AfterVisit(AST::Node &) override;

  bool ReportErrorWhenUseBeforeDefine(const location &, const std::string &);
  bool ReportErrorWhenViolateODR(const location &, const std::string &,
                                 const char *, int,
                                 const ptr<Type> & = MakeUnknownType());

  ptr<Type> NodeType(AST::Node &n) {
    if (auto id = dyn_cast<AST::Identifier>(&n))
      return SSTab().LookupSymbol(id->name);
    else if (auto expr = dyn_cast<AST::Expr>(&n)) {
      if (auto ref = expr->GetReference()) {
        if (auto id = dyn_cast<AST::Identifier>(ref))
          return SSTab().LookupSymbol(id->name);
      } else if (expr->op == "dataof") {
        if (auto ref = cast<AST::Expr>(expr->value_r)->GetReference()) {
          auto id = cast<AST::Identifier>(ref);
          if (!SSTab().LookupSymbol(id->name))  // make sure the symbol exists
            return nullptr;
          return SSTab().LookupSymbol(id->name + ".data");
        }
      }
    }
    return n.GetType();
  }

  void SetNodeType(AST::Node &n, const ptr<Type> &ty) {
    n.SetType(ty);
    if (trace_visit)
      os << "Set type of " << STR(n) << " as " << STR(*n.GetType()) << "\n";
  }

 public:
  EarlySemantics(std::ostream &o = std::cout)
      : os(o), trace_visit(std::getenv("TRACE_SEMA")) {}
  ~EarlySemantics() {}

  bool Visit(AST::MultiNodes &) override;
  bool Visit(AST::MultiValues &) override;
  bool Visit(AST::IntLiteral &) override;
  bool Visit(AST::Expr &) override;
  bool Visit(AST::MultiDimSpans &) override;
  bool Visit(AST::NamedTypeDecl &) override;
  bool Visit(AST::NamedVariableDecl &) override;
  bool Visit(AST::IntTuple &) override;
  bool Visit(AST::Assignment &) override;
  bool Visit(AST::IntIndex &) override;
  bool Visit(AST::DataType &) override;
  bool Visit(AST::Identifier &) override;
  bool Visit(AST::Parameter &) override;
  bool Visit(AST::ParamList &) override;
  bool Visit(AST::ParallelBy &) override;
  bool Visit(AST::WhereBind &) override;
  bool Visit(AST::WithIn &) override;
  bool Visit(AST::WithBlock &) override;
  bool Visit(AST::Memory &) override;
  bool Visit(AST::DMA &) override;
  bool Visit(AST::ChunkAt &) override;
  bool Visit(AST::Wait &) override;
  bool Visit(AST::Call &) override;
  bool Visit(AST::Return &) override;
  bool Visit(AST::ForeachBlock &) override;
  bool Visit(AST::FunctionDecl &) override;
  bool Visit(AST::ChoreoFunction &) override;
  bool Visit(AST::CppSourceCode &) override;
  bool Visit(AST::Program &) override;

  bool HasError();
};

}  // end namespace Choreo

#endif  // __CHOREO_EARLY_SEMANTICS_CHECK_HPP__
