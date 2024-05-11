#ifndef __CHOREO_SYMBOL_VALIDATOR_CHECK_HPP__
#define __CHOREO_SYMBOL_VALIDATOR_CHECK_HPP__

// This apply the type check and symbol table generation

#include "visitor.hpp"

namespace Choreo {

struct SymbolValidator : public Visitor {
 private:
  std::ostream &os;
  bool trace_visit = false;  // for debugging purpose only
  size_t error_count = 0;

 private:
  bool in_decl =
      false;  // we need context to judge if it is declaration or reference

  bool requires_return = false;  // only void function does not require return value
  bool found_return = false;

 private:
  bool BeforeVisit(AST::Node &) override;
  bool AfterVisit(AST::Node &) override;

  bool ReportErrorWhenUseBeforeDefine(const location &, const std::string &);
  bool ReportErrorWhenViolateODR(const location &, const std::string &,
                                 const char *, int);

 public:
  SymbolValidator(std::ostream &o = std::cout)
      : os(o), trace_visit(std::getenv("TRACE_VALI")) {}
  ~SymbolValidator() {}

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
  bool Visit(AST::RequireBind &) override;
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

#endif  // __CHOREO_SYMBOL_VALIDATOR_CHECK_HPP__
