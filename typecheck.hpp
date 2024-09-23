#ifndef __CHOREO_SEMANTIC_CHECK_HPP__
#define __CHOREO_SEMANTIC_CHECK_HPP__

// This apply the type check and symbol table generation

#include "visitor.hpp"

namespace Choreo {

struct TypeChecker : public VisitorWithSymTab {
 private:
  std::ostream &os;
  size_t error_count = 0;

 private:
  bool BeforeVisitImpl(AST::Node &) override;
  bool AfterVisitImpl(AST::Node &) override;

  bool ReportUnknown(AST::Node &, const char *, int);
  bool ReportUnknownSymbol(const std::string &, const location &, const char *,
                           int);

 public:
  TypeChecker(const ptr<SymbolTable> s_tab, std::ostream &o = std::cout)
      : VisitorWithSymTab("check", s_tab), os(o){}
  ~TypeChecker() {}

  bool Visit(AST::MultiNodes &) override;
  bool Visit(AST::MultiValues &) override;
  bool Visit(AST::IntLiteral &) override;
  bool Visit(AST::Boolean &) override;
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
  bool Visit(AST::SpanAs &) override;
  bool Visit(AST::DMA &) override;
  bool Visit(AST::ChunkAt &) override;
  bool Visit(AST::Wait &) override;
  bool Visit(AST::Call &) override;
  bool Visit(AST::Select &) override;
  bool Visit(AST::Return &) override;
  bool Visit(AST::LoopRange &) override;
  bool Visit(AST::ForeachBlock &) override;
  bool Visit(AST::FunctionDecl &) override;
  bool Visit(AST::ChoreoFunction &) override;
  bool Visit(AST::CppSourceCode &) override;
  bool Visit(AST::Program &) override;

  bool HasError();
};

}  // end namespace Choreo

#endif  // __CHOREO_SEMANTIC_CHECK_HPP__
