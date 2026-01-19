#ifndef __CHOREO_SEMANTIC_CHECK_HPP__
#define __CHOREO_SEMANTIC_CHECK_HPP__

// This apply the type check and symbol table generation

#include "derivation.hpp"
#include "visitor.hpp"

namespace Choreo {

struct SemaChecker : public TracedVisitorWithSymTab {
private:
  std::set<std::string> pending_async; // a simple check to detect async
                                       // entities that are not waited
  std::set<std::string> waited_async;  // a simple check to detect async
                                       // entities that are not waited

  AttributeDeriver input_deps{this, "input-deps", false};
  AttributeDeriver local_deps{this, "local-deps", false};

private:
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool ReportUnknown(AST::Node&, const char*, int, bool = false);
  bool ReportUnknownSymbol(const std::string&, const location&, const char*,
                           int);

  void EmitAssertion(const ValueItem&, const std::string&, const location&,
                     const ptr<AST::Node>&);

public:
  SemaChecker() : TracedVisitorWithSymTab("check") {}
  ~SemaChecker() {}

  bool VisitNode(AST::IntLiteral&) override;
  bool VisitNode(AST::FloatLiteral&) override;
  bool VisitNode(AST::BoolLiteral&) override;
  bool VisitNode(AST::Expr&) override;
  bool VisitNode(AST::MultiDimSpans&) override;
  bool VisitNode(AST::NamedTypeDecl&) override;
  bool VisitNode(AST::NamedVariableDecl&) override;
  bool VisitNode(AST::IntTuple&) override;
  bool VisitNode(AST::DataAccess&) override;
  bool VisitNode(AST::Assignment&) override;
  bool VisitNode(AST::IntIndex&) override;
  bool VisitNode(AST::DataType&) override;
  bool VisitNode(AST::Identifier&) override;
  bool VisitNode(AST::Parameter&) override;
  bool VisitNode(AST::ParallelBy&) override;
  bool VisitNode(AST::WithIn&) override;
  bool VisitNode(AST::SpanAs&) override;
  bool VisitNode(AST::DMA&) override;
  bool VisitNode(AST::MMA&) override;
  bool VisitNode(AST::ChunkAt&) override;
  bool VisitNode(AST::Wait&) override;
  bool VisitNode(AST::Trigger&) override;
  bool VisitNode(AST::Call&) override;
  bool VisitNode(AST::Rotate&) override;
  bool VisitNode(AST::Select&) override;
  bool VisitNode(AST::Return&) override;
};

} // end namespace Choreo

#endif // __CHOREO_SEMANTIC_CHECK_HPP__
