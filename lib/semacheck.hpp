#ifndef __CHOREO_SEMANTIC_CHECK_HPP__
#define __CHOREO_SEMANTIC_CHECK_HPP__

// This apply the type check and symbol table generation

#include "derivation.hpp"
#include "visitor.hpp"

#include <map>

namespace Choreo {

struct SemaChecker : public TracedVisitorWithSymTab {
private:
  std::set<std::string> pending_async; // a simple check to detect async
                                       // entities that are not waited
  std::set<std::string> waited_async;  // a simple check to detect async
                                       // entities that are not waited
  // Source-order state for first-class futures. pending_async/waited_async
  // retain the whole-function completeness check. Each dynamic future
  // instance is completed by an explicit wait or a native completion-event
  // binding. Data-future symbols may have multiple static wait sites in a
  // swap/rotation pipeline.
  std::set<std::string> live_async;
  std::set<std::string> observed_async;
  std::set<std::string> completed_async;

  AttributeDeriver input_deps{this, "input-deps", false};
  AttributeDeriver local_deps{this, "local-deps", false};
  std::vector<ValueItem> scope_pred_stack;
  std::vector<ParallelLevel> parallel_level_stack;
  std::vector<bool> cooperative_stack;
  std::map<std::string, AST::DMA*> shared_tensor_producers;
  std::map<std::string, AST::DMA*> async_dma_producers;

private:
  bool BeforeVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;
  bool InMidVisitImpl(AST::Node&) override;

  bool ReportUnknown(AST::Node&, const char*, int, bool = false);
  bool ReportUnknownSymbol(const std::string&, const location&, const char*,
                           int);

  void RecordAsyncProducer(const std::string&);
  bool ObserveAsyncFuture(const std::string&, const location&,
                          const std::string&);
  bool ConsumeAsyncFuture(const std::string&, const location&,
                          const std::string&);

  void CreateAssessment(const ValueItem&, const std::string&, const location&,
                        const ptr<AST::Node>&,
                        UsageType uty = UsageType::ShapeCompatibility,
                        AST::Node* emit_node = nullptr);
  ValueItem ActiveScopePredicate() const;
  bool ExpressionIsConstrained(const ValueItem&) const;

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
  bool VisitNode(AST::IfElseBlock&) override;
  bool VisitNode(AST::ParallelBy&) override;
  bool VisitNode(AST::Barrier&) override;
  bool VisitNode(AST::Fence&) override;
  bool VisitNode(AST::WithIn&) override;
  bool VisitNode(AST::SpanAs&) override;
  bool VisitNode(AST::DMA&) override;
  bool VisitNode(AST::BufferMap&) override;
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
