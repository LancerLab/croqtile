#ifndef __CHOREO_ASSERT_SITE_HPP__
#define __CHOREO_ASSERT_SITE_HPP__

/// AssertSite - hoist runtime assertions to the earliest safe placement site.
///
/// After SemaChecker produces assertions marked ENTRY, HOIST_SITE, or
/// USE_SITE, this pass walks each function body and moves every non-entry
/// assertion to the earliest site that is still after all referenced symbol
/// definitions.  Hoisting is conservative: assertions inside conditional
/// scopes (if/else, foreach, while) are never promoted past the enclosing
/// block boundary.
///
/// The pass runs inside PlanCodeGenStages(), after target-specific checks
/// and before the actual code generator.

#include "assess.hpp"
#include "visitor.hpp"

#include <set>
#include <unordered_map>
#include <vector>

namespace Choreo {

struct AssertSite : public VisitorWithSymTab {
  AssertSite() : VisitorWithSymTab("assert-site") {}
  ~AssertSite() {}

private:
  /// Map from scoped symbol name to the AST node that defines it.
  /// Built per function while traversing the body.
  std::unordered_map<std::string, AST::Node*> def_map;
  std::unordered_map<AST::Node*, AST::Node*> scope_map;
  std::unordered_map<AST::Node*, int> scope_level;
  std::unordered_map<AST::Node*, int> block_cost;
  std::vector<AST::Node*> scopes;
  std::unordered_map<std::string, std::set<AST::Node*>> barrier_map;

  /// Walk order counter, used to pick the *latest* defining node when an
  /// assertion references multiple symbols.
  size_t walk_order = 0;

  /// Walk order of the first statement in the function body (after params).
  size_t body_order = 0;

  /// Map from AST node pointer to its walk order.
  std::unordered_map<AST::Node*, size_t> node_order;

  bool allow_named_dim;

  /// Reset per-function state.
  void ResetFunction();

  /// After all definitions have been collected during the traversal, hoist
  /// eligible assertions for the current function.
  void HoistAssertions(AST::ChoreoFunction*);
  void EstimateAssertions();

  AST::Node* FirstStmtOf(AST::Block* stmts) const;
  size_t EstimateLoopTripCount(AST::Node* n) const;
  uint64_t EstimateAssertionCost(AST::Node* site) const;
  AssertionCost CategorizeCost(uint64_t cost) const;

  /// Record a definition of `scoped_name` at node `n`.
  void RecordDef(const std::string& scoped_name, AST::Node* n);

  /// Record a hoisting barrier for symbol `scoped_name` at node `n`.
  void RecordBarrier(const std::string& scoped_name, AST::Node* n);

  /// Record barriers for all symbols referenced in the predicate expression.
  bool RecordBarrierFromPred(const ptr<sbe::SymbolicExpression>&, AST::Node* n);
  /// Print a human-readable assertion report to stderr (used by
  /// --show-assess).  Called at the end of HoistAssertions().
  void PrintAssertionReport() const;

  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node& n) override;

  bool Visit(AST::Parameter& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Assignment& n) override;
  bool Visit(AST::ParallelBy& n) override;
  bool Visit(AST::WithBlock& n) override;
  bool Visit(AST::LoopRange& n) override;
  bool Visit(AST::IfElseBlock& n) override;
  bool Visit(AST::WhileBlock& n) override;
  bool Visit(AST::InThreadsBlock& n) override;
  bool Visit(AST::Identifier& n) override;
  bool Visit(AST::DataType& n) override;
};

} // end namespace Choreo

#endif // __CHOREO_ASSERT_SITE_HPP__
