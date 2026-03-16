#ifndef __CHOREO_ASSERT_SITE_HPP__
#define __CHOREO_ASSERT_SITE_HPP__

/// AssertSite -- hoist runtime assertions to the earliest safe placement site.
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

#include <unordered_map>
#include <vector>

namespace Choreo {

struct AssertSite : public VisitorWithSymTab {
  AssertSite() : VisitorWithSymTab("assertsite") {}
  ~AssertSite() {}

private:
  /// Map from scoped symbol name to the AST node that defines it.
  /// Built per function while traversing the body.
  std::unordered_map<std::string, AST::Node*> def_map;
  std::unordered_map<AST::Node*, AST::Node*> parent_map;
  std::vector<AST::Node*> parent_stack;

  /// Walk order counter, used to pick the *latest* defining node when an
  /// assertion references multiple symbols.
  size_t walk_order = 0;

  /// Map from AST node pointer to its walk order.
  std::unordered_map<AST::Node*, size_t> node_order;

  /// Reset per-function state.
  void ResetFunction();

  /// After all definitions have been collected during the traversal, hoist
  /// eligible assertions for the current function.
  void HoistAssertions();

  AST::Node* FirstStmtOf(const ptr<AST::MultiNodes>& stmts) const;
  size_t EstimateLoopTripCount(AST::Node* n) const;
  uint64_t EstimateAssertionCost(AST::Node* site) const;
  AssertionCost CategorizeCost(uint64_t cost) const;
  AssertionEmitPosition SiteEmitPosition(AST::Node* n) const;
  AST::Node* NextStatementInBlock(AST::Node* n) const;
  AST::Node* LaterNode(AST::Node* lhs, AST::Node* rhs) const;

  /// Return true when `container` (a MultiNodes statement list) is directly
  /// inside a conditional or iterative scope (IfElseBlock, ForeachBlock, or
  /// WhileBlock).  Used to prevent parameter-only assertions from being
  /// promoted to ENTRY when the access lives inside a guarded block.

  /// Find the MultiNodes (statement list) that directly contains `use`.
  AST::MultiNodes* FindStatementContainer(AST::Node* use) const;

  /// Walk up from `def` through `parent_map` until the immediate parent is
  /// `container`.  Returns that ancestor (the "statement" in `container` that
  /// contains `def`).  If `def` is already a direct child of `container`,
  /// returns `def`.  Returns `def` unchanged when no ancestor is found.
  ///
  /// This implements hoist-barrier case 3.iii: if a block statement contains
  /// a hoist barrier (definition or mutation) in its body, the block itself
  /// becomes the barrier at the surrounding scope level.
  AST::Node* BubbleToSiblingScope(AST::Node* def,
                                  AST::MultiNodes* container) const;

  /// Record a definition of `scoped_name` at node `n`.
  void RecordDef(const std::string& scoped_name, AST::Node* n);

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
};

} // end namespace Choreo

#endif // __CHOREO_ASSERT_SITE_HPP__
