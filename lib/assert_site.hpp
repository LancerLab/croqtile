#ifndef __CHOREO_ASSERT_SITE_HPP__
#define __CHOREO_ASSERT_SITE_HPP__

/// AssertSite — Classify USE_SITE / DEF_SITE assertions and compute their
/// emission placement nodes.
///
/// After SemaChecker produces assertions marked GLOBAL, USE_SITE, or
/// DEF_SITE, this pass walks each function body and does two things:
///
/// 1. For every USE_SITE assertion, set `Assertion.node` to the originating
///    AST node (already done by SemaChecker — kept for completeness).
///
/// 2. For every DEF_SITE assertion, determine the *defining statement* of
///    every symbol referenced in the assertion expression (via
///    `GetSymbols(expr)`), then set `Assertion.node` to the defining
///    statement that appears latest in the pre-order walk so that all
///    referenced symbols are guaranteed to be defined.
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

  /// Walk order counter, used to pick the *latest* defining node when an
  /// assertion references multiple symbols.
  size_t walk_order = 0;

  /// Map from AST node pointer to its walk order.
  std::unordered_map<AST::Node*, size_t> node_order;

  /// Reset per-function state.
  void ResetFunction();

  /// After all definitions have been collected during the traversal, resolve
  /// DEF_SITE assertion placement for the current function.
  void ResolveDefSiteAssertions();

  /// Record a definition of `scoped_name` at node `n`.
  void RecordDef(const std::string& scoped_name, AST::Node* n);

  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node& n) override;

  bool Visit(AST::Parameter& n) override;
  bool Visit(AST::NamedVariableDecl& n) override;
  bool Visit(AST::Assignment& n) override;
};

} // end namespace Choreo

#endif // __CHOREO_ASSERT_SITE_HPP__
