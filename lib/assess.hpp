#ifndef __CHOREO_ASSESS_HPP__
#define __CHOREO_ASSESS_HPP__

#include "loc.hpp"
#include "symvals.hpp"
#include <cassert>
#include <string>
#include <vector>

namespace Choreo {

namespace AST {
struct Node;
}

struct Visitor;
class FunctionContext;

enum class AssessType {
  ENTRY,
  DEF_SITE,
  USE_SITE,
};

enum class AssessPolicy {
  Error,
  Warn,
  ErrWarn,
};

enum class AssessRelation {
  EQ,
  NE,
};

struct AssessResult {
  bool passed = true;
  bool warned = false;
  bool inserted = false;
};

struct Assertion {
  ptr<sbe::SymbolicExpression> expr;

  AssessType type = AssessType::ENTRY;
  location loc;
  std::string message;
  AST::Node* node = nullptr;
  /// Optional override for the emission target node.  When set,
  /// BuildSiteAssertionMap uses this instead of `node` for mapping.
  /// This is needed when the classification node (e.g., an Expr whose
  /// BoundedType determines USE_SITE) does not receive AfterVisit in the AST
  /// traversal, but its parent (e.g., Select) does.
  AST::Node* emit_node = nullptr;

  /// Return the node to use for site-assertion emission mapping.
  AST::Node* EmitTarget() const { return emit_node ? emit_node : node; }
};

class Assessor {
private:
  std::vector<Assertion> assertions;
  Visitor* visitor = nullptr;

  /// Raw assertion insertion (no evaluation, no visitor required).
  void AddAssertion(const ptr<sbe::SymbolicExpression>& ar, const location& l,
                    const std::string& s, AssessType aty,
                    AST::Node* n = nullptr, AST::Node* en = nullptr);

  bool DebugOn() const;

public:
  /// Bind a visitor for diagnostic emission. Returns *this for chaining.
  Assessor& Bind(Visitor& v) {
    visitor = &v;
    return *this;
  }

  const std::vector<Assertion>& GetAssertions() const { return assertions; }

  std::vector<Assertion> GetAssertions(AssessType aty) const {
    std::vector<Assertion> output;
    output.reserve(assertions.size());
    for (const auto& as : assertions)
      if (as.type == aty) output.push_back(as);
    return output;
  }

  /// Evaluate a relational assessment and insert runtime assertion if needed.
  AssessResult Assess(AssessPolicy ap, AssessRelation rel, const ValueItem& lhs,
                      const ValueItem& rhs, const std::string& error_message,
                      const std::string& warn_message, AssessType aty,
                      const location& l, AST::Node* node = nullptr);

  /// Convenience overload with a single message for both error and warning.
  AssessResult Assess(AssessPolicy ap, AssessRelation rel, const ValueItem& lhs,
                      const ValueItem& rhs, const std::string& message,
                      AssessType aty, const location& l,
                      AST::Node* node = nullptr);

  /// Evaluate a boolean-expression assessment and insert runtime assertion if
  /// needed.
  AssessResult Assess(AssessPolicy ap, const ValueItem& bo,
                      const std::string& message, AssessType aty,
                      const location& l, AST::Node* node = nullptr,
                      AST::Node* emit_node = nullptr);
};

} // end namespace Choreo

#endif // __CHOREO_ASSESS_HPP__
