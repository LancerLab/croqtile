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
  GLOBAL,
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

  AssessType type = AssessType::GLOBAL;
  location loc;
  std::string message;
  AST::Node* node = nullptr;
};

class Assessor {
private:
  std::vector<Assertion> assertions;
  Visitor* visitor = nullptr;

  /// Raw assertion insertion (no evaluation, no visitor required).
  void AddAssertion(const ptr<sbe::SymbolicExpression>& ar, const location& l,
                    const std::string& s, AssessType aty,
                    AST::Node* n = nullptr) {
    assert(IsComputable(ar));
    assertions.push_back({ar, aty, l, s, n});
  }

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
                      const location& l, AST::Node* node = nullptr);
};

} // end namespace Choreo

#endif // __CHOREO_ASSESS_HPP__
