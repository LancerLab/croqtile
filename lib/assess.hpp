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
  HOIST_SITE,
  USE_SITE,
};

/// Semantic category of a safety assessment.
enum class UsageType {
  UnClassified,
  ShapeCompatibility, ///< Inter-operand shape contracts: broadcast-compat,
                      ///< matmul contraction dims, DMA shape equality.
  ElementAccess,      ///< Per-element index bounds: 0 <= index < dim.
  LoopBound,          ///< Iteration-space validity: dim > 0, stride != 0.
  HardwareConstraint, ///< Target-specific limits: DMA transfer size,
                      ///< alignment, thread count, padding ranges.
};

enum class AssertionEmitPosition {
  BEFORE_NODE,
  AFTER_NODE,
  IN_BLOCK,
};

enum class AssertionCost {
  NONE,
  ENTRY,
  LOW,
  MEDIUM,
  HIGH,
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

/// Compile-time evaluation outcome of a single bounds/safety check.
enum class AssessOutcome {
  STATIC_TRUE,  ///< Proven safe at compile time -- no code generated.
  STATIC_FALSE, ///< Proven unsafe at compile time -- compile error/warning.
  RUNTIME,      ///< Cannot evaluate -- runtime assertion emitted.
};

/// Record of every assessment evaluation, regardless of outcome.
struct AssessmentEntry {
  std::string message;
  location loc;
  AssessOutcome outcome;
  UsageType usage_type = UsageType::UnClassified;
  /// Index into Assessor::assertions (RUNTIME only); SIZE_MAX otherwise.
  size_t assertion_idx = static_cast<size_t>(-1);
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
  AST::Node* emit_node = nullptr;
  AssertionEmitPosition emit_position = AssertionEmitPosition::AFTER_NODE;
  uint64_t estimated_cost = 1;
  AssertionCost cost = AssertionCost::LOW;
  bool enabled = true;
  UsageType usage_type = UsageType::UnClassified;

  /// Return the node to use for site-assertion emission mapping.
  AST::Node* EmitTarget() const {
    assert(emit_node != nullptr);
    return emit_node;
  }
};

class Assessor {
private:
  std::vector<Assertion> assertions;
  Visitor* visitor = nullptr;

  /// Raw assertion insertion (no evaluation, no visitor required).
  void AddAssertion(const ptr<sbe::SymbolicExpression>& ar, const location& l,
                    const std::string& s, AssessType aty, UsageType uty,
                    AST::Node* n, AST::Node* en = nullptr);

  /// Record a single assessment evaluation to the ordered log.
  void LogAssessment(const std::string& msg, const location& l,
                     AssessOutcome outcome, UsageType uty,
                     size_t assertion_idx = static_cast<size_t>(-1));

  bool DebugOn() const;

  std::vector<AssessmentEntry> assessment_log;

public:
  /// Bind a visitor for diagnostic emission. Returns *this for chaining.
  Assessor& Bind(Visitor& v) {
    visitor = &v;
    return *this;
  }

  const std::vector<Assertion>& GetAssertions() const { return assertions; }
  const std::vector<AssessmentEntry>& GetAssessmentLog() const {
    return assessment_log;
  }

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
                      const std::string& warn_message, UsageType uty,
                      AssessType aty, const location& l, AST::Node* node);

  /// Convenience overload with a single message for both error and warning.
  AssessResult Assess(AssessPolicy ap, AssessRelation rel, const ValueItem& lhs,
                      const ValueItem& rhs, const std::string& message,
                      UsageType uty, AssessType aty, const location& l,
                      AST::Node* node);

  /// Evaluate a boolean-expression assessment and insert runtime assertion if
  /// needed.
  AssessResult Assess(AssessPolicy ap, const ValueItem& bo,
                      const std::string& message, UsageType uty, AssessType aty,
                      const location& l, AST::Node* node,
                      AST::Node* emit_node = nullptr,
                      const ValueItem& guard = GetInvalidValueItem());
};

} // end namespace Choreo

#endif // __CHOREO_ASSESS_HPP__
