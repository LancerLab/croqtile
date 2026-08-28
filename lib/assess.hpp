#ifndef __CHOREO_ASSESS_HPP__
#define __CHOREO_ASSESS_HPP__

#include "loc.hpp"
#include "symvals.hpp"
#include <cassert>
#include <initializer_list>
#include <optional>
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

/// Information dependence of an assessment's predicate (RQ4 capability-gap
/// analysis): what class of information the discharge relies on.
enum class AssessDependence {
  CONSTANT,         ///< No symbolic content (constants only).
  SCALAR_SYMBOLIC,  ///< Plain scalar symbols (params/dims) that survive
                    ///< lowering as runtime values.
  STRUCTURAL,       ///< References block/thread-parallel structure (symbols
                    ///< scoped under paraby_/inthreads_) that lowering
                    ///< dissolves into flat index arithmetic.
};

/// Resolution mechanism that decided an assessment (RQ6 mechanism
/// breakdown). CANONICAL: structural normalization/constant folding of the
/// predicate itself. INTERVAL: interval analysis over bounded-type ranges
/// (TryProveWithIntervals in the semantic checker).
enum class AssessMechanism {
  CANONICAL,
  INTERVAL,
};

/// Record of every assessment evaluation, regardless of outcome.
struct AssessmentEntry {
  std::string message;
  location loc;
  AssessOutcome outcome;
  UsageType usage_type = UsageType::UnClassified;
  AssessDependence dependence = AssessDependence::CONSTANT;
  AssessMechanism mechanism = AssessMechanism::CANONICAL;
  /// Index into Assessor::assertions (RUNTIME only); SIZE_MAX otherwise.
  size_t assertion_idx = static_cast<size_t>(-1);
};

struct AssessResult {
  bool passed = true;
  bool warned = false;
  bool inserted = false;
};

// Classify the information dependence of an assessment predicate from the
// symbols referenced by its (pre-decision) operands: structural symbols are
// those scoped under block/thread-parallel boundaries (paraby_/inthreads_),
// which exist only at the semantic stage.
inline AssessDependence ClassifyDependence(
    std::initializer_list<const ValueItem*> operands) {
  bool has_symbol = false;
  for (auto* op : operands) {
    if (!op || !IsValidValueItem(*op)) continue;
    for (const auto& sym : GetSymbols(*op)) {
      has_symbol = true;
      if (auto name = VISym(sym)) {
        if (name->find("paraby_") != std::string::npos ||
            name->find("inthreads_") != std::string::npos)
          return AssessDependence::STRUCTURAL;
      }
    }
  }
  return has_symbol ? AssessDependence::SCALAR_SYMBOLIC
                    : AssessDependence::CONSTANT;
}

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
                     AssessDependence dep = AssessDependence::CONSTANT,
                     size_t assertion_idx = static_cast<size_t>(-1),
                     AssessMechanism mech = AssessMechanism::CANONICAL);

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
                      const ValueItem& guard = GetInvalidValueItem(),
                      std::optional<AssessDependence> dep_override =
                          std::nullopt,
                      AssessMechanism mech = AssessMechanism::CANONICAL);
};

} // end namespace Choreo

#endif // __CHOREO_ASSESS_HPP__
