#include "assess.hpp"
#include "visitor.hpp"

using namespace Choreo;

namespace Choreo {

inline const std::string STR(const AssessType& at) {
  switch (at) {
  case AssessType::ENTRY: return "entry";
  case AssessType::HOIST_SITE: return "hoist";
  case AssessType::USE_SITE: return "use";
  default: break;
  }
  choreo_unreachable("unsupported assess type.");
  return "";
}

inline const std::string STR(const AssertionEmitPosition& pos) {
  switch (pos) {
  case AssertionEmitPosition::BEFORE_NODE: return "before";
  case AssertionEmitPosition::AFTER_NODE: return "after";
  default: break;
  }
  choreo_unreachable("unsupported assertion emit position.");
  return "";
}

inline const std::string STR(const AssessPolicy& ap) {
  switch (ap) {
  case AssessPolicy::Error: return "error";
  case AssessPolicy::Warn: return "warning";
  case AssessPolicy::ErrWarn: return "error-warning";
  default: break;
  }
  choreo_unreachable("unsupported assess type.");
  return "";
}

inline const std::string STR(const AssessRelation& ar) {
  switch (ar) {
  case AssessRelation::EQ: return "==";
  case AssessRelation::NE: return "!=";
  default: break;
  }
  choreo_unreachable("unsupported assess type.");
  return "";
}
} // namespace Choreo

void Assessor::AddAssertion(const ptr<sbe::SymbolicExpression>& ar,
                            const location& l, const std::string& s,
                            AssessType aty, AST::Node* n, AST::Node* en,
                            AST::Node* guard_site,
                            AssertionEmitPosition guard_site_position) {
  if (DebugOn())
    dbgs() << " +- runtime assertion: " << sbe::PSTR(ar)
           << ", type: " << STR(aty)
           << (guard_site ? (", guard-site: " + PSTR(guard_site) + " (" +
                             STR(guard_site_position) + ")")
                          : std::string())
           << "\n";

  assert(IsComputable(ar));
  assertions.push_back(
      {ar, aty, l, s, n, en, AssertionEmitPosition::AFTER_NODE, guard_site,
       guard_site_position});
}

bool Assessor::DebugOn() const {
  return (visitor && visitor->DebugIsEnabled());
}

AssessResult Assessor::Assess(AssessPolicy ap, AssessRelation rel,
                              const ValueItem& lhs, const ValueItem& rhs,
                              const std::string& error_message,
                              const std::string& warn_message, AssessType aty,
                              const location& l, AST::Node* node) {
  if (DebugOn())
    dbgs() << "[Assess] relation: " << STR(lhs) << STR(rel) << STR(rhs)
           << ", type: " << STR(aty) << ", policy: " << STR(ap)
           << ", node: " << PSTR(node) << "\n";

  assert(visitor && "Visitor not bound. Call Bind() before Assess.");
  auto pred =
      (rel == AssessRelation::EQ) ? sbe::oc_eq(lhs, rhs) : sbe::oc_ne(lhs, rhs);
  const auto warning_msg = warn_message.empty() ? error_message : warn_message;

  if (auto b = VIBool(pred)) {
    if (b.value() == false) {
      switch (ap) {
      case AssessPolicy::Error:
      case AssessPolicy::ErrWarn:
        visitor->Error1(l, error_message);
        return {false, false, false};
      case AssessPolicy::Warn:
        visitor->Warning(l, warning_msg);
        return {true, true, false};
      }
    }
    return {true, false, false};
  }

  bool strict_fail = false;
  bool may_fail = false;
  if (rel == AssessRelation::EQ) {
    strict_fail = sbe::must_ne(lhs, rhs);
    may_fail = sbe::may_ne(lhs, rhs);
  } else {
    strict_fail = sbe::must_eq(lhs, rhs);
    may_fail = sbe::may_eq(lhs, rhs);
  }

  switch (ap) {
  case AssessPolicy::Error:
    if (strict_fail) {
      visitor->Error1(l, error_message);
      return {false, false, false};
    }
    break;
  case AssessPolicy::Warn:
    if (strict_fail || may_fail) visitor->Warning(l, warning_msg);
    return {true, strict_fail || may_fail, false};
  case AssessPolicy::ErrWarn:
    if (strict_fail) {
      visitor->Error1(l, error_message);
      return {false, false, false};
    }
    if (may_fail) visitor->Warning(l, warning_msg);
    break;
  }

  AddAssertion(pred, l, error_message, aty, node);
  return {true, may_fail, true};
}

AssessResult Assessor::Assess(AssessPolicy ap, AssessRelation rel,
                              const ValueItem& lhs, const ValueItem& rhs,
                              const std::string& message, AssessType aty,
                              const location& l, AST::Node* node) {
  return Assess(ap, rel, lhs, rhs, message, message, aty, l, node);
}

AssessResult Assessor::Assess(AssessPolicy ap, const ValueItem& bo,
                              const std::string& message, AssessType aty,
                              const location& l, AST::Node* node,
                              AST::Node* emit_node, const ValueItem& guard,
                              AST::Node* guard_site,
                              AssertionEmitPosition guard_site_position) {
  if (DebugOn())
    dbgs() << "[Assess] " << STR(bo) << ", type: " << STR(aty)
           << ", policy: " << STR(ap) << ", node: " << PSTR(emit_node)
           << ", node: " << PSTR(emit_node) << "\n";
  assert(visitor && "Visitor not bound. Call Bind() before Assess.");

  if (ap == AssessPolicy::ErrWarn)
    choreo_unreachable(
        "ErrWarn policy is not allowed for boolean-expression assessment "
        "insertion.");

  auto pred = bo;
  if (pred) pred = pred->Normalize();

  auto norm_guard = guard;
  if (norm_guard) norm_guard = norm_guard->Normalize();
  if (auto gb = VIBool(norm_guard)) {
    if (gb.value() == false) return {true, false, false};
    norm_guard = GetInvalidValueItem();
  }

  if (auto b = VIBool(pred)) {
    if (b.value() == false) {
      if (IsValidValueItem(norm_guard)) {
        if (ap == AssessPolicy::Warn) return {true, false, false};
        AddAssertion(pred, l, message, aty, node, emit_node, guard_site,
                     guard_site_position);
        return {true, false, true};
      }
      if (ap == AssessPolicy::Error)
        visitor->Error1(l, message);
      else
        visitor->Warning(l, message);
      return {ap == AssessPolicy::Warn, ap == AssessPolicy::Warn, false};
    }
    return {true, false, false};
  }

  if (ap == AssessPolicy::Warn) return {true, false, false};

  AddAssertion(pred, l, message, aty, node, emit_node, guard_site,
               guard_site_position);
  return {true, false, true};
}
