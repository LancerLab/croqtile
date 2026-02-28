#include "assess.hpp"

#include "visitor.hpp"

using namespace Choreo;

AssessResult Assessor::Assess(
    AssessPolicy ap, AssessRelation rel, const ValueItem& lhs,
    const ValueItem& rhs, const std::string& error_message,
    const std::string& warn_message, AssessType aty, const location& l,
    AST::Node* node) {
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

AssessResult Assessor::Assess(
    AssessPolicy ap, AssessRelation rel, const ValueItem& lhs,
    const ValueItem& rhs, const std::string& message, AssessType aty,
    const location& l, AST::Node* node) {
  return Assess(ap, rel, lhs, rhs, message, message, aty, l, node);
}

AssessResult Assessor::Assess(
    AssessPolicy ap, const ValueItem& bo,
    const std::string& message, AssessType aty, const location& l,
    AST::Node* node) {
  assert(visitor && "Visitor not bound. Call Bind() before Assess.");
  if (ap == AssessPolicy::ErrWarn)
    choreo_unreachable(
        "ErrWarn policy is not allowed for boolean-expression assessment "
        "insertion.");

  auto pred = bo;
  if (pred) pred = pred->Normalize();

  if (auto b = VIBool(pred)) {
    if (b.value() == false) {
      if (ap == AssessPolicy::Error)
        visitor->Error1(l, message);
      else
        visitor->Warning(l, message);
      return {ap == AssessPolicy::Warn, ap == AssessPolicy::Warn, false};
    }
    return {true, false, false};
  }

  if (ap == AssessPolicy::Warn) return {true, false, false};

  AddAssertion(pred, l, message, aty, node);
  return {true, false, true};
}
