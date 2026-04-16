#ifndef __CHOREO_INTERVAL_HPP__
#define __CHOREO_INTERVAL_HPP__

#include "symbexpr.hpp"
#include "symvals.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Interval analysis library for the Choreo SBE engine.
//
// Provides half-open integer intervals [lo, hi) over int64_t with optional
// unbounded ends (nullopt = +/-infinity). An IntervalSet is a sorted,
// non-overlapping union of Intervals suitable for disjunctive constraints
// like (p < 3 || p >= 5).
//
// Constraint projection extracts the IntervalSet a specific symbolic
// variable must satisfy given an SBE boolean predicate built from scope
// guards and type bounds.

namespace Choreo {
namespace sbe {

// Runtime statistics for interval analysis.
struct IntervalStats {
  uint64_t project_calls = 0;
  uint64_t eval_pred_calls = 0;
  uint64_t proven_true = 0;
  uint64_t proven_false = 0;
  uint64_t unknown = 0;

  void Reset() { *this = {}; }
};

class IntervalProfiler {
public:
  IntervalStats stats;

  static IntervalProfiler& Get() {
    static IntervalProfiler profiler;
    return profiler;
  }

  void Reset() { stats.Reset(); }
  IntervalStats Snapshot() const { return stats; }
};

#define CHOREO_INTERVAL_STATS_INC(field)                                       \
  ++::Choreo::sbe::IntervalProfiler::Get().stats.field

// Half-open interval [lo, hi).  nullopt means unbounded in that direction.
struct Interval {
  std::optional<int64_t> lo; // inclusive lower bound
  std::optional<int64_t> hi; // exclusive upper bound

  static Interval Empty() { return {0, 0}; }
  static Interval Universe() { return {std::nullopt, std::nullopt}; }
  static Interval Point(int64_t v) { return {v, v + 1}; }
  static Interval HalfOpenBelow(int64_t hi) { return {std::nullopt, hi}; }
  static Interval HalfOpenAbove(int64_t lo) { return {lo, std::nullopt}; }
  static Interval HalfOpen(int64_t lo, int64_t hi) { return {lo, hi}; }

  bool IsEmpty() const {
    return lo.has_value() && hi.has_value() && *lo >= *hi;
  }

  bool IsUniverse() const { return !lo.has_value() && !hi.has_value(); }

  bool IsSingleton() const {
    return lo.has_value() && hi.has_value() && *hi == *lo + 1;
  }

  bool Contains(int64_t v) const {
    if (lo.has_value() && v < *lo) return false;
    if (hi.has_value() && v >= *hi) return false;
    return true;
  }

  bool operator==(const Interval& o) const {
    if (IsEmpty() && o.IsEmpty()) return true;
    return lo == o.lo && hi == o.hi;
  }
  bool operator!=(const Interval& o) const { return !(*this == o); }

  std::string ToString() const {
    if (IsEmpty()) return "{}";
    std::string s;
    s += lo.has_value() ? "[" + std::to_string(*lo) : "(-inf";
    s += ", ";
    s += hi.has_value() ? std::to_string(*hi) + ")" : "+inf)";
    return s;
  }
};

// Intersection of two single intervals.
inline Interval Intersect(const Interval& a, const Interval& b) {
  std::optional<int64_t> lo;
  if (a.lo.has_value() && b.lo.has_value())
    lo = std::max(*a.lo, *b.lo);
  else if (a.lo.has_value())
    lo = a.lo;
  else
    lo = b.lo;

  std::optional<int64_t> hi;
  if (a.hi.has_value() && b.hi.has_value())
    hi = std::min(*a.hi, *b.hi);
  else if (a.hi.has_value())
    hi = a.hi;
  else
    hi = b.hi;

  Interval r{lo, hi};
  if (r.IsEmpty()) return Interval::Empty();
  return r;
}

// Sorted, non-overlapping union of half-open intervals.
class IntervalSet {
  std::vector<Interval> ivs_;

  void Normalize() {
    ivs_.erase(std::remove_if(ivs_.begin(), ivs_.end(),
                              [](const Interval& i) { return i.IsEmpty(); }),
               ivs_.end());
    if (ivs_.empty()) return;

    std::sort(ivs_.begin(), ivs_.end(),
              [](const Interval& a, const Interval& b) {
                if (!a.lo.has_value() && b.lo.has_value()) return true;
                if (a.lo.has_value() && !b.lo.has_value()) return false;
                if (!a.lo.has_value() && !b.lo.has_value()) return false;
                return *a.lo < *b.lo;
              });

    std::vector<Interval> merged;
    merged.push_back(ivs_[0]);
    for (size_t i = 1; i < ivs_.size(); ++i) {
      auto& top = merged.back();
      auto& cur = ivs_[i];

      bool can_merge = false;
      if (!top.hi.has_value()) {
        can_merge = true;
      } else if (cur.lo.has_value()) {
        can_merge = (*cur.lo <= *top.hi);
      } else {
        can_merge = true;
      }

      if (can_merge) {
        if (!top.hi.has_value() || !cur.hi.has_value())
          top.hi = std::nullopt;
        else
          top.hi = std::max(*top.hi, *cur.hi);
        if (!top.lo.has_value() || !cur.lo.has_value()) top.lo = std::nullopt;
      } else {
        merged.push_back(cur);
      }
    }
    ivs_ = std::move(merged);
  }

public:
  IntervalSet() = default;
  explicit IntervalSet(const Interval& iv) {
    if (!iv.IsEmpty()) ivs_.push_back(iv);
  }
  explicit IntervalSet(std::vector<Interval> ivs) : ivs_(std::move(ivs)) {
    Normalize();
  }

  static IntervalSet MakeEmpty() { return IntervalSet(); }
  static IntervalSet MakeUniverse() {
    return IntervalSet(Interval::Universe());
  }
  static IntervalSet MakePoint(int64_t v) {
    return IntervalSet(Interval::Point(v));
  }

  bool IsEmpty() const { return ivs_.empty(); }
  bool IsUniverse() const { return ivs_.size() == 1 && ivs_[0].IsUniverse(); }

  bool IsSingleton() const { return ivs_.size() == 1 && ivs_[0].IsSingleton(); }

  std::optional<int64_t> SingletonValue() const {
    if (!IsSingleton()) return std::nullopt;
    return ivs_[0].lo;
  }

  const std::vector<Interval>& Intervals() const { return ivs_; }

  std::optional<int64_t> LowerBound() const {
    if (ivs_.empty()) return std::nullopt;
    return ivs_.front().lo;
  }

  std::optional<int64_t> UpperBound() const {
    if (ivs_.empty()) return std::nullopt;
    return ivs_.back().hi;
  }

  bool ProvablyLT(int64_t val) const {
    if (ivs_.empty()) return true;
    auto ub = UpperBound();
    return ub.has_value() && *ub <= val;
  }

  bool ProvablyLE(int64_t val) const {
    if (ivs_.empty()) return true;
    auto ub = UpperBound();
    return ub.has_value() && *ub <= val + 1;
  }

  bool ProvablyGE(int64_t val) const {
    if (ivs_.empty()) return true;
    auto lb = LowerBound();
    return lb.has_value() && *lb >= val;
  }

  bool ProvablyGT(int64_t val) const {
    if (ivs_.empty()) return true;
    auto lb = LowerBound();
    return lb.has_value() && *lb > val;
  }

  bool Contains(int64_t v) const {
    for (auto& iv : ivs_)
      if (iv.Contains(v)) return true;
    return false;
  }

  bool IsSubsetOf(const IntervalSet& other) const {
    auto diff = Intersect(*this, Complement(other));
    return diff.IsEmpty();
  }

  bool operator==(const IntervalSet& o) const { return ivs_ == o.ivs_; }
  bool operator!=(const IntervalSet& o) const { return !(*this == o); }

  std::string ToString() const {
    if (ivs_.empty()) return "{}";
    std::string s;
    for (size_t i = 0; i < ivs_.size(); ++i) {
      if (i > 0) s += " U ";
      s += ivs_[i].ToString();
    }
    return s;
  }

  // ---- Set operations (friend functions) ----

  friend IntervalSet Intersect(const IntervalSet& a, const IntervalSet& b) {
    if (a.IsEmpty() || b.IsEmpty()) return IntervalSet::MakeEmpty();

    std::vector<Interval> result;
    size_t i = 0, j = 0;
    while (i < a.ivs_.size() && j < b.ivs_.size()) {
      auto iv = Choreo::sbe::Intersect(a.ivs_[i], b.ivs_[j]);
      if (!iv.IsEmpty()) result.push_back(iv);

      bool advance_a;
      if (!a.ivs_[i].hi.has_value())
        advance_a = false;
      else if (!b.ivs_[j].hi.has_value())
        advance_a = true;
      else
        advance_a = *a.ivs_[i].hi <= *b.ivs_[j].hi;

      if (advance_a)
        ++i;
      else
        ++j;
    }
    return IntervalSet(std::move(result));
  }

  friend IntervalSet Unite(const IntervalSet& a, const IntervalSet& b) {
    std::vector<Interval> all;
    all.insert(all.end(), a.ivs_.begin(), a.ivs_.end());
    all.insert(all.end(), b.ivs_.begin(), b.ivs_.end());
    return IntervalSet(std::move(all));
  }

  friend IntervalSet Complement(const IntervalSet& s) {
    if (s.IsEmpty()) return IntervalSet::MakeUniverse();
    if (s.IsUniverse()) return IntervalSet::MakeEmpty();

    std::vector<Interval> result;
    std::optional<int64_t> prev_hi = std::nullopt;

    for (auto& iv : s.ivs_) {
      if (iv.lo.has_value()) {
        Interval gap{prev_hi, iv.lo};
        if (!gap.IsEmpty()) result.push_back(gap);
      }
      prev_hi = iv.hi;
    }

    if (prev_hi.has_value()) result.push_back(Interval{prev_hi, std::nullopt});

    return IntervalSet(std::move(result));
  }
};

// ---- Interval arithmetic on IntervalSet ----

inline IntervalSet IntervalAdd(const IntervalSet& a, const IntervalSet& b) {
  if (a.IsEmpty() || b.IsEmpty()) return IntervalSet::MakeEmpty();
  std::vector<Interval> result;
  for (auto& ai : a.Intervals()) {
    for (auto& bi : b.Intervals()) {
      std::optional<int64_t> lo, hi;
      if (ai.lo.has_value() && bi.lo.has_value()) lo = *ai.lo + *bi.lo;
      if (ai.hi.has_value() && bi.hi.has_value()) hi = *ai.hi + *bi.hi - 1;
      result.push_back({lo, hi});
    }
  }
  return IntervalSet(std::move(result));
}

inline IntervalSet IntervalSub(const IntervalSet& a, const IntervalSet& b) {
  if (a.IsEmpty() || b.IsEmpty()) return IntervalSet::MakeEmpty();
  std::vector<Interval> result;
  for (auto& ai : a.Intervals()) {
    for (auto& bi : b.Intervals()) {
      std::optional<int64_t> lo, hi;
      if (ai.lo.has_value() && bi.hi.has_value()) lo = *ai.lo - *bi.hi + 1;
      if (ai.hi.has_value() && bi.lo.has_value()) hi = *ai.hi - *bi.lo;
      result.push_back({lo, hi});
    }
  }
  return IntervalSet(std::move(result));
}

inline IntervalSet IntervalNeg(const IntervalSet& a) {
  if (a.IsEmpty()) return IntervalSet::MakeEmpty();
  std::vector<Interval> result;
  for (auto& ai : a.Intervals()) {
    std::optional<int64_t> lo, hi;
    if (ai.hi.has_value()) lo = -(*ai.hi) + 1;
    if (ai.lo.has_value()) hi = -(*ai.lo) + 1;
    result.push_back({lo, hi});
  }
  return IntervalSet(std::move(result));
}

inline IntervalSet IntervalMul(const IntervalSet& a, const IntervalSet& b) {
  if (a.IsEmpty() || b.IsEmpty()) return IntervalSet::MakeEmpty();

  if (!a.ProvablyGE(0) || !b.ProvablyGE(0)) return IntervalSet::MakeUniverse();

  std::vector<Interval> result;
  for (auto& ai : a.Intervals()) {
    for (auto& bi : b.Intervals()) {
      std::optional<int64_t> lo, hi;
      if (ai.lo.has_value() && bi.lo.has_value()) lo = *ai.lo * *bi.lo;
      if (ai.hi.has_value() && bi.hi.has_value())
        hi = (*ai.hi - 1) * (*bi.hi - 1) + 1;
      result.push_back({lo, hi});
    }
  }
  return IntervalSet(std::move(result));
}

// ---- Constraint projection ----
//
// Given an SBE boolean predicate and a target symbol name, extract the
// IntervalSet that the symbol must lie within for the predicate to hold.
//
// Handles: comparisons, AND, OR, NOT, and simple linear isolation
// (sym + k < c  =>  sym < c-k).

using ConstraintEnv = std::unordered_map<std::string, IntervalSet>;

namespace detail {

struct LinearDecomp {
  bool valid = false;
  int64_t offset = 0;
};

// Decompose an expression as (symbol + constant_offset).
inline LinearDecomp DecomposeLinear(const Operand& expr,
                                    const std::string& sym) {
  if (!expr) return {};
  auto norm = expr->Normalize();

  if (auto sv = dyn_cast<SymbolicValue>(norm)) {
    if (sv->Value() == sym) return {true, 0};
    return {};
  }

  if (auto bop = dyn_cast<BinaryOperation>(norm)) {
    auto lhs = bop->GetLeft();
    auto rhs = bop->GetRight();
    auto op = bop->GetOpCode();

    if (op == OpCode::ADD) {
      if (auto rv = VIInt(rhs)) {
        auto inner = DecomposeLinear(lhs, sym);
        if (inner.valid) return {true, inner.offset + *rv};
      }
      if (auto lv = VIInt(lhs)) {
        auto inner = DecomposeLinear(rhs, sym);
        if (inner.valid) return {true, inner.offset + *lv};
      }
    } else if (op == OpCode::SUBTRACT) {
      if (auto rv = VIInt(rhs)) {
        auto inner = DecomposeLinear(lhs, sym);
        if (inner.valid) return {true, inner.offset - *rv};
      }
    }
  }
  return {};
}

// Extract the interval for `sym` from a single comparison predicate.
inline IntervalSet ProjectAtom(const Operand& pred, const std::string& sym,
                               const ConstraintEnv& env) {
  if (!pred) return IntervalSet::MakeUniverse();

  auto norm = pred->Normalize();

  auto bop = dyn_cast<BinaryOperation>(norm);
  if (!bop) return IntervalSet::MakeUniverse();

  auto op = bop->GetOpCode();
  auto lhs = bop->GetLeft();
  auto rhs = bop->GetRight();

  auto TryProject = [&](const Operand& expr_side, const Operand& bound_side,
                        bool sym_on_left) -> IntervalSet {
    auto ld = DecomposeLinear(expr_side, sym);
    if (!ld.valid) return IntervalSet::MakeUniverse();

    std::optional<int64_t> bound_val;
    if (auto nv = VIInt(bound_side->Normalize())) {
      bound_val = *nv;
    } else if (auto sv = dyn_cast<SymbolicValue>(bound_side->Normalize())) {
      auto it = env.find(sv->Value());
      if (it != env.end()) {
        if (sym_on_left)
          bound_val = it->second.UpperBound();
        else
          bound_val = it->second.LowerBound();
      }
    }

    if (!bound_val.has_value()) return IntervalSet::MakeUniverse();

    int64_t bv = *bound_val;
    int64_t off = ld.offset;

    auto effective_op = op;
    if (!sym_on_left) {
      switch (op) {
      case OpCode::LT: effective_op = OpCode::GT; break;
      case OpCode::LE: effective_op = OpCode::GE; break;
      case OpCode::GT: effective_op = OpCode::LT; break;
      case OpCode::GE: effective_op = OpCode::LE; break;
      default: break;
      }
    }

    int64_t adjusted = bv - off;
    switch (effective_op) {
    case OpCode::LT: return IntervalSet(Interval::HalfOpenBelow(adjusted));
    case OpCode::LE: return IntervalSet(Interval::HalfOpenBelow(adjusted + 1));
    case OpCode::GT: return IntervalSet(Interval::HalfOpenAbove(adjusted + 1));
    case OpCode::GE: return IntervalSet(Interval::HalfOpenAbove(adjusted));
    case OpCode::EQ: return IntervalSet::MakePoint(adjusted);
    case OpCode::NE: return Complement(IntervalSet::MakePoint(adjusted));
    default: return IntervalSet::MakeUniverse();
    }
  };

  if (IsCompare(op)) {
    auto r1 = TryProject(lhs, rhs, true);
    if (!r1.IsUniverse()) return r1;
    auto r2 = TryProject(rhs, lhs, false);
    return r2;
  }

  return IntervalSet::MakeUniverse();
}

} // namespace detail

// Project a boolean SBE predicate onto a single symbolic variable,
// returning the IntervalSet that the variable must belong to.
inline IntervalSet ProjectConstraint(const Operand& pred,
                                     const std::string& sym,
                                     const ConstraintEnv& env = {}) {
  CHOREO_INTERVAL_STATS_INC(project_calls);

  if (!pred) return IntervalSet::MakeUniverse();
  auto norm = pred->Normalize();

  if (auto bv = dyn_cast<BooleanValue>(norm))
    return bv->Value() ? IntervalSet::MakeUniverse() : IntervalSet::MakeEmpty();

  auto bop = dyn_cast<BinaryOperation>(norm);
  if (!bop) {
    auto uop_ptr = dyn_cast<UnaryOperation>(norm);
    if (uop_ptr && uop_ptr->GetOpCode() == OpCode::NOT)
      return Complement(ProjectConstraint(uop_ptr->GetOperand(), sym, env));
    return IntervalSet::MakeUniverse();
  }

  auto op = bop->GetOpCode();

  if (op == OpCode::AND) {
    auto left = ProjectConstraint(bop->GetLeft(), sym, env);
    auto right = ProjectConstraint(bop->GetRight(), sym, env);
    return Intersect(left, right);
  }

  if (op == OpCode::OR) {
    auto left = ProjectConstraint(bop->GetLeft(), sym, env);
    auto right = ProjectConstraint(bop->GetRight(), sym, env);
    return Unite(left, right);
  }

  if (IsCompare(op)) return detail::ProjectAtom(pred, sym, env);

  return IntervalSet::MakeUniverse();
}

// Evaluate an SBE expression over intervals, returning the possible range.
inline IntervalSet EvalExprInterval(const Operand& expr,
                                    const ConstraintEnv& env) {
  if (!expr) return IntervalSet::MakeUniverse();
  auto norm = expr->Normalize();

  if (auto nv = dyn_cast<NumericValue>(norm))
    return IntervalSet::MakePoint(nv->Value());

  if (auto sv = dyn_cast<SymbolicValue>(norm)) {
    auto it = env.find(sv->Value());
    if (it != env.end()) return it->second;
    return IntervalSet::MakeUniverse();
  }

  if (auto bop = dyn_cast<BinaryOperation>(norm)) {
    auto lhs = EvalExprInterval(bop->GetLeft(), env);
    auto rhs = EvalExprInterval(bop->GetRight(), env);
    switch (bop->GetOpCode()) {
    case OpCode::ADD: return IntervalAdd(lhs, rhs);
    case OpCode::SUBTRACT: return IntervalSub(lhs, rhs);
    case OpCode::MULTIPLY: return IntervalMul(lhs, rhs);
    default: break;
    }
  }

  if (auto uop = dyn_cast<UnaryOperation>(norm)) {
    if (uop->GetOpCode() == OpCode::SUBTRACT)
      return IntervalNeg(EvalExprInterval(uop->GetOperand(), env));
  }

  return IntervalSet::MakeUniverse();
}

// Evaluate a boolean comparison predicate over interval environments.
// Returns: true if provably true, false if provably false, nullopt if unknown.
inline std::optional<bool> EvalPredInterval(const Operand& pred,
                                            const ConstraintEnv& env) {
  CHOREO_INTERVAL_STATS_INC(eval_pred_calls);

  if (!pred) return std::nullopt;
  auto norm = pred->Normalize();

  if (auto bv = dyn_cast<BooleanValue>(norm)) return bv->Value();

  if (auto uop = dyn_cast<UnaryOperation>(norm)) {
    if (uop->GetOpCode() == OpCode::NOT) {
      auto inner = EvalPredInterval(uop->GetOperand(), env);
      if (inner.has_value()) return !*inner;
      return std::nullopt;
    }
  }

  auto bop = dyn_cast<BinaryOperation>(norm);
  if (!bop) return std::nullopt;

  auto op = bop->GetOpCode();

  if (op == OpCode::AND) {
    auto l = EvalPredInterval(bop->GetLeft(), env);
    auto r = EvalPredInterval(bop->GetRight(), env);
    if (l.has_value() && *l == false) return false;
    if (r.has_value() && *r == false) return false;
    if (l.has_value() && r.has_value() && *l && *r) return true;
    return std::nullopt;
  }

  if (op == OpCode::OR) {
    auto l = EvalPredInterval(bop->GetLeft(), env);
    auto r = EvalPredInterval(bop->GetRight(), env);
    if (l.has_value() && *l == true) return true;
    if (r.has_value() && *r == true) return true;
    if (l.has_value() && r.has_value() && !*l && !*r) return false;
    return std::nullopt;
  }

  if (!IsCompare(op)) return std::nullopt;

  auto lhs = EvalExprInterval(bop->GetLeft(), env);
  auto rhs = EvalExprInterval(bop->GetRight(), env);

  if (lhs.IsEmpty() || rhs.IsEmpty()) return std::nullopt;

  // Half-open [a, b) over integers: min_val = a, max_val = b - 1.
  //   LT  true:  max(L) < min(R)  <=>  b <= c
  //   LT  false: min(L) >= max(R) <=>  a+1 >= d
  //   LE  true:  max(L) <= min(R) <=>  b <= c+1
  //   LE  false: min(L) > max(R)  <=>  a >= d
  //   GT  true:  min(L) > max(R)  <=>  a >= d
  //   GT  false: max(L) <= min(R) <=>  b <= c+1
  //   GE  true:  min(L) >= max(R) <=>  a+1 >= d
  //   GE  false: max(L) < min(R)  <=>  b <= c
  std::optional<bool> result;
  switch (op) {
  case OpCode::LT:
    if (lhs.UpperBound().has_value() && rhs.LowerBound().has_value() &&
        *lhs.UpperBound() <= *rhs.LowerBound())
      result = true;
    else if (lhs.LowerBound().has_value() && rhs.UpperBound().has_value() &&
             *lhs.LowerBound() + 1 >= *rhs.UpperBound())
      result = false;
    break;
  case OpCode::LE:
    if (lhs.UpperBound().has_value() && rhs.LowerBound().has_value() &&
        *lhs.UpperBound() <= *rhs.LowerBound() + 1)
      result = true;
    else if (lhs.LowerBound().has_value() && rhs.UpperBound().has_value() &&
             *lhs.LowerBound() >= *rhs.UpperBound())
      result = false;
    break;
  case OpCode::GT:
    if (lhs.LowerBound().has_value() && rhs.UpperBound().has_value() &&
        *lhs.LowerBound() >= *rhs.UpperBound())
      result = true;
    else if (lhs.UpperBound().has_value() && rhs.LowerBound().has_value() &&
             *lhs.UpperBound() <= *rhs.LowerBound() + 1)
      result = false;
    break;
  case OpCode::GE:
    if (lhs.LowerBound().has_value() && rhs.UpperBound().has_value() &&
        *lhs.LowerBound() + 1 >= *rhs.UpperBound())
      result = true;
    else if (lhs.UpperBound().has_value() && rhs.LowerBound().has_value() &&
             *lhs.UpperBound() <= *rhs.LowerBound())
      result = false;
    break;
  case OpCode::EQ:
    if (lhs.IsSingleton() && rhs.IsSingleton() &&
        *lhs.SingletonValue() == *rhs.SingletonValue())
      result = true;
    else if (Intersect(lhs, rhs).IsEmpty())
      result = false;
    break;
  case OpCode::NE:
    if (Intersect(lhs, rhs).IsEmpty())
      result = true;
    else if (lhs.IsSingleton() && rhs.IsSingleton() &&
             *lhs.SingletonValue() == *rhs.SingletonValue())
      result = false;
    break;
  default: break;
  }

  if (result.has_value()) {
    if (*result)
      CHOREO_INTERVAL_STATS_INC(proven_true);
    else
      CHOREO_INTERVAL_STATS_INC(proven_false);
  } else {
    CHOREO_INTERVAL_STATS_INC(unknown);
  }
  return result;
}

} // namespace sbe
} // namespace Choreo

#endif // __CHOREO_INTERVAL_HPP__
