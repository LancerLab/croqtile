#ifndef __CHOREO_SYMBOLIC_EXPRESSION_HPP__
#define __CHOREO_SYMBOLIC_EXPRESSION_HPP__

#include "aux.hpp"
#include "utils.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// This is a light-weight integer library that support both constant and
// symbolic values. It applies:
//
//  - folding,
//  - reordering (commutative),
//  - reassociation (associative).
//
// For example,
//
//    (1 + (3 * a) * (c * 2)) + 4;
// -> ((a * 3) * (c * 2) + 1) + 4;     (reorder)
// -> (a * ((3 * c) * 2)) + (1 + 4);   (reassociate)
// -> (a * ((c * 3) * 2)) + 5;         (reorder & fold)
// -> (a * (c * (3 * 2))) + 5;         (reassociate)
// -> (a * (c * 6)) + 5;               (reorder & fold)
//
// the last expression is named as the "normalized" symbolic expression, which
// can be used for expression comparison.
//
// Limitation: risk of overflow values
//
// Note: special thanks to deepseek for initiating the code

namespace Choreo {

// Supported operation types
enum class OpCode {
  NONE,
  /* arithmetic */
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  IRES,
  POWER,
  /* comparison */
  GT,
  LT,
  EQ,
  GE,
  LE,
  NE,
  /* ternary */
  SELECT,
};

inline static bool IsArith(OpCode op) {
  switch (op) {
  case OpCode::ADD:
  case OpCode::SUBTRACT:
  case OpCode::MULTIPLY:
  case OpCode::DIVIDE:
  case OpCode::IRES:
  case OpCode::POWER: return true;
  default: break;
  }
  return false;
}

inline static bool IsCompare(OpCode op) {
  switch (op) {
  case OpCode::GT:
  case OpCode::LT:
  case OpCode::EQ:
  case OpCode::GE:
  case OpCode::LE:
  case OpCode::NE: return true;
  default: break;
  }
  return false;
}

inline static std::string STR(OpCode tc) {
  switch (tc) {
  case OpCode::ADD: return "+";
  case OpCode::SUBTRACT: return "-";
  case OpCode::MULTIPLY: return "*";
  case OpCode::DIVIDE: return "/";
  case OpCode::IRES: return "%";
  case OpCode::POWER: return "^";
  case OpCode::GT: return ">";
  case OpCode::LT: return "<";
  case OpCode::EQ: return "==";
  case OpCode::GE: return ">=";
  case OpCode::LE: return "<=";
  case OpCode::NE: return "!=";
  case OpCode::SELECT: return "?";
  default: choreo_unreachable("unsupported opcode");
  }
  return "";
}

inline static OpCode ToOpCode(const std::string& op) {
  if (op == "+")
    return OpCode::ADD;
  else if (op == "-")
    return OpCode::SUBTRACT;
  else if (op == "*")
    return OpCode::MULTIPLY;
  else if (op == "/")
    return OpCode::DIVIDE;
  else if (op == "%")
    return OpCode::IRES;
  else
    choreo_unreachable("operation '" + op + "' is not supported.");
  return OpCode::NONE;
}

namespace sbe {

inline static int64_t gcd(int64_t a, int64_t b) {
  while (b != 0) {
    int64_t temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

inline static bool multipleof(int64_t a, int64_t b) { return gcd(a, b) == b; }

// Note: Same symbol names implies same value. Therefore scoped symbols are
// required.

// Forward declarations
class SymbolicExpression;
class NumericValue;
class SymbolicValue;
class BinaryOperation;

// Operand type can be either a numeric value, symbolic value, or another
// SymbolicExpression
using Operand = ptr<SymbolicExpression>;

// forward decls
inline Operand operator+(const Operand&, const Operand&);
inline Operand operator-(const Operand&, const Operand&);
inline Operand operator*(const Operand&, const Operand&);
inline Operand operator/(const Operand&, const Operand&);
inline Operand operator%(const Operand&, const Operand&);
// Operands comparison
// Note: Operand are partially ordered. Use it carefully.
//       - oc_lt(a, b) == false, it does not always mean 'a >= b',
//       - oc_lt(a, b) == true, 'a' must be strictly less than 'b'.
inline Operand oc_lt(const Operand&, const Operand&);
inline Operand oc_gt(const Operand&, const Operand&);
inline Operand oc_le(const Operand&, const Operand&);
inline Operand oc_ge(const Operand&, const Operand&);
inline Operand oc_eq(const Operand&, const Operand&);
inline Operand oc_ne(const Operand&, const Operand&);
inline int Compare(const SymbolicExpression&, const SymbolicExpression&);
inline bool operator<(const SymbolicExpression&, const SymbolicExpression&);
inline Operand nu(int64_t);
inline Operand bl(bool);
inline Operand sym(const std::string& name);
inline Operand bop(OpCode, const Operand&, const Operand&);
inline Operand sel(const Operand&, const Operand&, const Operand&);

inline static int64_t IsCommutative(OpCode op) {
  if (IsArith(op)) {
    switch (op) {
    case OpCode::ADD:
    case OpCode::MULTIPLY: return true;
    case OpCode::SUBTRACT:
    case OpCode::DIVIDE:
    case OpCode::IRES:
    case OpCode::POWER: return false;
    default: choreo_unreachable("unsupported opcode");
    }
  }
  return false;
}

inline static int64_t IsAssociative(OpCode op) {
  if (IsArith(op)) {
    switch (op) {
    case OpCode::ADD:
    case OpCode::MULTIPLY: return true;
    case OpCode::SUBTRACT:
    case OpCode::DIVIDE:
    case OpCode::IRES:
    case OpCode::POWER: return false;
    default: choreo_unreachable("unsupported opcode");
    }
  }
  return false;
}

class SymbolicExpression {
public:
  virtual ~SymbolicExpression() = default;
  virtual std::string ToString() const = 0;
  virtual bool IsNumeric() const = 0;
  virtual size_t Hash() const = 0;
  virtual bool operator==(const SymbolicExpression&) const = 0;
  virtual bool IsLeaf() const = 0;
  virtual void
  Apply(const std::function<void(const SymbolicExpression*)>& func) {
    func(this);
  }

public:
  virtual Operand Clone() const = 0;
  virtual Operand Fold() const = 0;
  virtual Operand Reorder() const = 0;
  virtual Operand Normalize() const = 0;
  virtual Operand Reassociate() const = 0;

public:
  // for runtime type disambiguation
  __UDT_TYPE_INFO_BASE__(notype)
};

inline static std::string STR(const SymbolicExpression& se) {
  return se.ToString();
}

inline static std::string PSTR(SymbolicExpression* pse) {
  return pse->ToString();
}

inline static std::string PSTR(const ptr<SymbolicExpression>& pse) {
  return pse->ToString();
}

class NumericValue : public SymbolicExpression,
                     public TypeIDProvider<NumericValue> {
public:
  NumericValue(int64_t value) : value(value) {}

  std::string ToString() const override { return std::to_string(value); }
  int64_t Value() const { return value; }
  size_t Hash() const override { return std::hash<int64_t>{}(Value()); }

  bool IsNumeric() const override { return true; }

  bool operator==(const SymbolicExpression& op) const override {
    if (auto nv = dyn_cast<NumericValue>(&op)) return nv->value == value;
    return false;
  }

public:
  bool IsLeaf() const override { return true; }
  Operand Clone() const override { return nu(value); };
  Operand Fold() const override { return Clone(); }
  Operand Reorder() const override { return Clone(); };
  Operand Normalize() const override { return Clone(); };
  Operand Reassociate() const override { return Clone(); };

private:
  int64_t value;

public:
  __UDT_TYPE_INFO__(SymbolicExpression, NumericValue)
};

class BooleanValue : public SymbolicExpression,
                     public TypeIDProvider<BooleanValue> {
public:
  BooleanValue(bool value) : value(value) {}

  std::string ToString() const override { return std::to_string(value); }
  bool Value() const { return value; }
  size_t Hash() const override { return std::hash<bool>{}(Value()); }

  bool IsNumeric() const override { return true; }

  bool operator==(const SymbolicExpression& op) const override {
    if (auto nv = dyn_cast<BooleanValue>(&op)) return nv->value == value;
    return false;
  }

public:
  bool IsLeaf() const override { return true; }
  Operand Clone() const override { return bl(value); };
  Operand Fold() const override { return Clone(); }
  Operand Reorder() const override { return Clone(); };
  Operand Normalize() const override { return Clone(); };
  Operand Reassociate() const override { return Clone(); };

private:
  bool value;

public:
  __UDT_TYPE_INFO__(SymbolicExpression, BooleanValue)
};

class SymbolicValue : public SymbolicExpression,
                      public TypeIDProvider<SymbolicValue> {
public:
  SymbolicValue(const std::string& name) : symbol(name) {}

  std::string ToString() const override { return symbol; }

  bool IsNumeric() const override { return false; }

  bool operator==(const SymbolicExpression& op) const override {
    if (auto sv = dyn_cast<SymbolicValue>(&op)) return sv->symbol == symbol;
    return false;
  }

  const std::string Value() const { return symbol; }
  size_t Hash() const override { return std::hash<std::string>{}(Value()); }

public:
  bool IsLeaf() const override { return true; }
  Operand Clone() const override { return sym(symbol); }
  Operand Fold() const override { return Clone(); }
  Operand Reorder() const override { return Clone(); };
  Operand Normalize() const override { return Clone(); };
  Operand Reassociate() const override { return Clone(); };

private:
  std::string symbol;

public:
  __UDT_TYPE_INFO__(SymbolicExpression, SymbolicValue)
};

class BinaryOperation : public SymbolicExpression,
                        public TypeIDProvider<BinaryOperation> {
private:
  OpCode op;
  Operand left;
  Operand right;

public:
  BinaryOperation(OpCode op, const Operand& left, const Operand& right)
      : op(op), left(left), right(right) {}

  std::string ToString() const override {
    return "(" + PSTR(left) + " " + STR(op) + " " + PSTR(right) + ")";
  }

  bool IsNumeric() const override {
    return left->IsNumeric() && right->IsNumeric();
  }

  bool operator==(const SymbolicExpression& expr) const override {
    if (auto se = dyn_cast<BinaryOperation>(&expr))
      if (se->op == op && (*se->left == *left) && (*se->right == *right))
        return true;
    return false;
  }

  size_t Hash() const override { return std::hash<std::string>{}(ToString()); }

public:
  bool IsLeaf() const override { return false; }
  const Operand GetLeft() const { return left; }
  const Operand GetRight() const { return right; }
  OpCode GetOpCode() const { return op; }

  Operand Clone() const override {
    return std::make_shared<BinaryOperation>(op, left->Clone(), right->Clone());
  }

  Operand Fold() const override {
    // handle comparison
    if (IsCompare(op)) {
      if (auto nu = dyn_cast<NumericValue>((left - right)->Normalize())) {
        switch (op) {
        case OpCode::GT:
          if (nu->Value() > 0)
            return bl(true);
          else
            return bl(false);
        case OpCode::LT:
          if (nu->Value() < 0)
            return bl(true);
          else
            return bl(false);
        case OpCode::EQ:
          if (nu->Value() == 0)
            return bl(true);
          else
            return bl(false);
        case OpCode::GE:
          if (nu->Value() >= 0)
            return bl(true);
          else
            return bl(false);
        case OpCode::LE:
          if (nu->Value() <= 0)
            return bl(true);
          else
            return bl(false);
        case OpCode::NE:
          if (nu->Value() <= 0)
            return bl(true);
          else
            return bl(false);
        default: choreo_unreachable("unsupported comparison.");
        }
      }
      // else no simpilification
    }

    auto simplifiedLeft = left->Fold();
    auto simplifiedRight = right->Fold();

    // If both operands are numeric, compute the result
    if (isa<NumericValue>(simplifiedLeft) &&
        isa<NumericValue>(simplifiedRight)) {
      auto lnv = cast<NumericValue>(simplifiedLeft);
      auto rnv = cast<NumericValue>(simplifiedRight);
      int64_t leftVal = lnv->Value();
      int64_t rightVal = rnv->Value();

      switch (op) {
      case OpCode::ADD: return nu(leftVal + rightVal);
      case OpCode::SUBTRACT: return nu(leftVal - rightVal);
      case OpCode::MULTIPLY: return nu(leftVal * rightVal);
      case OpCode::DIVIDE: {
        if (rightVal == 0) choreo_unreachable("Division by zero");
        return nu(leftVal / rightVal);
      }
      case OpCode::IRES:
        if (rightVal == 0) choreo_unreachable("Division by zero");
        return nu(leftVal % rightVal);
      case OpCode::POWER: return nu(std::pow(leftVal, rightVal));
      default: choreo_unreachable("Unknown operation");
      }
    }

    auto lnv = dyn_cast<NumericValue>(simplifiedLeft);
    auto rnv = dyn_cast<NumericValue>(simplifiedRight);
    // Handle special simplification cases
    // x + 0 = x, 0 + x = x
    if (op == OpCode::ADD) {
      if (rnv && (rnv->Value() == 0)) return simplifiedLeft;
      if (lnv && (lnv->Value() == 0)) return simplifiedRight;
    }
    // x - 0 = x
    else if (op == OpCode::SUBTRACT) {
      if (rnv && (rnv->Value() == 0)) return simplifiedLeft;
    }
    // x * 0 = 0, 0 * x = 0
    else if (op == OpCode::MULTIPLY) {
      if ((lnv && (lnv->Value() == 0)) || (rnv && (rnv->Value() == 0)))
        return 0;
      // x * 1 = x, 1 * x = x
      if (rnv && (rnv->Value() == 1)) return simplifiedLeft;
      if (lnv && (lnv->Value() == 1)) return simplifiedRight;
    }
    // x / x = 1, x / 1 = x, 0 / x = 0
    else if (op == OpCode::DIVIDE) {
      if (*simplifiedLeft == *simplifiedRight) return nu(1);
      if (rnv && (rnv->Value() == 1)) return simplifiedLeft;
      if (lnv && (lnv->Value() == 0)) return nu(0);
    }
    // x % x = 0 , x % 1 = 0, 0 % x = 0
    else if (op == OpCode::IRES) {
      if (*simplifiedLeft == *simplifiedRight) return nu(0);
      if (rnv && (rnv->Value() == 1)) return nu(0);
      if (lnv && (lnv->Value() == 0)) return nu(0);
    }
    // x^1 = x, 1^x = 1
    else if (op == OpCode::POWER) {
      if (rnv && (rnv->Value() == 1)) return simplifiedLeft;
      if (lnv && (lnv->Value() == 1)) return nu(1);
    }

    // If no simplification possible, return a new binary operation
    return std::make_shared<BinaryOperation>(op, simplifiedLeft,
                                             simplifiedRight);
  }

  Operand Normalize() const override {
    Operand expr = std::make_shared<BinaryOperation>(op, left, right);
    while (true) {
      auto new_expr = expr->Reorder()->Fold()->Reassociate()->Fold();
      if (*new_expr == *expr)
        return new_expr;
      else
        expr = new_expr;
    }
    choreo_unreachable("unexpected flow.");
    return nullptr;
  }

  Operand Reassociate() const override {
    if (left->IsLeaf() && right->IsLeaf()) return Clone();

    auto new_left = left->Reassociate();
    auto new_right = right->Reassociate();
    auto new_bin = std::make_shared<BinaryOperation>(op, new_left, new_right);
    if (!IsAssociative(op)) return new_bin;

    // find right-most and its parent
    auto RightMostOfLeft = [this](const ptr<BinaryOperation>& n, OpCode opc)
        -> std::tuple<std::shared_ptr<BinaryOperation>,
                      std::shared_ptr<BinaryOperation>, Operand> {
      if (n->left->IsLeaf()) return {nullptr, nullptr, nullptr};
      auto nleft = cast<BinaryOperation>(n->left);
      if (nleft->op != opc) return {nullptr, nullptr, nullptr};

      // right-most and its parent and grandparent
      ptr<BinaryOperation> pp_rmost = n;
      ptr<BinaryOperation> p_rmost = nleft;
      ptr<BinaryOperation> rmost = dyn_cast<BinaryOperation>(p_rmost->right);
      while (rmost && (rmost->op == opc)) { // step down
        pp_rmost = p_rmost;
        p_rmost = rmost;
        rmost = dyn_cast<BinaryOperation>(p_rmost->right);
      }
      return {pp_rmost, p_rmost, p_rmost->right};
    };

    // left-most and its parent and grandparent
    auto LeftMostOfRight = [](const ptr<BinaryOperation>& n, OpCode opc)
        -> std::tuple<std::shared_ptr<BinaryOperation>,
                      std::shared_ptr<BinaryOperation>, Operand> {
      if (n->right->IsLeaf()) return {nullptr, nullptr, nullptr};
      auto nright = cast<BinaryOperation>(n->right);
      if (nright->op != opc) return {nullptr, nullptr, nullptr};

      ptr<BinaryOperation> pp_lmost = n;
      ptr<BinaryOperation> p_lmost = nright;
      ptr<BinaryOperation> lmost = dyn_cast<BinaryOperation>(p_lmost->left);
      while (lmost && (lmost->op == opc)) { // step down
        pp_lmost = p_lmost;
        p_lmost = lmost;
        lmost = dyn_cast<BinaryOperation>(p_lmost->left);
      }
      return {pp_lmost, p_lmost, p_lmost->left};
    };

    if (new_left->IsLeaf()) {
      auto [pp, p, l] = LeftMostOfRight(new_bin, op);
      if (pp && p && l) {
        if ((isa<NumericValue>(new_left) && isa<NumericValue>(l)) ||
            (*new_left < *l)) {
          new_bin->left =
              std::make_shared<BinaryOperation>(op, new_left, l)->Fold();
          if (pp == new_bin)
            pp->right = p->right;
          else
            pp->left = p->right; // hoist the node
        }
      }
    } else if (new_right->IsLeaf()) {
      auto [pp, p, r] = RightMostOfLeft(new_bin, op);
      if (pp && p && r) {
        if ((isa<NumericValue>(r) && isa<NumericValue>(new_right)) ||
            (*r < *new_right)) {
          new_bin->right =
              std::make_shared<BinaryOperation>(op, r, new_right)->Fold();
          if (pp == new_bin)
            new_bin->left = p->left;
          else
            pp->right = p->left; // hoist the node
        }
      }
    } else {
      auto [ppl, pl, l] = LeftMostOfRight(new_bin, op);
      auto [ppr, pr, r] = RightMostOfLeft(new_bin, op);
      if (ppl && pl && l && ppr && pr && r) {
        if ((isa<NumericValue>(r) && isa<NumericValue>(l)) || (*r < *l)) {
          pl->left = std::make_shared<BinaryOperation>(op, r, l)
                         ->Fold(); // move to the right branch
          if (ppr == new_bin)
            ppr->left = pr->left;
          else
            ppr->right = pr->left;
        }
      }
    }

    return new_bin;
  }

  Operand Reorder() const override {
    auto l = left->Reorder();
    auto r = right->Reorder();

    if (auto lbop = dyn_cast<BinaryOperation>(l)) {
      auto a = lbop->GetLeft();
      auto b = lbop->GetRight();
      auto c = r;
      if (lbop->op == OpCode::MULTIPLY && op == OpCode::DIVIDE &&
          !a->IsNumeric() && b->IsNumeric() && c->IsNumeric()) {
        // simplify (a * b) / c
        auto bv = cast<NumericValue>(b)->Value();
        auto cv = cast<NumericValue>(c)->Value();
        auto gcd_val = gcd(bv, cv);
        if (gcd_val != 1)
          return (a * (b / nu(gcd_val))->Fold()) / (c / nu(gcd_val)->Fold());
      }
#if 0
      if (lbop->op == OpCode::DIVIDE && op == OpCode::MULTIPLY && (*b < *c || (b->IsNumeric() && c->IsNumeric()))) {
        // (a / b) * c -> a * (c / b), when a is not numeric
        auto bv = cast<NumericValue>(b)->Value();
        auto cv = cast<NumericValue>(c)->Value();
        if (gcd(bv, cv) == bv) return a * (c / b)->Fold();
      }
#endif
    }

    if (auto rbop = dyn_cast<BinaryOperation>(r)) {
      auto a = l;
      auto b = rbop->GetLeft();
      auto c = rbop->right;
      if (rbop->op == OpCode::DIVIDE && op == OpCode::DIVIDE &&
          !a->IsNumeric()) {
        if (*a == *b) {
          // a / (b / c) -> c,  when a == b
          return c;
        }
#if 0
        if (b->IsNumeric() && c->IsNumeric()) {
          auto bv = cast<NumericValue>(b)->Value();
          auto cv = cast<NumericValue>(c)->Value();
          if (cv > bv) {
            // a / (b / c) = a * (c / b)
            return a * (c / b);
          }
        }
#endif
      }
    }

    if (IsCommutative(op) && (*l < *r)) std::swap(l, r);

    return std::make_shared<BinaryOperation>(op, l, r);
  }

public:
  __UDT_TYPE_INFO__(SymbolicExpression, BinaryOperation)
};

class TernaryOperation : public SymbolicExpression,
                         public TypeIDProvider<TernaryOperation> {
private:
  OpCode op;
  Operand pred;
  Operand left;
  Operand right;

public:
  TernaryOperation(OpCode op, const Operand& p, const Operand& l,
                   const Operand& r)
      : op(op), pred(p), left(l), right(r) {
    // currently only support select
    assert(op == OpCode::SELECT);
  }

  std::string ToString() const override {
    return "(" + PSTR(pred) + " " + STR(op) + " " + PSTR(left) + " : " +
           PSTR(right) + ")";
  }

  bool IsNumeric() const override {
    // can be optimized
    return pred->IsNumeric() && left->IsNumeric() && right->IsNumeric();
  }

  bool operator==(const SymbolicExpression& expr) const override {
    if (auto se = dyn_cast<TernaryOperation>(&expr))
      if (se->op == op && (*se->pred == *pred) && (*se->left == *left) &&
          (*se->right == *right))
        return true;
    return false;
  }

  size_t Hash() const override { return std::hash<std::string>{}(ToString()); }

public:
  bool IsLeaf() const override { return false; }
  const Operand GetPred() const { return pred; }
  const Operand GetLeft() const { return left; }
  const Operand GetRight() const { return right; }
  OpCode GetOpCode() const { return op; }

  Operand Clone() const override {
    return std::make_shared<TernaryOperation>(op, pred->Clone(), left->Clone(),
                                              right->Clone());
  }

  Operand Fold() const override {
    auto npred = pred->Fold();
    auto nl = left->Fold();
    auto nr = right->Fold();
    if (auto p = dyn_cast<BooleanValue>(npred)) {
      if (p->Value() == true)
        return left->Fold();
      else
        return right->Fold();
    }

    return sel(npred, nl, nr);
  }

  Operand Normalize() const override {
    auto nv = Fold();
    if (!isa<TernaryOperation>(nv)) return nv->Normalize();
    return sel(pred->Normalize(), left->Normalize(), right->Normalize());
  }

  Operand Reassociate() const override { return Clone(); }
  Operand Reorder() const override { return Clone(); }

public:
  __UDT_TYPE_INFO__(SymbolicExpression, TernaryOperation)
};

namespace {

std::string GetHighRankString(const BinaryOperation& b) {
  std::string hrs;
  if (auto sv = dyn_cast<SymbolicValue>(b.GetLeft()))
    hrs = ((hrs > sv->Value()) ? hrs : sv->Value());
  if (auto sv = dyn_cast<SymbolicValue>(b.GetRight()))
    hrs = ((hrs > sv->Value()) ? hrs : sv->Value());

  if (auto sv = dyn_cast<BinaryOperation>(b.GetLeft()->Fold()))
    hrs = ((hrs > GetHighRankString(*sv)) ? hrs : GetHighRankString(*sv));
  if (auto sv = dyn_cast<BinaryOperation>(b.GetRight()->Fold()))
    hrs = ((hrs > GetHighRankString(*sv)) ? hrs : GetHighRankString(*sv));

  return hrs;
}

} // end anonymous namespace

// 0 - equal, positive - gt, negative - lt
// Note: string follows an alphabet ordering, where 'a' > 'b'.
inline int Compare(const SymbolicExpression& lhs,
                   const SymbolicExpression& rhs) {
  auto l = lhs.Fold();
  auto r = rhs.Fold();
  if (isa<NumericValue>(l)) {
    if (isa<NumericValue>(r))
      return 0;
    else if (isa<SymbolicValue>(r) || isa<BinaryOperation>(r))
      return -1;
  } else if (auto ls = dyn_cast<SymbolicValue>(&lhs)) {
    if (isa<NumericValue>(r))
      return 1;
    else if (auto rs = dyn_cast<SymbolicValue>(r))
      return -ls->Value().compare(rs->Value());
    else if (auto rb = dyn_cast<BinaryOperation>(r)) {
      auto hrs = GetHighRankString(*rb);
      return -ls->Value().compare(hrs);
    }
  } else if (auto lb = dyn_cast<BinaryOperation>(l)) {
    auto hrs = GetHighRankString(*lb);
    if (isa<NumericValue>(r))
      return 1;
    else if (auto rs = dyn_cast<SymbolicValue>(r))
      return -hrs.compare(rs->Value());
    else if (auto rb = dyn_cast<BinaryOperation>(r))
      return -hrs.compare(GetHighRankString(*rb));
  }
  choreo_unreachable("unsupported value.");
  return 0;
}

inline bool operator<(const SymbolicExpression& lhs,
                      const SymbolicExpression& rhs) {
  return Compare(lhs, rhs) < 0;
}
inline bool operator!=(const SymbolicExpression& lhs,
                       const SymbolicExpression& rhs) {
  return !(lhs == rhs);
}
inline bool operator==(const SymbolicExpression& lhs, int rhs) {
  return lhs == NumericValue(rhs);
}
inline bool operator!=(const SymbolicExpression& lhs, int rhs) {
  return !(lhs == rhs);
}
inline bool operator==(const SymbolicExpression& lhs, const std::string& rhs) {
  return lhs == SymbolicValue(rhs);
}
inline bool operator!=(const SymbolicExpression& lhs, const std::string& rhs) {
  return !(lhs == rhs);
}

inline Operand SimplifyExpression(const Operand& expr) {
  return expr->Normalize();
}

// Function to simplify an expression and return the result as a string
inline std::string
SimplifyAndPrint(const std::shared_ptr<SymbolicExpression>& expr) {
  auto simplified = expr->Normalize();
  return PSTR(simplified);
}

// Helper functions to create SymbolicExpressions
inline std::shared_ptr<SymbolicExpression> make_numeric(int64_t value) {
  return std::make_shared<NumericValue>(value);
}

inline std::shared_ptr<SymbolicExpression> make_boolean(bool value) {
  return std::make_shared<BooleanValue>(value);
}

inline Operand make_symbolic(const std::string& name) {
  return std::make_shared<SymbolicValue>(name);
}

inline Operand make_operation(OpCode op, const Operand& left,
                              const Operand& right) {
  return std::make_shared<BinaryOperation>(op, left, right);
}

inline Operand make_select(const Operand& pred, const Operand& left,
                           const Operand& right) {
  return std::make_shared<TernaryOperation>(OpCode::SELECT, pred, left, right);
}

// short-cuts
inline Operand nu(int64_t value) { return make_numeric(value); }

inline Operand bl(bool value) { return make_boolean(value); }

inline Operand sym(const std::string& name) { return make_symbolic(name); }

inline Operand bop(OpCode op, const Operand& left, const Operand& right) {
  return make_operation(op, left, right);
}

inline Operand sel(const Operand& pred, const Operand& left,
                   const Operand& right) {
  return make_select(pred, left, right);
}

inline Operand operator+(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::ADD, vi1, vi2)->Normalize();
}

inline Operand operator-(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::SUBTRACT, vi1, vi2)->Normalize();
}

inline Operand operator*(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::MULTIPLY, vi1, vi2)->Normalize();
}

inline Operand operator/(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::DIVIDE, vi1, vi2)->Normalize();
}

inline Operand operator%(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::IRES, vi1, vi2)->Normalize();
}

inline Operand oc_lt(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::LT, vi1, vi2)->Normalize();
}

inline Operand oc_gt(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::GT, vi1, vi2)->Normalize();
}

inline Operand oc_ge(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::GE, vi1, vi2)->Normalize();
}

inline Operand oc_le(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::LE, vi1, vi2)->Normalize();
}

inline Operand oc_eq(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::EQ, vi1, vi2)->Normalize();
}

inline Operand oc_ne(const Operand& vi1, const Operand& vi2) {
  return bop(OpCode::NE, vi1, vi2)->Normalize();
}

inline bool clt(const Operand& vi1, const Operand& vi2) {
  if (auto v = dyn_cast<BooleanValue>(oc_lt(vi1, vi2))) return v->Value();
  return false;
}
inline bool cgt(const Operand& vi1, const Operand& vi2) {
  if (auto v = dyn_cast<BooleanValue>(oc_gt(vi1, vi2))) return v->Value();
  return false;
}
inline bool cle(const Operand& vi1, const Operand& vi2) {
  if (auto v = dyn_cast<BooleanValue>(oc_le(vi1, vi2))) return v->Value();
  return false;
}
inline bool cge(const Operand& vi1, const Operand& vi2) {
  if (auto v = dyn_cast<BooleanValue>(oc_ge(vi1, vi2))) return v->Value();
  return false;
}
inline bool ceq(const Operand& vi1, const Operand& vi2) {
  if (auto v = dyn_cast<BooleanValue>(oc_eq(vi1, vi2))) return v->Value();
  return false;
}
inline bool cne(const Operand& vi1, const Operand& vi2) {
  if (auto v = dyn_cast<BooleanValue>(oc_ne(vi1, vi2))) return v->Value();
  return false;
}

template <typename T>
inline std::basic_ostream<T>& operator<<(std::basic_ostream<T>& os,
                                         const Operand& oprd) {
  if (!oprd)
    os << "nil";
  else
    os << oprd->ToString();
  return os;
}

class OperandHasher {
private:
  std::unordered_set<size_t> used_hashes;
  std::unordered_map<Operand, size_t> item2hash;
  std::unordered_map<size_t, Operand> hash2item;

public:
  size_t operator()(const Operand& oprd) {
    if (oprd == nullptr) choreo_unreachable("operand is null.");
    if (item2hash.count(oprd)) return item2hash[oprd];

    size_t content_hash = oprd->Hash();
    size_t sbe_hash =
        content_hash ^ (0x9e3779b9 + (content_hash << 6) + (content_hash >> 2));
    while (used_hashes.count(sbe_hash)) {
      if (*hash2item.at(sbe_hash) == *oprd) // existing symbolic expression
        return sbe_hash;
      ++sbe_hash; // avoid collision
    }
    used_hashes.insert(sbe_hash);
    item2hash.emplace(oprd, sbe_hash);
    hash2item.emplace(sbe_hash, oprd);
    return sbe_hash;
  }
};

} // end namespace sbe

} // end namespace Choreo

#endif //__CHOREO_SYMBOLIC_EXPRESSION_HPP__
