#ifndef __CHOREO_VALUE_NUMBERING_HPP__
#define __CHOREO_VALUE_NUMBERING_HPP__

#include <charconv>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "ast.hpp"
#include "typeresolve.hpp"
#include "types.hpp"
#include "valbind.hpp"
#include "visitor.hpp"

namespace Choreo {

class ShapeInference;

// Remove the prefix
inline std::string RemovePrefix(const std::string& str,
                                const std::string& prefix) {
  if (str.find(prefix) == 0) return str.substr(prefix.length());
  return str;
}

// retrieve the n-th element from the comma-separated input string
inline std::optional<std::string> GetNthElement(const std::string& input,
                                                int n) {
  std::istringstream iss(input);
  std::string token;
  int currentIndex = 0;

  // Iterate through the tokens in the string
  while (std::getline(iss, token, ',')) {
    if (currentIndex == n) {
      // We've found the token at the specified index
      return token;
    }
    currentIndex++;
  }

  // If the index is out of range, return an empty string
  return std::nullopt;
}

inline int CountElementsInSignature(const std::string& input) {
  if (input.empty()) return 0; // Return 0 if the string is empty
  int count = 1;
  for (char c : input) {
    if (c == ',') ++count; // Increment for each comma found
  }
  return count;
}

namespace valno {

class NumTy { // value number type
private:
  constexpr static int invalid_val = GetInvalidValueNumber();
  // none valno: unspecified value which can not be evaluate (but could be part
  // of multi-vns)
  constexpr static int none_val = std::numeric_limits<int>::max() - 1;
  // unknown valno: bottom value for exceptions
  constexpr static int unknown_val = -1;

private:
  int valno;

public:
  NumTy(int v = invalid_val) : valno(v) {}
  NumTy(const NumTy& n) = default;
  NumTy& operator=(const NumTy& n) = default;

  int Value() const { return valno; }
  bool IsValid() const { return valno != invalid_val; }
  bool IsInValid() const { return valno == invalid_val; }
  bool IsNone() const { return valno == none_val; }
  bool IsUnknown() const { return valno == unknown_val; }

  void Invalidate() { valno = invalid_val; }

  bool operator==(const NumTy& n) const { return valno == n.valno; }
  bool operator!=(const NumTy& n) const { return valno != n.valno; }

  const std::string ToString(bool textual = false) const {
    if (textual) {
      if (IsInValid()) return "inv";
      if (IsUnknown()) return "unk";
      if (IsNone()) return "nil";
    }
    return "#" + std::to_string(valno);
  }

public:
  const static NumTy None() { return NumTy(none_val); }
  const static NumTy Invalid() { return NumTy(invalid_val); }
  const static NumTy Unknown() { return NumTy(unknown_val); }
};

using OpTy = std::string;
inline const std::string STR(const OpTy& o) { return o; }

class UnknownSign;
class NoneSign;
class Signature {
protected:
  static constexpr const char* unknown_sign = "__valno_not_known__";
  static constexpr const char* none_sign = "__valno_not_specified__";

public:
  virtual size_t Count() const { return 1; }
  virtual const std::string ToString() const { return ""; }

  virtual bool operator==(const Signature&) const = 0;
  virtual bool operator!=(const Signature& s) const { return !operator==(s); }

public:
  static const UnknownSign Unknown();
  static const NoneSign None();

public:
  // for runtime type disambiguation
  __UDT_TYPE_INFO_BASE__(notype)
};

using SignTy = std::shared_ptr<Signature>;

class UnknownSign : public Signature, public TypeIDProvider<UnknownSign> {
public:
  UnknownSign() {}
  const std::string ToString() const override { return "__valno_not_known__"; }

  bool operator==(const Signature& st) const override {
    return isa<UnknownSign>(&st);
  }

public:
  __UDT_TYPE_INFO__(Signature, UnknownSign)
};

class NoneSign : public Signature, public TypeIDProvider<NoneSign> {
public:
  NoneSign() {}
  const std::string ToString() const override {
    return "__valno_not_specified__";
  }

  bool operator==(const Signature& st) const override {
    return isa<NoneSign>(&st);
  }

public:
  __UDT_TYPE_INFO__(Signature, NoneSign)
};

class ConstSign : public Signature, public TypeIDProvider<ConstSign> {
private:
  using Var = std::variant<int64_t, float, double, bool>;
  Var value;

public:
  ConstSign(int v) : value((int64_t)v) {}
  ConstSign(int64_t v) : value(v) {}
  ConstSign(float v) : value(v) {}
  ConstSign(double v) : value(v) {}
  ConstSign(bool v) : value(v) {}

  const std::variant<int64_t, float, double, bool>& Value() const {
    return value;
  }
  template <typename T>
  bool Holds() const {
    return std::holds_alternative<T>(value);
  }
  template <typename T>
  const T Get() const {
    return std::get<T>(value);
  }
  bool IsFloat() const {
    return std::holds_alternative<float>(value) ||
           std::holds_alternative<double>(value);
  }

  struct IsZeroVisitor {
    bool operator()(int64_t value) const { return value == 0; }
    bool operator()(float value) const { return value == 0.0f; }
    bool operator()(double value) const { return value == 0.0; }
    // Never consider bool as zero
    bool operator()(bool) const { return false; }
  };
  bool IsZero() { return std::visit(IsZeroVisitor{}, value); }

  struct IsDenormalVisitor {
    bool operator()(int64_t) const { return false; }
    bool operator()(bool) const { return false; }
    bool operator()(float value) const {
      return std::fpclassify(value) == FP_SUBNORMAL;
    }
    bool operator()(double value) const {
      return std::fpclassify(value) == FP_SUBNORMAL;
    }
  };
  bool IsDenormal() { return std::visit(IsDenormalVisitor{}, value); }

  struct IsNaNVisitor {
    bool operator()(int64_t) const { return false; }
    bool operator()(bool) const { return false; }
    bool operator()(float value) const {
      return std::fpclassify(value) == FP_NAN;
    }
    bool operator()(double value) const {
      return std::fpclassify(value) == FP_NAN;
    }
  };
  bool IsNaN() { return std::visit(IsNaNVisitor{}, value); }

  int64_t GetInt() const {
    if (!std::holds_alternative<int64_t>(value))
      choreo_unreachable("the const is not an integer.");
    return std::get<int64_t>(value);
  }
  bool GetBool() const {
    if (!std::holds_alternative<bool>(value))
      choreo_unreachable("the const is not a boolean.");
    return std::get<bool>(value);
  }
  float GetFloat() const {
    if (!std::holds_alternative<float>(value))
      choreo_unreachable("the const is not a float.");
    return std::get<float>(value);
  }
  double GetDouble() const {
    if (!std::holds_alternative<double>(value))
      choreo_unreachable("the const is not a double.");
    return std::get<double>(value);
  }
  // Visitor that converts any supported variant type to string
  struct ToStringVisitor {
    template <typename T>
    const std::string operator()(const T& value) const {
      if constexpr (std::is_same_v<T, bool>)
        return value ? "true" : "false";
      else if constexpr (std::is_same_v<T, float>)
        return "const_" + std::to_string(value) + "f";
      else
        return "const_" + std::to_string(value);
    }
  };

  const std::string ToString() const override {
    return std::visit(ToStringVisitor{}, value);
  }

public:
  // Type promotion rules
  template <typename T, typename U>
  using promoted_t = std::conditional_t<
      (std::is_same_v<T, double> || std::is_same_v<U, double>), double,
      std::conditional_t<
          (std::is_same_v<T, float> || std::is_same_v<U, float>), float,
          std::conditional_t<(std::is_same_v<T, int64_t> ||
                              std::is_same_v<U, int64_t>),
                             int64_t, std::common_type_t<T, U>>>>;

  // Arithmetic visitor with string opcode
  struct ArithmeticVisitor {
    const OpTy op;

    template <typename T, typename U>
    ptr<Signature> operator()(T a, U b) const {
      // Convert bools to int for arithmetic
      if constexpr (std::is_same_v<T, bool>) a = static_cast<int>(a);
      if constexpr (std::is_same_v<U, bool>) b = static_cast<int>(b);

      using ResultType = promoted_t<T, U>;
      auto promoted_a = static_cast<ResultType>(a);
      auto promoted_b = static_cast<ResultType>(b);

      if (op == "+")
        return std::make_shared<ConstSign>(promoted_a + promoted_b);
      if (op == "-")
        return std::make_shared<ConstSign>(promoted_a - promoted_b);
      if (op == "*")
        return std::make_shared<ConstSign>(promoted_a * promoted_b);
      if (op == "/") {
        if (promoted_b == 0) choreo_unreachable("divide by zero is found.");
        return std::make_shared<ConstSign>(promoted_a / promoted_b);
      }
      if (op == "%") {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {
          if (promoted_b == 0) choreo_unreachable("divide by zero is found.");
          return std::make_shared<ConstSign>(promoted_a % promoted_b);
        }
        return std::make_shared<UnknownSign>();
      }
      if (op == "cdiv") {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {
          if (promoted_b == 0) choreo_unreachable("divide by zero is found.");
          return std::make_shared<ConstSign>((promoted_a + promoted_b - 1) /
                                             promoted_b);
        }
        return std::make_shared<UnknownSign>();
      }
      if (op == "#") {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<U>)
          return std::make_shared<ConstSign>(promoted_a * promoted_b);
        return std::make_shared<UnknownSign>();
      }
      if (op == "#+") {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<U>)
          return std::make_shared<ConstSign>(promoted_a + promoted_b);
        return std::make_shared<UnknownSign>();
      }
      if (op == "#-") {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<U>)
          return std::make_shared<ConstSign>(promoted_a - promoted_b);
        return std::make_shared<UnknownSign>();
      }

      // Comparison operators return bool
      if (op == "<")
        return std::make_shared<ConstSign>(promoted_a < promoted_b);
      if (op == ">")
        return std::make_shared<ConstSign>(promoted_a > promoted_b);
      if (op == "==")
        return std::make_shared<ConstSign>(promoted_a == promoted_b);
      if (op == "!=")
        return std::make_shared<ConstSign>(promoted_a != promoted_b);
      if (op == "<=")
        return std::make_shared<ConstSign>(promoted_a <= promoted_b);
      if (op == ">=")
        return std::make_shared<ConstSign>(promoted_a >= promoted_b);

      return std::make_shared<UnknownSign>();
    }
  }; // ArithmeticVisitor

  bool operator==(const Signature& st) const override {
    if (auto csign = dyn_cast<ConstSign>(&st)) return value == csign->value;
    return false;
  }

public:
  __UDT_TYPE_INFO__(Signature, ConstSign)
};

class SymbolSign : public Signature, public TypeIDProvider<SymbolSign> {
private:
  const std::string symbol;

public:
  SymbolSign(const std::string& s) : symbol(s) {
    assert(PrefixedWith(s, "::"));
  }
  const std::string Value() const { return symbol; }
  const std::string ToString() const override { return symbol; }

  bool operator==(const Signature& st) const override {
    if (auto sign = dyn_cast<SymbolSign>(&st)) return symbol == sign->symbol;
    return false;
  }

public:
  __UDT_TYPE_INFO__(Signature, SymbolSign)
};

class OperationSign : public Signature, public TypeIDProvider<OperationSign> {
private:
  const OpTy opcode;
  std::vector<SignTy> operands;

public:
  template <
      typename... Args,
      typename = std::enable_if_t<
          (std::conjunction_v<std::is_same<std::decay_t<Args>, SignTy>...>)>>
  OperationSign(const OpTy& o, const Args&... ns) : opcode(o) {
    operands.reserve(sizeof...(ns));
    (operands.push_back(ns), ...);
  }

  const std::vector<SignTy>& OperandSigns() const { return operands; }
  const OpTy& Operation() const { return opcode; }
  bool IsOp(const OpTy& o) const { return opcode == o; }
  size_t OpCount() const { return operands.size(); }
  void Append(const SignTy& n) { operands.push_back(n); }
  const SignTy& At(size_t i) const {
    if (i > OpCount())
      choreo_unreachable("out of bound for an operation signature.");
    return operands[i];
  }

  const std::string ToString() const override {
    std::string res = STR(opcode);
    for (auto& op : operands) res += ":" + op->ToString();
    return res;
  }

  bool operator==(const Signature& st) const override {
    auto sign = dyn_cast<OperationSign>(&st);
    if (!sign) return false;
    if (!IsOp(sign->opcode)) return false;
    if (OpCount() != sign->OpCount()) return false;
    for (size_t i = 0; i < OpCount(); ++i)
      if (At(i) != sign->At(i)) return false;
    return true;
  }

public:
  __UDT_TYPE_INFO__(Signature, OperationSign)
};

// plural
class MultiSigns : public Signature, public TypeIDProvider<MultiSigns> {
private:
  std::vector<SignTy> signs;

public:
  // works for c++17
  template <
      typename... Args,
      typename = std::enable_if_t<
          (std::conjunction_v<std::is_same<std::decay_t<Args>, SignTy>...>)>>
  MultiSigns(const Args&... ns) {
    signs.reserve(sizeof...(ns));
    (signs.push_back(ns), ...);
  }

  // broadcast init
  MultiSigns(const SignTy& n, size_t count) {
    for (size_t i = 0; i < count; ++i) signs.push_back(n);
  }

  MultiSigns(const std::vector<SignTy>& vs) : signs(vs) {}

  size_t Count() const override { return signs.size(); }
  const std::vector<SignTy>& AllSigns() const { return signs; }
  const SignTy& At(size_t i) const { return signs.at(i); }
  const SignTy& At(const NumTy& n) const { return signs.at(n.Value()); }

  void Append(const SignTy& n) { signs.push_back(n); }
  void Append(const SignTy& n, size_t count) {
    for (size_t i = 0; i < count; ++i) signs.push_back(n);
  }

  bool operator==(const Signature& st) const override {
    auto sign = dyn_cast<MultiSigns>(&st);
    if (!sign) return false;
    if (Count() != sign->Count()) return false;
    for (size_t i = 0; i < Count(); ++i)
      if (At(i) != sign->At(i)) return false;
    return true;
  }

  const std::string ToString() const override {
    std::string res;
    for (size_t i = 0; i < signs.size(); ++i) {
      if (i > 0) res += ",";
      res += signs[i]->ToString();
    }
    return res;
  }

public:
  __UDT_TYPE_INFO__(Signature, MultiSigns)
};

inline const ptr<UnknownSign> unk_sn() {
  return std::make_shared<UnknownSign>();
}
inline const ptr<NoneSign> non_sn() { return std::make_shared<NoneSign>(); }

// short-hands
template <typename T>
inline const ptr<ConstSign> c_sn(const T& v) {
  return std::make_shared<ConstSign>(v);
}
inline const ptr<SymbolSign> s_sn(const std::string& s) {
  return std::make_shared<SymbolSign>(s);
}
template <typename... Args,
          typename = std::enable_if_t<(
              std::conjunction_v<std::is_same<std::decay_t<Args>, SignTy>...>)>>
inline const ptr<OperationSign> o_sn(const std::string& op, Args... ns) {
  return std::make_shared<OperationSign>(op, ns...);
}
template <typename... Args,
          typename = std::enable_if_t<(
              std::conjunction_v<std::is_same<std::decay_t<Args>, SignTy>...>)>>
inline const ptr<MultiSigns> m_sn(Args... ns) {
  return std::make_shared<MultiSigns>(ns...);
}
inline const ptr<MultiSigns> m_sn(const SignTy& n, size_t cnt) {
  return std::make_shared<MultiSigns>(n, cnt);
}
inline const ptr<MultiSigns> m_sn(const std::vector<SignTy>& v) {
  return std::make_shared<MultiSigns>(v);
}
inline bool IsValid(const SignTy& s) { return s != nullptr; }
inline bool IsUnknown(const SignTy& s) { return isa<UnknownSign>(s); }
inline bool IsNone(const SignTy& s) { return isa<NoneSign>(s); }
inline const ptr<ConstSign> CSign(const SignTy& s) {
  return dyn_cast<ConstSign>(s);
}
inline const ptr<SymbolSign> SSign(const SignTy& s) {
  return dyn_cast<SymbolSign>(s);
}
inline const ptr<OperationSign> OpSign(const SignTy& s) {
  return dyn_cast<OperationSign>(s);
}
inline const ptr<MultiSigns> MSign(const SignTy& s) {
  return dyn_cast<MultiSigns>(s);
}

} // end namespace valno

inline std::ostream& operator<<(std::ostream& os, const valno::NumTy& n) {
  os << n.ToString();
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const valno::SignTy& n) {
  os << n->ToString();
  return os;
}

} // end namespace Choreo

// make NumTy & SignTy to work with unordered_map
namespace std {
template <>
struct hash<Choreo::valno::NumTy> {
  size_t operator()(const Choreo::valno::NumTy& k) const {
    return std::hash<int>{}(k.Value());
  }
};

template <>
struct hash<Choreo::valno::SignTy> {
  size_t operator()(const Choreo::valno::SignTy& k) const {
    return std::hash<std::string>{}(k->ToString());
  }
};

// Add the matching equality comparator
template <>
struct equal_to<Choreo::valno::SignTy> {
  bool operator()(const Choreo::valno::SignTy& a,
                  const Choreo::valno::SignTy& b) const {
    // Handle null pointers if applicable
    if (!a || !b) return !a && !b;
    return a->ToString() == b->ToString();
  }
};

} // end namespace std

inline bool operator==(const Choreo::valno::SignTy& lhs,
                       const Choreo::valno::SignTy& rhs) {
  return lhs->operator==(*rhs);
}

inline bool operator!=(const Choreo::valno::SignTy& lhs,
                       const Choreo::valno::SignTy& rhs) {
  return !lhs->operator==(*rhs);
}

namespace Choreo {
namespace valno {

inline const std::string STR(const SignTy& s) { return s->ToString(); }
inline const std::string STR(const NumTy& v) { return v.ToString(); }

using Choreo::STR;

// Note:
//  1. A value number is 1-1 mapped with a constant signature.
//  2. If not representing constant, the value number and its signatures are 1-n
//  mapped, where multiple signatures could have a same value number.
//  valid expression.
class ValueNumberTable {
private:
  // a signature may either be inside the sign_pool or const_pool
  std::unordered_map<SignTy, NumTy> sign_pool;
  std::unordered_map<SignTy, NumTy> const_pool;
  std::unordered_map<NumTy, std::vector<SignTy>> value_nums;

private:
  int next_valno = 0;
  bool trace = false;

private:
  void Reset() {}

  bool IsConstant(const SignTy& s) const { return isa<ConstSign>(s); }

  bool ValueNumExists(const SignTy& expr) const {
    if (sign_pool.count(expr)) return true;
    return const_pool.count(expr) != 0;
  }

  bool SignatureExists(NumTy vn) const { return value_nums.count(vn) != 0; }

public:
  ValueNumberTable(bool t = false) : trace(t) {
    // Add special values
    const_pool.emplace(unk_sn(), NumTy::Unknown());
    const_pool.emplace(non_sn(), NumTy::None());
    value_nums[NumTy::Unknown()].push_back(unk_sn());
    value_nums[NumTy::None()].push_back(non_sn());
  }

  bool Exists(SignTy s) const { return ValueNumExists(s); }
  bool Exists(NumTy vn) const { return SignatureExists(vn); }

  NumTy GetValueNum(const SignTy& expr) const {
    if (sign_pool.count(expr)) return sign_pool.at(expr);
    if (const_pool.count(expr) == 0)
      choreo_unreachable("can not find valno of expression : " + STR(expr) +
                         ".");
    return const_pool.at(expr);
  }

  const SignTy& GetSignature(const NumTy& vn) const {
    if (!Exists(vn))
      choreo_unreachable("can not find signature of valno: " + STR(vn) + ".");

    return value_nums.at(vn).at(0); // use the first signature
  }

  // Add a new signature to an existing valno as its alias
  void Alias(NumTy vn, const SignTy& s) {
    if (!Exists(vn))
      choreo_unreachable("Alias fails: valno: " + STR(vn) +
                         " does not exists.");

    if (Exists(s))
      choreo_unreachable("Alias fails: signature: " + STR(s) + " exists.");

    if (IsConstant(s))
      const_pool.emplace(s, vn);
    else
      sign_pool.emplace(s, vn);

    value_nums.at(vn).push_back(s);
  }

  void ReAlias(NumTy vn, const SignTy& s) {
    if (!Exists(vn))
      choreo_unreachable("ReAlias fails: valno: " + STR(vn) +
                         " does not exists.");
    if (!Exists(s))
      choreo_unreachable("ReAlias fails: signature: " + STR(s) +
                         " does not exists.");
    if (IsConstant(s)) choreo_unreachable("ReAlias fails: constant.");

    sign_pool[s] = vn;
    auto& signs = value_nums[vn];
    signs.erase(std::remove(signs.begin(), signs.end(), s), signs.end());
  }

  // specific: take a dummy signature in (not associated with an invalid valno)
  void DummyGen(const SignTy& s) {
    assert(!IsConstant(s));

    if (Exists(s)) choreo_unreachable("signature: " + STR(s) + " exists.");

    sign_pool.emplace(s, GetInvalidValueNumber());
  }

  // Generate a valno for the new signature
  NumTy Generate(const SignTy& s) {
    // note: dummy sign can be re-generated
    if (Exists(s) && GetValueNum(s).IsValid())
      choreo_unreachable("signature: " + STR(s) + " exists.");

    // generate a new value number
    auto valno = next_valno++;

    // be defensive
    assert(value_nums.count(valno) == 0);

    if (IsConstant(s))
      const_pool.emplace(s, valno);
    else
      sign_pool.emplace(s, valno);

    value_nums.emplace(valno, std::vector<SignTy>{});
    value_nums[valno].push_back(s);

    return valno;
  }

  // Bind a valno to the existing (dummy) signature
  void BindDummy(const SignTy& s, NumTy v) {
    // note: only dummy sign can be re-generated
    assert(!IsConstant(s) && "unable to regen const.");

    if (!Exists(s))
      choreo_unreachable("signature: '" + STR(s) + "' does not exists.");
    if (!Exists(v))
      choreo_unreachable("valno: " + STR(s) + " does not exists.");

    if (GetValueNum(s).IsValid())
      choreo_unreachable("signature: " + STR(s) + " has a valid valno.");

    sign_pool[s] = v;

    value_nums[v].push_back(s);
  }

public:
  void EnterScope() {}
  void LeaveScope() {}

public:
  void Print(std::ostream& os) const {
    for (auto& item : sign_pool)
      os << "expr: \"" << item.first << "\", valno: " << item.second << "\n";
    for (auto& item : const_pool)
      os << "const: \"" << item.first << "\", valno: " << item.second << "\n";
  }
};

class ValueNumbering {
private:
  ShapeInference* visitor;

private:
  ValueNumberTable vntbl;

public:
  ValueNumberTable& Tabel() { return vntbl; }
  const ValueNumberTable& Tabel() const { return vntbl; }

private:
  bool need_bound = true;
  bool trace = false;

  std::optional<std::string> ref = std::nullopt;

public:
  explicit ValueNumbering(ShapeInference* v)
      : visitor(v), vntbl(CCtx().TraceValueNumbers()),
        trace(CCtx().TraceValueNumbers()) {}

  void EnterScope();
  void LeaveScope();

  void SetListReference(const std::string& r) { ref = r; }
  void ResetListReference() { ref.reset(); }

  // It binds a expression signature with an existing value number.
  void AssociateSignatureWithValueNumber(const SignTy&, const NumTy&);
  void AssociateSignatureWithInvalidValueNumber(const SignTy&);

  // rebind/modify the value number.
  // Caution: only used for scenario where the value number has not been
  // determined yet.
  void RebindSignatureWithValueNumber(const SignTy&, const NumTy&);

  // Directly get the value number from a signature. Abort when it fails.
  const NumTy GetValueNumberOfSignature(const SignTy&) const;

  // Generate the new value number from a signature. Abort when the value number
  // exists.
  const NumTy GenerateValueNumberFromSignature(const SignTy&);

  // Check if the value number exists for the signature
  bool HasValueNumberOfSignature(const SignTy&) const;

  // Check if the value number exists and is valid for the signature
  bool HasValidValueNumberOfSignature(const SignTy&);

  const NumTy GetOrGenValueNumberFromSignature(const SignTy&);

  // Retrieve the signature from a value number. About when fails.
  const SignTy GetSignatureFromValueNumber(const NumTy& vn) const {
    if (vn.IsUnknown()) return unk_sn();
    if (vn.IsNone()) return non_sn();

    if (!vntbl.Exists(vn))
      choreo_unreachable("value number " + STR(vn) +
                         " does not exists in the value number table.");
    return vntbl.GetSignature(vn);
  }

  // Simplify signature when possible, or else return origial signature
  const SignTy Simplify(const SignTy&);

  // Force to be multisigns
  const ptr<MultiSigns> ToMSign(const SignTy& s) const {
    if (auto ms = dyn_cast<MultiSigns>(s)) return ms;
    return m_sn(s);
  }

  bool ContainsZero(const SignTy& s) const {
    auto msn = ToMSign(s);
    for (auto& n : msn->AllSigns())
      if (auto csn = CSign(n))
        if (csn->IsZero()) return true;

    return false;
  }

  // Try to generate simplified signature. when fails, return 'unknown'
  const SignTy TryToSimplifyBinary(const OpTy&, const SignTy&, const SignTy&,
                                   bool = false);

  const SignTy SignBinaryCompositeValues(const location&, const OpTy&,
                                         const MultiSigns&, const MultiSigns&,
                                         bool = false);

  ValueItem GenValueItemFromSignature(const SignTy&);
  ValueItem GenValueItemFromValueNumber(const NumTy&);
  const ValueList GenValueListFromSignature(const SignTy&);
  const ValueList GenValueListFromValueNumber(const NumTy&);
  const SignTy ValueItemToSignature(const ValueItem&, bool = false);
  const SignTy ValueListToSignature(const ValueList&, bool = true);

public:
  ValBind::BindInfo<NumTy> bind_info; // TODO: to abondon

  void BindValueNumbers(NumTy vn0, NumTy vn1) {
    assert(vntbl.Exists(vn0) && vntbl.Exists(vn1) &&
           "invalid value number is provided.");

    AddBind(vn0, vn1);
  }

  // Bind two value numbers
  const ValBind::Binds<NumTy>::Set& GetBindSet(const NumTy& vn) {
    auto& ret = bind_info.GetSet(vn);
    return ret;
  }

  void AddBind(NumTy vn0, NumTy vn1) { bind_info.AddBind(vn0, vn1); }

public:
  // utilities
  const std::vector<NumTy> Flatten(const NumTy&) const;

  const std::string ScopeIndent();

  const NumTy NumSign(const SignTy& s) const {
    return GetValueNumberOfSignature(s);
  }
  const SignTy SignNum(const NumTy& n) const {
    return GetSignatureFromValueNumber(n);
  }

  const std::vector<NumTy> NumVector(const SignTy& sign) const {
    std::vector<NumTy> res;
    if (auto ms = dyn_cast<MultiSigns>(sign)) {
      for (auto s : ms->AllSigns()) res.push_back(NumSign(s));
    } else
      res.push_back(NumSign(sign));
    return res;
  }

  const std::vector<NumTy> NumVector(const NumTy& valno) const {
    assert(valno.IsValid() && "not a valid value number.");
    return NumVector(GetSignatureFromValueNumber(valno));
  }

  const std::vector<SignTy> Lift(const SignTy& sign) const {
    if (auto ms = ToMSign(sign)) return ms->AllSigns();
    return {sign};
  }

  const SignTy Concat(const SignTy& ls, const SignTy& rs) const {
    auto& lsns = Lift(ls);
    auto& rsns = Lift(rs);
    std::vector<SignTy> res;
    res.insert(res.end(), lsns.begin(), lsns.end());
    res.insert(res.end(), rsns.begin(), rsns.end());
    return m_sn(res);
  }

  const SignTy MakePluralSign(const std::vector<NumTy>& vns) const {
    auto s = m_sn();
    for (auto& n : vns) s->Append(SignNum(n));
    return s;
  }

  const NumTy MakePluralNum(const std::vector<NumTy>& vns) {
    return GetOrGenValueNumberFromSignature(MakePluralSign(vns));
  }

  const SignTy MakeOpSign(const std::string& op, const SignTy& lsn,
                          const SignTy& rsn) {
    auto osn = o_sn(op, lsn, rsn);
    return Simplify(osn);
  }

  const NumTy MakeOpNum(const std::string& op, const SignTy& lsn,
                        const SignTy& rsn) {
    return GetOrGenValueNumberFromSignature(MakeOpSign(op, lsn, rsn));
  }

  const std::string ToSTR(const SignTy& sn) const {
    // for better readability
    std::string res;
    if (auto msn = dyn_cast<MultiSigns>(sn)) {
      for (size_t i = 0; i < msn->Count(); ++i) {
        if (i > 0) res += ",";
        res += STR(NumSign(msn->At(i)));
      }
    } else if (auto osn = dyn_cast<OperationSign>(sn)) {
      res = STR(osn->Operation());
      for (auto& op : osn->OperandSigns()) res += ":" + STR(NumSign(op));
    } else
      res = STR(sn);

    return res;
  }

  const std::string ToSTR(const NumTy& vn) const { return STR(vn); }

  const SignTy MakeOpSign(const std::string& op, const NumTy& lvn,
                          const NumTy& rvn) {
    auto osn = o_sn(op, SignNum(lvn), SignNum(rvn));
    return Simplify(osn);
  }

  const NumTy MakeOpNum(const std::string& op, const NumTy& lvn,
                        const NumTy& rvn) {
    return GetOrGenValueNumberFromSignature(MakeOpSign(op, lvn, rvn));
  }

  void Print(std::ostream& os) const { vntbl.Print(os); }

private:
  void Error(const location& loc, const std::string& message);
  void Warning(const location& loc, const std::string& message);
};

} // end namespace valno

} // end namespace Choreo

#endif // __CHOREO_VALUE_NUMBERING_HPP__
