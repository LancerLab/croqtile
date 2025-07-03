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

using SignTy = std::string; // signature type. TODO: use structure

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

} // end namespace valno

inline std::ostream& operator<<(std::ostream& os, const valno::NumTy& n) {
  os << n.ToString();
  return os;
}

} // end namespace Choreo

// make NumTy to work with unordered_map
namespace std {
template <>
struct hash<Choreo::valno::NumTy> {
  size_t operator()(const Choreo::valno::NumTy& k) const {
    return std::hash<int>{}(k.Value());
  }
};
} // end namespace std

namespace Choreo {
namespace valno {

inline const SignTy UnknownSign() { return "__valno_not_known__"; }
inline bool IsUnknownSign(const SignTy& s) { return s == UnknownSign(); }

inline const SignTy NoneSign() { return SignTy("__valno_not_specified__"); }
inline bool IsNoneSign(const SignTy& s) { return s == NoneSign(); }

inline const std::string STR(const SignTy& s) { return s; }
inline const std::string STR(const NumTy& v) { return v.ToString(); }

using Choreo::STR;

// Assumptions:
//  1. A value number is 1-1 mapped with a constant signature.
//  2. If not representing constant, the value number and its signatures are 1-n
//  mapped, where multiple signaturea could have a same value number.
//  3. signatures are scoped. Any signature exists on scope stack indicates an
//  valid expression.
class ValueNumberTable {
private:
  // a signature may either be inside the scoped_pool or const_pool
  std::vector<std::unordered_map<SignTy, NumTy>> scoped_pool;
  std::unordered_map<SignTy, NumTy> const_pool;
  std::unordered_map<NumTy, std::vector<SignTy>> value_nums;

private:
  int next_valno = 0;
  bool trace = false;

private:
  void Reset() {}

  bool IsConstant(const SignTy& s) const { return PrefixedWith(s, "const_"); }

  bool ValueNumExists(const SignTy& expr) const {
    for (auto expr_valno = scoped_pool.rbegin();
         expr_valno != scoped_pool.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      return true;
    }
    return const_pool.count(expr) != 0;
  }

  bool SignatureExists(NumTy vn) const { return value_nums.count(vn) != 0; }

  bool InsertToSignTable(const SignTy& s, NumTy v) {
    bool done = false;
    for (auto expr_valno = scoped_pool.rbegin();
         expr_valno != scoped_pool.rend(); expr_valno++) {
      if (!expr_valno->count(s)) continue;
      (*expr_valno)[s] = v;
      done = true;
      break;
    }
    return done;
  }

public:
  ValueNumberTable(bool t = false) : trace(t) {
    // Add special values
    const_pool.emplace(UnknownSign(), NumTy::Unknown());
    const_pool.emplace(NoneSign(), NumTy::None());
    value_nums[NumTy::Unknown()].push_back(UnknownSign());
    value_nums[NumTy::None()].push_back(NoneSign());
  }

  bool Exists(SignTy s) const { return ValueNumExists(s); }
  bool Exists(NumTy vn) const { return SignatureExists(vn); }

  NumTy GetValueNum(const SignTy& expr) const {
    for (auto expr_valno = scoped_pool.rbegin();
         expr_valno != scoped_pool.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      return expr_valno->at(expr);
    }
    if (const_pool.count(expr) == 0)
      choreo_unreachable("can not find valno of expression : " + expr + ".");
    return const_pool.at(expr);
  }

  const SignTy& GetSignature(NumTy vn) const {
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
      scoped_pool.back().emplace(s, vn);

    value_nums.at(vn).push_back(s);
  }

  // specific: take a dummy signature in (not associated with an invalid valno)
  void DummyGen(const SignTy& s) {
    assert(!IsConstant(s));

    if (Exists(s)) choreo_unreachable("signature: " + STR(s) + " exists.");

    scoped_pool.back().emplace(s, GetInvalidValueNumber());
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
      scoped_pool.back().emplace(s, valno);

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

    for (auto expr_valno = scoped_pool.rbegin();
         expr_valno != scoped_pool.rend(); expr_valno++) {
      if (!expr_valno->count(s)) continue;
      (*expr_valno)[s] = v;
    }

    value_nums[v].push_back(s);
  }

  // Bind two value numbers
public:
  void EnterScope() { scoped_pool.push_back({}); }
  void LeaveScope() {
    assert(!scoped_pool.empty());

    for (auto& item : scoped_pool.back()) {
      // Dummy Signature is not associated with a valid valno
      if (!item.second.IsValid()) continue;

      auto& signs = value_nums[item.second];
      signs.erase(std::remove(signs.begin(), signs.end(), item.first),
                  signs.end());
      // remove the valno entry totally when no signatures is mapped
      if (signs.empty()) value_nums.erase(item.second);
    }

    // drop all the signatures in the frame
    scoped_pool.pop_back();

    // reset value number when leaving the function scope
    if (scoped_pool.size() <= 1) Reset();
  }

public:
  void Print(std::ostream& os) const {
    int scope = 0;
    for (auto& stack : scoped_pool) {
      os << scope++ << "\n";
      for (auto& item : stack)
        os << "expr: \"" << item.first << "\", valno: " << item.second << "\n";
    }
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
  NumTy VNReal(const std::string&) const;
  const SignTy RealSign(const std::string&) const;

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
  void AssociateSignatureWithValueNumber(const SignTy& sig, NumTy valno);
  void AssociateSignatureWithInvalidValueNumber(const SignTy& sig);
  // rebind/modify the value number.
  // Caution: only used for scenario where the value number has not been
  // determined yet.
  void RebindSignatureWithValueNumber(const SignTy& sig, NumTy valno);

  // Directly get the value number from a signature. Abort when it fails.
  NumTy GetValueNumberOfSignature(const SignTy&) const;

  // Generate the new value number from a signature. Abort when the value number
  // exists.
  NumTy GenerateValueNumberFromSignature(const SignTy& signature);

  // Check if the value number exists for the signature
  bool HasValueNumberOfSignature(const SignTy&) const;

  // Check if the value number exists and is valid for the signature
  bool HasValidValueNumberOfSignature(const SignTy&);

  NumTy GetOrGenValueNumberFromSignature(const SignTy&);

  // Retrieve the signature from a value number. About when fails.
  SignTy GetSignatureFromValueNumber(NumTy vn) const {
    if (vn.IsUnknown()) return UnknownSign();
    if (vn.IsNone()) return NoneSign();

    if (!vntbl.Exists(vn))
      choreo_unreachable("value number " + STR(vn) +
                         " does not exists in the value number table.");
    return vntbl.GetSignature(vn);
  }

  SignTy SignatureOfSymbol(SignTy sym) {
    return GetSignatureFromValueNumber(GetValueNumberOfSignature(sym));
  }

  const SignTy SimplifySignature(const location&, const SignTy&);

  std::optional<SignTy> TryToSimplifyBinary(const location&, const SignTy&,
                                            const SignTy&, const SignTy&,
                                            bool = false);

  const SignTy SignBinaryCompositeValues(const location&, const SignTy&,
                                         const SignTy&, const SignTy&,
                                         bool = false);

  ValueItem GenValueItemFromSignature(const SignTy&);
  ValueItem GenValueItemFromValueNumber(NumTy);
  const ValueList GenValueListFromSignature(const SignTy&);
  const ValueList GenValueListFromValueNumber(NumTy);
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
  const ValBind::Binds<NumTy>::Set& GetBindSet(NumTy vn) {
    auto& ret = bind_info.GetSet(vn);
    return ret;
  }

  void AddBind(NumTy vn0, NumTy vn1) { bind_info.AddBind(vn0, vn1); }

public:
  // retrieve the n-th element from the comma-separated input string
  NumTy GetNthValNo(const SignTy& input, NumTy n) const;
  const std::vector<NumTy> Flatten(NumTy) const;

  const std::string ScopeIndent();

  const std::vector<NumTy> AsVector(const SignTy& sign) const {
    std::vector<NumTy> mvn;
    if (!PrefixedWith(sign, "#"))
      mvn.push_back(GetValueNumberOfSignature(sign));
    else {
      auto parts = SplitStringByDelimiter(sign, ",");
      for (auto& p : parts) mvn.push_back(std::stoi(p.substr(1)));
    }
    return mvn;
  }

  const std::vector<NumTy> AsVector(NumTy valno) const {
    assert(valno.IsValid() && "not a valid value number.");
    return AsVector(GetSignatureFromValueNumber(valno));
  }

  void Print(std::ostream& os) const { vntbl.Print(os); }

private:
  void Error(const location& loc, const std::string& message);
  void Warning(const location& loc, const std::string& message);
};

// Given a multi-value signature, process each value
inline void ForeachValueNumber(const SignTy& sign,
                               std::function<void(NumTy, size_t)> lambda) {
  std::regex valuePattern("#(-?\\d+)");
  auto begin = std::sregex_iterator(sign.begin(), sign.end(), valuePattern);
  auto end = std::sregex_iterator();

  size_t matchIndex = 0;
  for (auto i = begin; i != end; ++i, ++matchIndex) {
    std::smatch match = *i;
    std::string matchStr = match.str(1); // Capture the number part of the match
    NumTy number{std::stoi(matchStr)};

    // Call the passed lambda function with the extracted string and its
    // index
    lambda(number, matchIndex);
  }
}

} // end namespace valno

} // end namespace Choreo

#endif // __CHOREO_VALUE_NUMBERING_HPP__
