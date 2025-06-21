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

inline constexpr int UnknownValue() { return -1; }
inline bool ValidVN(int vn) { return IsValidValueNumber(vn); }
inline void InvalidateVN(int& vn) { vn = GetInvalidValueNumber(); }
inline bool UnknownVN(int vn) { return vn == UnknownValue(); }
inline void SetUnknownVN(int& vn) { vn = UnknownValue(); }

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
using NumTy = int;          // value number type.

inline const std::string STR(const SignTy& s) { return s; }
inline const std::string STR(const NumTy& v) { return std::to_string(v); }
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
  ValueNumberTable(bool t = false) : trace(t) {}

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
    if (Exists(s) && ValidVN(GetValueNum(s)))
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

    if (ValidVN(GetValueNum(s)))
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
      if (!ValidVN(item.second)) continue;

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
        os << "expr: \"" << item.first << "\", valno: #" << item.second << "\n";
    }
    for (auto& item : const_pool)
      os << "const: \"" << item.first << "\", valno: #" << item.second << "\n";
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
  const SignTy NumCharToSign(const std::string&) const;

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

  NumTy GetOrGenValueNumberFromSignature(const SignTy& signature);

  // Retrieve the signature from a value number. About when fails.
  SignTy GetSignatureFromValueNumber(NumTy vn) const {
    if (vn == UnknownValue()) return "?";

    if (!vntbl.Exists(vn))
      choreo_unreachable("value number " + std::to_string(vn) +
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

  SignTy SignBinaryCompositeValues(const location&, const SignTy&,
                                   const SignTy&, const SignTy&, bool = false);

  ValueItem GenValueItemFromSignature(const SignTy&);
  ValueItem GenValueItemFromValueNumber(NumTy);
  const ValueList GenValueListFromSignature(const SignTy&);
  const ValueList GenValueListFromValueNumber(NumTy);
  SignTy ValueItemToSignature(const ValueItem&, bool = false);
  SignTy ValueListToSignature(const ValueList&, bool = true);

public:
  ValBind::BindInfo<NumTy> bind_info; // TODO: to abondon

  void BindValueNumbers(NumTy vn0, NumTy vn1) {
    assert(vntbl.Exists(vn0) && vntbl.Exists(vn1) &&
           "invalid value number is provided.");

    AddBind(vn0, vn1);
  }

  // Bind two value numbers
  const ValBind::Binds<int>::Set& GetBindSet(int vn) {
    auto& ret = bind_info.GetSet(vn);
    return ret;
  }

  void AddBind(int vn0, int vn1) { bind_info.AddBind(vn0, vn1); }

public:
  // retrieve the n-th element from the comma-separated input string
  NumTy GetNthValNo(const SignTy& input, NumTy n) const;
  const std::vector<NumTy> Flatten(NumTy) const;

  std::string ScopeIndent();

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
    assert(ValidVN(valno) && "not a valid value number.");
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
    int number = std::stoi(matchStr);

    // Call the passed lambda function with the extracted string and its
    // index
    lambda(number, matchIndex);
  }
}

} // end namespace valno

} // end namespace Choreo

#endif // __CHOREO_VALUE_NUMBERING_HPP__
