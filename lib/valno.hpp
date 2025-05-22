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

// retrieve the n-th element from the comma-seperated input string
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

class ShapeInference;
class ValueNumbering {
private:
  ShapeInference* visitor;
  std::vector<std::unordered_map<std::string, int>> expressionValueNumbers;
  std::vector<std::unordered_map<int, std::string>> valueNumberExpressions;
  ValBind::BindInfo<int> bind_info;

  std::vector<std::unordered_map<const AST::Node*, int>>
      nodeValueNumbers; // cache to direct map node to value number

  bool InternalHasExprValNo(const std::string& expr) {
    for (auto expr_valno = expressionValueNumbers.rbegin();
         expr_valno != expressionValueNumbers.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      return true;
    }
    return false;
  }

  int InternalGetExprValNo(const std::string& expr) {
    for (auto expr_valno = expressionValueNumbers.rbegin();
         expr_valno != expressionValueNumbers.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      return (*expr_valno)[expr];
    }
    choreo_unreachable("can not find valno of expression : " + expr + ".");
  }

  void InternalUpdateExprValNo(const std::string& expr, int val_no) {
    for (auto expr_valno = expressionValueNumbers.rbegin();
         expr_valno != expressionValueNumbers.rend(); expr_valno++) {
      if (!expr_valno->count(expr)) continue;
      (*expr_valno)[expr] = val_no;
      return;
    }
    assert(!expressionValueNumbers.empty() &&
           "empty expression value number map.");
    expressionValueNumbers.back()[expr] = val_no;
  }

  bool InternalHasValNoExpr(int vn) {
    for (auto valno_expr = valueNumberExpressions.rbegin();
         valno_expr != valueNumberExpressions.rend(); valno_expr++) {
      if (!valno_expr->count(vn)) continue;
      return true;
    }
    return false;
  }

  const std::string& InternalGetValNoExpr(int vn) {
    for (auto valno_expr = valueNumberExpressions.rbegin();
         valno_expr != valueNumberExpressions.rend(); valno_expr++) {
      if (!valno_expr->count(vn)) continue;
      return (*valno_expr)[vn];
    }
    choreo_unreachable(
        "can not find expression of valno: " + std::to_string(vn) + ".");
  }

  void InternalUpdateValNoExpr(int vn, const std::string& expr) {
    for (auto valno_expr = valueNumberExpressions.rbegin();
         valno_expr != valueNumberExpressions.rend(); valno_expr++) {
      if (!valno_expr->count(vn)) continue;
      (*valno_expr)[vn] = expr;
    }
    assert(!valueNumberExpressions.empty() &&
           "empty value number expression map.");
    valueNumberExpressions.back()[vn] = expr;
  }

  bool InternalHasNodeValNo(const AST::Node* node) {
    for (auto node_valno = nodeValueNumbers.rbegin();
         node_valno != nodeValueNumbers.rend(); node_valno++) {
      if (!node_valno->count(node)) continue;
      return true;
    }
    return false;
  }

  int InternalGetNodeValNo(const AST::Node* node) {
    for (auto node_valno = nodeValueNumbers.rbegin();
         node_valno != nodeValueNumbers.rend(); node_valno++) {
      if (!node_valno->count(node)) continue;
      return (*node_valno)[node];
    }
    choreo_unreachable("can not find valno of node: " + PSTR(node) + ".");
  }

  void InternalUpdateNodeValNo(const AST::Node* node, int vn) {
    for (auto node_valno = nodeValueNumbers.rbegin();
         node_valno != nodeValueNumbers.rend(); node_valno++) {
      if (!node_valno->count(node)) continue;
      (*node_valno)[node] = vn;
    }
    assert(!nodeValueNumbers.empty() && "empty node value number map.");
    nodeValueNumbers.back()[node] = vn;
  }

  int nextValueNumber = 0;

  bool trace = false;

  std::optional<std::string> ref = std::nullopt;

public:
  explicit ValueNumbering(ShapeInference* v)
      : visitor(v), trace(CCtx().TraceValueNumbers()) {}

  void EnterScope();
  void LeaveScope();

  void SetListReference(const std::string& r) { ref = r; }
  void ResetListReference() { ref.reset(); }

  // It binds a expression sigature with an existing value number.
  void AssociateSignatureWithValueNumber(const std::string& sig, int valno);
  void AssociateSignatureWithInvalidValueNumber(const std::string& sig);
  // rebind/modify the value number.
  // Caution: only used for scenario where the value number has not been
  // determined yet.
  void RebindSignatureWithValueNumber(const std::string& sig, int valno);

  std::optional<std::string> TryToSimplifyNodeSignature(const AST::Node& node);

  // Generate the signature for a node, simplify the signature when optimiz flag
  // is set.
  std::string GenerateNodeSignature(const AST::Node& node, bool optimiz = true);

  // special for bounded variables
  std::optional<std::string> GenerateSpecialNodeSignature(const AST::Node&);

  // Directly get the value number. Abort when it fails.
  int GetValueNumberForNode(const AST::Node&);

  // Generate the new value number. Abort when the value number exists.
  int GenerateValueNumberForNode(const AST::Node&);

  // Check if the value number exists for the node
  bool HasValueNumberForNode(const AST::Node&);

  // Directly get the value number from a signature. Abort when it fails.
  int GetValueNumberOfSignature(const std::string&);

  // Bind two value numbers
  void BindValueNumbers(int, int);

  // Bind two value numbers
  const ValBind::Binds<int>::Set& GetBindSet(int vn) {
    auto& ret = bind_info.GetSet(vn);
    return ret;
  }

  void AddBind(int vn0, int vn1) { bind_info.AddBind(vn0, vn1); }

  // Generate the new value number from a signature. Abort when the value number
  // exists.
  int GenerateValueNumberFromSignature(const std::string& signature);

  // Check if the value number exists for the signature
  bool HasValueNumberOfSignature(const std::string&);

  // Check if the value number exists and is valid for the signature
  bool HasValidValueNumberOfSignature(const std::string&);

  int GetOrInsertValueNumberFromSignature(const std::string& signature);

  // Symbol names related to the value numbering
  const std::string VNSymbolName(const AST::Identifier&) const;

  // Retrieve the signature from a value number. About when fails.
  std::string GetSignatureFromValueNumber(int vn) {
    if (vn == UnknownValue()) return "?";

    if (!InternalHasValNoExpr(vn))
      choreo_unreachable("value number " + std::to_string(vn) +
                         " does not exists in the value number table.");
    return InternalGetValNoExpr(vn);
  }

  std::string SignatureOfSymbol(std::string sym) {
    return GetSignatureFromValueNumber(GetValueNumberOfSignature(sym));
  }

  std::string GetSignatureForNode(const AST::Node& n) {
    return GetSignatureFromValueNumber(GetValueNumberForNode(n));
  }

  void Print(std::ostream& os) {
    int scope = 0;
    for (auto& stack : expressionValueNumbers) {
      os << scope++ << "\n";
      for (auto& item : stack)
        os << "expr: \"" << item.first << "\", value_no: #" << item.second
           << "\n";
    }
  }

  std::optional<std::string>
  SignBoundedOperation(const location&, const std::string&, const AST::Node&,
                       const AST::Node&, bool verbose);

  std::optional<std::string>
  TryToSimplifyBinary(const location&, const std::string&, const std::string&,
                      const std::string&, bool = false);

  std::string SignBinaryCompositeValues(const location&, const std::string&,
                                        const std::string&, const std::string&,
                                        bool = false);

  std::string ValueItemToSignature(const ValueItem&, bool);

private:
  std::string ScopeIndent();

  void Error(const location& loc, const std::string& message);
  void Warning(const location& loc, const std::string& message);
};

} // end namespace Choreo

#endif // __CHOREO_VALUE_NUMBERING_HPP__
