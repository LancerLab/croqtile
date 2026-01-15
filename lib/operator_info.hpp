#ifndef __CHOREO_OPERATOR_INFO_HPP__
#define __CHOREO_OPERATOR_INFO_HPP__

#include <unordered_map>

#include "aux.hpp"

namespace Choreo {

namespace Operator {

// TODO: should < be nonassoc? a < b < c
enum class Assoc { Left, Right, Nonassoc };

struct OpInfo {
  size_t precedence;
  Assoc assoc;
};

// https://en.cppreference.com/w/cpp/language/operator_precedence.html

inline const std::unordered_map<std::string, OpInfo> op_table = {
    // L = ExprSTR(l), R = ExprSTR(r), C = ExprSTR(c)
    // Value(x) = ValueSTR(x) is always valid to use directly (already wrapped
    // with parentheses if needed).

    // In fact, custom operators(e.g. sizeof, getith, cdiv...) can be classified
    // as native C++ operators. Therefore, their priority is not necessary here.
    // If these customoperators are encountered,then the corresponding native
    // C++ operators will be used instead.That is,their specific priorities
    // depend on the implementation rather than theoriginal definition.

    // clang-format off
    {"",       {0, Assoc::Left}},   // dummy operator with the lowest priority 

    {"?",      {1, Assoc::Right}},  // C ? L : R

    {"||",     {2, Assoc::Left}},

    {"&&",     {3, Assoc::Left}},

    {"|",      {4, Assoc::Left}},
    
    {"^",      {5, Assoc::Left}},
    
    {"&",      {6, Assoc::Left}},

    {"==",     {7, Assoc::Left}},
    {"!=",     {7, Assoc::Left}},
    
    {"<",      {8, Assoc::Left}},
    {"<=",     {8, Assoc::Left}},
    {">",      {8, Assoc::Left}},
    {">=",     {8, Assoc::Left}},
    
    {"<<",     {9, Assoc::Left}},
    {">>",     {9, Assoc::Left}},

    {"+",      {10, Assoc::Left}},
    {"-",      {10, Assoc::Left}},
    
    {"*",      {11, Assoc::Left}},
    {"/",      {11, Assoc::Left}},
    {"%",      {11, Assoc::Left}},

    {"++",     {12, Assoc::Right}},   // Prefix increment
    {"--",     {12, Assoc::Right}},   // Prefix decrement
    {"!",      {12, Assoc::Right}},
    {"~",      {12, Assoc::Right}},
    {"&",      {12,  Assoc::Right}},  // address-of
    
    {"[]",     {13,  Assoc::Right}},  // subscript, seems did not use
    // {"suf_++",     {x, x}} // suffix increment         
    // {"suf_--",     {x, x}} // suffix decrement
    
    /*
    {"#",      {x, Assoc::Left}}, // L * UB(r) + R
    {"getith", {x, Assoc::Left}}, // bv(idx) => `ub(bv)+idx` if idx<0
                                  // bv(idx) => `idx`        if idx>=0
    {"cdiv",   {x, Assoc::Left}}, // (L + R - 1) / R
    {"addrof", {x, Assoc::Right}},

    {"elemof", {x, Assoc::Left}}, // array[x][x]
    {"ref",    {x, Assoc::Left}},
    {"sizeof", {x, Assoc::Left}}, // UNUSED: Value(...)
    {"ubound", {x, Assoc::Left}}, // UNUSED: Value(ub(bv))
    {"dataof", {x, Assoc::Left}}, // UNUSED: future.data => id__buf__ or id.data()
    {"mdataof", {x, Assoc::Left}}, // UNUSED: future.mdata => id__mdata__ or id.mdata()

    {"#+",     {x, Assoc::Left}}, // the ub is changed
    {"#-",     {x, Assoc::Left}}, 
    {"#*",     {x, Assoc::Left}}, 
    {"#/",     {x, Assoc::Left}},
    {"#%",     {x, Assoc::Left}},

    the res will be gain from shapeinfer, so no op "dimof" at CodeGen.
    {"dimof",  {x, Assoc::Left}},  // mdspan(idx)
    
    note: have no compound arith assign: "+=", "-=", ...
    they are normalized to "x = x + y"
    */
    // clang-format on
};

inline size_t GetPrecedence(const std::string& op) {
  auto it = op_table.find(op);
  if (it == op_table.end())
    choreo_unreachable("unexpected op '" + op +
                       "' which cannot be found in op table.");
  return it->second.precedence;
}

inline Assoc GetAssociativity(const std::string& op) {
  auto it = op_table.find(op);
  if (it == op_table.end())
    choreo_unreachable("unexpected op '" + op +
                       "' which cannot be found in op table.");
  return it->second.assoc;
}

// child_op is the current op.
inline bool NeedParen(const std::string& child_op, const std::string& parent_op,
                      bool is_left_child = true) {
  size_t child_prec = GetPrecedence(child_op);
  size_t parent_prec = GetPrecedence(parent_op);

  if (child_prec > parent_prec) return false;
  if (child_prec < parent_prec) return true;

  Assoc parent_assoc = GetAssociativity(parent_op);
  if (parent_assoc == Assoc::Left && !is_left_child) return true;
  if (parent_assoc == Assoc::Right && is_left_child) return true;

  // special case: make clear output if nested ? expr
  if (child_op == "?" && parent_op == "?") return true;

  return false;
}

} // namespace Operator

} // end namespace Choreo

#endif //__CHOREO_OPERATOR_INFO_HPP__
