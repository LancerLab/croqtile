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
    // UNSUED means that the op is printed by oss directly, not OpExprSTR

    // clang-format off
    {"",       {1, Assoc::Left}},
    {"?",      {1, Assoc::Right}},  // C ? L : R

    {"||",     {2, Assoc::Left}},

    {"&&",     {3, Assoc::Left}},

    {"<",      {4, Assoc::Left}},
    {"<=",     {4, Assoc::Left}},
    {"==",     {4, Assoc::Left}},
    {">",      {4, Assoc::Left}},
    {">=",     {4, Assoc::Left}},
    {"!=",     {4, Assoc::Left}},

    {"+",      {5, Assoc::Left}},
    {"-",      {5, Assoc::Left}},

    {"getith", {6, Assoc::Left}}, // bv(idx) => `Value(bv)+idx` if idx<0
                                  // bv(idx) => `Value(bv) if idx>=0

    {"#",      {7, Assoc::Left}}, // L * UB(r) + R

    {"*",      {8, Assoc::Left}},
    {"/",      {8, Assoc::Left}},
    {"%",      {8, Assoc::Left}},

    // TODO: test
    {"cdiv",   {9, Assoc::Left}}, // (L + R - 1) / R

    {"!",      {10, Assoc::Right}},
    {"addrof", {10, Assoc::Right}},
    {"++",     {10, Assoc::Right}},
    {"--",     {10, Assoc::Right}},

    {"elemof", {11, Assoc::Left}}, // array[x][x]
    {"ref",    {11, Assoc::Left}},
    {"sizeof", {11, Assoc::Left}}, // UNUSED: Value(...)
    {"ubound", {11, Assoc::Left}}, // UNUSED: value(ub(bv))
    {"dataof", {11, Assoc::Left}}, // UNUSED: future.data => id__buf__ or id.data()

    {"#+",     {12, Assoc::Left}}, // specia case: UB arith.
    {"#-",     {12, Assoc::Left}}, // res is only related to the original bv,
    {"#*",     {12, Assoc::Left}}, // and contains no operator.
    {"#/",     {12, Assoc::Left}},
    {"#%",     {12, Assoc::Left}},
    // the res will be gain from shapeinfer, so no op "dimof" at CodeGen.
    // {"dimof",  {x, Assoc::Left}},  // mdspan(idx)
    
    // note: have no compound arith assign: "+=", "-=", ...
    // they are normalized to "x = x + y"

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

  return false;
}

} // namespace Operator

} // end namespace Choreo

#endif //__CHOREO_OPERATOR_INFO_HPP__
