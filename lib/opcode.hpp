#ifndef __CHOREO_OPCODE_HPP__
#define __CHOREO_OPCODE_HPP__

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>

#include "aux.hpp"

namespace Choreo {

struct Opcode {
  enum class Kind : uint8_t {
    None,
    Invalid,
    Ref,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    CeilDiv,
    LogicOr,
    LogicAnd,
    LogicNot,
    BitOr,
    BitAnd,
    BitXor,
    BitNot,
    Shl,
    Shr,
    UBound,
    UBoundAdd,
    UBoundSub,
    UBoundScale,
    UBoundDiv,
    UBoundMod,
    Lt,
    Gt,
    Eq,
    Ne,
    Le,
    Ge,
    Select,
    DimOf,
    GetIth,
    GetUBound,
    SizeOf,
    DataOf,
    MDataOf,
    AddrOf,
    PreInc,
    PreDec,
    Cast,
    Concat,
    ElemOf,
    UBoundInternal,
    UBoundAddInternal,
    UBoundSubInternal,
  };

private:
  Kind kind = Kind::Invalid;

public:
  constexpr Opcode() = default;
  constexpr Opcode(Kind k) : kind(k) {}
  explicit Opcode(std::string_view op) : kind(Parse(op)) {}

  constexpr Kind GetKind() const { return kind; }

  Opcode& operator=(Kind k) {
    kind = k;
    return *this;
  }

  constexpr bool operator==(Kind k) const { return kind == k; }
  constexpr bool operator!=(Kind k) const { return kind != k; }
  constexpr bool operator==(const Opcode& other) const {
    return kind == other.kind;
  }
  constexpr bool operator!=(const Opcode& other) const {
    return kind != other.kind;
  }
  constexpr bool operator<(const Opcode& other) const {
    return static_cast<uint8_t>(kind) < static_cast<uint8_t>(other.kind);
  }
  bool operator==(std::string_view other) const { return kind == Parse(other); }
  bool operator!=(std::string_view other) const { return kind != Parse(other); }

  static Kind Parse(std::string_view op) {
    if (op.empty()) return Kind::None;
    if (op == "ref") return Kind::Ref;
    if (op == "+") return Kind::Add;
    if (op == "-") return Kind::Sub;
    if (op == "*") return Kind::Mul;
    if (op == "/") return Kind::Div;
    if (op == "%") return Kind::Mod;
    if (op == "cdiv") return Kind::CeilDiv;
    if (op == "||") return Kind::LogicOr;
    if (op == "&&") return Kind::LogicAnd;
    if (op == "!") return Kind::LogicNot;
    if (op == "|") return Kind::BitOr;
    if (op == "&") return Kind::BitAnd;
    if (op == "^") return Kind::BitXor;
    if (op == "~") return Kind::BitNot;
    if (op == "<<") return Kind::Shl;
    if (op == ">>") return Kind::Shr;
    // `#`, `#+`, ... are source-level bound operators.
    if (op == "#") return Kind::UBound;
    if (op == "#+") return Kind::UBoundAdd;
    if (op == "#-") return Kind::UBoundSub;
    if (op == "#*") return Kind::UBoundScale;
    if (op == "#/") return Kind::UBoundDiv;
    if (op == "#%") return Kind::UBoundMod;
    if (op == "<") return Kind::Lt;
    if (op == ">") return Kind::Gt;
    if (op == "==") return Kind::Eq;
    if (op == "!=") return Kind::Ne;
    if (op == "<=") return Kind::Le;
    if (op == ">=") return Kind::Ge;
    if (op == "?") return Kind::Select;
    if (op == "dimof") return Kind::DimOf;
    if (op == "getith") return Kind::GetIth;
    if (op == "ubound") return Kind::GetUBound;
    if (op == "sizeof") return Kind::SizeOf;
    if (op == "dataof") return Kind::DataOf;
    if (op == "mdataof") return Kind::MDataOf;
    if (op == "addrof") return Kind::AddrOf;
    if (op == "++") return Kind::PreInc;
    if (op == "--") return Kind::PreDec;
    if (op == "cast") return Kind::Cast;
    if (op == "concat") return Kind::Concat;
    if (op == "elemof") return Kind::ElemOf;
    // `@`, `@+`, `@-` are value-numbering-only spellings used so printed
    // value numbers are not confused with the `#<valno>` prefix.
    if (op == "@") return Kind::UBoundInternal;
    if (op == "@+") return Kind::UBoundAddInternal;
    if (op == "@-") return Kind::UBoundSubInternal;
    choreo_unreachable("unexpected opcode '" + std::string(op) + "'.");
  }

  std::string ToString() const {
    switch (kind) {
    case Kind::None: return "";
    case Kind::Invalid: return "<invalid-opcode>";
    case Kind::Ref: return "ref";
    case Kind::Add: return "+";
    case Kind::Sub: return "-";
    case Kind::Mul: return "*";
    case Kind::Div: return "/";
    case Kind::Mod: return "%";
    case Kind::CeilDiv: return "cdiv";
    case Kind::LogicOr: return "||";
    case Kind::LogicAnd: return "&&";
    case Kind::LogicNot: return "!";
    case Kind::BitOr: return "|";
    case Kind::BitAnd: return "&";
    case Kind::BitXor: return "^";
    case Kind::BitNot: return "~";
    case Kind::Shl: return "<<";
    case Kind::Shr: return ">>";
    case Kind::UBound: return "#";
    case Kind::UBoundAdd: return "#+";
    case Kind::UBoundSub: return "#-";
    case Kind::UBoundScale: return "#*";
    case Kind::UBoundDiv: return "#/";
    case Kind::UBoundMod: return "#%";
    case Kind::Lt: return "<";
    case Kind::Gt: return ">";
    case Kind::Eq: return "==";
    case Kind::Ne: return "!=";
    case Kind::Le: return "<=";
    case Kind::Ge: return ">=";
    case Kind::Select: return "?";
    case Kind::DimOf: return "dimof";
    case Kind::GetIth: return "getith";
    case Kind::GetUBound: return "ubound";
    case Kind::SizeOf: return "sizeof";
    case Kind::DataOf: return "dataof";
    case Kind::MDataOf: return "mdataof";
    case Kind::AddrOf: return "addrof";
    case Kind::PreInc: return "++";
    case Kind::PreDec: return "--";
    case Kind::Cast: return "cast";
    case Kind::Concat: return "concat";
    case Kind::ElemOf: return "elemof";
    case Kind::UBoundInternal: return "@";
    case Kind::UBoundAddInternal: return "@+";
    case Kind::UBoundSubInternal: return "@-";
    }
    choreo_unreachable("unhandled opcode.");
  }
};

using Op = Opcode::Kind;

inline constexpr bool IsArith(const Opcode& op) {
  switch (op.GetKind()) {
  case Op::Add:
  case Op::Sub:
  case Op::Mul:
  case Op::Div:
  case Op::Mod:
  case Op::CeilDiv:
  case Op::UBound:
  case Op::UBoundAdd:
  case Op::UBoundSub:
  case Op::UBoundScale:
  case Op::UBoundDiv:
  case Op::UBoundMod:
  case Op::UBoundInternal:
  case Op::UBoundAddInternal:
  case Op::UBoundSubInternal: return true;
  default: return false;
  }
}

inline constexpr bool IsLogical(const Opcode& op) {
  switch (op.GetKind()) {
  case Op::LogicOr:
  case Op::LogicAnd:
  case Op::LogicNot: return true;
  default: return false;
  }
}

inline constexpr bool IsCompare(const Opcode& op) {
  switch (op.GetKind()) {
  case Op::Lt:
  case Op::Le:
  case Op::Eq:
  case Op::Gt:
  case Op::Ge:
  case Op::Ne: return true;
  default: return false;
  }
}

inline constexpr bool IsBitwise(const Opcode& op) {
  switch (op.GetKind()) {
  case Op::BitAnd:
  case Op::BitOr:
  case Op::BitXor:
  case Op::BitNot:
  case Op::Shl:
  case Op::Shr: return true;
  default: return false;
  }
}

inline constexpr bool IsUBArith(const Opcode& op) {
  switch (op.GetKind()) {
  case Op::UBound:
  case Op::UBoundAdd:
  case Op::UBoundSub:
  case Op::UBoundScale:
  case Op::UBoundDiv:
  case Op::UBoundMod:
  case Op::UBoundInternal:
  case Op::UBoundAddInternal:
  case Op::UBoundSubInternal: return true;
  default: return false;
  }
}

inline std::ostream& operator<<(std::ostream& os, const Opcode& op) {
  os << op.ToString();
  return os;
}

inline std::string STR(const Opcode& op) { return op.ToString(); }

inline bool operator==(std::string_view lhs, const Opcode& rhs) {
  return rhs == lhs;
}

inline bool operator!=(std::string_view lhs, const Opcode& rhs) {
  return rhs != lhs;
}

inline std::string operator+(std::string lhs, const Opcode& rhs) {
  lhs += rhs.ToString();
  return lhs;
}

inline std::string operator+(const Opcode& lhs, std::string rhs) {
  return lhs.ToString() + rhs;
}

} // namespace Choreo

namespace std {

template <>
struct hash<Choreo::Opcode> {
  size_t operator()(const Choreo::Opcode& op) const noexcept {
    return std::hash<int>{}(static_cast<int>(op.GetKind()));
  }
};

} // namespace std

#endif // __CHOREO_OPCODE_HPP__
