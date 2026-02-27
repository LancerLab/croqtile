#ifndef __CHOREO_TYPES_H__
#define __CHOREO_TYPES_H__

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>

// to avoid definition error
namespace Choreo {
enum class Storage {
  REG, /* register, normally not explicit */
  LOCAL,
  SHARED,
  GLOBAL /*device global*/,
  NODE /*cluster node*/,
  DEFAULT,
  NONE
};
enum class ParallelLevel {
  THREAD,
  GROUP,
  GROUPx4,
  BLOCK,
  DEVICE,
  TERM /* terminal machine in cluster*/,
  SEQ,
  NONE /*bottom*/,
  UNKNOWN /*top*/
};
enum class CompileTarget;
} // namespace Choreo

#include "aux.hpp"
#include "context.hpp"
#include "infra_utils.hpp"

namespace Choreo {

enum class BaseType {
  F64,
  F32,
  TF32,
  F16,
  BF16,
  F8_E4M3,
  F8_E5M2,
  F8_UE8M0,
  F8_UE4M3,
  F6_E2M3,
  F6_E3M2,
  F4_E2M1,
  U64,
  S64,
  U32,
  S32,
  U16,
  S16,
  U8,
  S8,
  U6,
  S6,
  U4,
  S4,
  U2,
  S2,
  BIN1,
  U1,
  // NUMERIC TYPES ENDS
  BOOL,
  UNKSCALAR, // unknown scalar, must be inferred
  // SCALAR TYPES ENDS
  INDEX,
  ITUPLE,
  PARTIAL,
  EVENT,
  ARRAY,
  VECTOR,
  ADDR,
  VOID,
  SPANNED,
  BOUNDED_INT,
  BOUNDED_ITUPLE,
  FUTURE,
  STRING,
  STREAM,
  FUNCTION,
  DEVICE,
  UNKNOWN, // must be inferred
  UNSPECVAL,
};

inline static int BaseTypeRank(BaseType bt) {
  switch (bt) {
  case BaseType::U64: return 15;
  case BaseType::S64: return 14;
  case BaseType::U32: return 13;
  case BaseType::S32: return 12;
  case BaseType::U16: return 11;
  case BaseType::S16: return 10;
  case BaseType::U8: return 9;
  case BaseType::S8: return 8;
  case BaseType::U6: return 7;
  case BaseType::S6: return 6;
  case BaseType::U4: return 5;
  case BaseType::S4: return 4;
  case BaseType::U2: return 3;
  case BaseType::S2: return 2;
  case BaseType::U1: return 1;
  case BaseType::BOOL: return 1;
  case BaseType::UNKSCALAR: return 0;
  case BaseType::UNKNOWN: return 0;
  case BaseType::UNSPECVAL: return 0;
  default: return -1;
  }
}

inline static bool ApprxEqual(BaseType lty, BaseType rty) {
  if (lty == BaseType::UNKNOWN || rty == BaseType::UNKNOWN) return true;
  return lty == rty;
}

inline static bool ApprxEqualRank(BaseType lty, BaseType rty) {
  if (lty == BaseType::UNKNOWN || rty == BaseType::UNKNOWN) return true;
  return lty == rty;
}

inline static bool IsIntegerType(BaseType bt) {
  return bt == BaseType::S64 || bt == BaseType::U64 || bt == BaseType::S32 ||
         bt == BaseType::U32 || bt == BaseType::S16 || bt == BaseType::U16 ||
         bt == BaseType::S8 || bt == BaseType::U8 || bt == BaseType::S6 ||
         bt == BaseType::U6 || bt == BaseType::S4 || bt == BaseType::U4 ||
         bt == BaseType::S2 || bt == BaseType::U2 || bt == BaseType::U1 ||
         bt == BaseType::BIN1;
}

// integrals include integer and bool
inline static bool IsIntegralType(BaseType bt) {
  return bt == BaseType::BOOL || IsIntegerType(bt);
}

inline static bool IsSignedType(BaseType bt) {
  assert(IsIntegerType(bt));
  return bt == BaseType::S64 || bt == BaseType::S32 || bt == BaseType::S16 ||
         bt == BaseType::S8 || bt == BaseType::S6 || bt == BaseType::S4 ||
         bt == BaseType::S2 || bt == BaseType::BIN1;
}

inline static bool IsUnsignedType(BaseType bt) {
  assert(IsIntegerType(bt));
  return !IsSignedType(bt);
}

inline static bool IsFloatType(BaseType bt) {
  return bt == BaseType::F64 || bt == BaseType::F32 || bt == BaseType::TF32 ||
         bt == BaseType::F16 || bt == BaseType::BF16 ||
         bt == BaseType::F8_E4M3 || bt == BaseType::F8_E5M2 ||
         bt == BaseType::F8_UE8M0 || bt == BaseType::F8_UE4M3 ||
         bt == BaseType::F6_E2M3 || bt == BaseType::F6_E3M2 ||
         bt == BaseType::F4_E2M1;
}

// note: NumericType does not contain bool
inline static bool IsNumericType(BaseType bt) {
  return IsIntegerType(bt) || IsFloatType(bt);
}

inline static bool IsScalarType(BaseType bt) {
  return IsNumericType(bt) || bt == BaseType::BOOL || bt == BaseType::UNKSCALAR;
}

inline static bool IsFloatSubByteType(BaseType bt) {
  return bt == BaseType::F6_E2M3 || bt == BaseType::F6_E3M2 ||
         bt == BaseType::F4_E2M1;
}

inline static bool IsIntegerSubByteType(BaseType bt) {
  return bt == BaseType::S6 || bt == BaseType::U6 || bt == BaseType::S4 ||
         bt == BaseType::U4 || bt == BaseType::S2 || bt == BaseType::U2 ||
         bt == BaseType::U1 || bt == BaseType::BIN1;
}

inline static bool IsSubByteType(BaseType bt) {
  return IsFloatSubByteType(bt) || IsIntegerSubByteType(bt);
}

inline static bool Compatible(const Storage& a, const Storage& b) {
  if (a == Storage::DEFAULT || a == Storage::GLOBAL)
    return (b == Storage::DEFAULT || b == Storage::GLOBAL);
  else
    return a == b;
}

inline static Storage ProjectStorage(const Storage& a) {
  if (a == Storage::DEFAULT)
    return Storage::GLOBAL;
  else
    return a;
}

enum class ParamAttr : uint16_t {
  NONE = 0,
  SHADOW_TO_GLOBAL = 1, // shadow the host memory to global
  GLOBAL_INPUT = 2,     // it is a global buffer
};

inline static std::string STR(ParamAttr at) {
  switch (at) {
  case ParamAttr::NONE: return "";
  case ParamAttr::SHADOW_TO_GLOBAL: return "shadow";
  case ParamAttr::GLOBAL_INPUT: return "global-input";
  default: choreo_unreachable("unknown parameter attribute.");
  }
  return "";
}

inline static size_t SizeOf(BaseType bt) {
  if (!IsScalarType(bt)) { choreo_unreachable("base type is not scalar."); }
  switch (bt) {
  case BaseType::F64: return sizeof(double);
  case BaseType::U64:
  case BaseType::S64: return 8;
  case BaseType::BOUNDED_INT:
  case BaseType::F32:
  case BaseType::TF32:
  case BaseType::U32:
  case BaseType::S32: return 4;
  case BaseType::F16:
  case BaseType::BF16:
  case BaseType::U16:
  case BaseType::S16: return 2;
  case BaseType::F8_E4M3:
  case BaseType::F8_E5M2:
  case BaseType::F8_UE8M0:
  case BaseType::F8_UE4M3:
  case BaseType::U8:
  case BaseType::S8:
  case BaseType::U6:
  case BaseType::S6:
  case BaseType::F6_E3M2:
  case BaseType::F6_E2M3:
  case BaseType::F4_E2M1:
  case BaseType::U4:
  case BaseType::S4:
  case BaseType::U2:
  case BaseType::S2: return 1;
  case BaseType::BIN1:
  case BaseType::U1:
  case BaseType::BOOL: return sizeof(bool);
  default: break;
  }
  return 0;
}

// utility functions to map types to strings, and the opposite.
inline static BaseType BaseTypeFromString(const std::string& input) {
  static const std::unordered_map<std::string, BaseType> typeMap = {
      {"f64", BaseType::F64},
      {"tf32", BaseType::TF32},
      {"f32", BaseType::F32},
      {"f16", BaseType::F16},
      {"bf16", BaseType::BF16},
      {"f8_e4m3", BaseType::F8_E4M3},
      {"f8_e5m2", BaseType::F8_E5M2},
      {"f8_ue8m0", BaseType::F8_UE8M0},
      {"f8_ue4m3", BaseType::F8_UE4M3},
      {"f6_e2m3", BaseType::F6_E2M3},
      {"f6_e3m2", BaseType::F6_E3M2},
      {"f4_e2m1", BaseType::F4_E2M1},
      {"u64", BaseType::U64},
      {"s64", BaseType::S64},
      {"u32", BaseType::U32},
      {"s32", BaseType::S32},
      {"u16", BaseType::U16},
      {"s16", BaseType::S16},
      {"u8", BaseType::U8},
      {"s8", BaseType::S8},
      {"u6", BaseType::U6},
      {"s6", BaseType::S6},
      {"u4", BaseType::U4},
      {"s4", BaseType::S4},
      {"u2", BaseType::U2},
      {"s2", BaseType::S2},
      {"bin1", BaseType::BIN1},
      {"u1", BaseType::U1},
      {"bool", BaseType::BOOL},
      {"index", BaseType::INDEX},
      {"partial", BaseType::PARTIAL},
      {"spanned", BaseType::SPANNED},
      {"bounded_int", BaseType::BOUNDED_INT},
      {"bounded_ituple", BaseType::BOUNDED_ITUPLE},
      {"FUTURE", BaseType::FUTURE},
      {"string", BaseType::STRING},
      {"stream", BaseType::STREAM},
      {"function", BaseType::FUNCTION},
      {"device", BaseType::DEVICE},
      {"index", BaseType::INDEX},
      {"ituple", BaseType::ITUPLE},
      {"event", BaseType::EVENT},
      {"array", BaseType::ARRAY},
      {"vector", BaseType::VECTOR},
      {"address", BaseType::ADDR},
      {"void", BaseType::VOID},
      {"unk", BaseType::UNKSCALAR},
      {"unknown", BaseType::UNKNOWN},
      {"inf", BaseType::UNSPECVAL},
  };

  auto it = typeMap.find(input);
  assert(it != typeMap.end() && "incorrect type string");

  return it->second;
}

// map from common device type string to BaseType
inline static BaseType DSTR2BT(std::string type_name) {
  static const std::map<std::string, BaseType> known_types = {
      {"char", BaseType::S8},          {"short", BaseType::S16},
      {"int", BaseType::S32},          {"long long", BaseType::S64},
      {"long long", BaseType::S64},    {"bool", BaseType::BOOL},
      {"unsigned char", BaseType::U8}, {"unsigned short", BaseType::U16},
      {"unsigned int", BaseType::U32}, {"unsigned long long", BaseType::U64},
      {"int8_t", BaseType::S16},       {"int16_t", BaseType::S16},
      {"int32_t", BaseType::S32},      {"int64_t", BaseType::S64},
      {"uint8_t", BaseType::U16},      {"uint16_t", BaseType::U16},
      {"uint32_t", BaseType::U32},     {"size_t", BaseType::U32},
      {"uint64_t", BaseType::U64},     {"float", BaseType::F32},
      {"double", BaseType::F64},       {"__bf16", BaseType::BF16},
      {"__fp16", BaseType::F16}};

  if (known_types.find(type_name) != known_types.end()) {
    return known_types.at(type_name);
  } else {
    return BaseType::UNKNOWN;
  }
}

namespace __internal__ {

inline static std::string GetStringFrom(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F64, "f64"},
      {BaseType::F32, "f32"},
      {BaseType::TF32, "tf32"},
      {BaseType::F16, "f16"},
      {BaseType::BF16, "bf16"},
      {BaseType::U64, "u64"},
      {BaseType::S64, "s64"},
      {BaseType::U32, "u32"},
      {BaseType::S32, "s32"},
      {BaseType::U16, "u16"},
      {BaseType::S16, "s16"},
      {BaseType::F8_E4M3, "f8_e4m3"},
      {BaseType::F8_E5M2, "f8_e5m2"},
      {BaseType::F8_UE8M0, "f8_ue8m0"},
      {BaseType::F8_UE4M3, "f8_ue4m3"},
      {BaseType::F6_E2M3, "f6_e2m3"},
      {BaseType::F6_E3M2, "f6_e3m2"},
      {BaseType::F4_E2M1, "f4_e2m1"},
      {BaseType::U8, "u8"},
      {BaseType::S8, "s8"},
      {BaseType::U6, "u6"},
      {BaseType::S6, "s6"},
      {BaseType::U4, "u4"},
      {BaseType::S4, "s4"},
      {BaseType::U2, "u2"},
      {BaseType::S2, "s2"},
      {BaseType::U1, "u1"},
      {BaseType::BIN1, "bin1"},
      {BaseType::BOOL, "bool"},
      {BaseType::ITUPLE, "ituple"},
      {BaseType::PARTIAL, "partial"},
      {BaseType::SPANNED, "spanned"},
      {BaseType::BOUNDED_INT, "bounded_int"},
      {BaseType::BOUNDED_ITUPLE, "bounded_ituple"},
      {BaseType::FUTURE, "FUTURE"},
      {BaseType::STRING, "string"},
      {BaseType::STREAM, "stream"},
      {BaseType::FUNCTION, "function"},
      {BaseType::DEVICE, "device"},
      {BaseType::INDEX, "index"},
      {BaseType::EVENT, "event"},
      {BaseType::ARRAY, "array"},
      {BaseType::VECTOR, "vector"},
      {BaseType::ADDR, "address"},
      {BaseType::VOID, "void"},
      {BaseType::UNKSCALAR, "unk"},
      {BaseType::UNKNOWN, "unknown"},
      {BaseType::UNSPECVAL, "inf"},
  };

  auto it = enumToString.find(dataType);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

inline static std::string GetStringFrom(Storage st) {
  static const std::unordered_map<Storage, std::string> enumToString = {
      {Storage::REG, "register"},    {Storage::LOCAL, "local"},
      {Storage::SHARED, "shared"},   {Storage::GLOBAL, "global"},
      {Storage::NODE, "node"},       {Storage::NONE, "none"},
      {Storage::DEFAULT, "default"},
  };

  auto it = enumToString.find(st);
  assert(it != enumToString.end() && "unsupported storage.");

  return it->second;
}

inline static std::string GetStringFrom(ParallelLevel st) {
  static const std::unordered_map<ParallelLevel, std::string> enumToString = {
      {ParallelLevel::THREAD, "thread"},   {ParallelLevel::GROUP, "group"},
      {ParallelLevel::GROUPx4, "group-4"}, {ParallelLevel::BLOCK, "block"},
      {ParallelLevel::DEVICE, "device"},   {ParallelLevel::TERM, "term"},
      {ParallelLevel::SEQ, "sequential"},  {ParallelLevel::NONE, "none"},
      {ParallelLevel::UNKNOWN, "unknown"},
  };

  auto it = enumToString.find(st);
  assert(it != enumToString.end() && "unsupported parallel level.");

  return it->second;
}

} // end namespace __internal__

inline static const std::string STR(size_t sz) { return std::to_string(sz); }
inline static const std::string STR(BaseType bt) {
  return __internal__::GetStringFrom(bt);
}
inline static const std::string STR(Storage st) {
  return __internal__::GetStringFrom(st);
}
inline static const std::string STR(ParallelLevel pl) {
  return __internal__::GetStringFrom(pl);
}

inline static std::ostream& operator<<(std::ostream& os, BaseType bt) {
  return os << STR(bt);
}

/*
  suppose
  f8:   E4M3:   <= 240
        E5M2:   <= 57344
    note: f8 may or may not have implicit leading bit!
  f16:  E5M11:  <= 65504
  bf16: E8M8:   <= 3.38×10^38
  f32:  E8M24:  <= 3.4×10^38
  f64:  E11M53: <= 1.79×10^308

  U8:  [0, 255]
  S8:  [-128, 127]
  U16: [0, 2^16-1]
  S16: [-2^8, 2^8-1]
  U32: [0, 2^32-1]
  S32: [-2^16, 2^16-1]
  U64: [0, 2^64-1]
  S64: [-2^32, 2^32-1]

  numeric conversions: may be unsafe
  (1) Value-preserving conversions: safe. `t` can exactly represent all
      possible values in `f`, e.g. s32 => s64
  (2) Reinterpretive conversions: unsafe. The value may be different, but no
      data is lost, e.g. s32(-9) => u32(4294967287) => s32(-9)
  (3) Lossy conversions: unsafe. Data may be lost during the conversion.
      e.g. f32(1.5f) => s32(1) => f32(1.0f)
      static_cast<long long>(static_cast<double>(10000000000000001LL))
       => 10000000000000000LL
*/

// The cast will not cause a range error, and not change the precision of the
// floating-point value
inline static bool IsValuePreservingCast(const BaseType f, const BaseType t) {
  using BT = BaseType;
  if (!IsNumericType(f) || !IsNumericType(t))
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");
  static const std::unordered_map<BT, std::unordered_set<BT>> table = {
      {BT::U8,
       {BT::U8, BT::U16, BT::S16, BT::U32, BT::S32, BT::U64, BT::S64, BT::BF16,
        BT::F16, BT::F32, BT::F64}},
      {BT::S8,
       {BT::S8, BT::S16, BT::S32, BT::S64, BT::BF16, BT::F16, BT::F32,
        BT::F64}},
      {BT::U16,
       {BT::U16, BT::U32, BT::S32, BT::U64, BT::S64, BT::F32, BT::F64}},
      {BT::S16, {BT::S16, BT::S32, BT::S64, BT::F32, BT::F64}},
      {BT::U32, {BT::U32, BT::U64, BT::S64, BT::F64}},
      {BT::S32, {BT::S32, BT::S64, BT::F64}},
      {BT::U64, {BT::U64}},
      {BT::S64, {BT::S64}},
      {BT::F8_E4M3, {BT::F8_E4M3, BT::BF16, BT::F16, BT::F32, BT::F64}},
      {BT::F8_E5M2, {BT::F8_E5M2, BT::BF16, BT::F16, BT::F32, BT::F64}},
      {BT::BF16, {BT::BF16, BT::F32, BT::F64}},
      {BT::F16, {BT::F16, BT::F32, BT::F64}},
      {BT::TF32, {BT::TF32, BT::F32, BT::F64}},
      {BT::F32, {BT::F32, BT::F64}},
      {BT::F64, {BT::F64}},
  };
  auto it = table.find(f);
  if (it != table.end() && it->second.count(t)) return true;
  return false;
}

// actually signed <=> unsigned
inline static bool IsReinterpretiveCast(const BaseType f, const BaseType t) {
  using BT = BaseType;
  if (!IsNumericType(f) || !IsNumericType(t))
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");
  static const std::unordered_map<BT, std::unordered_set<BT>> table = {
      {BT::U8, {BT::S8}},   {BT::S8, {BT::U8, BT::U16, BT::U32, BT::U64}},
      {BT::U16, {BT::S16}}, {BT::S16, {BT::U16, BT::U32, BT::U64}},
      {BT::U32, {BT::S32}}, {BT::S32, {BT::U32, BT::U64}},
      {BT::U64, {BT::S64}}, {BT::S64, {BT::U64}},
  };
  auto it = table.find(f);
  if (it != table.end() && it->second.count(t)) return true;
  return false;
}

inline static bool IsLossyCast(const BaseType f, const BaseType t) {
  using BT = BaseType;
  if (!IsNumericType(f) || !IsNumericType(t))
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");
  static const std::unordered_map<BT, std::unordered_set<BT>> table = {
      {BT::U8,
       {BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1}},
      {BT::S8,
       {BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1}},
      {BT::U16,
       {BT::U8, BT::S8, BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2,
        BT::F4_E2M1, BT::BF16, BT::F16}},
      {BT::S16,
       {BT::U8, BT::S8, BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2,
        BT::F4_E2M1, BT::BF16, BT::F16}},
      {BT::U32,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3,
        BT::F6_E3M2, BT::F4_E2M1, BT::BF16, BT::F16, BT::F32}},
      {BT::S32,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3,
        BT::F6_E3M2, BT::F4_E2M1, BT::BF16, BT::F16, BT::F32}},
      {BT::U64,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32, BT::F8_E4M3,
        BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1, BT::BF16, BT::F16,
        BT::F32, BT::F64}},
      {BT::S64,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32, BT::F8_E4M3,
        BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1, BT::BF16, BT::F16,
        BT::F32, BT::F64}},
      {BT::F8_E4M3,
       {BT::F8_E5M2, BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32,
        BT::U64, BT::S64}},
      {BT::F8_E5M2,
       {BT::F8_E4M3, BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32,
        BT::U64, BT::S64}},
      {BT::F6_E2M3,
       {BT::F6_E3M2, BT::F4_E2M1, BT::U8, BT::S8, BT::U16, BT::S16, BT::U32,
        BT::S32, BT::U64, BT::S64, BT::BF16, BT::F16, BT::F32, BT::F64}},
      {BT::F6_E3M2,
       {BT::F6_E2M3, BT::F4_E2M1, BT::U8, BT::S8, BT::U16, BT::S16, BT::U32,
        BT::S32, BT::U64, BT::S64, BT::BF16, BT::F16, BT::F32, BT::F64}},
      {BT::F4_E2M1,
       {BT::F6_E2M3, BT::F6_E3M2, BT::U8, BT::S8, BT::U16, BT::S16, BT::U32,
        BT::S32, BT::U64, BT::S64, BT::BF16, BT::F16, BT::F32, BT::F64}},
      {BT::BF16,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32, BT::U64, BT::S64,
        BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1,
        BT::F16}},
      {BT::F16,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32, BT::U64, BT::S64,
        BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1,
        BT::BF16}},
      {BT::F32,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32, BT::U64, BT::S64,
        BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1,
        BT::BF16, BT::F16}},
      {BT::F64,
       {BT::U8, BT::S8, BT::U16, BT::S16, BT::U32, BT::S32, BT::U64, BT::S64,
        BT::F8_E4M3, BT::F8_E5M2, BT::F6_E2M3, BT::F6_E3M2, BT::F4_E2M1,
        BT::BF16, BT::F16, BT::F32}},
  };
  auto it = table.find(f);
  if (it != table.end() && it->second.count(t)) return true;
  return false;
}

// safe version for pointers
template <typename T>
inline static const std::string PSTR(T* pt) {
  if (!pt) return "invalid";
  return STR(*pt);
}

template <typename T>
inline static const std::string PSTR(const ptr<T>& pt) {
  if (!pt) return "invalid";
  return STR(*pt);
}

using IntegerList = std::vector<int>;

struct Shape {
  static ValueListRepo values; // value numbers

  size_t val_no = GetInvalidValueNumber();
  size_t dim_count =
      GetInvalidRank(); // dim_count is used when no value appears

  void Invalidate() {
    val_no = GetInvalidUnsigned();
    dim_count = GetInvalidRank();
  }

  explicit Shape() {} // this initialize an invalid Shape
                      // The type must be deduced for use

  Shape(size_t n) : dim_count(n) {}
  // init a shape with n-'v's
  Shape(size_t n, const ValueItem& v) : dim_count(n) {
    ValueList vl(n);
    std::fill(vl.begin(), vl.end(), v);
    val_no = values.Insert(vl);
  }
  Shape(size_t n, int v) : dim_count(n) {
    ValueList vl(n);
    std::fill(vl.begin(), vl.end(), sbe::nu(v));
    val_no = values.Insert(vl);
  }
  Shape(size_t n, const std::string& v) : dim_count(n) {
    ValueList vl(n);
    std::fill(vl.begin(), vl.end(), sbe::sym(v));
    val_no = values.Insert(vl);
  }
  Shape(const ValueList& v) {
    val_no = values.Insert(v);
    dim_count = v.size();
  }
  // could be inconsistently sized, but only be verified with sema checker
  Shape(size_t n, const ValueList& v) : dim_count(n) {
    val_no = values.Insert(v);
  }

  // trivially copyable
  Shape(const Shape& s) = default;
  Shape(const Shape&& s) {
    val_no = s.val_no;
    dim_count = s.dim_count;
    Invalidate();
  }
  Shape& operator=(const Shape&) = default;

  size_t DimCount() const { return dim_count; }
  size_t Rank() const { return dim_count; }

  void Update() { dim_count = values[val_no].size(); }

  bool IsRankDummy() const { return IsUnknownRank(dim_count); }

  bool IsRanked() const {
    // ok to hold a valid rank only
    if (IsRankDummy()) return false;
    if (!IsValidValueNumber(val_no)) return IsValidRank(dim_count);
    return dim_count == values[val_no].size();
  }

  // has valid and consistent value
  bool IsValid() const {
    if (!IsValidValueNumber(val_no)) return false;
    return dim_count == values[val_no].size();
  }
  bool IsValidOrDummy() const { return IsValid() || IsRankDummy(); }

  const ValueList& Value() const {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    return values[val_no];
  }

  const ValueItem LeadingValue() const {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    return values[val_no].back();
  }

  const ValueList GenDenseStrides() const {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    ValueList dense_strides;

    auto stride = sbe::nu(1);
    dense_strides.push_back(stride); // leading-dimension is always 1
    if (Rank() > 1) {
      for (auto v = values[val_no].rbegin(); v != values[val_no].rend() - 1;
           ++v) {
        stride = (stride * (*v))->Normalize();
        dense_strides.push_back(stride);
      }
    }
    std::reverse(dense_strides.begin(), dense_strides.end());
    return dense_strides;
  }

  const Shape TrimDims(size_t n) const {
    if (n == 0) return *this;

    auto vals = Value();
    if (n >= vals.size())
      choreo_unreachable(
          "the dimension count trimmed is large than shape's rank.");

    vals.assign(vals.begin() + n, vals.end());
    return {dim_count - n, vals};
  }

  ValueList Value() {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    return values[val_no];
  }

  bool SameRankAs(const Shape& s) const {
    return IsRanked() && s.IsRanked() && (Rank() == s.Rank());
  }
  bool CompatibleRank(const Shape& s) const {
    return SameRankAs(s) || s.IsRankDummy() || IsRankDummy();
  }

  const ValueItem& ValueAt(size_t index) const {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    if (index > dim_count)
      choreo_unreachable("index '" + std::to_string(index) +
                         "' exceeds rank: " + std::to_string(dim_count) + ".");
    return values[val_no].at(index);
  }

  bool IsDynamic() const {
    for (auto v : Value())
      if (!isa<sbe::NumericValue>(v->Normalize()))
        return true; // a symbolic value represents that the value is decided
                     // at runtime
    return false;
  }

  // retrieve the dimensions that are dynamic
  std::vector<std::pair<size_t, ValueItem>> DynamicDimensions() const {
    std::vector<std::pair<size_t, ValueItem>> res;
    size_t i = 0;
    for (auto& v : Value()) {
      if (!v->IsNumeric() && !v->IsBoolean()) res.emplace_back(i, v);
      ++i;
    }
    return res;
  }

  // retrieve all symbol references
  const std::set<ValueItem> GetDynamicSymbols() const {
    return GetSymbols(Value());
  }

  std::optional<std::vector<size_t>> PosValList() const {
    std::vector<size_t> int_list;
    for (auto v : Value()) {
      if (auto pint = dyn_cast<sbe::NumericValue>(v)) {
        if (pint->Value() < 0) return std::nullopt;
        int_list.push_back(pint->Value());
      } else
        return std::nullopt;
    }
    return int_list;
  }

  const ValueItem ElementCountValue() const {
    assert(!Value().empty() && "no values inside the shape.");
    return MultiplyAll(Value());
  }

  size_t ElementCount() const {
    auto ecv = ElementCountValue();
    if (auto nv = dyn_cast<sbe::NumericValue>(ecv)) return nv->Value();
    choreo_unreachable("fail to get an integer list.");
    return 0;
  }

  const std::string ElemCountExprString(bool ULL_suffix = false) const {
    assert(!Value().empty() && "no values inside the shape.");
    return ElementCountValue()->ToString((ULL_suffix ? "ULL" : ""));
  }

  void Print(std::ostream& os) const {
    if (!IsValidValueNumber(val_no)) {
      os << "[]";
    } else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueList(Value(), os);
    }
  }

  void PrintSizeExpr(std::ostream& os) const {
    if (!IsValidValueNumber(val_no)) {
      os << "";
    } else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueListSizeExpr(Value(), os, nullptr, nullptr);
    }
  }

  void PrintAsList(std::ostream& os) const {
    if (!IsValidValueNumber(val_no))
      os << "{}";
    else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueList(Value(), os, "{", "}");
    }
  }

  void PrintPlain(std::ostream& os) const {
    if (!IsValidValueNumber(val_no))
      os << "";
    else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueList(Value(), os, nullptr, nullptr);
    }
  }

  std::string EmitTo(CompileTarget target) const;
};

inline bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.IsValid() && rhs.IsValid() && (lhs.Rank() == rhs.Rank()) &&
         IsValueListEqual(lhs.Value(), rhs.Value());
}

inline bool operator!=(const Shape& lhs, const Shape& rhs) {
  return !(lhs == rhs);
}

using MultiBounds = Shape; // using a shape as a multi-bound

inline MultiBounds operator-(const MultiBounds& lhs, const MultiBounds& rhs) {
  if (!lhs.IsValid() || !rhs.IsValid())
    choreo_unreachable("unexpected to substract invalid shapes.");

  ValueList vl;
  for (size_t idx = 0; idx < lhs.Rank(); ++idx)
    vl.push_back(lhs.ValueAt(idx) - rhs.ValueAt(idx));

  return {vl.size(), vl};
}

ValueList GenUninitValueList(size_t rank);

struct Type {
  BaseType bt = BaseType::UNKNOWN;
  std::unordered_map<std::string, std::string> note; // key-val note
  Type(BaseType t) : bt(t) {}
  virtual BaseType GetBaseType() const { return bt; }
  virtual size_t Dims() const = 0;
  virtual bool IsComplete() const = 0; // it is a partial or complete type
  // Types with/without sufficient info is of the same type. However, a type
  // with sufficient info is higher ranked. In type-inference, a type without
  // sufficient info should promoted to the one with sufficient info.
  virtual bool HasSufficientInfo() const { return false; }
  virtual bool operator==(const Type& t) const = 0;
  // sometimes it requires to ignore the memory for type's comparison
  virtual bool LogicalEqual(const Type& t) const { return operator==(t); }
  // in-precise comparison without considering the shape detail.
  // used in early semantics
  virtual bool ApprxEqual(const Type& t) const = 0;

  virtual const ptr<Type> Clone() const {
    auto nty = CloneImpl();
    nty->note = note;
    return nty;
  }
  virtual const ptr<Type> CloneImpl() const = 0;

  virtual bool HasNote(const std::string& k) const { return note.count(k); };
  virtual const std::string GetNote(const std::string& k) const {
    return note.at(k);
  };
  virtual void AddNote(const std::string& k, const std::string& v = "") {
    note.emplace(k, v);
  };
  virtual void Print(std::ostream&) const = 0;
  virtual const std::string Name() const = 0;

  // codegen util for emitting target's code in string format
  virtual std::string EmitTo(CompileTarget) const {
    assert(false && "Emit stringify not impled for this type");
    return "";
  }

  void PrintNote(std::ostream& os) const {
    if (note.empty()) return;
    os << "(";
    for (auto& item : note) {
      os << item.first;
      if (item.second.size() > 0) os << ":" << item.second;
    }
    os << ")";
  }

  // for runtime type disambiguation
  __UDT_TYPE_INFO_BASE__(notype)
};

inline bool operator!=(const Type& t1, const Type& t2) {
  return !t1.operator==(t2);
}

inline std::string STR(const Type& ty) {
  std::ostringstream oss;
  ty.Print(oss);
  return oss.str();
}

inline std::string STR(const Shape& s) {
  std::ostringstream oss;
  s.Print(oss);
  return oss.str();
}

// string as list
inline std::string LSTR(const Shape& s) {
  std::ostringstream oss;
  s.PrintAsList(oss);
  return oss.str();
}

// un-braced 'raw' string
inline std::string RSTR(const Shape& s) {
  std::ostringstream oss;
  s.PrintPlain(oss);
  return oss.str();
}

inline std::string RSTR(const ValueItem& vi) { return STR(vi); }

struct VoidType final : public Type, public TypeIDProvider<VoidType> {
  explicit VoidType() : Type(BaseType::VOID) {}
  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "void"; }
  const std::string Name() const override { return "void_type"; }
  bool HasSufficientInfo() const override { return true; }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<VoidType>();
  }
  bool operator==(const Type& ty) const override { return isa<VoidType>(&ty); }
  bool ApprxEqual(const Type& ty) const override { return isa<VoidType>(&ty); }

  __UDT_TYPE_INFO__(Type, VoidType)
};

struct AddrType final : public Type, public TypeIDProvider<AddrType> {
  explicit AddrType() : Type(BaseType::ADDR) {}
  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "address"; }
  const std::string Name() const override { return "addr_type"; }
  bool HasSufficientInfo() const override { return true; }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<AddrType>();
  }

  bool operator==(const Type& ty) const override { return isa<AddrType>(&ty); }
  bool ApprxEqual(const Type& ty) const override { return isa<AddrType>(&ty); }

  __UDT_TYPE_INFO__(Type, AddrType)
};

struct StreamType final : public Type, public TypeIDProvider<StreamType> {
  explicit StreamType() : Type(BaseType::STREAM) {}
  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "stream"; }
  const std::string Name() const override { return "stream_type"; }
  bool HasSufficientInfo() const override { return true; }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<StreamType>();
  }

  bool operator==(const Type& ty) const override {
    return isa<StreamType>(&ty);
  }
  bool ApprxEqual(const Type& ty) const override {
    return isa<StreamType>(&ty);
  }

  __UDT_TYPE_INFO__(Type, StreamType)
};

// The type is unknown. It requires type inference
struct UnknownType final : public Type, public TypeIDProvider<UnknownType> {
  explicit UnknownType() : Type(BaseType::UNKNOWN) {}
  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return false; }
  void Print(std::ostream& os) const override { os << "unknown"; }
  const std::string Name() const override { return "unknown_type"; }
  bool HasSufficientInfo() const override { return false; }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<UnknownType>();
  }

  // Not comparable
  bool operator==(const Type&) const override { return false; }
  bool ApprxEqual(const Type&) const override { return false; }

  __UDT_TYPE_INFO__(Type, UnknownType)
};

// NoValueType: the value is UNSPECIFIED at compile time. It could be even
// UNKNOWN at runtime (like the size of a stream input). As a result, a
// NoValueType value can not be utilized for any value operations, like
// evaluation of its value, valno numbering, and etc..
//
// Note: NoValueType is different with UnknownType since UnknownType acts as the
// bottom for type derivation, while NoValueType is a confirmed type.
struct NoValueType final : public Type, public TypeIDProvider<NoValueType> {
  explicit NoValueType() : Type(BaseType::UNSPECVAL) {}
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "Novalue"; }
  const std::string Name() const override { return "un_specified_value_type"; }
  bool HasSufficientInfo() const override { return true; }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<NoValueType>();
  }

  // Not comparable
  bool operator==(const Type&) const override { return false; }
  bool ApprxEqual(const Type&) const override { return false; }

  __UDT_TYPE_INFO__(Type, NoValueType)
};

struct PlaceHolderType final : public Type,
                               public TypeIDProvider<PlaceHolderType> {
  PlaceHolderType(BaseType t) : Type(t) {}
  size_t Dims() const override { return 0; }
  bool IsComplete() const override { return false; }
  void Print(std::ostream& os) const override {
    os << "placeholder<" << STR(GetBaseType()) << ">";
  }
  const std::string Name() const override { return "place_holder"; }
  bool HasSufficientInfo() const override { return false; }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<PlaceHolderType>(Type::GetBaseType());
  }

  bool operator==(const Type&) const override { return false; }
  // tolerate im-precise comparison
  bool ApprxEqual(const Type& t) const override {
    return t.GetBaseType() == GetBaseType();
  }

  __UDT_TYPE_INFO__(Type, PlaceHolderType)
};

struct ScalarType : public Type, public TypeIDProvider<ScalarType> {
  bool is_mutable = false;
  ScalarType(BaseType t, bool m) : Type(t), is_mutable(m) {
    assert(IsScalarType(t));
  }

  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return true; }
  virtual void SetBaseType(BaseType t) { bt = t; }
  virtual bool IsMutable() const { return is_mutable; }
  virtual void SetMutable(bool m) { is_mutable = m; }
  virtual ptr<ScalarType> Clone(bool m) const = 0;
  // virtual const ptr<Type> CloneImpl() const override = 0;

  bool operator==(const Type& ty) const override {
    if (auto sty = dyn_cast<ScalarType>(&ty)) {
      if (sty->is_mutable != is_mutable) return false;
      return bt == ty.bt;
    }
    return false;
  }

  bool ApprxEqual(const Type& ty) const override { return bt == ty.bt; }

  virtual bool IsFloat() const { return false; };
  virtual bool IsBoolInteger() const { return true; };

  const std::string Name() const override { return STR(bt); }
  void Print(std::ostream& os) const override {
    if (is_mutable) os << "mutable ";
    os << STR(bt);
  }

  __UDT_TYPE_INFO__(Type, ScalarType)
};

struct VectorType final : public Type, public TypeIDProvider<VectorType> {
  bool is_mutable = true;
  using FundamentalType = BaseType;
  FundamentalType e_type;
  size_t ec = 0;
  VectorType(BaseType e, size_t w) : Type(BaseType::VECTOR), e_type(e), ec(w) {
    assert(w > 0 && "vector width must be positive.");
  }

  size_t ElemCount() const { return ec; }
  BaseType ElemType() const { return e_type; }
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return true; }
  bool ApprxEqual(const Type& ty) const override {
    if (auto vt = dyn_cast<VectorType>(&ty)) {
      return vt->ec == ec && vt->e_type == e_type;
    }
    return false;
  }
  void Print(std::ostream& os) const override {
    if (is_mutable) os << "mutable ";
    os << "vector<" << STR(e_type) << "," << ec << ">";
  }
  const std::string Name() const override { return "vector"; }
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<VectorType>(e_type, ec);
  }
  bool operator==(const Type& ty) const override {
    if (auto vt = dyn_cast<VectorType>(&ty)) {
      return vt->ec == ec && vt->e_type == e_type;
    }
    return false;
  }

  __UDT_TYPE_INFO__(Type, VectorType)
};

inline bool ConvertibleToInt(const Type& ty);

struct ScalarFloatType;

struct ScalarIntegerType : public ScalarType,
                           public TypeIDProvider<ScalarIntegerType> {
  ScalarIntegerType(BaseType t, bool m) : ScalarType(t, m) {}
  bool LogicalEqual(const Type& ty) const override {
    return ConvertibleToInt(ty);
  }
  bool ApprxEqual(const Type& ty) const override {
    return isa<ScalarIntegerType>(&ty) || isa<ScalarFloatType>(&ty);
  }
  __UDT_TYPE_INFO__(ScalarType, ScalarIntegerType)
};

struct Bin1Type final : public ScalarIntegerType,
                        public TypeIDProvider<Bin1Type> {
  Bin1Type(bool m) : ScalarIntegerType(BaseType::BIN1, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<Bin1Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<Bin1Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, Bin1Type)
};

struct U1Type final : public ScalarIntegerType, public TypeIDProvider<U1Type> {
  U1Type(bool m) : ScalarIntegerType(BaseType::U1, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U1Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U1Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U1Type)
};

struct S2Type final : public ScalarIntegerType, public TypeIDProvider<S2Type> {
  S2Type(bool m) : ScalarIntegerType(BaseType::S2, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<S2Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<S2Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, S2Type)
};

struct U2Type final : public ScalarIntegerType, public TypeIDProvider<U2Type> {
  U2Type(bool m) : ScalarIntegerType(BaseType::U2, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U2Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U2Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U2Type)
};

struct S4Type final : public ScalarIntegerType, public TypeIDProvider<S4Type> {
  S4Type(bool m) : ScalarIntegerType(BaseType::S4, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<S4Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<S4Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, S4Type)
};

struct U4Type final : public ScalarIntegerType, public TypeIDProvider<U4Type> {
  U4Type(bool m) : ScalarIntegerType(BaseType::U4, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U4Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U4Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U4Type)
};

struct S6Type final : public ScalarIntegerType, public TypeIDProvider<S6Type> {
  S6Type(bool m) : ScalarIntegerType(BaseType::S6, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<S6Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<S6Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, S6Type)
};

struct U6Type final : public ScalarIntegerType, public TypeIDProvider<U6Type> {
  U6Type(bool m) : ScalarIntegerType(BaseType::U6, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U6Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U6Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U6Type)
};

struct S8Type final : public ScalarIntegerType, public TypeIDProvider<S8Type> {
  S8Type(bool m) : ScalarIntegerType(BaseType::S8, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<S8Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<S8Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, S8Type)
};

struct U8Type final : public ScalarIntegerType, public TypeIDProvider<U8Type> {
  U8Type(bool m) : ScalarIntegerType(BaseType::U8, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U8Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U8Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U8Type)
};

struct S16Type final : public ScalarIntegerType,
                       public TypeIDProvider<S16Type> {
  S16Type(bool m) : ScalarIntegerType(BaseType::S16, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<S16Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<S16Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, S16Type)
};

struct U16Type final : public ScalarIntegerType,
                       public TypeIDProvider<U16Type> {
  U16Type(bool m) : ScalarIntegerType(BaseType::U16, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U16Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U16Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U16Type)
};

struct S32Type final : public ScalarIntegerType,
                       public TypeIDProvider<S32Type> {
  S32Type(bool m) : ScalarIntegerType(BaseType::S32, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<S32Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<S32Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, S32Type)
};

using IntegerType = S32Type; // alias for S32Type, as it is the default

struct U32Type final : public ScalarIntegerType,
                       public TypeIDProvider<U32Type> {
  U32Type(bool m) : ScalarIntegerType(BaseType::U32, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U32Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U32Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U32Type)
};

struct S64Type final : public ScalarIntegerType,
                       public TypeIDProvider<S64Type> {
  S64Type(bool m) : ScalarIntegerType(BaseType::S64, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<S64Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<S64Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, S64Type)
};

struct U64Type final : public ScalarIntegerType,
                       public TypeIDProvider<U64Type> {
  U64Type(bool m) : ScalarIntegerType(BaseType::U64, m) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<U64Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<U64Type>(m);
  }

  __UDT_TYPE_INFO__(ScalarIntegerType, U64Type)
};

struct ScalarFloatType : public ScalarType,
                         public TypeIDProvider<ScalarFloatType> {
  ScalarFloatType(BaseType t, bool m) : ScalarType(t, m) {}
  bool IsFloat() const override { return true; }
  bool IsBoolInteger() const override { return false; }
  bool ApprxEqual(const Type& ty) const override {
    return isa<ScalarIntegerType>(&ty) || isa<ScalarFloatType>(&ty);
  }
  __UDT_TYPE_INFO__(ScalarType, ScalarFloatType)
};

struct FloatE2M1Type final : public ScalarFloatType,
                             public TypeIDProvider<FloatE2M1Type> {
  FloatE2M1Type(bool m) : ScalarFloatType(BaseType::F4_E2M1, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FloatE2M1Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatE2M1Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, FloatE2M1Type)
};

struct FloatE3M2Type final : public ScalarFloatType,
                             public TypeIDProvider<FloatE3M2Type> {
  FloatE3M2Type(bool m) : ScalarFloatType(BaseType::F6_E3M2, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FloatE3M2Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatE3M2Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, FloatE3M2Type)
};

struct FloatE2M3Type final : public ScalarFloatType,
                             public TypeIDProvider<FloatE2M3Type> {
  FloatE2M3Type(bool m) : ScalarFloatType(BaseType::F6_E2M3, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FloatE2M3Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatE2M3Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, FloatE2M3Type)
};

struct FloatUE8M0Type final : public ScalarFloatType,
                              public TypeIDProvider<FloatUE8M0Type> {
  FloatUE8M0Type(bool m) : ScalarFloatType(BaseType::F8_UE8M0, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FloatUE8M0Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatUE8M0Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, FloatUE8M0Type)
};

struct FloatUE4M3Type final : public ScalarFloatType,
                              public TypeIDProvider<FloatUE4M3Type> {
  FloatUE4M3Type(bool m) : ScalarFloatType(BaseType::F8_UE4M3, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FloatUE4M3Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatUE4M3Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, FloatUE4M3Type)
};

struct FloatE4M3Type final : public ScalarFloatType,
                             public TypeIDProvider<FloatE4M3Type> {
  FloatE4M3Type(bool m) : ScalarFloatType(BaseType::F8_E4M3, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FloatE4M3Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatE4M3Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, FloatE4M3Type)
};

struct FloatE5M2Type final : public ScalarFloatType,
                             public TypeIDProvider<FloatE5M2Type> {
  FloatE5M2Type(bool m) : ScalarFloatType(BaseType::F8_E5M2, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FloatE5M2Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatE5M2Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, FloatE5M2Type)
};

struct F16Type final : public ScalarFloatType, public TypeIDProvider<F16Type> {
  F16Type(bool m) : ScalarFloatType(BaseType::F16, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<F16Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<F16Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, F16Type)
};

struct BF16Type final : public ScalarFloatType,
                        public TypeIDProvider<BF16Type> {
  BF16Type(bool m) : ScalarFloatType(BaseType::BF16, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<BF16Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<BF16Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, BF16Type)
};

struct TF32Type final : public ScalarFloatType,
                        public TypeIDProvider<TF32Type> {
  TF32Type(bool m) : ScalarFloatType(BaseType::F32, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<TF32Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<TF32Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, TF32Type)
};

struct F32Type final : public ScalarFloatType, public TypeIDProvider<F32Type> {
  F32Type(bool m) : ScalarFloatType(BaseType::F32, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<F32Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<F32Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, F32Type)
};

struct F64Type final : public ScalarFloatType, public TypeIDProvider<F64Type> {
  F64Type(bool m) : ScalarFloatType(BaseType::F64, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<F64Type>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<F64Type>(m);
  }
  __UDT_TYPE_INFO__(ScalarFloatType, F64Type)
};

struct BooleanType final : public ScalarType,
                           public TypeIDProvider<BooleanType> {
  BooleanType(bool m) : ScalarType(BaseType::BOOL, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<BooleanType>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<BooleanType>(m);
  }

  __UDT_TYPE_INFO__(ScalarType, BooleanType)
};

struct UnknownScalarType final : public ScalarType,
                                 public TypeIDProvider<UnknownScalarType> {
  UnknownScalarType(bool m) : ScalarType(BaseType::UNKSCALAR, m) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<UnknownScalarType>(IsMutable());
  }
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<UnknownScalarType>(m);
  }

  bool operator==(const Type&) const override {
    choreo_unreachable("The unknown element type is not strictly comparable.");
  }
  bool ApprxEqual(const Type& ty) const override {
    return isa<ScalarType>(&ty);
  }

  __UDT_TYPE_INFO__(ScalarType, UnknownScalarType)
};

struct StringType : public Type, public TypeIDProvider<StringType> {
  StringType() : Type(BaseType::STRING) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<StringType>();
  }
  void Print(std::ostream& os) const override { os << "string"; }
  const std::string Name() const override { return "string"; }

  size_t Dims() const override { return 0; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return true; }

  bool operator==(const Type& ty) const override {
    return isa<StringType>(&ty);
  }
  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }

  __UDT_TYPE_INFO__(Type, StringType)
};

struct IndexType : public Type, public TypeIDProvider<IndexType> {
  IndexType() : Type(BaseType::INDEX) {}
  // note: index type takes 1 dim in mdspan/ituple declaration
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<IndexType>();
  }
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "idx"; }
  const std::string Name() const override { return "index"; }

  bool operator==(const Type& ty) const override { return isa<IndexType>(&ty); }
  bool ApprxEqual(const Type& ty) const override { return isa<IndexType>(&ty); }

  __UDT_TYPE_INFO__(Type, IndexType)
};

// ITuple is a dimensioned type
struct ITupleType : public Type, public TypeIDProvider<ITupleType> {
  size_t dim_count = GetInvalidRank();

  explicit ITupleType()
      : Type(BaseType::ITUPLE) {} // this initialize an invalid ITupleType
                                  // The Type must be deduced for use

  bool HasSufficientInfo() const override { return IsValidRank(dim_count); }

  ITupleType(size_t n) : Type(BaseType::ITUPLE), dim_count(n) {}

  size_t Dims() const override { return dim_count; }
  bool IsDimValid() const { return IsValidRank(dim_count); }
  bool IsComplete() const override { return true; }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<ITupleType>(dim_count);
  }

  void Print(std::ostream& os) const override {
    os << "ituple<";
    if (HasSufficientInfo())
      os << dim_count;
    else
      os << "unknown";
    os << ">";
  }

  const std::string Name() const override { return "ituple"; }

  bool operator==(const Type& ty) const override {
    if (auto itty = dyn_cast<ITupleType>(&ty)) {
      if ((Dims() == itty->Dims()) && HasSufficientInfo()) return true;
    }
    return false;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (auto itty = dyn_cast<ITupleType>(&ty)) {
      if (HasSufficientInfo() && itty->HasSufficientInfo())
        return (Dims() == itty->Dims());
      else
        return true;
    }
    return false;
  }

  bool LogicalEqual(const Type& ty) const override {
    if (ConvertibleToInt(*this))
      return ConvertibleToInt(ty);
    else
      return operator==(ty);
  }

  __UDT_TYPE_INFO__(Type, ITupleType)
};

struct MDSpanType : public Type, public TypeIDProvider<MDSpanType> {
  Shape value;

  MDSpanType(const Shape& v) : Type(BaseType::PARTIAL), value(v) {}

  void SetShape(const Shape& v) { value = v; }
  const Shape GetShape() { return value; }

  size_t Dims() const override { return value.Rank(); }

  // MDSpanType is an incomplete/partial type
  bool IsComplete() const override { return false; }

  // TODO: if the value is not evaluated, or can not be evaluated, not
  // sufficient information is obtained
  bool HasSufficientInfo() const override { return value.IsValid(); }

  bool operator==(const Type& ty) const override {
    if (&ty == this) return true;
    if (auto mty = dyn_cast<MDSpanType>(&ty)) {
      // Must consider the condition whn shapes are not accurately decided
      // (only the dim-count is available)
      if (mty->value.IsValid() && value.IsValid()) return mty->value == value;
      return mty->value.SameRankAs(value);
    }
    return false;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (&ty == this) return true;
    if (auto pty = dyn_cast<PlaceHolderType>(&ty))
      return pty->ApprxEqual(*this);

    if (auto mty = dyn_cast<MDSpanType>(&ty)) {
      if (mty->value.IsValid() && value.IsValid()) return mty->value == value;
      return mty->value.CompatibleRank(value);
    }
    return false;
  }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<MDSpanType>(value);
  }

  void Print(std::ostream& os) const override {
    os << "mdspan<";
    if (value.IsRanked()) os << Dims();
    os << "> " << STR(value);
  }

  const std::string Name() const override { return "mdspan"; }

  __UDT_TYPE_INFO__(Type, MDSpanType)
};

enum class Modality { MUST, MAY, NOT /*MUST_NOT*/ };
inline Modality operator&(const Modality l, const Modality r) {
  if (l == Modality::NOT) return l;
  if (l == Modality::MAY) {
    if (l == Modality::MUST)
      return l;
    else
      return r;
  }
  return r;
}
inline Modality operator|(const Modality l, const Modality r) {
  if (l == Modality::MUST) return l;
  if (l == Modality::MAY) {
    if (l == Modality::NOT)
      return l;
    else
      return r;
  }
  return r;
}

inline Modality operator&=(Modality& l, const Modality r) {
  l = l & r;
  return l;
}
inline Modality operator|=(Modality& l, const Modality r) {
  l = l | r;
  return l;
}
// providing the shape and stride, judge if it is contiguous
inline Modality IsContiguous(const Shape& shape, const ValueList& strides) {
  if (!shape.IsValid()) choreo_unreachable("the shape is not accessible.");
  if (shape.DimCount() != strides.size())
    choreo_unreachable("shape and stride does not match.");
  if (!sbe::ceq(strides.back(), sbe::nu(1))) return Modality::NOT;

  auto strd = sbe::nu(1);
  Modality p = Modality::MUST;
  for (int i = shape.DimCount() - 2; i >= 0; --i) {
    strd *= shape.Value()[i + 1];
    if (sbe::must_ne(strd, strides[i])) p &= Modality::NOT;
    if (sbe::may_ne(strd, strides[i])) p &= Modality::MAY;
  }
  return p;
}

struct SpannedType;
template <>
inline bool isa<SpannedType>(const Type*);
template <>
inline SpannedType* dyn_cast<SpannedType>(const Type* ty);

struct SpannedType : public Type, public TypeIDProvider<SpannedType> {
  using FundamentalType = BaseType;
  FundamentalType e_type;
  ptr<MDSpanType> s_type = nullptr;
  ValueList strides;
  Storage m_type;

  SpannedType(BaseType t, const ptr<MDSpanType>& s, const ValueList& strd,
              Storage m = Storage::DEFAULT)
      : Type(BaseType::SPANNED), e_type(t), s_type(s), strides(strd),
        m_type(m) {
    assert((s_type != nullptr) && "mdspan is not initialized.");
    assert(IsScalarType(e_type) && "element type must be a scalar type.");
  }

  const ptr<Type> CloneImpl() const override {
    assert(s_type);
    return std::make_shared<SpannedType>(
        e_type, cast<MDSpanType>(s_type->Clone()), strides, m_type);
  }

  BaseType ElementType() const { return e_type; }
  const ptr<MDSpanType> GetMDSpanType() { return s_type; }
  size_t Dims() const override { return s_type->Dims(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override {
    return s_type->HasSufficientInfo();
  }

  void SetStrides(const ValueList& strd) { strides = strd; }
  const ValueList& GetStrides() const { return strides; }
  Modality IsDense() const { return IsContiguous(GetShape(), GetStrides()); }

  bool operator==(const Type& ty) const override {
    if (&ty == this) return true;
    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    return t.e_type == e_type && *t.s_type == *s_type && t.strides == strides &&
           Compatible(t.m_type, m_type);
  }

  // Ignore the memory
  bool LogicalEqual(const Type& ty) const override {
    if (&ty == this) return true;
    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    return t.e_type == e_type && *t.s_type == *s_type;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (&ty == this) return true;
    if (auto pty = dyn_cast<PlaceHolderType>(&ty))
      return pty->ApprxEqual(*this);

    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    // should the equivalence of storage be checked?
    return Choreo::ApprxEqual(t.e_type, e_type) &&
           t.s_type->ApprxEqual(*s_type);
  }

  const Shape GetShape() const { return s_type->GetShape(); }
  const ptr<MDSpanType> GetMDSpanType() const { return s_type; }

  bool RuntimeShaped() const {
    assert(s_type && "missing the spanned type.");
    return GetShape().IsDynamic();
  }

  // use these interface when the shape is NOT runtime-shaped
  size_t ElementCount() const { return GetShape().ElementCount(); }
  ValueItem ElementCountValue() const { return GetShape().ElementCountValue(); }
  ValueItem ByteSizeValue() const {
    return (ElementCountValue() * sbe::nu(SizeOf(e_type)))->Normalize();
  }
  size_t ByteSize() const { return SizeOf(e_type) * GetShape().ElementCount(); }
  const ValueItem ElementSizeValue() const { return sbe::nu(SizeOf(e_type)); }

  const std::string ByteSizeExpression(bool ULL_suffix = false) const {
    if (RuntimeShaped())
      return "(" + GetShape().ElemCountExprString(ULL_suffix) + ") * " +
             std::to_string(SizeOf(e_type));
    else
      return std::to_string(ByteSize()) + (ULL_suffix ? "ULL" : "");
  }

  void SetStorage(Storage s) { m_type = s; }
  Storage GetStorage() const { return m_type; }

  void Print(std::ostream& os) const override;

  const std::string Name() const override { return "spanned"; }

  __UDT_TYPE_INFO__(Type, SpannedType)
};

struct DeviceDataType final : public Type,
                              public TypeIDProvider<DeviceDataType> {
  std::string name;
  std::string plain_name;
  std::string attr;
  BaseType data_type;
  bool is_pointer;
  std::string init_expr;
  size_t pointer_count = 0; // for pointer type, the pointer count

  DeviceDataType(std::string str, std::string at = "",
                 BaseType bt = BaseType::UNKNOWN, bool ip = false,
                 std::string init = "")
      : Type(BaseType::DEVICE), name(str), plain_name(str), attr(at),
        data_type(bt), is_pointer(ip), init_expr(init) {}

  DeviceDataType(std::string str, std::string plain_str, std::string at = "",
                 BaseType bt = BaseType::UNKNOWN, bool ip = false,
                 std::string init = "")
      : Type(BaseType::DEVICE), name(str), plain_name(plain_str), attr(at),
        data_type(bt), is_pointer(ip), init_expr(init) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<DeviceDataType>(name, plain_name, attr, data_type,
                                            is_pointer, init_expr);
  }

  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override {
    os << STR(data_type);
    if (is_pointer) os << " *";
  }
  const std::string Name() const override { return name; }
  const std::string PlainName() const { return plain_name; }

  bool HasSufficientInfo() const override { return true; }

  bool operator==(const Type& ty) const override {
    return isa<DeviceDataType>(&ty) && name == ((const DeviceDataType&)ty).name;
  }

  const std::string GetTypeStr() { return name; }
  void SetTypeStr(std::string s) { name = s; }

  BaseType GetDataType() { return data_type; }
  void SetDataType(BaseType bt) { data_type = bt; }

  bool IsPointerType() { return is_pointer; }
  void SetPointerType(bool ip) { is_pointer = ip; }

  bool Initized() const { return !init_expr.empty(); }

  // used to march with choreo type including scalar type and spanned type.
  bool ApprxEqual(const Type& ty) const override {
    if (data_type == BaseType::UNKNOWN) return false;
    if (ty.GetBaseType() == BaseType::UNKNOWN) return false;
    if (isa<ScalarType>(&ty)) {
      return ty.GetBaseType() == data_type ||
             IsValuePreservingCast(ty.GetBaseType(), data_type);
    }
    // spanned type with the same element type
    if (SpannedType* spanned_ty = dyn_cast<SpannedType>(&ty);
        spanned_ty && is_pointer) {
      if (spanned_ty->ElementType() == BaseType::UNKNOWN) return false;
      if (data_type == BaseType::VOID ||
          IsValuePreservingCast(spanned_ty->ElementType(), data_type))
        return true;
    }

    return false;
  }

  bool IsNaiveType() {
    return IsNumericType(data_type) || data_type == BaseType::VOID;
  }
  __UDT_TYPE_INFO__(Type, DeviceDataType)
};

struct BoundedType : public Type, public TypeIDProvider<BoundedType> {
  BoundedType(BaseType tc, const std::string& k = "", const std::string& v = "")
      : Type(tc) {
    if (k.size() > 0) AddNote(k, v);
  }
  virtual bool HasValidBound() const = 0;
  virtual const ValueItem& GetUpperBound() const = 0;
  virtual const MultiBounds GetUpperBounds() const = 0;
  virtual int GetStep() const = 0;
  virtual IntegerList GetSteps() const = 0;
  virtual int GetWidth() const = 0;
  virtual IntegerList GetWidths() const = 0;

  bool LogicalEqual(const Type& ty) const override {
    if (auto fty = dyn_cast<BoundedType>(&ty)) return Dims() == fty->Dims();
    return false;
  }

  void Print(std::ostream& os) const override { PrintNote(os); }

  __UDT_TYPE_INFO__(Type, BoundedType)
};

struct BoundedIntegerType final : public BoundedType,
                                  public TypeIDProvider<BoundedIntegerType> {
  ValueItem lbound = GetInvalidValueItem();
  ValueItem ubound = GetInvalidValueItem();
  int step = GetInvalidStep();
  int width = 1;

  BoundedIntegerType() : BoundedType(BaseType::BOUNDED_INT, "") {}
  BoundedIntegerType(int lb, int ub, int s = 1, const std::string& note = "")
      : BoundedIntegerType(sbe::nu(lb), sbe::nu(ub), s, note) {}
  BoundedIntegerType(const ValueItem& lexpr, const ValueItem& uexpr, int s = 1,
                     const std::string& note = "")
      : BoundedType(BaseType::BOUNDED_INT, note), lbound(lexpr), ubound(uexpr),
        step(s) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<BoundedIntegerType>(lbound, ubound, step);
  }
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return HasValidBound(); }
  bool HasValidBound() const override {
    return IsValidValueItem(lbound) && IsValidValueItem(ubound) &&
           IsValidStep(step);
  }
  ValueItem GetLowerBound() const { return lbound; }
  const ValueItem& GetUpperBound() const override { return ubound; }
  const MultiBounds GetUpperBounds() const override {
    return MultiBounds(1, ubound);
  }
  int GetStep() const override { return step; }
  IntegerList GetSteps() const override { return IntegerList(1, step); }
  int GetWidth() const override { return width; }
  IntegerList GetWidths() const override { return IntegerList(1, width); }
  bool operator==(const Type& ty) const override {
    if (!isa<BoundedIntegerType>(&ty)) return false;
    auto bty = (BoundedIntegerType&)ty;
    return (bty.lbound == lbound) && (bty.ubound == ubound) &&
           (bty.step == step);
  }

  bool ApprxEqual(const Type& ty) const override {
    // do not care about the bound expression
    return isa<BoundedIntegerType>(&ty);
  }

  void Print(std::ostream& os) const override {
    bool plain = true;
    if (!HasValidBound())
      os << "int->[unknown]";
    else
      os << "int->[" << STR(lbound) << "," << STR(ubound) << "]";
    if (plain == false) os << ":" << step;
    BoundedType::Print(os);
  }

  const std::string Name() const override { return "bounded-integer"; }

  __UDT_TYPE_INFO__(BoundedType, BoundedIntegerType)
};

struct BoundedITupleType final : public BoundedType,
                                 public TypeIDProvider<BoundedITupleType> {
  MultiBounds lbounds;
  MultiBounds ubounds;
  IntegerList steps;
  IntegerList widths;

  BoundedITupleType(const MultiBounds& l, const MultiBounds& u,
                    const IntegerList s, const std::string& k = "",
                    const std::string& v = "")
      : BoundedType(BaseType::BOUNDED_ITUPLE, k, v), lbounds(l), ubounds(u),
        steps(s) {
    if (lbounds.IsValid())
      assert((lbounds.DimCount() == ubounds.DimCount()) &&
             (lbounds.DimCount() == steps.size()) &&
             "expecting a valid bound.");
    else
      assert(!ubounds.IsValid() && steps.empty() &&
             "expecting an invalid bound.");
    widths = IntegerList(lbounds.DimCount(), 1);
  }

  BoundedITupleType(const MultiBounds& l, const MultiBounds& u,
                    const IntegerList s, const IntegerList w,
                    const std::string& k = "", const std::string& v = "")
      : BoundedType(BaseType::BOUNDED_ITUPLE, k, v), lbounds(l), ubounds(u),
        steps(s), widths(w) {
    if (lbounds.IsValid())
      assert((lbounds.DimCount() == ubounds.DimCount()) &&
             (lbounds.DimCount() == steps.size()) &&
             "expecting a valid bound.");
    else
      assert(!ubounds.IsValid() && steps.empty() &&
             "expecting an invalid bound.");
  }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<BoundedITupleType>(lbounds, ubounds, steps);
  }
  size_t Dims() const override { return ubounds.Rank(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return ubounds.IsValid(); }
  const MultiBounds GetLowerBounds() const { return lbounds; }
  const MultiBounds GetUpperBounds() const override { return ubounds; }
  const Shape GetSizes() const { return ubounds - lbounds; }
  int GetStep() const override { return steps[0]; }
  IntegerList GetSteps() const override { return steps; }
  int GetWidth() const override { return widths[0]; }
  IntegerList GetWidths() const override { return widths; }
  const ValueItem& GetUpperBound() const override { return ubounds.ValueAt(0); }
  const ValueItem& GetUpperBound(size_t idx) const {
    return ubounds.ValueAt(idx);
  }
  const ValueItem& GetLowerBound(size_t idx = 0) const {
    return lbounds.ValueAt(idx);
  }
  int GetStep(size_t idx) const { return steps[idx]; }
  int GetWidth(size_t idx) const { return widths[idx]; }
  bool IsPlain(size_t idx) const {
    return (*lbounds.ValueAt(idx) == *sbe::nu(0)) && (steps[idx] == 1);
  }
  bool HasValidBound() const override {
    return lbounds.IsValid() && ubounds.IsValid() && !steps.empty() &&
           (lbounds.DimCount() == ubounds.DimCount()) &&
           (lbounds.DimCount() == steps.size()) &&
           (steps.size() == widths.size());
  }

  bool operator==(const Type& ty) const override {
    if (!isa<BoundedITupleType>(&ty)) return false;
    auto& t = (BoundedITupleType&)ty;
    return (t.lbounds == lbounds) && (t.ubounds == ubounds) &&
           (t.steps == steps) && (t.widths == widths);
  }

  bool ApprxEqual(const Type& ty) const override {
    if (!isa<BoundedITupleType>(&ty)) return false;
    auto& t = (BoundedITupleType&)ty;
    return t.Dims() == Dims();
  }

  void Print(std::ostream& os) const override {
    if (!ubounds.IsRanked()) {
      os << "{invalid}";
      BoundedType::Print(os);
      return;
    }
    assert(Dims() > 0 && "dim of bounded ituple is incorrect.");
    os << "{int";
    for (size_t i = 1; i < Dims(); ++i) os << ",int";
    os << "}->";
    bool plain = true;
    for (size_t i = 1; i < Dims(); ++i) {
      if (!IsPlain(i)) {
        plain = false;
        break;
      }
    }

    if (plain) {
      os << STR(ubounds);
      if (steps[0] != 1) os << ", s = " << steps[0];
      if (widths[0] != 1) os << ", w = " << widths[0];
      BoundedType::Print(os);
      return;
    }

    os << "{[" << STR(lbounds.ValueAt(0)) << "," << STR(ubounds.ValueAt(0))
       << ")";

    for (size_t i = 1; i < Dims(); ++i) {
      os << ", [" << STR(lbounds.ValueAt(i)) << "," << STR(ubounds.ValueAt(i))
         << "):" << steps[i];
    }
    os << "}";

    BoundedType::Print(os);
  }

  const std::string Name() const override { return "bounded-ituple"; }

  __UDT_TYPE_INFO__(BoundedType, BoundedITupleType)
};

struct AsyncType : public Type, public TypeIDProvider<AsyncType> {
  AsyncType(BaseType t) : Type(t) {}
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }

  // can not have instance
  __UDT_TYPE_INFO__(Type, AsyncType)
};

struct EventType;
template <>
inline EventType* dyn_cast<EventType>(const Type*);

struct EventType : public AsyncType, public TypeIDProvider<EventType> {
  Storage scope;
  explicit EventType(Storage s) : AsyncType(BaseType::EVENT), scope(s) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<EventType>(scope);
  }
  bool HasSufficientInfo() const override { return true; }
  void Print(std::ostream& os) const override { os << STR(scope) << " event"; }
  const std::string Name() const override { return STR(scope) + " event"; }

  bool operator==(const Type& ty) const override {
    if (auto ety = dyn_cast<EventType>(&ty)) return scope == ety->scope;
    return false;
  }
  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }
  Storage GetStorage() const { return scope; }
  void SetStorage(Storage s) { scope = s; }

  __UDT_TYPE_INFO__(AsyncType, EventType)
};

struct FutureType : public AsyncType, public TypeIDProvider<FutureType> {
  ptr<SpannedType> psty =
      nullptr; // the spanned data associated with the future
  bool async;

  explicit FutureType(const ptr<SpannedType>& s, bool a)
      : AsyncType(BaseType::FUTURE), psty(s), async(a) {}
  const ptr<Type> CloneImpl() const override {
    return std::make_shared<FutureType>(cast<SpannedType>(psty->Clone()),
                                        async);
  }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return psty->HasSufficientInfo(); }
  const std::string Name() const override { return "future"; }
  const Shape GetShape() { return psty->GetShape(); }
  const ValueList& GetStrides() { return psty->GetStrides(); }
  const ptr<SpannedType>& GetSpannedType() const { return psty; }
  BaseType ElementType() const { return psty->ElementType(); }
  size_t Dims() const override { return psty->Dims(); }
  bool IsAsync() const { return async; }
  Storage GetStorage() const { return psty->GetStorage(); }

  bool operator==(const Type& ty) const override {
    if (auto fty = dyn_cast<FutureType>(&ty))
      return (fty->async == async) && (*fty->psty == *psty);
    else
      return false;
  }

  bool LogicalEqual(const Type& ty) const override {
    if (auto fty = dyn_cast<FutureType>(&ty))
      return (fty->async == async) && fty->psty->LogicalEqual(*psty);
    else
      return false;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (auto pty = dyn_cast<PlaceHolderType>(&ty))
      return pty->ApprxEqual(*this);

    if (auto fty = dyn_cast<FutureType>(&ty))
      return (fty->async == async) && (fty->psty->ApprxEqual(*psty));
    else
      return false;
  }

  void Print(std::ostream& os) const override {
    if (async)
      os << "async=>";
    else
      os << "sync=>";
    psty->Print(os);
  }

  __UDT_TYPE_INFO__(AsyncType, FutureType)
};

struct FunctionType : public Type, public TypeIDProvider<FunctionType> {
  ptr<Type> out_ty;
  std::vector<ptr<Type>> in_tys;

  FunctionType(const ptr<Type>& ot, const std::vector<ptr<Type>>& its)
      : Type(BaseType::FUNCTION), out_ty(ot), in_tys(its) {}

  const ptr<Type> CloneImpl() const override {
    std::vector<ptr<Type>> intys;
    for (auto ity : in_tys) intys.push_back(ity->Clone());
    return std::make_shared<FunctionType>(out_ty->Clone(), intys);
  }
  size_t Dims() const override {
    choreo_unreachable("a function can not have dimensions.");
    return 0;
  }
  bool IsComplete() const override { return true; }
  bool operator==(const Type& type) const override {
    if (auto t = dyn_cast<FunctionType>(&type)) {
      if (t->in_tys.size() != in_tys.size()) return false;
      for (size_t i = 0; i < in_tys.size(); ++i)
        if (*t->in_tys[i] != *in_tys[i]) return false;
      return *out_ty == *t->out_ty;
    }
    return false;
  }
  bool ApprxEqual(const Type& type) const override {
    if (auto t = dyn_cast<FunctionType>(&type)) {
      if (t->in_tys.size() != in_tys.size()) return false;
      for (size_t i = 0; i < in_tys.size(); ++i)
        if (!t->in_tys[i]->ApprxEqual(*in_tys[i])) return false;
      return out_ty->ApprxEqual(*t->out_ty);
    }
    return false;
  }
  void Print(std::ostream& os) const override {
    os << STR(*out_ty) << " (*)(";
    if (in_tys.size() > 0) os << STR(*in_tys[0]);
    for (size_t i = 1; i < in_tys.size(); ++i) {
      os << ", " << STR(*in_tys[i]);
    }
    os << ")";
  }
  const std::string Name() const override { return "function"; }

  __UDT_TYPE_INFO__(Type, FunctionType)
};

// ArrayType's size must be known in compile-time.
struct ArrayType : public Type, public TypeIDProvider<ArrayType> {
  ValueList dims;

  ArrayType(ValueList l) : Type(BaseType::ARRAY) { dims = l; }
  ArrayType(size_t rank) : Type(BaseType::ARRAY) {
    dims = GenUninitValueList(rank);
  }

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<ArrayType>(Dimensions());
  }

  size_t Dims() const override { return ArrayRank(); }

  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return true; }

  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }

  const std::string Name() const override { return "array"; }

  virtual const ptr<Type> ArrayElementType() const {
    return std::make_shared<NoValueType>();
  }
  virtual const ptr<Type> SubScriptType(size_t) {
    return std::make_shared<NoValueType>();
  };

  size_t ArrayRank() const { return dims.size(); }

  // array[n][m] - subscripting by 1  results in array[n]
  virtual ValueList SubScript(size_t dim_count) {
    if (dim_count > dims.size())
      choreo_unreachable("invalid subscription: not enough dimension.");
    return ValueList(dims.begin(), dims.begin() + dim_count);
  }

  // array[n][m] - subscripting by 1  the remainder dimensions is [m]
  virtual const ValueList RemainderDimensions(size_t dim_count) {
    if (dim_count > dims.size())
      choreo_unreachable("invalid subscription: not enough dimension.");
    return ValueList(dims.begin() + dim_count, dims.end());
  }

  virtual ValueItem Dimension(size_t idx) const { return dims.at(idx); }
  virtual const ValueList& Dimensions() const { return dims; }
  virtual ValueItem ElemCount() const {
    if (dims.size() == 0) {
      choreo_unreachable("invalid array.");
      return 0;
    }
    return MultiplyAll(dims);
  }

  virtual bool operator==(const Type& ty) const override {
    if (auto t = dyn_cast<ArrayType>(&ty))
      return IsValueListEqual(Dimensions(), t->Dimensions());
    return false;
  }

  virtual void Print(std::ostream& os) const override {
    os << "[";
    for (const auto& d : dims) os << "[" << STR(d) << "]";
    os << "]";
  }

  virtual void PrintAsCArray(std::ostream& os) const {
    for (const auto& d : dims) os << "[" << STR(d) << "]";
  }

  // for runtime type disambiguation
  __UDT_TYPE_INFO__(Type, ArrayType)
};

struct EventArrayType final : public ArrayType,
                              public TypeIDProvider<EventArrayType> {
  ptr<EventType> event;
  explicit EventArrayType(Storage s, const ValueList& ec)
      : ArrayType(ec), event(std::make_shared<EventType>(s)) {}
  explicit EventArrayType(Storage s, size_t rank)
      : ArrayType(rank), event(std::make_shared<EventType>(s)) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<EventArrayType>(event->GetStorage(),
                                            ArrayType::Dimensions());
  }

  const ptr<Type> ArrayElementType() const override {
    assert(event);
    return event;
  }
  const ptr<Type> SubScriptType(size_t subscription_count) override {
    auto arr = SubScript(subscription_count);
    if (arr.size() == 0)
      return std::make_shared<EventType>(event->GetStorage());
    else
      return std::make_shared<EventArrayType>(event->GetStorage(), arr);
  }

  size_t Dims() const override { return ArrayType::ArrayRank(); }

  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return true; }

  bool operator==(const Type& ty) const override {
    if (auto t = dyn_cast<EventArrayType>(&ty))
      return sbe::must_eq(t->ElemCount(), ElemCount());
    return false;
  }

  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }

  const std::string Name() const override { return event->Name() + " array"; }

  void Print(std::ostream& os) const override {
    event->Print(os);
    ArrayType::Print(os);
  }

  Storage GetStorage() const { return event->GetStorage(); }

  __UDT_TYPE_INFO__(ArrayType, EventArrayType)
};

struct SpannedArrayType final : public ArrayType,
                                public TypeIDProvider<SpannedArrayType> {
  ptr<SpannedType> spty = nullptr;
  explicit SpannedArrayType(BaseType ft, const ptr<MDSpanType>& s,
                            const ValueList& strd, Storage m,
                            const ValueList& ads)
      : ArrayType(ads), spty(std::make_shared<SpannedType>(ft, s, strd, m)) {}

  explicit SpannedArrayType(BaseType ft, const ptr<MDSpanType>& s,
                            const ValueList& strd, Storage m, size_t array_rank)
      : ArrayType(array_rank),
        spty(std::make_shared<SpannedType>(ft, s, strd, m)) {}

  const ptr<Type> CloneImpl() const override {
    return std::make_shared<SpannedArrayType>(
        spty->ElementType(), cast<MDSpanType>(spty->s_type->Clone()),
        spty->GetStrides(), spty->GetStorage(), ArrayType::Dimensions());
  }

  const ptr<Type> ArrayElementType() const override {
    assert(spty);
    return spty;
  }
  bool IsComplete() const override { return spty->IsComplete(); }
  bool HasSufficientInfo() const override { return spty->HasSufficientInfo(); }

  const ptr<Type> SubScriptType(size_t subscription_count) override {
    auto arr = SubScript(subscription_count);
    if (arr.size() == 0)
      return std::make_shared<SpannedType>(spty->e_type, spty->GetMDSpanType(),
                                           spty->GetStrides(),
                                           spty->GetStorage());
    else
      return std::make_shared<SpannedArrayType>(
          spty->e_type, spty->GetMDSpanType(), spty->GetStrides(),
          spty->GetStorage(), arr);
  }

  size_t Dims() const override { return ArrayType::ArrayRank(); }

  bool operator==(const Type& ty) const override {
    if (auto t = dyn_cast<SpannedArrayType>(&ty))
      return sbe::must_eq(t->ElemCount(), ElemCount());
    return false;
  }

  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }

  const std::string Name() const override { return spty->Name() + " array"; }

  void Print(std::ostream& os) const override {
    spty->Print(os);
    ArrayType::Print(os);
  }

  __UDT_TYPE_INFO__(ArrayType, SpannedArrayType)
};

// Specialization: use the same judgement as the corresponding scalar type
template <>
inline bool isa<EventType>(Type* ty) {
  if (!ty) return false;
  if (auto aty = dyn_cast<ArrayType>(ty))
    return aty->ArrayElementType()->IsType(EventType::TypeID());
  return ty->IsType(EventType::TypeID());
}

template <>
inline bool isa<EventType>(const ptr<Type>& ty) {
  return isa<EventType>(ty.get());
}

template <>
inline bool isa<EventType>(const Type* ty) {
  return isa<EventType>(const_cast<Type*>(ty));
}

template <>
inline bool isa<SpannedType>(Type* ty) {
  if (!ty) return false;
  if (auto aty = dyn_cast<ArrayType>(ty))
    return aty->ArrayElementType()->IsType(SpannedType::TypeID());
  return ty->IsType(SpannedType::TypeID());
}

template <>
inline bool isa<SpannedType>(const ptr<Type>& ty) {
  return isa<SpannedType>(ty.get());
}

template <>
inline bool isa<SpannedType>(const Type* ty) {
  return isa<SpannedType>(const_cast<Type*>(ty));
}

template <>
inline ptr<EventType> dyn_cast<EventType>(const ptr<Type>& ty) {
  if (auto aty = dyn_cast<ArrayType>(ty);
      aty && isa<EventType>(aty->ArrayElementType()))
    return std::static_pointer_cast<EventType>(aty->ArrayElementType());
  else if (isa<EventType>(ty))
    return std::static_pointer_cast<EventType>(ty);
  return nullptr;
}

template <>
inline EventType* dyn_cast<EventType>(Type* ty) {
  if (auto aty = dyn_cast<ArrayType>(ty);
      aty && isa<EventType>(aty->ArrayElementType()))
    return (EventType*)(aty->ArrayElementType().get());
  else if (isa<EventType>(ty))
    return (EventType*)ty;
  return nullptr;
}

template <>
inline EventType* dyn_cast<EventType>(const Type* ty) {
  return dyn_cast<EventType>(const_cast<Type*>(ty));
}

template <>
inline ptr<SpannedType> dyn_cast<SpannedType>(const ptr<Type>& ty) {
  if (auto aty = dyn_cast<ArrayType>(ty);
      aty && isa<SpannedType>(aty->ArrayElementType()))
    return std::static_pointer_cast<SpannedType>(aty->ArrayElementType());
  else if (isa<SpannedType>(ty))
    return std::static_pointer_cast<SpannedType>(ty);
  return nullptr;
}

template <>
inline SpannedType* dyn_cast<SpannedType>(Type* ty) {
  if (auto aty = dyn_cast<ArrayType>(ty);
      aty && isa<SpannedType>(aty->ArrayElementType()))
    return (SpannedType*)(aty->ArrayElementType().get());
  else if (isa<SpannedType>(ty))
    return (SpannedType*)ty;
  return nullptr;
}

template <>
inline SpannedType* dyn_cast<SpannedType>(const Type* ty) {
  return dyn_cast<SpannedType>(const_cast<Type*>(ty));
}

inline bool IsActualVectorType(const ptr<Type>& ty) {
  if (isa<VectorType>(ty)) return true;
  if (auto bty = dyn_cast<BoundedIntegerType>(ty))
    return bty->GetWidth() > 1;
  else if (auto bty = dyn_cast<BoundedITupleType>(ty))
    return bty->GetWidths()[bty->Dims() - 1] > 1;
  return false;
}

inline size_t SizeOf(const Type& ty) {
  if (isa<VoidType>(&ty)) return 0;
  if (isa<ScalarType>(&ty))
    return SizeOf(ty.bt);
  else if (isa<BoundedIntegerType>(&ty))
    return 4;
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return t->ByteSize();
  else if (auto vt = dyn_cast<VectorType>(&ty)) {
    return vt->ElemCount() * SizeOf(vt->ElemType());
  }

  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return 0;
}

// This util might be useful to keep symbolic form till runtime
inline std::string SizeExprOf(const Type& ty, bool ULL_suffix = false) {
  if (isa<VoidType>(&ty)) return {};
  if (isa<ScalarType>(&ty))
    return std::to_string(SizeOf(ty.bt));
  else if (isa<BoundedIntegerType>(&ty))
    return "4";
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return t->ByteSizeExpression(ULL_suffix);
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return {};
}

inline std::string ElemCountExprOf(const Type& ty) {
  if (isa<ScalarType>(&ty))
    return "1";
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return t->ElementCountValue()->ToString();
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return {};
}

inline BaseType GetBaseType(const Type& ty) {
  if (isa<VoidType>(&ty))
    return BaseType::VOID;
  else if (isa<ScalarType>(&ty))
    return ty.bt;
  else if (isa<BoundedIntegerType>(&ty))
    return BaseType::S32;
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return t->e_type;
  else if (auto vt = dyn_cast<VectorType>(&ty))
    return vt->ElemType();
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
}

inline BaseType ElementType(const ptr<Type>& ty) {
  if (auto t = dyn_cast<SpannedType>(ty))
    return t->e_type;
  else if (IsActualVectorType(ty)) {
    if (auto vty = dyn_cast<VectorType>(ty)) {
      return vty->ElemType();
    } else if (auto bv = dyn_cast<BoundedType>(ty)) {
      return BaseType::S32;
    }
  }
  choreo_unreachable(STR(*ty) + " does not have an element type.");
  return BaseType::UNKNOWN;
}

inline size_t ElementCount(const ptr<Type>& ty) {
  if (auto t = dyn_cast<SpannedType>(ty))
    return t->ElementCount();
  else if (IsActualVectorType(ty)) {
    if (auto vty = dyn_cast<VectorType>(ty)) {
      return vty->ElemCount();
    } else if (auto bit = dyn_cast<BoundedIntegerType>(ty)) {
      return bit->GetWidth();
    } else if (auto bit = dyn_cast<BoundedITupleType>(ty)) {
      return bit->GetWidths()[bit->Dims() - 1];
    }
  }
  choreo_unreachable(STR(*ty) + " does not have an element count.");
  return 0;
}

inline bool IsActualBoundedIntegerType(const ptr<Type>& ty) {
  if (auto bty = dyn_cast<BoundedType>(ty)) return bty->Dims() == 1;
  return false;
}

inline bool CanYieldAnInteger(const ptr<Type>& ty) {
  return isa<ScalarType>(ty) || IsActualBoundedIntegerType(ty) ||
         (isa<VectorType>(ty) || (isa<ITupleType>(ty) && ty->Dims() == 1));
}

inline bool CanYieldIndex(const ptr<Type>& ty) {
  return isa<ScalarIntegerType>(ty) || isa<BoundedType>(ty) ||
         (isa<ITupleType>(ty));
}

inline bool IntegersOnly(const ptr<Type>& ty) {
  return isa<ScalarIntegerType>(ty) || (isa<ITupleType>(ty));
}

inline bool ConvertibleToInt(const ptr<Type>& ty) {
  return ConvertibleToInt(*ty);
}

inline bool ConvertibleToInt(const Type& ty) {
  return isa<ScalarIntegerType>(&ty) || isa<BooleanType>(&ty) ||
         (isa<ITupleType>(&ty) && ty.Dims() == 1);
}

inline ValueItem GetSingleUpperBound(const ptr<Type>& ty) {
  if (!IsActualBoundedIntegerType(ty))
    choreo_unreachable("can not get the single upper bound for a " + PSTR(ty) +
                       " type.");
  return cast<BoundedType>(ty)->GetUpperBound();
}

inline int GetSingleStep(const ptr<Type>& ty) {
  if (!IsActualBoundedIntegerType(ty))
    choreo_unreachable("can not get the single stride for a " + PSTR(ty) +
                       " type.");
  return cast<BoundedType>(ty)->GetStep();
}

inline int GetSingleWidth(const ptr<Type>& ty) {
  if (!IsActualBoundedIntegerType(ty))
    choreo_unreachable("can not get the single width for a " + PSTR(ty) +
                       " type.");
  return cast<BoundedType>(ty)->GetWidth();
}

// utility functions to generate types
// Note: should always use utility functions
inline Shape GenInvalidShape() { return Shape(); }
inline Shape GenUnknownShape() { return Shape(GetUnknownRank()); }

// To indicate uninferred array dims.
inline ValueList GenUninitValueList(size_t rank) {
  return ValxN(sbe::nu(0), rank);
}

inline ptr<VoidType> MakeVoidType() { return std::make_shared<VoidType>(); }

inline ptr<AddrType> MakeAddrType() { return std::make_shared<AddrType>(); }

inline ptr<StreamType> MakeStreamType() {
  return std::make_shared<StreamType>();
}

inline ptr<UnknownScalarType> MakeUnknownScalarType(bool is_mutable = false) {
  return std::make_shared<UnknownScalarType>(is_mutable);
}

inline ptr<UnknownType> MakeUnknownType() {
  return std::make_shared<UnknownType>();
}

inline ptr<ScalarIntegerType> MakeScalarIntegerType(BaseType t,
                                                    bool m = false) {
  switch (t) {
  case BaseType::U1: return std::make_shared<U1Type>(m);
  case BaseType::BIN1: return std::make_shared<Bin1Type>(m);
  case BaseType::U2: return std::make_shared<U2Type>(m);
  case BaseType::S2: return std::make_shared<S2Type>(m);
  case BaseType::U4: return std::make_shared<U4Type>(m);
  case BaseType::S4: return std::make_shared<S4Type>(m);
  case BaseType::U6: return std::make_shared<U6Type>(m);
  case BaseType::S6: return std::make_shared<S6Type>(m);
  case BaseType::S8: return std::make_shared<S8Type>(m);
  case BaseType::U8: return std::make_shared<U8Type>(m);
  case BaseType::S16: return std::make_shared<S16Type>(m);
  case BaseType::U16: return std::make_shared<U16Type>(m);
  case BaseType::S32: return std::make_shared<S32Type>(m);
  case BaseType::U32: return std::make_shared<U32Type>(m);
  case BaseType::S64: return std::make_shared<S64Type>(m);
  case BaseType::U64: return std::make_shared<U64Type>(m);
  case BaseType::UNKNOWN: return std::make_shared<IntegerType>(m);
  default: {
    choreo_unreachable("unsupported base type for scalar integer type.");
    return nullptr;
  }
  }
}

inline ptr<IntegerType> MakeIntegerType(bool m = false) {
  return std::make_shared<IntegerType>(m);
}

inline const ptr<NoValueType> MakeNoValueType() {
  return std::make_shared<NoValueType>();
}

inline ptr<BooleanType> MakeBooleanType(bool m = false) {
  return std::make_shared<BooleanType>(m);
}

inline ptr<ScalarFloatType> MakeScalarFloatType(BaseType bt, bool m = false) {
  switch (bt) {
  case BaseType::F4_E2M1: return std::make_shared<FloatE2M1Type>(m);
  case BaseType::F6_E3M2: return std::make_shared<FloatE3M2Type>(m);
  case BaseType::F6_E2M3: return std::make_shared<FloatE2M3Type>(m);
  case BaseType::F8_UE8M0: return std::make_shared<FloatUE8M0Type>(m);
  case BaseType::F8_UE4M3: return std::make_shared<FloatUE4M3Type>(m);
  case BaseType::F8_E4M3: return std::make_shared<FloatE4M3Type>(m);
  case BaseType::F8_E5M2: return std::make_shared<FloatE5M2Type>(m);
  case BaseType::F16: return std::make_shared<F16Type>(m);
  case BaseType::BF16: return std::make_shared<BF16Type>(m);
  case BaseType::TF32: return std::make_shared<TF32Type>(m);
  case BaseType::F32: return std::make_shared<F32Type>(m);
  case BaseType::F64: return std::make_shared<F64Type>(m);
  default: choreo_unreachable("unsupported base type.");
  }
  return nullptr;
}

inline static ptr<Type> MakeScalarType(BaseType bt, bool m = false) {
  if (IsIntegerType(bt))
    return MakeScalarIntegerType(bt, m);
  else if (IsFloatType(bt))
    return MakeScalarFloatType(bt, m);
  else if (bt == BaseType::BOOL)
    return MakeBooleanType(m);
  else if (bt == BaseType::UNKSCALAR)
    return MakeUnknownScalarType(m);
  else {
    choreo_unreachable("unsupported base type for scalar type.");
    return nullptr;
  }
}

inline ptr<ScalarFloatType> MakeF32Type(bool m = false) {
  return MakeScalarFloatType(BaseType::F32, m);
}

inline ptr<ScalarFloatType> MakeF64Type(bool m = false) {
  return MakeScalarFloatType(BaseType::F64, m);
}

inline ptr<VectorType> MakeVectorType(BaseType bt, size_t n) {
  return std::make_shared<VectorType>(bt, n);
}

inline ptr<StringType> MakeStringType() {
  return std::make_shared<StringType>();
}

inline ptr<IndexType> MakeIndexType() { return std::make_shared<IndexType>(); }

inline ptr<ITupleType> MakeITupleType(size_t n) {
  return std::make_shared<ITupleType>(n);
}

inline ptr<ITupleType> MakeUninitITupleType() {
  return std::make_shared<ITupleType>();
}

inline ptr<MDSpanType> MakeUninitMDSpanType() {
  return std::make_shared<MDSpanType>(GenUnknownShape());
}

inline ptr<MDSpanType> MakeRankedMDSpanType(size_t n) {
  if (!IsValidRank(n)) return MakeUninitMDSpanType();
  return std::make_shared<MDSpanType>(Shape(n));
}

inline ptr<MDSpanType> MakeMDSpanType(const Shape& v) {
  return std::make_shared<MDSpanType>(v);
}

inline ptr<SpannedType>
MakeDenseSpannedType(BaseType t, const Shape& v,
                     const Storage& s = Storage::DEFAULT) {
  auto strides = v.GenDenseStrides();
  return std::make_shared<SpannedType>(t, MakeMDSpanType(v), strides, s);
}

inline ptr<SpannedType> MakeSpannedType(BaseType t, const Shape& v,
                                        const ValueList& strd,
                                        const Storage& s = Storage::DEFAULT) {
  return std::make_shared<SpannedType>(t, MakeMDSpanType(v), strd, s);
}

// all the values are fake. it is used only to indicate a spanned type without
// the shape detail
inline ptr<SpannedType> MakeDummySpannedType() {
  return MakeSpannedType(BaseType::UNKSCALAR, GenUnknownShape(), {},
                         Storage::DEFAULT);
}

inline ptr<SpannedType>
MakeUnRankedSpannedType(BaseType bt, Storage sto = Storage::DEFAULT) {
  // only care about the rank of span
  return MakeSpannedType(bt, GenUnknownShape(), {}, sto);
}

inline ptr<SpannedType> MakeRankedSpannedType(size_t n,
                                              BaseType bt = BaseType::UNKSCALAR,
                                              Storage sto = Storage::DEFAULT) {
  if (!IsValidRank(n)) MakeUnRankedSpannedType(bt, sto);
  // only care about the rank of span
  return MakeSpannedType(bt, Shape(n), {}, sto);
}

inline ptr<SpannedType>
MakeShapedDenseSpannedType(const Shape& s, BaseType bt = BaseType::UNKSCALAR) {
  // only care about the precise shape
  return MakeDenseSpannedType(bt, s, Storage::DEFAULT);
}

inline ptr<SpannedType>
MakeShapedStridedSpannedType(const Shape& s, const ValueList& strd,
                             BaseType bt = BaseType::UNKSCALAR) {
  // only care about the precise shape
  return MakeSpannedType(bt, s, strd, Storage::DEFAULT);
}

inline ptr<BoundedIntegerType> MakeBoundedIntegerType(int ub) {
  return std::make_shared<BoundedIntegerType>(sbe::nu(0), sbe::nu(ub));
}

inline ptr<BoundedIntegerType> MakeBoundedIntegerType(const std::string& ub) {
  return std::make_shared<BoundedIntegerType>(sbe::nu(0), sbe::sym(ub));
}

inline ptr<BoundedIntegerType> MakeBoundedIntegerType(const ValueItem& ub) {
  return std::make_shared<BoundedIntegerType>(sbe::nu(0), ub);
}

inline ptr<BoundedIntegerType> MakeUnknownBoundedIntegerType() {
  return std::make_shared<BoundedIntegerType>();
}

inline ptr<BoundedITupleType> MakeBoundedITupleType(const MultiBounds& ub,
                                                    const std::string& k = "",
                                                    const std::string& v = "") {
  MultiBounds lb(ub.DimCount(), sbe::nu(0));
  IntegerList s(ub.DimCount());
  std::fill(s.begin(), s.end(), 1);
  return std::make_shared<BoundedITupleType>(lb, ub, s, k, v);
}

inline ptr<BoundedITupleType> MakeBoundedITupleType(const MultiBounds& lb,
                                                    const MultiBounds& ub,
                                                    const std::string& k = "",
                                                    const std::string& v = "") {
  IntegerList s(ub.DimCount());
  std::fill(s.begin(), s.end(), 1);
  return std::make_shared<BoundedITupleType>(lb, ub, s, k, v);
}

inline ptr<BoundedITupleType> MakeBoundedITupleType(const MultiBounds& lb,
                                                    const MultiBounds& ub,
                                                    const IntegerList& il,
                                                    const std::string& k = "",
                                                    const std::string& v = "") {
  return std::make_shared<BoundedITupleType>(lb, ub, il, k, v);
}

inline ptr<BoundedITupleType>
MakeBoundedITupleType(const MultiBounds& lb, const MultiBounds& ub,
                      const IntegerList& il, const IntegerList& wl,
                      const std::string& k = "", const std::string& v = "") {
  return std::make_shared<BoundedITupleType>(lb, ub, il, wl, k, v);
}

inline ptr<BoundedITupleType> MakeUninitBoundedITupleType() {
  return std::make_shared<BoundedITupleType>(
      GenUnknownShape(), GenUnknownShape(), IntegerList(), "");
}

inline ptr<ArrayType> MakeArrayType(const ValueList& ad) {
  return std::make_shared<ArrayType>(ad);
}

inline ptr<ArrayType> MakeRankedArrayType(size_t rank) {
  return std::make_shared<ArrayType>(rank);
}

inline ptr<EventType> MakeEventType(Storage s) {
  return std::make_shared<EventType>(s);
}

inline ptr<EventArrayType> MakeEventArrayType(Storage s, const ValueList& ad) {
  return std::make_shared<EventArrayType>(s, ad);
}

inline ptr<SpannedArrayType>
MakeDenseSpannedArrayType(BaseType bt, const Shape& v, const ValueList& ad,
                          const Storage& s = Storage::DEFAULT) {
  auto strides = v.GenDenseStrides();
  return std::make_shared<SpannedArrayType>(bt, MakeMDSpanType(v), strides, s,
                                            ad);
}

inline ptr<SpannedArrayType>
MakeRankedSpannedArrayType(size_t n, const ValueList& ad,
                           BaseType bt = BaseType::UNKSCALAR,
                           const Storage& s = Storage::DEFAULT) {
  // only care about the rank of span
  return std::make_shared<SpannedArrayType>(bt, MakeMDSpanType(Shape(n)),
                                            ValueList{}, s, ad);
}

inline ptr<SpannedArrayType>
MakeUnRankedSpannedArrayType(BaseType bt, const ValueList& ad,
                             const Storage& sto = Storage::DEFAULT) {
  // only care about the rank of span
  return std::make_shared<SpannedArrayType>(
      bt, MakeMDSpanType(GenUnknownShape()), ValueList{}, sto, ad);
}

inline ptr<SpannedArrayType>
MakeStridedSpannedArrayType(BaseType ft, const Shape& v, const ValueList& strd,
                            const ValueList& ad,
                            const Storage& s = Storage::DEFAULT) {
  return std::make_shared<SpannedArrayType>(ft, MakeMDSpanType(v), strd, s, ad);
}

// like MakeDummySpannedType, but the array dim are real.;
inline ptr<SpannedArrayType> MakeDummySpannedArrayType(const ValueList& ad) {
  return std::make_shared<SpannedArrayType>(BaseType::UNKSCALAR,
                                            MakeMDSpanType(GenUnknownShape()),
                                            ValueList{}, Storage::DEFAULT, ad);
}

inline ptr<FutureType> MakeFutureType(const ptr<SpannedType>& v, bool async) {
  return std::make_shared<FutureType>(v, async);
}

inline ptr<FutureType> MakeRankedFutureType(size_t n, bool async) {
  return std::make_shared<FutureType>(MakeRankedSpannedType(n), async);
}

inline ptr<FutureType> MakeShapedFutureType(const Shape& v, bool async,
                                            const ValueList strd,
                                            BaseType bt = BaseType::UNKNOWN) {
  return std::make_shared<FutureType>(MakeShapedStridedSpannedType(v, strd, bt),
                                      async);
}

inline ptr<FutureType> MakeDummyFutureType(bool async) {
  return std::make_shared<FutureType>(MakeDummySpannedType(), async);
}

inline ptr<PlaceHolderType> MakePlaceHolderMDSpanType() {
  return std::make_shared<PlaceHolderType>(BaseType::PARTIAL);
}

inline ptr<PlaceHolderType> MakePlaceHolderSpannedType() {
  return std::make_shared<PlaceHolderType>(BaseType::SPANNED);
}

inline ptr<PlaceHolderType> MakePlaceHolderFutureType() {
  return std::make_shared<PlaceHolderType>(BaseType::FUTURE);
}

inline ptr<DeviceDataType> MakeDeviceDataType(const std::string& name,
                                              BaseType ty,
                                              const std::string& attr = "",
                                              bool ip = false,
                                              const std::string& init = "") {
  return std::make_shared<DeviceDataType>(name, attr, ty, ip, init);
}

// only return spanned type, scalar type, void type and unknown type
inline ptr<Type> MakeChoreoDataType(const ptr<DeviceDataType>& dt) {
  if (dt->is_pointer) {
    if (IsNumericType(dt->data_type) || dt->data_type == BaseType::VOID)
      return MakeUnRankedSpannedType(dt->data_type);
    else
      return MakeDummySpannedType();
  } else {
    if (IsNumericType(dt->data_type) || dt->data_type == BaseType::BOOL)
      return MakeScalarType(dt->data_type, true);
    else if (dt->data_type == BaseType::VOID)
      return MakeVoidType();
    else if (dt->data_type == BaseType::UNKNOWN)
      return MakeUnknownType();
    else
      choreo_unreachable("unsupported DeviceDataType.");
  }
}

inline ptr<FunctionType> MakeFunctionType(const ptr<Type> ot,
                                          const std::vector<ptr<Type>>& its) {
  return std::make_shared<FunctionType>(ot, its);
}

// map default to global
inline static ptr<Type> ShadowTypeStorage(const ptr<Type>& ty) {
  if (auto sty = dyn_cast<SpannedType>(ty)) {
    if (auto at = dyn_cast<ArrayType>(ty))
      return MakeStridedSpannedArrayType(sty->ElementType(), sty->GetShape(),
                                         sty->GetStrides(), at->dims,
                                         ProjectStorage(sty->GetStorage()));
    else
      return MakeSpannedType(sty->ElementType(), sty->GetShape(),
                             sty->GetStrides(),
                             ProjectStorage(sty->GetStorage()));
  } else if (auto fty = dyn_cast<FutureType>(ty)) {
    auto sty = fty->GetSpannedType();
    return MakeFutureType(MakeSpannedType(sty->ElementType(), sty->GetShape(),
                                          sty->GetStrides(),
                                          ProjectStorage(sty->GetStorage())),
                          fty->IsAsync());
  } else
    return ty;
}

inline static ptr<SpannedType> GetSpannedType(const ptr<Type>& ty) {
  if (auto fty = dyn_cast<FutureType>(ty))
    return fty->GetSpannedType();
  else if (auto sty = dyn_cast<SpannedType>(ty))
    return sty;
  else if (auto saty = dyn_cast<SpannedArrayType>(ty))
    return saty->spty;
  return nullptr;
}

inline static ptr<MDSpanType> GetMDSpanType(const ptr<Type>& ty) {
  if (auto fty = dyn_cast<FutureType>(ty))
    return fty->GetSpannedType()->GetMDSpanType();
  else if (auto sty = dyn_cast<SpannedType>(ty))
    return sty->GetMDSpanType();
  else if (auto mty = dyn_cast<MDSpanType>(ty))
    return mty;
  else
    return nullptr;
}

inline static Shape GetShape(const ptr<Type>& ty) {
  if (auto mty = dyn_cast<MDSpanType>(ty))
    return mty->GetShape();
  else if (auto sty = dyn_cast<SpannedType>(ty))
    return sty->GetShape();
  else if (auto fty = dyn_cast<FutureType>(ty))
    return fty->GetShape();
  else if (auto bty = dyn_cast<BoundedITupleType>(ty))
    return bty->GetSizes();

  return Shape(); // avoid warning
}

inline static ValueList GetArrayDimensions(const ptr<Type>& ty) {
  if (auto aty = dyn_cast<ArrayType>(ty)) return aty->Dimensions();
  return {};
}

inline static bool GeneralFutureType(const Type& ty) {
  return ty.GetBaseType() == BaseType::FUTURE;
}

inline static bool GeneralFutureType(const ptr<Type>& ty) {
  if (!ty) return false;
  return GeneralFutureType(*ty);
}

// if type a has better quality than type b
inline bool BetterQuality(const ptr<Type>& a, const ptr<Type>& b) {
  if (a->HasSufficientInfo() && !b->HasSufficientInfo()) return true;

  auto a_sty = GetSpannedType(a);
  auto b_sty = GetSpannedType(b);
  if (a_sty && b_sty &&
      ((b_sty->ElementType() == BaseType::UNKNOWN) &&
       (a_sty->ElementType() != BaseType::UNKNOWN)))
    return a_sty->GetShape() == b_sty->GetShape();

  if (!a->ApprxEqual(*b)) return false;
  if (*a == *b) return false;

  return false;
}

inline static BaseType GetUnderlyingType(const ptr<Type>& ty) {
  if (isa<ScalarType>(ty))
    return ty->GetBaseType();
  else if (auto sty = GetSpannedType(ty))
    return sty->ElementType();
  else if (CanYieldAnInteger(ty))
    return BaseType::S32;
  return BaseType::UNKNOWN;
}

struct PromoteResult {
  BaseType lty;
  BaseType rty;
};

inline PromoteResult PromoteType(BaseType lty, BaseType rty) {
  using BT = BaseType;

  for (const auto ty : {lty, rty})
    if (!(IsIntegerType(ty) || IsFloatType(ty) || ty == BaseType::UNKSCALAR))
      choreo_unreachable("unexpect type in promote: " + STR(ty));

  PromoteResult res{.lty = lty, .rty = rty};

  if (lty == BaseType::UNKSCALAR || rty == BaseType::UNKSCALAR) return res;

  if (lty == rty) return res;

  if (IsFloatType(lty) && IsFloatType(rty)) {
    // both floating-point
    if (IsValuePreservingCast(lty, rty))
      res.lty = res.rty = rty;
    else if (IsValuePreservingCast(rty, lty))
      res.lty = res.rty = lty;
    else
      choreo_unreachable("unexpect floating-point types in promote: " +
                         STR(lty) + " and " + STR(rty));

    return res;
  } else if (IsFloatType(lty) || IsFloatType(rty)) {
    // only one is floating-point
    if (IsFloatType(lty))
      res.rty = lty;
    else
      res.lty = rty;
    return res;
  } else {
    auto IntegralPromotion = [](BT bt) -> BT {
      // maybe need promote all the integer type smaller than int
      // to int first which is processed in cpp.
      return bt;
    };
    auto CanRepresent = [](BT signed_type, BT unsigned_type) -> bool {
      // return if `signed_type` can represent `unsigned_type`
      assert(IsSignedType(signed_type));
      assert(IsUnsignedType(unsigned_type));
      switch (signed_type) {
      case BT::S64:
        return unsigned_type == BT::U32 || unsigned_type == BT::U16 ||
               unsigned_type == BT::U8;
      case BT::S32: return unsigned_type == BT::U16 || unsigned_type == BT::U8;
      case BT::S16: return unsigned_type == BT::U8;
      default: return false;
      }
    };
    auto UnsignedVersion = [](BaseType signed_type) {
      assert(IsSignedType(signed_type));
      switch (signed_type) {
      case BaseType::S64: return BaseType::U64;
      case BaseType::S32: return BaseType::U32;
      case BaseType::S16: return BaseType::U16;
      case BaseType::S8: return BaseType::U8;
      default: choreo_unreachable("unexpect signed type!");
      }
    };

    lty = IntegralPromotion(lty);
    rty = IntegralPromotion(rty);

    int rank_l = BaseTypeRank(lty);
    int rank_r = BaseTypeRank(rty);

    // both unsigned or both
    if ((IsUnsignedType(lty) && IsUnsignedType(rty)) ||
        (IsSignedType(lty) && IsSignedType(rty))) {
      if (rank_l > rank_r)
        res.rty = lty;
      else
        res.lty = rty;
      return res;
    }

    // one unsigned, one signed
    if (IsUnsignedType(lty) && IsSignedType(rty)) {
      if (rank_l > rank_r) // use the unsigned
        res.rty = lty;
      else if (CanRepresent(rty, lty)) // rty can represent lty
        res.lty = rty;
      else // both to the unsigned version of rty
        res.lty = res.rty = UnsignedVersion(rty);
      return res;
    }
    if (IsSignedType(lty) && IsUnsignedType(rty)) {
      if (rank_r > rank_l)
        res.lty = rty;
      else if (CanRepresent(lty, rty))
        res.rty = lty;
      else
        res.lty = res.rty = UnsignedVersion(lty);
      return res;
    }

    assert(false);
    return res;
  }
}

inline static ptr<Type> MutateType(const ptr<Type>& ty) {
  auto sty = dyn_cast<ScalarType>(ty);
  if (!sty) return ty->Clone();
  return MakeScalarType(sty->GetBaseType(), true);
}

inline bool IsMutable(const Type& ty) {
  auto sty = dyn_cast<ScalarType>(&ty);
  if (!sty) {
    // spanned are always mutable
    if (isa<SpannedType>(&ty))
      return true;
    else
      return false;
  }
  return sty->IsMutable();
}

inline bool MutableType(const Type& ty) { return isa<ScalarType>(&ty); }

inline bool SupportIntListCollapse(const ptr<Type>& ty) {
  return isa<ScalarIntegerType>(ty) || isa<MDSpanType>(ty) ||
         isa<ITupleType>(ty);
}

inline bool CanYieldDimension(const ptr<Type>& ty) {
  return (isa<ScalarIntegerType>(ty) || (isa<ITupleType>(ty))) &&
         !IsMutable(*ty);
}

inline bool CompatibleRank(size_t r1, size_t r2) {
  return r1 == r2 || IsUnknownRank(r1) || IsUnknownRank(r2);
}

} // end namespace Choreo

#endif // __CHOREO_TYPES_H__
