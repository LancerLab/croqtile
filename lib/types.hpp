#ifndef __CHOREO_TYPES_H__
#define __CHOREO_TYPES_H__

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>

// to avoid definition error
namespace Choreo {
enum class Storage { SUB, LOCAL, SHARED, GLOBAL, DEFAULT, NONE };
enum class CompileTarget;
} // namespace Choreo

#include "aux.hpp"
#include "context.hpp"
#include "io.hpp"
#include "utils.hpp"

namespace Choreo {

enum class TypeCategory {
  INT,
  BOOL,
  HALF8,
  HALF,
  BFP16,
  FLOAT,
  DOUBLE,
  INDEX,
  ITUPLE,
  PARTIAL,
  SPANNED,
  BOUNDED_INT,
  BOUNDED_ITUPLE,
  VOID,
  FUTURE,
  EVENT,
  ARRAY,
  STRING,
  FUNCTION,
  UNKNOWN,
};

inline static std::string STR(TypeCategory tc) {
  switch (tc) {
  case TypeCategory::INT: return "INT";
  case TypeCategory::BOOL: return "BOOL";
  case TypeCategory::HALF8: return "HALF8";
  case TypeCategory::HALF: return "HALF";
  case TypeCategory::BFP16: return "BFP16";
  case TypeCategory::FLOAT: return "FLOAT";
  case TypeCategory::DOUBLE: return "DOUBLE";
  case TypeCategory::INDEX: return "INDEX";
  case TypeCategory::ITUPLE: return "ITUPLE";
  case TypeCategory::PARTIAL: return "PARTIAL";
  case TypeCategory::SPANNED: return "SPANNED";
  case TypeCategory::BOUNDED_INT: return "BOUNDED_INT";
  case TypeCategory::BOUNDED_ITUPLE: return "BOUNDED_ITUPLE";
  case TypeCategory::VOID: return "VOID";
  case TypeCategory::FUTURE: return "FUTURE";
  case TypeCategory::EVENT: return "EVENT";
  case TypeCategory::ARRAY: return "ARRAY";
  case TypeCategory::FUNCTION: return "FUNCTION";
  case TypeCategory::STRING: return "STRING";
  case TypeCategory::UNKNOWN: return "UNKNOWN";
  default: choreo_unreachable("unsupported type category.");
  }
  return "";
}

// BaseType, FundamentalType, and ScalarType
// if the type needs deduction, mark it as 'UNKNOWN'
enum class BaseType {
  F32,
  F16,
  BF16,
  F8,
  U32,
  S32,
  U16,
  S16,
  U8,
  S8,
  INT,
  BOOL,
  HALF8,
  HALF,
  BFP16,
  FLOAT,
  DOUBLE,
  ITUPLE,
  EVENT,
  ARRAY,
  VOID,
  UNKNOWN
};

enum class FundamentalType {
  F32 = (int)BaseType::F32,
  F16 = (int)BaseType::F16,
  BF16 = (int)BaseType::BF16,
  F8 = (int)BaseType::F8,
  U32 = (int)BaseType::U32,
  U16 = (int)BaseType::U16,
  U8 = (int)BaseType::U8,
  S32 = (int)BaseType::S32,
  S16 = (int)BaseType::S16,
  S8 = (int)BaseType::S8,
  UND = (int)BaseType::UNKNOWN,
};

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

inline BaseType TC2BT(TypeCategory tc) {
  switch (tc) {
  case TypeCategory::INT: return BaseType::INT;
  case TypeCategory::BOOL: return BaseType::BOOL;
  case TypeCategory::HALF8: return BaseType::HALF8;
  case TypeCategory::HALF: return BaseType::HALF;
  case TypeCategory::BFP16: return BaseType::BFP16;
  case TypeCategory::FLOAT: return BaseType::FLOAT;
  case TypeCategory::DOUBLE: return BaseType::DOUBLE;
  case TypeCategory::VOID: return BaseType::VOID;
  default:
    choreo_unreachable("unsupported mapping from type category to base type.");
  }

  return BaseType::UNKNOWN;
}

inline static size_t SizeOf(FundamentalType ft) {
  switch (ft) {
  case FundamentalType::F32:
  case FundamentalType::U32:
  case FundamentalType::S32: return 4;
  case FundamentalType::F16:
  case FundamentalType::BF16:
  case FundamentalType::U16:
  case FundamentalType::S16: return 2;
  case FundamentalType::F8:
  case FundamentalType::U8:
  case FundamentalType::S8: return 1;
  default: choreo_unreachable("fundamental type is not supported.");
  }
  return 0;
}

inline static size_t SizeOf(BaseType bt) {
  switch (bt) {
  case BaseType::DOUBLE: return sizeof(double);
  case BaseType::F32:
  case BaseType::U32:
  case BaseType::S32:
  case BaseType::INT:
  case BaseType::FLOAT: return 4;
  case BaseType::F16:
  case BaseType::BF16:
  case BaseType::U16:
  case BaseType::S16: return 2;
  case BaseType::F8:
  case BaseType::U8:
  case BaseType::S8: return 1;
  case BaseType::BOOL: return sizeof(bool);
  default: choreo_unreachable("base type is not supported.");
  }
  return 0;
}

// utility functions to map types to strings, and the opposite.
inline static BaseType BaseTypeFromString(const std::string& input) {
  static const std::unordered_map<std::string, BaseType> typeMap = {
      {"f32", BaseType::F32},     {"f16", BaseType::F16},
      {"bf16", BaseType::BF16},   {"u32", BaseType::U32},
      {"s32", BaseType::S32},     {"u16", BaseType::U16},
      {"s16", BaseType::S16},     {"u8", BaseType::U8},
      {"s8", BaseType::S8},       {"f8", BaseType::F8},
      {"half8", BaseType::HALF8}, {"half", BaseType::HALF},
      {"float", BaseType::FLOAT}, {"double", BaseType::DOUBLE},
      {"bfp16", BaseType::BFP16}, {"int", BaseType::INT},
      {"bool", BaseType::BOOL},   {"ituple", BaseType::ITUPLE},
      {"event", BaseType::EVENT}, {"array", BaseType::ARRAY},
      {"void", BaseType::VOID},   {"unknown", BaseType::UNKNOWN},
  };

  auto it = typeMap.find(input);
  assert(it != typeMap.end() && "incorrect type string");

  return it->second;
}

namespace __internal__ {

inline static std::string GetStringFrom(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F32, "f32"},     {BaseType::F16, "f16"},
      {BaseType::BF16, "bf16"},   {BaseType::U32, "u32"},
      {BaseType::S32, "s32"},     {BaseType::U16, "u16"},
      {BaseType::S16, "s16"},     {BaseType::U8, "u8"},
      {BaseType::S8, "s8"},       {BaseType::F8, "f8"},
      {BaseType::HALF8, "half8"}, {BaseType::HALF, "half"},
      {BaseType::BFP16, "bfp16"}, {BaseType::INT, "int"},
      {BaseType::FLOAT, "float"}, {BaseType::DOUBLE, "double"},
      {BaseType::BOOL, "bool"},   {BaseType::ITUPLE, "ituple"},
      {BaseType::EVENT, "event"}, {BaseType::ARRAY, "array"},
      {BaseType::VOID, "void"},   {BaseType::UNKNOWN, "unknown"},
  };

  auto it = enumToString.find(dataType);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

inline static std::string GetStringFrom(Storage st) {
  static const std::unordered_map<Storage, std::string> enumToString = {
      {Storage::SUB, "sub-local"}, {Storage::LOCAL, "local"},
      {Storage::GLOBAL, "global"}, {Storage::SHARED, "shared"},
      {Storage::NONE, "none"},     {Storage::DEFAULT, "default"},
  };

  auto it = enumToString.find(st);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

} // end namespace __internal__

inline static std::string STR(size_t sz) { return std::to_string(sz); }
inline static std::string STR(BaseType bt) {
  return __internal__::GetStringFrom(bt);
}
inline static std::string STR(FundamentalType ft) { return STR((BaseType)ft); }
inline static std::string STR(Storage st) {
  return __internal__::GetStringFrom(st);
}

// safe version for pointers
template <typename T>
inline static std::string PSTR(T* pt) {
  if (!pt) return "invalid";
  return STR(*pt);
}

template <typename T>
inline static std::string PSTR(const ptr<T>& pt) {
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
  Shape(const ValueList& v) { val_no = values.Insert(v); }
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

  bool IsRanked() const {
    // ok to hold a valid rank only
    if (!IsValidValueNumber(val_no)) return IsValidRank(dim_count);
    return dim_count == values[val_no].size();
  }

  bool IsValid() const {
    if (!IsValidValueNumber(val_no)) return false;
    return dim_count == values[val_no].size();
  }

  const ValueList& Value() const {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    return values[val_no];
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

  const ValueItem& ValueAt(size_t index) const {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    if (index > dim_count)
      choreo_unreachable("index '" + std::to_string(index) +
                         "' exceeds rank: " + std::to_string(dim_count) + ".");
    return values[val_no].at(index);
  }

  int NthInteger(size_t index) const {
    return cast<sbe::NumericValue>(Value().at(index))->Value();
  }

  std::optional<IntegerList> GetIntList() const {
    IntegerList int_list;
    for (auto v : Value()) {
      if (auto pint = dyn_cast<sbe::NumericValue>(v))
        int_list.push_back(pint->Value());
      else
        return std::nullopt;
    }
    return int_list;
  }

  bool IsDynamic() const {
    for (auto v : Value())
      if (!isa<sbe::NumericValue>(v))
        return true; // a symbolic value represents that the value is decided
                     // at runtime
    return false;
  }

  // retrieve the dimensions that are dynamic
  std::vector<std::pair<int, ValueExpr>> GetDynamicDims() const {
    std::vector<std::pair<int, ValueExpr>> res;
    size_t i = 0;
    for (auto& v : Value()) {
      if (!isa<sbe::NumericValue>(v)) res.emplace_back(i, PSTR(v));
      ++i;
    }
    return res;
  }

  std::string GetElementCountExpression(bool ULL_suffix = false) const {
    if (!IsDynamic())
      return std::to_string(ElementCount()) + (ULL_suffix ? "ULL" : "");

    assert(!Value().empty() && "no values inside the shape.");
    std::string res;
    res = "(" + ValueItemAsString(Value()[0], ULL_suffix) + ")";
    for (size_t i = 1; i < Value().size(); ++i)
      res += " * (" + ValueItemAsString(Value()[i], ULL_suffix) + ")";
    return res;
  }

  std::optional<std::vector<size_t>> GetUIntList() const {
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

  IntegerList IntList() const {
    auto ilist = GetIntList();
    if (!ilist) choreo_unreachable("fail to get an integer list.");
    return *ilist;
  }

  size_t ElementCount() const {
    auto ilist = IntList();
    size_t sz = 1;
    for (int s : ilist) {
      if (s < 0) choreo_unreachable("negative value is found.");
      sz *= s;
    }
    return sz;
  }

  ValueItem ElementCountValue() const {
    auto vi = sbe::nu(1);
    for (auto v : Value()) vi = vi * v;
    return vi;
  }

  void Print(std::ostream& os) const {
    if (!IsValidValueNumber(val_no)) {
      os << "[]";
    } else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueList(Value(), os);
    }
  }

  void PrintAsListSquared(std::ostream& os) const {
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
         isValueListEqual(lhs.Value(), rhs.Value());
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

//
struct Type {
  TypeCategory tc = TypeCategory::UNKNOWN;
  Type(TypeCategory t) : tc(t) {}

  virtual TypeCategory Category() const { return tc; }
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

  virtual void Print(std::ostream&) const = 0;
  virtual const std::string Name() const = 0;

  // codegen util for emitting target's code in string format
  virtual std::string EmitTo(CompileTarget) const {
    assert(false && "Emit stringify not impled for this type");
    return "";
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

inline std::string STR(const ValueItem& vi) {
  if (!vi) return "invalid";
  return vi->ToString();
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
  explicit VoidType() : Type(TypeCategory::VOID) {}
  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "void"; }
  const std::string Name() const override { return "void_type"; }
  bool HasSufficientInfo() const { return true; }

  bool operator==(const Type& ty) const override { return isa<VoidType>(&ty); }
  bool ApprxEqual(const Type& ty) const override { return isa<VoidType>(&ty); }

  __UDT_TYPE_INFO__(Type, VoidType)
};

// The type is unknown. It requires type inference
struct UnknownType final : public Type, public TypeIDProvider<UnknownType> {
  explicit UnknownType() : Type(TypeCategory::UNKNOWN) {}
  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return false; }
  void Print(std::ostream& os) const override { os << "unknown"; }
  const std::string Name() const override { return "unknown_type"; }
  bool HasSufficientInfo() const { return false; }

  // Not comparable
  bool operator==(const Type&) const override { return false; }
  bool ApprxEqual(const Type&) const override { return false; }

  __UDT_TYPE_INFO__(Type, UnknownType)
};

struct PlaceHolderType final : public Type,
                               public TypeIDProvider<PlaceHolderType> {
  PlaceHolderType(TypeCategory t) : Type(t) {}
  size_t Dims() const override { return 0; }
  bool IsComplete() const { return false; }
  void Print(std::ostream& os) const override {
    os << "placeholder<" << STR(Category()) << ">";
  }
  const std::string Name() const override { return "place_holder"; }
  bool HasSufficientInfo() const { return false; }

  bool operator==(const Type&) const override { return false; }
  // tolerate im-precise comparison
  bool ApprxEqual(const Type& t) const override {
    return t.Category() == Category();
  }

  __UDT_TYPE_INFO__(Type, PlaceHolderType)
};

struct ScalarType : public Type, public TypeIDProvider<ScalarType> {
  bool is_mutable = false;
  ScalarType(TypeCategory t, bool m) : Type(t), is_mutable(m) {}
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return true; }
  virtual bool IsMutable() const { return is_mutable; }
  virtual void SetMutable(bool m) { is_mutable = m; }
  virtual ptr<ScalarType> Clone(bool m) const = 0;

  bool operator==(const Type& ty) const override {
    if (auto sty = dyn_cast<ScalarType>(&ty))
      return sty->is_mutable == is_mutable;
    return false;
  }

  virtual bool IsFloat() const { return false; }
  virtual bool IsBoolInteger() const { return true; }
  // can not have instance

  void Print(std::ostream& os) const override {
    if (is_mutable) os << "mutable ";
  }

  __UDT_TYPE_INFO__(Type, ScalarType)
};

inline bool ConvertibleToInt(const Type& ty);

struct IntegerType final : public ScalarType,
                           public TypeIDProvider<IntegerType> {
  ValueItem value = GetInvalidValueItem(); // optional value expression
  IntegerType(bool m) : ScalarType(TypeCategory::INT, m) {}
  IntegerType(const ValueItem& vi, bool m)
      : ScalarType(TypeCategory::INT, m), value(vi) {}

  ptr<ScalarType> Clone(bool m) const {
    return std::make_shared<IntegerType>(m);
  }
  void Print(std::ostream& os) const override {
    ScalarType::Print(os);
    os << "int" << (IsValidValueItem(value) ? (" [" + STR(value) + "]") : "");
  }
  const std::string Name() const override { return "integer"; }
  std::optional<ValueItem> GetValidExpression() const {
    if (IsValidValueItem(value))
      return value;
    else
      return std::nullopt;
  }

  bool operator==(const Type& ty) const override {
    return isa<IntegerType>(&ty) && ScalarType::operator==(ty);
  }
  bool ApprxEqual(const Type& ty) const override {
    return isa<IntegerType>(&ty);
  }
  bool LogicalEqual(const Type& ty) const override {
    return ConvertibleToInt(ty);
  }

  __UDT_TYPE_INFO__(ScalarType, IntegerType)
};

struct ScalarFloatType : public ScalarType,
                         public TypeIDProvider<ScalarFloatType> {
  ScalarFloatType(TypeCategory t, bool m) : ScalarType(t, m) {}
  bool IsFloat() const override { return true; }
  bool IsBoolInteger() const override { return false; }
  __UDT_TYPE_INFO__(ScalarType, ScalarFloatType)
};

struct Half8Type final : public ScalarFloatType,
                         public TypeIDProvider<Half8Type> {
  Half8Type(bool m) : ScalarFloatType(TypeCategory::HALF8, m) {}
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<Half8Type>(m);
  }
  void Print(std::ostream& os) const override {
    ScalarType::Print(os);
    os << "half8";
  }
  const std::string Name() const override { return "half8"; }

  bool operator==(const Type& ty) const override {
    return isa<Half8Type>(&ty) && ScalarType::operator==(ty);
  }
  bool ApprxEqual(const Type& ty) const override { return isa<Half8Type>(&ty); }

  __UDT_TYPE_INFO__(ScalarFloatType, Half8Type)
};

struct HalfType final : public ScalarFloatType,
                        public TypeIDProvider<HalfType> {
  HalfType(bool m) : ScalarFloatType(TypeCategory::HALF, m) {}
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<HalfType>(m);
  }
  void Print(std::ostream& os) const override {
    ScalarType::Print(os);
    os << "half";
  }
  const std::string Name() const override { return "half"; }

  bool operator==(const Type& ty) const override {
    return isa<HalfType>(&ty) && ScalarType::operator==(ty);
  }
  bool ApprxEqual(const Type& ty) const override { return isa<HalfType>(&ty); }

  __UDT_TYPE_INFO__(ScalarFloatType, HalfType)
};

struct BFP16Type final : public ScalarFloatType,
                         public TypeIDProvider<BFP16Type> {
  BFP16Type(bool m) : ScalarFloatType(TypeCategory::BFP16, m) {}
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<BFP16Type>(m);
  }
  void Print(std::ostream& os) const override {
    ScalarType::Print(os);
    os << "bfp16";
  }
  const std::string Name() const override { return "bfp16"; }

  bool operator==(const Type& ty) const override {
    return isa<BFP16Type>(&ty) && ScalarType::operator==(ty);
  }
  bool ApprxEqual(const Type& ty) const override { return isa<BFP16Type>(&ty); }

  __UDT_TYPE_INFO__(ScalarFloatType, BFP16Type)
};

struct FloatType final : public ScalarFloatType,
                         public TypeIDProvider<FloatType> {
  FloatType(bool m) : ScalarFloatType(TypeCategory::FLOAT, m) {}
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<FloatType>(m);
  }
  void Print(std::ostream& os) const override {
    ScalarType::Print(os);
    os << "float";
  }
  const std::string Name() const override { return "float"; }

  bool operator==(const Type& ty) const override {
    return isa<FloatType>(&ty) && ScalarType::operator==(ty);
  }
  bool ApprxEqual(const Type& ty) const override { return isa<FloatType>(&ty); }

  __UDT_TYPE_INFO__(ScalarFloatType, FloatType)
};

struct DoubleType final : public ScalarFloatType,
                          public TypeIDProvider<DoubleType> {
  DoubleType(bool m) : ScalarFloatType(TypeCategory::DOUBLE, m) {}
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<DoubleType>(m);
  }
  void Print(std::ostream& os) const override {
    ScalarType::Print(os);
    os << "double";
  }
  const std::string Name() const override { return "double"; }

  bool operator==(const Type& ty) const override {
    return isa<DoubleType>(&ty) && ScalarType::operator==(ty);
  }
  bool ApprxEqual(const Type& ty) const override {
    return isa<DoubleType>(&ty);
  }

  __UDT_TYPE_INFO__(ScalarFloatType, DoubleType)
};

struct BooleanType final : public ScalarType,
                           public TypeIDProvider<BooleanType> {
  BooleanType(bool m) : ScalarType(TypeCategory::BOOL, m) {}
  ptr<ScalarType> Clone(bool m) const override {
    return std::make_shared<BooleanType>(m);
  }
  void Print(std::ostream& os) const override {
    ScalarType::Print(os);
    os << "bool";
  }
  const std::string Name() const override { return "boolean"; }

  bool operator==(const Type& ty) const override {
    return isa<BooleanType>(&ty) && ScalarType::operator==(ty);
  }
  bool ApprxEqual(const Type& ty) const override {
    return isa<BooleanType>(&ty);
  }
  __UDT_TYPE_INFO__(ScalarType, BooleanType)
};

struct StringType : public Type, public TypeIDProvider<StringType> {
  StringType() : Type(TypeCategory::STRING) {}
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
  IndexType() : Type(TypeCategory::INDEX) {}
  // note: index type takes 1 dim in mdspan/ituple declaration
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
      : Type(TypeCategory::ITUPLE) {} // this initialize an invalid ITupleType
                                      // The Type must be deduced for use

  bool HasSufficientInfo() const override { return IsValidRank(dim_count); }

  ITupleType(size_t n) : Type(TypeCategory::ITUPLE), dim_count(n) {}

  size_t Dims() const override { return dim_count; }
  bool IsDimValid() const { return IsValidRank(dim_count); }
  bool IsComplete() const override { return true; }

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

  MDSpanType(const Shape& v) : Type(TypeCategory::PARTIAL), value(v) {}

  void SetShape(const Shape& v) { value = v; }
  const Shape GetShape() { return value; }

  size_t Dims() const override { return value.Rank(); }

  // MDSpanType is an incomplete/partial type
  bool IsComplete() const override { return false; }

  // TODO: if the value is not evaluated, or can not be evaluated, not
  // sufficient information is obtained
  bool HasSufficientInfo() const override { return value.IsValid(); }

  bool operator==(const Type& ty) const override {
    if (!isa<MDSpanType>(&ty)) return false;
    auto& mty = ((const MDSpanType&)ty);
    // Must consider the condition whn shapes are not accurately decided
    // (only the dim-count is available)
    return (mty.value == value) || (!mty.value.IsValid() && !value.IsValid() &&
                                    mty.value.SameRankAs(value));
  }

  bool ApprxEqual(const Type& ty) const override {
    if (auto pty = dyn_cast<PlaceHolderType>(&ty))
      return pty->ApprxEqual(*this);

    if (auto sty = dyn_cast<MDSpanType>(&ty)) {
      if (!value.IsRanked() || !sty->value.IsRanked())
        return true; // it is ok when the shape is unknown
      else
        return ty.Dims() == Dims();
    }
    return false;
  }

  void Print(std::ostream& os) const override {
    os << "mdspan<";
    if (value.IsRanked()) os << Dims();
    os << ">";
    if (value.IsRanked()) os << " " << STR(value);
  }

  const std::string Name() const override { return "mdspan"; }

  __UDT_TYPE_INFO__(Type, MDSpanType)
};

struct SpannedType : public Type, public TypeIDProvider<SpannedType> {
  FundamentalType f_type;
  ptr<MDSpanType> s_type = nullptr;
  Storage m_type;

  SpannedType(FundamentalType ft, const ptr<MDSpanType>& s,
              Storage m = Storage::DEFAULT)
      : Type(TypeCategory::SPANNED), f_type(ft), s_type(s), m_type(m) {
    assert((s_type != nullptr) && "mdspan is not initialized.");
  }

  BaseType ElementType() const { return (BaseType)f_type; }
  size_t Dims() const override { return s_type->Dims(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override {
    return s_type->HasSufficientInfo();
  }

  bool operator==(const Type& ty) const override {
    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    return t.f_type == f_type && *t.s_type == *s_type &&
           Compatible(t.m_type, m_type);
  }

  // Ignore the memory
  bool LogicalEqual(const Type& ty) const override {
    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    return t.f_type == f_type && *t.s_type == *s_type;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (auto pty = dyn_cast<PlaceHolderType>(&ty))
      return pty->ApprxEqual(*this);

    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    // should the equivalence of storage be checked?
    return t.f_type == f_type && t.s_type->ApprxEqual(*s_type);
  }

  Shape GetShape() const { return s_type->GetShape(); }
  ptr<MDSpanType> GetMDSpanType() { return s_type; }

  bool RuntimeShaped() const {
    assert(s_type && "missing the spanned type.");
    return GetShape().IsDynamic();
  }

  // use these interface when the shape is NOT runtime-shaped
  size_t ElementCount() const { return GetShape().ElementCount(); }
  ValueItem ElementCountValue() const { return GetShape().ElementCountValue(); }
  size_t ByteSize() const { return SizeOf(f_type) * GetShape().ElementCount(); }

  const std::string ShapeSizeExpression(bool ULL_suffix = false) const {
    if (RuntimeShaped())
      return "(" + GetShape().GetElementCountExpression(ULL_suffix) + ")";
    else
      return std::to_string(ElementCount()) + (ULL_suffix ? "ULL" : "");
  }

  const std::string ByteSizeExpression(bool ULL_suffix = false) const {
    if (RuntimeShaped())
      return "(" + GetShape().GetElementCountExpression(ULL_suffix) + ") * " +
             std::to_string(SizeOf(f_type));
    else
      return std::to_string(ByteSize()) + (ULL_suffix ? "ULL" : "");
  }

  void SetStorage(Storage s) { m_type = s; }
  Storage GetStorage() { return m_type; }

  void Print(std::ostream& os) const override {
    if (m_type != Storage::NONE && m_type != Storage::DEFAULT)
      os << STR(m_type) << " ";
    os << STR(f_type) << " ";
    s_type->Print(os);
  }

  const std::string Name() const override { return "spanned"; }

  __UDT_TYPE_INFO__(Type, SpannedType)
};

struct BoundedType : public Type, public TypeIDProvider<BoundedType> {
  std::string note = ""; // some annotation to make
  BoundedType(TypeCategory tc, const std::string& n) : Type(tc), note(n) {}
  virtual bool HasValidBound() const = 0;
  virtual std::string GetNote() const { return note; };
  virtual void AppendNote(const std::string& n) { note += n; };
  virtual const ValueItem& GetUpperBound() const = 0;

  bool LogicalEqual(const Type& ty) const override {
    if (auto fty = dyn_cast<BoundedType>(&ty)) return Dims() == fty->Dims();
    return false;
  }

  void Print(std::ostream& os) const override {
    if (!note.empty()) os << "(" << note << ")";
  }

  __UDT_TYPE_INFO__(Type, BoundedType)
};

struct BoundedIntegerType final : public BoundedType,
                                  public TypeIDProvider<BoundedIntegerType> {
  ValueItem lbound = GetInvalidValueItem();
  ValueItem ubound = GetInvalidValueItem();
  int stride = GetInvalidStride();

  BoundedIntegerType() : BoundedType(TypeCategory::BOUNDED_INT, "") {}
  BoundedIntegerType(int lb, int ub, int s = 1, const std::string& note = "")
      : BoundedIntegerType(sbe::nu(lb), sbe::nu(ub), s, note) {}
  BoundedIntegerType(const ValueItem& lexpr, const ValueItem& uexpr, int s = 1,
                     const std::string& note = "")
      : BoundedType(TypeCategory::BOUNDED_INT, note), lbound(lexpr),
        ubound(uexpr), stride(s) {}

  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return HasValidBound(); }
  bool HasValidBound() const override {
    return IsValidValueItem(lbound) && IsValidValueItem(ubound) &&
           IsValidStride(stride);
  }
  ValueItem GetLowerBound() const { return lbound; }
  const ValueItem& GetUpperBound() const override { return ubound; }
  ValueItem GetStride() const { return ubound; }

  bool operator==(const Type& ty) const override {
    if (!isa<BoundedIntegerType>(&ty)) return false;
    auto bty = (BoundedIntegerType&)ty;
    return (bty.lbound == lbound) && (bty.ubound == ubound) &&
           (bty.stride == stride);
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
    if (plain == false) os << ":" << stride;
    BoundedType::Print(os);
  }

  const std::string Name() const override { return "bounded-integer"; }

  __UDT_TYPE_INFO__(BoundedType, BoundedIntegerType)
};

struct BoundedITupleType final : public BoundedType,
                                 public TypeIDProvider<BoundedITupleType> {
  MultiBounds lbounds;
  MultiBounds ubounds;
  IntegerList strides;

  BoundedITupleType(const MultiBounds& l, const MultiBounds& u,
                    const IntegerList s, const std::string& n = "")
      : BoundedType(TypeCategory::BOUNDED_ITUPLE, n), lbounds(l), ubounds(u),
        strides(s) {
    if (lbounds.IsValid())
      assert((lbounds.DimCount() == ubounds.DimCount()) &&
             (lbounds.DimCount() == strides.size()) &&
             "expecting a valid bound.");
    else
      assert(!ubounds.IsValid() && strides.empty() &&
             "expecting an invalid bound.");
  }

  size_t Dims() const override { return ubounds.Rank(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return ubounds.IsValid(); }
  const MultiBounds GetLowerBounds() const { return lbounds; }
  const MultiBounds GetUpperBounds() const { return ubounds; }
  const Shape GetSizes() const { return ubounds - lbounds; }
  IntegerList GetStrides() const { return strides; }
  const ValueItem& GetUpperBound() const override { return ubounds.ValueAt(0); }
  const ValueItem& GetUpperBound(size_t idx) const {
    return ubounds.ValueAt(idx);
  }
  const ValueItem& GetLowerBound(size_t idx) const {
    return lbounds.ValueAt(idx);
  }
  int GetStride(size_t idx) const { return strides[idx]; }
  bool IsPlain(size_t idx) const {
    return (*lbounds.ValueAt(idx) == *sbe::nu(0)) && (strides[idx] == 1);
  }
  bool HasValidBound() const override {
    return lbounds.IsValid() && ubounds.IsValid() && !strides.empty() &&
           (lbounds.DimCount() == ubounds.DimCount()) &&
           (lbounds.DimCount() == strides.size());
  }

  bool operator==(const Type& ty) const override {
    if (!isa<BoundedITupleType>(&ty)) return false;
    auto& t = (BoundedITupleType&)ty;
    return (t.lbounds == lbounds) && (t.ubounds == ubounds) &&
           (t.strides == strides);
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
      BoundedType::Print(os);
      return;
    }

    os << "{[" << STR(lbounds.ValueAt(0)) << "," << STR(ubounds.ValueAt(0))
       << ")";

    for (size_t i = 1; i < Dims(); ++i) {
      os << ", [" << STR(lbounds.ValueAt(i)) << "," << STR(ubounds.ValueAt(i))
         << "):" << strides[i];
    }
    os << "}";

    BoundedType::Print(os);
  }

  const std::string Name() const override { return "bounded-ituple"; }

  __UDT_TYPE_INFO__(BoundedType, BoundedITupleType)
};

struct AsyncType : public Type, public TypeIDProvider<AsyncType> {
  AsyncType(TypeCategory t) : Type(t) {}
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }

  // can not have instance
  __UDT_TYPE_INFO__(Type, AsyncType)
};

struct EventType : public AsyncType, public TypeIDProvider<EventType> {
  Storage scope;
  explicit EventType(Storage s) : AsyncType(TypeCategory::EVENT), scope(s) {}
  bool HasSufficientInfo() const override { return true; }
  void Print(std::ostream& os) const override { os << STR(scope) << " event"; }
  const std::string Name() const override { return STR(scope) + " event"; }

  bool operator==(const Type& ty) const override {
    if (auto ety = dyn_cast<EventType>(&ty)) return scope == ety->scope;
    return false;
  }
  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }
  virtual Storage GetStorage() const { return scope; }
  virtual void SetStorage(Storage s) { scope = s; }

  __UDT_TYPE_INFO__(AsyncType, EventType)
};

struct FutureType : public AsyncType, public TypeIDProvider<FutureType> {
  ptr<SpannedType> psty =
      nullptr; // the spanned data associated with the future
  bool async;

  explicit FutureType(const ptr<SpannedType>& s, bool a)
      : AsyncType(TypeCategory::FUTURE), psty(s), async(a) {}
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return psty->HasSufficientInfo(); }
  const std::string Name() const override { return "future"; }
  Shape GetShape() { return psty->GetShape(); }
  const ptr<SpannedType>& GetSpannedType() const { return psty; }
  size_t Dims() const override { return psty->Dims(); }
  bool IsAsync() const { return async; }

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
      : Type(TypeCategory::FUNCTION), out_ty(ot), in_tys(its) {}

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

// array is with a fixed-size
struct ArrayType;
template <>
inline ArrayType* dyn_cast<ArrayType, const Type>(const Type* n);

template <>
inline ptr<ArrayType> dyn_cast<ArrayType, Type>(const ptr<Type>& n);

// ArrayType is not consider as derived Type (avoid multi-inheritance)
// This hacks corresponding dyn_cast
struct ArrayType : public TypeIDProvider<ArrayType> {
  std::vector<size_t> dims;
  TypeCategory tc = TypeCategory::ARRAY;

  ArrayType(std::initializer_list<size_t> ds) {
    for (auto d : ds) {
      if (d == 0) choreo_unreachable("invalid dimension.");
      dims.push_back(d);
    }
  }

  explicit ArrayType(std::vector<size_t> ds) {
    for (auto d : ds) {
      if (d == 0) choreo_unreachable("invalid dimension.");
      dims.push_back(d);
    }
  }

  virtual const ptr<Type> SubScriptType(size_t) = 0;

  size_t ArrayRank() const { return dims.size(); }

  // array[n][m] - subscripting by 1  results in array[n]
  virtual const std::vector<size_t> SubScript(size_t dim_count) {
    if (dim_count > dims.size())
      choreo_unreachable("invalid subscription: not enough dimension.");
    return std::vector<size_t>(dims.begin(), dims.begin() + dim_count);
  }

  // array[n][m] - subscripting by 1  the remainder dimensions is [m]
  virtual const std::vector<size_t> RemainderDimensions(size_t dim_count) {
    if (dim_count > dims.size())
      choreo_unreachable("invalid subscription: not enough dimension.");
    return std::vector<size_t>(dims.begin() + dim_count, dims.end());
  }

  virtual size_t Dimension(size_t idx) const { return dims.at(idx); }
  virtual const std::vector<size_t>& Dimensions() { return dims; }
  virtual size_t ElemCount() const {
    if (dims.size() == 0) {
      choreo_unreachable("invalid array.");
      return 0;
    }

    size_t ec = 1;
    for (auto d : dims) ec *= d;

    return ec;
  }

  virtual bool operator==(const Type& ty) const {
    if (auto t = dyn_cast<ArrayType>(&ty)) {
      if (ArrayRank() != t->ArrayRank()) return false;
      for (size_t idx = 0; idx < ArrayRank(); ++idx)
        if (Dimension(idx) != t->Dimension(idx)) return false;
      return true;
    }
    return false;
  }

  virtual void Print(std::ostream& os) const {
    os << "[";
    for (auto d : dims) os << "[" << d << "]";
    os << "]";
  }

  virtual void PrintAsCArray(std::ostream& os) const {
    for (auto d : dims) os << "[" << d << "]";
  }

  // for runtime type disambiguation
  __UDT_TYPE_INFO_BASE1__(arraytype)
};

struct EventArrayType final : public ArrayType,
                              public EventType,
                              public TypeIDProvider<EventArrayType> {
  EventArrayType(Storage s, std::initializer_list<size_t> ec)
      : ArrayType(ec), EventType(s) {}
  explicit EventArrayType(Storage s, std::vector<size_t> ec)
      : ArrayType(ec), EventType(s) {}

  const ptr<Type> SubScriptType(size_t subscription_count) override {
    auto arr = SubScript(subscription_count);
    if (arr.size() == 0)
      return std::make_shared<EventType>(EventType::GetStorage());
    else
      return std::make_shared<EventArrayType>(EventType::GetStorage(), arr);
  }

  size_t Dims() const override { return ArrayType::ArrayRank(); }

  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return true; }

  bool operator==(const Type& ty) const override {
    if (auto t = dyn_cast<EventArrayType>(&ty))
      return t->ElemCount() == ElemCount();
    return false;
  }

  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }

  const std::string Name() const override {
    return EventType::Name() + " array";
  }

  void Print(std::ostream& os) const override {
    EventType::Print(os);
    ArrayType::Print(os);
  }

  __UDT_2TYPES_INFO__(ArrayType, EventType, EventArrayType)
};

struct SpannedArrayType final : public ArrayType,
                                public SpannedType,
                                public TypeIDProvider<SpannedArrayType> {
  SpannedArrayType(FundamentalType ft, const ptr<MDSpanType>& s, Storage m,
                   std::vector<size_t> ads)
      : ArrayType(ads), SpannedType(ft, s, m) {}
  bool IsComplete() const override { return SpannedType::IsComplete(); }
  bool HasSufficientInfo() const override {
    return SpannedType::HasSufficientInfo();
  }

  const ptr<Type> SubScriptType(size_t subscription_count) override {
    auto arr = SubScript(subscription_count);
    if (arr.size() == 0)
      return std::make_shared<SpannedType>(SpannedType::f_type,
                                           SpannedType::GetMDSpanType(),
                                           SpannedType::GetStorage());
    else
      return std::make_shared<SpannedArrayType>(SpannedType::f_type,
                                                SpannedType::GetMDSpanType(),
                                                SpannedType::GetStorage(), arr);
  }

  size_t Dims() const override { return ArrayType::ArrayRank(); }

  bool operator==(const Type& ty) const override {
    if (auto t = dyn_cast<SpannedArrayType>(&ty))
      return t->ElemCount() == ElemCount();
    return false;
  }

  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }

  const std::string Name() const override {
    return SpannedType::Name() + " array";
  }

  void Print(std::ostream& os) const override {
    SpannedType::Print(os);
    ArrayType::Print(os);
  }

  __UDT_2TYPES_INFO__(ArrayType, SpannedType, SpannedArrayType)
};

template <>
inline ArrayType* dyn_cast<ArrayType, const Type>(const Type* n) {
  if (auto ty = dyn_cast<EventArrayType>(n)) return static_cast<ArrayType*>(ty);
  if (auto ty = dyn_cast<SpannedArrayType>(n))
    return static_cast<ArrayType*>(ty);
  return nullptr;
}

template <>
inline ptr<ArrayType> dyn_cast<ArrayType, Type>(const ptr<Type>& n) {
  if (auto ty = dyn_cast<EventArrayType>(n))
    return std::static_pointer_cast<ArrayType>(ty);
  if (auto ty = dyn_cast<SpannedArrayType>(n))
    return std::static_pointer_cast<ArrayType>(ty);
  return nullptr;
}

inline size_t SizeOf(const Type& ty) {
  if (isa<VoidType>(&ty)) return 0;
  if (isa<IntegerType>(&ty))
    return 4;
  else if (isa<BooleanType>(&ty))
    return 4;
  else if (isa<BoundedIntegerType>(&ty))
    return 4;
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return t->ByteSize();
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return 0;
}

// This util might be useful to keep symbolic form till runtime
inline std::string SizeExprOf(const Type& ty, bool ULL_suffix = false) {
  if (isa<VoidType>(&ty)) return {};
  if (isa<IntegerType>(&ty))
    return "4";
  else if (isa<BooleanType>(&ty))
    return "4";
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
    return t->ShapeSizeExpression();
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return {};
}

inline BaseType GetBaseType(const Type& ty) {
  if (isa<VoidType>(&ty)) return BaseType::VOID;
  if (isa<IntegerType>(&ty))
    return BaseType::INT;
  else if (isa<FloatType>(&ty))
    return BaseType::FLOAT;
  else if (isa<DoubleType>(&ty))
    return BaseType::DOUBLE;
  else if (isa<BooleanType>(&ty))
    return BaseType::BOOL;
  else if (isa<BoundedIntegerType>(&ty))
    return BaseType::INT;
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return (BaseType)t->f_type;
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
}

inline bool IsActualBoundedIntegerType(const ptr<Type>& ty) {
  if (auto bty = dyn_cast<BoundedType>(ty)) return bty->Dims() == 1;
  return false;
}

inline bool CanYieldAnInteger(const ptr<Type>& ty) {
  return isa<ScalarType>(ty) || IsActualBoundedIntegerType(ty) ||
         (isa<ITupleType>(ty) && ty->Dims() == 1);
}

inline bool ConvertibleToInt(const ptr<Type>& ty) {
  return isa<ScalarType>(ty) || (isa<ITupleType>(ty) && ty->Dims() == 1);
}

inline bool ConvertibleToInt(const Type& ty) {
  return isa<ScalarType>(&ty) || (isa<ITupleType>(&ty) && ty.Dims() == 1);
}

inline ValueItem GetSingleUpperBound(const ptr<Type>& ty) {
  if (!IsActualBoundedIntegerType(ty))
    choreo_unreachable("can not get the single upper bound for a " + PSTR(ty) +
                       " type.");
  return cast<BoundedType>(ty)->GetUpperBound();
}

// utility functions to generate types
// Note: should always use utility functions
inline Shape GenUninitShape() { return Shape(); }

inline ptr<VoidType> MakeVoidType() { return std::make_shared<VoidType>(); }

inline ptr<UnknownType> MakeUnknownType() {
  return std::make_shared<UnknownType>();
}

inline ptr<IntegerType> MakeIntegerType(bool m = false) {
  return std::make_shared<IntegerType>(m);
}

inline ptr<IntegerType> MakeIntegerType(const Shape& s, bool m = false) {
  if (s.IsValid()) {
    assert(s.Rank() == 1);
    return std::make_shared<IntegerType>(s.ValueAt(0), m);
  }
  return MakeIntegerType(m);
}

inline ptr<BooleanType> MakeBooleanType(bool m = false) {
  return std::make_shared<BooleanType>(m);
}

inline ptr<ScalarFloatType> MakeScalarFloatType(BaseType bt, bool m = false) {
  switch (bt) {
  case BaseType::HALF8: return std::make_shared<Half8Type>(m);
  case BaseType::HALF: return std::make_shared<HalfType>(m);
  case BaseType::BFP16: return std::make_shared<BFP16Type>(m);
  case BaseType::FLOAT: return std::make_shared<FloatType>(m);
  case BaseType::DOUBLE: return std::make_shared<DoubleType>(m);
  default: choreo_unreachable("unsupported base type.");
  }
  return nullptr;
}

inline ptr<ScalarType> MakeScalarType(BaseType bt, bool m = false) {
  switch (bt) {
  case BaseType::INT: return std::make_shared<IntegerType>(m);
  case BaseType::BOOL: return std::make_shared<BooleanType>(m);
  case BaseType::HALF8: return std::make_shared<Half8Type>(m);
  case BaseType::HALF: return std::make_shared<HalfType>(m);
  case BaseType::BFP16: return std::make_shared<BFP16Type>(m);
  case BaseType::FLOAT: return std::make_shared<FloatType>(m);
  case BaseType::DOUBLE: return std::make_shared<DoubleType>(m);
  default: choreo_unreachable("unsupported base type.");
  }
  return nullptr;
}

inline ptr<ScalarFloatType> MakeFloatType(bool m = false) {
  return MakeScalarFloatType(BaseType::FLOAT, m);
}

inline ptr<ScalarFloatType> MakeDoubleType(bool m = false) {
  return MakeScalarFloatType(BaseType::DOUBLE, m);
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
  return std::make_shared<MDSpanType>(GenUninitShape());
}

inline ptr<MDSpanType> MakeRankedMDSpanType(size_t n) {
  if (!IsValidRank(n)) return MakeUninitMDSpanType();
  return std::make_shared<MDSpanType>(Shape(n));
}

inline ptr<MDSpanType> MakeMDSpanType(const Shape& v) {
  return std::make_shared<MDSpanType>(v);
}

inline ptr<SpannedType> MakeSpannedType(FundamentalType ft, const Shape& v,
                                        const Storage& s = Storage::DEFAULT) {
  return std::make_shared<SpannedType>(ft, MakeMDSpanType(v), s);
}

inline ptr<SpannedType> MakeSpannedType(BaseType ft, const Shape& v,
                                        const Storage& s = Storage::DEFAULT) {
  return MakeSpannedType((FundamentalType)ft, v, s);
}

// all the values are fake. it is used only to indicate a spanned type without
// the shape detail
inline ptr<SpannedType> MakeDummySpannedType() {
  return MakeSpannedType(BaseType::UNKNOWN, GenUninitShape(), Storage::DEFAULT);
}

inline ptr<SpannedType> MakeRankedSpannedType(size_t n,
                                              BaseType bt = BaseType::UNKNOWN,
                                              Storage sto = Storage::DEFAULT) {
  // only care about the rank of span
  return MakeSpannedType(bt, Shape(n), sto);
}

inline ptr<SpannedType> MakeShapedSpannedType(const Shape& s,
                                              BaseType bt = BaseType::UNKNOWN) {
  // only care about the precise shape
  return MakeSpannedType(bt, s, Storage::DEFAULT);
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
                                                    const std::string& n = "") {
  MultiBounds lb(ub.DimCount(), sbe::nu(0));
  IntegerList s(ub.DimCount());
  std::fill(s.begin(), s.end(), 1);
  return std::make_shared<BoundedITupleType>(lb, ub, s, n);
}

inline ptr<BoundedITupleType> MakeBoundedITupleType(const MultiBounds& lb,
                                                    const MultiBounds& ub,
                                                    const std::string& n = "") {
  IntegerList s(ub.DimCount());
  std::fill(s.begin(), s.end(), 1);
  return std::make_shared<BoundedITupleType>(lb, ub, s, n);
}

inline ptr<BoundedITupleType> MakeBoundedITupleType(const MultiBounds& lb,
                                                    const MultiBounds& ub,
                                                    const IntegerList& il,
                                                    const std::string& n = "") {
  return std::make_shared<BoundedITupleType>(lb, ub, il, n);
}

inline ptr<BoundedITupleType> MakeUninitBoundedITupleType() {
  return std::make_shared<BoundedITupleType>(GenUninitShape(), GenUninitShape(),
                                             IntegerList(), "");
}

inline ptr<EventType> MakeEventType(Storage s) {
  return std::make_shared<EventType>(s);
}

inline ptr<EventArrayType>
MakeEventArrayType(Storage s, std::initializer_list<size_t> ec) {
  return std::make_shared<EventArrayType>(s, ec);
}

inline ptr<EventArrayType> MakeEventArrayType(Storage s,
                                              const std::vector<size_t>& ad) {
  return std::make_shared<EventArrayType>(s, ad);
}

inline ptr<SpannedType>
MakeSpannedArrayType(BaseType ft, const Shape& v,
                     const std::vector<size_t> ad = {},
                     const Storage& s = Storage::DEFAULT) {
  return std::make_shared<SpannedArrayType>((FundamentalType)ft,
                                            MakeMDSpanType(v), s, ad);
}

inline ptr<FutureType> MakeFutureType(const ptr<SpannedType>& v, bool async) {
  return std::make_shared<FutureType>(v, async);
}

inline ptr<FutureType> MakeRankedFutureType(size_t n, bool async) {
  return std::make_shared<FutureType>(MakeRankedSpannedType(n), async);
}

inline ptr<FutureType> MakeShapedFutureType(const Shape& v, bool async) {
  return std::make_shared<FutureType>(MakeShapedSpannedType(v), async);
}

inline ptr<FutureType> MakeDummyFutureType(bool async) {
  return std::make_shared<FutureType>(MakeDummySpannedType(), async);
}

inline ptr<PlaceHolderType> MakePlaceHolderMDSpanType() {
  return std::make_shared<PlaceHolderType>(TypeCategory::PARTIAL);
}

inline ptr<PlaceHolderType> MakePlaceHolderSpannedType() {
  return std::make_shared<PlaceHolderType>(TypeCategory::SPANNED);
}

inline ptr<PlaceHolderType> MakePlaceHolderFutureType() {
  return std::make_shared<PlaceHolderType>(TypeCategory::FUTURE);
}

inline ptr<FunctionType> MakeFunctionType(const ptr<Type> ot,
                                          const std::vector<ptr<Type>>& its) {
  return std::make_shared<FunctionType>(ot, its);
}

// map default to global
inline static ptr<Type> ShadowTypeStorage(const ptr<Type>& ty) {
  if (auto sty = dyn_cast<SpannedType>(ty)) {
    if (auto at = dyn_cast<ArrayType>(ty))
      return MakeSpannedArrayType(sty->ElementType(), sty->GetShape(), at->dims,
                                  ProjectStorage(sty->GetStorage()));
    else
      return MakeSpannedType(sty->ElementType(), sty->GetShape(),
                             ProjectStorage(sty->GetStorage()));
  } else if (auto fty = dyn_cast<FutureType>(ty)) {
    auto sty = fty->GetSpannedType();
    return MakeFutureType(MakeSpannedType(sty->ElementType(), sty->GetShape(),
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
  else
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

inline static bool GeneralFutureType(const Type& ty) {
  return ty.Category() == TypeCategory::FUTURE;
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
    return TC2BT(ty->Category());
  else if (auto sty = GetSpannedType(ty))
    return sty->ElementType();
  else if (isa<ITupleType>(ty) &&
           ty->Dims() == 1) // special handling of ituple with dim-1
    return BaseType::INT;
  return BaseType::UNKNOWN;
}

inline static ptr<Type> MakeElemScalarType(BaseType bt, bool m = false) {
  switch (bt) {
  case BaseType::F32: return std::make_shared<FloatType>(m);
  case BaseType::F16: return std::make_shared<HalfType>(m);
  case BaseType::BF16: return std::make_shared<BFP16Type>(m);
  case BaseType::F8: return std::make_shared<Half8Type>(m);
  case BaseType::S32:
  case BaseType::U32:
  case BaseType::U16:
  case BaseType::S16:
  case BaseType::U8:
  case BaseType::S8:
    return std::make_shared<IntegerType>(m); // convert all implicitly?
  default: choreo_unreachable("unsupported base type: " + STR(bt) + ".");
  }
}

inline static ptr<Type> MutateType(const Type& ty) {
  auto sty = dyn_cast<ScalarType>(&ty);
  if (!sty) choreo_unreachable("can not mutate a '" + STR(ty) + "' type.");

  return MakeScalarType(TC2BT(sty->Category()));
}

inline bool IsMutable(const Type& ty) {
  auto sty = dyn_cast<ScalarType>(&ty);
  if (!sty) return false;
  return sty->IsMutable();
}

inline bool MutableType(const Type& ty) { return isa<ScalarType>(&ty); }

} // end namespace Choreo

#endif // __CHOREO_TYPES_H__
