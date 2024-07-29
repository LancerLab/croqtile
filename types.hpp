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
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "aux.hpp"
#include "enums.hpp"

namespace Choreo {

template <typename T>
using ptr = std::shared_ptr<T>;

enum class TypeCategory {
  INT,
  BOOL,
  INDEX,
  ITUPLE,
  PARTIAL,
  SPANNED,
  BOUNDED_INT,
  BOUNDED_ITUPLE,
  VOID,
  FUTURE,
  FUNCTION,
  UNKNOWN,
};

// BaseType, FundamentalType, and ScalarType
// if the type needs deduction, mark it as 'UNKNOWN'
enum class BaseType {
  F32,
  F16,
  BF16,
  U32,
  S32,
  U16,
  S16,
  U8,
  S8,
  INT,
  BOOL,
  ITUPLE,
  VOID,
  UNKNOWN
};

enum class FundamentalType {
  F32 = (int)BaseType::F32,
  F16 = (int)BaseType::F16,
  BF16 = (int)BaseType::BF16,
  U32 = (int)BaseType::U32,
  U16 = (int)BaseType::U16,
  U8 = (int)BaseType::U8,
  S32 = (int)BaseType::S32,
  S16 = (int)BaseType::S16,
  S8 = (int)BaseType::S8,
};

enum class Storage { LOCAL, SHARED, GLOBAL, DEFAULT, NONE };

enum Attribute : uint16_t {
  ATT_NONE = 0,
  ATT_SHADOW_TO_GLOBAL = 1,  // shadow the host memory to global
};

inline BaseType TC2BT(TypeCategory tc) {
  switch (tc) {
    case TypeCategory::INT:
      return BaseType::INT;
    case TypeCategory::BOOL:
      return BaseType::BOOL;
    case TypeCategory::VOID:
      return BaseType::VOID;
    default:
      choreo_unreachable(
          "unsupported mapping from type category to base type.");
  }

  return BaseType::UNKNOWN;
}

inline static size_t GetByteSizeOf(FundamentalType ft) {
  switch (ft) {
    case FundamentalType::F32:
    case FundamentalType::U32:
    case FundamentalType::S32:
      return 4;
    case FundamentalType::F16:
    case FundamentalType::BF16:
    case FundamentalType::U16:
    case FundamentalType::S16:
      return 2;
    case FundamentalType::U8:
    case FundamentalType::S8:
      return 1;
    default:
      choreo_unreachable("fundamental type is not supported.");
  }
  return 0;
}

// utility functions to map types to strings, and the opposite.
inline static BaseType BaseTypeFromString(const std::string& input) {
  static const std::unordered_map<std::string, BaseType> typeMap = {
      {"f32", BaseType::F32},   {"f16", BaseType::F16},
      {"bf16", BaseType::BF16}, {"u32", BaseType::U32},
      {"s32", BaseType::S32},   {"u16", BaseType::U16},
      {"s16", BaseType::S16},   {"u8", BaseType::U8},
      {"s8", BaseType::S8},     {"int", BaseType::INT},
      {"bool", BaseType::BOOL}, {"ituple", BaseType::ITUPLE},
      {"void", BaseType::VOID}, {"unknown", BaseType::UNKNOWN},
  };

  auto it = typeMap.find(input);
  assert(it != typeMap.end() && "incorrect type string");

  return it->second;
}

namespace __internal__ {

inline static std::string GetStringFrom(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F32, "f32"},   {BaseType::F16, "f16"},
      {BaseType::BF16, "bf16"}, {BaseType::U32, "u32"},
      {BaseType::S32, "s32"},   {BaseType::U16, "u16"},
      {BaseType::S16, "s16"},   {BaseType::U8, "u8"},
      {BaseType::S8, "s8"},     {BaseType::INT, "int"},
      {BaseType::BOOL, "bool"}, {BaseType::ITUPLE, "ituple"},
      {BaseType::VOID, "void"}, {BaseType::UNKNOWN, "unknown"},
  };

  auto it = enumToString.find(dataType);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

inline static std::string GetStringFrom(Storage st) {
  static const std::unordered_map<Storage, std::string> enumToString = {
      {Storage::LOCAL, "local"},     {Storage::GLOBAL, "global"},
      {Storage::SHARED, "shared"},   {Storage::NONE, "none"},
      {Storage::DEFAULT, "default"},
  };

  auto it = enumToString.find(st);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

}  // end namespace __internal__

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

// smart typeid provider suggested by GPT
template <typename T>
struct TypeIDProvider {
  static int __unique_id;
};

template <typename T>
int TypeIDProvider<T>::__unique_id;

// User defined type that utilize isa/cast/dyn_cast must place the macro inside
// its class definition
#define __UDT_TYPE_INFO__                                       \
  const std::string TypeNameString() override {                 \
    std::string name = __PRETTY_FUNCTION__;                     \
    std::regex prefix_regex("^.*Choreo::");                     \
    name = std::regex_replace(name, prefix_regex, "");          \
    std::regex suffix_regex("::TypeNameString.*$");             \
    name = std::regex_replace(name, suffix_regex, "");          \
    return name;                                                \
  }                                                             \
  static uint64_t TypeID() { return (uint64_t)(&__unique_id); } \
  uint64_t RuntimeID() const override { return (uint64_t)(&__unique_id); }

// LLVM-style type utility functions
//
// Note:
// To be simple, we do not handle any relationship about inheritance but only
// the extact (most-derived) type
//

template <typename T, typename U>
bool isa(U* n) {
  if (!n) return false;
  return T::TypeID() == n->RuntimeID();
}
template <typename T, typename U>
bool isa(const ptr<U>& n) {
  if (!n) return false;
  return T::TypeID() == n->RuntimeID();
}

template <typename T, typename U>
T* dyn_cast(U* n) {
  if (isa<T>(n))
    return (T*)n;
  else
    return nullptr;
}
template <typename T, typename U>
T* dyn_cast(const ptr<U>& n) {
  if (isa<T>(n))
    return (T*)(n.get());
  else
    return nullptr;
}

template <typename T, typename U>
T* cast(U* n) {
  if (isa<T>(n))
    return (T*)n;
  else {
    std::cerr << "type cast failure for incompatibility.\n";
    abort();
  }
}
template <typename T, typename U>
T* cast(const ptr<U>& n) {
  if (isa<T>(n))
    return (T*)(n.get());
  else {
    std::cerr << "type cast failure for incompatibility.\n";
    abort();
  }
}

// Define ValueList as a group of values
template <class T>
inline constexpr bool always_false = false;

namespace __internal {
static constexpr size_t INVALID_UNSIGNED = std::numeric_limits<size_t>::max();
static constexpr int INVALID_SIGNED = std::numeric_limits<int>::max();
static constexpr int UNKNOWN_SIGNED =
    std::numeric_limits<int>::min();  // represent literal value '?' only
}  // namespace __internal

inline constexpr size_t GetInvalidUnsigned() {
  return __internal::INVALID_UNSIGNED;
}
inline constexpr int GetInvalidSigned() { return __internal::INVALID_SIGNED; }
inline constexpr int GetInvalidValueNumber() {
  return __internal::INVALID_SIGNED;
}
inline constexpr int GetUnKnownInteger() { return __internal::UNKNOWN_SIGNED; }
inline constexpr size_t GetInvalidRank() {
  return __internal::INVALID_UNSIGNED;
}

inline constexpr bool IsValidUnsigned(size_t v) {
  return v != __internal::INVALID_UNSIGNED;
}
inline constexpr bool IsValidSigned(int v) {
  return v != __internal::INVALID_SIGNED;
}
inline constexpr bool IsValidValueNumber(int v) {
  return v != __internal::INVALID_SIGNED;
}
inline constexpr bool IsUnKnownInteger(int v) {
  return v == __internal::UNKNOWN_SIGNED;
}
inline constexpr bool IsValidRank(size_t v) {
  return v != __internal::INVALID_UNSIGNED;
}

using ValueExpr = std::string;
using ValueItem = std::variant<int, ValueExpr>;
using ValueList = std::vector<ValueItem>;

inline ValueItem GetInvalidValueItem() { return ValueItem{GetInvalidSigned()}; }
inline bool IsValidValueItem(const ValueItem& vi) {
  if (std::holds_alternative<ValueExpr>(vi)) return true;
  return IsValidSigned(std::get<int>(vi));
}

// specialization for ValueItem
template <typename T>
T* dyn_cast(ValueItem* vi) {
  if (std::holds_alternative<T>(*vi)) return &std::get<T>(*vi);
  return nullptr;
}
template <typename T>
const T* dyn_cast(const ValueItem* vi) {
  if (std::holds_alternative<T>(*vi)) return &std::get<T>(*vi);
  return nullptr;
}

template <typename T>
bool isa(ValueItem* vi) {
  if (std::holds_alternative<T>(*vi)) return true;
  return false;
}
template <typename T>
bool isa(const ValueItem* vi) {
  if (std::holds_alternative<T>(*vi)) return true;
  return false;
}

template <typename T>
T* cast(ValueItem* vi) {
  if (T* res = dyn_cast<T>(vi)) return res;
  choreo_unreachable("value item does not contain the type.");
  return nullptr;
}

template <typename T>
const T* cast(const ValueItem* vi) {
  if (auto* res = dyn_cast<T>(vi)) return res;
  choreo_unreachable("value item does not contain the type.");
  return nullptr;
}

struct ValueExprHasher {
  std::size_t operator()(const ValueExpr& v) const noexcept {
    return std::hash<std::string>{}(v);
  }
};

struct ValueItemHasher {
  std::size_t operator()(const ValueItem& var) const noexcept {
    std::size_t content_hash = std::visit(
        [](auto&& arg) -> std::size_t {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int>) {
            return std::hash<int>{}(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            return ValueExprHasher{}(arg);
          } else {
            static_assert(always_false<void>, "Unhandled type in variant");
            return 0;  // This line should theoretically never be reached.
          }
        },
        var);

    // Combine the content hash with the variant's index to differentiate types
    std::size_t type_index_hash = std::hash<size_t>{}(var.index());
    return content_hash ^ (type_index_hash + 0x9e3779b9 + (content_hash << 6) +
                           (content_hash >> 2));
  }
};

inline std::string ValueItemAsString(const ValueItem& vi) {
  if (auto pint = dyn_cast<int>(&vi)) return std::to_string(*pint);
  return *cast<ValueExpr>(&vi);
}

// some operations
inline ValueItem operator+(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return ValueItemAsString(vi1) + "+" + ValueItemAsString(vi2);
  return *cast<int>(&vi1) + *cast<int>(&vi2);
}

inline ValueItem operator-(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return ValueItemAsString(vi1) + "-" + ValueItemAsString(vi2);
  return *cast<int>(&vi1) - *cast<int>(&vi2);
}

inline ValueItem operator*(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return ValueItemAsString(vi1) + "*" + ValueItemAsString(vi2);
  return *cast<int>(&vi1) * *cast<int>(&vi2);
}

inline ValueItem operator/(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return ValueItemAsString(vi1) + "/" + ValueItemAsString(vi2);
  return *cast<int>(&vi1) / *cast<int>(&vi2);
}

inline ValueItem operator%(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return ValueItemAsString(vi1) + "%" + ValueItemAsString(vi2);
  return *cast<int>(&vi1) % *cast<int>(&vi2);
}

struct ValueListHasher {
  std::size_t operator()(const ValueList& val) const noexcept {
    std::size_t hash = 0;
    ValueItemHasher variantHasher;
    for (const auto& v : val)
      hash ^= variantHasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    return hash;
  }
};

inline bool IsValueItemEqual(const ValueItem& a, const ValueItem& b) {
  if (a.index() != b.index()) return false;  // Different types

  return a == b;
}

// Function to compare two ValueList
inline bool isValueListEqual(const ValueList& a, const ValueList& b) {
  if (a.size() != b.size()) return false;  // Different sizes

  for (size_t i = 0; i < a.size(); ++i)
    if (!IsValueItemEqual(a[i], b[i])) return false;  // Found a mismatch

  return true;  // All elements match
}

// Stores all the value lists. It generates unique value number for each list.
// lists with identical values are never replicated in the repo.
struct ValueListRepo {
  std::vector<ValueList> values;
  std::unordered_map<size_t, size_t> hash_index;

  size_t Insert(const ValueList& st) {
    size_t hash_val = ValueListHasher{}(st);

    auto InsertValueList = [this, &st](size_t vn) {
      values.push_back(st);
      size_t index = values.size() - 1;

      hash_index.emplace(vn, index);
    };

    // the value number does not exits, simply add the mdspan
    if (!hash_index.count(hash_val)) {
      InsertValueList(hash_val);
      return hash_val;
    }

    // value number exists
    while (hash_index.count(hash_val)) {
      // ValueList exists, return the value number directly
      if (isValueListEqual(values[hash_index[hash_val]], st)) return hash_val;

      // conflicting keys, rehash
      hash_val++;
    }

    InsertValueList(hash_val);
    return hash_val;
  }

  bool Exists(size_t hash_val) { return hash_index.count(hash_val); }

  const ValueList& operator[](size_t hash_val) {
    assert(Exists(hash_val) && "Value Number does not exist.");
    assert((hash_index.size() > hash_index[hash_val]) &&
           "Internal error: unexpected value number.");

    return values[hash_index[hash_val]];
  }
};

inline void PrintValueList(const ValueList& vl, std::ostream& os,
                           const char* lb = "[", const char* rb = "]") {
  auto print_variant = [&os](const ValueItem& vle) {
    if (vle.index() == 0)
      os << std::get<0>(vle);
    else
      os << std::get<1>(vle);
  };
  if (lb) os << lb;
  if (!vl.empty()) {
    print_variant(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) {
      os << ", ";
      print_variant(vl[i]);
    }
  }
  if (rb) os << rb;
}

inline void PrintValueListAccumulator(const ValueList& vl, std::ostream& os,
                           const char* lb = "[", const char* rb = "]") {
  auto print_variant = [&os](const ValueItem& vle) {
    if (vle.index() == 0)
      os << std::get<0>(vle);
    else
      os << std::get<1>(vle);
  };
  if (lb) os << lb;
  if (!vl.empty()) {
    print_variant(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) {
      os << " * ";
      print_variant(vl[i]);
    }
  }
  if (rb) os << rb;
}

struct Shape {
  static ValueListRepo values;  // value numbers

  size_t val_no = GetInvalidValueNumber();
  size_t dim_count =
      GetInvalidRank();  // dim_count is used when no value appears

  void Invalidate() {
    val_no = GetInvalidUnsigned();
    dim_count = GetInvalidRank();
  }

  explicit Shape() {}  // this initialize an invalid Shape
                       // The type must be deduced for use

  Shape(size_t n) : dim_count(n) {}
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
  constexpr Shape& operator=(const Shape&) = default;

  size_t Dims() const { return dim_count; }
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

  ValueList Value() {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    return values[val_no];
  }

  const ValueItem& ValueAt(size_t index) const {
    if (!IsValid()) choreo_unreachable("the shape is not accessible.");
    if (index > dim_count)
      choreo_unreachable("index '" + std::to_string(index) +
                         "' exceeds rank: " + std::to_string(dim_count) + ".");
    return values[val_no].at(index);
  }

  int NthInteger(size_t index) const {
    const ValueItem& vi = Value().at(index);
    return *cast<int>(const_cast<ValueItem*>(&vi));
  }

  std::optional<std::vector<int>> GetIntList() const {
    std::vector<int> int_list;
    for (auto v : Value()) {
      if (auto pint = dyn_cast<int>(&v))
        int_list.push_back(*pint);
      else
        return std::nullopt;
    }
    return int_list;
  }

  bool IsDynamic() const {
    for (auto v : Value())
      if (!isa<int>(&v))
        return true;  // a symbolic value represents that the value is decided
                      // at runtime
    return false;
  }

  // retrieve the dimensions that are dynamic
  std::unordered_map<int, ValueExpr> GetDynamicDims() const {
    std::unordered_map<int, ValueExpr> res;
    size_t i = 0;
    for (auto & v : Value()) {
      if (!isa<int>(&v))
        res.emplace(i, *cast<ValueExpr>(&v));
      ++i;
    }
    return res;
  }

  std::string GetSizeExpression() const {
    if (!IsDynamic()) return std::to_string(Size());

    assert(!Value().empty() && "no values inside the shape.");
    std::string res;
    res = ValueItemAsString(Value()[0]);
    for (size_t i = 1; i < Value().size(); ++i)
      res += " * " + ValueItemAsString(Value()[i]);
    return res;
  }

  std::optional<std::vector<size_t>> GetUIntList() const {
    std::vector<size_t> int_list;
    for (auto v : Value()) {
      if (auto pint = dyn_cast<int>(&v)) {
        if (*pint < 0) return std::nullopt;
        int_list.push_back(*pint);
      } else
        return std::nullopt;
    }
    return int_list;
  }

  std::vector<int> IntList() const {
    auto ilist = GetIntList();
    if (!ilist) choreo_unreachable("fail to get an integer list.");
    return *ilist;
  }

  size_t Size() const {
    auto ilist = IntList();
    size_t sz = 1;
    for (int s : ilist) {
      if (s < 0) choreo_unreachable("negative value is found.");
      sz *= s;
    }
    return sz;
  }

  void Print(std::ostream& os) const {
    if (!IsValidValueNumber(val_no)) {
      os << "[]";
    } else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueList(Value(), os);
    }
  }

  // os << [4096, 4096]
  void PrintAsCUDAShape(std::ostream& os) const {
    if (!IsValidValueNumber(val_no)) {
      os << "[]";
    } else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueList(Value(), os);
    }
  }

  // os << [4096 * 4096]
  void PrintAsCUDASize(std::ostream& os) const {
    if (!IsValidValueNumber(val_no)) {
      os << "[]";
    } else {
      assert(values.Exists(val_no) && "invalid value number.");
      PrintValueListAccumulator(Value(), os);
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

  std::string EmitTo(Target target) const;
};

inline bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.IsValid() && rhs.IsValid() && (lhs.Dims() == rhs.Dims()) &&
         isValueListEqual(lhs.Value(), rhs.Value());
}

//
struct Type {
  TypeCategory tc = TypeCategory::UNKNOWN;
  Type(TypeCategory t) : tc(t) {}

  virtual TypeCategory Category() const { return tc; }
  virtual size_t Dims() const = 0;
  virtual bool IsComplete() const = 0;  // it is a partial or compelete type
  // is the information enough for semantic check and code generation
  virtual bool HasSufficientInfo() const { return true; }
  virtual bool operator==(const Type& t) const = 0;
  // in-precise comparison without considering the shape detail.
  // used in early semantics
  virtual bool ApprxEqual(const Type& t) const = 0;

  virtual void Print(std::ostream&) const = 0;
  virtual const std::string Name() const = 0;

  // codegen util for emitting target's code in string format
  virtual std::string EmitTo(Target) const {
    assert(false && "Emit stringify not impled for this type");
  }

  // for runtime type disambiguition
  virtual const std::string TypeNameString() = 0;
  virtual uint64_t RuntimeID() const { return 0xDEADBEEFULL; }
  static uint64_t TypeID() { return 0xDEADBEEFULL; }
  // forbidden to have instance
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
  std::ostringstream oss;
  if (auto iv = dyn_cast<int>(&vi))
    oss << *iv;
  else
    oss << *cast<std::string>(&vi);
  return oss.str();
}

// string as list
inline std::string LSTR(const Shape& s) {
  std::ostringstream oss;
  s.PrintAsList(oss);
  return oss.str();
}

// >> [4096, 4096]
inline std::string CUDASHAPE(const Shape& s) {
  std::ostringstream oss;
  s.PrintAsCUDAShape(oss);
  return oss.str();
}

// >> [4096, 4096]
inline std::string CUDASIZE(const Shape& s) {
  std::ostringstream oss;
  s.PrintAsCUDASize(oss);
  return oss.str();
}

// un-braced 'raw' string
inline std::string RSTR(const Shape& s) {
  std::ostringstream oss;
  s.PrintPlain(oss);
  return oss.str();
}

inline std::string RSTR(const ValueItem& vi) {
  std::ostringstream oss;
  if (auto iv = dyn_cast<int>(&vi))
    oss << *iv;
  else
    oss << *cast<std::string>(&vi);
  return oss.str();
}

struct VoidType final : public Type, public TypeIDProvider<VoidType> {
  explicit VoidType() : Type(TypeCategory::VOID) {}
  size_t Dims() const override { return GetInvalidRank(); }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "void"; }
  const std::string Name() const override { return "void_type"; }
  bool HasSufficientInfo() const { return true; }

  bool operator==(const Type& ty) const override { return isa<VoidType>(&ty); }
  bool ApprxEqual(const Type& ty) const override { return isa<VoidType>(&ty); }

  __UDT_TYPE_INFO__
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

  __UDT_TYPE_INFO__
};

struct ScalarType : public Type {
  ScalarType(TypeCategory t) : Type(t) {}
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  // can not have instance
};

struct IntegerType : public ScalarType, public TypeIDProvider<IntegerType> {
  IntegerType() : ScalarType(TypeCategory::INT) {}
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "int"; }
  const std::string Name() const override { return "integer"; }

  bool operator==(const Type& ty) const override {
    return isa<IntegerType>(&ty);
  }
  bool ApprxEqual(const Type& ty) const override { return operator==(ty); }

  __UDT_TYPE_INFO__
};

struct BooleanType final : public ScalarType,
                           public TypeIDProvider<BooleanType> {
  BooleanType() : ScalarType(TypeCategory::BOOL) {}
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "bool"; }
  const std::string Name() const override { return "boolean"; }

  bool operator==(const Type& ty) const override {
    return isa<BooleanType>(&ty);
  }
  bool ApprxEqual(const Type& ty) const override {
    return isa<BooleanType>(&ty);
  }
  __UDT_TYPE_INFO__
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

  __UDT_TYPE_INFO__
};

// ITuple is a dimensioned type
struct ITupleType : public Type, public TypeIDProvider<ITupleType> {
  size_t dim_count = GetInvalidRank();

  explicit ITupleType()
      : Type(TypeCategory::ITUPLE) {}  // this initialize an invalid ITupleType
                                       // The Type must be deduced for use

  bool HasSufficientInfo() const { return IsValidRank(dim_count); }

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

  __UDT_TYPE_INFO__
};

struct MDSpanType : public Type, public TypeIDProvider<MDSpanType> {
  Shape value;

  MDSpanType(const Shape& v) : Type(TypeCategory::PARTIAL), value(v) {}

  void SetShape(const Shape& v) { value = v; }
  const Shape GetShape() { return value; }

  size_t Dims() const override { return value.Dims(); }

  // MDSpanType is an incomplete/partial type
  bool IsComplete() const override { return false; }

  // TODO: if the value is not evaluated, or can not be evaluated, not
  // sufficient information is obtained
  bool HasSufficientInfo() const override { return value.IsValid(); }

  bool operator==(const Type& ty) const override {
    if (!isa<MDSpanType>(&ty)) return false;
    return ((const MDSpanType&)ty).value == value;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (auto sty = dyn_cast<MDSpanType>(&ty)) {
      if (!value.IsRanked() || !sty->value.IsRanked())
        return true;  // it is ok when the shape is unknown
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

  std::string EmitTo(Target target) const override {
    std::ostringstream _os;
    // _os << "mdspan";
    if (value.IsValid()) {
      // value.Print(_os);
      // TODO(albert): for readibility, consider change stringify to emit
      _os << value.EmitTo(target);
    }
    return _os.str();
  }

  const std::string Name() const override { return "mdspan"; }

  __UDT_TYPE_INFO__
};

struct SpannedType final : public Type, public TypeIDProvider<SpannedType> {
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
    return t.f_type == f_type && *t.s_type == *s_type;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    return t.f_type == f_type && t.s_type->ApprxEqual(*s_type);
  }

  Shape GetShape() const { return s_type->GetShape(); }
  ptr<MDSpanType> GetMDSpanType() { return s_type; }

  bool RuntimeShaped() const {
    assert(s_type && "missing the spanned type.");
    return GetShape().IsDynamic();
  }

  size_t ByteSize() const { return GetByteSizeOf(f_type) * GetShape().Size(); }
  std::string ByteSizeExpression() const {
    if (RuntimeShaped())
      return GetShape().GetSizeExpression() + " * " +
             std::to_string(GetByteSizeOf(f_type));
    else
      return std::to_string(ByteSize());
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

  __UDT_TYPE_INFO__
};

struct BoundedType : public Type {
  std::string note = "";  // some annotation to make
  BoundedType(TypeCategory tc, const std::string& n) : Type(tc), note(n) {}
  virtual bool HasValidBound() const = 0;
  virtual std::string GetNote() const { return note; };
  virtual void AppendNote(const std::string& n) { note += n; };
};

struct BoundedIntegerType final : public BoundedType,
                                  public TypeIDProvider<BoundedIntegerType> {
  ValueItem bound = GetInvalidValueItem();

  BoundedIntegerType() : BoundedType(TypeCategory::BOUNDED_INT, "") {}
  BoundedIntegerType(const ValueItem& expr, const std::string& note = "")
      : BoundedType(TypeCategory::BOUNDED_INT, note), bound(expr) {}

  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return HasValidBound(); }
  bool HasValidBound() const override { return IsValidValueItem(bound); }
  ValueItem GetBound() const { return bound; }

  bool operator==(const Type& ty) const override {
    if (isa<BoundedIntegerType>(&ty)) return false;
    return ((BoundedIntegerType&)ty).bound == bound;
  }

  bool ApprxEqual(const Type& ty) const override {
    // do not care about the bound expression
    return isa<BoundedIntegerType>(&ty);
  }

  void Print(std::ostream& os) const override {
    if (HasValidBound())
      os << "int->[unknown]";
    else
      std::visit([this, &os](const auto& v) { os << "int->[0, " << v << ")"; },
                 bound);
  }

  const std::string Name() const override { return "bounded-integer"; }

  __UDT_TYPE_INFO__
};

struct BoundedITupleType final : public BoundedType,
                                 public TypeIDProvider<BoundedITupleType> {
  Shape bounds;
  BoundedITupleType(const Shape& s, const std::string& n = "")
      : BoundedType(TypeCategory::BOUNDED_ITUPLE, n), bounds(s) {}

  size_t Dims() const override { return bounds.Dims(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return bounds.IsValid(); }
  Shape GetBounds() const { return bounds; }
  const ValueItem& GetBound(size_t idx) const { return bounds.ValueAt(idx); }
  bool HasValidBound() const override { return bounds.IsValid(); }

  bool operator==(const Type& ty) const override {
    if (!isa<BoundedITupleType>(&ty)) return false;
    auto& t = (BoundedITupleType&)ty;
    return t.bounds == bounds;
  }

  bool ApprxEqual(const Type& ty) const override {
    if (!isa<BoundedITupleType>(&ty)) return false;
    auto& t = (BoundedITupleType&)ty;
    return t.Dims() == Dims();
  }

  void Print(std::ostream& os) const override {
    if (!bounds.IsRanked()) {
      os << "{invalid}";
      return;
    }
    assert(Dims() > 0 && "dim of bounded ituple is incorrect.");
    os << "{int";
    for (size_t i = 1; i < Dims(); ++i) os << ",int";
    os << "}->";
    bounds.Print(os);
  }

  const std::string Name() const override { return "bounded-ituple"; }

  __UDT_TYPE_INFO__
};

struct FutureType : public Type, public TypeIDProvider<FutureType> {
  ptr<SpannedType> psty =
      nullptr;  // the spanned data associated with the future
  bool async;

  FutureType(const ptr<SpannedType>& s, bool a)
      : Type(TypeCategory::FUTURE), psty(s), async(a) {}
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

  bool ApprxEqual(const Type& ty) const override {
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

  __UDT_TYPE_INFO__
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

  __UDT_TYPE_INFO__
};

#if 0
inline bool operator==(const Type& t1, const Type& t2) {
  return t1.operator==(t2);
}
#endif

inline size_t GetByteSizeOf(const Type& ty) {
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

inline std::string GetByteSizeExprOf(const Type& ty) {
  if (isa<VoidType>(&ty)) return {};
  if (isa<IntegerType>(&ty))
    return "4";
  else if (isa<BooleanType>(&ty))
    return "4";
  else if (isa<BoundedIntegerType>(&ty))
    return "4";
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return t->ByteSizeExpression();
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return {};
}

inline std::string GetBaseTypeStringOf(const Type& ty) {
  if (isa<VoidType>(&ty)) return "void";
  if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (isa<BoundedIntegerType>(&ty))
    return "int";
  else if (auto t = dyn_cast<SpannedType>(&ty))
    return STR(t->f_type);
  choreo_unreachable(STR(ty) + " does not imply runtime storage.");
  return 0;
}

inline bool IsScalarType(const ptr<Type>& ty) {
  return isa<IntegerType>(ty) && isa<BooleanType>(ty);
}

inline bool IsBoundedType(const ptr<Type>& ty) {
  return isa<BoundedIntegerType>(ty) || isa<BoundedITupleType>(ty);
}

// utility functions to generate types
// Note: should always use utility functions
inline Shape GenUninitShape() { return Shape(); }

inline ptr<VoidType> MakeVoidType() { return std::make_shared<VoidType>(); }

inline ptr<UnknownType> MakeUnknownType() {
  return std::make_shared<UnknownType>();
}

inline ptr<IntegerType> MakeIntegerType() {
  return std::make_shared<IntegerType>();
}

inline ptr<BooleanType> MakeBooleanType() {
  return std::make_shared<BooleanType>();
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
  return MakeSpannedType(BaseType::S32, GenUninitShape(), Storage::DEFAULT);
}

inline ptr<SpannedType> MakeRankedSpannedType(size_t n,
                                              BaseType bt = BaseType::S32,
                                              Storage sto = Storage::DEFAULT) {
  // only care about the rank of span
  return MakeSpannedType(bt, Shape(n), sto);
}

inline ptr<SpannedType> MakeShapedSpannedType(const Shape& s,
                                              BaseType bt = BaseType::S32) {
  // only care about the precise shape
  return MakeSpannedType(bt, s, Storage::DEFAULT);
}

inline ptr<BoundedIntegerType> MakeBoundedIntegerType(const ValueItem& ub) {
  return std::make_shared<BoundedIntegerType>(ub);
}

inline ptr<BoundedIntegerType> MakeUnknownBoundedIntegerType() {
  return std::make_shared<BoundedIntegerType>();
}

inline ptr<BoundedITupleType> MakeBoundedITupleType(const Shape& v,
                                                    const std::string& n = "") {
  return std::make_shared<BoundedITupleType>(v, n);
}

inline ptr<BoundedITupleType> MakeUninitBoundedITupleType() {
  return std::make_shared<BoundedITupleType>(GenUninitShape(), "");
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

inline ptr<FunctionType> MakeFunctionType(const ptr<Type> ot,
                                          const std::vector<ptr<Type>>& its) {
  return std::make_shared<FunctionType>(ot, its);
}

}  // end namespace Choreo

#endif  // __CHOREO_TYPES_H__
