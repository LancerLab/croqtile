#ifndef __CHOREO_TYPES_H__
#define __CHOREO_TYPES_H__

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Choreo {

template <typename T>
using ptr = std::shared_ptr<T>;

enum class TypeCategory {
  INT,
  BOOL,
  ITUPLE,
  PARTIAL,
  SPANNED,
  BOUNDED_INT,
  BOUNDED_ITUPLE,
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

// utility functions to map types to strings, and the opposite.
inline static BaseType getTypeFromString(const std::string& input) {
  static const std::unordered_map<std::string, BaseType> typeMap = {
      {"f32", BaseType::F32},   {"f16", BaseType::F16},
      {"bf16", BaseType::BF16}, {"u32", BaseType::U32},
      {"s32", BaseType::S32},   {"u16", BaseType::U16},
      {"s16", BaseType::S16},   {"u8", BaseType::U8},
      {"s8", BaseType::S8},     {"int", BaseType::INT},
      {"bool", BaseType::BOOL}, {"ituple", BaseType::ITUPLE},
  };

  auto it = typeMap.find(input);
  assert(it != typeMap.end() && "incorrect type string");

  return it->second;
}

inline static std::string getStringFrom(BaseType dataType) {
  static const std::unordered_map<BaseType, std::string> enumToString = {
      {BaseType::F32, "f32"},   {BaseType::F16, "f16"},
      {BaseType::BF16, "bf16"}, {BaseType::U32, "u32"},
      {BaseType::S32, "s32"},   {BaseType::U16, "u16"},
      {BaseType::S16, "s16"},   {BaseType::U8, "u8"},
      {BaseType::S8, "s8"},     {BaseType::INT, "int"},
      {BaseType::BOOL, "bool"}, {BaseType::ITUPLE, "ituple"}};

  auto it = enumToString.find(dataType);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
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
#define __UDT_TYPE_INFO__                                                     \
  const std::string NodeTypeString() override { return __PRETTY_FUNCTION__; } \
  static uint64_t TypeID() { return (uint64_t)(&__unique_id); }               \
  virtual uint64_t RuntimeID() const override {                               \
    return (uint64_t)(&__unique_id);                                          \
  }

// LLVM-style type utility functions
//
// Note:
// To be simple, we do not handle any relationship about inheritance but only
// the extact (most-derived) type
//

template <typename T, typename U>
bool isa(U* n) {
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
T* cast(U* n) {
  if (isa<T>(n))
    return (T*)n;
  else {
    std::cerr << "Cast failure for the type inconsistence." << std::endl;
    abort();
  }
}

// Define ValueList as a group of values
template <class T>
inline constexpr bool always_false = false;

static constexpr size_t __INVALID_VALUE__ = std::numeric_limits<size_t>::max();
static constexpr int __UNKNOWN_INTVAL__ = std::numeric_limits<int>::min();

using ValueListExpr = std::vector<std::string>;
using ValueListElem = std::variant<int, ValueListExpr>;
using ValueList = std::vector<ValueListElem>;

struct ValueListExprHasher {
  std::size_t operator()(const ValueListExpr& v) const noexcept {
    std::size_t hash = 0;
    std::hash<std::string> hasher;
    for (const auto& str : v) {
      hash ^= hasher(str) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

// Custom hasher for std::variant<int, std::vector<std::string>>
struct ValueListElemHasher {
  std::size_t operator()(const ValueListElem& var) const noexcept {
    std::size_t content_hash = std::visit(
        [](auto&& arg) -> std::size_t {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, int>) {
            return std::hash<int>{}(arg);
          } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            return ValueListExprHasher{}(arg);
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

struct ValueListHasher {
  std::size_t operator()(const ValueList& val) const noexcept {
    std::size_t hash = 0;
    ValueListElemHasher variantHasher;
    for (const auto& v : val)
      hash ^= variantHasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    return hash;
  }
};

// Helper function to compare two std::variant<int, std::vector<std::string>>
inline bool isValueListElemEqual(const ValueListElem& a,
                                 const ValueListElem& b) {
  if (a.index() != b.index()) return false;  // Different types

  if (a.index() == 0)
    return std::get<0>(a) == std::get<0>(b);  // Compare ints directly
  else if (a.index() == 1) {
    auto& expr_a = std::get<1>(a);
    auto& expr_b = std::get<1>(b);
    return expr_a.size() == expr_b.size() &&
           std::equal(expr_a.begin(), expr_a.end(), expr_b.begin());
  } else {
    assert(false && "Unhandled type in variant");
    return false;
  }
}

// Function to compare two ValueList
inline bool isValueListEqual(const ValueList& a, const ValueList& b) {
  if (a.size() != b.size()) return false;  // Different sizes

  for (size_t i = 0; i < a.size(); ++i)
    if (!isValueListElemEqual(a[i], b[i])) return false;  // Found a mismatch

  return true;  // All elements match
}

// Stores all the value lists. It generates unique value number for each list.
struct ValueListRepo {
  std::vector<ValueList> values;
  std::unordered_map<size_t, size_t> valno_index;

  size_t Insert(const ValueList& st) {
    size_t value_number = ValueListHasher{}(st);

    auto InsertValueList = [this, &st](int vn) {
      values.push_back(st);
      size_t index = values.size() - 1;

      valno_index.emplace(vn, index);
    };

    // the value number does not exits, simply add the mdspan
    if (!valno_index.count(value_number)) {
      InsertValueList(value_number);
      return value_number;
    }

    // value number exists
    while (valno_index.count(value_number)) {
      // ValueList exists, return the value number directly
      if (isValueListEqual(values[valno_index[value_number]], st))
        return value_number;

      // conflicting keys, rehash
      value_number++;
    }

    InsertValueList(value_number);
    return value_number;
  }

  bool Exists(size_t value_number) { return valno_index.count(value_number); }

  const ValueList& operator[](size_t value_number) {
    assert(Exists(value_number) && "Value Number does not exist.");
    assert((valno_index.size() > valno_index[value_number]) &&
           "Internal error: unexpected value number.");

    return values[valno_index[value_number]];
  }
};

inline void PrintValueList(const ValueList& vl, std::ostream& os) {
  auto print_variant = [&os](const ValueListElem& vle) {
    if (vle.index() == 0)
      os << std::get<0>(vle);
    else {
      os << "{";
      auto& arr = std::get<1>(vle);
      if (!arr.empty()) {
        os << "\"" << arr[0] << "\"";
        for (unsigned i = 1; i < arr.size(); ++i) os << " \"" << arr[i] << "\"";
      }
      os << "}";
    }
  };
  os << "[";
  if (!vl.empty()) {
    print_variant(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) {
      os << ", ";
      print_variant(vl[i]);
    }
  }
  os << "]";
}

// MDSpan is sized and dependent type (dependent on the others)
struct MDSpanValue {
  static ValueListRepo values;  // value numbers

  size_t val_no = __INVALID_VALUE__;
  size_t dim_count =
      __INVALID_VALUE__;  // dim_count is used when no value appears

  void Invalidate() {
    val_no = __INVALID_VALUE__;
    dim_count = __INVALID_VALUE__;
  }

  explicit MDSpanValue() {}  // this initialize an invalid MDSpanValue
                             // The type must be deduced for use

  MDSpanValue(size_t n) : dim_count(n) {}
  MDSpanValue(const ValueList& v) { val_no = values.Insert(v); }
  // could be inconsist sized, but only be verified with sema checker
  MDSpanValue(size_t n, const ValueList& v) : dim_count(n) {
    val_no = values.Insert(v);
  }

  size_t Dims() const { return dim_count; }
  void Update() { dim_count = values[val_no].size(); }
  bool IsValid() const {
    if (val_no == __INVALID_VALUE__)
      return dim_count != __INVALID_VALUE__;
    else
      return dim_count == values[val_no].size();
  }

  const ValueList& Value() const { return values[val_no]; }

  void Print(std::ostream& os) const {
    if (val_no == __INVALID_VALUE__)
      os << "{}";
    else {
      assert(values.Exists(val_no) && "bad value number.");
      PrintValueList(Value(), os);
    }
  }
};

inline bool operator==(const MDSpanValue& lhs, const MDSpanValue& rhs) {
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
  virtual bool HasSufficientInfo() const {
    return true;
  }  // is the information enough for semantic check and code generation
  virtual bool operator==(const Type& t) const {
    return (t.Dims() == Dims()) && (t.Category() == Category());
  }
  virtual void Print(std::ostream&) const = 0;
  virtual const std::string Name() const = 0;

  // for runtime type disambiguition
  virtual const std::string NodeTypeString() = 0;
  virtual uint64_t RuntimeID() const { return 0xDEADBEEFULL; }
  static uint64_t TypeID() { return 0xDEADBEEFULL; }
  // can not have instance
};

// The type is unknown. It requires type inference
struct UnknownType final : public Type, public TypeIDProvider<UnknownType> {
  explicit UnknownType() : Type(TypeCategory::UNKNOWN) {}
  size_t Dims() const override { return __INVALID_VALUE__; }
  bool IsComplete() const override { return false; }
  void Print(std::ostream& os) const override { os << "unknown_type"; }
  const std::string Name() const override { return "unknown"; }
  bool HasSufficientInfo() const { return false; }

  __UDT_TYPE_INFO__
};

struct ScalarType : public Type {
  ScalarType(TypeCategory t) : Type(t) {}
  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  // can not have instance
};

struct IntegerType : public ScalarType, public TypeIDProvider<IntegerType> {
  IntegerType(TypeCategory t = TypeCategory::INT) : ScalarType(t) {}
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "int"; }
  const std::string Name() const override { return "integer"; }

  __UDT_TYPE_INFO__
};

struct BooleanType final : public ScalarType,
                           public TypeIDProvider<BooleanType> {
  BooleanType() : ScalarType(TypeCategory::BOOL) {}
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "bool"; }
  const std::string Name() const override { return "boolean"; }

  __UDT_TYPE_INFO__
};

// ITuple is a dimensioned type
struct ITupleType : public Type, public TypeIDProvider<ITupleType> {
  size_t dim_count = __INVALID_VALUE__;

  explicit ITupleType()
      : Type(TypeCategory::ITUPLE) {}  // this initialize an invalid ITupleType
                                       // The Type must be deduced for use

  bool HasSufficientInfo() const { return dim_count != __INVALID_VALUE__; }

  ITupleType(int n, TypeCategory tc = TypeCategory::ITUPLE)
      : Type(tc), dim_count(n) {}

  size_t Dims() const override { return dim_count; }
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

  __UDT_TYPE_INFO__
};

struct MDSpanType : public Type, public TypeIDProvider<MDSpanType> {
  const MDSpanValue mdspan;

  MDSpanType(const MDSpanValue& v) : Type(TypeCategory::PARTIAL), mdspan(v) {}

  size_t Dims() const override {
    assert(mdspan.IsValid() && "Invalid mdspan defined.");
    return mdspan.Dims();
  }

  // MDSpanType is an incomplete/partial type
  bool IsComplete() const override { return false; }

  // TODO: if the value is not evaluated, or can not be evaluated, not
  // sufficient information is obtained
  bool HasSufficientInfo() const override { return true; }

  bool operator==(const Type& ty) const override {
    if (!isa<MDSpanType>(&ty)) return false;
    return ((const MDSpanType&)ty).mdspan == mdspan;
  }

  void Print(std::ostream& os) const override {
    os << "mdspan<";
    if (mdspan.IsValid()) os << Dims();
    os << "> ";
    mdspan.Print(os);
  }

  const std::string Name() const override { return "mdspan"; }

  __UDT_TYPE_INFO__
};

struct SpannedType final : public Type, public TypeIDProvider<SpannedType> {
  FundamentalType f_type;
  MDSpanType s_type;

  SpannedType(FundamentalType ft, const MDSpanType& s)
      : Type(TypeCategory::SPANNED), f_type(ft), s_type(s) {}

  size_t Dims() const override { return s_type.Dims(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const override { return s_type.HasSufficientInfo(); }
  bool operator==(const Type& ty) const override {
    if (!isa<SpannedType>(&ty)) return false;
    auto& t = (SpannedType&)ty;
    return t.f_type == f_type && t.s_type == s_type;
  }

  void Print(std::ostream& os) const override {
    os << getStringFrom((BaseType)f_type) << " ";
    s_type.Print(os);
  }

  const std::string Name() const override { return "spanned"; }

  __UDT_TYPE_INFO__
};

struct BoundedIntegerType final : public Type,
                                  public TypeIDProvider<BoundedIntegerType> {
  int bound = __UNKNOWN_INTVAL__;
  BoundedIntegerType(int b) : Type(TypeCategory::BOUNDED_INT), bound(b) {}

  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return bound != __UNKNOWN_INTVAL__; }

  bool operator==(const Type& ty) const override {
    if (isa<BoundedIntegerType>(&ty)) return false;
    return ((BoundedIntegerType&)ty).bound == bound;
  }

  void Print(std::ostream& os) const override {
    if (bound == __UNKNOWN_INTVAL__)
      os << "int->[unknown]";
    else
      os << "int->[0, " << bound << ")";
  }

  const std::string Name() const override { return "bounded-integer"; }

  __UDT_TYPE_INFO__
};

struct BoundedITupleType final : public Type,
                                 public TypeIDProvider<BoundedITupleType> {
  ITupleType ituple;
  SpannedType bounds;
  BoundedITupleType(size_t n, const SpannedType& s)
      : Type(TypeCategory::BOUNDED_ITUPLE), ituple(n), bounds(s) {
    assert(n == s.Dims() &&
           "ituple has a different bounded range with the associated mdspan.");
  }

  size_t Dims() const override { return ituple.Dims(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const {
    return ituple.HasSufficientInfo() && bounds.HasSufficientInfo();
  }

  bool operator==(const Type& ty) const override {
    if (!isa<BoundedITupleType>(&ty)) return false;
    auto& t = (BoundedITupleType&)ty;
    return (t.bounds == bounds) && t.ituple == ituple;
  }

  void Print(std::ostream& os) const override {
    ituple.Print(os);
    os << "->";
    bounds.Print(os);
  }

  const std::string Name() const override { return "bounded-ituple"; }

  __UDT_TYPE_INFO__
};

struct FutureType : public ScalarType, public TypeIDProvider<FutureType> {
  FutureType(TypeCategory t = TypeCategory::INT) : ScalarType(t) {}
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "fut"; }
  const std::string Name() const override { return "future"; }

  __UDT_TYPE_INFO__
};

// Utility functions to generate types
inline MDSpanValue GenUninitMDSpanValue() { return MDSpanValue(); }

inline ptr<UnknownType> MakeUnknownType() {
  return std::make_shared<UnknownType>();
}

inline ptr<IntegerType> MakeIntegerType() {
  return std::make_shared<IntegerType>();
}

inline ptr<BooleanType> MakeBooleanType() {
  return std::make_shared<BooleanType>();
}

inline ptr<ITupleType> MakeUninitITupleType() {
  return std::make_shared<ITupleType>();
}

inline ptr<MDSpanType> MakeUninitMDSpanType() {
  return std::make_shared<MDSpanType>(GenUninitMDSpanValue());
}

inline ptr<MDSpanType> MakeDimSizedMDSpanType(size_t n) {
  return std::make_shared<MDSpanType>(MDSpanValue(n));
}

inline ptr<SpannedType> MakeSpannedType(FundamentalType ft,
                                        const MDSpanValue& v) {
  return std::make_shared<SpannedType>(ft, v);
}

inline ptr<SpannedType> MakeSpannedType(BaseType ft, const MDSpanValue& v) {
  return MakeSpannedType((FundamentalType)ft, v);
}

inline ptr<FutureType> MakeFutureType() {
  return std::make_shared<FutureType>();
}

}  // end namespace Choreo

#endif  // __CHOREO_TYPES_H__
