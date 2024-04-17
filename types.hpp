#ifndef __CHOREO_TYPES_H__
#define __CHOREO_TYPES_H__

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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
  ITUPLE,
  PARTIAL,
  SPANNED,
  BOUNDED_INT,
  BOUNDED_ITUPLE,
  VOID,
  UNKNOWN,
  FUTURE,
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

// utility functions to map types to strings, and the opposite.
inline static BaseType getTypeFromString(const std::string& input) {
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

inline static std::string getStringFrom(BaseType dataType) {
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

inline static std::string getStringFrom(Storage st) {
  static const std::unordered_map<Storage, std::string> enumToString = {
      {Storage::LOCAL, "local"},     {Storage::GLOBAL, "global"},
      {Storage::SHARED, "shared"},   {Storage::NONE, "none"},
      {Storage::DEFAULT, "default"},
  };

  auto it = enumToString.find(st);
  assert(it != enumToString.end() && "unsupported type.");

  return it->second;
}

inline std::optional<std::string> PrefixedWith(const std::string& prefix,
                                               const std::string& str) {
  if (str.find(prefix) == 0)  // Check if 'prefix' is at the beginning
    return str.substr(prefix.length());  // Return the substring after 'prefix'
  else
    return std::nullopt;  // Return an empty string if 'prefix' is not at the
                          // beginning
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
    std::cerr << "Cast failure for the type inconsistence.\n";
    abort();
  }
}
template <typename T, typename U>
T* cast(const ptr<U>& n) {
  if (isa<T>(n))
    return (T*)(n.get());
  else {
    std::cerr << "Cast failure for the type inconsistence.\n";
    abort();
  }
}

// Define ValueList as a group of values
template <class T>
inline constexpr bool always_false = false;

static constexpr size_t __INVALID_VALUE__ = std::numeric_limits<size_t>::max();
static constexpr int __UNKNOWN_INTVAL__ = std::numeric_limits<int>::min();
static constexpr int __INVALID_INTVAL__ = std::numeric_limits<int>::max();

using ValueExpr = std::string;
using ValueItem = std::variant<int, ValueExpr>;
using ValueList = std::vector<ValueItem>;

// specialization for ValueItem
template <typename T>
T* dyn_cast(ValueItem* vi) {
  if (std::holds_alternative<T>(*vi)) return &std::get<T>(*vi);
  return nullptr;
}

template <typename T>
T* cast(ValueItem* vi) {
  if (T* res = dyn_cast<T>(vi)) return res;
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

struct ValueListHasher {
  std::size_t operator()(const ValueList& val) const noexcept {
    std::size_t hash = 0;
    ValueItemHasher variantHasher;
    for (const auto& v : val)
      hash ^= variantHasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    return hash;
  }
};

inline bool isValueItemEqual(const ValueItem& a, const ValueItem& b) {
  if (a.index() != b.index()) return false;  // Different types

  return a == b;
}

// Function to compare two ValueList
inline bool isValueListEqual(const ValueList& a, const ValueList& b) {
  if (a.size() != b.size()) return false;  // Different sizes

  for (size_t i = 0; i < a.size(); ++i)
    if (!isValueItemEqual(a[i], b[i])) return false;  // Found a mismatch

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

inline void PrintValueList(const ValueList& vl, std::ostream& os) {
  auto print_variant = [&os](const ValueItem& vle) {
    if (vle.index() == 0)
      os << std::get<0>(vle);
    else
      os << std::get<1>(vle);
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

// TODO(albert): pack this util function together with other emit purpose
// classes/methods
// TODO(albert): add emit target
inline void EmitValueListForFactor(const ValueList& vl, std::ostream& os) {
  auto print_variant = [&os](const ValueItem& vle) {
    if (vle.index() == 0)
      os << std::get<0>(vle);
    else
      os << std::get<1>(vle);
  };
  os << "{";
  if (!vl.empty()) {
    print_variant(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) {
      os << ", ";
      print_variant(vl[i]);
    }
  }
  os << "}";
}

// MDSpan is sized and dependent type (dependent on the others)
struct Shape {
  static ValueListRepo values;  // value numbers

  size_t val_no = __INVALID_VALUE__;
  size_t dim_count =
      __INVALID_VALUE__;  // dim_count is used when no value appears

  void Invalidate() {
    val_no = __INVALID_VALUE__;
    dim_count = __INVALID_VALUE__;
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
  bool IsValid() const {
    if (val_no == __INVALID_VALUE__)
      return dim_count != __INVALID_VALUE__;
    else
      return dim_count == values[val_no].size();
  }

  const ValueList& Value() const { return values[val_no]; }

  void Print(std::ostream& os) const {
    if (val_no == __INVALID_VALUE__) os << "[]";
    // PrintValueList(Value(), os);
    else {
      assert(values.Exists(val_no) && "bad value number.");
      PrintValueList(Value(), os);
    }
  }

  // util function for emit
  std::string EmitTo(Target target) const {
    (void)target;
    std::ostringstream _os;
    if (val_no == __INVALID_VALUE__) _os << "{}";
    // PrintValueList(Value(), _os);
    else {
      assert(values.Exists(val_no) && "bad value number.");
      EmitValueListForFactor(Value(), _os);
    }
    return _os.str();
  }
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
  virtual bool HasSufficientInfo() const {
    return true;
  }  // is the information enough for semantic check and code generation
  virtual bool operator==(const Type& t) const = 0;
  virtual void Print(std::ostream&) const = 0;
  virtual const std::string Name() const = 0;

  // codegen util for emitting target's code in string format
  virtual std::string EmitTo(Target) const {
    assert(false && "Emit stringify not impled for this type");
  }

  // for runtime type disambiguition
  virtual const std::string NodeTypeString() = 0;
  virtual uint64_t RuntimeID() const { return 0xDEADBEEFULL; }
  static uint64_t TypeID() { return 0xDEADBEEFULL; }
  // can not have instance
};

struct VoidType final : public Type, public TypeIDProvider<VoidType> {
  explicit VoidType() : Type(TypeCategory::VOID) {}
  size_t Dims() const override { return __INVALID_VALUE__; }
  bool IsComplete() const override { return true; }
  void Print(std::ostream& os) const override { os << "void_type"; }
  const std::string Name() const override { return "void"; }
  bool HasSufficientInfo() const { return true; }

  bool operator==(const Type& ty) const override { return isa<VoidType>(&ty); }

  __UDT_TYPE_INFO__
};

// The type is unknown. It requires type inference
struct UnknownType final : public Type, public TypeIDProvider<UnknownType> {
  explicit UnknownType() : Type(TypeCategory::UNKNOWN) {}
  size_t Dims() const override { return __INVALID_VALUE__; }
  bool IsComplete() const override { return false; }
  void Print(std::ostream& os) const override { os << "unknown_type"; }
  const std::string Name() const override { return "unknown"; }
  bool HasSufficientInfo() const { return false; }

  // Not comparable
  bool operator==(const Type&) const override { return false; }

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
  __UDT_TYPE_INFO__
};

// ITuple is a dimensioned type
struct ITupleType : public Type, public TypeIDProvider<ITupleType> {
  size_t dim_count = __INVALID_VALUE__;

  explicit ITupleType()
      : Type(TypeCategory::ITUPLE) {}  // this initialize an invalid ITupleType
                                       // The Type must be deduced for use

  bool HasSufficientInfo() const { return dim_count != __INVALID_VALUE__; }

  ITupleType(size_t n) : Type(TypeCategory::ITUPLE), dim_count(n) {}

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

  bool operator==(const Type& ty) const override {
    if (auto itty = dyn_cast<ITupleType>(&ty)) {
      if ((Dims() == itty->Dims()) && HasSufficientInfo()) return true;
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

  size_t Dims() const override {
    assert(value.IsValid() && "Invalid mdspan defined.");
    return value.Dims();
  }

  // MDSpanType is an incomplete/partial type
  bool IsComplete() const override { return false; }

  // TODO: if the value is not evaluated, or can not be evaluated, not
  // sufficient information is obtained
  bool HasSufficientInfo() const override { return true; }

  bool operator==(const Type& ty) const override {
    if (!isa<MDSpanType>(&ty)) return false;
    return ((const MDSpanType&)ty).value == value;
  }

  void Print(std::ostream& os) const override {
    os << "mdspan<";
    if (value.IsValid()) os << Dims();
    os << "> ";
    value.Print(os);
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

  Shape GetShape() { return s_type->GetShape(); }
  ptr<MDSpanType> GetMDSpanType() { return s_type; }

  void SetStorage(Storage s) { m_type = s; }
  Storage GetStorage() { return m_type; }

  void Print(std::ostream& os) const override {
    if (m_type != Storage::NONE && m_type != Storage::DEFAULT)
      os << getStringFrom(m_type) << " ";
    os << getStringFrom((BaseType)f_type) << " ";
    s_type->Print(os);
  }

  const std::string Name() const override { return "spanned"; }

  __UDT_TYPE_INFO__
};

struct BoundedIntegerType final : public Type,
                                  public TypeIDProvider<BoundedIntegerType> {
  ValueItem bound = __UNKNOWN_INTVAL__;
  BoundedIntegerType(int b) : Type(TypeCategory::BOUNDED_INT), bound(b) {}
  BoundedIntegerType(const std::string& expr)
      : Type(TypeCategory::BOUNDED_INT), bound(expr) {}

  size_t Dims() const override { return 1; }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const {
    return bound != ValueItem{__UNKNOWN_INTVAL__};
  }

  bool operator==(const Type& ty) const override {
    if (isa<BoundedIntegerType>(&ty)) return false;
    return ((BoundedIntegerType&)ty).bound == bound;
  }

  void Print(std::ostream& os) const override {
    if (bound == ValueItem{__UNKNOWN_INTVAL__})
      os << "int->[unknown]";
    else
      std::visit([this, &os](const auto& v) { os << "int->[0, " << v << ")"; },
                 bound);
  }

  const std::string Name() const override { return "bounded-integer"; }

  __UDT_TYPE_INFO__
};

struct BoundedITupleType final : public Type,
                                 public TypeIDProvider<BoundedITupleType> {
  Shape bounds;
  BoundedITupleType(const Shape& s)
      : Type(TypeCategory::BOUNDED_ITUPLE), bounds(s) {}

  size_t Dims() const override { return bounds.Dims(); }
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return bounds.IsValid(); }
  Shape GetBounds() const { return bounds; }

  bool operator==(const Type& ty) const override {
    if (!isa<BoundedITupleType>(&ty)) return false;
    auto& t = (BoundedITupleType&)ty;
    return t.bounds == bounds;
  }

  void Print(std::ostream& os) const override {
    if (Dims() > 0) {
      os << "{int";
      for (size_t i = 1; i < Dims(); ++i) os << ",int";
      os << "}->";
      bounds.Print(os);
    }
  }

  const std::string Name() const override { return "bounded-ituple"; }

  __UDT_TYPE_INFO__
};

struct FutureType : public ScalarType, public TypeIDProvider<FutureType> {
  Shape shape;  // the data shape associated with the future
  FutureType() : ScalarType(TypeCategory::FUTURE) {}
  FutureType(const Shape& mds) : ScalarType(TypeCategory::FUTURE), shape(mds) {}
  bool IsComplete() const override { return true; }
  bool HasSufficientInfo() const { return shape.IsValid(); }
  const std::string Name() const override { return "future"; }
  Shape GetShape() { return shape; }

  bool operator==(const Type& ty) const override {
    return isa<FutureType>(&ty);
  }

  void Print(std::ostream& os) const override {
    os << "async=>";
    shape.Print(os);
  }

  __UDT_TYPE_INFO__
};

#if 0
inline bool operator==(const Type& t1, const Type& t2) {
  return t1.operator==(t2);
}
#endif

inline bool operator!=(const Type& t1, const Type& t2) {
  return !t1.operator==(t2);
}

inline std::string STR(const Type& ty) {
  std::ostringstream oss;
  ty.Print(oss);
  return oss.str();
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

inline ptr<ITupleType> MakeITupleType(size_t n) {
  return std::make_shared<ITupleType>(n);
}

inline ptr<ITupleType> MakeUninitITupleType() {
  return std::make_shared<ITupleType>();
}

inline ptr<MDSpanType> MakeUninitMDSpanType() {
  return std::make_shared<MDSpanType>(GenUninitShape());
}

inline ptr<MDSpanType> MakeDimedMDSpanType(size_t n) {
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

inline ptr<BoundedITupleType> MakeBoundedITupleType(const Shape& v) {
  return std::make_shared<BoundedITupleType>(v);
}

inline ptr<FutureType> MakeFutureType(const Shape& v) {
  return std::make_shared<FutureType>(v);
}

inline ptr<FutureType> MakeFutureType() {
  return std::make_shared<FutureType>();
}

}  // end namespace Choreo

#endif  // __CHOREO_TYPES_H__
