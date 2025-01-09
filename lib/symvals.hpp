#ifndef __CHOREO_SYMBOL_VALUES_H__
#define __CHOREO_SYMBOL_VALUES_H__

#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Choreo {

// Define ValueList as a group of values
template <class T>
inline constexpr bool always_false = false;

namespace __internal {
static constexpr size_t INVALID_UNSIGNED = std::numeric_limits<size_t>::max();
static constexpr int INVALID_SIGNED = std::numeric_limits<int>::max();
static constexpr int UNKNOWN_SIGNED =
    std::numeric_limits<int>::min(); // represent literal value '?' only
static constexpr int UNKNOWN_FLOAT =
    std::numeric_limits<float>::min(); // represent literal value '?' only
static constexpr int UNKNOWN_DOUBLE =
    std::numeric_limits<double>::min(); // represent literal value '?' only
} // namespace __internal

inline constexpr size_t GetInvalidUnsigned() {
  return __internal::INVALID_UNSIGNED;
}
inline constexpr int GetInvalidSigned() { return __internal::INVALID_SIGNED; }
inline constexpr int GetUnKnownInteger() { return __internal::UNKNOWN_SIGNED; }
inline constexpr int GetUnKnownFloat() { return __internal::UNKNOWN_FLOAT; }
inline constexpr int GetUnKnownDouble() { return __internal::UNKNOWN_DOUBLE; }

inline constexpr bool IsValidUnsigned(size_t v) {
  return v != GetInvalidUnsigned();
}
inline constexpr bool IsValidSigned(int v) { return v != GetInvalidSigned(); }
inline constexpr bool IsUnKnownInteger(int v) {
  return v == GetUnKnownInteger();
}
inline constexpr bool IsUnKnownFloatPoint(float v) {
  return v == GetUnKnownFloat();
}
inline constexpr bool IsUnKnownFloatPoint(double v) {
  return v == GetUnKnownDouble();
}

inline constexpr size_t GetInvalidRank() { return GetInvalidUnsigned(); }
inline constexpr int GetInvalidValueNumber() { return GetInvalidSigned(); }
inline constexpr int GetInvalidBound() { return GetInvalidSigned(); }
inline constexpr int GetInvalidStride() { return GetInvalidSigned(); }

inline constexpr bool IsValidRank(size_t v) { return v != GetInvalidRank(); }
inline constexpr bool IsValidValueNumber(int v) {
  return v != GetInvalidValueNumber();
}
inline constexpr bool IsValidBound(int v) { return v != GetInvalidBound(); }
inline constexpr bool IsValidStride(int v) { return v != GetInvalidStride(); }

// ------------------------------------------------------------------------- //
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
            return 0; // This line should theoretically never be reached.
          }
        },
        var);

    // Combine the content hash with the variant's index to differentiate types
    std::size_t type_index_hash = std::hash<size_t>{}(var.index());
    return content_hash ^ (type_index_hash + 0x9e3779b9 + (content_hash << 6) +
                           (content_hash >> 2));
  }
};

inline std::string ValueItemAsString(const ValueItem& vi,
                                     bool ULL_suffix = false) {
  if (auto pint = dyn_cast<int>(&vi))
    return std::to_string(*pint) + (ULL_suffix ? "ULL" : "");
  return *cast<ValueExpr>(&vi);
}

template <typename T>
inline T GetValueAt(ValueList vlist, int idx) {
  return *(std::get_if<T>(&vlist[idx]));
};

// If vi is not int, wrap it with "(xxx)".
inline std::string WrapWithParentheses(const ValueItem& vi) {
  if (!isa<int>(&vi)) return "(" + ValueItemAsString(vi) + ")";
  return ValueItemAsString(vi);
}

// some operations
inline ValueItem operator+(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return WrapWithParentheses(vi1) + "+" + WrapWithParentheses(vi2);
  return *cast<int>(&vi1) + *cast<int>(&vi2);
}

inline ValueItem operator-(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return WrapWithParentheses(vi1) + "-" + WrapWithParentheses(vi2);
  return *cast<int>(&vi1) - *cast<int>(&vi2);
}

inline ValueItem operator*(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return WrapWithParentheses(vi1) + "*" + WrapWithParentheses(vi2);
  return *cast<int>(&vi1) * *cast<int>(&vi2);
}

inline ValueItem operator/(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return WrapWithParentheses(vi1) + "/" + WrapWithParentheses(vi2);
  return *cast<int>(&vi1) / *cast<int>(&vi2);
}

inline ValueItem operator%(const ValueItem& vi1, const ValueItem& vi2) {
  if (!isa<int>(&vi1) || !isa<int>(&vi2))
    return WrapWithParentheses(vi1) + "%" + WrapWithParentheses(vi2);
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
  if (a.index() != b.index()) return false; // Different types

  return a == b;
}

// Function to compare two ValueList
inline bool isValueListEqual(const ValueList& a, const ValueList& b) {
  if (a.size() != b.size()) return false; // Different sizes

  for (size_t i = 0; i < a.size(); ++i)
    if (!IsValueItemEqual(a[i], b[i])) return false; // Found a mismatch

  return true; // All elements match
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

inline void PrintValueListSizeExpr(const ValueList& vl, std::ostream& os,
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

} // end namespace Choreo
#endif // __CHOREO_SYMBOL_VALUES_H__
