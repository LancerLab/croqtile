#ifndef __CHOREO_SYMBOL_VALUES_H__
#define __CHOREO_SYMBOL_VALUES_H__

#include "symbexpr.hpp"
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace Choreo {

// Define ValueList as a group of values
template <class T>
inline constexpr bool always_false = false;

namespace __internal {
static constexpr size_t INVALID_UNSIGNED = std::numeric_limits<size_t>::max();
static constexpr size_t UNKNOWN_UNSIGNED =
    std::numeric_limits<size_t>::max() - 1;
static constexpr int INVALID_SIGNED = std::numeric_limits<int>::max();
static constexpr int UNKNOWN_SIGNED =
    std::numeric_limits<int>::min(); // represent literal value '?' only
static constexpr float INVALID_FLOAT = std::numeric_limits<float>::quiet_NaN();
static constexpr double INVALID_DOUBLE =
    std::numeric_limits<double>::quiet_NaN();
static constexpr float UNKNOWN_FLOAT = std::numeric_limits<float>::infinity();
static constexpr double UNKNOWN_DOUBLE =
    std::numeric_limits<double>::infinity();
} // namespace __internal

inline constexpr size_t GetInvalidUnsigned() {
  return __internal::INVALID_UNSIGNED;
}
inline constexpr size_t GetUnknownUnsigned() {
  return __internal::UNKNOWN_UNSIGNED;
}
inline constexpr int GetInvalidSigned() { return __internal::INVALID_SIGNED; }
inline constexpr float GetInvalidFloat() { return __internal::INVALID_FLOAT; }
inline constexpr double GetInvalidDouble() {
  return __internal::INVALID_DOUBLE;
}
inline constexpr int GetUnKnownInteger() { return __internal::UNKNOWN_SIGNED; }
inline constexpr float GetUnKnownFloat() { return __internal::UNKNOWN_FLOAT; }
inline constexpr double GetUnKnownDouble() {
  return __internal::UNKNOWN_DOUBLE;
}

inline constexpr bool IsValidUnsigned(size_t v) {
  return v != GetInvalidUnsigned();
}
inline constexpr bool IsValidSigned(int v) { return v != GetInvalidSigned(); }
inline constexpr bool IsUnKnownInteger(int v) {
  return v == GetUnKnownInteger();
}
inline constexpr bool IsUnKnownInteger(int64_t v) {
  return v == GetUnKnownInteger();
}
inline constexpr bool IsValidFloatPoint(float v) { return !std::isnan(v); }
inline constexpr bool IsValidFloatPoint(double v) { return !std::isnan(v); }
inline constexpr bool IsUnKnownFloatPoint(float v) { return std::isinf(v); }
inline constexpr bool IsUnKnownFloatPoint(double v) { return std::isinf(v); }

inline constexpr size_t GetInvalidRank() { return GetInvalidUnsigned(); }
inline constexpr size_t GetUnknownRank() { return GetUnknownUnsigned(); }
inline constexpr int GetInvalidValueNumber() { return GetInvalidSigned(); }
inline constexpr int GetInvalidBound() { return GetInvalidSigned(); }
inline constexpr int GetInvalidStep() { return GetInvalidSigned(); }

inline constexpr bool IsValidRank(size_t v) { return v != GetInvalidRank(); }
inline constexpr bool IsUnknownRank(size_t v) { return v == GetUnknownRank(); }
inline constexpr bool IsValidValueNumber(int v) {
  return v != GetInvalidValueNumber();
}
inline constexpr bool IsValidBound(int v) { return v != GetInvalidBound(); }
inline constexpr bool IsValidStep(int v) { return v != GetInvalidStep(); }

// ------------------------------------------------------------------------- //
using ValueItem = sbe::Operand;
using ValueList = std::vector<ValueItem>;

inline ValueItem GetInvalidValueItem() { return nullptr; }
inline bool IsValidValueItem(const ValueItem& vi) { return vi != nullptr; }
inline bool IsValidValueList(const ValueList& vl) {
  for (auto& vi : vl)
    if (!IsValidValueItem(vi)) return false;
  return true;
}

inline bool IsValueListNumeric(const ValueList& vl) {
  for (auto& vi : vl)
    if (!isa<sbe::NumericValue>(vi)) return false;
  return true;
}

inline bool IsValueListNumericOrBool(const ValueList& vl) {
  for (auto& vi : vl)
    if (!isa<sbe::NumericValue>(vi) && !isa<sbe::BooleanValue>(vi))
      return false;
  return true;
}

inline bool IsComputable(const ValueItem& vi) {
  assert(IsValidValueItem(vi));
  return vi->Computable();
}

inline bool IsComputable(const ValueList& vl) {
  for (auto& vi : vl)
    if (!IsComputable(vi)) return false;
  return true;
}

inline std::string STR(const ValueItem& vi) {
  if (!IsValidValueItem(vi)) return "invalid";
  return vi->ToString();
}

inline bool operator==(const ValueList& l, const ValueList& r) {
  if (l.size() != r.size()) return false;
  for (size_t i = 0; i < l.size(); ++i)
    if (!sbe::ceq(l[i], r[i])) return false;
  return true;
}

inline bool VIIsNil(const ValueItem& vi) { return isa<sbe::InvalidValue>(vi); }
inline static std::optional<int64_t> VIInt(const ValueItem& vi) {
  if (auto iv = dyn_cast<sbe::NumericValue>(vi)) return iv->Value();
  return std::nullopt;
}
inline static bool VIIsInt(const ValueItem& vi) {
  return VIInt(vi).has_value();
}
inline static std::optional<bool> VIBool(const ValueItem& vi) {
  if (auto iv = dyn_cast<sbe::BooleanValue>(vi)) return iv->Value();
  return std::nullopt;
}
inline static bool VIIsBool(const ValueItem& vi) {
  return VIBool(vi).has_value();
}

inline static std::optional<std::string> VISym(const ValueItem& vi) {
  if (auto iv = dyn_cast<sbe::SymbolicValue>(vi)) return iv->Value();
  return std::nullopt;
}

inline static bool VIIsSym(const ValueItem& vi) {
  return VISym(vi).has_value();
}

inline static std::shared_ptr<sbe::UnaryOperation> VIUop(const ValueItem& vi) {
  return dyn_cast<sbe::UnaryOperation>(vi);
}
inline static bool VIIsUop(const ValueItem& vi) { return VIUop(vi) != nullptr; }

inline static std::shared_ptr<sbe::BinaryOperation> VIBop(const ValueItem& vi) {
  return dyn_cast<sbe::BinaryOperation>(vi);
}
inline static bool VIIsBop(const ValueItem& vi) { return VIBop(vi) != nullptr; }

inline static std::shared_ptr<sbe::TernaryOperation>
VITop(const ValueItem& vi) {
  return dyn_cast<sbe::TernaryOperation>(vi);
}
inline static bool VIIsTop(const ValueItem& vi) { return VITop(vi) != nullptr; }

template <typename T>
inline T GetValueAt(ValueList vlist, int idx) {
  return *(std::get_if<T>(&vlist[idx]));
};

inline const std::set<ValueItem> GetSymbols(const ValueItem& vi) {
  if (VIIsSym(vi))
    return {vi};
  else if (auto uop = VIUop(vi)) {
    return GetSymbols(uop);
  } else if (auto bop = VIBop(vi)) {
    auto ls = GetSymbols(bop->GetLeft());
    auto rs = GetSymbols(bop->GetRight());
    ls.insert(rs.begin(), rs.end());
    return ls;
  } else if (auto bop = VITop(vi)) {
    auto ps = GetSymbols(bop->GetPred());
    auto ls = GetSymbols(bop->GetLeft());
    auto rs = GetSymbols(bop->GetRight());
    ps.insert(ls.begin(), ls.end());
    ps.insert(rs.begin(), rs.end());
    return ps;
  }

  return {};
}

inline const std::set<ValueItem> GetSymbols(const ValueList& vl) {
  std::set<ValueItem> res;
  for (auto& vi : vl) {
    auto vs = GetSymbols(vi);
    res.insert(vs.begin(), vs.end());
  }
  return res;
}

struct ValueListHasher {
  std::size_t operator()(const ValueList& val) const noexcept {
    std::size_t hash = 0;
    sbe::OperandHasher vi_hasher;
    for (const auto& v : val)
      hash ^= vi_hasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    return hash;
  }
};

inline bool IsValueItemEqual(const ValueItem& a, const ValueItem& b) {
  return sbe::ceq(a->Normalize(), b->Normalize());
}

inline bool IsValueItemEqual(int a, const ValueItem& b) {
  return sbe::ceq(sbe::nu(a), b->Normalize());
}

inline const ValueItem MultiplyAll(const ValueList& vl) {
  auto res = sbe::nu(1);
  for (auto vi : vl) res = res * vi;
  return res->Normalize();
}

// Function to compare two ValueList
inline bool IsValueListEqual(const ValueList& a, const ValueList& b) {
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
      if (IsValueListEqual(values[hash_index[hash_val]], st)) return hash_val;

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
  if (lb) os << lb;
  if (!vl.empty()) {
    os << STR(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) { os << ", " << vl[i]; }
  }
  if (rb) os << rb;
}

inline const std::string STR(const ValueList& vl) {
  std::ostringstream os;
  PrintValueList(vl, os, "", "");
  return os.str();
}

inline void PrintValueListSizeExpr(const ValueList& vl, std::ostream& os,
                                   const char* lb = "[", const char* rb = "]") {
  if (lb) os << lb;
  if (!vl.empty()) {
    os << vl[0];
    for (unsigned i = 1; i < vl.size(); ++i) {
      os << " * ";
      os << vl[i];
    }
  }
  if (rb) os << rb;
}

inline const ValueList ValxN(ValueItem vi, size_t count) {
  ValueList vl(count);
  std::fill(vl.begin(), vl.end(), vi);
  return vl;
}

inline const ValueList operator+(const ValueList& vl, const ValueItem& vi) {
  ValueList res;
  for (auto& i : vl) res.push_back((i + vi)->Normalize());
  return res;
}
inline const ValueList operator-(const ValueList& vl, const ValueItem& vi) {
  ValueList res;
  for (auto& i : vl) res.push_back((i - vi)->Normalize());
  return res;
}
inline const ValueList operator*(const ValueList& vl, const ValueItem& vi) {
  ValueList res;
  for (auto& i : vl) res.push_back((i * vi)->Normalize());
  return res;
}
inline const ValueList operator/(const ValueList& vl, const ValueItem& vi) {
  ValueList res;
  for (auto& i : vl) res.push_back((i / vi)->Normalize());
  return res;
}

inline const ValueList operator+(const ValueItem& vi, const ValueList& vl) {
  return vl + vi;
}
inline const ValueList operator*(const ValueItem& vi, const ValueList& vl) {
  return vl * vi;
}

inline const ValueList Reverse(const ValueList& vl) {
  ValueList res(vl.size());
  for (size_t i = 0; i < vl.size(); ++i) res[vl.size() - i - 1] = vl[i];
  return res;
}

// Trim the first-n value
inline const ValueList Trim(const ValueList& vl, size_t n = 1) {
  assert(vl.size() > n);
  ValueList res;
  for (size_t i = n; i < vl.size(); ++i) res.push_back(vl[i]);
  return res;
}

} // end namespace Choreo

#endif // __CHOREO_SYMBOL_VALUES_H__
