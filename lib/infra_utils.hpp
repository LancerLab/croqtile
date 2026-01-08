#ifndef __CHOREO_INFRA_UTILITY_HPP__
#define __CHOREO_INFRA_UTILITY_HPP__

#include "io.hpp"

namespace Choreo {

template <typename T>
using ptr = std::shared_ptr<T>;

// smart typeid provider suggested by GPT
template <typename T>
struct TypeIDProvider {
  static int __unique_id;
};

template <typename T>
int TypeIDProvider<T>::__unique_id;

// User defined type that utilize isa/cast/dyn_cast must place the macro inside
// its class definition
//
#define __UDT_TYPE_INFO__(PTYPE, TTYPE)                                        \
  const std::string TypeNameString() const override {                          \
    std::string name = __PRETTY_FUNCTION__;                                    \
    std::regex prefix_regex("^.*Choreo::");                                    \
    name = std::regex_replace(name, prefix_regex, "");                         \
    std::regex suffix_regex("::TypeNameString.*$");                            \
    name = std::regex_replace(name, suffix_regex, "");                         \
    return name;                                                               \
  }                                                                            \
  static uint64_t TypeID() {                                                   \
    return (uint64_t)&(TypeIDProvider<TTYPE>::__unique_id);                    \
  }                                                                            \
  bool IsType(uint64_t ty) const override {                                    \
    return (ty == (uint64_t)&(TypeIDProvider<TTYPE>::__unique_id)) ||          \
           PTYPE::IsType(ty);                                                  \
  }

#define __UDT_2TYPES_INFO__(PTYPE1, PTYPE2, TTYPE)                             \
  const std::string TypeNameString() const override {                          \
    std::string name = __PRETTY_FUNCTION__;                                    \
    std::regex prefix_regex("^.*Choreo::");                                    \
    name = std::regex_replace(name, prefix_regex, "");                         \
    std::regex suffix_regex("::TypeNameString.*$");                            \
    name = std::regex_replace(name, suffix_regex, "");                         \
    return name;                                                               \
  }                                                                            \
  static uint64_t TypeID() {                                                   \
    return (uint64_t)&(TypeIDProvider<TTYPE>::__unique_id);                    \
  }                                                                            \
  bool IsType(uint64_t ty) const override {                                    \
    return (ty == (uint64_t)&(TypeIDProvider<TTYPE>::__unique_id)) ||          \
           PTYPE1::IsType(ty) || PTYPE2::IsType(ty);                           \
  }

#define __UDT_TYPE_INFO_BASE__(NAME)                                           \
  virtual const std::string TypeNameString() const { return #NAME; }           \
  static uint64_t TypeID() { return 0xDEADBEEFULL; }                           \
  virtual bool IsType(uint64_t ty) const { return ty == 0xDEADBEEFULL; }

#define __UDT_TYPE_INFO_BASE1__(NAME)                                          \
  virtual const std::string TypeNameString() const { return #NAME; }           \
  static uint64_t TypeID() { return 0xBAADF00DULL; }                           \
  virtual bool IsType(uint64_t ty) const { return ty == 0xBAADF00DULL; }

// LLVM-style type utility functions
//
// Note:
// To be simple, we do not handle any relationship about inheritance but only
// the extact (most-derived) type
//

template <typename T, typename U>
bool isa(U* n) {
  if (!n) return false;
  return n->IsType(T::TypeID());
}
template <typename T, typename U>
bool isa(const ptr<U>& n) {
  if (!n) return false;
  return n->IsType(T::TypeID());
}

template <typename T, typename U>
T* dyn_cast(U* n) {
  if (isa<T>(n))
    return static_cast<T*>(const_cast<std::remove_cv_t<U>*>(n));
  else
    return nullptr;
}
template <typename T, typename U>
ptr<T> dyn_cast(const ptr<U>& n) {
  if (isa<T>(n))
    return std::static_pointer_cast<T>(n);
  else
    return nullptr;
}

template <typename T, typename U>
T* cast(U* n) {
  auto t = dyn_cast<T, U>(n);
  if (t == nullptr) {
    errs() << "type cast failure for incompatibility.\n";
    abort();
  }
  return t;
}

template <typename T, typename U>
ptr<T> cast(const ptr<U>& n) {
  auto t = dyn_cast<T, U>(n);
  if (t == nullptr) {
    errs() << "type cast failure for incompatibility.\n";
    abort();
  }
  return t;
}

// for debug purpose only
template <typename T, typename U>
T* cast_dbg(U* n) {
  auto t = dyn_cast<T, U>(n);
  if (t == nullptr) {
    errs() << "type cast failure for incompatibility: " << n->TypeNameString()
           << ".\n";
    abort();
  }
  return t;
}

template <typename T, typename U>
ptr<T> cast_dbg(const ptr<U>& n) {
  auto t = dyn_cast<T, U>(n);
  if (t == nullptr) {
    errs() << "type cast failure for incompatibility: " << n->TypeNameString()
           << ".\n";
    abort();
  }
  return t;
}

template <typename N>
const ptr<N> CloneP(const ptr<N>& i) {
  if (i == nullptr) return nullptr;
  return cast<N>(i->Clone());
}

} // end namespace Choreo

#endif // __CHOREO_INFRA_UTILITY_HPP__
