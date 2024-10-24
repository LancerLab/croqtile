#ifndef CATZILLA_CATZ_INDEX_H_
#define CATZILLA_CATZ_INDEX_H_

#include "trait.h"
#include <type_traits>

template <int Value>
struct Index;

struct IndexDyn {
  int value;

  constexpr IndexDyn(int v) : value(v) {}
  constexpr IndexDyn() : value(0) {}

  constexpr int operator()() const { return value; }

  static constexpr bool isStatic() { return false; }

  // Prefix increment operator
  constexpr IndexDyn &operator++() {
    ++value;
    return *this;
  }
};

template <int Value>
struct Index {
  static constexpr int value = Value;

  constexpr Index() {}
  constexpr Index(int v) {}

  constexpr int operator()() const { return value; }

  static constexpr bool isStatic() { return true; }

  constexpr Index<1 + Value> operator++() { return Index<Value + 1>(); }
};

// Primary template for is_index_like
template <typename T, typename = void>
struct is_index_like : std::false_type {};

// Specialization for IndexDyn
template <>
struct is_index_like<IndexDyn> : std::true_type {};

template <typename U>
struct is_index_like<U, std::enable_if_t<std::is_integral_v<
                            std::remove_const_t<decltype(U::value)>>>>
    : std::true_type {};

// Helper variable templates
template <typename T>
inline constexpr bool is_index_like_v = is_index_like<T>::value;

template <
    typename Lhs, typename Rhs,
    typename = std::enable_if_t<is_index_like_v<Lhs> && is_index_like_v<Rhs>>>
constexpr bool operator<(const Lhs &lhs, const Rhs &rhs) {
  return lhs.value < rhs.value;
}

#define DEFINE_BINARY_OPERATOR_FOR_INDEX(op)                                   \
  template <int Value, int OtherValue>                                         \
  constexpr auto operator op(const Index<Value> &lhs,                          \
                             const Index<OtherValue> &rhs) {                   \
    return Index<Value op OtherValue>();                                       \
  }                                                                            \
                                                                               \
  template <int Value>                                                         \
  constexpr auto operator op(const IndexDyn &lhs, const Index<Value> &rhs) {   \
    return IndexDyn(lhs.value op Value);                                       \
  }                                                                            \
                                                                               \
  template <int Value>                                                         \
  constexpr auto operator op(const Index<Value> &lhs, const IndexDyn &rhs) {   \
    return IndexDyn(lhs.value op rhs.value);                                   \
  }                                                                            \
                                                                               \
  constexpr auto operator op(const IndexDyn &lhs, const IndexDyn &rhs) {       \
    return IndexDyn(lhs.value op rhs.value);                                   \
  }                                                                            \
  // /* Index + int */                                                   \
  // template <int Value>                                                 \
  // constexpr auto operator op(const Index<Value>& lhs, int rhs) {      \
  //     return Index<Value op rhs>{};                                    \
  // }                                                                    \
  // /* int + Index */                                                    \
  // template <int Value>                                                 \
  // constexpr auto operator op(int lhs, const Index<Value>& rhs) {      \
  //     return Index<lhs op Value>{};                                    \
  // }                                                                    \

#define DEFINE_POINTER_OPERATOR_FOR_INDEX(op)                                  \
  /* need to constraint allowed type */                                        \
  /* float* + Index */                                                         \
  template <int Value>                                                         \
  constexpr __device__ float *operator op(float *ptr,                          \
                                          const Index<Value> &index) {         \
    return ptr op index.value;                                                 \
  }                                                                            \
  /* Index + float* */                                                         \
  template <int Value>                                                         \
  constexpr __device__ float *operator op(const Index<Value> &index,           \
                                          float *ptr) {                        \
    return ptr op index.value;                                                 \
  }                                                                            \
  constexpr __device__ float *operator op(float *ptr, const IndexDyn &index) { \
    return ptr op index.value;                                                 \
  }                                                                            \
  /* Index + float* */                                                         \
  constexpr __device__ float *operator op(const IndexDyn &index, float *ptr) { \
    return ptr op index.value;                                                 \
  }

DEFINE_BINARY_OPERATOR_FOR_INDEX(+)
DEFINE_BINARY_OPERATOR_FOR_INDEX(-)
DEFINE_BINARY_OPERATOR_FOR_INDEX(*)
DEFINE_BINARY_OPERATOR_FOR_INDEX(/)
DEFINE_BINARY_OPERATOR_FOR_INDEX(%)

DEFINE_POINTER_OPERATOR_FOR_INDEX(+)
DEFINE_POINTER_OPERATOR_FOR_INDEX(-)

// define special ceil-div
template <int Value, int OtherValue>
constexpr auto ceil_div(const Index<Value> &lhs, const Index<OtherValue> &rhs) {
  return Index<(Value + OtherValue - 1) / OtherValue>();
}

constexpr auto ceil_div(const IndexDyn &lhs, const IndexDyn &rhs) {
  return IndexDyn((lhs.value + rhs.value - 1) / rhs.value);
}

template <int Value>
constexpr auto ceil_div(const Index<Value> &lhs, const IndexDyn &rhs) {
  return IndexDyn((lhs.value + rhs.value - 1) / rhs.value);
}

template <int Value>
constexpr auto ceil_div(const IndexDyn &lhs, const Index<Value> &rhs) {
  return IndexDyn((lhs.value + rhs - 1) / rhs);
}

template <int Value>
constexpr auto make_index() {
  return Index<Value>();
}

constexpr auto make_index(int value) {
  return IndexDyn(value); // 返回 IndexDyn 实例
}

/////////// macros for convinience
///

#define DEFINE_INDEX(n) using I##n = Index<n>;
using I = IndexDyn;

// 展开所有 2 的幂次别名，从 1 到 1024
DEFINE_INDEX(0)
DEFINE_INDEX(1)
DEFINE_INDEX(2)
DEFINE_INDEX(4)
DEFINE_INDEX(8)
DEFINE_INDEX(16)
DEFINE_INDEX(32)
DEFINE_INDEX(64)
DEFINE_INDEX(128)
DEFINE_INDEX(256)
DEFINE_INDEX(512)
DEFINE_INDEX(1024)

#endif // CATZILLA_CATZ_INDEX_H_
