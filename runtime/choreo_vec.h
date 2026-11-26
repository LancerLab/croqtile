#ifndef __CHOREO_VEC_RUNTIME_H__
#define __CHOREO_VEC_RUNTIME_H__

// Fixed-length logical vector values used by the explicit vec DSL.
//
// This header is a value-semantics fallback. It deliberately does not own
// memory access, gather/scatter, masked-memory lowering, or target builtins;
// target codegen remains responsible for those. Every lane operation below is
// expanded at compile time and must not introduce a runtime loop.

#include <cstddef>
#include <utility>

#ifndef __co_any__
  #define __co_any__
#endif

namespace choreo {

template <std::size_t N>
struct mask {
  static_assert(N > 0, "a vector mask needs at least one lane.");

  bool lanes[N];

  __co_any__ bool operator[](std::size_t i) const { return lanes[i]; }
  __co_any__ bool& operator[](std::size_t i) { return lanes[i]; }
};

template <typename T, std::size_t N>
struct simd {
  static_assert(N > 0, "a vector needs at least one lane.");

  T lanes[N];

  __co_any__ const T& operator[](std::size_t i) const { return lanes[i]; }
  __co_any__ T& operator[](std::size_t i) { return lanes[i]; }
};

namespace detail {

template <typename T, std::size_t N, std::size_t... I>
__co_any__ simd<T, N> splat_impl(T value, std::index_sequence<I...>) {
  return simd<T, N>{{(static_cast<void>(I), value)...}};
}

template <typename T, std::size_t N, std::size_t... I>
__co_any__ simd<T, N> iota_impl(int start, int stride,
                                std::index_sequence<I...>) {
  return simd<T, N>{{static_cast<T>(start + static_cast<int>(I) * stride)...}};
}

#define CHOREO_VEC_BINARY_IMPL(NAME, OP)                                       \
  template <typename T, std::size_t N, std::size_t... I>                       \
  __co_any__ simd<T, N> NAME(const simd<T, N>& lhs, const simd<T, N>& rhs,     \
                             std::index_sequence<I...>) {                      \
    return simd<T, N>{{(lhs[I] OP rhs[I])...}};                                \
  }

CHOREO_VEC_BINARY_IMPL(add, +)
CHOREO_VEC_BINARY_IMPL(sub, -)
CHOREO_VEC_BINARY_IMPL(mul, *)
CHOREO_VEC_BINARY_IMPL(div, /)
CHOREO_VEC_BINARY_IMPL(mod, %)
CHOREO_VEC_BINARY_IMPL(bit_and, &)
CHOREO_VEC_BINARY_IMPL(bit_or, |)
CHOREO_VEC_BINARY_IMPL(bit_xor, ^)
CHOREO_VEC_BINARY_IMPL(shl, <<)
CHOREO_VEC_BINARY_IMPL(shr, >>)

#undef CHOREO_VEC_BINARY_IMPL

template <typename T, std::size_t N, std::size_t... I>
__co_any__ simd<T, N> neg_impl(const simd<T, N>& value,
                               std::index_sequence<I...>) {
  return simd<T, N>{{(-value[I])...}};
}

template <typename T, std::size_t N, std::size_t... I>
__co_any__ simd<T, N> bitnot_impl(const simd<T, N>& value,
                                  std::index_sequence<I...>) {
  return simd<T, N>{{(~value[I])...}};
}

#define CHOREO_VEC_COMPARE_IMPL(NAME, OP)                                      \
  template <typename T, std::size_t N, std::size_t... I>                       \
  __co_any__ mask<N> NAME(const simd<T, N>& lhs, const simd<T, N>& rhs,        \
                          std::index_sequence<I...>) {                         \
    return mask<N>{{(lhs[I] OP rhs[I])...}};                                   \
  }

CHOREO_VEC_COMPARE_IMPL(eq, ==)
CHOREO_VEC_COMPARE_IMPL(ne, !=)
CHOREO_VEC_COMPARE_IMPL(lt, <)
CHOREO_VEC_COMPARE_IMPL(le, <=)
CHOREO_VEC_COMPARE_IMPL(gt, >)
CHOREO_VEC_COMPARE_IMPL(ge, >=)

#undef CHOREO_VEC_COMPARE_IMPL

template <std::size_t N, std::size_t... I>
__co_any__ mask<N> mask_not_impl(const mask<N>& value,
                                 std::index_sequence<I...>) {
  return mask<N>{{(!value[I])...}};
}

template <std::size_t N, std::size_t... I>
__co_any__ mask<N> mask_and_impl(const mask<N>& lhs, const mask<N>& rhs,
                                 std::index_sequence<I...>) {
  return mask<N>{{(lhs[I] && rhs[I])...}};
}

template <std::size_t N, std::size_t... I>
__co_any__ mask<N> mask_or_impl(const mask<N>& lhs, const mask<N>& rhs,
                                std::index_sequence<I...>) {
  return mask<N>{{(lhs[I] || rhs[I])...}};
}

template <std::size_t N, std::size_t... I>
__co_any__ mask<N> mask_xor_impl(const mask<N>& lhs, const mask<N>& rhs,
                                 std::index_sequence<I...>) {
  return mask<N>{{(lhs[I] != rhs[I])...}};
}

template <std::size_t N, std::size_t... I>
__co_any__ mask<N> splat_mask_impl(bool value, std::index_sequence<I...>) {
  return mask<N>{{(static_cast<void>(I), value)...}};
}

template <typename T, std::size_t N, std::size_t... I>
__co_any__ simd<T, N> select_impl(const mask<N>& predicate,
                                  const simd<T, N>& lhs, const simd<T, N>& rhs,
                                  std::index_sequence<I...>) {
  return simd<T, N>{{(predicate[I] ? lhs[I] : rhs[I])...}};
}

template <typename U, typename T, std::size_t N, std::size_t... I>
__co_any__ simd<U, N> cast_impl(const simd<T, N>& value,
                                std::index_sequence<I...>) {
  return simd<U, N>{{static_cast<U>(value[I])...}};
}

} // namespace detail

template <typename T, std::size_t N>
__co_any__ simd<T, N> splat(T value) {
  return detail::splat_impl<T, N>(value, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
__co_any__ simd<T, N> iota(int start, int stride) {
  return detail::iota_impl<T, N>(start, stride, std::make_index_sequence<N>{});
}

#define CHOREO_VEC_BINARY(OP, NAME)                                            \
  template <typename T, std::size_t N>                                         \
  __co_any__ simd<T, N> operator OP(const simd<T, N>& lhs,                     \
                                    const simd<T, N>& rhs) {                   \
    return detail::NAME(lhs, rhs, std::make_index_sequence<N>{});              \
  }                                                                            \
  template <typename T, std::size_t N>                                         \
  __co_any__ simd<T, N> operator OP(const simd<T, N>& lhs, T rhs) {            \
    return lhs OP splat<T, N>(rhs);                                            \
  }                                                                            \
  template <typename T, std::size_t N>                                         \
  __co_any__ simd<T, N> operator OP(T lhs, const simd<T, N>& rhs) {            \
    return splat<T, N>(lhs) OP rhs;                                            \
  }

CHOREO_VEC_BINARY(+, add)
CHOREO_VEC_BINARY(-, sub)
CHOREO_VEC_BINARY(*, mul)
CHOREO_VEC_BINARY(/, div)
CHOREO_VEC_BINARY(%, mod)
CHOREO_VEC_BINARY(&, bit_and)
CHOREO_VEC_BINARY(|, bit_or)
CHOREO_VEC_BINARY(^, bit_xor)
CHOREO_VEC_BINARY(<<, shl)
CHOREO_VEC_BINARY(>>, shr)

#undef CHOREO_VEC_BINARY

template <typename T, std::size_t N>
__co_any__ simd<T, N> operator-(const simd<T, N>& value) {
  return detail::neg_impl(value, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
__co_any__ simd<T, N> operator~(const simd<T, N>& value) {
  return detail::bitnot_impl(value, std::make_index_sequence<N>{});
}

#define CHOREO_VEC_COMPARE(OP, NAME)                                           \
  template <typename T, std::size_t N>                                         \
  __co_any__ mask<N> operator OP(const simd<T, N>& lhs,                        \
                                 const simd<T, N>& rhs) {                      \
    return detail::NAME(lhs, rhs, std::make_index_sequence<N>{});              \
  }                                                                            \
  template <typename T, std::size_t N>                                         \
  __co_any__ mask<N> operator OP(const simd<T, N>& lhs, T rhs) {               \
    return lhs OP splat<T, N>(rhs);                                            \
  }                                                                            \
  template <typename T, std::size_t N>                                         \
  __co_any__ mask<N> operator OP(T lhs, const simd<T, N>& rhs) {               \
    return splat<T, N>(lhs) OP rhs;                                            \
  }

CHOREO_VEC_COMPARE(==, eq)
CHOREO_VEC_COMPARE(!=, ne)
CHOREO_VEC_COMPARE(<, lt)
CHOREO_VEC_COMPARE(<=, le)
CHOREO_VEC_COMPARE(>, gt)
CHOREO_VEC_COMPARE(>=, ge)

#undef CHOREO_VEC_COMPARE

template <std::size_t N>
__co_any__ mask<N> operator!(const mask<N>& value) {
  return detail::mask_not_impl(value, std::make_index_sequence<N>{});
}

template <std::size_t N>
__co_any__ mask<N> operator&(const mask<N>& lhs, const mask<N>& rhs) {
  return detail::mask_and_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <std::size_t N>
__co_any__ mask<N> operator|(const mask<N>& lhs, const mask<N>& rhs) {
  return detail::mask_or_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <std::size_t N>
__co_any__ mask<N> operator^(const mask<N>& lhs, const mask<N>& rhs) {
  return detail::mask_xor_impl(lhs, rhs, std::make_index_sequence<N>{});
}

template <std::size_t N>
__co_any__ mask<N> splat_mask(bool value) {
  return detail::splat_mask_impl<N>(value, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N>
__co_any__ simd<T, N> select(const mask<N>& predicate, const simd<T, N>& lhs,
                             const simd<T, N>& rhs) {
  return detail::select_impl(predicate, lhs, rhs,
                             std::make_index_sequence<N>{});
}

template <typename U, typename T, std::size_t N>
__co_any__ simd<U, N> cast(const simd<T, N>& value) {
  return detail::cast_impl<U>(value, std::make_index_sequence<N>{});
}

} // namespace choreo

#endif // __CHOREO_VEC_RUNTIME_H__
