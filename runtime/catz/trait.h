#ifndef CATZILLA_CATZ_TRAIT_H_
#define CATZILLA_CATZ_TRAIT_H_

#include <cassert>
#include <iostream>
#include <type_traits>

template <typename T>
struct is_compile_time_constant {
  static constexpr bool value =
      std::is_integral<T>::value &&
      std::is_const<std::remove_reference_t<T>>::value;
};

template <typename T>
constexpr bool is_compile_time_constant_v = is_compile_time_constant<T>::value;

template <typename T, T Value, typename = void>
struct is_integral_constant_convertible : std::false_type {};

// 2. If it can be used as a template parameter, match this specialization
// version
template <typename T, T Value>
struct is_integral_constant_convertible<
    T, Value, std::void_t<decltype(std::integral_constant<T, Value>{})>>
    : std::true_type {};

// 3. Simplified version to check if a variable can be converted
template <typename T, T Value>
constexpr bool is_integral_constant_convertible_v =
    is_integral_constant_convertible<T, Value>::value;

template <typename T>
constexpr bool is_modifiable_variable =
    !std::is_const_v<std::remove_reference_t<T>>;

template <typename T>
constexpr bool is_immutable_v = std::is_const_v<std::remove_reference_t<T>>;

template <typename T>
struct is_allowed_type
    : std::disjunction<std::is_same<T, float>, std::is_same<T, half>,
                       std::is_same<T, __nv_bfloat16>,
                       std::is_same<T, float4>> {};

#endif // CATZILLA_CATZ_TRAIT_H_
