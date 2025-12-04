#ifndef __CHOREO_H__
#define __CHOREO_H__

#if __cplusplus < 201703L
// #error "Choreo requires C++17 or later"
#endif

#include <algorithm>
#include <cmath>            // For fp16
#include <cstdint>          // For fixed-width integer types
#include <initializer_list> // for std::initializer_list
#include <iostream>         // report error
#include <map>
#include <memory>
#include <random>

#ifdef __TOPSCC__

#define __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__
// #define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
#define __co_device__ __device__
#define __co_host__ __host__
#define __co_any__ __device__ __host__

#elif defined(__CHOREO_TARGET_CUTE__)
#ifdef __USE_CUDA_TYPE__
#include "cuda.h"
#if CUDA_VERSION >= 11000
#define __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__
#include "cuda_bf16.h"
#endif

#define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
#include "cuda_fp16.h"

#if CUDA_VERSION >= 11080
#define __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
#if CUDA_VERSION >= 12090
#define __CHOREO_TARGET_NATIVE_FP8_E8M0_SUPPORT__
#endif
#include "cuda_fp8.h"
#endif

#if CUDA_VERSION >= 12090
#define __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
#define __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
#include "cuda_fp4.h"
#include "cuda_fp6.h"
#endif

#define __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
#else // __USE_CUTE_TYPE__
#define __CHOREO_TARGET_NATIVE_TFLOAT32_SUPPORT__
#define __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__
#define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
#define __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
#define __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
#define __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
#define __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
#endif

#include "cute/tensor.hpp"
#include <cuda/barrier>
#include <mma.h>

#define __co_device__
#define __co_host__
#define __co_any__

#else

#define __co_device__
#define __co_host__
#define __co_any__

#endif // TOPSCC and CUTE

#if __GCU_ARCH__ == 400
#define __CHOREO_BLOCK_SINGLE__                                                \
  threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&                  \
      subThreadIdx.x == 0 && subThreadIdx.y == 0 && subThreadIdx.z == 0
#define __CHOREO_GROUP_SINGLE__                                                \
  subThreadIdx.x == 0 && subThreadIdx.y == 0 && subThreadIdx.z == 0
#elif defined(__CHOREO_TARGET_CUTE__)
#define __CHOREO_BLOCK_SINGLE__                                                \
  threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
#define __CHOREO_GROUP_SINGLE__(GSIZE) (threadIdx.x % GSIZE) == 0
#else
#define __CHOREO_BLOCK_SINGLE__                                                \
  threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0
#define __CHOREO_GROUP_SINGLE__ "invalid to use sublocal predicate"
#endif

#define __cok__ namespace choreo

namespace choreo {

constexpr size_t __inf__ = (size_t)((1LL << 32) - 1);

inline void choreo_assert(bool p, const char* msg, const char* file = __FILE__,
                          int line = __LINE__) {
  if (!p) {
    std::cerr << file << ":" << line << ": choreo assertion abort: " << msg
              << std::endl;
    std::abort();
  }
  return;
}

inline void runtime_check(bool p, const char* msg) {
  if (!p) {
    std::cerr << "choreo runtime check failed: " << msg << std::endl;
    std::abort();
  }
  return;
}

inline void runtime_check(bool p, const std::string& msg) {
  if (!p) {
    std::cerr << "choreo runtime check failed: " << msg << std::endl;
    std::abort();
  }
  return;
}

#ifdef __TOPSCC__
template <typename T>
__co_device__ inline void fill(T* begin, T* end, const T& value) {
  for (size_t idx = 0; idx < end - begin; ++idx) begin[idx] = value;
  // TODO: OPT
}

template <typename T>
__co_device__ inline void fill_n(T* begin, size_t n, const T& value) {
  for (size_t idx = 0; idx < n; ++idx) begin[idx] = value;
  // TODO: OPT
}
#endif // __TOPSCC__

template <typename T>
__co_host__ inline void fill(T* begin, T* end, const T& value) {
  std::fill(begin, end, value);
}

template <typename T>
__co_host__ inline void fill_n(T* begin, size_t n, const T& value) {
  std::fill_n(begin, n, value);
}

namespace {

template <typename T, size_t N>
class SimpleArray {
  static_assert(N > 0, "can not create 0-dim array");

public:
  // Constructor for brace-initialization
  __co_any__ SimpleArray(std::initializer_list<T> init) {
    std::size_t num_elements = init.size();
    if (num_elements == 1) {
      fill(data, data + N, *init.begin());
    } else {
      for (size_t i = 0; i < num_elements && i < N; ++i)
        data[i] = *(init.begin() + i);
    }
  }

  __co_any__ SimpleArray(const SimpleArray&) = default;
  __co_any__ SimpleArray& operator=(const SimpleArray&) = default;
  __co_any__ ~SimpleArray() = default;

  // Returns the element at specified index
  __co_any__ T& operator[](uint32_t index) { return data[index]; }

  // Returns the element at specified index (const version)
  __co_any__ const T& operator[](uint32_t index) const { return data[index]; }

  // Returns the number of elements in the array
  __co_any__ constexpr uint32_t size() const noexcept { return N; }

  // Returns a pointer to the underlying array serving as element storage
  __co_any__ T* begin() { return data; }
  __co_any__ const T* begin() const { return data; }

  __co_any__ T* end() { return data + N; }
  __co_any__ const T* end() const { return data + N; }

  void fill_random() { fill_random(data, std::is_floating_point<T>()); }

private:
  T data[N];

  template <typename U>
  typename std::enable_if<std::is_floating_point<U>::value>::type
  fill_random(U (&array)[N], std::true_type) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // floating-point range [-1.0, 1.0)
    std::uniform_real_distribution<U> rand_func(-1.0,
                                                1.0); // range [-1.0, 1.0)

    std::generate_n(&array[0], N, [&]() { return rand_func(gen); });
  }

  // if T is integer, use std::uniform_int_distribution
  template <typename U>
  typename std::enable_if<std::is_integral<U>::value>::type
  fill_random(U (&array)[N], std::false_type) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // integers range [-100, 100]
    std::uniform_int_distribution<U> rand_func(-100, 100);

    std::generate_n(&array[0], N, [&]() { return rand_func(gen); });
  }
};

template <typename T, size_t N, size_t M>
__co_any__ inline static bool operator==(const SimpleArray<T, N>& l,
                                         const SimpleArray<T, M>& r) {
  if constexpr (N != M)
    return false;
  else {
    for (size_t i = 0; i < N; ++i)
      if (l.data[i] != r.data[i]) return false;
    return true;
  }
}

} // end anonymous namespace

template <int Rank>
using mdspan = SimpleArray<size_t, Rank>;

template <size_t N>
inline std::ostream& operator<<(std::ostream& os, const mdspan<N>& s) {
  for (size_t i = 0; i < N; ++i) os << s[i] << " ";
  return os;
}

template <size_t Rank>
inline size_t span_size(const mdspan<Rank>& s) {
  size_t sz = 1;
  for (size_t i = 0; i < Rank; ++i) sz *= s[i];
  return sz;
}

namespace {

// For multi-dimensional array reference
template <typename T, size_t N>
class ArrayProxy {
  T* data;
  const mdspan<N>* dims;
  size_t offset;

public:
  __co_any__ ArrayProxy(T* arr, const mdspan<N>& dimensions, size_t off)
      : data(arr), dims(&dimensions), offset(off) {}

  template <size_t M = N>
  typename std::enable_if<(M == 1),
                          T&>::type // make sure to return the reference type
      __co_any__
      operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < (*dims)[0], "Index out of bounds", __FILE__,
                  __LINE__);

    // Direct element access
    return data[offset + (size_t)index];
  }

  template <size_t M = N>
  typename std::enable_if<(M > 1), ArrayProxy<T, N - 1>>::type __co_any__
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < (*dims)[0], "Index out of bounds", __FILE__,
                  __LINE__);

    // Recurse with reduced dimensionality
    const auto& sub_dims =
        *reinterpret_cast<const mdspan<N - 1>*>(&((*dims)[1]));
    return ArrayProxy<T, N - 1>(data, sub_dims,
                                (offset + (size_t)index) * (*dims)[1]);
  }
};

} // end anonymous namespace

// Floating-point types
using f64 = double;
using f32 = float;

#ifdef __CHOREO_TARGET_NATIVE_TFLOAT32_SUPPORT__
// TF32 is only used in tensor core in CUDA and CUTE
#if defined(__USE_CUTE_TYPE__)
using cute::tfloat32_t;
#else
#error "TF32 type is not supported on this target."
#endif
using tf32 = tfloat32_t;
#endif

// Function to convert float to half precision bits
// Refer to https://en.wikipedia.org/wiki/Half-precision_floating-point_format
//    and https://en.wikipedia.org/wiki/Single-precision_floating-point_format
template <typename T, typename F>
__co_any__ inline static T __f32_to_f16(F value) {
  static_assert(sizeof(F) == 4, "source is not a float.");
  static_assert(sizeof(T) == 2, "target is not a half float.");

  uint32_t fltInt32 = *reinterpret_cast<uint32_t*>(&value);
  uint32_t sign = (fltInt32 >> 31) & 0x1;
  uint32_t exponent = ((fltInt32 >> 23) & 0xFF); // 8-bit exponent
  uint32_t fraction = fltInt32 & 0x7FFFFF;       // 23-bit fraction
  uint16_t resultBits = 0;

  if (exponent == 0x0 && fraction == 0x0) { // Zero
    resultBits = sign << 15;
    return *reinterpret_cast<T*>(&resultBits);
  }
  if (exponent == 0x0 && fraction != 0x0) { // Subnormal for float32
    // Subnormal float32 is all zero in float16
    resultBits = sign << 15;
    return *reinterpret_cast<T*>(&resultBits);
  }
  if (exponent == 0xFF && fraction == 0x0) { // Infinity
    resultBits = (sign << 15) | (0x1F << 10);
    return *reinterpret_cast<T*>(&resultBits);
  }
  if (exponent - 0x70 > 0x0 && exponent - 0x70 < 0x1F) { // Normalized value
    // Only exponent within [-14, 15] could be convert to normalized float16
    // Otherwise it will be inf
    // Why 0x70(112)? 112 = 127 - 15
    resultBits = (sign << 15) | (((exponent - 0x70) & 0x1F) << 10) |
                 ((fraction & 0x7FE000) >> 13);
    return *reinterpret_cast<T*>(&resultBits);
  } else { // Rest cases are all NaN.
    // This strategy is not quite appropriate and needs improvement.
    auto nanFraction = (fraction & 0x7FE000) >> 13;
    if (nanFraction == 0) { nanFraction += 1; }
    resultBits = (sign << 15) | (0x1F << 10) | nanFraction;
    return *reinterpret_cast<T*>(&resultBits);
  }
  return *reinterpret_cast<T*>(&resultBits);
}

// Function to convert half precision bits to float
// Refer to https://en.wikipedia.org/wiki/Half-precision_floating-point_format
//    and https://en.wikipedia.org/wiki/Single-precision_floating-point_format
template <typename T, typename F>
__co_any__ inline static T __f16_to_f32(F value) {
  static_assert(sizeof(T) == 4, "target is not a float.");
  static_assert(sizeof(F) == 2, "source is not a half float.");

  int16_t fltInt16 = *(int16_t*)&value;
  uint32_t sign = (fltInt16 >> 15) & 0x1;
  uint32_t exponent = ((fltInt16 >> 10) & 0x1F); // 5-bit exponent
  uint32_t fraction = fltInt16 & 0x3FF;          // 10-bit fraction
  uint32_t resultBits = 0;

  if (exponent == 0x0 && fraction == 0x0) { // Zero
    resultBits = sign << 31;
  }
  if (exponent == 0x0 && fraction != 0x0) { // Subnormal for float16
    // Subnormal float16 is normalized in float32.
    // Why 0x89(137)? 137 = 127 + 23 - 13
    // Why (fraction - 1)? Minus the implicit "1" from normalized
    resultBits = (sign << 31) | (0x89) << 23 | ((fraction - 1) << 13);
  }
  if (exponent > 0x0 && exponent < 0x1F) { // Normalized value
    // Why 112? 112 = 127 - 15
    resultBits = (sign << 31) | (exponent + 112) << 23 | (fraction << 13);
  }
  if (exponent == 0x1F && fraction != 0) { // Infinity or NaN
    resultBits = (sign << 31) | 0x7F800000 | (fraction << 13);
  }
  return *reinterpret_cast<T*>(&resultBits);
}

struct co_native_base {
  uint64_t data;
};

#ifndef __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__
// this f16 accepts literal initialization, but without arith support
class f16 {
private:
  uint16_t bits;

public:
  // Default constructor
  __co_any__ f16() = default;

  // Constructor for conversion from float
  __co_any__ f16(float value) { bits = __f32_to_f16<uint16_t>(value); }

  // Constructor for conversion from double
  __co_any__ f16(double value) {
    bits = __f32_to_f16<uint16_t>(static_cast<float>(value));
  }

  // Implicit conversion from float
  __co_any__ f16& operator=(float value) {
    bits = __f32_to_f16<uint16_t>(value);
    return *this;
  }

  // Implicit conversion from double
  __co_any__ f16& operator=(double value) {
    bits = __f32_to_f16<uint16_t>(static_cast<float>(value));
    return *this;
  }

  template <typename T>
  __co_any__ bool operator==(T value) {
    if constexpr (std::is_same<T, f16>::value) {
      auto valueF = (float)value;
      if (std::isnan(valueF)) { return std::isnan(__f16_to_f32<float>(bits)); }
      return __f16_to_f32<float>(bits) == valueF;
    } else {
      auto valueF = static_cast<float>(value);
      if (std::isnan(valueF)) { return std::isnan(__f16_to_f32<float>(bits)); }
      return __f16_to_f32<float>(bits) == valueF;
    }
  }

  template <typename T>
  __co_any__ bool operator>(T value) {
    if constexpr (std::is_same<T, f16>::value) {
      auto valueF = (float)value;
      if (std::isnan(valueF)) { return std::isnan(__f16_to_f32<float>(bits)); }
      return __f16_to_f32<float>(bits) > valueF;
    } else {
      auto valueF = static_cast<float>(value);
      if (std::isnan(valueF)) { return std::isnan(__f16_to_f32<float>(bits)); }
      return __f16_to_f32<float>(bits) > valueF;
    }
  }

  template <typename T>
  __co_any__ bool operator<(T value) {
    if constexpr (std::is_same<T, f16>::value) {
      auto valueF = (float)value;
      if (std::isnan(valueF)) { return std::isnan(__f16_to_f32<float>(bits)); }
      return __f16_to_f32<float>(bits) < valueF;
    } else {
      auto valueF = static_cast<float>(value);
      if (std::isnan(valueF)) { return std::isnan(__f16_to_f32<float>(bits)); }
      return __f16_to_f32<float>(bits) < valueF;
    }
  }

  // Method to get the float value from the f16 object
  __co_any__ operator float() const { return __f16_to_f32<float>(bits); }
};

using half = unsigned short; // device f16 type simulation

inline std::ostream& operator<<(std::ostream& os, const f16& v) {
  os << (float)v;
  return os;
}

#else
#if defined(__USE_CUTE_TYPE__)
using f16 = cute::half_t;
using half = cute::half_t;
#elif defined(__USE_CUDA_TYPE__)
using f16 = __half;
using half = __half;
#elif defined(__TOPSCC__)
using f16 = __fp16;
using half = __fp16;
#else
#error "half float is not supported on this target."
#endif
#endif // __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__

__co_any__ inline static f16 f32_to_f16(f32 value) {
  return __f32_to_f16<f16>(value);
}

__co_any__ inline static f32 f16_to_f32(f16 value) {
  return __f16_to_f32<f32>(value);
}

#ifndef __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
class bf16 {
private:
  uint16_t bits; // Storage for the half-precision bits

public:
  // Default constructor
  __co_any__ bf16() = default;

  // Constructor for conversion from float
  __co_any__ bf16(float value) { bits = floatToHalfBits(value); }

  // Constructor for conversion from double
  __co_any__ bf16(double value) {
    bits = floatToHalfBits(static_cast<float>(value));
  }

  // Implicit conversion from float
  __co_any__ bf16& operator=(float value) {
    bits = floatToHalfBits(value);
    return *this;
  }

  // Implicit conversion from double
  __co_any__ bf16& operator=(double value) {
    bits = floatToHalfBits(static_cast<float>(value));
    return *this;
  }

  __co_any__ bool operator==(double value) {
    auto valueF = static_cast<float>(value);
    if (std::isnan(valueF)) { return std::isnan(halfBitsToFloat(bits)); }
    return halfBitsToFloat(bits) == valueF;
  }

  template <typename T>
  __co_any__ bool operator==(T value) {
    if constexpr (std::is_same<T, bf16>::value) {
      auto valueF = (float)value;
      if (std::isnan(valueF)) { return std::isnan(halfBitsToFloat(bits)); }
      return halfBitsToFloat(bits) == valueF;
    } else {
      auto valueF = static_cast<float>(value);
      if (std::isnan(valueF)) { return std::isnan(halfBitsToFloat(bits)); }
      return halfBitsToFloat(bits) == valueF;
    }
  }

  template <typename T>
  __co_any__ bool operator>(T value) {
    if constexpr (std::is_same<T, bf16>::value) {
      auto valueF = (float)value;
      if (std::isnan(valueF)) { return std::isnan(halfBitsToFloat(bits)); }
      return halfBitsToFloat(bits) > valueF;
    } else {
      auto valueF = static_cast<float>(value);
      if (std::isnan(valueF)) { return std::isnan(halfBitsToFloat(bits)); }
      return halfBitsToFloat(bits) > valueF;
    }
  }

  template <typename T>
  __co_any__ bool operator<(T value) {
    if constexpr (std::is_same<T, bf16>::value) {
      auto valueF = (float)value;
      if (std::isnan(valueF)) { return std::isnan(halfBitsToFloat(bits)); }
      return halfBitsToFloat(bits) < valueF;
    } else {
      auto valueF = static_cast<float>(value);
      if (std::isnan(valueF)) { return std::isnan(halfBitsToFloat(bits)); }
      return halfBitsToFloat(bits) < valueF;
    }
  }

  // Function to convert float to half precision bits (naive and placeholder)
  __co_any__ static uint16_t floatToHalfBits(float value) {
    // Simplified conversion: this does not handle rounding, infinities, or NaNs
    // correctly In practice, use a library or a fully implemented conversion
    // function
    int32_t fltInt32 = *((int32_t*)&value);
    return (fltInt32 & 0xFFFF0000) >> 16;
  }

  // Function to convert half precision bits to float (naive and placeholder)
  __co_any__ static float halfBitsToFloat(uint16_t bits) {
    int32_t fltInt32 = ((uint32_t)bits) << 16;
    return *((float*)&fltInt32);
  }

  // Method to get the float value from the bf16 object
  __co_any__ operator float() const { return halfBitsToFloat(bits); }
};

using bfloat16 = bf16;
using bfp16 = bf16;

inline std::ostream& operator<<(std::ostream& os, const bf16& v) {
  os << (float)v;
  return os;
}

#else // __CHOREO_TARGET_NATIVE_BF16_SUPPORT__
#ifdef __CHOREO_TARGET_CUTE__
#ifdef __USE_CUTE_TYPE__
using __bf16 = cute::bfloat16_t;
#else
using __bf16 = __nv_bfloat16;
#endif
#endif
using bf16 = __bf16;
using bfp16 = __bf16;
using bfloat16 = __bf16;

// Check for __bf16 support
#if !defined(__TOPSCC__) && !defined(__clang__) && !defined(__GNUC__) &&       \
    !defined(__CUDACC__)
#error                                                                         \
    "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
#elif (defined(__clang__) && __clang_major__ < 11) ||                          \
    (defined(__GNUC__) && __GNUC__ < 11)
#error                                                                         \
    "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
#endif // defined...

#endif // __CHOREO_TARGET_NATIVE_BF16_SUPPORT__

#ifndef BF16_SUPPORTED
//#error \
//    "Compiler does not support __bf16. Please use a compiler that supports __bf16 or define a fallback type."
#endif

#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
#if defined(__USE_CUTE_TYPE__)
using cute::float_e4m3_t;
using cute::float_e5m2_t;
using cute::float_ue4m3_t;
using cute::float_ue8m0_t;
#elif defined(__USE_CUDA_TYPE__)
using float_e4m3_t = __nv_fp8_e4m3;
using float_e5m2_t = __nv_fp8_e5m2;
#ifdef __CHOREO_TARGET_NATIVE_FP8_E8M0_SUPPORT__
using float_ue8m0_t = __nv_fp8_e8m0;
#else
using float_ue8m0_t =
    choreo::co_native_base; // Placeholder for unsupported type
#endif
using float_ue4m3_t =
    choreo::co_native_base; // Placeholder for unsupported type
#elif defined(__TOPSCC__) || __GCU_ARCH__ >= 400
// TODO
#else
#error "FP8 E4M3 support requires CUTE Target."
#endif
using f8 = float_e4m3_t; // define f8 as float_e4m3_t
using f8_e4m3 = float_e4m3_t;
using f8_e5m2 = float_e5m2_t;
using f8_ue8m0 = float_ue8m0_t;
using f8_ue4m3 = float_ue4m3_t;
#endif // __CHOREO_TARGET_NATIVE_FP8_SUPPORT__

#ifdef __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
#if defined(__USE_CUTE_TYPE__)
using cute::float_e2m1_t;
#elif defined(__USE_CUDA_TYPE__)
using float_e2m1_t = __nv_fp4_e2m1;
#elif defined(__TOPSCC__) || __GCU_ARCH__ >= 400
// TODO
#else
#error "FP4 is not supported on this target."
#endif
using f4_e2m1 = float_e2m1_t;
#endif // __CHOREO_TARGET_NATIVE_FP4_SUPPORT__

#ifdef __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
#if defined(__USE_CUTE_TYPE__)
using cute::float_e2m3_t;
using cute::float_e3m2_t;
#elif defined(__USE_CUDA_TYPE__)
using float_e3m2_t = __nv_fp6_e3m2;
using float_e2m3_t = __nv_fp6_e2m3;
#elif defined(__TOPSCC__) || __GCU_ARCH__ >= 400
// TODO
#else
#error "FP6 is not supported on this target."
#endif
using f6_e3m2 = float_e3m2_t;
using f6_e2m3 = float_e2m3_t;
#endif // __CHOREO_TARGET_NATIVE_FP6_SUPPORT__

// Unsigned integer types
using u64 = uint64_t; // 64-bit unsigned integer
using u32 = uint32_t; // 32-bit unsigned integer
using u16 = uint16_t; // 16-bit unsigned integer
using u8 = uint8_t;   // 8-bit unsigned integer

// Signed integer types
using s64 = int64_t; // 64-bit signed integer
using s32 = int32_t; // 32-bit signed integer
using s16 = int16_t; // 16-bit signed integer
using s8 = int8_t;   // 8-bit signed integer

// Sub-Byte integer types
#ifdef __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
#if defined(__USE_CUDA_TYPE__) || defined(__USE_CUTE_TYPE__)
using cute::bin1_t;
using cute::int2b_t;
using cute::int4b_t;
using cute::int6b_t;
using cute::uint1b_t;
using cute::uint2b_t;
using cute::uint4b_t;
using cute::uint6b_t;
#else
#error "Sub-Byte integer types is not supported on this target."
#endif
using bin1 = bin1_t;
using s2 = int2b_t;
using s4 = int4b_t;
using s6 = int6b_t;
using u1 = uint1b_t;
using u2 = uint2b_t;
using u4 = uint4b_t;
using u6 = uint6b_t;
#endif // __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__

namespace utils {

// template <typename U>
// inline void fill_random(U*, size_t, U, U);

// specializations
// f64
template <typename U>
inline typename std::enable_if<std::is_same<U, double>::value, void>::type
fill_random(U* array, size_t N, U lb, U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<U> rand_func(lb, ub); // [lb, ub)

  std::generate_n(&array[0], N, [&]() { return rand_func(gen); });
}

// f32
template <typename U>
inline typename std::enable_if<std::is_same<U, float>::value, void>::type
fill_random(U* array, size_t N, U lb, U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<U> rand_func(lb, ub); // [lb, ub)

  std::generate_n(&array[0], N, [&]() { return rand_func(gen); });
}

// f16
template <typename U>
inline typename std::enable_if<std::is_same<U, f16>::value, void>::type
fill_random(U* array, size_t N, U lb, U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rand_func(
      static_cast<float>(lb),
      static_cast<float>(ub)); // [-1.0, 1.0)
  std::generate_n(&array[0], N, [&]() { return f16(rand_func(gen)); });
}

// bf16
template <typename U>
inline typename std::enable_if<std::is_same<U, bf16>::value, void>::type
fill_random(U* array, size_t N, U lb, U ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rand_func(
      static_cast<float>(lb),
      static_cast<float>(ub)); // [-1.0, 1.0)

  std::generate_n(&array[0], N, [&]() { return bf16(rand_func(gen)); });
}

// f16/bf16 with float lb/ub
template <typename U>
inline typename std::enable_if<
    std::is_same<U, f16>::value || std::is_same<U, bf16>::value, void>::type
fill_random(U* array, size_t N, float lb, float ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rand_func(lb, ub);

  std::generate_n(&array[0], N, [&]() { return U(rand_func(gen)); });
}

#ifdef __CHOREO_TARGET_NATIVE_FP8_SUPPORT__
// float_e4m3_t float_e5m2_t
template <typename U>
inline typename std::enable_if<std::is_same<U, float_e4m3_t>::value ||
                                   std::is_same<U, float_e5m2_t>::value,
                               void>::type
fill_random(U* array, size_t N, float lb, float ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rand_func(lb, ub); // [lb, ub)
  std::generate_n(&array[0], N, [&]() { return U(rand_func(gen)); });
}
#endif

#ifdef __CHOREO_TARGET_NATIVE_FP6_SUPPORT__
template <typename U>
inline typename std::enable_if<std::is_same<U, float_e3m2_t>::value ||
                                   std::is_same<U, float_e2m3_t>::value,
                               void>::type
fill_random(U* array, size_t N, float lb, float ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rand_func(lb, ub); // [lb, ub)

  std::generate_n(&array[0], N, [&]() { return U(rand_func(gen)); });
}
#endif

#ifdef __CHOREO_TARGET_NATIVE_FP4_SUPPORT__
template <typename U>
inline typename std::enable_if<std::is_same<U, float_e2m1_t>::value, void>::type
fill_random(U* array, size_t N, float lb, float ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rand_func(lb, ub); // [lb, ub)

  std::generate_n(&array[0], N, [&]() { return U(rand_func(gen)); });
}
#endif

#ifdef __CHOREO_TARGET_NATIVE_SUB_BYTE_INTEGRAL_SUPPORT__
// tiny integer types
template <typename U>
inline typename std::enable_if<
    std::is_same<U, uint4b_t>::value || std::is_same<U, uint6b_t>::value ||
        std::is_same<U, uint2b_t>::value || std::is_same<U, uint1b_t>::value ||
        std::is_same<U, int6b_t>::value || std::is_same<U, int4b_t>::value ||
        std::is_same<U, int2b_t>::value || std::is_same<U, bin1_t>::value,
    void>::type
fill_random(U* array, size_t N, int lb, int ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> rand_func(lb, ub); // [lb, ub)

  std::generate_n(&array[0], N,
                  [&]() { return static_cast<U>(rand_func(gen)); });
}
#endif

// s32/u32 ...
// if T is integer, utilize std::uniform_int_distribution
template <typename U>
inline typename std::enable_if<std::is_integral<U>::value, void>::type
fill_random(U* array, size_t N, int lb, int ub) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> rand_func(lb, ub); // [lb, ub]

  std::generate_n(&array[0], N, [&]() { return U(rand_func(gen)); });
}
} // end namespace utils

// A 'spanned_view' is a memview of data. It is ranked, but no necessary to have
// compile-time dimensions
template <typename T, size_t Rank>
class spanned_view {
  static_assert(Rank != 0, "unexpected 0-dims.");
  T* ptr = nullptr;
  const mdspan<Rank> dims;

public:
  __co_any__ explicit spanned_view(T* d, const mdspan<Rank>& s)
      : ptr(d), dims(s) {}

  constexpr size_t rank() const { return Rank; }
  __co_any__ const mdspan<Rank>& shape() const { return dims; }

  __co_any__ size_t element_count() const { return span_size(dims); }
  __co_any__ size_t bytes() const { return element_count() * sizeof(T); }
  __co_any__ T* data() { return ptr; }
  __co_any__ T* data() const { return ptr; }

  // allow multi-dim-style access, be like: a[1][3]
  template <size_t M = Rank>
  typename std::enable_if<(M == 1),
                          T&>::type // make sure to return the reference type
      __co_any__
      operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < dims[0], "Index out of bounds", __FILE__,
                  __LINE__);
    return ptr[index];
  }

  template <size_t M = Rank>
  typename std::enable_if<(M > 1), ArrayProxy<T, Rank - 1>>::type __co_any__
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < dims[0], "Index out of bounds", __FILE__,
                  __LINE__);
    const auto& sub_dims =
        *reinterpret_cast<const mdspan<Rank - 1>*>(&(dims[1]));
    return ArrayProxy<T, Rank - 1>(ptr, sub_dims, (size_t)index * dims[1]);
  }

  __co_any__ friend bool operator==(const spanned_view& l,
                                    const spanned_view& r) {
    if (l.dims != r.dims) return false;

    for (size_t i = 0; i < l.element_count(); ++i)
      if (l.ptr[i] != r.ptr[i]) return false;

    return true;
  }

  template <typename U>
  __co_any__ void fill(U value) {
    fill_n(this->data(), this->element_count(), static_cast<T>(value));
  }

  template <typename U>
  __co_host__ void fill_random(U lb, U ub) {
    utils::fill_random(this->data(), this->element_count(), static_cast<T>(lb),
                       static_cast<T>(ub));
  }
};

template <typename T>
using spanned_data_deleter_t = void (*)(T*);

template <typename T>
using spanned_data_unique_ptr = std::unique_ptr<T, spanned_data_deleter_t<T>>;

// A 'spanned_data' is similar to 'spanned_view' but manage memory
template <typename T, size_t Rank>
class spanned_data {
public:
  using unique_ptr_t = spanned_data_unique_ptr<T>;

private:
  unique_ptr_t ptr = nullptr; // this is used as the output
  mdspan<Rank> dims;

public:
  explicit spanned_data(unique_ptr_t&& raw, const mdspan<Rank>& s)
      : ptr(std::move(raw)), dims(s) {}

  spanned_data(const spanned_data&) = delete; // move only
  spanned_data& operator=(const spanned_data&) = delete;

  spanned_data(spanned_data&& sd) : ptr(std::move(sd.ptr)), dims(sd.dims) {}

  constexpr size_t rank() const { return Rank; }
  const mdspan<Rank>& shape() const { return dims; }

  size_t element_count() const { return span_size(dims); }
  size_t bytes() const { return element_count() * sizeof(T); }
  T* data() { return ptr.get(); }

  // allow multi-dim-style access, be like: a[1][3]
  template <size_t M = Rank>
  typename std::enable_if<(M == 1),
                          T&>::type // make sure to return the reference type
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < dims[0], "Index out of bounds", __FILE__,
                  __LINE__);
    return *(data() + index);
  }

  template <size_t M = Rank>
  typename std::enable_if<(M > 1), ArrayProxy<T, Rank - 1>>::type
  operator[](int index) {
    choreo_assert(index >= 0, "Index out of bounds", __FILE__, __LINE__);
    choreo_assert((size_t)index < dims[0], "Index out of bounds", __FILE__,
                  __LINE__);
    const auto& sub_dims =
        *reinterpret_cast<const mdspan<Rank - 1>*>(&(dims[1]));
    return ArrayProxy<T, Rank - 1>(ptr.get(), sub_dims,
                                   (size_t)index * dims[1]);
  }

  friend bool operator==(const spanned_data& l, const spanned_data& r) {
    if (l.dims != r.dims) return false;

    for (size_t i = 0; i < l.element_count(); ++i)
      if (l.ptr[i] != r.ptr[i]) return false;

    return true;
  }
  template <typename U>
  __co_any__ void fill(U value) {
    fill_n(this->data(), this->element_count(), static_cast<T>(value));
  }

  template <typename U>
  __co_host__ void fill_random(U lb, U ub) {
    utils::fill_random(this->data(), this->element_count(), lb, ub);
  }

  __co_host__ spanned_view<T, Rank> view() {
    return spanned_view<T, Rank>(data(), dims);
  }
};

template <size_t Rank>
__co_any__ mdspan<Rank> make_mdspan(const std::initializer_list<size_t>& init) {
  return mdspan<Rank>(init);
}

// note: spanned_view does not invoke copy. Instead, it associates data with a
// multi-dimension view of memory
template <size_t Rank, typename T>
__co_any__ spanned_view<T, Rank>
make_spanview(T* ptr, std::initializer_list<size_t> init) {
  return spanned_view<T, Rank>(ptr, make_mdspan<Rank>(init));
}

template <typename T, size_t N>
__co_any__ spanned_view<T, 1> make_spanview(T (&arr)[N]) {
  return spanned_view<T, 1>((T*)arr, {N});
}

template <typename T, size_t N, size_t M>
__co_any__ spanned_view<T, 2> make_spanview(T (&arr)[N][M]) {
  return spanned_view<T, 2>((T*)arr, {N, M});
}

template <typename T, size_t Rank>
__co_host__ spanned_data<T, Rank>
make_spandata(std::initializer_list<size_t> init) {
  size_t element_count = 1;
  for (auto& value : init) element_count *= value;
  choreo_assert(element_count > 0, "error: invalid dimensions.", __FILE__,
                __LINE__);

  T* raw_ptr = nullptr;
#ifdef __TOPSCC__
  // host memory optimization
  runtime_check(!topsHostMalloc(&raw_ptr, element_count * sizeof(T)),
                "[choreo-rt] failed to allocate memory.");
  auto del = [](T* p) {
    runtime_check(!topsHostFree(p), "[choreo-rt] failed to free memory.");
  };
#else
  raw_ptr = new T[element_count];
  auto del = [](T* p) { delete[] p; };
#endif
  spanned_data_unique_ptr<T> ptr(raw_ptr, del);
  return spanned_data<T, Rank>(std::move(ptr), make_mdspan<Rank>(init));
}

// alternative interface
template <typename T, typename... Dims,
          typename = typename std::enable_if<
              (std::is_convertible<Dims, size_t>::value && ...)>::type>
__co_host__ auto make_spandata(Dims... dims) {
  constexpr size_t Rank = sizeof...(Dims);
  return make_spandata<T, Rank>({static_cast<size_t>(dims)...});
}

// converting from vector to another type
template <size_t Rank, typename T>
auto copy_as_spanned(T* ptr, std::initializer_list<size_t> init) {
  size_t element_count = 1;
  for (auto& value : init) element_count *= value;
  choreo_assert(element_count > 0, "error: invalid dimensions.", __FILE__,
                __LINE__);

  auto parr = new T[element_count];
  std::copy(ptr, ptr + element_count, parr);
  auto del = [](T* p) { delete[] p; };
  spanned_data_unique_ptr<T> uptr((T*)parr, del);
  auto res = spanned_data<T, Rank>(std::move(uptr), make_mdspan<Rank>(init));
  choreo_assert(res.bytes() == element_count * sizeof(T),
                "error: element_count does not match.", __FILE__, __LINE__);
  return res;
}

template <size_t Rank, typename T>
auto copy_as_spanned(T* ptr, const mdspan<Rank> dims) {
  size_t element_count = span_size(dims);
  auto parr = new T[element_count];
  std::copy(ptr, ptr + element_count, parr);
  auto del = [](T* p) { delete[] p; };
  spanned_data_unique_ptr<T> uptr((T*)parr, del);
  auto res = spanned_data<T, Rank>(std::move(uptr), dims);
  choreo_assert(res.bytes() == element_count * sizeof(T),
                "error: element_count does not match.", __FILE__, __LINE__);
  return res;
}

struct HeapSimulator {
  using Range = std::pair<size_t, size_t>;
  struct Buffer {
    size_t size;
    std::vector<Range> ranges;
    std::string buffer_id;
    bool Interfere(const Buffer& other) const {
      for (const auto& [as, ae] : this->ranges)
        for (const auto& [bs, be] : other.ranges)
          if (as <= be && bs <= ae) return true;
      return false;
    }
  };
  using Chunk = Buffer;
  using Chunks = std::vector<Chunk>;

  // memory allocation result
  struct Result {
    std::map<std::string, size_t> chunk_offsets; // offset of each buffer
    size_t heap_size;                            // total memory size
  };

  // global decreasing size best fit allocate algorithm
  // (support arbitrary alignment)
  Result GlobalDecreasingSizeBestFitAllocate(const std::vector<Chunk>& chunks,
                                             size_t alignment = 0) {
    Result result;
    result.heap_size = 0;

    size_t length = chunks.size();

    auto AlignUp = [alignment](size_t x) -> size_t {
      if (alignment == 0) return x;
      return (x + alignment - 1) / alignment * alignment;
    };

    // sort by size in descending order
    // TODO: use idx or pointer rather than Chunk
    std::vector<Chunk> sorted_chunks = chunks;
    std::sort(sorted_chunks.begin(), sorted_chunks.end(),
              [](const Chunk& a, const Chunk& b) { return a.size > b.size; });

    // build interference graph - represent which buffers' lifetime overlap
    // TODO: O(n^2) maybe can be optimized
    std::vector<std::vector<bool>> interference_graph(
        length, std::vector<bool>(length, false));

    for (size_t i = 0; i < length; ++i)
      for (size_t j = i + 1; j < length; ++j)
        if (sorted_chunks[i].Interfere(sorted_chunks[j])) {
          interference_graph[i][j] = true;
          interference_graph[j][i] = true;
        }

    // assign space for each buffer
    std::map<size_t, size_t> assigned_offsets;

    using Range = std::pair<size_t, size_t>;

    for (size_t i = 0; i < length; ++i) {
      const Chunk& chunk = sorted_chunks[i];

      // collect the allocated regions that overlap with the current buffer
      std::vector<Range> forbidden_ranges;
      for (size_t j = 0; j < i; ++j) {
        if (interference_graph[i][j] && assigned_offsets.count(j)) {
          // the current buffer and the buffer in j-th position overlap in
          // lifetime, so they can't be allocated to the same position
          forbidden_ranges.push_back(
              {assigned_offsets[j],
               assigned_offsets[j] + sorted_chunks[j].size});
        }
      }

      // sort the forbidden ranges by the start position
      std::sort(forbidden_ranges.begin(), forbidden_ranges.end());

      // merge the overlapping forbidden ranges
      if (!forbidden_ranges.empty()) {
        std::vector<Range> merged_ranges;
        merged_ranges.push_back(forbidden_ranges[0]);

        for (size_t j = 1; j < forbidden_ranges.size(); ++j) {
          auto& last = merged_ranges.back();
          const auto& current = forbidden_ranges[j];

          if (current.first <= last.second)
            last.second = std::max(last.second, current.second);
          else
            merged_ranges.push_back(current);
        }

        forbidden_ranges = std::move(merged_ranges);
      }

      // find the first valid position that satisfies the alignment
      // requirement
      size_t pos = 0;
      pos = AlignUp(pos);

      bool found_valid_position = false;
      for (size_t j = 0; j <= forbidden_ranges.size(); ++j) {
        // check if the current position is valid
        if (j == forbidden_ranges.size() ||
            pos + chunk.size <= forbidden_ranges[j].first) {
          found_valid_position = true;
          break;
        }

        // update the position to the current forbidden range
        pos = forbidden_ranges[j].second;
        // ensure the new position satisfies the alignment requirement
        pos = AlignUp(pos);
      }

      if (!found_valid_position) {
        // this should not happen in normal cases, because we always can find
        // a position after all forbidden ranges but just in case, we should
        // handle this situation
        std::cerr << "Error: Could not find valid position for buffer "
                  << chunk.buffer_id << std::endl;
        // indicate allocation failed
        result.chunk_offsets[chunk.buffer_id] = (size_t)-1;
        continue;
      }

      // assign the aligned offset to the current buffer
      size_t aligned_offset = pos;
      assigned_offsets.emplace(i, aligned_offset);

      // update the result
      result.chunk_offsets[chunk.buffer_id] = aligned_offset;
      result.heap_size =
          std::max(result.heap_size, aligned_offset + chunk.size);
    }

    // ensure the final heap size also satisfies the alignment requirement
    result.heap_size = AlignUp(result.heap_size);

    return result;
  }

  Result Allocate(const std::vector<Chunk>& chunks, int64_t alignment = 0) {
    return GlobalDecreasingSizeBestFitAllocate(chunks, alignment);
  }
};

// For API check: abend on failures
static __attribute__((always_inline)) inline void abend_false(bool p) {
  if (!p) std::abort();
}

static __attribute__((always_inline)) inline void abend_true(bool p) {
  if (p) std::abort();
}

static __attribute__((always_inline)) inline void verify_device_status() {
#ifdef __CUDA__
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error after kernel: %s\n", cudaGetErrorString(err));
    std::abort();
  }
#endif
}

// target specific definations
#ifdef __TOPSCC__
template <typename T>
__device__ static int inline __addr2int__(T* v) {
  return static_cast<int>(reinterpret_cast<long long>(v));
}
#else
template <typename T>
static int inline __addr2int__(T* v) {
  return static_cast<int>(reinterpret_cast<std::uintptr_t>(v));
}
#endif

#ifdef __TOPSCC__

} // end namespace choreo

#include <krt/builtins.h>

namespace choreo {

// --- light-weight choreo-topscc device library --- //

__device__ __attribute__((always_inline)) static inline void __co_abort__() {
#ifdef __CHOREO_USE_TOPS_ABORT__
  tops::abort();
#else
  abort();
#endif
}

#endif // __TOPSCC__

#ifdef __TOPSCC__

#if __GCU_ARCH__ == 400
using choreo_dte_ctx_t = tops_dte_ctx_base_s;
__device__ __forceinline__ void tops_init_dte(tops_dte_ctx_base_s* ctx) {
  ctx->init_comm();
}
__device__ __forceinline__ void tops_destroy_dte(tops_dte_ctx_base_s* ctx) {
  ctx->destroy_comm();
}
#else
using choreo_dte_ctx_t = tops_dte_ctx_t;
#endif

// choreo device future
struct future {
  choreo_dte_ctx_t* ctx = nullptr;
  tops::event e;
  void* d = nullptr; // data: future's user must guarantee it is valid

  // for runtime check purpose
  //
  // ST_NONE -> ST_INITED -> ST_TRIGGERED -> ST_WAITED
  //                              ^              |
  //                              +--------------+
  enum Status {
    ST_NONE = 0,
    ST_INITED = 1,
    ST_TRIGGERED = 2,
    ST_WAITED = 3,
  };
  Status s = ST_NONE;
  const char* name = nullptr;
  // source code locations
  unsigned line = 0;
  unsigned column = 0;

  __device__ future(choreo_dte_ctx_t& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr)
      : ctx(&dte), d(data), s(ST_NONE), name(n), line(l), column(c) {}
#if __GCU_ARCH__ == 400
  __device__ future(tops::local_dte& dte, const char* n, unsigned l, unsigned c,
                    void* data = nullptr)
      : ctx(&dte), d(data), s(ST_NONE), name(n), line(l), column(c) {}
  __device__ future(tops::shared_dte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr)
      : ctx(&dte), d(data), s(ST_NONE), name(n), line(l), column(c) {}
  __device__ future(tops::private_dte& dte, const char* n, unsigned l,
                    unsigned c, void* data = nullptr)
      : ctx(&dte), d(data), s(ST_NONE), name(n), line(l), column(c) {}
#endif

  // context is retrieved to invoke data operations
  __device__ auto get_ctx() {
    if (s == ST_NONE) {
      tops_init_dte(ctx);
      s = ST_INITED;
    }
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is not initialized.\n",
             line, column);
      __co_abort__();
    }
    return ctx;
  }

  // when async, an event is obtained for later waiting
  __device__ void set_event(tops::event& ev) {
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "is triggered on an in-flight event.\n",
             line, column);
      __co_abort__();
    } else if (s == ST_NONE) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is not initialized before triggering.\n",
             line, column);
      __co_abort__();
    }
    if (ev.ctx != ctx) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used inconsistently.\n",
             line, column);
      __co_abort__();
    }

    e = ev;
    s = ST_TRIGGERED;
  }

  // when sync, no wait is required. simply change the status
  __device__ void set_nowait() {
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    }
    s = ST_WAITED;
  }

  __device__ void set_data(void* data) { d = data; }
  __device__ void set_event_data(tops::event& ev, void* data) {
    set_event(ev);
    set_data(data);
  }

  __device__ void wait() {
    if (s == ST_TRIGGERED) {
      tops::wait(e);
      s = ST_WAITED;
    } else if (s == ST_WAITED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has been waited multiple times.\n",
             line, column);
      __co_abort__();
    } else if (s == ST_INITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    } else
      assert(s == ST_NONE); // waiting on not triggered future is acceptable
  }

#if 0
  __device__ tops::event& event() {
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] internal error: future (defined at line %u:%u) is not associated with an event.\n",
             line, column);
      __co_abort__();
    }
    return e;
  }
#endif

  __device__ void* data() {
    if (!d) {
      printf("[choreo-rt] internal error: future (defined at line %u:%u) is "
             "not associated with a data.\n",
             line, column);
      __co_abort__();
    }
    if (s == ST_TRIGGERED) {
      // TODO: requires krt %s support to print future name
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) is "
             "not waited before using.\n",
             line, column);
      __co_abort__();
    }
    return d;
  }

  __device__ ~future() {
    if (s == ST_TRIGGERED) {
      // TODO: requires krt %s support to print future name
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has never been waited.\n",
             line, column);
      __co_abort__();
    }
    if (s >= ST_INITED) tops_destroy_dte(ctx);
  }
  __device__ future(const future& f) = delete;
  __device__ future(future&& f) = delete;
  __device__ future& operator=(const future& f) = delete;
};

__device__ static inline void swap(future& a, future& b) {
  auto ctx = a.ctx;
  auto e = a.e;
  auto d = a.d;
  auto s = a.s;
  auto l = a.line;
  auto c = a.column;

  a.ctx = b.ctx;
  a.e = b.e;
  a.d = b.d;
  a.s = b.s;
  a.line = b.line;
  a.column = b.column;

  b.ctx = ctx;
  b.e = e;
  b.d = d;
  b.s = s;
  b.line = l;
  b.column = c;
}

#endif

#ifdef __CHOREO_TARGET_CUTE__

// SM90+ (Hopper+) - TMA barrier and token
struct TMAAtom {
  cuda::barrier<cuda::thread_scope_block>* bar;
  cuda::barrier<cuda::thread_scope_block>::arrival_token tok;
  //  TMAAtom(cuda::barrier<cuda::thread_scope_block> *b): bar(b) {}
  __device__ auto& barrier() { return *bar; }
  __device__ auto& token() { return tok; }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
using TMALoadAtom = cute::SM90_TMA_LOAD;
using TMAStoreAtom = cute::SM90_TMA_STORE;

#endif

using AsyncCopyAtom = cute::AutoCopyAsync;

__device__ __attribute__((always_inline)) static inline void __co_abort__() {
  __trap();
}

// this facilitate the wait-N implementation for sm_80+
// it is must be warp-wise since async copy is warp-wise.
// a ring with 6 elements is 8-bytes. Therefore it may cost up-to 256-bytes
// (32-warps) shared memory
struct future;
template <int N>
struct future_ring {
  static_assert(N < UCHAR_MAX, "ring size is too large.");
  int8_t ring[N];

  uint8_t head = 0; // lastest commit
  uint8_t tail = 0; // oldest commit

  __device__ void commit(future*);
  __device__ int discard(future*);
  __device__ void init() {
    head = 0;
    tail = 0;
  }
};

using AtomType = void; // erase the type

// choreo device future
struct future {

  AtomType* atom = nullptr;
  void* d = nullptr; // data: future's user must guarantee it is valid

  bool is_tma = false;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  future_ring<6>* ring;
  int8_t id;
#else
  // make host compilation happy
  future_ring<6>* ring;
  int8_t id;
#endif
  __device__ void set_ring(future_ring<6>* r) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (__CHOREO_GROUP_SINGLE__(32)) {
      if (!r) return;
      ring = r + (threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y) /
                     32;
    }
#else
// make host compilation happy
#endif
  }
  // for runtime check purpose
  //
  // ST_NONE -> ST_INITED -> ST_TRIGGERED -> ST_WAITED
  //                              ^              |
  //                              +--------------+
  enum Status {
    ST_NONE = 0,
    ST_INITED = 1,
    ST_TRIGGERED = 2,
    ST_WAITED = 3,
  };

#ifdef __CHOREO_DMA_DIAGNOSIS__
  Status s = ST_NONE;
  const char* name = nullptr;
  // source code locations
  unsigned line = 0;
  unsigned column = 0;

  __device__ future(const char* n, unsigned l, unsigned c, void* data = nullptr)
      : d(data), s(ST_NONE), name(n), line(l), column(c) {}
#else
  __device__ future(void* data = nullptr) : d(data) {}
#endif //__CHOREO_DMA_DIAGNOSIS__

  // context is retrieved to invoke data operations
  __device__ auto get_atom() {
#ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_NONE) s = ST_INITED;
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is not initialized.\n",
             line, column);
      __co_abort__();
    }
#endif // __CHOREO_DMA_DIAGNOSIS__
    return atom;
  }

  // when async, an event is obtained for later waiting
  __device__ void set_atom(AtomType* a) {
#ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "is triggered on an in-flight event.\n",
             line, column);
      __co_abort__();
    } else if (s != ST_NONE) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "has been initialized before setting atom.\n",
             line, column);
      __co_abort__();
    }
#endif // __CHOREO_DMA_DIAGNOSIS__

    atom = a;
    s = ST_INITED;
  }

  // when sync, no wait is required. simply change the status
  __device__ void set_nowait() {
#ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    }
    s = ST_WAITED;
#endif // __CHOREO_DMA_DIAGNOSIS__
  }

  __device__ void set_data(void* data) { d = data; }
  __device__ void set_atom_data(AtomType* a, void* data) {
    set_atom(a);
    set_data(data);
  }

  __device__ void wait_impl() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (is_tma) {
      auto& barrier = ((TMAAtom*)atom)->barrier();
      auto& token = ((TMAAtom*)atom)->token();
      barrier.wait(std::move(token));
      return;
    }
    // cautious: must be warp based
    if (__CHOREO_GROUP_SINGLE__(32)) {
      assert(ring && "ring is invalid.");
      int discard_count = ring->discard(this);
      switch (discard_count) {
      case -1: break;
      case 1: cute::cp_async_wait<1>(); break;
      case 2: cute::cp_async_wait<2>(); break;
      case 3: cute::cp_async_wait<3>(); break;
      case 4: cute::cp_async_wait<4>(); break;
      case 5: cute::cp_async_wait<5>(); break;
      default:
#ifdef __CHOREO_DMA_DIAGNOSIS__
        printf("[choreo-rt] Unable to wait the %d futures (current defined at "
               "line %u:%u).\n",
               discard_count, line, column);
#else
        printf("[choreo-rt] Unable to wait the %d futures.\n", discard_count);
#endif // DIAGNOSIS
        __co_abort__();
        break;
      }
    }
#else
// cuda host compilation
#endif
  }

  __device__ void trigger() {
#ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s != ST_INITED && s != ST_WAITED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has been triggered without atom set.\n",
             line, column);
      __co_abort__();
    }
#endif // __CHOREO_DMA_DIAGNOSIS__
    s = ST_TRIGGERED;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // cautious: must be warp based
    if (__CHOREO_GROUP_SINGLE__(32)) {
      assert(ring && "ring is invalid.");
      ring->commit(this);
    }
#else
// cuda host compilation
#endif
  }

  __device__ void wait() {
#ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED) {
      s = ST_WAITED;
    } else if (s == ST_WAITED) {
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has been waited multiple times.\n",
             line, column);
      __co_abort__();
    } else if (s == ST_INITED) {
      printf("[choreo-rt] Internal error: future (defined at line %u:%u) "
             "is used incorrectly.\n",
             line, column);
      __co_abort__();
    } else
      assert(s == ST_NONE); // waiting on not triggered future is acceptable
#endif                      // __CHOREO_DMA_DIAGNOSIS__
    wait_impl();
  }

  __device__ void* data() {
#ifdef __CHOREO_DMA_DIAGNOSIS__
    if (!d) {
      printf("[choreo-rt] internal error: future (defined at line %u:%u) is "
             "not associated with a data.\n",
             line, column);
      __co_abort__();
    }
    if (s == ST_TRIGGERED) {
      // TODO: requires krt %s support to print future name
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) is "
             "not waited before using.\n",
             line, column);
      __co_abort__();
    }
#endif // __CHOREO_DMA_DIAGNOSIS__
    return d;
  }

  __device__ void destroy() {}

  __device__ ~future() {
#ifdef __CHOREO_DMA_DIAGNOSIS__
    if (s == ST_TRIGGERED) {
      // TODO: requires krt %s support to print future name
      printf("[choreo-rt] Error is detected: future (defined at line %u:%u) "
             "has never been waited.\n",
             line, column);
      __co_abort__();
    }
    if (s >= ST_INITED) destroy();
#else
    destroy();
#endif // __CHOREO_DMA_DIAGNOSIS__
  }
  __device__ future(const future& f) = delete;
  __device__ future(future&& f) = delete;
  __device__ future& operator=(const future& f) = delete;
};

template <int N>
inline __device__ void future_ring<N>::commit(future* f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // the uniqueness of id is guaranteed by choreo
  ring[head] = f->id;
  head = (head + 1) % N;

#ifdef __CHOREO_DEBUG_FUTURE_RING__
  printf("committed feature: %d, [%d, %d)\n", f->id, tail, head);
#endif

#else
// cuda host compilation
#endif // CUDA_ARCH
}

template <int N>
inline __device__ int future_ring<N>::discard(future* f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

#ifdef __CHOREO_DEBUG_FUTURE_RING__
  printf("discarding feature: %d, [%d, %d)\n", f->id, tail, head);
#endif

  uint8_t p = tail;
  while (p != head) {
    if (ring[p] == f->id) {
      int size = (p + 1 + N - tail) % N;
      tail = (p + 1) % N;
      return size;
    } else
      p = (p + 1) % N;
  }

  // the ring is now empty
  if (tail == head) p = (p + 1) % N;

  while (p != tail) {
    // has been discarded already
    if (ring[p] == f->id)
      return -1;
    else
      p = (p + 1) % N;
  }

  printf("[choreo-rt] Internal error: future %d (defined at line %u:%u) "
         "is not committed.\n",
         f->id, f->line, f->column);

  //  __co_abort__();
#else
// cuda host compilation
#endif // CUDA_ARCH
  return -1;
}

__device__ static inline void swap(future& a, future& b) {
  auto atom = a.atom;
  auto d = a.d;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  future_ring<6>* ring = a.ring;
  int8_t id = a.id;
#else
#endif

#ifdef __CHOREO_DMA_DIAGNOSIS__
  auto name = a.name;
  auto s = a.s;
  auto l = a.line;
  auto c = a.column;
#endif

  a.atom = b.atom;
  a.d = b.d;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  a.ring = b.ring;
  a.id = b.id;
#else
#endif

#ifdef __CHOREO_DMA_DIAGNOSIS__
  a.name = b.name;
  a.s = b.s;
  a.line = b.line;
  a.column = b.column;
#endif

  b.atom = atom;
  b.d = d;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  b.ring = ring;
  b.id = id;
#else
#endif

#ifdef __CHOREO_DMA_DIAGNOSIS__
  b.name = name;
  b.s = s;
  b.line = l;
  b.column = c;
#endif
}

// ------------------- C++17 utilities -------------------
template <class...>
using void_t = void;

template <class T, class = void>
struct has_value : std::false_type {};
template <class T>
struct has_value<T, void_t<decltype(T::value)>> : std::true_type {};

// std::conjunction for C++14/17
template <class...>
struct conj : std::true_type {};
template <class B1>
struct conj<B1> : B1 {};
template <class B1, class... Bn>
struct conj<B1, Bn...> : std::conditional<B1::value, conj<Bn...>, B1>::type {};

// ------------------- compile-time stride checks -------------------
// One-stride divisibility check (only meaningful if stride is static
// cute::C<...>)
template <int Nelems, class StrideT>
struct stride_ok_ct
    : std::bool_constant<has_value<StrideT>::value &&
                         ((int(StrideT::value) % Nelems == 0) ||
                          (Nelems % int(StrideT::value) == 0))> {};

// Fold over stride tuple at indices I...
template <int Nelems, class Strides, std::size_t... I>
struct all_strides_ok_ct_impl {
  using type = conj<stride_ok_ct<
      Nelems,
      typename std::remove_cv<typename std::remove_reference<
          decltype(std::get<I>(std::declval<Strides>()))>::type>::type>...>;
  static constexpr bool value = type::value;
};

// Are ALL strides compile-time constants? (no divisibility yet)
template <class Strides, std::size_t... I>
struct all_strides_are_static_impl {
  using type =
      conj<has_value<typename std::remove_cv<typename std::remove_reference<
          decltype(std::get<I>(std::declval<Strides>()))>::type>::type>...>;
  static constexpr bool value = type::value;
};

// Trait: for a given Src tensor, can we prove Bits-wide vector is OK at compile
// time?
template <int Bits, class Src>
struct layout_vec_ok_ct {
  using E = typename Src::value_type;
  static constexpr int Nelems = Bits / (8 * int(sizeof(E)));
  using Strides = decltype(std::declval<Src>().layout().stride());
  static constexpr std::size_t R =
      decltype(size(std::declval<Strides>()))::value;

  static constexpr bool value =
      all_strides_ok_ct_impl<Nelems, Strides,
                             std::make_index_sequence<R>{}>::value;
};

// Trait: are ALL strides of Src compile-time constants?
template <class Src>
struct all_strides_are_static {
  using Strides = decltype(std::declval<Src>().layout().stride());
  static constexpr std::size_t R =
      decltype(size(std::declval<Strides>()))::value;

  static constexpr bool value =
      all_strides_are_static_impl<Strides,
                                  std::make_index_sequence<R>{}>::value;
};

// ------------------- runtime pointer alignment (bytes) -------------------
template <int Bits, class Ptr>
CUTE_HOST_DEVICE bool aligned_at_least(Ptr p) {
  constexpr std::uintptr_t A = Bits / 8;
  return (reinterpret_cast<std::uintptr_t>(p) % A) == 0;
}

// ------------------- universal copy (any rank, C++17) -------------------
template <class Src, class Dst>
CUTE_HOST_DEVICE void opt_copy(const Src& src, Dst& dst) {
#if 0
  // adjust these two lines if your Engine exposes pointers differently
  auto src_ptr = std::get<0>(src.data());
  auto dst_ptr = std::get<0>(dst.data());

  // If all strides are static, we can safely *attempt* wide vectors:
  if constexpr (all_strides_are_static<Src>::value) {
    // 128-bit (e.g., 4x int/float) if strides OK at compile time AND pointers aligned at runtime
    if constexpr (layout_vec_ok_ct<128, Src>::value) {
      if (aligned_at_least<128>(src_ptr) && aligned_at_least<128>(dst_ptr)) {
static_assert(false, "path 1\n");
        copy(cute::AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst);
        return;
      }
    }
    // 64-bit next
    if constexpr (layout_vec_ok_ct<64, Src>::value) {
static_assert(false, "path 2\n");
      if (aligned_at_least<64>(src_ptr) && aligned_at_least<64>(dst_ptr)) {
        copy(cute::AutoVectorizingCopyWithAssumedAlignment<64>{}, src, dst);
        return;
      }
    }
  }
#endif

  // Fallback: 32-bit (scalar element width) - always safe for any
  // shape/stride/alignment
  copy(cute::AutoVectorizingCopyWithAssumedAlignment<32>{}, src, dst);
}

// TODO: move to choreo_mma_wrapper.h
// ------------------- inline mma PTX -------------------
template <class TensorA, class TensorB>
inline __device__ void mma_sync_aligned_m8n8k4_row_col_f64_f64_f64_f64(
    double& d0, double& d1, TensorA const& A, TensorB const& B,
    const double& c0, const double& c1) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int A_row = lane >> 2;
  int A_col = lane % 4;
  int B_row = lane % 4;
  int B_col = lane >> 2;
  // TODO: load matrix
  double A_frag = A(A_row, A_col);
  double B_frag = B(B_row, B_col);
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
               "{%0, %1}, "
               "{%2}, "
               "{%3}, "
               "{%4, %5};\n"
               : "=d"(d0), "=d"(d1)
               : "d"(A_frag), "d"(B_frag), "d"(c0), "d"(c1));
}

template <class TensorD>
inline __device__ void
mma_sync_aligned_m8n8k4_row_col_f64_f64_f64_f64_store(double& d0, double& d1,
                                                      TensorD const& D) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int D_row = lane >> 2;
  int D_col0 = (lane % 4) * 2;
  int D_col1 = D_col0 + 1;
  D(D_row, D_col0) = d0;
  D(D_row, D_col1) = d1;
}

template <class TensorA, class TensorB>
inline __device__ void mma_sync_aligned_m8n8k4_row_col_f16_f16_f16_f16(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, TensorA const& A,
    TensorB const& B, const uint32_t& c0, const uint32_t& c1,
    const uint32_t& c2, const uint32_t& c3) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int A_row;
  if (lane < 16)
    A_row = lane % 4;
  else
    A_row = lane % 4 + 4;
  auto A_u32 = cute::recast<uint32_t>(A);
  uint32_t a0 = A_u32(A_row, 0);
  uint32_t a1 = A_u32(A_row, 1);
  int B_col;
  if (lane < 16)
    B_col = lane % 4;
  else
    B_col = lane % 4 + 4;
  uint32_t b0 = (uint32_t(reinterpret_cast<uint16_t&>(B(0, B_col))) << 16) |
                uint16_t(reinterpret_cast<uint16_t&>(B(1, B_col)));
  uint32_t b1 = (uint32_t(reinterpret_cast<uint16_t&>(B(2, B_col))) << 16) |
                uint16_t(reinterpret_cast<uint16_t&>(B(3, B_col)));
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
               "{%0, %1,  %2,  %3},"
               "{%4, %5},"
               "{%6, %7},"
               "{%8, %9, %10, %11};\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
                 "r"(c3));
}

template <class TensorD>
inline __device__ void mma_sync_aligned_m8n8k4_row_col_f16_f16_f16_f16_store(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, TensorD const& D) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int D_row;
  if (lane < 16)
    D_row = lane % 4;
  else
    D_row = lane % 4 + 4;
  auto D_u32 = cute::recast<uint32_t>(D);
  D_u32(D_row, 0) = d0;
  D_u32(D_row, 1) = d1;
  D_u32(D_row, 2) = d2;
  D_u32(D_row, 3) = d3;
}

template <class TensorA, class TensorB>
inline __device__ void mma_sync_aligned_m16n8k8_row_col_f16_f16_f16_f16(
    uint32_t& d0, uint32_t& d1, TensorA const& A, TensorB const& B,
    const uint32_t& c0, const uint32_t& c1) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int A_row0 = gid;     // for 0, 1
  int A_row1 = gid + 8; // for 2, 3
  int A_col = tid_in_group;
  auto A_u32 = cute::recast<uint32_t>(A);
  uint32_t a0 = A_u32(A_row0, A_col);
  uint32_t a1 = A_u32(A_row1, A_col);
  int B_row = tid_in_group * 2;
  int B_col = gid;
  uint32_t b0 = (uint32_t(reinterpret_cast<uint16_t&>(B(B_row, B_col))) << 16) |
                uint16_t(reinterpret_cast<uint16_t&>(B(B_row + 1, B_col)));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
               "{%0, %1},"
               "{%2, %3},"
               "{%4},"
               "{%5, %6};\n"
               : "=r"(d0), "=r"(d1)
               : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
}

template <class TensorD>
inline __device__ void mma_sync_aligned_m16n8k8_row_col_f16_f16_f16_f16_store(
    uint32_t& d0, uint32_t& d1, TensorD const& D) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int D_row0 = gid;
  int D_row1 = gid + 8;
  int D_col = tid_in_group;
  auto D_u32 = cute::recast<uint32_t>(D);
  D_u32(D_row0, D_col) = d0;
  D_u32(D_row1, D_col) = d1;
}

template <class TensorA, class TensorB>
inline __device__ void mma_sync_aligned_m16n8k8_row_col_f32_bf16_bf16_f32(
    float& d0, float& d1, float& d2, float& d3, TensorA const& A,
    TensorB const& B, const float& c0, const float& c1, const float& c2,
    const float& c3) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int A_row0 = gid;     // for 0, 1
  int A_row1 = gid + 8; // for 2, 3
  int A_col = tid_in_group;
  auto A_u32 = cute::recast<uint32_t>(A);
  uint32_t a0 = A_u32(A_row0, A_col);
  uint32_t a1 = A_u32(A_row1, A_col);
  int B_row = tid_in_group * 2;
  int B_col = gid;
  uint32_t b0 = (uint32_t(reinterpret_cast<uint16_t&>(B(B_row, B_col))) << 16) |
                uint16_t(reinterpret_cast<uint16_t&>(B(B_row + 1, B_col)));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
               "{%0,  %1,  %2,  %3},"
               "{%4,  %5},"
               "{%6},"
               "{%7,  %8,  %9,  %10};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

template <class TensorD>
inline __device__ void mma_sync_aligned_m16n8k8_row_col_f32_bf16_bf16_f32_store(
    float& d0, float& d1, float& d2, float& d3, TensorD const& D) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int D_row0 = gid;
  int D_row1 = gid + 8;
  int D_col = tid_in_group * 2;
  D(D_row0, D_col) = d0;
  D(D_row0, D_col + 1) = d1;
  D(D_row1, D_col) = d2;
  D(D_row1, D_col + 1) = d3;
}

template <class TensorA, class TensorB>
inline __device__ void mma_sync_aligned_m16n8k16_row_col_f16_f16_f16_f16(
    uint32_t& d0, uint32_t& d1, TensorA const& A, TensorB const& B,
    const uint32_t& c0, const uint32_t& c1) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int A_row0 = gid;              // for 0 and 1, 4 and 5
  int A_row1 = gid + 8;          // for 2 and 3, 6 and 7
  int A_col0 = tid_in_group;     // for 0, 1, 2 and 3
  int A_col1 = tid_in_group + 4; // for 4, 5, 6 and 7
  auto A_u32 = cute::recast<uint32_t>(A);
  uint32_t a0 = A_u32(A_row0, A_col0);
  uint32_t a1 = A_u32(A_row1, A_col0);
  uint32_t a2 = A_u32(A_row0, A_col1);
  uint32_t a3 = A_u32(A_row1, A_col1);
  int B_row0 = tid_in_group * 2;
  int B_row1 = tid_in_group * 2 + 8;
  int B_col = gid;
  uint32_t b0 =
      (uint32_t(reinterpret_cast<uint16_t&>(B(B_row0, B_col))) << 16) |
      uint16_t(reinterpret_cast<uint16_t&>(B(B_row0 + 1, B_col)));
  uint32_t b1 =
      (uint32_t(reinterpret_cast<uint16_t&>(B(B_row1, B_col))) << 16) |
      uint16_t(reinterpret_cast<uint16_t&>(B(B_row1 + 1, B_col)));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0,  %1},"
               "{%2,  %3,  %4,  %5},"
               "{%6,  %7},"
               "{%8,  %9};\n"
               : "=r"(d0), "=r"(d1)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0),
                 "r"(c1));
}

template <class TensorD>
inline __device__ void mma_sync_aligned_m16n8k16_row_col_f16_f16_f16_f16_store(
    uint32_t& d0, uint32_t& d1, TensorD const& D) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int D_row0 = gid;
  int D_row1 = gid + 8;
  int D_col = tid_in_group;
  auto D_u32 = cute::recast<uint32_t>(D);
  D_u32(D_row0, D_col) = d0;
  D_u32(D_row1, D_col) = d1;
}

template <class TensorA, class TensorB>
inline __device__ void mma_sync_aligned_m16n8k16_row_col_f32_bf16_bf16_f32(
    float& d0, float& d1, float& d2, float& d3, TensorA const& A,
    TensorB const& B, const float& c0, const float& c1, const float& c2,
    const float& c3) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int A_row0 = gid;              // for 0 and 1, 4 and 5
  int A_row1 = gid + 8;          // for 2 and 3, 6 and 7
  int A_col0 = tid_in_group;     // for 0, 1, 2 and 3
  int A_col1 = tid_in_group + 4; // for 4, 5, 6 and 7
  auto A_u32 = cute::recast<uint32_t>(A);
  uint32_t a0 = A_u32(A_row0, A_col0);
  uint32_t a1 = A_u32(A_row1, A_col0);
  uint32_t a2 = A_u32(A_row0, A_col1);
  uint32_t a3 = A_u32(A_row1, A_col1);
  int B_row0 = tid_in_group * 2;
  int B_row1 = tid_in_group * 2 + 8;
  int B_col = gid;
  uint32_t b0 =
      (uint32_t(reinterpret_cast<uint16_t&>(B(B_row0, B_col))) << 16) |
      uint16_t(reinterpret_cast<uint16_t&>(B(B_row0 + 1, B_col)));
  uint32_t b1 =
      (uint32_t(reinterpret_cast<uint16_t&>(B(B_row1, B_col))) << 16) |
      uint16_t(reinterpret_cast<uint16_t&>(B(B_row1 + 1, B_col)));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,  %1,  %2,  %3},"
               "{%4,  %5,  %6,  %7},"
               "{%8,  %9},"
               "{%10, %11, %12, %13};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
                 "f"(c1), "f"(c2), "f"(c3));
}

template <class TensorD>
inline __device__ void
mma_sync_aligned_m16n8k16_row_col_f32_bf16_bf16_f32_store(float& d0, float& d1,
                                                          float& d2, float& d3,
                                                          TensorD const& D) {
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);
  int lane = threadIdx.x & 31;
  int gid = lane >> 2;
  int tid_in_group = lane % 4;
  int D_row0 = gid;
  int D_row1 = gid + 8;
  int D_col = tid_in_group * 2;
  D(D_row0, D_col) = d0;
  D(D_row0, D_col + 1) = d1;
  D(D_row1, D_col) = d2;
  D(D_row1, D_col + 1) = d3;
}

#endif // __CHOREO_TARGET_CUTE__

#if defined(__TOPSCC__) || defined(__CHOREO_TARGET_CUTE__)

template <typename T>
struct is_future : std::false_type {};
template <>
struct is_future<future> : std::true_type {};

template <typename T, typename... Rest>
__device__ void inline LeftRotateFutures(T& first, T& second, Rest&... rest) {
  static_assert(is_future<T>::value,
                "All arguments must be of type choreo::future");
  static_assert((is_future<Rest>::value && ...),
                "All arguments must be of type choreo::future");

  // swap the pointers
  swap(first, second);

  if constexpr (sizeof...(rest) > 0) LeftRotateFutures(second, rest...);
}

template <typename... Futures>
__device__ inline void rotate(Futures&... f) {
  static_assert(sizeof...(f) > 1, "rotate futures less than 1.");
  LeftRotateFutures(f...);
}
#endif

} // end namespace choreo

#endif // __CHOREO_H__
